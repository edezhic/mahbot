//! Configuration: hardcoded defaults in this module, overlayed from `core.db`
//! by [`crate::config_db`]. [`load_or_init`] seeds
//! `ConfigData::STRUCT_FIELDS_DEFAULT` into [`CONFIG`], then [`reload_from_db`]
//! overlays persisted values.
//!
//! Two **independent** resolution chains — a `config_kv` entry cannot override
//! a per-model routing, and vice versa:
//!
//! * KV chain: `config_kv` table → hardcoded `const`. Covers the
//!   `string_config_fields!` fields and the two model slots. Accessor
//!   annotations: `non_empty` → `Option<String>` (empty/whitespace → `None`);
//!   `or(DEFAULT)` → `String`; `list_or(default = …)` → `Vec<String>` from a
//!   newline-separated list, and a `, raw` suffix also generates
//!   `<field>_raw`, the list exactly as configured with no default injected.
//!   [`reload_from_db`] applies rows through [`ConfigData::set_string_field`];
//!   an absent key stays `None` so the accessor resolves its fallback. Fields
//!   outside this chain (e.g. `model_routings`) have their own table.
//! * Routing chain: `config_model_routing` row → [`default_model_routing`] →
//!   OpenRouter auto-routing. Stored in [`ConfigData::model_routings`]; every
//!   chat request is built through [`ConfigReload::model_routing`].
//!
//! `provider_endpoint` (self-hosted OpenAI-compatible: Ollama, LM Studio,
//! llama.cpp, vLLM, LiteLLM) is honored at runtime and falls back to
//! OpenRouter when unset. `provider_endpoint_key` is that endpoint's own
//! optional key — many servers are keyless — and is **only** ever sent to it,
//! never to OpenRouter. Media features (image/video generation, catalogs) always
//! use `DEFAULT_PROVIDER_ENDPOINT` + `provider_key` regardless of the custom chat
//! endpoint.
//!
//! [`ConfigReload::role_model`] maps `Role::Manager`, `Assistant`, `Discovery`,
//! `Engineer` and `Maintainer` to the manager slot; every other role to the
//! worker slot. Unset slots fall back to their `DEFAULT_*_MODEL` constant.
//!
//! Rows in `config_kv` without a [`ConfigData`] field are lazily purged on
//! reload: each unknown key is logged at `debug`, then garbage rows are
//! deleted best-effort (debug on success, warn without failing boot). The
//! intentional shared namespaces — nightly-discovery marker, temp-cleanup
//! cadence, update notice, low-disk notice ladder — are left untouched.
//! Downgrade resurrection is therefore lost: orphans are deleted on first
//! reboot instead of re-appearing. The settings path cannot create new ones
//! because [`write_kv_and_update_config`] rejects unknown keys before any DB
//! write. [`crate::providers::compatible`] writes the resolved provider order
//! into the OpenRouter body (`None` sends no preference at all).

use crate::Role;
use crate::util::{UnwrapPoison, is_http_url};
use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use tokio::fs;

pub(crate) const DEFAULT_PROVIDER_ENDPOINT: &str = "https://openrouter.ai/api/v1";

/// Pinned by `default_model_identifier_is_pinned` so a typo fails the suite instead of reaching
/// a provider request.
const DEFAULT_CHAT_MODEL: &str = "deepseek/deepseek-v4.1-flash-1017";
pub(crate) const DEFAULT_MANAGER_MODEL: &str = DEFAULT_CHAT_MODEL;
pub(crate) const DEFAULT_WORKER_MODEL: &str = DEFAULT_CHAT_MODEL;

// Every value a release ever compiled as a slot default, newest first. A stored value equal to one
// was never chosen by the user (a deliberate pick of the same id is indistinguishable and moves
// too), so `migrate_old_default_models` rewrites it on boot. Per-slot lists keep a genuine pick of
// the other slot's old default intact.
const OLD_DEFAULT_MANAGER_MODELS: &[&str] = &[
    "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4-pro-0813",
];
const OLD_DEFAULT_WORKER_MODELS: &[&str] = &[
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4-flash-0731",
];

/// Used by every account with no explicit pick. The retired predecessor
/// (`google/gemini-3.1-flash-image`) is rewritten out of the stored picker list and every
/// account's pick by catalog entry `50`.
pub(crate) const DEFAULT_IMAGE_GEN_MODEL: &str = "google/gemini-nano-banana-2.1";
/// Used by every account with no explicit pick; a stored pick is never rewritten.
pub(crate) const DEFAULT_VIDEO_MODEL: &str = "google/gemini-omni-1.2-flash";

/// Fresh-install seeded image-generation model list: newline-separated, in picker order; mirrors
/// the curated set the live install ships.
const FRESH_INSTALL_IMAGE_GEN_MODELS: &str = "google/gemini-nano-banana-2.1\nmicrosoft/mai-image-2.6\nopenai/gpt-image-2.5-sunburst\nqwen/qwen-image-3-pro";

/// Fresh-install seeded video-generation model list: newline-separated, in picker order, the
/// default model first (as in the image list).
const FRESH_INSTALL_VIDEO_MODELS: &str =
    "google/gemini-omni-1.2-flash\nbytedance/seedance-2.0-mini\nminimax/hailuo-3";

#[derive(Debug, Clone, PartialEq)]
pub struct ModelRouting {
    pub model: String,
    pub provider_order: Option<String>,
}

impl ModelRouting {
    /// Find-or-push: update the fields `set_field` mutates on the entry matching `model`, or push a
    /// new one defaulted to `None`. An existing entry's other fields are preserved.
    pub(crate) fn upsert(
        routings: &mut Vec<ModelRouting>,
        model: impl Into<String>,
        set_field: impl FnOnce(&mut ModelRouting),
    ) {
        let model = model.into();
        if let Some(existing) = routings.iter_mut().find(|mr| mr.model == model) {
            set_field(existing);
        } else {
            let mut new = ModelRouting {
                model,
                provider_order: None,
            };
            set_field(&mut new);
            routings.push(new);
        }
    }
}

/// All runtime-configurable fields, protected by an [`RwLock`] in [`ConfigReload`].
/// Accessors return owned strings because the lock guard cannot escape their scope.
///
/// Adding a persisted field needs two steps, in order: declare it here, then
/// add it to the `string_config_fields!` invocation in this file — that macro
/// emits `STRUCT_FIELDS_DEFAULT` (`const Self { … }`, so a missing entry is a
/// compile error), [`ConfigData::string_fields()`], [`ConfigData::set_string_field()`],
/// the `CONFIG_KEY_<FIELD>` consts and the typed accessors. Use those consts,
/// never raw key literals, so a rename stays compiler-tied.
///
/// **Every** `Option<String>` here must be in the macro: there is no
/// "transient" `Option<String>` outside it. A field not meant to persist
/// should use another type or live elsewhere — otherwise it still lands in
/// `string_fields()` and gets written by `reload_from_db`. The same guard
/// covers the UX asymmetry: the GUI Settings page reads the whole
/// [`ConfigReload::snapshot`], while persisting goes only through the
/// macro-generated [`ConfigData::string_fields`], so a missing entry would look
/// editable and silently discard its value on save.
#[derive(Debug, Clone)]
pub struct ConfigData {
    pub provider_key: Option<String>,
    pub provider_endpoint: Option<String>,
    /// The custom endpoint's own key (many are keyless); never sent to OpenRouter.
    pub provider_endpoint_key: Option<String>,
    /// Model slot for the Manager group (Manager, Assistant, Discovery, Engineer, Maintainer).
    pub manager_model: Option<String>,
    /// Model slot for all worker roles (Analyst, Worker, QA, Reviewer, Sanitation).
    pub worker_model: Option<String>,
    /// Newline-separated picker list.
    pub image_gen_models: Option<String>,
    /// Newline-separated picker list.
    pub video_models: Option<String>,
    /// DeepSeek platform API key (`api.deepseek.com`) for the web-search tool —
    /// not the OpenRouter key the product's DeepSeek models travel through.
    pub deepseek_key: Option<String>,
    pub exa_key: Option<String>,
    /// Web search provider selection: `"deepseek"` or `"exa"` (case-insensitive).
    /// When `None`, no explicit choice was made — text searches then go to
    /// DeepSeek and picture searches to Exa when that key is configured.
    pub web_search_provider: Option<String>,
    /// Hot-reloaded on save.
    pub telegram_bot_token: Option<String>,
    /// Which Bot API the Telegram transport talks to: `"local"` serves the bot
    /// from the product's own self-hosted server, anything else (or no row) is
    /// Telegram's cloud API. This row is the durable decision of a switch —
    /// the local-server supervisor reads it at every start and finishes an
    /// unfinished switch before the bot runs.
    pub telegram_local_server: Option<String>,
    /// `api_id` of the owner's Telegram application (`my.telegram.org`), read
    /// only by that server.
    pub telegram_api_id: Option<String>,
    pub telegram_api_hash: Option<String>,
    /// Journal of a switch to or from the local server that has not finished
    /// yet (JSON, see `channels::telegram_local`). Present only while a switch
    /// is in progress or blocked, so its presence is what makes a crashed
    /// switch resumable.
    pub telegram_local_migration: Option<String>,
    /// Enable local audio transcription (every platform).
    ///
    /// Absence of the row means enabled; catalog `69` writes `"false"` into
    /// every install that never wrote it, so an untouched installation ships
    /// disabled. When enabled and the model set is on disk or can be fetched,
    /// audio transcription runs fully locally on the inference crate's
    /// recognition pair (`crate::speech`). Audio never leaves the machine.
    ///
    /// Set to `"false"` to keep the recogniser's words out of what the model reads:
    /// an audio marker is replaced with just the icon combo and its temp file is
    /// deleted, so a voice message is not recoverable. The recogniser itself still
    /// runs where its output is not words — a video's or a recording's sound
    /// labels come from the same pass, and a media timeline keeps its frames and
    /// its labels while leaving the speech lines out.
    pub audio_transcription_use_local: Option<String>,
    /// Voice assistant (wake word detection and commands); `"true"` enables voice mode.
    pub voice_enabled: Option<String>,
    /// Enable speech output for the admin's Assistant (default: `"false"`). Set to
    /// `"true"` to let it speak short updates aloud through the `say` tool on this
    /// machine. Nothing is ever read aloud on its own.
    pub tts_enabled: Option<String>,
    /// JSON wake word enrollment (schema 3: prototype + calibration). Owned exclusively by the
    /// voice pipeline; an older record is refused and the owner enrols again.
    pub wake_word_templates: Option<String>,
    /// Onboarding state machine: "init" | "finished".
    /// Absent = Init (fresh install). Set by the onboarding flow; existing
    /// installs keep the `finished` value seeded by the retired catalog's
    /// one-time migration. A legacy `"welcomed"` value reads as `Finished`.
    pub onboarding_state: Option<String>,
    /// Start at login: `"false"` when the owner turned the Settings switch off,
    /// anything else (or no row) when it is on. The decision outlives installs and
    /// updates; `crate::autostart` owns what the system's registration is.
    pub autolaunch: Option<String>,
    /// Update by itself: `"false"` when the owner turned the Settings switch off,
    /// anything else (or no row) when it is on. It only gates the unattended
    /// trigger — the release check, the changelog and the manual update are unaffected.
    pub autoupdate: Option<String>,
    pub model_routings: Vec<ModelRouting>,
}

/// Generate `string_fields()`, `set_string_field()`, the const `STRUCT_FIELDS_DEFAULT`, the
/// `CONFIG_KEY_<FIELD>` constants **and** the typed accessors on [`ConfigReload`] from one
/// annotated field list, so adding or removing a field updates every sync item at once.
///
/// Each field is declared as `$field [$annotation]` with `non_empty`, `or($default)`,
/// `list_or(default = $default)` or `list_or(default = $default, raw)` — the last also emitting
/// a raw `<field>_raw` accessor with no default injected.
///
/// The `CONFIG_KEY_<FIELD>` constant pastes the name from the field token and
/// `stringify!`s the same token, so a rename changes both and stale use sites
/// fail to compile instead of silently dropping their side effect.
macro_rules! string_config_fields {
    (
        $(
            $field:ident [ $($annotation:tt)* ]
        ),* $(,)?
    ) => {
        ::paste::paste! {
            $(
                pub const [<CONFIG_KEY_ $field:upper>]: &str = stringify!($field);
            )*
        }

        impl ConfigData {
            /// Default-initialised [`ConfigData`], used by [`ConfigReload::const_new`].
            /// A `const Self { … }` requires every field, so a struct–macro
            /// mismatch is a compile error before any test runs.
            pub(crate) const STRUCT_FIELDS_DEFAULT: Self = Self {
                $($field: None,)*
                model_routings: Vec::new(),
            };

            /// All string-valued fields as (db_key, current_value) pairs;
            /// `model_routings` is excluded — separate table.
            #[must_use]
            pub fn string_fields(&self) -> Vec<(&'static str, Option<&str>)> {
                vec![$((stringify!($field), self.$field.as_deref())),*]
            }

            /// Set a string field by its database key, `true` when recognised.
            /// The value is stored as-is: call [`Self::normalize`] to collapse
            /// empty/whitespace-only values to `None`.
            #[must_use]
            pub fn set_string_field(&mut self, key: &str, value: &str) -> bool {
                match key {
                    $(stringify!($field) => self.$field = Some(value.to_owned()),)*
                    _ => return false,
                }
                true
            }

            /// Allocation-free counterpart to [`set_string_field`], mirroring its
            /// arms so both agree on the generated key set; `None` for unknown
            /// keys. The value is as stored — [`Self::normalize`] first.
            #[must_use]
            pub(crate) fn get_string_field(&self, key: &str) -> Option<&str> {
                match key {
                    $(stringify!($field) => self.$field.as_deref(),)*
                    _ => None,
                }
            }

            /// Whether `key` names a known field — checked against the generated
            /// field set, not the value (on [`STRUCT_FIELDS_DEFAULT`] every value
            /// is `None`, so [`get_string_field`] would always answer no).
            #[must_use]
            fn has_string_field(&self, key: &str) -> bool {
                match key {
                    $(stringify!($field) => true,)*
                    _ => false,
                }
            }

            /// Trim every string field in place and collapse empty/whitespace to
            /// `None`. [`set_string_field`] stores as-is, so this is the canonical
            /// normalisation point — call it before the config is used.
            fn normalize_string_fields(&mut self) {
                $(self.$field = non_empty(self.$field.take());)*
            }
        }

        // Both passes are needed: normalize() writes canonical values so the
        // persist paths' `!=` comparisons see no spurious diffs, and the
        // accessors below re-normalise raw values.
        impl ConfigReload {
            $(
                string_config_fields!(@accessor $field $($annotation)*);
            )*
        }
    };

    // non_empty: Option<String>, empty/whitespace collapsed to None.
    (@accessor $field:ident non_empty) => {
        #[doc = concat!(
            "Returns the configured `", stringify!($field),
            "`, with empty/whitespace values collapsed to `None`."
        )]
        #[must_use]
        pub fn $field(&self) -> Option<String> {
            non_empty(self.read().$field.clone())
        }
    };

    // or(DEFAULT): String, falling back to the given default constant.
    (@accessor $field:ident or($default:expr)) => {
        #[doc = concat!(
            "Returns the configured `", stringify!($field),
            "`, falling back to the default if unset."
        )]
        #[must_use]
        pub fn $field(&self) -> String {
            resolve_or(self.read().$field.clone(), $default)
        }
    };

    // list_or(default = <const>): Vec<String> from a newline-separated list,
    // falling back to `$default`. `, raw` also emits `<field>_raw` — the stored
    // list with no default injected, for a caller that must tell "nothing was
    // chosen" apart from "the default was chosen".
    (@accessor $field:ident list_or(default = $default:expr)) => {
        string_config_fields!(@list_accessor $field, $default);
    };
    (@accessor $field:ident list_or(default = $default:expr, raw)) => {
        string_config_fields!(@list_accessor $field, $default);
        string_config_fields!(@raw_accessor $field);
    };

    (@list_accessor $field:ident, $default:expr) => {
        #[doc = concat!(
            "Returns the list of available `", stringify!($field), "`.",
            "\n\nIf unset or the parsed newline-separated list is empty,",
            " falls back to a built-in default."
        )]
        #[must_use]
        pub fn $field(&self) -> Vec<String> {
            let guard = self.read();
            resolve_list_or(guard.$field.as_deref(), $default)
        }
    };

    (@raw_accessor $field:ident) => {
        ::paste::paste! {
            #[doc = concat!(
                "Returns the `", stringify!($field), "` list exactly as configured",
                " (empty when unset), with no default injected."
            )]
            #[must_use]
            pub fn [<$field _raw>](&self) -> Vec<String> {
                parse_newline_list(self.read().$field.as_deref().unwrap_or_default())
            }
        }
    };
}

string_config_fields! {
    provider_key [non_empty],
    provider_endpoint [or(DEFAULT_PROVIDER_ENDPOINT)],
    provider_endpoint_key [non_empty],
    manager_model [or(DEFAULT_MANAGER_MODEL)],
    worker_model [or(DEFAULT_WORKER_MODEL)],
    image_gen_models [list_or(default = DEFAULT_IMAGE_GEN_MODEL, raw)],
    video_models [list_or(default = DEFAULT_VIDEO_MODEL)],
    deepseek_key [non_empty],
    exa_key [non_empty],
    web_search_provider [non_empty],
    telegram_bot_token [non_empty],
    telegram_local_server [non_empty],
    telegram_api_id [non_empty],
    telegram_api_hash [non_empty],
    telegram_local_migration [non_empty],
    audio_transcription_use_local [non_empty],
    voice_enabled [non_empty],
    tts_enabled [non_empty],
    wake_word_templates [non_empty],
    onboarding_state [non_empty],
    autolaunch [non_empty],
    autoupdate [non_empty],
}

impl ConfigData {
    /// Trim inner fields of `Vec` entries and collapse empty to `None` — the
    /// counterpart of [`normalize_string_fields()`], which stops at the top level.
    fn normalize_entries(&mut self) {
        for mr in &mut self.model_routings {
            mr.provider_order = non_empty(mr.provider_order.take());
        }
    }

    /// Canonical form: trim top-level fields and `Vec`-entry inner fields,
    /// collapsing empty to `None`, then sort `model_routings` by model name. Every
    /// caller building a [`ConfigData`] MUST call this before swapping into
    /// [`CONFIG`] so the in-memory value is path-independent.
    pub(crate) fn normalize(&mut self) {
        self.normalize_string_fields();
        self.normalize_entries();
        self.model_routings.sort_by(|a, b| a.model.cmp(&b.model));
    }
}

/// Trim, and `None` when empty or whitespace-only. The canonical primitive.
#[must_use]
pub(crate) fn trimmed_or_none(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Newline-separated string into non-empty, trimmed entries.
#[must_use]
pub(crate) fn parse_newline_list(s: &str) -> Vec<String> {
    s.split('\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Empty or whitespace-only becomes `None`; the value is trimmed.
#[must_use]
fn non_empty(val: Option<String>) -> Option<String> {
    val.and_then(|s| trimmed_or_none(&s))
}

/// Whether a stored on/off switch reads as on: an absent row means on, and only a
/// stored `false` reads off — trimmed and case-insensitive, so a row written by hand
/// reads the way one written here does.
#[must_use]
pub(crate) fn flag_on(val: Option<&str>) -> bool {
    !val.is_some_and(|val| val.trim().eq_ignore_ascii_case("false"))
}

/// `val` when non-empty after trimming, else `fallback`.
#[must_use]
pub(crate) fn resolve_or(val: Option<String>, fallback: &str) -> String {
    non_empty(val).unwrap_or(fallback.to_string())
}

/// Parsed non-empty lines of `list_field`, else the single-element
/// `[default_value]`.
#[must_use]
fn resolve_list_or(list_field: Option<&str>, default_value: &str) -> Vec<String> {
    if let Some(raw) = list_field {
        let parsed = parse_newline_list(raw);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    vec![default_value.to_string()]
}

/// Normalize an endpoint URL for comparison: trim, lowercase scheme and host
/// but not the path, strip trailing slashes and one trailing
/// `/chat/completions`. The default OpenRouter URL and these trivial variants
/// must never count as a custom endpoint. Suffix matching is case-sensitive
/// (`/Chat/Completions` stays). Scheme-less input is only trimmed/stripped, and
/// empty input normalizes to `""`.
///
/// SSoT: a change here intentionally moves fetch-URL construction at every call
/// site (catalog, compatible provider).
#[must_use]
pub(crate) fn normalize_endpoint_url(url: &str) -> String {
    let t = url.trim().trim_end_matches('/');
    // The suffix has no trailing slash after it is stripped.
    let t = t
        .strip_suffix("/chat/completions")
        .unwrap_or(t)
        .trim_end_matches('/');
    // Lowercase scheme + host, preserve the path as-is.
    if let Some(scheme_end) = t.find("://") {
        let scheme_and_dots = &t[..scheme_end + 3];
        let rest = &t[scheme_end + 3..];
        let authority_end = rest.find('/').unwrap_or(rest.len());
        let authority = &rest[..authority_end];
        let path = &rest[authority_end..];
        format!(
            "{}{}{}",
            scheme_and_dots.to_ascii_lowercase(),
            authority.to_ascii_lowercase(),
            path
        )
    } else {
        // No scheme — validation rejects such values anyway; return unchanged.
        t.to_string()
    }
}

/// Whether `endpoint` is the default one, modulo the variants
/// [`normalize_endpoint_url`] folds away.
#[must_use]
pub(crate) fn is_default_endpoint(endpoint: &str) -> bool {
    normalize_endpoint_url(endpoint) == normalize_endpoint_url(DEFAULT_PROVIDER_ENDPOINT)
}

#[must_use]
pub(crate) fn is_custom_endpoint(endpoint: &str) -> bool {
    !is_default_endpoint(endpoint)
}

/// Whether the LLM provider is configured: a non-empty OpenRouter key OR a
/// genuinely custom endpoint (keyless ones count; the default URL never does).
/// Gates pipeline pickup and the GUI boot-to-Settings redirect.
#[must_use]
pub(crate) fn provider_configured() -> bool {
    is_custom_endpoint(&CONFIG.provider_endpoint()) || CONFIG.provider_key().is_some()
}

/// A model provider is configured and onboarding finished. A product message
/// landing before either is a guaranteed failing turn, so its producer stays
/// silent instead.
#[must_use]
pub(crate) fn assistant_can_run() -> bool {
    provider_configured() && CONFIG.onboarding_stage() == OnboardingState::Finished
}

/// The snapshot's persisted endpoint when set, else the default.
#[must_use]
pub(crate) fn effective_chat_endpoint(config: &ConfigData) -> String {
    resolve_or(config.provider_endpoint.clone(), DEFAULT_PROVIDER_ENDPOINT)
}

/// Chat credential under `config`: the custom endpoint's own key when one is
/// active (`None` = keyless, no Authorization header), else the OpenRouter key.
/// The OpenRouter key is NEVER sent to a custom endpoint.
#[must_use]
pub(crate) fn chat_credential(config: &ConfigData) -> Option<String> {
    let ep = effective_chat_endpoint(config);
    if is_custom_endpoint(&ep) {
        non_empty(config.provider_endpoint_key.clone())
    } else {
        non_empty(config.provider_key.clone())
    }
}

/// Global reloadable config singleton.
///
/// The `storage_root` is immutable after startup; all other fields live in an
/// `RwLock<ConfigData>` that can be atomically swapped at runtime.
pub static CONFIG: ConfigReload = ConfigReload::const_new();

/// Serializes every settled-field persist (settings-page autosave), held across
/// the whole persist — including the provider warmup call — so read-modify-write
/// sequences and side-effect settles (warmup, Telegram restart) cannot race and
/// clobber each other's rows with stale snapshots.
static CONFIG_PERSIST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn persist_lock() -> &'static tokio::sync::Mutex<()> {
    CONFIG_PERSIST_LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Whether `core.db` was absent on disk before the config store was opened this
/// boot, as recorded by [`load_or_init`]. [`reload_from_db`] seeds defaults into
/// brand-new databases only — existing ones receive zero writes (hard constraint).
static CONFIG_DB_FRESH_AT_BOOT: AtomicBool = AtomicBool::new(false);

/// Reloadable configuration with atomic swap. Readers never block each other;
/// writes happen only at startup and on GUI-driven saves.
pub struct ConfigReload {
    storage_root: OnceLock<PathBuf>,
    inner: RwLock<ConfigData>,
}

impl ConfigReload {
    #[must_use]
    pub const fn const_new() -> Self {
        Self {
            storage_root: OnceLock::new(),
            inner: RwLock::new(ConfigData::STRUCT_FIELDS_DEFAULT),
        }
    }

    /// # Panics
    /// Panics if storage root has not been set.
    pub fn global_storage_root(&self) -> PathBuf {
        self.storage_root
            .get()
            .expect("CONFIG storage_root not initialized")
            .clone()
    }

    /// `None` instead of panicking when no root is set — for paths that tolerate
    /// an uninitialized config (tests).
    #[must_use]
    pub fn try_storage_root(&self) -> Option<PathBuf> {
        self.storage_root.get().cloned()
    }

    pub(crate) fn set_storage_root(&self, root: PathBuf) {
        self.storage_root
            .set(root)
            .expect("CONFIG storage_root already set");
    }

    /// `Err` instead of panicking when already set — the root may have been set
    /// by a previously-running test.
    #[cfg(test)]
    pub(crate) fn try_set_storage_root(&self, root: PathBuf) -> std::result::Result<(), PathBuf> {
        self.storage_root.set(root)
    }

    /// Get a read-locked snapshot of the current config.
    /// Prefer the typed accessors below for individual fields.
    fn read(&self) -> RwLockReadGuard<'_, ConfigData> {
        self.inner.read().unwrap_poison()
    }

    /// Replace the entire config atomically (used during startup and reload).
    pub(crate) fn swap(&self, new_config: ConfigData) {
        *self.inner.write().unwrap_poison() = new_config;
    }

    /// Get a full clone of the current config for serialisation / GUI display.
    #[must_use]
    pub fn snapshot(&self) -> ConfigData {
        self.read().clone()
    }

    /// The current onboarding state (derived from the `onboarding_state` field).
    #[must_use]
    pub fn onboarding_stage(&self) -> OnboardingState {
        OnboardingState::from_raw(self.onboarding_state().as_deref())
    }

    /// In-memory only: no DB write and no provider warmup, so callers must
    /// persist separately (e.g. [`crate::config_db::ConfigStore::set_kv`]).
    /// `true` when the key was recognised; unknown keys are ignored for
    /// forward compatibility.
    #[must_use]
    pub fn set_string_field(&self, key: &str, value: &str) -> bool {
        let mut guard = self.inner.write().unwrap_poison();
        guard.set_string_field(key, value)
    }

    /// Find-or-push in memory; `None` `provider_order` removes the row, mirroring
    /// the DELETE-if-empty persistence path.
    fn set_model_routing_row(&self, model: &str, provider_order: Option<String>) {
        let mut guard = self.inner.write().unwrap_poison();
        if provider_order.is_none() {
            guard.model_routings.retain(|mr| mr.model != model);
        } else {
            ModelRouting::upsert(&mut guard.model_routings, model, |mr| {
                mr.provider_order = provider_order;
            });
        }
    }

    /// The raw stored row, unlike [`Self::model_routing`] which resolves the
    /// effective one. `None` lets the settings page tell "no override" apart
    /// from "override cleared" when mirroring settled rows.
    pub(crate) fn model_routing_by_key(&self, model: &str) -> Option<ModelRouting> {
        let guard = self.read();
        guard
            .model_routings
            .iter()
            .find(|mr| mr.model == model)
            .cloned()
    }

    /// The effective routing: the stored order (Settings → Provider → Models settings →
    /// Provider Routing) wins, else [`default_model_routing`], so a cleared field behaves
    /// exactly as if that provider had been set.
    #[must_use]
    pub(crate) fn model_routing(&self, model: &str) -> ModelRouting {
        // No stored order → the model's default provider; see `default_model_routing`.
        let provider_order = self
            .model_routing_by_key(model)
            .and_then(|mr| mr.provider_order)
            .or_else(|| default_model_routing(model).map(str::to_string));
        ModelRouting {
            model: model.to_string(),
            provider_order,
        }
    }

    /// The manager group (Manager, Assistant, Discovery, Engineer, Maintainer)
    /// uses the manager slot; every other role the worker slot. Unset slots fall
    /// back to their code default.
    #[must_use]
    pub fn role_model(&self, role: Role) -> String {
        match role {
            Role::Manager
            | Role::Assistant
            | Role::Discovery
            | Role::Engineer
            | Role::Maintainer => self.manager_model(),
            _ => self.worker_model(),
        }
    }
}

pub fn default_config_dir() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return Ok(PathBuf::from(home).join(".mahbot"));
    }

    let home = UserDirs::new()
        .map(|u| u.home_dir().to_path_buf())
        .context("Could not find home directory")?;
    Ok(home.join(".mahbot"))
}

/// Resolve `~/.mahbot` as the storage root, create it, seed hardcoded defaults
/// and store the result in [`CONFIG`]. The caller must then call
/// [`reload_from_db`], and providers must be initialised **after** it, or API
/// keys and model settings do not take effect.
pub async fn load_or_init() -> Result<()> {
    let mahbot_dir = default_config_dir()?;
    fs::create_dir_all(&mahbot_dir)
        .await
        .context("Failed to create config directory")?;

    CONFIG.set_storage_root(mahbot_dir.clone());

    // Probed BEFORE any store open creates it. Probing any other location would
    // classify every existing install as fresh and re-seed it on each boot,
    // violating the zero-write hard constraint.
    CONFIG_DB_FRESH_AT_BOOT.store(config_db_is_fresh(&mahbot_dir), Ordering::Release);

    // Start with hardcoded defaults — reload_from_db() will overlay
    // any persisted values from the config store (called later in bootstrap).
    CONFIG.swap(ConfigData::STRUCT_FIELDS_DEFAULT);

    tracing::info!(
        "Config system initialised (storage root: {}).",
        mahbot_dir.display()
    );
    Ok(())
}

/// `true` when the consolidated domain database
/// ([`crate::db::CONSOLIDATED_DB_NAME`]) does not exist yet. Must be probed
/// BEFORE any store open creates it, and must check the exact path the open
/// uses (`init_all_stores` → `open_store` → `store_db_path`).
fn config_db_is_fresh(mahbot_dir: &std::path::Path) -> bool {
    !crate::db::store_db_path(mahbot_dir, "config").exists()
}

/// The provider that serves a model unless the user chose one: a DeepSeek-family
/// slug (case-insensitive substring) → OpenRouter's lowercase `deepseek`; every
/// other model stays auto-routed.
///
/// **DO NOT DELETE, WEAKEN OR "SIMPLIFY" THIS RULE WITHOUT AN EXPLICIT USER
/// REQUEST.** Which host serves a model decides prompt-cache locality, so
/// auto-routing a DeepSeek model across hosts is a large billing regression.
pub(crate) fn default_model_routing(model: &str) -> Option<&'static str> {
    model
        .to_ascii_lowercase()
        .contains("deepseek")
        .then_some("deepseek")
}

/// Seed the rows a fresh install must show: the image/video generation model
/// lists (the Settings GUI reads the raw snapshot fields, not the
/// default-resolving `list_or` accessors) and the default slots' routing. The
/// active selection is per-user (`crate::users`), so no singular key is seeded.
/// The transcription-off flag belongs to catalog `69`, the single owner of that
/// initial value. `fresh` is the pre-open discriminator from [`load_or_init`] —
/// the seed writes nothing to existing databases.
async fn seed_fresh_install_defaults(
    fresh: bool,
    store: &crate::config_db::ConfigStore,
) -> Result<()> {
    if !fresh {
        return Ok(());
    }
    store
        .set_kv(CONFIG_KEY_IMAGE_GEN_MODELS, FRESH_INSTALL_IMAGE_GEN_MODELS)
        .await?;
    store
        .set_kv(CONFIG_KEY_VIDEO_MODELS, FRESH_INSTALL_VIDEO_MODELS)
        .await?;
    // These rows duplicate what an empty field resolves to, but are kept so the
    // Settings routing field renders the value. Only the manager/worker slots
    // are seeded: the selection is per-user.
    for default_model in [DEFAULT_MANAGER_MODEL, DEFAULT_WORKER_MODEL] {
        if let Some(provider) = default_model_routing(default_model) {
            store
                .save_model_routing(default_model, Some(provider))
                .await?;
        }
    }
    tracing::info!(
        "Fresh config database: seeded fresh-install defaults (image/video generation model sets; provider routing for the default model slots)"
    );
    Ok(())
}

/// Consume the boot discriminator and seed when it fired, mirroring the
/// `load_or_init` → `reload_from_db` handoff, so the seed lands once per fresh
/// boot only. Separate so the consumption path is testable without the global store.
async fn seed_fresh_install_defaults_from_flag(
    store: &crate::config_db::ConfigStore,
) -> Result<()> {
    let fresh = CONFIG_DB_FRESH_AT_BOOT.swap(false, Ordering::AcqRel);
    seed_fresh_install_defaults(fresh, store).await
}

/// Idempotent: rewrite persisted manager/worker rows holding any historical
/// compiled default. Every value a release ever shipped is listed, so an
/// installation that skipped releases migrates too. Exact match on the full
/// provider-prefixed string — accepted residual: a deliberate pick of an id that
/// was also a historical default is indistinguishable and moves. A rewrite also
/// seeds the new default's routing row so the Settings field renders it. A no-op
/// once no row matches, so safe on every boot.
async fn migrate_old_default_models(store: &crate::config_db::ConfigStore) -> Result<()> {
    let mut changed_manager = 0;
    for old in OLD_DEFAULT_MANAGER_MODELS {
        changed_manager += store
            .migrate_kv_if_equals(CONFIG_KEY_MANAGER_MODEL, old, DEFAULT_MANAGER_MODEL)
            .await?;
    }
    let mut changed_worker = 0;
    for old in OLD_DEFAULT_WORKER_MODELS {
        changed_worker += store
            .migrate_kv_if_equals(CONFIG_KEY_WORKER_MODEL, old, DEFAULT_WORKER_MODEL)
            .await?;
    }
    for (changed, model) in [
        (changed_manager, DEFAULT_MANAGER_MODEL),
        (changed_worker, DEFAULT_WORKER_MODEL),
    ] {
        if changed > 0
            && let Some(provider) = default_model_routing(model)
        {
            store.save_model_routing(model, Some(provider)).await?;
        }
    }
    if changed_manager > 0 || changed_worker > 0 {
        tracing::info!(
            manager_rows = changed_manager,
            worker_rows = changed_worker,
            "Migrated stored old default models to the current defaults"
        );
    }
    Ok(())
}

/// Overlay persisted settings on top of the hardcoded defaults, atomically. Called
/// at startup, after `config_db` init.
pub async fn reload_from_db() -> Result<()> {
    let store = crate::config_db::store();
    let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;

    // The config store open above already ran catalog 69, which owns the
    // transcription-off flag.
    seed_fresh_install_defaults_from_flag(store).await?;

    // Rewrite persisted old-default manager/worker rows to the new default
    // before the overlay below, so this boot serves the migrated value.
    migrate_old_default_models(store).await?;

    let kvs = store.get_all_kv().await?;
    for (key, value) in &kvs {
        if !config.set_string_field(key, value) {
            tracing::debug!(key, "Unknown config key, ignoring");
        }
    }
    purge_unknown_kv_orphans(store, &kvs).await;

    let routings = store.get_all_model_routings().await?;
    config.model_routings = routings;

    // Normalised and sorted so the in-memory value matches what the
    // per-field persistence paths write.
    config.normalize();

    CONFIG.swap(config);
    tracing::info!("Config reloaded from DB");
    Ok(())
}

/// Delete `config_kv` rows whose key is no longer a recognized field, so a stale
/// value cannot resurface if the key is reintroduced. The shared namespaces above
/// are left untouched. Delete failures are tolerated (transient, retried on the
/// next reload).
async fn purge_unknown_kv_orphans(store: &crate::config_db::ConfigStore, kvs: &[(String, String)]) {
    let orphans: Vec<String> = kvs
        .iter()
        .filter(|(key, _)| {
            !ConfigData::STRUCT_FIELDS_DEFAULT.has_string_field(key)
                && key != crate::workspace::NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY
                && key != crate::temp::TEMP_CLEANUP_LAST_RUN_KV_KEY
                && key != crate::temp::TEMP_CLEANUP_MODE_KV_KEY
                && key != crate::self_update::UPDATE_LAST_RUN_VERSION_KV_KEY
                && key != crate::self_update::UPDATE_PENDING_CHANGELOG_KV_KEY
                && key != crate::disk_notice::LADDER_KV_KEY
        })
        .map(|(key, _)| key.clone())
        .collect();
    if !orphans.is_empty() {
        tracing::debug!(keys=?orphans, "Purging unknown config_kv orphans");
    }
    for key in orphans {
        match store.delete_kv(&key).await {
            Ok(()) => tracing::debug!(key, "Purged unknown config_kv orphan"),
            Err(e) => {
                tracing::warn!(key, error=%e, "Failed to purge unknown config_kv orphan (transient, ignoring)");
            }
        }
    }
}

// Per-field persistence (settings-page autosave): one KV row or one routing row per settled
// field, never a whole-config rewrite. Every function here runs under [`persist_lock`],
// validates BEFORE writing (a failure leaves the DB and CONFIG untouched and propagates for
// inline display), applies side effects only when a value settles (never per keystroke) and
// returns the canonical persisted value (trimmed, `None` as `""`) for the caller to re-sync its
// snapshot. `wake_word_templates` is excluded everywhere: the voice pipeline
// (`persist_enrollment`) owns the key, so a GUI write would be a dual-writer race.

/// The canonical persisted value plus an optional non-fatal warning (e.g. an unreachable custom
/// endpoint that was saved anyway). `pub` because it flows through the public
/// [`crate::gui::settings::SettingsMessage`] persist-result variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistOutcome {
    pub value: String,
    pub warning: Option<String>,
}

/// Persist one settled field; `key` must be a `config_kv` key
/// ([`ConfigData::string_fields`]).
pub async fn persist_settled_string_field(key: &str, value: &str) -> Result<PersistOutcome> {
    let _guard = persist_lock().lock().await;
    let trimmed = value.trim().to_string();

    // Defense in depth: no control renders for this key, but refuse it
    // structurally so no future caller can write it here.
    if key == CONFIG_KEY_WAKE_WORD_TEMPLATES {
        return Ok(PersistOutcome {
            value: CONFIG.wake_word_templates().unwrap_or_default(),
            warning: None,
        });
    }

    let canonical = trimmed_or_none(&trimmed).unwrap_or_default();

    match key {
        // The active image model is per-user (`crate::users`) and is NOT
        // re-validated here: the image catalog is endpoint-keyed, so validating
        // committed models against the new endpoint could deadlock a switch to a
        // disjoint catalog (endpoint rejects until the model changes, model
        // rejects until the endpoint changes). Telegram picker taps validate
        // against the default endpoint instead, so a switch is two independently
        // valid steps — endpoint first, then model.
        CONFIG_KEY_PROVIDER_ENDPOINT => {
            let probe = probe_validate_write(key, &trimmed).await?;

            // Settling a default or empty endpoint drops the custom key row —
            // it is only meaningful while a custom endpoint is active. The
            // endpoint row is written FIRST, so a failed write leaves the key row
            // intact; a failed cascade leaves a stale row that is never read, so
            // the cleanup is best-effort and never fails the save.
            if trimmed.is_empty() || crate::config::is_default_endpoint(&trimmed) {
                let store = crate::config_db::store();
                match store.delete_kv(CONFIG_KEY_PROVIDER_ENDPOINT_KEY).await {
                    Ok(()) => {
                        let _ = CONFIG.set_string_field(CONFIG_KEY_PROVIDER_ENDPOINT_KEY, "");
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "failed to clear custom endpoint key row (harmless orphan)"
                        );
                    }
                }
            }

            // Best-effort and custom-only: an unreachable custom endpoint is
            // saved anyway (with a warning) so a self-hosted server can be
            // configured before it is reachable. A revert to the default is NOT
            // probed, so a transient OpenRouter outage cannot warn on it.
            let effective = effective_chat_endpoint(&probe);
            let warning = if crate::config::is_custom_endpoint(&effective) {
                match crate::providers::warmup_provider_from_config(&probe).await {
                    Ok(()) => None,
                    Err(e) => {
                        let msg = format!("Saved, but {effective} is unreachable right now: {e:#}");
                        tracing::warn!(key, endpoint = %effective, "Provider warmup failed (non-fatal, value saved): {e:#}");
                        Some(msg)
                    }
                }
            } else {
                None
            };

            // The only warmup this save performs — recreate_all must not fire a
            // second background one for the same endpoint.
            crate::providers::recreate_all(&CONFIG.snapshot(), false);
            return Ok(PersistOutcome {
                value: canonical,
                warning,
            });
        }
        // No warn-and-save on key fields: warmup never validates keys (a GET ignoring auth
        // status), so these saves only validate + write and let `recreate_all` rebuild the
        // provider. That keeps the custom-endpoint warning channel off the key fields.
        CONFIG_KEY_PROVIDER_KEY | CONFIG_KEY_PROVIDER_ENDPOINT_KEY => {
            let _ = probe_validate_write(key, &trimmed).await?;
            crate::providers::recreate_all(&CONFIG.snapshot(), true);
            return Ok(PersistOutcome {
                value: canonical,
                warning: None,
            });
        }
        // Hot-reloads the listener after the write.
        CONFIG_KEY_TELEGRAM_BOT_TOKEN => {
            let old_token = CONFIG.telegram_bot_token();
            let new_token = trimmed_or_none(&trimmed);
            if new_token != old_token
                && let Some(ref token) = new_token
            {
                crate::channels::telegram::TelegramChannel::validate_token(token).await?;
            }
            write_kv_and_update_config(CONFIG_KEY_TELEGRAM_BOT_TOKEN, &trimmed).await?;
            let persisted = CONFIG.telegram_bot_token();
            if persisted != old_token
                && let Err(e) =
                    crate::channels::telegram::restart_telegram_listener(persisted.as_deref())
            {
                // The token is written and takes effect at the next start; only
                // the hot-reload failed, so the save stands. No warning: that slot
                // belongs to the endpoint arm.
                tracing::warn!(
                    key,
                    "Telegram listener restart failed (non-fatal, value saved): {e:#}"
                );
            }
        }
        // Everything else is read dynamically at use time — persist only.
        _ => {
            write_kv_and_update_config(key, &trimmed).await?;
        }
    }

    Ok(PersistOutcome {
        value: canonical,
        warning: None,
    })
}

/// Probe→validate→write for the provider arms: snapshot the live CONFIG, apply
/// `key = trimmed`, normalize, validate, then write the KV row and mirror it into
/// CONFIG. Returns the probe — the live config with only this field applied, so
/// concurrent changes to other fields are never clobbered.
///
/// # Precondition
/// The caller holds [`persist_lock`]; this helper does not acquire it.
///
/// Provider arms only — telegram/image_model/others validate differently. The
/// probe is consumed only by the endpoint arm (for warmup); the key arm
/// discards it via `let _ = …`.
async fn probe_validate_write(key: &str, trimmed: &str) -> Result<ConfigData> {
    let mut probe = CONFIG.snapshot();
    let _ = probe.set_string_field(key, trimmed);
    probe.normalize();
    validate_config(&probe)?;
    write_kv_and_update_config(key, trimmed).await?;
    Ok(probe)
}

/// Persist a settled routing `provider_order` (`""` clears it), returning the
/// canonical value. Routing applies only to the manager and worker slots.
pub async fn persist_settled_routing_order(model: &str, order: &str) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let order = trimmed_or_none(order);
    save_routing_row(model, order).await
}

/// Write a single routing row (UPSERT or DELETE-if-empty) and mirror it into
/// the in-memory CONFIG. Caller holds [`persist_lock`].
async fn save_routing_row(model: &str, order: Option<String>) -> Result<String> {
    let store = crate::config_db::store();
    store.save_model_routing(model, order.as_deref()).await?;
    CONFIG.set_model_routing_row(model, order.clone());
    Ok(order.unwrap_or_default())
}

/// Write a single `config_kv` row (delete when the value is empty after
/// trimming) and mirror it into the in-memory CONFIG. Caller holds
/// [`persist_lock`].
async fn write_kv_and_update_config(key: &str, trimmed: &str) -> Result<()> {
    // Reject unknown keys BEFORE touching the DB: `set_string_field` is lenient
    // (returns `false`, CONFIG untouched), so writing the row anyway would orphan
    // a DB entry and silently diverge DB from CONFIG.
    if !ConfigData::STRUCT_FIELDS_DEFAULT.has_string_field(key) {
        anyhow::bail!("unknown config field: {key}");
    }
    let store = crate::config_db::store();
    if trimmed.is_empty() {
        store.delete_kv(key).await?;
    } else {
        store.set_kv(key, trimmed).await?;
    }
    let _ = CONFIG.set_string_field(key, trimmed);
    Ok(())
}

/// The onboarding state machine's durable value (lowercase in `config_kv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnboardingState {
    /// Fresh install, no provider configured yet.
    Init,
    /// Onboarding complete: the greeting was auto-sent and the Assistant drives
    /// onboarding from here.
    Finished,
}

impl OnboardingState {
    #[must_use]
    pub fn from_raw(s: Option<&str>) -> Self {
        match s {
            // Legacy `"welcomed"` (Support ran onboarding) reads as `Finished`:
            // nothing writes it anymore, and a user stuck there is deliberately
            // NOT migrated back to `Init`. Other unknown/corrupt values fail
            // toward `Init`.
            Some("welcomed" | "finished") => Self::Finished,
            _ => Self::Init,
        }
    }
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Finished => "finished",
        }
    }
}

/// Reject common misconfigurations before persisting.
///
/// # Precondition
/// [`ConfigData::normalize`] MUST have run: every `Option<String>` is assumed
/// trimmed, empty collapsed to `None`, for **every** field regardless of its
/// accessor annotation.
fn validate_config(config: &ConfigData) -> Result<()> {
    if let Some(ref ep) = config.provider_endpoint {
        if !is_http_url(ep) {
            anyhow::bail!(
                "Provider endpoint must be a valid URL starting with https:// or http://"
            );
        }
        // Empty-host rejection lives here, not in `is_http_url` (prefix-only by
        // design). Beyond it, nothing is rejected: `https://exa mple.com` passes
        // and is deferred to warmup — do not over-tighten. IPv6 literals and
        // userinfo are out of scope. The authority is trimmed twice (overall
        // `normalize` + here) to catch whitespace-only hosts like
        // `https://   /v1`.
        let rest = ep
            .strip_prefix("https://")
            .or_else(|| ep.strip_prefix("http://"))
            .unwrap();
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = rest[..auth_end].trim();
        if authority.is_empty() || authority.starts_with(':') {
            anyhow::bail!(
                "Provider endpoint must be a valid URL starting with https:// or http://"
            );
        }
    }

    if let Some(ref key) = config.provider_key
        && key.contains("...")
    {
        anyhow::bail!("Provider key is still the placeholder value — please set a real key");
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn model_routing(model: &str, provider_order: Option<&str>) -> ModelRouting {
    ModelRouting {
        model: model.into(),
        provider_order: provider_order.map(String::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A typo in the pinned default fails here, before any request is built.
    #[test]
    fn default_model_identifier_is_pinned() {
        let reload = ConfigReload::const_new();
        assert_eq!(
            reload.manager_model(),
            "deepseek/deepseek-v4.1-flash-1017",
            "a fresh manager slot must resolve to the pinned default"
        );
        assert_eq!(
            reload.worker_model(),
            "deepseek/deepseek-v4.1-flash-1017",
            "a fresh worker slot must resolve to the pinned default"
        );
        assert_eq!(
            default_model_routing(DEFAULT_CHAT_MODEL),
            Some("deepseek"),
            "the default must keep the DeepSeek provider pin"
        );
    }

    #[test]
    fn onboarding_from_raw_maps_legacy_welcomed_to_finished() {
        assert_eq!(
            OnboardingState::from_raw(Some("welcomed")),
            OnboardingState::Finished
        );
        assert_eq!(
            OnboardingState::from_raw(Some("finished")),
            OnboardingState::Finished
        );
        assert_eq!(
            OnboardingState::from_raw(Some("garbage")),
            OnboardingState::Init
        );
        assert_eq!(OnboardingState::from_raw(None), OnboardingState::Init);
    }

    /// Every accessor pattern through the ONE reload singleton, plus the role slots: the manager
    /// slot serves every role that plans and the worker slot every role that does the work, so a
    /// mis-wired role is caught here rather than by a run. The setter only stores — `normalize`
    /// collapses — and an unknown key is refused, not silently dropped.
    #[test]
    #[expect(clippy::too_many_lines)] // one section per accessor pattern the macro generates
    fn config_accessors_resolve_defaults_and_override_them() {
        let reload = ConfigReload::const_new();

        assert_eq!(reload.provider_key(), None, "unset provider_key is None");
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(config.set_string_field("provider_key", "sk-test"));
        reload.swap(config);
        assert_eq!(reload.provider_key(), Some("sk-test".to_string()));

        reload.swap(ConfigData::STRUCT_FIELDS_DEFAULT);
        assert_eq!(
            reload.manager_model(),
            DEFAULT_MANAGER_MODEL,
            "unset manager_model falls back to default"
        );
        assert_eq!(
            reload.worker_model(),
            DEFAULT_WORKER_MODEL,
            "unset worker_model falls back to default"
        );

        let mut custom_cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(custom_cfg.set_string_field("provider_endpoint", "https://custom.example/v1"));
        reload.swap(custom_cfg);
        assert_eq!(
            reload.provider_endpoint(),
            "https://custom.example/v1",
            "or field honors a persisted custom value"
        );
        reload.swap(ConfigData::STRUCT_FIELDS_DEFAULT);
        assert_eq!(reload.provider_endpoint(), DEFAULT_PROVIDER_ENDPOINT);

        let mut empty = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(empty.set_string_field("provider_key", ""));
        reload.swap(empty);
        assert_eq!(
            reload.provider_key(),
            None,
            "empty string is collapsed to None"
        );

        reload.swap(ConfigData::STRUCT_FIELDS_DEFAULT);
        assert_eq!(
            reload.image_gen_models(),
            vec![DEFAULT_IMAGE_GEN_MODEL.to_string()],
            "unset image_gen_models falls back to default"
        );
        // The raw accessor injects no default, so a caller cannot mistake the
        // setter's fallback for a chosen model.
        assert!(
            reload.image_gen_models_raw().is_empty(),
            "unset image_gen_models_raw must be empty"
        );

        let mut list_config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(list_config.set_string_field("image_gen_models", "model-a\nmodel-b\nmodel-c"));
        reload.swap(list_config);
        assert_eq!(
            reload.image_gen_models(),
            vec!["model-a", "model-b", "model-c"]
        );
        assert_eq!(
            reload.image_gen_models_raw(),
            vec!["model-a", "model-b", "model-c"],
            "raw accessor returns exactly the configured entries, in order"
        );

        let mut raw = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(raw.set_string_field("provider_key", ""));
        assert_eq!(
            raw.get_string_field("provider_key"),
            Some(""),
            "empty string stored as-is by set_string_field"
        );
        let _ = raw.set_string_field("provider_key", "   ");
        assert_eq!(
            raw.get_string_field("provider_key"),
            Some("   "),
            "whitespace-only string stored as-is by set_string_field"
        );
        raw.normalize();
        assert!(
            raw.get_string_field("provider_key").is_none(),
            "normalize() collapses an empty/whitespace value to None"
        );
        assert!(
            !raw.set_string_field("nonexistent_key", "value"),
            "a key that is no field is refused"
        );

        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(config.set_string_field("manager_model", "manager-slot-model"));
        assert!(config.set_string_field("worker_model", "worker-slot-model"));
        reload.swap(config);

        for role in [
            Role::Manager,
            Role::Assistant,
            Role::Discovery,
            Role::Engineer,
            Role::Maintainer,
        ] {
            assert_eq!(reload.role_model(role), "manager-slot-model");
        }
        for role in [
            Role::Analyst,
            Role::Worker,
            Role::Qa,
            Role::Reviewer,
            Role::Sanitation,
        ] {
            assert_eq!(reload.role_model(role), "worker-slot-model");
        }
    }

    /// Trivial variants of the default OpenRouter URL (trailing slash, padding,
    /// scheme/host case, `/chat/completions` suffix) must never count as custom.
    #[test]
    fn endpoint_normalization_default_vs_custom() {
        assert!(is_default_endpoint(DEFAULT_PROVIDER_ENDPOINT));
        assert!(is_default_endpoint("https://openrouter.ai/api/v1/"));
        assert!(is_default_endpoint("  https://openrouter.ai/api/v1  "));
        assert!(is_default_endpoint("HTTPS://OPENROUTER.AI/api/v1"));
        assert!(is_default_endpoint(
            "https://openrouter.ai/api/v1/chat/completions"
        ));
        assert!(is_default_endpoint(
            "https://openrouter.ai/api/v1/chat/completions/"
        ));
        assert!(is_custom_endpoint("http://localhost:8080/v1"));
        assert!(is_custom_endpoint("https://custom.example/v1"));
        assert!(!is_custom_endpoint(DEFAULT_PROVIDER_ENDPOINT));
    }

    /// The OpenRouter key is only ever used for the default endpoint; a keyless
    /// custom endpoint sends NO Authorization header.
    #[test]
    fn chat_credential_isolates_keys_per_endpoint() {
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert_eq!(chat_credential(&cfg).as_deref(), Some("sk-or"));

        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert!(cfg.set_string_field("provider_endpoint_key", "sk-custom"));
        assert_eq!(chat_credential(&cfg).as_deref(), Some("sk-custom"));

        // Keyless custom endpoint: the OR key must never be sent.
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert!(cfg.set_string_field("provider_endpoint_key", ""));
        assert_eq!(chat_credential(&cfg), None);

        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert_eq!(chat_credential(&cfg), None);

        assert_eq!(chat_credential(&ConfigData::STRUCT_FIELDS_DEFAULT), None);
    }

    /// Exactly one entry by model name: one upsert never rewrites a sibling and
    /// never adds a duplicate row.
    #[test]
    fn upsert_targets_one_entry_by_its_model() {
        let mut routings = vec![
            model_routing("test-router-a", Some("OpenAi")),
            model_routing("test-router-b", Some("Anthropic")),
        ];

        ModelRouting::upsert(&mut routings, "test-router-a", |mr| {
            mr.provider_order = Some("Google".into());
        });
        assert_eq!(routings.len(), 2, "no spurious push");
        assert_eq!(routings[0].provider_order, Some("Google".into()));
        assert_eq!(routings[1].provider_order, Some("Anthropic".into()));

        ModelRouting::upsert(&mut routings, "test-router-b", |mr| {
            mr.provider_order = Some("OpenAi".into());
        });
        assert_eq!(routings[0].provider_order, Some("Google".into()));
        assert_eq!(routings[1].provider_order, Some("OpenAi".into()));

        ModelRouting::upsert(&mut routings, "test-router-a", |mr| {
            mr.provider_order = None;
        });
        assert_eq!(routings.len(), 2);
        assert_eq!(
            routings[0].provider_order, None,
            "a field cleared to None is cleared"
        );

        ModelRouting::upsert(&mut routings, "test-router-c", |mr| {
            mr.provider_order = Some("OpenAi".into());
        });
        assert_eq!(routings.len(), 3);
        assert_eq!(routings[2].model, "test-router-c");
        assert_eq!(routings[2].provider_order, Some("OpenAi".into()));
    }

    /// Regression for an ordering bug: `validate_config` (untrimmed
    /// `starts_with`) once ran *before* `normalize`. What validation REFUSES is
    /// pinned through the settle-and-persist path
    /// (`config_db::tests::test_persist_settled_rejects_invalid_values_before_writing`),
    /// which also holds that nothing is written when it refuses.
    #[test]
    fn validate_config_accepts_a_url_however_it_was_padded() {
        let mut config = ConfigData {
            provider_endpoint: Some("https://openrouter.ai/api/v1".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        validate_config(&config).unwrap();

        let mut config = ConfigData {
            provider_endpoint: Some("  https://openrouter.ai/api/v1   ".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        validate_config(&config).unwrap();
    }

    /// Every key carrying a persist side effect (or the structural guard) must
    /// still be a real config field — the set a rename in
    /// `string_config_fields!` must keep in sync.
    #[test]
    fn side_effect_config_keys_are_real_fields() {
        let known: Vec<&'static str> = ConfigData::STRUCT_FIELDS_DEFAULT
            .string_fields()
            .iter()
            .map(|(k, _)| *k)
            .collect();
        for key in [
            CONFIG_KEY_PROVIDER_ENDPOINT,
            CONFIG_KEY_PROVIDER_KEY,
            CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
            CONFIG_KEY_TELEGRAM_BOT_TOKEN,
            CONFIG_KEY_WAKE_WORD_TEMPLATES,
        ] {
            assert!(
                known.contains(&key),
                "side-effect key '{key}' must be a real config field"
            );
        }
    }

    /// Drives the real boot chain rather than hand-picked `fresh` values: the
    /// probe, the boot flag, the store open that creates `core.db`, then the
    /// flag consumption + seed.
    ///
    /// Regression guard for the wrong-path probe: opening the config store must
    /// flip the discriminator to "not fresh" — probing a top-level
    /// `<root>/config.db` would still report "fresh" and re-seed every existing
    /// install on every boot.
    #[tokio::test]
    async fn fresh_config_db_seeds_defaults_only_when_new() {
        let fresh_root = tempfile::TempDir::new().unwrap();
        let fresh = config_db_is_fresh(fresh_root.path());
        assert!(
            fresh,
            "a storage root with no config store file must be classified fresh"
        );
        CONFIG_DB_FRESH_AT_BOOT.store(fresh, Ordering::Release);

        // Opening the store creates the exact file the probe must check.
        let fresh_store = crate::config_db::ConfigStore::open(fresh_root.path())
            .await
            .unwrap();
        assert!(
            !config_db_is_fresh(fresh_root.path()),
            "an existing config store file must not be classified fresh"
        );

        seed_fresh_install_defaults_from_flag(&fresh_store)
            .await
            .unwrap();
        assert!(
            !CONFIG_DB_FRESH_AT_BOOT.load(Ordering::Acquire),
            "the boot discriminator must be consumed by the seed"
        );
        assert_eq!(
            fresh_store.get_all_kv().await.unwrap(),
            vec![
                (
                    "audio_transcription_use_local".to_string(),
                    "false".to_string()
                ),
                (
                    "image_gen_models".to_string(),
                    "google/gemini-nano-banana-2.1\nmicrosoft/mai-image-2.6\nopenai/gpt-image-2.5-sunburst\nqwen/qwen-image-3-pro"
                        .to_string()
                ),
                (
                    "video_models".to_string(),
                    "google/gemini-omni-1.2-flash\nbytedance/seedance-2.0-mini\nminimax/hailuo-3"
                        .to_string()
                ),
            ],
            "a fresh config database must hold catalog 69's transcription row plus the fresh-install defaults"
        );
        // Manager and worker share one default, so the passes collapse into one row.
        assert_eq!(
            fresh_store.get_all_model_routings().await.unwrap(),
            vec![model_routing(DEFAULT_MANAGER_MODEL, Some("deepseek"))],
            "a fresh config database must seed routing rows for default models \
             with a known provider (others get none)"
        );

        let (existing_store, existing_dir) =
            crate::open_test_store!(crate::config_db::ConfigStore, "config");
        let fresh_existing = config_db_is_fresh(existing_dir.path());
        assert!(
            !fresh_existing,
            "an existing install must not be classified fresh"
        );
        CONFIG_DB_FRESH_AT_BOOT.store(fresh_existing, Ordering::Release);
        seed_fresh_install_defaults_from_flag(&existing_store)
            .await
            .unwrap();
        assert_eq!(
            existing_store.get_all_kv().await.unwrap(),
            vec![(
                "audio_transcription_use_local".to_string(),
                "false".to_string()
            )],
            "an existing config database receives only catalog 69's transcription row; the seed adds nothing"
        );
        assert!(
            existing_store
                .get_all_model_routings()
                .await
                .unwrap()
                .is_empty(),
            "an existing config database must receive zero routing rows (no backfill)"
        );
    }

    /// Only exact full-string matches are rewritten, each seeding the new
    /// default's routing row; re-running is a no-op.
    #[tokio::test]
    async fn migrate_old_default_models_rewrites_every_previous_default() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        // A stale video-transcription row and an unrelated key must survive.
        store
            .set_kv("video_transcription_model", "qwen/qwen3.7-flash")
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_PROVIDER_KEY, "sk-example")
            .await
            .unwrap();

        // Every manager default of every earlier release, not just the last one.
        for old in [
            "deepseek/deepseek-v4.1-flash",
            "z-ai/glm-5.3-flash",
            "deepseek/deepseek-v4-flash-vision-exp",
            "deepseek/deepseek-v4-pro-0813",
        ] {
            store.set_kv(CONFIG_KEY_MANAGER_MODEL, old).await.unwrap();
            migrate_old_default_models(&store).await.unwrap();
            assert_eq!(
                store
                    .get_kv(CONFIG_KEY_MANAGER_MODEL)
                    .await
                    .unwrap()
                    .as_deref(),
                Some(DEFAULT_MANAGER_MODEL),
                "manager slot holding the retired default {old} must be migrated"
            );
        }
        for old in [
            "deepseek/deepseek-v4.1-flash",
            "deepseek/deepseek-v4-flash-vision-exp",
            "deepseek/deepseek-v4-flash-0731",
        ] {
            store.set_kv(CONFIG_KEY_WORKER_MODEL, old).await.unwrap();
            migrate_old_default_models(&store).await.unwrap();
            assert_eq!(
                store
                    .get_kv(CONFIG_KEY_WORKER_MODEL)
                    .await
                    .unwrap()
                    .as_deref(),
                Some(DEFAULT_WORKER_MODEL),
                "worker slot holding the retired default {old} must be migrated"
            );
        }

        assert_eq!(
            store
                .get_kv("video_transcription_model")
                .await
                .unwrap()
                .as_deref(),
            Some("qwen/qwen3.7-flash"),
            "migrate only rewrites manager/worker old-default rows"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_PROVIDER_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("sk-example"),
            "unrelated config keys must never be touched"
        );
        // Both slots share one default, so one shared routing row.
        assert_eq!(
            store.get_all_model_routings().await.unwrap(),
            vec![model_routing(DEFAULT_MANAGER_MODEL, Some("deepseek"))],
            "migrating a slot to the new default must upsert its routing row"
        );

        migrate_old_default_models(&store).await.unwrap();
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_MANAGER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some(DEFAULT_MANAGER_MODEL),
            "re-running the migration must be a no-op"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_WORKER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some(DEFAULT_WORKER_MODEL),
            "re-running the migration must be a no-op"
        );
        assert_eq!(
            store.get_all_model_routings().await.unwrap(),
            vec![model_routing(DEFAULT_MANAGER_MODEL, Some("deepseek"))],
            "re-running the migration must not add or drop the migrated routing row"
        );
    }

    /// Regression for the retired `video_transcription_model` slot: the orphan
    /// purge must delete a pre-hardcode row so a stale value cannot resurface if
    /// the key returns. Known keys and shared namespaces survive.
    #[tokio::test]
    async fn purge_unknown_kv_orphans_deletes_stale_video_transcription_row() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");
        store
            .set_kv("video_transcription_model", "qwen/qwen3.7-flash")
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_PROVIDER_KEY, "sk-example")
            .await
            .unwrap();
        store
            .set_kv(
                crate::workspace::NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY,
                "2026-08-29",
            )
            .await
            .unwrap();
        store
            .set_kv(
                crate::temp::TEMP_CLEANUP_LAST_RUN_KV_KEY,
                "2026-08-29T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .set_kv(crate::temp::TEMP_CLEANUP_MODE_KV_KEY, "daily")
            .await
            .unwrap();
        store
            .set_kv(crate::self_update::UPDATE_LAST_RUN_VERSION_KV_KEY, "1.2.3")
            .await
            .unwrap();
        store
            .set_kv(crate::self_update::UPDATE_PENDING_CHANGELOG_KV_KEY, "{}")
            .await
            .unwrap();
        store
            .set_kv(crate::disk_notice::LADDER_KV_KEY, "2@2026-10-28T00:00:00Z")
            .await
            .unwrap();

        let kvs = store.get_all_kv().await.unwrap();
        purge_unknown_kv_orphans(&store, &kvs).await;

        assert_eq!(
            store.get_kv("video_transcription_model").await.unwrap(),
            None,
            "the stale video-transcription row must be purged"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_PROVIDER_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("sk-example"),
            "known config keys must survive the purge"
        );
        assert_eq!(
            store
                .get_kv(crate::workspace::NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("2026-08-29"),
            "preserved shared namespaces must survive the purge"
        );
        assert_eq!(
            store
                .get_kv(crate::temp::TEMP_CLEANUP_LAST_RUN_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("2026-08-29T00:00:00Z"),
            "temp-cleanup cadence state must survive the purge"
        );
        assert_eq!(
            store
                .get_kv(crate::temp::TEMP_CLEANUP_MODE_KV_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("daily"),
            "temp-cleanup mode must survive the purge"
        );
        for key in [
            crate::self_update::UPDATE_LAST_RUN_VERSION_KV_KEY,
            crate::self_update::UPDATE_PENDING_CHANGELOG_KV_KEY,
            crate::disk_notice::LADDER_KV_KEY,
        ] {
            assert!(
                store.get_kv(key).await.unwrap().is_some(),
                "state shared with another subsystem must survive the purge: {key}"
            );
        }
    }

    /// A no-op when a slot holds the target default, a genuine override, or a
    /// custom-endpoint installation's own settings: a historical default of the
    /// *other* slot is not this slot's default.
    #[tokio::test]
    async fn migrate_old_default_models_never_touches_a_genuine_choice() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        // A custom-endpoint installation keeps its endpoint, key and models.
        let installation = [
            (CONFIG_KEY_PROVIDER_ENDPOINT, "https://llm.example/v1"),
            (CONFIG_KEY_PROVIDER_ENDPOINT_KEY, "sk-local"),
            (CONFIG_KEY_MANAGER_MODEL, "my-stack/manager"),
            (CONFIG_KEY_WORKER_MODEL, "my-stack/worker"),
        ];
        for (key, value) in installation {
            store.set_kv(key, value).await.unwrap();
        }

        migrate_old_default_models(&store).await.unwrap();

        for (key, value) in installation {
            assert_eq!(
                store.get_kv(key).await.unwrap().as_deref(),
                Some(value),
                "{key} must survive the default-model migration"
            );
        }
        assert!(
            store.get_all_model_routings().await.unwrap().is_empty(),
            "nothing was migrated, so no routing row may be written"
        );

        // Already at the target default.
        store
            .set_kv(CONFIG_KEY_MANAGER_MODEL, DEFAULT_MANAGER_MODEL)
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_WORKER_MODEL, "user/private-model")
            .await
            .unwrap();

        migrate_old_default_models(&store).await.unwrap();

        assert_eq!(
            store
                .get_kv(CONFIG_KEY_MANAGER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some(DEFAULT_MANAGER_MODEL),
            "a slot already at the target must be untouched"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_WORKER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some("user/private-model"),
            "a genuine user override must be preserved"
        );
        assert!(
            store.get_all_model_routings().await.unwrap().is_empty(),
            "no routing writes may happen when nothing was migrated"
        );

        // A historical default of the *other* slot must survive as a pick.
        store
            .set_kv(CONFIG_KEY_MANAGER_MODEL, "deepseek/deepseek-v4-flash-0731")
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_WORKER_MODEL, "z-ai/glm-5.3-flash")
            .await
            .unwrap();

        migrate_old_default_models(&store).await.unwrap();

        assert_eq!(
            store
                .get_kv(CONFIG_KEY_MANAGER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some("deepseek/deepseek-v4-flash-0731"),
            "a manager pick equal to a worker-only old default must be preserved"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_WORKER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some("z-ai/glm-5.3-flash"),
            "a worker pick equal to a manager-only old default must be preserved"
        );
        assert!(
            store.get_all_model_routings().await.unwrap().is_empty(),
            "no routing writes may happen when nothing was migrated"
        );
    }

    /// A stored override wins, else the model's default provider; a non-DeepSeek
    /// model stays auto-routed. Normalisation runs first, so a stored order is
    /// trimmed and a whitespace-only one is as unset as an empty one.
    #[test]
    fn model_routing_applies_the_default_provider_when_unset() {
        let config = ConfigReload::const_new();
        let mut data = ConfigData::STRUCT_FIELDS_DEFAULT;
        data.model_routings = vec![
            model_routing("deepseek/deepseek-v4.1-flash", Some("  GMICloud  ")),
            model_routing("deepseek/deepseek-v4-flash-0731", Some("")),
            model_routing("deepseek/deepseek-v4.1-pro", Some("   ")),
            model_routing("qwen/qwen3.8-flash", Some("  ")),
        ];
        data.normalize();
        config.swap(data);

        assert_eq!(
            config
                .model_routing("deepseek/deepseek-v4.1-flash")
                .provider_order
                .as_deref(),
            Some("GMICloud"),
            "an explicitly set provider must win, trimmed of its padding"
        );
        for model in [
            "deepseek/deepseek-v4.1-pro",
            "deepseek-ai/DeepSeek-V3",
            "deepseek/deepseek-v4-flash-0731",
        ] {
            assert_eq!(
                config.model_routing(model).provider_order.as_deref(),
                Some("deepseek"),
                "{model} has no explicit setting — an empty or whitespace-only order \
                 is no setting — and must fall back to the canonical DeepSeek slug"
            );
        }
        assert_eq!(
            config.model_routing("qwen/qwen3.8-flash").provider_order,
            None,
            "a non-DeepSeek model must stay auto-routed"
        );

        // The built-in default model carries its provider pin, so a fresh
        // install's manager requests are routed rather than left to the
        // provider's own choice.
        assert_eq!(
            default_model_routing(DEFAULT_CHAT_MODEL),
            Some("deepseek"),
            "the default must keep the DeepSeek provider pin"
        );
    }
}

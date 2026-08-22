//! Configuration system with three independent resolution chains.
//!
//! # Architecture overview
//!
//! The config system has a layered structure: hardcoded defaults in this module
//! provide the base values, and the [`ConfigReload`] singleton is then overlayed
//! with persisted values from the `config.db` Turso database (via
//! [`crate::config_db`]).
//!
//! At startup, [`load_or_init`] seeds `ConfigData::STRUCT_FIELDS_DEFAULT` into the
//! global [`CONFIG`]. Then [`reload_from_db`] overlays persisted values from the
//! three database tables on top of those defaults.
//!
//! # Resolution chains
//!
//! The configuration system has three independent resolution chains. They are
//! **independent** — a KV entry in Chain 1 cannot override a model slot in
//! Chain 2, and a slot entry in Chain 2 cannot override a per-model routing
//! in Chain 3. Each chain applies to different fields.
//!
//! ## Chain 1: KV-overridable string fields
//!
//! `config_kv` table → hardcoded default (`const` in this module)
//!
//! The fields listed in the `string_config_fields!` invocation
//! belong to this chain. Their accessor methods (generated on [`ConfigReload`])
//! each follow a per-field annotation:
//!
//! * `non_empty` — returns `Option<String>`, collapses empty/whitespace to `None`.
//! * `or(DEFAULT)` — returns `String`, falls back to a compile-time constant
//!   (e.g. `DEFAULT_PROVIDER_ENDPOINT`).
//! * `list_or(fallback = …, default = …)` — returns `Vec<String>`, parses a
//!   newline-separated list, falling back to a singular field then to a hardcoded
//!   default.
//!
//! At reload time [`reload_from_db`] loads key–value pairs from the `config_kv`
//! table (via [`crate::config_db::ConfigStore::get_all_kv`]) and applies them
//! through [`ConfigData::set_string_field`]. Any key absent from the table
//! remains `None`, and the accessor resolves the hardcoded fallback.
//!
//! Fields **not** in this chain (e.g. `model_routings`) have their own
//! dedicated table and reload path.
//!
//! ### Custom chat-completions endpoint
//!
//! `provider_endpoint` is an `or(DEFAULT_PROVIDER_ENDPOINT)` field: a
//! persisted value (a self-hosted OpenAI-compatible endpoint — Ollama,
//! LM Studio, llama.cpp, vLLM, LiteLLM) is honored at runtime, falling back
//! to OpenRouter when unset. `provider_endpoint_key` is the custom
//! endpoint's own optional API key (many self-hosted servers are keyless) —
//! it is only ever sent to the custom endpoint, never to OpenRouter. Media
//! features (image/video generation, catalogs, media transcription) always
//! use `DEFAULT_PROVIDER_ENDPOINT` + `provider_key` regardless of the
//! custom chat endpoint.
//!
//! ## Chain 2: Three model slots
//!
//! `config_kv` table → hardcoded slot default (`const` in this module)
//!
//! The three model slots — `manager_model`, `worker_model`,
//! `video_transcription_model` — are ordinary Chain 1 fields.
//! [`ConfigReload::role_model`] maps every role onto exactly one slot:
//!
//! > `Role::Manager` | `Role::Assistant` → manager slot; every other role
//! > (including Artist) → worker slot. The video-transcription slot backs
//! > only video transcription — no role uses it.
//!
//! Unset slots fall back to their `DEFAULT_*_MODEL` constant. Historical
//! per-role overrides are no longer used — legacy rows are inert ghosts if
//! present.
//!
//! ## Chain 3: Per-model provider routing
//!
//! `config_model_routing` table → `None` defaults
//!
//! Stored in [`ConfigData::model_routings`] as a [`Vec<ModelRouting>`][ModelRouting],
//! loaded at reload time from the `config_model_routing` table. Checked via
//! [`ConfigReload::model_routing`]. When no entry exists, the returned
//! [`ModelRouting`] has `provider_order` `None`. The provider layer (in
//! [`crate::providers`]) resolves this `None` value at request time when
//! building the OpenAI-compatible chat request.
//!
//! # Persistence layer
//!
//! The tables live in `config.db` and are managed by [`crate::config_db`]:
//!
//! | Table | Read | Write |
//! |---|---|---|
//! | `config_kv` | [`crate::config_db::ConfigStore::get_all_kv`] | [`crate::config_db::ConfigStore::set_kv`] |
//! | `config_model_routing` | [`crate::config_db::ConfigStore::get_all_model_routings`] | [`crate::config_db::ConfigStore::save_model_routing`] |
//!
//! # Orphaned database keys
//!
//! Rows in `config_kv` without a corresponding [`ConfigData`] field are
//! lazily purged on reload: [`reload_from_db`] logs each unknown key at
//! `debug`, then best-effort deletes garbage rows (debug on success, warn
//! on transient failure without failing boot) while the two intentional
//! shared namespaces `nightly_discovery_last_pass_at`
//! (`src/workspace.rs:1572`) and `telegram_role_pin:*`
//! (`src/channels/telegram.rs:515`) are left untouched. Downgrade
//! resurrection is therefore lost — under the previous ghost policy orphans
//! were retained and would re-appear on downgrade, now they are deleted on
//! first reboot. New orphans cannot be created via the settings path because
//! [`write_kv_and_update_config`] rejects unknown keys before any DB write.
//!
//! # See also
//!
//! * [`crate::config_db`] — database persistence for all three chains.
//! * [`crate::role`] — [`crate::role::RoleInfo`] definitions with per-role defaults.
//! * [`crate::providers::compatible`] — where `None` routing fields are resolved
//!   at the provider layer.

use crate::Role;
use crate::util::{UnwrapPoison, is_http_url};
use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use tokio::fs;

// ── Hardcoded defaults ───────────────────────────────────────────

pub(crate) const DEFAULT_PROVIDER_ENDPOINT: &str = "https://openrouter.ai/api/v1";

pub(crate) const DEFAULT_MANAGER_MODEL: &str = "deepseek/deepseek-v4-flash-vision-exp";
pub(crate) const DEFAULT_WORKER_MODEL: &str = "deepseek/deepseek-v4-flash-vision-exp";

// The previous compiled defaults for the manager/worker slots, kept only for
// the one-time migration in `migrate_old_default_models`.
const OLD_DEFAULT_MANAGER_MODEL: &str = "deepseek/deepseek-v4-pro-0813";
const OLD_DEFAULT_WORKER_MODEL: &str = "deepseek/deepseek-v4-flash-0731";
pub(crate) const DEFAULT_VIDEO_TRANSCRIPTION_MODEL: &str = "qwen/qwen3.7-flash";

const DEFAULT_IMAGE_GEN_MODEL: &str = "google/gemini-3.1-flash-image";
const DEFAULT_VIDEO_MODEL: &str = "minimax/hailuo-3";

/// Fresh-install seeded image-generation model list (newline-separated, in
/// picker order; the first entry is the active model). Mirrors the curated
/// set the live install ships.
const FRESH_INSTALL_IMAGE_GEN_MODELS: &str =
    "google/gemini-3.1-flash-image\nmicrosoft/mai-image-2.5\nqwen/qwen-image-3-pro";

/// Fresh-install seeded video-generation model list (newline-separated, in
/// picker order; the second entry is the active model). Mirrors the curated
/// set the live install ships.
const FRESH_INSTALL_VIDEO_MODELS: &str = "bytedance/seedance-2.0-mini\nminimax/hailuo-3";

pub(crate) const DEFAULT_TTS_LANGUAGE: &str = "na";

// ── Named config structs ───────────────────────────────────────────

/// A per-model provider routing rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRouting {
    pub model: String,
    pub provider_order: Option<String>,
}

impl ModelRouting {
    /// Find-or-push: update a subset of fields on an existing entry matching
    /// `model`, or push a new entry (all fields defaulted to `None`).
    ///
    /// Only the field(s) mutated inside `set_field` are touched — if the
    /// entry already exists its other fields are preserved unchanged.
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

// ── ConfigData — the reloadable inner config ─────────────────────

/// All runtime-configurable fields, protected by an [`RwLock`] in [`ConfigReload`].
///
/// Every accessor returns an owned [`String`] (or [`Option<String>`]) because the
/// lock guard cannot escape the accessor's scope. Clone is cheap — these are
/// short strings read infrequently.
///
/// ## Adding a new persisted `Option<String>` field
///
/// Follow all three steps in order:
///
/// 1. **Field declaration** — add the field here on [`ConfigData`].
/// 2. **Macro** — add the field name to the `string_config_fields!` invocation in this file.
///    The macro generates `ConfigData::STRUCT_FIELDS_DEFAULT` (used by [`ConfigReload::const_new`]),
///    [`ConfigData::string_fields()`], [`ConfigData::set_string_field()`], and — for every field —
///    a `pub const CONFIG_KEY_<FIELD>: &str` key constant from this list.  The compiler
///    enforces that every field on [`ConfigData`] is present in `STRUCT_FIELDS_DEFAULT`,
///    so forgetting this step is a compile error.  Use the emitted `CONFIG_KEY_<FIELD>` consts
///    at write sites / match arms instead of raw string literals — a rename then stays
///    compiler-tied at every use site.
/// 3. **Typed accessor** — automatically generated. The `string_config_fields!` macro
///    produces typed accessor methods on [`ConfigReload`] based on each field's
///    annotation (`non_empty`, `or(DEFAULT)`, or `list_or(...)`) — no manual
///    accessor code is needed.
///
/// ## All `Option<String>` fields must be in the macro
///
/// EVERY `Option<String>` field on [`ConfigData`] **must** appear in the
/// `string_config_fields!` invocation — the compiler enforces this through
/// `STRUCT_FIELDS_DEFAULT` (a `const Self { … }` that initialises every
/// field).  There is no such thing as a "transient" `Option<String>` field
/// that lives outside the macro.
///
/// Fields that should NOT be persisted as config KV pairs (runtime-only
/// caches, reconstructed state) will still appear in `string_fields()`
/// and thus be written/read by the per-field persist paths / `reload_from_db`.
/// If you truly need an unpersisted value, use a different type or a
/// separate data structure — not an `Option<String>` on [`ConfigData`].
///
/// ## UX asymmetry warning
///
/// The GUI Settings page reads [`ConfigData`] directly via [`ConfigReload::snapshot`]
/// (all fields).  But the per-field persist functions persist fields **only** through
/// [`ConfigData::string_fields`], which is macro-generated.  A field missing from
/// the macro would appear editable in the GUI but silently discard its value on
/// every save.  The compiler guard on `ConfigData::STRUCT_FIELDS_DEFAULT` prevents this.
#[derive(Debug, Clone)]
pub struct ConfigData {
    /// API key for the LLM provider.
    pub provider_key: Option<String>,
    /// Base URL for the OpenAI-compatible LLM provider.
    pub provider_endpoint: Option<String>,
    /// API key for the custom chat-completions endpoint (optional — many
    /// self-hosted servers are keyless). Only ever sent to the custom endpoint,
    /// never to OpenRouter.
    pub provider_endpoint_key: Option<String>,
    /// Model slot for the Manager role.
    pub manager_model: Option<String>,
    /// Model slot for all worker roles (Engineer, Analyst, Coder, QA,
    /// Reviewer, Discovery, Maintainer, Sanitation).
    pub worker_model: Option<String>,
    /// Model slot for video transcription (no role uses it).
    pub video_transcription_model: Option<String>,
    /// Image generation model.
    pub image_gen_model: Option<String>,
    /// Newline-separated list of available image generation models (for selection UI).
    pub image_gen_models: Option<String>,
    /// Video model — shared by the video_gen and video_edit tools.
    pub video_model: Option<String>,
    /// Newline-separated list of available video models (for selection UI).
    pub video_models: Option<String>,
    /// Firecrawl API key for web search.
    pub firecrawl_key: Option<String>,
    /// Exa API key for web search (alternative to Firecrawl).
    pub exa_key: Option<String>,
    /// Web search provider selection: "firecrawl" or "exa" (case-insensitive).
    /// When `None`, auto-selects based on which keys are configured (Firecrawl wins on tie).
    pub web_search_provider: Option<String>,
    /// Telegram Bot API token (hot-reloaded on save).
    pub telegram_bot_token: Option<String>,
    /// Enable local Qwen3-ASR audio transcription.
    ///
    /// When `true` (default) and the model is cached or can be downloaded, audio
    /// transcription runs fully locally via the `qwen-asr` crate with Qwen3-ASR-0.6B.
    /// Audio never leaves the machine.
    ///
    /// Set to `"false"` to disable audio transcription entirely — audio markers
    /// are replaced with just the icon combo and the temp file is deleted, so
    /// voice messages are not recoverable.
    pub audio_transcription_use_local: Option<String>,
    /// Enable voice assistant (wake word detection and voice commands).
    /// Set to `"true"` to enable voice mode.
    pub voice_enabled: Option<String>,
    /// Enable text-to-speech for agent responses (default: `"false"`).
    /// Set to `"true"` to enable. When enabled, agent responses are spoken
    /// aloud via the OS-native audio player when the responding role matches
    /// the user's active GUI role.
    pub tts_enabled: Option<String>,
    /// Language tag for TTS synthesis (default: `"na"` — language-agnostic).
    /// Supported codes: en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr,
    /// hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk,
    /// vi, na.
    ///
    /// The default `"na"` works well for any language. Set to `"en"` for
    /// optimal English pronunciation, or to your language's code for better
    /// results in that language.
    pub tts_language: Option<String>,
    /// JSON-serialized wake word enrollment (v2 schema: prototype + calibration)
    /// for the voice assistant.  Owned exclusively by the voice pipeline.
    pub wake_word_templates: Option<String>,
    /// Per-model provider routing.
    pub model_routings: Vec<ModelRouting>,
}

// ── String config field mapping ──────────────────────────────────
//
// The four runtime sync items (`STRUCT_FIELDS_DEFAULT`, `string_fields()`,
// `set_string_field()`, `normalize_string_fields()`) plus the typed accessors
// on [`ConfigReload`]
// are all generated from a single annotated field-name declaration by the
// `string_config_fields!` macro — adding or removing a field in the
// macro invocation updates all items automatically, eliminating the
// entire class of sync bugs.
//
// The macro additionally emits a `pub const CONFIG_KEY_<FIELD>` constant per
// field, alongside the sync items, so hand-written persist paths reference
// keys by name instead of raw literals.
//
// ══ Structural protection ═════════════════════════════════════════
//
// `STRUCT_FIELDS_DEFAULT` is a `const Self { … }` that initialises every
// [`ConfigData`] field.  The compiler requires every field to be present,
// so adding a field to [`ConfigData`] without adding it to the macro is
// a **compile error**.  This eliminates the silent-drift class entirely
// — no manual count constants or runtime tests needed.
//
// ══ UX asymmetry ═══════════════════════════════════════════════════
//
// The GUI Settings page reads [`ConfigData`] via [`ConfigReload::snapshot`],
// which clones every struct field directly.  But the per-field persist
// functions persist **only** through [`ConfigData::string_fields`]
// (macro-generated).  A field missing from the macro would appear editable
// in the UI but silently discard on persist.  The compiler guard above
// prevents this.
//
// ══ Per-field accessor patterns ═════════════════════════════════════
//
// Each field is annotated with one of three patterns:
//
// * `non_empty` — returns `Option<String>`, collapses empty/whitespace to `None`.
// * `or(DEFAULT)` — returns `String`, falls back to the given default constant.
// * `list_or(fallback = <field>, default = <const>)` — returns `Vec<String>`,
//   parses a newline-separated list, falls back to the named field then
//   the default constant.
//
// Generated accessors live on `impl ConfigReload`, created by `string_config_fields!`.

/// Generate the runtime sync methods `string_fields()` and `set_string_field()`,
/// the const `STRUCT_FIELDS_DEFAULT`, **and** the typed accessors on [`ConfigReload`]
/// — all from a single annotated list of `Option<String>` field names.
///
/// Each field is declared as `$field [$annotation]` where `$annotation` is one of
/// `non_empty`, `or($default)`, or `list_or(fallback = $fallback, default = $default)`.
///
/// All generated items are guaranteed to stay synchronised because they expand
/// from the same source.
///
/// The macro also emits one `pub const CONFIG_KEY_<FIELD>: &str` per field
/// (name pasted from the field token, value `stringify!` of the same token),
/// so write sites and match arms can reference the key without raw literals —
/// a rename in the invocation changes the constant and its value together.
///
/// ## Structural drift protection
///
/// The generated [`ConfigData::STRUCT_FIELDS_DEFAULT`] is a `const` value that
/// initialises **every** field on [`ConfigData`] — both the listed `Option<String>`
/// fields and the `Vec` fields.  Because `Self { ... }` in a const requires all
/// fields, the **compiler** catches a struct–macro mismatch: adding a field to
/// [`ConfigData`] without adding it to the macro invocation produces a compile
/// error.  This eliminates the entire class of silent-drift bugs without manual
/// count constants or runtime tests.
macro_rules! string_config_fields {
    // ── Entry point: parse annotated field list ─────────────────
    (
        $(
            $field:ident [ $($annotation:tt)* ]
        ),* $(,)?
    ) => {
        // ── Config-key constants — single source of truth ──────
        //
        // Each field emits a `CONFIG_KEY_<FIELD>` const: the name is pasted
        // from the field token (`:upper`), the value `stringify!`s the same
        // token, so a rename changes both together and stale use sites fail
        // to compile instead of silently dropping their side effect.
        ::paste::paste! {
            $(
                pub const [<CONFIG_KEY_ $field:upper>]: &str = stringify!($field);
            )*
        }

        impl ConfigData {
            /// Default-initialised [`ConfigData`] with all `Option<String>` fields
            /// set to `None` and `Vec` fields empty.
            ///
            /// Used by [`ConfigReload::const_new`] so that adding a new field to
            /// the struct **and** the macro invocation is the *only* step needed
            /// — the const automatically stays in sync.
            ///
            /// ## Compiler enforcement
            ///
            /// Because this is a `const Self { … }`, the compiler requires every
            /// field on [`ConfigData`] to be present.  Adding a field to the
            /// struct without adding it here (via the macro) is a **compile
            /// error** — the entire silent-drift class is caught before a test
            /// ever runs.
            pub(crate) const STRUCT_FIELDS_DEFAULT: Self = Self {
                $($field: None,)*
                model_routings: Vec::new(),
            };

            /// Return all string-valued config fields as (db_key, current_value) pairs.
            ///
            /// `model_routings` is **not** included here — it lives in a
            /// separate database table (`config_model_routing`).
            #[must_use]
            pub fn string_fields(&self) -> Vec<(&'static str, Option<&str>)> {
                vec![$((stringify!($field), self.$field.as_deref())),*]
            }

            /// Set a string field by its database key. Returns `true` if the key was
            /// recognised and the field was updated, `false` for unknown keys.
            ///
            /// The value is stored as-is without normalization — call [`Self::normalize`]
            /// before using the config to collapse empty/whitespace-only values to `None`.
            ///
            /// `model_routings` is **not** handled here — it lives in a separate
            /// database table (`config_model_routing`).
            #[must_use]
            pub fn set_string_field(&mut self, key: &str, value: &str) -> bool {
                match key {
                    $(stringify!($field) => self.$field = Some(value.to_owned()),)*
                    _ => return false,
                }
                true
            }

            /// Normalise all string fields in place: trim whitespace and collapse
            /// empty or whitespace-only values to `None`.
            ///
            /// Unlike [`set_string_field`], which stores values as-is, this is the
            /// canonical normalization point — callers that set individual fields
            /// should ensure [`Self::normalize`] is called before using the config.
            fn normalize_string_fields(&mut self) {
                $(self.$field = non_empty(self.$field.take());)*
            }
        }

        // ── Generate typed accessors on ConfigReload ────────────
        //
        // Both passes needed: `normalize()` writes canonical values so
        // the per-field persist paths' `!=` comparisons against current
        // CONFIG accessors see no spurious diffs; accessors below
        // re-normalise raw values. See `config_reload_accessors_roundtrip`.
        impl ConfigReload {
            $(
                string_config_fields!(@accessor $field $($annotation)*);
            )*
        }
    };

    // ── Accessor pattern: non_empty ─────────────────────────────
    //
    // Returns Option<String>, collapses empty/whitespace to None.
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

    // ── Accessor pattern: or(DEFAULT) ───────────────────────────
    //
    // Returns String, falls back to the given default constant.
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

    // ── Accessor pattern: list_or(fallback = <field>, default = <const>) ──
    //
    // Returns Vec<String>. Tries parsing `$field` as a newline-separated list.
    // If non-empty, returns the parsed entries. Otherwise falls back to the
    // named `$fallback` field, then to the hardcoded `$default` constant.
    (@accessor $field:ident list_or(fallback = $fallback:ident, default = $default:expr)) => {
        #[doc = concat!(
            "Returns the list of available `", stringify!($field), "`.",
            "\n\nIf unset or the parsed newline-separated list is empty,",
            " falls back to `", stringify!($fallback),
            "`, then to a built-in default."
        )]
        #[must_use]
        pub fn $field(&self) -> Vec<String> {
            let guard = self.read();
            resolve_list_or(
                guard.$field.as_deref(),
                guard.$fallback.clone(),
                $default,
            )
        }
    };
}

string_config_fields! {
    provider_key [non_empty],
    provider_endpoint [or(DEFAULT_PROVIDER_ENDPOINT)],
    provider_endpoint_key [non_empty],
    manager_model [or(DEFAULT_MANAGER_MODEL)],
    worker_model [or(DEFAULT_WORKER_MODEL)],
    video_transcription_model [or(DEFAULT_VIDEO_TRANSCRIPTION_MODEL)],
    image_gen_model [or(DEFAULT_IMAGE_GEN_MODEL)],
    image_gen_models [list_or(fallback = image_gen_model, default = DEFAULT_IMAGE_GEN_MODEL)],
    video_model [or(DEFAULT_VIDEO_MODEL)],
    video_models [list_or(fallback = video_model, default = DEFAULT_VIDEO_MODEL)],
    firecrawl_key [non_empty],
    exa_key [non_empty],
    web_search_provider [non_empty],
    telegram_bot_token [non_empty],
    audio_transcription_use_local [non_empty],
    voice_enabled [non_empty],
    tts_enabled [non_empty],
    tts_language [or(DEFAULT_TTS_LANGUAGE)],
    wake_word_templates [non_empty],
}

impl ConfigData {
    /// Normalise inner `Option<String>` fields of `Vec` entries in place:
    /// trim whitespace and collapse empty/whitespace-only values to `None`.
    ///
    /// This is the Vec-entry counterpart of [`normalize_string_fields()`] —
    /// the macro-generated method only touches top-level `Option<String>` fields,
    /// not the inner fields of [`ModelRouting`] entries.
    fn normalize_entries(&mut self) {
        for mr in &mut self.model_routings {
            mr.provider_order = non_empty(mr.provider_order.take());
        }
    }

    /// Apply canonical normalisation + sorting to the in-memory
    /// representation so it is consistent across all persistence paths.
    ///
    /// The sequence is:
    /// 1. Trim top-level `Option<String>` fields and collapse empty → `None`.
    /// 2. Trim inner fields on `Vec` entries (`[ModelRouting]`)
    ///    and collapse empty → `None`.
    /// 3. Sort `model_routings` by model name.
    ///
    /// Every caller that produces a newly-built [`ConfigData`] must call
    /// this before swapping into the global [`CONFIG`] so that the in-memory
    /// representation is the same regardless of which code path produced it.
    pub(crate) fn normalize(&mut self) {
        self.normalize_string_fields();
        self.normalize_entries();
        self.model_routings.sort_by(|a, b| a.model.cmp(&b.model));
    }
}

// ── Config value helpers ────────────────────────────────────────────

/// Trim a string and return `None` if empty or whitespace-only.
/// This is the canonical primitive for string trimming helpers.
#[must_use]
pub(crate) fn trimmed_or_none(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Parse a newline-separated string into a vector of non-empty, trimmed entries.
#[must_use]
pub(crate) fn parse_newline_list(s: &str) -> Vec<String> {
    s.split('\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Treat an empty or whitespace-only string as `None`.
/// The value is trimmed before being returned.
/// Delegates to [`trimmed_or_none`].
#[must_use]
pub(crate) fn non_empty(val: Option<String>) -> Option<String> {
    val.and_then(|s| trimmed_or_none(&s))
}

/// Resolve a value with a fallback: use `val` if non-empty (after trimming), else `fallback`.
#[must_use]
pub(crate) fn resolve_or(val: Option<String>, fallback: &str) -> String {
    non_empty(val).unwrap_or(fallback.to_string())
}

/// Parse a newline-separated list field, falling back to a singular field, then to a hardcoded
/// default.
///
/// If `list_field` is `Some` and contains at least one non-empty line (after trimming), the parsed
/// lines are returned as a `Vec<String>`. Otherwise a single-element vec containing the resolved
/// value of `fallback_field` (or `default_value`) is returned.
#[must_use]
pub(crate) fn resolve_list_or(
    list_field: Option<&str>,
    fallback_field: Option<String>,
    default_value: &str,
) -> Vec<String> {
    if let Some(raw) = list_field {
        let parsed = parse_newline_list(raw);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    vec![resolve_or(fallback_field, default_value)]
}

/// Normalize an endpoint URL for comparison: trim surrounding whitespace,
/// lowercase the scheme and host (not the path), strip trailing slashes, and
/// strip a trailing `/chat/completions` suffix. The default OpenRouter URL and
/// trivial variants (trailing slash, whitespace, scheme/host case,
/// chat-completions suffix) must never count as a custom endpoint.
#[must_use]
pub(crate) fn normalize_endpoint_url(url: &str) -> String {
    let t = url.trim().trim_end_matches('/');
    // Strip one trailing `/chat/completions` suffix (after slash-stripping the
    // suffix has no trailing slash). Handles a URL ending exactly in
    // `/chat/completions`.
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

/// Whether `endpoint` is the default OpenRouter chat-completions endpoint
/// (modulo trivial variants — see [`normalize_endpoint_url`]).
#[must_use]
pub(crate) fn is_default_endpoint(endpoint: &str) -> bool {
    normalize_endpoint_url(endpoint) == normalize_endpoint_url(DEFAULT_PROVIDER_ENDPOINT)
}

/// Whether `endpoint` is a genuinely non-default (custom) endpoint.
#[must_use]
pub(crate) fn is_custom_endpoint(endpoint: &str) -> bool {
    !is_default_endpoint(endpoint)
}

/// Whether a custom chat-completions endpoint is currently active in the
/// global CONFIG (default URL and trivial variants never count as custom).
#[must_use]
pub(crate) fn custom_endpoint_active() -> bool {
    is_custom_endpoint(&CONFIG.provider_endpoint())
}

/// Effective chat-completions endpoint for a config snapshot: the persisted
/// endpoint when set (custom), else the default.
#[must_use]
pub(crate) fn effective_chat_endpoint(config: &ConfigData) -> String {
    resolve_or(config.provider_endpoint.clone(), DEFAULT_PROVIDER_ENDPOINT)
}

/// Credential for chat-completions requests under `config`: the custom
/// endpoint's own key when a custom endpoint is active (None = keyless, no
/// Authorization header), otherwise the OpenRouter key. The OpenRouter key is
/// NEVER sent to a custom endpoint.
#[must_use]
pub(crate) fn chat_credential(config: &ConfigData) -> Option<String> {
    let ep = effective_chat_endpoint(config);
    if is_custom_endpoint(&ep) {
        non_empty(config.provider_endpoint_key.clone())
    } else {
        non_empty(config.provider_key.clone())
    }
}

/// Whether the LLM provider is configured: a non-empty OpenRouter key OR an
/// active custom endpoint (keyless custom endpoints included). Used by the
/// pipeline pickup gate and the GUI boot-to-Settings redirect.
#[must_use]
pub(crate) fn provider_configured() -> bool {
    custom_endpoint_active() || CONFIG.provider_key().is_some()
}

// ── ConfigReload — global singleton ──────────────────────────────

/// Global reloadable config singleton.
///
/// The `storage_root` is immutable after startup; all other fields live in an
/// `RwLock<ConfigData>` that can be atomically swapped at runtime.
pub static CONFIG: ConfigReload = ConfigReload::const_new();

/// Serializes all per-field config persistence (settings-page autosave).
///
/// Every settled-field persist runs inside this lock so that:
/// - read-modify-write sequences (routing rows, endpoint/key probes built
///   from the live config) never interleave with each other;
/// - side-effect settles (provider warmup + recreate, Telegram listener
///   restart) cannot race, which would let two full config rewrites with
///   stale snapshots clobber each other's freshly-written rows.
///
/// The lock is held across the entire persist — including the provider
/// warmup network call — so each settle validates and writes against the
/// latest committed state.
static CONFIG_PERSIST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn persist_lock() -> &'static tokio::sync::Mutex<()> {
    CONFIG_PERSIST_LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Fresh-install discriminator: set during [`load_or_init`] to
/// whether `config.db` did not exist on disk before the config store was
/// opened this boot. [`reload_from_db`] seeds the fresh-install defaults
/// into brand-new config databases only — existing databases receive zero
/// writes (hard constraint).
static CONFIG_DB_FRESH_AT_BOOT: AtomicBool = AtomicBool::new(false);

/// Reloadable configuration with atomic swap capability.
///
/// The inner [`ConfigData`] is protected by an [`RwLock`] so readers don't
/// block each other. Writes happen only during startup and GUI-driven config
/// saves.
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

    // ── Storage root (set once at startup, immutable thereafter) ─

    /// # Panics
    /// Panics if storage root has not been set.
    pub fn global_storage_root(&self) -> PathBuf {
        self.storage_root
            .get()
            .expect("CONFIG storage_root not initialized")
            .clone()
    }

    /// Like [`Self::global_storage_root`], but returns `None` instead of panicking if
    /// storage root has not been set yet. Useful for code paths that can tolerate
    /// an uninitialized config (e.g., graceful degradation in tests).
    #[must_use]
    pub fn try_storage_root(&self) -> Option<PathBuf> {
        self.storage_root.get().cloned()
    }

    pub(crate) fn set_storage_root(&self, root: PathBuf) {
        self.storage_root
            .set(root)
            .expect("CONFIG storage_root already set");
    }

    /// Like [`set_storage_root`], but returns `Err` instead of panicking if
    /// already set. Useful in test environments where the root may have been
    /// set by a previously-running test.
    #[cfg(test)]
    pub(crate) fn try_set_storage_root(&self, root: PathBuf) -> std::result::Result<(), PathBuf> {
        self.storage_root.set(root)
    }

    // ── Snapshot access ─────────────────────────────────────────

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

    /// Update a single string config field in-memory.
    ///
    /// This is intentionally lightweight — it only mutates the in-memory
    /// [`ConfigData`] without touching the database or triggering provider
    /// warmup. Callers are responsible for persisting the change to the
    /// config DB separately (e.g. via [`crate::config_db::ConfigStore::set_kv`]).
    ///
    /// Returns `true` if the key was recognised, `false` otherwise (unknown
    /// keys are silently ignored for forward compatibility).
    #[must_use]
    pub fn set_string_field(&self, key: &str, value: &str) -> bool {
        let mut guard = self.inner.write().unwrap_poison();
        guard.set_string_field(key, value)
    }

    /// Apply a single model-routing row to the in-memory config (find-or-push).
    ///
    /// `None` on `provider_order` removes the row (mirroring the
    /// DELETE-if-empty persistence path).
    pub(crate) fn set_model_routing_row(&self, model: &str, provider_order: Option<String>) {
        let mut guard = self.inner.write().unwrap_poison();
        if provider_order.is_none() {
            guard.model_routings.retain(|mr| mr.model != model);
        } else {
            ModelRouting::upsert(&mut guard.model_routings, model, |mr| {
                mr.provider_order = provider_order;
            });
        }
    }

    // ── Provider routing (per-model) ──────────────────────────

    /// Find the per-model routing row by model key, if one exists.
    ///
    /// Unlike [`Self::model_routing`] (which always returns a row, defaulting
    /// missing entries to all-`None` fields), this returns `None` when no row
    /// is configured — the settings page uses it to tell "no override" apart
    /// from "all-None override" when mirroring settled rows.
    pub(crate) fn model_routing_by_key(&self, model: &str) -> Option<ModelRouting> {
        let guard = self.read();
        guard
            .model_routings
            .iter()
            .find(|mr| mr.model == model)
            .cloned()
    }

    /// Look up the provider routing config for a given model.
    ///
    /// Returns a [`ModelRouting`] with the model field populated from the lookup
    /// parameter. When no routing is configured, all fields except `model` are `None`.
    #[must_use]
    pub fn model_routing(&self, model: &str) -> ModelRouting {
        self.model_routing_by_key(model)
            .unwrap_or_else(|| ModelRouting {
                model: model.to_string(),
                provider_order: None,
            })
    }

    // ── Role model resolution (three slots) ─────────────────────

    /// Resolve the configured model for a role from the three model slots.
    ///
    /// Manager and Assistant use the manager slot; every other role (including
    /// Artist) uses the worker slot. Unset slots fall back to their code
    /// default.
    #[must_use]
    pub fn role_model(&self, role: Role) -> String {
        match role {
            Role::Manager | Role::Assistant => self.manager_model(),
            _ => self.worker_model(),
        }
    }
}

// ── Startup / reload / save ──────────────────────────────────────

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

/// Load (or initialise) the config system.
///
/// 1. Resolves `~/.mahbot` as the storage root.
/// 2. Creates the directory if needed.
/// 3. Seeds runtime config with hardcoded defaults.
/// 4. Stores the result in the global [`CONFIG`] singleton.
///
/// The caller must subsequently call [`reload_from_db`] to load any
/// persisted configuration from `config.db`. Providers must be
/// initialised **after** `reload_from_db` so API keys and model
/// settings take effect.
pub async fn load_or_init() -> Result<()> {
    let mahbot_dir = default_config_dir()?;
    fs::create_dir_all(&mahbot_dir)
        .await
        .context("Failed to create config directory")?;

    CONFIG.set_storage_root(mahbot_dir.clone());

    // Fresh-install discriminator: capture whether the config store file exists
    // BEFORE any store open creates it, so reload_from_db() can seed
    // fresh-install defaults into brand-new databases only. The
    // probe must check the exact path the store open uses (`<root>/db/config.db`
    // — see [`crate::turso::store_db_path`]); probing any other location would
    // classify every existing install as fresh and re-seed it on each boot,
    // violating the zero-write hard constraint.
    CONFIG_DB_FRESH_AT_BOOT.store(config_db_is_fresh(&mahbot_dir), Ordering::Release);

    // Start with hardcoded defaults — reload_from_db() will overlay
    // any persisted values from config.db (called later in bootstrap).
    CONFIG.swap(ConfigData::STRUCT_FIELDS_DEFAULT);

    tracing::info!(
        "Config system initialised (storage root: {}).",
        mahbot_dir.display()
    );
    Ok(())
}

/// Fresh-install discriminator: `true` when the config store
/// file does not yet exist at its real location (`<root>/db/config.db` — see
/// [`crate::turso::store_db_path`]).
///
/// Must be probed BEFORE any store open creates the file, and must check the
/// exact path the open uses (`init_all_stores` → `open_store` →
/// `store_db_path`). A probe of any other location would classify existing
/// installs as fresh and re-seed them on every boot, violating the zero-write
/// hard constraint.
fn config_db_is_fresh(mahbot_dir: &std::path::Path) -> bool {
    !crate::turso::store_db_path(mahbot_dir, "config").exists()
}

/// Seed the fresh-install defaults into a brand-new config database
///
/// A fresh install must not load or download any audio model until the user
/// enables a feature, so `audio_transcription_use_local` is seeded to
/// `"false"` (the existing reading semantics are unchanged: absence of the
/// row = enabled, `"false"` = disabled). It must also show populated model
/// pickers: the Settings GUI reads the raw snapshot fields
/// (`config.image_gen_models` / `config.image_gen_model`,
/// `config.video_models` / `config.video_model`), not the default-resolving
/// `list_or`/`or` accessors, so the image/video generation model lists and
/// their active selections are seeded too. `fresh` is the pre-open
/// file-existence discriminator captured in [`load_or_init`] — existing
/// databases receive zero writes.
async fn seed_fresh_install_defaults(
    fresh: bool,
    store: &crate::config_db::ConfigStore,
) -> Result<()> {
    if !fresh {
        return Ok(());
    }
    store
        .set_kv(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "false")
        .await?;
    store
        .set_kv(CONFIG_KEY_IMAGE_GEN_MODEL, DEFAULT_IMAGE_GEN_MODEL)
        .await?;
    store
        .set_kv(CONFIG_KEY_IMAGE_GEN_MODELS, FRESH_INSTALL_IMAGE_GEN_MODELS)
        .await?;
    store
        .set_kv(CONFIG_KEY_VIDEO_MODEL, DEFAULT_VIDEO_MODEL)
        .await?;
    store
        .set_kv(CONFIG_KEY_VIDEO_MODELS, FRESH_INSTALL_VIDEO_MODELS)
        .await?;
    // default model slots hosted by DeepSeek route through the
    // DeepSeek provider on fresh installs. The rows are explicit and editable
    // (clearing the field in the Settings UI returns the model to auto).
    // Derived from the default-model constants filtered by the lowercase
    // `deepseek/` prefix so the seed follows the defaults if they ever change;
    // every other default model gets no routing override (OpenRouter
    // auto-routes). The video-transcription default gets no routing seed —
    // routing applies only to the manager and worker model slots. Existing
    // installs receive zero writes — the `fresh` guard above already returned
    // for them.
    for default_model in [DEFAULT_MANAGER_MODEL, DEFAULT_WORKER_MODEL] {
        if default_model.starts_with("deepseek/") {
            store
                .save_model_routing(default_model, Some("DeepSeek"))
                .await?;
        }
    }
    tracing::info!(
        "Fresh config database: seeded fresh-install defaults (audio transcription off; image/video generation model sets; DeepSeek routing for deepseek/* default models)"
    );
    Ok(())
}

/// Consume the boot fresh-install discriminator and seed the fresh-install
/// defaults when it fired.
///
/// Mirrors the `load_or_init` → `reload_from_db` handoff exactly: the flag is
/// set from [`config_db_is_fresh`] during boot and consumed (reset) here, so
/// the seed lands once per fresh boot only. Kept as a separate function so the
/// flag-consumption path is testable without the global store.
async fn seed_fresh_install_defaults_from_flag(
    store: &crate::config_db::ConfigStore,
) -> Result<()> {
    let fresh = CONFIG_DB_FRESH_AT_BOOT.swap(false, Ordering::AcqRel);
    seed_fresh_install_defaults(fresh, store).await
}

/// One-time idempotent migration: rewrite persisted
/// `config_kv` rows that still hold the *previous* manager/worker default
/// model to the new vision-capable default. Matching is exact on the full
/// provider-prefixed string, so only rows the user never overrode are
/// touched; the video-transcription slot and every other key are untouched.
/// Safe to run on every boot — a no-op once no row matches.
async fn migrate_old_default_models(store: &crate::config_db::ConfigStore) -> Result<()> {
    let changed_manager = store
        .migrate_kv_if_equals(
            CONFIG_KEY_MANAGER_MODEL,
            OLD_DEFAULT_MANAGER_MODEL,
            DEFAULT_MANAGER_MODEL,
        )
        .await?;
    let changed_worker = store
        .migrate_kv_if_equals(
            CONFIG_KEY_WORKER_MODEL,
            OLD_DEFAULT_WORKER_MODEL,
            DEFAULT_WORKER_MODEL,
        )
        .await?;
    if changed_manager > 0 || changed_worker > 0 {
        tracing::info!(
            manager_rows = changed_manager,
            worker_rows = changed_worker,
            "Migrated stored old default models to the vision-capable default"
        );
    }
    Ok(())
}

/// Reload config from the `config.db` database, atomically swapping the
/// runtime config. Called at startup (after config_db init) to overlay
/// persisted settings on top of hardcoded defaults.
pub async fn reload_from_db() -> Result<()> {
    let store = crate::config_db::store();
    let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;

    // Fresh-install seed: a brand-new config
    // database gets the transcription-off default (so no audio model is
    // downloaded or loaded at boot) plus the image/video generation model
    // lists and active selections (so the Settings GUI pickers are populated).
    // Existing installs are never written (the flag is only set when config.db
    // did not exist before the store open).
    seed_fresh_install_defaults_from_flag(store).await?;

    // Rewrite persisted old-default manager/worker rows to the new default
    // before the overlay below, so this boot serves the migrated value.
    migrate_old_default_models(store).await?;

    let kvs = store.get_all_kv().await?;
    // Keep in sync with crate::workspace::NIGHTLY_DISCOVERY_LAST_PASS_KV_KEY and
    // crate::channels::telegram::ROLE_PIN_KV_PREFIX (private constants).
    let mut unknown_garbage_keys: Vec<String> = Vec::new();
    for (key, value) in &kvs {
        if !config.set_string_field(key, value) {
            tracing::debug!(key, "Unknown config key, ignoring");
            // Preserved shared namespaces — leave untouched.
            if key != "nightly_discovery_last_pass_at" && !key.starts_with("telegram_role_pin:") {
                unknown_garbage_keys.push(key.clone());
            }
        }
    }
    if !unknown_garbage_keys.is_empty() {
        tracing::debug!(keys=?unknown_garbage_keys, "Purging unknown config_kv orphans");
    }
    for key in unknown_garbage_keys {
        match store.delete_kv(&key).await {
            Ok(()) => tracing::debug!(key, "Purged unknown config_kv orphan"),
            Err(e) => {
                tracing::warn!(key, error=%e, "Failed to purge unknown config_kv orphan (transient, ignoring)");
            }
        }
    }

    let routings = store.get_all_model_routings().await?;
    config.model_routings = routings;

    // Normalise and sort so the in-memory representation matches the
    // per-field persistence paths (see the "Per-field persistence" section).
    config.normalize();

    CONFIG.swap(config);
    tracing::info!("Config reloaded from DB");
    Ok(())
}

// ── Per-field persistence (settings-page autosave) ─────────────────
//
// Each settled config field is persisted individually — a single KV row or a
// single routing row — instead of a whole-config rewrite. Every function
// in this section:
//
// 1. Runs under [`persist_lock`] so read-modify-write sequences and
//    side-effect settles never interleave (see the lock's doc comment).
// 2. Validates the settled value BEFORE anything is written; on failure the
//    DB and the in-memory CONFIG are untouched and the error propagates to
//    the caller for inline display.
// 3. Applies targeted side effects only for the fields that need them
//    (provider/transcriber re-init, Telegram listener reload) — never per
//    keystroke, only when a value settles.
// 4. Returns the canonical persisted value (trimmed, `None` collapsed to
//    `""`) so the caller can re-sync its display snapshot — the settings
//    page writes exactly this value back into the editable snapshot for
//    every field type.
//
// `wake_word_templates` is excluded from every path: it is owned exclusively
// by the voice pipeline (`persist_enrollment` in `voice.rs`), which writes
// the key directly, and any GUI write would create a dual-writer race.

/// Outcome of persisting a settled config field: the canonical persisted value
/// plus an optional non-fatal warning (e.g. an unreachable custom endpoint that
/// was saved anyway).
///
/// `pub` (not `pub(crate)`) because it flows through the public
/// [`crate::gui::settings::SettingsMessage`] enum's persist-result variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistOutcome {
    pub value: String,
    pub warning: Option<String>,
}

/// Persist a single settled string config field.
///
/// `key` must be a `config_kv` key (see [`ConfigData::string_fields`]).
/// Returns the canonical persisted value plus an optional non-fatal warning.
pub async fn persist_settled_string_field(key: &str, value: &str) -> Result<PersistOutcome> {
    let _guard = persist_lock().lock().await;
    let trimmed = value.trim().to_string();

    // Defense in depth: the settings page renders no control for this key, but
    // refuse it structurally so no future caller can ever write it here.
    if key == CONFIG_KEY_WAKE_WORD_TEMPLATES {
        return Ok(PersistOutcome {
            value: CONFIG.wake_word_templates().unwrap_or_default(),
            warning: None,
        });
    }

    match key {
        // Provider endpoint changes re-init the provider and transcriber.
        // Ordering: validate (structural) → write → cascade → warmup
        // (best-effort: an unreachable custom endpoint saves with a warning,
        // so a self-hosted server can be configured before it is reachable) →
        // recreate (runtime switches to the saved
        // endpoint even when unreachable — only structural validation and DB
        // writes can fail a save now). The probe is the live config with only
        // this field applied, so concurrent changes to other fields are never
        // clobbered (the persist lock serializes settles anyway).
        //
        // Note: the active image model is deliberately NOT re-validated here.
        // The image-model catalog is endpoint-keyed, and validating the
        // committed model against the new endpoint would deadlock a provider
        // switch to a disjoint catalog (endpoint rejects until the model
        // changes, model rejects until the endpoint changes). The image model
        // is instead validated when it itself settles, against the then-
        // committed endpoint — so a switch is two steps (endpoint first, then
        // model), each independently valid.
        CONFIG_KEY_PROVIDER_ENDPOINT => {
            let mut probe = CONFIG.snapshot();
            let _ = probe.set_string_field(key, &trimmed);
            probe.normalize();
            validate_config(&probe)?;

            write_kv_and_update_config(key, &trimmed).await?;

            // Cascade: settling a default or empty endpoint removes the custom
            // endpoint's key row — the key is only meaningful while a custom
            // endpoint is active. Covers both the toggle-off "" and typing the
            // default URL back. The endpoint row is written FIRST so a failure
            // on the write leaves the key row intact (consistent state); a
            // failure on the cascade leaves a stale key row, which is never
            // read (harmless per the orphaned-key policy) — so the cleanup is
            // best-effort and never fails the save.
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

            // Warmup is best-effort AND custom-only: an unreachable custom
            // endpoint is saved anyway (with a warning) so a self-hosted
            // server can be configured before it is reachable. Reverting to
            // the default (empty or default URL — toggle-off / typing the
            // default back) is NOT probed: a transient OpenRouter outage
            // must not surface an 'unreachable' warning on a deliberate
            // revert.
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

            // The foreground warmup above (or its deliberate skip on revert)
            // is the only warmup this save performs — recreate_all must not
            // fire a second background warmup of the same endpoint.
            crate::providers::recreate_all(&CONFIG.snapshot(), false).await;
            return Ok(PersistOutcome {
                value: trimmed_or_none(&trimmed).unwrap_or_default(),
                warning,
            });
        }
        // Provider key changes re-init the provider with the new credential.
        // No warn-and-save here: warmup is a connection-pool pre-warm that
        // never validates keys (a GET that ignores auth status), so key saves
        // are structurally validated + written, and `recreate_all` rebuilds the
        // provider with the new credential (warmup runs as a non-fatal
        // background task inside it). This keeps the req-9 custom-endpoint
        // warning channel off the key fields — a warmup failure while a custom
        // endpoint is active must not surface an 'unreachable' warning under
        // the OpenRouter key field.
        CONFIG_KEY_PROVIDER_KEY | CONFIG_KEY_PROVIDER_ENDPOINT_KEY => {
            let mut probe = CONFIG.snapshot();
            let _ = probe.set_string_field(key, &trimmed);
            probe.normalize();
            validate_config(&probe)?;

            write_kv_and_update_config(key, &trimmed).await?;
            crate::providers::recreate_all(&CONFIG.snapshot(), true).await;
            return Ok(PersistOutcome {
                value: trimmed_or_none(&trimmed).unwrap_or_default(),
                warning: None,
            });
        }
        // Telegram token change hot-reloads the listener after the write.
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
            if persisted != old_token {
                crate::channels::telegram::restart_telegram_listener(persisted.as_deref()).await?;
            }
        }
        // The active image model must exist in the endpoint-keyed catalog
        // (fail-open when the catalog is unreachable — matching the
        // generation tool's semantics). A cleared model falls back to the
        // default — the model that would actually be used — so it is
        // validated too.
        //
        // Image models always run on OpenRouter — the catalog is
        // endpoint-keyed on the default, so validation never
        // consults a custom chat endpoint.
        CONFIG_KEY_IMAGE_GEN_MODEL => {
            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string();
            let model_opt = trimmed_or_none(&trimmed);
            let model: &str = model_opt.as_deref().unwrap_or(DEFAULT_IMAGE_GEN_MODEL);
            if model != CONFIG.image_gen_model() {
                crate::tools::image_catalog::validate_image_model_for_endpoint(model, &endpoint)
                    .await?;
            }
            write_kv_and_update_config(CONFIG_KEY_IMAGE_GEN_MODEL, &trimmed).await?;
        }
        // Video transcription model changes rebuild the media transcriber (no
        // provider warmup — the provider is unaffected by this) — the
        // transcriber captures the model at build time.
        CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL => {
            write_kv_and_update_config(key, &trimmed).await?;
            crate::providers::recreate_media_transcriber();
        }
        // Everything else is read dynamically at use time — persist only.
        _ => {
            write_kv_and_update_config(key, &trimmed).await?;
        }
    }

    Ok(PersistOutcome {
        value: trimmed_or_none(&trimmed).unwrap_or_default(),
        warning: None,
    })
}

/// Persist a settled per-model routing `provider_order` (`""` clears it).
///
/// Returns the canonical order value. Routing applies only to the manager and
/// worker model slots — the video-transcription model never consults routing,
/// so persisting a routing row for it is a no-op for the transcriber.
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
    // Reject unknown keys BEFORE touching the DB: `set_string_field` is
    // deliberately lenient (returns `false` for unrecognised keys, leaving
    // CONFIG untouched), and writing a row the in-memory config never
    // mirrors would create an orphaned DB entry. Defensive — the settings
    // page only renders known keys — but keeps a typo'd or future key from
    // silently diverging the DB and CONFIG.
    if !ConfigData::STRUCT_FIELDS_DEFAULT
        .string_fields()
        .iter()
        .any(|(known, _)| *known == key)
    {
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

/// Validate a [`ConfigData`] before persisting — rejecting common misconfigurations.
///
/// # Precondition
/// [`ConfigData::normalize`] MUST have been called before this function.
/// All `Option<String>` fields are assumed to be already trimmed, with
/// empty/whitespace-only values collapsed to `None` by
/// [`normalize_string_fields`][ConfigData::normalize_string_fields] (which `normalize` calls unconditionally for **every** field regardless
/// of its per-field annotation — `non_empty`, `or(…)`, or `list_or(…)`).
fn validate_config(config: &ConfigData) -> Result<()> {
    if let Some(ref ep) = config.provider_endpoint {
        if !is_http_url(ep) {
            anyhow::bail!(
                "Provider endpoint must be a valid URL starting with https:// or http://"
            );
        }
        // Empty-host rejection — only in config validation, not in
        // `is_http_url` (which intentionally stays prefix-only). For values
        // where `is_http_url(ep)` is true (scheme is case-sensitive
        // `http://`/`https://`), extract authority = substring after `://` up
        // to first `/`, `?`, `#` or end, trim, and reject if empty or
        // starts-with `:`.
        //
        // Intentionally NOT rejected here beyond the empty-host check:
        // `https://exa mple.com` (embedded space) passes this check and is
        // deferred to warmup — do not over-tighten to reject it here.
        // IPv6 literals (`[::1]`) and userinfo (`user:pass@host`) are out of
        // scope for this check.
        // The authority double-trim (overall `normalize` trim + this
        // `authority.trim()`) is required to catch `https://   /v1` where the
        // authority is whitespace-only.
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

// ── Test helpers ──────────────────────────────────────────────

/// Construct a [`ModelRouting`] for tests.
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

    /// All string keys that [`ConfigData::string_fields`] returns must be
    /// round-trippable through [`ConfigData::set_string_field`]: setting each
    /// individually and reading back via [`ConfigData::string_fields`] must
    /// produce the same value.
    ///
    /// The test is self-maintaining: it generates synthetic values from each
    /// field's key, so adding a field to `string_config_fields!` automatically
    /// covers it without manual test-data upkeep.
    #[test]
    fn string_fields_roundtrip() {
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;

        // Verify the initial state: all fields are None.
        for (_key, value) in config.string_fields() {
            assert!(value.is_none(), "field should start as None");
        }

        // Set each field to a synthetic value derived from its key and verify
        // it round-trips back through string_fields.  Using synthetic values
        // keeps the test self-maintaining — adding a field to the macro
        // automatically covers it without separate test-data upkeep.
        let keys: Vec<&str> = config.string_fields().iter().map(|(k, _)| *k).collect();
        for &key in &keys {
            let test_value = format!("test-{key}");
            let recognized = config.set_string_field(key, &test_value);
            assert!(recognized, "key '{key}' should be recognized");

            // Find this key in string_fields and verify the value matches.
            let found = config
                .string_fields()
                .iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| *v);
            assert_eq!(
                found,
                Some(test_value.as_str()),
                "value for '{key}' should match after set"
            );
        }

        // ── Normalization is handled by normalize(), not set_string_field ──
        // set_string_field stores the raw value as-is.
        let _ = config.set_string_field("provider_key", "");
        let pk = config
            .string_fields()
            .iter()
            .find(|(k, _)| *k == "provider_key")
            .and_then(|(_, v)| *v);
        assert_eq!(
            pk,
            Some(""),
            "empty string stored as-is by set_string_field"
        );

        let _ = config.set_string_field("provider_key", "   ");
        let pk = config
            .string_fields()
            .iter()
            .find(|(k, _)| *k == "provider_key")
            .and_then(|(_, v)| *v);
        assert_eq!(
            pk,
            Some("   "),
            "whitespace-only string stored as-is by set_string_field"
        );

        // After normalize(), empty/whitespace values are collapsed to None.
        config.normalize();
        let pk = config
            .string_fields()
            .iter()
            .find(|(k, _)| *k == "provider_key")
            .and_then(|(_, v)| *v);
        assert!(pk.is_none(), "normalize() collapses empty string to None");

        // Unknown key returns false.
        assert!(!config.set_string_field("nonexistent_key", "value"));
    }

    /// Smoke test: macro-generated accessors roundtrip correctly for one
    /// representative field of each pattern (`non_empty`, `or`, `list_or`).
    ///
    /// Structural sync (every field has a correctly-typed accessor) is guaranteed
    /// at compile time by the macro — this test only verifies runtime semantics.
    #[test]
    fn config_reload_accessors_roundtrip() {
        let reload = ConfigReload::const_new();

        // ── non_empty: returns None when unset, Some(value) when set ──
        assert_eq!(reload.provider_key(), None, "unset provider_key is None");
        let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(config.set_string_field("provider_key", "sk-test"));
        reload.swap(config);
        assert_eq!(reload.provider_key(), Some("sk-test".to_string()));

        // ── or: falls back to default when unset ──
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
        assert_eq!(
            reload.video_transcription_model(),
            DEFAULT_VIDEO_TRANSCRIPTION_MODEL,
            "unset video_transcription_model falls back to default"
        );

        // ── or: persisted provider_endpoint is honored (custom endpoint) ──
        let mut custom_cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(custom_cfg.set_string_field("provider_endpoint", "https://custom.example/v1"));
        reload.swap(custom_cfg);
        assert_eq!(
            reload.provider_endpoint(),
            "https://custom.example/v1",
            "or field honors a persisted custom value"
        );
        // and falls back to the default when unset
        reload.swap(ConfigData::STRUCT_FIELDS_DEFAULT);
        assert_eq!(reload.provider_endpoint(), DEFAULT_PROVIDER_ENDPOINT);

        // ── non_empty: empty/whitespace → None ──
        let mut empty = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(empty.set_string_field("provider_key", ""));
        reload.swap(empty);
        assert_eq!(
            reload.provider_key(),
            None,
            "empty string is collapsed to None"
        );

        // ── list_or: falls back to active model when list is unset ──
        reload.swap(ConfigData::STRUCT_FIELDS_DEFAULT);
        assert_eq!(
            reload.image_gen_models(),
            vec![DEFAULT_IMAGE_GEN_MODEL.to_string()],
            "unset image_gen_models falls back to active model"
        );

        // When list is set, returns parsed entries
        let mut list_config = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(list_config.set_string_field("image_gen_models", "model-a\nmodel-b\nmodel-c"));
        reload.swap(list_config);
        assert_eq!(
            reload.image_gen_models(),
            vec!["model-a", "model-b", "model-c"]
        );
    }

    #[test]
    fn trimmed_or_none_trims_whitespace() {
        // trimmed_or_none is the canonical primitive — trims and returns None
        // for empty or whitespace-only strings.
        assert_eq!(trimmed_or_none("  value  "), Some("value".to_string()));
        assert_eq!(trimmed_or_none(" "), None);
        assert_eq!(trimmed_or_none(""), None);
    }

    /// Endpoint normalization: trivial variants of the default
    /// OpenRouter URL — trailing slash, surrounding whitespace, uppercase
    /// scheme/host, a trailing `/chat/completions` suffix — must never count
    /// as a custom endpoint, while a genuinely different URL must.
    #[test]
    fn endpoint_normalization_default_vs_custom() {
        // Exact default.
        assert!(is_default_endpoint(DEFAULT_PROVIDER_ENDPOINT));
        // Trailing slash.
        assert!(is_default_endpoint("https://openrouter.ai/api/v1/"));
        // Surrounding whitespace.
        assert!(is_default_endpoint("  https://openrouter.ai/api/v1  "));
        // Uppercase scheme/host.
        assert!(is_default_endpoint("HTTPS://OPENROUTER.AI/api/v1"));
        // Chat-completions suffix (both bare and trailing-slash variants).
        assert!(is_default_endpoint(
            "https://openrouter.ai/api/v1/chat/completions"
        ));
        assert!(is_default_endpoint(
            "https://openrouter.ai/api/v1/chat/completions/"
        ));
        // A genuinely custom endpoint.
        assert!(is_custom_endpoint("http://localhost:8080/v1"));
        assert!(is_custom_endpoint("https://custom.example/v1"));
        assert!(!is_custom_endpoint(DEFAULT_PROVIDER_ENDPOINT));
    }

    /// Chat-credential key isolation: the OpenRouter key is
    /// only ever used for the default endpoint. A custom endpoint uses its
    /// own key; when that key is empty NO Authorization header is sent
    /// (keyless servers) — the OpenRouter key must never reach a custom
    /// endpoint.
    #[test]
    fn chat_credential_isolates_keys_per_endpoint() {
        // Default endpoint → OpenRouter key.
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert_eq!(chat_credential(&cfg).as_deref(), Some("sk-or"));

        // Custom endpoint with its own key → the custom key (OR key ignored).
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert!(cfg.set_string_field("provider_endpoint_key", "sk-custom"));
        assert_eq!(chat_credential(&cfg).as_deref(), Some("sk-custom"));

        // Custom endpoint, keyless (empty custom key), OR key set → None:
        // the OR key must never be sent to a custom endpoint.
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_key", "sk-or"));
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert!(cfg.set_string_field("provider_endpoint_key", ""));
        assert_eq!(chat_credential(&cfg), None);

        // Custom endpoint, keyless, no OR key → None (keyless operation).
        let mut cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(cfg.set_string_field("provider_endpoint", "http://localhost:8080/v1"));
        assert_eq!(chat_credential(&cfg), None);

        // Default endpoint, no key → None.
        assert_eq!(chat_credential(&ConfigData::STRUCT_FIELDS_DEFAULT), None);
    }

    // NOTE: Per-struct normalize tests (`model_routing_normalize`) have been
    // intentionally removed as redundant.  The `normalize()` method is a
    // one-line delegation to `non_empty()` with no conditional logic.  The
    // `non_empty` / `trimmed_or_none` primitive is covered exhaustively by
    // `trimmed_or_none_trims_whitespace` above, and the end-to-end integration
    // through `normalize_entries()` is covered by `normalize_entries_works`
    // below.  If a new normalization scenario is added, it should be added to
    // the primitive test AND exercised through the integration test — there is
    // no need for per-struct test duplication.

    /// Verify that [`ConfigData::normalize_entries`] normalises every entry in
    /// `model_routings`.
    #[test]
    fn normalize_entries_works() {
        let mut config = ConfigData {
            model_routings: vec![
                ModelRouting {
                    model: "test-model".into(),
                    provider_order: Some("   ".into()),
                },
                ModelRouting {
                    model: "test-model-2".into(),
                    provider_order: Some("  OpenAi,  Anthropic  ".into()),
                },
            ],
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };

        config.normalize_entries();

        // Routing: whitespace-only provider_order → None
        assert_eq!(config.model_routings[0].provider_order, None);

        // Routing: trimmed provider_order preserved
        assert_eq!(
            config.model_routings[1].provider_order,
            Some("OpenAi,  Anthropic".into())
        );
    }

    // ── Upsert three-scenario tests ─────────────────────────
    //
    // `ModelRouting::upsert` is tested across three scenarios:
    //   1. updates_existing — existing entry, upsert sets the target field
    //   2. pushes_new_entry — empty vec, new entry is pushed with the
    //      target field set
    //   3. can_set_none — existing entry has the field set to non-None;
    //      clearing it via None removes the value

    #[test]
    fn upsert_model_routing_fields() {
        // 1. updates_existing — set provider_order
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"))];
            ModelRouting::upsert(&mut items, "test-model", |item| {
                item.provider_order = Some("Anthropic".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].provider_order,
                Some("Anthropic".into()),
                "[provider_order] target field updated"
            );
        }

        // 2. pushes_new_entry
        {
            let mut items = vec![];
            ModelRouting::upsert(&mut items, "test-model", |item| {
                item.provider_order = Some("OpenAi".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].model, "test-model");
            assert_eq!(
                items[0].provider_order,
                Some("OpenAi".into()),
                "[provider_order] set on new entry"
            );
        }

        // 3. can_set_none — clear provider_order
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"))];
            ModelRouting::upsert(&mut items, "test-model", |item| item.provider_order = None);
            assert_eq!(
                items[0].provider_order, None,
                "[provider_order] cleared to None"
            );
        }
    }

    #[test]
    fn upsert_multiple_entries_independent_keys() {
        let mut routings = vec![
            model_routing("test-router-a", Some("OpenAi")),
            model_routing("test-router-b", Some("Anthropic")),
        ];

        // Each upsert targets exactly one entry by key.
        ModelRouting::upsert(&mut routings, "test-router-a", |mr| {
            mr.provider_order = Some("Google".into());
        });
        assert_eq!(routings[0].provider_order, Some("Google".into()));
        assert_eq!(routings[1].provider_order, Some("Anthropic".into()));

        ModelRouting::upsert(&mut routings, "test-router-b", |mr| {
            mr.provider_order = Some("OpenAi".into());
        });
        assert_eq!(routings[0].provider_order, Some("Google".into()));
        assert_eq!(routings[1].provider_order, Some("OpenAi".into()));

        // Total entries unchanged — no spurious pushes.
        assert_eq!(routings.len(), 2);
    }

    // ── validate_config tests ──────────────────────────────────────

    /// A valid URL (trimmed) passes validation.
    #[test]
    fn validate_config_accepts_valid_url() {
        let mut config = ConfigData {
            provider_endpoint: Some("https://openrouter.ai/api/v1".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        validate_config(&config).unwrap();
    }

    /// A whitespace-padded URL passes validation after `normalize` normalises
    /// it.  This is a regression test for the latent ordering bug where
    /// `validate_config` (which used untrimmed `starts_with`) ran *before*
    /// `normalize` (which trims).  The fix ensures `normalize` always runs
    /// first, so validation only ever sees canonical values.
    #[test]
    fn validate_config_accepts_whitespace_padded_url_after_normalize() {
        let mut config = ConfigData {
            provider_endpoint: Some("  https://openrouter.ai/api/v1   ".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        // After normalize the value is trimmed — validation sees the canonical form.
        validate_config(&config).unwrap();
    }

    /// A URL without scheme is rejected regardless of whitespace.
    #[test]
    fn validate_config_rejects_url_without_scheme() {
        let mut config = ConfigData {
            provider_endpoint: Some("not-a-url".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("Provider endpoint must be a valid URL"),
            "expected URL scheme error, got: {err}",
        );
    }

    /// A placeholder provider key is rejected.
    #[test]
    fn validate_config_rejects_placeholder_key() {
        let mut config = ConfigData {
            provider_key: Some("sk-or-v1-...".into()),
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        config.normalize();
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("placeholder"),
            "expected placeholder error, got: {err}",
        );
    }

    /// The per-field persist path must never write or delete
    /// `wake_word_templates` — the key is owned exclusively by the voice
    /// pipeline (`persist_enrollment`), and `persist_settled_string_field`
    /// refuses it structurally (defense in depth) so no future caller can
    /// accidentally create a dual-writer race.
    ///
    /// This test swaps the shared global CONFIG, so it joins the
    /// `config_persist` serial group used by the config_db persist tests —
    /// an unserialized restore swap could clobber a concurrent serialized
    /// test's CONFIG writes and fail its asserts nondeterministically.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn persist_settled_string_field_refuses_wake_word_templates() {
        // Templates enrolled by the voice pipeline.
        let template_json = r#"{"classifier":null}"#;
        let original = CONFIG.snapshot();
        let mut enrolled = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(enrolled.set_string_field("wake_word_templates", template_json));
        CONFIG.swap(enrolled);

        // The guard returns the current templates and never touches the DB —
        // the call must not error and must not alter CONFIG.
        let result = persist_settled_string_field("wake_word_templates", "garbage").await;
        assert_eq!(
            result.unwrap().value,
            template_json,
            "guard must return the current templates unchanged"
        );
        assert_eq!(
            CONFIG.wake_word_templates(),
            Some(template_json.to_string()),
            "CONFIG wake_word_templates must be untouched"
        );

        CONFIG.swap(original);
    }

    /// The persist side-effect arms and the `wake_word_templates` guard are
    /// wired through `CONFIG_KEY_*` constants (compile-tied: a rename in
    /// `string_config_fields!` changes both the constant name and its value,
    /// so every arm referencing the old name fails to compile). This test
    /// pins the vocabulary contract: every key that carries a persist side
    /// effect (or the structural guard) must still be a real config field,
    /// and documents the exact set a rename must keep in sync.
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
            CONFIG_KEY_IMAGE_GEN_MODEL,
            CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL,
            CONFIG_KEY_WAKE_WORD_TEMPLATES,
        ] {
            assert!(
                known.contains(&key),
                "side-effect key '{key}' must be a real config field"
            );
        }
    }

    /// Fresh-install seed: the fresh-install
    /// defaults land in brand-new config databases only — existing databases
    /// receive zero writes.
    ///
    /// The seeded set is the transcription-off default (so no audio model is
    /// downloaded or loaded at boot) plus the image/video generation model
    /// lists and active selections (so the Settings GUI model pickers, which
    /// read the raw snapshot fields, are populated on fresh installs).
    ///
    /// This test drives the real boot chain end-to-end instead of hand-picking
    /// `fresh` values: the discriminator probe (`config_db_is_fresh` — the
    /// exact function `load_or_init` uses), the boot flag it feeds, the store
    /// open that creates `db/config.db`, and the flag-consumption + seed
    /// (`seed_fresh_install_defaults_from_flag`, the `reload_from_db` path).
    ///
    /// Regression guard for the wrong-path probe: opening the config store
    /// must flip the discriminator to "not fresh" — a probe of any other
    /// location (e.g. a top-level `<root>/config.db`) would still report
    /// "fresh" here and re-seed every existing install on every boot.
    #[tokio::test]
    async fn fresh_config_db_seeds_defaults_only_when_new() {
        // ── Fresh install: the store file does not exist yet ──
        let fresh_root = tempfile::TempDir::new().unwrap();
        let fresh = config_db_is_fresh(fresh_root.path());
        assert!(
            fresh,
            "a storage root with no config store file must be classified fresh"
        );
        CONFIG_DB_FRESH_AT_BOOT.store(fresh, Ordering::Release);

        // Opening the config store creates <root>/db/config.db — the exact
        // file the probe must check. This is the regression pin for the
        // original bug: probing <root>/config.db would still report "fresh"
        // here and re-seed existing installs on every boot.
        let fresh_store = crate::config_db::ConfigStore::open(fresh_root.path())
            .await
            .unwrap();
        assert!(
            !config_db_is_fresh(fresh_root.path()),
            "an existing config store file must not be classified fresh"
        );

        // reload_from_db's path: consume the boot flag and seed.
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
                    "image_gen_model".to_string(),
                    "google/gemini-3.1-flash-image".to_string()
                ),
                (
                    "image_gen_models".to_string(),
                    "google/gemini-3.1-flash-image\nmicrosoft/mai-image-2.5\nqwen/qwen-image-3-pro"
                        .to_string()
                ),
                ("video_model".to_string(), "minimax/hailuo-3".to_string()),
                (
                    "video_models".to_string(),
                    "bytedance/seedance-2.0-mini\nminimax/hailuo-3".to_string()
                ),
            ],
            "a fresh config database must be seeded with the fresh-install defaults"
        );
        assert_eq!(
            fresh_store.get_all_model_routings().await.unwrap(),
            vec![model_routing(DEFAULT_MANAGER_MODEL, Some("DeepSeek"))],
            "a fresh config database must seed a single DeepSeek routing row for \
             the vision-capable default model shared by the manager and worker \
             slots (others get none)"
        );

        // ── Existing install: the store file already exists ──
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
        assert!(
            existing_store.get_all_kv().await.unwrap().is_empty(),
            "an existing config database must receive zero writes"
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

    /// One-time migration: persisted `config_kv` rows holding the previous
    /// manager/worker default model are rewritten to the new vision-capable
    /// default, and only those rows (and only when they match the exact old
    /// default) are touched. Re-running is a no-op.
    #[tokio::test]
    async fn migrate_old_default_models_rewrites_only_stale_slot_rows() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        // Existing-install snapshot at the time of the default swap: both slots
        // still hold the previous defaults; the video-transcription slot and an
        // unrelated key must remain untouched.
        store
            .set_kv(CONFIG_KEY_MANAGER_MODEL, OLD_DEFAULT_MANAGER_MODEL)
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_WORKER_MODEL, OLD_DEFAULT_WORKER_MODEL)
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL, "qwen/qwen3.7-flash")
            .await
            .unwrap();
        store
            .set_kv(CONFIG_KEY_PROVIDER_KEY, "sk-example")
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
            "manager slot holding the old default must be migrated"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_WORKER_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some(DEFAULT_WORKER_MODEL),
            "worker slot holding the old default must be migrated"
        );
        assert_eq!(
            store
                .get_kv(CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL)
                .await
                .unwrap()
                .as_deref(),
            Some("qwen/qwen3.7-flash"),
            "the video-transcription slot must never be touched"
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

        // Idempotent: a second run changes nothing.
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
    }

    /// The migration must be a no-op when a slot already holds the target
    /// default or a genuine user override — it must never overwrite a
    /// user-configured model.
    #[tokio::test]
    async fn migrate_old_default_models_noop_for_target_and_user_overrides() {
        let (store, _dir) = crate::open_test_store!(crate::config_db::ConfigStore, "config");

        // Already at the target default.
        store
            .set_kv(CONFIG_KEY_MANAGER_MODEL, DEFAULT_MANAGER_MODEL)
            .await
            .unwrap();
        // A genuine user override (not the old default) must be preserved.
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
    }
}

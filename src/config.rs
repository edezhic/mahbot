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
//! **independent** — a KV entry in Chain 1 cannot override a per-role model in
//! Chain 2, and a per-role entry in Chain 2 cannot override a per-model routing
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
//! Fields **not** in this chain (e.g. `per_role_configs`, `model_routings`) have
//! their own dedicated tables and reload paths.
//!
//! ## Chain 2: Per-role model and reasoning effort
//!
//! `config_role` table → [`crate::role::RoleInfo::default_model`] / [`crate::role::RoleInfo::default_reasoning_effort`]
//!
//! Stored in [`ConfigData::per_role_configs`] as a [`Vec<RoleConfig>`][RoleConfig],
//! loaded at reload time from the `config_role` table. Checked at request time by
//! [`ConfigReload::role_model`] and [`ConfigReload::role_reasoning_effort`] with
//! the priority:
//!
//! > Per-role override → [`role_info`]`(role).default_*`
//!
//! When no matching [`RoleConfig`] entry exists, the role's built-in default from
//! [`role_info`] (defined in [`crate::role`]) is returned.
//!
//! ## Chain 3: Per-model provider routing
//!
//! `config_model_routing` table → `None` defaults
//!
//! Stored in [`ConfigData::model_routings`] as a [`Vec<ModelRouting>`][ModelRouting],
//! loaded at reload time from the `config_model_routing` table. Checked via
//! [`ConfigReload::model_routing`]. When no entry exists, all fields on the
//! returned [`ModelRouting`] — `provider_order` and `allow_fallbacks` — are `None`.
//! The provider layer (in [`crate::providers`]) resolves these `None` values at
//! request time when building the OpenAI-compatible chat request.
//!
//! # Persistence layer
//!
//! The three tables live in `config.db` and are managed by [`crate::config_db`]:
//!
//! | Table | Read | Write |
//! |---|---|---|
//! | `config_kv` | [`crate::config_db::ConfigStore::get_all_kv`] | [`crate::config_db::ConfigStore::set_kv`] |
//! | `config_role` | [`crate::config_db::ConfigStore::get_all_role_configs`] | [`crate::config_db::ConfigStore::save_role_config`] |
//! | `config_model_routing` | [`crate::config_db::ConfigStore::get_all_model_routings`] | [`crate::config_db::ConfigStore::save_model_routing`] |
//!
//! # Orphaned database keys
//!
//! Rows in `config_kv` without a corresponding [`ConfigData`] field are
//! silently ignored and require no migration. Some legacy keys are still read
//! as migration sources in [`reload_from_db`] when the target field is unset;
//! those rows are kept intentionally (a downgrade would resurrect them) and
//! must not be deleted. Orphaned rows are harmless and are naturally
//! overwritten if a future config key reuses the name.
//!
//! # See also
//!
//! * [`crate::config_db`] — database persistence for all three chains.
//! * [`crate::role`] — [`crate::role::RoleInfo`] definitions with per-role defaults.
//! * [`crate::providers::compatible`] — where `None` routing fields are resolved
//!   at the provider layer.

use crate::Role;
use crate::role::role_info;
use crate::util::{UnwrapPoison, is_http_url};
use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use tokio::fs;

// ── Hardcoded defaults ───────────────────────────────────────────

pub(crate) const DEFAULT_PROVIDER_ENDPOINT: &str = "https://openrouter.ai/api/v1";

const DEFAULT_IMAGE_GEN_MODEL: &str = "google/gemini-3.1-flash-image";
const DEFAULT_VIDEO_MODEL: &str = "minimax/hailuo-3";
pub(crate) const DEFAULT_MEDIA_TRANSCRIPTION_MODEL: &str = "qwen/qwen3.6-plus";
pub(crate) const DEFAULT_TTS_LANGUAGE: &str = "na";

/// Default adaptive k multiplier for wake word detection.
const DEFAULT_ADAPTIVE_K: &str = "2.5";

// ── Named config structs ───────────────────────────────────────────

/// A per-role model & reasoning-effort override.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleConfig {
    pub role: String,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
}

// NOTE: RoleConfig::upsert and ModelRouting::upsert are structurally identical
// by design. The ~13 lines of shared find-or-push logic are below the abstraction
// threshold — a trait, macro, or generic function would add more conceptual surface
// area than the duplication it eliminates. Keep both methods direct and concrete.

impl RoleConfig {
    /// Find-or-push: update a subset of fields on an existing entry matching
    /// `role`, or push a new entry (all fields defaulted to `None`).
    ///
    /// Only the field(s) mutated inside `set_field` are touched — if the
    /// entry already exists its other fields are preserved unchanged.
    pub(crate) fn upsert(
        configs: &mut Vec<RoleConfig>,
        role: impl Into<String>,
        set_field: impl FnOnce(&mut RoleConfig),
    ) {
        let role = role.into();
        if let Some(existing) = configs.iter_mut().find(|rc| rc.role == role) {
            set_field(existing);
        } else {
            let mut new = RoleConfig {
                role,
                model: None,
                reasoning_effort: None,
            };
            set_field(&mut new);
            configs.push(new);
        }
    }
}

/// A per-model provider routing rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRouting {
    pub model: String,
    pub provider_order: Option<String>,
    pub allow_fallbacks: Option<bool>,
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
                allow_fallbacks: None,
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
///    [`ConfigData::string_fields()`], and [`ConfigData::set_string_field()`] from this list.  The compiler
///    enforces that every field on [`ConfigData`] is present in `STRUCT_FIELDS_DEFAULT`,
///    so forgetting this step is a compile error.
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
    /// Media transcription model (images and videos).
    pub media_transcription_model: Option<String>,
    /// Media transcription provider routing.
    pub media_transcription_provider: Option<String>,
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
    /// Adaptive threshold k multiplier for wake word detection (default: `"2.5"`).
    /// The adaptive threshold is computed as `mean + k × std` over a running
    /// window of recent per-frame classifier scores.  Range: [1.0, 4.0].
    pub adaptive_k: Option<String>,
    /// Per-role model overrides.
    pub per_role_configs: Vec<RoleConfig>,
    /// Per-model provider routing.
    /// `allow_fallbacks` is `None` when unset (defaults to `false` at request time).
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
// Each field is annotated with one of four patterns:
//
// * `non_empty` — returns `Option<String>`, collapses empty/whitespace to `None`.
// * `or(DEFAULT)` — returns `String`, falls back to the given default constant.
// * `fixed(DEFAULT)` — returns `String`, ALWAYS the given constant (the
//   persisted field value is not honored; used when a field stays in the
//   schema for future use but only one value is currently supported).
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
/// `non_empty`, `or($default)`, `fixed($default)`, or `list_or(fallback = $fallback, default = $default)`.
///
/// All generated items are guaranteed to stay synchronised because they expand
/// from the same source.
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
                per_role_configs: Vec::new(),
                model_routings: Vec::new(),
            };

            /// Return all string-valued config fields as (db_key, current_value) pairs.
            ///
            /// `per_role_configs` and `model_routings` are **not** included here:
            /// the former lives in a separate database table (`config_role`),
            /// the latter in `config_model_routing`.
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
            /// `per_role_configs` and `model_routings` are **not** handled here —
            /// they live in separate database tables (`config_role` and `config_model_routing`).
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

    // ── Accessor pattern: fixed(DEFAULT) ────────────────────────
    //
    // Returns String, ALWAYS the given constant — the persisted field
    // value is not honored at runtime. Used when a config field stays in
    // the schema (future use) but only one value is currently supported.
    (@accessor $field:ident fixed($default:expr)) => {
        #[doc = concat!(
            "Returns the hardcoded `", stringify!($field),
            "` value `", stringify!($default),
            "` — any persisted value is not honored while only this ",
            "value is supported."
        )]
        #[must_use]
        pub fn $field(&self) -> String {
            $default.to_string()
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
    provider_endpoint [fixed(DEFAULT_PROVIDER_ENDPOINT)],
    media_transcription_model [or(DEFAULT_MEDIA_TRANSCRIPTION_MODEL)],
    media_transcription_provider [non_empty],
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
    adaptive_k [or(DEFAULT_ADAPTIVE_K)],
}

impl ConfigData {
    /// Normalise inner `Option<String>` fields of `Vec` entries in place:
    /// trim whitespace and collapse empty/whitespace-only values to `None`.
    ///
    /// This is the Vec-entry counterpart of [`normalize_string_fields()`] —
    /// the macro-generated method only touches top-level `Option<String>` fields,
    /// not the inner fields of [`RoleConfig`] and [`ModelRouting`] entries.
    fn normalize_entries(&mut self) {
        for rc in &mut self.per_role_configs {
            rc.model = non_empty(rc.model.take());
            rc.reasoning_effort = non_empty(rc.reasoning_effort.take());
        }
        for mr in &mut self.model_routings {
            mr.provider_order = non_empty(mr.provider_order.take());
        }
    }

    /// Apply canonical normalisation + sorting to the in-memory
    /// representation so it is consistent across all persistence paths.
    ///
    /// The sequence is:
    /// 1. Trim top-level `Option<String>` fields and collapse empty → `None`.
    /// 2. Trim inner fields on `Vec` entries (`[RoleConfig]`, `[ModelRouting]`)
    ///    and collapse empty → `None`.
    /// 3. Sort `per_role_configs` by role name.
    /// 4. Sort `model_routings` by model name.
    ///
    /// Every caller that produces a newly-built [`ConfigData`] must call
    /// this before swapping into the global [`CONFIG`] so that the in-memory
    /// representation is the same regardless of which code path produced it.
    pub(crate) fn normalize(&mut self) {
        self.normalize_string_fields();
        self.normalize_entries();
        self.per_role_configs.sort_by(|a, b| a.role.cmp(&b.role));
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

// ── ConfigReload — global singleton ──────────────────────────────

/// Global reloadable config singleton.
///
/// The `storage_root` is immutable after startup; all other fields live in an
/// `RwLock<ConfigData>` that can be atomically swapped at runtime.
pub static CONFIG: ConfigReload = ConfigReload::const_new();

/// Serializes all per-field config persistence (settings-page autosave).
///
/// Every settled-field persist runs inside this lock so that:
/// - read-modify-write sequences (per-role rows, endpoint/key probes built
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

    /// Find the per-role override row by role key (string form), if one exists.
    ///
    /// Unlike [`Self::find_role_config`] (which takes the [`Role`] enum),
    /// this accepts the raw role key as stored in `config_role.role`, so the
    /// settings page and persistence layer can look rows up without
    /// re-parsing the key.
    pub(crate) fn role_config_by_key(&self, role: &str) -> Option<RoleConfig> {
        let guard = self.read();
        guard
            .per_role_configs
            .iter()
            .find(|rc| rc.role == role)
            .cloned()
    }

    /// Apply a single role-config row to the in-memory config (find-or-push).
    ///
    /// When both override fields are `None` the row is removed — an all-None
    /// row is indistinguishable from having no override, and the removal keeps
    /// the in-memory representation identical to what [`reload_from_db`]
    /// would load from the DB after the matching row deletion.
    pub(crate) fn set_role_config_row(
        &self,
        role: &str,
        model: Option<String>,
        reasoning_effort: Option<String>,
    ) {
        let mut guard = self.inner.write().unwrap_poison();
        if model.is_none() && reasoning_effort.is_none() {
            guard.per_role_configs.retain(|rc| rc.role != role);
        } else {
            RoleConfig::upsert(&mut guard.per_role_configs, role, |rc| {
                rc.model = model;
                rc.reasoning_effort = reasoning_effort;
            });
        }
    }

    /// Apply a single model-routing row to the in-memory config (find-or-push).
    ///
    /// `Some(false)` on `allow_fallbacks` is meaningful and preserved; only
    /// `None` + `None` removes the row (mirroring [`ConfigReload::set_role_config_row`]).
    pub(crate) fn set_model_routing_row(
        &self,
        model: &str,
        provider_order: Option<String>,
        allow_fallbacks: Option<bool>,
    ) {
        let mut guard = self.inner.write().unwrap_poison();
        if provider_order.is_none() && allow_fallbacks.is_none() {
            guard.model_routings.retain(|mr| mr.model != model);
        } else {
            ModelRouting::upsert(&mut guard.model_routings, model, |mr| {
                mr.provider_order = provider_order;
                mr.allow_fallbacks = allow_fallbacks;
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
        let guard = self.read();
        if let Some(mr) = guard.model_routings.iter().find(|mr| mr.model == model) {
            mr.clone()
        } else {
            ModelRouting {
                model: model.to_string(),
                provider_order: None,
                allow_fallbacks: None,
            }
        }
    }

    // ── Per-role model resolution ───────────────────────────────

    /// Find the per-role override config for a given role, if one exists.
    ///
    /// Returns `None` when no matching override is configured in
    /// `per_role_configs`.
    fn find_role_config(&self, role: Role) -> Option<RoleConfig> {
        let role_key: &str = role.into();
        let guard = self.read();
        guard
            .per_role_configs
            .iter()
            .find(|rc| rc.role == role_key)
            .cloned()
    }

    /// Resolve the configured model for a role.
    ///
    /// Priority: per-role override → role info default.
    #[must_use]
    pub fn role_model(&self, role: Role) -> String {
        if let Some(rc) = self.find_role_config(role)
            && let Some(ref m) = rc.model
        {
            return m.clone();
        }
        role_info(&role).default_model.to_string()
    }

    /// Resolve the configured reasoning effort for a role.
    ///
    /// Priority: per-role override → role info default.
    /// Always returns a value — every role has a non-empty default defined
    /// in [`role_info`].
    #[must_use]
    pub fn role_reasoning_effort(&self, role: Role) -> String {
        if let Some(rc) = self.find_role_config(role)
            && let Some(ref r) = rc.reasoning_effort
        {
            return r.clone();
        }
        role_info(&role).default_reasoning_effort.to_string()
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

    // Start with hardcoded defaults — reload_from_db() will overlay
    // any persisted values from config.db (called later in bootstrap).
    CONFIG.swap(ConfigData::STRUCT_FIELDS_DEFAULT);

    tracing::info!(
        "Config system initialised (storage root: {}).",
        mahbot_dir.display()
    );
    Ok(())
}

/// First value in `kvs` whose key matches any of `legacy` (in order).
fn first_legacy_value<'a>(kvs: &'a [(String, String)], legacy: &[&str]) -> Option<&'a str> {
    legacy.iter().find_map(|k| {
        kvs.iter()
            .find(|(kk, _)| kk.as_str() == *k)
            .map(|(_, v)| v.as_str())
    })
}

/// Reload config from the `config.db` database, atomically swapping the
/// runtime config. Called at startup (after config_db init) to overlay
/// persisted settings on top of hardcoded defaults.
pub async fn reload_from_db() -> Result<()> {
    let store = crate::config_db::store();
    let mut config = ConfigData::STRUCT_FIELDS_DEFAULT;

    let kvs = store.get_all_kv().await?;
    for (key, value) in &kvs {
        if !config.set_string_field(key, value) {
            tracing::debug!(key, "Unknown config key, ignoring");
        }
    }

    // Migrate the legacy split video config keys into the unified video_model
    // field: the old video-edit value takes precedence over the old video-gen
    // value. The legacy keys remain as orphaned config_kv rows (harmless; a
    // downgrade would resurrect them — accepted).
    if config.video_model.is_none() {
        config.video_model =
            first_legacy_value(&kvs, &["video_edit_model", "video_gen_model"]).map(String::from);
    }

    // Migrate the legacy image_transcription_* keys into media_transcription_*
    // (the transcriber now handles images and videos). The legacy keys remain
    // as orphaned config_kv rows (harmless; a downgrade would resurrect them).
    if config.media_transcription_model.is_none() {
        config.media_transcription_model =
            first_legacy_value(&kvs, &["image_transcription_model"]).map(String::from);
    }
    if config.media_transcription_provider.is_none() {
        config.media_transcription_provider =
            first_legacy_value(&kvs, &["image_transcription_provider"]).map(String::from);
    }

    let roles = store.get_all_role_configs().await?;
    config.per_role_configs = roles;

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
// single role/routing row — instead of a whole-config rewrite. Every function
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
//    `""`; `"true"`/`"false"` for the routing-allow toggle) so the caller
//    can re-sync its display snapshot — the settings page writes exactly
//    this value back into the editable snapshot for every field type.
//
// `wake_word_templates` is excluded from every path: it is owned exclusively
// by the voice pipeline (`persist_enrollment` in `voice.rs`), which writes
// the key directly, and any GUI write would create a dual-writer race.

/// Persist a single settled string config field.
///
/// `key` must be a `config_kv` key (see [`ConfigData::string_fields`]).
/// Returns the canonical persisted value.
pub async fn persist_settled_string_field(key: &str, value: &str) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let trimmed = value.trim().to_string();

    // Defense in depth: the settings page renders no control for this key, but
    // refuse it structurally so no future caller can ever write it here.
    if key == stringify!(wake_word_templates) {
        return Ok(CONFIG.wake_word_templates().unwrap_or_default());
    }

    match key {
        // Provider endpoint/key changes re-init the provider and transcriber.
        // Ordering: validate → warmup (pre-commit) → write → recreate
        // (post-commit). The probe is the live config with only this field
        // applied, so concurrent changes to other fields are never clobbered
        // (the persist lock serializes settles anyway).
        //
        // Note: the active image model is deliberately NOT re-validated here.
        // The image-model catalog is endpoint-keyed, and validating the
        // committed model against the new endpoint would deadlock a provider
        // switch to a disjoint catalog (endpoint rejects until the model
        // changes, model rejects until the endpoint changes). The image model
        // is instead validated when it itself settles, against the then-
        // committed endpoint — so a switch is two steps (endpoint first, then
        // model), each independently valid.
        "provider_endpoint" => {
            let mut probe = CONFIG.snapshot();
            let _ = probe.set_string_field("provider_endpoint", &trimmed);
            probe.normalize();
            validate_config(&probe)?;

            crate::providers::warmup_provider_from_config(&probe).await?;
            write_kv_and_update_config("provider_endpoint", &trimmed).await?;
            crate::providers::recreate_all(&CONFIG.snapshot()).await?;
        }
        "provider_key" => {
            let mut probe = CONFIG.snapshot();
            let _ = probe.set_string_field("provider_key", &trimmed);
            probe.normalize();
            validate_config(&probe)?;
            crate::providers::warmup_provider_from_config(&probe).await?;
            write_kv_and_update_config("provider_key", &trimmed).await?;
            crate::providers::recreate_all(&CONFIG.snapshot()).await?;
        }
        // Telegram token change hot-reloads the listener after the write.
        "telegram_bot_token" => {
            let old_token = CONFIG.telegram_bot_token();
            let new_token = trimmed_or_none(&trimmed);
            if new_token != old_token
                && let Some(ref token) = new_token
            {
                crate::channels::telegram::TelegramChannel::validate_token(token).await?;
            }
            write_kv_and_update_config("telegram_bot_token", &trimmed).await?;
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
        "image_gen_model" => {
            let endpoint = CONFIG.provider_endpoint();
            let model_opt = trimmed_or_none(&trimmed);
            let model: &str = model_opt.as_deref().unwrap_or(DEFAULT_IMAGE_GEN_MODEL);
            if model != CONFIG.image_gen_model() {
                crate::tools::image_catalog::validate_image_model_for_endpoint(model, &endpoint)
                    .await?;
            }
            write_kv_and_update_config("image_gen_model", &trimmed).await?;
        }
        // Transcription settings rebuild the media transcriber (no provider
        // warmup — the provider is unaffected by these) and kick the local
        // transcriber load when it becomes enabled.
        "media_transcription_model" | "media_transcription_provider" => {
            write_kv_and_update_config(key, &trimmed).await?;
            crate::providers::recreate_media_transcriber();
        }
        "audio_transcription_use_local" => {
            write_kv_and_update_config(key, &trimmed).await?;
            if trimmed != "false" && !crate::audio::local_transcriber::is_loaded() {
                crate::audio::local_transcriber::try_init_from_cache().await;
            }
        }
        // Everything else is read dynamically at use time — persist only.
        _ => {
            write_kv_and_update_config(key, &trimmed).await?;
        }
    }

    Ok(trimmed_or_none(&trimmed).unwrap_or_default())
}

/// Persist a settled per-role model override (`""` clears it).
///
/// Preserves the role row's current `reasoning_effort` from the live config,
/// so a reasoning-effort edit and a model edit on the same role never clobber
/// each other regardless of arrival order. Returns the canonical model value.
pub async fn persist_settled_role_model(role: &str, model: &str) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let model = trimmed_or_none(model);
    let reasoning = CONFIG
        .role_config_by_key(role)
        .and_then(|rc| rc.reasoning_effort);
    save_role_row(role, model, reasoning).await
}

/// Persist a settled per-role reasoning-effort override.
///
/// Preserves the role row's current `model` from the live config. Returns the
/// canonical effort value (the settled effort trimmed, `None` collapsed to
/// `""` — NOT the row's preserved model).
pub async fn persist_settled_role_reasoning(role: &str, effort: &str) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let effort = trimmed_or_none(effort);
    let model = CONFIG.role_config_by_key(role).and_then(|rc| rc.model);
    save_role_row(role, model, effort.clone()).await?;
    Ok(effort.unwrap_or_default())
}

/// Persist a settled per-model routing `provider_order` (`""` clears it).
///
/// Preserves the row's current `allow_fallbacks` from the live config.
/// Returns the canonical order value.
pub async fn persist_settled_routing_order(model: &str, order: &str) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let order = trimmed_or_none(order);
    let allow = CONFIG.model_routing(model).allow_fallbacks;
    save_routing_row(model, order, allow).await
}

/// Persist a settled per-model routing `allow_fallbacks` toggle.
///
/// Preserves the row's current `provider_order` from the live config.
/// Returns the canonical allow value (`"true"`/`"false"`) so the caller can
/// re-sync its display snapshot — the row's order is unchanged by this call.
pub async fn persist_settled_routing_allow(model: &str, allow: bool) -> Result<String> {
    let _guard = persist_lock().lock().await;
    let order = CONFIG.model_routing(model).provider_order;
    save_routing_row(model, order, Some(allow)).await?;
    Ok(allow.to_string())
}

/// Write a single role row (UPSERT or DELETE-if-empty) and mirror it into the
/// in-memory CONFIG. Caller holds [`persist_lock`].
async fn save_role_row(
    role: &str,
    model: Option<String>,
    reasoning_effort: Option<String>,
) -> Result<String> {
    let store = crate::config_db::store();
    store
        .save_role_config(role, model.as_deref(), reasoning_effort.as_deref())
        .await?;
    CONFIG.set_role_config_row(role, model.clone(), reasoning_effort);
    Ok(model.unwrap_or_default())
}

/// Write a single routing row (UPSERT or DELETE-if-empty) and mirror it into
/// the in-memory CONFIG. Caller holds [`persist_lock`].
async fn save_routing_row(
    model: &str,
    order: Option<String>,
    allow: Option<bool>,
) -> Result<String> {
    let store = crate::config_db::store();
    store
        .save_model_routing(model, order.as_deref(), allow)
        .await?;
    CONFIG.set_model_routing_row(model, order.clone(), allow);
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
    if let Some(ref ep) = config.provider_endpoint
        && !is_http_url(ep)
    {
        anyhow::bail!("Provider endpoint must be a valid URL starting with https:// or http://");
    }

    if let Some(ref key) = config.provider_key
        && key.contains("...")
    {
        anyhow::bail!("Provider key is still the placeholder value — please set a real key");
    }

    Ok(())
}

// ── Test helpers ──────────────────────────────────────────────

/// Construct a [`RoleConfig`] for tests.
#[cfg(test)]
pub(crate) fn role_config(
    role: &str,
    model: Option<&str>,
    reasoning_effort: Option<&str>,
) -> RoleConfig {
    RoleConfig {
        role: role.into(),
        model: model.map(String::from),
        reasoning_effort: reasoning_effort.map(String::from),
    }
}

/// Construct a [`ModelRouting`] for tests.
#[cfg(test)]
pub(crate) fn model_routing(
    model: &str,
    provider_order: Option<&str>,
    allow_fallbacks: Option<bool>,
) -> ModelRouting {
    ModelRouting {
        model: model.into(),
        provider_order: provider_order.map(String::from),
        allow_fallbacks,
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
    /// representative field of each pattern (`non_empty`, `or`, `fixed`,
    /// `list_or`).
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
            reload.media_transcription_model(),
            DEFAULT_MEDIA_TRANSCRIPTION_MODEL,
            "unset media_transcription_model falls back to default"
        );

        // ── fixed: always returns the constant, ignoring persisted values ──
        let mut fixed_cfg = ConfigData::STRUCT_FIELDS_DEFAULT;
        assert!(fixed_cfg.set_string_field("provider_endpoint", "https://custom.example/v1"));
        reload.swap(fixed_cfg);
        assert_eq!(
            reload.provider_endpoint(),
            DEFAULT_PROVIDER_ENDPOINT,
            "fixed field ignores a persisted custom value"
        );

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

    // NOTE: Per-struct normalize tests (`role_config_normalize`,
    // `model_routing_normalize`) have been intentionally removed as
    // redundant.  Both `normalize()` methods are one-line delegations to
    // `non_empty()` with no conditional logic.  The `non_empty` / `trimmed_or_none`
    // primitive is covered exhaustively by `trimmed_or_none_trims_whitespace`
    // above, and the end-to-end integration through `normalize_entries()` is
    // covered by `normalize_entries_works` below.  If a new normalization
    // scenario is added, it should be added to the primitive test AND
    // exercised through the integration test — there is no need for
    // per-struct test duplication.

    /// Verify that [`ConfigData::normalize_entries`] normalises every entry in
    /// `per_role_configs` and `model_routings`.
    #[test]
    fn normalize_entries_works() {
        let mut config = ConfigData {
            per_role_configs: vec![
                RoleConfig {
                    role: "engineer".into(),
                    model: Some(String::new()),
                    reasoning_effort: Some("  high  ".into()),
                },
                RoleConfig {
                    role: "manager".into(),
                    model: Some("  test-model  ".into()),
                    reasoning_effort: None,
                },
            ],
            model_routings: vec![ModelRouting {
                model: "test-model".into(),
                provider_order: Some("   ".into()),
                allow_fallbacks: None,
            }],
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };

        config.normalize_entries();

        // First role: empty model → None, whitespace reasoning_effort → trimmed
        assert_eq!(config.per_role_configs[0].model, None);
        assert_eq!(
            config.per_role_configs[0].reasoning_effort,
            Some("high".into())
        );

        // Second role: trimmed model preserved, None stays None
        assert_eq!(config.per_role_configs[1].model, Some("test-model".into()));
        assert_eq!(config.per_role_configs[1].reasoning_effort, None);

        // Routing: whitespace-only provider_order → None
        assert_eq!(config.model_routings[0].provider_order, None);
    }

    // ── Upsert three-scenario tests ─────────────────────────
    //
    // Each upsert method (RoleConfig::upsert, ModelRouting::upsert)
    // is tested across three scenarios:
    //   1. updates_existing — existing entry, upsert sets one field,
    //      the other field is preserved unchanged
    //   2. pushes_new_entry — empty vec, new entry is pushed with the
    //      target field set and the other field None
    //   3. can_set_none — existing entry has both fields set to non-None;
    //      clearing one field via None leaves the other field unchanged

    #[test]
    fn upsert_role_config_fields() {
        // 1a. updates_existing — set model preserves reasoning_effort
        {
            let mut items = vec![role_config("engineer", Some("old"), Some("high"))];
            RoleConfig::upsert(&mut items, "engineer", |item| {
                item.model = Some("new".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].model,
                Some("new".into()),
                "[model] target field updated"
            );
            assert_eq!(
                items[0].reasoning_effort,
                Some("high".into()),
                "[model] other field preserved"
            );
        }
        // 1b. updates_existing — set reasoning_effort preserves model
        {
            let mut items = vec![role_config("engineer", Some("test-model"), Some("low"))];
            RoleConfig::upsert(&mut items, "engineer", |item| {
                item.reasoning_effort = Some("high".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].reasoning_effort,
                Some("high".into()),
                "[reasoning_effort] target field updated"
            );
            assert_eq!(
                items[0].model,
                Some("test-model".into()),
                "[reasoning_effort] other field preserved"
            );
        }

        // 2a. pushes_new_entry — model set, reasoning_effort is None
        {
            let mut items = vec![];
            RoleConfig::upsert(&mut items, "engineer", |item| {
                item.model = Some("test-model".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].role, "engineer");
            assert_eq!(
                items[0].model,
                Some("test-model".into()),
                "[model] set on new entry"
            );
            assert_eq!(items[0].reasoning_effort, None);
        }
        // 2b. pushes_new_entry — reasoning_effort set, model is None
        {
            let mut items = vec![];
            RoleConfig::upsert(&mut items, "engineer", |item| {
                item.reasoning_effort = Some("high".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].role, "engineer");
            assert_eq!(
                items[0].reasoning_effort,
                Some("high".into()),
                "[reasoning_effort] set on new entry"
            );
            assert_eq!(items[0].model, None);
        }

        // 3a. can_set_none — clear model, reasoning_effort preserved
        {
            let mut items = vec![role_config("engineer", Some("test-model"), Some("high"))];
            RoleConfig::upsert(&mut items, "engineer", |item| item.model = None);
            assert_eq!(items[0].model, None, "[model] cleared to None");
            assert_eq!(
                items[0].reasoning_effort,
                Some("high".into()),
                "[model] other field preserved when clearing"
            );
        }
        // 3b. can_set_none — clear reasoning_effort, model preserved
        {
            let mut items = vec![role_config("engineer", Some("test-model"), Some("high"))];
            RoleConfig::upsert(&mut items, "engineer", |item| item.reasoning_effort = None);
            assert_eq!(
                items[0].reasoning_effort, None,
                "[reasoning_effort] cleared to None"
            );
            assert_eq!(
                items[0].model,
                Some("test-model".into()),
                "[reasoning_effort] other field preserved when clearing"
            );
        }
    }

    #[test]
    fn upsert_model_routing_fields() {
        // 1a. updates_existing — set provider_order preserves allow_fallbacks
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"), Some(true))];
            ModelRouting::upsert(&mut items, "test-model", |item| {
                item.provider_order = Some("Anthropic".into());
            });
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].provider_order,
                Some("Anthropic".into()),
                "[provider_order] target field updated"
            );
            assert_eq!(
                items[0].allow_fallbacks,
                Some(true),
                "[provider_order] other field preserved"
            );
        }
        // 1b. updates_existing — set allow_fallbacks preserves provider_order
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"), Some(true))];
            ModelRouting::upsert(&mut items, "test-model", |item| {
                item.allow_fallbacks = Some(false);
            });
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].allow_fallbacks,
                Some(false),
                "[allow_fallbacks] target field updated"
            );
            assert_eq!(
                items[0].provider_order,
                Some("OpenAi".into()),
                "[allow_fallbacks] other field preserved"
            );
        }

        // 2a. pushes_new_entry — provider_order set, allow_fallbacks is None
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
            assert_eq!(items[0].allow_fallbacks, None);
        }
        // 2b. pushes_new_entry — allow_fallbacks set, provider_order is None
        {
            let mut items = vec![];
            ModelRouting::upsert(&mut items, "test-model", |item| {
                item.allow_fallbacks = Some(false);
            });
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].model, "test-model");
            assert_eq!(
                items[0].allow_fallbacks,
                Some(false),
                "[allow_fallbacks] set on new entry"
            );
            assert_eq!(items[0].provider_order, None);
        }

        // 3a. can_set_none — clear provider_order, allow_fallbacks preserved
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"), Some(true))];
            ModelRouting::upsert(&mut items, "test-model", |item| item.provider_order = None);
            assert_eq!(
                items[0].provider_order, None,
                "[provider_order] cleared to None"
            );
            assert_eq!(
                items[0].allow_fallbacks,
                Some(true),
                "[provider_order] other field preserved when clearing"
            );
        }
        // 3b. can_set_none — clear allow_fallbacks, provider_order preserved
        {
            let mut items = vec![model_routing("test-model", Some("OpenAi"), Some(true))];
            ModelRouting::upsert(&mut items, "test-model", |item| item.allow_fallbacks = None);
            assert_eq!(
                items[0].allow_fallbacks, None,
                "[allow_fallbacks] cleared to None"
            );
            assert_eq!(
                items[0].provider_order,
                Some("OpenAi".into()),
                "[allow_fallbacks] other field preserved when clearing"
            );
        }
    }

    #[test]
    fn upsert_multiple_entries_independent_keys() {
        let mut configs = vec![
            role_config("engineer", Some("model-a"), None),
            role_config("manager", Some("model-b"), Some("high")),
        ];
        let mut routings = vec![
            model_routing("test-router-a", Some("OpenAi"), None),
            model_routing("test-router-b", Some("Anthropic"), Some(true)),
        ];

        // Each upsert targets exactly one entry by key.
        RoleConfig::upsert(&mut configs, "engineer", |c| {
            c.model = Some("model-c".into());
        });
        assert_eq!(configs[0].model, Some("model-c".into()));
        assert_eq!(configs[1].model, Some("model-b".into()));

        RoleConfig::upsert(&mut configs, "manager", |c| {
            c.reasoning_effort = Some("low".into());
        });
        assert_eq!(configs[1].reasoning_effort, Some("low".into()));
        assert_eq!(configs[0].reasoning_effort, None);

        ModelRouting::upsert(&mut routings, "test-router-a", |mr| {
            mr.provider_order = Some("Google".into());
        });
        assert_eq!(routings[0].provider_order, Some("Google".into()));
        assert_eq!(routings[1].provider_order, Some("Anthropic".into()));

        ModelRouting::upsert(&mut routings, "test-router-b", |mr| {
            mr.allow_fallbacks = Some(false);
        });
        assert_eq!(routings[0].allow_fallbacks, None);
        assert_eq!(routings[1].allow_fallbacks, Some(false));

        // Total entries unchanged — no spurious pushes.
        assert_eq!(configs.len(), 2);
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
            result.unwrap(),
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
}

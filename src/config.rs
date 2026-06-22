use crate::Role;
use crate::role::role_info;
use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use tokio::fs;

// ── Hardcoded defaults ───────────────────────────────────────────

pub(crate) const DEFAULT_PROVIDER_ENDPOINT: &str = "https://openrouter.ai/api/v1";

const DEFAULT_IMAGE_GEN_MODEL: &str = "google/gemini-3.1-flash-image-preview";
const DEFAULT_VIDEO_GEN_MODEL: &str = "google/veo-3.1-lite";
pub(crate) const DEFAULT_IMAGE_TRANSCRIPTION_MODEL: &str = "qwen/qwen3.6-plus";
pub(crate) const DEFAULT_AUDIO_TRANSCRIPTION_MODEL: &str = "xiaomi/mimo-v2.5";

// ── Named config structs ───────────────────────────────────────────

/// A per-role model & reasoning-effort override.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleConfig {
    pub role: String,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
}

/// A per-model provider routing rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRouting {
    pub model: String,
    pub provider_order: Option<String>,
    pub allow_fallbacks: Option<bool>,
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
/// 2. **Macro** — add the field name to the `string_config_fields!` invocation (~line 200).
///    The macro generates [`ConfigData::STRUCT_FIELDS_DEFAULT`] (used by [`ConfigReload::const_new`]),
///    [`ConfigData::string_fields()`], and [`ConfigData::set_string_field()`] from this list.  The compiler
///    enforces that every field on [`ConfigData`] is present in `STRUCT_FIELDS_DEFAULT`,
///    so forgetting this step is a compile error.
/// 3. **Typed accessor** — write a typed accessor on [`ConfigReload`] (see
///    "String config accessors" below) so that the public API stays uniform.
///
/// ## Transient `Option<String>` fields (non-persisted)
///
/// If a new `Option<String>` field is a runtime-only cache or otherwise NOT
/// meant to be persisted as a config KV pair, do NOT add it to
/// `string_config_fields!` and do NOT add it to [`ConfigData::STRUCT_FIELDS_DEFAULT`].
/// It will not be persisted through `save_and_reload` — this is intentional.
///
/// ## UX asymmetry warning
///
/// The GUI Settings page reads [`ConfigData`] directly via [`ConfigReload::snapshot`]
/// (all fields).  But [`save_and_reload`] persists fields **only** through
/// [`ConfigData::string_fields`], which is macro-generated.  A field missing from
/// the macro would appear editable in the GUI but silently discard its value on
/// every save.  The compiler guard on [`ConfigData::STRUCT_FIELDS_DEFAULT`] prevents this.
#[derive(Debug, Clone, Default)]
pub struct ConfigData {
    /// API key for the LLM provider.
    pub provider_key: Option<String>,
    /// Base URL for the OpenAI-compatible LLM provider.
    pub provider_endpoint: Option<String>,
    /// Image transcription model.
    pub image_transcription_model: Option<String>,
    /// Audio transcription model.
    pub audio_transcription_model: Option<String>,
    /// OpenRouter provider routing for vision/transcription requests.
    pub transcription_provider: Option<String>,
    /// Audio transcription provider routing.
    pub audio_transcription_provider: Option<String>,
    /// Image generation model.
    pub image_gen_model: Option<String>,
    /// Newline-separated list of available image generation models (for selection UI).
    pub image_gen_models: Option<String>,
    /// Video generation model.
    pub video_gen_model: Option<String>,
    /// Newline-separated list of available video generation models (for selection UI).
    pub video_gen_models: Option<String>,
    /// Exa API key for web search.
    pub exa_key: Option<String>,
    /// Telegram Bot API token (hot-reloaded on save).
    pub telegram_bot_token: Option<String>,
    /// Per-role model overrides.
    pub per_role_configs: Vec<RoleConfig>,
    /// Per-model provider routing.
    /// `allow_fallbacks` is `None` when unset (defaults to `false` at request time).
    pub model_routings: Vec<ModelRouting>,
}

// ── String config field mapping ──────────────────────────────────
//
// The three runtime sync items (`STRUCT_FIELDS_DEFAULT`, `string_fields()`,
// `set_string_field()`) plus the test-only `STRING_CONFIG_KEYS` constant
// are generated from a single field-name declaration by the
// `string_config_fields!` macro — adding or removing a field in the
// macro invocation updates all four automatically, eliminating the
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
// which clones every struct field directly.  But [`save_and_reload`]
// persists **only** through [`ConfigData::string_fields`] (macro-generated).
// A field missing from the macro would appear editable in the UI but
// silently discard on save.  The compiler guard above prevents this.
//
// Per-field accessors on [`ConfigReload`] are written explicitly
// below (see "String config accessors") so IDE navigation works.

/// Generate the runtime sync methods `string_fields()` and `set_string_field()`,
/// the const `STRUCT_FIELDS_DEFAULT`, and the test-only `STRING_CONFIG_KEYS`
/// constant from a single list of `Option<String>` field names.
///
/// Each field name corresponds to a field on [`ConfigData`] and becomes the
/// database key (via `stringify!`).  All generated items are guaranteed to stay
/// synchronised because they expand from the same source.
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
    ($($field:ident),* $(,)?) => {
        /// All database keys that [`ConfigData::string_fields`] recognises.
        #[cfg(test)]
        pub(crate) const STRING_CONFIG_KEYS: &[&str] = &[$(stringify!($field)),*];

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
            /// The value is passed through `non_empty` so that empty strings are
            /// stored as `None` (matching the read-path semantics of [`reload_from_db`]).
            ///
            /// `per_role_configs` and `model_routings` are **not** handled here —
            /// they live in separate database tables (`config_role` and `config_model_routing`).
            #[must_use]
            pub fn set_string_field(&mut self, key: &str, value: &str) -> bool {
                match key {
                    $(stringify!($field) => self.$field = non_empty(Some(value.to_owned())),)*
                    _ => return false,
                }
                true
            }

            /// Normalise all string fields in place: trim whitespace and collapse
            /// empty or whitespace-only values to `None`.
            ///
            /// This is equivalent to passing every field through [`set_string_field`]
            /// but avoids the intermediate enumeration, allocation, and key dispatch.
            /// The behaviour is identical — each field is run through [`non_empty`].
            pub(crate) fn normalize_string_fields(&mut self) {
                $(self.$field = non_empty(self.$field.take());)*
            }
        }
    };
}

string_config_fields! {
    provider_key,
    provider_endpoint,
    image_transcription_model,
    audio_transcription_model,
    transcription_provider,
    audio_transcription_provider,
    image_gen_model,
    image_gen_models,
    video_gen_model,
    video_gen_models,
    exa_key,
    telegram_bot_token,
}

// ── Config value helpers ────────────────────────────────────────────

/// Trim a string and return `None` if empty or whitespace-only.
/// This is the canonical primitive for string trimming helpers.
#[must_use]
pub(crate) fn trim_non_empty(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Treat an empty or whitespace-only string as `None`.
/// The value is trimmed before being returned.
/// Delegates to [`trim_non_empty`].
#[must_use]
fn non_empty(val: Option<String>) -> Option<String> {
    val.and_then(|s| trim_non_empty(&s))
}

/// Resolve a value with a fallback: use `val` if non-empty (after trimming), else `fallback`.
#[must_use]
fn resolve_or(val: Option<String>, fallback: &str) -> String {
    non_empty(val).unwrap_or_else(|| fallback.to_string())
}

/// Parse a newline-separated list field, falling back to a singular field, then to a hardcoded
/// default.
///
/// If `list_field` is `Some` and contains at least one non-empty line (after trimming), the parsed
/// lines are returned as a `Vec<String>`. Otherwise a single-element vec containing the resolved
/// value of `fallback_field` (or `default_value`) is returned.
#[must_use]
fn resolve_list_or(
    list_field: Option<&String>,
    fallback_field: Option<String>,
    default_value: &str,
) -> Vec<String> {
    if let Some(raw) = list_field {
        let parsed: Vec<String> = raw
            .split('\n')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    vec![resolve_or(fallback_field, default_value)]
}

/// Expand a leading tilde (`~`) to the user's home directory.
///
/// Checks `$HOME` first (Unix, Git Bash on Windows), then `$USERPROFILE`
/// (cmd.exe / PowerShell). If neither is set, returns the path unchanged
/// (which means `~`-prefixed entries will be skipped by callers that
/// check for expansion success).
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix('~') {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
        if let Ok(home) = home {
            return PathBuf::from(home).join(stripped.trim_start_matches('/'));
        }
    }
    PathBuf::from(path)
}

// ── ConfigReload — global singleton ──────────────────────────────

/// Global reloadable config singleton.
///
/// Replaces the old `OnceCell<Config>` (write-once). The `storage_root` is
/// immutable after startup; all other fields live in an `Arc<RwLock<ConfigData>>`
/// that can be atomically swapped at runtime.
pub static CONFIG: ConfigReload = ConfigReload::const_new();

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

    pub(crate) fn set_storage_root(&self, root: PathBuf) {
        self.storage_root
            .set(root)
            .expect("CONFIG storage_root already set");
    }

    // ── Snapshot access ─────────────────────────────────────────

    /// Get a read-locked snapshot of the current config.
    /// Prefer the typed accessors below for individual fields.
    fn read(&self) -> RwLockReadGuard<'_, ConfigData> {
        self.inner.read().expect("CONFIG inner poisoned")
    }

    /// Replace the entire config atomically (used during startup and reload).
    pub(crate) fn swap(&self, new_config: ConfigData) {
        *self.inner.write().expect("CONFIG inner poisoned") = new_config;
    }

    /// Get a full clone of the current config for serialisation / GUI display.
    #[must_use]
    pub fn snapshot(&self) -> ConfigData {
        self.read().clone()
    }

    /// Update a single string config field in-memory and atomically apply it.
    ///
    /// This is intentionally lightweight — it only mutates the in-memory
    /// [`ConfigData`] without touching the database or triggering provider
    /// warmup. Callers are responsible for persisting the change to the
    /// config DB separately (e.g. via [`crate::config_db::store().set_kv()`]).
    ///
    /// Returns `true` if the key was recognised, `false` otherwise (unknown
    /// keys are silently ignored for forward compatibility).
    #[must_use]
    pub fn set_string_field_and_apply(&self, key: &str, value: &str) -> bool {
        let mut config = self.snapshot();
        let recognized = config.set_string_field(key, value);
        if recognized {
            *self.inner.write().expect("CONFIG inner poisoned") = config;
        }
        recognized
    }

    // ── Provider routing (per-model) ──────────────────────────

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
            && !m.is_empty()
        {
            return m.clone();
        }
        role_info(&role).default_model.to_string()
    }

    /// Resolve the configured reasoning effort for a role.
    ///
    /// Returns `None` when unset (model defaults apply).
    /// Priority: per-role override → role info default.
    #[must_use]
    pub fn role_reasoning_effort(&self, role: Role) -> Option<String> {
        if let Some(rc) = self.find_role_config(role)
            && let Some(ref r) = rc.reasoning_effort
            && !r.is_empty()
        {
            return Some(r.clone());
        }
        Some(role_info(&role).default_reasoning_effort.to_string())
    }

    // ── String config accessors ──────────────────────────────────
    //
    // These are written explicitly (rather than generated by a macro) so that
    // IDE navigation (Ctrl+Click → definition) works and error messages point
    // to the actual accessor, not macro-expanded code.
    //
    // The compiler catches an accessor referencing a non-existent `ConfigData`
    // field, and `accessor_count_matches_field_count` covers the
    // forward direction: every field listed in `string_config_fields!` has a
    // corresponding accessor match arm.  The reverse direction — a field added
    // to `ConfigData` but omitted from `string_config_fields!` — is caught by
    // the compiler via [`ConfigData::STRUCT_FIELDS_DEFAULT`], which is a `const
    // Self { … }` that must list every struct field.  Adding a field to the
    // struct without adding it to the macro produces a compile error.

    /// Get the configured provider API key, with empty/whitespace values collapsed to `None`.
    #[must_use]
    pub fn provider_key(&self) -> Option<String> {
        non_empty(self.read().provider_key.clone())
    }

    /// Get the configured provider endpoint, falling back to the default.
    #[must_use]
    pub fn provider_endpoint(&self) -> String {
        resolve_or(
            self.read().provider_endpoint.clone(),
            DEFAULT_PROVIDER_ENDPOINT,
        )
    }

    /// Get the configured image transcription model, falling back to the default.
    #[must_use]
    pub fn image_transcription_model(&self) -> String {
        resolve_or(
            self.read().image_transcription_model.clone(),
            DEFAULT_IMAGE_TRANSCRIPTION_MODEL,
        )
    }

    /// Get the configured audio transcription model, falling back to the default.
    #[must_use]
    pub fn audio_transcription_model(&self) -> String {
        resolve_or(
            self.read().audio_transcription_model.clone(),
            DEFAULT_AUDIO_TRANSCRIPTION_MODEL,
        )
    }

    /// Get the configured transcription provider, with empty/whitespace values collapsed to
    /// `None`.
    #[must_use]
    pub fn transcription_provider(&self) -> Option<String> {
        non_empty(self.read().transcription_provider.clone())
    }

    /// Get the configured audio transcription provider, with empty/whitespace values
    /// collapsed to `None`.
    #[must_use]
    pub fn audio_transcription_provider(&self) -> Option<String> {
        non_empty(self.read().audio_transcription_provider.clone())
    }

    /// Get the configured image generation model, falling back to the default.
    #[must_use]
    pub fn image_gen_model(&self) -> String {
        resolve_or(self.read().image_gen_model.clone(), DEFAULT_IMAGE_GEN_MODEL)
    }

    /// Get the configured video generation model, falling back to the default.
    #[must_use]
    pub fn video_gen_model(&self) -> String {
        resolve_or(self.read().video_gen_model.clone(), DEFAULT_VIDEO_GEN_MODEL)
    }

    /// Get the list of available image generation models for selection UI.
    ///
    /// Returns the parsed newline-separated list from `image_gen_models` if set and
    /// non-empty, otherwise falls back to a vec containing the currently active model
    /// (or the hardcoded default).
    #[must_use]
    pub fn image_gen_models(&self) -> Vec<String> {
        let guard = self.read();
        resolve_list_or(
            guard.image_gen_models.as_ref(),
            guard.image_gen_model.clone(),
            DEFAULT_IMAGE_GEN_MODEL,
        )
    }

    /// Get the list of available video generation models for selection UI.
    ///
    /// Returns the parsed newline-separated list from `video_gen_models` if set and
    /// non-empty, otherwise falls back to a vec containing the currently active model
    /// (or the hardcoded default).
    #[must_use]
    pub fn video_gen_models(&self) -> Vec<String> {
        let guard = self.read();
        resolve_list_or(
            guard.video_gen_models.as_ref(),
            guard.video_gen_model.clone(),
            DEFAULT_VIDEO_GEN_MODEL,
        )
    }

    /// Get the configured Exa API key, with empty/whitespace values collapsed to `None`.
    #[must_use]
    pub fn exa_key(&self) -> Option<String> {
        non_empty(self.read().exa_key.clone())
    }

    /// Get the configured Telegram bot token, with empty/whitespace values collapsed to
    /// `None`.
    #[must_use]
    pub fn telegram_bot_token(&self) -> Option<String> {
        non_empty(self.read().telegram_bot_token.clone())
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
    CONFIG.swap(ConfigData::default());

    tracing::info!(
        "Config system initialised (storage root: {}).",
        mahbot_dir.display()
    );
    Ok(())
}

/// Reload config from the `config.db` database, atomically swapping the
/// runtime config. Called both at startup (after config_db init) and after
/// a GUI-driven save.
pub async fn reload_from_db() -> Result<()> {
    let store = crate::config_db::store();
    let mut config = ConfigData::default();

    // Load key-value pairs
    let kvs = store.get_all_kv().await?;
    for (key, value) in &kvs {
        if !config.set_string_field(key, value) {
            tracing::debug!(key, "Unknown config key, ignoring");
        }
    }

    // Load per-role configs
    let roles = store.get_all_role_configs().await?;
    config.per_role_configs = roles;

    // Load per-model provider routings
    let routings = store.get_all_model_routings().await?;
    config.model_routings = routings;

    // Atomically swap
    CONFIG.swap(config);
    tracing::info!("Config reloaded from DB");
    Ok(())
}

/// Persist a [`ConfigData`] snapshot to the config database, reload runtime
/// config, and recreate provider/transcriber singletons.
///
/// Atomicity guarantee: provider recreation is attempted **before** committing
/// to DB. If warmup fails (bad key, network error), the DB and CONFIG are
/// unchanged and the existing provider continues serving.
///
/// If the Telegram bot token changed, the listener is hot-reloaded after the
/// config is persisted — no full application restart required.
///
/// Flow:
/// 1. Validate config values
/// 2. Validate new Telegram token (if changed) — fails early without DB mutation
/// 3. Warm-up a temporary provider (no global swap yet)
/// 4. On success: write to DB, reload CONFIG, swap singletons
/// 5. If Telegram token changed: hot-reload the listener
pub async fn save_and_reload(config: &ConfigData) -> Result<()> {
    // Validate
    validate_config(config)?;

    // Capture old Telegram token BEFORE we mutate DB so we can detect
    // changes and trigger hot-reload after persistence succeeds.
    let old_token = CONFIG.telegram_bot_token();

    // If the token changed and the new token is non-empty, validate it
    // early — before any DB write. If validation fails, nothing has been
    // mutated and the existing listener keeps running.
    if config.telegram_bot_token != old_token
        && let Some(ref new_token) = config.telegram_bot_token
    {
        crate::channels::telegram::TelegramChannel::validate_token(new_token).await?;
    }

    // ── Pre-commit warmup (no global swap) ─────────────────────
    // If this fails, nothing has changed — CONFIG, PROVIDER, and DB are untouched.
    crate::providers::warmup_provider_from_config(config).await?;

    // ── Persist to DB ─────────────────────────────────────────
    let store = crate::config_db::store();

    // Write all KV pairs
    for (key, value) in config.string_fields() {
        if let Some(v) = value.filter(|v| !v.is_empty()) {
            store.set_kv(key, v).await?;
        } else {
            store.delete_kv(key).await?;
        }
    }

    // Write per-role configs AND per-model routings inside a single
    // transaction so a crash between writes doesn't leave inconsistent state.
    {
        let tx = store.begin_tx().await?;
        // ── config_role ──
        for role in Role::all_roles() {
            tx.execute(
                "DELETE FROM config_role WHERE role = ?1",
                crate::turso::params![role.as_str()],
            )
            .await?;
        }
        for rc in &config.per_role_configs {
            tx.execute(
                "INSERT INTO config_role (role, model, reasoning_effort) VALUES (?1, ?2, ?3)",
                turso::params![
                    rc.role.as_str(),
                    rc.model.as_deref(),
                    rc.reasoning_effort.as_deref()
                ],
            )
            .await?;
        }

        // ── config_model_routing ──
        tx.execute("DELETE FROM config_model_routing", crate::turso::params![])
            .await?;
        for mr in &config.model_routings {
            let allow_int = mr.allow_fallbacks.map(i32::from);
            tx.execute(
                "INSERT INTO config_model_routing (model, provider_order, allow_fallbacks) \
                 VALUES (?1, ?2, ?3)",
                crate::turso::params![mr.model.as_str(), mr.provider_order.as_deref(), allow_int,],
            )
            .await?;
        }

        tx.commit().await?;
    }

    // ── Commit to runtime ─────────────────────────────────────
    // Warmup succeeded above — now persist runtime config and swap singletons.
    //
    // We swap the in-memory `config` directly instead of re-reading from DB
    // (which would be redundant).  Apply the same normalisation that
    // reload_from_db's read path would perform so the behaviour is identical.
    let mut config = config.clone();

    // Normalise string fields: trim whitespace, collapse empty → None.
    // This matches reload_from_db's set_string_field → non_empty pipeline.
    config.normalize_string_fields();
    // Sort to match DB ORDER BY clauses (get_all_role_configs / get_all_model_routings).
    config.per_role_configs.sort_by(|a, b| a.role.cmp(&b.role));
    config.model_routings.sort_by(|a, b| a.model.cmp(&b.model));

    // Capture the new token before swapping so we can detect changes below.
    let new_token = config.telegram_bot_token.clone();
    CONFIG.swap(config);
    tracing::info!("Config reloaded from DB");
    crate::providers::recreate_all().await?;

    // ── Hot-reload Telegram listener if token changed ─────────
    // Do this AFTER the DB and CONFIG have been updated so the
    // running listener reflects the persisted state.
    if new_token != old_token {
        tracing::info!(
            old = ?old_token,
            new = ?new_token,
            "Telegram bot token changed — restarting listener",
        );
        crate::channels::telegram::restart_telegram_listener(new_token.as_deref()).await?;
    }

    tracing::info!("Config saved and reloaded successfully");
    Ok(())
}

/// Validate a [`ConfigData`] before persisting.
fn validate_config(config: &ConfigData) -> Result<()> {
    // Validate endpoint URL — basic sanity check
    if let Some(ref ep) = config.provider_endpoint
        && !ep.trim().is_empty()
        && !ep.starts_with("https://")
        && !ep.starts_with("http://")
    {
        anyhow::bail!("Provider endpoint must be a valid URL starting with https:// or http://");
    }

    // Validate API key — reject the placeholder "sk-..." pattern
    if let Some(ref key) = config.provider_key
        && (key.trim() == "sk-..." || key.trim().starts_with("sk-.."))
    {
        anyhow::bail!("Provider key is still the placeholder value — please set a real key");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared test values for every string config field, used by multiple roundtrip tests.
    /// Each entry is a (db_key, test_value) pair.
    const STRING_FIELD_TEST_VALUES: &[(&str, &str)] = &[
        ("provider_key", "sk-test-key"),
        ("provider_endpoint", "https://example.com/api"),
        ("image_transcription_model", "gpt-4-vision"),
        ("audio_transcription_model", "whisper-1"),
        ("transcription_provider", "OpenAI"),
        ("audio_transcription_provider", "Deepgram"),
        ("image_gen_model", "dall-e-3"),
        ("image_gen_models", "dall-e-3\nstable-diffusion\nmidjourney"),
        ("video_gen_model", "sora"),
        ("video_gen_models", "sora\npika\nrunway"),
        ("exa_key", "exa-test-key"),
        ("telegram_bot_token", "123:abc"),
    ];

    /// All string keys that [`ConfigData::string_fields`] returns must be
    /// round-trippable through [`ConfigData::set_string_field`]: setting each
    /// individually and reading back via [`ConfigData::string_fields`] must
    /// produce the same value (with empty strings collapsed to `None` via
    /// [`non_empty`]).
    #[test]
    fn string_fields_roundtrip() {
        let mut config = ConfigData::default();

        // Verify the initial state: all fields are None.
        for (_key, value) in config.string_fields() {
            assert!(value.is_none(), "field should start as None");
        }

        // Set each field to a known value via set_string_field and verify
        // it round-trips back through string_fields.
        // Defensive completeness check — every known key must be exercised.
        assert_eq!(
            STRING_FIELD_TEST_VALUES.len(),
            STRING_CONFIG_KEYS.len(),
            "STRING_FIELD_TEST_VALUES must cover every entry in STRING_CONFIG_KEYS",
        );

        for &(key, value) in STRING_FIELD_TEST_VALUES {
            let recognized = config.set_string_field(key, value);
            assert!(recognized, "key '{key}' should be recognized");

            // Find this key in string_fields and verify the value matches.
            let found = config
                .string_fields()
                .iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| *v);
            assert_eq!(
                found,
                Some(value),
                "value for '{key}' should match after set"
            );
        }

        // Setting an empty string should result in None (non_empty semantics).
        let _ = config.set_string_field("provider_key", "");
        let pk = config
            .string_fields()
            .iter()
            .find(|(k, _)| *k == "provider_key")
            .and_then(|(_, v)| *v);
        assert!(pk.is_none(), "empty string should be stored as None");

        // Whitespace-only should also be None.
        let _ = config.set_string_field("provider_key", "   ");
        let pk = config
            .string_fields()
            .iter()
            .find(|(k, _)| *k == "provider_key")
            .and_then(|(_, v)| *v);
        assert!(
            pk.is_none(),
            "whitespace-only string should be stored as None"
        );

        // Unknown key returns false.
        assert!(!config.set_string_field("nonexistent_key", "value"));
    }

    /// All [`ConfigReload`] accessors exist and return the expected types.
    ///
    /// Calling each accessor explicitly catches signature regressions
    /// (e.g. `Option<String>` vs `String`). Iterates [`STRING_CONFIG_KEYS`] with a
    /// self-validating match so that adding a new field to `STRING_CONFIG_KEYS`,
    /// `string_fields()`, and `set_string_field()` without adding a match arm
    /// produces a clear failure ("unknown config key '{key}'"), rather than a
    /// cryptic "expected 10, got 11".
    #[test]
    fn accessor_count_matches_field_count() {
        let reload = ConfigReload::const_new();
        for &key in STRING_CONFIG_KEYS {
            match key {
                // ── non_empty accessors (Option<String>) ──
                "provider_key" => {
                    let _: Option<String> = reload.provider_key();
                }
                // ── or_default accessors (String) ──
                "provider_endpoint" => {
                    let _: String = reload.provider_endpoint();
                }
                "image_transcription_model" => {
                    let _: String = reload.image_transcription_model();
                }
                "audio_transcription_model" => {
                    let _: String = reload.audio_transcription_model();
                }
                "image_gen_model" => {
                    let _: String = reload.image_gen_model();
                }
                "image_gen_models" => {
                    let _: Vec<String> = reload.image_gen_models();
                }
                "video_gen_model" => {
                    let _: String = reload.video_gen_model();
                }
                "video_gen_models" => {
                    let _: Vec<String> = reload.video_gen_models();
                }
                // ── non_empty accessors (Option<String>) ──
                "transcription_provider" => {
                    let _: Option<String> = reload.transcription_provider();
                }
                "audio_transcription_provider" => {
                    let _: Option<String> = reload.audio_transcription_provider();
                }
                "exa_key" => {
                    let _: Option<String> = reload.exa_key();
                }
                "telegram_bot_token" => {
                    let _: Option<String> = reload.telegram_bot_token();
                }
                _ => {
                    panic!("unknown config key '{key}' — add a ConfigReload accessor and match arm")
                }
            }
        }
    }

    /// Values set via [`ConfigData::set_string_field`] round-trip correctly through the
    /// typed [`ConfigReload`] accessors, and the two accessor patterns (non_empty,
    /// or_default) behave as documented.
    #[test]
    fn config_reload_accessors_roundtrip() {
        let reload = ConfigReload::const_new();
        let mut config = ConfigData::default();

        // Set every string field via the DB interface (set_string_field)
        // Defensive completeness check — every known key must be exercised.
        assert_eq!(
            STRING_FIELD_TEST_VALUES.len(),
            STRING_CONFIG_KEYS.len(),
            "STRING_FIELD_TEST_VALUES must cover every entry in STRING_CONFIG_KEYS",
        );
        for &(key, value) in STRING_FIELD_TEST_VALUES {
            assert!(
                config.set_string_field(key, value),
                "set_string_field failed for '{key}'"
            );
        }
        reload.swap(config);

        // ── non_empty: returns the set value ──
        assert_eq!(reload.provider_key(), Some("sk-test-key".to_string()));
        assert_eq!(reload.transcription_provider(), Some("OpenAI".to_string()));
        assert_eq!(
            reload.audio_transcription_provider(),
            Some("Deepgram".to_string())
        );
        assert_eq!(reload.exa_key(), Some("exa-test-key".to_string()));
        assert_eq!(reload.telegram_bot_token(), Some("123:abc".to_string()));

        // ── or_default: falls back when field is unset ──
        reload.swap(ConfigData::default());
        assert_eq!(
            reload.provider_endpoint(),
            DEFAULT_PROVIDER_ENDPOINT,
            "unset provider_endpoint should fall back to default"
        );
        assert_eq!(
            reload.image_transcription_model(),
            DEFAULT_IMAGE_TRANSCRIPTION_MODEL,
            "unset image_transcription_model should fall back to default"
        );
        assert_eq!(
            reload.audio_transcription_model(),
            DEFAULT_AUDIO_TRANSCRIPTION_MODEL,
            "unset audio_transcription_model should fall back to default"
        );
        assert_eq!(
            reload.image_gen_model(),
            DEFAULT_IMAGE_GEN_MODEL,
            "unset image_gen_model should fall back to default"
        );
        assert_eq!(
            reload.video_gen_model(),
            DEFAULT_VIDEO_GEN_MODEL,
            "unset video_gen_model should fall back to default"
        );

        // ── Vec accessors: when list field is unset, falls back to active model ──
        assert_eq!(
            reload.image_gen_models(),
            vec![DEFAULT_IMAGE_GEN_MODEL.to_string()],
            "unset image_gen_models should fall back to active model"
        );
        assert_eq!(
            reload.video_gen_models(),
            vec![DEFAULT_VIDEO_GEN_MODEL.to_string()],
            "unset video_gen_models should fall back to active model"
        );

        // When list field is set, returns parsed entries
        let mut list_config = ConfigData::default();
        assert!(list_config.set_string_field("image_gen_models", "model-a\nmodel-b\nmodel-c"));
        assert!(list_config.set_string_field("video_gen_models", "vid-x\nvid-y"));
        reload.swap(list_config);
        assert_eq!(
            reload.image_gen_models(),
            vec!["model-a", "model-b", "model-c"]
        );
        assert_eq!(reload.video_gen_models(), vec!["vid-x", "vid-y"]);

        // ── non_empty: empty/whitespace → None ──
        let mut empty_config = ConfigData::default();
        let _ = empty_config.set_string_field("provider_key", "");
        let _ = empty_config.set_string_field("transcription_provider", "");
        let _ = empty_config.set_string_field("audio_transcription_provider", "   ");
        let _ = empty_config.set_string_field("exa_key", "");
        let _ = empty_config.set_string_field("telegram_bot_token", "   ");
        reload.swap(empty_config);
        assert_eq!(reload.provider_key(), None);
        assert_eq!(reload.transcription_provider(), None);
        assert_eq!(reload.audio_transcription_provider(), None);
        assert_eq!(reload.exa_key(), None);
        assert_eq!(reload.telegram_bot_token(), None);

        // Directly assign empty provider_key to the struct field (bypassing
        // set_string_field) to verify the accessor itself applies non_empty.
        let pk_config = ConfigData {
            provider_key: Some(String::new()),
            ..Default::default()
        };
        reload.swap(pk_config);
        assert_eq!(
            reload.provider_key(),
            None,
            "empty string in struct field is collapsed to None by non_empty accessor"
        );
    }

    #[test]
    fn trim_non_empty_trims_whitespace() {
        // trim_non_empty is the canonical primitive — trims and returns None
        // for empty or whitespace-only strings.
        assert_eq!(trim_non_empty("  value  "), Some("value".to_string()));
        assert_eq!(trim_non_empty(" "), None);
        assert_eq!(trim_non_empty(""), None);

        // non_empty delegates to trim_non_empty — the only unique behaviour is
        // the Option→&str unwrap via and_then.
        assert_eq!(non_empty(None), None);

        // resolve_or delegates through non_empty — the only unique behaviour is
        // the fallback on None via unwrap_or_else.
        assert_eq!(resolve_or(None, "fallback"), "fallback");
    }

    #[test]
    fn set_string_field_and_apply_updates_in_memory() {
        let reload = ConfigReload::const_new();

        // Unknown key returns false and does nothing
        assert!(!reload.set_string_field_and_apply("nonexistent", "value"));

        // Known key returns true and updates the in-memory value
        assert!(reload.set_string_field_and_apply("image_gen_model", "test-model"));
        assert_eq!(reload.image_gen_model(), "test-model");

        // Empty string stores as None (via non_empty)
        assert!(reload.set_string_field_and_apply("image_gen_model", ""));
        assert_eq!(
            reload.image_gen_model(),
            "google/gemini-3.1-flash-image-preview"
        );
    }
}

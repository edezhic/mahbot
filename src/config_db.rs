//! Config key-value, per-role model overrides, and per-model routing rules
//! stored in `config.db`.
//!
//! Three tables:
//! - `config_kv` — generic key-value string pairs for runtime configuration.
//! - `config_role` — per-role model and reasoning_effort overrides.
//! - `config_model_routing` — per-model provider order and fallback settings.

use crate::config::{ModelRouting, RoleConfig};
use crate::turso::{self};
use anyhow::Result;

crate::define_store! {
    /// Global config store.
    pub static CONFIG_STORE: ConfigStore,
    db_name = "config",
    schema = SCHEMA,
    expect = "CONFIG_STORE not initialized — call init_global() first",
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS config_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_role (
    role             TEXT PRIMARY KEY,
    model            TEXT,
    reasoning_effort TEXT
);

CREATE TABLE IF NOT EXISTS config_model_routing (
    model              TEXT PRIMARY KEY,
    provider_order     TEXT,
    allow_fallbacks    INTEGER
);";

// ── Column index constants ──────────────────────────────────

// config_kv table (2-column SELECT: key, value)
crate::columns! {
    KV_COLUMNS [KV] {
        KEY   => "key",
        VALUE => "value",
    }
}

// config_role table (3-column SELECT: role, model, reasoning_effort)
crate::columns! {
    ROLE_CONFIG_COLUMNS [RC] {
        ROLE             => "role",
        MODEL            => "model",
        REASONING_EFFORT => "reasoning_effort",
    }
}

// config_model_routing table (3-column SELECT: model, provider_order, allow_fallbacks)
crate::columns! {
    MODEL_ROUTING_COLUMNS [MR] {
        MODEL           => "model",
        PROVIDER_ORDER  => "provider_order",
        ALLOW_FALLBACKS => "allow_fallbacks",
    }
}

// ── Shared row-parsing helpers ──────────────────────────────────

/// Parse a `RoleConfig` from a `config_role` row.
fn role_config_from_row(row: &turso::Row) -> Result<RoleConfig, ::turso::Error> {
    let role = row.get::<String>(COL_RC_ROLE)?;
    let model = row.get::<Option<String>>(COL_RC_MODEL)?;
    let reasoning_effort = row.get::<Option<String>>(COL_RC_REASONING_EFFORT)?;
    Ok(RoleConfig {
        role,
        model,
        reasoning_effort,
    })
}

/// Parse a `ModelRouting` from a `config_model_routing` row.
fn model_routing_from_row(row: &turso::Row) -> Result<ModelRouting, ::turso::Error> {
    let model = row.get::<String>(COL_MR_MODEL)?;
    let provider_order = row.get::<Option<String>>(COL_MR_PROVIDER_ORDER)?;
    let allow_fallbacks = row.get::<Option<bool>>(COL_MR_ALLOW_FALLBACKS)?;
    Ok(ModelRouting {
        model,
        provider_order,
        allow_fallbacks,
    })
}

/// Parse a `(key, value)` pair from a `config_kv` row.
fn kv_from_row(row: &turso::Row) -> Result<(String, String), ::turso::Error> {
    let key = row.get::<String>(COL_KV_KEY)?;
    let value = row.get::<String>(COL_KV_VALUE)?;
    Ok((key, value))
}

// ── UPSERT SQL constants ──────────────────────────────────

const SET_KV_SQL: &str = "INSERT INTO config_kv (key, value) VALUES (?1, ?2) \
     ON CONFLICT(key) DO UPDATE SET value = excluded.value";

const DELETE_KV_SQL: &str = "DELETE FROM config_kv WHERE key = ?1";

// ── Per-row UPSERT / DELETE (config_role, config_model_routing) ──

const UPSERT_ROLE_CONFIG_SQL: &str = "INSERT INTO config_role (role, model, reasoning_effort) VALUES (?1, ?2, ?3) \
     ON CONFLICT(role) DO UPDATE SET \
         model = excluded.model, reasoning_effort = excluded.reasoning_effort";

const DELETE_ROLE_CONFIG_SQL: &str = "DELETE FROM config_role WHERE role = ?1";

const UPSERT_MODEL_ROUTING_SQL: &str = "INSERT INTO config_model_routing (model, provider_order, allow_fallbacks) \
     VALUES (?1, ?2, ?3) \
     ON CONFLICT(model) DO UPDATE SET \
         provider_order = excluded.provider_order, allow_fallbacks = excluded.allow_fallbacks";

const DELETE_MODEL_ROUTING_SQL: &str = "DELETE FROM config_model_routing WHERE model = ?1";

impl ConfigStore {
    // ── config_kv ────────────────────────────────────────────

    /// Upsert a key-value pair.
    pub async fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(SET_KV_SQL, turso::params![key, value])
            .await?;
        Ok(())
    }

    /// Delete a key-value pair. Succeeds even if the key does not exist.
    pub async fn delete_kv(&self, key: &str) -> Result<()> {
        self.conn
            .execute(DELETE_KV_SQL, turso::params![key])
            .await?;
        Ok(())
    }

    /// Get a single value by key; returns `None` when the key is absent.
    pub async fn get_kv(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT value FROM config_kv WHERE key = ?1",
                turso::params![key],
                |row| row.get::<String>(0),
            )
            .await
    }

    /// Get all key-value pairs.
    pub async fn get_all_kv(&self) -> Result<Vec<(String, String)>> {
        self.get_all_rows(KV_COLUMNS, "config_kv", "key", kv_from_row)
            .await
    }

    // ── config_role ──────────────────────────────────────────

    /// Get all role config rows.
    pub async fn get_all_role_configs(&self) -> Result<Vec<RoleConfig>> {
        self.get_all_rows(
            ROLE_CONFIG_COLUMNS,
            "config_role",
            "role",
            role_config_from_row,
        )
        .await
    }

    // ── config_model_routing ──────────────────────────────────

    /// Get all model routing rows.
    pub async fn get_all_model_routings(&self) -> Result<Vec<ModelRouting>> {
        self.get_all_rows(
            MODEL_ROUTING_COLUMNS,
            "config_model_routing",
            "model",
            model_routing_from_row,
        )
        .await
    }

    // ── per-row save (role configs / model routings) ─────────────
    //
    // Used by the settings page's per-field autosave: each editable row is
    // persisted individually (UPSERT, or DELETE once both override columns
    // are `None` — an all-None row is indistinguishable from having no
    // override, and the role's built-in defaults resolve identically either
    // way).

    /// Save a single `config_role` row: UPSERT, or DELETE when both override
    /// fields are `None` (an all-None row is indistinguishable from having no
    /// override — the role's built-in defaults resolve identically either way).
    pub async fn save_role_config(
        &self,
        role: &str,
        model: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> Result<()> {
        if model.is_none() && reasoning_effort.is_none() {
            self.conn
                .execute(DELETE_ROLE_CONFIG_SQL, turso::params![role])
                .await?;
        } else {
            self.conn
                .execute(
                    UPSERT_ROLE_CONFIG_SQL,
                    turso::params![role, model, reasoning_effort],
                )
                .await?;
        }
        Ok(())
    }

    /// Save a single `config_model_routing` row: UPSERT, or DELETE when both
    /// override fields are `None`. `Some(false)` on `allow_fallbacks` is
    /// meaningful (explicitly disables fallbacks at request time) and is
    /// therefore preserved — only `None` + `None` deletes.
    pub async fn save_model_routing(
        &self,
        model: &str,
        provider_order: Option<&str>,
        allow_fallbacks: Option<bool>,
    ) -> Result<()> {
        if provider_order.is_none() && allow_fallbacks.is_none() {
            self.conn
                .execute(DELETE_MODEL_ROUTING_SQL, turso::params![model])
                .await?;
        } else {
            let allow_int = allow_fallbacks.map(i32::from);
            self.conn
                .execute(
                    UPSERT_MODEL_ROUTING_SQL,
                    turso::params![model, provider_order, allow_int],
                )
                .await?;
        }
        Ok(())
    }

    // ── config_kv ────────────────────────────────────────────

    /// Execute a read-only query with a row mapper, collecting all results into
    /// a `Vec`.  Shared implementation for all `get_all_*` methods.
    ///
    /// # Correctness
    ///
    /// `columns`, `table`, and `order_by` are always compile-time string
    /// literals supplied by the caller; they are never user-provided, so the
    /// `format!` injection is benign.
    async fn get_all_rows<T, E>(
        &self,
        columns: &str,
        table: &str,
        order_by: &str,
        parser: impl FnMut(&turso::Row) -> std::result::Result<T, E> + Send + 'static,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let sql = format!("SELECT {columns} FROM {table} ORDER BY {order_by}");
        self.conn
            .query_map_strict(&sql, turso::params![], parser)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{model_routing, role_config};
    use tempfile::TempDir;

    async fn setup() -> (ConfigStore, TempDir) {
        crate::open_test_store!(ConfigStore, "config")
    }

    // ── config_kv lifecycle ──────────────────────────────────
    //
    // KV storage uses the production set_kv/get_kv/delete_kv/get_all_kv path.

    #[tokio::test]
    async fn test_config_kv_lifecycle() {
        let (store, _dir) = setup().await;

        // 1. empty state
        let val = store.get_kv("nonexistent").await.unwrap();
        assert!(val.is_none(), "get_kv should return None for missing key");

        // 2. insert
        store.set_kv("alpha", "first").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert_eq!(
            val,
            Some("first".to_string()),
            "get_kv should return inserted value"
        );

        // 3. overwrite
        store.set_kv("alpha", "updated").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert_eq!(
            val,
            Some("updated".to_string()),
            "set_kv should overwrite existing key"
        );

        // 4. get_all with multiple items (sorted by key)
        store.set_kv("beta", "second").await.unwrap();
        let all = store.get_all_kv().await.unwrap();
        assert_eq!(
            all,
            vec![
                ("alpha".to_string(), "updated".to_string()),
                ("beta".to_string(), "second".to_string()),
            ],
            "get_all_kv should return all pairs sorted by key"
        );

        // 5. delete
        store.delete_kv("alpha").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert!(val.is_none(), "get_kv should return None after delete");

        // 6. delete non-existent key (no-op, must not error)
        store.delete_kv("never-existed").await.unwrap();

        // 7. delete already-deleted key (also a no-op)
        store.delete_kv("alpha").await.unwrap();

        // 8. remaining item still present
        let all = store.get_all_kv().await.unwrap();
        assert_eq!(
            all,
            vec![("beta".to_string(), "second".to_string())],
            "only the undeleted item should remain"
        );
    }

    // ── Per-row save (settings-page autosave) ─────────────────────

    /// `save_role_config` UPSERTs a row and DELETE-if-empty removes it once
    /// both override columns are `None`; clearing a single column keeps the
    /// row.
    #[tokio::test]
    async fn test_save_role_config_upsert_and_delete_if_empty() {
        let (store, _dir) = setup().await;

        // New row.
        store
            .save_role_config("engineer", Some("m1"), None)
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![role_config("engineer", Some("m1"), None)],
            "new row persisted"
        );

        // UPSERT over the same role replaces both columns.
        store
            .save_role_config("engineer", Some("m2"), Some("high"))
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![role_config("engineer", Some("m2"), Some("high"))],
            "upsert replaces both columns"
        );

        // Clearing only the model keeps the row (reasoning still set).
        store
            .save_role_config("engineer", None, Some("medium"))
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![role_config("engineer", None, Some("medium"))],
            "single-column clear keeps the row"
        );

        // Both None → row deleted (all-None row == no override).
        store
            .save_role_config("engineer", None, None)
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert!(all.is_empty(), "all-None row deleted");
    }

    /// `save_model_routing` UPSERTs a row and DELETE-if-empty removes it only
    /// when BOTH columns are `None` — `Some(false)` on `allow_fallbacks` is
    /// meaningful (explicitly disables fallbacks) and must survive.
    #[tokio::test]
    async fn test_save_model_routing_upsert_and_delete_if_empty() {
        let (store, _dir) = setup().await;

        // `Some(false)` is meaningful: row persists with order None.
        store
            .save_model_routing("test-model", None, Some(false))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![model_routing("test-model", None, Some(false))],
            "Some(false) allow_fallbacks persists with None order"
        );

        // UPSERT replaces both columns.
        store
            .save_model_routing("test-model", Some("OpenAI"), Some(true))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![model_routing("test-model", Some("OpenAI"), Some(true))],
            "upsert replaces both columns"
        );

        // Clearing only the order keeps the row (allow_fallbacks still set).
        store
            .save_model_routing("test-model", None, Some(true))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![model_routing("test-model", None, Some(true))],
            "single-column clear keeps the row"
        );

        // Both None → row deleted.
        store
            .save_model_routing("test-model", None, None)
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert!(all.is_empty(), "all-None row deleted");
    }

    /// The per-field persist path must never overwrite or delete the
    /// `wake_word_templates` key — it is owned exclusively by the voice
    /// pipeline (`persist_enrollment`). `persist_settled_string_field`
    /// refuses the key structurally; this test verifies both the guard and
    /// that persisting an unrelated field leaves the enrolled templates row
    /// untouched.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_never_touches_wake_word_templates() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        // Restore the global CONFIG on exit: the firecrawl_key persist below
        // mutates it, and leaving the key set with its DB row deleted would
        // leak a phantom web-search key into parallel agent tests.
        let original = crate::config::CONFIG.snapshot();

        // Simulate a freshly-enrolled template set (clean slate first — the
        // global test store is shared across serialized persist tests).
        let template_json =
            r#"{"templates":[{"name":"hey","embeddings":[[0.1]],"threshold":0.5}]}"#;
        store
            .set_kv("wake_word_templates", template_json)
            .await
            .unwrap();

        // 1. The guard itself: calling persist on the key is a no-op.
        crate::config::persist_settled_string_field("wake_word_templates", "garbage")
            .await
            .unwrap();
        let saved = store.get_kv("wake_word_templates").await.unwrap();
        assert_eq!(
            saved.as_deref(),
            Some(template_json),
            "guard must not overwrite the enrolled templates"
        );

        // 2. An unrelated field persist must not delete the templates row.
        crate::config::persist_settled_string_field("firecrawl_key", "fc-test-key")
            .await
            .unwrap();
        let saved = store.get_kv("wake_word_templates").await.unwrap();
        assert_eq!(
            saved.as_deref(),
            Some(template_json),
            "persisting another field must not delete wake_word_templates"
        );

        // 3. The unrelated field itself must be persisted.
        assert_eq!(
            store.get_kv("firecrawl_key").await.unwrap().as_deref(),
            Some("fc-test-key"),
            "the unrelated field itself must be persisted"
        );

        // Clean up the shared store.
        let _ = store.delete_kv("wake_word_templates").await;
        let _ = store.delete_kv("firecrawl_key").await;
        crate::config::CONFIG.swap(original);
    }

    // ── Per-field persist orchestration (global store) ─────────────
    //
    // `persist_settled_*` writes through the global `CONFIG_STORE` and the
    // global `CONFIG` singleton, so these tests share `init_test_stores()`
    // and are serialized (and clean up their rows) to avoid cross-test
    // interference.

    /// Invalid settled values are rejected before anything is written: the
    /// endpoint must be a valid URL and the provider key must not be the
    /// placeholder — the safety requirement "an invalid settled value must
    /// never be persisted".
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_rejects_invalid_values_before_writing() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let _ = store.delete_kv("provider_endpoint").await;
        let _ = store.delete_kv("provider_key").await;

        let err = crate::config::persist_settled_string_field("provider_endpoint", "not-a-url")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("valid URL"),
            "endpoint without scheme rejected: {err}"
        );
        assert!(
            store.get_kv("provider_endpoint").await.unwrap().is_none(),
            "rejected endpoint must not be written"
        );

        let err = crate::config::persist_settled_string_field("provider_key", "sk-or-v1-...")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("placeholder"),
            "placeholder key rejected: {err}"
        );
        assert!(
            store.get_kv("provider_key").await.unwrap().is_none(),
            "rejected key must not be written"
        );
    }

    /// A plain field (no side effects) persists its trimmed canonical value,
    /// and clearing it deletes the row.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_plain_field_trims_and_clears() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let _ = store.delete_kv("exa_key").await;
        // Restore the global CONFIG on exit: the persists below mutate
        // `CONFIG.exa_key` and must not leak the value into parallel tests.
        let original = crate::config::CONFIG.snapshot();

        let persisted = crate::config::persist_settled_string_field("exa_key", "  exa-abc  ")
            .await
            .unwrap();
        assert_eq!(persisted, "exa-abc", "canonical trimmed value returned");
        assert_eq!(
            store.get_kv("exa_key").await.unwrap().as_deref(),
            Some("exa-abc"),
            "trimmed value persisted"
        );
        assert_eq!(
            crate::config::CONFIG.exa_key().as_deref(),
            Some("exa-abc"),
            "in-memory CONFIG mirrors the persisted value"
        );

        let persisted = crate::config::persist_settled_string_field("exa_key", "   ")
            .await
            .unwrap();
        assert_eq!(persisted, "", "cleared value returns empty");
        assert!(
            store.get_kv("exa_key").await.unwrap().is_none(),
            "cleared value deletes the row"
        );
        crate::config::CONFIG.swap(original);
    }

    /// A per-role model settle preserves the row's reasoning effort from the
    /// live config (the two columns are edited independently), and clearing
    /// the model keeps the row when reasoning is still set.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_role_model_preserves_reasoning_effort() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let original = crate::config::CONFIG.snapshot();

        // Seed the live row (as a prior reasoning-effort settle would).
        crate::config::CONFIG.set_role_config_row(
            "engineer",
            Some("old-model".into()),
            Some("high".into()),
        );
        store
            .save_role_config("engineer", Some("old-model"), Some("high"))
            .await
            .unwrap();

        let persisted = crate::config::persist_settled_role_model("engineer", "new-model")
            .await
            .unwrap();
        assert_eq!(persisted, "new-model");
        let rows = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            rows,
            vec![role_config("engineer", Some("new-model"), Some("high"))],
            "model updated, reasoning_effort preserved"
        );

        // Clearing the model keeps the row (reasoning still set).
        crate::config::persist_settled_role_model("engineer", "")
            .await
            .unwrap();
        let rows = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            rows,
            vec![role_config("engineer", None, Some("high"))],
            "cleared model keeps the reasoning_effort row"
        );

        // Clearing the reasoning too deletes the empty row.
        crate::config::persist_settled_role_reasoning("engineer", "")
            .await
            .unwrap();
        let rows = store.get_all_role_configs().await.unwrap();
        assert!(rows.is_empty(), "all-None row removed");

        crate::config::CONFIG.swap(original);
    }

    /// A routing allow-fallbacks settle preserves the row's provider order,
    /// and `Some(false)` survives (explicitly disabling fallbacks is
    /// meaningful at request time).
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_routing_allow_preserves_order() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let original = crate::config::CONFIG.snapshot();

        crate::config::CONFIG.set_model_routing_row(
            "test-model",
            Some("OpenAI".into()),
            Some(false),
        );
        store
            .save_model_routing("test-model", Some("OpenAI"), Some(false))
            .await
            .unwrap();

        let persisted = crate::config::persist_settled_routing_allow("test-model", true)
            .await
            .unwrap();
        assert_eq!(persisted, "true", "canonical allow value returned");
        let rows = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            rows,
            vec![model_routing("test-model", Some("OpenAI"), Some(true))],
            "allow_fallbacks updated, provider_order preserved"
        );

        crate::config::persist_settled_routing_order("test-model", "")
            .await
            .unwrap();
        let rows = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            rows,
            vec![model_routing("test-model", None, Some(true))],
            "cleared order keeps the allow_fallbacks row"
        );

        crate::config::persist_settled_routing_allow("test-model", false)
            .await
            .unwrap();
        let rows = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            rows,
            vec![model_routing("test-model", None, Some(false))],
            "Some(false) is meaningful and preserved"
        );

        crate::config::persist_settled_routing_allow("test-model", false)
            .await
            .unwrap();
        crate::config::CONFIG.set_model_routing_row("test-model", None, None);
        store
            .save_model_routing("test-model", None, None)
            .await
            .unwrap();
        let rows = store.get_all_model_routings().await.unwrap();
        assert!(rows.is_empty(), "all-None routing row removed");

        crate::config::CONFIG.swap(original);
    }
}

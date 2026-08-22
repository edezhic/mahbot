//! Config key-value pairs and per-model routing rules stored in `config.db`.
//!
//! Two tables:
//! - `config_kv` — generic key-value string pairs for runtime configuration.
//! - `config_model_routing` — per-model provider order.
//!
//! The `config_role` table (per-role model overrides) remains in the schema
//! but is intentionally unreferenced since mahbot-1822: its rows are inert
//! orphans (schema unchanged, rows untouched, no read or write path).

use crate::config::{
    CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, CONFIG_KEY_VOICE_ENABLED, ModelRouting,
};
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

// config_model_routing table (2-column SELECT: model, provider_order)
crate::columns! {
    MODEL_ROUTING_COLUMNS [MR] {
        MODEL          => "model",
        PROVIDER_ORDER => "provider_order",
    }
}

// ── Shared row-parsing helpers ──────────────────────────────────

/// Parse a `ModelRouting` from a `config_model_routing` row.
fn model_routing_from_row(row: &turso::Row) -> Result<ModelRouting, ::turso::Error> {
    let model = row.get::<String>(COL_MR_MODEL)?;
    let provider_order = row.get::<Option<String>>(COL_MR_PROVIDER_ORDER)?;
    Ok(ModelRouting {
        model,
        provider_order,
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

const MIGRATE_KV_IF_EQUALS_SQL: &str = "UPDATE config_kv SET value = ?3 \
     WHERE key = ?1 AND value = ?2";

// ── Per-row UPSERT / DELETE (config_model_routing) ──

const UPSERT_MODEL_ROUTING_SQL: &str = "INSERT INTO config_model_routing (model, provider_order) \
     VALUES (?1, ?2) \
     ON CONFLICT(model) DO UPDATE SET provider_order = excluded.provider_order";

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

    /// Conditionally rewrite a `config_kv` value: set `value` to `new_value`
    /// only when the existing row holds exactly `old_value`. Returns the
    /// number of rows changed (`0` when absent, already at `new_value`, or
    /// holding a different value). Idempotent by construction: once every
    /// matching row is rewritten, later calls are no-ops. Returns the count
    /// (rather than `()`) so the caller can log when it actually changes
    /// something.
    pub async fn migrate_kv_if_equals(
        &self,
        key: &str,
        old_value: &str,
        new_value: &str,
    ) -> Result<u64> {
        let changed = self
            .conn
            .execute(
                MIGRATE_KV_IF_EQUALS_SQL,
                turso::params![key, old_value, new_value],
            )
            .await?;
        Ok(changed)
    }

    /// Persist the transcription toggle and its wake-word cascade atomically
    /// (mahbot-1825).
    ///
    /// Row semantics are unchanged: enabling deletes the `audio_transcription_use_local`
    /// row (absence = enabled), disabling writes `"false"`. When turning
    /// transcription OFF while wake word was on, `voice_enabled` is deleted in the
    /// same transaction — a failure then rolls back both keys, so the settings UI
    /// can never diverge from the DB.
    pub async fn set_transcription_toggle(
        &self,
        transcription_enabled: bool,
        cascade_voice_off: bool,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if transcription_enabled {
            tx.execute(
                DELETE_KV_SQL,
                turso::params![CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL],
            )
            .await?;
        } else {
            tx.execute(
                SET_KV_SQL,
                turso::params![CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "false"],
            )
            .await?;
        }
        if cascade_voice_off {
            tx.execute(DELETE_KV_SQL, turso::params![CONFIG_KEY_VOICE_ENABLED])
                .await?;
        }
        tx.commit().await?;
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

    // ── per-row save (model routings) ────────────────────────────
    //
    // Used by the settings page's per-field autosave: each editable row is
    // persisted individually (UPSERT, or DELETE once the order column is
    // `None` — an all-None row is indistinguishable from having no override,
    // and the provider layer's built-in defaults resolve identically either
    // way).

    /// Save a single `config_model_routing` row: UPSERT, or DELETE when the
    /// `provider_order` column is `None`.
    pub async fn save_model_routing(
        &self,
        model: &str,
        provider_order: Option<&str>,
    ) -> Result<()> {
        if provider_order.is_none() {
            self.conn
                .execute(DELETE_MODEL_ROUTING_SQL, turso::params![model])
                .await?;
        } else {
            self.conn
                .execute(
                    UPSERT_MODEL_ROUTING_SQL,
                    turso::params![model, provider_order],
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
    use crate::config::model_routing;
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

    /// `migrate_kv_if_equals` only rewrites a row whose value equals `old_value`,
    /// and reports the affected-row count (0 = no-op).
    #[tokio::test]
    async fn test_migrate_kv_if_equals_conditional_rewrite() {
        let (store, _dir) = setup().await;

        // Row with the old value → rewritten, reports 1.
        store.set_kv("slot", "OLD").await.unwrap();
        let changed = store
            .migrate_kv_if_equals("slot", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 1, "exact match must be rewritten");
        assert_eq!(store.get_kv("slot").await.unwrap().as_deref(), Some("NEW"));

        // Already at target → no-op, reports 0.
        let changed = store
            .migrate_kv_if_equals("slot", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 0, "already at target must be untouched");

        // A different value (user override) → untouched, reports 0.
        store.set_kv("slot", "OTHER").await.unwrap();
        let changed = store
            .migrate_kv_if_equals("slot", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 0, "a different value must be untouched");
        assert_eq!(
            store.get_kv("slot").await.unwrap().as_deref(),
            Some("OTHER")
        );

        // Absent key → no-op, reports 0.
        let changed = store
            .migrate_kv_if_equals("absent", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 0, "an absent key must be a no-op");
    }

    // ── Per-row save (settings-page autosave) ─────────────────────

    /// `save_model_routing` UPSERTs a row with a provider order and DELETEs it
    /// once the order is cleared (`None` order == no override).
    #[tokio::test]
    async fn test_save_model_routing_upsert_and_delete_if_empty() {
        let (store, _dir) = setup().await;

        // New row.
        store
            .save_model_routing("test-model", Some("OpenAI"))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![model_routing("test-model", Some("OpenAI"))],
            "new row persisted"
        );

        // UPSERT replaces the order column.
        store
            .save_model_routing("test-model", Some("Anthropic"))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![model_routing("test-model", Some("Anthropic"))],
            "upsert replaces the order column"
        );

        // Order cleared → row deleted.
        store.save_model_routing("test-model", None).await.unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert!(all.is_empty(), "cleared order deletes the row");
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
        assert_eq!(
            persisted.value, "exa-abc",
            "canonical trimmed value returned"
        );
        assert!(
            persisted.warning.is_none(),
            "plain field carries no warning"
        );
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
        assert_eq!(persisted.value, "", "cleared value returns empty");
        assert!(
            store.get_kv("exa_key").await.unwrap().is_none(),
            "cleared value deletes the row"
        );
        crate::config::CONFIG.swap(original);
    }

    /// An unreachable custom chat endpoint is saved anyway (mahbot-1884): the
    /// persist path warns (non-fatal) instead of rejecting, so a self-hosted
    /// endpoint can be configured before its server is reachable. Port 1 →
    /// immediate connection refused — fast, no external network.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_unreachable_custom_endpoint_saves_with_warning() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let _ = store.delete_kv("provider_endpoint").await;
        let _ = store.delete_kv("provider_endpoint_key").await;
        // Restore the global CONFIG on exit: the persist below mutates it.
        let original = crate::config::CONFIG.snapshot();
        // The persist path rebuilds the provider/transcriber singletons too —
        // snapshot them so the test leaves the process globals untouched.
        let prev_provider = crate::providers::snapshot_provider_for_test();
        let prev_transcriber = crate::providers::snapshot_transcriber_for_test();

        let outcome = crate::config::persist_settled_string_field(
            "provider_endpoint",
            "http://127.0.0.1:1/v1",
        )
        .await
        .expect("unreachable endpoint must still save");
        assert_eq!(
            outcome.value, "http://127.0.0.1:1/v1",
            "the unreachable value is the canonical persisted value"
        );
        let warning = outcome
            .warning
            .expect("unreachable endpoint must carry a warning");
        assert!(
            warning.contains("unreachable"),
            "warning should mention unreachability, got: {warning}"
        );
        assert_eq!(
            store.get_kv("provider_endpoint").await.unwrap().as_deref(),
            Some("http://127.0.0.1:1/v1"),
            "saved despite being unreachable"
        );

        // Clean up the shared store and restore the global CONFIG +
        // provider/transcriber singletons.
        let _ = store.delete_kv("provider_endpoint").await;
        let _ = store.delete_kv("provider_endpoint_key").await;
        crate::config::CONFIG.swap(original);
        crate::providers::restore_provider_for_test(prev_provider);
        crate::providers::restore_transcriber_for_test(prev_transcriber);
    }
}

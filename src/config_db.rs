//! Config key-value pairs and per-model routing rules stored in the consolidated domain
//! database (`core.db`): `config_kv` for generic string pairs, `config_model_routing` for
//! per-model provider order.

use crate::config::ModelRouting;
use crate::config::{CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, CONFIG_KEY_VOICE_ENABLED};
use crate::db::{self};
use anyhow::Result;

crate::define_store! {
    pub static CONFIG_STORE: ConfigStore,
    expect = "CONFIG_STORE not initialized — call init_all_stores() first",
}

crate::columns! {
    KV_COLUMNS [KV] {
        KEY   => "key",
        VALUE => "value",
    }
}

crate::columns! {
    MODEL_ROUTING_COLUMNS [MR] {
        MODEL          => "model",
        PROVIDER_ORDER => "provider_order",
    }
}

fn model_routing_from_row(row: &db::Row) -> Result<ModelRouting, ::turso::Error> {
    let model = row.get::<String>(COL_MR_MODEL)?;
    let provider_order = row.get::<Option<String>>(COL_MR_PROVIDER_ORDER)?;
    Ok(ModelRouting {
        model,
        provider_order,
    })
}

fn kv_from_row(row: &db::Row) -> Result<(String, String), ::turso::Error> {
    let key = row.get::<String>(COL_KV_KEY)?;
    let value = row.get::<String>(COL_KV_VALUE)?;
    Ok((key, value))
}

const UPDATE_KV_SQL: &str = "UPDATE config_kv SET value = ?1 WHERE key = ?2";

const INSERT_KV_SQL: &str = "INSERT INTO config_kv (key, value) VALUES (?1, ?2) \
     ON CONFLICT(key) DO NOTHING";

const DELETE_KV_SQL: &str = "DELETE FROM config_kv WHERE key = ?1";

const MIGRATE_KV_IF_EQUALS_SQL: &str = "UPDATE config_kv SET value = ?3 \
     WHERE key = ?1 AND value = ?2";

const UPDATE_MODEL_ROUTING_SQL: &str =
    "UPDATE config_model_routing SET provider_order = ?1 WHERE model = ?2";

const INSERT_MODEL_ROUTING_SQL: &str = "INSERT INTO config_model_routing (model, provider_order) \
     VALUES (?1, ?2) \
     ON CONFLICT(model) DO NOTHING";

const DELETE_MODEL_ROUTING_SQL: &str = "DELETE FROM config_model_routing WHERE model = ?1";

impl ConfigStore {
    pub async fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .upsert_row(
                UPDATE_KV_SQL,
                || db::params![value, key],
                INSERT_KV_SQL,
                db::params![key, value],
            )
            .await?;
        Ok(())
    }

    pub async fn delete_kv(&self, key: &str) -> Result<()> {
        self.conn.execute(DELETE_KV_SQL, db::params![key]).await?;
        Ok(())
    }

    /// Set `value` to `new_value` only when the existing row holds exactly `old_value`,
    /// returning the number of rows changed (`0` when absent, already at `new_value`, or
    /// holding a different value). Idempotent by construction, and the count lets the caller
    /// log when it actually changes something.
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
                db::params![key, old_value, new_value],
            )
            .await?;
        Ok(changed)
    }

    /// Persist the transcription toggle and its wake-word cascade atomically. Enabling
    /// deletes the `audio_transcription_use_local` row (absence = enabled), disabling writes
    /// `"false"`; turning transcription OFF while the wake word was on deletes
    /// `voice_enabled` in the same transaction, so a failure rolls back both keys and the
    /// settings UI can never diverge from the DB.
    pub async fn set_transcription_toggle(
        &self,
        transcription_enabled: bool,
        cascade_voice_off: bool,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if transcription_enabled {
            tx.execute(
                DELETE_KV_SQL,
                db::params![CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL],
            )
            .await?;
        } else {
            tx.upsert_row(
                UPDATE_KV_SQL,
                || db::params!["false", CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL],
                INSERT_KV_SQL,
                db::params![CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, "false"],
            )
            .await?;
        }
        if cascade_voice_off {
            tx.execute(DELETE_KV_SQL, db::params![CONFIG_KEY_VOICE_ENABLED])
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_kv(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT value FROM config_kv WHERE key = ?1",
                db::params![key],
                |row| row.get::<String>(0),
            )
            .await
    }

    pub async fn get_all_kv(&self) -> Result<Vec<(String, String)>> {
        self.get_all_rows(KV_COLUMNS, "config_kv", "key", kv_from_row)
            .await
    }

    pub async fn get_all_model_routings(&self) -> Result<Vec<ModelRouting>> {
        self.get_all_rows(
            MODEL_ROUTING_COLUMNS,
            "config_model_routing",
            "model",
            model_routing_from_row,
        )
        .await
    }

    // Used by the settings page's per-field autosave: each editable row is persisted
    // individually (UPSERT, or DELETE once the order column is `None` — an all-None row is
    // indistinguishable from no override, and `ConfigReload::model_routing` resolves a missing
    // order to the model's default provider either way).

    /// UPSERT a single `config_model_routing` row, or DELETE it when `provider_order` is `None`.
    pub async fn save_model_routing(
        &self,
        model: &str,
        provider_order: Option<&str>,
    ) -> Result<()> {
        if provider_order.is_none() {
            self.conn
                .execute(DELETE_MODEL_ROUTING_SQL, db::params![model])
                .await?;
        } else {
            self.conn
                .upsert_row(
                    UPDATE_MODEL_ROUTING_SQL,
                    || db::params![provider_order, model],
                    INSERT_MODEL_ROUTING_SQL,
                    db::params![model, provider_order],
                )
                .await?;
        }
        Ok(())
    }

    /// Read-only query with a row mapper, collecting all results — shared by every
    /// `get_all_*`. `columns`, `table` and `order_by` are compile-time literals supplied by
    /// the caller, never user-provided, so the `format!` injection is benign.
    async fn get_all_rows<T, E>(
        &self,
        columns: &str,
        table: &str,
        order_by: &str,
        parser: impl FnMut(&db::Row) -> std::result::Result<T, E> + Send + 'static,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let sql = format!("SELECT {columns} FROM {table} ORDER BY {order_by}");
        self.conn
            .query_map_strict(&sql, db::params![], parser)
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

    // KV storage uses the production set_kv/get_kv/delete_kv/get_all_kv path.

    #[tokio::test]
    async fn test_config_kv_lifecycle() {
        let (store, _dir) = setup().await;

        let val = store.get_kv("nonexistent").await.unwrap();
        assert!(val.is_none(), "get_kv should return None for missing key");

        store.set_kv("alpha", "first").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert_eq!(
            val,
            Some("first".to_string()),
            "get_kv should return inserted value"
        );

        store.set_kv("alpha", "updated").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert_eq!(
            val,
            Some("updated".to_string()),
            "set_kv should overwrite existing key"
        );

        // get_all with multiple items (sorted by key). Only the keys this test owns are
        // pinned — migrations seed config rows of their own.
        store.set_kv("beta", "second").await.unwrap();
        let all = store.get_all_kv().await.unwrap();
        let mine: Vec<(String, String)> = all
            .iter()
            .filter(|(k, _)| k == "alpha" || k == "beta")
            .cloned()
            .collect();
        assert_eq!(
            mine,
            vec![
                ("alpha".to_string(), "updated".to_string()),
                ("beta".to_string(), "second".to_string()),
            ],
            "get_all_kv should return this test's pairs in key order"
        );
        assert!(
            all.windows(2).all(|w| w[0].0 <= w[1].0),
            "get_all_kv should return pairs sorted by key"
        );

        store.delete_kv("alpha").await.unwrap();
        let val = store.get_kv("alpha").await.unwrap();
        assert!(val.is_none(), "get_kv should return None after delete");

        store.delete_kv("never-existed").await.unwrap();

        store.delete_kv("alpha").await.unwrap();

        let all = store.get_all_kv().await.unwrap();
        let mine: Vec<(String, String)> = all
            .iter()
            .filter(|(k, _)| k == "alpha" || k == "beta")
            .cloned()
            .collect();
        assert_eq!(
            mine,
            vec![("beta".to_string(), "second".to_string())],
            "only the undeleted item should remain"
        );
    }

    /// `migrate_kv_if_equals` only rewrites a row whose value equals `old_value`, and reports
    /// the affected-row count (0 = no-op).
    #[tokio::test]
    async fn test_migrate_kv_if_equals_conditional_rewrite() {
        let (store, _dir) = setup().await;

        store.set_kv("slot", "OLD").await.unwrap();
        let changed = store
            .migrate_kv_if_equals("slot", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 1, "exact match must be rewritten");
        assert_eq!(store.get_kv("slot").await.unwrap().as_deref(), Some("NEW"));

        let changed = store
            .migrate_kv_if_equals("slot", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 0, "already at target must be untouched");

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

        let changed = store
            .migrate_kv_if_equals("absent", "OLD", "NEW")
            .await
            .unwrap();
        assert_eq!(changed, 0, "an absent key must be a no-op");
    }

    /// `save_model_routing` UPSERTs a row with a provider order and DELETEs it once the
    /// order is cleared.
    #[tokio::test]
    async fn test_save_model_routing_upsert_and_delete_if_empty() {
        let (store, _dir) = setup().await;

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

        store.save_model_routing("test-model", None).await.unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert!(all.is_empty(), "cleared order deletes the row");
    }

    /// The per-field persist path must never overwrite or delete the `wake_word_templates`
    /// key — the voice pipeline (`persist_enrollment`) owns it exclusively, and
    /// `persist_settled_string_field` refuses it structurally.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_never_touches_wake_word_templates() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        // Restore the global CONFIG on exit: the persist below mutates it, and a leftover key
        // with its DB row deleted would leak a phantom key into parallel agent tests.
        let original = crate::config::CONFIG.snapshot();

        let template_json =
            r#"{"templates":[{"name":"hey","embeddings":[[0.1]],"threshold":0.5}]}"#;
        store
            .set_kv("wake_word_templates", template_json)
            .await
            .unwrap();

        // The guard itself: persisting the key is a no-op — the DB row and the in-memory
        // CONFIG alike.
        crate::config::CONFIG.swap({
            let mut enrolled = crate::config::ConfigData::STRUCT_FIELDS_DEFAULT;
            assert!(enrolled.set_string_field("wake_word_templates", template_json));
            enrolled
        });
        let guarded = crate::config::persist_settled_string_field("wake_word_templates", "garbage")
            .await
            .unwrap();
        assert_eq!(
            guarded.value, template_json,
            "the guard reports the enrolled templates, not the refused value"
        );
        assert_eq!(
            crate::config::CONFIG.wake_word_templates().as_deref(),
            Some(template_json),
            "the guard must not alter the in-memory CONFIG either"
        );
        let saved = store.get_kv("wake_word_templates").await.unwrap();
        assert_eq!(
            saved.as_deref(),
            Some(template_json),
            "guard must not overwrite the enrolled templates"
        );

        crate::config::persist_settled_string_field("deepseek_key", "ds-test-key")
            .await
            .unwrap();
        let saved = store.get_kv("wake_word_templates").await.unwrap();
        assert_eq!(
            saved.as_deref(),
            Some(template_json),
            "persisting another field must not delete wake_word_templates"
        );

        assert_eq!(
            store.get_kv("deepseek_key").await.unwrap().as_deref(),
            Some("ds-test-key"),
            "the unrelated field itself must be persisted"
        );

        let _ = store.delete_kv("wake_word_templates").await;
        let _ = store.delete_kv("deepseek_key").await;
        crate::config::CONFIG.swap(original);
    }

    // `persist_settled_*` writes through the global `CONFIG_STORE` and `CONFIG` singleton,
    // so these tests share `init_test_stores()` and are serialized (cleaning their rows up) to
    // avoid cross-test interference.

    /// Invalid settled values are rejected before anything is written: the endpoint must be a
    /// valid URL and the provider key must not be the placeholder.
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

    /// A plain field persists its trimmed canonical value, and clearing it deletes the row.
    #[tokio::test]
    #[serial_test::serial(config_persist)]
    async fn test_persist_settled_plain_field_trims_and_clears() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let _ = store.delete_kv("exa_key").await;
        // Restore the global CONFIG on exit: the persists below mutate `CONFIG.exa_key` and
        // must not leak the value into parallel tests.
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

    /// An unreachable custom chat endpoint is saved anyway: the
    /// persist path warns (non-fatal) instead of rejecting, so a self-hosted
    /// endpoint can be configured before its server is reachable. Port 1 →
    /// immediate connection refused — fast, no external network.
    #[tokio::test]
    #[serial_test::serial(config_persist, provider)] // snapshots/restores the process-global PROVIDER singleton
    async fn test_persist_settled_unreachable_custom_endpoint_saves_with_warning() {
        crate::util::test::init_test_stores().await;
        let store = crate::config_db::store();
        let _ = store.delete_kv("provider_endpoint").await;
        let _ = store.delete_kv("provider_endpoint_key").await;
        // Restore the global CONFIG on exit: the persist below mutates it.
        let original = crate::config::CONFIG.snapshot();
        // The persist path rebuilds the provider singleton too — snapshot it
        // so the test leaves the process global untouched.
        let prev_provider = crate::providers::snapshot_provider_for_test();

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

        // Clean up the shared store and restore the global CONFIG + provider
        // singleton.
        let _ = store.delete_kv("provider_endpoint").await;
        let _ = store.delete_kv("provider_endpoint_key").await;
        crate::config::CONFIG.swap(original);
        crate::providers::restore_provider_for_test(prev_provider);
    }
}

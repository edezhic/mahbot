//! Config key-value and per-role model overrides stored in `config.db`.
//!
//! Two tables:
//! - `config_kv` — generic key-value string pairs for runtime configuration.
//! - `config_role` — per-role model and reasoning_effort overrides.

use crate::config::{CONFIG, ModelRouting, RoleConfig};
use crate::turso::{self, Connection};
use anyhow::Result;
use std::path::Path;
use tokio::sync::OnceCell;

/// Global config store.
pub static CONFIG_STORE: OnceCell<ConfigStore> = OnceCell::const_new();

/// Initialize the global config store.
pub async fn init_global() -> Result<()> {
    let root = CONFIG.global_storage_root();
    turso::register_global_store(&CONFIG_STORE, "CONFIG_STORE", || ConfigStore::open(&root)).await
}

/// Turso-backed config storage.
#[derive(Clone, Debug)]
pub struct ConfigStore {
    pub(crate) conn: Connection,
}

/// Returns a reference to the global config store.
///
/// # Panics
///
/// Panics if the config store has not been initialized. All production code
/// initializes the store before any access, so this is a programming error.
#[must_use]
pub fn store() -> &'static ConfigStore {
    CONFIG_STORE
        .get()
        .expect("CONFIG_STORE not initialized — call init_global() first")
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

impl ConfigStore {
    /// Open (or create) the config database at `root/db/config.db`.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/config.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;
        Ok(Self { conn })
    }

    /// Begin a transaction that serializes all subsequent operations until
    /// committed or rolled back. The returned guard keeps the connection locked.
    pub async fn begin_tx(&self) -> Result<turso::TxGuard<'_>> {
        Ok(self.conn.begin_tx().await?)
    }

    // ── config_kv ────────────────────────────────────────────

    /// Upsert a key-value pair.
    pub async fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO config_kv (key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                turso::params![key, value],
            )
            .await?;
        Ok(())
    }

    /// Delete a key-value pair. Succeeds even if the key does not exist.
    pub async fn delete_kv(&self, key: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM config_kv WHERE key = ?1", turso::params![key])
            .await?;
        Ok(())
    }

    /// Get all key-value pairs.
    pub async fn get_all_kv(&self) -> Result<Vec<(String, String)>> {
        let rows = self
            .conn
            .query(
                "SELECT key, value FROM config_kv ORDER BY key",
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| Ok((turso::row_text(&row, 0)?, turso::row_text(&row, 1)?)))
            .collect()
    }

    // ── config_role ──────────────────────────────────────────

    /// Get all role config rows.
    pub async fn get_all_role_configs(&self) -> Result<Vec<RoleConfig>> {
        let rows = self
            .conn
            .query(
                "SELECT role, model, reasoning_effort FROM config_role ORDER BY role",
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                let role = turso::row_text(&row, 0)?;
                let model = match row.get_value(1)? {
                    turso::Value::Text(s) => Some(s),
                    _ => None,
                };
                let reasoning_effort = match row.get_value(2)? {
                    turso::Value::Text(s) => Some(s),
                    _ => None,
                };
                Ok(RoleConfig {
                    role,
                    model,
                    reasoning_effort,
                })
            })
            .collect()
    }

    // ── config_model_routing ──────────────────────────────────

    /// Get all model routing rows.
    pub async fn get_all_model_routings(&self) -> Result<Vec<ModelRouting>> {
        let rows = self
            .conn
            .query(
                "SELECT model, provider_order, allow_fallbacks FROM config_model_routing ORDER BY model",
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                let model = turso::row_text(&row, 0)?;
                let provider_order = match row.get_value(1)? {
                    turso::Value::Text(s) => Some(s),
                    _ => None,
                };
                let allow_fallbacks = match row.get_value(2)? {
                    turso::Value::Integer(i) => Some(i != 0),
                    _ => None,
                };
                Ok(ModelRouting {
                    model,
                    provider_order,
                    allow_fallbacks,
                })
            })
            .collect()
    }
}

#[cfg(test)]
impl ConfigStore {
    // Get a single key-value pair by key. Returns `None` if not found.
    async fn get_kv(&self, key: &str) -> Result<Option<String>> {
        match self
            .conn
            .query_row(
                "SELECT value FROM config_kv WHERE key = ?1",
                turso::params![key],
                |row| row.get_value(0),
            )
            .await
        {
            Ok(turso::Value::Text(v)) => Ok(Some(v)),
            Ok(turso::Value::Null) | Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Ok(other) => anyhow::bail!("unexpected value type for key '{key}': {other:?}"),
            Err(e) => Err(e.into()),
        }
    }

    // Get the role config overrides for a role.
    // Returns `None` if no row exists for the role.
    async fn get_role_config(&self, role: &str) -> Result<Option<RoleConfig>> {
        let role_owned = role.to_string();
        match self
            .conn
            .query_row(
                "SELECT model, reasoning_effort FROM config_role WHERE role = ?1",
                turso::params![role],
                |row| {
                    let model = match row.get_value(0)? {
                        turso::Value::Text(s) => Some(s),
                        turso::Value::Null => None,
                        other => {
                            return Err(::turso::Error::Error(format!(
                                "expected text or null for model, got {other:?}"
                            )));
                        }
                    };
                    let reasoning_effort = match row.get_value(1)? {
                        turso::Value::Text(s) => Some(s),
                        turso::Value::Null => None,
                        other => {
                            return Err(::turso::Error::Error(format!(
                                "expected text or null for reasoning_effort, got {other:?}"
                            )));
                        }
                    };
                    Ok(RoleConfig {
                        role: role_owned,
                        model,
                        reasoning_effort,
                    })
                },
            )
            .await
        {
            Ok(rc) => Ok(Some(rc)),
            Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // Upsert a role config row. Passing `None` for a field sets it to NULL.
    async fn set_role_config(
        &self,
        role: &str,
        model: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO config_role (role, model, reasoning_effort) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(role) DO UPDATE SET \
                     model = excluded.model, \
                     reasoning_effort = excluded.reasoning_effort",
                turso::params![role, model, reasoning_effort],
            )
            .await?;
        Ok(())
    }

    // Delete a role config row. Succeeds even if the role does not exist.
    async fn delete_role_config(&self, role: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM config_role WHERE role = ?1",
                turso::params![role],
            )
            .await?;
        Ok(())
    }

    // Get the model routing config for a model.
    // Returns `None` if no row exists for the model.
    async fn get_model_routing(&self, model: &str) -> Result<Option<ModelRouting>> {
        let model_owned = model.to_string();
        match self
            .conn
            .query_row(
                "SELECT provider_order, allow_fallbacks FROM config_model_routing WHERE model = ?1",
                turso::params![model],
                |row| {
                    let provider_order = match row.get_value(0)? {
                        turso::Value::Text(s) => Some(s),
                        turso::Value::Null => None,
                        other => {
                            return Err(::turso::Error::Error(format!(
                                "expected text or null for provider_order, got {other:?}"
                            )));
                        }
                    };
                    let allow_fallbacks = match row.get_value(1)? {
                        turso::Value::Integer(i) => Some(i != 0),
                        turso::Value::Null => None,
                        other => {
                            return Err(::turso::Error::Error(format!(
                                "expected integer or null for allow_fallbacks, got {other:?}"
                            )));
                        }
                    };
                    Ok(ModelRouting {
                        model: model_owned,
                        provider_order,
                        allow_fallbacks,
                    })
                },
            )
            .await
        {
            Ok(mr) => Ok(Some(mr)),
            Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // Upsert a model routing row.
    async fn set_model_routing(
        &self,
        model: &str,
        provider_order: Option<&str>,
        allow_fallbacks: Option<bool>,
    ) -> Result<()> {
        let allow_fallbacks_int = allow_fallbacks.map(i32::from);
        self.conn
            .execute(
                "INSERT INTO config_model_routing (model, provider_order, allow_fallbacks) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(model) DO UPDATE SET \
                     provider_order = excluded.provider_order, \
                     allow_fallbacks = excluded.allow_fallbacks",
                turso::params![model, provider_order, allow_fallbacks_int],
            )
            .await?;
        Ok(())
    }

    // Delete a model routing row. Succeeds even if the model does not exist.
    async fn delete_model_routing(&self, model: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM config_model_routing WHERE model = ?1",
                turso::params![model],
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (ConfigStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = ConfigStore::open(dir.path()).await.unwrap();
        (store, dir)
    }

    #[tokio::test]
    async fn test_get_kv_empty() {
        let (store, _dir) = setup().await;
        let val = store.get_kv("nonexistent").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_kv() {
        let (store, _dir) = setup().await;
        store.set_kv("foo", "bar").await.unwrap();
        let val = store.get_kv("foo").await.unwrap();
        assert_eq!(val, Some("bar".to_string()));
    }

    #[tokio::test]
    async fn test_set_kv_overwrites() {
        let (store, _dir) = setup().await;
        store.set_kv("key1", "value1").await.unwrap();
        store.set_kv("key1", "value2").await.unwrap();
        let val = store.get_kv("key1").await.unwrap();
        assert_eq!(val, Some("value2".to_string()));
    }

    #[tokio::test]
    async fn test_delete_kv() {
        let (store, _dir) = setup().await;
        store.set_kv("todelete", "yep").await.unwrap();
        store.delete_kv("todelete").await.unwrap();
        let val = store.get_kv("todelete").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_delete_kv_nonexistent() {
        let (store, _dir) = setup().await;
        store.delete_kv("ghost").await.unwrap();
    }

    #[tokio::test]
    async fn test_get_all_kv() {
        let (store, _dir) = setup().await;
        store.set_kv("a", "1").await.unwrap();
        store.set_kv("b", "2").await.unwrap();
        let all = store.get_all_kv().await.unwrap();
        assert_eq!(
            all,
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn test_get_role_config_empty() {
        let (store, _dir) = setup().await;
        let val = store.get_role_config("engineer").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_role_config() {
        let (store, _dir) = setup().await;
        store
            .set_role_config("engineer", Some("gpt-4"), Some("high"))
            .await
            .unwrap();
        let val = store.get_role_config("engineer").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "engineer".to_string(),
                model: Some("gpt-4".to_string()),
                reasoning_effort: Some("high".to_string()),
            })
        );
    }

    #[tokio::test]
    async fn test_set_role_config_partial() {
        let (store, _dir) = setup().await;
        store
            .set_role_config("coder", Some("claude-3"), None)
            .await
            .unwrap();
        let val = store.get_role_config("coder").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "coder".to_string(),
                model: Some("claude-3".to_string()),
                reasoning_effort: None,
            })
        );
    }

    #[tokio::test]
    async fn test_set_role_config_overwrites() {
        let (store, _dir) = setup().await;
        store
            .set_role_config("qa", Some("model-a"), Some("low"))
            .await
            .unwrap();
        store
            .set_role_config("qa", Some("model-b"), None)
            .await
            .unwrap();
        let val = store.get_role_config("qa").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "qa".to_string(),
                model: Some("model-b".to_string()),
                reasoning_effort: None,
            })
        );
    }

    #[tokio::test]
    async fn test_get_all_role_configs() {
        let (store, _dir) = setup().await;
        store
            .set_role_config("role1", Some("m1"), None)
            .await
            .unwrap();
        store
            .set_role_config("role2", Some("m2"), Some("low"))
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![
                RoleConfig {
                    role: "role1".to_string(),
                    model: Some("m1".to_string()),
                    reasoning_effort: None,
                },
                RoleConfig {
                    role: "role2".to_string(),
                    model: Some("m2".to_string()),
                    reasoning_effort: Some("low".to_string()),
                },
            ]
        );
    }

    #[tokio::test]
    async fn test_delete_role_config() {
        let (store, _dir) = setup().await;
        store
            .set_role_config("reviewer", Some("o1"), Some("medium"))
            .await
            .unwrap();
        store.delete_role_config("reviewer").await.unwrap();
        let val = store.get_role_config("reviewer").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_delete_role_config_nonexistent() {
        let (store, _dir) = setup().await;
        store.delete_role_config("nobody").await.unwrap();
    }

    // ── config_model_routing ─────────────────────────────────

    #[tokio::test]
    async fn test_get_model_routing_empty() {
        let (store, _dir) = setup().await;
        let val = store.get_model_routing("nonexistent-model").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_model_routing() {
        let (store, _dir) = setup().await;
        store
            .set_model_routing("deepseek/deepseek-v4-flash", Some("OpenRouter"), Some(true))
            .await
            .unwrap();
        let val = store
            .get_model_routing("deepseek/deepseek-v4-flash")
            .await
            .unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "deepseek/deepseek-v4-flash".to_string(),
                provider_order: Some("OpenRouter".to_string()),
                allow_fallbacks: Some(true),
            })
        );
    }

    #[tokio::test]
    async fn test_set_model_routing_partial() {
        let (store, _dir) = setup().await;
        store
            .set_model_routing("qwen/qwen3.6-plus", Some("DeepSeek"), None)
            .await
            .unwrap();
        let val = store.get_model_routing("qwen/qwen3.6-plus").await.unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "qwen/qwen3.6-plus".to_string(),
                provider_order: Some("DeepSeek".to_string()),
                allow_fallbacks: None,
            })
        );
    }

    #[tokio::test]
    async fn test_set_model_routing_overwrites() {
        let (store, _dir) = setup().await;
        store
            .set_model_routing("gpt-4", Some("OpenAI"), Some(true))
            .await
            .unwrap();
        store
            .set_model_routing("gpt-4", Some("Azure"), Some(false))
            .await
            .unwrap();
        let val = store.get_model_routing("gpt-4").await.unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "gpt-4".to_string(),
                provider_order: Some("Azure".to_string()),
                allow_fallbacks: Some(false),
            })
        );
    }

    #[tokio::test]
    async fn test_get_all_model_routings() {
        let (store, _dir) = setup().await;
        store
            .set_model_routing("model-a", Some("ProviderA"), Some(true))
            .await
            .unwrap();
        store
            .set_model_routing("model-b", None, Some(false))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![
                ModelRouting {
                    model: "model-a".to_string(),
                    provider_order: Some("ProviderA".to_string()),
                    allow_fallbacks: Some(true),
                },
                ModelRouting {
                    model: "model-b".to_string(),
                    provider_order: None,
                    allow_fallbacks: Some(false),
                },
            ]
        );
    }

    #[tokio::test]
    async fn test_delete_model_routing() {
        let (store, _dir) = setup().await;
        store
            .set_model_routing("todelete", Some("X"), Some(false))
            .await
            .unwrap();
        store.delete_model_routing("todelete").await.unwrap();
        let val = store.get_model_routing("todelete").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn test_delete_model_routing_nonexistent() {
        let (store, _dir) = setup().await;
        store.delete_model_routing("ghost-model").await.unwrap();
    }
}

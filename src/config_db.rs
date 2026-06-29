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

// ── SQL statement constants ──────────────────────────────────

/// UPSERT a key-value pair in config_kv.
const UPSERT_KV_SQL: &str = "INSERT INTO config_kv (key, value) VALUES (?1, ?2) \
    ON CONFLICT(key) DO UPDATE SET value = excluded.value";

/// Delete a key-value pair from config_kv by key. Succeeds even if the key does not exist.
const DELETE_KV_SQL: &str = "DELETE FROM config_kv WHERE key = ?1";

impl ConfigStore {
    /// Begin a transaction that serializes all subsequent operations until
    /// committed or rolled back. The returned guard keeps the connection locked.
    pub async fn begin_tx(&self) -> Result<turso::TxGuard<'_>> {
        Ok(self.conn.begin_tx().await?)
    }

    // ── config_kv ────────────────────────────────────────────

    /// Upsert a key-value pair.
    pub async fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(UPSERT_KV_SQL, turso::params![key, value])
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

    /// Get all key-value pairs.
    pub async fn get_all_kv(&self) -> Result<Vec<(String, String)>> {
        let rows = self
            .conn
            .query(
                &format!("SELECT {KV_COLUMNS} FROM config_kv ORDER BY key"),
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                Ok((
                    row.get::<String>(COL_KV_KEY)?,
                    row.get::<String>(COL_KV_VALUE)?,
                ))
            })
            .collect()
    }

    // ── config_role ──────────────────────────────────────────

    /// Get all role config rows.
    pub async fn get_all_role_configs(&self) -> Result<Vec<RoleConfig>> {
        let rows = self
            .conn
            .query(
                &format!("SELECT {ROLE_CONFIG_COLUMNS} FROM config_role ORDER BY role"),
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                let role = row.get::<String>(COL_RC_ROLE)?;
                let model = row.get::<Option<String>>(COL_RC_MODEL)?;
                let reasoning_effort = row.get::<Option<String>>(COL_RC_REASONING_EFFORT)?;
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
                &format!("SELECT {MODEL_ROUTING_COLUMNS} FROM config_model_routing ORDER BY model"),
                turso::params![],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                let model = row.get::<String>(COL_MR_MODEL)?;
                let provider_order = row.get::<Option<String>>(COL_MR_PROVIDER_ORDER)?;
                let allow_fallbacks = row.get::<Option<bool>>(COL_MR_ALLOW_FALLBACKS)?;
                Ok(ModelRouting {
                    model,
                    provider_order,
                    allow_fallbacks,
                })
            })
            .collect()
    }

    // ── batch save (role configs + model routings) ──────────────

    /// Atomically replace all role configs and model routings in a single
    /// transaction.
    ///
    /// Old rows are deleted with a blanket `DELETE` (no per-role iteration),
    /// which is both simpler and prevents stale rows from surviving when a role
    /// is removed from the enum.  The enclosing transaction guarantees that a
    /// crash or error before commit rolls back to the prior state — partial
    /// writes from the two tables are never visible.
    pub async fn save_role_and_routing_configs(
        &self,
        role_configs: &[RoleConfig],
        model_routings: &[ModelRouting],
    ) -> Result<()> {
        let tx = self.begin_tx().await?;
        self.save_role_and_routing_configs_tx(&tx, role_configs, model_routings)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Same as [`save_role_and_routing_configs`] but operates on an existing
    /// transaction provided by the caller.  The caller is responsible for
    /// calling `commit()` or `rollback()` on the [`turso::TxGuard`].
    ///
    /// # Deadlock safety
    ///
    /// This method does NOT call `begin_tx()` or `commit()` internally —
    /// it uses the supplied `tx` directly, making it safe to call from within
    /// an outer transaction without deadlocking on the connection mutex.
    pub(crate) async fn save_role_and_routing_configs_tx(
        &self,
        tx: &turso::TxGuard<'_>,
        role_configs: &[RoleConfig],
        model_routings: &[ModelRouting],
    ) -> Result<()> {
        tx.execute("DELETE FROM config_role", turso::params![])
            .await?;
        for rc in role_configs {
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
        tx.execute("DELETE FROM config_model_routing", turso::params![])
            .await?;
        for mr in model_routings {
            let allow_int = mr.allow_fallbacks.map(i32::from);
            tx.execute(
                "INSERT INTO config_model_routing (model, provider_order, allow_fallbacks) \
                 VALUES (?1, ?2, ?3)",
                turso::params![mr.model.as_str(), mr.provider_order.as_deref(), allow_int],
            )
            .await?;
        }
        Ok(())
    }

    // ── config_kv — tx-aware variants ─────────────────────────

    /// Upsert a key-value pair within an existing transaction.
    /// Like [`set_kv`] but executes on the supplied [`turso::TxGuard`].
    pub(crate) async fn set_kv_tx(
        &self,
        tx: &turso::TxGuard<'_>,
        key: &str,
        value: &str,
    ) -> Result<()> {
        tx.execute(UPSERT_KV_SQL, turso::params![key, value])
            .await?;
        Ok(())
    }

    /// Delete a key-value pair within an existing transaction.
    /// Like [`delete_kv`] but executes on the supplied [`turso::TxGuard`].
    /// Succeeds even if the key does not exist.
    pub(crate) async fn delete_kv_tx(&self, tx: &turso::TxGuard<'_>, key: &str) -> Result<()> {
        tx.execute(DELETE_KV_SQL, turso::params![key]).await?;
        Ok(())
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
        self.conn
            .query_optional(
                "SELECT model, reasoning_effort FROM config_role WHERE role = ?1",
                turso::params![role],
                |row| {
                    let model = row.get::<Option<String>>(0)?;
                    let reasoning_effort = row.get::<Option<String>>(1)?;
                    Ok::<RoleConfig, ::turso::Error>(RoleConfig {
                        role: role_owned,
                        model,
                        reasoning_effort,
                    })
                },
            )
            .await
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
        self.conn
            .query_optional(
                "SELECT provider_order, allow_fallbacks FROM config_model_routing WHERE model = ?1",
                turso::params![model],
                |row| {
                    let provider_order = row.get::<Option<String>>(0)?;
                    let allow_fallbacks = row.get::<Option<bool>>(1)?;
                    Ok::<ModelRouting, ::turso::Error>(ModelRouting {
                        model: model_owned,
                        provider_order,
                        allow_fallbacks,
                    })
                },
            )
            .await
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
        crate::open_test_store!(ConfigStore, "config")
    }

    // ── config_role lifecycle ──────────────────────────────────

    #[tokio::test]
    async fn test_config_role_lifecycle() {
        let (store, _dir) = setup().await;

        // 1. empty state
        let val = store.get_role_config("nonexistent-role").await.unwrap();
        assert!(
            val.is_none(),
            "get_role_config should return None for missing role"
        );

        // 2. insert with all fields
        store
            .set_role_config("engineer", Some("gpt-4"), Some("high"))
            .await
            .unwrap();
        let val = store.get_role_config("engineer").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "engineer".into(),
                model: Some("gpt-4".into()),
                reasoning_effort: Some("high".into()),
            }),
            "get_role_config should return inserted config",
        );

        // 3. partial set (Some → None on optional field)
        store
            .set_role_config("engineer", Some("gpt-5"), None)
            .await
            .unwrap();
        let val = store.get_role_config("engineer").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "engineer".into(),
                model: Some("gpt-5".into()),
                reasoning_effort: None,
            }),
            "set_role_config with None reasoning_effort should clear it",
        );

        // 4. overwrite with both fields
        store
            .set_role_config("engineer", Some("claude-4"), Some("low"))
            .await
            .unwrap();
        let val = store.get_role_config("engineer").await.unwrap();
        assert_eq!(
            val,
            Some(RoleConfig {
                role: "engineer".into(),
                model: Some("claude-4".into()),
                reasoning_effort: Some("low".into()),
            }),
            "set_role_config should fully overwrite existing row",
        );

        // 5. get_all with multiple items (sorted by role)
        store
            .set_role_config("reviewer", Some("o1"), None)
            .await
            .unwrap();
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![
                RoleConfig {
                    role: "engineer".into(),
                    model: Some("claude-4".into()),
                    reasoning_effort: Some("low".into()),
                },
                RoleConfig {
                    role: "reviewer".into(),
                    model: Some("o1".into()),
                    reasoning_effort: None,
                },
            ],
            "get_all_role_configs should return all rows sorted by role",
        );

        // 6. delete
        store.delete_role_config("engineer").await.unwrap();
        let val = store.get_role_config("engineer").await.unwrap();
        assert!(
            val.is_none(),
            "get_role_config should return None after delete"
        );

        // 7. delete non-existent key (no-op, must not error)
        store.delete_role_config("never-existed").await.unwrap();

        // 8. delete already-deleted key (also a no-op)
        store.delete_role_config("engineer").await.unwrap();

        // 9. remaining item still present
        let all = store.get_all_role_configs().await.unwrap();
        assert_eq!(
            all,
            vec![RoleConfig {
                role: "reviewer".into(),
                model: Some("o1".into()),
                reasoning_effort: None,
            }],
            "only the undeleted role config should remain",
        );
    }

    // ── config_model_routing lifecycle ─────────────────────────

    #[tokio::test]
    async fn test_config_model_routing_lifecycle() {
        let (store, _dir) = setup().await;

        // 1. empty state
        let val = store.get_model_routing("nonexistent-model").await.unwrap();
        assert!(
            val.is_none(),
            "get_model_routing should return None for missing model"
        );

        // 2. insert with all fields
        store
            .set_model_routing("gpt-4", Some("OpenAI"), Some(true))
            .await
            .unwrap();
        let val = store.get_model_routing("gpt-4").await.unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "gpt-4".into(),
                provider_order: Some("OpenAI".into()),
                allow_fallbacks: Some(true),
            }),
            "get_model_routing should return inserted config",
        );

        // 3. partial set (Some → None on optional field)
        store
            .set_model_routing("gpt-4", Some("Azure"), None)
            .await
            .unwrap();
        let val = store.get_model_routing("gpt-4").await.unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "gpt-4".into(),
                provider_order: Some("Azure".into()),
                allow_fallbacks: None,
            }),
            "set_model_routing with None allow_fallbacks should clear it",
        );

        // 4. overwrite with both fields
        store
            .set_model_routing("gpt-4", Some("OpenRouter"), Some(false))
            .await
            .unwrap();
        let val = store.get_model_routing("gpt-4").await.unwrap();
        assert_eq!(
            val,
            Some(ModelRouting {
                model: "gpt-4".into(),
                provider_order: Some("OpenRouter".into()),
                allow_fallbacks: Some(false),
            }),
            "set_model_routing should fully overwrite existing row",
        );

        // 5. get_all with multiple items (sorted by model)
        store
            .set_model_routing("claude-3", None, Some(true))
            .await
            .unwrap();
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![
                ModelRouting {
                    model: "claude-3".into(),
                    provider_order: None,
                    allow_fallbacks: Some(true),
                },
                ModelRouting {
                    model: "gpt-4".into(),
                    provider_order: Some("OpenRouter".into()),
                    allow_fallbacks: Some(false),
                },
            ],
            "get_all_model_routings should return all rows sorted by model",
        );

        // 6. delete
        store.delete_model_routing("gpt-4").await.unwrap();
        let val = store.get_model_routing("gpt-4").await.unwrap();
        assert!(
            val.is_none(),
            "get_model_routing should return None after delete"
        );

        // 7. delete non-existent key (no-op, must not error)
        store.delete_model_routing("never-existed").await.unwrap();

        // 8. delete already-deleted key (also a no-op)
        store.delete_model_routing("gpt-4").await.unwrap();

        // 9. remaining item still present
        let all = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            all,
            vec![ModelRouting {
                model: "claude-3".into(),
                provider_order: None,
                allow_fallbacks: Some(true),
            }],
            "only the undeleted model routing should remain",
        );
    }

    // ── config_kv lifecycle ──────────────────────────────────

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

    // ── save_role_and_routing_configs ──────────────────────────

    /// Save role configs and model routings, then verify the saved data
    /// matches input. Input slices are sorted to match DB ORDER BY.
    async fn assert_save_configs(
        store: &ConfigStore,
        role_configs: &[RoleConfig],
        model_routings: &[ModelRouting],
    ) {
        let mut roles = role_configs.to_vec();
        let mut routings = model_routings.to_vec();
        roles.sort_by(|a, b| a.role.cmp(&b.role));
        routings.sort_by(|a, b| a.model.cmp(&b.model));
        store
            .save_role_and_routing_configs(&roles, &routings)
            .await
            .unwrap();
        let saved_roles = store.get_all_role_configs().await.unwrap();
        assert_eq!(saved_roles, roles, "saved role configs should match input");
        let saved_routings = store.get_all_model_routings().await.unwrap();
        assert_eq!(
            saved_routings, routings,
            "saved model routings should match input"
        );
    }

    #[tokio::test]
    async fn test_save_role_and_routing_configs_initial_save() {
        let (store, _dir) = setup().await;
        assert_save_configs(
            &store,
            &[
                RoleConfig {
                    role: "engineer".to_string(),
                    model: Some("gpt-4".to_string()),
                    reasoning_effort: Some("high".to_string()),
                },
                RoleConfig {
                    role: "reviewer".to_string(),
                    model: Some("claude-3".to_string()),
                    reasoning_effort: None,
                },
            ],
            &[ModelRouting {
                model: "gpt-4".to_string(),
                provider_order: Some("OpenAI".to_string()),
                allow_fallbacks: Some(true),
            }],
        )
        .await;
    }

    #[tokio::test]
    async fn test_save_role_and_routing_configs_replaces_old_rows() {
        let (store, _dir) = setup().await;

        // Pre-insert initial data
        assert_save_configs(
            &store,
            &[RoleConfig {
                role: "manager".to_string(),
                model: Some("old-model".to_string()),
                reasoning_effort: Some("low".to_string()),
            }],
            &[ModelRouting {
                model: "old-model".to_string(),
                provider_order: Some("OldProvider".to_string()),
                allow_fallbacks: Some(false),
            }],
        )
        .await;

        // Replace with completely different data
        assert_save_configs(
            &store,
            &[RoleConfig {
                role: "qa".to_string(),
                model: Some("new-model".to_string()),
                reasoning_effort: None,
            }],
            &[
                ModelRouting {
                    model: "fallback-model".to_string(),
                    provider_order: None,
                    allow_fallbacks: None,
                },
                ModelRouting {
                    model: "new-model".to_string(),
                    provider_order: Some("NewProvider".to_string()),
                    allow_fallbacks: Some(true),
                },
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn test_save_role_and_routing_configs_empty_slices() {
        let (store, _dir) = setup().await;

        // Pre-insert some data via individual set calls
        store
            .set_role_config("should-be-cleared", Some("x"), None)
            .await
            .unwrap();
        store
            .set_model_routing("should-be-cleared", Some("y"), Some(true))
            .await
            .unwrap();

        // Save with empty slices — should clear both tables
        assert_save_configs(&store, &[], &[]).await;
    }

    #[tokio::test]
    async fn test_save_role_and_routing_configs_duplicate_key_returns_err() {
        let (store, dir) = setup().await;

        // Open a second connection *before* any transaction starts on the
        // first one, so the schema DDL write can complete without lock
        // contention.  Because SQLite's default isolation ensures that a
        // separate connection never sees uncommitted changes from another
        // connection, reading from `second` after a failed transaction proves
        // that no durable trace was left.
        let second = ConfigStore::open(dir.path()).await.unwrap();

        // Pre-populate with known data.
        let mut original_roles = vec![
            RoleConfig {
                role: "alpha".to_string(),
                model: Some("m1".to_string()),
                reasoning_effort: None,
            },
            RoleConfig {
                role: "beta".to_string(),
                model: Some("m2".to_string()),
                reasoning_effort: Some("low".to_string()),
            },
        ];
        let mut original_routings = vec![ModelRouting {
            model: "m1".to_string(),
            provider_order: Some("ProviderX".to_string()),
            allow_fallbacks: Some(true),
        }];
        original_roles.sort_by(|a, b| a.role.cmp(&b.role));
        original_routings.sort_by(|a, b| a.model.cmp(&b.model));
        store
            .save_role_and_routing_configs(&original_roles, &original_routings)
            .await
            .unwrap();

        // Attempt to save rows with a duplicate role key.
        // The second INSERT will hit the PRIMARY KEY constraint and fail.
        let conflicting_roles = vec![
            RoleConfig {
                role: "collides".to_string(),
                model: Some("first".to_string()),
                reasoning_effort: None,
            },
            RoleConfig {
                role: "collides".to_string(),
                model: Some("second".to_string()),
                reasoning_effort: Some("high".to_string()),
            },
        ];
        let result = store
            .save_role_and_routing_configs(&conflicting_roles, &[])
            .await;
        assert!(result.is_err(), "duplicate role key should cause an error");

        // Read from the separate connection — only committed state is visible,
        // proving the failed transaction had no permanent effect.
        let saved_roles = second.get_all_role_configs().await.unwrap();
        assert_eq!(
            saved_roles, original_roles,
            "role configs should be unchanged after rollback"
        );

        let saved_routings = second.get_all_model_routings().await.unwrap();
        assert_eq!(
            saved_routings, original_routings,
            "model routings should be unchanged after rollback"
        );
    }
}

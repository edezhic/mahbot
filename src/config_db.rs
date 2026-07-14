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
        let tx = self.conn.begin_tx().await?;
        Self::save_role_and_routing_configs_tx(&tx, role_configs, model_routings).await?;
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
        tx: &turso::TxGuard<'_>,
        role_configs: &[RoleConfig],
        model_routings: &[ModelRouting],
    ) -> Result<()> {
        tx.execute("DELETE FROM config_role", turso::params![])
            .await?;
        let insert_role_sql =
            format!("INSERT INTO config_role ({ROLE_CONFIG_COLUMNS}) VALUES (?1, ?2, ?3)");
        for rc in role_configs {
            tx.execute(
                &insert_role_sql,
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
        let insert_routing_sql = format!(
            "INSERT INTO config_model_routing ({MODEL_ROUTING_COLUMNS}) VALUES (?1, ?2, ?3)"
        );
        for mr in model_routings {
            let allow_int = mr.allow_fallbacks.map(i32::from);
            tx.execute(
                &insert_routing_sql,
                turso::params![mr.model.as_str(), mr.provider_order.as_deref(), allow_int],
            )
            .await?;
        }
        Ok(())
    }

    // ── config_kv — tx-aware variants ─────────────────────────

    /// Upsert a key-value pair within an existing transaction.
    /// Like [`set_kv`] but executes on the supplied [`turso::TxGuard`].
    pub(crate) async fn set_kv_tx(tx: &turso::TxGuard<'_>, key: &str, value: &str) -> Result<()> {
        tx.execute(SET_KV_SQL, turso::params![key, value]).await?;
        Ok(())
    }

    /// Delete a key-value pair within an existing transaction.
    /// Like [`delete_kv`] but executes on the supplied [`turso::TxGuard`].
    /// Succeeds even if the key does not exist.
    pub(crate) async fn delete_kv_tx(tx: &turso::TxGuard<'_>, key: &str) -> Result<()> {
        tx.execute(DELETE_KV_SQL, turso::params![key]).await?;
        Ok(())
    }

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
        let rows = self.conn.query_map(&sql, turso::params![], parser).await?;
        Ok(rows
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?)
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

    // ── Parameterised lifecycle tests (role + routing) ─────────
    //
    // Both config_role and config_model_routing share the same 6-step lifecycle
    // against save_role_and_routing_configs.  A macro eliminates the structural
    // duplication while keeping the per-type data fully explicit.
    //
    // Each step tuple is (roles, routings, expected, message):
    //   roles/routings  — data passed to save_role_and_routing_configs
    //   expected        — value that get_all_role_configs / get_all_model_routings
    //                     must return after the save
    //   message         — assertion label for the step

    macro_rules! lifecycle_test {
        (
            $name:ident,
            $getter:ident,
            [$((
                $roles:expr,
                $routings:expr,
                $expected:expr,
                $msg:expr
            )),+ $(,)?]
        ) => {
            #[tokio::test]
            #[allow(clippy::too_many_lines)]
            async fn $name() {
                let (store, _dir) = setup().await;

                // 1. empty state
                let all = store.$getter().await.unwrap();
                assert!(
                    all.is_empty(),
                    "get_all should return empty vec for empty table"
                );

                $(
                    let roles: Vec<RoleConfig> = $roles;
                    let routings: Vec<ModelRouting> = $routings;
                    let expected = $expected;

                    store
                        .save_role_and_routing_configs(&roles, &routings)
                        .await
                        .unwrap();

                    let all = store.$getter().await.unwrap();
                    assert_eq!(all, expected, $msg);
                )+
            }
        };
    }

    lifecycle_test!(
        test_config_role_lifecycle,
        get_all_role_configs,
        [
            (
                vec![role_config("engineer", Some("gpt-4"), Some("high"))],
                vec![],
                vec![role_config("engineer", Some("gpt-4"), Some("high"))],
                "save should persist item"
            ),
            (
                vec![role_config("engineer", Some("gpt-5"), None)],
                vec![],
                vec![role_config("engineer", Some("gpt-5"), None)],
                "save with None nullable should persist NULL"
            ),
            (
                vec![role_config("engineer", Some("claude-4"), Some("low"))],
                vec![],
                vec![role_config("engineer", Some("claude-4"), Some("low"))],
                "save should fully replace existing row"
            ),
            (
                vec![
                    role_config("engineer", Some("claude-4"), Some("low")),
                    role_config("reviewer", Some("o1"), None),
                ],
                vec![],
                vec![
                    role_config("engineer", Some("claude-4"), Some("low")),
                    role_config("reviewer", Some("o1"), None),
                ],
                "get_all should return all rows sorted by key"
            ),
            (
                vec![role_config("reviewer", Some("o1"), None)],
                vec![],
                vec![role_config("reviewer", Some("o1"), None),],
                "only the saved item should remain after replacement"
            ),
        ]
    );

    lifecycle_test!(
        test_config_model_routing_lifecycle,
        get_all_model_routings,
        [
            (
                vec![],
                vec![model_routing("gpt-4", Some("OpenAI"), Some(true))],
                vec![model_routing("gpt-4", Some("OpenAI"), Some(true))],
                "save should persist item"
            ),
            (
                vec![],
                vec![model_routing("gpt-4", Some("Azure"), None)],
                vec![model_routing("gpt-4", Some("Azure"), None)],
                "save with None nullable should persist NULL"
            ),
            (
                vec![],
                vec![model_routing("gpt-4", Some("OpenRouter"), Some(false))],
                vec![model_routing("gpt-4", Some("OpenRouter"), Some(false))],
                "save should fully replace existing row"
            ),
            (
                vec![],
                vec![
                    model_routing("claude-3", None, Some(true)),
                    model_routing("gpt-4", Some("OpenRouter"), Some(false)),
                ],
                vec![
                    model_routing("claude-3", None, Some(true)),
                    model_routing("gpt-4", Some("OpenRouter"), Some(false)),
                ],
                "get_all should return all rows sorted by key"
            ),
            (
                vec![],
                vec![model_routing("claude-3", None, Some(true))],
                vec![model_routing("claude-3", None, Some(true)),],
                "only the saved item should remain after replacement"
            ),
        ]
    );

    // ── config_kv lifecycle ──────────────────────────────────
    //
    // KV storage uses the production set_kv/delete_kv/get_all_kv path.
    // For individual value lookups we use an inline get_kv helper (defined below).

    #[tokio::test]
    async fn test_config_kv_lifecycle() {
        // Inline helper for single-key lookup.
        async fn get_kv(store: &ConfigStore, key: &str) -> Result<Option<String>> {
            store
                .conn
                .query_optional(
                    "SELECT value FROM config_kv WHERE key = ?1",
                    ::turso::params![key],
                    |row| row.get::<String>(0),
                )
                .await
        }

        let (store, _dir) = setup().await;

        // 1. empty state
        let val = get_kv(&store, "nonexistent").await.unwrap();
        assert!(val.is_none(), "get_kv should return None for missing key");

        // 2. insert
        store.set_kv("alpha", "first").await.unwrap();
        let val = get_kv(&store, "alpha").await.unwrap();
        assert_eq!(
            val,
            Some("first".to_string()),
            "get_kv should return inserted value"
        );

        // 3. overwrite
        store.set_kv("alpha", "updated").await.unwrap();
        let val = get_kv(&store, "alpha").await.unwrap();
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
        let val = get_kv(&store, "alpha").await.unwrap();
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
    async fn test_save_role_and_routing_configs_replaces_old_rows() {
        let (store, _dir) = setup().await;

        // Pre-insert initial data
        assert_save_configs(
            &store,
            &[role_config("manager", Some("old-model"), Some("low"))],
            &[model_routing("old-model", Some("OldProvider"), Some(false))],
        )
        .await;

        // Replace with completely different data
        assert_save_configs(
            &store,
            &[role_config("qa", Some("new-model"), None)],
            &[
                model_routing("fallback-model", None, None),
                model_routing("new-model", Some("NewProvider"), Some(true)),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn test_save_role_and_routing_configs_empty_slices() {
        let (store, _dir) = setup().await;

        // Pre-insert some data via the production batch path
        store
            .save_role_and_routing_configs(
                &[role_config("should-be-cleared", Some("x"), None)],
                &[model_routing("should-be-cleared", Some("y"), Some(true))],
            )
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
            role_config("alpha", Some("m1"), None),
            role_config("beta", Some("m2"), Some("low")),
        ];
        let mut original_routings = vec![model_routing("m1", Some("ProviderX"), Some(true))];
        original_roles.sort_by(|a, b| a.role.cmp(&b.role));
        original_routings.sort_by(|a, b| a.model.cmp(&b.model));
        store
            .save_role_and_routing_configs(&original_roles, &original_routings)
            .await
            .unwrap();

        // Attempt to save rows with a duplicate role key.
        // The second INSERT will hit the PRIMARY KEY constraint and fail.
        let conflicting_roles = vec![
            role_config("collides", Some("first"), None),
            role_config("collides", Some("second"), Some("high")),
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

//! Shared test utilities for initializing global stores and building test
//! tickets.
//!
//! Provides a single temporary directory shared across all test initializations,
//! instead of the previous per-module pattern that leaked a separate temp dir
//! per store.
//!
//! The shared test root is created once per process; on unix it is removed at
//! process exit via an exit hook (pass, failure, and per-test panics all exit
//! the process normally), and on other platforms it is left for the OS temp
//! sweep. Since global [`OnceCell`]s can only be set once, each store is
//! initialized at most once per test run.
//!
//! Also provides [`TicketBuilder`], a builder for creating test tickets that was
//! historically defined in the `board` module and imported from there by sibling
//! modules. Moved here so all test infrastructure lives in one place.
//!
//! Also provides [`JobRowBuilder`], a builder for inserting test `jobs` rows
//! (the durability/resume substrate in the consolidated domain database) that
//! mirrors the production INSERT in [`crate::jobs::spawn_job`].
//!
//! Also provides [`create_test_workspace`] (inserting a workspace into the test DB)
//! and [`init_management_test_stores`] (initializing all stores plus the manager
//! queue), relocated from `management.rs` tests so they are discoverable alongside
//! the rest of the shared test infrastructure.

#![cfg(test)]

use crate::db;
use crate::pipeline::board::{BoardStore, Ticket, TicketParams, TicketPhase};
use crate::util::UnwrapPoison;
use crate::workspace::test_ws_named;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Shared test root directory, created once per process.
///
/// Removed at process exit on unix via [`register_test_root_cleanup`]; on
/// other platforms it is left for the OS temp sweep. This is the process-level
/// shared root only — per-test temp dirs (`open_test_store!`,
/// `init_temp_repo`, per-test `TempDir`s) keep their own create/drop
/// lifecycle.
static TEST_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// PID of the process that created [`TEST_ROOT`].
///
/// Guards the exit-time cleanup: `atexit` handlers registered before `fork()`
/// run in the child too, and a fork child must never remove its parent's
/// live root.
#[cfg(unix)]
static TEST_ROOT_CREATOR_PID: OnceLock<libc::pid_t> = OnceLock::new();

/// Mutex serializing env-var-modifying tests to prevent thread-safety
/// issues with `std::env::set_var` (which is `unsafe` in Rust 2024).
///
/// All test modules that modify environment variables should use this
/// shared lock to prevent data races between concurrent tests.
pub fn env_lock() -> &'static std::sync::Mutex<()> {
    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// Shared lock serializing tests that install the retry-policy override and /
/// or the fake provider (scoped-retry tests).
///
/// Both [`crate::retry::swap_test_retry_policy`] (via
/// [`install_test_retry_policy`]) and
/// [`crate::providers::swap_provider_for_test`] mutate process-global state;
/// tests using either must hold this lock for their duration.
///
/// # Panic safety
///
/// The lock is poison-tolerant: a test that panics while holding it poisons
/// the mutex, and the next caller recovers the guard via
/// [`UnwrapPoison::unwrap_poison`] instead of panicking — a single test
/// failure cannot cascade into every subsequent retry test.
pub fn retry_tests_lock() -> std::sync::MutexGuard<'static, ()> {
    static RETRY_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    RETRY_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_poison()
}

// ── Retry-policy guard for scoped-retry tests ──────────────

/// RAII guard restoring the previous test retry-policy override on drop.
///
/// Returned by [`install_test_retry_policy`] — holding it (typically
/// `let _policy_guard`) restores the pre-test override (or unset state) when
/// the test finishes, including on panic, so a tiny test policy never leaks
/// into other tests. Mirrors [`FakeProviderGuard`].
#[must_use]
pub(crate) struct RetryPolicyGuard {
    previous: Option<crate::retry::RetryPolicy>,
}

impl Drop for RetryPolicyGuard {
    fn drop(&mut self) {
        crate::retry::restore_test_retry_policy(self.previous.take());
    }
}

/// Install a test retry-policy override for the duration of the returned
/// guard.
///
/// Callers must hold [`retry_tests_lock()`] for the duration and keep the
/// returned guard alive — dropping it restores the previous override.
pub(crate) fn install_test_retry_policy(policy: crate::retry::RetryPolicy) -> RetryPolicyGuard {
    let previous = crate::retry::swap_test_retry_policy(policy);
    RetryPolicyGuard { previous }
}

// ── Fake provider for scoped-retry tests ──────────────────

/// Scripted [`crate::Provider`] test double for the scoped retry loops.
///
/// Each [`chat_scoped`](crate::Provider::chat_scoped) call pops the next
/// scripted outcome: either `Ok(ChatResponse)` or an error carrying an
/// explicit [`FailureClass`] (so classification is exercised without HTTP).
/// Every request's `Debug` fingerprint is recorded so tests can assert
/// byte-identical retries vs. parse-failure re-prompts.
pub(crate) struct FakeProvider {
    script: std::sync::Mutex<
        std::collections::VecDeque<Result<crate::ChatResponse, crate::providers::ScopedCallError>>,
    >,
    /// `Debug` fingerprint of every request received (byte-identity checks).
    pub request_fingerprints: std::sync::Mutex<Vec<String>>,
    /// `Debug` fingerprint of just each request's message list — the
    /// byte-stable-prefix property (append-only growth of the continuation
    /// tail) is asserted on this. Per-message `Debug` joined with NUL (which
    /// cannot appear in `Debug` output — control chars are escaped), so a
    /// plain `starts_with` prefix check holds for appended-only growth.
    pub request_messages: std::sync::Mutex<Vec<String>>,
}

impl FakeProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            script: std::sync::Mutex::new(std::collections::VecDeque::new()),
            request_fingerprints: std::sync::Mutex::new(Vec::new()),
            request_messages: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Push a scripted successful response with the given text.
    #[must_use]
    pub(crate) fn ok(self, text: &str) -> Self {
        self.ok_with_finish(text, None)
    }

    /// Push a scripted successful response with an explicit finish_reason
    /// (e.g. "length" to simulate provider truncation).
    #[must_use]
    pub(crate) fn ok_with_finish(self, text: &str, finish_reason: Option<&str>) -> Self {
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text: Some(text.to_string()),
                finish_reason: finish_reason.map(str::to_string),
                ..crate::ChatResponse::default()
            }));
        self
    }

    /// Push a scripted reasoning-only response (empty content, no tool calls,
    /// reasoning fields set) — the reasoning-only-stop class. `reasoning` is
    /// mirrored into both `reasoning` and `reasoning_content` (the DeepSeek
    /// wire shape).
    #[must_use]
    pub(crate) fn ok_reasoning_only(self, reasoning: &str, finish_reason: Option<&str>) -> Self {
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text: None,
                reasoning: Some(crate::Reasoning {
                    reasoning: Some(reasoning.to_string()),
                    reasoning_content: Some(reasoning.to_string()),
                    reasoning_details: None,
                }),
                finish_reason: finish_reason.map(str::to_string),
                ..crate::ChatResponse::default()
            }));
        self
    }

    /// Like [`Self::ok_reasoning_only`] but with real provider usage on the
    /// envelope — a real reasoning-only response always carries usage, and the
    /// session-length recording tests need usage-bearing fakes.
    #[must_use]
    pub(crate) fn ok_reasoning_only_with_usage(
        self,
        reasoning: &str,
        finish_reason: Option<&str>,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text: None,
                reasoning: Some(crate::Reasoning {
                    reasoning: Some(reasoning.to_string()),
                    reasoning_content: Some(reasoning.to_string()),
                    reasoning_details: None,
                }),
                finish_reason: finish_reason.map(str::to_string),
                usage: Some(crate::ProviderUsage {
                    input_tokens: Some(input_tokens),
                    output_tokens: Some(output_tokens),
                    ..crate::ProviderUsage::default()
                }),
                ..crate::ChatResponse::default()
            }));
        self
    }

    /// Push a scripted successful response with the given text and real
    /// provider usage (input/output tokens).
    #[must_use]
    pub(crate) fn ok_with_usage(self, text: &str, input_tokens: u64, output_tokens: u64) -> Self {
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text: Some(text.to_string()),
                usage: Some(crate::ProviderUsage {
                    input_tokens: Some(input_tokens),
                    output_tokens: Some(output_tokens),
                    ..crate::ProviderUsage::default()
                }),
                ..crate::ChatResponse::default()
            }));
        self
    }

    /// Push a scripted tool-call response (empty text, one tool call) — the
    /// valid empty-text tool-call turn.
    #[must_use]
    pub(crate) fn ok_tool_call(self, name: &str) -> Self {
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text: None,
                tool_calls: vec![crate::ToolCall {
                    id: "call_test".to_string(),
                    name: name.to_string(),
                    arguments: serde_json::json!({}),
                }],
                finish_reason: Some("tool_calls".to_string()),
                ..crate::ChatResponse::default()
            }));
        self
    }

    /// Push a scripted failure with the given granular class.
    #[must_use]
    pub(crate) fn err(self, class: crate::retry::FailureClass, msg: &str) -> Self {
        let inner = anyhow::anyhow!("{msg}");
        let record = crate::retry::RetryFailureRecord::new_simple(class, &inner, None);
        self.script
            .lock()
            .unwrap()
            .push_back(Err(crate::providers::ScopedCallError::new(
                inner, record, class,
            )));
        self
    }

    /// Push a scripted non-retryable HTTP failure with a raw provider body,
    /// formatted exactly like the production `HttpError` display
    /// (`"{context} API error ({status}): {body}"`), so detection logic that
    /// scans the error-trail text (e.g. the input-image-rejection strip) sees
    /// the same shape as a real provider rejection.
    #[must_use]
    pub(crate) fn err_http(self, status: u16, body: &str) -> Self {
        let msg = format!("OpenRouter API error ({status}): {body}");
        self.err(crate::retry::FailureClass::NonRetryable, &msg)
    }
}

#[async_trait::async_trait]
impl crate::Provider for FakeProvider {
    async fn chat(&self, _request: crate::ChatRequest) -> anyhow::Result<crate::ChatResponse> {
        // The scoped tests always go through `chat_scoped`; this is a
        // safety-net implementation for the trait's default path.
        Ok(crate::ChatResponse::default())
    }

    // `ScopedCallError` is deliberately large (full diagnostics payload); the
    // scripted-Result shape is the point of this test double.
    async fn chat_scoped(
        &self,
        request: crate::ChatRequest,
        _idle_timeout: std::time::Duration,
        _deadline: std::time::Instant,
    ) -> Result<crate::ChatResponse, crate::providers::ScopedCallError> {
        self.request_fingerprints
            .lock()
            .unwrap()
            .push(format!("{request:?}"));
        self.request_messages.lock().unwrap().push(
            request
                .messages
                .iter()
                .map(|m| format!("{m:?}"))
                .collect::<Vec<_>>()
                .join("\u{0}"),
        );
        self.script.lock().unwrap().pop_front().unwrap_or_else(|| {
            Ok(crate::ChatResponse {
                text: Some("unscripted default".to_string()),
                ..crate::ChatResponse::default()
            })
        })
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// RAII guard restoring the previous global provider on drop.
///
/// Returned by [`install_fake_provider`] — holding it (typically `let _guard`)
/// restores the pre-test provider (or unset state) when the test finishes,
/// including on panic, so the fake never leaks into other tests. The restore is
/// conditional on the provider not having been replaced by a concurrent test
/// (see [`restore_provider_for_test_if`]), so a guard never clobbers a newer
/// fake installed by another test.
#[must_use]
pub(crate) struct FakeProviderGuard {
    previous: Option<Arc<dyn crate::Provider>>,
    installed: Arc<dyn crate::Provider>,
}

impl Drop for FakeProviderGuard {
    fn drop(&mut self) {
        crate::providers::restore_provider_for_test_if(&self.installed, self.previous.take());
    }
}

/// Install a [`FakeProvider`] as the global provider for tests.
///
/// Callers must hold [`retry_tests_lock()`] for the duration and keep the
/// returned guard alive — dropping it (on test end or panic) restores the
/// previous provider without clobbering a newer fake installed concurrently.
pub(crate) fn install_fake_provider(provider: Arc<dyn crate::Provider>) -> FakeProviderGuard {
    let previous = crate::providers::swap_provider_for_test(provider.clone());
    FakeProviderGuard {
        previous,
        installed: provider,
    }
}

/// Install a caller-owned logs store for `record_llm_*` stat writes for the
/// duration of the returned guard (mirrors [`install_fake_provider`]).
///
/// The boot `LOG_STORE` is a process-global `OnceCell` set by `init_tracing`,
/// which tests never run — so stat recording is normally a silent no-op.
/// This seam redirects it to a test store so end-to-end tests can assert on
/// the recorded rows.
///
/// While the guard is alive the redirect is PROCESS-GLOBAL: every
/// `record_llm_*` write in any concurrently-running test lands in this store.
/// Callers must hold [`retry_tests_lock()`] for the duration (same convention
/// as [`install_fake_provider`]) and filter queries by a test-unique
/// `agent_id` — otherwise writes leak across tests.
pub(crate) fn install_test_log_store(store: crate::logs::LogStore) -> TestLogStoreGuard {
    let previous = crate::stats::swap_test_log_store(Some(store));
    TestLogStoreGuard { previous }
}

/// RAII guard restoring the previous test log-store override on drop
/// (including during a panic), so it never leaks into other tests.
#[must_use]
pub(crate) struct TestLogStoreGuard {
    previous: Option<crate::logs::LogStore>,
}

impl Drop for TestLogStoreGuard {
    fn drop(&mut self) {
        crate::stats::swap_test_log_store(self.previous.take());
    }
}

/// RAII guard that restores an environment variable to its original value
/// on drop, including during a panic (unwind safety).
///
/// Created by [`set_env_var`]. Holds the shared [`env_lock()`] for the
/// entire duration to serialize concurrent env access across tests.
///
/// # Panic safety
///
/// The `Drop` implementation restores the original value even if the
/// enclosing scope panics, preventing test-isolation leaks. This is the
/// key advantage over a closure-based `with_env_var` helper.
pub struct EnvVarGuard {
    /// Serializes concurrent env access — held for the guard's entire lifetime.
    _lock: std::sync::MutexGuard<'static, ()>,
    key: String,
    /// Original value captured before mutation. Stored as [`OsString`] to
    /// preserve arbitrary (non-UTF-8) env var values on restore.
    original: Option<std::ffi::OsString>,
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: We hold the env_lock, prohibiting concurrent env writes
        // from other test threads while we restore the original value.
        unsafe {
            match &self.original {
                Some(val) => std::env::set_var(&self.key, val),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Set an environment variable for the duration of the returned guard.
///
/// The environment variable `key` is immediately set to `value` (or
/// removed if `value` is `None`). When the returned [`EnvVarGuard`] is
/// dropped — including on panic — the original value is restored.
///
/// Acquires the shared [`env_lock()`] to prevent data races with other
/// tests that manipulate environment variables.
///
/// # Example
///
/// ```ignore
/// let _guard = set_env_var("CARGO_HOME", Some("/custom/cargo"));
/// let path = resolve_cargo_bin_path();
/// // _guard drops here, restoring CARGO_HOME to its original value
/// ```
#[must_use]
pub fn set_env_var(key: &str, value: Option<&str>) -> EnvVarGuard {
    let guard = env_lock().lock().unwrap_poison();
    let original = std::env::var_os(key);
    // SAFETY: Protected by the env_lock acquired above — no concurrent
    // env writes can happen from other test threads.
    unsafe {
        match value {
            Some(val) => std::env::set_var(key, val),
            None => std::env::remove_var(key),
        }
    }
    EnvVarGuard {
        _lock: guard,
        key: key.to_owned(),
        original,
    }
}

/// The single process-level test temp root.
///
/// All test-owned process-global state that needs a temp directory — the
/// CONFIG storage root (stores, user workspaces, models) and the embedder
/// test config — resolves under this one root, so the whole process owns
/// exactly one test temp directory that self-cleans on normal exit (unix).
pub(crate) fn test_root() -> &'static PathBuf {
    TEST_ROOT.get_or_init(|| {
        let tmp = tempfile::TempDir::new().expect("failed to create test temp dir");
        let path = tmp.path().to_path_buf();
        // Deliberately leak the TempDir guard: keeping it around would race
        // with the exit-time removal below (both would try to delete the same
        // directory). The directory is instead removed by the atexit handler
        // (unix) or reclaimed by the OS temp sweep otherwise.
        std::mem::forget(tmp);
        #[cfg(unix)]
        register_test_root_cleanup();
        path
    })
}

/// Register exit-time removal of the shared test root (unix).
///
/// Runs on normal process exit — pass, failure, and per-test panics alike
/// (the test harness catches panics and the process still exits via
/// `exit()`, which runs atexit handlers under the default `panic = "unwind"`
/// profile). Killed processes (abort, signals) skip atexit handlers entirely
/// — leftovers from killed test processes are intentionally out of scope
/// (the OS reclaims them).
///
/// Race safety: each test process creates its own unique root (one per
/// process via [`TEST_ROOT`]) and removes only that one. The handler is
/// panic-free and guarded by the creator PID, so a `fork()` child spawned by
/// a test never removes its parent's root (exec'd children replace the
/// process image and lose the atexit list anyway).
#[cfg(unix)]
fn register_test_root_cleanup() {
    // Runs exactly once per process (called only from the TEST_ROOT
    // get_or_init closure), so no double-registration guard is needed.
    TEST_ROOT_CREATOR_PID.get_or_init(|| unsafe { libc::getpid() });
    // Registration failure is non-fatal: the directory simply leaks to the
    // OS temp sweep (the pre-change behavior).
    unsafe {
        libc::atexit(cleanup_test_root);
    }
}

/// Panic-free atexit handler removing the shared test root.
#[cfg(unix)]
extern "C" fn cleanup_test_root() {
    // SAFETY: getpid is safe to call from an atexit handler.
    let current_pid = unsafe { libc::getpid() };
    if TEST_ROOT_CREATOR_PID.get().copied() != Some(current_pid) {
        return;
    }
    if let Some(path) = TEST_ROOT.get() {
        // Panic-free by construction: remove_dir_all returns a Result, and
        // errors (e.g. ENOENT, leftovers) are ignored — the OS temp sweep
        // reclaims anything remaining.
        let _ = std::fs::remove_dir_all(path);
    }
}

/// Open a temporary store for testing.
///
/// Creates a temporary directory and opens the given store inside it.
/// Returns `(store, TempDir)`.  The `TempDir` MUST be held for the store's
/// lifetime (typically bound to `_tmp` / `_dir` in the calling test).
///
/// `store_name` is used in the panic message if opening the store fails,
/// so it should be a human-readable identifier (e.g. `"workspace"`, `"board"`).
///
/// This is a macro (not a generic function) because Rust's type system cannot
/// express the lifetime relationship between a closure argument and the future
/// returned by `async fn open(path: &Path) -> Result<T>` — the future captures
/// a borrow of the argument, which would require higher-ranked lifetime bounds
/// that `FnOnce` / `AsyncFnOnce` cannot express in the current edition without
/// boxing.
///
/// # Panics
///
/// Panics if the temporary directory cannot be created, or if opening the
/// store fails.
#[macro_export]
macro_rules! open_test_store {
    ($store:ty, $store_name:expr) => {{
        let tmp = ::tempfile::TempDir::new().expect("temp dir for test store");
        let store = <$store>::open(tmp.path()).await.unwrap_or_else(|e| {
            ::std::panic!(
                "failed to open test {store_name} store: {e:?}",
                store_name = $store_name
            )
        });
        (store, tmp)
    }};
}

/// Convenience helper to create a test ticket with just a title and phase.
///
/// Reduces the common boilerplate:
/// ```ignore
/// let id = TicketBuilder::new(&store, &ws)
///     .title("My Ticket")
///     .phase(TicketPhase::Backlog)
///     .create()
///     .await
///     .expect("create my ticket");
/// ```
/// to:
/// ```ignore
/// let id = make_ticket(&store, &ws, "My Ticket", TicketPhase::Backlog).await;
/// ```
///
/// For tickets that need `.desc()` or `.prereqs()`, use [`TicketBuilder`]
/// directly.
///
/// # Panics
///
/// Panics if the ticket cannot be created. The panic message includes the title
/// and phase for debugging.
pub(crate) async fn make_ticket(
    store: &BoardStore,
    ws: &crate::Workspace,
    title: &str,
    phase: TicketPhase,
) -> String {
    TicketBuilder::new(store, ws)
        .title(title)
        .phase(phase)
        .create()
        .await
        .unwrap_or_else(|e| panic!("make_ticket({title}, {phase}) failed: {e}"))
}

/// Fetch a ticket by ID, panicking if the DB query fails or the ticket
/// does not exist.
///
/// Replaces the common test boilerplate:
/// ```ignore
/// let ticket = store.get_ticket(&id).await.expect("get").expect("exists");
/// ```
/// with the more concise:
/// ```ignore
/// let ticket = expect_ticket(&store, &id).await;
/// ```
///
/// # Panics
///
/// Panics if the DB query fails or the ticket is not found (returns `None`).
/// The panic originates from within this helper function.
pub async fn expect_ticket(store: &BoardStore, id: &str) -> Ticket {
    store
        .get_ticket(id)
        .await
        .expect("BoardStore::get_ticket query failed")
        .expect("expected ticket to exist")
}

/// Fetch a ticket's phase by ID, panicking if the DB query fails or the
/// ticket does not exist.
///
/// Replaces the common test boilerplate:
/// ```ignore
/// let phase = store.get_ticket_phase(&id).await.expect("query").expect("exists");
/// ```
/// with the more concise:
/// ```ignore
/// let phase = expect_ticket_phase(&store, &id).await;
/// ```
///
/// # Panics
///
/// Panics if the DB query fails or the ticket is not found (returns `None`).
/// The panic originates from within this helper function.
pub async fn expect_ticket_phase(store: &BoardStore, id: &str) -> TicketPhase {
    store
        .get_ticket_phase(id)
        .await
        .expect("BoardStore::get_ticket_phase query failed")
        .expect("expected ticket phase to exist")
}

/// Builder for creating test tickets with common defaults.
///
/// Defaults: `desc="desc"`, `phase=Backlog`, `prerequisites=[]`, `reporter="test"`,
/// `embedding=None`. Title is required (no default) via `.title()`.
///
/// # Examples
/// ```ignore
/// // Simple ticket with defaults
/// TicketBuilder::new(&store, &ws).title("A").create().await?;
///
/// // Custom phase and prerequisites
/// TicketBuilder::new(&store, &ws)
///     .title("B")
///     .phase(TicketPhase::InDevelopment)
///     .prereqs(&[a_id, b_id])
///     .create().await?;
/// ```
pub(crate) struct TicketBuilder<'a> {
    store: &'a BoardStore,
    ws: crate::Workspace,
    title: String,
    desc: String,
    phase: TicketPhase,
    prereqs: Vec<String>,
    reporter: String,
    embedding: Option<Vec<u8>>,
    priority: i64,
}

impl<'a> TicketBuilder<'a> {
    /// Start building a test ticket for `store` in workspace `ws`.
    pub(crate) fn new(store: &'a BoardStore, ws: &crate::Workspace) -> Self {
        Self {
            store,
            ws: ws.clone(),
            title: String::new(),
            desc: "desc".into(),
            phase: TicketPhase::Backlog,
            prereqs: Vec::new(),
            reporter: "test".into(),
            embedding: None,
            priority: 1,
        }
    }

    /// Set the ticket title (required).
    pub(crate) fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the description (default: `"desc"`).
    pub(crate) fn desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    /// Set the phase (default: [`TicketPhase::Backlog`]).
    pub(crate) fn phase(mut self, phase: TicketPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Set prerequisites (default: empty).
    pub(crate) fn prereqs(mut self, prereqs: &[String]) -> Self {
        self.prereqs = prereqs.to_vec();
        self
    }

    /// Set the priority (default: 1).
    pub(crate) fn priority(mut self, priority: i64) -> Self {
        self.priority = priority;
        self
    }

    /// Set the reporter (default: `"test"`).
    pub(crate) fn reporter(mut self, reporter: impl Into<String>) -> Self {
        self.reporter = reporter.into();
        self
    }

    /// Create the ticket with the accumulated parameters.
    pub(crate) async fn create(self) -> anyhow::Result<String> {
        let (store, params) = self.into_parts();
        store.create_ticket(&params).await
    }

    fn into_parts(self) -> (&'a BoardStore, TicketParams) {
        (
            self.store,
            TicketParams {
                title: self.title,
                description: self.desc,
                workspace_name: self.ws.name,
                phase: self.phase,
                prerequisites: self.prereqs,
                reporter: self.reporter,
                embedding: self.embedding,
                priority: self.priority,
            },
        )
    }
}

/// Builder for inserting a test `jobs` row (the durability/resume substrate
/// in the consolidated domain database).
///
/// Mirrors the production INSERT in [`crate::jobs::spawn_job`] — when the
/// `jobs` schema changes, update both this builder and the production insert.
///
/// Required schema columns live in the constructor (`id`, `kind`, `role`,
/// `workspace_name`); the columns the historical fixtures varied between
/// (user identity, channel, retry count) are opt-in setters, and `task`
/// defaults to `''`. Timestamps are required via [`Self::timestamps`] and
/// transcribed verbatim — the helper never generates timestamps internally
/// (tests assert exact equality on them).
///
/// # Panics
///
/// [`Self::insert`] panics if [`Self::timestamps`] was not called.
pub(crate) struct JobRowBuilder<'a> {
    conn: &'a crate::db::Connection,
    id: String,
    kind: String,
    role: String,
    workspace_name: String,
    task: String,
    user_name: Option<String>,
    channel: Option<String>,
    retry_count: Option<i64>,
    timestamps: Option<String>,
}

impl<'a> JobRowBuilder<'a> {
    /// Start a builder for a `jobs` row with the schema-required columns.
    pub(crate) fn new(
        conn: &'a crate::db::Connection,
        id: impl Into<String>,
        kind: impl Into<String>,
        role: impl Into<String>,
        workspace_name: impl Into<String>,
    ) -> Self {
        Self {
            conn,
            id: id.into(),
            kind: kind.into(),
            role: role.into(),
            workspace_name: workspace_name.into(),
            task: String::new(),
            user_name: None,
            channel: None,
            retry_count: None,
            timestamps: None,
        }
    }

    /// Set `task` (default: `''`).
    pub(crate) fn task(mut self, task: impl Into<String>) -> Self {
        self.task = task.into();
        self
    }

    /// Set `user_name` explicitly (default: column omitted → schema default
    /// `''`). Pass an empty string to force the explicit `''` value.
    pub(crate) fn user_name(mut self, user_name: impl Into<String>) -> Self {
        self.user_name = Some(user_name.into());
        self
    }

    /// Set `channel` explicitly (default: column omitted → schema default
    /// `''`). Pass an empty string to force the explicit `''` value.
    pub(crate) fn channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    /// Set `created_at` AND `updated_at` (both required, transcribed
    /// verbatim — the helper never generates timestamps internally).
    pub(crate) fn timestamps(mut self, timestamps: impl Into<String>) -> Self {
        self.timestamps = Some(timestamps.into());
        self
    }

    /// Insert the row.
    pub(crate) async fn insert(self) -> anyhow::Result<()> {
        let Self {
            conn,
            id,
            kind,
            role,
            workspace_name,
            task,
            user_name,
            channel,
            retry_count,
            timestamps,
        } = self;
        let mut columns = vec!["id", "kind", "task", "workspace_name", "role"];
        let mut values: Vec<crate::db::Value> = vec![
            crate::db::Value::Text(id),
            crate::db::Value::Text(kind),
            crate::db::Value::Text(task),
            crate::db::Value::Text(workspace_name),
            crate::db::Value::Text(role),
        ];
        if let Some(user_name) = user_name {
            columns.push("user_name");
            values.push(crate::db::Value::Text(user_name));
        }
        if let Some(channel) = channel {
            columns.push("channel");
            values.push(crate::db::Value::Text(channel));
        }
        if let Some(retry_count) = retry_count {
            columns.push("retry_count");
            values.push(crate::db::Value::Integer(retry_count));
        }
        let timestamps = timestamps.expect(
            "JobRowBuilder::insert: `.timestamps()` is required — the helper never generates timestamps internally",
        );
        columns.push("created_at");
        columns.push("updated_at");
        values.push(crate::db::Value::Text(timestamps.clone()));
        values.push(crate::db::Value::Text(timestamps));
        let placeholders = vec!["?"; values.len()].join(", ");
        let table = "jobs";
        let sql = format!(
            "INSERT INTO {table} ({}) VALUES ({placeholders})",
            columns.join(", ")
        );
        conn.execute(&sql, values).await?;
        Ok(())
    }
}

/// Initialize all global test stores (session, board, workspace, users,
/// config, stats, chat_history) with a shared temp directory by delegating
/// to [`db::init_all_stores`].
///
/// Also initializes the search engine registry (required by workspace store)
/// and sets the CONFIG storage root.
///
/// Idempotent — subsequent calls are no-ops.
pub async fn init_test_stores() {
    static INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    INIT.get_or_init(|| async {
        // Set CONFIG storage root (no-op if already set by another test)
        let _ = crate::config::CONFIG.try_set_storage_root(test_root().clone());

        // search_engine is sync — must be initialized before workspace
        crate::search_engine::init_global();

        // chronicle is sync — lightweight allocation, no DB I/O.
        crate::pipeline::chronicle::init_global();

        crate::db::init_all_stores()
            .await
            .expect("failed to initialize test stores (see chained error for per-store details)");
    })
    .await;
}

/// Initialize all stores needed by management tests that interact with
/// the ticket buffer.
///
/// Calls [`init_test_stores`] (all test DBs) then initializes the global
/// message_router. The router is required by callers that
/// exercise the pipeline `notify_ticket` path, which
/// enqueues notifications via [`crate::agent::message_router::route`].
///
/// # Panics
///
/// Panics if [`init_test_stores`] has not been called first (the route
/// depends on stores being available), or if initialization of the
/// router fails.
///
/// # Idempotency note
///
/// [`init_test_stores`] is idempotent (uses a [`tokio::sync::OnceCell`]).
/// [`crate::agent::message_router::init_global`] initialises the router's internal
/// [`HashMap`](std::collections::HashMap) (via [`std::sync::OnceLock`]).
/// No consumer loops are spawned until the first [`route`](crate::agent::message_router::route)
/// to each agent ID.
pub async fn init_management_test_stores() {
    init_test_stores().await;

    let _ = crate::agent::message_router::init_global();
}

/// Create a test workspace by inserting it into the test DB and returning
/// a [`Workspace`](crate::Workspace) struct with the given `path` and `name`.
///
/// Parameters are `(path, name)` to match the convention of
/// [`test_ws_named`](crate::workspace::test_ws_named).
///
/// # Precondition
///
/// [`init_test_stores`] must be called before this function — the
/// workspace store's [`OnceCell`](tokio::sync::OnceCell) panics if
/// accessed before initialization.
///
/// # Panics
///
/// Panics if the workspace store is not initialized, or if the INSERT
/// SQL query fails.
pub async fn create_test_workspace(path: &str, name: &str) -> crate::Workspace {
    let now = crate::db::now();
    crate::workspace::store()
        .conn
        .execute(
            "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            db::params![name, path, now.clone(), now, 0],
        )
        .await
        .expect("insert test workspace");
    test_ws_named(path, name)
}

/// Create a temporary directory initialized as a git repository with a
/// committed file named `test.txt` (containing `"line1\nline2\nline3\n"`)
/// and a single commit titled `"Initial commit"`.
///
/// The returned [`TempDir`](tempfile::TempDir) MUST be kept alive (bound to
/// `_dir` or similar) for the returned [`PathBuf`] to remain valid.
///
/// # Panics
///
/// Panics if `git` is not available, or if any git command fails.
pub(crate) fn init_temp_repo() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let repo_path = dir.path().to_path_buf();

    // git init
    let status = std::process::Command::new("git")
        .args(["init"])
        .current_dir(&repo_path)
        .status()
        .expect("git init");
    assert!(status.success());

    // Set user config (required for commit)
    for (key, value) in [("user.name", "Test"), ("user.email", "test@test.com")] {
        let status = std::process::Command::new("git")
            .args(["config", key, value])
            .current_dir(&repo_path)
            .status()
            .expect("git config");
        assert!(status.success());
    }

    // Create a file and make initial commit
    std::fs::write(repo_path.join("test.txt"), b"line1\nline2\nline3\n").expect("write test file");
    let status = std::process::Command::new("git")
        .args(["add", "-A"])
        .current_dir(&repo_path)
        .status()
        .expect("git add");
    assert!(status.success());
    let status = std::process::Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(&repo_path)
        .status()
        .expect("git commit");
    assert!(status.success());

    (dir, repo_path)
}

// ── EnvVarGuard tests ─────────────────────────────────────────────────
//
// These tests use `unsafe { std::env::set_var/remove_var }` directly for
// setup and cleanup rather than going through the guard — ironic given the
// helper's purpose. This is an accepted trade-off: we're testing the
// wrapper's correctness, and the env_lock is independently exercised by
// every non-helper caller. Unique-per-test variable names prevent
// collisions within this module, but per std::env docs even writes to
// *different* variables are a data race without the lock.

#[cfg(test)]
mod env_var_guard_tests {
    use super::*;

    #[test]
    fn sets_and_restores_to_absent() {
        let guard = set_env_var("MAHBOT_TEST_SET_RESTORE", Some("hello"));
        assert_eq!(std::env::var("MAHBOT_TEST_SET_RESTORE"), Ok("hello".into()));
        drop(guard);

        // Variable was absent before the guard (unique name, first use).
        // The guard should restore to that state.
        assert!(
            std::env::var_os("MAHBOT_TEST_SET_RESTORE").is_none(),
            "guard should restore env var to absent on drop"
        );
    }

    #[test]
    fn removes_env_var() {
        // SAFETY: This bypasses env_lock (unique var name mitigates but
        // doesn't eliminate the race — see mod-level doc above). We need
        // pre-existing state to test the removal+restore path.
        unsafe {
            std::env::set_var("MAHBOT_TEST_REMOVE", "present");
        }

        let guard = set_env_var("MAHBOT_TEST_REMOVE", None);
        assert!(
            std::env::var_os("MAHBOT_TEST_REMOVE").is_none(),
            "set_env_var(key, None) should remove the variable"
        );
        drop(guard);

        // Original should be restored.
        assert_eq!(
            std::env::var("MAHBOT_TEST_REMOVE"),
            Ok("present".into()),
            "guard should restore the original value on drop"
        );

        // SAFETY: Bypasses env_lock (same trade-off as above).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_REMOVE");
        }
    }

    #[test]
    fn captures_and_restores_original_value() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::set_var("MAHBOT_TEST_CAPTURE", "original");
        }

        let guard = set_env_var("MAHBOT_TEST_CAPTURE", Some("override"));
        assert_eq!(std::env::var("MAHBOT_TEST_CAPTURE"), Ok("override".into()));
        drop(guard);

        assert_eq!(
            std::env::var("MAHBOT_TEST_CAPTURE"),
            Ok("original".into()),
            "guard should restore the original value on drop"
        );

        // SAFETY: Bypasses env_lock (same trade-off).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_CAPTURE");
        }
    }

    #[test]
    fn restores_on_panic() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_PANIC_ABSENT");
        }

        let result = std::panic::catch_unwind(|| {
            let _guard = set_env_var("MAHBOT_TEST_PANIC_ABSENT", Some("panic-value"));
            panic!("intentional panic");
        });
        assert!(result.is_err());

        assert!(
            std::env::var_os("MAHBOT_TEST_PANIC_ABSENT").is_none(),
            "MAHBOT_TEST_PANIC_ABSENT should be absent after panic-restore"
        );
    }

    #[test]
    fn restores_original_on_panic() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::set_var("MAHBOT_TEST_PANIC_ORIGINAL", "original");
        }

        let result = std::panic::catch_unwind(|| {
            let _guard = set_env_var("MAHBOT_TEST_PANIC_ORIGINAL", Some("panic-value"));
            panic!("intentional panic");
        });
        assert!(result.is_err());

        assert_eq!(
            std::env::var("MAHBOT_TEST_PANIC_ORIGINAL"),
            Ok("original".into()),
            "should restore original value after panic"
        );

        // SAFETY: Bypasses env_lock (same trade-off).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_PANIC_ORIGINAL");
        }
    }
}

#[cfg(test)]
mod retry_policy_guard_tests {
    use super::*;

    /// The tiny test policy's `max_attempts` (3) — distinguishes it from the
    /// production default (13) when asserting restore semantics.
    const TINY_MAX_ATTEMPTS: u32 = 3;

    #[test]
    fn installs_and_restores_on_drop() {
        let _lock = retry_tests_lock();
        assert_eq!(
            crate::retry::RetryPolicy::current().max_attempts,
            crate::retry::DEFAULT_RETRY_MAX_ATTEMPTS
        );
        let guard = install_test_retry_policy(crate::retry::tiny_test_policy());
        assert_eq!(
            crate::retry::RetryPolicy::current().max_attempts,
            TINY_MAX_ATTEMPTS
        );
        drop(guard);
        assert_eq!(
            crate::retry::RetryPolicy::current().max_attempts,
            crate::retry::DEFAULT_RETRY_MAX_ATTEMPTS,
            "guard must restore the pre-test override on drop"
        );
    }

    #[test]
    fn restores_on_panic() {
        let _lock = retry_tests_lock();
        let result = std::panic::catch_unwind(|| {
            let _guard = install_test_retry_policy(crate::retry::tiny_test_policy());
            assert_eq!(
                crate::retry::RetryPolicy::current().max_attempts,
                TINY_MAX_ATTEMPTS
            );
            panic!("intentional panic while holding the policy guard");
        });
        assert!(result.is_err());
        assert_eq!(
            crate::retry::RetryPolicy::current().max_attempts,
            crate::retry::DEFAULT_RETRY_MAX_ATTEMPTS,
            "a panicking test must not leak the tiny policy into later tests"
        );
    }

    #[test]
    fn lock_recovers_after_poison() {
        // A test that panics while holding retry_tests_lock poisons the std
        // Mutex; the next acquire must recover the guard (PoisonError
        // into_inner) instead of panicking — one failure cannot cascade.
        let result = std::panic::catch_unwind(|| {
            let _lock = retry_tests_lock();
            panic!("intentional panic while holding the retry lock");
        });
        assert!(result.is_err());
        let _lock = retry_tests_lock(); // must not panic despite poisoning
    }
}

/// Deterministic per-pixel noise that defeats PNG compression (used by the
/// reference-image loader and body-budget tests to build reliably large files).
pub fn noisy_png(width: u32, height: u32) -> Vec<u8> {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
        let mut v = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
        v ^= v >> 13;
        v ^= v << 17;
        v ^= v >> 5;
        Rgb([
            (v & 0xFF) as u8,
            ((v >> 8) & 0xFF) as u8,
            ((v >> 16) & 0xFF) as u8,
        ])
    });
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

//! Shared test infrastructure: global stores, builders and guards.
//!
//! One temporary directory is shared across all test initializations rather than one per store. The
//! root is created once per process and removed at exit on unix; since global [`OnceCell`]s can
//! only be set once, each store is initialized at most once per test run.
//!
//! Also here: [`TicketBuilder`] and [`TicketFixture`] (tickets), [`JobRowBuilder`] (mirroring the
//! production INSERT in [`crate::jobs::spawn_job`]), [`create_test_workspace`],
//! [`init_management_test_stores`], [`seed_session_row`] (the session store's `batch_append` is
//! module-private), [`ProbeFile`], and [`rs_files_under`] with [`rel_source_path`] for the
//! source-scanning tripwires.

#![cfg(test)]

use crate::db;
use crate::pipeline::board::{
    BoardStore, Ticket, TicketComment, TicketLabel, TicketParams, TicketPhase,
};
use crate::util::UnwrapPoison;
use crate::workspace::test_ws_named;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Shared test root directory, created once per process and removed at exit on unix via
/// [`register_test_root_cleanup`]. Per-test temp dirs keep their own lifecycle.
static TEST_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// PID of the process that created [`TEST_ROOT`]: `atexit` handlers registered before `fork()` run
/// in the child too, and a fork child must never remove its parent's live root.
#[cfg(unix)]
static TEST_ROOT_CREATOR_PID: OnceLock<libc::pid_t> = OnceLock::new();

/// Mutex serializing env-var-modifying tests: `std::env::set_var` is `unsafe` in Rust 2024, and
/// every test module touching the environment must take this lock.
pub fn env_lock() -> &'static std::sync::Mutex<()> {
    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// A probe file that removes itself on drop.
pub(crate) struct ProbeFile(pub PathBuf);

impl Drop for ProbeFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// Test-global exclusion is via [`serial_test`] keys: `provider` guards the fake-provider and
// retry-policy globals, `drain` the shutdown drain flag. A test needs a key whenever it MUTATES
// the guarded global or its assertions depend on its value — a reader asserting non-drain
// behaviour needs `drain` as much as a mutator does (a test awaiting a routed manager envelope:
// routing skips the live route while the flag is set, so its wait would never resolve). Keys may
// be listed in any order, since the derive macro sorts them at expansion. Tests touching no
// guarded global carry no key and must not share process-global state of their own.

/// The process-global store lock: tests writing the shared `OnceCell` stores hold it for their
/// whole body, so no two touch the same rows at once. Async-aware, so a guard may be held across
/// `.await` points.
pub(crate) static STORE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// RAII guard restoring the previous test retry-policy override on drop, on panic included, so a
/// tiny test policy never leaks into other tests. Mirrors [`FakeProviderGuard`].
#[must_use]
pub(crate) struct RetryPolicyGuard {
    previous: Option<crate::retry::RetryPolicy>,
}

impl Drop for RetryPolicyGuard {
    fn drop(&mut self) {
        crate::retry::restore_test_retry_policy(self.previous.take());
    }
}

/// Install a test retry-policy override for the duration of the returned guard. The caller must be
/// annotated `#[serial_test::serial(provider)]` and keep the guard alive.
pub(crate) fn install_test_retry_policy(policy: crate::retry::RetryPolicy) -> RetryPolicyGuard {
    let previous = crate::retry::swap_test_retry_policy(policy);
    RetryPolicyGuard { previous }
}

/// Scripted [`crate::Provider`] test double for the scoped retry loops: each
/// [`chat_scoped`](crate::Provider::chat_scoped) call pops the next outcome — `Ok(ChatResponse)`
/// or an error carrying an explicit [`FailureClass`], so classification needs no HTTP. Every
/// request's `Debug` fingerprint is recorded for byte-identity assertions.
pub(crate) struct FakeProvider {
    script: std::sync::Mutex<
        std::collections::VecDeque<Result<crate::ChatResponse, crate::providers::ScopedCallError>>,
    >,
    /// `Debug` fingerprint of every request received (byte-identity checks).
    pub request_fingerprints: std::sync::Mutex<Vec<String>>,
    /// `Debug` fingerprint of just each request's message list, on which the byte-stable-prefix
    /// property is asserted: per-message `Debug` joined with NUL (which cannot appear in `Debug`
    /// output), so a plain `starts_with` check holds for appended-only growth.
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

    #[must_use]
    pub(crate) fn ok(self, text: &str) -> Self {
        self.ok_with_finish(text, None)
    }

    /// Push a successful response with an explicit finish_reason (e.g. "length").
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

    /// Push a reasoning-only response (empty content, no tool calls): `reasoning` is mirrored into
    /// both `reasoning` and `reasoning_content`, the DeepSeek wire shape.
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

    /// Like [`Self::ok_reasoning_only`] but with real usage on the envelope, which a real
    /// reasoning-only response always carries.
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

    /// Push text AND one or more tool calls (ids `call_test_0`, … in order).
    #[must_use]
    pub(crate) fn ok_text_and_tool_calls(
        self,
        text: &str,
        calls: &[(&str, serde_json::Value)],
    ) -> Self {
        let tool_calls = calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| crate::ToolCall {
                id: format!("call_test_{i}"),
                name: name.to_string(),
                arguments: args.clone(),
            })
            .collect();
        let text = Some(text.to_string());
        self.script
            .lock()
            .unwrap()
            .push_back(Ok(crate::ChatResponse {
                text,
                tool_calls,
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

    /// Push a non-retryable HTTP failure formatted exactly like the production `HttpError` display
    /// (`"{context} API error ({status}): {body}"`), so error-trail scanning sees a real shape.
    #[must_use]
    pub(crate) fn err_http(self, status: u16, body: &str) -> Self {
        let msg = format!("OpenRouter API error ({status}): {body}");
        self.err(crate::retry::FailureClass::InvalidRequest, &msg)
    }
}

#[async_trait::async_trait]
impl crate::Provider for FakeProvider {
    // `ScopedCallError` is deliberately large (full diagnostics payload); the
    // scripted-Result shape is the point of this test double.
    async fn chat_scoped(
        &self,
        request: crate::ChatRequest,
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

/// RAII guard restoring the previous global provider on drop, on panic included, so the fake never
/// leaks into other tests. The restore is conditional on the provider not having been replaced
/// meanwhile (see [`restore_provider_for_test_if`]), so it never clobbers a newer fake.
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

/// Install a [`FakeProvider`] as the global provider for tests. The caller must be annotated
/// `#[serial_test::serial(provider)]` and keep the returned guard alive.
pub(crate) fn install_fake_provider(provider: Arc<dyn crate::Provider>) -> FakeProviderGuard {
    let previous = crate::providers::swap_provider_for_test(provider.clone());
    FakeProviderGuard {
        previous,
        installed: provider,
    }
}

/// RAII guard for [`install_retry_seam`]: holds the tiny retry-policy override and the fake
/// provider until dropped, restoring both at scope exit. Fields are ordered so the drop restores
/// provider → policy, the reverse of install. The caller must be annotated
/// `#[serial_test::serial(provider)]`.
#[must_use]
pub(crate) struct RetrySeam {
    _fake: FakeProviderGuard,
    _retry: RetryPolicyGuard,
}

/// Scoped install of the full retry-provider test seam: [`crate::retry::tiny_test_policy`] and
/// `provider` as the global provider, both restored when the [`RetrySeam`] drops. The caller must
/// be annotated `#[serial_test::serial(provider)]`.
pub(crate) fn install_retry_seam(provider: impl crate::Provider + 'static) -> RetrySeam {
    install_retry_seam_dyn(Arc::new(provider))
}

/// [`install_retry_seam`] for an already-arc'd provider.
pub(crate) fn install_retry_seam_dyn(provider: Arc<dyn crate::Provider>) -> RetrySeam {
    let retry = install_test_retry_policy(crate::retry::tiny_test_policy());
    let fake = install_fake_provider(provider);
    RetrySeam {
        _fake: fake,
        _retry: retry,
    }
}

/// Recording [`crate::Channel`] test double: every sent message is pushed into a shared buffer,
/// `listen` is a no-op. `name` is required because the registry keys channels by name and silently
/// drops duplicates, so call sites pass distinct names.
pub(crate) struct SpyChannel {
    name: &'static str,
    sent: Arc<std::sync::Mutex<Vec<crate::SendMessage>>>,
}

impl SpyChannel {
    /// Returns the spy and its shared sent-messages buffer.
    #[must_use]
    pub(crate) fn new(
        name: &'static str,
    ) -> (Self, Arc<std::sync::Mutex<Vec<crate::SendMessage>>>) {
        let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                name,
                sent: Arc::clone(&sent),
            },
            sent,
        )
    }
}

#[async_trait::async_trait]
impl crate::Channel for SpyChannel {
    async fn send(
        &self,
        message: &crate::SendMessage,
    ) -> Result<crate::SendOutcome, crate::SendFailure> {
        self.sent.lock().unwrap_poison().push(message.clone());
        Ok(crate::SendOutcome::default())
    }

    async fn listen(
        &self,
        _tx: tokio::sync::mpsc::Sender<crate::ChannelMessage>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Install a caller-owned logs store for the telemetry queue's writes for the duration of the
/// returned guard (mirrors [`install_fake_provider`]).
///
/// The boot `LOG_STORE` is a `OnceCell` set by `init_tracing`, which tests never run, so telemetry
/// recording is normally a silent no-op; this seam redirects EVERY record the queue writes to a
/// test store. The redirect is PROCESS-GLOBAL: callers must hold the serial_test keys covering
/// every global their test touches and filter queries by a test-unique identifier. A test reading
/// those rows must first await [`crate::telemetry::drain_for_test`] — no test runs the boot that
/// starts the drain task.
pub(crate) fn install_test_log_store(store: crate::logs::LogStore) -> TestLogStoreGuard {
    let previous = crate::telemetry::swap_test_log_store(Some(store));
    TestLogStoreGuard { previous }
}

/// RAII guard restoring the previous test log-store override on drop, panic included.
#[must_use]
pub(crate) struct TestLogStoreGuard {
    previous: Option<crate::logs::LogStore>,
}

impl Drop for TestLogStoreGuard {
    fn drop(&mut self) {
        crate::telemetry::swap_test_log_store(self.previous.take());
    }
}

/// RAII guard restoring an environment variable to its original value on drop, panic included —
/// the key advantage over a closure-based helper. Created by [`set_env_var`]; holds the shared
/// [`env_lock()`] for its entire duration to serialize concurrent env access.
pub struct EnvVarGuard {
    /// Serializes concurrent env access — held for the guard's entire lifetime.
    _lock: std::sync::MutexGuard<'static, ()>,
    key: String,
    /// Original value, as [`OsString`] to preserve non-UTF-8 values on restore.
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

/// Set an environment variable for the duration of the returned guard: `key` is set to `value` (or
/// removed when `value` is `None`) and restored on drop, panic included. Acquires the shared
/// [`env_lock()`] to prevent races with other tests.
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

/// Make a file executable (0o755). Unix only, like the tests that need it:
/// every one of them drives a `sh` stub, and the file's mode is the only way to
/// make one runnable there.
#[cfg(unix)]
pub fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, PermissionsExt::from_mode(0o755)).unwrap();
}

/// Wait — briefly — for a pid to stop existing. `kill(pid, 0)` is the existence
/// probe: it sends no signal. Unix only, like the stops it checks.
#[cfg(unix)]
pub async fn wait_for_death(pid: i32) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        // SAFETY: signal 0 performs the permission/existence check only.
        if unsafe { libc::kill(pid, 0) } != 0 {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("pid {pid} is still alive");
}

/// 1×1 red PNG as a native data-URI — a payload that survives the media
/// gate's base64 + bounded raster decode; used wherever tests need a marker
/// the provider promotes to a vision part.
pub(crate) fn tiny_png_data_uri() -> String {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("test PNG must encode");
    format!("data:image/png;base64,{}", STANDARD.encode(&buf))
}

/// The single process-level test temp root: all test-owned global state needing a temp directory
/// (the CONFIG storage root: stores, user workspaces, models) resolves under it, so the process
/// owns one directory that self-cleans on normal exit (unix).
pub(crate) fn test_root() -> &'static PathBuf {
    TEST_ROOT.get_or_init(|| {
        let tmp = tempfile::TempDir::new().expect("failed to create test temp dir");
        let path = tmp.path().to_path_buf();
        // Deliberately leaked: keeping the TempDir guard would race with the exit-time removal
        // below. The atexit handler removes the directory (unix), else the OS temp sweep does.
        std::mem::forget(tmp);
        #[cfg(unix)]
        register_test_root_cleanup();
        path
    })
}

/// Register exit-time removal of the shared test root (unix).
///
/// Runs on normal exit — pass, failure and per-test panics alike, since the harness catches them
/// and the process still exits via `exit()`. Killed processes skip atexit handlers. Each process
/// creates its own root and removes only that one, and the handler is panic-free and guarded by the
/// creator PID, so a `fork()` child never removes its parent's root.
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

/// Open a temporary store for testing: `(store, TempDir)`, and the `TempDir` MUST be held for the
/// store's lifetime (typically bound to `_tmp` / `_dir`).
///
/// This is a macro, not a generic function, because the lifetime relationship between a closure
/// argument and the future of `async fn open(path: &Path) -> Result<T>` cannot be expressed by
/// `FnOnce` / `AsyncFnOnce` without boxing. Panics if the directory cannot be created or the store
/// fails to open.
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

/// Create a test ticket with just a title and phase; use [`TicketBuilder`] directly for `.desc()`
/// or `.prereqs()`. Panics if the ticket cannot be created, naming the title and phase.
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

/// Fetch a ticket by ID, panicking if the query fails or the ticket does not exist.
pub async fn expect_ticket(store: &BoardStore, id: &str) -> Ticket {
    store
        .get_ticket(id)
        .await
        .expect("BoardStore::get_ticket query failed")
        .expect("expected ticket to exist")
}

/// Fetch a ticket's phase by ID, panicking if the query fails or the ticket does not exist.
pub async fn expect_ticket_phase(store: &BoardStore, id: &str) -> TicketPhase {
    store
        .get_ticket_phase(id)
        .await
        .expect("BoardStore::get_ticket_phase query failed")
        .expect("expected ticket phase to exist")
}

/// Builder for test tickets: `desc="desc"`, `phase=Backlog`, `prerequisites=[]`,
/// `reporter="test"`, with the title required via `.title()`.
pub(crate) struct TicketBuilder<'a> {
    store: &'a BoardStore,
    ws: crate::Workspace,
    title: String,
    desc: String,
    phase: TicketPhase,
    prereqs: Vec<String>,
    reporter: String,
    priority: i64,
}

impl<'a> TicketBuilder<'a> {
    pub(crate) fn new(store: &'a BoardStore, ws: &crate::Workspace) -> Self {
        Self {
            store,
            ws: ws.clone(),
            title: String::new(),
            desc: "desc".into(),
            phase: TicketPhase::Backlog,
            prereqs: Vec::new(),
            reporter: "test".into(),
            priority: 1,
        }
    }

    pub(crate) fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub(crate) fn desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    pub(crate) fn phase(mut self, phase: TicketPhase) -> Self {
        self.phase = phase;
        self
    }

    pub(crate) fn prereqs(mut self, prereqs: &[String]) -> Self {
        self.prereqs = prereqs.to_vec();
        self
    }

    pub(crate) fn priority(mut self, priority: i64) -> Self {
        self.priority = priority;
        self
    }

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
                priority: self.priority,
                label: TicketLabel::Feat,
            },
        )
    }
}

/// In-memory [`Ticket`] fixture, for tests that need a `Ticket` value without a
/// store — distinct from the DB-insert [`TicketBuilder`].
pub(crate) struct TicketFixture(Ticket);

impl TicketFixture {
    pub(crate) fn new(id: &str, phase: TicketPhase) -> Self {
        Self(Ticket {
            id: id.into(),
            title: "Test ticket".into(),
            description: String::new(),
            phase,
            workspace_name: "test_ws".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            comments: Vec::new(),
            prerequisites: Vec::new(),
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            priority: 0,
            label: TicketLabel::Feat,
            reviewed_head: None,
            reviewed_tree: None,
            done_at: None,
            bounce_count: 0,
        })
    }

    pub(crate) fn title(mut self, title: &str) -> Self {
        self.0.title = title.into();
        self
    }

    pub(crate) fn workspace(mut self, workspace: &str) -> Self {
        self.0.workspace_name = workspace.into();
        self
    }

    pub(crate) fn archived(mut self, archived: bool) -> Self {
        self.0.is_archived = archived;
        self
    }
    pub(crate) fn label(mut self, label: TicketLabel) -> Self {
        self.0.label = label;
        self
    }

    pub(crate) fn comment(mut self, role: &str, content: &str, created_at: &str) -> Self {
        self.0.comments.push(TicketComment {
            role: role.into(),
            content: content.into(),
            created_at: created_at.into(),
        });
        self
    }

    pub(crate) fn build(self) -> Ticket {
        self.0
    }
}

/// Builder for inserting a test `jobs` row (the durability/resume substrate
/// in the consolidated domain database).
///
/// Mirrors the production INSERT in [`crate::jobs::spawn_job`] — when the
/// `jobs` schema changes, update both this builder and the production insert.
///
/// Required schema columns live in the constructor (`id`, `kind`, `role`,
/// `workspace_name`); the columns the fixtures vary between (user identity,
/// channel, retry count) are opt-in setters, and `task`
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

        // workspace_index is sync — must be initialized before workspace
        crate::workspace_index::init_global();

        // chronicle is sync — lightweight allocation, no DB I/O.
        crate::pipeline::chronicle::init_global();

        crate::db::init_all_stores()
            .await
            .expect("failed to initialize test stores (see chained error for per-store details)");

        // Start the CDC-driven chronicle subscriber so the timeline is
        // materialized from ticket change events in tests too.
        crate::pipeline::chronicle::start_subscriber();
    })
    .await;
}

/// Initialize all stores needed by management tests that interact with
/// the chronicle timeline.
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

/// Seed one session row directly via SQL (the store's `batch_append` is
/// `session`-module private).
pub(crate) async fn seed_session_row(
    conn: &crate::db::Connection,
    agent_id: &str,
    role: &str,
    content: &str,
) {
    conn.execute(
        "INSERT INTO sessions (agent_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
        db::params![agent_id, role, content, crate::db::now()],
    )
    .await
    .unwrap();
}

/// Make `dir` a repository git answers for, with no commit of its own — for the
/// nested repositories a test builds inside a folder it created itself.
///
/// # Panics
///
/// Panics if `git` is not available or refuses to initialize.
pub(crate) fn init_git_repo(dir: &Path) {
    let status = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(dir)
        .status()
        .expect("git init");
    assert!(status.success());
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
    init_git_repo(&repo_path);

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

/// Every `.rs` file under `dir`, recursively, in directory-walk order — the file
/// list the source-scanning tripwires walk (`db`'s store-handle guard and
/// `tools::shell::tree`'s window guarantee).
///
/// # Panics
///
/// Panics if `dir` cannot be read.
pub(crate) fn rs_files_under(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read source directory") {
            let path = entry.expect("read directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    walk(dir, &mut files);
    files
}

/// Where a source file sits relative to `manifest_dir`, spelled with `/`
/// separators — how the tripwires compare a walked file against their literal
/// lists, so the comparison does not depend on the host's separators.
///
/// # Panics
///
/// Panics if `file` is not under `manifest_dir`.
pub(crate) fn rel_source_path(manifest_dir: &Path, file: &Path) -> String {
    file.strip_prefix(manifest_dir)
        .expect("source files live under the manifest dir")
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// The one table assertion the case tables in this crate share: every
/// `contains` needle is present in `result` and every `not_contains` needle
/// absent, naming the case whose row broke — the name is the only thing a
/// reader of a table-driven failure has to go on.
pub(crate) fn assert_contains_not_contains(
    name: &str,
    result: &str,
    contains: &[&str],
    not_contains: &[&str],
) {
    for needle in contains {
        assert!(
            result.contains(needle),
            "[{name}] expected contains {needle:?}\n  got: {result:?}",
        );
    }
    for needle in not_contains {
        assert!(
            !result.contains(needle),
            "[{name}] expected NOT contains {needle:?}\n  got: {result:?}",
        );
    }
}

// These tests use `unsafe { std::env::set_var/remove_var }` directly rather than the guard — they
// test the wrapper's correctness. Accepted trade-off: unique-per-test names prevent collisions
// within this module, though per std::env docs even writes to different variables are a data race
// without the lock.

#[cfg(test)]
mod env_var_guard_tests {
    use super::*;

    #[test]
    fn sets_and_restores_the_value_it_replaced() {
        // Absent before the guard (unique name, first use): drop returns it to absent.
        let guard = set_env_var("MAHBOT_TEST_SET_RESTORE", Some("hello"));
        assert_eq!(std::env::var("MAHBOT_TEST_SET_RESTORE"), Ok("hello".into()));
        drop(guard);
        assert!(
            std::env::var_os("MAHBOT_TEST_SET_RESTORE").is_none(),
            "guard should restore env var to absent on drop"
        );

        // Present before the guard: drop puts the original value back.
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

    /// A panicking test must not leak either kind of previous state: absent
    /// stays absent, a value comes back.
    ///
    /// One guard at a time: [`set_env_var`] holds the process-wide
    /// [`env_lock`] for the guard's lifetime, so two live guards would deadlock
    /// the thread against itself.
    #[test]
    fn restores_the_previous_state_on_panic() {
        // SAFETY: Bypasses env_lock (unique var name — see mod-level doc).
        unsafe {
            std::env::remove_var("MAHBOT_TEST_PANIC_ABSENT");
            std::env::set_var("MAHBOT_TEST_PANIC_ORIGINAL", "original");
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
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    fn installs_and_restores_on_drop() {
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
    #[serial_test::serial(provider)] // serializes the process-global fake provider (providers::PROVIDER)
    fn restores_on_panic() {
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
}

/// Deterministic per-pixel noise that defeats image compression (used by the
/// reference-image loader and body-budget tests to build reliably large files).
fn noisy_image(width: u32, height: u32) -> image::RgbImage {
    use image::{ImageBuffer, Rgb};
    ImageBuffer::from_fn(width, height, |x, y| {
        let mut v = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
        v ^= v >> 13;
        v ^= v << 17;
        v ^= v >> 5;
        Rgb([
            (v & 0xFF) as u8,
            ((v >> 8) & 0xFF) as u8,
            ((v >> 16) & 0xFF) as u8,
        ])
    })
}

/// Encode `img` as a PNG.
fn png_of(img: image::RgbImage) -> Vec<u8> {
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

/// A PNG of deterministic noise.
pub fn noisy_png(width: u32, height: u32) -> Vec<u8> {
    png_of(noisy_image(width, height))
}

/// A single-colour PNG of the given size: many pixels, few bytes — for tests about dimensions
/// rather than about file size.
pub fn solid_png(width: u32, height: u32) -> Vec<u8> {
    png_of(image::RgbImage::from_pixel(
        width,
        height,
        image::Rgb([40, 90, 160]),
    ))
}

/// A 2x1 JPEG of a red and a blue pixel, stored unrotated: the smallest picture whose orientation
/// is visible in its dimensions.
pub fn red_blue_strip_jpeg() -> Vec<u8> {
    let strip = image::RgbImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 0, 255])
        }
    });
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut out)
        .encode_image(&image::DynamicImage::ImageRgb8(strip))
        .unwrap();
    out
}

/// Inject an EXIF orientation tag (0x0112 = 6 → Rotate90) into a JPEG right after the SOI marker.
/// The stored pixels stay unrotated; a viewer (or the compression path) must rotate them per the
/// tag.
pub fn jpeg_with_exif_orientation(jpeg: &[u8]) -> Vec<u8> {
    assert!(jpeg.starts_with(&[0xFF, 0xD8]));
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1, 0x00, 0x22]);
    out.extend_from_slice(b"Exif\0\0");
    // Little-endian TIFF: header (8) + IFD (2 count + 12 entry + 4 next).
    out.extend_from_slice(&[
        0x49, 0x49, 0x2A, 0x00, // "II", 42
        0x08, 0x00, 0x00, 0x00, // IFD offset = 8
        0x01, 0x00, // 1 entry
        0x12, 0x01, 0x03, 0x00, // tag 0x0112, type SHORT
        0x01, 0x00, 0x00, 0x00, // count = 1
        0x06, 0x00, 0x00, 0x00, // value = 6 (Rotate90)
        0x00, 0x00, 0x00, 0x00, // next IFD = none
    ]);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// The canonical path as the resolvers hand it on: the platform's
/// [`std::fs::canonicalize`] with the Windows verbatim prefix stripped.
pub fn canonical_without_verbatim_prefix(path: &Path) -> PathBuf {
    crate::util::strip_verbatim_prefix(&std::fs::canonicalize(path).expect("canonicalize"))
}

/// The `mahbot` binary this test's own build put beside the harness:
/// `target/<profile>/mahbot` from `target/<profile>/deps/<test binary>`.
///
/// `None` when the binary was never built (a `cargo test --lib` run that did not
/// build the bin), which each lane reports its own way. Unix-gated with the lanes
/// that drive it.
#[cfg(unix)]
pub(crate) fn built_binary() -> Option<PathBuf> {
    let harness = std::env::current_exe().ok()?;
    let candidate = harness.parent()?.parent()?.join("mahbot");
    candidate.is_file().then_some(candidate)
}

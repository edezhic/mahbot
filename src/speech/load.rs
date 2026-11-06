//! One lazily-opened engine over the machine-wide model set: the lifecycle both speech engines
//! run — nothing loads at boot, the first operation that needs an engine loads it (from the
//! files on disk, or by fetching them first), and an engine idle for [`IDLE_RELEASE`] is
//! dropped, then loaded again by its next use.
//!
//! The state is [`crate::util::model_state::ModelState`]: Uninit → Loading → Ready →
//! Released / Failed. One engine handle exists per [`Lazy`], cloned out from under its lock
//! rather than held across a pass; serialisation is each engine's own — the synthesiser's
//! single pass lock, the recogniser's transcription permit and the listener's mutex. A caller
//! that needs an engine either waits out a load already in flight ([`Lazy::ensure_loaded`]) or
//! is answered from the state; a fetch is never waited for.
//!
//! A release is taken only when nothing holds the engine ([`Lazy::try_release`]): a pass keeps
//! the handle it cloned, and a caller waiting for one is counted. `Failed` never ends a surface:
//! the next use re-opens a set that is on disk and starts the fetch again for one that is not, so
//! an idle release can never leave an engine silently off.

use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context as _, Result};
use futures_util::FutureExt as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::speech::models::{self, Purpose};

/// Retry sleep base: 5 seconds, doubled per attempt.
const RETRY_BASE_SECS: u64 = 5;

/// Attempts one background fetch-and-open chain makes before it gives up and reports `Failed`.
const MAX_LOAD_RETRIES: u32 = 12;

/// How often a caller that needs an engine checks whether an in-flight load has settled.
const LOAD_WAIT_POLL: Duration = Duration::from_millis(100);

/// The longest a caller waits for a load already in flight: a set on disk reads in seconds, and
/// a fetch is not waited on at all.
const LOAD_WAIT_LIMIT: Duration = Duration::from_mins(2);

/// How long an engine stays loaded with no use before it is dropped and its memory handed back.
pub(crate) const IDLE_RELEASE: Duration = Duration::from_mins(5);

/// Milliseconds since the process's first engine use, the clock the idle timer reads.
fn now_millis() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// One engine and the lifecycle that opens it. `open` is the engine's own constructor, run off
/// the async threads: it maps a pair of files and compiles graphs.
pub(crate) struct Lazy<E: Send + 'static> {
    purpose: Purpose,
    open: fn() -> Result<E>,
    engine: Mutex<Option<Arc<Mutex<E>>>>,
    state: AtomicModelState,
    /// True once a chain has moved past checking the files on disk to fetching them. A caller
    /// waiting for `Loading` stops waiting when this is set: a cache read takes seconds, a
    /// fetch takes minutes, and whoever just asked gets an answer either way.
    fetching: AtomicBool,
    /// When the engine was last used, in [`now_millis`] terms; the idle release reads it.
    last_use: AtomicU64,
    /// Callers between `ensure_loaded` and the handle they cloned: a release waits them out.
    busy: AtomicUsize,
}

impl<E: Send + 'static> Lazy<E> {
    pub(crate) const fn new(purpose: Purpose, open: fn() -> Result<E>) -> Self {
        Self {
            purpose,
            open,
            engine: Mutex::new(None),
            state: AtomicModelState::new(ModelState::Uninit),
            fetching: AtomicBool::new(false),
            last_use: AtomicU64::new(0),
            busy: AtomicUsize::new(0),
        }
    }

    fn title(&self) -> &'static str {
        self.purpose.title()
    }

    pub(crate) fn state(&self) -> ModelState {
        self.state.load(Ordering::Acquire)
    }

    pub(crate) fn is_loaded(&self) -> bool {
        self.state.is_ready()
    }

    pub(crate) fn is_loading(&self) -> bool {
        self.state() == ModelState::Loading
    }

    pub(crate) fn is_failed(&self) -> bool {
        self.state() == ModelState::Failed
    }

    /// The opened engine, or `None` until a load published one. The `Arc` is cloned and the
    /// lock released before the caller's blocking pass, so the handle is never held across it.
    /// Taking a handle counts as a use: the idle timer restarts here.
    pub(crate) fn engine(&self) -> Option<Arc<Mutex<E>>> {
        self.touch();
        self.engine.lock().unwrap_poison().clone()
    }

    /// The opened engine, loading it first when it was released or never started: the caller
    /// needs the engine for a pass, so a load on disk is waited out. `None` when the set cannot
    /// be loaded (a fetch was started, or the load failed).
    pub(crate) async fn acquire(&'static self, dir: &Path) -> Option<Arc<Mutex<E>>> {
        let _busy = Busy::new(&self.busy);
        if !self.ensure_loaded(dir).await {
            return None;
        }
        // A release that raced the start of this window is healed by the load above; otherwise
        // `Busy` kept one away and the engine is still there.
        self.engine()
    }

    /// Restart the idle timer.
    fn touch(&self) {
        self.last_use.store(now_millis(), Ordering::Release);
    }

    /// Whether the engine has gone [`IDLE_RELEASE`] without a use.
    fn idle_for_release(&self) -> bool {
        let idle = u64::try_from(IDLE_RELEASE.as_millis()).unwrap_or(u64::MAX);
        now_millis().saturating_sub(self.last_use.load(Ordering::Acquire)) >= idle
    }

    /// Drop the engine when it has had no use for [`IDLE_RELEASE`], returning whether it was
    /// released. A pass holds a cloned handle and a caller waiting for one is counted, so a
    /// release never takes an engine out from under either.
    pub(crate) fn release_if_idle(&self) -> bool {
        if !self.state.is_ready() {
            return false;
        }
        // A pass in flight is a use: the window restarts while it runs, so an engine is never
        // released the moment a long pass ends. This is read every tick, so the window is exact
        // to within one poll.
        if self.in_use() {
            self.touch();
            return false;
        }
        self.idle_for_release() && self.try_release("after an idle period")
    }

    /// Whether a pass holds a cloned handle right now.
    fn in_use(&self) -> bool {
        self.engine
            .lock()
            .unwrap_poison()
            .as_ref()
            .is_some_and(|engine| Arc::strong_count(engine) > 1)
    }

    /// Drop the engine whatever its idle time, unless a pass holds it: the process-wide memory
    /// clear needs every engine cold. Returns whether the engine is cold afterwards.
    pub(crate) fn release(&self) -> bool {
        if self.state.is_ready() && !self.try_release("for the process-wide memory clear") {
            return false;
        }
        self.is_cold()
    }

    /// Whether nothing is resident and nothing is loading.
    pub(crate) fn is_cold(&self) -> bool {
        !matches!(self.state(), ModelState::Ready | ModelState::Loading)
    }

    /// Drop the engine and mark it released, unless something holds it. `why` completes the log
    /// line, so an idle release and the clear's forced one never read the same.
    fn try_release(&self, why: &'static str) -> bool {
        if self.busy.load(Ordering::Acquire) > 0 {
            return false;
        }
        let mut guard = self.engine.lock().unwrap_poison();
        let Some(engine) = guard.as_ref() else {
            return false;
        };
        // A cloned handle is a pass in flight: it keeps the engine until it is done.
        if Arc::strong_count(engine) > 1 {
            return false;
        }
        // The state moves before the slot empties, so a caller that read `Ready` a moment ago
        // sees `Released` on its next look instead of finding nothing under `Ready`.
        if !self
            .state
            .transition(ModelState::Ready, ModelState::Released)
        {
            return false;
        }
        let taken = guard.take();
        drop(guard);
        drop(taken);
        info!("{}: released {why}", self.title());
        true
    }

    /// Test-only: pin the lifecycle state, for a caller exercising the states a real load takes
    /// minutes to reach.
    #[cfg(test)]
    pub(crate) fn test_set_state(&self, state: ModelState) {
        self.state.store(state, Ordering::Release);
    }

    /// Retry after a `Failed` state: reset it and re-run the background chain. Returns
    /// `true` when a retry was initiated.
    pub(crate) fn retry_init(&'static self) -> bool {
        if !self
            .state
            .transition(ModelState::Failed, ModelState::Uninit)
        {
            return false;
        }
        self.spawn_background_init();
        true
    }

    /// Move `Uninit`/`Released` to `Loading` and start the chain's file check from a clean slate:
    /// a caller waiting for this load must not be answered by the previous one's fetch. The flag
    /// is cleared right after the transition, so a waiter that observes the new state can read the
    /// previous chain's value for a couple of instructions at worst.
    fn begin_load(&self) -> bool {
        let started = self
            .state
            .transition(ModelState::Uninit, ModelState::Loading)
            || self
                .state
                .transition(ModelState::Released, ModelState::Loading);
        if started {
            self.fetching.store(false, Ordering::Release);
        }
        started
    }

    /// Retry after a `Failed` state, or start the engine when it has not been asked for yet: a
    /// load that failed never stays off until someone opens Settings.
    pub(crate) fn retry(&'static self) {
        if !self.retry_init() {
            self.spawn_background_init();
        }
    }

    /// Demand the engine: start the load-or-fetch chain in the background. A no-op when it is
    /// already ready or already loading — the caller has already decided the engine is needed
    /// (a first use, a saved toggle, an explicit retry), and nothing here is awaited. A
    /// released engine is requested again, which is what keeps a listening session alive after
    /// an idle release.
    pub(crate) fn spawn_background_init(&'static self) {
        if !self.begin_load() {
            debug!("{}: background init skipped", self.title());
            return;
        }
        tokio::spawn(async move {
            // A panic must never leave the state stuck in Loading, silently disabling the
            // engine until a restart.
            if std::panic::AssertUnwindSafe(self.load_retry_loop())
                .catch_unwind()
                .await
                .is_err()
            {
                warn!(
                    "{}: background init panicked — marking failed",
                    self.title()
                );
                self.state.store(ModelState::Failed, Ordering::Release);
            }
        });
    }

    /// The awaited "make sure it is loaded": a load already in flight is waited out, not
    /// reported unavailable — a voice press starts the load a moment before the pass that
    /// needs its result. Only a set already on disk is opened here; a fetch is spawned detached
    /// and answers `false` immediately.
    ///
    /// A failed load over files that are on disk is retried here, and a failed fetch is started
    /// again: an engine released for being idle, or a fetch that ran out of attempts, must never
    /// stay off until an explicit retry.
    pub(crate) async fn ensure_loaded(&'static self, dir: &Path) -> bool {
        loop {
            match self.state() {
                ModelState::Ready => {
                    self.touch();
                    return true;
                }
                ModelState::Loading => {
                    return models::usable(dir, self.purpose) && self.wait_until_settled().await;
                }
                ModelState::Failed => {
                    if !models::usable(dir, self.purpose) {
                        // A fetch that gave up: the next use starts it again, so a failed download
                        // is never terminal for a surface that never opens Settings.
                        self.retry_init();
                        return false;
                    }
                    // A lost race means another task moved the state; the loop re-reads it.
                    let _ = self
                        .state
                        .transition(ModelState::Failed, ModelState::Uninit);
                }
                ModelState::Released | ModelState::Uninit => {
                    if !models::usable(dir, self.purpose) {
                        // What is left is a fetch, which the caller does not wait for: let the
                        // background chain own every later transition.
                        self.spawn_background_init();
                        return false;
                    }
                    if !self.begin_load() {
                        continue;
                    }
                    let loaded = {
                        let _guard = ModelLoadGuard::new(&self.state);
                        self.open_and_publish().await
                    };
                    match loaded {
                        Ok(()) => return true,
                        Err(e) => {
                            warn!("{}: {e:#}", self.title());
                            return false;
                        }
                    }
                }
            }
        }
    }

    /// Obtain the set once, with a notice before a fetch that really begins, and open the
    /// engine — the command line's own entry point: one attempt, no retry loop, and a failure
    /// is the caller's to report.
    pub(crate) async fn ensure_ready_once(&self, announce: impl FnOnce(&Path)) -> Result<()> {
        let mut said = Some(announce);
        let mut fetch_starting = |dir: &Path| {
            if let Some(say) = said.take() {
                say(dir);
            }
        };
        self.load_or_fetch(&mut fetch_starting).await
    }

    /// Wait for an in-flight load to settle: ends as soon as the answer is known — `Ready`,
    /// `Failed`/`Uninit`/`Released`, a chain that has moved on to fetching, or
    /// [`LOAD_WAIT_LIMIT`].
    async fn wait_until_settled(&self) -> bool {
        let deadline = tokio::time::Instant::now() + LOAD_WAIT_LIMIT;
        loop {
            match self.state() {
                ModelState::Ready => return true,
                // Failed: the load being waited on ended in failure. Uninit: an explicit retry
                // reset it, so the load this call waited on is gone. Released: the engine was
                // dropped between the load and this wait.
                ModelState::Failed | ModelState::Uninit | ModelState::Released => return false,
                ModelState::Loading if self.fetching.load(Ordering::Acquire) => return false,
                ModelState::Loading => {}
            }
            if tokio::time::Instant::now() >= deadline {
                warn!("{}: the model load did not settle in time", self.title());
                return false;
            }
            tokio::time::sleep(LOAD_WAIT_POLL).await;
        }
    }

    /// One background chain: check-or-fetch the set, open the engine, retry with backoff until
    /// the attempt budget is spent.
    async fn load_retry_loop(&'static self) {
        let _guard = ModelLoadGuard::new(&self.state);
        let mut delay = Duration::from_secs(RETRY_BASE_SECS);
        for attempt in 1..=MAX_LOAD_RETRIES {
            info!(
                "{}: model load attempt {attempt}/{MAX_LOAD_RETRIES}",
                self.title()
            );
            match self
                .load_or_fetch(&mut |_dir| {
                    self.fetching.store(true, Ordering::Release);
                })
                .await
            {
                Ok(()) => return,
                Err(e) => warn!("{}: {e:#}", self.title()),
            }
            // The next attempt checks the files again before it can fetch, so a waiter is back to
            // waiting on a load rather than being told a download is running.
            self.fetching.store(false, Ordering::Release);
            // A shutdown or drain ends the chain rather than sleeping through it.
            if !crate::shutdown::sleep_or_shutdown_or_drain(delay).await {
                return;
            }
            delay = (delay * 2).min(Duration::from_mins(2));
        }
        warn!(
            "{}: the model set could not be loaded; giving up",
            self.title()
        );
        self.state.store(ModelState::Failed, Ordering::Release);
    }

    /// Check the set, fetch what is missing or wrong, and open the engine over it. The only
    /// chain that may download.
    async fn load_or_fetch(&self, fetch_starting: &mut impl FnMut(&Path)) -> Result<()> {
        // A load that finds the set whole is a read, which a caller waits out; the flag is
        // what tells it a fetch has begun instead.
        self.fetching.store(false, Ordering::Release);
        let dir = models::root()?;
        let client = models::download_client()?;
        models::ensure_in(&dir, self.purpose, &client, fetch_starting).await?;
        self.open_and_publish().await
    }

    /// Open the engine over the checked set and publish it as ready.
    async fn open_and_publish(&self) -> Result<()> {
        let open = self.open;
        let engine = tokio::task::spawn_blocking(open)
            .await
            .context("the engine's load task panicked")??;
        *self.engine.lock().unwrap_poison() = Some(Arc::new(Mutex::new(engine)));
        self.touch();
        self.state.store(ModelState::Ready, Ordering::Release);
        info!("{}: the engine is ready", self.title());
        Ok(())
    }
}

/// Counts one caller between asking for an engine and taking its handle, so a release cannot
/// take the engine out from under a request that is still waiting for it.
struct Busy<'a>(&'a AtomicUsize);

impl<'a> Busy<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the fake engine's constructor must match Lazy::new's fallible signature"
)]
mod tests {
    use super::*;
    use crate::speech::models::write_sparse_set;

    /// The engine a test opens: the counter is how a reload is seen.
    struct Fake;

    fn set_on_disk() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("a temp dir");
        write_sparse_set(dir.path(), Purpose::Synthesis);
        dir
    }

    /// A release never takes an engine a pass holds, and the next use loads it again — the
    /// release-to-reload cycle the idle timer drives.
    #[tokio::test]
    async fn a_held_engine_is_not_released_and_a_released_one_is_loaded_again() {
        static ENGINE: Lazy<Fake> = Lazy::new(Purpose::Synthesis, open);
        static OPENS: AtomicUsize = AtomicUsize::new(0);
        fn open() -> Result<Fake> {
            OPENS.fetch_add(1, Ordering::AcqRel);
            Ok(Fake)
        }

        let dir = set_on_disk();
        assert!(ENGINE.acquire(dir.path()).await.is_some());
        assert!(ENGINE.is_loaded());
        assert!(!ENGINE.release_if_idle(), "an engine just used is not idle");

        let held = ENGINE.engine().expect("the handle");
        assert!(
            !ENGINE.try_release("the test"),
            "a pass holding a handle keeps the engine"
        );
        drop(held);
        assert!(ENGINE.try_release("the test"));
        assert_eq!(ENGINE.state(), ModelState::Released);
        assert!(ENGINE.engine().is_none(), "the released engine is gone");

        assert!(ENGINE.acquire(dir.path()).await.is_some());
        assert!(ENGINE.is_loaded());
        assert_eq!(OPENS.load(Ordering::Acquire), 2, "the next use reloaded it");
    }

    /// A load that failed over a set on disk is retried by the next use rather than staying off.
    #[tokio::test]
    async fn a_failed_load_over_files_on_disk_is_retried_by_the_next_use() {
        static ENGINE: Lazy<Fake> = Lazy::new(Purpose::Synthesis, open);
        static OPENS: AtomicUsize = AtomicUsize::new(0);
        fn open() -> Result<Fake> {
            OPENS.fetch_add(1, Ordering::AcqRel);
            Ok(Fake)
        }

        let dir = set_on_disk();
        ENGINE.test_set_state(ModelState::Failed);
        assert!(ENGINE.ensure_loaded(dir.path()).await);
        assert!(ENGINE.is_loaded());
        assert_eq!(OPENS.load(Ordering::Acquire), 1);
    }

    /// A load that starts after a fetch is a load on disk: it must not inherit the previous
    /// chain's "this is a download" flag, or its waiters would be refused instead of waiting.
    #[test]
    fn a_new_load_does_not_inherit_the_previous_fetch() {
        static ENGINE: Lazy<Fake> = Lazy::new(Purpose::Synthesis, open);
        fn open() -> Result<Fake> {
            Ok(Fake)
        }

        ENGINE.fetching.store(true, Ordering::Release);
        ENGINE.test_set_state(ModelState::Released);
        assert!(ENGINE.begin_load());
        assert!(!ENGINE.fetching.load(Ordering::Acquire));
    }
}

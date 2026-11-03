//! One lazily-opened engine over the machine-wide model set: the lifecycle both speech engines
//! run — nothing loads at boot, the first operation that needs an engine loads it (from the
//! files on disk, or by fetching them first), and a terminal `Failed` is left only by an
//! explicit retry.
//!
//! The state is [`crate::util::model_state::ModelState`]: Uninit → Loading → Ready / Failed.
//! One engine handle exists per [`Lazy`], cloned out from under its lock rather than held across
//! a pass; serialisation is each engine's own — the synthesiser's single pass lock, the
//! recogniser's one per consumer. A caller that needs an engine either waits out a load already
//! in flight ([`Lazy::ensure_loaded`]) or is answered from the state; a fetch is never waited
//! for.

use crate::util::UnwrapPoison;
use crate::util::model_state::{AtomicModelState, ModelLoadGuard, ModelState};
use anyhow::{Context as _, Result};
use futures_util::FutureExt as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::speech::models::{self, Purpose};

/// Retry sleep base: 5 seconds, doubled per attempt.
const RETRY_BASE_SECS: u64 = 5;

/// Attempts one background fetch-and-open chain makes before `Failed` is terminal.
const MAX_LOAD_RETRIES: u32 = 12;

/// How often a caller that needs an engine checks whether an in-flight load has settled.
const LOAD_WAIT_POLL: Duration = Duration::from_millis(100);

/// The longest a caller waits for a load already in flight: a set on disk reads in seconds, and
/// a fetch is not waited on at all.
const LOAD_WAIT_LIMIT: Duration = Duration::from_mins(2);

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
}

impl<E: Send + 'static> Lazy<E> {
    pub(crate) const fn new(purpose: Purpose, open: fn() -> Result<E>) -> Self {
        Self {
            purpose,
            open,
            engine: Mutex::new(None),
            state: AtomicModelState::new(ModelState::Uninit),
            fetching: AtomicBool::new(false),
        }
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
    pub(crate) fn engine(&self) -> Option<Arc<Mutex<E>>> {
        self.engine.lock().unwrap_poison().clone()
    }

    /// Test-only: pin the lifecycle state, for a caller exercising the states a real load takes
    /// minutes to reach.
    #[cfg(test)]
    pub(crate) fn test_set_state(&self, state: ModelState) {
        self.state.store(state, Ordering::Release);
    }

    /// Retry after a terminal `Failed` state: reset it and re-run the background chain. Returns
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

    /// Demand the engine: start the load-or-fetch chain in the background. A no-op when it is
    /// already ready, already loading, or terminally failed — the caller has already decided
    /// the engine is needed (a first use, a saved toggle, an explicit retry), and nothing here
    /// is awaited.
    pub(crate) fn spawn_background_init(&'static self) {
        if !self
            .state
            .transition(ModelState::Uninit, ModelState::Loading)
        {
            debug!("{}: background init skipped", self.purpose.title());
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
                    self.purpose.title()
                );
                self.state.store(ModelState::Failed, Ordering::Release);
            }
        });
    }

    /// The awaited "make sure it is loaded": a load already in flight is waited out, not
    /// reported unavailable — a voice press starts the load a moment before the pass that
    /// needs its result. Only a set already on disk is opened here; a fetch is spawned detached
    /// and answers `false` immediately.
    pub(crate) async fn ensure_loaded(&'static self, dir: &Path) -> bool {
        match self.state.load(Ordering::Acquire) {
            ModelState::Ready => return true,
            ModelState::Uninit => {}
            _ => return models::present(dir, self.purpose) && self.wait_until_settled().await,
        }
        if !self
            .state
            .transition(ModelState::Uninit, ModelState::Loading)
        {
            return self.state.load(Ordering::Acquire) == ModelState::Ready;
        }
        if models::present(dir, self.purpose) {
            match self.load_from_disk(dir).await {
                Ok(()) => return true,
                Err(e) => warn!("{}: {e:#}", self.purpose.title()),
            }
        }
        // What is left is a fetch, which the caller does not wait for: hand the state back
        // and let the background chain own every later transition.
        self.state.store(ModelState::Uninit, Ordering::Release);
        self.spawn_background_init();
        false
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
    /// `Failed`/`Uninit`, a chain that has moved on to fetching, or [`LOAD_WAIT_LIMIT`].
    async fn wait_until_settled(&self) -> bool {
        let deadline = tokio::time::Instant::now() + LOAD_WAIT_LIMIT;
        loop {
            match self.state.load(Ordering::Acquire) {
                ModelState::Ready => return true,
                // Failed: the load being waited on ended in failure. Uninit: an explicit retry
                // reset it, so the load this call waited on is gone.
                ModelState::Failed | ModelState::Uninit => return false,
                ModelState::Loading if self.fetching.load(Ordering::Acquire) => return false,
                ModelState::Loading => {}
            }
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    "{}: the model load did not settle in time",
                    self.purpose.title()
                );
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
                self.purpose.title()
            );
            match self
                .load_or_fetch(&mut |_dir| {
                    self.fetching.store(true, Ordering::Release);
                })
                .await
            {
                Ok(()) => return,
                Err(e) => warn!("{}: {e:#}", self.purpose.title()),
            }
            // A shutdown or drain ends the chain rather than sleeping through it.
            if !crate::shutdown::sleep_or_shutdown_or_drain(delay).await {
                return;
            }
            delay = (delay * 2).min(Duration::from_mins(2));
        }
        warn!(
            "{}: the model set could not be loaded; giving up",
            self.purpose.title()
        );
        self.state.store(ModelState::Failed, Ordering::Release);
    }

    /// Open the engine over a set already on disk — verified, never fetched: a caller is
    /// waiting for the pass, and a fetch takes minutes.
    async fn load_from_disk(&self, dir: &Path) -> Result<()> {
        models::verify_set(dir, self.purpose).await?;
        self.open_and_publish().await
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

    /// Open the engine over the verified set and publish it as ready.
    async fn open_and_publish(&self) -> Result<()> {
        let open = self.open;
        let engine = tokio::task::spawn_blocking(open)
            .await
            .context("the engine's load task panicked")??;
        *self.engine.lock().unwrap_poison() = Some(Arc::new(Mutex::new(engine)));
        self.state.store(ModelState::Ready, Ordering::Release);
        info!("{}: the engine is ready", self.purpose.title());
        Ok(())
    }
}

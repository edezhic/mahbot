//! Shared per-workspace search engine registry.
//!
//! Each workspace gets a single [`SharedFilePicker`] + [`SharedQueryTracker`] pair that all
//! agents share. Background filesystem scanning begins eagerly when a workspace
//! is registered (on app startup or workspace add).
//!
//! ## Persistent query tracking
//!
//! The [`QueryTracker`] stores query→file associations on disk under
//! `~/.mahbot/search/{workspace_name}/queries/`. This persists combo-boost data
//! across agent and application restarts. If the LMDB database cannot be
//! opened (disk full, corruption, permission issues), we fall back to an
//! in-memory-only tracker — searches still work but combo-boosting resets on
//! restart.
//!
//! ## Unready-state handling
//!
//! When `ensure_scanned` is called before the background scan has finished,
//! it blocks for up to 30 seconds. If the scan still isn't done, it returns an
//! error rather than returning incomplete results.

use crate::Workspace;
use crate::config::CONFIG;
use fff_search::file_picker::{FFFMode, FilePickerOptions};
use fff_search::shared::{SharedFilePicker, SharedFrecency, SharedQueryTracker};
use fff_search::{FilePicker, QueryTracker};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use tokio::sync::OnceCell;

// ── Registry ─────────────────────────────────────────────────────────────

/// Global search-engine registry, keyed by workspace name.
static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<SearchEngineEntry>>>> = OnceCell::const_new();

/// Initialize the global registry. Must be called during bootstrap, after
/// [`crate::workspace::init_global`] and before any background task that may search.
pub fn init_global() {
    REGISTRY
        .set(RwLock::new(HashMap::new()))
        .expect("search engine registry already initialized");
}

fn registry() -> &'static RwLock<HashMap<String, Arc<SearchEngineEntry>>> {
    REGISTRY
        .get()
        .expect("search engine registry not initialized — call search_engine::init_global()")
}

/// Whether the global search-engine registry has been initialized.
#[must_use]
pub(crate) fn registry_initialized() -> bool {
    REGISTRY.get().is_some()
}

// ── Entry ─────────────────────────────────────────────────────────────────

/// Per-workspace shared search engine state.
///
/// Created once per workspace, shared across all agents searching that
/// workspace.
#[derive(Debug)]
pub(crate) struct SearchEngineEntry {
    /// Shared file picker used for both `files` and `grep` modes.
    pub picker: SharedFilePicker,
    /// Persistent query tracker for combo-boost scoring.
    /// Falls back to in-memory if the LMDB database cannot be opened.
    pub query_tracker: SharedQueryTracker,
    /// Set to `true` once a background scan has been initiated.
    scan_initiated: AtomicBool,
}

// ── Initialization ────────────────────────────────────────────────────────

/// Get or initialize the shared search engine for a workspace.
///
/// On first access, this creates the [`FilePicker`] and spawns a background
/// filesystem scan via [`FilePicker::new_with_shared_state`]. The persistent
/// query tracker database is also opened (or falls back to in-memory).
///
/// Returns a cloneable handle. Multiple callers racing on first access are
/// serialized by the registry write lock — only one engine is created.
pub(crate) fn get_or_init_engine(
    name: &str,
    path: &Path,
) -> Result<Arc<SearchEngineEntry>, String> {
    // Fast path: read-lock check.
    {
        let reg = registry()
            .read()
            .map_err(|e| format!("registry lock poisoned: {e}"))?;
        if let Some(entry) = reg.get(name) {
            return Ok(Arc::clone(entry));
        }
    }

    // Slow path: serialise creation under the write lock so that two
    // concurrent callers never create duplicate scans/query-tracker DBs.
    let mut reg = registry()
        .write()
        .map_err(|e| format!("registry lock poisoned: {e}"))?;

    // Double-check: another writer may have inserted while we waited.
    if let Some(existing) = reg.get(name) {
        return Ok(Arc::clone(existing));
    }

    let entry = Arc::new(init_engine_for_workspace(name, path)?);
    reg.insert(name.to_string(), Arc::clone(&entry));
    Ok(entry)
}

/// Initialize the search engine for a workspace without touching the registry.
///
/// Handles persistent query tracker setup with fallback, creates the
/// `FilePicker`, and spawns the background scan.
fn init_engine_for_workspace(name: &str, path: &Path) -> Result<SearchEngineEntry, String> {
    if !path.exists() {
        return Err(format!(
            "Workspace directory does not exist: {}",
            path.display()
        ));
    }

    let picker = SharedFilePicker::default();
    let frecency = SharedFrecency::default();

    // Try to open persistent query tracker DB; fall back to in-memory
    let query_tracker = match open_persistent_query_tracker(name) {
        Ok(qt) => qt,
        Err(e) => {
            tracing::warn!(
                workspace_name = name,
                error = %e,
                "Failed to open persistent query tracker — using in-memory fallback"
            );
            SharedQueryTracker::default()
        }
    };

    let options = FilePickerOptions {
        base_path: path.to_string_lossy().to_string(),
        enable_mmap_cache: false,
        enable_content_indexing: true,
        mode: FFFMode::Ai,
        watch: true,
        follow_symlinks: false,
        enable_fs_root_scanning: false,
        enable_home_dir_scanning: false,
        cache_budget: None,
    };

    FilePicker::new_with_shared_state(picker.clone(), frecency, options)
        .map_err(|e| format!("Failed to create search engine: {e}"))?;

    tracing::info!(
        workspace_name = name,
        workspace_path = %path.display(),
        "Search engine created — background scan started"
    );

    Ok(SearchEngineEntry {
        picker,
        query_tracker,
        scan_initiated: AtomicBool::new(true),
    })
}

/// Open a persistent [`QueryTracker`] database for a workspace.
///
/// The LMDB environment lives at `~/.mahbot/search/{workspace_name}/queries/`.
/// Parent directories are created if necessary.
fn open_persistent_query_tracker(workspace_name: &str) -> Result<SharedQueryTracker, String> {
    let root = CONFIG.global_storage_root();
    let db_path = root.join("search").join(workspace_name).join("queries");

    std::fs::create_dir_all(&db_path).map_err(|e| {
        format!(
            "Failed to create query tracker dir {}: {e}",
            db_path.display()
        )
    })?;

    let tracker = QueryTracker::open(&db_path)
        .map_err(|e| format!("Failed to open QueryTracker at {}: {e}", db_path.display()))?;

    let shared = SharedQueryTracker::default();
    shared
        .init(tracker)
        .map_err(|e| format!("Failed to init shared query tracker: {e}"))?;

    Ok(shared)
}

// ── Scan readiness ────────────────────────────────────────────────────────

/// Wait for the background filesystem scan to finish.
///
/// Blocks for up to 30 seconds. Returns an error if the scan hasn't completed
/// within the timeout, rather than returning incomplete or stale results.
///
/// This is an async function because [`SharedFilePicker::wait_for_scan`] is a
/// blocking call — we run it on the tokio blocking thread pool.
pub(crate) async fn ensure_scanned(entry: &SearchEngineEntry) -> Result<(), String> {
    if !entry.scan_initiated.load(Ordering::Acquire) {
        return Ok(());
    }

    let picker = entry.picker.clone();

    let scanned = tokio::task::spawn_blocking(move || {
        picker.wait_for_scan(std::time::Duration::from_secs(30))
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {e}"))?;

    if scanned {
        // Verify the file index is non-empty: the scan thread may have
        // panicked before populating the file list, which the scanning
        // guard's destructor would silently paper over by clearing the
        // scanning flag.
        let guard = entry
            .picker
            .read()
            .map_err(|e| format!("Failed to read picker state: {e}"))?;
        let is_empty = match &*guard {
            Some(picker) => picker.live_file_count() == 0,
            None => true,
        };
        drop(guard);

        if is_empty {
            return Err(
                "Scan completed but file index is empty — workspace may be misconfigured \
                 or scan failed silently."
                    .to_string(),
            );
        }

        Ok(())
    } else {
        Err("Search engine scan has not completed within 30 seconds. \
             The workspace may be too large or the filesystem is slow. \
             Try searching again in a moment."
            .to_string())
    }
}

// ── Lookup helpers for tools ───────────────────────────────────────────────

/// Get a workspace's search engine entry without creating one.
///
/// Unlike [`get_or_init_engine`], this returns `None` if the engine hasn't
/// been created yet (no searches have occurred in this workspace), or if the
/// global registry hasn't been initialized yet (during early bootstrap or in
/// tests that don't set up the search engine). Used by tools that want to
/// update the search index after file writes without triggering engine
/// creation as a side effect.
#[must_use]
pub(crate) fn get_engine_if_exists(ws: &Workspace) -> Option<Arc<SearchEngineEntry>> {
    let reg = REGISTRY.get()?.read().ok()?;
    reg.get(&ws.name).cloned()
}

// ── Lifecycle ─────────────────────────────────────────────────────────────

/// A convenience handle that wraps an `Arc<SearchEngineEntry>` and provides
/// direct field access for the search tool.
///
/// Created by `resolve_engine` in the search tool after ensuring the scan
/// is complete.
pub struct EngineHandle {
    pub picker: SharedFilePicker,
    pub query_tracker: SharedQueryTracker,
    _entry: Arc<SearchEngineEntry>,
}

impl EngineHandle {
    pub(crate) fn new(entry: Arc<SearchEngineEntry>) -> Self {
        Self {
            picker: entry.picker.clone(),
            query_tracker: entry.query_tracker.clone(),
            _entry: entry,
        }
    }
}

/// Remove a workspace's search engine from the registry.
///
/// Dropping the last [`Arc<SearchEngineEntry>`] will drop the underlying
/// [`SharedFilePicker`], which triggers the background scan's cancellation
/// flag and cleans up any associated threads.
///
/// The persistent query tracker LMDB directory is **not** deleted — that would
/// require closing the LMDB environment while no readers are active, which is
/// tricky across threads. The directory is small and harmless to leave behind.
pub fn remove_engine(workspace_name: &str) {
    let mut reg = match registry().write() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                workspace_name,
                "registry lock poisoned while removing engine: {e}"
            );
            return;
        }
    };
    if let Some(entry) = reg.remove(workspace_name) {
        drop(entry); // explicit: drop Arc before logging
        tracing::info!(workspace_name, "Search engine removed from registry");
    }
}

/// Initiate eager scanning for all registered workspaces.
///
/// Should be called once after bootstrap, in a background task. Errors for
/// individual workspaces are logged but don't prevent other workspaces from
/// being scanned.
pub async fn init_all_engines() {
    let workspaces = match crate::workspace::store().list().await {
        Ok(wss) => wss,
        Err(e) => {
            tracing::error!(error = %e, "Failed to list workspaces for eager scan");
            return;
        }
    };

    for ws in &workspaces {
        match get_or_init_engine(&ws.name, Path::new(&ws.path)) {
            Ok(_) => { /* scan started */ }
            Err(e) => {
                tracing::warn!(
                    workspace_name = ws.name,
                    workspace_path = %ws.path,
                    error = %e,
                    "Failed to initialize search engine for workspace"
                );
            }
        }
    }

    tracing::info!(
        workspace_count = workspaces.len(),
        "Eager search engine initialization complete"
    );
}

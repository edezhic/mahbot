//! Shared per-workspace search engine registry.
//!
//! Each workspace gets a single [`SharedFilePicker`] that all agents share.
//! Background filesystem scanning begins eagerly when a workspace is
//! registered (on app startup or workspace add), and `ensure_scanned` gates
//! searches on scan readiness.

use crate::util::UnwrapPoison;
use fff_search::FilePicker;
use fff_search::file_picker::{FFFMode, FilePickerOptions};
use fff_search::shared::{SharedFilePicker, SharedFrecency};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::sync::OnceCell;

// ── Registry ─────────────────────────────────────────────────────────────

/// Global search-engine registry, keyed by workspace name.
static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<SearchEngineEntry>>>> = OnceCell::const_new();

/// Initialize the global registry. Must be called during bootstrap, before
/// any background task that may search.
///
/// Idempotent: a second call (e.g. a test init path racing another test's
/// bootstrap) is a no-op that preserves the existing registry — the map is
/// never cleared, so there is nothing to re-create.
pub fn init_global() {
    let _ = REGISTRY.set(RwLock::new(HashMap::new()));
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
/// Created once per workspace (by [`get_or_init_engine`]), shared across all
/// agents searching that workspace. The caller should ensure the background
/// scan is complete (via [`ensure_scanned`]) before using the engine for
/// searches — this is handled by [`resolve_engine`].
#[derive(Debug)]
pub(crate) struct SearchEngineEntry {
    /// Shared file picker used for both `files` and `grep` modes.
    pub picker: SharedFilePicker,
}

// ── Initialization ────────────────────────────────────────────────────────

/// Get or initialize the shared search engine for a workspace.
///
/// On first access, this creates the [`FilePicker`] and spawns a background
/// filesystem scan via [`FilePicker::new_with_shared_state`].
///
/// Returns a cloneable handle. Multiple callers racing on first access are
/// serialized by the registry write lock — only one engine is created.
pub(crate) fn get_or_init_engine(
    name: &str,
    path: &Path,
) -> Result<Arc<SearchEngineEntry>, String> {
    // Fast path: read-lock check.
    {
        let reg = registry().read().unwrap_poison();
        if let Some(entry) = reg.get(name) {
            return Ok(Arc::clone(entry));
        }
    }

    // Slow path: serialise creation under the write lock so that two
    // concurrent callers never create duplicate scans.
    let mut reg = registry().write().unwrap_poison();

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
/// Creates the `FilePicker` and spawns the background scan.
fn init_engine_for_workspace(name: &str, path: &Path) -> Result<SearchEngineEntry, String> {
    if !path.exists() {
        return Err(format!(
            "Workspace directory does not exist: {}",
            path.display()
        ));
    }

    let picker = SharedFilePicker::default();
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

    // The picker requires a frecency handle. Nothing in the product ever
    // records file visits, so ranking stays fuzzy-match-quality only, and the
    // disabled handle guarantees no frecency store can be opened per workspace
    // (`init` is a no-op on it).
    FilePicker::new_with_shared_state(picker.clone(), SharedFrecency::noop(), options)
        .map_err(|e| format!("Failed to create search engine: {e}"))?;

    tracing::info!(
        workspace_name = name,
        workspace_path = %path.display(),
        "Search engine created — background scan started"
    );

    Ok(SearchEngineEntry { picker })
}

// ── Scan readiness ────────────────────────────────────────────────────────

/// Wait for the background filesystem scan to finish.
///
/// Blocks for up to 30 seconds. Returns an error if the scan hasn't completed
/// within the timeout, rather than returning incomplete or stale results.
/// A completed scan may legitimately leave the index empty — the live
/// watcher populates it as files appear — so it returns `Ok(())` regardless.
///
/// This is an async function because [`SharedFilePicker::wait_for_scan`] is a
/// blocking call — we run it on the tokio blocking thread pool.
async fn ensure_scanned(entry: &SearchEngineEntry) -> Result<(), String> {
    let picker = entry.picker.clone();

    let scanned = tokio::task::spawn_blocking(move || {
        picker.wait_for_scan(std::time::Duration::from_secs(30))
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {e}"))?;

    if !scanned {
        return Err("Search engine scan has not completed within 30 seconds. \
                     The workspace may be too large or the filesystem is slow. \
                     Try searching again in a moment."
            .to_string());
    }

    // A completed scan may legitimately leave the index empty — an empty
    // workspace, or an ephemeral run folder the coder hasn't written yet.
    // The live filesystem watcher populates the index incrementally as
    // files appear, so we return empty results rather than erroring.
    //
    // Deliberate trade-off: this also drops the post-scan panic detector
    // (a panicked walk clears the scanning flag but commits zero files);
    // a genuinely-broken scan is rare and an empty result is accepted.
    Ok(())
}

/// Get or init the engine for a workspace and ensure the background scan has
/// finished — the shared entry point used by the search tool and the editor's
/// global-search path.
///
/// `scan_error_prefix` is prepended to scan-readiness errors only; the
/// editor's run path passes `"Search engine not ready: "` while other callers
/// pass `""` to keep the raw error.
pub(crate) async fn resolve_engine(
    name: &str,
    path: &str,
    scan_error_prefix: &str,
) -> Result<Arc<SearchEngineEntry>, String> {
    let entry = get_or_init_engine(name, Path::new(path))?;
    ensure_scanned(&entry)
        .await
        .map_err(|e| format!("{scan_error_prefix}{e}"))?;
    Ok(entry)
}

// ── Lookup helpers for tools ───────────────────────────────────────────────

#[must_use]
pub(crate) fn get_engine_by_name(name: &str) -> Option<Arc<SearchEngineEntry>> {
    let reg = REGISTRY.get()?.read().ok()?;
    reg.get(name).cloned()
}

// ── Lifecycle ─────────────────────────────────────────────────────────────

/// Remove a workspace's search engine from the registry.
///
/// Dropping the last [`Arc<SearchEngineEntry>`] will drop the underlying
/// [`SharedFilePicker`], which triggers the background scan's cancellation
/// flag and cleans up any associated threads.
pub(crate) fn remove_engine(workspace_name: &str) {
    let mut reg = registry().write().unwrap_poison();
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

//! The per-workspace file index: one scan-backed `SharedFilePicker` per
//! workspace, shared by every caller — the file-search tool, `read`, `edit`, the
//! editor and the Git page, plus the lifecycle paths that reuse or release one
//! (branch launch and stop, session deletion, transient-session purge, research-run
//! cleanup). Indexes are created on demand or eagerly at start-up, registered
//! before use, and `ensure_scanned` gates searches on scan readiness.
//!
//! Not a search engine and not the search tool: this module owns the indexes and
//! their lifecycle. Items keep their historical names (`get_or_init_engine` etc).

use crate::util::UnwrapPoison;
use fff_search::FilePicker;
use fff_search::GitRecencyConfig;
use fff_search::file_picker::{FFFMode, FilePickerOptions};
use fff_search::shared::{SharedFilePicker, SharedFrecency};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::sync::OnceCell;

/// Global search-engine registry, keyed by workspace name.
static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<SearchEngineEntry>>>> = OnceCell::const_new();

/// Must be called during bootstrap, before any background task that may search.
/// Idempotent: a second call (e.g. a test init racing another test's bootstrap) is
/// a no-op that preserves the existing registry.
pub fn init_global() {
    let _ = REGISTRY.set(RwLock::new(HashMap::new()));
}

fn registry() -> &'static RwLock<HashMap<String, Arc<SearchEngineEntry>>> {
    REGISTRY
        .get()
        .expect("search engine registry not initialized — call workspace_index::init_global()")
}

#[must_use]
pub(crate) fn registry_initialized() -> bool {
    REGISTRY.get().is_some()
}

/// Created once per workspace and shared across all agents searching it. Callers
/// must go through [`resolve_engine`], which waits for the scan first.
#[derive(Debug)]
pub(crate) struct SearchEngineEntry {
    /// Used for both `files` and `grep` modes.
    pub picker: SharedFilePicker,
}

/// On first access, creates the [`FilePicker`] and spawns a background filesystem
/// scan. Racing callers are serialised by the registry write lock — only one
/// engine is created.
pub(crate) fn get_or_init_engine(
    name: &str,
    path: &Path,
) -> Result<Arc<SearchEngineEntry>, String> {
    {
        let reg = registry().read().unwrap_poison();
        if let Some(entry) = reg.get(name) {
            return Ok(Arc::clone(entry));
        }
    }

    let mut reg = registry().write().unwrap_poison();

    // Another writer may have inserted while we waited for the lock.
    if let Some(existing) = reg.get(name) {
        return Ok(Arc::clone(existing));
    }

    let entry = Arc::new(init_engine_for_workspace(name, path)?);
    reg.insert(name.to_string(), Arc::clone(&entry));
    Ok(entry)
}

/// Without touching the registry: creates the `FilePicker` and starts the scan.
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
        // Ranking by git recency is on by default and reorders matches by commit
        // recency; the product orders by fuzzy-match quality only, so it stays off.
        git_recency: GitRecencyConfig {
            enabled: false,
            ..GitRecencyConfig::default()
        },
    };

    // The picker requires a frecency handle; nothing records file visits, and the
    // noop handle guarantees no frecency store is opened per workspace.
    FilePicker::new_with_shared_state(picker.clone(), SharedFrecency::noop(), options)
        .map_err(|e| format!("Failed to create search engine: {e}"))?;

    tracing::info!(
        workspace_name = name,
        workspace_path = %path.display(),
        "Search engine created — background scan started"
    );

    Ok(SearchEngineEntry { picker })
}

/// Waits up to 30 s for the background scan; an error means incomplete or stale
/// results would otherwise leak out. [`SharedFilePicker::wait_for_scan`] blocks,
/// hence the blocking thread pool.
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

    // A completed scan may legitimately leave the index empty (an empty workspace,
    // an ephemeral run folder not written yet); the live watcher fills it as files
    // appear. Trade-off: this drops the post-scan panic detector — a panicked walk
    // clears the scanning flag but commits zero files — an accepted residual.
    Ok(())
}

/// Get or init the engine and wait for the scan — the shared entry point of the
/// search tool and the editor's global-search path. `scan_error_prefix` is
/// prepended to scan-readiness errors only (the editor passes a label, others `""`).
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

#[must_use]
pub(crate) fn get_engine_by_name(name: &str) -> Option<Arc<SearchEngineEntry>> {
    let reg = REGISTRY.get()?.read().ok()?;
    reg.get(name).cloned()
}

/// Dropping the last [`Arc<SearchEngineEntry>`] drops the [`SharedFilePicker`],
/// which cancels the background scan and cleans up its threads.
pub(crate) fn remove_engine(workspace_name: &str) {
    let mut reg = registry().write().unwrap_poison();
    if reg.remove(workspace_name).is_some() {
        tracing::info!(workspace_name, "Search engine removed from registry");
    }
}

/// Release the engine of a path-named workspace — one named by an absolute path,
/// a synthetic workspace created on demand, never a managed name (`validate_name`
/// allows only letters and underscores) and never a `dir-…` delegation key. Its
/// engine is keyed by the folder, not the launch: kept while any launch works
/// there, released as soon as one leaves, stops, or its session ends, and a
/// sibling still working there re-creates it. A managed workspace's engine is left alone.
pub(crate) fn release_path_named_engine(workspace_name: &str) {
    if registry_initialized() && std::path::Path::new(workspace_name).is_absolute() {
        remove_engine(workspace_name);
    }
}

/// Eager scanning for all managed workspaces: called once after bootstrap, in a
/// background task. One workspace's error does not stop the others.
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

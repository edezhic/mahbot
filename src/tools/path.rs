//! Path validation and resolution functions for tool operations.
//!
//! This module implements the path boundary for all tool file operations. The
//! boundary itself is the tool's [`PathAccess`], fixed when a role's toolset is
//! built: the guest Assistant's workspace-only frame, the pipeline roles' read
//! allowlist (temp files, dependency caches) and the admin Assistant's
//! unrestricted access.
//!
//! Two product rules hold whatever the level. The service's own live stores are
//! refused by location ([`StoreFile`]) for every read and every write. A write
//! is refused inside a registered project workspace too — but that second rule
//! binds only the unrestricted writer ([`unrestricted_write_refusal`]): a
//! confined write is inside its own workspace, where the case never arises.

use anyhow::Context;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// What a file tool may reach. The level is fixed when a role's toolset is
/// built ([`crate::Role::tools`]), so it is a property of the tool instance and
/// never of a call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PathAccess {
    /// The guest Assistant: the workspace alone.
    Workspace,
    /// The pipeline roles, and every general read: the workspace plus the read
    /// allowlist (dependency caches, SDK headers, temp roots, spill files).
    /// Writes have no extra allowed paths, so this reads as [`Self::Workspace`]
    /// for them.
    Allowlisted,
    /// The admin Assistant: any path the shell could name, minus the two places
    /// the product keeps to itself (see [`StoreFile`] and
    /// [`unrestricted_write_refusal`]).
    Unrestricted,
}

/// True when `s` contains glob metacharacters (`*`, `?`, `[`, and optionally `]`).
///
/// `include_close_bracket` preserves per-site semantics: the read tool treats a
/// lone `]` as a glob signal, the shell readonly sandbox does not.
#[must_use]
pub(crate) fn contains_glob(s: &str, include_close_bracket: bool) -> bool {
    s.contains(['*', '?', '[']) || (include_close_bracket && s.contains(']'))
}

/// Shell-quote a string for safe interpolation into a POSIX shell command.
///
/// Wraps the value in single quotes; embedded `'` is escaped as `'\''`
/// (terminate, insert escaped quote, resume). Single quotes suppress all
/// expansion, so spaces, `$`, backticks, backslashes, and glob characters
/// pass through literally.
pub(crate) fn shell_quote(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{escaped}'")
}

/// Canonicalize the parent directory of `path` and join the original file name.
///
/// This is the common canonicalization strategy used by both
/// [`resolve_directory_read_fallback`] and [`resolve_write_target`]: the parent
/// directory is canonicalized (to resolve symlinks in the directory chain) while
/// the final file component is preserved as-is (the file itself may not exist
/// yet for write operations).
///
/// Returns an error if `path` has no parent or file_name component, or if
/// [`tokio::fs::canonicalize`] fails on the parent directory. The verbatim prefix
/// is dropped (see [`crate::util::strip_verbatim_prefix`]), so the tool hands on
/// the plain spelling.
async fn canonicalize_parent_and_join(path: &Path) -> std::io::Result<PathBuf> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "no parent directory")
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "no file name"))?;
    let canon_parent = tokio::fs::canonicalize(parent).await?;
    Ok(crate::util::strip_verbatim_prefix(&canon_parent).join(name))
}

/// Fallback directory resolution when full-path [`canonicalize`] fails with
/// `NotFound` but the lexical path still exists as a directory.
///
/// Uses parent canonicalization + final component (same strategy as write-mode
/// path resolution) so existing directories are listable even when agents omit
/// a trailing `/` or when full-path canonicalization fails on edge-case paths.
async fn resolve_directory_read_fallback(full_path: &Path) -> Option<PathBuf> {
    let meta = tokio::fs::symlink_metadata(full_path).await.ok()?;
    if !meta.is_dir() {
        return None;
    }

    let resolved = canonicalize_parent_and_join(full_path).await.ok()?;
    if tokio::fs::symlink_metadata(&resolved)
        .await
        .is_ok_and(|m| m.is_dir())
    {
        return Some(resolved);
    }

    Some(full_path.to_path_buf())
}

/// Resolve and validate a file target for write/edit operations.
///
/// For every access but [`PathAccess::Unrestricted`], path validation is
/// [`is_path_safe_for_workspace`] (pre- and post-canonicalization): the write
/// stays inside the workspace, where the extra *read* paths (spill files,
/// dependency caches) are not writable either.
///
/// Additional security:
/// 1. Canonicalize the **parent** directory only — the file itself may not exist yet.
/// 2. Symlink check: if the target exists and is a symlink, refuse (unlike reading,
///    where `canonicalize` resolves through symlinks safely).
/// 3. If `ensure_parent` is `true`, creates parent directories before canonicalizing.
///
/// [`PathAccess::Unrestricted`] (the admin Assistant) instead reaches anything
/// the shell could, minus two places, and resolves the file the write would
/// really land in — see [`resolve_unrestricted_write`].
///
/// See [`resolve_read_target`] for the read-side counterpart.
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub(crate) async fn resolve_write_target(
    workspace_root: &Path,
    path: &str,
    ensure_parent: bool,
    access: PathAccess,
) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root, access)?;

    if access == PathAccess::Unrestricted {
        return resolve_unrestricted_write(workspace_root, &full_path, path, ensure_parent).await;
    }

    // The live stores are an invariant of the product, not a rule of a role:
    // refused by location for every holder, before anything is created. It can
    // only bite here if a workspace covers the store directory.
    if let Some(StoreFile::Live) = product_store_file(&full_path) {
        return Err(live_store_refusal("write to", path));
    }

    // Pre-canonicalization check — the workspace frame, no extra allowed paths
    if !is_path_safe_for_workspace(path, workspace_root) {
        anyhow::bail!(
            "forbidden: cannot write to {path}: outside the workspace — hint: writes are only \
             allowed inside the workspace root ({}); use a path under it",
            workspace_root.display()
        );
    }

    let Some(parent) = full_path.parent() else {
        anyhow::bail!("Invalid path: missing parent directory");
    };

    if ensure_parent {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent directories")?;
    }

    // Canonicalize parent only — the file itself may not exist yet
    let resolved_target = canonicalize_parent_and_join(&full_path)
        .await
        .context("Failed to resolve file path")?;

    // The post-canonicalization half of the store rule, as on the read side: a
    // parent that is a symlink into the store directory is not visible in the
    // spelling, and the resolved file is what would be opened.
    if let Some(StoreFile::Live) = product_store_file(&resolved_target) {
        return Err(live_store_refusal("write to", path));
    }

    // Re-extract canonicalized parent for the post-canonicalization security check.
    let Some(resolved_parent) = resolved_target.parent() else {
        anyhow::bail!("Invalid canonicalized path: missing parent directory");
    };

    if !is_path_safe_for_workspace(&resolved_parent.to_string_lossy(), workspace_root) {
        anyhow::bail!(
            "forbidden: cannot write to {}: resolves outside the workspace — hint: writes are \
             only allowed inside the workspace root ({}); check for symlinks pointing outside it",
            resolved_parent.display(),
            workspace_root.display()
        );
    }

    // Explicit symlink refusal (read resolves symlinks via canonicalize instead)
    if let Ok(meta) = tokio::fs::symlink_metadata(&resolved_target).await
        && meta.file_type().is_symlink()
    {
        anyhow::bail!(
            "forbidden: cannot write to {}: it is a symlink — hint: write to the symlink's \
             resolved target directly, provided it stays inside the workspace",
            resolved_target.display()
        );
    }

    refuse_non_regular_target(&resolved_target, path).await?;

    Ok(resolved_target)
}

/// Refuse a write whose target already exists as anything but a regular file:
/// a device, a socket or a FIFO has no bounded read — and the edit tool reads
/// the file it edits — and a directory is not what any caller means to write
/// over.
async fn refuse_non_regular_target(resolved_target: &Path, requested: &str) -> anyhow::Result<()> {
    if let Ok(meta) = tokio::fs::symlink_metadata(resolved_target).await
        && !meta.is_file()
    {
        anyhow::bail!(
            "forbidden: cannot write to {requested}: {} already exists and is not a regular file \
             — hint: only a regular file can be written; create a file inside a directory instead \
             of over it",
            resolved_target.display()
        );
    }
    Ok(())
}

/// Resolve a write for the admin's Assistant: any path the shell could name,
/// minus two places — the product's own store files, and the registered project
/// workspaces, whose work belongs to their own Manager.
///
/// The decision is made on the file the write would really land in: the deepest
/// existing ancestor is canonicalized, so every symlink and `..` on the way —
/// the final component included — is settled before the checks read the path.
/// Nothing is created on disk until every check has passed, so a refused write
/// leaves no directory behind.
async fn resolve_unrestricted_write(
    own_workspace: &Path,
    full_path: &Path,
    requested: &str,
    ensure_parent: bool,
) -> anyhow::Result<PathBuf> {
    let resolved = resolve_through_existing_ancestor(full_path)
        .await
        .with_context(|| format!("Failed to resolve file path: {}", full_path.display()))?;
    // A dangling final component is a link the walk above cannot settle (its
    // target does not exist yet): it is followed to the file it names, which is
    // the file the write really lands in — the shell would create that same
    // file through the link.
    let resolved = follow_final_links(resolved).await?;

    // The list is read before anything is judged, so an unreadable registry is
    // a refusal rather than a silently empty one.
    let projects = registered_projects(requested).await?;
    // Only the resolved file is judged: the file the write really lands in is
    // what decides, never the spelling it came in as.
    if let Some(err) = unrestricted_write_refusal(own_workspace, &resolved, requested, &projects) {
        return Err(err);
    }

    refuse_non_regular_target(&resolved, requested).await?;

    if ensure_parent && let Some(parent) = resolved.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create parent directories")?;
    }

    Ok(resolved)
}

/// The refusal a write of the admin's Assistant meets, or `None` when the write
/// is allowed. Both refusals are by location and neither by content: the
/// product's own store files (never opened to find out what they are), and the
/// registered project workspaces, whose work belongs to their own manager.
/// `own_workspace` — the admin's own working folder — outranks the second rule,
/// so a personal folder that happens to sit inside a registered project area
/// stays writable.
fn unrestricted_write_refusal(
    own_workspace: &Path,
    candidate: &Path,
    requested: &str,
    projects: &[(String, PathBuf)],
) -> Option<anyhow::Error> {
    match product_store_file(candidate) {
        Some(StoreFile::Live) => return Some(live_store_refusal("write to", requested)),
        Some(StoreFile::Copy) => {
            return Some(anyhow::anyhow!(
                "forbidden: cannot write to {requested}: it is a copy the service keeps of its own \
                 databases — hint: only the service's own copies are closed; every other database \
                 (its format notwithstanding) is an ordinary file"
            ));
        }
        None => {}
    }
    if is_within_ignoring_case(candidate, &crate::util::canonical_or_self(own_workspace)) {
        return None;
    }
    projects
        .iter()
        .find(|(_, path)| is_within_ignoring_case(candidate, path))
        .map(|(name, _)| {
            anyhow::anyhow!(
                "forbidden: cannot write to {requested}: it is inside the registered project \
                 workspace '{name}' — hint: work in a project goes through that project's manager \
                 (`send_message_to_manager`); the admin's own workspace and any unregistered \
                 directory stay writable"
            )
        })
}

/// The registered project workspaces, by name and path.
///
/// An unreadable list is an error, never an empty one: a write that cannot see
/// which areas belong to projects is refused rather than guessed at. The
/// consequence is accepted — while the product's stores are unavailable, the
/// admin cannot write into its own memory either.
async fn registered_projects(requested: &str) -> anyhow::Result<Vec<(String, PathBuf)>> {
    registered_projects_from(crate::users::registered_workspaces().await, requested)
}

/// [`registered_projects`] over an already-read registry, so the refusal an
/// unreadable one meets is decided without the store.
fn registered_projects_from(
    read: anyhow::Result<Vec<crate::Workspace>>,
    requested: &str,
) -> anyhow::Result<Vec<(String, PathBuf)>> {
    read.map(|projects| {
        projects
            .into_iter()
            .map(|ws| (ws.name, PathBuf::from(ws.path)))
            .collect()
    })
    .map_err(|e| {
        anyhow::anyhow!(
            "forbidden: cannot write to {requested}: the registered project workspaces cannot be \
             read ({e}) — hint: the refusal is the safe answer for a write; retry once the \
             product's own stores are available"
        )
    })
}

/// Canonicalize the deepest existing ancestor of `path` and re-append the
/// components below it — the resolution a target that does not exist yet needs,
/// so every symlink and `..` on the way is settled before any check reads the
/// path. Nothing is created on disk.
///
/// The walk keeps the caller's own spelling: a `..` behind a symlinked component
/// is settled by the filesystem, as the shell settles it. Only the components
/// below the deepest existing ancestor are resolved lexically (they do not
/// exist, so no symlink can hide in them) — where a component on the way is a
/// *dangling* symlink, no ancestor including it canonicalizes and the tail stays
/// lexical, so the path lands beside the link where the shell's own open would
/// fail. Both the checks and the write use the same resolved path, so no
/// boundary is crossed.
async fn resolve_through_existing_ancestor(path: &Path) -> std::io::Result<PathBuf> {
    let components: Vec<std::path::Component<'_>> = path.components().collect();
    let mut last_error = None;
    for keep in (1..=components.len()).rev() {
        let mut ancestor = PathBuf::new();
        for component in &components[..keep] {
            ancestor.push(component.as_os_str());
        }
        match tokio::fs::canonicalize(&ancestor).await {
            Ok(resolved) => {
                let mut out = crate::util::strip_verbatim_prefix(&resolved);
                for component in &components[keep..] {
                    out.push(component.as_os_str());
                }
                return Ok(normalize_path(&out));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => last_error = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no existing ancestor for {}", path.display()),
        )
    }))
}

/// How many links a resolved path may end in before the chain is refused, as
/// the shell refuses a link loop.
const MAX_FINAL_LINKS: usize = 32;

/// Settle a chain of symlinks at the END of `resolved` — the part
/// [`resolve_through_existing_ancestor`] cannot canonicalize, because the file
/// it names does not exist yet. Each link is replaced by the file it points at
/// and resolved in turn, so the returned path is a name that is not a link (or
/// does not exist at all). Nothing is created.
async fn follow_final_links(resolved: PathBuf) -> std::io::Result<PathBuf> {
    let mut resolved = resolved;
    for _ in 0..MAX_FINAL_LINKS {
        match tokio::fs::symlink_metadata(&resolved).await {
            Ok(meta) if meta.file_type().is_symlink() => {
                let target = tokio::fs::read_link(&resolved).await?;
                let next = match resolved.parent() {
                    Some(parent) if !target.is_absolute() => parent.join(&target),
                    _ => target,
                };
                resolved = resolve_through_existing_ancestor(&next).await?;
            }
            _ => return Ok(resolved),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "too many levels of symbolic links resolving {} — hint: break the link loop and \
             retry",
            resolved.display()
        ),
    ))
}

/// Resolve and validate a file path for read operations.
///
/// `access` decides the boundary (see [`check_path_read_allowed`]), checked
/// pre- and post-canonicalization. Whatever the access, a path that is neither
/// a regular file nor a directory is refused: the read tool holds no bounded
/// read for a device, a socket or a channel.
///
/// Key differences from [`resolve_write_target`]:
/// - Canonicalizes the **full path**, not just the parent (file must exist).
/// - No `ensure_parent` parameter — parent creation is a write-only concept.
/// - No explicit symlink refusal — `tokio::fs::canonicalize` resolves symlinks,
///   so the post-canonicalization check catches escapes via the resolved path.
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub(crate) async fn resolve_read_target(
    workspace_root: &Path,
    path: &str,
    access: PathAccess,
) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root, access)?;

    // Pre-canonicalization check — allows EXTRA_READ_ALLOWED paths
    // (temp files, dependency source directories) outside the workspace
    check_path_read_allowed(path, workspace_root, access)?;

    // Canonicalize full path (file must exist). Resolves symlinks,
    // so the post-canonicalization check catches escapes.
    let resolved_path = match tokio::fs::canonicalize(&full_path).await {
        Ok(resolved) => crate::util::strip_verbatim_prefix(&resolved),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            resolve_directory_read_fallback(&full_path)
                .await
                .ok_or_else(|| anyhow::anyhow!("File not found: {}", full_path.display()))?
        }
        Err(e) => {
            return Err(match e.kind() {
                std::io::ErrorKind::PermissionDenied => {
                    anyhow::anyhow!("Permission denied: {}", full_path.display())
                }
                _ => anyhow::anyhow!("Failed to resolve file path: {}: {e}", full_path.display()),
            });
        }
    };

    check_path_read_allowed(&resolved_path.to_string_lossy(), workspace_root, access)?;

    // The read tool opens regular files and lists directories. Anything else —
    // a device, a socket, a channel — has no bounded read and is refused for
    // every role: the refusal is about what the tool can hold open, not about
    // policy (which is what `access` decides above).
    let meta = tokio::fs::metadata(&resolved_path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", resolved_path.display()))?;
    if !meta.is_file() && !meta.is_dir() {
        anyhow::bail!(
            "forbidden: cannot read {path}: {} is neither a regular file nor a directory — hint: \
             the read tool reads regular files and lists directories only",
            resolved_path.display()
        );
    }

    Ok(resolved_path)
}

/// Lexically normalize a path by resolving `.` and `..` components
/// without filesystem access (no I/O).
///
/// This is a pure lexical transformation — it does not resolve symlinks,
/// verify existence, or canonicalize. It is safe to call on paths that
/// do not yet exist (e.g. target paths for write operations).
///
/// Algorithm:
/// - `RootDir` components establish the root anchor.
/// - `CurDir` (`.`) components are dropped.
/// - `ParentDir` (`..`) components pop the last `Normal` component if
///   one exists; if no `Normal` component remains and the path is
///   absolute (has a root), the `..` is silently dropped (can't go above
///   root); if relative, excess `..` are preserved.
/// - `Normal` components are pushed sequentially.
///
/// # Limitations
///
/// - Does not resolve symlinks. If a symlink within an allowed root
///   points outside the root, lexical normalization cannot detect the
///   escape — only filesystem-level canonicalization can.
/// - A Windows `Prefix` component (`C:`, `\\?\C:`) is carried through as an
///   absolute anchor — `..` cannot escape past it — and is otherwise left
///   untouched, verbatim spelling included.
#[must_use]
pub(crate) fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut normalized: Vec<Component<'_>> = Vec::new();
    let mut is_absolute = false;

    for component in path.components() {
        match component {
            Component::RootDir => {
                is_absolute = true;
                normalized.push(component);
            }
            Component::Prefix(prefix) => {
                is_absolute = true;
                normalized.push(Component::Prefix(prefix));
            }
            Component::CurDir => {
                // Skip `.` components
            }
            Component::Normal(_) => {
                normalized.push(component);
            }
            Component::ParentDir => {
                // Try to pop the last Normal component.
                if let Some(last) = normalized.last() {
                    if matches!(last, Component::Normal(_)) {
                        normalized.pop();
                    } else {
                        // Last component is RootDir, Prefix, or another ParentDir.
                        // For absolute paths: can't go above root → drop `..`.
                        // For relative paths: keep excess `..` as meaningful prefix.
                        if !is_absolute {
                            normalized.push(component);
                        }
                    }
                } else {
                    // Empty normalized — relative path starting with `..`.
                    if !is_absolute {
                        normalized.push(component);
                    }
                }
            }
        }
    }

    let mut result = PathBuf::new();
    for component in &normalized {
        result.push(component.as_os_str());
    }
    result
}

/// Check whether a path is under any of the given roots, after tilde expansion
/// and lexical normalization.
///
/// The path may contain a leading `~` (user-provided input before
/// canonicalization). In that case the `~` is expanded to the user's
/// home directory before comparing against the (already-expanded) roots.
///
/// The path is normalized (`.`, `..` resolved) *after* tilde expansion so that
/// `../` segments introduced by tilde expansion or present in the original path
/// cannot escape the allowed roots via lexical traversal. The comparison itself is
/// [`crate::util::is_within`], which settles the spelling on both sides.
pub(crate) fn is_path_under_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let expanded = crate::util::expand_tilde(&path.to_string_lossy());
    let normalized = normalize_path(&expanded);
    roots
        .iter()
        .any(|root| crate::util::is_within(&normalized, root))
}

/// The canonical allowed temp/scratch roots (shared by the read-path
/// allowlists and the read-only shell guard).
#[must_use]
pub(crate) fn allowed_temp_roots() -> Vec<PathBuf> {
    ALLOWED_TEMP_ROOTS.clone()
}

/// Check whether `path` is under an [`EXTRA_READ_ALLOWED`] directory.
fn is_path_in_extra_allowed(path: &Path) -> bool {
    is_path_under_roots(path, &EXTRA_READ_ALLOWED)
}

/// Whether two paths refer to the same file or directory.
///
/// First compares by direct structural equality; if that fails, falls back
/// to canonicalizing both paths and comparing the canonical forms.
/// Returns `false` if either path cannot be canonicalized.
fn paths_same_or_canonical(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// Whether `path` is an OS temp directory root (not merely nested under temp).
///
/// Deliberately permissive: it accepts every spelling any platform uses for the
/// temp location, while the daemon's own temp environment (`crate::temp`) holds
/// only the names this platform actually pins and models.
fn is_os_temp_root(path: &Path) -> bool {
    let check_path = crate::util::expand_tilde(&path.to_string_lossy());

    if paths_same_or_canonical(&check_path, &std::env::temp_dir()) {
        return true;
    }

    for var in ["TMPDIR", "TEMP", "TMP"] {
        if let Ok(val) = std::env::var(var) {
            let env_path = PathBuf::from(val);
            if paths_same_or_canonical(&check_path, &env_path) {
                return true;
            }
        }
    }

    #[cfg(unix)]
    {
        for prefix in ["/tmp", "/private/tmp", "/var/tmp"] {
            if paths_same_or_canonical(&check_path, Path::new(prefix)) {
                return true;
            }
        }
        // macOS per-user temp root: /var/folders/XX/YY/T
        let lossy = check_path.to_string_lossy();
        let parts: Vec<&str> = lossy.trim_start_matches('/').split('/').collect();
        if parts.len() == 5 && parts[0] == "var" && parts[1] == "folders" && parts[4] == "T" {
            return true;
        }
    }

    false
}

/// Format a spill filename with a random 4-digit hex identifier.
pub(crate) fn format_spill_filename() -> String {
    format!("spill_{:04x}.txt", rand::random::<u16>())
}

/// Whether `name` is a mahbot shell spill/full log name: `spill_XXXX.txt`
/// (4-digit hex) or a `.full.log` suffix.
fn is_mahbot_spill_filename(name: &str) -> bool {
    if name.ends_with(".full.log") {
        return true;
    }
    name.strip_prefix("spill_")
        .and_then(|s| s.strip_suffix(".txt"))
        .is_some_and(|hex| hex.len() == 4 && hex.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Whether `path` has the spill filename and `.agent` parent layout (ignoring temp root).
fn is_mahbot_spill_shaped(path: &Path) -> bool {
    if !path.is_absolute() {
        return false;
    }
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if !is_mahbot_spill_filename(file_name) {
        return false;
    }
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        == Some(".agent")
}

/// Whether the grandparent directory of `path` is an OS temp/scratch root.
///
/// This is the "on OS temp" half of the spill-file check: for a spill path
/// like `/tmp/.agent/spill_ab12.txt`, the grandparent is `/tmp/`, which is
/// an OS temp root.  The shape check (`.agent` parent + spill filename) is
/// separate — see [`is_mahbot_spill_shaped`].
fn is_grandparent_temp_root(path: &Path) -> bool {
    path.parent()
        .and_then(|p| p.parent())
        .is_some_and(is_os_temp_root)
}

/// The product's own storage root — where its stores, and the copies it keeps
/// of them, live. Read from the running configuration (set at startup; a test
/// run resolves its own root), so no path is hard-coded here.
fn storage_root() -> Option<PathBuf> {
    crate::config::CONFIG
        .try_storage_root()
        .or_else(|| crate::config::default_config_dir().ok())
}

/// Sidecar suffixes of a store file — the write-ahead log and its journal
/// siblings. They belong to the store and are refused with it.
const STORE_SIDECAR_SUFFIXES: &[&str] = &["-wal", "-shm", "-journal"];

/// A store file of the product, identified by its location alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StoreFile {
    /// One of the two live stores. The service holds whole-file locks on them,
    /// so opening one from inside the service — briefly, to read a header, to
    /// find out whether it is a database — silently releases those locks for
    /// the rest of the process's life. Never opened, by any tool, for any
    /// reason.
    Live,
    /// A copy the service keeps for itself: the quarantine and pre-reindex
    /// families, and the rebuild's own temp file. Static, so it is readable as
    /// an ordinary file; only the unrestricted writer's refusal covers it.
    Copy,
}

/// Whether `path` is inside `base`, comparing every component without ASCII
/// case.
///
/// The store and project rules are location rules, and the volumes they run on
/// (macOS, Windows) compare file names without case, while a canonicalized path
/// keeps the spelling the caller typed — so a differently-cased spelling of a
/// protected place is that place, and the rule must answer it the same way. On a
/// case-sensitive volume the two spellings are two directories; the comparison
/// still answers as if they were one, which over-refuses where it backs a
/// refusal (accepted — a rule that limits rather than fences the shell) and
/// over-permits where it backs the admin's own-folder exemption (a case-varied
/// sibling of that folder inside a registered project stays writable). The
/// supported volumes cannot tell the two apart.
#[must_use]
fn is_within_ignoring_case(candidate: &Path, base: &Path) -> bool {
    let candidate = crate::util::strip_verbatim_prefix(candidate);
    let base = crate::util::strip_verbatim_prefix(base);
    let mut candidate = candidate.components();
    let mut base = base.components();
    loop {
        match (candidate.next(), base.next()) {
            (_, None) => return true,
            (Some(candidate), Some(base)) => {
                if !candidate.as_os_str().eq_ignore_ascii_case(base.as_os_str()) {
                    return false;
                }
            }
            (None, Some(_)) => return false,
        }
    }
}

/// Classify `path` as one of the product's own store files under the store
/// directory — **by location and name only**, and without case. Deciding what a
/// file is by opening it is exactly what the live-store rule forbids, so the
/// store file itself is never opened here (only its directory's spelling is
/// settled).
fn store_file_at(root: &Path, path: &Path) -> Option<StoreFile> {
    // The store directory is matched in both spellings: a caller's resolved
    // path carries the canonical one (`/private/tmp` where the storage root is
    // spelled `/tmp`), and the rule must not depend on which of the two a path
    // happened to come in as.
    let db_dir = crate::db::store_dir(root);
    let candidate = normalize_path(&crate::util::expand_tilde(&path.to_string_lossy()));
    let under = is_within_ignoring_case(&candidate, &db_dir)
        || std::fs::canonicalize(&db_dir)
            .is_ok_and(|canonical| is_within_ignoring_case(&candidate, &canonical));
    if !under {
        return None;
    }
    let name = path.file_name()?.to_str()?;
    let base = STORE_SIDECAR_SUFFIXES
        .iter()
        .find_map(|suffix| crate::util::strip_suffix_ignoring_case(name, suffix))
        .unwrap_or(name);
    let lowered = base.to_ascii_lowercase();
    // `{store}.db` is a live store — only the two the service runs on, so a
    // file left over from a retired store name is an ordinary file.
    if let Some(stem) = lowered.strip_suffix(".db")
        && (stem == crate::db::CONSOLIDATED_DB_NAME || stem == crate::db::LOG_DB_NAME)
    {
        return Some(StoreFile::Live);
    }
    // `{store}.db.{copy}` is a copy the service keeps of one — of ANY store,
    // since it quarantines a retired name exactly as it quarantines a live one.
    // The naming contract is `db::debug`'s (and the rebuild's own, `db`'s), so
    // a copy is recognised through them rather than through a second spelling
    // of either; both are matched without case, like the rest of this rule.
    if let Some((_, rest)) = lowered.split_once(".db")
        && rest.starts_with(crate::db::REBUILD_TEMP_MARKER)
    {
        return Some(StoreFile::Copy);
    }
    crate::db::debug::parse_family_name(base).map(|_| StoreFile::Copy)
}

/// [`store_file_at`] against the running storage root. `None` when the root
/// cannot be resolved at all (no `HOME` and no user directories) — a case a
/// running service cannot be in, since it resolves the root before any tool
/// runs.
fn product_store_file(path: &Path) -> Option<StoreFile> {
    store_file_at(&storage_root()?, path)
}

/// The live-store refusal, worded once so the read and write sides cannot drift.
fn live_store_refusal(action: &str, path: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "forbidden: cannot {action} {path}: it is one of the product's own live databases, or a \
         journal tail of one — hint: the service holds locks on them, and opening one from inside \
         the running service silently releases those locks for the rest of the process; the \
         product's own data is read through its debug surface (`mahbot debug`, or the admin's \
         `mahbot_debug` tool) instead"
    )
}

/// Whether `path` names an existing file or directory once resolved the way a
/// read resolves it — the test that tells a literal path whose own name carries
/// glob metacharacters from a pattern.
pub(crate) async fn literal_path_exists(
    workspace_root: &Path,
    path: &str,
    access: PathAccess,
) -> bool {
    let Ok(full_path) = resolve_tool_path_with_base(path, workspace_root, access) else {
        return false;
    };
    tokio::fs::try_exists(full_path).await.unwrap_or(false)
}

/// Whether a read of `path` is a wildcard listing rather than a read of the file
/// it names: the path carries glob metacharacters (`* ? [ ]`) AND names nothing.
///
/// A file whose own name holds one (`report[1].txt`) is an ordinary path the
/// shell could name — the edit tool takes it — so the read does too, and only a
/// pattern that names no file falls through to the workspace listing.
pub(crate) async fn is_wildcard_read(
    workspace_root: &Path,
    path: &str,
    access: PathAccess,
) -> bool {
    contains_glob(path, true) && !literal_path_exists(workspace_root, path, access).await
}

/// Check that a path is allowed by the read policy of `access`.
///
/// Every level is refused the product's own live stores. Beyond that,
/// [`PathAccess::Workspace`] allows the workspace alone,
/// [`PathAccess::Unrestricted`] allows anything, and
/// [`PathAccess::Allowlisted`] allows the workspace plus [`EXTRA_READ_ALLOWED`]
/// (temp files, dependency caches, SDK headers) and spill files on a temp root.
fn check_path_read_allowed(
    path: &str,
    workspace_root: &Path,
    access: PathAccess,
) -> anyhow::Result<()> {
    // The live stores are an invariant of the product, not a rule of a role:
    // refused by location, before anything is opened, for every holder and
    // every access level. Copies are ordinary files and pass.
    if let Some(StoreFile::Live) = product_store_file(Path::new(path)) {
        return Err(live_store_refusal("read", path));
    }

    if access == PathAccess::Unrestricted {
        return Ok(());
    }

    if access == PathAccess::Workspace {
        if is_path_safe_for_workspace(path, workspace_root) {
            return Ok(());
        }
        anyhow::bail!(
            "forbidden: cannot read {path}: outside the workspace — hint: in this mode only \
             paths inside the workspace root ({}) are readable",
            workspace_root.display()
        );
    }

    let path_buf = Path::new(path);

    // Early-return for spill files on an OS temp root (allowed unconditionally).
    // If a path looks like a spill file but is NOT on a temp root, reject it.
    if is_mahbot_spill_shaped(path_buf) {
        if !is_grandparent_temp_root(path_buf) {
            anyhow::bail!(
                "forbidden: cannot read {path}: spill-shaped paths are only readable from an OS \
                 temp root — hint: use the exact background-session output path the shell tool \
                 returned"
            );
        }
        return Ok(());
    }

    if !is_path_safe_for_workspace(path, workspace_root) && !is_path_in_extra_allowed(path_buf) {
        anyhow::bail!(
            "forbidden: cannot read {path}: outside the allowed read envelope — hint: reads are \
             allowed inside the workspace root ({}) and whitelisted system directories (OS temp \
             roots, dependency caches, toolchain/SDK sources); use a path under one of those",
            workspace_root.display()
        );
    }
    Ok(())
}

/// Helper for [`EXTRA_READ_ALLOWED`] initialization: canonicalizes `raw` and
/// pushes both the canonical and raw paths (if they differ) into `dirs`,
/// ensuring no duplicates. On macOS `/tmp` → `/private/tmp` symlink, this
/// ensures both `/tmp` and `/private/tmp` are in the allowed set so that
/// both the raw and resolved forms match during [`resolve_read_target`]'s
/// pre- and post-canonicalization checks.
fn add_path_with_canonical(dirs: &mut Vec<PathBuf>, raw: PathBuf) {
    if dirs.contains(&raw) {
        return;
    }
    match std::fs::canonicalize(&raw) {
        Ok(canonical) => {
            if !dirs.contains(&canonical) {
                dirs.push(canonical.clone());
            }
            if canonical != raw {
                dirs.push(raw);
            }
        }
        Err(_) => {
            dirs.push(raw);
        }
    }
}

/// Map of XDG subdirectory (under `~`) to the corresponding environment variable.
/// Used to generate alternative paths when e.g. `$XDG_CACHE_HOME` is set to a
/// non-default location.
const XDG_SUBDIR_TO_ENV: &[(&str, &str)] = &[
    (".cache/", "XDG_CACHE_HOME"),
    (".config/", "XDG_CONFIG_HOME"),
    (".local/share/", "XDG_DATA_HOME"),
    (".local/state/", "XDG_STATE_HOME"),
];

/// For a `~`-prefixed path that starts with an XDG subdirectory
/// (e.g. `~/.cache/pypoetry/`), generate the alternative path using the
/// corresponding XDG environment variable if it's set and different from
/// the default.
///
/// Returns `None` if the path doesn't start with a known XDG subdirectory,
/// or if the corresponding env var is unset.
fn xdg_variant_path(tilde_path: &str) -> Option<String> {
    xdg_variant_path_with(|var| std::env::var(var).ok(), tilde_path)
}

/// Pure, getter-driven variant of [`xdg_variant_path`]: reads environment
/// values through `get` so tests can drive it without mutating the process
/// environment. Production passes `|var| std::env::var(var).ok()`.
#[must_use]
fn xdg_variant_path_with(get: impl Fn(&str) -> Option<String>, tilde_path: &str) -> Option<String> {
    for (xdg_subdir, env_var) in XDG_SUBDIR_TO_ENV {
        if let Some(suffix) = tilde_path
            .strip_prefix("~/")
            .and_then(|p| p.strip_prefix(xdg_subdir))
            && let Some(xdg_dir) = get(env_var)
        {
            let xdg_dir = xdg_dir.trim_end_matches('/');
            return Some(format!("{xdg_dir}/{suffix}"));
        }
    }
    None
}

/// Roots whose location is spelled by an environment variable, mapped to the
/// subpaths allowed beneath them. Two kinds share the shape: toolchain homes,
/// whose default is a location under HOME that can be relocated entirely
/// (`CARGO_HOME`, `RUSTUP_HOME`, …), and the Windows system roots, whose
/// defaults hard-code the system drive — those variables exist only on Windows,
/// so elsewhere they are unset and contribute nothing, while the rows
/// themselves stay compiled and covered on every platform. A subpath is a
/// `/`-separated component list joined with the host's own separator; an empty
/// subpath means the whole variable prefix is allowed.
const ENV_DERIVED_ROOTS: &[(&str, &[&str])] = &[
    ("CARGO_HOME", &["registry/src/", "git/checkouts/"]),
    ("RUSTUP_HOME", &["toolchains/"]),
    ("GOMODCACHE", &[""]),
    ("GOPATH", &["pkg/mod/"]),
    ("GRADLE_USER_HOME", &["caches/"]),
    ("JAVA_HOME", &["include/"]),
    ("GOROOT", &["src/"]),
    ("ProgramData", &["chocolatey/lib"]),
    (
        "SystemDrive",
        &[
            "msys64/mingw64/include",
            "msys64/ucrt64/include",
            "msys64/clang64/include",
            "msys64/usr/include",
        ],
    ),
    ("ProgramFiles(x86)", &["Windows Kits"]),
    ("ProgramFiles", &["Microsoft Visual Studio"]),
];

/// Pure, I/O-free construction of env-derived read-allowed paths.
/// For each `(var, subpaths)` in [`ENV_DERIVED_ROOTS`], when `get(var)`
/// yields a non-empty value, emits `<value><sep><subpath>` for each subpath (an
/// empty subpath yields the bare, trailing-separator-trimmed value). The
/// separator is spelled explicitly because several of these values are a bare
/// drive prefix (`C:`), and `C:` is drive-RELATIVE — a plain join onto it is
/// not a join at all. An unset or empty variable contributes nothing: fail
/// closed rather than allowlist a fabricated root. The env getter is injected
/// so tests can drive it without mutating the process environment; production
/// passes `|var| std::env::var(var).ok()`.
#[must_use]
fn env_derived_allowed_paths(get: impl Fn(&str) -> Option<String>) -> Vec<String> {
    let sep = std::path::MAIN_SEPARATOR;
    let mut paths = Vec::new();
    for &(var, subpaths) in ENV_DERIVED_ROOTS {
        let Some(value) = get(var) else {
            continue;
        };
        let value = value.trim_end_matches(['\\', '/']);
        if value.is_empty() {
            continue;
        }
        for &subpath in subpaths {
            let relative = subpath.replace('/', std::path::MAIN_SEPARATOR_STR);
            if relative.is_empty() {
                paths.push(value.to_string());
            } else {
                paths.push(format!("{value}{sep}{relative}"));
            }
        }
    }
    paths
}

/// Enumerate `<root>/<child>/<suffix>` JVM header roots at init. The system
/// JVM locations are scoped to `include/` headers only (never the whole JDK
/// tree), so the individual JDK directories must be enumerated one level.
/// Nothing is added when the root doesn't exist (fail closed).
fn jvm_include_roots(root: &str, suffix: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .filter_map(std::io::Result::ok)
        .map(|entry| entry.path().join(suffix))
        .collect()
}

/// Extra read-allowed roots: dependency caches, toolchain and SDK sources,
/// package-manager stores.
/// Paths starting with `~` are expanded at init time. Paths that don't
/// exist on the current platform are harmless (they fail canonicalization
/// and just get added as-is, never matching any read request).
const EXTRA_ALLOWED_RAW_PATHS: &[&str] = &[
    // ── Rust (Cargo) ────────────────────────────────────────
    "~/.cargo/registry/src/",
    "~/.cargo/git/checkouts/",
    "~/.rustup/toolchains/",
    // ── Python ──────────────────────────────────────────────
    "~/.local/lib/",
    "~/Library/Python/",
    "~/AppData/Roaming/Python/",
    "~/AppData/Local/Programs/Python/",
    "/usr/local/lib/",
    "/usr/lib/",
    "/Library/Frameworks/Python.framework/Versions/",
    "/opt/homebrew/lib/",
    "~/anaconda3/",
    "~/miniconda3/",
    "/opt/anaconda3/",
    "/opt/miniconda3/",
    "~/AppData/Local/conda/",
    "~/.cache/pypoetry/",
    "~/Library/Caches/pypoetry/",
    "~/AppData/Local/pypoetry/",
    "~/.local/share/virtualenvs/",
    "~/.cache/pipenv/",
    "~/Library/Caches/pipenv/",
    "~/AppData/Local/pipenv/",
    "~/.cache/uv/",
    "~/.local/share/uv/",
    "~/AppData/Local/uv/",
    "~/.rye/",
    // ── JavaScript / TypeScript ─────────────────────────────
    "~/.bun/install/cache/",
    "~/.local/share/pnpm/",
    "~/Library/pnpm/",
    "~/AppData/Local/pnpm/",
    "~/AppData/Roaming/npm/",
    "~/.npm/",
    "~/.nvm/",
    "~/.volta/",
    "~/.cache/yarn/",
    "~/Library/Caches/Yarn/",
    "~/AppData/Local/Yarn/",
    "~/.pnpm-store/",
    // ── Go ──────────────────────────────────────────────────
    "~/go/pkg/mod/",
    "/usr/local/go/",
    "/opt/homebrew/opt/go/",
    "/usr/local/opt/go/",
    // ── JVM ─────────────────────────────────────────────────
    // Note: JDK `src.zip` is intentionally NOT readable — archives are not
    // extracted by the read tool, so don't advertise it in docs.
    "~/.m2/repository/",
    "~/.gradle/caches/",
    // System JVM locations (`/Library/Java/JavaVirtualMachines/`,
    // `/usr/lib/jvm/`) are NOT whole-tree allowlisted — they are scoped to
    // `include/` headers only via init-time enumeration in EXTRA_READ_ALLOWED
    // (see `jvm_include_roots`).
    // ── Ruby ────────────────────────────────────────────────
    "~/.gem/",
    "~/.local/share/gem/",
    "~/.bundle/",
    // ── PHP (Composer) ──────────────────────────────────────
    "~/.composer/",
    "~/.cache/composer/",
    "~/Library/Caches/composer/",
    "~/AppData/Local/composer/",
    // ── C/C++ ───────────────────────────────────────────────
    "~/.conan/",
    "~/.conan2/",
    "/usr/local/Cellar/",
    "/opt/homebrew/Cellar/",
    "/usr/local/Homebrew/Library/Taps/",
    "/usr/include/",
    "/usr/local/include/",
    "/opt/homebrew/include/",
    "/opt/homebrew/opt/",
    "/opt/homebrew/Frameworks/",
    "/usr/local/opt/",
    "/usr/local/Frameworks/",
    // The Windows installs of these (chocolatey, msys2, the Windows SDK, MSVC)
    // normally live under the system drive, so they are derived from
    // [`ENV_DERIVED_ROOTS`] rather than hard-coded here.
    // ── SDK roots ────────────────────────────────────────────
    "/Applications/Xcode.app/Contents/Developer/",
    "/Library/Developer/CommandLineTools/",
    // ── Swift ───────────────────────────────────────────────
    "~/.swiftpm/",
    "~/Library/Developer/Xcode/DerivedData/",
    // ── Dart / Flutter ──────────────────────────────────────
    "~/.pub-cache/",
    // ── Elixir / Erlang ─────────────────────────────────────
    "~/.hex/",
    "~/.mix/",
    // ── Haskell ─────────────────────────────────────────────
    "~/.cabal/",
    "~/.local/state/cabal/",
    "~/.stack/",
    "~/AppData/Local/stack/",
    "~/AppData/Roaming/stack/",
    // ── Lua (LuaRocks) ──────────────────────────────────────
    "~/.luarocks/",
    "~/.cache/luarocks/",
    "~/Library/Caches/luarocks/",
    "~/AppData/Local/luarocks/",
    // ── R ───────────────────────────────────────────────────
    "~/Library/R/",
    "~/R/",
    "~/Documents/R/",
    // ── OCaml (opam) ────────────────────────────────────────
    "~/.opam/",
    // ── Julia ───────────────────────────────────────────────
    "~/.julia/",
    // ── Nix ─────────────────────────────────────────────────
    "/nix/store/",
    // ── System package managers ─────────────────────────────
    "/opt/local/",
    "~/.local/pipx/",
];

/// Allowed filesystem roots for scratch/temp files (single source of truth).
///
/// Used by read-path allowlists and read-only shell redirect / scratch-write policy.
static ALLOWED_TEMP_ROOTS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = Vec::new();
    add_path_with_canonical(&mut dirs, std::env::temp_dir());
    // The unix shared temp directories. Elsewhere they are not temp roots at
    // all, and (never canonicalizing) they would sit in the allowlist as raw
    // paths, so they are gated rather than merely documented.
    #[cfg(unix)]
    {
        add_path_with_canonical(&mut dirs, PathBuf::from("/tmp"));
        add_path_with_canonical(&mut dirs, PathBuf::from("/private/tmp"));
        add_path_with_canonical(&mut dirs, PathBuf::from("/var/tmp"));
    }
    // Explicit spill directory (usually under `temp_dir()`; documents intent).
    add_path_with_canonical(&mut dirs, std::env::temp_dir().join(".agent"));
    // The legacy (pre-pin) OS temp dir — the one captured before the daemon
    // pinned the temp environment to the private root: on macOS the darwin user
    // temp dir (`/var/folders/.../T`), where bare `mktemp -d` STILL lands
    // because it ignores the temp variables, and on Windows the user's own temp
    // directory. The read allowlist and the cleaner's scan roots
    // (`crate::temp`) are built from this same value, so the guard's coverage
    // stays exactly as broad as before the pin (no write restrictions are added
    // by the consolidation).
    if let Some(legacy) = crate::temp::legacy_temp_dir() {
        add_path_with_canonical(&mut dirs, legacy.to_path_buf());
        add_path_with_canonical(&mut dirs, legacy.join(".agent"));
    }
    dirs
});

/// Paths under any of these directories are allowed for reading (temp dir,
/// dependency caches, SDK headers, etc.). Paths are canonicalized at init
/// to handle symlinks (e.g. macOS `/tmp` → `/private/tmp`) so that
/// `is_path_in_extra_allowed()` matches paths resolved by `resolve_read_target`.
///
/// Both the canonicalized and raw paths are included because `resolve_read_target`
/// validates the path twice — once before canonicalization (raw user-provided path)
/// and once after — and both validations may bypass via `EXTRA_READ_ALLOWED`.
///
/// `~`-prefixed entries are expanded at init time using
/// [`crate::util::expand_tilde`]. If `$HOME` (and `$USERPROFILE` on Windows)
/// is unset, `~`-prefixed entries are skipped. Entries that follow XDG Base
/// Directory conventions (`~/.cache/`, `~/.local/share/`, `~/.local/state/`,
/// `~/.config/`) also generate variants using the corresponding `$XDG_*`
/// environment variable when set. Roots whose location is carried by an
/// environment variable — relocated toolchain homes and the Windows system
/// roots a Windows install would otherwise place on the system drive — are
/// likewise emitted from the variable value when set (see
/// [`ENV_DERIVED_ROOTS`]).
static EXTRA_READ_ALLOWED: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = ALLOWED_TEMP_ROOTS.clone();

    // Dependency source directories (cross-platform)
    for raw_path in EXTRA_ALLOWED_RAW_PATHS {
        if raw_path.starts_with('~') {
            let expanded = crate::util::expand_tilde(raw_path);
            // Skip if expansion didn't work (HOME unset → literal ~ kept)
            if expanded.to_string_lossy().starts_with('~') {
                continue;
            }
            add_path_with_canonical(&mut dirs, expanded);

            // XDG variant (e.g. ~/.cache/pypoetry → $XDG_CACHE_HOME/pypoetry)
            if let Some(xdg_path) = xdg_variant_path(raw_path) {
                add_path_with_canonical(&mut dirs, PathBuf::from(xdg_path));
            }
        } else {
            add_path_with_canonical(&mut dirs, PathBuf::from(raw_path));
        }
    }

    // Env-derived roots (whole-prefix replacement, e.g. $CARGO_HOME/registry/src
    // or $ProgramData/chocolatey/lib when those variables are set).
    for raw_path in env_derived_allowed_paths(|var| std::env::var(var).ok()) {
        add_path_with_canonical(&mut dirs, PathBuf::from(raw_path));
    }

    // System JVM `include/` header roots: enumerated one level at init so the
    // JDK trees are scoped to headers only, not whole-directory allowlisted.
    // Nothing is added when the root is absent (fail closed).
    for header_root in
        jvm_include_roots("/Library/Java/JavaVirtualMachines", "Contents/Home/include")
            .into_iter()
            .chain(jvm_include_roots("/usr/lib/jvm", "include"))
    {
        add_path_with_canonical(&mut dirs, header_root);
    }

    dirs
});

// ── Path validation ──────────────────────────

/// Check whether `path` is safe to access within the given `workspace_root`.
///
/// This is the central security gate for all file-path operations: it performs
/// a purely lexical check — no filesystem I/O — so it cannot detect symlink-based
/// escapes. The caller is responsible for that post-canonicalization validation
/// (see [`resolve_read_target`] and [`resolve_write_target`]).
///
/// The workspace comparison runs through [`crate::util::is_within`], so a stored
/// root and a canonicalized candidate that differ only by the Windows verbatim
/// prefix still match.
#[must_use]
fn is_path_safe_for_workspace(path: &str, workspace_root: &Path) -> bool {
    let path = path.trim();
    if path.is_empty() {
        return true; // empty after trim → relative, safe
    }
    // Bare tilde is shorthand for workspace root (see resolve_tool_path_with_base)
    if path == "~" {
        return true;
    }
    if path.contains('\0') {
        return false;
    }
    if Path::new(path)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return false;
    }
    if path.starts_with('~') && !path.starts_with("~/") {
        return false;
    }
    let expanded_path = crate::util::expand_tilde(path);
    if expanded_path.is_absolute() {
        // Lexical prefix check — no sync I/O.
        //
        // Note: workspace_root is pre-canonicalized at workspace registration
        // (see canonicalize_workspace_path in workspace.rs). The lexical check
        // may reject absolute paths whose prefix is a symlink into the workspace
        // (e.g. /tmp/… when the real workspace is /private/tmp/… on macOS), but
        // this is harmless: agents use relative paths, and the post-canonicalization
        // checks in resolve_read_target / resolve_write_target catch any symlink
        // escapes that would bypass this pre-check.
        crate::util::is_within(&expanded_path, workspace_root)
    } else {
        // Relative path without parent-dir components — always safe
        true
    }
}

/// Resolve a user path segment against `workspace_root`.
///
/// A bare `~` means the workspace root — except for the admin's Assistant,
/// whose `~` is the shell's: the user's home directory, bare or with a path
/// under it (`~/notes.md`). `~user` is refused there rather than answered with
/// `$HOME/user`, which is not the file the shell would name; the other access
/// levels keep their existing wording, where the frame refuses it.
fn resolve_tool_path_with_base(
    path: &str,
    workspace_root: &Path,
    access: PathAccess,
) -> anyhow::Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Ok(workspace_root.to_path_buf());
    }

    if access == PathAccess::Unrestricted {
        if is_foreign_tilde(trimmed) {
            anyhow::bail!(
                "forbidden: cannot resolve {path}: `~user` is not expanded — hint: only `~` and \
                 `~/…` are, and they mean your own home directory; spell another user's path \
                 absolutely"
            );
        }
        if trimmed.starts_with('~') {
            let home = crate::util::expand_tilde(trimmed);
            if !home.is_absolute() {
                anyhow::bail!(
                    "forbidden: cannot resolve {path}: no home directory is set — hint: spell the \
                     path absolutely"
                );
            }
            return Ok(home);
        }
    } else if trimmed == "~" {
        return Ok(workspace_root.to_path_buf());
    }

    let expanded = crate::util::expand_tilde(trimmed);
    if expanded.is_absolute() {
        return Ok(expanded);
    }
    Ok(workspace_root.join(expanded))
}

/// A `~`-prefixed spelling that names another user (`~user…`) rather than the
/// caller's own home — [`crate::util::expand_tilde`] would answer it with
/// `$HOME/user…`, a file the caller did not name.
fn is_foreign_tilde(path: &str) -> bool {
    path.starts_with('~') && path != "~" && !path.starts_with("~/")
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::canonical_without_verbatim_prefix;
    use tempfile::TempDir;

    // ── Path validation: is_path_safe_for_workspace ─────────────────────

    #[expect(clippy::too_many_lines)]
    #[test]
    fn is_path_safe_for_workspace_all_cases() {
        struct Case {
            name: &'static str,
            path: &'static str,
            safe: bool,
        }

        // Most security invariants are workspace-independent and can be
        // verified against a minimal "." base.
        let dot_cases = [
            Case {
                name: "traversal_etc_passwd",
                path: "../etc/passwd",
                safe: false,
            },
            Case {
                name: "traversal_via_foo",
                path: "foo/../etc/passwd",
                safe: false,
            },
            Case {
                name: "my_dot_dot_file",
                path: "my..file.txt",
                safe: true,
            },
            Case {
                name: "null_byte",
                path: "file\0.txt",
                safe: false,
            },
            Case {
                name: "proc_self_root",
                path: "/proc/self/root/etc/passwd",
                safe: false,
            },
            Case {
                name: "docker_socket",
                path: "/var/run/docker.sock",
                safe: false,
            },
            Case {
                name: "home_ssh_key_outside_workspace",
                path: "~/.ssh/id_rsa",
                safe: false,
            },
            Case {
                name: "home_gnupg_outside_workspace",
                path: "~/.gnupg/secring.gpg",
                safe: false,
            },
            Case {
                name: "tilde_user_ssh",
                path: "~root/.ssh/id_rsa",
                safe: false,
            },
            Case {
                name: "tilde_nobody",
                path: "~nobody",
                safe: false,
            },
            Case {
                name: "root_slash",
                path: "/",
                safe: false,
            },
            Case {
                name: "anything_absolute",
                path: "/anything",
                safe: false,
            },
            Case {
                name: "absolute_tmp",
                path: "/tmp",
                safe: false,
            },
            Case {
                name: "absolute_var_log",
                path: "/var/log",
                safe: false,
            },
            Case {
                name: "whitespace_absolute",
                path: "  /etc/passwd",
                safe: false,
            },
            Case {
                name: "tab_absolute",
                path: "\t/etc/passwd",
                safe: false,
            },
            Case {
                name: "whitespace_tilde_user",
                path: "  ~root/.ssh/id_rsa",
                safe: false,
            },
            Case {
                name: "whitespace_traversal",
                path: "  ../foo",
                safe: false,
            },
            Case {
                name: "whitespace_only_empty",
                path: "  ",
                safe: true,
            },
            Case {
                name: "bare_tilde",
                path: "~",
                safe: true,
            },
            Case {
                name: "leading_whitespace_tilde",
                path: "  ~",
                safe: true,
            },
            Case {
                name: "trailing_whitespace_tilde",
                path: "~  ",
                safe: true,
            },
            Case {
                name: "both_whitespace_tilde",
                path: "  ~  ",
                safe: true,
            },
        ];
        let base = Path::new(".");
        for case in &dot_cases {
            assert_eq!(
                is_path_safe_for_workspace(case.path, base),
                case.safe,
                "case: {}",
                case.name
            );
        }

        // Workspace-relative checks (ws_cases below) require a real
        // TempDir because they exercise workspace-scope behavior that
        // can't be tested against a static "." base.
        let tmp = TempDir::new().expect("tempdir");
        let ws = tmp.path().to_path_buf();

        // Absolute path inside the workspace (only expressible at runtime)
        assert!(
            is_path_safe_for_workspace(ws.join("test.txt").to_str().unwrap(), &ws),
            "absolute path inside workspace should be safe"
        );

        let ws_cases = [
            Case {
                name: "relative_in_workspace",
                path: "relative.txt",
                safe: true,
            },
            Case {
                name: "relative_src_main",
                path: "src/main.rs",
                safe: true,
            },
            Case {
                name: "relative_deep_nested",
                path: "deep/nested/dir/file.txt",
                safe: true,
            },
            Case {
                name: "dot_gitignore",
                path: ".gitignore",
                safe: true,
            },
            Case {
                name: "dot_env",
                path: ".env",
                safe: true,
            },
            Case {
                name: "empty_string",
                path: "",
                safe: true,
            },
            Case {
                name: "ws_traversal_etc_passwd",
                path: "../etc/passwd",
                safe: false,
            },
            Case {
                name: "double_traversal_ssh",
                path: "../../root/.ssh/id_rsa",
                safe: false,
            },
            Case {
                name: "triple_traversal_shadow",
                path: "foo/../../../etc/shadow",
                safe: false,
            },
            Case {
                name: "dot_dot_alone",
                path: "..",
                safe: false,
            },
            Case {
                name: "bare_tilde_in_workspace",
                path: "~",
                safe: true,
            },
        ];
        for case in &ws_cases {
            assert_eq!(
                is_path_safe_for_workspace(case.path, &ws),
                case.safe,
                "case: {}",
                case.name
            );
        }

        // Paths outside the workspace (even temp dirs and dependency paths)
        // must be blocked — the extra-read bypass is read-only and must not
        // affect write-path checks.
        {
            let temp_file = std::env::temp_dir().join("test-spill.txt");
            assert!(
                !is_path_safe_for_workspace(temp_file.to_str().unwrap(), &ws),
                "Temp file should be blocked by base check"
            );
        }
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                !is_path_safe_for_workspace(&dep_path, &ws),
                "Dependency path should be blocked by base check"
            );
        }
    }

    /// Containment is judged in one spelling, so a candidate that differs from
    /// the root only by the Windows verbatim prefix is still accepted — in both
    /// directions — while a path outside the root is still refused. The
    /// separator is a forward slash so the assertion means the same thing on
    /// both platforms (Windows accepts either separator), which is what makes
    /// it host-runnable evidence for the comparison logic.
    #[test]
    fn containment_accepts_either_spelling_but_is_case_sensitive() {
        assert!(crate::util::is_within(
            Path::new("C:/ws/a/b"),
            Path::new(r"\\?\C:/ws")
        ));
        assert!(crate::util::is_within(
            Path::new(r"\\?\C:/ws/a/b"),
            Path::new("C:/ws")
        ));
        assert!(!crate::util::is_within(
            Path::new(r"\\?\C:/other"),
            Path::new("C:/ws")
        ));
        // Ordinary name components are compared exactly — no case folding — so a
        // differently-cased directory is a sibling of the base, not inside it. The
        // drive letter cannot witness this: Windows folds it while parsing a path,
        // while unix compares it as an ordinary name component, so a drive-case
        // assertion holds on one host and fails on the other.
        assert!(!crate::util::is_within(
            Path::new("C:/ws/Docs/a"),
            Path::new("C:/ws/docs")
        ));
    }

    /// The location rules' own comparison: same shape as [`crate::util::is_within`]
    /// but case-blind, so a differently-cased spelling of a protected place is
    /// that place — the bypass a byte-exact comparison left open on the
    /// case-insensitive volumes the product ships on.
    #[test]
    fn location_containment_is_case_blind() {
        assert!(is_within_ignoring_case(
            Path::new("/Storage/DB/Core.db"),
            Path::new("/storage/db")
        ));
        assert!(is_within_ignoring_case(
            Path::new(r"\\?\C:/ws/a/b"),
            Path::new("C:/WS")
        ));
        // A sibling that merely shares the name prefix is still outside, in
        // either spelling.
        assert!(!is_within_ignoring_case(
            Path::new("/storage/db-other/core.db"),
            Path::new("/storage/db")
        ));
        assert!(!is_within_ignoring_case(
            Path::new("/other/core.db"),
            Path::new("/storage/db")
        ));
    }

    // ── is_path_under_roots / allowed_temp_roots tests ───────────────────

    /// The roots gate settles the verbatim spelling the way containment does: a
    /// root stored in either form is honoured, while a sibling directory that
    /// merely shares the name prefix is not.
    #[test]
    fn verbatim_root_spelling_is_settled_by_the_roots_gate() {
        let roots = [PathBuf::from("C:/ws")];
        let verbatim_roots = [PathBuf::from(r"\\?\C:/ws")];
        assert!(is_path_under_roots(Path::new(r"\\?\C:/ws/a/b"), &roots));
        assert!(is_path_under_roots(Path::new("C:/ws/a/b"), &verbatim_roots));
        assert!(!is_path_under_roots(Path::new("C:/ws-other"), &roots));
    }

    #[test]
    fn is_path_under_allowed_temp_covers_common_roots() {
        let roots = allowed_temp_roots();
        let temp = std::env::temp_dir();
        let spill = temp.join(".agent/spill_test.txt");
        assert!(is_path_under_roots(&temp.join("scratch.txt"), &roots));
        assert!(is_path_under_roots(&spill, &roots));
        assert!(is_path_under_roots(Path::new("/tmp/out.txt"), &roots));
        assert!(is_path_under_roots(Path::new("/var/tmp/out.txt"), &roots));
        assert!(!is_path_under_roots(Path::new("relative.txt"), &roots));
        assert!(!is_path_under_roots(Path::new("/etc/passwd"), &roots));
        // Path traversal via `..` must be blocked.
        assert!(
            !is_path_under_roots(Path::new("/tmp/../etc/passwd"), &roots),
            "Path traversal via /tmp/../etc/passwd must be blocked"
        );
        assert!(
            !is_path_under_roots(Path::new("/tmp/../../../etc/passwd"), &roots),
            "Deep path traversal must be blocked"
        );
        // Traversal that stays within temp after normalization should be allowed.
        assert!(
            is_path_under_roots(Path::new("/tmp/../tmp/file.txt"), &roots),
            "Traversal back into temp should be allowed"
        );
    }

    // ── check_path_read_allowed: spill / extra-read ────────────────────

    #[test]
    fn check_path_read_allowed_all_cases() {
        struct Case {
            name: &'static str,
            path: String,
            allowed: bool,
        }

        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        let mut cases: Vec<Case> = Vec::new();

        let spill = std::env::temp_dir().join(".agent/spill_ab12.txt");
        cases.push(Case {
            name: "temp_agent_spill",
            path: spill.to_string_lossy().to_string(),
            allowed: true,
        });
        let full_log = std::env::temp_dir().join(".agent/12345_cargo_check.full.log");
        cases.push(Case {
            name: "full_log_spill",
            path: full_log.to_string_lossy().to_string(),
            allowed: true,
        });

        let in_workspace = workspace.join(".agent/spill_ab12.txt");
        cases.push(Case {
            name: "workspace_spill_rejected",
            path: in_workspace.to_string_lossy().to_string(),
            allowed: false,
        });
        cases.push(Case {
            name: "etc_passwd_rejected",
            path: "/etc/passwd".to_string(),
            allowed: false,
        });
        cases.push(Case {
            name: "var_tmp_allowed",
            path: "/var/tmp/mahbot-test.txt".to_string(),
            allowed: true,
        });

        for case in &cases {
            let result = check_path_read_allowed(&case.path, &workspace, PathAccess::Allowlisted);
            assert_eq!(
                result.is_ok(),
                case.allowed,
                "case: {} — path: {}",
                case.name,
                case.path
            );
        }

        // macOS per-user temp path (e.g. /var/folders/xx/yy/T/…) should be
        // detected as a temp-root spill even when it doesn't match the active
        // temp_dir() at-test-time (which may be /private/var/…).
        // Inline rather than table-driven because this path relies on
        // is_os_temp_root's Unix-specific /var/folders/… detection and
        // #[cfg(unix)] wrapping a table push() would be more awkward.
        #[cfg(unix)]
        {
            let mac_spill = PathBuf::from("/var/folders/xx/yy/T/.agent/spill_cd34.txt");
            assert!(
                check_path_read_allowed(
                    &mac_spill.to_string_lossy(),
                    &workspace,
                    PathAccess::Allowlisted
                )
                .is_ok(),
                "macOS-shaped spill path should be allowed"
            );
        }

        // Non-temp spill-shaped path outside the workspace must be rejected.
        // Inline rather than table-driven for the same reason as the
        // macOS-spill block above — the path relies on is_os_temp_root's
        // Unix-specific logic and the #[cfg(unix)] guard is cleaner here.
        #[cfg(unix)]
        {
            let outside = PathBuf::from("/usr/local/.agent/spill_ab12.txt");
            assert!(
                check_path_read_allowed(
                    &outside.to_string_lossy(),
                    &workspace,
                    PathAccess::Allowlisted
                )
                .is_err(),
                "non-temp spill-shaped path should be rejected"
            );
        }
    }

    // ── EXTRA_READ_ALLOWED tests ──────────────────────────────────────

    #[test]
    fn extra_allowed_all_cases() {
        struct Case {
            name: &'static str,
            path: &'static str,
            allowed: bool,
        }

        // Force LazyLock init and verify it's populated (preserved from the
        // removed extra_allowed_init_does_not_panic for diagnostic clarity).
        assert!(
            !EXTRA_READ_ALLOWED.is_empty(),
            "EXTRA_READ_ALLOWED should not be empty"
        );

        // Paths inside known dependency directories should match via
        // is_path_in_extra_allowed.
        let cases = [
            Case {
                name: "cargo_registry_tilde",
                path: "~/.cargo/registry/src/some-crate/src/lib.rs",
                allowed: true,
            },
            Case {
                name: "system_python_site_packages",
                path: "/usr/local/lib/python3.12/site-packages/requests/models.py",
                allowed: true,
            },
        ];
        for case in &cases {
            assert_eq!(
                is_path_in_extra_allowed(Path::new(case.path)),
                case.allowed,
                "case: {}",
                case.name
            );
        }

        // Expanded home-dir path (requires $HOME at runtime)
        if let Ok(home) = std::env::var("HOME") {
            let expanded = PathBuf::from(&home).join(".cargo/registry/src/some-crate/src/lib.rs");
            assert!(
                is_path_in_extra_allowed(&expanded),
                "expanded cargo registry path should match"
            );
        }

        // Prefix strictness: sibling of an allowed root should NOT match
        #[cfg(unix)]
        {
            assert!(
                !is_path_in_extra_allowed(Path::new("/usr/local/lib_evil/foo")),
                "Sibling of allowed root should not match"
            );
            assert!(
                !is_path_in_extra_allowed(Path::new("/usr/local/lib64/foo")),
                "Numeric suffix should not match"
            );
        }

        // No literal tilde paths in the allowlist itself
        for dir in &*EXTRA_READ_ALLOWED {
            let s = dir.to_string_lossy();
            assert!(
                !s.starts_with('~'),
                "Literal tilde path should never be stored: {s}"
            );
        }
    }

    /// `check_path_read_allowed` permits paths under dependency source
    /// directories (e.g. cargo registry, pip site-packages) even when
    /// they're outside the workspace.
    #[test]
    fn check_path_read_allowed_extra_dependency_paths() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        // Temp dir file — should be allowed for read via extra_allowed
        let temp_file = std::env::temp_dir().join("test-read.txt");
        let temp_str = temp_file.to_string_lossy().to_string();
        assert!(
            check_path_read_allowed(&temp_str, &workspace, PathAccess::Allowlisted).is_ok(),
            "Temp file should be allowed for read"
        );

        // Dependency path
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                check_path_read_allowed(&dep_path, &workspace, PathAccess::Allowlisted).is_ok(),
                "Dependency path should be allowed for read"
            );

            // ~-prefixed version (pre-canonicalization check)
            let tilde_input = "~/.cargo/registry/src/crate-0.1.0/src/lib.rs";
            assert!(
                check_path_read_allowed(tilde_input, &workspace, PathAccess::Allowlisted).is_ok(),
                "~-prefixed dependency path should be allowed for read"
            );
        }
    }

    // ── Env-derived allowed-path tests ─────────────────────────────────

    #[test]
    fn env_derived_allowed_paths_pure() {
        // Unset/unknown vars are skipped.
        assert!(env_derived_allowed_paths(|_| None).is_empty());

        // CARGO_HOME with subpaths.
        let paths = env_derived_allowed_paths(|var| match var {
            "CARGO_HOME" => Some("/opt/cargo".to_string()),
            _ => None,
        });
        assert_eq!(
            paths,
            vec![
                "/opt/cargo/registry/src/".to_string(),
                "/opt/cargo/git/checkouts/".to_string(),
            ]
        );

        // Trailing slash on the value is trimmed before joining subpaths.
        let paths = env_derived_allowed_paths(|var| match var {
            "CARGO_HOME" => Some("/opt/cargo/".to_string()),
            _ => None,
        });
        assert_eq!(
            paths,
            vec![
                "/opt/cargo/registry/src/".to_string(),
                "/opt/cargo/git/checkouts/".to_string(),
            ]
        );

        // Empty subpath → the bare, trailing-slash-trimmed value.
        let paths = env_derived_allowed_paths(|var| match var {
            "GOMODCACHE" => Some("/modcache".to_string()),
            _ => None,
        });
        assert_eq!(paths, vec!["/modcache".to_string()]);

        // Non-empty subpath appends the subpath to the value.
        let paths = env_derived_allowed_paths(|var| match var {
            "GOROOT" => Some("/go".to_string()),
            _ => None,
        });
        assert_eq!(paths, vec!["/go/src/".to_string()]);

        // Empty value is skipped.
        assert!(
            env_derived_allowed_paths(|var| match var {
                "GOMODCACHE" => Some(String::new()),
                _ => None,
            })
            .is_empty()
        );

        // A system root contributes each of its subpaths, spelled with the
        // host's own separator.
        let sep = std::path::MAIN_SEPARATOR;
        let paths = env_derived_allowed_paths(|var| match var {
            "ProgramData" => Some("/data".to_string()),
            "ProgramFiles" => Some("/program files".to_string()),
            _ => None,
        });
        assert_eq!(
            paths,
            vec![
                format!("/data{sep}chocolatey{sep}lib"),
                format!("/program files{sep}Microsoft Visual Studio"),
            ]
        );

        // A trailing separator on the value does not double up.
        assert_eq!(
            env_derived_allowed_paths(|var| match var {
                "ProgramData" => Some("/data/".to_string()),
                _ => None,
            }),
            vec![format!("/data{sep}chocolatey{sep}lib")]
        );

        // A BARE drive value (`SystemDrive=C:`) is joined with an explicit
        // separator: `C:` is drive-RELATIVE, so a plain join would produce
        // `C:msys64…` — a different path, and the reason this formatting exists.
        assert_eq!(
            env_derived_allowed_paths(|var| match var {
                "SystemDrive" => Some("C:".to_string()),
                _ => None,
            }),
            vec![
                format!("C:{sep}msys64{sep}mingw64{sep}include"),
                format!("C:{sep}msys64{sep}ucrt64{sep}include"),
                format!("C:{sep}msys64{sep}clang64{sep}include"),
                format!("C:{sep}msys64{sep}usr{sep}include"),
            ]
        );
    }

    #[test]
    fn xdg_variant_path_with_pure() {
        let get = |var: &str| match var {
            "XDG_CACHE_HOME" => Some("/xdg".to_string()),
            _ => None,
        };

        // XDG subpath maps to the env-derived path.
        assert_eq!(
            xdg_variant_path_with(get, "~/.cache/pypoetry/"),
            Some("/xdg/pypoetry/".to_string())
        );

        // Non-XDG path returns None.
        assert_eq!(xdg_variant_path_with(get, "~/.cargo/registry/src/"), None);

        // Unset var returns None.
        assert_eq!(xdg_variant_path_with(|_| None, "~/.cache/foo/"), None);
    }

    #[test]
    fn access_levels_each_reach_what_they_should() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        let outside = "/etc/passwd";
        let dependency = "~/.cargo/registry/src/crate-0.1.0/src/lib.rs";
        let credential = "~/.ssh/id_rsa";

        // The guest Assistant reaches the workspace and nothing else.
        assert!(check_path_read_allowed("src/main.rs", &workspace, PathAccess::Workspace).is_ok());
        for path in [outside, dependency, credential, "/tmp/note.txt"] {
            assert!(
                check_path_read_allowed(path, &workspace, PathAccess::Workspace).is_err(),
                "the workspace-only read must refuse {path}"
            );
        }

        // A pipeline role reaches the workspace and its dependency caches —
        // including a name that used to be a credential denial, inside the
        // workspace where its own envelope already reached.
        for path in ["src/main.rs", dependency, "src/id_rsa"] {
            assert!(
                check_path_read_allowed(path, &workspace, PathAccess::Allowlisted).is_ok(),
                "the general read must allow {path}"
            );
        }
        assert!(
            check_path_read_allowed(outside, &workspace, PathAccess::Allowlisted).is_err(),
            "the general read stays inside its envelope"
        );
        assert!(
            check_path_read_allowed(credential, &workspace, PathAccess::Allowlisted).is_err(),
            "a credential place outside the envelope stays out of a pipeline role's reach"
        );

        // The admin Assistant reaches the machine — `..`, symlinks and
        // "forbidden" names included, all of which the other levels refuse.
        for path in [
            outside,
            dependency,
            credential,
            "/tmp/note.txt",
            "../sibling/file.txt",
            "~/.aws/credentials",
        ] {
            assert!(
                check_path_read_allowed(path, &workspace, PathAccess::Unrestricted).is_ok(),
                "the unrestricted read must allow {path}"
            );
        }
    }

    // ── Store-file tests ───────────────────────────────────────────────

    /// A store file is recognised from its name and its place, never from its
    /// contents: the two live stores and their tails are refused, the recovery
    /// copies are recognised as copies, and everything else in the directory —
    /// including a stale `config.db` or a file left by a retired store name — is
    /// an ordinary file. Name and place are matched without case, because the
    /// volumes the service runs on compare them without case: a differently-cased
    /// spelling is the same file.
    #[test]
    fn store_files_are_classified_by_name_and_place() {
        let root = Path::new("/storage");
        let db = root.join("db");

        for name in [
            "core.db",
            "logs.db",
            "core.db-wal",
            "logs.db-wal",
            "core.db-shm",
            "logs.db-journal",
            // The same files as the volume sees them.
            "Core.db",
            "CORE.DB",
            "core.DB-wal",
            "Logs.Db-SHM",
        ] {
            assert_eq!(
                store_file_at(root, &db.join(name)),
                Some(StoreFile::Live),
                "{name} is a live store's own file"
            );
        }

        for name in [
            "core.db.quarantine-20260101T101010Z-42",
            "core.db.quarantine-20260101T101010Z-42-1",
            "logs.db.quarantine-20260101T101010Z-42-wal",
            "core.db.pre-reindex-20260101T101010Z-42",
            "core.db.pre-reindex-20260101T101010Z-42-wal",
            "core.db.rebuild-20260101T101010Z",
            // The same copies as the volume sees them.
            "Core.db.Quarantine-20260101t101010z-42",
            "LOGS.DB.Pre-Reindex-20260101T101010z-42",
            // A copy of ANY store: the service quarantines a retired store name
            // exactly as it quarantines a live one.
            "board.db.quarantine-20260101T101010Z-42",
            "sessions.db.pre-reindex-20260101T101010Z-42",
            "stats.db.rebuild-20260101T101010Z",
        ] {
            assert_eq!(
                store_file_at(root, &db.join(name)),
                Some(StoreFile::Copy),
                "{name} is a copy the service keeps of its own store"
            );
        }

        for name in [
            "config.db",
            ".DS_Store",
            "core.db.quarantine-nonsense",
            "some_other.db",
            "core.txt",
            // A retired store name is not a live store: the service runs on the
            // consolidated file and the logs file, and everything else in the
            // directory is an ordinary file.
            "board.db",
            "board.db-wal",
            "sessions.db",
        ] {
            assert_eq!(
                store_file_at(root, &db.join(name)),
                None,
                "{name} is an ordinary file"
            );
        }

        // The directory's own spelling does not decide either.
        assert_eq!(
            store_file_at(root, &root.join("DB/core.db")),
            Some(StoreFile::Live),
            "a store file under a differently-cased store directory is the same file"
        );

        // Outside `db/`, the same names mean nothing.
        assert_eq!(store_file_at(root, &root.join("core.db")), None);
        assert_eq!(
            store_file_at(root, &root.join("models/core.db")),
            None,
            "a store name outside the store directory is an ordinary file"
        );
    }

    /// The read refusal is wired to the running storage root for every access
    /// level — the product invariant, not a rule of a role.
    #[test]
    fn live_stores_are_refused_at_every_access_level() {
        let Some(root) = storage_root() else {
            return; // no HOME and no user directory: nothing to refuse
        };
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        for name in [
            "core.db",
            "logs.db",
            "core.db-wal",
            "Core.db",
            "core.DB-WAL",
        ] {
            let path = root.join("db").join(name);
            let path = path.to_string_lossy().to_string();
            for access in [
                PathAccess::Workspace,
                PathAccess::Allowlisted,
                PathAccess::Unrestricted,
            ] {
                let err = check_path_read_allowed(&path, &workspace, access)
                    .expect_err("a live store must be refused");
                assert!(
                    err.to_string().contains("live databases"),
                    "{name} at {access:?}: {err}"
                );
            }
        }

        // A copy is readable — only writing to it is refused.
        let copy = root.join("db/core.db.quarantine-20260101T101010Z-42");
        assert!(
            check_path_read_allowed(
                &copy.to_string_lossy(),
                &workspace,
                PathAccess::Unrestricted
            )
            .is_ok(),
            "a snapshot is an ordinary read"
        );
    }

    #[test]
    fn check_path_read_allowed_new_dependency_roots() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        let allowed = [
            "~/.rustup/toolchains/1.98.0/lib/rustlib/src/rust/library/core/src/lib.rs",
            "~/.m2/repository/org/example/foo/1.0/foo-1.0.jar",
            "~/.gradle/caches/modules-2/files-2.1/org.example/foo/1.0/foo-1.0.jar",
            "~/.npm/_cacache/content-v2/sha512/ab/foo",
            "~/.nvm/versions/node/v22/lib/node_modules/foo/index.js",
            "~/.volta/tools/image/node/current/bin/node",
            "~/.mix/archives/foo-1.0.ez",
            "/usr/include/stdio.h",
            "/opt/homebrew/include/zlib.h",
            "/opt/homebrew/opt/openssl/include/openssl/ssl.h",
            "/usr/local/include/foo.h",
            "/usr/local/go/src/fmt/print.go",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/stdio.h",
        ];
        for path in allowed {
            assert!(
                check_path_read_allowed(path, &workspace, PathAccess::Allowlisted).is_ok(),
                "expected allowed: {path}"
            );
        }

        // System JVM locations are allowlisted via init-time I/O enumeration
        // (scoped to `include/` headers only, never the whole JDK tree), so a
        // lexically-nonexistent JDK name can no longer be asserted as allowed.
        // If the system JVM root exists, verify the enumerated header for every
        // child passes; otherwise skip (the test must not depend on a specific
        // JDK being installed).
        if let Ok(entries) = std::fs::read_dir("/Library/Java/JavaVirtualMachines") {
            for entry in entries.filter_map(std::io::Result::ok) {
                let header = entry
                    .path()
                    .join("Contents/Home/include/jni.h")
                    .to_string_lossy()
                    .into_owned();
                assert!(
                    check_path_read_allowed(&header, &workspace, PathAccess::Allowlisted).is_ok(),
                    "expected enumerated JVM header allowed: {header}"
                );
            }
        }

        // Gradle root (outside caches/) is NOT covered: the allowlist is
        // deliberately as narrow as the caches themselves.
        let err = check_path_read_allowed(
            "~/.gradle/gradle.properties",
            &workspace,
            PathAccess::Allowlisted,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("outside the allowed read envelope"),
            "~/.gradle/gradle.properties should fall through to the generic error: {err}"
        );
    }

    /// The admin's write refusals, which are the only two: the product's own
    /// store files, and the registered project workspaces — with the admin's
    /// own folder outranking the second rule.
    #[test]
    fn unrestricted_write_refusals() {
        let own = Path::new("/storage/userspaces/admin");
        let project = PathBuf::from("/work/proj");
        let projects = vec![("proj".to_string(), project.clone())];

        let err =
            unrestricted_write_refusal(own, &project.join("src/main.rs"), "src/main.rs", &projects)
                .expect("a write inside a project must be refused");
        assert!(err.to_string().contains("'proj'"), "{err}");
        assert!(
            err.to_string().contains("manager"),
            "the refusal must name the route: {err}"
        );

        // The volume compares names without case, so a differently-cased
        // spelling of the project is the same place and must be refused too —
        // the bypass a byte-exact comparison left open.
        let err = unrestricted_write_refusal(
            own,
            &PathBuf::from("/WORK/Proj/src/main.rs"),
            "src/main.rs",
            &projects,
        )
        .expect("a differently-cased project spelling must be refused");
        assert!(err.to_string().contains("'proj'"), "{err}");

        // Everything else is an ordinary file: another tree, another user's
        // folder, and a database of any kind — the product's own stores
        // excepted, and only by location.
        for allowed in [
            "/work/other/src/main.rs",
            "/storage/userspaces/bob/notes.md",
            "/home/bob/user.db",
            "/var/lib/postgresql/data/base.db",
        ] {
            assert!(
                unrestricted_write_refusal(own, Path::new(allowed), "x", &projects).is_none(),
                "{allowed} must be writable"
            );
        }

        // The admin's own folder outranks the project rule, even when it sits
        // inside a registered area.
        let nested = project.join("personal");
        assert!(
            unrestricted_write_refusal(&nested, &nested.join("MEMORY.md"), "MEMORY.md", &projects)
                .is_none(),
            "the admin's own workspace stays writable inside a registered area"
        );
        assert!(
            unrestricted_write_refusal(
                &nested,
                &project.join("shared.txt"),
                "shared.txt",
                &projects
            )
            .is_some(),
            "a sibling of the admin's own folder is still the project's"
        );
    }

    /// The admin's own folder outranks the project rule even when the two are
    /// spelled differently: the candidate arrives resolved, so the folder is
    /// compared in its canonical spelling rather than as it was stored.
    #[cfg(unix)]
    #[test]
    fn the_admin_own_folder_outranks_a_project_it_is_spelled_differently_in() {
        let tmp = TempDir::new().unwrap();
        let base = std::fs::canonicalize(tmp.path()).expect("the temp dir resolves");
        let project = base.join("real/proj");
        std::fs::create_dir_all(project.join("admin")).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&project, &link).unwrap();
        let projects = vec![("proj".to_string(), project.clone())];

        // The folder kept under the link's spelling, the candidate resolved.
        let own = link.join("admin");
        assert!(
            unrestricted_write_refusal(
                &own,
                &project.join("admin/notes.md"),
                "notes.md",
                &projects
            )
            .is_none(),
            "the admin's own folder stays writable"
        );
        assert!(
            unrestricted_write_refusal(&own, &project.join("shared.md"), "shared.md", &projects)
                .is_some(),
            "a sibling of the admin's own folder is still the project's"
        );
    }

    /// The project rule end to end, through the store the registry is read
    /// from: the wiring a pure-helper test cannot reach.
    #[tokio::test]
    async fn unrestricted_write_consults_the_registered_projects() {
        crate::util::test::init_test_stores().await;
        let (dir, own) = test_workspace().await;
        let project_dir = TempDir::new().expect("tempdir");
        let project = tokio::fs::canonicalize(project_dir.path())
            .await
            .expect("the project directory resolves");
        crate::util::test::create_test_workspace(&project.to_string_lossy(), "path-policy-project")
            .await;

        // A write inside the project is refused, naming the project and the
        // route, and creates nothing on the way.
        let inside = project.join("src/main.rs");
        let err = resolve_write_target(
            &own,
            &inside.to_string_lossy(),
            true,
            PathAccess::Unrestricted,
        )
        .await
        .expect_err("a write inside a registered project must be refused");
        assert!(err.to_string().contains("path-policy-project"), "{err}");
        assert!(err.to_string().contains("manager"), "{err}");
        assert!(
            !project.join("src").exists(),
            "a refused write creates nothing"
        );

        // What decides is the file the write lands in: a spelling that only
        // passes through the project on its way out is an ordinary write.
        let through = project.join("../elsewhere/new.txt");
        let resolved = resolve_write_target(
            &own,
            &through.to_string_lossy(),
            false,
            PathAccess::Unrestricted,
        )
        .await
        .expect("a write that only passes through the project lands outside it");
        assert!(!resolved.starts_with(&project), "{resolved:?}");

        drop((dir, project_dir));
    }

    /// An unreadable project registry refuses the write with a plain message
    /// rather than an accident — the accepted price is that the admin's own
    /// memory is closed with it.
    #[test]
    fn an_unreadable_project_registry_refuses_the_write() {
        let read = Err(anyhow::anyhow!("the workspace store is not open"));
        let err = registered_projects_from(read, "notes/todo.md")
            .expect_err("an unreadable registry must refuse the write");
        let message = err.to_string();
        assert!(
            message.starts_with("forbidden: cannot write to notes/todo.md"),
            "{message}"
        );
        assert!(message.contains("cannot be read"), "{message}");
        assert!(message.contains("retry once"), "{message}");
    }

    /// The store half of the same rule: the live stores and their tails are
    /// refused, the copies the service keeps are refused too, and nothing is
    /// opened to decide any of it.
    #[test]
    fn unrestricted_write_refuses_store_files() {
        let Some(root) = storage_root() else {
            return;
        };
        let own = TempDir::new().expect("tempdir");
        let own = own.path();
        let projects: Vec<(String, PathBuf)> = Vec::new();

        for name in [
            "core.db",
            "logs.db",
            "core.db-wal",
            "logs.db-shm",
            "Core.db",
            "LOGS.DB",
        ] {
            let candidate = root.join("db").join(name);
            let err = unrestricted_write_refusal(own, &candidate, name, &projects)
                .expect("a live store must be refused");
            assert!(err.to_string().contains("live databases"), "{name}: {err}");
        }

        for name in [
            "core.db.quarantine-20260101T101010Z-42",
            "logs.db.pre-reindex-20260101T101010Z-42",
            "core.db.rebuild-20260101T101010Z",
            "Core.db.Quarantine-20260101t101010z-42",
            "board.db.quarantine-20260101T101010Z-42",
        ] {
            let candidate = root.join("db").join(name);
            let err = unrestricted_write_refusal(own, &candidate, name, &projects)
                .expect("a copy the service keeps must be refused");
            assert!(
                err.to_string().contains("a copy the service keeps"),
                "{name}: {err}"
            );
        }
    }

    /// A `~` reading, for the admin's Assistant alone: the home directory, bare
    /// or with a path under it. Every other access keeps the workspace-root
    /// reading, and `~user` is refused with the way out rather than answered
    /// with `$HOME/user`.
    #[tokio::test]
    async fn tilde_is_the_home_directory_for_the_admin_only() {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return;
        };
        let home = tokio::fs::canonicalize(&home)
            .await
            .expect("the home directory resolves");
        let (dir, ws) = test_workspace().await;

        assert_eq!(
            resolve_read_target(&ws, "~", PathAccess::Unrestricted)
                .await
                .expect("the admin's `~` is readable"),
            home,
            "the admin's bare `~` is the home directory"
        );
        assert_eq!(
            resolve_read_target(&ws, "~", PathAccess::Allowlisted)
                .await
                .expect("`~` is the workspace root for every other access"),
            ws,
        );
        assert_eq!(
            resolve_read_target(&ws, "~/..", PathAccess::Unrestricted)
                .await
                .expect("`~/…` reads from the home directory"),
            home.parent().expect("the home directory has a parent"),
        );

        // Another user's `~` never becomes `$HOME/user`: it is refused, and the
        // refusal says how to spell the path instead.
        for access in [PathAccess::Unrestricted, PathAccess::Allowlisted] {
            let read = resolve_read_target(&ws, "~other/notes.md", access).await;
            let err = read.expect_err("`~user` must be refused");
            if access == PathAccess::Unrestricted {
                assert!(err.to_string().contains("absolutely"), "{err}");
            }
            let err = resolve_write_target(&ws, "~other/notes.md", false, access)
                .await
                .expect_err("`~user` must be refused for writes too");
            if access == PathAccess::Unrestricted {
                assert!(err.to_string().contains("absolutely"), "{err}");
            }
        }

        drop(dir);
    }

    /// The admin's write reaches what the frame refuses everyone else (a `..`
    /// spelling, a database of another product, the admin's own memory),
    /// creates parents on the way, and leaves nothing behind when it is refused.
    #[tokio::test]
    async fn unrestricted_write_reaches_outside_without_leaving_traces() {
        crate::util::test::init_test_stores().await;
        let (dir, own) = test_workspace().await;
        let outside = TempDir::new().expect("tempdir");
        let outside = tokio::fs::canonicalize(outside.path())
            .await
            .expect("the temp dir resolves");

        // A `..` spelling and a brand-new tree outside the workspace.
        let target = own.join("..").join("elsewhere/deep/new.txt");
        let resolved = resolve_write_target(
            &own,
            &target.to_string_lossy(),
            true,
            PathAccess::Unrestricted,
        )
        .await
        .expect("an unrestricted write reaches outside the workspace");
        assert!(resolved.ends_with("elsewhere/deep/new.txt"), "{resolved:?}");
        assert!(resolved.is_absolute(), "the `..` was settled: {resolved:?}");
        tokio::fs::write(&resolved, "hello").await.unwrap();

        // The admin's own workspace and a database of someone else's — the
        // same format as the product's, and an ordinary file for all that.
        for path in [own.join("MEMORY.md"), outside.join("user.db")] {
            let resolved = resolve_write_target(
                &own,
                &path.to_string_lossy(),
                true,
                PathAccess::Unrestricted,
            )
            .await
            .unwrap_or_else(|e| panic!("{} must be writable: {e}", path.display()));
            tokio::fs::write(&resolved, "written").await.unwrap();
            assert!(resolved.exists(), "{resolved:?}");
        }

        drop((dir, outside));
    }

    /// Every symlink the admin's write meets is settled by the file the write
    /// would land in: a link into a private-key folder is writable, a `..`
    /// behind a link follows the target, a dangling link is followed to the file
    /// it names, and a link that leads to a live store is refused.
    #[cfg(unix)]
    #[tokio::test]
    async fn unrestricted_write_settles_every_symlink_by_its_target() {
        use std::os::unix::fs::symlink;
        crate::util::test::init_test_stores().await;
        let (dir, own) = test_workspace().await;
        let outside = TempDir::new().expect("tempdir");
        let outside = tokio::fs::canonicalize(outside.path())
            .await
            .expect("the temp dir resolves");

        let keys = outside.join("keys");
        tokio::fs::create_dir_all(&keys).await.unwrap();
        symlink(&keys, own.join("link-to-keys")).unwrap();
        let via_link = own.join("link-to-keys/id_rsa");
        let resolved = resolve_write_target(
            &own,
            &via_link.to_string_lossy(),
            true,
            PathAccess::Unrestricted,
        )
        .await
        .expect("a symlink into a private-key folder is writable");
        assert_eq!(resolved, keys.join("id_rsa"));
        tokio::fs::write(&resolved, "PRIVATE KEY BODY")
            .await
            .unwrap();

        // `..` behind a symlinked component is the filesystem's to settle, as it
        // is the shell's: the write lands beside the link's target, not beside
        // the link.
        let through_link = own.join("link-to-keys/../through-link.txt");
        let resolved = resolve_write_target(
            &own,
            &through_link.to_string_lossy(),
            false,
            PathAccess::Unrestricted,
        )
        .await
        .expect("a `..` behind a symlinked component is settled by the filesystem");
        assert_eq!(
            resolved,
            outside.join("through-link.txt"),
            "the `..` is the link target's parent, not the link's"
        );

        // A link whose target does not exist yet is followed to that target: the
        // shell would create it through the link, so the write lands there and
        // the link itself is left alone.
        let dangling = own.join("not-yet.txt");
        symlink(outside.join("made-by-link.txt"), &dangling).unwrap();
        let resolved = resolve_write_target(
            &own,
            &dangling.to_string_lossy(),
            true,
            PathAccess::Unrestricted,
        )
        .await
        .expect("a dangling link is followed to the file it names");
        assert_eq!(resolved, outside.join("made-by-link.txt"));
        // The link is not what gets written: the file the link names is.
        tokio::fs::write(&resolved, "through the link")
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_link(&dangling)
                .await
                .expect("the link is still there"),
            outside.join("made-by-link.txt")
        );
        assert_eq!(
            tokio::fs::read_to_string(&dangling).await.unwrap(),
            "through the link",
            "reading the link reaches the file the write landed in"
        );

        // The same link aimed at a live store is refused, and the refusal
        // creates nothing anywhere on the way.
        let Some(root) = storage_root() else {
            return;
        };
        let store_dir = root.join("db");
        let linked_dir = own.join("link-to-store");
        symlink(&store_dir, &linked_dir).unwrap();
        let err = resolve_write_target(
            &own,
            &linked_dir.join("sub/core.db").to_string_lossy(),
            true,
            PathAccess::Unrestricted,
        )
        .await
        .expect_err("a live store is refused through a link too");
        assert!(err.to_string().contains("live databases"), "{err}");
        assert!(
            !store_dir.join("sub").exists(),
            "a refused write must not create directories"
        );

        // A link as the final component is the same question: what counts is
        // the file the write would land in.
        let file_link = own.join("store-link.db");
        symlink(store_dir.join("core.db"), &file_link).unwrap();
        let err = resolve_write_target(
            &own,
            &file_link.to_string_lossy(),
            false,
            PathAccess::Unrestricted,
        )
        .await
        .expect_err("a link to a live store is refused too");
        assert!(err.to_string().contains("live databases"), "{err}");

        drop((dir, outside));
    }

    // ── Path resolution tests ──────────────────────────────────────────

    /// Create a temporary workspace and return `(TempDir, canonical_ws_path)`.
    /// The `TempDir` guard must be held alive for the test duration.
    async fn test_workspace() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().expect("tempdir");
        let ws_raw = tmp.path().join("ws");
        tokio::fs::create_dir(&ws_raw).await.unwrap();
        let ws = tokio::fs::canonicalize(&ws_raw).await.unwrap();
        (tmp, ws)
    }

    #[tokio::test]
    async fn resolve_read_target_file_exists() {
        let (_tmp, ws) = test_workspace().await;
        let file_path = ws.join("existing.txt");
        tokio::fs::write(&file_path, "hello").await.unwrap();

        let result = resolve_read_target(&ws, "existing.txt", PathAccess::Allowlisted).await;
        assert!(
            result.is_ok(),
            "Should resolve existing file: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert_eq!(
            resolved,
            canonical_without_verbatim_prefix(&file_path),
            "should resolve to the canonical path"
        );
    }

    #[tokio::test]
    async fn resolve_read_target_existing_subdirectory_without_trailing_slash() {
        let (_tmp, ws) = test_workspace().await;
        let sub = ws.join("nested");
        tokio::fs::create_dir_all(&sub).await.unwrap();
        tokio::fs::write(sub.join("leaf.txt"), "hello")
            .await
            .unwrap();

        let result = resolve_read_target(&ws, "nested", PathAccess::Allowlisted).await;
        assert!(
            result.is_ok(),
            "Should resolve existing directory without trailing slash: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert_eq!(resolved, canonical_without_verbatim_prefix(&sub));
    }

    #[tokio::test]
    async fn resolve_read_target_file_not_found() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_read_target(&ws, "nonexistent.txt", PathAccess::Allowlisted).await;
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("File not found"),
            "Should report File not found: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_read_target_permission_denied() {
        use std::os::unix::fs::PermissionsExt;

        let (_tmp, ws) = test_workspace().await;
        let restricted_dir = ws.join("secret");
        tokio::fs::create_dir(&restricted_dir).await.unwrap();
        let file_path = restricted_dir.join("file.txt");
        tokio::fs::write(&file_path, "secret").await.unwrap();

        // Remove search permission from directory so canonicalize can't enter it
        std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o000)).unwrap();

        let result = resolve_read_target(&ws, "secret/file.txt", PathAccess::Allowlisted).await;

        // Restore permissions so TempDir can clean up
        let _ = std::fs::set_permissions(&restricted_dir, std::fs::Permissions::from_mode(0o755));

        assert!(result.is_err(), "Should fail with Permission denied");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Permission denied"),
            "Should mention Permission denied: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_read_target_symlink_resolution() {
        let (_tmp, ws) = test_workspace().await;
        let secret = ws.join("secret.txt");
        tokio::fs::write(&secret, "content").await.unwrap();
        let link = ws.join("link.txt");
        std::os::unix::fs::symlink(&secret, &link).unwrap();

        let result = resolve_read_target(&ws, "link.txt", PathAccess::Allowlisted).await;
        assert!(result.is_ok(), "Should resolve symlink: {:?}", result.err());

        let resolved = result.unwrap();
        assert_eq!(
            resolved,
            canonical_without_verbatim_prefix(&link),
            "should resolve to the canonical path"
        );
    }

    #[tokio::test]
    async fn resolve_read_target_extra_allowed_path() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = tmp.path().join("ws");
        tokio::fs::create_dir(&ws).await.unwrap();

        let spill_dir = std::env::temp_dir().join(".agent");
        tokio::fs::create_dir_all(&spill_dir).await.unwrap();
        let spill_file = spill_dir.join("spill_ab12.txt");
        tokio::fs::write(&spill_file, "spill content")
            .await
            .unwrap();
        let spill_str = spill_file.to_string_lossy().to_string();

        let result = resolve_read_target(&ws, &spill_str, PathAccess::Allowlisted).await;
        let _ = tokio::fs::remove_file(&spill_file).await;

        assert!(
            result.is_ok(),
            "Should allow extra read paths (e.g. /tmp): {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn resolve_read_target_workspace_only_rejects_extra_allowed_path() {
        let tmp = TempDir::new().expect("tempdir");
        let ws = tmp.path().join("ws");
        tokio::fs::create_dir(&ws).await.unwrap();

        let spill_dir = std::env::temp_dir().join(".agent");
        tokio::fs::create_dir_all(&spill_dir).await.unwrap();
        let spill_file = spill_dir.join("spill_ef56.txt");
        tokio::fs::write(&spill_file, "spill content")
            .await
            .unwrap();
        let spill_str = spill_file.to_string_lossy().to_string();

        // The general read still permits the extra-allowed spill path.
        let result = resolve_read_target(&ws, &spill_str, PathAccess::Allowlisted).await;
        assert!(
            result.is_ok(),
            "the general read should allow an extra-allowed path: {:?}",
            result.err()
        );

        // The workspace-only read rejects it — the only allowed paths are
        // in-workspace.
        let result = resolve_read_target(&ws, &spill_str, PathAccess::Workspace).await;
        assert!(
            result.is_err(),
            "the workspace-only read must reject an extra-allowed path"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("forbidden: cannot read") && err.contains("outside the workspace"),
            "the workspace-only rejection should cite its envelope: {err}"
        );

        let _ = tokio::fs::remove_file(&spill_file).await;
    }

    #[tokio::test]
    async fn resolve_read_target_non_canonicalized_workspace_root() {
        // When workspace_root is not canonicalized (e.g. /tmp vs /private/tmp on macOS),
        // reading via relative path should still succeed because resolve_tool_path_with_base
        // joins the (non-canonical) root with the relative path, and canonicalize resolves
        // the full path before the post-check.
        let tmp = TempDir::new().expect("tempdir");
        let ws_dir = tmp.path().join("ws");
        tokio::fs::create_dir(&ws_dir).await.unwrap();
        let file_path = ws_dir.join("hello.txt");
        tokio::fs::write(&file_path, "world").await.unwrap();

        // Use the non-canonicalized ws_dir as workspace_root.
        let result = resolve_read_target(&ws_dir, "hello.txt", PathAccess::Allowlisted).await;
        assert!(
            result.is_ok(),
            "Read should succeed even with non-canonicalized root: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        // The resolved path is canonical; verify the content is correct.
        let content = tokio::fs::read_to_string(&resolved).await.unwrap();
        assert_eq!(content, "world", "should read the correct file content");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_in_existing_dir() {
        let (_tmp, ws) = test_workspace().await;
        let subdir = ws.join("subdir");
        tokio::fs::create_dir(&subdir).await.unwrap();

        let result =
            resolve_write_target(&ws, "subdir/new_file.rs", false, PathAccess::Workspace).await;
        assert!(
            result.is_ok(),
            "Should resolve new file in existing dir: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(
            crate::util::is_within(&resolved, &ws),
            "Path should be within workspace: {resolved:?}"
        );
        assert_eq!(resolved.file_name().unwrap(), "new_file.rs");
        // The file should NOT exist yet
        assert!(!resolved.exists(), "File should not exist yet");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_new_dir_with_ensure_parent() {
        let (_tmp, ws) = test_workspace().await;

        let result =
            resolve_write_target(&ws, "a/b/c/new_file.rs", true, PathAccess::Workspace).await;
        assert!(
            result.is_ok(),
            "Should create parent directories: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(
            crate::util::is_within(&resolved, &ws),
            "Path should be within workspace: {resolved:?}"
        );
        assert_eq!(resolved.file_name().unwrap(), "new_file.rs");
        // Parent chain should exist
        assert!(
            ws.join("a/b/c").exists(),
            "Parent directories should be created"
        );
        // File should NOT exist yet
        assert!(!resolved.exists(), "File should not exist yet");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_new_dir_no_ensure_parent() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_write_target(
            &ws,
            "nonexistent_dir/new_file.rs",
            false,
            PathAccess::Workspace,
        )
        .await;
        assert!(result.is_err(), "Should fail when parent doesn't exist");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to resolve file path")
                || err.to_string().contains("No such file or directory"),
            "Should mention resolution failure: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_write_target_symlink_refusal() {
        let (_tmp, ws) = test_workspace().await;
        // Create a symlink at the file target location
        let link = ws.join("malicious_link.txt");
        std::os::unix::fs::symlink("/etc/passwd", &link).unwrap();

        let result =
            resolve_write_target(&ws, "malicious_link.txt", false, PathAccess::Workspace).await;
        assert!(result.is_err(), "Should refuse to write through symlink");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("it is a symlink"),
            "Error should mention the symlink refusal: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_write_target_outside_workspace_rejected() {
        let (_tmp, ws) = test_workspace().await;
        let outside = PathBuf::from("/tmp/outside_write_test.txt");

        let result = resolve_write_target(
            &ws,
            &outside.to_string_lossy(),
            false,
            PathAccess::Workspace,
        )
        .await;
        assert!(result.is_err(), "Should reject write outside workspace");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("forbidden: cannot write to")
                && err.to_string().contains("outside the workspace"),
            "Error should cite the write gate: {err}"
        );
    }

    // ── normalize_path tests ──────────────────────────────────────────────

    /// Path normalization (`/`, `.`, `..` resolution) is exercised via a
    /// data-driven table so the identity, dot-component removal, simple `..`,
    /// root-limit no-op, excess-`..` preservation for relative paths, and
    /// complex traversal rules are all covered in one place.
    #[expect(clippy::too_many_lines)]
    #[test]
    fn normalize_path_all_cases() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: &'static str,
        }

        let cases = [
            // Paths without `.` or `..` should be unchanged.
            Case {
                name: "identity_root",
                input: "/",
                expected: "/",
            },
            Case {
                name: "identity_tmp",
                input: "/tmp",
                expected: "/tmp",
            },
            Case {
                name: "identity_tmp_file",
                input: "/tmp/file.txt",
                expected: "/tmp/file.txt",
            },
            Case {
                name: "identity_relative",
                input: "relative/path",
                expected: "relative/path",
            },
            Case {
                name: "identity_single",
                input: "single",
                expected: "single",
            },
            // Dot components are dropped.
            Case {
                name: "dot_tmp_file",
                input: "/tmp/./file.txt",
                expected: "/tmp/file.txt",
            },
            Case {
                name: "dot_leading_tmp_file",
                input: "/./tmp/./file.txt",
                expected: "/tmp/file.txt",
            },
            Case {
                name: "dot_relative",
                input: "./relative/./path",
                expected: "relative/path",
            },
            Case {
                name: "dot_slash_root",
                input: "/.",
                expected: "/",
            },
            Case {
                name: "dot_only",
                input: ".",
                expected: "",
            },
            // Simple `..` resolution.
            Case {
                name: "dotdot_tmp_etc",
                input: "/tmp/../etc/passwd",
                expected: "/etc/passwd",
            },
            Case {
                name: "dotdot_deep_etc",
                input: "/tmp/foo/../../etc",
                expected: "/etc",
            },
            Case {
                name: "dotdot_tmp_root",
                input: "/tmp/..",
                expected: "/",
            },
            // Can't go above root — excess `..` after root are dropped.
            Case {
                name: "dotdot_above_root_tmp",
                input: "/../tmp",
                expected: "/tmp",
            },
            Case {
                name: "dotdot_root_file",
                input: "/tmp/../../tmp/file.txt",
                expected: "/tmp/file.txt",
            },
            Case {
                name: "dotdot_above_root_only",
                input: "/../../..",
                expected: "/",
            },
            // For relative paths, excess `..` are preserved as meaningful prefix.
            Case {
                name: "rel_excess_dotdot_foo",
                input: "../../foo",
                expected: "../../foo",
            },
            Case {
                name: "rel_excess_dotdot_only",
                input: "../../..",
                expected: "../../..",
            },
            Case {
                name: "rel_excess_dotdot_bar",
                input: "foo/../../bar",
                expected: "../bar",
            },
            Case {
                name: "rel_excess_dotdot_d",
                input: "a/b/c/../../d",
                expected: "a/d",
            },
            // Complex traversals, preserving the inline `->` transformation
            // illustrations for the expected result.
            // /a/b/c/../d/./e/../../f → /a/b/f
            Case {
                name: "complex_abs",
                input: "/a/b/c/../d/./e/../../f",
                expected: "/a/b/f",
            },
            // a/./b/./c/../d/../../e → a/e
            Case {
                name: "complex_rel",
                input: "a/./b/./c/../d/../../e",
                expected: "a/e",
            },
            // Relative path with excess `..`: a/b/../../../../c → ../../c
            //   (pop past 'a' into excess `..` preserved for relative paths)
            Case {
                name: "complex_rel_excess",
                input: "a/b/../../../../c",
                expected: "../../c",
            },
            // Empty and dot-only paths.
            Case {
                name: "empty_string",
                input: "",
                expected: "",
            },
            Case {
                name: "dot_only_empty",
                input: ".",
                expected: "",
            },
            Case {
                name: "dot_slash_empty",
                input: "/.",
                expected: "/",
            },
        ];

        for case in &cases {
            assert_eq!(
                normalize_path(Path::new(case.input)),
                Path::new(case.expected),
                "case: {}",
                case.name
            );
        }
    }

    // ── is_path_under_roots / allowed_temp_roots (traversal cluster) ──────

    /// Allow/reject semantics for [`is_path_under_roots`] against
    /// [`allowed_temp_roots`], covering path traversal, tilde expansion, clean
    /// temp paths, absolute non-temp paths, harmless dot components, and the
    /// root limit. The common-roots coverage test is kept separate.
    #[expect(clippy::too_many_lines)]
    #[test]
    fn is_path_under_allowed_temp_all_cases() {
        struct Case {
            name: &'static str,
            path: &'static str,
            allowed: bool,
        }

        let cases = [
            // Core traversal: /tmp/../etc/passwd starts with /tmp but normalizes to /etc/passwd
            Case {
                name: "traversal_tmp_etc",
                path: "/tmp/../etc/passwd",
                allowed: false,
            },
            Case {
                name: "traversal_deep_etc",
                path: "/tmp/../../../../etc/passwd",
                allowed: false,
            },
            // Deeper traversal: /tmp/foo/../../etc/passwd → /etc/passwd
            Case {
                name: "traversal_foo_etc",
                path: "/tmp/foo/../../etc/passwd",
                allowed: false,
            },
            Case {
                name: "traversal_dot_etc_shadow",
                path: "/tmp/./../etc/shadow",
                allowed: false,
            },
            // Traversal that stays within temp should still be allowed.
            // /tmp/../tmp/file.txt → /tmp/file.txt
            Case {
                name: "traversal_back_into_tmp",
                path: "/tmp/../tmp/file.txt",
                allowed: true,
            },
            // /tmp/foo/../../tmp/file.txt → /tmp/file.txt
            Case {
                name: "traversal_foo_back_tmp",
                path: "/tmp/foo/../../tmp/file.txt",
                allowed: true,
            },
            // /tmp/../../tmp/../tmp/bar → /tmp/bar
            Case {
                name: "traversal_deep_back_tmp",
                path: "/tmp/../../tmp/../tmp/bar",
                allowed: true,
            },
            // Tilde expansion + traversal: ~ expands to $HOME (e.g. /Users/username).
            // ~/../tmp/file.txt → /Users/username/../tmp/file.txt → /Users/tmp/file.txt
            //   NOT under /tmp → rejected
            Case {
                name: "tilde_traversal_tmp",
                path: "~/../tmp/file.txt",
                allowed: false,
            },
            // ~/../../tmp/file.txt → /Users/username/../../tmp/file.txt → /tmp/file.txt
            //   IS under /tmp → allowed
            Case {
                name: "tilde_traversal_tmp_allowed",
                path: "~/../../tmp/file.txt",
                allowed: true,
            },
            // ~/../etc/passwd → /Users/username/../etc/passwd → /Users/etc/passwd
            //   NOT under /tmp → rejected
            Case {
                name: "tilde_traversal_etc",
                path: "~/../etc/passwd",
                allowed: false,
            },
            // ~/../../etc/passwd → /Users/username/../../etc/passwd → /etc/passwd
            //   NOT under /tmp → rejected
            Case {
                name: "tilde_traversal_etc_deep",
                path: "~/../../etc/passwd",
                allowed: false,
            },
            // Ensure normal temp paths still work (no regression).
            Case {
                name: "clean_tmp_out",
                path: "/tmp/out.txt",
                allowed: true,
            },
            Case {
                name: "clean_private_tmp_out",
                path: "/private/tmp/out.txt",
                allowed: true,
            },
            Case {
                name: "clean_var_tmp_out",
                path: "/var/tmp/out.txt",
                allowed: true,
            },
            // Relative paths still rejected (can't start_with absolute roots).
            Case {
                name: "clean_relative_rejected",
                path: "relative.txt",
                allowed: false,
            },
            Case {
                name: "clean_dotdot_tmp_rejected",
                path: "../tmp/out.txt",
                allowed: false,
            },
            // Absolute paths outside temp are still rejected.
            Case {
                name: "non_temp_etc_passwd",
                path: "/etc/passwd",
                allowed: false,
            },
            Case {
                name: "non_temp_usr_bin",
                path: "/usr/bin/foo",
                allowed: false,
            },
            Case {
                name: "non_temp_var_log",
                path: "/var/log/system.log",
                allowed: false,
            },
            // Dot components in isolation should not affect the result.
            Case {
                name: "dot_harmless_tmp_file",
                path: "/tmp/./file.txt",
                allowed: true,
            },
            Case {
                name: "dot_harmless_multi",
                path: "/tmp/./././file.txt",
                allowed: true,
            },
            Case {
                name: "dot_harmless_foo",
                path: "/tmp/foo/./../file.txt",
                allowed: true,
            },
            Case {
                name: "dot_harmless_etc_rejected",
                path: "/tmp/./../etc/passwd",
                allowed: false,
            },
            // Paths that would normalize to just "/" or above root should be rejected.
            Case {
                name: "root_limit_tmp",
                path: "/tmp/../../../",
                allowed: false,
            },
            Case {
                name: "root_limit_absolute",
                path: "/../../../../../",
                allowed: false,
            },
        ];

        for case in &cases {
            assert_eq!(
                is_path_under_roots(Path::new(case.path), &allowed_temp_roots()),
                case.allowed,
                "case: {}",
                case.name
            );
        }
    }

    // ── shell_quote / contains_glob ─────────────────────────────────────

    #[test]
    fn shell_quoting_edge_cases() {
        // Simple path
        assert_eq!(shell_quote("/tmp/dir"), "'/tmp/dir'");
        // Path with spaces
        assert_eq!(shell_quote("/my dir/file"), "'/my dir/file'");
        // Path with single quote
        assert_eq!(shell_quote("/it's dir"), "'/it'\\''s dir'");
        // Path with dollar sign
        assert_eq!(shell_quote("/$dir"), "'/$dir'");
        // Path with backtick
        assert_eq!(shell_quote("/`dir`"), "'/`dir`'");
        // Path with backslash
        assert_eq!(shell_quote("/dir\\name"), "'/dir\\name'");
        // Empty string
        assert_eq!(shell_quote(""), "''");
        // Already quoted — just wraps
        assert_eq!(shell_quote("normal"), "'normal'");
    }

    #[test]
    fn contains_glob_detects_wildcards() {
        assert!(contains_glob("src/*.rs", true));
        assert!(contains_glob("lib?.rs", true));
        assert!(!contains_glob("src/main.rs", true));
    }
}

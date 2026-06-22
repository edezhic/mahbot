//! Path validation and resolution functions for tool operations.
//!
//! This module implements the path security boundary for all tool file operations.
//! It enforces workspace-scoped access, temp-file allowlists, dependency-cache
//! access for reading, spill-file detection, and symlink/symlink-escape protection.

use anyhow::Context;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Canonicalize the parent directory of `path` and join the original file name.
///
/// This is the common canonicalization strategy used by both
/// [`resolve_directory_read_fallback`] and [`resolve_write_target`]: the parent
/// directory is canonicalized (to resolve symlinks in the directory chain) while
/// the final file component is preserved as-is (the file itself may not exist
/// yet for write operations).
///
/// Returns an error if `path` has no parent or file_name component, or if
/// [`tokio::fs::canonicalize`] fails on the parent directory.
async fn canonicalize_parent_and_join(path: &Path) -> std::io::Result<PathBuf> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "no parent directory")
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "no file name"))?;
    let canon_parent = tokio::fs::canonicalize(parent).await?;
    Ok(canon_parent.join(name))
}

/// Fallback directory resolution when full-path [`canonicalize`] fails with
/// `NotFound` but the lexical path still exists as a directory.
///
/// Uses parent canonicalization + final component (same strategy as write-mode
/// path resolution) so existing directories are listable even when agents omit
/// a trailing `/` or when full-path canonicalization fails on edge-case paths.
pub(crate) async fn resolve_directory_read_fallback(full_path: &Path) -> Option<PathBuf> {
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
/// Path security is enforced via [`is_path_safe_for_workspace`] (pre- and
/// post-canonicalization). Extra read paths (spill files, dependency caches)
/// are **not** allowed for writes.
///
/// Additional security:
/// 1. Canonicalize the **parent** directory only — the file itself may not exist yet.
/// 2. Symlink check: if the target exists and is a symlink, refuse (unlike reading,
///    where `canonicalize` resolves through symlinks safely).
/// 3. If `ensure_parent` is `true`, creates parent directories before canonicalizing.
///
/// See [`resolve_read_target`] for the read-side counterpart.
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub(crate) async fn resolve_write_target(
    workspace_root: &Path,
    path: &str,
    ensure_parent: bool,
) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root);

    // Pre-canonicalization check — strict, no extra allowed paths
    if !is_path_safe_for_workspace(path, workspace_root) {
        anyhow::bail!("Path not allowed by security policy: {path}");
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

    // Re-extract canonicalized parent for the post-canonicalization security check.
    // The helper guarantees the result has a parent component.
    let resolved_parent = resolved_target.parent().unwrap();

    if !is_path_safe_for_workspace(&resolved_parent.to_string_lossy(), workspace_root) {
        anyhow::bail!(
            "Path not allowed by security policy: {}",
            resolved_parent.display()
        );
    }

    // Explicit symlink refusal (read resolves symlinks via canonicalize instead)
    if let Ok(meta) = tokio::fs::symlink_metadata(&resolved_target).await
        && meta.file_type().is_symlink()
    {
        anyhow::bail!(
            "Refusing to write through symlink: {}",
            resolved_target.display()
        );
    }

    Ok(resolved_target)
}

/// Resolve and validate a file path for read operations.
///
/// Path security is enforced by `check_path_read_allowed` (pre- and
/// post-canonicalization), permitting `EXTRA_READ_ALLOWED` paths (temp files,
/// dependency source directories) in addition to workspace-scoped paths.
///
/// Key differences from [`resolve_write_target`]:
/// - Canonicalizes the **full path**, not just the parent (file must exist).
/// - No `ensure_parent` parameter — parent creation is a write-only concept.
/// - No explicit symlink refusal — `tokio::fs::canonicalize` resolves symlinks,
///   so the post-canonicalization check catches escapes via the resolved path.
/// - Also allows `EXTRA_READ_ALLOWED` paths (e.g. /tmp files, dependency caches).
///
/// Returns `Ok(path)` on success, or an error message to propagate to the agent.
pub(crate) async fn resolve_read_target(
    workspace_root: &Path,
    path: &str,
) -> anyhow::Result<PathBuf> {
    let full_path = resolve_tool_path_with_base(path, workspace_root);

    // Pre-canonicalization check — allows EXTRA_READ_ALLOWED paths
    // (temp files, dependency source directories) outside the workspace
    check_path_read_allowed(path, workspace_root)?;

    // Canonicalize full path (file must exist). Resolves symlinks,
    // so the post-canonicalization check catches escapes.
    let resolved_path = match tokio::fs::canonicalize(&full_path).await {
        Ok(resolved) => resolved,
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

    check_path_read_allowed(&resolved_path.to_string_lossy(), workspace_root)?;

    Ok(resolved_path)
}

/// If the path starts with `~`, expand it to the user's home directory.
/// Otherwise returns the original path as-is.
///
/// This is used by path-check helpers that compare user-provided (pre-canonicalization)
/// paths against init-time expanded allowlist entries.
fn expand_tilde_for_path_check(path: &Path) -> Cow<'_, Path> {
    if path.to_str().is_some_and(|s| s.starts_with('~')) {
        Cow::Owned(crate::config::expand_tilde(&path.to_string_lossy()))
    } else {
        Cow::Borrowed(path)
    }
}

/// Check whether a path is under any of the given roots, after tilde expansion.
///
/// The path may contain a leading `~` (user-provided input before
/// canonicalization). In that case the `~` is expanded to the user's
/// home directory before comparing against the (already-expanded) roots.
fn is_path_under_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let check_path = expand_tilde_for_path_check(path);
    roots.iter().any(|root| check_path.starts_with(root))
}

/// Check whether `path` is under an [`EXTRA_READ_ALLOWED`] directory.
fn is_path_in_extra_allowed(path: &Path) -> bool {
    is_path_under_roots(path, &EXTRA_READ_ALLOWED)
}

/// Whether `path` looks like an OS temp/scratch directory (checked at read time).
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
fn is_os_temp_root(path: &Path) -> bool {
    let check_path = expand_tilde_for_path_check(path);

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

/// Whether `path` is a mahbot shell spill/raw log under an OS temp directory.
fn is_mahbot_spill_filename(name: &str) -> bool {
    if name.ends_with(".raw.log") {
        return true;
    }
    let Some(hex) = name
        .strip_prefix("spill_")
        .and_then(|s| s.strip_suffix(".txt"))
    else {
        return false;
    };
    hex.len() == 4 && hex.chars().all(|c| c.is_ascii_hexdigit())
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

/// Whether `path` is a mahbot spill file written under `$TEMP/.agent/`.
fn is_mahbot_spill_file(path: &Path) -> bool {
    if !is_mahbot_spill_shaped(path) {
        return false;
    }
    path.parent()
        .and_then(|p| p.parent())
        .is_some_and(is_os_temp_root)
}

/// Check that a path is allowed by the read-path security policy.
///
/// The path must be either within the workspace (via [`is_path_safe_for_workspace`])
/// or under one of the [`EXTRA_READ_ALLOWED`] directories (temp files,
/// dependency caches, SDK headers, etc.).
fn check_path_read_allowed(path: &str, workspace_root: &Path) -> anyhow::Result<()> {
    let path_buf = Path::new(path);
    if is_mahbot_spill_shaped(path_buf) && !is_mahbot_spill_file(path_buf) {
        anyhow::bail!("Path not allowed by security policy: {path}");
    }
    if !is_path_safe_for_workspace(path, workspace_root)
        && !is_path_in_extra_allowed(path_buf)
        && !is_mahbot_spill_file(path_buf)
    {
        anyhow::bail!("Path not allowed by security policy: {path}");
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
    for (xdg_subdir, env_var) in XDG_SUBDIR_TO_ENV {
        if let Some(suffix) = tilde_path
            .strip_prefix("~/")
            .and_then(|p| p.strip_prefix(xdg_subdir))
            && let Ok(xdg_dir) = std::env::var(env_var)
        {
            let xdg_dir = xdg_dir.trim_end_matches('/');
            return Some(format!("{xdg_dir}/{suffix}"));
        }
    }
    None
}

/// All paths from the ticket that should be allowed for reading.
/// Paths starting with `~` are expanded at init time. Paths that don't
/// exist on the current platform are harmless (they fail canonicalization
/// and just get added as-is, never matching any read request).
const EXTRA_ALLOWED_RAW_PATHS: &[&str] = &[
    // ── Rust (Cargo) ────────────────────────────────────────
    "~/.cargo/registry/src/",
    "~/.cargo/git/checkouts/",
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
    // ── Go ──────────────────────────────────────────────────
    "~/go/pkg/mod/",
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
    r"C:\ProgramData\chocolatey\lib\",
    r"C:\msys64\mingw64\include\",
    r"C:\msys64\ucrt64\include\",
    r"C:\msys64\clang64\include\",
    r"C:\msys64\usr\include\",
    // ── Windows SDK / MSVC ──────────────────────────────────
    r"C:\Program Files (x86)\Windows Kits\",
    r"C:\Program Files\Microsoft Visual Studio\",
    // ── Swift ───────────────────────────────────────────────
    "~/.swiftpm/",
    "~/Library/Developer/Xcode/DerivedData/",
    // ── Dart / Flutter ──────────────────────────────────────
    "~/.pub-cache/",
    // ── Elixir / Erlang ─────────────────────────────────────
    "~/.hex/",
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

/// Approved filesystem roots for scratch/temp files (single source of truth).
///
/// Used by read-path allowlists and read-only shell redirect / scratch-write policy.
static APPROVED_TEMP_ROOTS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = Vec::new();
    add_path_with_canonical(&mut dirs, std::env::temp_dir());
    add_path_with_canonical(&mut dirs, PathBuf::from("/tmp"));
    add_path_with_canonical(&mut dirs, PathBuf::from("/private/tmp"));
    add_path_with_canonical(&mut dirs, PathBuf::from("/var/tmp"));
    // Explicit spill directory (usually under `temp_dir()`; documents intent).
    add_path_with_canonical(&mut dirs, std::env::temp_dir().join(".agent"));
    dirs
});

/// Check whether `path` is under an approved temp/scratch root.
#[must_use]
pub(crate) fn is_path_under_allowed_temp(path: &Path) -> bool {
    is_path_under_roots(path, &APPROVED_TEMP_ROOTS)
}

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
/// [`crate::config::expand_tilde`]. If `$HOME` (and `$USERPROFILE` on Windows)
/// is unset, `~`-prefixed entries are skipped. Entries that follow XDG Base
/// Directory conventions (`~/.cache/`, `~/.local/share/`, `~/.local/state/`,
/// `~/.config/`) also generate variants using the corresponding `$XDG_*`
/// environment variable when set.
static EXTRA_READ_ALLOWED: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs = APPROVED_TEMP_ROOTS.clone();

    // Dependency source directories (cross-platform)
    for raw_path in EXTRA_ALLOWED_RAW_PATHS {
        if raw_path.starts_with('~') {
            let expanded = crate::config::expand_tilde(raw_path);
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

    dirs
});

// ── Path validation ──────────────────────────

/// Check whether `path` is safe to access within the given `workspace_root`.
///
/// This is the central security gate for all file-path operations: it performs
/// a purely lexical check — no filesystem I/O — so it cannot detect symlink-based
/// escapes. The caller is responsible for that post-canonicalization validation
/// (see [`resolve_read_target`] and [`resolve_write_target`]).
#[must_use]
pub(crate) fn is_path_safe_for_workspace(path: &str, workspace_root: &Path) -> bool {
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
    let lower = path.to_lowercase();
    if lower.contains("..%2f") || lower.contains("%2f..") {
        return false;
    }
    if path.starts_with('~') && path != "~" && !path.starts_with("~/") {
        return false;
    }
    let expanded_path = crate::config::expand_tilde(path);
    if expanded_path.is_absolute() {
        // Lexical prefix check — no sync I/O.
        //
        // Safety: workspace_root is pre-canonicalized at workspace registration
        // (see canonicalize_workspace_path in workspace.rs). The lexical check
        // may reject absolute paths whose prefix is a symlink into the workspace
        // (e.g. /tmp/… when the real workspace is /private/tmp/… on macOS), but
        // this is harmless: agents use relative paths, and the post-canonicalization
        // checks in resolve_read_target / resolve_write_target catch any symlink
        // escapes that would bypass this pre-check.
        expanded_path.starts_with(workspace_root)
    } else {
        // Relative path without parent-dir components — always safe
        true
    }
}

/// Resolve a user path segment against `workspace_root`.
#[must_use]
pub(crate) fn resolve_tool_path_with_base(path: &str, workspace_root: &Path) -> PathBuf {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "~" {
        return workspace_root.to_path_buf();
    }
    let expanded = crate::config::expand_tilde(trimmed);
    if expanded.is_absolute() {
        return expanded;
    }
    workspace_root.join(expanded)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Path validation tests ──────────────────────────────────────────

    #[test]
    fn path_traversal_edge_cases() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("../etc/passwd", base));
        assert!(!is_path_safe_for_workspace("foo/../etc/passwd", base));
        assert!(is_path_safe_for_workspace("my..file.txt", base));
        assert!(!is_path_safe_for_workspace(
            "foo/..%2f..%2fetc/passwd",
            base
        ));
    }

    #[test]
    fn path_blocked_system_and_sensitive() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("file\0.txt", base));
        assert!(!is_path_safe_for_workspace(
            "/proc/self/root/etc/passwd",
            base
        ));
        assert!(!is_path_safe_for_workspace("/var/run/docker.sock", base));
        assert!(!is_path_safe_for_workspace("~/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("~/.gnupg/secring.gpg", base));
        assert!(!is_path_safe_for_workspace("~root/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("~nobody", base));
    }

    #[test]
    fn checklist_path_blocking() {
        let base = Path::new(".");
        assert!(!is_path_safe_for_workspace("/", base));
        assert!(!is_path_safe_for_workspace("/anything", base));
        assert!(!is_path_safe_for_workspace("/tmp", base));
        assert!(!is_path_safe_for_workspace("/var/log", base));
        // Leading whitespace bypasses (all three variants)
        assert!(!is_path_safe_for_workspace("  /etc/passwd", base));
        assert!(!is_path_safe_for_workspace("\t/etc/passwd", base));
        assert!(!is_path_safe_for_workspace("  ~root/.ssh/id_rsa", base));
        assert!(!is_path_safe_for_workspace("  ../foo", base));
        // Whitespace-only paths are treated as empty (relative, safe)
        assert!(is_path_safe_for_workspace("  ", base));

        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace(
            workspace.join("test.txt").to_str().unwrap(),
            &workspace
        ));
        assert!(is_path_safe_for_workspace("relative.txt", &workspace));
    }

    #[test]
    fn bare_tilde_is_allowed_as_workspace_root_shorthand() {
        // Bare ~ is shorthand for the workspace root — is_path_safe_for_workspace
        // must accept it, matching resolve_tool_path_with_base's behaviour.
        // Write operations are still protected by the post-canonicalization
        // parent check in resolve_write_target (the parent of workspace root
        // is outside the workspace).
        let base = Path::new(".");
        // Pathological: bare tilde with no workspace context
        assert!(is_path_safe_for_workspace("~", base));
        // Bare tilde with leading/trailing whitespace (trimmed before check)
        assert!(is_path_safe_for_workspace("  ~", base));
        assert!(is_path_safe_for_workspace("~  ", base));
        assert!(is_path_safe_for_workspace("  ~  ", base));

        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace("~", &workspace));

        // ~/… still correctly resolved relative to home and blocked if outside
        // workspace (sensitive files like .ssh should never pass)
        assert!(!is_path_safe_for_workspace("~/.ssh/id_rsa", &workspace));
        assert!(!is_path_safe_for_workspace(
            "~/.gnupg/secring.gpg",
            &workspace
        ));
    }

    #[test]
    fn path_allows_relative_and_blocks_absolute() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(is_path_safe_for_workspace("src/main.rs", &workspace));
        assert!(is_path_safe_for_workspace(
            "deep/nested/dir/file.txt",
            &workspace
        ));
        assert!(is_path_safe_for_workspace(".gitignore", &workspace));
        assert!(is_path_safe_for_workspace(".env", &workspace));
        assert!(is_path_safe_for_workspace("", &workspace));
        assert!(!is_path_safe_for_workspace("../etc/passwd", &workspace));
        assert!(!is_path_safe_for_workspace(
            "../../root/.ssh/id_rsa",
            &workspace
        ));
        assert!(!is_path_safe_for_workspace(
            "foo/../../../etc/shadow",
            &workspace
        ));
        assert!(!is_path_safe_for_workspace("..", &workspace));
    }

    // ── is_path_under_allowed_temp tests ────────────────────────────────

    #[test]
    fn is_path_under_allowed_temp_covers_common_roots() {
        let temp = std::env::temp_dir();
        let spill = temp.join(".agent/spill_test.txt");
        assert!(is_path_under_allowed_temp(&temp.join("scratch.txt")));
        assert!(is_path_under_allowed_temp(&spill));
        assert!(is_path_under_allowed_temp(Path::new("/tmp/out.txt")));
        assert!(is_path_under_allowed_temp(Path::new("/var/tmp/out.txt")));
        assert!(!is_path_under_allowed_temp(Path::new("relative.txt")));
        assert!(!is_path_under_allowed_temp(Path::new("/etc/passwd")));
    }

    // ── check_path_read_allowed tests ──────────────────────────────────

    #[test]
    fn check_path_read_allowed_var_tmp() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();
        assert!(
            check_path_read_allowed("/var/tmp/mahbot-test.txt", &workspace).is_ok(),
            "/var/tmp should be readable via EXTRA_READ_ALLOWED"
        );
    }

    #[test]
    fn is_mahbot_spill_file_allows_temp_agent_spills() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        let spill = std::env::temp_dir().join(".agent/spill_ab12.txt");
        assert!(
            check_path_read_allowed(&spill.to_string_lossy(), &workspace).is_ok(),
            "spill under current temp_dir should be allowed: {}",
            spill.display()
        );

        #[cfg(unix)]
        {
            let mac_spill = PathBuf::from("/var/folders/xx/yy/T/.agent/spill_cd34.txt");
            assert!(
                check_path_read_allowed(&mac_spill.to_string_lossy(), &workspace).is_ok(),
                "macOS-shaped spill path should be allowed"
            );
        }

        let raw_log = std::env::temp_dir().join(".agent/12345_cargo_check.raw.log");
        assert!(
            check_path_read_allowed(&raw_log.to_string_lossy(), &workspace).is_ok(),
            "raw.log spill should be allowed"
        );
    }

    #[test]
    fn is_mahbot_spill_file_rejects_workspace_and_system_paths() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        let in_workspace = workspace.join(".agent/spill_ab12.txt");
        assert!(
            check_path_read_allowed(&in_workspace.to_string_lossy(), &workspace).is_err(),
            "workspace .agent spill should be rejected: {}",
            in_workspace.display()
        );

        #[cfg(unix)]
        {
            let outside = PathBuf::from("/usr/local/.agent/spill_ab12.txt");
            assert!(
                check_path_read_allowed(&outside.to_string_lossy(), &workspace).is_err(),
                "non-temp spill-shaped path should be rejected"
            );
        }

        assert!(
            check_path_read_allowed("/etc/passwd", &workspace).is_err(),
            "/etc/passwd should be rejected"
        );
    }

    // ── EXTRA_READ_ALLOWED tests ──────────────────────────────────────

    /// Init doesn't panic even with many non-existent paths (e.g. Windows
    /// paths on macOS, or missing optional toolchains).
    #[test]
    fn extra_allowed_init_does_not_panic() {
        // Force initialization of the LazyLock.
        let dirs = &*EXTRA_READ_ALLOWED;
        // The temp dirs + all dependency paths should be present.
        // On a typical dev machine, at least a few entries should exist.
        assert!(!dirs.is_empty(), "EXTRA_READ_ALLOWED should not be empty");
    }

    /// No literal `~` path is ever stored in the allowlist — all `~`-prefixed
    /// entries are expanded at init time (or skipped if `$HOME` is unset).
    #[test]
    fn extra_allowed_no_literal_tilde() {
        let dirs = &*EXTRA_READ_ALLOWED;
        for dir in dirs {
            let s = dir.to_string_lossy();
            assert!(
                !s.starts_with('~'),
                "Literal tilde path should never be stored: {s}"
            );
        }
    }

    /// `is_path_in_extra_allowed` matches `~`-prefixed user input against
    /// the expanded allowlist entries. This simulates the pre-canonicalization
    /// check in `resolve_read_target`.
    #[test]
    fn extra_allowed_tilde_input_matches() {
        // Use a known-expanded path from the allowlist to construct a ~ variant
        if let Ok(home) = std::env::var("HOME") {
            // ~/.cargo/registry/src/ should be in the allowlist (expanded to $HOME/.cargo/registry/src/)
            let tilde_input = "~/.cargo/registry/src/some-crate/src/lib.rs";
            assert!(
                is_path_in_extra_allowed(Path::new(tilde_input)),
                "~-prefixed path should match expanded allowlist entry"
            );

            // Verify the expanded path also matches (post-canonicalization)
            let expanded = PathBuf::from(&home).join(".cargo/registry/src/some-crate/src/lib.rs");
            assert!(
                is_path_in_extra_allowed(&expanded),
                "Expanded path should match allowlist entry"
            );
        }
    }

    /// Verify prefix-matching semantics — a file inside a cargo registry
    /// should match, while a sibling of the allowlist root should not.
    #[test]
    fn extra_allowed_prefix_matching() {
        // File inside an allowed dependency tree should match
        assert!(
            is_path_in_extra_allowed(Path::new(
                "/usr/local/lib/python3.12/site-packages/requests/models.py"
            )),
            "File under /usr/local/lib/ should match"
        );
        // Sibling of an allowed root should NOT match (prefix strictness)
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
            check_path_read_allowed(&temp_str, &workspace).is_ok(),
            "Temp file should be allowed for read"
        );

        // Dependency path
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                check_path_read_allowed(&dep_path, &workspace).is_ok(),
                "Dependency path should be allowed for read"
            );

            // ~-prefixed version (pre-canonicalization check)
            let tilde_input = "~/.cargo/registry/src/crate-0.1.0/src/lib.rs";
            assert!(
                check_path_read_allowed(tilde_input, &workspace).is_ok(),
                "~-prefixed dependency path should be allowed for read"
            );
        }
    }

    /// `is_path_safe_for_workspace` blocks paths outside the workspace even
    /// when they're in [`EXTRA_READ_ALLOWED`] — the extra-read bypass is
    /// read-only and must not affect write-path checks.
    #[test]
    fn is_path_safe_for_workspace_blocks_extra_dependency_paths() {
        let tmp = TempDir::new().expect("tempdir");
        let workspace = tmp.path().to_path_buf();

        // Temp dir spill files — should be blocked by is_path_safe_for_workspace
        let temp_file = std::env::temp_dir().join("test-spill.txt");
        let temp_str = temp_file.to_string_lossy().to_string();
        assert!(
            !is_path_safe_for_workspace(&temp_str, &workspace),
            "Temp file should be blocked by base check"
        );

        // Dependency path (if $HOME is set)
        if let Ok(home) = std::env::var("HOME") {
            let dep_path = format!("{home}/.cargo/registry/src/crate-0.1.0/src/lib.rs");
            assert!(
                !is_path_safe_for_workspace(&dep_path, &workspace),
                "Dependency path should be blocked by base check"
            );
        }
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

        let result = resolve_read_target(&ws, "existing.txt").await;
        assert!(
            result.is_ok(),
            "Should resolve existing file: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        let canonical = tokio::fs::canonicalize(&file_path).await.unwrap();
        assert_eq!(resolved, canonical, "should resolve to the canonical path");
    }

    #[tokio::test]
    async fn resolve_read_target_existing_subdirectory_without_trailing_slash() {
        let (_tmp, ws) = test_workspace().await;
        let sub = ws.join("nested");
        tokio::fs::create_dir_all(&sub).await.unwrap();
        tokio::fs::write(sub.join("leaf.txt"), "hello")
            .await
            .unwrap();

        let result = resolve_read_target(&ws, "nested").await;
        assert!(
            result.is_ok(),
            "Should resolve existing directory without trailing slash: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        let canonical = tokio::fs::canonicalize(&sub).await.unwrap();
        assert_eq!(resolved, canonical);
    }

    #[tokio::test]
    async fn resolve_read_target_file_not_found() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_read_target(&ws, "nonexistent.txt").await;
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

        let result = resolve_read_target(&ws, "secret/file.txt").await;

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

        let result = resolve_read_target(&ws, "link.txt").await;
        assert!(result.is_ok(), "Should resolve symlink: {:?}", result.err());

        let resolved = result.unwrap();
        let canonical = tokio::fs::canonicalize(&link).await.unwrap();
        assert_eq!(resolved, canonical, "should resolve to the canonical path");
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

        let result = resolve_read_target(&ws, &spill_str).await;
        let _ = tokio::fs::remove_file(&spill_file).await;

        assert!(
            result.is_ok(),
            "Should allow extra read paths (e.g. /tmp): {:?}",
            result.err()
        );
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
        let result = resolve_read_target(&ws_dir, "hello.txt").await;
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

        let result = resolve_write_target(&ws, "subdir/new_file.rs", false).await;
        assert!(
            result.is_ok(),
            "Should resolve new file in existing dir: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(resolved.starts_with(&ws), "Path should be within workspace");
        assert_eq!(resolved.file_name().unwrap(), "new_file.rs");
        // The file should NOT exist yet
        assert!(!resolved.exists(), "File should not exist yet");
    }

    #[tokio::test]
    async fn resolve_write_target_new_file_new_dir_with_ensure_parent() {
        let (_tmp, ws) = test_workspace().await;

        let result = resolve_write_target(&ws, "a/b/c/new_file.rs", true).await;
        assert!(
            result.is_ok(),
            "Should create parent directories: {:?}",
            result.err()
        );
        let resolved = result.unwrap();
        assert!(resolved.starts_with(&ws), "Path should be within workspace");
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

        let result = resolve_write_target(&ws, "nonexistent_dir/new_file.rs", false).await;
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

        let result = resolve_write_target(&ws, "malicious_link.txt", false).await;
        assert!(result.is_err(), "Should refuse to write through symlink");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("symlink"),
            "Error should mention symlink: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_write_target_outside_workspace_rejected() {
        let (_tmp, ws) = test_workspace().await;
        let outside = PathBuf::from("/tmp/outside_write_test.txt");

        let result = resolve_write_target(&ws, &outside.to_string_lossy(), false).await;
        assert!(result.is_err(), "Should reject write outside workspace");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Path not allowed"),
            "Error should mention security policy: {err}"
        );
    }
}

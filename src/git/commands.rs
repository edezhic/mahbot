//! Git subprocess wrappers — run git commands and parse porcelain/numstat output.
//!
//! Provides `run_git_*` async functions that operate on a repository path
//! and return results. For diff output parsing, see [`crate::git::diff_parse`].

use anyhow::Context;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tracing::warn;

use crate::tools::shell::{internal_env_pairs, program_command};
use crate::util::unquote_c_style;

/// Result of a successful `git commit` — the full hash and line stats.
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Full 40-character commit SHA from `git rev-parse HEAD`.
    pub hash: String,
    /// Total lines added across all changed files.
    pub lines_added: i64,
    /// Total lines removed across all changed files.
    pub lines_removed: i64,
}

impl CommitInfo {
    /// Return the first 7 characters of the commit hash, or the full hash
    /// if it's shorter than 7 characters.
    #[must_use]
    pub fn short_hash(&self) -> &str {
        self.hash.get(..7).unwrap_or(&self.hash)
    }
}

/// Maximum size (1 MiB) for reading untracked files in [`run_git_worktree_snapshot`]
/// and the diff page's untracked-file display.
///
/// Larger untracked files are surfaced via [`DiffStats::huge_binary_file_count`]
/// in the footer rather than line-counted; the diff page still shows a
/// placeholder instead of reading them.
pub(crate) const MAX_UNTRACKED_SIZE: u64 = 1024 * 1024;

/// Cap on the total number of untracked files classified (line-counted or
/// oversized/binary-detected) per [`run_git_worktree_snapshot`] refresh.
/// Files beyond the cap are counted in [`DiffStats::omitted_file_count`].
const MAX_UNTRACKED_FILES_COUNTED: usize = 500;
/// Cap on the total bytes read from untracked files per
/// [`run_git_worktree_snapshot`] refresh.
const MAX_UNTRACKED_BYTES_READ: u64 = 10 * 1024 * 1024;

/// Classification of an untracked file read for diff display / line counting.
///
/// Ordering is load-bearing: size is checked from metadata before reading,
/// then null-byte detection, then UTF-8 validity. Unreadable paths are skipped
/// entirely — only `TooLarge` and `Binary` produce GUI placeholders.
pub(crate) enum UntrackedFileRead {
    Text(String),
    TooLarge(u64),
    Binary,
    Skip,
}

/// Classify an in-memory untracked-file buffer whose size already passed the
/// [`MAX_UNTRACKED_SIZE`] gate. Returns `None` for binary (a null byte, or
/// invalid UTF-8 without a null byte), or the UTF-8 text otherwise. The binary
/// rule lives here so the snapshot and the diff window cannot disagree about it.
fn classify_untracked_content(content: Vec<u8>) -> Option<String> {
    if content.contains(&0) {
        return None;
    }
    String::from_utf8(content).ok()
}

/// Read and classify an untracked file whose metadata the caller already holds.
/// [`read_untracked_file`] fetches it; the snapshot passes the one it already
/// read for its byte budget, so neither stats twice.
async fn read_untracked_file_with_meta(path: &Path, meta: &std::fs::Metadata) -> UntrackedFileRead {
    if !meta.is_file() {
        return UntrackedFileRead::Skip;
    }
    if meta.len() > MAX_UNTRACKED_SIZE {
        return UntrackedFileRead::TooLarge(meta.len());
    }
    let Ok(content) = tokio::fs::read(path).await else {
        return UntrackedFileRead::Skip;
    };
    // Invalid UTF-8 without a null byte is still binary — never Skip, or the
    // GUI would silently drop the file instead of showing a placeholder.
    classify_untracked_content(content).map_or(UntrackedFileRead::Binary, UntrackedFileRead::Text)
}

/// Read and classify one untracked file for the diff window: fetch its
/// metadata, then delegate to [`read_untracked_file_with_meta`] — the shared
/// regular-file, size-gate, read and classification core that the snapshot's
/// line counting also uses.
pub(crate) async fn read_untracked_file(path: &Path) -> UntrackedFileRead {
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return UntrackedFileRead::Skip;
    };
    read_untracked_file_with_meta(path, &meta).await
}

/// Whether to discard changes in a single file or an entire directory tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardTarget {
    File,
    /// Recursively discard all changes within a directory (and its subdirectories).
    Directory,
}

/// Check if a directory contains a `.git` entry (file for worktrees, directory otherwise).
#[must_use]
pub fn is_git_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

/// Run `git show -m <hash> --no-color --find-renames --format=` — the diff a
/// historical commit introduced.
///
/// `-m` splits merge commits into per-parent diffs, producing standard unified
/// diff blocks that `parse_git_diff` can handle. This may produce duplicate file
/// paths in the tree sidebar (one `diff --git` block per parent), which is
/// expected — each parent comparison is a different diff. `--format=`
/// suppresses the commit log header.
pub(crate) async fn run_git_show_diff(repo_path: &Path, hash: &str) -> anyhow::Result<String> {
    run_git_command(
        repo_path,
        &[
            "show",
            "-m",
            hash,
            "--no-color",
            "--find-renames",
            "--format=",
        ],
    )
    .await
}

/// Run `git diff <base> --no-color --find-renames` against the working tree.
///
/// `base` is `HEAD`, or this repository's empty tree on an unborn HEAD (see
/// [`run_git_empty_tree`]). `--find-renames` is explicit so rename detection
/// matches [`run_git_worktree_numstat`] regardless of repository config.
pub(crate) async fn run_git_worktree_diff(
    repo_path: &Path,
    unborn_head: bool,
) -> anyhow::Result<String> {
    let base = if unborn_head {
        run_git_empty_tree(repo_path).await?
    } else {
        "HEAD".to_string()
    };
    run_git_command(repo_path, &["diff", &base, "--no-color", "--find-renames"]).await
}

/// Run `git status --porcelain --untracked-files=normal` and return the output.
///
/// The untracked mode is pinned because it is otherwise the owner's
/// `status.showUntrackedFiles` (global config is still read; only
/// `safe.directory` is injected). With it set to `no`, this reader — the
/// sanitation clean-tree check and its file listing, and the reviewed-base dirty
/// guard — would call a tree whose only changes are untracked files clean, and
/// sanitation would move the ticket to Done without committing its new files.
/// `normal` is git's own mode when `-u` is absent, so a default configuration
/// sees no change, and it keeps the dirty checks cheap. Git's promise that the
/// porcelain format is unchanged across Git versions and regardless of user
/// configuration is about the format, not about which files are selected, so the
/// flag is not redundant.
///
/// The plain, newline-delimited form is deliberate: the same output feeds the
/// sanitation file listing and [`has_unstaged_changes`], whose byte-level
/// worktree column test must keep counting an entry whose path does not parse.
/// Under `-z` the whole status is one NUL-delimited stream — a single "line" to
/// that test, which would then read only the first record and could report a
/// dirty tree as clean. Quoting and the ` -> ` delimiter the parsers handle are
/// the price of that gate.
pub async fn run_git_status(repo_path: &Path) -> anyhow::Result<String> {
    run_git_command(
        repo_path,
        &["status", "--porcelain", "--untracked-files=normal"],
    )
    .await
}

/// Run `git status --porcelain --untracked-files=all`.
///
/// `-uall` lists individual files inside untracked directories instead of the
/// collapsed `dir/` entry, so one status reading names every untracked file the
/// footer counts and the diff window displays. Paths are repo-root-relative.
///
/// This is the snapshot's one unbounded read: git walks every untracked
/// directory, and every returned path is kept. The complete file set is the
/// contract — a collapsed directory cannot be expanded without losing files —
/// while the per-file content reads are capped by the `MAX_UNTRACKED_*` limits.
async fn run_git_status_all(repo_path: &Path) -> anyhow::Result<String> {
    run_git_command(
        repo_path,
        &["status", "--porcelain", "--untracked-files=all"],
    )
    .await
}

/// Run `git show HEAD:<path>` (when `commit_ref` is `None`)
/// or `git show <commit_ref>:<path>` (when `Some`) and return the file content.
///
/// Returns `None` if the file does not exist at that ref (new/untracked files,
/// or root-commit `~1` which has no parent), or if any git error occurs.
///
/// **`~1` parent refs**: The caller constructs the parent hash. To get the parent
/// version, pass `commit_ref = Some(&format!("{hash}~1"))`.
pub async fn run_git_show(
    repo_path: &Path,
    file_path: &str,
    commit_ref: Option<&str>,
) -> Option<String> {
    let show_arg = if let Some(hash) = commit_ref {
        format!("{hash}:{file_path}")
    } else {
        format!("HEAD:{file_path}")
    };
    run_git_command(repo_path, &["show", &show_arg]).await.ok()
}

/// Resolve the repository top-level directory (the `.git`-containing ancestor).
///
/// `repo_path` may be the repo root or a subdirectory of it. Walks upward to
/// the first ancestor that contains a `.git` entry — a directory for normal
/// repos, or a *file* for linked worktrees/submodules, hence [`is_git_repo`]
/// (which uses `exists()`, not `is_dir()`).
///
/// Deliberately does NOT use `git rev-parse --show-toplevel`: repo discovery
/// itself runs the dubious-ownership check this exists to satisfy, so it would
/// recurse into the same failure (circular). Canonicalizes the resolved path
/// where possible so the value matches git's own realpath-normalized discovery,
/// and drops the Windows verbatim prefix for the same reason — the value is
/// handed to git ([`inject_safe_directory`]) and must be spelled the way git
/// spells it, not the way `std::fs::canonicalize` does on Windows.
pub(crate) fn resolve_git_top_level(repo_path: &Path) -> Option<PathBuf> {
    let dir = repo_path.ancestors().find(|dir| is_git_repo(dir))?;
    Some(crate::util::strip_verbatim_prefix(
        &std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()),
    ))
}

/// Inject the repo top-level into git's per-process config as `safe.directory`.
///
/// Uses the process-scoped `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_0`/
/// `GIT_CONFIG_VALUE_0` env form — never writes the user's global gitconfig and
/// never uses `*` (which would fully opt out of the CVE ownership guard). The
/// value is scoped to the single resolved repo top-level. On git < 2.38 the env
/// form is silently ignored by git, which is acceptable for the supported
/// target as long as the repo is owned by the running user.
fn inject_safe_directory(cmd: &mut std::process::Command, repo_path: &Path) {
    let Some(top_level) = resolve_git_top_level(repo_path) else {
        return;
    };
    cmd.env("GIT_CONFIG_COUNT", "1");
    cmd.env("GIT_CONFIG_KEY_0", "safe.directory");
    cmd.env("GIT_CONFIG_VALUE_0", top_level.as_path());
}

/// Create a [`std::process::Command`] for `git` with the product's own internal
/// environment — the shared setup of every git subprocess in this module.
///
/// The subprocess environment is cleared and re-populated with the reduced
/// variable set the product uses for its own work (see [`internal_env_pairs`])
/// to prevent credential leakage (CWE-200). `LC_ALL=C` is set for consistent
/// locale behavior across all git invocations. [`git_command`] wraps this for
/// async callers; [`tracked_paths`] runs it directly because it answers in the
/// middle of a blocking directory walk.
///
/// `repo_root` is the workspace/`current_dir` path for repo-bound invocations;
/// when present, the resolved repo top-level is injected as `safe.directory`
/// (see [`inject_safe_directory`]) so git works on repos owned by a different
/// UID. Pass `None` for repo-agnostic invocations (e.g. [`git_is_installed`]),
/// which must not receive a repo-path value.
///
/// Callers should add further configuration (args, current_dir, stdio, etc.)
/// and then spawn or execute the command.
fn git_std_command(repo_root: Option<&Path>) -> std::process::Command {
    // `program_command` applies the pairs it names `git` under. No directory is
    // passed: the internal list is absolute, so the `current_dir` callers set
    // afterwards cannot change which program it resolves to.
    let mut cmd = program_command("git", &internal_env_pairs(), None);
    cmd.env("LC_ALL", "C");
    if let Some(path) = repo_root {
        inject_safe_directory(&mut cmd, path);
    }
    cmd
}

/// Create a [`tokio::process::Command`] for `git` — the async form of
/// [`git_std_command`], and the only entry point for production async git
/// subprocess creation in this module.
fn git_command(repo_root: Option<&Path>) -> tokio::process::Command {
    git_std_command(repo_root).into()
}

/// Every UTF-8 path `repo`'s index holds, repository-relative and `/`-separated as
/// git spells them — git's own answer to "is this path tracked?", which no ignore
/// rule overrides. An entry whose name is not UTF-8 is left out, as a walked path
/// that is not UTF-8 never reaches the snapshot either.
///
/// `None` when git cannot answer: not installed, not a repository, an unreadable
/// index, or a failed run. A caller reads that as "git cannot say", never as an
/// error — the paths the rules would hide then stay visible. Synchronous on purpose
/// — its only caller asks the question in the middle of a directory walk on the
/// blocking pool — and `GIT_OPTIONAL_LOCKS=0` keeps the read from taking the index
/// lock, so the inspected repository is left untouched.
pub(crate) fn tracked_paths(repo: &Path) -> Option<Vec<String>> {
    let mut cmd = git_std_command(Some(repo));
    cmd.current_dir(repo)
        // `core.fsmonitor` would run a command this repository's own config
        // names; a passive read of the index never needs that.
        .args(["-c", "core.fsmonitor=false", "ls-files", "-z", "--cached"])
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stderr(Stdio::null());
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        output
            .stdout
            .split(|&byte| byte == 0)
            .filter_map(|path| std::str::from_utf8(path).ok())
            .filter(|path| !path.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// Run a git command without any interpretation of the exit code.
///
/// Returns the raw [`std::process::Output`] so each caller can interpret the
/// exit status as appropriate; [`run_git_command_env`] is the checked-output
/// sibling.
///
/// **The product's own internal environment**: The subprocess environment is
/// cleared and re-populated with the reduced variable set the product uses for
/// its own work (see [`internal_env_pairs`] for details). This prevents leaking
/// API keys and other secrets into child processes (CWE-200), but it also means
/// variables like `SSH_AUTH_SOCK` and `GIT_SSH_COMMAND` are **not**
/// inherited. That is git's own containment and nothing else's: an agent's
/// shell command is handed the owner's environment (see
/// [`crate::shell_env`]), while this internal work keeps the reduced one. Use
/// SSH config (`~/.ssh/config`) for SSH-based git remotes rather than
/// environment variables. ([`run_git_network`] is the one exception: it
/// restores `SSH_AUTH_SOCK` so SSH-agent remotes still work.)
pub(crate) async fn run_git_output(
    repo_path: &Path,
    args: &[&str],
) -> anyhow::Result<std::process::Output> {
    run_git_output_env(repo_path, &[], args).await
}

/// [`run_git_output`] with extra environment variables — used to point git at a
/// throwaway index file instead of the repository's own.
async fn run_git_output_env(
    repo_path: &Path,
    env: &[(&str, &OsStr)],
    args: &[&str],
) -> anyhow::Result<std::process::Output> {
    let mut cmd = git_command(Some(repo_path));
    for (name, value) in env {
        cmd.env(name, value);
    }
    cmd.args(args).current_dir(repo_path);
    // Tokio's `output()` inherits stdin (unlike std's, which nulls it), and no
    // git command here reads stdin; `hash-object --stdin` needs EOF, not a
    // terminal. Null it so a command can never wait on the parent's input.
    cmd.stdin(Stdio::null());
    cmd.output()
        .await
        .with_context(|| format!("Failed to run git {}", args.join(" ")))
}

/// Run a git command and return stdout as string on success.
///
/// Returns an error if git exits with a non-zero status.
pub async fn run_git_command(repo_path: &Path, args: &[&str]) -> anyhow::Result<String> {
    run_git_command_env(repo_path, &[], args).await
}

/// [`run_git_command`] with extra environment variables set — used to point git
/// at a throwaway index file instead of the repository's own.
async fn run_git_command_env(
    repo_path: &Path,
    env: &[(&str, &OsStr)],
    args: &[&str],
) -> anyhow::Result<String> {
    let output = run_git_output_env(repo_path, env, args).await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git {} failed: {stderr}", args.join(" "));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run a git command and return its trimmed stdout.
///
/// Shared by the helpers that return a single scalar (hash, branch, message).
async fn run_git_trimmed(repo_path: &Path, args: &[&str]) -> anyhow::Result<String> {
    Ok(run_git_command(repo_path, args).await?.trim().to_string())
}

/// [`run_git_trimmed`] with extra environment variables set.
async fn run_git_trimmed_env(
    repo_path: &Path,
    env: &[(&str, &OsStr)],
    args: &[&str],
) -> anyhow::Result<String> {
    Ok(run_git_command_env(repo_path, env, args)
        .await?
        .trim()
        .to_string())
}

/// Check if `git status --porcelain` output contains any unstaged changes.
///
/// Looks for any line where the worktree status column (second character) is
/// non-space — an unstaged modification, deletion, or untracked file (`??` has
/// `?` there). Fully-staged-only entries (`M `, `A `) are ignored.
///
/// The test is byte-level, not a record parse, on purpose: an entry whose path no
/// parser accepts still counts as a change, so a dirty tree is never mistaken for
/// a clean one. `!!` ignored lines would count too; the callers' invocation simply
/// never asks git for ignored files.
///
/// This is a pure parsing function with no I/O — callers run `git status`
/// themselves and pass the captured stdout here.
#[must_use]
pub fn has_unstaged_changes(porcelain: &str) -> bool {
    porcelain.lines().any(|line| {
        let line = line.trim_end();
        if line.is_empty() {
            return false;
        }
        // Check the worktree status column (byte index 1).
        // For `??` (untracked) lines, both columns are `?` — the second is non-space.
        // For ` M`, `MM`, ` D`, etc., the second column is non-space.
        // For `M `, `A `, etc., the second column is space (fully staged).
        line.as_bytes().get(1).is_some_and(|&b| b != b' ')
    })
}

/// Stage all changes in the working tree via `git add -A`.
///
/// This stages tracked modifications, untracked files, and file deletions.
/// Unlike [`run_git_commit`], this does NOT commit — it only stages changes
/// into the index.
///
/// Returns the stdout of the git command on success.
pub async fn run_git_add_all(repo_path: &Path) -> anyhow::Result<String> {
    run_git_command(repo_path, &["add", "-A"]).await
}

/// Run `git rev-parse HEAD` and return the trimmed commit hash.
///
/// Errors when HEAD cannot be resolved (e.g. a repository with no commits
/// yet). Callers that need unknown-as-`None` semantics should use `.ok()`.
pub async fn run_git_head(repo_path: &Path) -> anyhow::Result<String> {
    run_git_trimmed(repo_path, &["rev-parse", "HEAD"]).await
}

/// Run `git write-tree` and return the trimmed index tree hash.
///
/// The tree hash identifies the exact staged content (index). Errors when the
/// index cannot be written (e.g. unmerged entries). Callers that need
/// unknown-as-`None` semantics should use `.ok()`.
pub async fn run_git_write_tree(repo_path: &Path) -> anyhow::Result<String> {
    run_git_trimmed(repo_path, &["write-tree"]).await
}

/// The current content identity of the working tree: `(HEAD, tree)`, where the
/// tree is the one a `git add -A` would stage (tracked modifications, untracked
/// files, deletions, `.gitignore` honoured).
///
/// Read-only: the entries go into a throwaway index seeded from the
/// repository's own, so the repository's index and working tree are left exactly
/// as they were found — a caller may take this reading without changing what an
/// agent's `git status`/`git diff` reports or the state a later step inherits.
/// The blobs a tree needs land in the object store, as they do for any
/// `git add`, and stay unreferenced until something stages that content.
/// The value it returns is what [`run_git_write_tree`] reports after a real
/// `git add -A` of the same content.
pub async fn run_git_worktree_identity(repo_path: &Path) -> anyhow::Result<(String, String)> {
    let dir = tempfile::tempdir().context("Failed to create a throwaway git index")?;
    let index = dir.path().join("index");
    // Seed from the repository's own index: without the seed, a path that is
    // tracked but matched by `.gitignore` is dropped (ignore rules govern making
    // new index entries), so the tree would differ from the real `git add -A`
    // this reading has to agree with. A split index needs nothing extra: its
    // shared file is named after its own content and looked up through the
    // repository, not relative to the index that links to it.
    //
    // A repository with no index yet is normal — git creates the throwaway one
    // from nothing. A failed copy is warned, because the tree would then be
    // built unseeded and could disagree with the real `git add -A`. A repository
    // git cannot resolve at all is not reported here: the `add -A` below fails
    // on the same condition and the caller reports the round's missing identity.
    if let Ok(path) = run_git_trimmed(repo_path, &["rev-parse", "--git-path", "index"]).await {
        let path = PathBuf::from(path);
        let path = if path.is_absolute() {
            path
        } else {
            repo_path.join(path)
        };
        match tokio::fs::copy(&path, &index).await {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                warn!(error = %e, path = %path.display(), "Could not seed the throwaway git index");
            }
            _ => {}
        }
    }
    let env = [("GIT_INDEX_FILE", index.as_os_str())];
    run_git_trimmed_env(repo_path, &env, &["add", "-A"]).await?;
    let tree = run_git_trimmed_env(repo_path, &env, &["write-tree"]).await?;
    Ok((run_git_head(repo_path).await?, tree))
}

/// Stage all changes and commit with the given message.
/// Runs `git add -A` followed by `git commit -m "<msg>"`,
/// then captures the full SHA via `git rev-parse HEAD` and
/// line stats via `git diff --numstat`.
pub async fn run_git_commit(repo_path: &Path, message: &str) -> anyhow::Result<CommitInfo> {
    // Stage all changes (tracked, untracked, removed) in the worktree.
    run_git_add_all(repo_path).await?;

    run_git_command(repo_path, &["commit", "-m", message])
        .await
        .context("Failed to commit changes")?;

    // Capture the full 40-char SHA — reliable source, not abbreviated.
    let hash = match run_git_head(repo_path).await {
        Ok(hash) => hash,
        Err(e) => {
            warn!(
                error = %e,
                "git rev-parse HEAD failed after successful commit — commit exists, returning unknown hash"
            );
            return Ok(CommitInfo {
                hash: "unknown".into(),
                lines_added: 0,
                lines_removed: 0,
            });
        }
    };

    // Capture line stats via --numstat. Try HEAD~1..HEAD first.
    let (lines_added, lines_removed) =
        if let Ok(stats) = parse_numstat(repo_path, &["HEAD~1..HEAD"]).await {
            stats
        } else if let Ok(base) = run_git_empty_tree(repo_path).await {
            // HEAD~1 doesn't exist (first commit) — diff against this
            // repository's own empty tree.
            parse_numstat(repo_path, &[&base, "HEAD"])
                .await
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };
    Ok(CommitInfo {
        hash,
        lines_added,
        lines_removed,
    })
}

/// Broadcast for pipeline auto-commit notifications. The pipeline sends the
/// repo path after a successful auto-commit (sanitation → Done); the GUI
/// subscription consumes it and refreshes the footer promptly, since a commit
/// is a ref-only change the file watcher never reports. Lives here (not in the
/// GUI) so the pipeline can signal without depending on the GUI layer.
pub(crate) static GIT_COMMIT_TX: OnceLock<tokio::sync::broadcast::Sender<PathBuf>> =
    OnceLock::new();

/// Initialise the git-commit broadcast. Called from [`crate::gui::init_git_commit_tx`]
/// in `main` before the iced application runs.
pub(crate) fn init_git_commit_tx() {
    let _ = GIT_COMMIT_TX.set(tokio::sync::broadcast::channel(64).0);
}

/// Notify the GUI that a pipeline auto-commit completed on `repo_path`, so the
/// git footer refreshes promptly instead of waiting for the periodic timer.
pub(crate) fn notify_git_commit(repo_path: &Path) {
    if let Some(tx) = GIT_COMMIT_TX.get() {
        let _ = tx.send(repo_path.to_path_buf());
    }
}

/// A single entry from `git diff --numstat` or `git show --numstat` output.
///
/// `additions` and `deletions` are `None` for binary files (where git outputs `-`
/// instead of a line count). Regular files always have `Some` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumstatEntry {
    /// Lines added, or `None` if the file is binary.
    pub additions: Option<i64>,
    /// Lines deleted, or `None` if the file is binary.
    pub deletions: Option<i64>,
    /// File path as printed by git.
    pub path: String,
}

/// Parse the output of `git diff --numstat` or `git show --numstat`.
///
/// Returns a vector of [`NumstatEntry`] values for each file.
/// Binary files (displayed as `-\t-\t<path>`) have `additions` and `deletions`
/// set to `None` so callers can distinguish them from regular entries with zero
/// changes. Lines that don't match the expected 3-field format are silently skipped.
///
/// This is a pure parsing function with no I/O — callers run git themselves
/// and pass the captured stdout here.
#[must_use]
pub fn parse_numstat_lines(stdout: &str) -> Vec<NumstatEntry> {
    let mut result = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: <additions>\t<deletions>\t<path>
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() != 3 {
            continue;
        }

        let additions_str = parts[0];
        let deletions_str = parts[1];
        let path = parts[2].to_string();

        // Binary files are displayed as "-\t-\t<path>"
        // If either field is "-", treat the file as binary (both fields None).
        // This is defensive: git always outputs "-\t-" for both fields on binary
        // files, but we handle the mixed case conservatively.
        if additions_str == "-" || deletions_str == "-" {
            result.push(NumstatEntry {
                additions: None,
                deletions: None,
                path,
            });
            continue;
        }

        let additions: i64 = additions_str.parse().unwrap_or(0);
        let deletions: i64 = deletions_str.parse().unwrap_or(0);
        result.push(NumstatEntry {
            additions: Some(additions),
            deletions: Some(deletions),
            path,
        });
    }
    result
}

/// Run `git diff --numstat <range...>` and return the per-file entries.
/// Callers pass only the range (e.g. `&["HEAD"]` or `&["<empty-tree>", "HEAD"]`).
async fn run_git_diff_numstat(
    repo_path: &Path,
    range: &[&str],
) -> anyhow::Result<Vec<NumstatEntry>> {
    let mut args = vec!["diff", "--numstat"];
    args.extend_from_slice(range);
    let stdout = run_git_command(repo_path, &args).await?;
    Ok(parse_numstat_lines(&stdout))
}

/// Run `git diff --numstat <range...>` and sum the line stats across all files.
///
/// Returns `Ok((lines_added, lines_removed))` on success (even if the diff
/// is empty). Returns the git error message on failure.
async fn parse_numstat(repo_path: &Path, range: &[&str]) -> anyhow::Result<(i64, i64)> {
    let entries = run_git_diff_numstat(repo_path, range).await?;

    let mut lines_added: i64 = 0;
    let mut lines_removed: i64 = 0;

    for entry in entries {
        // Binary files have None values — they contribute 0 lines.
        if let Some(added) = entry.additions {
            lines_added += added;
        }
        if let Some(removed) = entry.deletions {
            lines_removed += removed;
        }
    }

    Ok((lines_added, lines_removed))
}

/// Check if git is installed.
pub async fn git_is_installed() -> bool {
    let mut cmd = git_command(None);
    cmd.arg("--version");
    cmd.output().await.is_ok_and(|o| o.status.success())
}

/// Check if a git repo has any commits (spawn/git failures count as `false`).
async fn git_has_commits(repo_path: &Path) -> bool {
    run_git_head(repo_path).await.is_ok()
}

/// Print this repository's empty tree hash.
///
/// `git hash-object -t tree --stdin` reads EOF (the subprocess wrapper closes
/// stdin) and prints the empty tree's hash, so the value follows the
/// repository's object format (SHA-1 or SHA-256) instead of hardcoding SHA-1's
/// `4b825d…`. Used as the base for an unborn HEAD.
async fn run_git_empty_tree(repo_path: &Path) -> anyhow::Result<String> {
    run_git_trimmed(repo_path, &["hash-object", "-t", "tree", "--stdin"]).await
}

/// Get the current branch name (e.g. `main`, `feature/xyz`).
pub async fn run_git_current_branch(repo_path: &Path) -> anyhow::Result<String> {
    run_git_trimmed(repo_path, &["rev-parse", "--abbrev-ref", "HEAD"]).await
}

/// What the branch and its upstream each carry without the other.
#[derive(Debug, Clone, Copy)]
struct UpstreamCounts {
    /// Commits the upstream has that `HEAD` does not: what a pull would take.
    behind: usize,
    /// Commits `HEAD` has that the upstream does not: what a push to it owes.
    ahead: usize,
}

/// Measure the branch against `upstream`.
async fn ahead_behind(repo_path: &Path, upstream: &str) -> anyhow::Result<UpstreamCounts> {
    let counts = run_git_trimmed(
        repo_path,
        &[
            "rev-list",
            "--count",
            "--left-right",
            &format!("{upstream}...HEAD"),
        ],
    )
    .await?;
    let (behind, ahead) = counts
        .split_once('\t')
        .with_context(|| format!("git rev-list printed {counts:?}"))?;
    Ok(UpstreamCounts {
        behind: behind.trim().parse()?,
        ahead: ahead.trim().parse()?,
    })
}

/// Where the sync control pushes the checked-out branch, and what it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitPushTarget {
    /// The remote that accepts pushes and receives the branch.
    pub remote: String,
    /// Commits a press carries: what `remote` lacks of the branch as the update
    /// step leaves it, or every commit the branch carries when it is published or
    /// when `remote` could not be asked.
    pub commits: usize,
    /// Whether `remote` has a branch of this name: `Some(true)` it does not, so
    /// the push publishes it and the update step is skipped; `Some(false)` it
    /// does; `None` when `remote` could not be asked, so neither is claimed.
    pub publish: Option<bool>,
}

/// Where the checked-out branch can be pushed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitPushDestination {
    /// A remote accepts pushes: the one a plain `git push` reaches, the branch's
    /// tracked remote, or the only push-capable remote.
    Target(GitPushTarget),
    /// No remote accepts pushes, or several do: nothing is guessed, and the
    /// control says why instead of staying hidden.
    Nowhere {
        /// Why no destination was chosen.
        reason: String,
    },
}

/// The checked-out branch's remote state, as the footer's sync control shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitRemoteState {
    /// The branch this state describes: the footer caches it across refreshes,
    /// and a branch switched since then makes it stale rather than wrong.
    pub branch: String,
    /// Commits the branch would pull from its own update source.
    pub behind: usize,
    /// Where a push would go.
    pub push: GitPushDestination,
}

impl GitRemoteState {
    /// Whether the sync control has anything to offer: commits to pull, commits
    /// to push, or a branch no reachable remote can take (which the control says
    /// plainly rather than hiding).
    #[must_use]
    pub fn has_sync_work(&self) -> bool {
        match &self.push {
            GitPushDestination::Nowhere { .. } => true,
            GitPushDestination::Target(target) => self.behind > 0 || target.commits > 0,
        }
    }
}

/// Resolve the checked-out branch's remote state for the footer's sync control.
///
/// `behind` is measured against the branch's own upstream (what a pull would
/// take), while the push destination is a separate question: a branch may track a
/// fetch-only remote, or nothing at all, and still be pushable. `None` when
/// `HEAD` names no branch — the control concerns the checked-out branch.
pub async fn run_git_remote_state(repo_path: &Path) -> anyhow::Result<Option<GitRemoteState>> {
    let Some(lane) = resolve_push_lane(repo_path).await? else {
        return Ok(None);
    };
    Ok(Some(GitRemoteState {
        branch: lane.branch,
        behind: lane.upstream_counts.map_or(0, |counts| counts.behind),
        push: lane.destination,
    }))
}

/// The remotes git resolves for a branch: the one it tracks for updates and the
/// one a plain `git push` sends to.
struct BranchRemotes {
    /// The remote the branch tracks, even when its remote-tracking ref is gone;
    /// `None` when it tracks nothing.
    upstream: Option<String>,
    /// The remote-tracking ref the upstream names, when that ref exists — the
    /// exact local measure of what a pull would take and of what a push to the
    /// upstream still owes.
    upstream_ref: Option<String>,
    /// The remote a plain `git push` sends to, which honours
    /// `branch.<name>.pushRemote` and `remote.pushDefault`.
    push: Option<String>,
}

/// The checked-out branch's push lane: the branch, git's own view of its
/// remotes, and where a push of it would go.
struct PushLane {
    branch: String,
    remotes: BranchRemotes,
    /// The branch's distance from its upstream ref, when it has one.
    upstream_counts: Option<UpstreamCounts>,
    destination: GitPushDestination,
}

/// Resolve where the checked-out branch can be pushed; `None` when `HEAD` names
/// no branch (detached or unborn).
///
/// Where a push goes is decided by which remote accepts pushes, not by what the
/// branch tracks: a tracked remote that is fetch-only is the branch's update
/// source, never a push destination. The remote git itself would push to comes
/// first, so the destination the control names is the one a plain `git push`
/// reaches.
async fn resolve_push_lane(repo_path: &Path) -> anyhow::Result<Option<PushLane>> {
    let Some(branch) = checked_out_branch(repo_path).await else {
        return Ok(None);
    };
    let pushable: Vec<String> = remotes(repo_path)
        .await?
        .into_iter()
        .filter_map(|(name, accepts_pushes)| accepts_pushes.then_some(name))
        .collect();
    let own = branch_remotes(repo_path, &branch).await?;
    let upstream_counts = match &own.upstream_ref {
        Some(upstream) => Some(ahead_behind(repo_path, upstream).await?),
        None => None,
    };
    // A press fast-forwards the branch from its own update source before pushing,
    // so the count belongs to the tip it carries then: the upstream it can
    // fast-forward to, or the branch itself when the update cannot move it there
    // (no upstream ref, already level, or diverged).
    let push_tip = match (&own.upstream_ref, upstream_counts) {
        (Some(upstream), Some(counts)) if counts.behind > 0 && counts.ahead == 0 => {
            upstream.as_str()
        }
        _ => "HEAD",
    };
    // Prefer the remote git itself would push to, then the tracked one, then the
    // only push-capable remote; a fetch-only remote is never chosen.
    let accepts = |name: Option<&str>| {
        name.filter(|name| pushable.iter().any(|remote| remote == name))
            .map(str::to_string)
    };
    let destination_remote = accepts(own.push.as_deref())
        .or_else(|| accepts(own.upstream.as_deref()))
        .or_else(|| match pushable.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        });
    let destination = match destination_remote.as_deref() {
        Some(remote) => {
            // The upstream ref the fetch keeps current is exact, so a push to
            // the branch's own upstream needs no network round trip.
            let tracked_destination = own.upstream.as_deref() == Some(remote);
            let position = match upstream_counts {
                Some(counts) if tracked_destination => DestinationPosition {
                    publish: Some(false),
                    commits: counts.ahead,
                },
                _ => destination_position(repo_path, remote, &branch, push_tip).await?,
            };
            GitPushDestination::Target(GitPushTarget {
                remote: remote.to_string(),
                commits: position.commits,
                publish: position.publish,
            })
        }
        None => GitPushDestination::Nowhere {
            reason: match pushable.as_slice() {
                [] => "no remote accepts pushes".to_string(),
                several => format!("several remotes accept pushes ({})", several.join(", ")),
            },
        },
    };
    Ok(Some(PushLane {
        branch,
        remotes: own,
        upstream_counts,
        destination,
    }))
}

/// The branch `HEAD` names, if it names one (a detached or unborn `HEAD` does
/// not: `rev-parse --abbrev-ref` prints `HEAD`, or fails).
async fn checked_out_branch(repo_path: &Path) -> Option<String> {
    run_git_current_branch(repo_path)
        .await
        .ok()
        .filter(|branch| branch != "HEAD")
}

/// What the destination answered for a branch: what it would take, and whether it
/// already carries it at all.
struct DestinationPosition {
    /// Whether the destination has no branch of this name: `None` when it could
    /// not be asked.
    publish: Option<bool>,
    /// Commits a press carries.
    commits: usize,
}

/// Ask `remote` what it carries for `branch`, and measure the push against it.
///
/// The destination is asked directly (`ls-remote`) rather than trusting the
/// local `refs/remotes/<remote>/<branch>`: a narrowed `remote.<remote>.fetch`
/// refspec never creates that ref, and the refresh's fetch follows the branch's
/// own upstream, so that ref is stale in exactly the cases the control exists
/// for. The destination's tip may not be in this repository at all, so its copy
/// of the branch is fetched — into the remote-tracking ref, so the count holds
/// whatever refspecs are configured — when the tip's commit is unknown.
///
/// A destination that cannot be reached, or whose objects cannot be fetched, does
/// not fail the state and does not claim a publish: the branch is reported with
/// the commits it carries, so it stays visible, and a press reports git's own
/// reason.
async fn destination_position(
    repo_path: &Path,
    remote: &str,
    branch: &str,
    push_tip: &str,
) -> anyhow::Result<DestinationPosition> {
    // The full ref, not the short name: `ls-remote` also matches a short name
    // against the tail of other refs (`main` finds `feature/main` too).
    let wanted = format!("refs/heads/{branch}");
    let listing = match run_git_network(
        repo_path,
        &["ls-remote", "--branches", remote, &wanted],
        FETCH_TIMEOUT,
    )
    .await
    {
        Ok(output) => output.stdout,
        Err(e) => {
            tracing::debug!(error = %e, %remote, %branch, "git ls-remote failed");
            return unreported_position(repo_path).await;
        }
    };
    let tip = listing.lines().find_map(|line| {
        let (tip, refname) = line.split_once('\t')?;
        (refname.trim() == wanted).then(|| tip.trim().to_string())
    });
    let Some(tip) = tip else {
        // The destination has no such branch: the push publishes it, so the count
        // is everything the branch carries (nothing is pulled first).
        return Ok(DestinationPosition {
            publish: Some(true),
            commits: count_commits(repo_path, "HEAD").await?,
        });
    };
    if !rev_exists(repo_path, &tip).await? && !fetch_destination(repo_path, remote, branch).await {
        return unreported_position(repo_path).await;
    }
    Ok(DestinationPosition {
        publish: Some(false),
        commits: count_commits(repo_path, &format!("{tip}..{push_tip}")).await?,
    })
}

/// The position to report when the destination's copy of the branch is unknown —
/// it could not be reached, or its tip could not be fetched: every commit the
/// branch carries, so a pushable branch is never hidden behind a zero, and no
/// publish claim that the destination was never asked about.
async fn unreported_position(repo_path: &Path) -> anyhow::Result<DestinationPosition> {
    Ok(DestinationPosition {
        publish: None,
        commits: count_commits(repo_path, "HEAD").await?,
    })
}

/// Fetch `remote`'s copy of `branch` into its remote-tracking ref. `false` when
/// another fetch is already running or this one failed — the caller then reports
/// what the local refs say.
async fn fetch_destination(repo_path: &Path, remote: &str, branch: &str) -> bool {
    if FETCH_IN_FLIGHT.swap(true, Ordering::AcqRel) {
        tracing::debug!(%remote, %branch, "skipping the destination fetch: another fetch is running");
        return false;
    }
    let _guard = FetchInFlightGuard;
    let refspec = format!("+refs/heads/{branch}:refs/remotes/{remote}/{branch}");
    match run_git_network(
        // `--no-write-fetch-head` keeps the fetch from clobbering a concurrent
        // pull's merge source; `--no-tags` keeps a probe from following tags.
        repo_path,
        &[
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "--no-tags",
            remote,
            &refspec,
        ],
        FETCH_TIMEOUT,
    )
    .await
    {
        Ok(_) => true,
        Err(e) => {
            tracing::debug!(error = %e, %remote, %branch, "git fetch of the push destination failed");
            false
        }
    }
}

/// Remotes configured in `repo_path` and whether a push can reach each, in
/// `git remote -v` order.
///
/// A remote is push-capable only when every push URL it lists is usable — git
/// prints one `(push)` line per push URL, and a fetch-only remote disables
/// pushing by giving it a dummy push URL.
async fn remotes(repo_path: &Path) -> anyhow::Result<Vec<(String, bool)>> {
    let output = run_git_command(repo_path, &["remote", "-v"]).await?;
    let mut remotes: Vec<(String, bool)> = Vec::new();
    for line in output.lines() {
        let Some((name, target)) = line.split_once('\t') else {
            continue;
        };
        let Some(url) = target.strip_suffix(" (push)") else {
            continue;
        };
        let usable = push_destination_accepts_pushes(repo_path, url);
        match remotes.iter_mut().find(|(remote, _)| remote == name) {
            Some((_, all_usable)) => *all_usable &= usable,
            None => remotes.push((name.to_string(), usable)),
        }
    }
    Ok(remotes)
}

/// Whether a push URL can actually receive a push.
///
/// Git has no "read-only remote" flag: the convention is a dummy push URL
/// (`remote.<name>.pushurl = no-push`, `DISABLED`, …), which makes a remote
/// fetch-only by sending pushes nowhere. Such a URL is a bare word naming no
/// filesystem path — the convention's own caveat is "as long as you don't have
/// a folder of the same name", so the existence probe mirrors exactly that.
/// Everything else is a real transport (a `scheme://` URL, including `git+ssh://`
/// and custom `git-remote-*` helpers, an scp-like `host:path`, or a path).
fn push_destination_accepts_pushes(repo_path: &Path, url: &str) -> bool {
    let url = url.trim();
    if url.is_empty() {
        return false;
    }
    if url.contains([':', '/', '\\']) || url.starts_with(['.', '~']) {
        return true;
    }
    repo_path.join(url).exists()
}

/// The remotes git resolves for `branch`: the one it tracks for updates, that
/// upstream's remote-tracking ref when it exists, and the one a plain
/// `git push` sends to.
///
/// Git is asked for its own resolution (`%(upstream…)`, `%(push…)`) rather than
/// re-reading the branch's config, so a local `.` upstream, a defaulted remote
/// and a configured `pushRemote`/`pushDefault` are all handled by the same truth
/// the push itself uses.
async fn branch_remotes(repo_path: &Path, branch: &str) -> anyhow::Result<BranchRemotes> {
    let listing = run_git_command(
        repo_path,
        &[
            "for-each-ref",
            "--format=%(upstream)%09%(upstream:remotename)%09%(push:remotename)",
            &format!("refs/heads/{branch}"),
        ],
    )
    .await?;
    let mut fields = listing.trim_end().split('\t');
    let upstream_ref = fields.next().unwrap_or_default();
    let upstream = fields.next().unwrap_or_default();
    let push = fields.next().unwrap_or_default();
    // `%(upstream)` names what the config points at, not proof the ref exists: a
    // pruned or never-fetched tracking ref leaves the name behind while
    // `@{upstream}` stays unresolvable.
    let upstream_ref = if !upstream_ref.is_empty() && rev_exists(repo_path, upstream_ref).await? {
        Some(upstream_ref.to_string())
    } else {
        None
    };
    Ok(BranchRemotes {
        upstream: Some(upstream)
            .filter(|name| !name.is_empty())
            .map(str::to_string),
        upstream_ref,
        push: Some(push)
            .filter(|name| !name.is_empty())
            .map(str::to_string),
    })
}

/// Number of commits `rev` reaches.
async fn count_commits(repo_path: &Path, rev: &str) -> anyhow::Result<usize> {
    let count = run_git_trimmed(repo_path, &["rev-list", "--count", rev]).await?;
    count
        .parse()
        .with_context(|| format!("git rev-list --count {rev} printed {count:?}"))
}

/// Whether `rev` resolves to a commit in this repository.
/// Whether `rev` resolves to a commit in this repository.
///
/// `--verify` alone accepts any well-formed object id as a revision without the
/// object existing, so the peeled `^{commit}` form is what actually proves it.
async fn rev_exists(repo_path: &Path, rev: &str) -> anyhow::Result<bool> {
    Ok(run_git_output(
        repo_path,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{rev}^{{commit}}"),
        ],
    )
    .await?
    .status
    .success())
}

/// Hard timeout for a single network round trip — a `git fetch`, or the
/// destination probe that asks a remote what it carries. An unreachable/slow
/// remote must not hang the remote sync indefinitely.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// Single-flight guard for network fetches ([`run_git_fetch`] and the
/// destination probe): at most one fetch runs at a time so concurrent fetches
/// cannot overlap/contend on the same ref locks.
static FETCH_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// RAII guard that clears [`FETCH_IN_FLIGHT`] when the fetch task finishes
/// (including on error/timeout/cancellation).
struct FetchInFlightGuard;
impl Drop for FetchInFlightGuard {
    fn drop(&mut self) {
        FETCH_IN_FLIGHT.store(false, Ordering::Release);
    }
}

/// Run `git fetch` so remote refs are refreshed before the branch's remote state
/// is computed (a fetch-free comparison only sees the last-fetched remote state).
///
/// The subprocess hardening (including restoring `SSH_AUTH_SOCK` for SSH-agent
/// remotes) lives in [`run_git_network`]; the fetch is time-boxed by
/// [`FETCH_TIMEOUT`] and single-flight guarded.
///
/// Callers most often discard the report and only log a failure (the subsequent
/// remote-state read still works against local refs).
pub async fn run_git_fetch(repo_path: &Path) -> anyhow::Result<String> {
    // Single-flight: if a fetch is already running, skip this one rather than
    // queueing it (it would only contend on the same ref locks).
    if FETCH_IN_FLIGHT.swap(true, Ordering::AcqRel) {
        return Ok(String::new());
    }
    let _guard = FetchInFlightGuard;
    run_git_network(repo_path, &["fetch", "--quiet"], FETCH_TIMEOUT)
        .await
        .map(GitNetworkOutput::report)
}

/// What a [`run_git_network`] step printed, both streams trimmed.
struct GitNetworkOutput {
    /// git's progress and report lines — a push reports its ref updates and
    /// "Everything up-to-date" here.
    stderr: String,
    /// The command's own answer, e.g. `ls-remote`'s ref list.
    stdout: String,
}

impl GitNetworkOutput {
    /// The human report, stderr first: a push prints its ref report before its
    /// trailing upstream notice on stdout, so reading stdout alone would show a
    /// real push as nothing to do.
    fn report(self) -> String {
        [self.stderr, self.stdout]
            .into_iter()
            .filter(|stream| !stream.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Run a network git command (fetch/pull/push/ls-remote) with a hard timeout.
///
/// The subprocess environment is the product's own internal one (see
/// [`git_command`]), then `GIT_TERMINAL_PROMPT=0` prevents git/credential
/// helpers from popping an interactive (osxkeychain) prompt for a
/// credential-less remote, and `SSH_AUTH_SOCK` is restored (if present) so
/// SSH-agent remotes still work — the reduced base drops it.
///
/// On timeout the direct `git` child is killed and reaped; a remote helper it
/// spawned (e.g. `git-remote-https`) is not signalled directly but terminates
/// when its communication pipe closes, so it cannot hold the command open past
/// the timeout. A non-zero exit reports git's own stderr, so a refused push
/// carries git's reason.
///
/// The two streams stay apart: a command's answer (`ls-remote`) is parsed from
/// stdout, while [`GitNetworkOutput::report`] joins both for the human report.
async fn run_git_network(
    repo_path: &Path,
    args: &[&str],
    timeout: Duration,
) -> anyhow::Result<GitNetworkOutput> {
    let mut cmd = git_command(Some(repo_path));
    cmd.args(args).current_dir(repo_path);
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    // Restore the SSH agent socket for SSH remotes (dropped by the internal environment).
    if let Some(sock) = std::env::var_os("SSH_AUTH_SOCK") {
        cmd.env("SSH_AUTH_SOCK", sock);
    }
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    // Kill the child if this task is cancelled (e.g. on shutdown) so a running
    // network operation cannot outlive the runtime and orphan.
    cmd.kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to spawn git {}", args.join(" ")))?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();

    // Wait for the child while draining its pipes concurrently, all inside the
    // timeout — a slow/unreachable remote (or a subprocess that outlives git
    // holding the pipes) cannot hang the caller. On timeout, kill and reap so
    // the process does not linger as a zombie.
    let wait = child.wait();
    let read_stdout = async {
        if let Some(out) = &mut stdout {
            let _ = out.read_to_end(&mut stdout_buf).await;
        }
    };
    let read_stderr = async {
        if let Some(err) = &mut stderr {
            let _ = err.read_to_end(&mut stderr_buf).await;
        }
    };

    let status = if let Ok(res) = tokio::time::timeout(timeout, async {
        tokio::join!(wait, read_stdout, read_stderr).0
    })
    .await
    {
        res.with_context(|| format!("git {} wait failed", args.join(" ")))?
    } else {
        let _ = child.kill().await;
        let _ = child.wait().await;
        anyhow::bail!(
            "git {} timed out after {}s",
            args.join(" "),
            timeout.as_secs()
        );
    };

    if !status.success() {
        anyhow::bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&stderr_buf)
        );
    }

    Ok(GitNetworkOutput {
        stderr: String::from_utf8_lossy(&stderr_buf).trim().to_string(),
        stdout: String::from_utf8_lossy(&stdout_buf).trim().to_string(),
    })
}

/// Aggregate working-tree diff statistics for the GUI footer.
///
/// `added`/`removed` are the exact line counts used by the footer (+X / -Y)
/// and by reviewer churn calibration, summed from every file in
/// [`GitWorktreeSnapshot::changes`]. `huge_binary_file_count` counts oversized
/// (> [`MAX_UNTRACKED_SIZE`]) and binary *untracked* files. It is a separate
/// indicator and is deliberately NOT folded into `added`/`removed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffStats {
    /// Total lines added (tracked diff + line-counted untracked text files).
    pub added: i64,
    /// Total lines removed (tracked diff only).
    pub removed: i64,
    /// Count of oversized/binary untracked files surfaced as a separate
    /// "N files" indicator, not line counts.
    pub huge_binary_file_count: usize,
    /// Untracked files the bounded worktree read left out (both surfaces
    /// render [`Self::omitted_note`] for this).
    pub omitted_file_count: usize,
}

impl DiffStats {
    /// The one description of what the bounded untracked read left out; both the
    /// footer and the diff window render this exact note.
    pub(crate) fn omitted_note(&self) -> Option<String> {
        (self.omitted_file_count > 0).then(|| {
            format!(
                "{} uncommitted file{} left out (worktree read cap)",
                self.omitted_file_count,
                if self.omitted_file_count == 1 {
                    ""
                } else {
                    "s"
                }
            )
        })
    }
}

/// How an untracked file was classified for display / line counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UntrackedKind {
    Text,
    Binary,
    TooLarge(u64),
}

/// One file in the settled uncommitted change set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeChange {
    /// Path relative to the repository root (git reports root-relative paths,
    /// even from a subdirectory workspace).
    pub path: String,
    /// Rename/copy source, when git detected one.
    pub old_path: Option<String>,
    /// Lines added; 0 when the content has no lines (binary, oversized).
    pub added: i64,
    /// Lines removed; 0 likewise.
    pub removed: i64,
    /// `Some` exactly when the file is untracked.
    pub untracked: Option<UntrackedKind>,
}

/// Git file status for coloring the file tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitFileStatus {
    /// File has uncommitted modifications (M in porcelain output).
    Modified,
    /// File is untracked (?? in porcelain output) or newly added (A).
    Added,
}

/// Run `git diff --numstat -z --find-renames <base>` and parse it into the
/// tracked part of the change manifest.
///
/// `-z` keeps paths NUL-delimited and unquoted, so the returned paths are the
/// raw repo-root-relative names.
async fn run_git_worktree_numstat(
    repo_path: &Path,
    base: &str,
) -> anyhow::Result<Vec<WorktreeChange>> {
    let stdout = run_git_command(
        repo_path,
        &["diff", "--numstat", "-z", "--find-renames", base],
    )
    .await?;
    Ok(parse_numstat_z(&stdout))
}

/// Parse `git diff --numstat -z` output.
///
/// A normal record is `added\tremoved\t<path>\0`; a rename/copy record leaves
/// the path field empty and follows it with `<old>\0<new>\0`. `-` (binary) counts
/// as 0 lines. The trailing empty field after the final NUL is ignored.
fn parse_numstat_z(stdout: &str) -> Vec<WorktreeChange> {
    let mut fields = stdout.split('\0');
    let mut result = Vec::new();
    while let Some(field) = fields.next() {
        if field.is_empty() {
            continue;
        }
        let mut parts = field.splitn(3, '\t');
        let added = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let removed = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let Some(path) = parts.next() else {
            continue;
        };
        let (path, old_path) = if path.is_empty() {
            let (Some(old), Some(new)) = (fields.next(), fields.next()) else {
                break;
            };
            (new.to_string(), Some(old.to_string()))
        } else {
            (path.to_string(), None)
        };
        result.push(WorktreeChange {
            path,
            old_path,
            added,
            removed,
            untracked: None,
        });
    }
    result
}

/// Working-tree snapshot: aggregate diff stats, per-file porcelain statuses,
/// and the settled change set, from one `git status --porcelain
/// --untracked-files=all` reading plus one numstat pass.
///
/// The statuses are the same map the editor's file-tree coloring uses (once
/// forwarded by `GitState`); `changes` is what both the footer line counter and
/// the diff window draw from; and the stats drive the footer (+N / -M) and
/// reviewer churn calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitWorktreeSnapshot {
    /// Aggregate line stats, summed from [`Self::changes`].
    pub stats: DiffStats,
    /// Per-file porcelain status, keyed by the path exactly as git reports it
    /// (repo-root-relative).
    pub file_statuses: HashMap<String, GitFileStatus>,
    /// The repo has no commits (unborn HEAD): tracked changes are measured
    /// against the empty tree but `unborn_head` still tells churn consumers
    /// there is no valid baseline. Consumers that predate the unborn-HEAD
    /// tolerance and must keep their old error semantics (reviewer churn)
    /// check this instead of treating the snapshot as a valid diff baseline.
    pub unborn_head: bool,
    /// Every uncommitted file: tracked entries in numstat order, then untracked
    /// files in status order.
    pub changes: Vec<WorktreeChange>,
    /// Repository top level; untracked status paths resolve against this.
    pub repo_root: PathBuf,
}

/// A manifest entry for an untracked file. `added` is its line count (0 for
/// content with no lines); an untracked file never removes lines.
fn untracked_change(path: &str, kind: UntrackedKind, added: i64) -> WorktreeChange {
    WorktreeChange {
        path: path.to_owned(),
        old_path: None,
        added,
        removed: 0,
        untracked: Some(kind),
    }
}

/// Classify each `??` status path into a [`WorktreeChange`], returning the
/// changes and how many files the per-refresh caps left out.
///
/// `repo_root` resolves the root-relative status paths. Only regular files are
/// classified; collapsed nested repositories (`dir/`) are already excluded by
/// [`parse_untracked_from_porcelain`], so they never inflate the omitted tally.
/// The metadata read here gates the byte budget *before* the content read; the
/// read and its binary/text verdict are shared with the diff window through
/// [`read_untracked_file_with_meta`], so the two cannot drift on what a file is.
async fn classify_untracked_changes(
    repo_root: &Path,
    untracked: &[String],
) -> (Vec<WorktreeChange>, usize) {
    let mut changes = Vec::new();
    let mut files_processed = 0usize;
    let mut bytes_read = 0u64;
    let mut omitted = 0usize;
    for path in untracked {
        if files_processed >= MAX_UNTRACKED_FILES_COUNTED || bytes_read >= MAX_UNTRACKED_BYTES_READ
        {
            omitted += 1;
            continue;
        }
        let full_path = repo_root.join(path);
        let Ok(meta) = tokio::fs::metadata(&full_path).await else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        files_processed += 1;
        if meta.len() > MAX_UNTRACKED_SIZE {
            changes.push(untracked_change(
                path,
                UntrackedKind::TooLarge(meta.len()),
                0,
            ));
            continue;
        }
        if bytes_read + meta.len() > MAX_UNTRACKED_BYTES_READ {
            omitted += 1;
            continue;
        }
        match read_untracked_file_with_meta(&full_path, &meta).await {
            UntrackedFileRead::Text(text) => {
                bytes_read += meta.len();
                #[expect(clippy::cast_possible_wrap)] // line count fits i64
                let added = text.lines().count() as i64;
                changes.push(untracked_change(path, UntrackedKind::Text, added));
            }
            UntrackedFileRead::Binary => {
                bytes_read += meta.len();
                changes.push(untracked_change(path, UntrackedKind::Binary, 0));
            }
            // A file that grew between the metadata read and the content read.
            UntrackedFileRead::TooLarge(size) => {
                changes.push(untracked_change(path, UntrackedKind::TooLarge(size), 0));
            }
            UntrackedFileRead::Skip => {}
        }
    }
    (changes, omitted)
}

/// Capture the working-tree snapshot for a repository path.
///
/// A `git status --porcelain --untracked-files=all` failure is fatal: broken or
/// missing git never yields a snapshot, so callers keep their last-known-good
/// state (or default churn).
///
/// Tracked changes come from one numstat pass against `HEAD`; a missing HEAD
/// (unborn repo) measures them against this repository's empty tree instead.
/// Untracked files (`??` entries) are classified here — line-counted text,
/// binary, or oversized — from `repo_root`, because git reports status paths
/// relative to the repository root even when the workspace is a subdirectory.
/// Status enumeration itself is unbounded (the complete file set is the
/// point); the per-file content reads are bounded by the `MAX_UNTRACKED_*`
/// caps, and files those caps leave out are reported through
/// `omitted_file_count`.
pub async fn run_git_worktree_snapshot(repo_path: &Path) -> anyhow::Result<GitWorktreeSnapshot> {
    let repo_root =
        resolve_git_top_level(repo_path).context("Failed to resolve the repository top-level")?;
    let porcelain = run_git_status_all(&repo_root).await?;
    let file_statuses = parse_git_status_porcelain(&porcelain);
    let untracked = parse_untracked_from_porcelain(&porcelain);

    let (tracked, unborn_head) = match run_git_worktree_numstat(&repo_root, "HEAD").await {
        Ok(entries) => (entries, false),
        // `git_has_commits` distinguishes an unborn HEAD (a freshly
        // `git init`ed repo with no commits) from other numstat failures — an
        // unborn repo still yields the full change set, measured against the
        // empty tree.
        Err(_) if !git_has_commits(&repo_root).await => {
            let base = run_git_empty_tree(&repo_root).await?;
            (
                run_git_worktree_numstat(&repo_root, &base)
                    .await
                    .context("Failed to compute working-tree diff stats")?,
                true,
            )
        }
        Err(e) => return Err(e.context("Failed to compute working-tree diff stats")),
    };

    let mut changes = tracked;
    let (untracked_changes, omitted) = classify_untracked_changes(&repo_root, &untracked).await;
    changes.extend(untracked_changes);

    let mut stats = DiffStats {
        added: 0,
        removed: 0,
        huge_binary_file_count: 0,
        omitted_file_count: omitted,
    };
    for change in &changes {
        stats.added += change.added;
        stats.removed += change.removed;
        if matches!(
            change.untracked,
            Some(UntrackedKind::Binary | UntrackedKind::TooLarge(_))
        ) {
            stats.huge_binary_file_count += 1;
        }
    }

    Ok(GitWorktreeSnapshot {
        stats,
        file_statuses,
        unborn_head,
        changes,
        repo_root,
    })
}

/// Hard timeout for a `git pull`/`git push` from the sync control.
const SYNC_TIMEOUT: Duration = Duration::from_secs(300);

/// What a sync did, and — when nothing could take a push — why not.
#[derive(Debug, Clone)]
pub struct GitSyncReport {
    /// git's own report from the steps that ran (empty when none did).
    pub output: String,
    /// Why nothing was pushed, when no remote could take the branch.
    pub not_pushed: Option<String>,
}

/// Sync the checked-out branch: update it from its own update source, then push
/// it wherever a push can actually go.
///
/// The destination is resolved here rather than taken from the footer's cached
/// state, so a branch switched since the last refresh is pushed where *it* can
/// go, not where the previous branch could. The branch is updated from its own
/// tracking ref first — unless it has none, or the push publishes a new branch,
/// which the destination has no copy of to update from. A branch that tracks
/// nothing records its upstream on the remote that received the push; an existing
/// tracking — of a fetch-only remote, say — is left alone, because it is the
/// branch's update source, not a push destination. Returns what git reported,
/// from both streams.
pub async fn run_git_sync(repo_path: &Path) -> anyhow::Result<GitSyncReport> {
    let lane = resolve_push_lane(repo_path)
        .await?
        .context("no branch is checked out")?;
    let publish = matches!(&lane.destination, GitPushDestination::Target(target) if target.publish == Some(true));
    let update = if lane.remotes.upstream_ref.is_some() && !publish {
        Some(
            run_git_network(repo_path, &["pull", "--ff-only"], SYNC_TIMEOUT)
                .await?
                .report(),
        )
    } else {
        None
    };
    let (not_pushed, push) = match &lane.destination {
        GitPushDestination::Nowhere { reason } => (Some(reason.clone()), None),
        GitPushDestination::Target(target) => {
            // A plain `git push` reaches the branch's own push destination — the
            // one the control names whenever it accepts pushes — so it keeps
            // honouring the configured push remote, push refspecs and
            // `push.default`. Everywhere else the destination is named
            // explicitly, and a branch that tracks nothing records its upstream
            // on the remote that received the push.
            let record_upstream = lane.remotes.upstream.is_none();
            let plain =
                !record_upstream && lane.remotes.push.as_deref() == Some(target.remote.as_str());
            let report = if plain {
                run_git_network(repo_path, &["push"], SYNC_TIMEOUT)
                    .await?
                    .report()
            } else {
                let mut args = vec!["push"];
                if record_upstream {
                    args.push("--set-upstream");
                }
                args.push(target.remote.as_str());
                args.push(lane.branch.as_str());
                run_git_network(repo_path, &args, SYNC_TIMEOUT)
                    .await?
                    .report()
            };
            (None, Some(report))
        }
    };
    Ok(GitSyncReport {
        output: update
            .into_iter()
            .chain(push)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        not_pushed,
    })
}

/// Discard all changes to a path — restores tracked files from HEAD, unstages
/// staged new files, and removes untracked files. Uses a 3-step git command
/// sequence to handle ALL file states:
///
/// 1. `git checkout HEAD -- <path>`
///    — restores tracked files from HEAD (handles Modified, Deleted, Renamed).
///    For files staged in the index (Added) that don't exist in HEAD, checkout
///    removes them from both index and working tree. Untracked files fail with
///    "did not match" — absorbed below.
///
/// 2. `git reset HEAD -- <path>`
///    — unstages staged new (Added) files so that `git clean` can remove them.
///    For files already restored by checkout this is a no-op. Errors
///    (e.g. untracked files not in index) are absorbed.
///
/// 3. `git clean -f[d] -- <path>`
///    — removes untracked files. `-f` for files, `-fd` for directories (recurses
///    into subdirectories). Errors (file already tracked, already removed by
///    checkout) are absorbed.
///
/// `path` is repo-root-relative (as git reports it), which is exactly what the
/// diff window lists. Every pathspec uses `:(top,literal)`: `top` resolves it
/// against the repository root whatever the caller's working directory — a
/// subdirectory workspace would otherwise rewrite or delete a same-named path
/// beneath itself — and `literal` keeps glob characters in file names from
/// matching other files.
///
/// All errors from all three steps are absorbed. After the sequence, the working
/// tree is verified via
/// `git status --porcelain --untracked-files=normal -- <path>`: if the output is
/// empty the operation succeeded; otherwise the remaining changes are reported.
/// The untracked mode is pinned for the same reason [`run_git_status`] pins it —
/// the owner's `status.showUntrackedFiles` must not decide this answer — so a
/// leftover `git clean` deliberately keeps (a nested repository needs `-ff`) is
/// reported as the failure it is instead of a silent success.
///
/// Note: `git reset HEAD` also exits with non-zero status when the path is
/// outside the repository — callers should validate paths before calling this.
pub async fn run_git_discard(
    repo_path: &Path,
    path: &str,
    target: DiscardTarget,
) -> anyhow::Result<()> {
    let path = format!(":(top,literal){path}");
    let _ = run_git_command(repo_path, &["checkout", "HEAD", "--", &path]).await;

    let _ = run_git_command(repo_path, &["reset", "HEAD", "--", &path]).await;

    let clean_args: &[&str] = match target {
        DiscardTarget::Directory => &["clean", "-fd", "--", &path],
        DiscardTarget::File => &["clean", "-f", "--", &path],
    };
    let _ = run_git_command(repo_path, clean_args).await;

    // Verify: check if any changes remain.
    let status_args = [
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        &path,
    ];
    match run_git_command(repo_path, &status_args).await {
        Ok(status) if status.trim().is_empty() => Ok(()),
        Ok(status) => anyhow::bail!("Changes remain after discard:\n{}", status.trim()),
        Err(e) => anyhow::bail!("Discard ran but verification failed: {e}"),
    }
}

/// Get the last commit's subject via `git log -1 --format=%s`.
///
/// If `commit_hash` is `Some`, get the message for that specific commit
/// instead of HEAD.
pub async fn run_git_commit_message(
    repo_path: &Path,
    commit_hash: Option<&str>,
) -> anyhow::Result<String> {
    let mut args = vec!["log", "-1", "--format=%s"];
    if let Some(hash) = commit_hash {
        args.push(hash);
    }
    run_git_trimmed(repo_path, &args).await
}

/// List new or untracked files in the working tree.
///
/// Runs [`run_git_status`], so it reads the same pinned untracked mode as every
/// other reader here, and selects the entries a sanitation pass has to inspect:
/// untracked (`??`) files and files staged as new (`A`). An untracked directory
/// arrives as its single collapsed `dir/` entry: this list goes into the
/// sanitation prompt uncapped, so the per-file `-uall` form would let one
/// un-ignored build directory blow the prompt budget, while the collapsed entry
/// already names the directory and keeps the tree dirty. An entry whose quoted
/// path does not decode is skipped rather than reported as raw text (see
/// [`PorcelainRecord`]) — the wrong-path alternative, at the theoretical cost of
/// an empty list letting the sanitation commit proceed uninspected.
pub(crate) async fn list_new_or_untracked_files(repo_path: &Path) -> anyhow::Result<Vec<String>> {
    let porcelain = run_git_status(repo_path).await?;
    Ok(parse_new_files_from_porcelain(&porcelain))
}

/// One parsed `git status --porcelain` record: both status columns and the path.
///
/// The record is `<XY><space><path>`, `X` the index status and `Y` the worktree
/// status. Git C-quotes a path that needs it, quoting each side of a rename or
/// copy pair independently, so the path comes back unquoted; a rename or copy
/// yields its **destination** — the name that exists in the working tree — for
/// either column, never the `old -> new` text. A record too short to hold a path,
/// or one whose quoted path has escapes that do not decode, yields `None` and is
/// skipped rather than reported as its raw text. A field not wrapped in quotes on
/// both ends is taken literally — git always wraps a quoted path on both ends, so
/// the only such field is an unquoted path (the status map reads it the same way).
///
/// This is the record parser of the two path listings. The status map parses its
/// own records because it classifies rather than lists, sharing only
/// [`split_rename_paths`]. Where the views legitimately differ:
///
/// - A trailing-slash entry is a directory, not a file. The
///   `--untracked-files=normal` reading of [`list_new_or_untracked_files`]
///   collapses an untracked directory to one `dir/` entry, so
///   [`parse_new_files_from_porcelain`] keeps it; the snapshot's
///   `--untracked-files=all` reading lists a directory's contents and grows a
///   trailing slash only for a collapsed nested repository, so
///   [`parse_untracked_from_porcelain`] drops it; the map strips the slash to key
///   a tree row.
/// - A deleted entry (`D` in either column) is stored by no view: the map skips
///   it, neither listing selects it — except `AD`, whose `A` column the new-files
///   list selects and the map still skips — and a deletion registers only in the
///   byte-level [`has_unstaged_changes`], which reads columns rather than records
///   (so ` D` counts and `D ` does not).
/// - ` A`, the intent-to-add state of `git add -N`, is `Added` in the map and
///   excluded by both listings, which select on the index column.
struct PorcelainRecord {
    /// Index (staged) status column.
    index: char,
    /// Worktree (unstaged) status column.
    worktree: char,
    /// Path relative to the repository root, unquoted.
    path: String,
}

/// Parse one `git status --porcelain` record — see [`PorcelainRecord`].
///
/// The two status columns must be followed by their separator space; anything
/// else is not a record.
fn parse_porcelain_record(line: &str) -> Option<PorcelainRecord> {
    let mut chars = line.chars();
    let index = chars.next()?;
    let worktree = chars.next()?;
    if chars.next()? != ' ' {
        return None;
    }
    let field = chars.as_str();
    if field.is_empty() {
        return None;
    }
    let path = if index == 'R' || index == 'C' || worktree == 'R' || worktree == 'C' {
        split_rename_paths(field)?
    } else {
        unquote_c_style(field)?
    };
    Some(PorcelainRecord {
        index,
        worktree,
        path,
    })
}

/// The paths of the records `select` accepts, in git's order.
fn parse_porcelain_paths(
    porcelain: &str,
    mut select: impl FnMut(&PorcelainRecord) -> bool,
) -> Vec<String> {
    porcelain
        .lines()
        .filter_map(parse_porcelain_record)
        .filter(|record| select(record))
        .map(|record| record.path)
        .collect()
}

/// The destination path of a rename or copy record, from its `<old> -> <new>`
/// field.
///
/// The delimiter is the first ` -> ` outside a quoted span: git quotes each side
/// independently, so neither "the field starts quoted" nor "the last ` -> ` wins"
/// identifies it — a quoted source with a plain destination, a plain source with a
/// quoted destination, and a quoted destination containing the delimiter all split
/// here. The source is only scanned, never returned: the recorded path is the one
/// that exists in the working tree. `None` when there is no delimiter or the
/// destination is not a path, and the caller skips the record.
fn split_rename_paths(field: &str) -> Option<String> {
    let mut quoted = false;
    let mut escaped = false;
    for (at, c) in field.char_indices() {
        if escaped {
            escaped = false;
        } else if quoted && c == '\\' {
            escaped = true;
        } else if c == '"' {
            quoted = !quoted;
        } else if !quoted && let Some(destination) = field[at..].strip_prefix(" -> ") {
            return unquote_c_style(destination).filter(|path| !path.is_empty());
        }
    }
    None
}

/// New/added file paths from a `git status --porcelain` reading.
///
/// Selects `??` (untracked) records and index-column `A` ones: `A ` (staged,
/// clean), `AM` (staged, then modified), and `AD` (added, then deleted). Excludes
/// ` A` — the intent-to-add state of `git add -N` — which is a real status but
/// carries a space in the index column, so nothing is staged.
///
/// To list only truly untracked files, use [`parse_untracked_from_porcelain`].
#[must_use]
fn parse_new_files_from_porcelain(porcelain: &str) -> Vec<String> {
    parse_porcelain_paths(porcelain, |record| {
        record.index == 'A' || (record.index == '?' && record.worktree == '?')
    })
}

/// Untracked file paths from the snapshot's `--untracked-files=all` reading.
///
/// Selects `??` records only, so staged-as-new files the tracked half of the
/// snapshot already covers are not listed twice. A trailing-slash entry is a
/// collapsed nested repository rather than a file and is dropped — see
/// [`PorcelainRecord`].
#[must_use]
fn parse_untracked_from_porcelain(porcelain: &str) -> Vec<String> {
    parse_porcelain_paths(porcelain, |record| {
        record.index == '?' && record.worktree == '?'
    })
    .into_iter()
    .filter(|path| !path.ends_with('/'))
    .collect()
}

/// Parse `git status --porcelain` output into a map of file path → `GitFileStatus`.
///
/// The porcelain format uses a two-column status (index + worktree). If either
/// column is `M` the file is [`GitFileStatus::Modified`]; otherwise staged-new
/// (`A`) and untracked (`??`) entries both map to [`GitFileStatus::Added`] — so
/// the unmerged `AA`, `AU` and `UA` store `Added` through their `A` column. If a
/// file appears both modified and added, `Modified` wins. A rename in either
/// column keys its destination as [`GitFileStatus::Modified`] (see
/// [`split_rename_paths`]); a copy is skipped, as is a rename field that does not
/// split. A deleted entry (`D` in either column — `D `, ` D`, `AD`, `DU`, `DD`)
/// is skipped, the deletion taking precedence over an `A` column. Everything else
/// (`UU`, typechange) stores nothing. Handles git's C-style quoting for paths
/// with special characters, and a path that does not unquote is skipped.
///
/// Keys are the paths as emitted by `git status --porcelain` —
/// repo-root-relative, with a trailing slash stripped so they match a tree row.
/// When the workspace is the repository root, this is the same key space as the
/// editor's file tree.
///
/// The differences from the path listings are stated on [`PorcelainRecord`].
#[must_use]
pub(crate) fn parse_git_status_porcelain(output: &str) -> HashMap<String, GitFileStatus> {
    let mut map: HashMap<String, GitFileStatus> = HashMap::new();

    for line in output.lines() {
        let trimmed = line.trim_end();
        if trimmed.len() < 2 {
            continue;
        }

        let chars: Vec<char> = trimmed.chars().take(2).collect();
        if chars.len() < 2 {
            continue;
        }

        let ix = chars[0];
        let wt = chars[1];

        // Skip deleted files — they don't appear in the working tree.
        if ix == 'D' || wt == 'D' {
            continue;
        }

        // Rename entries: "R  old_path -> new_path". Git quotes each side
        // separately, so the split cannot assume a shared quoting style. The
        // worktree column is treated the same way as a robustness rule: ` R`
        // ("renamed in work tree") is documented but not observed in practice, and
        // if it ever appears, dropping the row would be worse than keying its
        // destination. The destination is the path in the tree either way.
        if ix == 'R' || wt == 'R' {
            let Some(new_path) = split_rename_paths(trimmed[2..].trim_start()) else {
                continue;
            };
            map.insert(new_path, GitFileStatus::Modified);
            continue;
        }

        // Copies are skipped: the classification below knows only M and A.
        if ix == 'C' || wt == 'C' {
            continue;
        }

        // Extract path: strip first 2 chars (status columns) and leading space.
        let path = trimmed[2..].trim_start();
        if path.is_empty() {
            continue;
        }

        let status = if ix == 'M' || wt == 'M' {
            GitFileStatus::Modified
        } else if ix == 'A' || wt == 'A' || (ix == '?' && wt == '?') {
            GitFileStatus::Added
        } else {
            // Clean or ignored — don't store.
            continue;
        };

        let Some(path) = unquote_c_style(path) else {
            continue;
        };
        // Strip trailing slash — git appends '/' for untracked directories
        // (e.g., `?? new_dir/`), but tree node full_path has no trailing slash.
        let path = path.strip_suffix('/').unwrap_or(&path).to_string();

        // For entries with multiple lines referencing the same file (e.g., staged +
        // unstaged), keep the most "interesting" status: Modified > Added.
        let entry = map.entry(path).or_insert(status);
        if status == GitFileStatus::Modified && *entry == GitFileStatus::Added {
            *entry = GitFileStatus::Modified;
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::init_temp_repo;

    /// The worktree identity is a pure reading (the repository's own index and
    /// status are left as found) that agrees with a real `git add -A` — the value
    /// the reviewed-base skip check compares an index tree against.
    #[tokio::test]
    async fn worktree_identity_is_read_only_and_matches_a_real_add() {
        let (_dir, repo_path) = init_temp_repo();
        std::fs::write(repo_path.join("tracked.txt"), b"changed\n").unwrap();
        std::fs::write(repo_path.join("untracked.txt"), b"new\n").unwrap();

        let status_before = run_git_status(&repo_path).await.unwrap();
        let index_before = run_git_write_tree(&repo_path).await.unwrap();
        let (head, tree) = run_git_worktree_identity(&repo_path).await.unwrap();

        assert_eq!(
            status_before,
            run_git_status(&repo_path).await.unwrap(),
            "the reading must not change the working-tree status",
        );
        assert_eq!(
            index_before,
            run_git_write_tree(&repo_path).await.unwrap(),
            "the reading must not touch the repository's own index",
        );

        run_git_add_all(&repo_path).await.unwrap();
        assert_eq!(
            tree,
            run_git_write_tree(&repo_path).await.unwrap(),
            "the identity must be what a real add -A stages",
        );
        assert_eq!(head, run_git_head(&repo_path).await.unwrap());
    }

    /// Both cases the seed exists for: a split index (whose shared file git
    /// resolves through the repository, not next to the copy) and a tracked file
    /// matched by `.gitignore`, which an unseeded throwaway index drops.
    #[tokio::test]
    async fn worktree_identity_seeds_a_split_index_and_ignored_tracked_paths() {
        let (_dir, repo_path) = init_temp_repo();
        let status = std::process::Command::new("git")
            .args(["update-index", "--split-index"])
            .current_dir(&repo_path)
            .status()
            .expect("git update-index");
        assert!(status.success());
        std::fs::write(repo_path.join(".gitignore"), b"test.txt\n").unwrap();
        std::fs::write(repo_path.join("test.txt"), b"v2\n").unwrap();
        std::fs::write(repo_path.join("untracked.txt"), b"new\n").unwrap();

        let (_, tree) = run_git_worktree_identity(&repo_path).await.unwrap();
        run_git_add_all(&repo_path).await.unwrap();
        assert_eq!(
            tree,
            run_git_write_tree(&repo_path).await.unwrap(),
            "the identity must be what a real add -A stages",
        );
    }

    // ── safe.directory resolver (CVE-2022-24765 ancestor walk) ───

    /// One rule at every entry point it is asked from: the walk resolves the
    /// `.git`-containing ancestor, whether it is asked from the repository root,
    /// from a subdirectory below it, or from a directory whose `.git` is a
    /// *file* (linked worktree / submodule marker — the walk uses `exists()`,
    /// not `is_dir()`) — and reports no repository outside one.
    #[test]
    fn resolve_git_top_level_cases() {
        let (_dir, repo_path) = init_temp_repo();
        let subdir = repo_path.join("src");
        std::fs::create_dir(&subdir).expect("create subdir");
        let worktree = tempfile::tempdir().expect("create temp dir");
        std::fs::write(worktree.path().join(".git"), b"gitdir: /elsewhere")
            .expect("write .git file");
        let outside = tempfile::tempdir().expect("create temp dir");

        let root = Some(crate::util::test::canonical_without_verbatim_prefix(
            &repo_path,
        ));
        let as_worktree = Some(crate::util::test::canonical_without_verbatim_prefix(
            worktree.path(),
        ));
        let cases: &[(&str, &Path, Option<&PathBuf>)] = &[
            ("repository root", &repo_path, root.as_ref()),
            ("subdirectory below it", &subdir, root.as_ref()),
            (".git file marker", worktree.path(), as_worktree.as_ref()),
            ("outside any repository", outside.path(), None),
        ];
        for (name, path, expected) in cases {
            assert_eq!(
                resolve_git_top_level(path).as_ref(),
                *expected,
                "case {name}: {}",
                path.display()
            );
        }
    }

    // ── Safe.directory injection (dubious-ownership fix) ──────────

    /// Build a git command with a hermetic config environment: `safe.directory`
    /// must come only from the injected `GIT_CONFIG_*` env, never from the
    /// developer's global/system gitconfig (which may already list the repo or
    /// use `*`). `GIT_TEST_ASSUME_DIFFERENT_OWNER=1` simulates a repo owned by a
    /// different UID without requiring `chown`/root.
    fn hermetic_git_command(repo_root: Option<&Path>) -> tokio::process::Command {
        let mut cmd = git_command(repo_root);
        cmd.env("GIT_CONFIG_GLOBAL", "/dev/null");
        cmd.env("GIT_CONFIG_NOSYSTEM", "1");
        cmd.env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1");
        cmd
    }

    /// Simulates a repo owned by a different UID via git's
    /// `GIT_TEST_ASSUME_DIFFERENT_OWNER` hook: the injected top-level
    /// `safe.directory` must make git operable.
    #[tokio::test]
    async fn test_safe_directory_injection_bypasses_dubious_ownership() {
        let (_dir, repo_path) = init_temp_repo();

        let mut cmd = hermetic_git_command(Some(&repo_path));
        cmd.arg("status").arg("--porcelain").current_dir(&repo_path);
        let out = cmd.output().await.expect("run git status");
        assert!(
            out.status.success(),
            "git status must succeed with safe.directory injected: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[tokio::test]
    async fn test_without_safe_directory_fails_dubious_ownership() {
        let (_dir, repo_path) = init_temp_repo();

        let mut cmd = hermetic_git_command(None);
        cmd.arg("status").arg("--porcelain").current_dir(&repo_path);
        let out = cmd.output().await.expect("run git status");
        assert!(
            !out.status.success(),
            "git must refuse dubious ownership without safe.directory injection"
        );
    }

    /// The injected value must be the repo *top-level* even when the invocation
    /// runs from a subdirectory workspace — git's own discovery normalizes to
    /// the top-level, so a subdir value would not match.
    #[tokio::test]
    async fn test_safe_directory_injects_top_level_for_subdir_workspace() {
        let (_dir, repo_path) = init_temp_repo();
        let subdir = repo_path.join("src");
        std::fs::create_dir(&subdir).expect("create subdir");

        let mut cmd = hermetic_git_command(Some(&subdir));
        cmd.arg("status").arg("--porcelain").current_dir(&subdir);
        let out = cmd.output().await.expect("run git status");
        assert!(
            out.status.success(),
            "git from a repo subdir must match the repo top-level safe.directory: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // ── Integration tests for run_git_* functions ────────────────

    #[tokio::test]
    async fn test_run_git_head_and_write_tree_fingerprint() {
        let (_dir, repo_path) = init_temp_repo();
        let head1 = run_git_head(&repo_path).await.expect("repo has commits");
        let tree1 = run_git_write_tree(&repo_path)
            .await
            .expect("index writable");

        // Staged content change → new index tree, unchanged HEAD.
        std::fs::write(repo_path.join("test.txt"), b"line1\nline2\n").expect("write file");
        run_git_add_all(&repo_path).await.expect("git add");
        let head2 = run_git_head(&repo_path).await.expect("repo has commits");
        let tree2 = run_git_write_tree(&repo_path)
            .await
            .expect("index writable");
        assert_eq!(head1, head2, "staging must not change HEAD");
        assert_ne!(tree1, tree2, "staging must change the index tree");
    }

    #[tokio::test]
    async fn test_run_git_head_none_without_commits() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let repo_path = dir.path().to_path_buf();

        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&repo_path)
            .status()
            .expect("git init");
        assert!(status.success());

        let head = run_git_head(&repo_path).await.ok();
        assert!(head.is_none(), "commit-less repo must not resolve HEAD");
    }

    /// `git_has_commits` and `run_git_commit_message` both resolve against the
    /// same HEAD: no commit → neither answers; one commit → both do; a named
    /// hash reads that commit's message.
    #[tokio::test]
    async fn test_git_has_commits_and_commit_message_read_through_head() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let empty = dir.path().to_path_buf();
        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&empty)
            .status()
            .expect("git init");
        assert!(status.success());
        assert!(!git_has_commits(&empty).await, "no commits yet");
        assert!(
            run_git_commit_message(&empty, None).await.is_err(),
            "a commit-less repo has no message to read"
        );

        let (_dir, repo_path) = init_temp_repo();
        assert!(
            git_has_commits(&repo_path).await,
            "the fixture commits once"
        );
        assert_eq!(
            run_git_commit_message(&repo_path, None)
                .await
                .expect("commit message without hash"),
            "Initial commit"
        );

        commit_file(&repo_path, "test.txt", "Second commit");
        let second = rev_parse(&repo_path, "HEAD");
        assert_eq!(
            run_git_commit_message(&repo_path, Some(&second))
                .await
                .expect("commit message with hash"),
            "Second commit",
            "a named hash reads that commit's subject"
        );
    }

    #[tokio::test]
    async fn test_run_git_worktree_snapshot_clean_tree() {
        let (_dir, repo_path) = init_temp_repo();
        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(snap.stats.added, 0);
        assert_eq!(snap.stats.removed, 0);
        assert_eq!(snap.stats.huge_binary_file_count, 0);
        assert_eq!(snap.stats.omitted_file_count, 0);
        assert!(snap.changes.is_empty(), "clean tree has no changes");
        assert!(snap.file_statuses.is_empty(), "clean tree has no statuses");
        assert_eq!(
            snap.repo_root,
            std::fs::canonicalize(&repo_path).unwrap(),
            "the snapshot resolves the repository root"
        );
    }

    #[tokio::test]
    async fn test_run_git_worktree_snapshot_with_changes() {
        let (_dir, repo_path) = init_temp_repo();
        // Modify: change "line2" to "line2 modified", add "line4"
        std::fs::write(
            repo_path.join("test.txt"),
            b"line1\nline2 modified\nline3\nline4\n",
        )
        .expect("write modified file");
        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(snap.stats.added, 2, "two lines added (modified + new line)");
        assert_eq!(snap.stats.removed, 1, "one line removed (line2)");
        assert_eq!(snap.stats.huge_binary_file_count, 0);
        assert_eq!(
            snap.changes,
            vec![WorktreeChange {
                path: "test.txt".into(),
                old_path: None,
                added: 2,
                removed: 1,
                untracked: None,
            }]
        );
        assert_eq!(
            snap.file_statuses.get("test.txt"),
            Some(&GitFileStatus::Modified)
        );
    }

    /// A mode-only tracked change has no hunks and `0\t0` numstat; it is still
    /// one listed change, with no line counts.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_git_worktree_snapshot_lists_mode_only_change_without_counts() {
        use std::os::unix::fs::PermissionsExt;

        let (_dir, repo_path) = init_temp_repo();
        let path = repo_path.join("test.txt");
        let mut permissions = std::fs::metadata(&path)
            .expect("test file metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).expect("make test file executable");

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!((snap.stats.added, snap.stats.removed), (0, 0));
        assert_eq!(
            snap.changes,
            vec![WorktreeChange {
                path: "test.txt".into(),
                old_path: None,
                added: 0,
                removed: 0,
                untracked: None,
            }]
        );
    }

    #[tokio::test]
    async fn test_run_git_worktree_snapshot_counts_untracked_directory_files() {
        let (_dir, repo_path) = init_temp_repo();

        // Create an untracked directory with a normal file (3 lines) and a
        // binary file. `--untracked-files=all` lists both files individually.
        std::fs::create_dir_all(repo_path.join("new_dir")).expect("create directory");
        std::fs::write(
            repo_path.join("new_dir/a.rs"),
            b"fn foo() {\n    bar();\n}\n",
        )
        .expect("write normal file");
        std::fs::write(repo_path.join("new_dir/binary.bin"), b"line1\x00\n")
            .expect("write binary file");

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(
            snap.stats.added, 3,
            "should count lines from directory files"
        );
        assert_eq!(snap.stats.removed, 0, "no removed lines");
        assert_eq!(
            snap.stats.huge_binary_file_count, 1,
            "binary file inside the directory surfaces as huge/binary"
        );
        assert_eq!(
            snap.changes,
            vec![
                WorktreeChange {
                    path: "new_dir/a.rs".into(),
                    old_path: None,
                    added: 3,
                    removed: 0,
                    untracked: Some(UntrackedKind::Text),
                },
                WorktreeChange {
                    path: "new_dir/binary.bin".into(),
                    old_path: None,
                    added: 0,
                    removed: 0,
                    untracked: Some(UntrackedKind::Binary),
                },
            ]
        );
        assert!(
            !snap.changes.iter().any(|c| c.path == "new_dir/"),
            "the directory itself is never a change"
        );
    }

    /// Both ways an untracked file becomes "huge": a NUL byte (binary) and a
    /// file over [`MAX_UNTRACKED_SIZE`]. Neither is line-counted, and both feed
    /// `huge_binary_file_count`, which the churn measurement reads.
    #[tokio::test]
    async fn test_run_git_worktree_snapshot_marks_binary_and_oversized_untracked_files_huge() {
        let (_dir, repo_path) = init_temp_repo();
        std::fs::write(repo_path.join("binary.bin"), b"line1\nline2\x00\n")
            .expect("write binary file");
        let size = usize::try_from(MAX_UNTRACKED_SIZE).expect("cap fits usize") + 1;
        std::fs::write(repo_path.join("large.bin"), vec![b'a'; size]).expect("write large file");

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(snap.stats.added, 0, "neither file is line-counted");
        assert_eq!(snap.stats.removed, 0, "no removed lines");
        assert_eq!(
            snap.stats.huge_binary_file_count, 2,
            "both are surfaced as huge/binary"
        );
        assert_eq!(
            snap.changes,
            vec![
                WorktreeChange {
                    path: "binary.bin".into(),
                    old_path: None,
                    added: 0,
                    removed: 0,
                    untracked: Some(UntrackedKind::Binary),
                },
                WorktreeChange {
                    path: "large.bin".into(),
                    old_path: None,
                    added: 0,
                    removed: 0,
                    untracked: Some(UntrackedKind::TooLarge(
                        u64::try_from(size).expect("size fits u64")
                    )),
                },
            ]
        );
    }

    /// A subdirectory workspace still reads the whole repository and reports
    /// repo-root-relative paths.
    #[tokio::test]
    async fn test_run_git_worktree_snapshot_resolves_root_relative_paths_from_subdir() {
        let (_dir, repo_path) = init_temp_repo();
        let sub = repo_path.join("sub");
        std::fs::create_dir(&sub).expect("create subdir");
        std::fs::write(sub.join("inside.txt"), b"one\ntwo\n").expect("write inside file");
        std::fs::write(repo_path.join("at-root.txt"), b"root\n").expect("write root file");

        let snap = run_git_worktree_snapshot(&sub)
            .await
            .expect("worktree snapshot");
        let paths: Vec<&str> = snap.changes.iter().map(|c| c.path.as_str()).collect();
        assert!(
            paths.contains(&"sub/inside.txt"),
            "a subdirectory workspace still reports repo-root-relative paths: {paths:?}"
        );
        assert!(
            paths.contains(&"at-root.txt"),
            "the reading is repo-wide: {paths:?}"
        );
        assert_eq!(snap.stats.added, 3, "both files are counted");
        assert_eq!(snap.repo_root, std::fs::canonicalize(&repo_path).unwrap());
    }

    #[expect(clippy::cast_possible_wrap)] // MAX_UNTRACKED_FILES_COUNTED is a small compile-time cap
    #[tokio::test]
    async fn test_run_git_worktree_snapshot_bounded_file_count_cap() {
        let (_dir, repo_path) = init_temp_repo();

        // An untracked directory with more files than the per-refresh cap —
        // only the first `MAX_UNTRACKED_FILES_COUNTED` files are counted, the
        // rest are reported as omitted.
        std::fs::create_dir_all(repo_path.join("big_dir")).expect("create directory");
        for i in 0..(MAX_UNTRACKED_FILES_COUNTED + 50) {
            std::fs::write(
                repo_path.join("big_dir").join(format!("f{i}")),
                format!("x{i}\n"),
            )
            .expect("write counted file");
        }

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(
            snap.stats.added, MAX_UNTRACKED_FILES_COUNTED as i64,
            "only the first cap files are line-counted"
        );
        assert_eq!(snap.stats.removed, 0);
        assert_eq!(snap.stats.huge_binary_file_count, 0);
        assert_eq!(
            snap.changes.len(),
            MAX_UNTRACKED_FILES_COUNTED,
            "the manifest only carries files inside the cap"
        );
        assert_eq!(
            snap.stats.omitted_file_count,
            MAX_UNTRACKED_FILES_COUNTED + 50 - MAX_UNTRACKED_FILES_COUNTED,
            "files beyond the cap are reported as omitted"
        );
    }

    #[tokio::test]
    async fn test_run_git_worktree_snapshot_gitignore_respected_in_directory() {
        let (_dir, repo_path) = init_temp_repo();

        // .gitignore is committed (tracked) so it is not itself counted.
        std::fs::write(repo_path.join(".gitignore"), b"ignored.log\n").expect("write gitignore");
        run_git_command(&repo_path, &["add", ".gitignore"])
            .await
            .expect("git add .gitignore");
        run_git_command(&repo_path, &["commit", "-m", "add .gitignore"])
            .await
            .expect("git commit .gitignore");

        // An untracked directory with a kept file and a gitignored file.
        std::fs::create_dir_all(repo_path.join("new_dir")).expect("create directory");
        std::fs::write(repo_path.join("new_dir/keep.rs"), b"aa\nbb\n").expect("write kept file");
        std::fs::write(repo_path.join("new_dir/ignored.log"), b"ignored\n")
            .expect("write ignored file");

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(
            snap.stats.added, 2,
            "gitignored file inside a directory is not counted"
        );
        assert_eq!(snap.stats.removed, 0);
        assert_eq!(snap.stats.huge_binary_file_count, 0);
        assert_eq!(snap.changes.len(), 1, "only the kept file is a change");
        assert_eq!(snap.changes[0].path, "new_dir/keep.rs");
    }

    #[tokio::test]
    async fn test_run_git_worktree_snapshot_unborn_head() {
        // A freshly `git init`ed repo with no commits has no HEAD; numstat
        // against the empty tree still carries staged and untracked files.
        let dir = tempfile::tempdir().expect("create temp dir");
        let repo_path = dir.path().to_path_buf();
        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&repo_path)
            .status()
            .expect("git init");
        assert!(status.success());

        std::fs::write(repo_path.join("first.rs"), b"fn main() {}\n")
            .expect("write untracked file");
        std::fs::write(repo_path.join("staged.rs"), b"a\nb\n").expect("write staged file");
        run_git_command(&repo_path, &["add", "staged.rs"])
            .await
            .expect("git add staged file");

        let snap = run_git_worktree_snapshot(&repo_path)
            .await
            .expect("worktree snapshot");
        assert_eq!(
            snap.stats.added, 3,
            "staged and untracked lines both counted despite unborn HEAD"
        );
        assert_eq!(snap.stats.removed, 0);
        assert_eq!(snap.stats.huge_binary_file_count, 0);
        assert!(snap.unborn_head, "unborn HEAD must be flagged");
        assert_eq!(
            snap.file_statuses.get("first.rs"),
            Some(&GitFileStatus::Added),
            "untracked file status flows on an unborn repo"
        );
        let staged = snap
            .changes
            .iter()
            .find(|c| c.path == "staged.rs")
            .expect("staged file is in the change set");
        assert_eq!((staged.added, staged.removed), (2, 0));
        assert!(staged.untracked.is_none(), "a staged file is tracked");
        assert_eq!(
            snap.changes
                .iter()
                .find(|c| c.path == "first.rs")
                .and_then(|c| c.untracked),
            Some(UntrackedKind::Text)
        );
    }

    #[test]
    fn parse_numstat_z_handles_normal_binary_and_rename_records() {
        let stdout = "3\t1\tfile.txt\0-\t-\timage.bin\0\
                      2\t0\t\0old.txt\0new.txt\0";
        let entries = parse_numstat_z(stdout);
        assert_eq!(
            entries.len(),
            3,
            "trailing NUL and empty fields are ignored"
        );
        assert_eq!(entries[0].path, "file.txt");
        assert_eq!((entries[0].added, entries[0].removed), (3, 1));
        assert_eq!(entries[0].old_path, None);
        assert_eq!(entries[1].path, "image.bin");
        assert_eq!((entries[1].added, entries[1].removed), (0, 0));
        assert_eq!(entries[2].path, "new.txt");
        assert_eq!(entries[2].old_path.as_deref(), Some("old.txt"));
        assert_eq!((entries[2].added, entries[2].removed), (2, 0));
    }
    // ── Push destination resolution and sync ─────────────────────

    /// Create a bare repository under `root` and register it as remote `name`
    /// of `repo_path`; returns the bare repo's path.
    fn add_bare_remote(root: &Path, repo_path: &Path, name: &str) -> std::path::PathBuf {
        let bare = root.join(format!("{name}.git"));
        let status = std::process::Command::new("git")
            .args(["init", "--bare"])
            .arg(&bare)
            .status()
            .expect("git init --bare");
        assert!(status.success());
        let status = std::process::Command::new("git")
            .args(["remote", "add", name])
            .arg(&bare)
            .current_dir(repo_path)
            .status()
            .expect("git remote add");
        assert!(status.success());
        bare
    }

    /// Run a plain git command in `repo_path`, asserting it succeeded.
    fn git_ok(repo_path: &Path, args: &[&str]) -> std::process::Output {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(repo_path)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn rev_parse(repo_path: &Path, rev: &str) -> String {
        String::from_utf8_lossy(&git_ok(repo_path, &["rev-parse", rev]).stdout)
            .trim()
            .to_string()
    }

    /// Write `name` and commit it, so the checked-out branch gains a commit.
    fn commit_file(repo_path: &Path, name: &str, message: &str) {
        std::fs::write(repo_path.join(name), b"new\n").expect("write");
        git_ok(repo_path, &["add", "-A"]);
        git_ok(repo_path, &["commit", "-m", message]);
    }

    /// The remote git resolves `branch`'s upstream to, if it tracks one.
    async fn tracked(repo_path: &Path, branch: &str) -> Option<String> {
        branch_remotes(repo_path, branch)
            .await
            .expect("branch remotes")
            .upstream
    }

    /// The remote state of the branch checked out in `repo_path`.
    async fn remote_state(repo_path: &Path) -> GitRemoteState {
        run_git_remote_state(repo_path)
            .await
            .expect("remote state")
            .expect("a branch is checked out")
    }

    /// An upstream-less branch on a repo with one pushable remote publishes to
    /// it: the push carries every commit, records the upstream, and the report
    /// is a real push (never reported as nothing to do).
    #[tokio::test]
    async fn publish_untracked_branch_to_single_pushable_remote() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        let carried = count_commits(&repo_path, "HEAD").await.expect("count");

        let state = remote_state(&repo_path).await;
        assert_eq!(state.behind, 0);
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.commits, carried, "publish carries every commit");
        assert_eq!(target.publish, Some(true), "origin has no such branch yet");
        assert!(state.has_sync_work());

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert!(
            !report.output.trim().is_empty(),
            "a real push must report something"
        );
        assert!(
            report.output.contains(&branch),
            "the report names the branch: {}",
            report.output
        );
        git_ok(
            &origin,
            &["rev-parse", "--verify", &format!("refs/heads/{branch}")],
        );
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("origin"),
            "the publish records the upstream"
        );
        assert!(
            !remote_state(&repo_path).await.has_sync_work(),
            "in sync again: nothing left to offer"
        );

        // Once the branch tracks its new remote, a later commit is an ordinary
        // update-and-push to the same destination.
        commit_file(&repo_path, "after.txt", "after the publish");
        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.publish, Some(false));
        assert_eq!(target.commits, 1);
        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD")
        );
    }

    /// The count is what a press pushes: an update-and-push fast-forwards from the
    /// branch's own update source first, and the commits the fast-forward brings
    /// go to the destination too.
    #[tokio::test]
    async fn update_and_push_counts_the_commits_a_press_pushes() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "upstream", &branch]);
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        git_ok(&repo_path, &["push", "origin", &branch]);
        // The update source moves on, so a press fast-forwards the branch and then
        // pushes what it carries.
        git_ok(&repo_path, &["switch", "-c", "upstream-side"]);
        commit_file(&repo_path, "moved.txt", "upstream moved on");
        commit_file(&repo_path, "moved-again.txt", "upstream moved on again");
        git_ok(&repo_path, &["push", "upstream", &format!("HEAD:{branch}")]);
        git_ok(&repo_path, &["switch", &branch]);
        git_ok(
            &repo_path,
            &["remote", "set-url", "--push", "upstream", "no-push"],
        );
        git_ok(&repo_path, &["fetch", "upstream"]);

        let state = remote_state(&repo_path).await;
        assert_eq!(state.behind, 2);
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.publish, Some(false));
        assert_eq!(
            target.commits, 2,
            "the commits the fast-forward brings are pushed too"
        );

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD"),
            "origin received the fast-forwarded branch"
        );
    }

    /// The destination's copy is fetched when its tip is unknown here, so the count
    /// is exact even though no local ref has seen that tip.
    #[tokio::test]
    async fn destination_advanced_elsewhere_is_fetched_before_counting() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "origin", &branch]);
        // Another clone moves the destination on, so its tip is not in this
        // repository's object store.
        let elsewhere = remotes.path().join("elsewhere");
        git_ok(
            remotes.path(),
            &[
                "clone",
                "--quiet",
                origin.to_str().expect("origin path"),
                elsewhere.to_str().expect("elsewhere path"),
            ],
        );
        for (key, value) in [("user.name", "Elsewhere"), ("user.email", "e@test.com")] {
            git_ok(&elsewhere, &["config", key, value]);
        }
        git_ok(
            &elsewhere,
            &["checkout", "-B", &branch, &format!("origin/{branch}")],
        );
        commit_file(&elsewhere, "elsewhere.txt", "origin moved on");
        git_ok(&elsewhere, &["push", "origin", &format!("HEAD:{branch}")]);
        let elsewhere_head = rev_parse(&elsewhere, "HEAD");
        commit_file(&repo_path, "local.txt", "only the local branch has this");
        assert!(
            !rev_exists(&repo_path, &elsewhere_head)
                .await
                .expect("rev exists"),
            "the destination's new tip starts out unknown here"
        );

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.publish, Some(false));
        assert_eq!(target.commits, 1, "only the local commit is not on origin");
        assert_eq!(
            rev_parse(&repo_path, &format!("refs/remotes/origin/{branch}")),
            elsewhere_head,
            "the probe fetched the destination's copy"
        );
    }

    /// The destination is asked directly: a branch it already carries is never
    /// announced as a publish just because this repository has no local ref for
    /// it (a narrowed `remote.<name>.fetch` refspec never creates one).
    #[tokio::test]
    async fn destination_branch_is_detected_without_a_local_ref() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "origin", &branch]);
        git_ok(
            &repo_path,
            &["update-ref", "-d", &format!("refs/remotes/origin/{branch}")],
        );
        commit_file(&repo_path, "later.txt", "only the local branch has this");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(
            target.publish,
            Some(false),
            "origin carries the branch even though no local ref says so"
        );
        assert_eq!(target.commits, 1);

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("origin")
        );
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD")
        );
    }

    /// The probe matches the branch's own ref: a nested ref on the destination
    /// (`feature/<branch>`) is not mistaken for the branch itself.
    #[tokio::test]
    async fn nested_destination_branch_is_not_mistaken_for_ours() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(
            &repo_path,
            &[
                "push",
                "origin",
                &format!("HEAD:refs/heads/feature/{branch}"),
            ],
        );

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(
            target.publish,
            Some(true),
            "origin has no branch named {branch}"
        );
        assert_eq!(
            target.commits,
            count_commits(&repo_path, "HEAD").await.expect("count")
        );
    }

    /// A publish skips the update step: the destination has no such branch to
    /// update from, so the branch's commits go out as they are.
    #[tokio::test]
    async fn publish_skips_the_update_step() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "upstream", &branch]);
        // The upstream moves on, so a pull would change the branch.
        git_ok(&repo_path, &["switch", "-c", "upstream-side"]);
        commit_file(&repo_path, "remote.txt", "upstream moved on");
        git_ok(&repo_path, &["push", "upstream", &format!("HEAD:{branch}")]);
        git_ok(&repo_path, &["switch", &branch]);
        git_ok(
            &repo_path,
            &["remote", "set-url", "--push", "upstream", "no-push"],
        );
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        git_ok(&repo_path, &["fetch", "upstream"]);
        let published_head = rev_parse(&repo_path, "HEAD");

        let state = remote_state(&repo_path).await;
        assert_eq!(state.behind, 1);
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.publish, Some(true), "origin has no such branch yet");

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&repo_path, "HEAD"),
            published_head,
            "a publish does not update the branch"
        );
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            published_head,
            "origin received exactly what the branch carried"
        );
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("upstream"),
            "the fetch-only tracking is left alone"
        );
    }

    /// A branch tracking a push-capable remote keeps that remote as its
    /// destination, and has nothing to offer once in sync.
    #[tokio::test]
    async fn tracked_push_capable_remote_stays_the_destination() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let upstream = add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "upstream", &branch]);
        commit_file(&repo_path, "pushed.txt", "second");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "upstream");
        assert_eq!(target.publish, Some(false));
        assert_eq!(target.commits, 1);

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert!(
            report.output.contains(&branch),
            "a real push reports itself: {}",
            report.output
        );
        assert_eq!(
            rev_parse(&upstream, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD"),
            "the tracked remote received the commit"
        );
        assert!(
            !remote_state(&repo_path).await.has_sync_work(),
            "in sync: the control hides, as before"
        );
    }

    /// An unreachable destination does not hide the branch and does not claim a
    /// publish: the branch is reported with the commits it carries, its reachability
    /// is not asserted, and a press reports git's own reason.
    #[tokio::test]
    async fn unreachable_destination_still_offers_a_push() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        std::fs::remove_dir_all(&origin).expect("remove the remote");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.publish, None, "the destination was never asked");
        assert_eq!(
            target.commits,
            count_commits(&repo_path, "HEAD").await.expect("count"),
            "the branch is reported with the commits it carries"
        );
        assert!(state.has_sync_work());
    }

    /// The destination the control names is the one a plain `git push` reaches:
    /// a configured `branch.<name>.pushRemote` wins over the tracked remote.
    #[tokio::test]
    async fn configured_push_remote_is_the_named_destination() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "origin");
        let backup = add_bare_remote(remotes.path(), &repo_path, "backup");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "origin", &branch]);
        git_ok(&repo_path, &["push", "backup", &branch]);
        git_ok(
            &repo_path,
            &["config", &format!("branch.{branch}.pushRemote"), "backup"],
        );
        commit_file(&repo_path, "pushed.txt", "only the push remote lacks this");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "backup", "the control names the push remote");
        assert_eq!(target.publish, Some(false));
        assert_eq!(target.commits, 1);

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&backup, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD"),
            "the named destination received the commit"
        );
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("origin"),
            "the tracking is left alone"
        );
    }

    /// A push remote whose push destination is disabled is never used: the push
    /// goes to the tracked remote that accepts pushes.
    #[tokio::test]
    async fn fetch_only_push_remote_falls_back_to_the_tracked_remote() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let upstream = add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "origin", &branch]);
        // Git's own way of making a remote fetch-only, this time as the branch's
        // configured push remote.
        git_ok(
            &repo_path,
            &["remote", "set-url", "--push", "upstream", "no-push"],
        );
        git_ok(
            &repo_path,
            &["config", &format!("branch.{branch}.pushRemote"), "upstream"],
        );
        commit_file(
            &repo_path,
            "pushed.txt",
            "only the tracked remote lacks this",
        );

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(target.commits, 1);

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD")
        );
        let upstream_refs =
            String::from_utf8_lossy(&git_ok(&upstream, &["for-each-ref"]).stdout).to_string();
        assert!(
            upstream_refs.trim().is_empty(),
            "the fetch-only remote received nothing: {upstream_refs}"
        );
    }

    /// A branch tracking a fetch-only upstream still pushes to the push-capable
    /// remote and keeps its fetch-only tracking.
    #[tokio::test]
    async fn fetch_only_upstream_stays_tracked_while_pushing_to_origin() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "upstream", &branch]);
        // Git's own way of making a remote fetch-only.
        git_ok(
            &repo_path,
            &["remote", "set-url", "--push", "upstream", "no-push"],
        );
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        git_ok(&repo_path, &["push", "origin", &branch]);
        commit_file(&repo_path, "pushed.txt", "Only origin lacks this");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(
            target.publish,
            Some(false),
            "origin already carries the branch"
        );
        assert_eq!(target.commits, 1);

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none(), "origin accepts pushes");
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("upstream"),
            "a fetch-only tracking is left alone"
        );
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD"),
            "origin received the commit"
        );
    }

    /// A branch whose tracking ref is gone (deleted remote branch, pruned or
    /// narrowed fetch) still resolves: the destination is asked directly, and the
    /// press republishes without overwriting the branch's tracking config.
    #[tokio::test]
    async fn missing_tracking_ref_still_resolves_and_publishes() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "origin", &branch]);
        git_ok(&origin, &["config", "receive.denyDeleteCurrent", "ignore"]);
        git_ok(&repo_path, &["push", "origin", "--delete", &branch]);

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Target(target) = &state.push else {
            panic!("expected a push target, got {:?}", state.push);
        };
        assert_eq!(target.remote, "origin");
        assert_eq!(
            target.publish,
            Some(true),
            "origin no longer carries the branch"
        );

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(report.not_pushed.is_none());
        assert_eq!(
            rev_parse(&origin, &format!("refs/heads/{branch}")),
            rev_parse(&repo_path, "HEAD")
        );
        assert_eq!(
            tracked(&repo_path, &branch).await.as_deref(),
            Some("origin"),
            "the existing tracking config is left alone"
        );
    }

    /// A branch that tracks a fetch-only upstream is still updated from it when
    /// no remote can take a push: the update step is not a guess.
    #[tokio::test]
    async fn behind_branch_still_fast_forwards_when_nowhere_to_push() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "upstream");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "-u", "upstream", &branch]);
        // Two pushable remotes make the destination ambiguous.
        add_bare_remote(remotes.path(), &repo_path, "origin");
        add_bare_remote(remotes.path(), &repo_path, "backup");

        // Advance the upstream from a side branch, so the checked-out branch is
        // behind its own update source; only then make the remote fetch-only.
        git_ok(&repo_path, &["switch", "-c", "upstream-side"]);
        commit_file(&repo_path, "remote.txt", "upstream moved on");
        git_ok(&repo_path, &["push", "upstream", &format!("HEAD:{branch}")]);
        git_ok(&repo_path, &["switch", &branch]);
        git_ok(
            &repo_path,
            &["remote", "set-url", "--push", "upstream", "no-push"],
        );
        git_ok(&repo_path, &["fetch", "upstream"]);
        let behind_head = rev_parse(&repo_path, "HEAD");

        let state = remote_state(&repo_path).await;
        assert_eq!(state.behind, 1);
        assert!(matches!(state.push, GitPushDestination::Nowhere { .. }));
        assert!(state.has_sync_work());

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert!(
            report.not_pushed.is_some(),
            "nothing can take a push, and the report says so"
        );
        assert_ne!(
            rev_parse(&repo_path, "HEAD"),
            behind_head,
            "the branch was fast-forwarded from its own update source"
        );
        assert_eq!(
            rev_parse(&repo_path, "HEAD"),
            rev_parse(&repo_path, &format!("upstream/{branch}"))
        );
    }

    /// With no remote at all, nothing is guessed and the control says so.
    #[tokio::test]
    async fn no_pushable_remote_reports_nowhere() {
        let (_dir, repo_path) = init_temp_repo();
        let state = remote_state(&repo_path).await;
        let GitPushDestination::Nowhere { reason } = &state.push else {
            panic!("expected nowhere, got {:?}", state.push);
        };
        assert!(reason.contains("push"), "reason mentions pushes: {reason}");
        assert!(
            state.has_sync_work(),
            "the control says why rather than hiding"
        );

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert_eq!(report.not_pushed.as_deref(), Some(reason.as_str()));
        assert!(report.output.is_empty(), "nothing ran: {}", report.output);
    }

    /// Two pushable remotes are ambiguous — neither is picked, nothing is
    /// pushed, and the control still says why.
    #[tokio::test]
    async fn several_pushable_remotes_are_not_guessed() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        let origin = add_bare_remote(remotes.path(), &repo_path, "origin");
        let backup = add_bare_remote(remotes.path(), &repo_path, "backup");

        let state = remote_state(&repo_path).await;
        let GitPushDestination::Nowhere { reason } = &state.push else {
            panic!("expected nowhere, got {:?}", state.push);
        };
        assert!(
            reason.contains("origin") && reason.contains("backup"),
            "the reason names both remotes: {reason}"
        );
        assert!(
            state.has_sync_work(),
            "a branch with no single destination still shows the control"
        );

        let report = run_git_sync(&repo_path).await.expect("sync");
        assert_eq!(report.not_pushed.as_deref(), Some(reason.as_str()));
        assert!(report.output.is_empty(), "nothing ran: {}", report.output);
        for remote in [&origin, &backup] {
            let refs =
                String::from_utf8_lossy(&git_ok(remote, &["for-each-ref"]).stdout).to_string();
            assert!(refs.trim().is_empty(), "nothing was pushed: {refs}");
        }
    }

    /// Dummy push URLs (git's fetch-only convention) are unusable; every real
    /// transport and existing local path is usable.
    #[test]
    fn push_destination_accepts_pushes_distinguishes_dummy_urls() {
        let (_dir, repo_path) = init_temp_repo();
        for url in ["no-push", "no_push", "DISABLED", "", "  "] {
            assert!(
                !push_destination_accepts_pushes(&repo_path, url),
                "{url:?} must not accept pushes"
            );
        }
        for url in [
            "../remote.git",
            "/abs/x.git",
            "https://example.com/x.git",
            "git@example.com:x.git",
            "ssh://example.com/x.git",
            "ssh+git://example.com/x.git",
            "git+ssh://example.com/x.git",
            "codecommit://example/repo",
            "file:///tmp/x.git",
        ] {
            assert!(
                push_destination_accepts_pushes(&repo_path, url),
                "{url} must accept pushes"
            );
        }
        // A bare word only counts when it names an existing local path.
        std::fs::create_dir(repo_path.join("local-remote.git")).expect("create dir");
        assert!(push_destination_accepts_pushes(
            &repo_path,
            "local-remote.git"
        ));
    }

    /// A refused push surfaces git's own reason, not a generic failure.
    #[tokio::test]
    async fn refused_push_reports_git_reason() {
        let (_dir, repo_path) = init_temp_repo();
        let remotes = tempfile::tempdir().expect("remote dir");
        add_bare_remote(remotes.path(), &repo_path, "origin");
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        git_ok(&repo_path, &["push", "origin", &branch]);

        // A commit only the remote gets, then a different one locally — the
        // branches diverge and a plain push is refused.
        git_ok(&repo_path, &["switch", "-c", "remote-only"]);
        commit_file(&repo_path, "remote.txt", "remote only");
        git_ok(&repo_path, &["push", "origin", &format!("HEAD:{branch}")]);
        git_ok(&repo_path, &["switch", &branch]);
        commit_file(&repo_path, "local.txt", "local only");

        let err = run_git_sync(&repo_path)
            .await
            .expect_err("a diverged push must be refused")
            .to_string();
        assert!(
            err.contains("rejected") || err.contains("non-fast-forward"),
            "git's own refusal reason must survive: {err}"
        );
    }

    // ── porcelain listings — new/untracked file paths ─────────────

    /// Shared porcelain test input with a mix of untracked (`??`), staged-as-new
    /// (`A `), modified, and C-quoted special-character paths.
    const PORCELAIN_INPUT: &str = "\
?? new_file.rs
M  modified.rs
?? another_new.py
A  staged_new.js
?? dir/untracked.txt
 M working_tree_only.txt
?? temp.log
AM staged_then_modified.js
 A working_tree_new.txt
?? \"file\\tname.rs\"
A  \"staged\\\"file.js\"
?? \"file\\\\backslash.rs\"
";

    /// The two listings over one mixed reading, and the entries both must
    /// refuse. `parse_new_files_from_porcelain` takes untracked and
    /// staged-as-new records; `parse_untracked_from_porcelain` takes `??` only,
    /// so a staged file is not listed twice. Both unquote C-quoted paths and
    /// both report nothing for a reading with no new file — where a record is
    /// too short, does not unquote, or names a collapsed nested repository
    /// rather than a file.
    ///
    /// The byte-level unstaged check still counts an entry no parser accepts —
    /// see `has_unstaged_changes_cases`.
    #[test]
    fn porcelain_listings_classify_and_skip_unusable_entries() {
        assert_eq!(
            parse_new_files_from_porcelain(PORCELAIN_INPUT),
            [
                "new_file.rs",
                "another_new.py",
                "staged_new.js",
                "dir/untracked.txt",
                "temp.log",
                "staged_then_modified.js",
                "file\tname.rs",
                "staged\"file.js",
                "file\\backslash.rs",
            ],
            "untracked (`??`) and staged-as-new (`A`) records, C-quotes unquoted"
        );
        assert_eq!(
            parse_untracked_from_porcelain(PORCELAIN_INPUT),
            [
                "new_file.rs",
                "another_new.py",
                "dir/untracked.txt",
                "temp.log",
                "file\tname.rs",
                "file\\backslash.rs",
            ],
            "`??` records only, so staged-as-new files are not listed twice"
        );

        // Readings that list nothing: only modified/deleted records, and lines
        // too short to carry a path.
        for porcelain in [
            "M  modified.rs\n M working_tree_only.txt\nD  deleted.rs\n A working_tree_new.txt\n",
            "A",
            "A ",
            "?? ",
            "??",
        ] {
            assert!(
                parse_new_files_from_porcelain(porcelain).is_empty(),
                "no new file in {porcelain:?}"
            );
            assert!(
                parse_untracked_from_porcelain(porcelain).is_empty(),
                "no untracked file in {porcelain:?}"
            );
        }

        // A path that does not unquote is skipped by both listings.
        let unusable = "?? \"broken\\q.rs\"\n?? good.rs\nA  \"also\\qbroken.rs\"\n";
        assert_eq!(parse_untracked_from_porcelain(unusable), ["good.rs"]);
        assert_eq!(parse_new_files_from_porcelain(unusable), ["good.rs"]);

        // A collapsed nested repository is one `dir/` entry, not a file, and
        // must not reach the manifest (or be counted as an omitted file).
        assert_eq!(
            parse_untracked_from_porcelain("?? nested/\n?? real.rs\n"),
            ["real.rs"]
        );
    }

    // ── parse_git_status_porcelain — per-file status map ─────────

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_parse_git_status_porcelain() {
        struct Case {
            /// Short label for failure messages.
            name: &'static str,
            /// Raw git status --porcelain output.
            input: &'static str,
            /// Expected entries: (path, Some(status)) asserts the file has that
            /// status; (path, None) asserts the file is absent from the map.
            /// An empty slice asserts the entire map is empty.
            expected: &'static [(&'static str, Option<GitFileStatus>)],
        }
        let cases: &[Case] = &[
            Case {
                name: "unstaged modified file",
                input: " M src/main.rs\n",
                expected: &[("src/main.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "staged added file",
                input: "A  new_file.rs\n",
                expected: &[("new_file.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "untracked file",
                input: "?? new_file.rs\n",
                expected: &[("new_file.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "intent-to-add ( A) is added",
                input: " A intent.rs\n",
                expected: &[("intent.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "staged and unstaged modified (MM)",
                input: "MM both.rs\n",
                expected: &[("both.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "staged added + unstaged modified (AM)",
                input: "AM partial.rs\n",
                expected: &[("partial.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename (old -> new)",
                input: "R  old.rs -> new.rs\n",
                expected: &[("new.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename with arrow in old path",
                input: "R  \"old -> name.rs\" -> \"new -> name.rs\"\n",
                expected: &[("new -> name.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename from a quoted source to a plain destination",
                input: "R  \"a b.txt\" -> plain.txt\n",
                expected: &[("plain.txt", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename into a quoted destination containing the delimiter",
                input: "R  plain.txt -> \"new -> name.txt\"\n",
                expected: &[("new -> name.txt", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename carried in the worktree column",
                input: " R old.rs -> new.rs\n",
                expected: &[("new.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "copy skipped",
                input: "C  old.rs -> copy.rs\n",
                expected: &[],
            },
            Case {
                name: "unmerged add/add (AA) is added",
                input: "AA both.rs\n",
                expected: &[("both.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "added then deleted (AD) skipped",
                input: "AD gone.rs\n",
                expected: &[],
            },
            Case {
                name: "typechange and unmerged (UU) skipped",
                input: " T link\nUU conflict.rs\n",
                expected: &[],
            },
            Case {
                name: "untracked directory (trailing slash stripped)",
                input: "?? new_dir/\n",
                expected: &[("new_dir", Some(GitFileStatus::Added))],
            },
            Case {
                name: "quoted path with spaces",
                input: " M \"path with spaces.rs\"\n",
                expected: &[("path with spaces.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "deleted file (unstaged) skipped",
                input: " D gone.rs\n",
                expected: &[],
            },
            Case {
                name: "deleted file (staged) skipped",
                input: "D  gone.rs\n",
                expected: &[],
            },
            Case {
                name: "clean (unrecognized status) not present",
                input: "   clean.rs\n",
                expected: &[("clean.rs", None)],
            },
            Case {
                name: "multiple entries same file — modified wins over added",
                input: "A  dup.rs\n M dup.rs\n",
                expected: &[("dup.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "multiple entries same file — added sticks",
                input: "?? dup.rs\nA  dup.rs\n",
                expected: &[("dup.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "empty output",
                input: "",
                expected: &[],
            },
            Case {
                name: "mixed statuses",
                input: concat!(
                    " M src/main.rs\n",
                    "?? new_file.rs\n",
                    "A  staged.rs\n",
                    " D deleted.rs\n",
                ),
                expected: &[
                    ("src/main.rs", Some(GitFileStatus::Modified)),
                    ("new_file.rs", Some(GitFileStatus::Added)),
                    ("staged.rs", Some(GitFileStatus::Added)),
                    ("deleted.rs", None),
                ],
            },
        ];

        for case in cases {
            let map = parse_git_status_porcelain(case.input);

            if case.expected.is_empty() {
                assert!(
                    map.is_empty(),
                    "case '{}' (input={:?}): expected empty map, got {:#?}",
                    case.name,
                    case.input,
                    map
                );
            } else {
                let expected_count = case.expected.iter().filter(|(_, s)| s.is_some()).count();
                assert_eq!(
                    map.len(),
                    expected_count,
                    "case '{}' (input={:?}): map has unexpected entries",
                    case.name,
                    case.input,
                );
                for &(path, expected_status) in case.expected {
                    match expected_status {
                        Some(status) => {
                            assert_eq!(
                                map.get(path),
                                Some(&status),
                                "case '{}' (input={:?}): path={:?}",
                                case.name,
                                case.input,
                                path
                            );
                        }
                        None => {
                            assert!(
                                !map.contains_key(path),
                                "case '{}' (input={:?}): path={:?} should be absent, got {:?}",
                                case.name,
                                case.input,
                                path,
                                map.get(path)
                            );
                        }
                    }
                }
            }
        }
    }

    /// A rename into a name that itself contains the ` -> ` delimiter: git quotes
    /// only the destination, so splitting at the quoted side's own delimiter
    /// would key a bogus path.
    #[cfg(unix)]
    #[tokio::test]
    async fn git_status_rename_into_a_name_with_the_delimiter_keys_its_destination() {
        let (_dir, repo_path) = init_temp_repo();
        run_git_command(&repo_path, &["config", "status.renames", "true"])
            .await
            .unwrap();
        run_git_command(&repo_path, &["mv", "test.txt", "new -> name.txt"])
            .await
            .unwrap();

        let porcelain = run_git_status(&repo_path).await.unwrap();
        assert!(
            porcelain.starts_with("R  test.txt -> \"new -> name.txt\""),
            "git's quoting is what this pins, got {porcelain:?}"
        );

        let map = parse_git_status_porcelain(&porcelain);
        assert_eq!(
            map.get("new -> name.txt"),
            Some(&GitFileStatus::Modified),
            "got {map:?}"
        );
    }

    // ── parse_numstat_lines — numstat line parsing ──

    /// numstat line parsing: normal, binary, malformed-skip, and empty inputs.
    #[test]
    fn parse_numstat_lines_cases() {
        let cases: &[(&str, &str, Vec<NumstatEntry>)] = &[
            (
                "normal",
                "10\t3\tsrc/main.rs\n0\t1\tsrc/lib.rs\n42\t7\tCargo.toml\n",
                vec![
                    NumstatEntry {
                        additions: Some(10),
                        deletions: Some(3),
                        path: "src/main.rs".to_string(),
                    },
                    NumstatEntry {
                        additions: Some(0),
                        deletions: Some(1),
                        path: "src/lib.rs".to_string(),
                    },
                    NumstatEntry {
                        additions: Some(42),
                        deletions: Some(7),
                        path: "Cargo.toml".to_string(),
                    },
                ],
            ),
            (
                "binary",
                "-\t-\timage.png\n42\t7\tsrc/main.rs\n",
                vec![
                    NumstatEntry {
                        additions: None,
                        deletions: None,
                        path: "image.png".to_string(),
                    },
                    NumstatEntry {
                        additions: Some(42),
                        deletions: Some(7),
                        path: "src/main.rs".to_string(),
                    },
                ],
            ),
            (
                "skips_malformed",
                "\n\n10\t3\tsrc/main.rs\n\t\t\nnot-enough-fields\n",
                vec![NumstatEntry {
                    additions: Some(10),
                    deletions: Some(3),
                    path: "src/main.rs".to_string(),
                }],
            ),
            ("empty_string", "", vec![]),
            ("empty_lines", "\n\n\n", vec![]),
        ];
        for (name, input, expected) in cases {
            assert_eq!(parse_numstat_lines(input), *expected, "case: {name}");
        }
    }

    // ── run_git_discard — discard modifications ──

    /// Discard changes to a modified tracked file — restores it to HEAD.
    #[tokio::test]
    async fn test_run_git_discard_modified_file() {
        let (_dir, repo_path) = init_temp_repo();

        // Modify the tracked file.
        std::fs::write(repo_path.join("test.txt"), b"modified content\n")
            .expect("write modified file");

        // Confirm file is dirty.
        let status = run_git_command(&repo_path, &["status", "--porcelain", "test.txt"])
            .await
            .expect("status before discard");
        assert!(
            !status.trim().is_empty(),
            "file should be dirty before discard"
        );

        // Discard the modification.
        run_git_discard(&repo_path, "test.txt", DiscardTarget::File)
            .await
            .expect("run_git_discard should succeed");

        // File should be clean now.
        let status = run_git_command(&repo_path, &["status", "--porcelain", "test.txt"])
            .await
            .expect("status after discard");
        assert!(
            status.trim().is_empty(),
            "file should be clean after discard"
        );

        // Content should match the committed version.
        let content = std::fs::read_to_string(repo_path.join("test.txt")).expect("read file");
        assert_eq!(
            content, "line1\nline2\nline3\n",
            "content should be restored to HEAD"
        );
    }

    /// Untracked content is removed by a discard, in both target shapes: a new
    /// file, and a directory that goes recursively.
    #[tokio::test]
    async fn test_run_git_discard_untracked_file_and_directory() {
        let (_dir, repo_path) = init_temp_repo();
        let new_path = repo_path.join("new_file.rs");
        std::fs::write(&new_path, b"fn new() {}").expect("write new file");
        let sub_dir = repo_path.join("subdir");
        std::fs::create_dir(&sub_dir).expect("create subdir");
        let sub_file = sub_dir.join("nested.rs");
        std::fs::write(&sub_file, b"fn nested() {}").expect("write nested file");

        run_git_discard(&repo_path, "new_file.rs", DiscardTarget::File)
            .await
            .expect("run_git_discard should succeed");
        run_git_discard(&repo_path, "subdir", DiscardTarget::Directory)
            .await
            .expect("run_git_discard should succeed");

        assert!(
            !new_path.exists(),
            "new file should be removed after discard"
        );
        assert!(
            !sub_dir.exists(),
            "directory and its contents should be removed after discard"
        );
    }

    /// Diff-window paths are repo-root-relative. Discarding from a subdirectory
    /// workspace must target the root file, never a same-named file beneath it.
    #[tokio::test]
    async fn test_run_git_discard_from_subdirectory_targets_the_root_path() {
        let (_dir, repo_path) = init_temp_repo();
        let sub = repo_path.join("sub");
        std::fs::create_dir(&sub).expect("create subdir");
        let nested = sub.join("test.txt");
        std::fs::write(&nested, b"nested\n").expect("write nested file");
        std::fs::write(repo_path.join("test.txt"), b"modified\n").expect("modify root file");

        run_git_discard(&sub, "test.txt", DiscardTarget::File)
            .await
            .expect("run_git_discard should succeed");

        assert_eq!(
            std::fs::read_to_string(repo_path.join("test.txt")).expect("read root file"),
            "line1\nline2\nline3\n",
            "the root file is restored"
        );
        assert!(
            nested.exists(),
            "the same-named file under the workspace is untouched"
        );
    }

    /// The untracked mode is pinned on the status readers and the discard
    /// verification, so the owner's `status.showUntrackedFiles` cannot decide
    /// what the pipeline sees: with it off, untracked files stay visible to both
    /// readings and a leftover discard must still be reported as a failure.
    #[tokio::test]
    async fn status_readers_pin_the_untracked_mode() {
        let (_dir, repo_path) = init_temp_repo();
        std::fs::create_dir(repo_path.join("sub")).unwrap();
        std::fs::write(repo_path.join("sub/file.txt"), b"new\n").unwrap();
        // A nested repository, which `git clean -fd` deliberately keeps (it needs
        // `-ff`), so the discard below leaves an untracked entry behind.
        let nested = repo_path.join("nested");
        std::fs::create_dir(&nested).unwrap();
        run_git_command(&nested, &["init", "-q"]).await.unwrap();
        run_git_command(&repo_path, &["config", "status.showUntrackedFiles", "no"])
            .await
            .unwrap();

        let porcelain = run_git_status(&repo_path).await.unwrap();
        assert!(
            porcelain.lines().any(|line| line == "?? sub/"),
            "untracked files must stay visible with the owner's mode off:\n{porcelain}"
        );

        assert_eq!(
            list_new_or_untracked_files(&repo_path).await.unwrap(),
            ["nested/", "sub/"],
            "the listing must keep the collapsed directory entry, not read as empty"
        );

        let err = run_git_discard(&repo_path, "nested", DiscardTarget::Directory)
            .await
            .expect_err("a leftover must be reported, not passed off as a success");
        assert!(
            err.to_string().contains("Changes remain after discard"),
            "unexpected error: {err}"
        );
    }

    // ── has_unstaged_changes ─────────────────────────────────────────

    /// Empty, fully-staged, unstaged, dual-status, untracked, mixed,
    /// trailing-newline, and unparseable-path porcelain inputs.
    #[test]
    fn has_unstaged_changes_cases() {
        let cases: &[(&str, &str, bool)] = &[
            ("empty_string", "", false),
            ("empty_lines", "\n\n", false),
            ("fully_staged_modified", "M  Cargo.toml\n", false),
            ("fully_staged_added", "A  src/lib.rs\n", false),
            (
                "fully_staged_mixed",
                "M  Cargo.toml\nA  src/lib.rs\nD  old.rs\n",
                false,
            ),
            ("unstaged_modified", " M src/lib.rs\n", true),
            ("unstaged_deleted", " D src/old.rs\n", true),
            ("dual_status_modified", "MM src/lib.rs\n", true),
            ("dual_status_added", "AM src/new.rs\n", true),
            ("dual_status_deleted", "MD src/old.rs\n", true),
            ("untracked_file", "?? new_file.rs\n", true),
            ("untracked_directory", "?? dir/untracked.txt\n", true),
            (
                "mixed_only_staged",
                "M  Cargo.toml\nA  src/main.rs\n",
                false,
            ),
            (
                "mixed_staged_and_unstaged",
                "M  Cargo.toml\n M src/main.rs\n?? new.rs\n",
                true,
            ),
            ("trailing_newline_unstaged", " M file.rs\n", true),
            ("trailing_newline_staged", "M  file.rs\n", false),
            // A path no parser accepts must still count: this gate runs on
            // bytes, not records, so a malformed entry can never read as clean.
            ("unparseable_quoted_path", " M \"broken\\q.rs\"\n", true),
            (
                "unparseable_path_after_staged",
                "M  file.rs\n M \"broken\\q.rs\"\n",
                true,
            ),
        ];
        for (name, input, expected) in cases {
            assert_eq!(has_unstaged_changes(input), *expected, "case: {name}");
        }
    }
}

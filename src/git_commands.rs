//! Git subprocess wrappers — run git commands and parse porcelain/numstat output.
//!
//! Provides `run_git_*` async functions that operate on a repository path
//! and return results. For diff output parsing, see [`crate::diff_parse`].

use std::collections::HashSet;
use std::path::Path;
use tracing::warn;

use crate::tools::shell::apply_safe_env;
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

/// Run `git diff HEAD --no-color --find-renames` (when `commit_ref` is `None`)
/// or `git show -m <commit_ref> --no-color --find-renames --format=""` (when `Some`).
///
/// `-m` splits merge commits into per-parent diffs, producing standard unified diff
/// blocks that `parse_git_diff` can handle. This may produce duplicate file paths in the
/// tree sidebar (one `diff --git` block per parent), which is expected — each parent
/// comparison is a different diff.
/// `--format=""` suppresses the commit log header.
pub async fn run_git_diff(repo_path: &Path, commit_ref: Option<&str>) -> Result<String, String> {
    if let Some(hash) = commit_ref {
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
    } else {
        run_git_command(repo_path, &["diff", "HEAD", "--no-color", "--find-renames"]).await
    }
}

/// Run `git status --porcelain` and return the output.
pub async fn run_git_status(repo_path: &Path) -> Result<String, String> {
    run_git_command(repo_path, &["status", "--porcelain"]).await
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

/// Create a [`tokio::process::Command`] for `git` with a sanitized environment.
///
/// The subprocess environment is cleared and re-populated with only safe
/// environment variables (see [`apply_safe_env`]) to prevent credential
/// leakage (CWE-200). `LC_ALL=C` is set for consistent locale behavior
/// across all git invocations. This is the only entry point for production git
/// subprocess creation in this module — all callers must use this helper.
///
/// Callers should add further configuration (args, current_dir, stdio, etc.)
/// and then spawn or execute the command.
fn git_command() -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("git");
    apply_safe_env(&mut cmd);
    cmd.env("LC_ALL", "C");
    cmd
}

/// Run a git command without any interpretation of the exit code.
///
/// Shared by [`run_git_command`] and [`git_has_commits`] to avoid
/// duplicating the spawn + output + decode pattern. Returns the raw
/// [`std::process::Output`] so each caller can interpret the exit
/// status as appropriate.
///
/// **Environment sanitization**: The subprocess environment is cleared
/// and re-populated with only a safe set of environment variables
/// (see [`apply_safe_env`] for details). This prevents leaking API keys
/// and other secrets into child processes (CWE-200), but it also means
/// variables like `SSH_AUTH_SOCK` and `GIT_SSH_COMMAND` are **not**
/// inherited. This is consistent with the shell tool's behavior — use
/// SSH config (`~/.ssh/config`) for SSH-based git remotes rather than
/// environment variables.
pub(crate) async fn run_git_output(
    repo_path: &Path,
    args: &[&str],
) -> Result<std::process::Output, String> {
    let mut cmd = git_command();
    cmd.args(args).current_dir(repo_path);
    cmd.output()
        .await
        .map_err(|e| format!("Failed to run git: {e}"))
}

/// Run a git command and return stdout as string on success.
///
/// Returns an error if git exits with a non-zero status.
pub async fn run_git_command(repo_path: &Path, args: &[&str]) -> Result<String, String> {
    let output = run_git_output(repo_path, args).await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Git command failed: {stderr}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run a git command with data piped to stdin.
///
/// Like [`run_git_output`], but pipes the given lines to the subprocess's stdin
/// before collecting output. Returns the raw [`std::process::Output`] so
/// callers can interpret exit codes as appropriate for their use case.
///
/// The `name` parameter is used to identify the subcommand in error messages
/// (e.g., `"check-ignore"`).
///
/// **Environment sanitization**: Same as [`run_git_output`] — the subprocess
/// environment is cleared and re-populated with only a safe set of variables.
pub(crate) async fn run_git_with_stdin(
    repo_path: &Path,
    args: &[&str],
    stdin_lines: &[String],
    name: &str,
) -> Result<std::process::Output, String> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;

    let mut cmd = git_command();
    cmd.args(args)
        .current_dir(repo_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn git {name}: {e}"))?;

    // Write all lines to stdin, then close it.
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("Failed to capture stdin for git {name}"))?;
    if !stdin_lines.is_empty() {
        let input = stdin_lines.join("\n");
        stdin
            .write_all(input.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to git {name} stdin: {e}"))?;
    }
    drop(stdin);

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for git {name}: {e}"))?;

    Ok(output)
}

/// Run `git check-ignore --stdin` with the given file paths.
/// Pipes paths via stdin and returns the set of paths that are ignored.
///
/// Exit code 0 means some paths matched (output lists them, one per line).
/// Exit code 1 means no paths were ignored (not a failure — returns empty set).
/// Any other exit code is treated as an error.
pub async fn run_git_check_ignore(
    repo_path: &Path,
    paths: &[String],
) -> Result<HashSet<String>, String> {
    let output = run_git_with_stdin(
        repo_path,
        &["check-ignore", "--stdin"],
        paths,
        "check-ignore",
    )
    .await?;

    // Exit code 1 means "no files ignored" — not a failure, return empty set.
    if output.status.code() == Some(1) {
        return Ok(HashSet::new());
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Git check-ignore failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let ignored: HashSet<String> = stdout.lines().map(ToString::to_string).collect();
    Ok(ignored)
}

/// Stage all changes and commit with the given message.
/// Runs `git add -A` followed by `git commit -m "<msg>"`,
/// then captures the full SHA via `git rev-parse HEAD` and
/// line stats via `git diff --numstat`.
pub async fn run_git_commit(repo_path: &Path, message: &str) -> Result<CommitInfo, String> {
    // Stage all changes (tracked, untracked, removed) in the worktree.
    run_git_command(repo_path, &["add", "-A"]).await?;

    run_git_command(repo_path, &["commit", "-m", message])
        .await
        .map_err(|e| format!("Commit failed: {}", e.trim()))?;

    // Capture the full 40-char SHA — reliable source, not abbreviated.
    let hash = match run_git_command(repo_path, &["rev-parse", "HEAD"]).await {
        Ok(out) => out.trim().to_string(),
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
        if let Ok(stats) = parse_numstat(repo_path, &["diff", "--numstat", "HEAD~1..HEAD"]).await {
            stats
        } else {
            // HEAD~1 doesn't exist (first commit) — fall back to the empty tree hash.
            parse_numstat(
                repo_path,
                &[
                    "diff",
                    "--numstat",
                    "4b825dc642cb6eb9a060e54bf899dcee6a7b9e2a",
                    "HEAD",
                ],
            )
            .await
            .unwrap_or((0, 0))
        };
    Ok(CommitInfo {
        hash,
        lines_added,
        lines_removed,
    })
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

/// Run `git diff --numstat <args...>` and sum the line stats across all files.
///
/// Returns `Ok((lines_added, lines_removed))` on success (even if the diff
/// is empty). Returns the git error message on failure.
async fn parse_numstat(repo_path: &Path, args: &[&str]) -> Result<(i64, i64), String> {
    let stdout = run_git_command(repo_path, args).await?;

    let mut lines_added: i64 = 0;
    let mut lines_removed: i64 = 0;

    for entry in parse_numstat_lines(&stdout) {
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
    let mut cmd = git_command();
    cmd.arg("--version");
    cmd.output().await.is_ok_and(|o| o.status.success())
}

/// Check if a git repo has any commits.
pub async fn git_has_commits(repo_path: &Path) -> Result<bool, String> {
    let output = run_git_output(repo_path, &["rev-list", "-n", "1", "HEAD"]).await?;

    // If the command fails, there are no commits or something is wrong
    if !output.status.success() {
        return Ok(false);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(!stdout.trim().is_empty())
}

/// Get the current branch name (e.g. `main`, `feature/xyz`).
pub async fn run_git_current_branch(repo_path: &Path) -> Result<String, String> {
    run_git_command(repo_path, &["rev-parse", "--abbrev-ref", "HEAD"])
        .await
        .map(|s| s.trim().to_string())
}

/// Get behind/ahead counts against the upstream branch.
///
/// Returns `(behind, ahead)`. If there is no upstream configured, returns
/// `(0, 0)` without error.
pub async fn run_git_behind_ahead(repo_path: &Path) -> Result<(usize, usize), String> {
    match run_git_command(
        repo_path,
        &["rev-list", "--count", "--left-right", "HEAD...@{upstream}"],
    )
    .await
    {
        Ok(out) => {
            let parts: Vec<&str> = out.trim().split('\t').collect();
            if parts.len() == 2 {
                let ahead = parts[0].parse::<usize>().unwrap_or(0);
                let behind = parts[1].parse::<usize>().unwrap_or(0);
                Ok((behind, ahead))
            } else {
                Ok((0, 0))
            }
        }
        Err(e) if e.contains("no upstream") || e.contains("upstream") => Ok((0, 0)),
        Err(e) => Err(e),
    }
}

/// Run `git diff --numstat HEAD` and return the total added/removed lines.
///
/// Delegates to `parse_numstat`.
pub async fn run_git_diff_stats(repo_path: &Path) -> Result<(i64, i64), String> {
    parse_numstat(repo_path, &["diff", "--numstat", "HEAD"]).await
}

/// Sync with remote: `git pull --ff-only` then `git push`.
///
/// Returns the combined output of both commands.
pub async fn run_git_sync(repo_path: &Path) -> Result<String, String> {
    let pull_out = run_git_command(repo_path, &["pull", "--ff-only"]).await?;
    let push_out = run_git_command(repo_path, &["push"]).await?;
    let combined = if pull_out.trim().is_empty() {
        push_out
    } else if push_out.trim().is_empty() {
        pull_out
    } else {
        format!("{pull_out}\n{push_out}")
    };
    Ok(combined)
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
/// All errors from all three steps are absorbed. After the sequence, the working
/// tree is verified via `git status --porcelain -- <path>`: if the output is
/// empty the operation succeeded; otherwise the remaining changes are reported.
///
/// Note: `git reset HEAD` also exits with non-zero status when the path is
/// outside the repository — callers should validate paths before calling this.
pub async fn git_discard(
    repo_path: &Path,
    path: &str,
    target: DiscardTarget,
) -> Result<(), String> {
    let _ = run_git_command(repo_path, &["checkout", "HEAD", "--", path]).await;

    let _ = run_git_command(repo_path, &["reset", "HEAD", "--", path]).await;

    let clean_args: &[&str] = match target {
        DiscardTarget::Directory => &["clean", "-fd", "--", path],
        DiscardTarget::File => &["clean", "-f", "--", path],
    };
    let _ = run_git_command(repo_path, clean_args).await;

    // Verify: check if any changes remain.
    match run_git_command(repo_path, &["status", "--porcelain", "--", path]).await {
        Ok(status) if status.trim().is_empty() => Ok(()),
        Ok(status) => Err(format!("Changes remain after discard:\n{}", status.trim())),
        Err(e) => Err(format!("Discard ran but verification failed: {e}")),
    }
}

/// Get the last commit's message via `git log -1 --format=%s`.
///
/// If `commit_hash` is `Some`, get the message for that specific commit
/// instead of HEAD.
pub async fn run_git_commit_message(
    repo_path: &Path,
    commit_hash: Option<&str>,
) -> Result<String, String> {
    let mut args = vec!["log", "-1", "--format=%s"];
    if let Some(hash) = commit_hash {
        args.push(hash);
    }
    let out = run_git_command(repo_path, &args).await?;
    Ok(out.trim().to_string())
}

/// List new or untracked files in the working tree.
///
/// Delegates to [`run_git_status`] to run `git status --porcelain`, then passes
/// the output to [`parse_new_files_from_porcelain`] for parsing.
///
/// Catches both `??` (untracked) and any entry starting with `A` (staged as new,
/// including `A ` clean staged and `AM` staged+modified).
pub(crate) async fn list_new_or_untracked_files(repo_path: &Path) -> Result<Vec<String>, String> {
    let porcelain = run_git_status(repo_path).await?;
    Ok(parse_new_files_from_porcelain(&porcelain))
}

/// Shared helper for parsing file paths from `git status --porcelain` output.
///
/// Iterates over lines, applies `predicate` to select relevant entries, then
/// extracts the path portion (starting at index 3 after the 2-char status
/// prefix and space). Guarded with `get(3..)` to avoid panics on malformed
/// input (empty lines, truncated porcelain entries).
fn parse_porcelain_paths(porcelain: &str, predicate: impl FnMut(&&str) -> bool) -> Vec<String> {
    porcelain
        .lines()
        .filter(predicate)
        // Note: porcelain lines are at minimum 4 chars (<XY><space><path>), but
        // we guard with `get()` to prevent panics on malformed input.
        .filter_map(|line| {
            let path = line.get(3..)?;
            if path.is_empty() {
                None
            } else {
                Some(unquote_c_style(path).unwrap_or_else(|| path.to_string()))
            }
        })
        .collect()
}

/// Parse new/added file paths from `git status --porcelain` output.
///
/// Returns file paths for entries where the index status indicates a new file:
/// - `??` — untracked file
/// - `A ` at position 0 — staged as new (first char is `A`)
///
/// This correctly catches `A ` (staged, clean) and `AM` (staged as new, then
/// modified in working tree) because both start with `A`. The porcelain format
/// is `<XY><space><path>` where X = index status, Y = working tree status.
/// Path always starts at index 3.
///
/// Excludes ` A` (not tracked, added only to working tree — this is a file
/// that exists but is not tracked by git; it falls under `??` instead).
///
/// To parse only truly untracked files (those prefixed with `?? `), use
/// [`parse_untracked_from_porcelain`] instead.
#[must_use]
pub(crate) fn parse_new_files_from_porcelain(porcelain: &str) -> Vec<String> {
    parse_porcelain_paths(porcelain, |line| {
        line.starts_with("?? ") || line.starts_with('A')
    })
}

/// Parse only truly untracked file paths from `git status --porcelain` output.
///
/// Returns file paths for entries where the porcelain status is `?? ` (untracked
/// file not in the index). Unlike [`parse_new_files_from_porcelain`], this does
/// *not* include staged-as-new (`A `) files — it only catches entries starting
/// with `?? `.
///
/// Use this when you only want files that are truly untracked and do not want
/// overlap with staged-as-new files that might already be present from a
/// `git diff HEAD` parse.
#[must_use]
pub(crate) fn parse_untracked_from_porcelain(porcelain: &str) -> Vec<String> {
    parse_porcelain_paths(porcelain, |line| line.starts_with("?? "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::init_temp_repo;

    // ── Unit tests for CommitInfo::short_hash ────────────────────

    #[test]
    fn short_hash_truncates_long_hash() {
        let info = CommitInfo {
            hash: "abc1234def5678".to_string(),
            lines_added: 0,
            lines_removed: 0,
        };
        assert_eq!(info.short_hash(), "abc1234");
    }

    #[test]
    fn short_hash_returns_full_hash_when_short() {
        let info = CommitInfo {
            hash: "abc12".to_string(),
            lines_added: 0,
            lines_removed: 0,
        };
        assert_eq!(info.short_hash(), "abc12");
    }

    #[test]
    fn short_hash_returns_full_hash_when_exactly_7() {
        let info = CommitInfo {
            hash: "abc1234".to_string(),
            lines_added: 0,
            lines_removed: 0,
        };
        assert_eq!(info.short_hash(), "abc1234");
    }

    // ── Integration tests for run_git_* functions ────────────────

    #[tokio::test]
    async fn test_git_has_commits_true() {
        let (_dir, repo_path) = init_temp_repo();
        let has = git_has_commits(&repo_path).await.expect("git_has_commits");
        assert!(has, "repo with initial commit should have commits");
    }

    #[tokio::test]
    async fn test_git_has_commits_false() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let repo_path = dir.path().to_path_buf();

        // `git init` without any commit — empty repo
        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(&repo_path)
            .status()
            .expect("git init");
        assert!(status.success());

        let has = git_has_commits(&repo_path).await.expect("git_has_commits");
        assert!(!has, "empty repo should not have commits");
    }

    #[tokio::test]
    async fn test_run_git_current_branch_default() {
        let (_dir, repo_path) = init_temp_repo();
        let branch = run_git_current_branch(&repo_path).await.expect("branch");
        assert!(!branch.is_empty(), "branch name should not be empty");
    }

    #[tokio::test]
    async fn test_run_git_behind_ahead_no_upstream() {
        let (_dir, repo_path) = init_temp_repo();
        let (behind, ahead) = run_git_behind_ahead(&repo_path)
            .await
            .expect("behind/ahead");
        assert_eq!(behind, 0);
        assert_eq!(ahead, 0);
    }

    #[tokio::test]
    async fn test_run_git_diff_stats_clean_tree() {
        let (_dir, repo_path) = init_temp_repo();
        let (added, removed) = run_git_diff_stats(&repo_path).await.expect("diff stats");
        assert_eq!(added, 0);
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn test_run_git_diff_stats_with_changes() {
        let (_dir, repo_path) = init_temp_repo();
        // Modify: change "line2" to "line2 modified", add "line4"
        std::fs::write(
            repo_path.join("test.txt"),
            b"line1\nline2 modified\nline3\nline4\n",
        )
        .expect("write modified file");
        let (added, removed) = run_git_diff_stats(&repo_path).await.expect("diff stats");
        assert_eq!(added, 2, "two lines added (modified + new line)");
        assert_eq!(removed, 1, "one line removed (line2)");
    }

    #[tokio::test]
    async fn test_run_git_list_branches_single() {
        let (_dir, repo_path) = init_temp_repo();
        let out = run_git_command(&repo_path, &["branch", "--format=%(refname:short)"])
            .await
            .expect("list branches");
        let branches: Vec<String> = out.lines().map(ToString::to_string).collect();
        assert_eq!(branches.len(), 1, "single branch in new repo");
    }

    #[tokio::test]
    async fn test_run_git_switch_and_create_branch() {
        let (_dir, repo_path) = init_temp_repo();
        // Note the default branch name before creating a new one
        let default_branch = run_git_current_branch(&repo_path)
            .await
            .expect("current branch");

        // Create and switch to a new branch
        run_git_command(&repo_path, &["switch", "-c", "feature/test"])
            .await
            .expect("create branch");
        // Verify we're on the new branch
        let current = run_git_current_branch(&repo_path)
            .await
            .expect("current branch");
        assert_eq!(current, "feature/test");
        // Verify it appears in the branch list
        let out = run_git_command(&repo_path, &["branch", "--format=%(refname:short)"])
            .await
            .expect("list branches");
        let branches: Vec<String> = out.lines().map(ToString::to_string).collect();
        assert!(branches.contains(&"feature/test".to_string()));
        // Switch back to the default branch
        run_git_command(&repo_path, &["switch", default_branch.as_str()])
            .await
            .expect("switch back");
        let switched = run_git_current_branch(&repo_path)
            .await
            .expect("current branch");
        assert_eq!(switched, default_branch, "should be back on default branch");
    }

    #[tokio::test]
    async fn test_run_git_commit_message() {
        let (_dir, repo_path) = init_temp_repo();

        // Without hash — should return HEAD's message
        let msg = run_git_commit_message(&repo_path, None)
            .await
            .expect("commit message without hash");
        assert_eq!(msg, "Initial commit");

        // Create a second commit
        std::fs::write(repo_path.join("test.txt"), b"line1\nline2\n").expect("write test file");
        let status = std::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&repo_path)
            .status()
            .expect("git add");
        assert!(status.success());
        let status = std::process::Command::new("git")
            .args(["commit", "-m", "Second commit"])
            .current_dir(&repo_path)
            .status()
            .expect("git commit");
        assert!(status.success());

        // Get the second commit's hash
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo_path)
            .output()
            .expect("git rev-parse");
        let second_hash = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // With hash — should return that commit's message
        let msg = run_git_commit_message(&repo_path, Some(&second_hash))
            .await
            .expect("commit message with hash");
        assert_eq!(msg, "Second commit");

        // Without hash should now return HEAD (second commit)
        let msg = run_git_commit_message(&repo_path, None)
            .await
            .expect("commit message without hash");
        assert_eq!(msg, "Second commit");
    }

    #[tokio::test]
    async fn test_run_git_sync_no_remote() {
        let (_dir, repo_path) = init_temp_repo();
        // No remote configured — run_git_sync should return an error
        // from git pull --ff-only (no remote) rather than panicking.
        let result = run_git_sync(&repo_path).await;
        assert!(result.is_err(), "sync without remote should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("remote") || err.contains("push") || err.contains("pull"),
            "error should mention remote/push/pull: {err}"
        );
    }

    // ── parse_new_files_from_porcelain — combined untracked + staged-as-new ──

    /// Shared porcelain test input with a mix of untracked (`??`), staged-as-new
    /// (`A `), modified, and C-quoted special-character paths.  Used by both
    /// `parse_new_files_from_porcelain` and `parse_untracked_from_porcelain` tests.
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

    /// Verify that `parse_new_files_from_porcelain` correctly extracts untracked
    /// and staged-as-new file paths from git porcelain output, including C-quoted
    /// paths with special characters (tab, double-quote, non-ASCII).
    #[test]
    fn parse_new_files_from_porcelain_extracts_new_files() {
        let porcelain = PORCELAIN_INPUT;

        let files = parse_new_files_from_porcelain(porcelain);

        assert_eq!(files.len(), 9);
        assert!(files.contains(&"new_file.rs".to_string()));
        assert!(files.contains(&"another_new.py".to_string()));
        assert!(files.contains(&"staged_new.js".to_string()));
        assert!(files.contains(&"dir/untracked.txt".to_string()));
        assert!(files.contains(&"temp.log".to_string()));
        assert!(files.contains(&"staged_then_modified.js".to_string()));
        // C-quoted paths should be properly unquoted:
        assert!(files.contains(&"file\tname.rs".to_string()));
        assert!(files.contains(&"staged\"file.js".to_string()));
        // Backslash in filename (\\) unquotes to single backslash:
        assert!(files.contains(&"file\\backslash.rs".to_string()));
        // These should be excluded:
        assert!(!files.contains(&"modified.rs".to_string()));
        assert!(!files.contains(&"working_tree_only.txt".to_string()));
        assert!(!files.contains(&"working_tree_new.txt".to_string()));
    }

    /// Verify that porcelain parsing returns empty for both clean output
    /// (no new/untracked files) and malformed/truncated lines.
    #[test]
    fn parse_new_files_from_porcelain_returns_empty() {
        let porcelain = "\
M  modified.rs
 M working_tree_only.txt
D  deleted.rs
 A working_tree_new.txt
";

        let files = parse_new_files_from_porcelain(porcelain);
        assert!(
            files.is_empty(),
            "Should be empty when no new/untracked files"
        );

        // Malformed/truncated lines that match the filter but are too short
        // for path extraction should also produce empty results without panicking.
        let short_lines = ["A", "A ", "?? ", "??"];
        for &bad_line in &short_lines {
            let files = parse_new_files_from_porcelain(bad_line);
            assert!(
                files.is_empty(),
                "Malformed line {bad_line:?} should produce empty result, got {files:?}"
            );
        }

        // Also test the ??-only variant with the same short lines.
        for &bad_line in &short_lines {
            let files = parse_untracked_from_porcelain(bad_line);
            assert!(
                files.is_empty(),
                "Malformed line {bad_line:?} should produce empty result from ??-only parser, got {files:?}"
            );
        }
    }

    // ── parse_untracked_from_porcelain — ??-only untracked parsing ──

    /// Verify that `parse_untracked_from_porcelain` catches only `?? ` (truly
    /// untracked) entries and excludes `A ` (staged-as-new) files.
    #[test]
    fn parse_untracked_from_porcelain_returns_only_untracked() {
        let porcelain = PORCELAIN_INPUT;

        let files = parse_untracked_from_porcelain(porcelain);

        // Should include only ??-prefixed entries (6 total)
        assert_eq!(files.len(), 6);
        assert!(files.contains(&"new_file.rs".to_string()));
        assert!(files.contains(&"another_new.py".to_string()));
        assert!(files.contains(&"dir/untracked.txt".to_string()));
        assert!(files.contains(&"temp.log".to_string()));
        // C-quoted paths should be properly unquoted:
        assert!(files.contains(&"file\tname.rs".to_string()));
        assert!(files.contains(&"file\\backslash.rs".to_string()));
        // These should be excluded:
        assert!(!files.contains(&"staged_new.js".to_string()));
        assert!(!files.contains(&"staged_then_modified.js".to_string()));
        assert!(!files.contains(&"modified.rs".to_string()));
        assert!(!files.contains(&"working_tree_only.txt".to_string()));
        assert!(!files.contains(&"working_tree_new.txt".to_string()));
        assert!(!files.contains(&"staged\"file.js".to_string()));
    }

    /// Verify that `parse_untracked_from_porcelain` returns empty for output
    /// containing only staged-as-new or modified files (no ?? entries).
    #[test]
    fn parse_untracked_from_porcelain_no_untracked() {
        let porcelain = "\
M  modified.rs
A  staged_new.js
 M working_tree_only.txt
AM staged_then_modified.js
 A working_tree_new.txt
";

        let files = parse_untracked_from_porcelain(porcelain);
        assert!(
            files.is_empty(),
            "Should be empty when no `?? ` entries present"
        );
    }

    // ── parse_numstat_lines — numstat line parsing ──

    /// Normal file modifications.
    #[test]
    fn parse_numstat_lines_normal() {
        let output = "10\t3\tsrc/main.rs\n0\t1\tsrc/lib.rs\n42\t7\tCargo.toml\n";
        let entries = parse_numstat_lines(output);
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0],
            NumstatEntry {
                additions: Some(10),
                deletions: Some(3),
                path: "src/main.rs".to_string()
            }
        );
        assert_eq!(
            entries[1],
            NumstatEntry {
                additions: Some(0),
                deletions: Some(1),
                path: "src/lib.rs".to_string()
            }
        );
        assert_eq!(
            entries[2],
            NumstatEntry {
                additions: Some(42),
                deletions: Some(7),
                path: "Cargo.toml".to_string()
            }
        );
    }

    /// Binary files are represented as (None, None).
    #[test]
    fn parse_numstat_lines_binary() {
        let output = "-\t-\timage.png\n42\t7\tsrc/main.rs\n";
        let entries = parse_numstat_lines(output);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            NumstatEntry {
                additions: None,
                deletions: None,
                path: "image.png".to_string()
            }
        );
        assert_eq!(
            entries[1],
            NumstatEntry {
                additions: Some(42),
                deletions: Some(7),
                path: "src/main.rs".to_string()
            }
        );
    }

    /// Empty lines and malformed lines are silently skipped.
    #[test]
    fn parse_numstat_lines_skips_malformed() {
        let output = "\n\n10\t3\tsrc/main.rs\n\t\t\nnot-enough-fields\n";
        let entries = parse_numstat_lines(output);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            NumstatEntry {
                additions: Some(10),
                deletions: Some(3),
                path: "src/main.rs".to_string()
            }
        );
    }

    /// Empty output produces an empty vector.
    #[test]
    fn parse_numstat_lines_empty() {
        assert!(parse_numstat_lines("").is_empty());
        assert!(parse_numstat_lines("\n\n\n").is_empty());
    }

    // ── run_git_with_stdin — stdin-piping helper ──

    /// Verify that stdin piping works by hashing content via stdin.
    #[tokio::test]
    async fn test_run_git_with_stdin_pipes_stdin() {
        let (_dir, repo_path) = init_temp_repo();

        let output = run_git_with_stdin(
            &repo_path,
            &["hash-object", "--stdin"],
            &["hello world".to_string()],
            "hash-object",
        )
        .await
        .unwrap();

        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        // git hash-object --stdin outputs the SHA-1 hash followed by a newline
        assert!(!stdout.trim().is_empty(), "Expected a non-empty hash");
    }

    /// Verify empty stdin lines produce a valid (empty) output.
    #[tokio::test]
    async fn test_run_git_with_stdin_empty_lines() {
        let (_dir, repo_path) = init_temp_repo();

        let output = run_git_with_stdin(
            &repo_path,
            &["hash-object", "--stdin"],
            &[] as &[String],
            "hash-object",
        )
        .await
        .unwrap();

        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        // hash-object with empty/absent stdin still outputs a hash (empty blob hash)
        assert!(
            !stdout.trim().is_empty(),
            "Expected a non-empty hash for empty input"
        );
    }

    // ── run_git_check_ignore — .gitignore matching ──

    /// A path matching .gitignore should be reported as ignored.
    #[tokio::test]
    async fn test_run_git_check_ignore_matches_ignored_path() {
        let (_dir, repo_path) = init_temp_repo();

        std::fs::write(repo_path.join(".gitignore"), "*.log\n").unwrap();

        let ignored = run_git_check_ignore(&repo_path, &["test.log".to_string()])
            .await
            .unwrap();
        assert!(
            ignored.contains("test.log"),
            "test.log should be ignored by *.log pattern"
        );
    }

    /// A path not matching .gitignore should return an empty set.
    #[tokio::test]
    async fn test_run_git_check_ignore_non_ignored_path() {
        let (_dir, repo_path) = init_temp_repo();

        std::fs::write(repo_path.join(".gitignore"), "*.log\n").unwrap();

        let ignored = run_git_check_ignore(&repo_path, &["test.txt".to_string()])
            .await
            .unwrap();
        assert!(
            ignored.is_empty(),
            "test.txt should not be ignored by *.log pattern"
        );
    }

    /// Empty path list should return an empty set.
    #[tokio::test]
    async fn test_run_git_check_ignore_empty_paths() {
        let (_dir, repo_path) = init_temp_repo();

        let ignored = run_git_check_ignore(&repo_path, &[]).await.unwrap();
        assert!(
            ignored.is_empty(),
            "Empty path list should produce empty result"
        );
    }

    // ── git_discard — discard modifications ──

    /// Discard changes to a modified tracked file — restores it to HEAD.
    #[tokio::test]
    async fn test_git_discard_modified_file() {
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
        git_discard(&repo_path, "test.txt", DiscardTarget::File)
            .await
            .expect("git_discard should succeed");

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

    /// Discard a new untracked file — removes it from the working tree.
    #[tokio::test]
    async fn test_git_discard_new_file() {
        let (_dir, repo_path) = init_temp_repo();

        // Create a new untracked file.
        let new_path = repo_path.join("new_file.rs");
        std::fs::write(&new_path, b"fn new() {}").expect("write new file");
        assert!(new_path.exists(), "new file should exist before discard");

        // Discard the untracked file.
        git_discard(&repo_path, "new_file.rs", DiscardTarget::File)
            .await
            .expect("git_discard should succeed");

        // File should be removed.
        assert!(
            !new_path.exists(),
            "new file should be removed after discard"
        );
    }

    /// Discard a directory with untracked content — recursively removes it.
    #[tokio::test]
    async fn test_git_discard_directory() {
        let (_dir, repo_path) = init_temp_repo();

        // Create a directory with an untracked file inside.
        let sub_dir = repo_path.join("subdir");
        std::fs::create_dir(&sub_dir).expect("create subdir");
        let sub_file = sub_dir.join("nested.rs");
        std::fs::write(&sub_file, b"fn nested() {}").expect("write nested file");
        assert!(sub_file.exists(), "nested file should exist before discard");

        // Discard the directory.
        git_discard(&repo_path, "subdir", DiscardTarget::Directory)
            .await
            .expect("git_discard should succeed");

        // Directory and its contents should be gone.
        assert!(
            !sub_dir.exists(),
            "directory should be removed after discard"
        );
    }

    /// Discard a clean file — succeeds as a no-op.
    #[tokio::test]
    async fn test_git_discard_clean_file() {
        let (_dir, repo_path) = init_temp_repo();

        let result = git_discard(&repo_path, "test.txt", DiscardTarget::File).await;
        assert!(result.is_ok(), "discarding a clean file should succeed");
    }
}

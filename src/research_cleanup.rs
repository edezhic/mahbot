//! Research-run cleanup: per-run temp folders, results archive, the cleaner
//! ticket (sanitizer report), and the periodic sweeps (run roots + artist
//! generated/uploads keep-detection).
//!
//! ## Run roots
//!
//! Every deep-research run gets a per-run folder under
//! `<temp_dir()>/mahbot-research/{job_id}` — inside the readonly-shell's
//! allowed temp roots, so analysts can write scratch files there. The folder
//! is created idempotently at dispatch AND boot resume (a resume after an OS
//! temp cleanup recreates it; lost prototypes are fail-open). A `.keep` file
//! inside the folder disables sweeping (manual escape hatch).
//!
//! ## Sweeps
//!
//! `sweep_run_roots` deletes run folders whose jobs row is gone (no liveness)
//! and whose pending_jobs envelope is delivered or older than the 7-day cap.
//! `sweep_media` deletes generated/uploads files in userspaces that no Artist
//! session mentions (keep-detection is strictly session-based — solution 1).
//! Both fold their deletion counts into the cleanup loop's `Result<u64>`.
//!
//! ## Cleaner ticket
//!
//! At terminalization the run's accumulated shell commands are filtered to
//! outside-zone write-intent commands (everything outside the per-run folder,
//! including OS-temp scratch) and reported as ONE Backlog ticket
//! (`reporter = "cleaner"`, title marked `[cleanup {run_id}] ...` — dedup key).

use crate::Workspace;
use crate::board::{TicketParams, TicketPhase};
use crate::config;
use crate::turso::{self, params};
use crate::util::scrub_credentials;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

// ── Constants ─────────────────────────────────────────────────────────────

/// Freshness grace for run-root sweeps: a folder younger than this is never
/// swept (covers the mkdir → INSERT jobs race on dispatch).
const RUN_ROOT_GRACE: Duration = Duration::from_hours(1);
/// Hard cap on how long an undelivered pending_jobs envelope keeps a run
/// folder: a stuck envelope cannot hold the folder forever.
const PENDING_GUARD_CAP: Duration = Duration::from_hours(7 * 24);
/// Cleaner-ticket description cap (bytes). Over-long reports are truncated
/// with an explicit marker so the ticket body never swallows the Manager's
/// rendering budget.
const CLEANER_TICKET_DESC_CAP: usize = 32 * 1024;
const CLEANER_TRUNCATION_MARKER: &str = "... [обрезано]";
/// Per-tick artist-session scan budget (bytes of session content). Typical
/// bases (~3 MB) fit in one tick; pathological growth is cut across ticks.
const MEDIA_SCAN_BUDGET_BYTES: usize = 10 * 1024 * 1024;
/// Video extensions for case-insensitive keep-matching.
const MEDIA_VIDEO_EXTS: &[&str] = &[".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"];

// ── Run-root helpers ──────────────────────────────────────────────────────

/// Base directory holding all per-run folders. Deliberately under the daemon's
/// `temp_dir()` — the readonly-shell validator only permits scratch writes
/// under the OS temp roots (a `$TMPDIR`-relative path would land elsewhere:
/// the shell's TMPDIR is `/tmp` while the daemon's temp_dir() is
/// `/var/folders/...` on macOS).
#[must_use]
pub(crate) fn research_root_base() -> PathBuf {
    std::env::temp_dir().join("mahbot-research")
}

/// Absolute per-run folder path for a job.
#[must_use]
pub(crate) fn run_root_path(job_id: &str) -> PathBuf {
    research_root_base().join(job_id)
}

/// Create the per-run folder (idempotent — never delete+recreate: a boot
/// resume must keep whatever survived). Returns the CANONICAL absolute path:
/// on macOS `temp_dir()` is `/var/folders/...` which resolves to
/// `/private/var/folders/...` — the zone filter must compare against the same
/// form the shell/coder workspace sees (the workspace path is canonicalized).
pub(crate) async fn ensure_run_root(job_id: &str) -> PathBuf {
    let path = run_root_path(job_id);
    if let Err(e) = tokio::fs::create_dir_all(&path).await {
        tracing::warn!(job = %job_id, error = %e, "Failed to create run root — analyst scratch writes may fail");
    }
    tokio::fs::canonicalize(&path).await.unwrap_or(path)
}

/// Relative file paths under the run root — the report's prototype list.
/// A missing/empty folder yields an empty list (boot resume after an OS temp
/// cleanup loses prototypes — fail-open, noted in the report).
pub(crate) async fn run_root_files(job_id: &str) -> Vec<String> {
    let root = run_root_path(job_id);
    let mut out: Vec<String> = list_files(&root)
        .await
        .into_iter()
        .filter_map(|p| {
            p.strip_prefix(&root)
                .ok()
                .map(|r| r.to_string_lossy().into_owned())
        })
        .collect();
    out.sort();
    out
}

// ── Results archive (results.md) ──────────────────────────────────────────

/// Write the run's archived result to `<storage_root>/research/results/{run_id}.md`
/// (question + delivered result, including partial terminalizations).
/// Overwrites idempotently on resume; a failed write is fail-open (the
/// terminalization is never blocked on the archive). The storage root comes
/// from CONFIG when set (test stores point it at a temp dir — no test writes
/// into the real `~/.mahbot`), falling back to the default config dir.
pub(crate) async fn write_results_md(job_id: &str, question: &str, result: &str) {
    let root = config::CONFIG
        .try_storage_root()
        .or_else(|| config::default_config_dir().ok());
    let Some(root) = root else {
        tracing::warn!(job = %job_id, "results.md skipped — no config dir");
        return;
    };
    let dir = root.join("research").join("results");
    let path = dir.join(format!("{job_id}.md"));
    let content =
        format!("# Research {job_id}\n\n## Question\n\n{question}\n\n## Result\n\n{result}\n");
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        tracing::warn!(job = %job_id, error = %e, "results.md: failed to create archive dir");
    } else if let Err(e) = tokio::fs::write(&path, content).await {
        tracing::warn!(job = %job_id, error = %e, "Failed to write results.md — run result not archived");
    }
}

// ── Sanitizer: shell-command collection ───────────────────────────────────

/// Extract every shell tool-call command from a session history. The canonical
/// [`decode_native_history_message`](crate::session::decode_native_history_message)
/// already un-nests provider-wrapped JSON arguments — read them directly. Never
/// fails: unreadable sessions contribute nothing (fail-open).
fn commands_from_history(history: &[crate::ChatMessage]) -> Vec<String> {
    let mut out = Vec::new();
    for msg in history {
        let Some(decoded) = crate::session::decode_native_history_message(msg) else {
            continue;
        };
        let crate::session::DecodedNativeHistoryMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } = decoded
        else {
            continue;
        };
        for call in calls {
            // The persisted history holds the RAW model call — normalize the
            // same way execution does. Gate on the NAME first (no arg clone
            // for the common non-shell calls); shell calls then pay for the
            // arg-key remap (`cmd`/`script` → `command` — a name-only match
            // finds no `command` key in the raw args).
            if crate::tools::normalize_tool_name(&call.name) != "shell" {
                continue;
            }
            let (_, args) = crate::tools::normalize_tool_call(&call.name, call.arguments.clone());
            if let Some(cmd) = args.get("command").and_then(serde_json::Value::as_str) {
                out.push(cmd.to_string());
            }
        }
    }
    out
}

/// Collect outside-zone shell commands from the persisted sessions of the given
/// agents (incremental stage collection — early sessions of long runs are
/// TTL'd, so the research orchestrator collects right after each round). The
/// zone filter is applied HERE — at capture time — so in-zone commands never
/// bloat the checkpointed state blob (they can never appear in a report).
pub(crate) async fn collect_agent_shell_commands(
    agent_ids: &[String],
    run_root: &Path,
) -> Vec<String> {
    let store = crate::session::store();
    let mut out = Vec::new();
    for id in agent_ids {
        let history = store.load(id).await;
        out.extend(
            commands_from_history(&history)
                .into_iter()
                .filter(|c| is_outside_zone_write(c, run_root)),
        );
    }
    out
}

/// Write-intent command with ANY extractable target outside the per-run
/// folder. Unattributable/relative targets count as IN-zone: the coder's
/// relative writes (`cat > script.py` with cwd = run_root) must never land in
/// the report — reporting them inverts the accepted false-negative direction
/// into systematic noise. `> /dev/null` is not a write. In-zone writes (target
/// under the run root, raw or canonical form) are dropped. ALL cleanable
/// targets are checked: a shell truncates every redirect target, not just the
/// last — `cat > /tmp/a > {run_root}/b` truncates the outside /tmp/a even
/// though stdout lands in-zone, and `cmd1 > /tmp/out && cmd2 > {run_root}/in`
/// writes both. Reporting on any-outside is the safe (over-report) direction.
fn is_outside_zone_write(cmd: &str, run_root: &Path) -> bool {
    is_write_intent_command(cmd)
        && write_targets(cmd)
            .into_iter()
            .any(|t| t != "/dev/null" && !target_in_zone(&t, run_root))
}

/// Lexically resolve `.`/`..` path components (no fs access). `Path::starts_with`
/// compares raw components, so `{run_root}/../escape/f.txt` would otherwise
/// short-circuit the zone check as in-zone. A leading-`..` absolute path
/// (`/../etc/x`) stays ABSOLUTE — `PathBuf::pop()` on the root is a no-op — and
/// normalizes to `/etc/x`, which the zone check classifies outside-zone
/// (over-report, safe direction).
fn normalize_lexical(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True when the write target lies under the run root. Compares BOTH the raw
/// and the canonicalized form of the target: on macOS `/var` → `/private/var`,
/// so a coder writing the non-canonical form of the run root (`/var/folders/…`)
/// must not read as outside-zone (false-positive cleaner ticket). The target
/// may not exist yet (it is being written) — canonicalize the nearest existing
/// ancestor and re-join the remainder. Runs on the blocking pool: the capture
/// path is async and `canonicalize` is a blocking syscall.
///
/// Accepted residual: a SYMLINK inside the run root pointing outside escapes
/// the raw fast-path below (the lexical prefix check treats it as in-zone, and
/// the canonicalize walk only runs for the non-prefix case). Resolving every
/// in-zone symlink would require per-target lstat walks — a false-negative
/// (missed outside write) within the accepted heuristic class, safe direction.
fn target_in_zone(target: &str, run_root: &Path) -> bool {
    crate::util::with_block_in_place(|| {
        let target = normalize_lexical(Path::new(target));
        if target.starts_with(run_root) {
            return true;
        }
        // Canonicalize the longest existing prefix of the target (the file
        // itself usually does not exist yet) and compare that form against
        // the run root.
        let mut suffix: Vec<std::ffi::OsString> = Vec::new();
        let mut cur = target.as_path();
        loop {
            if let Ok(canon) = std::fs::canonicalize(cur) {
                let mut joined = canon;
                for s in suffix.iter().rev() {
                    joined.push(s);
                }
                return joined.starts_with(run_root);
            }
            let Some(name) = cur.file_name().map(std::ffi::OsStr::to_os_string) else {
                return false;
            };
            suffix.push(name);
            let Some(parent) = cur.parent() else {
                return false;
            };
            cur = parent;
        }
    })
}

// ── Sanitizer: write-intent + zone filtering ──────────────────────────────

/// Conservative write-intent patterns. False positives are kept in the report
/// (over-keep is the safe direction); relative-path writes without cwd
/// attribution are the accepted false-negative.
const WRITE_INTENT_PATTERNS: &[&str] = &[
    "cat >", "cat >>", "tee ", "cp ", "mv ", "mkdir ", "touch ", "install ", "curl -o", "curl -O",
    "wget ", "wget -O", "scp ", "unzip ", "tar -x", "tar x", "<<", ">>", "printf >", "echo >",
    "> /", "> ~", "> $",
];

fn is_write_intent_command(cmd: &str) -> bool {
    WRITE_INTENT_PATTERNS.iter().any(|p| cmd.contains(p))
}

/// First non-flag token after `pos` — operands of a verb with leading flags.
fn first_non_flag_after<'a>(toks: &[&'a str], pos: usize) -> Option<&'a str> {
    toks[pos + 1..]
        .iter()
        .copied()
        .find(|t| !t.starts_with('-'))
}

/// Best-effort extraction of ALL write targets (absolute paths only).
/// Relative targets cannot be attributed without the shell cwd — skipped
/// (accepted false-negative). Verb targets (tee/cp/mv/install/mkdir/touch/
/// tar/unzip/curl/wget) are collected alongside redirects: a shell truncates
/// every redirect target, so `cat > /tmp/a > {run_root}/b` writes BOTH files
/// and an in-zone verb target must not mask an outside redirect (or vice
/// versa). The caller applies the zone filter per target.
///
/// The all-targets guarantee is scoped to REDIRECTS: the verb blocks use
/// `position()` (first verb occurrence) and read operands to the end of the
/// command, so a verb-to-verb chain (`cp A /tmp/out && cp C {run_root}/in`)
/// reports only the last operand — accepted heuristic class (the same masking
/// closed for redirects remains for verb chains; chained commands are rare in
/// sanitizer-relevant history).
///
/// FROZEN heuristic — do not extend: every shell-grammar edge case added here
/// costs a review round, and the design scoped this as a conservative pattern
/// check with accepted false-negatives. Known residuals kept intentionally:
/// cwd-relative paths, `install -t` flag destinations, `grep -o/-O` over-report
/// noise, `&>` combined redirects, and verb-to-verb chains.
fn write_targets(cmd: &str) -> Vec<String> {
    let toks: Vec<&str> = cmd.split_whitespace().collect();
    let mut out = Vec::new();
    // Redirect forms (`> TARGET`, `>> TARGET`, `2> TARGET`, `2>> TARGET`) —
    // skip fd redirects (`2>&1`, `&>`, `>&`) and `/dev/null` (NOT a write — a
    // real write suffixed with `> /dev/null 2>&1`, the common silent-download
    // pattern, must report the real target, not /dev/null). Relative targets
    // yield None — keep scanning: an earlier relative redirect must not mask a
    // later absolute one (`echo hi > b > /tmp/x` reports /tmp/x).
    for (i, t) in toks.iter().enumerate() {
        if (*t == ">" || *t == ">>" || *t == "2>" || *t == "2>>")
            && !toks.get(i + 1).is_some_and(|n| n.starts_with('&'))
            && let Some(t) = toks.get(i + 1)
            && let Some(target) = clean_target(t)
            && target != "/dev/null"
        {
            out.push(target);
        }
        if *t == "cat>"
            && let Some(t) = toks.get(i + 1)
            && let Some(target) = clean_target(t)
            && target != "/dev/null"
        {
            out.push(target);
        }
        if let Some(rest) = t.strip_prefix("cat>")
            && let Some(target) = clean_target(rest)
            && target != "/dev/null"
        {
            out.push(target);
        }
    }
    // `tee TARGET` — first non-flag argument.
    if let Some(pos) = toks.iter().position(|t| *t == "tee")
        && let Some(t) = first_non_flag_after(&toks, pos)
        && let Some(target) = clean_target(t)
    {
        out.push(target);
    }
    // `cp/mv/install SRC... DST` — the destination is the LAST non-flag
    // operand (flags like `-r`/`-f`/`-m MODE`/`--preserve` precede them;
    // option VALUES like `-m 644` are non-flag tokens but never last).
    // Operands stop at the first redirect (`cp a b > /dev/null` must not
    // treat `/dev/null` as a cp destination).
    for verb in ["cp", "mv", "install"] {
        if let Some(pos) = toks.iter().position(|t| *t == verb) {
            let operands: Vec<&str> = toks[pos + 1..]
                .iter()
                .copied()
                .take_while(|t| !t.contains('>'))
                .filter(|t| !t.starts_with('-'))
                .collect();
            if let Some(t) = operands.last()
                && let Some(target) = clean_target(t)
            {
                out.push(target);
            }
        }
    }
    // `mkdir TARGET` / `touch TARGET` — first non-flag argument.
    for verb in ["mkdir", "touch"] {
        if let Some(pos) = toks.iter().position(|t| *t == verb)
            && let Some(t) = first_non_flag_after(&toks, pos)
            && let Some(target) = clean_target(t)
        {
            out.push(target);
        }
    }
    // `tar -x ... (-C|--directory) TARGET` — extraction directory.
    if let Some(pos) = toks.iter().position(|t| *t == "tar") {
        if let Some(dir) = toks[pos + 1..]
            .iter()
            .find_map(|t| t.strip_prefix("--directory="))
            && let Some(target) = clean_target(dir)
        {
            out.push(target);
        }
        if let Some(cpos) = toks[pos + 1..].iter().position(|t| *t == "-C")
            && let Some(t) = toks.get(pos + 1 + cpos + 1)
            && let Some(target) = clean_target(t)
        {
            out.push(target);
        }
    }
    // `unzip ARCHIVE -d TARGET`.
    if let Some(pos) = toks.iter().position(|t| *t == "unzip")
        && let Some(dpos) = toks[pos + 1..].iter().position(|t| *t == "-d")
        && let Some(t) = toks.get(pos + 1 + dpos + 1)
        && let Some(target) = clean_target(t)
    {
        out.push(target);
    }
    // `curl -o TARGET` / `wget -O TARGET`.
    if let Some(pos) = toks.iter().position(|t| *t == "-o" || *t == "-O")
        && let Some(t) = toks.get(pos + 1)
        && let Some(target) = clean_target(t)
    {
        out.push(target);
    }
    out
}

fn clean_target(tok: &str) -> Option<String> {
    // Trim BOTH leading and trailing quotes: `cat > "/tmp/x"` and
    // `cat > "$TMPDIR/x"` (idiomatic double-quoted form) carry the quote
    // through split_whitespace — a leading quote would read as relative.
    let single_quoted = tok.starts_with('\'');
    let cleaned = tok
        .trim_start_matches(['\'', '"'])
        .trim_end_matches([';', '|', '&', ')', '\'', '"']);
    let expanded = crate::util::expand_tilde(cleaned);
    // The session shell's TMPDIR is /tmp (a known constant — the daemon's
    // temp_dir() differs on macOS) and $HOME is the daemon user's home;
    // expand both so `$TMPDIR/x` resolves to the OS-temp category decision 6
    // mandates reporting instead of escaping as an unattributable (in-zone)
    // relative target. Braced and plain forms both resolve. Single-quoted
    // tokens are shell LITERALS (`cat > '$TMPDIR/x'` names a literal
    // `$TMPDIR/x`) — no expansion (an over-report of the OS-temp category);
    // `$HOME` is only expanded when set (an empty HOME would resolve
    // `$HOME/x` to `/x`, a false-positive root write).
    let expanded = if single_quoted {
        expanded.to_string_lossy().into_owned()
    } else {
        let home = std::env::var("HOME").unwrap_or_default();
        let home = if home.is_empty() { "$HOME" } else { &home };
        expanded
            .to_string_lossy()
            .replace("${TMPDIR}", "/tmp")
            .replace("$TMPDIR", "/tmp")
            .replace("${HOME}", home)
            .replace("$HOME", home)
    };
    let path = Path::new(&expanded);
    path.is_absolute()
        .then(|| path.to_string_lossy().to_string())
}

/// Escape backticks/newlines so a value renders inside the cleaner-ticket
/// body (or title) without breaking the markdown list or injecting raw
/// markdown — downstream pipeline agents read the body as task instructions.
fn escape_md(s: &str) -> String {
    s.replace('`', "\\`").replace('\n', "\\n")
}

/// Build the sanitizer report from PRE-FILTERED outside-zone write-intent
/// commands (capture-time [`collect_agent_shell_commands`] is the single
/// filter — re-applying it here would be a no-op second pass). `None` when
/// nothing was captured — no ticket is created.
fn build_cleanup_report(
    job_id: &str,
    ws_name: &str,
    question: &str,
    run_root: &Path,
    commands: &[String],
) -> Option<String> {
    if commands.is_empty() {
        return None;
    }
    let mut out = String::new();
    let run_root_str = run_root.display();
    // Same escaping as the command lines below: a multi-line/backtick-bearing
    // question or workspace name would break the markdown list structure in a
    // body that downstream pipeline agents read as task instructions (lower
    // risk than commands — Manager-authored — but the same vector).
    let ws_name_esc = escape_md(ws_name);
    let question_esc = escape_md(question);
    let _ = std::fmt::write(
        &mut out,
        format_args!(
            "Research run {job_id} (workspace: {ws_name_esc}) left write-intent activity \
             outside its per-run folder.\n\nQuestion: {question_esc}\n\nPer-run folder: \
             {run_root_str}\n\nOutside-zone write-intent commands (scrubbed, newest first):\n"
        ),
    );
    // Newest first: the ticket body truncates the BEGINNING of the rendered
    // report (truncate_bytes + marker), so the newest evidence must be at the
    // top — a truncated ticket must never show the oldest surviving commands
    // while dropping the most recent (the state blob keeps newest too).
    // Escaping: backticks/newlines from multi-line heredoc commands would
    // otherwise break the markdown list and inject raw markdown into a ticket
    // body that downstream pipeline agents read as task instructions
    // (credentials are scrubbed first, the container is not).
    for cmd in commands.iter().rev() {
        let escaped = escape_md(&scrub_credentials(cmd));
        let _ = std::fmt::write(&mut out, format_args!("- `{escaped}`\n"));
    }
    let _ = std::fmt::write(
        &mut out,
        format_args!(
            "\nNote: everything outside the per-run folder counts — including OS-temp \
             scratch. The per-run folder itself is swept separately after the delivery \
             grace; this ticket covers only the outside-zone writes above.\n"
        ),
    );
    Some(out)
}

// ── Cleaner ticket ────────────────────────────────────────────────────────

/// Create ONE Backlog ticket per research run when the sanitizer found
/// outside-zone writes. Deduped by the `[cleanup {run_id}]` title marker so a
/// crash between ticket creation and terminalization cannot double-create.
/// Board store uninitialized → skip with a log (same as purge).
pub(crate) async fn maybe_create_cleaner_ticket(
    job_id: &str,
    ws: &Workspace,
    question: &str,
    commands: &[String],
) -> Result<()> {
    let Some(board) = crate::board::BOARD.get() else {
        tracing::info!(job = %job_id, "Cleaner ticket skipped — board store not initialized");
        return Ok(());
    };
    let run_root = tokio::fs::canonicalize(run_root_path(job_id))
        .await
        .unwrap_or_else(|_| run_root_path(job_id));
    let Some(report) = build_cleanup_report(job_id, &ws.name, question, &run_root, commands) else {
        tracing::info!(job = %job_id, "No outside-zone writes — no cleaner ticket");
        return Ok(());
    };

    let marker = format!("[cleanup {job_id}]");
    let terminal = crate::board::TERMINAL_PHASES
        .iter()
        .map(|p| format!("'{}'", p.as_ref()))
        .collect::<Vec<_>>()
        .join(", ");
    // `instr` = literal substring match — a `LIKE '%...%'` would treat the `_`
    // in a NanoID job_id as a single-char wildcard and could false-match a
    // different run's ticket.
    let dup = board
        .conn
        .query_optional(
            &format!(
                "SELECT 1 FROM tickets WHERE instr(title, ?1) > 0 AND is_archived = 0 \
                 AND phase NOT IN ({terminal}) LIMIT 1"
            ),
            params![marker.clone()],
            |_| Ok::<(), anyhow::Error>(()),
        )
        .await?
        .is_some();
    if dup {
        tracing::info!(job = %job_id, "Cleaner ticket already open — skipping");
        return Ok(());
    }

    let mut description = report;
    if description.len() > CLEANER_TICKET_DESC_CAP {
        let keep = crate::util::truncate_bytes(&description, CLEANER_TICKET_DESC_CAP);
        description = format!("{keep}{CLEANER_TRUNCATION_MARKER}");
    }
    let params = TicketParams {
        // Workspace names are user-controlled — same escaping as the body.
        title: format!("{marker} Research cleanup: {}", escape_md(&ws.name)),
        description,
        workspace_name: ws.name.clone(),
        phase: TicketPhase::Backlog,
        prerequisites: Vec::new(),
        reporter: "cleaner".to_string(),
        embedding: None,
        priority: 1,
    };
    let id = board.create_ticket(&params).await?;
    tracing::info!(
        job = %job_id,
        ticket = %id,
        "Cleaner ticket created — full pipeline re-entry (accepted LLM-cost multiplier)"
    );
    Ok(())
}

// ── Sweep: run roots ──────────────────────────────────────────────────────

/// Sweep run-root folders: delete folders with no live jobs row whose
/// pending_jobs envelope is delivered (or older than the cap). Production
/// entry — base is the daemon's temp dir.
pub async fn sweep_run_roots() -> Result<u64> {
    sweep_run_roots_at(&research_root_base()).await
}

/// Run-root sweep over an explicit base (injectable for tests).
#[allow(clippy::too_many_lines)] // per-entry guards (liveness, grace, .keep) are sequential and inline
pub(crate) async fn sweep_run_roots_at(base: &Path) -> Result<u64> {
    let conn = &crate::session::store().conn;
    let mut deleted = 0u64;
    let Ok(mut entries) = tokio::fs::read_dir(base).await else {
        return Ok(0);
    };
    loop {
        // A single entry-read failure must not abort the whole tick (the OS
        // temp cleaner may remove a folder mid-iteration).
        let entry = match entries.next_entry().await {
            Ok(Some(e)) => e,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(error = %e, "Run-root sweep: read_dir entry failed — skipping");
                continue;
            }
        };
        let path = entry.path();
        // OS temp race between read_dir and stat — skip, never abort the tick.
        let Ok(file_type) = entry.file_type().await else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Some(job_id) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        // .keep escape hatch.
        if tokio::fs::try_exists(path.join(".keep"))
            .await
            .unwrap_or(false)
        {
            continue;
        }
        // Freshness grace — covers the mkdir → INSERT jobs race AND the
        // purge-vs-live-run hazard. Residual: only the top-level mtime is
        // watched (subdir writes don't bump it) — jobs-row liveness below is
        // the primary guard.
        let Ok(modified) = entry
            .metadata()
            .await
            .map(|m| m.modified().unwrap_or(SystemTime::UNIX_EPOCH))
        else {
            continue;
        };
        if SystemTime::now()
            .duration_since(modified)
            .map_or(true, |d| d < RUN_ROOT_GRACE)
        {
            continue;
        }
        // Liveness: a jobs row with id = folder name keeps the folder.
        let has_job = conn
            .query_optional(
                "SELECT 1 FROM jobs WHERE id = ?1",
                params![job_id.clone()],
                |_| Ok::<(), anyhow::Error>(()),
            )
            .await?
            .is_some();
        if has_job {
            continue;
        }
        // Undelivered pending envelope keeps the folder — hard 7-day cap.
        let pending_created: Option<String> = conn
            .query_optional(
                "SELECT created_at FROM pending_jobs WHERE id = ?1",
                params![job_id.clone()],
                |r| r.get::<String>(0),
            )
            .await?;
        if let Some(created) = pending_created {
            let keep = turso::parse_utc_timestamp(&created).ok().is_none_or(|dt| {
                SystemTime::now()
                    .duration_since(dt.into())
                    .map_or(true, |d| d < PENDING_GUARD_CAP)
            });
            if keep {
                continue;
            }
        }
        // Dead: remove search-engine registry entry + LMDB query-tracker tree,
        // then the folder itself. Failures retry on the next tick.
        if crate::search_engine::registry_initialized() {
            crate::search_engine::remove_engine(&job_id);
        }
        if let Some(root) = config::CONFIG.try_storage_root() {
            let tracker_dir = root.join("search").join(&job_id);
            match tokio::fs::remove_dir_all(&tracker_dir).await {
                Ok(()) => {}
                // The tracker dir only exists when a coder/searcher actually
                // ran — a dead run that never searched has none.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(job = %job_id, error = %e, "Run-root sweep: failed to remove query-tracker dir");
                }
            }
        }
        match tokio::fs::remove_dir_all(&path).await {
            Ok(()) => {
                deleted += 1;
                tracing::info!(job = %job_id, "Run-root swept");
            }
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "Run-root sweep: folder removal failed — retrying next tick");
            }
        }
    }
    Ok(deleted)
}

// ── Sweep: artist generated/uploads keep-detection ────────────────────────

/// Per-user keep-scan cursor (in-memory; a restart resets it — safe, because
/// files only become deletion candidates after a full coverage pass within
/// one process lifetime). tokio Mutex: the sweep holds the guard across
/// awaits (DB + fs) and the cleanup-loop future must stay `Send`.
static MEDIA_CURSORS: tokio::sync::Mutex<Option<HashMap<String, MediaCursor>>> =
    tokio::sync::Mutex::const_new(None);

#[derive(Default)]
struct MediaCursor {
    /// Agent id → `session_metadata.last_activity` at scan time. Keyed by
    /// content VERSION, NOT index or bare id: a session scanned on an earlier
    /// tick whose content grew (new file mentions) must be re-scanned before
    /// the deletion pass — deleting against a stale keep-set is the unsafe
    /// direction. An unchanged activity skips the DB re-read entirely (the
    /// per-tick scan budget stays available for other users — no starvation).
    /// `last_activity` is written by [`crate::turso::now`] (chrono AutoSi —
    /// fractional-second precision when non-zero), so the practical residual
    /// of an unchanged activity hiding a content append is sub-microsecond —
    /// far narrower than a whole-second window.
    scanned: HashMap<String, String>,
    /// Agent id → stripped content contribution. REPLACED on re-scan, never
    /// re-appended: `last_activity` bumps on every message append, so an
    /// active artist session would otherwise duplicate its whole history into
    /// the keep-set every tick — daemon memory growth plus a premature hit of
    /// the overflow cap that permanently disables the user's sweep.
    session_content: HashMap<String, String>,
    /// Entire session base scanned (files become candidates only then).
    /// Stays true after the deletion pass — the keep-set grows incrementally
    /// and the sweep never forces a full re-scan cycle.
    covered: bool,
    /// Rebuilt from `session_content` after each scan — the keep-set
    /// (paths/basenames are matched against it).
    content: String,
    /// Set when the accumulated content hit the hard cap — the keep-set is
    /// incomplete, so the deletion pass is SKIPPED (keep everything: the safe
    /// direction is never deleting a file whose mention was not scanned).
    overflowed: bool,
}

/// Reset the in-memory media-sweep cursors (tests only — production never
/// needs a reset: a restart clears the process-global anyway).
#[cfg(test)]
pub(crate) async fn reset_media_cursors() {
    if let Some(map) = MEDIA_CURSORS.lock().await.as_mut() {
        map.clear();
    }
}

/// Strip `data:` URI blobs from session content — a data URI is never a file
/// mention. Conservative: on regex trouble the blob stays (over-keep).
fn strip_data_uris(content: &str) -> String {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"data:[A-Za-z0-9+/=:;,._-]+").expect("data URI regex compiles")
    });
    RE.replace_all(content, "").into_owned()
}

/// Recursively enumerate REGULAR files under `dir` (missing dir → empty).
/// Entry-level symlinks are skipped ENTIRELY (no-follow): a symlinked
/// directory planted inside generated/uploads must not be traversed — the
/// deletion pass would otherwise collect (and later delete) real files
/// OUTSIDE the userspace root, violating "never delete a file whose mention
/// was not scanned" at the filesystem level. A top-level generated/uploads
/// dir that is itself a symlink is guarded by the CALLER (`symlink_metadata`
/// before the walk — `read_dir` would resolve it). Runs on the blocking pool
/// — the sweep runs on the async runtime.
async fn list_files(dir: &Path) -> Vec<PathBuf> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in rd.flatten() {
                // `DirEntry::file_type` does NOT follow symlinks — a symlink
                // reports as a symlink, not as its target's type, so both
                // symlinked dirs and files are skipped by the two arms below.
                let Ok(ft) = entry.file_type() else {
                    continue;
                };
                let path = entry.path();
                if ft.is_dir() {
                    walk(&path, out);
                } else if ft.is_file() {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(&dir, &mut out);
        out
    })
    .await
    .unwrap_or_default()
}

/// Artist sessions for a user: `session_metadata` rows with role=artist and
/// the user's name (agent id + last_activity, most-recent-first — the activity
/// doubles as the content-change signal for the scan cursor). A DB error is
/// returned (NOT swallowed as empty — an empty list would trivially satisfy
/// full coverage and trigger mass deletion against an empty keep-set).
async fn artist_session_ids(user_name: &str) -> anyhow::Result<Vec<(String, String)>> {
    let rows = crate::session::store()
        .conn
        .query(
            "SELECT agent_id, last_activity FROM session_metadata WHERE role = 'artist' \
             AND user_name = ?1 ORDER BY last_activity DESC",
            params![user_name],
        )
        .await?;
    // Fail CLOSED on a malformed row: silently dropping it would let full
    // coverage complete without scanning that session — its file mentions
    // would be missed and its files deleted (the same deletion-safety class
    // as the DB-error fail-open above).
    let mut out = Vec::with_capacity(rows.len());
    for r in &rows {
        let agent_id = r.get::<String>(0)?;
        let last_activity = r.get::<String>(1)?;
        out.push((agent_id, last_activity));
    }
    Ok(out)
}

/// One artist session's message content (fail-open only via the `Result` —
/// a DB read failure must not masquerade as an empty session: the sweep would
/// advance coverage and delete files the unread session mentions). Fail
/// CLOSED on a malformed row (same deletion-safety class as
/// [`artist_session_ids`]: a row failing to decode loses its mentions from
/// the keep-set while the session still counts as scanned).
async fn artist_session_content(agent_id: &str) -> anyhow::Result<String> {
    let rows = crate::session::store()
        .conn
        .query(
            "SELECT content FROM sessions WHERE agent_id = ?1 ORDER BY id ASC",
            params![agent_id],
        )
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for r in &rows {
        out.push(r.get::<String>(0)?);
    }
    Ok(out.join("\n"))
}

/// Sweep generated/uploads under `~/.mahbot/userspaces/` — production entry.
pub async fn sweep_media() -> Result<u64> {
    let root = config::default_config_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("mahbot_userspaces"))
        .join("userspaces");
    sweep_media_at(&root).await
}

/// Artist-media sweep over an explicit userspaces root (injectable for tests).
pub(crate) async fn sweep_media_at(userspaces_root: &Path) -> Result<u64> {
    sweep_media_at_budgeted(userspaces_root, MEDIA_SCAN_BUDGET_BYTES).await
}

/// Budgeted variant — tests inject a tiny per-tick budget to force the scan
/// across multiple ticks (full-coverage gating).
async fn sweep_media_at_budgeted(userspaces_root: &Path, budget_bytes: usize) -> Result<u64> {
    let mut deleted = 0u64;
    let mut budget = budget_bytes;
    let mut guard = MEDIA_CURSORS.lock().await;
    let cursor_map = guard.get_or_insert_with(HashMap::new);
    let Ok(mut user_dirs) = tokio::fs::read_dir(userspaces_root).await else {
        return Ok(0);
    };
    // Prune cursors of users whose userspace dir vanished — but only after a
    // COMPLETE pass: a budget-exhausted or entry-error tick saw only a prefix
    // of the dirs, so pruning against that partial set would drop live
    // cursors (a re-scan is harmless, but the cursor's coverage state is
    // real work thrown away).
    let mut seen_users: HashSet<String> = HashSet::new();
    let mut complete_pass = true;
    loop {
        let user_entry = match user_dirs.next_entry().await {
            Ok(Some(e)) => e,
            Ok(None) => break,
            // A single entry-read failure (OS temp race) must not abort the
            // whole multi-user tick — skip it, consistent with the per-entry
            // `continue` pattern used for file_type/metadata errors elsewhere.
            Err(e) => {
                complete_pass = false;
                tracing::warn!(error = %e, "Media sweep: read_dir entry failed — skipping");
                continue;
            }
        };
        let user_path = user_entry.path();
        let Ok(file_type) = user_entry.file_type().await else {
            // Transient stat error — this user was never scanned, so the pass
            // is NOT complete: pruning cursors now would drop a live one.
            complete_pass = false;
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Some(user_name) = user_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(std::string::ToString::to_string)
        else {
            // Unrepresentable dir name — never scanned, so the pass is not
            // complete (consistent with the entry-error and file_type-error
            // arms above).
            complete_pass = false;
            continue;
        };
        seen_users.insert(user_name.clone());
        // NOTE: no overflowed early-skip here — sweep_user_media must run so
        // the session-deletion reset (a /clear re-enables an overflowed user)
        // can fire; the overflowed check happens AFTER that reset, inside
        // sweep_user_media (returning before the scan/deletion phases).
        if budget == 0 {
            complete_pass = false;
            break; // budget exhausted — continue on the next tick
        }
        deleted += sweep_user_media(&user_name, &user_path, &mut budget, cursor_map).await;
    }
    if complete_pass {
        cursor_map.retain(|u, _| seen_users.contains(u));
    }
    Ok(deleted)
}

#[allow(clippy::too_many_lines)] // per-phase guards (fail-open, empty-base, overflow, budget, coverage) are sequential and inline
async fn sweep_user_media(
    user_name: &str,
    user_path: &Path,
    budget: &mut usize,
    cursors: &mut HashMap<String, MediaCursor>,
) -> u64 {
    let cursor = cursors.entry(user_name.to_string()).or_default();
    // Fail-open on DB trouble: a transient read failure must not be confused
    // with "no artist sessions" — the latter trivially satisfies coverage and
    // would delete EVERY file. Skip the user for this tick instead.
    let Ok(session_ids) = artist_session_ids(user_name).await else {
        tracing::warn!(user = %user_name, "Media sweep: artist-session query failed — user skipped for this tick");
        return 0;
    };
    // A DELETED session (e.g. /clear) invalidates the accumulated keep-set:
    // its mentions would otherwise keep files forever (accepted consequence
    // "/clear → файлы-кандидаты"). Reset the cursor so the remaining base is
    // re-scanned from scratch — reorders/additions never reset (incremental).
    // The overflowed flag is reset here too: a shrunk base can be re-scanned,
    // so /clear must re-enable an overflowed user's sweep (without this, the
    // flag would be permanently sticky for the process lifetime).
    let ids_now: HashSet<&str> = session_ids.iter().map(|(id, _)| id.as_str()).collect();
    if cursor
        .scanned
        .keys()
        .any(|id| !ids_now.contains(id.as_str()))
    {
        cursor.scanned.clear();
        cursor.session_content.clear();
        cursor.content.clear();
        cursor.covered = false;
        cursor.overflowed = false;
    }
    // No artist sessions → nothing was ever scanned → deleting everything
    // would violate the safe direction ("never delete a file whose mention
    // was not scanned"). Keep the user's files. Deliberate conflict with the
    // accepted "/clear → файлы-кандидаты" consequence: a FULL /clear (zero
    // sessions) is indistinguishable from a brand-new user, so its files are
    // kept too (safe direction wins — never delete on no evidence; the
    // rotation fires only when at least one session remains).
    if session_ids.is_empty() {
        tracing::info!(user = %user_name, "Media sweep: no artist sessions — files kept");
        return 0;
    }
    // Overflowed keep-set (incomplete evidence): the safe direction is never
    // deleting — skip the scan and deletion phases for this tick. The flag is
    // only cleared by a session deletion above (a /clear re-enables the
    // sweep); a content-shrink without deletion stays disabled.
    if cursor.overflowed {
        return 0;
    }

    // Scan phase: advance through the session base within the tick budget.
    // Sessions whose activity is unchanged since the last scan are skipped
    // WITHOUT a DB re-read — their mentions are already in the keep-set and
    // the budget stays available for other users. `total` tracks the keep-set
    // size incrementally (the content rebuild is deferred below, so
    // `cursor.content.len()` would be stale across multiple changes in one
    // tick).
    let mut total = cursor.content.len();
    let mut changed = false;
    for (id, activity) in &session_ids {
        if cursor
            .scanned
            .get(id)
            .is_some_and(|saved| saved == activity)
        {
            continue;
        }
        if *budget == 0 {
            break;
        }
        let Ok(content) = artist_session_content(id).await else {
            tracing::warn!(user = %user_name, "Media sweep: session read failed — user skipped for this tick");
            return 0;
        };
        *budget = budget.saturating_sub(content.len());
        let stripped = strip_data_uris(&content);
        // REPLACE this session's prior contribution (a grown re-scan must not
        // re-append its whole history — last_activity bumps per append, so an
        // active session would duplicate its mentions every tick and blow the
        // cap without a real change to the keep-set).
        let prior = cursor.session_content.get(id).map_or(0, String::len);
        let new_total = total - prior + stripped.len();
        if new_total > MEDIA_SCAN_BUDGET_BYTES * 4 {
            // Hard cap on the accumulated keep-set. The set is now
            // incomplete — mark overflowed so this user's sweep is
            // disabled (dropping old content could delete mentioned
            // files — the unsafe direction). Record the overflowing
            // session in `scanned` too: a later /clear of it must
            // trigger the deletion-reset (which clears `overflowed`),
            // otherwise the flag stays permanently sticky.
            cursor.overflowed = true;
            cursor.scanned.insert(id.clone(), activity.clone());
            break;
        }
        total = new_total;
        cursor.session_content.insert(id.clone(), stripped);
        cursor.scanned.insert(id.clone(), activity.clone());
        changed = true;
    }
    // Rebuild the keep-set ONCE after the scan loop (a per-session rebuild is
    // O(total) per change — quadratic across many changed sessions; bounded by
    // the overflow cap but still a needless copy every tick).
    if changed {
        cursor.content = cursor
            .session_content
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
    }
    // Full coverage only when EVERY session's stored activity matches the
    // current one — a session whose content grew but was not yet re-scanned
    // (tick budget ran out) keeps `covered` false: the deletion pass must
    // never run against a stale keep-set.
    cursor.covered = session_ids
        .iter()
        .all(|(id, activity)| cursor.scanned.get(id) == Some(activity));

    // Deletion phase: only after FULL coverage AND a complete keep-set. The
    // cursor is NOT reset after the pass — the keep-set grows incrementally
    // (new/changed sessions extend it), so the full session base is never
    // re-scanned per cycle and later users are never starved of the budget.
    // Standing cost (design-accepted): the deletion pass itself is unbudgeted
    // — it re-lists every generated/uploads file and runs one `contains` per
    // file every 10-min tick for every covered user (~seconds of CPU for
    // heavy users near the cap; tracked at rollout).
    let mut deleted = 0u64;
    if cursor.covered && !cursor.overflowed {
        // Basename uniqueness across this user's generated+uploads UNION
        // (exact, case-sensitive) — the only path for basename matching.
        let mut base_counts: HashMap<String, usize> = HashMap::new();
        let mut files: Vec<PathBuf> = Vec::new();
        for dir in ["generated", "uploads"] {
            let d = user_path.join(dir);
            // `symlink_metadata` (no-follow): a top-level generated/uploads
            // that is ITSELF a symlink must not be traversed — `read_dir`
            // resolves the top-level path, so the per-entry no-follow inside
            // `list_files` cannot catch it (files outside the userspace root
            // could be collected and deleted).
            let Ok(md) = tokio::fs::symlink_metadata(&d).await else {
                continue;
            };
            if md.file_type().is_symlink() {
                continue;
            }
            for f in list_files(&d).await {
                if let Some(b) = f.file_name().and_then(|n| n.to_str()) {
                    *base_counts.entry(b.to_string()).or_default() += 1;
                }
                files.push(f);
            }
        }
        // Lowercase the keep-set ONCE — the video case-insensitive fallback
        // reuses it for every file (no O(files × content) re-allocation).
        let content_lower = cursor.content.to_ascii_lowercase();
        for f in files {
            if is_mentioned(&f, &cursor.content, &content_lower, &base_counts) {
                continue;
            }
            match tokio::fs::remove_file(&f).await {
                Ok(()) => deleted += 1,
                Err(e) => {
                    tracing::warn!(file = %f.display(), error = %e, "Media sweep: delete failed");
                }
            }
        }
    }
    deleted
}

/// Keep-detection for one file: mentioned by basename when that basename is
/// unique in the user's generated+uploads union. An AMBIGUOUS basename
/// (duplicate across the union) is KEPT — the design's safe direction
/// ("иначе файл сохраняется — пере-держать"): a bare-basename mention cannot
/// be attributed to one of the duplicates, so deleting either could destroy a
/// mentioned file. Video extensions match case-insensitively (`content_lower`
/// is the precomputed lowercase keep-set — shared by every file).
/// Absolute-path mentions need no separate check: any path mention contains
/// the file's basename (subsumed by the basename check), and ambiguous
/// basenames are always kept by the rule above.
fn is_mentioned(
    file: &Path,
    content: &str,
    content_lower: &str,
    base_counts: &HashMap<String, usize>,
) -> bool {
    let Some(base) = file.file_name().and_then(|n| n.to_str()) else {
        return true; // unrepresentable name — keep (safe)
    };
    if base_counts.get(base) != Some(&1) {
        return true; // ambiguous basename — keep (over-keep is the safe direction)
    }
    if content.contains(base) {
        return true;
    }
    // Case-insensitive fallback for video extensions (suffixes are already
    // dotted — no per-file format! allocation).
    let lower_base = base.to_ascii_lowercase();
    MEDIA_VIDEO_EXTS.iter().any(|e| lower_base.ends_with(e)) && content_lower.contains(&lower_base)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Open the sessions/board stores in a temp dir for sweep tests.
    async fn init_stores() {
        crate::util::test::init_management_test_stores().await;
    }

    /// Media-sweep fixture: fresh stores + a temp userspaces root with the
    /// user's generated/uploads dirs created. Returns the userspaces root.
    async fn media_fixture(user: &str) -> tempfile::TempDir {
        init_stores().await;
        let userspaces = tempfile::tempdir().unwrap();
        for sub in ["generated", "uploads"] {
            tokio::fs::create_dir_all(userspaces.path().join(user).join(sub))
                .await
                .unwrap();
        }
        userspaces
    }

    /// Write placeholder files under `userspaces/{user}/generated`.
    async fn write_gen_files(userspaces: &Path, user: &str, names: &[&str]) -> Vec<PathBuf> {
        let gdir = userspaces.join(user).join("generated");
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            let p = gdir.join(n);
            tokio::fs::write(&p, "x").await.unwrap();
            out.push(p);
        }
        out
    }

    /// Insert a fake jobs row (or pending_jobs row) in the test sessions store.
    async fn insert_jobs_row(id: &str, now: &str) {
        crate::session::store()
            .conn
            .execute(
                "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
                 VALUES (?1, 'research', 'launched', '', 'ws', 'manager', ?2, ?2)",
                params![id, now],
            )
            .await
            .unwrap();
    }

    async fn insert_pending_row(id: &str, created: &str) {
        crate::session::store()
            .conn
            .execute(
                "INSERT INTO pending_jobs (id, target_agent_id, kind, envelope, role, created_at) \
                 VALUES (?1, 'manager_ws', 'research_result', '{}', 'manager', ?2)",
                params![id, created],
            )
            .await
            .unwrap();
    }

    fn stale_time() -> String {
        (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339()
    }

    fn stale_time_8d() -> String {
        (chrono::Utc::now() - chrono::Duration::days(8)).to_rfc3339()
    }

    #[tokio::test]
    async fn sweep_run_roots_keeps_live_and_fresh_and_keep_marked() {
        init_stores().await;
        let base = tempfile::tempdir().unwrap();
        let now = crate::turso::now();
        let live = base.path().join("run_live");
        let fresh = base.path().join("run_fresh");
        let kept = base.path().join("run_keep");
        let pending = base.path().join("run_pending");
        let dead = base.path().join("run_dead");
        let stuck = base.path().join("run_stuck");
        for p in [&live, &fresh, &kept, &pending, &dead, &stuck] {
            tokio::fs::create_dir_all(p).await.unwrap();
        }
        insert_jobs_row("run_live", &now).await;
        // run_pending: undelivered envelope within the cap → kept.
        insert_pending_row("run_pending", &stale_time()).await;
        // run_dead: delivered envelope (no pending row) → swept.
        // run_stuck: undelivered pending envelope OLDER than the 7-day cap → swept.
        insert_pending_row("run_stuck", &stale_time_8d()).await;
        // Old mtime on the folders that must NOT be grace-protected.
        let old = std::time::SystemTime::now() - Duration::from_hours(3);
        for p in [&kept, &pending, &dead, &stuck] {
            std::fs::File::open(p).unwrap().set_modified(old).unwrap();
        }
        // .keep escape hatch on an otherwise-dead folder.
        tokio::fs::write(kept.join(".keep"), "").await.unwrap();

        let n = sweep_run_roots_at(base.path()).await.unwrap();
        assert_eq!(n, 2, "the dead and over-cap-stuck run-roots are swept");
        assert!(live.exists(), "live jobs row keeps the folder");
        assert!(fresh.exists(), "freshness grace keeps the folder");
        assert!(
            kept.exists(),
            ".keep escape hatch protects an old dead folder"
        );
        assert!(
            pending.exists(),
            "undelivered envelope within the cap keeps the folder"
        );
        assert!(!dead.exists(), "dead run-root swept");
        assert!(
            !stuck.exists(),
            "stuck envelope beyond the 7-day cap is swept"
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_keeps_mentioned_and_removes_unmentioned_after_full_scan() {
        let userspaces = media_fixture("alice").await;
        let up = userspaces.path().join("alice").join("uploads");
        let files =
            write_gen_files(userspaces.path(), "alice", &["image_1.png", "image_2.png"]).await;
        let mentioned = &files[0];
        let orphan = &files[1];
        let upload_orphan = up.join("photo_1.jpg");
        tokio::fs::write(&upload_orphan, "x").await.unwrap();

        // One artist session mentioning the first file by absolute path.
        let session = format!("[IMAGE:{}]", mentioned.canonicalize().unwrap().display());
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, created_at, last_activity, user_name, workspace_name, role) \
             VALUES ('artist_a', ?1, ?1, 'alice', 'personal:alice', 'artist')",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) VALUES ('artist_a', 'assistant', ?1, ?2)",
            params![session, now],
        )
        .await
        .unwrap();

        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 2, "both unmentioned files deleted");
        assert!(mentioned.exists(), "mentioned file kept");
        assert!(!orphan.exists());
        assert!(!upload_orphan.exists());

        // A second pass after a NEW artist-session mention: the freshly
        // mentioned orphan stays; the still-unmentioned image_3 is deleted.
        tokio::fs::write(&orphan, "x").await.unwrap();
        let image3 = files[0].parent().unwrap().join("image_3.png");
        tokio::fs::write(&image3, "x").await.unwrap();
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES ('artist_a', 'assistant', ?1, ?2)",
            params![
                format!("[IMAGE:{}]", orphan.canonicalize().unwrap().display()),
                now
            ],
        )
        .await
        .unwrap();
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "only the unmentioned image_3 deleted");
        assert!(orphan.exists(), "now-mentioned orphan kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_deleted_session_rotates_files() {
        let userspaces = media_fixture("clr").await;
        let files = write_gen_files(userspaces.path(), "clr", &["f_a.png", "f_b.png"]).await;
        insert_artist_session(
            "artist_clr1",
            "clr",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_clr2",
            "clr",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        // Tick 1: both sessions scanned — both files mentioned, nothing deleted.
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 0);
        assert!(files[0].exists());
        assert!(files[1].exists());
        // /clear deletes artist_clr1's session: its stale mention must not
        // keep f_a forever — the cursor resets and f_a becomes a candidate.
        crate::session::store()
            .conn
            .execute(
                "DELETE FROM session_metadata WHERE agent_id = 'artist_clr1'",
                (),
            )
            .await
            .unwrap();
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "f_a's only mention was in the cleared session");
        assert!(!files[0].exists());
        assert!(
            files[1].exists(),
            "f_b still mentioned by the surviving session"
        );
    }

    #[test]
    fn sanitizer_zone_and_write_intent_filter() {
        let run_root = Path::new("/var/folders/xx/mahbot-research/run_x");
        // Absolute outside-zone writes are reported.
        assert!(is_outside_zone_write("cat > /tmp/scratch.txt", run_root));
        assert!(is_outside_zone_write(
            "curl -o /var/tmp/x.bin https://example.com/x",
            run_root
        ));
        assert!(is_outside_zone_write("cp -r /run_root/a /tmp/b", run_root));
        assert!(is_outside_zone_write("mkdir -p /tmp/scratch_dir", run_root));
        // In-zone absolute writes are dropped.
        assert!(!is_outside_zone_write(
            &format!("cat > {}/in_zone.txt", run_root.display()),
            run_root,
        ));
        assert!(!is_outside_zone_write(
            &format!("mkdir -p {}/subdir", run_root.display()),
            run_root,
        ));
        // Relative targets (coder cwd = run_root) are NOT outside — reporting
        // them inverts the accepted false-negative direction into noise.
        assert!(!is_outside_zone_write("cat > script.py", run_root));
        // `> /dev/null` is not a write.
        assert!(!is_outside_zone_write("echo hi > /dev/null", run_root));
        // Read-only commands are not write-intent.
        assert!(!is_outside_zone_write("grep -r foo src", run_root));
        // Canonicalized root matches canonicalized targets (/var → /private/var).
        let canon = Path::new("/private/var/folders/xx/mahbot-research/run_x");
        assert!(!is_outside_zone_write(
            "cat > /private/var/folders/xx/mahbot-research/run_x/f.txt",
            canon,
        ));
        // `> /dev/null` never masks a real outside write.
        assert!(is_outside_zone_write(
            "curl -o /var/tmp/x.bin https://example.com/x > /dev/null 2>&1",
            canon,
        ));
        assert!(!is_outside_zone_write(
            &format!("curl -o {}/f.bin https://x > /dev/null", canon.display()),
            canon,
        ));
        // $TMPDIR-expanded targets land in the OS-temp outside-zone category.
        assert!(is_outside_zone_write("cat > $TMPDIR/scratch.txt", canon));
        // Quoted and braced temp forms resolve the same way.
        assert!(is_outside_zone_write("cat > \"/tmp/scratch.txt\"", canon));
        assert!(is_outside_zone_write(
            "cat > \"$TMPDIR/scratch.txt\"",
            canon
        ));
        assert!(is_outside_zone_write("cat > ${TMPDIR}/scratch.txt", canon));
        // Single-quoted literals are NOT expanded (a literal `$TMPDIR/x`
        // filename is relative → in-zone, no over-report).
        assert!(!is_outside_zone_write("cat > '$TMPDIR/scratch.txt'", canon));
        // `..` traversal out of the run root is NOT in-zone (Path::starts_with
        // is lexical — the components are normalized first).
        assert!(is_outside_zone_write(
            &format!("cat > {}/../escape/f.txt", canon.display()),
            canon,
        ));
        assert!(!is_outside_zone_write(
            &format!("cat > {}/sub/../f.txt", canon.display()),
            canon,
        ));
        // ALL cleanable targets are checked: a shell truncates every redirect
        // target, so an in-zone last redirect must not mask an earlier outside
        // one — and vice versa.
        assert!(is_outside_zone_write(
            &format!("cat > {}/in.txt > /tmp/out.txt", canon.display()),
            canon,
        ));
        assert!(is_outside_zone_write(
            &format!("cat > /tmp/out.txt > {}/in.txt", canon.display()),
            canon,
        ));
        // Bare stderr redirects are write-intent end-to-end: the `"> /"`
        // WRITE_INTENT_PATTERNS entry is what gates `cmd 2> /tmp/x` through
        // to the target extraction (a pattern-list edit would silently drop
        // stderr writes from the cleaner report).
        assert!(is_outside_zone_write("cmd 2> /tmp/err.log", canon));
        assert!(
            !is_outside_zone_write(&format!("cmd 2> {}/err.log", canon.display()), canon,),
            "an in-zone stderr redirect stays in-zone"
        );
    }

    #[cfg(unix)]
    #[test]
    fn sanitizer_zone_canonical_asymmetry() {
        use std::os::unix::fs::symlink;
        // A write through a NON-canonical path form of the run root (symlink
        // alias) must still read as in-zone: the target's longest existing
        // ancestor is canonicalized before the comparison (the /var →
        // /private/var asymmetry on macOS).
        let td = tempfile::tempdir().unwrap();
        let real = td.path().join("real");
        std::fs::create_dir_all(&real).unwrap();
        let run_root = std::fs::canonicalize(&real).unwrap();
        let link = td.path().join("link");
        symlink(&real, &link).unwrap();
        assert!(!is_outside_zone_write(
            &format!("cat > {}/f.txt", link.display()),
            &run_root,
        ));
        // An outside write stays outside (no false in-zone).
        let outside = td.path().join("outside.txt");
        assert!(is_outside_zone_write(
            &format!("cat > {}", outside.display()),
            &run_root,
        ));
    }

    #[test]
    fn sanitizer_report_formats_and_scrubs_prefiltered_commands() {
        let run_root = Path::new("/tmp/mahbot-research/run_x");
        // Production flow feeds the report ONLY capture-time-filtered commands
        // (the zone/write-intent filter lives in `collect_agent_shell_commands`
        // — the report itself never re-filters; that overlap is covered by
        // `sanitizer_zone_and_write_intent_filter` above). Pass pre-filtered
        // input here.
        let filtered = vec!["cat > /tmp/scratch.txt".to_string()];
        let report = build_cleanup_report("run_x", "ws", "q?", run_root, &filtered);
        let report = report.expect("outside-zone writes exist");
        assert!(report.contains("/tmp/scratch.txt"));
        assert!(report.contains("newest first"));
        // Multi-line/backtick-bearing question and workspace name are escaped
        // the same way as commands (they render inside the same markdown body).
        let hostile = build_cleanup_report(
            "run_x",
            "ws`bad",
            "q1\n- injected list item\n`code`",
            run_root,
            &filtered,
        )
        .unwrap();
        assert!(
            hostile.contains("ws\\`bad"),
            "ws_name backtick escaped (backslash + backtick)"
        );
        assert!(
            hostile.contains("q1\\n- injected list item"),
            "question newline escaped to literal \\n — no raw line break to start a list item"
        );
        assert!(hostile.contains("\\`code\\`"), "question backticks escaped");
        // Credential scrubbing: a command with an api_key is hidden in the report.
        let secret =
            "curl -o /tmp/out.bin \"https://api.example.com/data?api_key=SECRET_API_KEY_123\""
                .to_string();
        let report = build_cleanup_report("run_x", "ws", "q?", run_root, &[secret]);
        let report = report.unwrap();
        assert!(!report.contains("SECRET_API_KEY_123"));
        // Nothing captured → no ticket.
        assert!(build_cleanup_report("run_x", "ws", "q?", run_root, &[]).is_none());
    }

    #[test]
    fn commands_from_history_normalizes_aliased_calls() {
        // Native assistant payloads with RAW model calls: alias names AND
        // non-canonical arg keys must both be normalized — execution remaps
        // `bash`/`run_terminal_cmd` → `shell` and `cmd`/`script` → `command`,
        // while the persisted history holds the raw model form (this exact
        // bug class cost two review rounds).
        let native = |calls: &str| {
            crate::ChatMessage::assistant(format!(r#"{{"content":"","tool_calls":{calls}}}"#))
        };
        let history = vec![
            native(r#"[{"id":"1","name":"bash","arguments":{"cmd":"cat > /tmp/x"}}]"#),
            native(r#"[{"id":"2","name":"run_terminal_cmd","arguments":{"script":"tee /tmp/y"}}]"#),
            native(r#"[{"id":"3","name":"shell","arguments":{"command":"mkdir -p /tmp/z"}}]"#),
            // Non-shell calls are gated by name before any arg work.
            native(r#"[{"id":"4","name":"search","arguments":{"query":"foo"}}]"#),
            // Provider-wrapped nested-JSON arguments are un-nested too.
            native(r#"[{"id":"5","name":"shell","arguments":"{\"command\":\"touch /tmp/w\"}"}]"#),
            crate::ChatMessage::user("not a tool call"),
        ];
        assert_eq!(
            commands_from_history(&history),
            vec![
                "cat > /tmp/x",
                "tee /tmp/y",
                "mkdir -p /tmp/z",
                "touch /tmp/w",
            ],
            "alias names and arg keys normalized; non-shell calls skipped"
        );
    }

    #[tokio::test]
    async fn cleaner_ticket_created_once_and_deduped() {
        init_stores().await;
        crate::util::test::create_test_workspace("/tmp/test_ws_cleaner", "test_ws").await;
        let ws = crate::workspace::test_ws_named("/tmp/test_ws_cleaner", "test_ws");
        let commands = vec!["cat > /tmp/scratch.txt".to_string()];

        maybe_create_cleaner_ticket("run_dedup", &ws, "q?", &commands)
            .await
            .unwrap();
        maybe_create_cleaner_ticket("run_dedup", &ws, "q?", &commands)
            .await
            .unwrap();

        let board = crate::board::BOARD.get().expect("board initialized");
        let rows = board
            .conn
            .query(
                "SELECT title, reporter FROM tickets WHERE workspace_name = 'test_ws'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "exactly one cleaner ticket per run");
        let title: String = rows[0].get(0).unwrap();
        let reporter: String = rows[0].get(1).unwrap();
        assert!(title.contains("[cleanup run_dedup]"), "{title}");
        assert_eq!(reporter, "cleaner");

        // No outside-zone writes → no ticket.
        maybe_create_cleaner_ticket("run_clean", &ws, "q?", &[])
            .await
            .unwrap();
        let rows = board
            .conn
            .query(
                "SELECT COUNT(*) FROM tickets WHERE instr(title, '[cleanup run_clean]') > 0",
                (),
            )
            .await
            .unwrap();
        assert_eq!(rows[0].get::<i64>(0).unwrap(), 0);

        // Dedup is LITERAL: a different run whose id differs only at the '_'
        // position must NOT be deduped by run_dedup's marker (a LIKE wildcard
        // would false-match it).
        maybe_create_cleaner_ticket("runXdedup", &ws, "q?", &commands)
            .await
            .unwrap();
        let rows = board
            .conn
            .query(
                "SELECT COUNT(*) FROM tickets WHERE workspace_name = 'test_ws'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            2,
            "distinct runs get distinct cleaner tickets"
        );
    }

    #[test]
    fn write_target_parses_redirects_and_verbs() {
        assert_eq!(write_targets("cat > /tmp/a.txt << EOF"), vec!["/tmp/a.txt"]);
        assert_eq!(write_targets("tee /var/tmp/x.log"), vec!["/var/tmp/x.log"]);
        assert_eq!(write_targets("cp /tmp/src /tmp/dst"), vec!["/tmp/dst"]);
        assert_eq!(
            write_targets("cp -r /tmp/src /tmp/dst"),
            vec!["/tmp/dst"],
            "flag-style cp — destination is the LAST operand"
        );
        assert_eq!(write_targets("mv -f /tmp/a /tmp/b"), vec!["/tmp/b"]);
        assert_eq!(
            write_targets("install -m 644 /tmp/a /tmp/b"),
            vec!["/tmp/b"],
            "option values (644) are not the destination"
        );
        assert_eq!(write_targets("mkdir -p /tmp/newdir"), vec!["/tmp/newdir"]);
        assert_eq!(write_targets("touch /tmp/marker"), vec!["/tmp/marker"]);
        assert_eq!(
            write_targets("tar -xzf a.tgz -C /tmp/dst"),
            vec!["/tmp/dst"]
        );
        assert_eq!(
            write_targets("tar -xzf a.tgz --directory=/tmp/dst"),
            vec!["/tmp/dst"]
        );
        assert_eq!(write_targets("unzip a.zip -d /tmp/dst"), vec!["/tmp/dst"]);
        assert_eq!(write_targets("tar -xzf a.tgz"), Vec::<String>::new());
        assert_eq!(
            write_targets("curl -o /tmp/y.bin https://x"),
            vec!["/tmp/y.bin"]
        );
        assert_eq!(write_targets("wget -O /tmp/z https://x"), vec!["/tmp/z"]);
        assert_eq!(write_targets("echo hi"), Vec::<String>::new());
        assert_eq!(
            write_targets("grep foo 2>&1"),
            Vec::<String>::new(),
            "fd redirect is not a write target"
        );
        assert_eq!(
            write_targets("echo hi > b > /tmp/x"),
            vec!["/tmp/x"],
            "a relative first redirect must not mask a later absolute one"
        );
        assert_eq!(
            write_targets("echo hi > b"),
            Vec::<String>::new(),
            "only-relative redirects have no absolute target"
        );
        // `/dev/null` is not a write — the real target after it is reported
        // (the common silent-download pattern).
        assert_eq!(
            write_targets("curl -o /tmp/y.bin https://x > /dev/null 2>&1"),
            vec!["/tmp/y.bin"],
            "> /dev/null must not short-circuit the real -o target"
        );
        assert_eq!(
            write_targets("cp /tmp/a /tmp/b > /dev/null"),
            vec!["/tmp/b"],
            "cp suffixed with > /dev/null reports the cp destination"
        );
        assert_eq!(
            write_targets("echo hi > /dev/null"),
            Vec::<String>::new(),
            "> /dev/null alone is not a write"
        );
        // $TMPDIR/$HOME expansion (session TMPDIR=/tmp, a known constant).
        assert_eq!(
            write_targets("cat > $TMPDIR/x.log"),
            vec!["/tmp/x.log"],
            "$TMPDIR resolves to the OS-temp category decision 6 mandates"
        );
        assert_eq!(write_targets("curl -o $TMPDIR/x https://y"), vec!["/tmp/x"]);
        // ALL cleanable targets are collected — a shell truncates every
        // redirect target, not just the last one.
        assert_eq!(
            write_targets("cat > /tmp/a > /tmp/b"),
            vec!["/tmp/a", "/tmp/b"],
            "both redirect targets are writes"
        );
        assert_eq!(
            write_targets("cat > /tmp/out && echo hi > /tmp/in"),
            vec!["/tmp/out", "/tmp/in"],
            "redirects across &&-chained commands are both collected"
        );
        // No-space `cat>` form.
        assert_eq!(write_targets("cat>/tmp/x"), vec!["/tmp/x"]);
        assert_eq!(write_targets("cat> /tmp/x"), vec!["/tmp/x"]);
        // Single-quoted literals are not expanded (no over-report).
        assert_eq!(
            write_targets("cat > '$TMPDIR/x'"),
            Vec::<String>::new(),
            "single quotes make $TMPDIR a literal filename (relative → in-zone)"
        );
        // Bare stderr redirects create/truncate a file — extracted too
        // (`2>&1` stays skipped: the `&`-prefixed next token).
        assert_eq!(
            write_targets("cmd 2> /tmp/err.log"),
            vec!["/tmp/err.log"],
            "a bare 2> redirect is a write target"
        );
        assert_eq!(
            write_targets("cmd 2>/tmp/err.log"),
            Vec::<String>::new(),
            "no-space 2> form is outside the accepted pattern set"
        );
    }

    /// Insert one artist session (metadata + one message) for a user.
    async fn insert_artist_session(agent_id: &str, user: &str, content: &str) {
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, created_at, last_activity, user_name, workspace_name, role) \
             VALUES (?1, ?2, ?2, ?3, ?4, 'artist')",
            params![agent_id, now.clone(), user, format!("personal:{user}")],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES (?1, 'assistant', ?2, ?3)",
            params![agent_id, content, now],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_gates_deletion_on_full_coverage() {
        let userspaces = media_fixture("gates").await;
        let files = write_gen_files(
            userspaces.path(),
            "gates",
            &["f_a.png", "f_b.png", "f_c.png"],
        )
        .await;
        // Each session mentions its own file; the per-tick budget (1 byte)
        // fits exactly one session per tick.
        insert_artist_session(
            "artist_g1",
            "gates",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_g2",
            "gates",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;

        // Tick 1: only artist_b (newest) is scanned — f_c is unmentioned but
        // must NOT be deleted until the whole session base is covered.
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 0, "no deletion before full coverage");
        assert!(files[2].exists());

        // Tick 2: the remaining session is scanned, coverage completes, and
        // only then does the deletion pass run.
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 1, "unmentioned f_c deleted after full coverage");
        assert!(files[0].exists());
        assert!(files[1].exists());
        assert!(!files[2].exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_cursor_survives_list_reorder() {
        let userspaces = media_fixture("reorder").await;
        let files = write_gen_files(userspaces.path(), "reorder", &["f_a.png", "f_b.png"]).await;
        insert_artist_session(
            "artist_r1",
            "reorder",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_r2",
            "reorder",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;

        // Tick 1: scans artist_b (newest) only.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0
        );

        // artist_a gets newer activity — the ordered list flips. The cursor is
        // keyed by scanned agent id, so artist_a is still scanned on tick 2
        // (an index cursor would re-scan artist_b and declare coverage with
        // artist_a's mention missing from the keep-set — deleting f_a).
        let conn = &crate::session::store().conn;
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_r1'",
            params![crate::turso::now()],
        )
        .await
        .unwrap();
        let n = sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap();
        assert_eq!(n, 0, "both files mentioned — nothing to delete");
        assert!(
            files[0].exists(),
            "f_a mentioned in the re-ordered scan kept"
        );
        assert!(files[1].exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_strips_data_uris() {
        let userspaces = media_fixture("duri").await;
        let files = write_gen_files(userspaces.path(), "duri", &["thumb.png"]).await;
        // The filename appears only INSIDE a data URI — not a file mention.
        insert_artist_session("artist_d1", "duri", "data:image/png;base64,AAAAthumb.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1);
        assert!(
            !files[0].exists(),
            "data-URI-embedded name is not a mention"
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_keep_set_is_per_user() {
        let userspaces = media_fixture("pualice").await;
        let alice_files = write_gen_files(
            userspaces.path(),
            "pualice",
            &["alice_1.png", "alice_2.png"],
        )
        .await;
        let bgen = userspaces.path().join("pubob").join("generated");
        tokio::fs::create_dir_all(&bgen).await.unwrap();
        let bf = bgen.join("bob_pic.png");
        tokio::fs::write(&bf, "x").await.unwrap();
        // bob's sessions mention alice's a1 by absolute path — that must NOT
        // keep alice's files (keep-sets are per-user).
        insert_artist_session("artist_iso1", "pualice", "a log line with no file mentions").await;
        insert_artist_session(
            "artist_iso2",
            "pubob",
            &format!(
                "[IMAGE:{}]",
                alice_files[0].canonicalize().unwrap().display()
            ),
        )
        .await;
        insert_artist_session(
            "artist_iso3",
            "pubob",
            &format!("[IMAGE:{}]", bf.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(
            n, 2,
            "alice's files deleted — bob's mention is out of zone; bob's own kept"
        );
        assert!(!alice_files[0].exists());
        assert!(!alice_files[1].exists());
        assert!(bf.exists(), "bob's mentioned file kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_empty_session_base_keeps_files() {
        let userspaces = media_fixture("guard").await;
        let files = write_gen_files(userspaces.path(), "guard", &["legacy.png"]).await;
        // Zero artist sessions: coverage would be vacuously true and delete
        // every file — but nothing was ever scanned, so nothing is deleted
        // (the safe direction). Brand-new users / legacy uploads are kept.
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "no artist sessions → no deletion");
        assert!(files[0].exists(), "unscanned files are never deleted");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_rescans_grown_session_before_deleting() {
        let userspaces = media_fixture("grow").await;
        let files = write_gen_files(
            userspaces.path(),
            "grow",
            &["f_a.png", "f_b.png", "f_c.png"],
        )
        .await;
        insert_artist_session(
            "artist_grow1",
            "grow",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        insert_artist_session(
            "artist_grow2",
            "grow",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        // Deterministic scan order: grow2 is the newest session (scanned first).
        let conn = &crate::session::store().conn;
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_grow1'",
            params![(chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339()],
        )
        .await
        .unwrap();
        reset_media_cursors().await;

        // Tick 1 (budget 1 byte): scans the newest session (grow2) only.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0,
            "no deletion before full coverage"
        );
        assert!(files[2].exists());

        // grow2's content GROWS after it was scanned: a new file f_c is
        // mentioned. Its last_activity bumps — the cursor must re-scan it
        // before the deletion pass, or f_c (mentioned only in the new tail)
        // would be deleted against the stale keep-set.
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES ('artist_grow2', 'assistant', ?1, ?2)",
            params![
                format!("[IMAGE:{}]", files[2].canonicalize().unwrap().display()),
                crate::turso::now()
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'artist_grow2'",
            params![crate::turso::now()],
        )
        .await
        .unwrap();

        // Tick 2 (budget 1 byte): grow2 (changed activity) is re-scanned —
        // its new mention reaches the keep-set; grow1 is not yet scanned so
        // coverage is incomplete and nothing is deleted.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0
        );
        assert!(
            files[2].exists(),
            "mention in the grown session not yet scanned"
        );
        // Re-scan REPLACES the session's prior contribution instead of
        // re-appending its whole history (last_activity bumps per append — an
        // active session would otherwise duplicate its mentions every tick and
        // blow the overflow cap): grow2's old mention appears exactly once.
        let cursors = MEDIA_CURSORS.lock().await;
        let c = cursors
            .as_ref()
            .expect("cursor map initialized")
            .get("grow")
            .expect("grow cursor");
        let fb_mention = format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display());
        assert_eq!(
            c.content.matches(&fb_mention).count(),
            1,
            "re-scan replaced, not re-appended"
        );
        drop(cursors);
        // Tick 3: grow1 is scanned, coverage completes, and f_c is kept.
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            0,
            "all three files mentioned — nothing to delete"
        );
        assert!(files[0].exists());
        assert!(files[1].exists());
        assert!(
            files[2].exists(),
            "file mentioned in the grown session kept"
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_no_rescan_cycle_does_not_starve_later_users() {
        let userspaces = media_fixture("starve1").await;
        let dirs = ["starve1", "starve2"];
        for u in dirs {
            if u != "starve1" {
                let gdir = userspaces.path().join(u).join("generated");
                tokio::fs::create_dir_all(&gdir).await.unwrap();
            }
            let orphan = format!("{u}_orphan.png");
            write_gen_files(userspaces.path(), u, &[&orphan]).await;
            insert_artist_session(
                &format!("artist_{u}"),
                u,
                "a log line with no file mentions",
            )
            .await;
        }
        reset_media_cursors().await;
        // Budget 1 byte fits exactly one session per tick. After a covered
        // user's deletion pass the cursor must NOT reset (a full re-scan
        // would consume the whole budget every tick and starve the second
        // user in read_dir order).
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            1
        );
        assert_eq!(
            sweep_media_at_budgeted(userspaces.path(), 1).await.unwrap(),
            1,
            "second user swept on the next tick — covered user consumed no budget"
        );
        for u in dirs {
            assert!(
                !userspaces
                    .path()
                    .join(u)
                    .join("generated")
                    .join(format!("{u}_orphan.png"))
                    .exists(),
                "{u}'s unmentioned file deleted"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_basename_unique_across_dirs() {
        let userspaces = media_fixture("base").await;
        let up = userspaces.path().join("base").join("uploads");
        tokio::fs::create_dir_all(&up).await.unwrap();
        let g = userspaces
            .path()
            .join("base")
            .join("generated")
            .join("pic.png");
        let u = up.join("pic.png");
        tokio::fs::write(&g, "x").await.unwrap();
        tokio::fs::write(&u, "x").await.unwrap();
        // A bare basename mention is ambiguous (duplicate across the
        // generated+uploads UNION) — the design's safe direction keeps BOTH
        // ("иначе файл сохраняется — пере-держать"): the mention cannot be
        // attributed to one duplicate, so deleting either could destroy a
        // mentioned file.
        insert_artist_session("artist_b1", "base", "here is pic.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "ambiguous basename keeps both files");
        assert!(g.exists());
        assert!(u.exists());

        // An absolute mention of one duplicate still keeps the OTHER: its
        // basename remains ambiguous across the union (the absolute path
        // matching is per-file, the ambiguity rule is per-union).
        insert_artist_session(
            "artist_b2",
            "base",
            &format!("[IMAGE:{}]", u.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(
            n, 0,
            "ambiguous duplicate kept even when only one is mentioned"
        );
        assert!(g.exists());
        assert!(u.exists());
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_prunes_cursors_of_vanished_users() {
        let userspaces = media_fixture("gone").await;
        write_gen_files(userspaces.path(), "gone", &["f_a.png"]).await;
        insert_artist_session("artist_gone1", "gone", "no file mentions").await;
        reset_media_cursors().await;
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 1);
        assert!(
            MEDIA_CURSORS
                .lock()
                .await
                .as_ref()
                .unwrap()
                .contains_key("gone")
        );
        // The userspace dir vanishes — the next complete pass prunes its
        // cursor (bounded per-user memory; a re-created dir starts fresh).
        tokio::fs::remove_dir_all(userspaces.path().join("gone"))
            .await
            .unwrap();
        assert_eq!(sweep_media_at(userspaces.path()).await.unwrap(), 0);
        assert!(
            !MEDIA_CURSORS
                .lock()
                .await
                .as_ref()
                .unwrap()
                .contains_key("gone")
        );
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_video_case_insensitive() {
        let userspaces = media_fixture("video").await;
        let files = write_gen_files(userspaces.path(), "video", &["clip.mp4", "Photo.PNG"]).await;
        // Mentions in a different case: video matches case-insensitively,
        // non-video extensions do not.
        insert_artist_session("artist_v1", "video", "[VIDEO:CLIP.MP4] photo.png").await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1);
        assert!(
            files[0].exists(),
            "video kept via case-insensitive basename"
        );
        assert!(!files[1].exists(), "case-mismatched non-video not kept");
    }

    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_overflowed_cleared_by_clear_rotation() {
        let userspaces = media_fixture("ovf").await;
        let files = write_gen_files(userspaces.path(), "ovf", &["f_a.png", "f_b.png"]).await;
        // A bloated session (>4× scan budget) overflows the keep-set cap:
        // nothing can be safely deleted while it is in the base.
        let huge = "x".repeat(MEDIA_SCAN_BUDGET_BYTES * 4 + 1);
        insert_artist_session(
            "artist_ovf1",
            "ovf",
            &format!(
                "[IMAGE:{}] {huge}",
                files[0].canonicalize().unwrap().display()
            ),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 0, "overflowed keep-set never deletes");
        assert!(files[0].exists());
        assert!(files[1].exists());
        // /clear rotation: the overflowed session is deleted; the cursor
        // resets (incl. the overflowed flag), so a fresh base re-enables the
        // sweep — f_a (mentioned only in the cleared session) becomes a
        // candidate, f_b survives via the surviving session.
        crate::session::store()
            .conn
            .execute(
                "DELETE FROM session_metadata WHERE agent_id = 'artist_ovf1'",
                (),
            )
            .await
            .unwrap();
        insert_artist_session(
            "artist_ovf2",
            "ovf",
            &format!("[IMAGE:{}]", files[1].canonicalize().unwrap().display()),
        )
        .await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "overflowed user re-enabled after /clear rotation");
        assert!(!files[0].exists());
        assert!(files[1].exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;
        let userspaces = media_fixture("sym").await;
        // A symlinked directory INSIDE generated/ pointing OUTSIDE the
        // userspace root: traversal must not follow it — the deletion pass
        // would otherwise collect (and delete) real files outside the root,
        // violating "never delete a file whose mention was not scanned" at
        // the filesystem level. The symlink itself is also never a deletion
        // candidate (skipped by the no-follow walk).
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("victim.png");
        tokio::fs::write(&victim, "x").await.unwrap();
        let link = userspaces
            .path()
            .join("sym")
            .join("generated")
            .join("linked");
        symlink(outside.path(), &link).unwrap();
        // An artist session mentioning ONE real file: the deletion pass must
        // actually run (the unmentioned file below is a candidate) — without
        // sessions the empty-base guard short-circuits and the no-follow
        // logic would never be exercised.
        let files = write_gen_files(userspaces.path(), "sym", &["kept.png", "orphan.png"]).await;
        insert_artist_session(
            "artist_sym",
            "sym",
            &format!("[IMAGE:{}]", files[0].canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "orphan deleted; symlink and outside victim untouched");
        assert!(files[0].exists(), "mentioned file kept");
        assert!(!files[1].exists(), "unmentioned file deleted");
        assert!(victim.exists(), "file outside the userspace root untouched");
        assert!(link.exists(), "symlink itself kept");
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(media_sweep)] // shared MEDIA_CURSORS static + session store
    async fn sweep_media_skips_top_level_symlinked_dir() {
        use std::os::unix::fs::symlink;
        let userspaces = media_fixture("topsym").await;
        // The generated/ dir ITSELF is a symlink to an outside tree: read_dir
        // on the top-level path would resolve it, so the entry-level no-follow
        // inside list_files cannot catch it — the symlink_metadata guard must.
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("victim.png");
        tokio::fs::write(&victim, "x").await.unwrap();
        let gen_dir = userspaces.path().join("topsym").join("generated");
        tokio::fs::remove_dir_all(&gen_dir).await.unwrap();
        symlink(outside.path(), &gen_dir).unwrap();
        // An artist session mentioning ONE uploads file proves the deletion
        // pass ran (the unmentioned orphan below is a candidate) while the
        // symlinked generated/ tree stays untouched.
        let up = userspaces.path().join("topsym").join("uploads");
        let kept = up.join("kept.png");
        let orphan = up.join("orphan.png");
        tokio::fs::write(&kept, "x").await.unwrap();
        tokio::fs::write(&orphan, "x").await.unwrap();
        insert_artist_session(
            "artist_topsym",
            "topsym",
            &format!("[IMAGE:{}]", kept.canonicalize().unwrap().display()),
        )
        .await;
        reset_media_cursors().await;
        let n = sweep_media_at(userspaces.path()).await.unwrap();
        assert_eq!(n, 1, "orphan deleted; symlinked generated tree untouched");
        assert!(kept.exists(), "mentioned uploads file kept");
        assert!(!orphan.exists(), "unmentioned uploads file deleted");
        assert!(
            victim.exists(),
            "file behind the top-level symlink untouched"
        );
        assert!(gen_dir.exists(), "top-level symlink itself kept");
    }
}

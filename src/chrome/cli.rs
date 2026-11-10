//! `mahbot chrome` — browser automation CLI over the shared core.
//!
//! Runs before temp-root/lock init, alongside the daemon, so it needs no
//! initialized config. Sessions are namespaced `mahbot-chrome-*` and can never
//! collide with the interactive tool's `agent-tab-*`.
//!
//! stdout is exactly ONE line of JSON (the [`OutEnvelope`]); stderr carries
//! diagnostics. Exit codes: 0 success, 1 site/data step failure (schedulable),
//! 2 environment failure (fix it, don't blind-retry), 3 usage error, 128+signal
//! when interrupted or terminated — then no envelope. Steps run to the two
//! clocks of the [chrome policy](crate::chrome), as [`StepClocks`] derives them.
//! A tab an `open`/`click` step creates in a NAMED session leaves chrome-use's
//! group before the envelope, silently skipped when the browser will not answer
//! ([`crate::tools::chrome_groups`]).
//!
//! Without `--session` a call runs in a throwaway session, closed right after the
//! envelope. A named session's lifetime, `--keep` and its sweeps are
//! [`crate::chrome::cli_sessions`]'s; this module records each call's use and
//! holds its in-flight mark (an hour) while the call runs.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::chrome::actions::{self, CliFlag};
use crate::chrome::cli_sessions;
use crate::chrome::contract::{
    ChromeResponse, ERROR_PAGE_PROBE_JS, EXPECT_TIMEOUT_NOTE, ErrorPageProbe, ExpectOutcome,
    OutEnvelope, OutKind, chrome_leftover_pipe_note, chrome_use_warning, classify_call_failure,
    eval_count, eval_result, expect_outcome, extract_net_error_code, extract_output,
    extract_snapshot_text, is_session_unresponsive_error, is_unreachable_tab_error,
    net_error_phrase, parse_error_page_probe, parse_first, sanitize_timeout_message,
    self_launched_browser_cli_error, self_launched_browser_note, truncated_output_error,
    unreachable_tab_message, with_condition_timeout_note,
};
use crate::chrome::files::OneFileTarget;
use crate::chrome::forms::{
    ExpectCond, ExtractGate, WaitTarget, count_eval_js, describe, expect_args, extract_gate,
    parse_count_op, parse_predicate, parse_state, text_value_argv, validate_extract_getters,
    wait_args, wait_target,
};
use crate::chrome::spawn::{
    CliRun, CliSpawn, clear_group_cleanup_backstop, set_group_cleanup_backstop, spawn_cli,
};
use crate::chrome::{
    CHROME_USE_DECLARED_BUDGET, CHROME_USE_OWN_BUDGET, CLI_EPHEMERAL_PREFIX, CLI_SESSION_PREFIX,
    ChromeCallClocks, CliRecovery, DEFAULT_OPEN_TIMEOUT, DEFAULT_STEP_TIMEOUT, KILL_SLACK,
    SESSION_STOP_TIMEOUT, clocks, is_blank_page_url, kill_bound, probe_clocks,
};
use crate::tools::chrome_daemon::{
    CliStatus, SessionRecovery, cli_path, cli_probe, cli_version, ensure_ready_for_actions,
    readiness, recover_unresponsive_session, run_cli_bounded_at,
};
use crate::tools::chrome_groups::{self, Opened};
use crate::util::{TOOL_OUTPUT_BUDGET_BYTES, truncate_sandwich};
use serde_json::{Value, json};

/// Product-side bound on the daemon-free `session list` enumerate: it only asks
/// the daemon what sessions exist, so it stays well under chrome-use's budget.
const SESSION_LIST_TIMEOUT: Duration = Duration::from_secs(8);

/// `session status` probe bound: the product's own bounded liveness question,
/// declared to chrome-use ([`probe_clocks`]) rather than a longer bound it would
/// never let run. No answer within it is [`Liveness::Unresponsive`]. Opt-in.
const SESSION_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// The manual recovery flow for a wedged named session, naming only verbs that
/// exist in the mahbot surface.
const SESSION_RECOVERY_FLOW: &str = "`mahbot chrome session stop <name>`, then re-run the action with \
     `--session <name>` to re-create it (cookies persist in the profile; open \
     tabs do not)";

/// The sentence appended to a wedge-shaped envelope: what the automatic recovery
/// did, or the manual flow that remains. Shares the tool's own summary.
fn wedge_recovery_note(recovery: SessionRecovery) -> String {
    match recovery {
        SessionRecovery::Stopped => recovery.summary().to_string(),
        SessionRecovery::Unanswered | SessionRecovery::NotStarted => {
            // The summary says what happened; this appends the manual flow left.
            format!(
                "{} If it is still wedged, recover with {SESSION_RECOVERY_FLOW}.",
                recovery.summary()
            )
        }
    }
}

/// What [`recover_session_wedge`] should do for one envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WedgeAction {
    /// chrome-use's own session-unresponsive verdict: recover directly.
    Recover,
    /// The product's own clock ended the step: confirm liveness first, so a
    /// merely slow site never costs its session.
    Probe,
    /// No wedge evidence: leave the session (and the envelope) alone.
    Leave,
}

/// The wedge fact is WHO ended the step, not the envelope's kind: chrome-use's
/// own session-unresponsive verdict is a diagnosis, so it recovers directly, while
/// a deadline label counts only when the PRODUCT's clock ended the step
/// ([`CLI_STEP_KILLED_BY_CLOCK`]) — otherwise it is chrome-use's honest timeout,
/// which must never cost a session its tabs. Ephemeral sessions are left to the
/// caller. Pure so tests pin the keying.
#[must_use]
fn wedge_action(env: &OutEnvelope, named: bool, killed_by_clock: bool) -> WedgeAction {
    if !named {
        return WedgeAction::Leave;
    }
    if env.kind == OutKind::Environment
        && envelope_error(env).is_some_and(is_session_unresponsive_error)
    {
        return WedgeAction::Recover;
    }
    if matches!(env.kind, OutKind::Timeout | OutKind::Redesign) && killed_by_clock {
        return WedgeAction::Probe;
    }
    WedgeAction::Leave
}

fn envelope_error(env: &OutEnvelope) -> Option<&str> {
    env.payload.get("error").and_then(Value::as_str)
}

/// Append a sentence to an envelope's `error` text; a no-op when it carries none.
fn append_error_note(env: &mut OutEnvelope, note: &str) {
    if let Some(obj) = env.payload.as_object_mut()
        && let Some(err) = obj.get("error").and_then(Value::as_str)
    {
        obj.insert("error".into(), json!(format!("{err} — {note}")));
    }
}

/// Recover a wedge the envelope points at (bounded), putting what was done into its
/// error text ([`wedge_action`] says which envelopes count).
async fn recover_session_wedge(
    env: &mut OutEnvelope,
    session: &str,
    named: bool,
    killed_by_clock: bool,
) {
    let note = match wedge_action(env, named, killed_by_clock) {
        WedgeAction::Leave => return,
        WedgeAction::Recover => wedge_recovery_note(recover_unresponsive_session(session).await),
        WedgeAction::Probe => {
            // The product's kill preempted chrome-use's verdict: confirm with the same bounded
            // `get url` probe the `session status` verb uses.
            match probe_session_liveness(session).await {
                Liveness::Unresponsive => {
                    wedge_recovery_note(recover_unresponsive_session(session).await)
                }
                Liveness::Answered => {
                    "The session answered a liveness probe, so this was a slow call rather than a \
                     wedged session."
                        .to_string()
                }
                Liveness::Inconclusive => {
                    "The session's liveness could not be established, so its daemon was left \
                     alone; retry the call, and stop the session by hand if it stays \
                     unresponsive."
                        .to_string()
                }
            }
        }
    };
    append_error_note(env, &note);
}

/// What one bounded liveness probe of a named session established.
#[derive(Debug, Clone, Copy)]
enum Liveness {
    /// The session produced a normal answer: it is alive, so a clock-killed step
    /// was a slow call rather than a wedge.
    Answered,
    /// No answer within [`SESSION_PROBE_TIMEOUT`]: the session is wedged.
    Unresponsive,
    /// The probe could not run or establish anything (no binary, spawn failure,
    /// unparseable envelope): nothing is recovered on its account.
    Inconclusive,
}

/// Bounded liveness probe of a named session (read-only `get url`).
/// [`Liveness::Inconclusive`] is neither answered nor unresponsive: no recovery.
async fn probe_session_liveness(session: &str) -> Liveness {
    let Some(path) = cli_path() else {
        return Liveness::Inconclusive;
    };
    match spawn_step(
        &path,
        &["get", "url"],
        Some(session),
        StepClocks::probe(SESSION_PROBE_TIMEOUT),
        None,
    )
    .await
    {
        Ok(_) => Liveness::Answered,
        Err(f) if f.kind == OutKind::Timeout => Liveness::Unresponsive,
        Err(_) => Liveness::Inconclusive,
    }
}

/// Top-level `mahbot chrome -h` — rendered from the shared action registry.
#[must_use]
fn top_help() -> String {
    let mut out = String::from("mahbot chrome — browser automation CLI\n\n");
    out.push_str("Usage:\n");
    out.push_str("  mahbot chrome <action> [options]\n\n");
    out.push_str("Actions:\n");
    for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
        let Some(cli) = d.cli.as_ref() else {
            continue;
        };
        if cli.syntax.is_empty() {
            let _ = writeln!(out, "  {}", d.name);
        } else {
            let _ = writeln!(out, "  {} {}", d.name, cli.syntax);
        }
        let _ = writeln!(out, "      {}", d.purpose);
    }
    out.push_str("\nGlobal flags:\n");
    out.push_str(
        "  --session <name>   use/name a session — letters, digits, '-' or '_' only (not valid for status / session subcommands)\n",
    );
    out.push_str(
        "  --keep             keep the named session open until it is stopped — only with --session, and only on the call that creates the session (not valid for status / session subcommands)\n\n",
    );
    out.push_str("Sessions:\n");
    let minutes = cli_sessions::LIFETIME_MINUTES;
    let _ = write!(
        out,
        "  Without --session a call runs in a throwaway session whose tabs are closed right after the call.\n  With --session <name> the session's tabs live {minutes} minutes after its last use and are then closed;\n  any later call with the same name starts those {minutes} minutes again. With --keep the session never\n  expires: it stays until 'mahbot chrome session stop <name>'. 'mahbot chrome session list' shows the\n  sessions this CLI created, whether each is kept, and how long since it was last used.\n  A second call is not the first one's page: without --session, 'screenshot' or 'upload' after a\n  separate 'open' sees a blank tab — name the same --session in both calls.\n  Stopping, restarting or updating mahbot closes no browser tabs.\n\n"
    );
    out.push_str("Output (stdout, one JSON line):\n");
    out.push_str(
        "  {\"schema\":1,\"action\":\"...\",\"ok\":true|false,\"kind\":\"...\",...payload}\n\n",
    );
    out.push_str("Exit codes:\n");
    out.push_str("  0  ok / empty (including a legitimately empty region)\n");
    out.push_str("  1  site/data step failure (timeout, network, redesign, not-found, error)\n");
    out.push_str("  2  environment failure (no chrome-use CLI/relay/Chrome/display)\n");
    out.push_str("  3  usage error\n");
    out.push_str("  128+N  interrupted by signal N (a throwaway session is closed first)\n\n");
    out.push_str("Help:\n");
    out.push_str("  mahbot chrome -h            this help\n");
    out.push_str("  mahbot chrome <action> -h   per-action help (flags, kinds, examples)\n");
    out
}

/// Per-action `mahbot chrome <action> -h`.
#[must_use]
fn action_help(name: &str) -> String {
    let Some(d) = actions::desc(name) else {
        return String::new();
    };
    let Some(cli) = d.cli.as_ref() else {
        return String::new();
    };

    let mut out = String::new();
    let _ = write!(out, "mahbot chrome {} — {}\n\n", d.name, d.purpose);
    out.push_str("Usage:\n");
    if cli.syntax.is_empty() {
        let _ = writeln!(out, "  mahbot chrome {}", d.name);
    } else {
        let _ = writeln!(out, "  mahbot chrome {} {}", d.name, cli.syntax);
    }

    let flag_rows: Vec<(&str, &str)> = cli
        .flags()
        .map(|f| (f.spelling, f.description.as_str()))
        .collect();
    if !flag_rows.is_empty() {
        let width = flag_rows.iter().map(|(f, _)| f.len()).max().unwrap_or(0);
        out.push_str("\nFlags:\n");
        for (flag, desc) in flag_rows {
            let _ = writeln!(out, "  {flag:<width$}  {desc}");
        }
    }

    out.push_str("\nKinds (stdout \"kind\" → exit code):\n");
    let kinds = cli
        .kinds
        .iter()
        .map(|k| format!("{} ({})", k.as_str(), k.exit_code()))
        .collect::<Vec<_>>()
        .join(" · ");
    let _ = writeln!(out, "  {kinds}");

    if !cli.details.is_empty() {
        let _ = write!(out, "\n{}\n", cli.details);
    }

    out.push_str("\nExamples:\n");
    for ex in cli.examples {
        let _ = writeln!(out, "  {ex}");
    }
    out
}

/// A per-action help request: the first positional names a CLI action and a later
/// token is `-h`/`--help`; flags before the action word are skipped, and from `--`
/// on every token is literal text, never help. Returns the action's registry name.
fn action_help_request(args: &[String]) -> Option<&'static str> {
    let (i, word) = first_positional(args)?;
    let d = actions::desc(word)?;
    d.cli
        .as_ref()
        .is_some_and(|_| {
            args[i + 1..]
                .iter()
                .take_while(|t| *t != "--")
                .any(|t| t == "-h" || t == "--help")
        })
        .then_some(d.name)
}

/// A session the CLI resolved for an action that runs in one.
#[derive(Clone)]
struct CliSession {
    name: String,
    /// `true` when the call owns a throwaway session — anonymous and browser-driving —
    /// whose tabs are closed when the call ends (or, interrupted, before it exits).
    ephemeral: bool,
    /// The global `--keep` request: mark the session as having no lifetime. Only the
    /// call that creates the record can set it ([`cli_sessions::note_use`]).
    keep: bool,
}

struct Invocation {
    action: Action,
    session: Option<String>,
}

/// One parsed `mahbot chrome` action.
///
/// `timeout` (on `Open`/`Wait`/`Expect`) is the deadline declared for the step;
/// `bound` on the rest is the user's own `--timeout`, or `None` — those verbs
/// forward nothing chrome-use honours, so a bound below the declared clock is
/// refused and a longer one widens only the product's kill ([`StepClocks`]).
enum Action {
    Status,
    Open {
        url: String,
        expect: Option<String>,
        structural: bool,
        timeout: Duration,
    },
    Count {
        selector: String,
        bound: Option<Duration>,
    },
    Wait {
        target: WaitTarget,
        timeout: Duration,
    },
    Expect {
        cond: ExpectCond,
        timeout: Duration,
    },
    Eval {
        js: String,
        bound: Option<Duration>,
    },
    Extract {
        schema_file: String,
        limit: Option<usize>,
        bound: Option<Duration>,
    },
    Click {
        selector: String,
        if_present: bool,
        bound: Option<Duration>,
    },
    Fill {
        selector: String,
        source: TextInput,
        bound: Option<Duration>,
    },
    Type {
        selector: String,
        text: String,
        key_events: bool,
        bound: Option<Duration>,
    },
    Press {
        key: String,
        selector: Option<String>,
        hold: Option<u64>,
        bound: Option<Duration>,
    },
    Upload {
        selector: String,
        files: Vec<String>,
        bound: Option<Duration>,
    },
    Screenshot {
        output_dir: Option<String>,
    },
    SessionStop {
        name: String,
        force: bool,
    },
    SessionStatus {
        name: String,
    },
    /// The sessions this CLI created, their mark and time since last use.
    SessionList,
}

impl Action {
    /// Whether this action drives the browser in a session; the report/repair verbs
    /// (`status`, `session …`) run in none — the parser and the help both read it.
    fn runs_in_session(&self) -> bool {
        session_refusal_for(self.name()).is_none()
    }

    /// The shared-registry name of this action — the envelope's `action` field and the
    /// help lookup key; the `session` subcommands emit as `"session"`.
    fn name(&self) -> &'static str {
        match self {
            Action::Status => "status",
            Action::Open { .. } => "open",
            Action::Count { .. } => "count",
            Action::Wait { .. } => "wait",
            Action::Expect { .. } => "expect",
            Action::Eval { .. } => "eval",
            Action::Extract { .. } => "extract",
            Action::Click { .. } => "click",
            Action::Fill { .. } => "fill",
            Action::Type { .. } => "type",
            Action::Press { .. } => "press",
            Action::Upload { .. } => "upload",
            Action::Screenshot { .. } => "screenshot",
            Action::SessionStop { .. } | Action::SessionStatus { .. } | Action::SessionList => {
                "session"
            }
        }
    }
}

enum TextInput {
    Inline(String),
    File(String),
    Stdin,
}

impl TextInput {
    /// The chrome-use argv suffix carrying the value (leading-dash inline
    /// text shielded with `--`; see [`forms::text_value_argv`]).
    fn argv(&self) -> Vec<String> {
        match self {
            Self::Inline(t) => text_value_argv(t),
            Self::File(p) => vec!["--file".into(), p.clone()],
            Self::Stdin => vec!["--stdin".into()],
        }
    }
}

/// A failed chrome-use step: the classified [`OutKind`] plus its error text (empty
/// for a mahbot-side kill); wedge recovery is applied later ([`recover_session_wedge`]).
#[derive(Debug)]
struct StepFailure {
    kind: OutKind,
    message: String,
}

impl StepFailure {
    /// The failure envelope for `action`, layering the failure detail onto `base`: a
    /// deadline kill reports the product's own `timeout_ms` plus a factual `error`; any
    /// other failure the chrome-use error with remediation layered on, and a Network
    /// failure its net error token as `error_code` when there is one.
    fn envelope(self, action: &str, base: Value, clocks: StepClocks) -> OutEnvelope {
        let mut obj = match base {
            Value::Object(m) => m,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("error".into(), other);
                m
            }
        };
        if self.kind == OutKind::Timeout && self.message.is_empty() {
            obj.insert("timeout_ms".into(), json!(clocks.call.kill.as_millis()));
            obj.insert(
                "error".into(),
                json!(deadline_error(
                    clocks.call.chrome_side,
                    clocks.call.kill,
                    "the step did not complete"
                )),
            );
        } else {
            let message = if is_unreachable_tab_error(&self.message) {
                unreachable_tab_message(&self.message)
            } else {
                self.message.clone()
            };
            obj.insert("error".into(), json!(message));
        }
        if self.kind == OutKind::Network
            && let Some(code) = extract_net_error_code(&self.message)
        {
            obj.insert("error_code".into(), json!(code));
        }
        out_env(action, false, self.kind, Value::Object(obj))
    }
}

type StepOutcome = Result<ChromeResponse, StepFailure>;

/// Whether any session-scoped chrome-use spawn was attempted this process: a
/// throwaway can exist only after one, so cleanup is skipped (no stderr noise).
static CHROME_USE_SPAWNED: AtomicBool = AtomicBool::new(false);

/// Whether a step this process ran exited while a process it left behind still held
/// the output channel ([`CliOutput::leftover_pipes`]). ONE invocation per process, so
/// this is its record; [`run_action`] turns it into a note
/// ([`chrome_leftover_pipe_note`]), even on failure, never a failure.
static CHROME_USE_LEFTOVER_PIPES: AtomicBool = AtomicBool::new(false);

/// Whether the PRODUCT's own clock ended a step whose outcome IS the caller's verdict
/// ([`CliRun::TimedOut`]) above the `chrome_side` clock chrome-use was working to — so
/// chrome-use was cut off before classifying anything, which is what
/// [`recover_session_wedge`] keys on. Only a step that may run the relay self-heal
/// ([`CliRecovery::Allowed`]) attributes; probes and best-effort sub-steps
/// ([`CliRecovery::Suppressed`]) never can. [`run_action`] resets the mark before the
/// step and reads it right after, so a later step cannot erase a decisive one's mark.
static CLI_STEP_KILLED_BY_CLOCK: AtomicBool = AtomicBool::new(false);

/// Whether this call created the session's record. Read through [`record_materialized`]
/// by every early exit to drop a record whose session never reached the browser — a
/// kept one would otherwise never be reaped.
static SESSION_RECORD_CREATED: AtomicBool = AtomicBool::new(false);

/// `mahbot chrome` CLI entry — returns the process exit code.
pub async fn run_cli(args: &[String]) -> i32 {
    // Re-enterable pub API: clear the previous call's spawn bookkeeping.
    CHROME_USE_SPAWNED.store(false, Ordering::Relaxed);
    SESSION_RECORD_CREATED.store(false, Ordering::Relaxed);
    CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
    CLI_STEP_KILLED_BY_CLOCK.store(false, Ordering::Relaxed);
    clear_group_cleanup_backstop();
    if let Some(a) = args.first()
        && (a == "-h" || a == "--help")
    {
        print!("{}", top_help());
        return 0;
    }
    // A `-h`/`--help` after the action word is per-action help, anywhere before the first
    // `--` (`open <url> --help`, `--session x open -h`); from that `--` on the tokens are
    // the action's text (`type '#q' -- --help` types it) and nowhere else. A `-h` before
    // any action word falls through to top-level help.
    if let Some(action) = action_help_request(args) {
        print!("{}", action_help(action));
        return 0;
    }

    // The lifetime's CLI-side trigger: it works where no service process runs.
    crate::chrome::cli_sessions::sweep_expired().await;

    let (invocation, keep) = match parse_invocation(args) {
        Ok(parsed) => parsed,
        Err(msg) => {
            let action = recognized_action(args);
            eprintln!("mahbot chrome: {msg}");
            eprintln!("run 'mahbot chrome --help' for usage.");
            out_env(&action, false, OutKind::Usage, json!({ "error": msg })).emit();
            return 3;
        }
    };

    let session = call_session(&invocation, keep);
    // The action's call races the interruption signals, so an interrupted call still
    // closes the throwaway session it created.
    let mut call = Box::pin(dispatch(&invocation, &session));
    let mut interrupted = Box::pin(wait_for_interruption());
    let outcome = tokio::select! {
        biased;
        outcome = &mut call => outcome,
        code = &mut interrupted => {
            // Drop the in-flight call first so its chrome-use child cannot race the close.
            drop(call);
            close_throwaway_then_exit(&session, code).await
        }
    };
    outcome.envelope.emit();
    // The action is answered and its envelope is out: what is left is best-effort cleanup,
    // raced against a second interruption, which must end the process at once. A record
    // left by an aborted close is closed by the lifetime sweep; a record this call created
    // that never reached the browser is dropped here.
    let materialized = record_materialized();
    if !materialized && let Some(s) = outcome.session.as_ref() {
        cli_sessions::forget(&s.name);
    }
    if outcome.marked
        && materialized
        && let Some(s) = outcome.session.as_ref()
    {
        let display = cli_sessions::display_name(&s.name);
        eprintln!(
            "mahbot chrome: session '{display}' is kept open — stop it with 'mahbot chrome session stop {display}'"
        );
    }
    if let Some(s) = outcome.session.as_ref()
        && CHROME_USE_SPAWNED.load(Ordering::Relaxed)
    {
        if s.ephemeral {
            // The close runs after the envelope, so a helper it leaves behind can only be
            // reported here, like every other diagnostic of that close.
            CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
            if close_ephemeral_unless_interrupted(&s.name).await
                && CHROME_USE_LEFTOVER_PIPES.load(Ordering::Relaxed)
            {
                let display = cli_sessions::display_name(&s.name);
                eprintln!("mahbot chrome: {}", chrome_leftover_pipe_note(display));
            }
        } else if !outcome.tail_ran
            && let Some(cli) = tail_cli(&outcome.envelope, s)
        {
            // A named session outlives the command, so the groups this step used are still there
            // to leave collapsed — unless the step failed on its environment, where the tail
            // cannot reach the browser either ([`tail_cli`]). Cosmetic and silent.
            chrome_groups::collapse_own_groups(&cli, &s.name).await;
        }
    }
    // The call is over: its in-flight mark goes, so a sweep may now close the session on
    // the ordinary clock. A forgotten session has nothing to clear.
    if let Some(s) = outcome.session.as_ref() {
        cli_sessions::note_end(&s.name);
    }
    outcome.envelope.kind.exit_code()
}

/// The session this call runs in, resolved once for the interrupt path and the action
/// alike; only a browser-driving action owns a throwaway session.
fn call_session(invocation: &Invocation, keep: bool) -> CliSession {
    let (name, ephemeral) = resolve_session(invocation.session.as_deref());
    CliSession {
        name,
        ephemeral: ephemeral && invocation.action.runs_in_session(),
        keep,
    }
}

/// What `dispatch` produced: the envelope, the session it ran in (for the close and
/// collapse tails), whether this call set `--keep`, and whether the tab tail ran.
struct DispatchOutcome {
    envelope: OutEnvelope,
    session: Option<CliSession>,
    marked: bool,
    tail_ran: bool,
}

impl DispatchOutcome {
    /// An envelope from a verb that runs in no session and records nothing.
    fn plain(envelope: OutEnvelope) -> Self {
        Self {
            envelope,
            session: None,
            marked: false,
            tail_ran: false,
        }
    }
}

/// The wait an action's own call races against: an interrupt ends the CLI, whose
/// throwaway session must be closed first. Resolves to the `128+signal` exit code.
async fn wait_for_interruption() -> i32 {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let (Ok(mut interrupt), Ok(mut terminate), Ok(mut hangup)) = (
            signal(SignalKind::interrupt()),
            signal(SignalKind::terminate()),
            signal(SignalKind::hangup()),
        ) else {
            return std::future::pending::<i32>().await;
        };
        tokio::select! {
            _ = interrupt.recv() => 130,
            _ = terminate.recv() => 143,
            _ = hangup.recv() => 129,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        130
    }
}

/// An interrupted call still closes the throwaway session it created (a named one is
/// not its to close), then exits `128+signal`; a second interruption ends the process
/// at once instead of waiting the close out. Unmaterialized records are dropped.
async fn close_throwaway_then_exit(session: &CliSession, code: i32) -> ! {
    if session.ephemeral && CHROME_USE_SPAWNED.load(Ordering::Relaxed) {
        CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
        close_ephemeral_unless_interrupted(&session.name).await;
    } else if !record_materialized() {
        cli_sessions::forget(&session.name);
    }
    // The process is going: its mark must not outlive it (a sweep calls a dead pid's mark
    // stale); a named session stays alive on the ordinary clock.
    cli_sessions::note_end(&session.name);
    std::process::exit(code);
}

/// Whether this call's record describes a session that reached the browser: no record
/// was created, or a session-scoped step of this call was attempted.
fn record_materialized() -> bool {
    !SESSION_RECORD_CREATED.load(Ordering::Relaxed) || CHROME_USE_SPAWNED.load(Ordering::Relaxed)
}

/// [`close_ephemeral`], abandoned when a second interruption arrives: an answered call
/// must not keep the process alive for its bound. `true` when it completed.
async fn close_ephemeral_unless_interrupted(name: &str) -> bool {
    tokio::select! {
        () = close_ephemeral(name) => true,
        _ = wait_for_interruption() => false,
    }
}

/// The flags `word` declares — the same rows its help prints. The session-wide
/// pair is not here: [`session_refusal_for`] carries that decision.
fn declared_flags(word: &str) -> &'static [CliFlag] {
    actions::desc(word)
        .and_then(|d| d.cli.as_ref())
        .map_or(&[], |c| c.declared.as_slice())
}

/// Why `word`'s action refuses the session-wide pair, or `None` when it takes them.
fn session_refusal_for(word: &str) -> Option<&'static str> {
    actions::desc(word)?.cli.as_ref()?.session_refusal
}

/// Parse one command line into the action invocation and the global `--keep` request
/// (which needs a session name and is applied when the call resolves its session).
fn parse_invocation(args: &[String]) -> Result<(Invocation, bool), String> {
    let (session, keep, remaining) = extract_global_flags(args)?;
    let action_word = remaining
        .first()
        .ok_or_else(|| "no action given".to_string())?;
    let rest = &remaining[1..];
    let allowed: Vec<&CliFlag> = declared_flags(action_word).iter().collect();
    // The one declaration decision on which actions take the session-wide pair.
    if session.is_some()
        && let Some(refusal) = session_refusal_for(action_word)
    {
        return Err(refusal.to_string());
    }
    let invocation = match action_word.as_str() {
        "status" => parse_status(rest, &allowed),
        "open" => parse_open(session.as_deref(), rest, &allowed),
        "count" => parse_count(session.as_deref(), rest, &allowed),
        "wait" => parse_wait(session.as_deref(), rest, &allowed),
        "expect" => parse_expect(session.as_deref(), rest, &allowed),
        "eval" => parse_eval(session.as_deref(), rest, &allowed),
        "extract" => parse_extract(session.as_deref(), rest, &allowed),
        "click" => parse_click(session.as_deref(), rest, &allowed),
        "fill" => parse_fill(session.as_deref(), rest, &allowed),
        "type" => parse_type(session.as_deref(), rest, &allowed),
        "press" => parse_press(session.as_deref(), rest, &allowed),
        "upload" => parse_upload(session.as_deref(), rest, &allowed),
        "screenshot" => parse_screenshot(session.as_deref(), rest, &allowed),
        "session" => parse_session(rest, &allowed),
        other => Err(format!("unknown action '{other}'")),
    }?;
    if keep {
        // The mark needs both a browser-driving action and a name, checked once the action is
        // known so the refusal never names a flag the action itself forbids.
        if !invocation.action.runs_in_session() {
            return Err(format!("--keep is not valid for {action_word}"));
        }
        if invocation.session.is_none() {
            return Err(
                "--keep needs a session name: an unnamed call closes its session when it ends — pass --session <name>"
                    .to_string(),
            );
        }
    }
    Ok((invocation, keep))
}

/// Pull the global `--session` (last wins) and `--keep` out of anywhere before `--`,
/// returning the rest with everything from `--` on kept verbatim.
fn extract_global_flags(args: &[String]) -> Result<(Option<String>, bool, Vec<String>), String> {
    let mut session: Option<String> = None;
    let mut keep = false;
    let mut remaining = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        // `--` ends option parsing (the fill/type escape hatch): the rest is verbatim text.
        if a == "--" {
            remaining.extend_from_slice(&args[i..]);
            break;
        }
        if a == "--session" {
            i += 1;
            let val = args
                .get(i)
                .ok_or_else(|| "missing value for --session".to_string())?;
            validate_session_name(val)?;
            session = Some(val.clone());
            i += 1;
        } else if let Some(val) = a.strip_prefix("--session=") {
            if val.is_empty() {
                return Err("missing value for --session".to_string());
            }
            validate_session_name(val)?;
            session = Some(val.to_string());
            i += 1;
        } else if a == "--keep" {
            keep = true;
            i += 1;
        } else {
            remaining.push(a.clone());
            i += 1;
        }
    }
    Ok((session, keep, remaining))
}

/// Index and token of the first positional argument: skips `--session <name>`
/// (with its value), `--session=<name>`, and any other `-`-prefixed flag.
fn first_positional(args: &[String]) -> Option<(usize, &str)> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--session" {
            i += 2;
        } else if a.starts_with('-') {
            i += 1;
        } else {
            return Some((i, a));
        }
    }
    None
}

/// The action word to put on a usage envelope — the first non-flag token when
/// it is a recognized action, else `"usage"`.
fn recognized_action(args: &[String]) -> String {
    match first_positional(args) {
        Some((_, a)) if actions::is_cli_action(a) => a.to_string(),
        _ => "usage".to_string(),
    }
}

/// Split action args into positionals and a `--flag` map (`--flag value` and
/// `--flag=value`); `accepted` is the action's declaration deciding values.
fn parse_flags(args: &[String], accepted: &[&CliFlag]) -> Result<(Vec<String>, Flags), String> {
    parse_flags_impl(args, accepted, false)
}

/// [`parse_flags`] for fill/type: a single-dash token is literal positional text (they
/// have no single-dash flags), so `fill '#q' -tail` needs no `--` shield.
fn parse_flags_text(
    args: &[String],
    accepted: &[&CliFlag],
) -> Result<(Vec<String>, Flags), String> {
    parse_flags_impl(args, accepted, true)
}

fn parse_flags_impl(
    args: &[String],
    accepted: &[&CliFlag],
    single_dash_is_text: bool,
) -> Result<(Vec<String>, Flags), String> {
    let mut positionals = Vec::new();
    let mut flags = Flags::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        // `--` ends option parsing: the rest are positionals verbatim (dash-leading text).
        if a == "--" {
            positionals.extend(args[i + 1..].iter().cloned());
            break;
        }
        if let Some(body) = a.strip_prefix("--") {
            let (name, eq_value) = match body.split_once('=') {
                Some((n, v)) => (n, Some(v)),
                None => (body, None),
            };
            let Some(flag) = accepted.iter().find(|f| f.name() == name) else {
                return Err(format!("unknown flag --{name}"));
            };
            if flag.takes_value {
                let value = match eq_value {
                    Some(v) if !v.is_empty() => v.to_string(),
                    Some(_) => return Err(format!("missing value for --{name}")),
                    None => {
                        i += 1;
                        args.get(i)
                            .ok_or_else(|| format!("missing value for --{name}"))?
                            .clone()
                    }
                };
                flags.values.insert(name.to_string(), value);
            } else {
                if eq_value.is_some() {
                    return Err(format!("flag --{name} does not take a value"));
                }
                flags.bools.insert(name.to_string());
            }
            i += 1;
            continue;
        }
        if a.starts_with('-') && a.len() > 1 && !single_dash_is_text {
            return Err(format!("unknown flag {a}"));
        }
        positionals.push(a.clone());
        i += 1;
    }
    Ok((positionals, flags))
}

#[derive(Default)]
struct Flags {
    values: HashMap<String, String>,
    bools: HashSet<String>,
}

impl Flags {
    #[must_use]
    fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    #[must_use]
    fn has(&self, name: &str) -> bool {
        self.bools.contains(name)
    }
}

fn parse_status(rest: &[String], allowed: &[&CliFlag]) -> Result<Invocation, String> {
    // `status` declares no flag of its own, so anything flag-shaped here is refused.
    let (positionals, _) = parse_flags(rest, allowed)?;
    reject_extra_positionals(&positionals, 0)?;
    Ok(Invocation {
        action: Action::Status,
        session: None,
    })
}

fn parse_open(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let url = take_positional(&positionals, 0, "url")?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Open {
            url,
            expect: flags.value("expect").map(String::from),
            structural: flags.has("structural"),
            timeout: parse_forwarded_timeout(&flags)?.unwrap_or(DEFAULT_OPEN_TIMEOUT),
        },
        session: session.map(String::from),
    })
}

fn parse_count(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let selector = take_positional(&positionals, 0, "selector")?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Count {
            selector,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_wait(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let target = wait_target(
        positionals.first().map(String::as_str),
        flags.value("url"),
        flags.value("text"),
    )?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Wait {
            target,
            timeout: parse_forwarded_timeout(&flags)?.unwrap_or(DEFAULT_STEP_TIMEOUT),
        },
        session: session.map(String::from),
    })
}

fn parse_expect(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let cond = parse_expect_cond(&positionals)?;
    Ok(Invocation {
        action: Action::Expect {
            cond,
            timeout: parse_forwarded_timeout(&flags)?.unwrap_or(DEFAULT_STEP_TIMEOUT),
        },
        session: session.map(String::from),
    })
}

/// chrome-use 1.5.101 selector-first expect grammar, validated against the safe
/// allowlist: `<sel> <visible|hidden|present>`, `count <sel> <op> <n>`,
/// `text|value <sel> <pred> <value>`, `attr <sel> <name> <pred> <value>`,
/// `url <pred> <pattern>`; trailing payload words join with spaces.
#[expect(clippy::too_many_lines)]
fn parse_expect_cond(ps: &[String]) -> Result<ExpectCond, String> {
    let usage = "expect <selector> <visible|hidden|present> | count <sel> <op> <n> | text|value <sel> <equals|contains|matches> <value> | attr <sel> <name> <pred> <value> | url <pred> <pattern>";
    let first = ps
        .first()
        .map(String::as_str)
        .ok_or_else(|| format!("missing condition — {usage}"))?;
    match first {
        "count" => {
            if ps.len() != 4 {
                return Err("expect count takes exactly: count <selector> <op> <n>".to_string());
            }
            let op = parse_count_op(&ps[2]).ok_or_else(|| {
                format!(
                    "invalid count op '{}' — use == != > < >= <= (or eq ne gt lt ge le)",
                    ps[2]
                )
            })?;
            let n: u64 = ps[3]
                .parse()
                .map_err(|_| format!("count comparison needs a number, got '{}'", ps[3]))?;
            Ok(ExpectCond::Count {
                selector: ps[1].clone(),
                op,
                n,
            })
        }
        "text" | "value" => {
            if ps.len() < 4 {
                return Err(format!(
                    "expect {first} takes: {first} <selector> <equals|contains|matches> <value>"
                ));
            }
            let predicate = parse_predicate(&ps[2]).ok_or_else(|| {
                format!(
                    "invalid predicate '{}' — use equals|contains|matches",
                    ps[2]
                )
            })?;
            let value = ps[3..].join(" ");
            let selector = ps[1].clone();
            Ok(if first == "text" {
                ExpectCond::Text {
                    selector,
                    predicate,
                    value,
                }
            } else {
                ExpectCond::Value {
                    selector,
                    predicate,
                    value,
                }
            })
        }
        "attr" => {
            if ps.len() < 5 {
                return Err(
                    "expect attr takes: attr <selector> <name> <equals|contains|matches> <value>"
                        .to_string(),
                );
            }
            let predicate = parse_predicate(&ps[3]).ok_or_else(|| {
                format!(
                    "invalid predicate '{}' — use equals|contains|matches",
                    ps[3]
                )
            })?;
            Ok(ExpectCond::Attr {
                selector: ps[1].clone(),
                name: ps[2].clone(),
                predicate,
                value: ps[4..].join(" "),
            })
        }
        "url" => {
            if ps.len() < 3 {
                return Err("expect url takes: url <equals|contains|matches> <pattern>".to_string());
            }
            let predicate = parse_predicate(&ps[1]).ok_or_else(|| {
                format!(
                    "invalid predicate '{}' — use equals|contains|matches",
                    ps[1]
                )
            })?;
            Ok(ExpectCond::Url {
                predicate,
                pattern: ps[2..].join(" "),
            })
        }
        sel => {
            if ps.len() != 2 {
                return Err(format!("expect takes a selector and one state — {usage}"));
            }
            let state = parse_state(&ps[1]).ok_or_else(|| {
                format!(
                    "unsupported condition '{}' — allowed states: visible|hidden|present (count/text/value/attr/url have their own forms)",
                    ps[1]
                )
            })?;
            Ok(ExpectCond::State {
                selector: sel.to_string(),
                state,
            })
        }
    }
}

fn parse_eval(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let js = take_positional(&positionals, 0, "js")?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Eval {
            js,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_extract(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    if !positionals.is_empty() {
        return Err(format!("unexpected argument '{}'", positionals[0]));
    }
    let schema_file = flags
        .value("schema-file")
        .ok_or_else(|| "missing required --schema-file <path>".to_string())?
        .to_string();
    Ok(Invocation {
        action: Action::Extract {
            schema_file,
            limit: parse_limit_flag(&flags)?,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_click(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let selector = take_positional(&positionals, 0, "selector")?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Click {
            selector,
            if_present: flags.has("if-present"),
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_fill(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags_text(rest, allowed)?;
    let usage = "fill <selector> <text | --file <path> | --stdin>";
    let selector = match positionals.first() {
        Some(s) => s.clone(),
        None => return Err(format!("missing selector — {usage}")),
    };
    let inline = positionals.get(1..).unwrap_or_default();
    let file = flags.value("file");
    let stdin = flags.has("stdin");
    let source = match (file, stdin, inline.is_empty()) {
        (Some(f), false, true) => TextInput::File(f.to_string()),
        (None, true, true) => TextInput::Stdin,
        (None, false, false) => TextInput::Inline(inline.join(" ")),
        _ => return Err(format!("exactly one text source required — {usage}")),
    };
    Ok(Invocation {
        action: Action::Fill {
            selector,
            source,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_type(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags_text(rest, allowed)?;
    let usage = "type <selector> <text> [--key-events]";
    let selector = match positionals.first() {
        Some(s) => s.clone(),
        None => return Err(format!("missing selector — {usage}")),
    };
    let text = positionals.get(1..).unwrap_or_default().join(" ");
    if text.is_empty() {
        return Err(format!("missing text — {usage}"));
    }
    Ok(Invocation {
        action: Action::Type {
            selector,
            text,
            key_events: flags.has("key-events"),
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_press(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let usage = "press <key> [--selector <sel>] [--hold <ms>]";
    let key = match positionals.first() {
        Some(s) => s.clone(),
        None => return Err(format!("missing key — {usage}")),
    };
    reject_extra_positionals(&positionals, 1)?;
    let hold = match flags.value("hold") {
        Some(v) => Some(
            v.parse::<u64>()
                .map_err(|_| format!("--hold must be an integer (ms): {v}"))?,
        ),
        None => None,
    };
    Ok(Invocation {
        action: Action::Press {
            key,
            selector: flags.value("selector").map(String::from),
            hold,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

/// `upload <selector> <path...>` — the paths are resolved at dispatch against this
/// command's directory, so a bad path reports why it failed, not a shape error.
fn parse_upload(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    let usage = "upload <selector> <path...>";
    let selector = match positionals.first() {
        Some(s) => s.clone(),
        None => return Err(format!("missing selector — {usage}")),
    };
    let files: Vec<String> = positionals[1..].to_vec();
    if files.is_empty() {
        return Err(format!("missing local file path — {usage}"));
    }
    Ok(Invocation {
        action: Action::Upload {
            selector,
            files,
            bound: parse_bound_timeout(&flags)?,
        },
        session: session.map(String::from),
    })
}

/// `screenshot [--output-dir <dir>]` — the destination is resolved at dispatch
/// ([`crate::cli::resolve_output_dir`]): existing, the caller's own, never overwritten.
fn parse_screenshot(
    session: Option<&str>,
    rest: &[String],
    allowed: &[&CliFlag],
) -> Result<Invocation, String> {
    let (positionals, flags) = parse_flags(rest, allowed)?;
    reject_extra_positionals(&positionals, 0)?;
    Ok(Invocation {
        action: Action::Screenshot {
            output_dir: flags.value("output-dir").map(String::from),
        },
        session: session.map(String::from),
    })
}

fn parse_session(rest: &[String], allowed: &[&CliFlag]) -> Result<Invocation, String> {
    let sub = rest.first().map(String::as_str).ok_or_else(|| {
        "usage: session stop <name> | session status <name> | session list".to_string()
    })?;
    if !matches!(sub, "stop" | "status" | "list") {
        return Err(format!(
            "unknown session subcommand '{sub}' (expected 'stop', 'status' or 'list')"
        ));
    }
    // A flag scoped to one subcommand belongs to that subcommand alone.
    let accepted: Vec<&CliFlag> = allowed
        .iter()
        .copied()
        .filter(|f| f.scope.is_none_or(|scope| scope == sub))
        .collect();
    let (positionals, flags) = parse_flags(&rest[1..], &accepted)?;
    let action = match sub {
        "stop" => {
            let name = take_positional(&positionals, 0, "name")?;
            validate_session_name(&name)?;
            reject_extra_positionals(&positionals, 1)?;
            Action::SessionStop {
                name,
                force: flags.has("force"),
            }
        }
        "status" => {
            let name = take_positional(&positionals, 0, "name")?;
            validate_session_name(&name)?;
            reject_extra_positionals(&positionals, 1)?;
            Action::SessionStatus { name }
        }
        _ => {
            reject_extra_positionals(&positionals, 0)?;
            Action::SessionList
        }
    };
    Ok(Invocation {
        action,
        session: None,
    })
}

fn take_positional(positionals: &[String], idx: usize, name: &str) -> Result<String, String> {
    positionals
        .get(idx)
        .cloned()
        .ok_or_else(|| format!("missing positional: {name}"))
}

fn reject_extra_positionals(positionals: &[String], expected: usize) -> Result<(), String> {
    if positionals.len() > expected {
        return Err(format!("unexpected argument '{}'", positionals[expected]));
    }
    Ok(())
}

/// The value of a forwarded `--timeout` — the deadline this verb IS given (`wait`,
/// `expect`, `open`'s `--expect` wait), so chrome-use honours it in full. At or above
/// [`CHROME_USE_OWN_BUDGET`] it is refused: chrome-use would run out of tolerance and
/// its session-unresponsive verdict (losing the tabs) would replace the honest one.
/// `None` when the user gave none (the verb's own default then applies).
fn parse_forwarded_timeout(flags: &Flags) -> Result<Option<Duration>, String> {
    let Some(secs) = parse_timeout_secs(flags)? else {
        return Ok(None);
    };
    if secs >= CHROME_USE_OWN_BUDGET.as_secs() {
        return Err(format!(
            "--timeout {secs}s is at or above chrome-use's own client tolerance ({}s): \
             chrome-use cannot work past it, so a deadline that long makes it run out of \
             tolerance instead of reporting its own reason, and its session-unresponsive \
             verdict replaces the honest one. Use less than {}s.",
            CHROME_USE_OWN_BUDGET.as_secs(),
            CHROME_USE_OWN_BUDGET.as_secs()
        ));
    }
    Ok(Some(Duration::from_secs(secs)))
}

/// The value of a bound `--timeout` — every verb chrome-use takes no per-call deadline
/// for (count, eval, extract, click, fill, type, press, upload). A bound below the clock
/// the product declares to chrome-use ([`CHROME_USE_DECLARED_BUDGET`]) cannot be
/// honoured and is refused rather than discarded; at or above it the call is never cut
/// off below it, and above it only the product's own kill widens
/// ([`StepClocks::tool_clock`]). No upper limit: the in-flight mark keeps the session
/// alive for an hour.
fn parse_bound_timeout(flags: &Flags) -> Result<Option<Duration>, String> {
    let Some(secs) = parse_timeout_secs(flags)? else {
        return Ok(None);
    };
    let declared = CHROME_USE_DECLARED_BUDGET.as_secs();
    if secs < declared {
        return Err(format!(
            "--timeout {secs}s cannot be honoured: chrome-use takes no per-call deadline for this \
             verb, so the call runs to the {declared}s clock mahbot declares to chrome-use and \
             mahbot never cuts the call off below it — the reported bound would not be the one you \
             asked for. Give at least {declared}s (that is the shortest bound the call can be given), \
             or omit --timeout. `wait`/`expect` — and `open`'s --expect wait — are the verbs that \
             forward a deadline chrome-use honours in full."
        ));
    }
    Ok(Some(Duration::from_secs(secs)))
}

/// The user's own `--timeout` in seconds, or `None`: the one parser behind
/// [`parse_forwarded_timeout`] and [`parse_bound_timeout`]. There is no per-step default
/// — the declared clock is [`CHROME_USE_DECLARED_BUDGET`] unless the verb forwards a
/// deadline (`wait`/`expect`), and `wait`/`expect`/`open` apply their own default at
/// their parse site.
fn parse_timeout_secs(flags: &Flags) -> Result<Option<u64>, String> {
    match flags.value("timeout") {
        Some(v) => {
            let secs: u64 = v
                .parse()
                .map_err(|_| format!("--timeout must be an integer: {v}"))?;
            if secs < 1 {
                return Err("--timeout must be at least 1 second".to_string());
            }
            Ok(Some(secs))
        }
        None => Ok(None),
    }
}

fn parse_limit_flag(flags: &Flags) -> Result<Option<usize>, String> {
    match flags.value("limit") {
        Some(v) => {
            let n: usize = v
                .parse()
                .map_err(|_| format!("--limit must be an integer: {v}"))?;
            Ok(Some(n))
        }
        None => Ok(None),
    }
}

/// Resolve the session name and whether it is ephemeral: a named `--session` becomes
/// `mahbot-chrome-<name>` (idempotent); no flag yields an ephemeral
/// `mahbot-chrome-ephemeral-<suffix>`. The value is validated at extraction.
fn resolve_session(flag: Option<&str>) -> (String, bool) {
    match flag {
        Some(name) => {
            let name = if name.starts_with(CLI_SESSION_PREFIX) {
                name.to_string()
            } else {
                format!("{CLI_SESSION_PREFIX}{name}")
            };
            (name, false)
        }
        None => (
            format!("{CLI_EPHEMERAL_PREFIX}{}", crate::generate_suffix()),
            true,
        ),
    }
}

/// Session namespaces the CLI must never stop without an explicit `--force`: the
/// interactive tool's per-run sessions and link enrichment's.
const PROTECTED_SESSION_PREFIXES: [&str; 2] = ["agent-tab-", "link-enricher-"];

/// Validate a caller-supplied session name. It becomes part of a command recipe the
/// agent is told to run (`session stop <name> --force` in a leftover note), so it is
/// restricted to chrome-use's own alphabet (`validation::is_valid_session_name`:
/// alphanumerics, '-' and '_'). A '.' is refused because chrome-use refuses it on
/// `session stop`, leaving a session neither cleared by hand nor recovered.
fn validate_session_name(name: &str) -> Result<(), String> {
    let plain = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'));
    if plain {
        Ok(())
    } else {
        Err(format!(
            "invalid session name {name:?} — use letters, digits, '-' or '_'"
        ))
    }
}

/// Resolve a session name for a session-word subcommand: a bare name is prefixed like
/// `--session`, while CLI-namespace and protected names pass through as-is (status may
/// probe protected sessions read-only; only stop gates on `--force`).
fn resolve_session_target(name: &str) -> String {
    if name.starts_with(CLI_SESSION_PREFIX)
        || PROTECTED_SESSION_PREFIXES
            .iter()
            .any(|p| name.starts_with(p))
    {
        return name.to_string();
    }
    format!("{CLI_SESSION_PREFIX}{name}")
}

/// `session stop` gating on top of [`resolve_session_target`].
fn resolve_stop_target(name: &str, force: bool) -> Result<String, String> {
    let target = resolve_session_target(name);
    if !force
        && PROTECTED_SESSION_PREFIXES
            .iter()
            .any(|p| target.starts_with(p))
    {
        return Err(format!(
            "{name} is not a mahbot-chrome session; pass --force to stop it anyway"
        ));
    }
    Ok(target)
}

/// The caller-operand checks the pre-action arm runs BEFORE the readiness gate, so a
/// usage error is never buried under an environment refusal: `open`'s URL and
/// `--expect`, an attach's paths, a capture's destination. Each verb re-runs them.
fn validate_action_args(action: &Action) -> Option<OutEnvelope> {
    match action {
        Action::Open { url, expect, .. } => open_args(url, expect.as_deref()).err(),
        Action::Upload {
            selector, files, ..
        } => resolve_upload_files(files)
            .err()
            .map(|(kind, reason)| upload_refusal(selector, files, kind, &reason)),
        Action::Screenshot { output_dir } => crate::cli::resolve_output_dir(output_dir.as_deref())
            .err()
            .map(|error| {
                out_env(
                    "screenshot",
                    false,
                    OutKind::Usage,
                    json!({ "error": error }),
                )
            }),
        _ => None,
    }
}

/// Resolve an attach's paths against this command's directory — run by the pre-flight
/// (before any environment probe) and again by the verb itself.
fn resolve_upload_files(files: &[String]) -> Result<Vec<std::path::PathBuf>, (OutKind, String)> {
    let cwd = std::env::current_dir().map_err(|e| {
        (
            OutKind::Environment,
            format!("the directory this command was run in cannot be read: {e}"),
        )
    })?;
    files
        .iter()
        .map(|raw| {
            crate::chrome::files::resolve_local_file(raw, &cwd)
                .map_err(|reason| (OutKind::Usage, reason))
        })
        .collect()
}

/// One parsed call: the verbs that run in no session are answered as they are; every
/// other action drives the browser in the call's session via [`run_action`].
async fn dispatch(invocation: &Invocation, session: &CliSession) -> DispatchOutcome {
    if invocation.action.runs_in_session() {
        run_action(&invocation.action, session).await
    } else {
        // Still clocked for their envelope; none opens a page, so no tab report is kept.
        let started = Instant::now();
        let (mut env, _) = execute(&invocation.action, &session.name).await;
        stamp_elapsed(&mut env, started);
        DispatchOutcome::plain(env)
    }
}

/// One action that drives the browser in its session: the pre-flight, the record of
/// this call's use, the step, and the envelope's post-step facts. `elapsed_ms` covers
/// the pre-flight, `open`'s baseline read and the step.
async fn run_action(action: &Action, session: &CliSession) -> DispatchOutcome {
    // The action's own clock: every envelope it returns is stamped with it.
    let started = Instant::now();
    let use_outcome = match preflight(action, session, started).await {
        Ok(outcome) => outcome,
        Err(refused) => return refused,
    };
    // The session exists from here on: chrome-use is handed the lifetime as its cleanup
    // backstop, or the cleanup switched off for a kept one.
    set_group_cleanup_backstop(cli_sessions::backstop_ms(use_outcome.kept));
    CLI_STEP_KILLED_BY_CLOCK.store(false, Ordering::Relaxed);
    // `open` names no tab in its reply, so the tabs the session already had are read
    // before the step; an unreadable answer only costs the moving. The pre-flight already
    // refused a step its own arguments reject, so no listing is spent on one.
    let tabs_before = match action {
        Action::Open { .. } if !session.ephemeral => tabs_before_open(&session.name).await,
        _ => None,
    };
    let (mut env, opened_tab) = execute(action, &session.name).await;
    // Read the kill attribution before ANY recovery: the probe's own steps cannot
    // re-attribute this action's ending.
    let killed_by_clock = CLI_STEP_KILLED_BY_CLOCK.load(Ordering::Relaxed);
    // The resolved session and the call's wall time ride every envelope ([`situate`]).
    situate(&mut env, &session.name, started);
    // A process chrome-use left behind holding a step's output channel is a NOTE whatever
    // the outcome, alongside `error`, never changing the kind or the exit code.
    if CHROME_USE_LEFTOVER_PIPES.load(Ordering::Relaxed) {
        surface_leftover_note(&mut env, &session.name);
    }
    // A session that stopped answering is RECOVERED here — the one place the resolved name
    // is in scope. It runs before the group tail below: a wedged session answers no listing,
    // so the tail would spend its bound on a daemon about to be replaced.
    recover_session_wedge(&mut env, &session.name, !session.ephemeral, killed_by_clock).await;
    // The tab this step created leaves chrome-use's group here, before the caller sees the
    // result, and the session's groups are collapsed ([`chrome_groups`]). Best-effort and
    // silent; a step with no established tab leaves it to the post-envelope tail.
    let opened = match (tabs_before.as_ref(), opened_tab.as_deref()) {
        (Some(before), _) => Some(Opened::Baseline(before)),
        (None, Some(tab)) => Some(Opened::Tab(tab)),
        (None, None) => None,
    };
    let mut tail_ran = false;
    if let (Some(opened), Some(cli)) = (opened, tail_cli(&env, session).as_deref()) {
        tail_ran = true;
        chrome_groups::leave_command_line_tabs(cli, &session.name, opened).await;
    }
    DispatchOutcome {
        envelope: env,
        session: Some(session.clone()),
        marked: use_outcome.marked,
        tail_ran,
    }
}

/// The tabs the session held before an `open` step: its baseline. `None` when there is
/// nothing to compare against (the browser would not answer, no helper).
async fn tabs_before_open(session: &str) -> Option<std::collections::BTreeSet<i64>> {
    let cli = cli_path()?;
    chrome_groups::created_tabs(&cli, session).await
}

/// The chrome-use binary a step's group tail drives, when the step's session can be
/// reached at all: a named session whose step ran and whose helper spawned.
fn tail_cli(env: &OutEnvelope, session: &CliSession) -> Option<std::path::PathBuf> {
    if env.kind == OutKind::Environment
        || !CHROME_USE_SPAWNED.load(Ordering::Relaxed)
        || session.ephemeral
    {
        return None;
    }
    cli_path()
}

/// Every action's own work in one place. `opened` is the tab a verb's reply named as one
/// its page opened — the caller's own local, never process state.
async fn execute(action: &Action, name: &str) -> (OutEnvelope, Option<String>) {
    let mut opened = None;
    let env = match action {
        Action::Status => status().await,
        Action::SessionStop { name, force } => session_stop(name, *force).await,
        Action::SessionStatus { name } => session_status(name).await,
        Action::SessionList => session_list(),
        Action::Open {
            url,
            expect,
            structural,
            timeout,
        } => open(url, expect.as_deref(), *structural, *timeout, name).await,
        Action::Count { selector, bound } => count(selector, *bound, name).await,
        Action::Wait { target, timeout } => wait(target, *timeout, name).await,
        Action::Expect { cond, timeout } => expect(cond, *timeout, name).await,
        Action::Eval { js, bound } => eval(js, *bound, name).await,
        Action::Extract {
            schema_file,
            limit,
            bound,
        } => extract(schema_file, *limit, *bound, name).await,
        Action::Click {
            selector,
            if_present,
            bound,
        } => {
            let (env, tab) = click(selector, *if_present, *bound, name).await;
            opened = tab;
            env
        }
        Action::Fill {
            selector,
            source,
            bound,
        } => fill(selector, source, *bound, name).await,
        Action::Type {
            selector,
            text,
            key_events,
            bound,
        } => r#type(selector, text, *key_events, *bound, name).await,
        Action::Press {
            key,
            selector,
            hold,
            bound,
        } => press(key, selector.as_deref(), *hold, *bound, name).await,
        Action::Upload {
            selector,
            files,
            bound,
        } => upload(selector, files, *bound, name).await,
        Action::Screenshot { output_dir } => screenshot(output_dir.as_deref(), name).await,
    };
    (env, opened)
}

/// The action's pre-flight: the pure argument checks, the keep contract, the helper's
/// presence, this call's record of use and the readiness gate. An `Err` is the refusal
/// envelope; the early refusals come before anything is recorded.
async fn preflight(
    action: &Action,
    session: &CliSession,
    started: Instant,
) -> Result<cli_sessions::UseOutcome, DispatchOutcome> {
    let name = session.name.as_str();
    // A malformed URL or `--expect` selector is a usage error whatever the environment, so
    // it is answered without a readiness probe that would bury it.
    if let Some(mut refused) = validate_action_args(action) {
        situate(&mut refused, name, started);
        return Err(DispatchOutcome::plain(refused));
    }
    // A session's lifetime contract is fixed at creation: `--keep` on an existing unmarked
    // session is a usage error, answered before any environment probe.
    let recorded = match cli_sessions::check_keep(name, session.keep) {
        Ok(recorded) => recorded,
        Err(error) => return Err(refusal(action, name, OutKind::Usage, &error, started)),
    };
    // The pre-action gate: no action is dispatched until the connection to the owner's REAL
    // browser is established, so work never lands in a browser chrome-use launched itself.
    // The probe/cleanup verbs are answered before it. A missing helper is checked BEFORE the
    // gate, so it reports its own precise refusal rather than the long readiness one.
    let Some(path) = cli_path() else {
        return Err(refusal(
            action,
            name,
            OutKind::Environment,
            "chrome-use CLI not found",
            started,
        ));
    };
    // `--keep` on an unseen session creates only if chrome-use is not already running it:
    // one predating the record keeps the idle cleanup its daemon started with.
    if session.keep && !recorded && session_running(&path, name).await {
        return Err(refusal(
            action,
            name,
            OutKind::Usage,
            &cli_sessions::mark_refusal(name),
            started,
        ));
    }
    // This call's use restarts the lifetime, applies `--keep` and adds the in-flight mark:
    // from here no sweep closes the session, not even during the gate below.
    let use_outcome = match cli_sessions::note_use(name, session.keep) {
        Ok(outcome) => outcome,
        Err(error) => return Err(refusal(action, name, OutKind::Usage, &error, started)),
    };
    SESSION_RECORD_CREATED.store(use_outcome.created, Ordering::Relaxed);
    if let Err(why) = ensure_ready_for_actions().await {
        // No step: the mark goes and an unmaterialized record is dropped, as the tail does.
        if record_materialized() {
            cli_sessions::note_end(name);
        } else {
            cli_sessions::forget(name);
        }
        return Err(refusal(action, name, OutKind::Environment, &why, started));
    }
    Ok(use_outcome)
}

/// An action's refusal envelope: printed on stderr, carrying the resolved session (so a
/// swallowed or defaulted `--session` is observable) and the action's own clock.
fn refusal(
    action: &Action,
    name: &str,
    kind: OutKind,
    error: &str,
    started: Instant,
) -> DispatchOutcome {
    eprintln!("mahbot chrome: {error}");
    let mut env = out_env(action.name(), false, kind, json!({ "error": error }));
    situate(&mut env, name, started);
    DispatchOutcome::plain(env)
}

/// The facts every action's envelope carries: the resolved session (so a defaulted
/// `--session` is observable) and the wall time; `timeout_ms` stays separate.
fn situate(env: &mut OutEnvelope, name: &str, started: Instant) {
    surface_session(env, name);
    stamp_elapsed(env, started);
}

/// Insert the leftover-helper note as a top-level field. It carries a `session stop`
/// recipe, so it names the session the way a person writes it
/// ([`cli_sessions::display_name`]) — which `session stop` resolves.
fn surface_leftover_note(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        let name = cli_sessions::display_name(name);
        obj.insert("leftover".into(), json!(chrome_leftover_pipe_note(name)));
    }
}

/// Insert the session's name as a top-level envelope field: the name a person reads
/// ([`cli_sessions::display_name`]), never the record's own key.
fn surface_session(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("session".into(), json!(cli_sessions::display_name(name)));
    }
}

/// Stamp one action envelope with how long the call took, from the instant its clock
/// started: every envelope of an action that RAN carries it, because a duration is a
/// fact about the call. The one exception is a usage refusal of a command line that
/// never parsed ([`run_cli`]).
fn stamp_elapsed(env: &mut OutEnvelope, started: Instant) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("elapsed_ms".into(), json!(started.elapsed().as_millis()));
    }
}

/// The clocks one CLI step runs to. The two-clock rule is
/// [the chrome policy](crate::chrome)'s; each constructor states which shape a step of
/// this surface takes. A product kill reports its own `kill` as `timeout_ms`.
#[derive(Debug, Clone, Copy)]
struct StepClocks {
    call: ChromeCallClocks,
    /// Whether chrome-use's own relay self-heal may run inside this step.
    recovery: CliRecovery,
}

impl StepClocks {
    /// Assemble a step's clocks once for the step — the derived pair plus that same
    /// binding, so a step's kill and the policy it spawns with can never disagree.
    fn of(declared: Option<Duration>, recovery: CliRecovery) -> Self {
        Self {
            call: clocks(declared, recovery),
            recovery,
        }
    }

    /// `wait`/`expect`: the step forwards its declared deadline as `--timeout`, which
    /// chrome-use honours in full, so that value IS the chrome-side clock.
    fn forwarded(deadline: Duration) -> Self {
        Self::of(Some(deadline), CliRecovery::Allowed)
    }

    /// Every other agent-facing verb (click, count, eval, extract, fill, type, press,
    /// upload): no `--timeout` chrome-use honours is forwarded, so
    /// [`CHROME_USE_DECLARED_BUDGET`] is the clock it is given. `own_bound` is the caller's
    /// `--timeout`, accepted only at or above that clock; a larger one widens the kill
    /// alone, since declaring it to chrome-use would make it fail as a wedged session.
    fn tool_clock(own_bound: Option<Duration>) -> Self {
        let recovery = CliRecovery::Allowed;
        let bound = own_bound.unwrap_or(CHROME_USE_DECLARED_BUDGET);
        Self {
            call: ChromeCallClocks {
                chrome_side: CHROME_USE_DECLARED_BUDGET,
                kill: kill_bound(bound, recovery),
            },
            recovery,
        }
    }

    /// The composite operation's OWN step (`open`'s navigation): chrome-use runs it to its
    /// own tolerance, so the declared clock is [`CHROME_USE_DECLARED_BUDGET`] and the kill
    /// rides [`crate::chrome::kill_bound`] above it — the operation may outlive `--timeout`.
    fn operation() -> Self {
        Self::of(None, CliRecovery::Allowed)
    }

    /// A step of an operation the product bounds itself: `bound` is the product-side budget,
    /// `chrome_side` the clock declared to chrome-use. A probe in all but name (see the
    /// [chrome policy](crate::chrome)): the relay self-heal is suppressed and the kill rides
    /// [`KILL_SLACK`] above `bound`, so a product bound ends it, never chrome-use's verdict.
    /// Its failure never fails the operation.
    fn bounded(chrome_side: Duration, bound: Duration) -> Self {
        let recovery = CliRecovery::Suppressed;
        Self {
            call: ChromeCallClocks {
                chrome_side,
                kill: kill_bound(bound, recovery),
            },
            recovery,
        }
    }

    /// The product's own probe (`session stop`, `session status`): `bound` IS the kill and
    /// [`probe_clocks`] declares it to chrome-use (capped at [`CHROME_USE_DECLARED_BUDGET`]);
    /// the relay self-heal is suppressed — the documented exception.
    fn probe(bound: Duration) -> Self {
        let recovery = CliRecovery::Suppressed;
        Self {
            call: probe_clocks(bound),
            recovery,
        }
    }
}

/// Run one chrome-use step under `clocks` and classify its output. `session` scopes the
/// call; `None` leaves it unscoped; `input` pipes a stdin payload.
async fn spawn_step(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    clocks: StepClocks,
    input: Option<&[u8]>,
) -> StepOutcome {
    // A session-scoped spawn attempt may materialize the ephemeral session (a
    // vanished-binary race still sets this; the close is then a harmless no-op).
    if session.is_some() {
        CHROME_USE_SPAWNED.store(true, Ordering::Relaxed);
    }
    let failed = |kind: OutKind, message: String| Err(StepFailure { kind, message });
    match spawn_cli(CliSpawn {
        path,
        args,
        session,
        json: true,
        capture_stderr: true,
        timeout: clocks.call.kill,
        cancel_kills: true,
        input: input.map(<[u8]>::to_vec),
        chrome_side: clocks.call.chrome_side,
        recovery: clocks.recovery,
    })
    .await
    {
        CliRun::SpawnFailure => failed(
            OutKind::Environment,
            "chrome-use CLI could not be spawned/found".to_string(),
        ),
        CliRun::TimedOut => {
            // The product's own clock ended this step: wedge evidence only when the step's outcome
            // IS the caller's verdict ([`CliRecovery::Allowed`]) and its kill rode above the clock
            // chrome-use worked to — otherwise chrome-use did not answer its own deadline
            // (see [`CLI_STEP_KILLED_BY_CLOCK`]).
            if matches!(clocks.recovery, CliRecovery::Allowed)
                && clocks.call.kill > clocks.call.chrome_side
            {
                CLI_STEP_KILLED_BY_CLOCK.store(true, Ordering::Relaxed);
            }
            Err(StepFailure {
                kind: OutKind::Timeout,
                message: String::new(),
            })
        }
        CliRun::Output(out) => {
            // The child's exit decides the step; a process it left behind holding the
            // output channel is recorded here as a NOTE, never a failure. OR-ed, not stored.
            CHROME_USE_LEFTOVER_PIPES.fetch_or(out.leftover_pipes, Ordering::Relaxed);
            classify_step_output(
                args.first() == Some(&"expect"),
                out.truncated,
                out.status.code(),
                out.status.success(),
                &out.stdout,
                String::from_utf8_lossy(&out.stderr).trim(),
            )
        }
    }
}

/// Classify one chrome-use output into a step outcome. `expect` is the one action whose
/// envelope can succeed on a non-zero exit (the verdict rides in `data`), so there — and
/// nowhere else — envelope-success is trusted over the code. `truncated` names an answer
/// the product cut off; the parse is the shared tolerant one ([`parse_first`]). Pure so
/// tests pin the mapping.
fn classify_step_output(
    expect_style: bool,
    truncated: bool,
    exit_code: Option<i32>,
    status_success: bool,
    stdout: &[u8],
    stderr: &str,
) -> StepOutcome {
    let failed = |kind: OutKind, message: String| Err(StepFailure { kind, message });
    // Fallback message for a non-success exit / unparseable stdout.
    let fallback = || fallback_step_message(exit_code, stderr);
    let classified = |resp: ChromeResponse| {
        let code = resp.code.clone();
        let msg = resp
            .error
            .filter(|e| !e.trim().is_empty())
            .unwrap_or_else(fallback);
        let kind = classify_call_failure(code.as_deref(), &msg);
        failed(kind, sanitize_timeout_message(kind, &msg))
    };
    let parsed: Option<ChromeResponse> = parse_first::<ChromeResponse>(stdout);
    if !status_success {
        return match parsed {
            Some(resp) if expect_style && resp.is_success() => {
                self_launched_failure(&resp).map_or(Ok(resp), Err)
            }
            Some(resp) => classified(resp),
            None if truncated => failed(OutKind::Error, truncated_output_error()),
            None => {
                // The stderr fallback can carry a timeout phrasing plus the canned hint
                // (chrome-use prints diagnostics there when stdout is non-JSON), so it is
                // classified and sanitized.
                let msg = fallback();
                let kind = classify_call_failure(None, &msg);
                failed(kind, sanitize_timeout_message(kind, &msg))
            }
        };
    }
    match parsed {
        Some(resp) if resp.is_success() => self_launched_failure(&resp).map_or(Ok(resp), Err),
        Some(resp) => classified(resp),
        None => failed(
            OutKind::Error,
            if truncated {
                truncated_output_error()
            } else if stderr.is_empty() {
                "chrome-use returned non-JSON output".to_string()
            } else {
                stderr.to_string()
            },
        ),
    }
}

/// A successful step whose envelope carries chrome-use's browser-replacement note is a
/// plain failure: the command ran in a browser chrome-use launched ITSELF, so the page
/// state the caller assumed is not there. [`OutKind::Environment`] (rc 2), and
/// deliberately not the daemon-down path — only the real-browser connection was lost.
fn self_launched_failure(resp: &ChromeResponse) -> Option<StepFailure> {
    self_launched_browser_note(resp).map(|note| StepFailure {
        kind: OutKind::Environment,
        message: self_launched_browser_cli_error(&note),
    })
}

#[must_use]
fn fallback_step_message(status_code: Option<i32>, stderr: &str) -> String {
    if stderr.is_empty() {
        match status_code {
            Some(c) => format!("chrome-use exited with code {c}"),
            None => "chrome-use exited with a non-zero status".to_string(),
        }
    } else {
        stderr.to_string()
    }
}

/// The `out_env` kind contract: a kind emitted for a registered CLI action must be
/// declared in that action's kind list, so help cannot drift. Pure, so it is testable in
/// every profile; the `"usage"` pseudo-action is exempt.
fn kind_contract_holds(action: &str, kind: OutKind) -> bool {
    match actions::desc(action) {
        None => true,
        Some(d) => d.cli.as_ref().is_none_or(|c| c.kinds.contains(&kind)),
    }
}

#[must_use]
fn out_env(action: &str, ok: bool, kind: OutKind, payload: Value) -> OutEnvelope {
    debug_assert!(
        kind_contract_holds(action, kind),
        "kind {} not declared for action {action} — update the shared registry",
        kind.as_str()
    );
    OutEnvelope {
        action: action.to_string(),
        ok,
        kind,
        payload,
    }
}

/// Resolve the chrome-use CLI path, or build the environment-failure envelope. Only the
/// product's own copy is resolved ([`crate::tools::chrome_daemon::cli_path`]), never one
/// on the owner's PATH, so a foreign copy is never run.
fn require_cli(
    action: &str,
    request: serde_json::Map<String, Value>,
) -> Result<std::path::PathBuf, OutEnvelope> {
    cli_path().ok_or_else(|| {
        let mut payload = request;
        payload.insert("error".into(), json!("chrome-use CLI not found"));
        out_env(action, false, OutKind::Environment, Value::Object(payload))
    })
}

/// The request fields an action reports when its own envelope cannot be built. Pairs make
/// the payload an object by construction, so a refusal loses neither half.
fn request(
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> serde_json::Map<String, Value> {
    fields
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// `status` — pure preflight: the readiness report and RAW probes only (never the verbs
/// and the gate that recover). It states the `verdict` the report amounts to: `ready`
/// (rc 0), `blocked` (rc 2 — a fact established the connection cannot work), `not-proven`
/// (rc 0 — nothing established, and the gate would still dispatch).
async fn status() -> OutEnvelope {
    // One `--version` spawn in the happy path: a parsed version proves the CLI is
    // available; `cli_probe` only re-runs to classify WHY it didn't.
    let (chrome_use, version_ok) = match cli_version().await {
        Some(v) => (v.to_string(), true),
        None => match cli_path() {
            None => ("missing".to_string(), false),
            Some(_) => match cli_probe().await {
                CliStatus::Available => ("present".to_string(), true),
                CliStatus::Missing => ("missing".to_string(), false),
                CliStatus::Transient(f) => (f.to_string(), false),
            },
        },
    };
    // One bounded, daemon-free readiness snapshot (cached for HEALTH_TTL), every fact
    // tri-state, the display fact included.
    let readiness = readiness().await;

    let mut payload = serde_json::Map::new();
    payload.insert("chrome_use".into(), json!(chrome_use));
    payload.insert("relay_up".into(), json!(readiness.relay_up));
    payload.insert("chrome_running".into(), json!(readiness.chrome_running));
    payload.insert("display".into(), json!(readiness.display));
    payload.insert(
        "ready_for_actions".into(),
        json!(readiness.ready_for_actions()),
    );
    // The three-way state in one word, so a machine read does not infer it from the report
    // text: `ready` is PROVEN, `blocked` ruled out, `not-proven` neither.
    payload.insert(
        "verdict".into(),
        json!(if readiness.ready_for_actions() {
            "ready"
        } else if readiness.blocked() {
            "blocked"
        } else {
            "not-proven"
        }),
    );
    payload.insert("readiness".into(), json!(readiness.report()));

    if readiness.ready_for_actions() {
        return out_env("status", true, OutKind::Ok, Value::Object(payload));
    }
    // Three states, mirroring the pre-action gate. A CLI that cannot state its version, and a
    // fact RULING the connection out, are environment failures; a snapshot that established
    // nothing is a reporting gap, not a broken environment.
    let error = if !version_ok {
        Some(format!(
            "chrome-use CLI: {chrome_use}. {}",
            readiness.refusal()
        ))
    } else if readiness.blocked() {
        Some(readiness.refusal())
    } else {
        None
    };
    let Some(error) = error else {
        return out_env("status", true, OutKind::Ok, Value::Object(payload));
    };
    eprintln!("mahbot chrome: {error}");
    payload.insert("error".into(), json!(error));
    out_env(
        "status",
        false,
        OutKind::Environment,
        Value::Object(payload),
    )
}

/// The honest `error` text for a committed Chrome error page: the specific cause when a net
/// error code was extracted, the generic reason otherwise.
fn network_failure_error(code: Option<&str>) -> String {
    match code {
        Some(c) => {
            format!(
                "Chrome rendered an error page — {c} ({})",
                net_error_phrase(c)
            )
        }
        None => "Chrome rendered an error page (site unreachable or refused)".to_string(),
    }
}

/// The factual `error` text for a product-side deadline kill; `what` names the step. It
/// says the PRODUCT's bound ended the call, never that chrome-use timed out, and on a step
/// whose outcome is the caller's verdict it adds that the kill rode above the declared
/// clock, so a truncated call is not mistaken for chrome-use's verdict.
fn deadline_error(chrome_side: Duration, kill: Duration, what: &str) -> String {
    let kill = kill.as_millis();
    if kill > chrome_side.as_millis() {
        format!(
            "deadline reached after {kill}ms — {what}: that is the product's own bound, which \
             rode above the {}ms deadline chrome-use itself was working to, so chrome-use never \
             reported its own reason",
            chrome_side.as_millis()
        )
    } else {
        // The product bounded this step itself: its probe bound or the operation's budget.
        format!(
            "deadline reached after {kill}ms — {what}: that is the product's own bound for the \
             step (chrome-use was given {}ms), so a product-side bound — not chrome-use's \
             verdict — ended the call",
            chrome_side.as_millis()
        )
    }
}

/// Post-navigation settle cap for `open`'s plain path: a best-effort
/// `wait --load networkidle`, so the next step is not racing a settling SPA.
const SETTLE_CAP: Duration = Duration::from_secs(10);

/// Budget reserved for content capture, so the settle cannot drop the snapshot.
const CAPTURE_RESERVE: Duration = Duration::from_millis(2500);

/// Below this a settle spawn is not worth its startup cost — skipped.
const MIN_SETTLE_BUDGET: Duration = Duration::from_secs(1);

/// Settle budget for `open`'s plain path: capped at [`SETTLE_CAP`], leaving
/// [`CAPTURE_RESERVE`], `None` when the remainder is too small for both. Pure.
fn settle_budget(remaining: Duration) -> Option<Duration> {
    let budget = remaining.saturating_sub(CAPTURE_RESERVE).min(SETTLE_CAP);
    (budget >= MIN_SETTLE_BUDGET).then_some(budget)
}

/// The pure argument checks of `open`, run by [`preflight`] (before the readiness gate)
/// and by [`open`] itself: the URL must be one this product may navigate to, and
/// `--expect` goes through the shared wait-target policy — both before navigating.
fn open_args(url: &str, expect: Option<&str>) -> Result<Option<WaitTarget>, OutEnvelope> {
    if let Err(e) = crate::util::http::validate_url(url) {
        return Err(out_env(
            "open",
            false,
            OutKind::Usage,
            json!({ "url": url, "error": e.to_string() }),
        ));
    }
    expect
        .map(|sel| wait_target(Some(sel), None, None))
        .transpose()
        .map_err(|e| {
            out_env(
                "open",
                false,
                OutKind::Usage,
                json!({ "url": url, "error": e }),
            )
        })
}

/// `open` — navigate to `url`, optionally wait for `--expect` (redesign-aware via
/// `--structural`), report the committed URL and best-effort attach the content. On the
/// plain path a best-effort settle follows, capped at [`SETTLE_CAP`]. `timeout` bounds the
/// waits the operation drives, NOT the navigation: the operation may outlive it.
#[expect(clippy::too_many_lines)]
async fn open(
    url: &str,
    expect: Option<&str>,
    structural: bool,
    timeout: Duration,
    session: &str,
) -> OutEnvelope {
    let wait_for = match open_args(url, expect) {
        Ok(wait_for) => wait_for,
        Err(refusal) => return refusal,
    };
    let path = match require_cli("open", request([("url", json!(url))])) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let started = Instant::now();
    let total = timeout + KILL_SLACK;
    // The operation's own step: neither `timeout` nor `total` bounds the NAVIGATION —
    // chrome-use runs it to the clock mahbot declares, and killing it on the operation's
    // budget would lose chrome-use's own reason for the failure. The kill rides `kill_bound`
    // above that budget; `total` (`timeout` plus the error-page slack) is the operation's own
    // bookkeeping for the probe, the settle and the capture.
    let operation = StepClocks::operation();
    let resp = match spawn_step(&path, &["open", url], Some(session), operation, None).await {
        Ok(resp) => resp,
        Err(f) => return f.envelope("open", json!({ "url": url }), operation),
    };
    let final_url = resp
        .data
        .as_ref()
        .and_then(|d| d.get("url"))
        .and_then(Value::as_str)
        .unwrap_or(url)
        .to_string();
    if is_blank_page_url(&final_url) {
        return out_env(
            "open",
            false,
            OutKind::Network,
            json!({
                "url": url,
                "error": "navigation never committed — tab still on about:blank"
            }),
        );
    }
    // One of the operation's best-effort sub-steps (the policy exception): reserved up to the
    // 2 s slack so it runs even when the navigation spent the deadline, its kill riding that
    // slack above it. One eval yields the error-page verdict and the net error code.
    let probe_budget = total.saturating_sub(started.elapsed()).min(KILL_SLACK);
    if probe_budget >= Duration::from_millis(500)
        && let Ok(probe) = spawn_step(
            &path,
            &["eval", ERROR_PAGE_PROBE_JS],
            Some(session),
            StepClocks::bounded(CHROME_USE_DECLARED_BUDGET, probe_budget),
            None,
        )
        .await
        && let Some(ErrorPageProbe {
            is_error_page: true,
            code,
        }) = parse_error_page_probe(&probe)
    {
        let mut payload = json!({
            "url": url,
            "error": network_failure_error(code.as_deref()),
        });
        if let Some(code) = code {
            payload["error_code"] = json!(code);
        }
        return out_env("open", false, OutKind::Network, payload);
    }
    if let Some(target) = wait_for {
        let target_desc = target.describe();
        let mut payload = json!({ "url": final_url, "target": target_desc });
        // A missed `--expect` after a committed navigation is a suspected redesign when
        // `--structural` was declared, a plain timeout otherwise.
        let (timeout_kind, hint) = if structural {
            (
                OutKind::Redesign,
                "possible DOM redesign or structural change",
            )
        } else {
            (
                OutKind::Timeout,
                "selector not found — may be structural change, empty region, or content-dependent",
            )
        };
        // The wait's declared deadline is what REMAINS of the caller's `--timeout`, never the
        // bookkeeping `total`, whose slack belongs to the error-page probe. A declaration at or
        // above chrome-use's tolerance is already refused at parse time.
        let wait_budget = timeout.saturating_sub(started.elapsed());
        // No budget left for the wait (or its capture): the same timeout envelope, reporting the
        // OPERATION's own budget, since no deadline was ever declared to chrome-use for it.
        if wait_budget < Duration::from_millis(250) {
            payload["timeout_ms"] = json!(timeout.as_millis());
            payload["error"] = json!(format!(
                "no budget left for the `--expect` wait: the operation's own --timeout budget \
                 ({}ms) was spent by the navigation and the steps around it before the wait \
                 could be given any of it, so chrome-use was never given a deadline for that \
                 step — this is the product's own bound on `open`, not chrome-use's verdict. \
                 Raise --timeout or retry now that the page is open.",
                timeout.as_millis()
            ));
            payload["hint"] = json!(hint);
            return out_env("open", false, timeout_kind, payload);
        }
        // The wait forwards its remaining budget as `--timeout`, so that IS the step's chrome-side
        // clock, and its failure IS the caller's verdict: relay self-heal allowed.
        let wait_clocks = StepClocks::forwarded(wait_budget);
        let wargs = wait_args(&target, wait_budget.as_millis());
        let refs: Vec<&str> = wargs.iter().map(String::as_str).collect();
        let waited = spawn_step(&path, &refs, Some(session), wait_clocks, None).await;
        // The page is open regardless of the wait outcome, so content rides on failure envelopes
        // too — it is what diagnoses a redesign. Slow waits simply exhaust it.
        let content =
            capture_open_content(&path, session, total.saturating_sub(started.elapsed())).await;
        if let Some(content) = content {
            payload["content"] = json!(content);
        }
        return match waited {
            Ok(_) => out_env("open", true, OutKind::Ok, payload),
            Err(f) if f.kind == OutKind::Timeout => {
                // chrome-use's verdict reports the deadline it was given; a product kill
                // reports its own bound.
                let reported = if f.message.is_empty() {
                    wait_clocks.call.kill
                } else {
                    wait_clocks.call.chrome_side
                };
                payload["timeout_ms"] = json!(reported.as_millis());
                // The chrome-side message when chrome-use phrased it; a factual deadline
                // text on a product kill.
                if f.message.is_empty() {
                    payload["error"] = json!(deadline_error(
                        wait_clocks.call.chrome_side,
                        wait_clocks.call.kill,
                        "the target never appeared"
                    ));
                } else {
                    payload["error"] = json!(f.message);
                }
                payload["hint"] = json!(hint);
                out_env("open", false, timeout_kind, payload)
            }
            Err(f) => f.envelope("open", payload, wait_clocks),
        };
    }
    // Best-effort settle: heavy SPAs keep the network busy right after the navigation commits,
    // so the next step would otherwise race a still-settling page. Raw argv (the `wait` action
    // does not expose `--load`); its clock is the declared AGENT_BROWSER_DEFAULT_TIMEOUT and
    // the product's own bound the remaining budget, so the kill rides `KILL_SLACK` above it.
    // Discarded: a settle timeout never downgrades a committed navigation.
    if let Some(budget) = settle_budget(total.saturating_sub(started.elapsed())) {
        let _ = spawn_step(
            &path,
            &["wait", "--load", "networkidle"],
            Some(session),
            StepClocks::bounded(budget, budget),
            None,
        )
        .await;
    }
    // Content capture is best-effort and bounded: a failure, exhaustion or content-free page
    // simply omits `content`, never downgrading a successful navigation. Its product bound is
    // the remaining budget, the kill riding `KILL_SLACK` above it.
    let mut payload = json!({ "url": final_url });
    let budget = total.saturating_sub(started.elapsed());
    if let Some(content) = capture_open_content(&path, session, budget).await {
        payload["content"] = json!(content);
    }
    out_env("open", true, OutKind::Ok, payload)
}

/// Best-effort compact page snapshot for the `open` envelope, the content form the
/// interactive chrome tool surfaces: bounded ([`StepClocks::bounded`]), skipped under 500 ms,
/// and a failed step or content-free page yields none. Byte-truncated so the single-line
/// envelope stays valid (`TOOL_OUTPUT_BUDGET_BYTES`, plus its marker).
async fn capture_open_content(path: &Path, session: &str, budget: Duration) -> Option<String> {
    if budget < Duration::from_millis(500) {
        return None;
    }
    let resp = spawn_step(
        path,
        &["snapshot", "-c"],
        Some(session),
        StepClocks::bounded(CHROME_USE_DECLARED_BUDGET, budget),
        None,
    )
    .await
    .ok()?;
    let text = resp.data.as_ref().and_then(extract_snapshot_text)?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(truncate_sandwich(
        text,
        TOOL_OUTPUT_BUDGET_BYTES,
        "page content",
    ))
}

/// Run the count-eval shim for `selector`, shared by `count` and the `extract` honest-empty
/// gate: a step of the action that called it, so it runs to its clocks.
async fn count_via_eval(
    path: &Path,
    selector: &str,
    session: &str,
    clocks: StepClocks,
) -> Result<u64, StepFailure> {
    let js = count_eval_js(selector);
    let args = ["eval".to_string(), js];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let resp = spawn_step(path, &refs, Some(session), clocks, None).await?;
    eval_count(&resp).ok_or(StepFailure {
        kind: OutKind::Error,
        message: "count eval returned a non-numeric result".to_string(),
    })
}

/// `count` — eval shim over `querySelectorAll` (chrome-use has no `count` verb).
async fn count(selector: &str, bound: Option<Duration>, session: &str) -> OutEnvelope {
    let path = match require_cli("count", request([("selector", json!(selector))])) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let clocks = StepClocks::tool_clock(bound);
    match count_via_eval(&path, selector, session, clocks).await {
        Ok(n) => {
            let kind = if n == 0 { OutKind::Empty } else { OutKind::Ok };
            out_env(
                "count",
                true,
                kind,
                json!({ "selector": selector, "count": n }),
            )
        }
        Err(f) => f.envelope("count", json!({ "selector": selector }), clocks),
    }
}

/// `wait` — bounded wait for a safe [`WaitTarget`] (the numeric sleep form is rejected at
/// parse). `--timeout` IS chrome-use's deadline, so its honest error surfaces at the declared
/// clock; the kill rides the relay-recovery window above it.
async fn wait(target: &WaitTarget, timeout: Duration, session: &str) -> OutEnvelope {
    let base = request([("target", json!(target.describe()))]);
    let path = match require_cli("wait", base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let clocks = StepClocks::forwarded(timeout);
    let args = wait_args(target, timeout.as_millis());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(_) => out_env(
            "wait",
            true,
            OutKind::Ok,
            json!({ "target": target.describe() }),
        ),
        Err(mut f) => {
            with_condition_timeout_note("wait", f.kind, &mut f.message);
            f.envelope("wait", Value::Object(base), clocks)
        }
    }
}

/// Map a parsed expect verdict to the envelope: pass → rc 0, deadline expiration →
/// timeout, plain false → error. Pure so tests pin the mapping.
fn expect_envelope(condition: &str, outcome: ExpectOutcome) -> OutEnvelope {
    let mut payload = json!({ "condition": condition });
    if let Some(actual) = outcome.actual {
        payload["actual"] = actual;
    }
    if outcome.pass {
        payload["pass"] = json!(true);
        return out_env("expect", true, OutKind::Ok, payload);
    }
    payload["pass"] = json!(false);
    let kind = if outcome.timed_out {
        payload["timed_out"] = json!(true);
        payload["error"] = json!(format!(
            "condition was not met within the deadline{EXPECT_TIMEOUT_NOTE}"
        ));
        OutKind::Timeout
    } else {
        payload["error"] = json!("condition is false");
        OutKind::Error
    };
    out_env("expect", false, kind, payload)
}

/// `expect` — assert a condition with a bounded wait. The envelope, not the exit code, is
/// authoritative: a failed assertion arrives as `success:true` (see [`spawn_step`]).
async fn expect(cond: &ExpectCond, timeout: Duration, session: &str) -> OutEnvelope {
    let condition = describe(cond);
    let base = request([("condition", json!(condition))]);
    let path = match require_cli("expect", base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Same rule as wait: the forwarded `--timeout` is chrome-use's own clock.
    let clocks = StepClocks::forwarded(timeout);
    let args = expect_args(cond, timeout.as_millis());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => match resp.data.as_ref().and_then(expect_outcome) {
            Some(outcome) => expect_envelope(&condition, outcome),
            None => out_env(
                "expect",
                false,
                OutKind::Error,
                json!({ "condition": condition, "error": "expect returned an unrecognized payload" }),
            ),
        },
        Err(mut f) => {
            with_condition_timeout_note("expect", f.kind, &mut f.message);
            f.envelope("expect", Value::Object(base), clocks)
        }
    }
}

/// `eval` — run JS and emit the unwrapped result as a JSON value (number, string, object,
/// or null).
async fn eval(js: &str, bound: Option<Duration>, session: &str) -> OutEnvelope {
    let path = match require_cli("eval", request([("js", json!(js))])) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &["eval", js], Some(session), clocks, None).await {
        Ok(resp) => {
            let result = eval_result(&resp).cloned().unwrap_or(Value::Null);
            out_env("eval", true, OutKind::Ok, json!({ "result": result }))
        }
        Err(f) => f.envelope("eval", json!({ "js": js }), clocks),
    }
}

/// `extract` — schema-driven rows extraction, gated by a count so an empty region is
/// reported honestly (rows-mode `extract` returns phantom rows on a 0 match).
async fn extract(
    schema_file: &str,
    limit: Option<usize>,
    bound: Option<Duration>,
    session: &str,
) -> OutEnvelope {
    let schema = match std::fs::read_to_string(schema_file) {
        Ok(s) => match serde_json::from_str::<Value>(&s) {
            Ok(v) => v,
            Err(e) => {
                return out_env(
                    "extract",
                    false,
                    OutKind::Usage,
                    json!({ "error": format!("schema-file is not valid JSON: {e}") }),
                );
            }
        },
        Err(e) => {
            return out_env(
                "extract",
                false,
                OutKind::Usage,
                json!({ "error": format!("cannot read --schema-file: {e}") }),
            );
        }
    };

    if let Err(err) = validate_extract_getters(&schema) {
        return out_env("extract", false, OutKind::Usage, json!({ "error": err }));
    }

    let path = match require_cli("extract", serde_json::Map::new()) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let clocks = StepClocks::tool_clock(bound);
    // Honest-empty gate: when the rows selector matches 0, report empty without invoking
    // chrome-use's phantom-row `extract`.
    if let Some(rows_sel) = schema.get("rows").and_then(Value::as_str) {
        let count = count_via_eval(&path, rows_sel, session, clocks).await;
        match extract_gate(count) {
            ExtractGate::Empty => {
                return out_env(
                    "extract",
                    true,
                    OutKind::Empty,
                    extract_output(&json!([]), limit),
                );
            }
            ExtractGate::Proceed => {}
            ExtractGate::Fail(f) => return f.envelope("extract", json!({}), clocks),
        }
    }

    // `--schema-file` is the installed chrome-use's own documented form; rows-mode answers
    // `{extracted, count, origin, meta}`, single-object mode the bare object.
    let args = [
        "extract".to_string(),
        "--schema-file".to_string(),
        schema_file.to_string(),
    ];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => out_env(
            "extract",
            true,
            OutKind::Ok,
            extract_output(resp.data.as_ref().unwrap_or(&Value::Null), limit),
        ),
        Err(f) => f.envelope("extract", json!({}), clocks),
    }
}

/// `click` — click a selector, forwarding `--if-present` verbatim when set (chrome-use
/// treats an `--if-present` miss as a no-op success).
async fn click(
    selector: &str,
    if_present: bool,
    bound: Option<Duration>,
    session: &str,
) -> (OutEnvelope, Option<String>) {
    let path = match require_cli("click", request([("selector", json!(selector))])) {
        Ok(p) => p,
        Err(e) => return (e, None),
    };
    let mut args = vec!["click".to_string(), selector.to_string()];
    if if_present {
        args.push("--if-present".to_string());
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => {
            // A page the click opens is reported by chrome-use (its reply names the tab); the tail
            // takes it out of the group, which is why the id rides back with the envelope.
            let opened = resp
                .data
                .as_ref()
                .and_then(|data| data.pointer("/openedTab/tabId"))
                .and_then(Value::as_str)
                .map(String::from);
            (
                out_env("click", true, OutKind::Ok, json!({ "selector": selector })),
                opened,
            )
        }
        Err(f) => (
            f.envelope("click", json!({ "selector": selector }), clocks),
            None,
        ),
    }
}

/// Read this process's stdin to EOF for `fill --stdin`, capped so an oversized or
/// accidental stream is a usage error, not a memory event.
const STDIN_TEXT_CAP: usize = 10 * 1024 * 1024;
async fn read_stdin_capped() -> Result<Vec<u8>, String> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    let mut stdin = tokio::io::stdin().take((STDIN_TEXT_CAP + 1) as u64);
    stdin
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("cannot read stdin: {e}"))?;
    if buf.len() > STDIN_TEXT_CAP {
        return Err(format!(
            "stdin text exceeds the {STDIN_TEXT_CAP} byte cap — write it to a file and use --file"
        ));
    }
    if buf.is_empty() {
        return Err(
            "no text on stdin — pipe it, e.g. `cat post.md | mahbot chrome fill \".editor\" --stdin`"
                .to_string(),
        );
    }
    Ok(buf)
}

/// The fill/type/press success handling, pure for tests: a success carrying chrome-use's
/// degraded-success `warning` ([`chrome_use_warning`]) is kind error (rc 1) with the warning
/// surfaced — the action may not have taken effect. A clean success emits kind ok. Browser
/// replacement never reaches here ([`classify_step_output`] already failed it).
fn text_input_ok_envelope(action: &str, base: &Value, resp: ChromeResponse) -> OutEnvelope {
    let warning = chrome_use_warning(&resp);
    let mut payload = base.as_object().cloned().unwrap_or_default();
    if let Some(d) = resp.data {
        payload.insert("data".into(), d);
    }
    if let Some(warning) = warning {
        payload.insert("warning".into(), warning);
        return out_env(action, false, OutKind::Error, Value::Object(payload));
    }
    out_env(action, true, OutKind::Ok, Value::Object(payload))
}

/// `fill` — clear + verified fill. Exactly one text source (inline text, --file
/// passthrough, or mahbot-piped --stdin).
async fn fill(
    selector: &str,
    source: &TextInput,
    bound: Option<Duration>,
    session: &str,
) -> OutEnvelope {
    let action = "fill";
    let base = request([("selector", json!(selector))]);
    let path = match require_cli(action, base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut argv = vec!["fill".to_string(), selector.to_string()];
    argv.extend(source.argv());
    let input = match source {
        TextInput::Stdin => match read_stdin_capped().await {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                return out_env(
                    action,
                    false,
                    OutKind::Usage,
                    json!({ "selector": selector, "error": e }),
                );
            }
        },
        TextInput::File(f) => {
            if !std::path::Path::new(f).is_file() {
                return out_env(
                    action,
                    false,
                    OutKind::Usage,
                    json!({ "selector": selector, "error": format!("--file does not exist: {f}") }),
                );
            }
            None
        }
        TextInput::Inline(_) => None,
    };
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, input.as_deref()).await {
        Ok(resp) => text_input_ok_envelope(action, &Value::Object(base.clone()), resp),
        Err(f) => f.envelope(action, Value::Object(base), clocks),
    }
}

/// `type` — character-by-character typing (appends, does not clear).
async fn r#type(
    selector: &str,
    text: &str,
    key_events: bool,
    bound: Option<Duration>,
    session: &str,
) -> OutEnvelope {
    let action = "type";
    let base = request([("selector", json!(selector)), ("text", json!(text))]);
    let path = match require_cli(action, base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut argv = vec!["type".to_string(), selector.to_string()];
    if key_events {
        argv.push("--key-events".to_string());
    }
    // Leading-dash text rides after the `--` shield; action flags must come before it.
    argv.extend(text_value_argv(text));
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => text_input_ok_envelope(action, &Value::Object(base.clone()), resp),
        Err(f) => f.envelope(action, Value::Object(base), clocks),
    }
}

/// `press` — press a key at the focused element, optionally focusing `--selector` first
/// (a no-op for non-focusable elements) and holding `--hold` ms before release.
async fn press(
    key: &str,
    selector: Option<&str>,
    hold: Option<u64>,
    bound: Option<Duration>,
    session: &str,
) -> OutEnvelope {
    let action = "press";
    let mut base_obj = serde_json::Map::new();
    base_obj.insert("key".into(), json!(key));
    if let Some(sel) = selector {
        base_obj.insert("selector".into(), json!(sel));
    }
    if let Some(h) = hold {
        base_obj.insert("hold_ms".into(), json!(h));
    }
    let base = base_obj;
    let path = match require_cli(action, base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut argv = vec!["press".to_string(), key.to_string()];
    if let Some(sel) = selector {
        argv.extend(["--selector".to_string(), sel.to_string()]);
    }
    if let Some(h) = hold {
        argv.extend(["--hold".to_string(), h.to_string()]);
    }
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => text_input_ok_envelope(action, &Value::Object(base.clone()), resp),
        Err(f) => f.envelope(action, Value::Object(base), clocks),
    }
}

/// The delivery verdict a failed attach carries: a kill at the step's clock learned nothing,
/// and a reported failure may have streamed part of a file — so neither claims the files
/// are on the input.
const ATTACH_INCOMPLETE_NOTE: &str =
    "the attach did not complete — treat the files as not attached";
const DELIVERY_UNCONFIRMED_NOTE: &str =
    "delivery is not confirmed — treat the files as not attached";

/// The payload an attach reports: the target and the paths it was asked to send.
fn upload_base(selector: &str, files: &[String]) -> serde_json::Map<String, Value> {
    let mut base = request([("selector", json!(selector))]);
    base.insert("files".into(), json!(files));
    base
}

/// An attach that never reached the browser, on the same payload as a success.
fn upload_refusal(selector: &str, files: &[String], kind: OutKind, error: &str) -> OutEnvelope {
    let mut payload = upload_base(selector, files);
    payload.insert("error".into(), json!(error));
    out_env("upload", false, kind, Value::Object(payload))
}

/// An attach's envelope from chrome-use's answer: the `uploaded`/`attached` counts, the
/// absolute files read, and its `warning` verbatim when it could not confirm the input still
/// holds them. A stated zero upload is the failure envelope.
fn upload_ok_envelope(
    selector: &str,
    files: &[String],
    resolved: &[std::path::PathBuf],
    response: &ChromeResponse,
    unchecked: bool,
) -> OutEnvelope {
    let (uploaded, attached) = crate::chrome::files::upload_counts(response);
    let mut payload = upload_base(selector, files);
    payload.insert(
        "absolute_files".into(),
        json!(
            resolved
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        ),
    );
    if let Some(n) = uploaded {
        payload.insert("uploaded".into(), json!(n));
    }
    if let Some(n) = attached {
        payload.insert("attached".into(), json!(n));
    }
    if let Some(warning) = chrome_use_warning(response) {
        payload.insert("warning".into(), warning);
    }
    // A stated zero is the opposite of delivery: the attach failed whatever the exit said.
    if uploaded == Some(0) {
        payload.insert(
            "error".into(),
            json!(format!(
                "the browser side reported 0 of {} file(s) uploaded — nothing reached the page",
                resolved.len()
            )),
        );
        return out_env("upload", false, OutKind::Error, Value::Object(payload));
    }
    if unchecked {
        payload.insert(
            "note".into(),
            json!(crate::chrome::files::unchecked_several_files_note()),
        );
    }
    out_env("upload", true, OutKind::Ok, Value::Object(payload))
}

/// `upload` — attach local files to the page's file input, the label that pairs with it, or
/// the container holding it. Paths resolve against this command's directory and are handed
/// to chrome-use absolute, which is what the result reports. A `warning` it raises is a
/// degraded success, never a claim about the page's own validation.
async fn upload(
    selector: &str,
    files: &[String],
    bound: Option<Duration>,
    session: &str,
) -> OutEnvelope {
    let action = "upload";
    let base = upload_base(selector, files);
    // The pre-flight has already refused a path the caller cannot read (usage,
    // whatever the environment); this is the resolved list it hands chrome-use.
    let resolved = match resolve_upload_files(files) {
        Ok(resolved) => resolved,
        Err((kind, reason)) => return upload_refusal(selector, files, kind, &reason),
    };
    let path = match require_cli(action, base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Several files at a one-file input is a state a real selection can never
    // produce: the page would see a FileList no user could have chosen.
    let mut unchecked = false;
    match probe_one_file_target(&path, selector, resolved.len(), session).await {
        OneFileTarget::Refused(reason) => {
            return upload_refusal(selector, files, OutKind::Usage, &reason);
        }
        OneFileTarget::Unjudged => unchecked = true,
        OneFileTarget::Proceed => {}
    }
    let argv = crate::chrome::files::upload_argv(selector, &resolved);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => upload_ok_envelope(selector, files, &resolved, &resp, unchecked),
        Err(f) => {
            let note = if f.kind == OutKind::Timeout {
                DELIVERY_UNCONFIRMED_NOTE
            } else {
                ATTACH_INCOMPLETE_NOTE
            };
            let mut env = f.envelope(action, Value::Object(base), clocks);
            append_error_note(&mut env, note);
            env
        }
    }
}

/// Ask the page the one-file question the files module puts to a several-file attach. It is
/// the product's own bounded question, so it runs to [`DEFAULT_STEP_TIMEOUT`].
async fn probe_one_file_target(
    path: &Path,
    selector: &str,
    files: usize,
    session: &str,
) -> OneFileTarget {
    let Some(probe) = crate::chrome::files::one_file_probe(selector, files) else {
        return OneFileTarget::Proceed;
    };
    let Ok(resp) = spawn_step(
        path,
        &["eval", &probe],
        Some(session),
        StepClocks::probe(DEFAULT_STEP_TIMEOUT),
        None,
    )
    .await
    else {
        return OneFileTarget::Unjudged;
    };
    crate::chrome::files::judge_one_file_target(
        selector,
        crate::chrome::files::single_file_input(&resp),
    )
}

/// A screenshot's reserved file: removed on every path that is not a reported capture —
/// spawn failure, corrupt write, interrupted call. [`ReservedPicture::commit`] hands it over.
struct ReservedPicture {
    path: std::path::PathBuf,
    keep: bool,
}

impl ReservedPicture {
    fn commit(mut self) {
        self.keep = true;
    }
}

impl Drop for ReservedPicture {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// `screenshot` — capture the visible page into a produced PNG. The destination is the
/// caller's own directory ([`crate::cli::resolve_output_dir`]), resolved here, and the
/// picture is written under a name nothing occupies yet. The result reports the absolute
/// path written, in the platform's own spelling.
async fn screenshot(output_dir: Option<&str>, session: &str) -> OutEnvelope {
    let action = "screenshot";
    let dir = match crate::cli::resolve_output_dir(output_dir) {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(error) => {
            return out_env(action, false, OutKind::Usage, json!({ "error": error }));
        }
    };
    let path = match require_cli(action, serde_json::Map::new()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // The claim IS the collision check: chrome-use writes over the file it is handed.
    let (file, file_path) = match crate::util::create_unique_file(&dir, "screenshot.png").await {
        Ok(claimed) => claimed,
        Err(e) => {
            return out_env(
                action,
                false,
                OutKind::Error,
                json!({ "error": format!("cannot write into {}: {e}", dir.display()) }),
            );
        }
    };
    drop(file);
    // The reservation is removed on every path that does not become a reported capture —
    // including a call interrupted mid-step, whose future is dropped here.
    let picture = ReservedPicture {
        path: file_path.clone(),
        keep: false,
    };
    let written = file_path.to_string_lossy().into_owned();
    let clocks = StepClocks::tool_clock(None);
    match spawn_step(
        &path,
        &["screenshot", written.as_str()],
        Some(session),
        clocks,
        None,
    )
    .await
    {
        Ok(resp) => {
            let (width, height) = match crate::chrome::files::validate_png(&file_path) {
                Ok(size) => size,
                Err(e) => {
                    return out_env(
                        action,
                        false,
                        OutKind::Error,
                        json!({ "error": e.to_string() }),
                    );
                }
            };
            let bytes = std::fs::metadata(&file_path).map_or(0, |m| m.len());
            let mut payload = serde_json::Map::new();
            payload.insert("path".into(), json!(written));
            payload.insert("width".into(), json!(width));
            payload.insert("height".into(), json!(height));
            payload.insert("bytes".into(), json!(bytes));
            // The page the pixels came from: an unaddressed throwaway tab captures
            // `about:blank`, and the caller must see that rather than trust success.
            if let Some(origin) = resp.data.as_ref().and_then(|d| d.get("origin")) {
                payload.insert("origin".into(), origin.clone());
            }
            // chrome-use warns when the pixels did not come from the page the caller thinks it
            // captured; the file is written, so the capture stands and the warning is repeated.
            if let Some(warning) = chrome_use_warning(&resp) {
                payload.insert("warning".into(), warning);
            }
            picture.commit();
            out_env(action, true, OutKind::Ok, Value::Object(payload))
        }
        Err(f) => f.envelope(action, json!({}), clocks),
    }
}

/// `session stop` — name-gated stop of a CLI session (protects `agent-tab-*` unless
/// `--force`). The envelope echoes the name the owner wrote.
async fn session_stop(name: &str, force: bool) -> OutEnvelope {
    let target = match resolve_stop_target(name, force) {
        Ok(t) => t,
        Err(error) => {
            eprintln!("mahbot chrome: {error}");
            return out_env(
                "session",
                false,
                OutKind::Usage,
                json!({ "session": name, "error": error }),
            );
        }
    };
    let shown = cli_sessions::display_name(&target);
    let path = match require_cli("session", request([("session", json!(shown))])) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // A product probe ([`StepClocks::probe`]), not agent work: bounded by the shared
    // [`SESSION_STOP_TIMEOUT`] with the relay self-heal suppressed.
    let clocks = StepClocks::probe(SESSION_STOP_TIMEOUT);
    match spawn_step(&path, &["session", "stop", &target], None, clocks, None).await {
        Ok(_) => {
            // The mark holds only until an explicit stop.
            cli_sessions::forget(&target);
            out_env("session", true, OutKind::Ok, json!({ "session": shown }))
        }
        Err(f) => f.envelope("session", json!({ "session": shown }), clocks),
    }
}

/// Whether chrome-use is running this session right now: the daemon-free `session list`
/// read. `false` when the list cannot be read — an unknown answer must not refuse.
async fn session_running(path: &std::path::Path, name: &str) -> bool {
    match spawn_step(
        path,
        &["session", "list"],
        None,
        StepClocks::probe(SESSION_LIST_TIMEOUT),
        None,
    )
    .await
    {
        Ok(resp) => session_listed(&resp, name),
        Err(_) => false,
    }
}

/// `session list` — the sessions this CLI created: name, keep mark, time since last use.
/// Reads the local record only, never the browser.
fn session_list() -> OutEnvelope {
    let rows = cli_sessions::list();
    let sessions: Vec<Value> = rows
        .iter()
        .map(|row| {
            json!({
                "session": row.name,
                "kept": row.kept,
                "idle_seconds": row.idle_seconds,
            })
        })
        .collect();
    out_env(
        "session",
        true,
        OutKind::Ok,
        json!({ "sessions": sessions }),
    )
}

/// `session status <name>` — opt-in liveness probe: a daemon-free preflight reports a stopped
/// session as `empty` (rc 0), a listed one is probed with a real bounded `get url`. A timeout
/// there is wedge evidence, so it re-classifies as Environment (rc 2) and names the recovery.
/// This verb stops nothing; the action verbs and the watchdog recover such a session
/// themselves ([`recover_unresponsive_session`]).
async fn session_status(name: &str) -> OutEnvelope {
    let target = resolve_session_target(name);
    // What the envelopes echo: the name the owner wrote, as `session list` shows it.
    let shown = cli_sessions::display_name(&target);
    let path = match require_cli("session", request([("session", json!(shown))])) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // A diagnostic is still a later call with the same name: it restarts the lifetime and
    // spawns the daemon with the backstop its mark asks for. Foreign sessions are left alone.
    if let Some(kept) = cli_sessions::touch(&target) {
        set_group_cleanup_backstop(cli_sessions::backstop_ms(kept));
    }
    // Both steps are the product's own probes ([`StepClocks::probe`]).
    let enumerate = StepClocks::probe(SESSION_LIST_TIMEOUT);
    let list = match spawn_step(&path, &["session", "list"], None, enumerate, None).await {
        Ok(resp) => resp,
        Err(f) => {
            return f.envelope(
                "session",
                json!({ "session": shown, "stage": "enumerate" }),
                enumerate,
            );
        }
    };
    if !session_listed(&list, &target) {
        return out_env(
            "session",
            true,
            OutKind::Empty,
            json!({ "session": shown, "detail": "session is not running — nothing to probe" }),
        );
    }
    let probe = StepClocks::probe(SESSION_PROBE_TIMEOUT);
    match spawn_step(&path, &["get", "url"], Some(&target), probe, None).await {
        Ok(resp) => {
            let url = resp
                .data
                .as_ref()
                .and_then(|d| d.get("url"))
                .and_then(Value::as_str)
                .unwrap_or("?");
            out_env(
                "session",
                true,
                OutKind::Ok,
                json!({ "session": shown, "status": "usable", "url": url }),
            )
        }
        Err(mut f) => {
            // Any timeout on the probe is wedge evidence (it reads the current URL, never
            // navigates): re-classify as Environment with chrome-use's own phrasing. It does
            // NOT auto-recover — it is the diagnostic — so its text names the manual flow.
            // Other causes pass through.
            if f.kind == OutKind::Timeout {
                f = StepFailure {
                    kind: OutKind::Environment,
                    message: format!(
                        "session unresponsive: the session stopped answering — no answer to the \
                         liveness probe (get url) within {}s. This diagnostic stops nothing; \
                         recover it with {SESSION_RECOVERY_FLOW}, or let the next action verb \
                         recover it automatically.",
                        SESSION_PROBE_TIMEOUT.as_secs()
                    ),
                };
            }
            // A protected-namespace wedge needs `--force` to stop: make the recovery flow
            // actionable for that edge. Only wedges, and only protected ones, get the note.
            append_protected_stop_note(&mut f.message, &target);
            f.envelope("session", json!({ "session": shown }), probe)
        }
    }
}

/// Append the protected-namespace `--force` note to a message that reads as a wedge on a
/// protected (`agent-tab-*` / `link-enricher-*`) target: the recovery flow's
/// `session stop <name>` is only actionable with `--force` there. Pure, so tests pin both.
fn append_protected_stop_note(message: &mut String, target: &str) {
    if PROTECTED_SESSION_PREFIXES
        .iter()
        .any(|p| target.starts_with(p))
        && is_session_unresponsive_error(message)
    {
        write!(
            message,
            " (protected namespace: stopping '{target}' requires `session stop {target} --force`)"
        )
        .expect("writing to a String cannot fail");
    }
}

/// Tolerance-first membership check over a `session list` envelope: only an explicit failure
/// verdict rules the target out. Handles both shapes — latest
/// `{"ok":true,"sessions":[{"name":..}]}` and legacy `{"data":{"sessions":["name"]}}`.
fn session_listed(resp: &ChromeResponse, target: &str) -> bool {
    if resp.verdict() == Some(false) {
        return false;
    }
    let names = resp
        .extra
        .get("sessions")
        .or_else(|| resp.data.as_ref().and_then(|d| d.get("sessions")));
    let Some(Value::Array(arr)) = names else {
        return false;
    };
    arr.iter().any(|entry| match entry {
        Value::String(s) => s == target,
        Value::Object(_) => entry.get("name").and_then(Value::as_str) == Some(target),
        _ => false,
    })
}

/// Best-effort close of an ephemeral session AFTER the envelope — never changes the exit
/// code; diagnostics go to stderr. A successful close forgets the record.
async fn close_ephemeral(name: &str) {
    // Every line here is read by a person, so the session is named as its owner would.
    let display = cli_sessions::display_name(name);
    let Some(path) = cli_path() else {
        eprintln!(
            "mahbot chrome: could not close ephemeral session '{display}' (chrome-use CLI not found)"
        );
        return;
    };
    // A lifecycle close, not agent work: [`run_cli_bounded_at`] bounds it with a product clock
    // and suppresses the relay self-heal, like the CLI's other probes.
    match run_cli_bounded_at(
        &path,
        &["session", "stop", name],
        None,
        SESSION_STOP_TIMEOUT,
    )
    .await
    {
        CliRun::Output(out) => {
            // A helper the close leaves holding the call's channel is recorded for [`run_cli`] to
            // report: the action envelope went out before this ran.
            CHROME_USE_LEFTOVER_PIPES.fetch_or(out.leftover_pipes, Ordering::Relaxed);
            if out.status.success() {
                cli_sessions::forget(name);
            } else {
                eprintln!("mahbot chrome: failed to close ephemeral session '{display}'");
            }
        }
        CliRun::SpawnFailure => {
            eprintln!(
                "mahbot chrome: could not spawn chrome-use to close ephemeral session '{display}'"
            );
        }
        CliRun::TimedOut => {
            eprintln!("mahbot chrome: timed out closing ephemeral session '{display}'");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The clock arithmetic has one home; tests here derive their expectations
    // from it instead of repeating its literals.
    use crate::chrome::kill_bound;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().copied().map(String::from).collect()
    }

    /// [`parse_invocation`] for a test that only cares about the action.
    fn parse_action(args: &[String]) -> Result<Invocation, String> {
        parse_invocation(args).map(|(invocation, _)| invocation)
    }

    /// The rows `help` prints under `Flags:` — `(spelling, description)`, in order. An
    /// action whose help has no flags block yields no rows.
    fn printed_flag_rows(help: &str) -> Vec<(&str, &str)> {
        let Some((_, tail)) = help.split_once("\nFlags:\n") else {
            return Vec::new();
        };
        tail.lines()
            .take_while(|l| !l.is_empty())
            .filter_map(|l| l.strip_prefix("  ")?.split_once("  "))
            .collect()
    }

    /// The machine names of the flag rows a rendered help prints, in row order.
    fn printed_flag_names(help: &str) -> Vec<&str> {
        printed_flag_rows(help)
            .into_iter()
            .filter_map(|(spelling, _)| flag_name_in(spelling))
            .collect()
    }

    /// The flag a printed spelling or usage/example token names: `--expect <sel>` →
    /// `expect`, `[--force]` → `force`, `--limit=5` → `limit`.
    fn flag_name_in(token: &str) -> Option<&str> {
        let t = token.trim_matches(['[', '(', ')', ']', ',', '|']);
        let name = t
            .trim_start_matches('-')
            .split([' ', '='])
            .next()
            .unwrap_or_default();
        (t.starts_with('-') && !name.is_empty()).then_some(name)
    }

    /// The `Usage:` line of a rendered action help, without its `mahbot chrome <action>`.
    fn printed_usage(help: &str) -> String {
        let line = help
            .split_once("\nUsage:\n")
            .and_then(|(_, tail)| tail.lines().find(|l| !l.trim().is_empty()))
            .unwrap_or_default()
            .trim();
        let rest = line.strip_prefix("mahbot chrome ").unwrap_or(line);
        rest.split_once(' ')
            .map_or(String::new(), |(_, after)| after.to_string())
    }

    /// The example command lines a rendered action help prints (one per line).
    fn printed_examples(help: &str) -> Vec<String> {
        let Some((_, tail)) = help.split_once("\nExamples:\n") else {
            return Vec::new();
        };
        tail.lines()
            .filter_map(|l| l.strip_prefix("  ").map(str::to_string))
            .collect()
    }

    /// Rough shell-word split of a documented example: the examples are short quoted
    /// lines (`"#main"`, `'document.title'`), not scripts.
    fn shell_words(line: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut quote: Option<char> = None;
        let mut in_word = false;
        for c in line.chars() {
            match quote {
                Some(q) if c == q => quote = None,
                Some(_) => cur.push(c),
                None if c == '\'' || c == '"' => {
                    quote = Some(c);
                    in_word = true;
                }
                None if c.is_whitespace() => {
                    if in_word {
                        out.push(std::mem::take(&mut cur));
                        in_word = false;
                    }
                }
                None => {
                    cur.push(c);
                    in_word = true;
                }
            }
        }
        if in_word {
            out.push(cur);
        }
        out
    }

    /// The action words the top-level help advertises, in printed order: an action's line is
    /// indented two spaces, its purpose line four more, so only the shallower lines count.
    fn advertised_actions(help: &str) -> Vec<String> {
        let Some((_, tail)) = help.split_once("\nActions:\n") else {
            return Vec::new();
        };
        let body = tail
            .split_once("\nGlobal flags:")
            .map_or(tail, |(body, _)| body);
        body.lines()
            .filter_map(|l| l.strip_prefix("  "))
            .filter(|rest| !rest.starts_with(' '))
            .filter_map(|rest| rest.split_whitespace().next().map(str::to_string))
            .collect()
    }

    /// The session-wide pair — handled outside the common flag path (`extract_global_flags`),
    /// so it has its own guard rather than a shape probe.
    const SESSION_PAIR: [&str; 2] = ["session", "keep"];

    #[test]
    fn eval_count_unwraps_chrome_use_result_envelope() {
        let resp = |data: Value| ChromeResponse {
            success: Some(true),
            ok: None,
            data: Some(data),
            error: None,
            code: None,
            retryable: None,
            extra: serde_json::Map::default(),
        };
        // chrome-use 1.5.101 wraps eval output as {origin, result}.
        assert_eq!(
            eval_count(&resp(json!({"origin": "https://x", "result": 7}))),
            Some(7)
        );
        // Tolerance: bare value, text form, string-wrapped number.
        assert_eq!(eval_count(&resp(json!(3))), Some(3));
        assert_eq!(
            eval_count(&resp(json!({"origin": "https://x", "result": "12"}))),
            Some(12)
        );
        assert_eq!(eval_count(&resp(json!("0"))), Some(0));
        // A non-numeric result is not a count.
        assert_eq!(
            eval_count(&resp(json!({"origin": "https://x", "result": "abc"}))),
            None
        );
        assert_eq!(eval_count(&resp(Value::Null)), None);
        // eval_result unwraps the wrapped JS value for the eval action too,
        // including the chrome-error probe's boolean.
        let wrapped = resp(json!({"origin": "https://x", "result": {"a": 1}}));
        assert_eq!(eval_result(&wrapped), Some(&json!({"a": 1})));
        let boolean = resp(json!({"origin": "https://x", "result": true}));
        assert_eq!(eval_result(&boolean).and_then(Value::as_bool), Some(true));
        // An object that merely carries a `result` key is NOT the wrapper.
        let bare = json!({"result": "page payload", "other": 1});
        assert_eq!(eval_result(&resp(bare.clone())), Some(&bare));
    }

    /// A bare session name is prefixed exactly once, and no name at all yields
    /// an ephemeral CLI-namespaced session.
    #[test]
    fn resolve_session_prefixes_a_bare_name_once() {
        {
            let case = "resolve session named prefixing is idempotent";
            let (name, ephemeral) = resolve_session(Some("foo"));
            assert_eq!(
                name, "mahbot-chrome-foo",
                "case {case}: a bare name gains the namespace"
            );
            assert!(!ephemeral, "case {case}: a named session is not ephemeral");

            let (name, ephemeral) = resolve_session(Some("mahbot-chrome-foo"));
            assert_eq!(
                name, "mahbot-chrome-foo",
                "case {case}: an already-prefixed name is left alone"
            );
            assert!(!ephemeral, "case {case}: a named session is not ephemeral");
        }

        {
            let case = "resolve session ephemeral uses cli prefix";
            let (name, ephemeral) = resolve_session(None);
            assert!(
                name.starts_with(CLI_EPHEMERAL_PREFIX),
                "case {case}: no name yields the CLI's ephemeral prefix: {name}"
            );
            assert!(ephemeral, "case {case}: it is ephemeral");
        }
    }

    /// The alphabet is chrome-use's own for `session stop` (`is_valid_session_name`:
    /// alphanumerics, '-' and '_'). A name with a '.' parses and opens fine, but chrome-use then
    /// refuses to stop it, leaving a session nothing can clear — refused here, where the caller
    /// can still choose another.
    #[test]
    fn session_names_stay_within_the_alphabet_chrome_use_can_stop() {
        for name in ["qa-2933", "qa_2933", "Qa2933"] {
            assert!(
                validate_session_name(name).is_ok(),
                "{name} must be accepted"
            );
        }
        for name in ["qa.2933", "qa 2933", "", "../qa", "qa/2933", "qa;stop"] {
            assert!(
                validate_session_name(name).is_err(),
                "{name:?} must be refused — chrome-use will not stop such a session"
            );
        }
        assert!(
            parse_action(&[
                "session".into(),
                "stop".into(),
                "qa.2933".into(),
                "--force".into()
            ])
            .is_err(),
            "the refusal must also be wired into the stop verb, not only --session"
        );
    }

    #[test]
    fn step_clocks_ride_the_kill_above_the_clock_they_declare() {
        // The arithmetic has one home ([`crate::chrome::clocks`] / [`crate::chrome::kill_bound`]);
        // these assertions are about which shape the CLI takes, derived from it.
        let chrome_own = clocks(None, CliRecovery::Allowed);

        // The operation's own step (`open`'s navigation): chrome-use runs it to its own client
        // tolerance, and the operation's budget does NOT bound the kill, so a navigation still in
        // progress can report its own reason.
        let navigation = StepClocks::operation();
        assert_eq!(navigation.call.chrome_side, CHROME_USE_DECLARED_BUDGET);
        assert_eq!(
            navigation.call.kill,
            kill_bound(CHROME_USE_DECLARED_BUDGET, CliRecovery::Allowed)
        );

        // A verb chrome-use accepts no `--timeout` for: with no user bound the kill is the one the
        // declared clock derives, and a user bound IS the bound the call runs to.
        let no_bound = StepClocks::tool_clock(None);
        assert_eq!(no_bound.call.chrome_side, CHROME_USE_DECLARED_BUDGET);
        assert_eq!(no_bound.call.kill, chrome_own.kill);
        let at_the_clock = StepClocks::tool_clock(Some(CHROME_USE_DECLARED_BUDGET));
        assert_eq!(at_the_clock.call.kill, chrome_own.kill);

        // A LONGER one (`eval --timeout 300`) widens the kill alone: the clock DECLARED to
        // chrome-use stays under its own tolerance, which nothing can raise without making it fail
        // as a wedged session. The product's kill is what the envelope reports as `timeout_ms`.
        let wide = StepClocks::tool_clock(Some(Duration::from_secs(300)));
        assert_eq!(wide.call.chrome_side, CHROME_USE_DECLARED_BUDGET);
        assert_eq!(
            wide.call.kill,
            kill_bound(Duration::from_secs(300), CliRecovery::Allowed)
        );
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
        }
        .envelope("eval", json!({ "js": "1" }), wide);
        assert_eq!(env.payload["timeout_ms"], json!(wide.call.kill.as_millis()));
        let declared_ms = format!("{}ms", CHROME_USE_DECLARED_BUDGET.as_millis());
        assert!(
            env.payload["error"]
                .as_str()
                .is_some_and(|e| e.contains(&declared_ms)),
            "the kill must ride above the clock the call really ran to: {}",
            env.payload["error"]
        );

        // A bounded step of the operation: a probe in everything but name — it suppresses
        // the relay self-heal and its kill rests [`KILL_SLACK`] above the product's bound.
        let probe = StepClocks::bounded(CHROME_USE_DECLARED_BUDGET, Duration::from_secs(2));
        assert_eq!(probe.call.kill, Duration::from_secs(2) + KILL_SLACK);
        assert!(matches!(probe.recovery, CliRecovery::Suppressed));
        let settle = StepClocks::bounded(Duration::from_secs(6), Duration::from_secs(6));
        assert_eq!(settle.call.chrome_side, Duration::from_secs(6));
        assert_eq!(settle.call.kill, Duration::from_secs(6) + KILL_SLACK);
        // A bound above chrome-use's own clock is still the PRODUCT's bound: the step is
        // cut off at it.
        let oversized = StepClocks::bounded(CHROME_USE_DECLARED_BUDGET, Duration::from_secs(600));
        assert_eq!(oversized.call.kill, Duration::from_secs(600) + KILL_SLACK);
        assert!(oversized.call.kill > chrome_own.kill);
    }

    #[test]
    fn settle_budget_caps_and_reserves_capture() {
        // Plenty of room: capped at SETTLE_CAP.
        assert_eq!(settle_budget(Duration::from_secs(22)), Some(SETTLE_CAP));
        // Tight page: the capture reserve is honored first.
        assert_eq!(
            settle_budget(Duration::from_secs(4)),
            Some(Duration::from_millis(1500))
        );
        // Exactly at the skip threshold still runs; a hair under is skipped.
        assert_eq!(
            settle_budget(Duration::from_millis(3500)),
            Some(MIN_SETTLE_BUDGET)
        );
        assert_eq!(settle_budget(Duration::from_millis(3499)), None);
        assert_eq!(settle_budget(Duration::ZERO), None);
    }

    #[test]
    fn session_stop_gating_resolves_and_protects() {
        // CLI-namespace names pass through; bare names are prefixed.
        assert_eq!(
            resolve_stop_target("mahbot-chrome-foo", false).unwrap(),
            "mahbot-chrome-foo"
        );
        assert_eq!(
            resolve_stop_target("foo", false).unwrap(),
            "mahbot-chrome-foo"
        );
        // Protected namespaces are refused without --force, stopped as-is with it.
        assert!(resolve_stop_target("agent-tab-1", false).is_err());
        assert_eq!(
            resolve_stop_target("agent-tab-1", true).unwrap(),
            "agent-tab-1"
        );
        assert!(resolve_stop_target("link-enricher-7", false).is_err());
    }

    #[test]
    fn resolve_session_target_resolves_namespaces() {
        // A bare name is prefixed like `--session`; both namespace names pass through as-is.
        assert_eq!(resolve_session_target("docs"), "mahbot-chrome-docs");
        assert_eq!(
            resolve_session_target("mahbot-chrome-foo"),
            "mahbot-chrome-foo"
        );
        assert_eq!(resolve_session_target("agent-tab-1"), "agent-tab-1");
        assert_eq!(resolve_session_target("link-enricher-7"), "link-enricher-7");
    }

    /// The session subcommands parse into their own actions and reject what is not theirs.
    #[test]
    fn session_parsing_accepts_its_shapes_and_rejects_the_rest() {
        {
            let case = "session status parses and rejects";
            // status parses into SessionStatus.
            let inv = parse_action(&["session".into(), "status".into(), "docs".into()])
                .expect("session status parses");
            match inv.action {
                Action::SessionStatus { name } => {
                    assert_eq!(name, "docs", "case {case}: the name is the positional");
                }
                _ => panic!("case {case}: expected SessionStatus"),
            }

            // The global --session flag is rejected for session subcommands.
            let err = parse_action(&[
                "session".into(),
                "status".into(),
                "docs".into(),
                "--session".into(),
                "x".into(),
            ])
            .err()
            .expect("--session must be rejected for session subcommands");
            assert_eq!(
                err, "--session is not valid for session stop/status/list",
                "case {case}: --session is not a session subcommand's flag"
            );

            // `--force` is `stop`'s flag alone: the other subcommands accept no flags.
            let allowed: Vec<&CliFlag> = declared_flags("session").iter().collect();
            let err = parse_session(
                &["status".into(), "docs".into(), "--force".into()],
                &allowed,
            )
            .err()
            .expect("--force must be rejected for session status");
            assert_eq!(
                err, "unknown flag --force",
                "case {case}: --force belongs to stop alone"
            );

            // Extra positionals are rejected for status.
            assert!(
                parse_session(&["status".into(), "a".into(), "b".into()], &allowed).is_err(),
                "case {case}: status takes one name"
            );

            // An unknown sub names both verbs.
            let err = parse_action(&["session".into(), "destroy".into(), "docs".into()])
                .err()
                .expect("unknown sub must be rejected");
            assert!(
                err.contains(
                    "unknown session subcommand 'destroy' (expected 'stop', 'status' or 'list')"
                ),
                "case {case}: unexpected error: {err}"
            );
        }

        {
            let case = "session listed handles both shapes";
            // Latest shape: the top-level `sessions` array of objects (lands in `extra`).
            let latest: ChromeResponse = serde_json::from_value(json!({
                "ok": true, "sessions": [{ "name": "mahbot-chrome-docs" }]
            }))
            .expect("deserialize latest shape");
            assert!(
                session_listed(&latest, "mahbot-chrome-docs"),
                "case {case}: the latest shape lists the session"
            );
            assert!(
                !session_listed(&latest, "mahbot-chrome-other"),
                "case {case}: a name it does not carry is not listed"
            );

            // Legacy shape: `data.sessions` array of strings.
            let legacy: ChromeResponse = serde_json::from_value(json!({
                "success": true, "data": { "sessions": ["mahbot-chrome-docs"] }
            }))
            .expect("deserialize legacy shape");
            assert!(
                session_listed(&legacy, "mahbot-chrome-docs"),
                "case {case}: the legacy shape lists the session too"
            );

            // An explicit failure verdict rules the target out.
            let failed: ChromeResponse =
                serde_json::from_value(json!({ "success": false })).expect("deserialize failure");
            assert!(
                !session_listed(&failed, "mahbot-chrome-docs"),
                "case {case}: a failure verdict rules the target out"
            );

            // No sessions key at all → not listed.
            let no_sessions: ChromeResponse =
                serde_json::from_value(json!({ "success": true, "data": {} }))
                    .expect("deserialize empty payload");
            assert!(
                !session_listed(&no_sessions, "mahbot-chrome-docs"),
                "case {case}: no sessions key lists nothing"
            );
        }
    }

    #[test]
    fn wedge_keying_uses_who_ended_the_step_not_the_kind_alone() {
        let env = |kind: OutKind, error: &str| {
            out_env(
                "open",
                false,
                kind,
                json!({ "url": "https://x", "error": error }),
            )
        };
        // chrome-use's own session-unresponsive classification (Environment) is the one
        // wedge signal needing no confirmation: chrome-use diagnosed it, not cut off.
        let diagnosed = env(
            OutKind::Environment,
            "session unresponsive: no response within 45s",
        );
        assert_eq!(wedge_action(&diagnosed, true, false), WedgeAction::Recover);
        let cdp = env(
            OutKind::Environment,
            "CDP session is unresponsive after attaching (Connection reset).",
        );
        assert_eq!(wedge_action(&cdp, true, false), WedgeAction::Recover);

        // A Timeout chrome-use itself reported — what a failed `expect`/`wait` produces
        // normally — is NOT a wedge: stopping that session would only lose its tabs.
        let verdict = env(OutKind::Timeout, "Wait timed out after 15000ms");
        assert_eq!(wedge_action(&verdict, true, false), WedgeAction::Leave);

        // A Timeout the PRODUCT's own clock ended is only a candidate: chrome-use was cut
        // off before classifying, so the bounded probe has to confirm it.
        let killed = env(OutKind::Timeout, "deadline reached after 92000ms");
        assert_eq!(wedge_action(&killed, true, true), WedgeAction::Probe);

        // `open --expect --structural` labels a deadline expiration Redesign, so it keys the same.
        let redesign = env(
            OutKind::Redesign,
            "possible DOM redesign or structural change",
        );
        assert_eq!(wedge_action(&redesign, true, true), WedgeAction::Probe);
        assert_eq!(wedge_action(&redesign, true, false), WedgeAction::Leave);

        // Other kinds and messages are no evidence, and an ephemeral session is never touched.
        let relay = env(OutKind::Environment, "relay isn't connected");
        assert_eq!(wedge_action(&relay, true, true), WedgeAction::Leave);
        let missing = env(OutKind::Error, "element not found");
        assert_eq!(wedge_action(&missing, true, false), WedgeAction::Leave);
        let net = env(OutKind::Network, "net::ERR_CONNECTION_REFUSED");
        assert_eq!(wedge_action(&net, true, true), WedgeAction::Leave);
        assert_eq!(wedge_action(&killed, false, true), WedgeAction::Leave);
    }

    /// The end of the keying: an envelope left for [`WedgeAction::Leave`] reaches the caller
    /// untouched — no session stop, no annotation.
    #[tokio::test]
    async fn wedge_recovery_leaves_a_normal_timeout_alone() {
        let failure =
            |error: &str| out_env("expect", false, OutKind::Timeout, json!({ "error": error }));
        let mut env = failure("condition not met");
        recover_session_wedge(&mut env, "mahbot-chrome-docs", true, false).await;
        assert_eq!(env.payload["error"], "condition not met");

        let mut env = failure("condition not met");
        recover_session_wedge(&mut env, "mahbot-chrome-ephemeral-x", false, true).await;
        assert_eq!(env.payload["error"], "condition not met");
    }

    /// The spawn-level classification, case by case: the envelope decides over the exit code (a
    /// failed expect assertion arrives as `success:true` with exit 1), a browser chrome-use
    /// launched itself is never a success, an unparseable answer names itself, and the canned
    /// stale-relay hint is stripped from Timeout-classified text only.
    #[expect(clippy::too_many_lines)]
    #[test]
    fn classify_step_output_decides_from_the_envelope_and_sanitises() {
        enum Want {
            Ok,
            Verdict {
                pass: bool,
                timed_out: bool,
                actual: Option<Value>,
            },
            Fail(OutKind, String),
            FailKind(OutKind),
        }

        struct Case {
            name: &'static str,
            expect_style: bool,
            truncated: bool,
            exit_code: Option<i32>,
            status_success: bool,
            stdout: Vec<u8>,
            stderr: String,
            want: Want,
        }

        let envelope = |body: Value| serde_json::to_vec(&body).expect("serialize envelope");
        let hint = "Hint: the session's browser connection is unresponsive (likely a stale \
                    relay/service-worker mid-session). Reconnect with `connect`, or close the \
                    session and reopen it.";
        let replacement = "This session's previous browser is gone (its browser connection was dead) \
                    and a fresh one was launched for this command";

        let cases = [
            Case {
                // expect pass=false: success:true + exit 1 → the verdict is returned.
                name: "expect_verdict_survives_a_non_zero_exit",
                expect_style: true,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(
                    json!({"success": true, "data": {"pass": false, "actual": 3, "timedOut": true}}),
                ),
                stderr: String::new(),
                want: Want::Verdict {
                    pass: false,
                    timed_out: true,
                    actual: Some(json!(3)),
                },
            },
            Case {
                // …but only for expect: the same payload elsewhere is a generic failure.
                name: "same_payload_outside_expect_is_a_generic_failure",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(json!({"success": true, "data": {"pass": false}})),
                stderr: String::new(),
                want: Want::Fail(OutKind::Error, "chrome-use exited with code 1".into()),
            },
            Case {
                // expect un-evaluable (no browser): success:false + exit 2 → environment.
                name: "expect_un_evaluable_is_an_environment_failure",
                expect_style: true,
                truncated: false,
                exit_code: Some(2),
                status_success: false,
                stdout: envelope(
                    json!({"success": false, "error": "Browser not launched", "code": "browser_not_launched"}),
                ),
                stderr: String::new(),
                want: Want::Fail(OutKind::Environment, "Browser not launched".into()),
            },
            Case {
                // Honest timeout classification on a failed wait.
                name: "failed_wait_is_a_timeout",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(
                    json!({"success": false, "error": "Wait timed out after 15000ms"}),
                ),
                stderr: String::new(),
                want: Want::Fail(OutKind::Timeout, "Wait timed out after 15000ms".into()),
            },
            Case {
                // Happy path: exit 0 + success envelope.
                name: "success_envelope_on_a_zero_exit",
                expect_style: false,
                truncated: false,
                exit_code: Some(0),
                status_success: true,
                stdout: envelope(json!({"success": true, "data": {}})),
                stderr: String::new(),
                want: Want::Ok,
            },
            Case {
                // Unparseable stdout: zero exit → non-JSON error.
                name: "unparseable_stdout_on_a_zero_exit",
                expect_style: false,
                truncated: false,
                exit_code: Some(0),
                status_success: true,
                stdout: b"garbage".to_vec(),
                stderr: String::new(),
                want: Want::Fail(OutKind::Error, "chrome-use returned non-JSON output".into()),
            },
            Case {
                // Unparseable stdout: non-zero exit → stderr fallback, classified.
                name: "unparseable_stdout_falls_back_to_classified_stderr",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: b"garbage".to_vec(),
                stderr: "relay is not connected".into(),
                want: Want::Fail(OutKind::Environment, "relay is not connected".into()),
            },
            Case {
                // Timeout envelope: hint stripped, honest prefix kept.
                name: "timeout_envelope_loses_the_canned_hint",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(
                    json!({"success": false, "error": format!("Wait timed out after 15000ms. {hint}")}),
                ),
                stderr: String::new(),
                want: Want::Fail(OutKind::Timeout, "Wait timed out after 15000ms".into()),
            },
            Case {
                // A non-timeout envelope carrying the hint keeps it verbatim (only
                // Timeout-classified messages are rewritten).
                name: "non_timeout_envelope_keeps_the_hint",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(
                    json!({"success": false, "error": format!("some other failure. {hint}")}),
                ),
                stderr: String::new(),
                want: Want::Fail(OutKind::Error, format!("some other failure. {hint}")),
            },
            Case {
                // The stderr fallback is classified and sanitized too: a timeout
                // phrasing with the canned hint in stderr → hint stripped.
                name: "stderr_fallback_loses_the_canned_hint",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: b"garbage".to_vec(),
                stderr: format!("Wait timed out after 15000ms. {hint}"),
                want: Want::Fail(OutKind::Timeout, "Wait timed out after 15000ms".into()),
            },
            Case {
                // A successful envelope carrying the browser-replacement note is a failure, never
                // a quiet success — and NOT the daemon-down path (the daemon itself is fine).
                name: "self_launched_browser_note_is_an_environment_failure",
                expect_style: false,
                truncated: false,
                exit_code: Some(0),
                status_success: true,
                stdout: envelope(json!({"success": true, "warning": replacement})),
                stderr: String::new(),
                want: Want::Fail(
                    OutKind::Environment,
                    self_launched_browser_cli_error(replacement),
                ),
            },
            Case {
                // The expect-style success path (envelope over a non-zero exit) applies the same
                // rule.
                name: "self_launched_browser_note_fails_the_expect_style_path",
                expect_style: true,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: envelope(
                    json!({"success": true, "warning": replacement, "data": {"pass": true}}),
                ),
                stderr: String::new(),
                want: Want::FailKind(OutKind::Environment),
            },
            Case {
                // A degraded-success warning is NOT browser replacement: the step still succeeds.
                name: "degraded_success_warning_stays_a_success",
                expect_style: false,
                truncated: false,
                exit_code: Some(0),
                status_success: true,
                stdout: envelope(
                    json!({"success": true, "warning": "no key listeners on the focused element"}),
                ),
                stderr: String::new(),
                want: Want::Ok,
            },
            Case {
                // An answer the product cut off names the truncation.
                name: "truncated_answer_names_the_truncation",
                expect_style: false,
                truncated: true,
                exit_code: Some(0),
                status_success: true,
                stdout: b"garbage".to_vec(),
                stderr: String::new(),
                want: Want::Fail(OutKind::Error, truncated_output_error()),
            },
            Case {
                // A genuinely malformed one keeps the generic text.
                name: "malformed_answer_keeps_the_generic_text",
                expect_style: false,
                truncated: false,
                exit_code: Some(0),
                status_success: true,
                stdout: b"garbage".to_vec(),
                stderr: String::new(),
                want: Want::Fail(OutKind::Error, "chrome-use returned non-JSON output".into()),
            },
            Case {
                // A non-zero exit with an unparsed, truncated answer names the truncation too.
                name: "truncated_answer_beats_the_stderr_fallback",
                expect_style: false,
                truncated: true,
                exit_code: Some(1),
                status_success: false,
                stdout: b"garbage".to_vec(),
                stderr: "some stderr".into(),
                want: Want::Fail(OutKind::Error, truncated_output_error()),
            },
            Case {
                // …while a genuinely malformed one still falls back to stderr.
                name: "malformed_answer_falls_back_to_stderr",
                expect_style: false,
                truncated: false,
                exit_code: Some(1),
                status_success: false,
                stdout: b"garbage".to_vec(),
                stderr: "some stderr".into(),
                want: Want::Fail(OutKind::Error, "some stderr".into()),
            },
        ];

        for case in cases {
            let r = classify_step_output(
                case.expect_style,
                case.truncated,
                case.exit_code,
                case.status_success,
                &case.stdout,
                &case.stderr,
            );
            match case.want {
                Want::Ok => assert!(r.is_ok(), "{}: must be a success, got {r:?}", case.name),
                Want::Verdict {
                    pass,
                    timed_out,
                    actual,
                } => {
                    let Ok(resp) = r else {
                        panic!(
                            "{}: the expect verdict must survive the non-zero exit",
                            case.name
                        );
                    };
                    assert_eq!(
                        resp.data.as_ref().and_then(expect_outcome).map(|o| (
                            o.pass,
                            o.timed_out,
                            o.actual
                        )),
                        Some((pass, timed_out, actual)),
                        "{}: verdict",
                        case.name
                    );
                }
                Want::Fail(kind, message) => assert_eq!(
                    r.map(|resp| resp.data.clone())
                        .map_err(|f| (f.kind, f.message.clone())),
                    Err((kind, message)),
                    "{}",
                    case.name
                ),
                Want::FailKind(kind) => match r {
                    Ok(_) => panic!("{}: must not classify as success", case.name),
                    Err(f) => assert_eq!(f.kind, kind, "{}", case.name),
                },
            }
        }
    }

    #[test]
    fn step_failure_envelope_reports_timeout_vs_error() {
        // A declared (forwarded) deadline kill: the envelope reports the product's own bound as
        // `timeout_ms`, and the error names that bound and the clock chrome-use worked to.
        let clocks = StepClocks::forwarded(Duration::from_secs(10));
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
        }
        .envelope("wait", json!({ "selector": ".x" }), clocks);
        assert_eq!(
            env.payload["timeout_ms"],
            json!(clocks.call.kill.as_millis())
        );
        let kill = kill_bound(Duration::from_secs(10), CliRecovery::Allowed).as_millis();
        assert_eq!(
            env.payload["error"],
            format!(
                "deadline reached after {kill}ms — the step did not complete: that is the \
                 product's own bound, which rode above the 10000ms deadline chrome-use itself \
                 was working to, so chrome-use never reported its own reason"
            )
        );

        // Any other failure reports the chrome-use error text, no timeout_ms.
        let env = StepFailure {
            kind: OutKind::Network,
            message: "net::ERR_NAME_NOT_RESOLVED".into(),
        }
        .envelope(
            "open",
            json!({ "url": "https://x" }),
            StepClocks::operation(),
        );
        assert_eq!(env.payload["error"], "net::ERR_NAME_NOT_RESOLVED");
        assert_eq!(env.payload["error_code"], "ERR_NAME_NOT_RESOLVED");
        assert!(env.payload.get("timeout_ms").is_none());

        // A Network failure with no recognizable net error token gets no error_code.
        let env = StepFailure {
            kind: OutKind::Network,
            message: "connection refused by peer".into(),
        }
        .envelope(
            "open",
            json!({ "url": "https://x" }),
            StepClocks::operation(),
        );
        assert!(env.payload.get("error_code").is_none());
    }

    #[test]
    fn expect_envelope_maps_pass_timeout_and_false() {
        // pass → rc 0 / ok.
        let env = expect_envelope(
            "'#main' is visible",
            ExpectOutcome {
                pass: true,
                actual: Some(json!({"tag": "h1"})),
                timed_out: false,
            },
        );
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
        assert_eq!(env.payload["pass"], json!(true));
        assert_eq!(env.payload["actual"], json!({"tag": "h1"}));

        // deadline expiration → timeout.
        let env = expect_envelope(
            "count('.card') == 3",
            ExpectOutcome {
                pass: false,
                actual: Some(json!(0)),
                timed_out: true,
            },
        );
        assert!(!env.ok);
        assert_eq!(env.kind, OutKind::Timeout);
        assert_eq!(env.payload["pass"], json!(false));
        assert_eq!(env.payload["timed_out"], json!(true));
        assert_eq!(
            env.payload["error"],
            json!(
                "condition was not met within the deadline — consider verifying the condition or allowing more time"
            )
        );

        // plain false → error (no timed_out key).
        let env = expect_envelope(
            "url contains \"dashboard\"",
            ExpectOutcome {
                pass: false,
                actual: Some(json!("about:blank")),
                timed_out: false,
            },
        );
        assert!(!env.ok);
        assert_eq!(env.kind, OutKind::Error);
        assert_eq!(env.payload["pass"], json!(false));
        assert!(env.payload.get("timed_out").is_none());

        // `actual: None` omits the actual key.
        let env = expect_envelope(
            "'#main' is present",
            ExpectOutcome {
                pass: true,
                actual: None,
                timed_out: false,
            },
        );
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
        assert_eq!(env.payload["pass"], json!(true));
        assert!(env.payload.get("actual").is_none());
    }

    /// The `--timeout` of a verb that DECLARES its value to chrome-use (`wait`/`expect`, and
    /// `open` through its `--expect` wait) is capped below [`CHROME_USE_OWN_BUDGET`]: at or above
    /// the tolerance chrome-use runs OUT of tolerance instead of answering, so its
    /// session-unresponsive classification — which stops the session and loses its tabs —
    /// replaces the honest "condition was not met". Refused at parse time, not silently
    /// mis-reported. A verb forwarding nothing is refused the OTHER way: nothing can honour a
    /// bound below the declared clock, while a bound at or above it is one the call really runs to.
    #[test]
    fn timeout_flag_is_capped_below_chrome_use_client_tolerance() {
        let err = parse_action(&["wait".into(), "#x".into(), "--timeout".into(), "45".into()])
            .err()
            .expect("a deadline at the tolerance must be refused");
        assert!(
            err.contains("45") && err.contains("client tolerance"),
            "err must be the tolerance refusal, naming the value: {err}"
        );

        assert!(
            parse_action(&["wait".into(), "#x".into(), "--timeout".into(), "44".into(),]).is_ok(),
            "one second below the tolerance must parse"
        );

        for args in [
            &["expect".into(), "#x".into(), "visible".into()][..],
            &["open".into(), "https://example.com".into()][..],
        ] {
            let mut argv = args.to_vec();
            argv.extend(["--timeout".into(), "45".into()]);
            let err = parse_action(&argv)
                .err()
                .unwrap_or_else(|| panic!("{argv:?} must refuse a deadline at the tolerance"));
            // The tolerance refusal itself: it names the value asked for and the tolerance crossed.
            assert!(
                err.contains("45") && err.contains("client tolerance"),
                "{argv:?} must be refused on the value: {err}"
            );
        }

        // A bound below the clock mahbot declares cannot be honoured: it is refused, with the value
        // asked for and the shortest one that IS honoured.
        let declared = CHROME_USE_DECLARED_BUDGET.as_secs().to_string();
        for action in [
            &["count".into(), ".x".into()][..],
            &["eval".into(), "1+1".into()][..],
            &["extract".into(), "--schema-file".into(), "s.json".into()][..],
            &["click".into(), "#a".into()][..],
            &["fill".into(), "#a".into(), "t".into()][..],
            &["type".into(), "#a".into(), "t".into()][..],
            &["press".into(), "Enter".into()][..],
            &["upload".into(), "#f".into(), "a.txt".into()][..],
        ] {
            let mut argv = action.to_vec();
            argv.extend(["--timeout".into(), "5".into()]);
            let err = parse_action(&argv)
                .err()
                .unwrap_or_else(|| panic!("{argv:?} must refuse an unhonourable bound"));
            assert!(
                err.contains("--timeout 5s") && err.contains(&declared),
                "{argv:?} must name the refused value and the shortest honoured one: {err}"
            );
        }
        // At the declared clock it parses, and it is the bound the call runs to.
        let at_the_clock = parse_action(&[
            "eval".into(),
            "1+1".into(),
            "--timeout".into(),
            declared.clone(),
        ])
        .expect("a bound at the declared clock is honoured");
        assert!(matches!(
            at_the_clock.action,
            Action::Eval { bound: Some(b), .. } if b == CHROME_USE_DECLARED_BUDGET
        ));

        let wide = parse_action(&[
            "eval".into(),
            "1+1".into(),
            "--timeout".into(),
            "300".into(),
        ])
        .expect("a verb that forwards no deadline takes any bound");
        assert!(matches!(
            wide.action,
            Action::Eval { bound: Some(b), .. } if b == Duration::from_secs(300)
        ));
    }

    #[test]
    fn parse_rejects_usage_errors() {
        assert!(parse_action(&["bogus".into()]).is_err());
        assert!(parse_action(&["open".into()]).is_err()); // missing url
        assert!(parse_action(&["open".into(), "https://x".into(), "--bogus".into()]).is_err());
        assert!(parse_action(&["count".into()]).is_err()); // missing selector
        assert!(parse_action(&["extract".into()]).is_err()); // missing --schema-file
        assert!(parse_action(&["status".into(), "--session".into(), "x".into()]).is_err());
        assert!(parse_action(&["session".into(), "stop".into()]).is_err()); // missing name
        assert!(
            parse_action(&[
                "open".into(),
                "https://x".into(),
                "--timeout".into(),
                "abc".into()
            ])
            .is_err()
        );
        assert!(parse_action(&["open".into(), "https://x".into(), "--timeout=0".into()]).is_err());
        assert!(parse_action(&["session".into(), "destroy".into(), "x".into()]).is_err());
        // wait must reject the numeric silent-sleep form and mixed targets.
        assert!(parse_action(&["wait".into(), "5000".into()]).is_err());
        assert!(
            parse_action(&[
                "wait".into(),
                "--url".into(),
                "a".into(),
                "--text".into(),
                "b".into()
            ])
            .is_err()
        );
        assert!(parse_action(&["wait".into(), "--url".into(), "a".into(), ".x".into()]).is_err());
        // wait rejects extra positionals (previously silently ignored).
        assert!(parse_action(&["wait".into(), "#main".into(), "extra".into()]).is_err());
        // expect rejects missing / off-allowlist conditions.
        assert!(parse_action(&["expect".into()]).is_err());
        assert!(parse_action(&["expect".into(), "#a".into(), "gone".into()]).is_err());
        assert!(
            parse_action(&[
                "expect".into(),
                "count".into(),
                ".c".into(),
                "~=".into(),
                "2".into()
            ])
            .is_err()
        );
        assert!(
            parse_action(&[
                "expect".into(),
                "count".into(),
                ".c".into(),
                "==".into(),
                "x".into()
            ])
            .is_err()
        );
        assert!(
            parse_action(&[
                "expect".into(),
                "text".into(),
                "h1".into(),
                "starts".into(),
                "x".into()
            ])
            .is_err()
        );
        assert!(
            parse_action(&["expect".into(), "count".into(), ".c".into(), "==".into()]).is_err()
        );
    }

    /// `--` stays a true end-of-options terminator: every token after it is
    /// verbatim text, even one that names a flag.
    #[test]
    fn parse_dash_separator_takes_text_verbatim() {
        // Post-`--` tokens are verbatim text, never scanned for `--session` by either layer.
        let after_dash = argv(&["fill", "#q", "--", "--session", "sneaky", "rest"]);
        let (session, _, remaining) =
            extract_global_flags(&after_dash).expect("the terminator parses");
        assert_eq!(session, None, "--session after -- is text, not a binding");
        assert_eq!(remaining, after_dash, "every post--- token stays verbatim");
        let inv =
            parse_action(&argv(&["fill", "#q", "--", "--foo"])).expect("fill -- separator parses");
        match inv.action {
            Action::Fill { source, .. } => assert!(
                matches!(source, TextInput::Inline(t) if t == "--foo"),
                "fill takes the post--- token as inline text"
            ),
            _ => panic!("expected Fill"),
        }
        let inv = parse_action(&argv(&["type", "#q", "--", "--session", "sneaky", "rest"]))
            .expect("type with --session-looking text parses");
        assert_eq!(inv.session, None, "--session is never bound after --");
        match inv.action {
            Action::Type { text, .. } => assert_eq!(
                text, "--session sneaky rest",
                "parse_flags_text takes them as positionals"
            ),
            _ => panic!("expected Type"),
        }
    }

    /// Single-dash text is accepted in fill/type (they have no single-dash flags), while other
    /// actions keep rejecting it as an unknown flag.
    #[test]
    fn fill_type_accept_single_dash_text() {
        let inv = parse_action(&argv(&["fill", "#q", "-tail"])).expect("fill parses");
        match inv.action {
            Action::Fill { source, .. } => assert!(
                matches!(source, TextInput::Inline(t) if t == "-tail"),
                "fill takes -tail as inline text"
            ),
            _ => panic!("expected Fill"),
        }
        // A `--session` after the text still binds globally.
        let inv = parse_action(&argv(&["fill", "#q", "-tail", "--session", "abc"]))
            .expect("fill with trailing --session parses");
        assert_eq!(
            inv.session.as_deref(),
            Some("abc"),
            "a later --session still binds"
        );
        match inv.action {
            Action::Fill { source, .. } => assert!(
                matches!(source, TextInput::Inline(t) if t == "-tail"),
                "the trailing flag does not become text"
            ),
            _ => panic!("expected Fill"),
        }
        let inv = parse_action(&argv(&["type", "#q", "-dashprobe"])).expect("type parses");
        match inv.action {
            Action::Type { text, .. } => {
                assert_eq!(text, "-dashprobe", "type takes a single-dash token as text");
            }
            _ => panic!("expected Type"),
        }
        // Other actions still reject single-dash tokens as unknown flags, and
        // fill/type still reject unknown double-dash flags.
        assert!(
            parse_action(&argv(&["count", ".x", "-bogus"])).is_err(),
            "count still refuses a single-dash token"
        );
        assert!(
            parse_action(&argv(&["open", "https://x", "-bogus"])).is_err(),
            "open still refuses a single-dash token"
        );
        assert!(
            parse_action(&argv(&["fill", "#q", "--bogus"])).is_err(),
            "fill still refuses an unknown double-dash flag"
        );
    }

    /// The resolved session lands as a top-level field, under the name a person reads.
    #[test]
    fn surface_session_adds_top_level_field() {
        let mut env = out_env("fill", true, OutKind::Ok, json!({ "selector": "#q" }));
        surface_session(&mut env, "mahbot-chrome-abc");
        assert_eq!(
            env.payload["session"],
            json!("abc"),
            "the namespace prefix is not part of the name"
        );
        let wire: Value = serde_json::from_str(&env.to_json()).expect("wire json");
        assert_eq!(wire["session"], json!("abc"), "the field is on the wire");
        assert_eq!(wire["kind"], json!("ok"), "the payload stays flattened");
    }

    /// Read-back and key-listener notes are expected; a `warning` makes the envelope an error.
    #[test]
    fn a_text_input_envelope_is_ok_unless_a_warning_says_otherwise() {
        {
            let case = "text input ok envelope clean success";
            let base = json!({ "selector": "#q" });
            let resp = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "value": "filled" })),
                ..ChromeResponse::default()
            };
            let env = text_input_ok_envelope("fill", &base, resp);
            assert!(env.ok, "case {case}: a clean success stays a success");
            assert_eq!(env.kind, OutKind::Ok, "case {case}: its kind is ok");
            assert_eq!(
                env.payload["data"]["value"],
                json!("filled"),
                "case {case}: the answer's data is carried"
            );
            assert!(
                env.payload.get("warning").is_none(),
                "case {case}: no warning is invented"
            );
        }

        {
            let case = "text input ok envelope read back and key listeners without warning are ok";
            // chrome-use emits `readBack` on EVERY readable `type` success and `keyListeners`
            // on every listener-dependent `press` — not the degraded signal. Without its
            // `warning` field these are clean successes (rc 0).
            let typed = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "typed": "hi", "readBack": "hi" })),
                ..ChromeResponse::default()
            };
            let env =
                text_input_ok_envelope("type", &json!({ "selector": "#q", "text": "hi" }), typed);
            assert!(env.ok, "case {case}: a read-back alone is not degraded");
            assert_eq!(
                env.kind,
                OutKind::Ok,
                "case {case}: a read-back alone is ok"
            );

            let press = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "key": "ArrowDown", "keyListeners": 3 })),
                ..ChromeResponse::default()
            };
            let env = text_input_ok_envelope("press", &json!({ "key": "ArrowDown" }), press);
            assert!(
                env.ok,
                "case {case}: a key-listener count alone is not degraded"
            );
            assert_eq!(
                env.kind,
                OutKind::Ok,
                "case {case}: a key-listener count alone is ok"
            );
        }

        {
            let case = "text input ok envelope warning in data is error";
            let base = json!({ "selector": "#q", "text": "hi" });
            let resp = ChromeResponse {
                success: Some(true),
                data: Some(json!({
                    "typed": "hi", "readBack": "h",
                    "warning": "the field does not contain what was typed",
                })),
                ..ChromeResponse::default()
            };
            let env = text_input_ok_envelope("type", &base, resp);
            assert!(!env.ok, "case {case}: a warning in data is not a success");
            assert_eq!(env.kind, OutKind::Error, "case {case}: it is an error");
            assert_eq!(
                env.payload["warning"],
                json!("the field does not contain what was typed"),
                "case {case}: the warning is surfaced"
            );
        }

        {
            let case = "text input ok envelope warning at top level is error";
            let base = json!({ "key": "Enter" });
            let resp = ChromeResponse {
                success: Some(true),
                data: Some(json!({})),
                extra: serde_json::Map::from_iter([(
                    "warning".to_string(),
                    json!("no key listeners"),
                )]),
                ..ChromeResponse::default()
            };
            let env = text_input_ok_envelope("press", &base, resp);
            assert!(!env.ok, "case {case}: a top-level warning is not a success");
            assert_eq!(env.kind, OutKind::Error, "case {case}: it is an error");
            assert_eq!(
                env.payload["warning"],
                json!("no key listeners"),
                "case {case}: the top-level warning is surfaced"
            );
        }
    }

    fn upload_payload() -> (Vec<String>, Vec<std::path::PathBuf>) {
        (
            vec!["a.txt".to_string()],
            vec![std::path::PathBuf::from("/tmp/a.txt")],
        )
    }

    /// An upload envelope reports the delivery it made and every warning beside it.
    #[test]
    fn an_upload_envelope_reports_delivery_and_its_warnings() {
        {
            let case = "upload envelope reports the counts and the absolute files";
            let (files, resolved) = upload_payload();
            let kept = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "uploaded": 1, "attached": 1, "selector": "#f" })),
                ..ChromeResponse::default()
            };
            let env = upload_ok_envelope("#f", &files, &resolved, &kept, false);
            assert!(env.ok, "case {case}: a delivered upload is a success");
            assert_eq!(env.kind, OutKind::Ok, "case {case}: its kind is ok");
            assert_eq!(
                env.payload["uploaded"],
                json!(1),
                "case {case}: the browser's uploaded count is reported"
            );
            assert_eq!(
                env.payload["attached"],
                json!(1),
                "case {case}: the browser's attached count is reported"
            );
            assert_eq!(
                env.payload["absolute_files"],
                json!(["/tmp/a.txt"]),
                "case {case}: the absolute files are reported"
            );
            assert!(
                env.payload.get("warning").is_none(),
                "case {case}: no warning is invented"
            );
            assert!(
                env.payload.get("note").is_none(),
                "case {case}: no note is invented"
            );
        }

        {
            let case = "upload envelope fails a zero delivery";
            let (files, resolved) = upload_payload();
            let nothing = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "uploaded": 0, "attached": 0, "selector": "#f" })),
                ..ChromeResponse::default()
            };
            let env = upload_ok_envelope("#f", &files, &resolved, &nothing, false);
            assert!(!env.ok, "case {case}: a stated zero is not a success");
            assert_eq!(env.kind, OutKind::Error, "case {case}: it is an error");
            assert!(
                env.payload["error"]
                    .as_str()
                    .unwrap()
                    .contains("nothing reached"),
                "case {case}: the error says nothing reached the page: {:?}",
                env.payload["error"]
            );
        }

        {
            let case = "upload envelope notes a skipped one file check";
            let (files, resolved) = upload_payload();
            let kept = ChromeResponse {
                success: Some(true),
                data: Some(json!({ "uploaded": 1, "attached": 1, "selector": "#f" })),
                ..ChromeResponse::default()
            };
            let env = upload_ok_envelope("#f", &files, &resolved, &kept, true);
            assert_eq!(
                env.payload["note"],
                json!(crate::chrome::files::unchecked_several_files_note()),
                "case {case}: the skipped check is a note on the success"
            );
        }

        {
            let case = "upload envelope keeps a consumed input a success with its warning";
            let (files, resolved) = upload_payload();
            let consumed = ChromeResponse {
                success: Some(true),
                data: Some(json!({
                    "uploaded": 1, "attached": 0,
                    "warning": "upload events were delivered, but the file input is now empty",
                })),
                ..ChromeResponse::default()
            };
            let env = upload_ok_envelope("#f", &files, &resolved, &consumed, false);
            assert!(env.ok, "case {case}: a consumed input is still a success");
            assert_eq!(env.kind, OutKind::Ok, "case {case}: its kind stays ok");
            assert_eq!(
                env.payload["attached"],
                json!(0),
                "case {case}: the browser's attached count is kept honest"
            );
            assert_eq!(
                env.payload["warning"],
                json!("upload events were delivered, but the file input is now empty"),
                "case {case}: the warning is surfaced"
            );
        }
    }

    #[test]
    fn recognized_action_falls_back_to_usage() {
        assert_eq!(
            recognized_action(&["open".into(), "https://x".into()]),
            "open"
        );
        assert_eq!(recognized_action(&["bogus".into()]), "usage");
        assert_eq!(
            recognized_action(&["--session".into(), "s".into(), "count".into(), ".x".into()]),
            "count"
        );
        assert_eq!(
            recognized_action(&["--session=s".into(), "count".into(), ".x".into()]),
            "count"
        );
    }

    #[test]
    fn top_help_lists_every_cli_action_with_purpose() {
        let help = top_help();
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            assert!(help.contains(d.name), "top help missing action {}", d.name);
            assert!(
                help.contains(d.purpose),
                "top help missing purpose for {}",
                d.name
            );
        }
        assert!(help.contains("--keep"), "top help must name --keep");
        let lifetime = format!("{} minutes", cli_sessions::LIFETIME_MINUTES);
        assert!(
            help.contains(&lifetime),
            "top help must state the session lifetime, from the constant"
        );
        assert!(
            action_help("session").contains(&lifetime),
            "the session action's own help must state the same lifetime"
        );
        assert!(
            help.contains("session list"),
            "top help must name the session list verb"
        );
    }

    #[test]
    fn keep_requires_a_session_and_only_applies_to_action_calls() {
        let err = parse_invocation(&argv(&["--keep", "open", "https://x"]))
            .err()
            .expect("keep without a session must be refused");
        assert!(err.contains("--session"), "{err}");
        let (inv, keep) =
            parse_invocation(&argv(&["--keep", "--session", "docs", "open", "https://x"]))
                .expect("keep with a session parses");
        assert!(keep);
        assert_eq!(inv.session.as_deref(), Some("docs"));
        assert!(
            !parse_invocation(&argv(&["open", "https://x"]))
                .expect("open parses")
                .1
        );
        // The mark is not offered to the report/repair verbs: they refuse `--session`
        // outright, and a bare `--keep` on one is refused for the action itself.
        assert!(parse_invocation(&argv(&["--keep", "--session", "docs", "status"])).is_err());
        let err = parse_invocation(&argv(&["--keep", "status"]))
            .err()
            .expect("keep on a session-less action must be refused");
        assert!(err.contains("--keep is not valid for status"), "{err}");
    }

    #[test]
    fn action_help_covers_flags_kinds_and_examples() {
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let cli = d.cli.as_ref().expect("filtered to cli actions");
            let help = action_help(d.name);
            assert!(help.contains("Usage:"), "{}: missing Usage", d.name);
            assert!(help.contains("Kinds"), "{}: missing Kinds", d.name);
            assert!(help.contains("Examples:"), "{}: missing Examples", d.name);
            for k in cli.kinds {
                assert!(
                    help.contains(&format!("{} ({})", k.as_str(), k.exit_code())),
                    "{}: missing kind {}",
                    d.name,
                    k.as_str()
                );
            }
        }
        assert!(action_help("open").contains("--expect"));
        assert!(action_help("open").contains("redesign"));
        assert!(!action_help("status").contains("Flags:"));
    }

    #[test]
    fn printed_flag_rows_match_the_declaration() {
        let mut compared = 0;
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let cli = d.cli.as_ref().expect("filtered above");
            let help = action_help(d.name);
            let mut printed = printed_flag_names(&help);
            let mut declared: Vec<&str> = cli.flags().map(CliFlag::name).collect();
            printed.sort_unstable();
            declared.sort_unstable();
            assert_eq!(
                printed, declared,
                "{}: printed flag rows vs the declaration",
                d.name
            );
            compared += declared.len();
        }
        assert!(compared > 0, "the guard must have seen flags to compare");
    }

    #[test]
    fn printed_flags_have_the_shape_the_parser_enforces() {
        let mut probed = 0;
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let cli = d.cli.as_ref().expect("filtered above");
            let help = action_help(d.name);
            for (spelling, _) in printed_flag_rows(&help) {
                let name = flag_name_in(spelling).expect("a printed row spells a flag");
                if SESSION_PAIR.contains(&name) {
                    continue; // its own guard: it never reaches the common flag path
                }
                // The printed placeholder is the shape a reader is promised; the declaration's
                // `takes_value` must agree with it, and the parser enforce that shape through
                // its own shape error alone.
                let printed_takes_value = spelling
                    .split_whitespace()
                    .nth(1)
                    .is_some_and(|p| p.starts_with('<'));
                let flag = cli
                    .flags()
                    .find(|f| f.name() == name)
                    .expect("printed rows are declared");
                assert_eq!(
                    printed_takes_value, flag.takes_value,
                    "{}/--{name}: printed shape vs declared arity",
                    d.name
                );
                let mut argv = vec![d.name.to_string()];
                if let Some(scope) = flag.scope {
                    argv.push(scope.to_string());
                }
                let expected = if printed_takes_value {
                    argv.push(format!("--{name}"));
                    format!("missing value for --{name}")
                } else {
                    argv.push(format!("--{name}=1"));
                    format!("flag --{name} does not take a value")
                };
                assert_eq!(
                    parse_invocation(&argv).err().as_deref(),
                    Some(expected.as_str()),
                    "{}/--{name}: the parser must enforce the printed shape",
                    d.name
                );
                probed += 1;
            }
        }
        assert!(probed > 0, "the guard must have probed flags");
    }

    #[test]
    fn session_flags_are_printed_exactly_where_the_parser_takes_them() {
        let mut checked = [0usize; 2]; // takes them, refuses them
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let help = action_help(d.name);
            let printed = printed_flag_names(&help);
            let prints_session = printed.contains(&"session");
            let prints_keep = printed.contains(&"keep");
            assert_eq!(
                prints_session, prints_keep,
                "{}: the pair is printed as a pair",
                d.name
            );
            // A documented example is a valid invocation, so it proves acceptance.
            let example = printed_examples(&help).first().map_or_else(
                || panic!("{}: no documented example to probe the pair with", d.name),
                |example| shell_words(example),
            );
            assert_eq!(
                &example[..2],
                &["mahbot", "chrome"],
                "{}: example is a chrome command",
                d.name
            );
            let tail = &example[2..];
            let with = |prefix: &[&str]| {
                let mut line: Vec<String> = prefix.iter().map(|s| (*s).to_string()).collect();
                line.extend_from_slice(tail);
                line
            };
            if prints_session {
                checked[0] += 1;
                assert!(
                    parse_invocation(&with(&["--session", "docs"])).is_ok(),
                    "{}: prints --session so it must take it",
                    d.name
                );
                assert!(
                    parse_invocation(&with(&["--keep", "--session", "docs"])).is_ok(),
                    "{}: prints --keep so it must take it",
                    d.name
                );
            } else {
                checked[1] += 1;
                let err = parse_invocation(&with(&["--session", "docs"]))
                    .err()
                    .expect("refuses --session");
                assert!(
                    err.starts_with("--session is not valid for "),
                    "{}: --session refusal message: {err}",
                    d.name
                );
                let err = parse_invocation(&with(&["--keep"]))
                    .err()
                    .expect("refuses --keep");
                assert_eq!(
                    err,
                    format!("--keep is not valid for {}", d.name),
                    "{}: --keep refusal message",
                    d.name
                );
            }
        }
        assert!(
            checked.iter().all(|side| *side > 0),
            "the guard must have seen both sides: {checked:?} (takes, refuses)"
        );
    }

    #[test]
    fn documented_examples_parse() {
        let mut ran = 0;
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            for example in printed_examples(&action_help(d.name)) {
                let words = shell_words(&example);
                assert_eq!(&words[..2], &["mahbot", "chrome"], "{}: {example}", d.name);
                let (invocation, _) = parse_invocation(&words[2..]).unwrap_or_else(|err| {
                    panic!("{}: documented example refuses: {example}: {err}", d.name)
                });
                // The parsed action's name is also the declaration key the session-wide decision
                // reads.
                assert_eq!(
                    invocation.action.name(),
                    d.name,
                    "{}: parsed action",
                    d.name
                );
                ran += 1;
            }
        }
        assert!(ran > 0, "the guard must have run examples");
    }

    #[test]
    fn usage_and_examples_name_only_declared_flags() {
        let mut seen = 0;
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let cli = d.cli.as_ref().expect("filtered above");
            let help = action_help(d.name);
            let declared: Vec<&str> = cli.flags().map(CliFlag::name).collect();
            let usage = printed_usage(&help);
            for token in usage.split_whitespace() {
                if let Some(name) = flag_name_in(token) {
                    assert!(
                        declared.contains(&name),
                        "{}: usage names --{name}, which is not declared",
                        d.name
                    );
                    seen += 1;
                }
            }
            for example in printed_examples(&help) {
                let words = shell_words(&example);
                for token in words[2..].iter().take_while(|t| *t != "--") {
                    if let Some(name) = flag_name_in(token) {
                        assert!(
                            declared.contains(&name),
                            "{}: example names --{name}, which is not declared",
                            d.name
                        );
                        seen += 1;
                    }
                }
            }
        }
        // Description prose is deliberately NOT scanned; `session`'s and `wait`'s details mention
        // other actions' flags and this guard passes.
        assert!(
            seen > 0,
            "the guard must have seen flags in usage lines and examples"
        );
    }

    #[test]
    fn every_advertised_action_is_accepted_by_the_parser() {
        let advertised = advertised_actions(&top_help());
        assert!(!advertised.is_empty(), "top help advertises actions");
        for word in advertised {
            if let Err(err) = parse_invocation(std::slice::from_ref(&word)) {
                assert!(
                    !err.starts_with("unknown action"),
                    "the help advertises '{word}' but the parser rejects it: {err}"
                );
            }
        }
    }

    #[test]
    fn action_help_request_intercepts_per_action_help() {
        assert_eq!(action_help_request(&argv(&["open", "-h"])), Some("open"));
        assert_eq!(
            action_help_request(&argv(&["open", "https://x", "--help"])),
            Some("open")
        );
        assert_eq!(
            action_help_request(&argv(&["--session", "docs", "open", "-h"])),
            Some("open")
        );
        assert_eq!(action_help_request(&argv(&["open"])), None);
        assert_eq!(action_help_request(&argv(&["snapshot", "-h"])), None);
        assert_eq!(action_help_request(&argv(&["frobnicate", "-h"])), None);
        assert_eq!(action_help_request(&argv(&["--timeout", "5"])), None);
        // A `--` after the action word ends help lookup: what follows it is text.
        assert_eq!(
            action_help_request(&argv(&["type", "#q", "-h"])),
            Some("type")
        );
        assert_eq!(
            action_help_request(&argv(&["type", "#q", "--", "--help"])),
            None
        );
        assert_eq!(
            action_help_request(&argv(&["fill", "#q", "--", "-h"])),
            None
        );
        assert_eq!(
            action_help_request(&argv(&["count", ".x", "--", "--help"])),
            None
        );
        // A `--` before the action word changes nothing.
        assert_eq!(
            action_help_request(&argv(&["--", "open", "-h"])),
            Some("open")
        );
        // And the shielded token really is the text.
        let inv =
            parse_action(&argv(&["type", "#q", "--", "--help"])).expect("shielded text parses");
        match inv.action {
            Action::Type { text, .. } => assert_eq!(text, "--help"),
            _ => panic!("expected Type"),
        }
    }

    #[test]
    fn out_env_kind_contract_is_enforced() {
        // Every declared kind satisfies the contract.
        for d in actions::ACTIONS.iter().filter(|a| a.cli.is_some()) {
            let cli = d.cli.as_ref().expect("filtered above");
            for k in cli.kinds {
                assert!(
                    kind_contract_holds(d.name, *k),
                    "{}: declared kind {} fails the contract",
                    d.name,
                    k.as_str()
                );
            }
        }
        // An undeclared kind for a registered action violates it.
        assert!(!kind_contract_holds("status", OutKind::Network));
        // Unregistered pseudo-actions ("usage") are exempt.
        assert!(kind_contract_holds("usage", OutKind::Usage));
    }
}

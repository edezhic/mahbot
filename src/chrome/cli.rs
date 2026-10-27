//! `mahbot chrome` — browser automation CLI over the shared core.
//!
//! Dispatched from `main()` before temp-root/lock init (runs alongside the
//! daemon), so it must not rely on the `/tmp/mahbot` temp root or initialized
//! config; chrome-use resolves to the product's own copy at its standard
//! install directory, which needs no config to work out.
//! Sessions are namespaced `mahbot-chrome-*` so they can never collide with
//! the interactive tool's `agent-tab-*`.
//!
//! stdout is exactly ONE line of JSON (the [`OutEnvelope`]); stderr carries
//! human-readable diagnostics. Exit codes follow the contract: 0 success,
//! 1 site/data step failure (schedulable), 2 environment failure (fix the
//! environment, don't blind-retry), 3 usage error — and 128+signal when the call was
//! interrupted or terminated, in which case no envelope is emitted. Every step runs to the two
//! clocks the [chrome policy](crate::chrome) defines — chrome-use's clock,
//! declared to it, and the product's own kill above that — as [`StepClocks`]
//! derives them for this surface. The wait action never exposes `--load`; its one
//! raw-argv use is `open`'s internal best-effort post-navigation settle. A tab an `open` or `click`
//! step creates in a NAMED session is left as an ordinary tab, outside chrome-use's group, before
//! the envelope is out; the whole rule, including which verbs report their tab, is
//! [`crate::tools::chrome_groups`]'. That move needs the session's own tabs, and when the browser
//! will not answer for them the tab stays inside chrome-use's group: the step still succeeds, and
//! the miss is silent — the command line's tail reports nothing ([`crate::tools::chrome_groups`]
//! says why). An ephemeral session is closed right after its command and its tab closes with it.
//!
//! ## Session lifetime
//!
//! Without `--session` a call runs in a throwaway session, closed right after the
//! envelope and, if the call is interrupted or terminated, before it exits (a hard kill
//! is covered by the lifetime). A named session's lifetime, the `--keep` mark and the
//! sweeps that enforce them are [`crate::chrome::cli_sessions`]'s; this module records
//! each call's use, adds the session's in-flight mark while the call runs — which is what
//! keeps every sweep away from it, for as long as the mark holds (an hour; a call running
//! longer than that falls back to the ordinary lifetime) — and hands
//! chrome-use the same number as its cleanup backstop.

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

/// Product-side bound on the daemon-free `session list` enumerate `session
/// status` opens with: it only asks the daemon what sessions exist (it never
/// drives the browser), so a probe bound well under chrome-use's own budget is
/// what the product wants there — the relay self-heal would be pure waste.
const SESSION_LIST_TIMEOUT: Duration = Duration::from_secs(8);

/// `session status` probe bound: the product's own bounded liveness question —
/// "does the session answer at all" — so it deliberately does not wait out
/// chrome-use's own client tolerance, and declares that same short bound to
/// chrome-use ([`probe_clocks`]) rather than a longer one it would never let run.
/// A session that does not answer within it is [`Liveness::Unresponsive`] to
/// [`probe_session_liveness`]. Opt-in — never part of the default per-command path.
const SESSION_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// The manual recovery flow for a wedged named session — named only where the
/// automatic recovery did not confirm or never started a stop
/// ([`wedge_recovery_note`]). References only verbs that exist in the mahbot
/// surface.
const SESSION_RECOVERY_FLOW: &str = "`mahbot chrome session stop <name>`, then re-run the action with \
     `--session <name>` to re-create it (cookies persist in the profile; open \
     tabs do not)";

/// The sentence appended to a wedge-shaped envelope: what the automatic recovery
/// did, or — where the stop was only issued and not confirmed, or never started —
/// the manual flow that remains. Shares the recovery summary with the tool so the
/// two surfaces cannot drift.
fn wedge_recovery_note(recovery: SessionRecovery) -> String {
    match recovery {
        SessionRecovery::Stopped => recovery.summary().to_string(),
        SessionRecovery::Unanswered | SessionRecovery::NotStarted => {
            // The summary itself says what happened (the stop was issued and
            // keeps running, or none could be started); what is appended is only
            // the manual flow left to the caller.
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

/// The wedge fact is WHO ended the step, not the envelope's kind on its own.
/// chrome-use's own session-unresponsive classification is a diagnosis, so it
/// recovers directly. A deadline-expiration label — Timeout, or the Redesign
/// `open --expect --structural` gives a deadline expiration — counts only when
/// the PRODUCT's own clock ended the step ([`CLI_STEP_KILLED_BY_CLOCK`]): only
/// then was chrome-use cut off before it could classify anything, leaving the
/// session's liveness genuinely unknown, whereas the same labels also carry
/// chrome-use's own honest timeout verdicts. Every other failure is left alone —
/// a failed `expect` / `open --expect` / `wait` produces a Timeout as its normal
/// verdict, and neither it nor an unprobed error may stop a session and lose its
/// tabs. Ephemeral sessions are left to the caller's own cleanup (the `<name>`
/// verbs are unactionable for them).
///
/// Pure so tests pin the keying.
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

/// The envelope's `error` text, when it carries one.
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

/// Recover a wedge the envelope points at (bounded) and put what was done into
/// its error text (see [`wedge_action`] for which envelopes count). An ephemeral
/// session is left to the caller's own cleanup, and a session the agent chose is
/// the only one whose recovery text can name the manual flow.
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
            // The product's own kill preempted chrome-use's verdict, so the wedge
            // is only a candidate: confirm it with the same bounded `get url`
            // liveness probe the `session status` verb uses.
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
    /// The probe could not be run, or its outcome could not be established (no
    /// binary, spawn failure, unparseable envelope): nothing was learned about
    /// the session, so nothing is recovered on its account.
    Inconclusive,
}

/// Bounded liveness probe of a named session (read-only `get url`, the probe the
/// `session status` verb uses). [`Liveness::Inconclusive`] is deliberately not
/// [`Liveness::Answered`] and not [`Liveness::Unresponsive`]: a probe that could
/// not run answers nothing about the session, so its daemon is left alone.
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

/// A per-action help request: the first positional token names a CLI action
/// and a token after it is `-h`/`--help`. Flags/session values before the
/// action word are skipped. Within the tokens after the action word everything
/// from `--` on is literal text, never help. Returns the action's registry name.
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

/// Parsed CLI invocation.
struct Invocation {
    action: Action,
    session: Option<String>,
}

/// One parsed `mahbot chrome` action.
///
/// `timeout` on `Open`/`Wait`/`Expect` is the deadline the action declares for
/// the step (the user's `--timeout`, or the action's own default — forwarded to
/// chrome-use for wait/expect, and for `open` the budget that bounds the waits the
/// operation drives); `bound` on the rest is the user's own `--timeout`, or `None`
/// when they gave none — those verbs forward no `--timeout` chrome-use honours, so
/// their clock is the one mahbot declares to chrome-use (its own client tolerance
/// less the 2 s margin), a bound below that clock is REFUSED as a usage error
/// (nothing could honour it) and a longer one widens only the product's own kill
/// (see [`StepClocks`], [`parse_bound_timeout`]).
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
    /// Whether this action drives the browser in a session. The verbs that report and
    /// repair (`status`, `session …`) run in none — the same declaration field
    /// ([`CliHelp::session_refusal`]) that decides whether their help prints the
    /// session-wide rows and whether the parser takes them.
    fn runs_in_session(&self) -> bool {
        session_refusal_for(self.name()).is_none()
    }

    /// The shared-registry name of this action — the envelope's `action` field
    /// and the help lookup key. The `session` subcommands emit under
    /// `"session"`, the word the owner typed.
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

/// Where `fill` gets its text.
enum TextInput {
    /// Inline positional text (multi-word joined with spaces).
    Inline(String),
    /// `--file <path>` passthrough — chrome-use reads the file.
    File(String),
    /// `--stdin` — mahbot reads its own stdin and pipes the bytes.
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

/// A failed chrome-use step: the classified [`OutKind`] plus the error text
/// (empty for a mahbot-side deadline kill). The session-wedge recovery and its
/// text are applied after dispatch (see [`recover_session_wedge`]) — the one
/// place the resolved session name is in scope.
#[derive(Debug)]
struct StepFailure {
    kind: OutKind,
    message: String,
}

impl StepFailure {
    /// The failure envelope for `action`, layering the failure detail onto
    /// `base` params: a deadline kill reports `timeout_ms` (the product's own
    /// bound for the step — its kill) plus a factual `error` that says the
    /// product's own bound ended the call and which clock chrome-use was working
    /// to; any other failure reports the chrome-use `error` text with the
    /// accurate remediation layered on (unreachable-tab guidance), and Network
    /// failures carry the extracted net error token as `error_code` when the
    /// message contains one. The session-wedge recovery and its text are applied
    /// later, where the session name is known (see
    /// [`recover_session_wedge`]).
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

/// Outcome of one spawned chrome-use step.
type StepOutcome = Result<ChromeResponse, StepFailure>;

/// Whether any session-scoped chrome-use spawn has been attempted this
/// process — an ephemeral session can only exist after such a spawn, so
/// cleanup is skipped (no stderr noise) when every failure happened before
/// any spawn attempt. Never read before `run_cli` resets it.
static CHROME_USE_SPAWNED: AtomicBool = AtomicBool::new(false);

/// Whether any chrome-use step this process ran exited while a process it left
/// behind still held the step's output channel
/// ([`CliOutput::leftover_pipes`]) — including the ephemeral close's own spawn,
/// whose envelope is already out, so [`run_cli`] reports that leftover on stderr.
/// The CLI handles ONE invocation per process, so this is the invocation's whole
/// record; [`run_action`] turns a step's leftover into a note on the envelope
/// ([`chrome_leftover_pipe_note`]) — on a failure too — and never into a failure.
/// Never read before `run_cli` resets it.
static CHROME_USE_LEFTOVER_PIPES: AtomicBool = AtomicBool::new(false);

/// Whether the PRODUCT's own clock ended a step whose outcome IS the caller's
/// verdict ([`CliRun::TimedOut`]) ABOVE the clock chrome-use itself was working
/// to — the step's kill rode above its `chrome_side` (every step declares that
/// clock to chrome-use through `AGENT_BROWSER_DEFAULT_TIMEOUT`, see
/// [`crate::chrome::spawn::apply_chrome_side`]). chrome-use was then cut off
/// before it could classify anything, so what a named session's liveness is
/// remains unknown to it, which is exactly what [`recover_session_wedge`] keys
/// on. ONE rule decides the mark, applied where the step ends ([`spawn_step`]):
/// only a step that may run chrome-use's relay self-heal
/// ([`CliRecovery::Allowed`]) attributes anything, and only when its kill sat
/// above that `chrome_side`. The product's own probes and the operation's
/// best-effort sub-steps are [`CliRecovery::Suppressed`] — the product bounds
/// them on purpose and their outcome never becomes the envelope — so they can
/// never attribute. [`run_action`] resets the mark before the step and reads it
/// right after it returns; a later step of a composite operation cannot
/// erase an earlier decisive step's mark, and the wedge probe's own steps cannot
/// leak into it.
static CLI_STEP_KILLED_BY_CLOCK: AtomicBool = AtomicBool::new(false);

/// Whether this call created the session's record (rather than restarting one that
/// already existed). Read through [`record_materialized`] by everything that ends the call
/// early — the interrupt path and a pre-flight refusal — to drop a record whose session
/// never reached the browser, exactly as the normal tail does, so a kept or interrupted
/// session cannot leave a row nothing will ever reap. Never read before `run_cli` resets it.
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
    // A `-h`/`--help` after the action word is per-action help — anywhere before
    // the first `--`, not just immediately after the action word (so
    // `open <url> --help` and `--session x open -h` work too). From that `--` on
    // the tokens are the action's text, never help (`type '#q' -- --help` types
    // it); on a verb with no literal text such a token stays the usage error it
    // always was. A `-h` before any action word falls through to top-level help /
    // normal parsing: no valid invocation starts with a `-h`-looking token, and
    // parse_flags rejects those as usage errors, so per-action help only turns
    // would-be usage errors into help.
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
    // The action's own call races the platform's interruption signals, so an
    // interrupted call still closes the throwaway session it created.
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
    // The action is answered and its envelope is out: what is left is best-effort cleanup, and
    // its closes are raced against a second interruption, which must end the process at once
    // instead of waiting one out. A record left by an aborted close is durable and the lifetime
    // sweep closes it. A step whose tab tail ran collapsed its groups before the envelope, where
    // the caller can see its result ([`run_action`]); any other step collapses its groups here,
    // after the answer — including an `open` whose baseline the browser would not answer.
    //
    // A record this call created that no session-scoped chrome-use spawn was ever
    // attempted for describes nothing: drop it rather than let it linger — especially a
    // kept one, which has no lifetime to expire it.
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
            // The close runs after the envelope is emitted, so a helper it leaves
            // behind cannot be noted in it — it is reported here, like every other
            // diagnostic of that close.
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
            // to leave collapsed — unless the step failed on its environment, where the tail cannot
            // reach the browser either ([`tail_cli`]). A step whose tab tail ran did this before
            // the envelope ([`run_action`], [`chrome_groups`]); the rest do it here, including an
            // `open` whose baseline the browser would not answer. Cosmetic and best-effort: it
            // reports nothing.
            chrome_groups::collapse_own_groups(&cli, &s.name).await;
        }
    }
    // The call is over: its in-flight mark goes, so the session is idle from here — and a
    // sweep that waited the whole call out may now close it on the ordinary clock. A
    // session that was just forgotten (an unmaterialized record, a closed throwaway) has
    // nothing to clear.
    if let Some(s) = outcome.session.as_ref() {
        cli_sessions::note_end(&s.name);
    }
    outcome.envelope.kind.exit_code()
}

/// The session this call runs in, resolved once for the interrupt path and the action
/// alike. Only an action that drives the browser owns a throwaway session: the verbs
/// that report and repair run in none, so an interrupted one has nothing to close.
fn call_session(invocation: &Invocation, keep: bool) -> CliSession {
    let (name, ephemeral) = resolve_session(invocation.session.as_deref());
    CliSession {
        name,
        ephemeral: ephemeral && invocation.action.runs_in_session(),
        keep,
    }
}

/// What `dispatch` produced: the envelope, the session the action ran in (for the close/collapse
/// tails), whether this call set the keep mark, and whether the tab tail ran before the envelope
/// with a client resolved for it — its collapse is then done, and the post-envelope one is skipped.
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

/// The wait an action's own call races against: Ctrl-C, a closed terminal or a
/// parent's kill ends the CLI, whose throwaway session must still be closed first.
/// Resolves to the conventional 128+signal exit code; a failed registration leaves
/// the platform's own handling in place (and this waits forever).
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
/// not the call's to close), then exits with the interruption's `128+signal` code. A
/// second interruption ends the process at once instead of waiting the close out — the
/// close is raced against it ([`close_ephemeral_unless_interrupted`]). A record this call
/// created whose session never reached the browser describes nothing and is dropped, as
/// it is without an interruption.
async fn close_throwaway_then_exit(session: &CliSession, code: i32) -> ! {
    if session.ephemeral && CHROME_USE_SPAWNED.load(Ordering::Relaxed) {
        CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
        close_ephemeral_unless_interrupted(&session.name).await;
    } else if !record_materialized() {
        cli_sessions::forget(&session.name);
    }
    // The process is going: its mark must not outlive it, even though a sweep would call
    // the mark stale as soon as this pid is gone (a named session is unaffected by the
    // interruption and stays alive on the ordinary clock).
    cli_sessions::note_end(&session.name);
    std::process::exit(code);
}

/// Whether this call's record describes a session that actually reached the browser: no
/// record was created (a later call's record already described one), or a session-scoped
/// step of this call was attempted ([`CHROME_USE_SPAWNED`]).
fn record_materialized() -> bool {
    !SESSION_RECORD_CREATED.load(Ordering::Relaxed) || CHROME_USE_SPAWNED.load(Ordering::Relaxed)
}

/// [`close_ephemeral`], abandoned when a second interruption arrives first: the call is
/// already answered (or already interrupted), so a close the user interrupted again must
/// not keep the process alive for its bound. `true` when the close ran to completion.
async fn close_ephemeral_unless_interrupted(name: &str) -> bool {
    tokio::select! {
        () = close_ephemeral(name) => true,
        _ = wait_for_interruption() => false,
    }
}

// ── Parsing ──────────────────────────────────────────────────────

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
    // The declaration is the one decision on which actions take the session-wide
    // pair: the help renders their rows from the same field.
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
        // The mark is a property of a named session the call drives, so it needs both a
        // browser-driving action and a name — checked once the action is known, so the
        // refusal never names a flag the action itself forbids.
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

/// Pull the global `--session` and `--keep` flags out of anywhere before the `--`
/// end-of-options marker, returning the session value (last occurrence wins), the
/// keep request, and the remaining tokens (everything from `--` on kept verbatim).
fn extract_global_flags(args: &[String]) -> Result<(Option<String>, bool, Vec<String>), String> {
    let mut session: Option<String> = None;
    let mut keep = false;
    let mut remaining = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        // `--` ends option parsing (the fill/type text escape hatch): the
        // rest is verbatim text, never scanned for --session.
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

/// Split action args into positional tokens and a `--flag` map, handling both
/// `--flag value` and `--flag=value`. `accepted` is the action's declaration:
/// each flag's `takes_value` decides whether a value is read.
fn parse_flags(args: &[String], accepted: &[&CliFlag]) -> Result<(Vec<String>, Flags), String> {
    parse_flags_impl(args, accepted, false)
}

/// [`parse_flags`] for the fill/type text actions: a single-dash token is
/// taken as literal positional text (these actions have no single-dash
/// flags, so `fill '#q' -tail` fills the text `-tail` naturally instead of
/// demanding the `--` shield). Unknown `--flags` are still rejected.
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
        // `--` ends option parsing: the rest are positionals verbatim (the
        // escape hatch for a leading-dash fill/type text).
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

/// Positional/flag accumulator for one action.
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

/// chrome-use 1.5.101 selector-first expect grammar, validated against the
/// safe allowlist: `<sel> <visible|hidden|present>`, `count <sel> <op> <n>`,
/// `text|value <sel> <pred> <value>`, `attr <sel> <name> <pred> <value>`,
/// `url <pred> <pattern>`. Trailing words of a text/value/attr/url payload
/// are joined with spaces (shell quoting is the normal path).
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

/// `upload <selector> <path...>` — the paths stay as the caller wrote them here;
/// the action resolves each against this command's working directory at dispatch,
/// so a bad path is reported with the reason it failed, not as a shape error.
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
/// ([`crate::cli::resolve_output_dir`]), the same rule every producing command
/// line follows: it must already exist, be the caller's own, and the picture is
/// never written over anything.
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

/// The value of a forwarded `--timeout` — the deadline this verb IS given
/// (`wait`, `expect`, and `open`, whose `--expect` wait forwards its remaining
/// budget), so chrome-use honours it in full. A value at or above chrome-use's own
/// client tolerance ([`CHROME_USE_OWN_BUDGET`]) is refused: chrome-use cannot work
/// past that tolerance, so a deadline at it makes the tool run out of tolerance
/// instead of reporting its own reason, and its session-unresponsive verdict
/// (which stops the session and loses its tabs) replaces the honest one. At least
/// 1 s, and `None` when the user gave no `--timeout` (the verb's own default then
/// applies at its parse site).
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

/// The value of a bound `--timeout` — every verb chrome-use takes no per-call
/// deadline for (count, eval, extract, click, fill, type, press, upload). Nothing
/// can make chrome-use give up earlier on those, so a bound below the clock the
/// product declares to chrome-use ([`CHROME_USE_DECLARED_BUDGET`]) CANNOT be
/// honoured: it is refused as a usage error rather than accepted and silently
/// discarded. A bound at or above that clock is accepted: the call is never cut off
/// below it, and above the clock it widens the product's own kill
/// ([`StepClocks::tool_clock`]).
/// There is no upper limit here: while the call runs, its session is not closed — the call
/// says so through `cli_sessions`' in-flight mark, honoured for an hour (a call longer than
/// that falls back to the session's ordinary lifetime).
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

/// The user's own `--timeout` in seconds, or `None` when they gave none — the one
/// seconds parser behind [`parse_forwarded_timeout`] and [`parse_bound_timeout`],
/// which the parse sites call by name. There is deliberately no per-step default
/// any more: a step's declared clock is [`CHROME_USE_DECLARED_BUDGET`] unless the
/// verb forwards a deadline chrome-use honours in full (`wait`/`expect`), and the
/// actions that declare a deadline of their own (`wait`, `expect`, `open`) apply
/// their default at their parse site.
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

// ── Session resolution ───────────────────────────────────────────

/// Resolve the session name and whether it is ephemeral: a named `--session`
/// becomes `mahbot-chrome-<name>` (idempotent for an already-prefixed name);
/// no flag yields an ephemeral `mahbot-chrome-ephemeral-<suffix>` session.
///
/// The flag's own value is validated at extraction ([`extract_global_flags`]),
/// so a name reaching here is already a plain identifier.
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

/// Session namespaces the CLI must never stop without an explicit `--force`:
/// the interactive tool's per-run sessions and link enrichment's sessions.
const PROTECTED_SESSION_PREFIXES: [&str; 2] = ["agent-tab-", "link-enricher-"];

/// Validate a caller-supplied session name. The name becomes part of a command
/// recipe the agent is told to run (`mahbot chrome session stop <name> --force`
/// in a leftover note), so it is restricted to the plain identifier the tool
/// mints for its own sessions; anything else is a usage error rather than a
/// name that could turn that recipe into a different command. The alphabet is
/// chrome-use's own (`validation::is_valid_session_name`: alphanumerics, '-' and
/// '_'): a '.' is refused, because chrome-use refuses it on `session stop` — a
/// name the product accepted but chrome-use will not stop is a session that can
/// be neither cleared by hand nor recovered, its daemon and tabs left behind.
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

/// Resolve a session name for a session-word subcommand. A bare name is
/// treated as a CLI session name and prefixed (consistent with `--session`);
/// CLI-namespace and protected-namespace names pass through as-is (status may
/// probe protected sessions read-only; only stop gates them behind --force).
/// The name is validated when the invocation is parsed
/// ([`validate_session_name`]).
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

// ── Dispatch ─────────────────────────────────────────────────────

/// The caller-operand checks the pre-action arm runs BEFORE the readiness gate —
/// the usage errors an action can answer without probing the environment, so they
/// are never buried under an environment refusal: `open`'s URL and `--expect`
/// validation, an attach's local paths, and a capture's destination. Each verb
/// re-runs its own check for the values it actually uses.
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

/// Resolve an attach's paths against the directory this command was run in — the
/// caller's own operand check, run by the pre-flight (before any environment probe)
/// and again by the verb for the files it hands chrome-use.
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

/// One parsed call: the verbs that run in no session (they report and repair) are answered
/// as they are; every other action drives the browser in the call's session through
/// [`run_action`].
async fn dispatch(invocation: &Invocation, session: &CliSession) -> DispatchOutcome {
    if invocation.action.runs_in_session() {
        run_action(&invocation.action, session).await
    } else {
        // These calls run no step, but they are still clocked for their envelope. None of them
        // opens a page, so the tab report the browser actions keep has no reader here.
        let started = Instant::now();
        let (mut env, _) = execute(&invocation.action, &session.name).await;
        stamp_elapsed(&mut env, started);
        DispatchOutcome::plain(env)
    }
}

/// One action that drives the browser in its session: the pre-flight, the record of this
/// call's use, the step itself, and the envelope's post-step facts. The action's clock starts
/// before the pre-flight and is stamped onto the envelope right after the step, so `elapsed_ms`
/// covers the pre-flight, `open`'s baseline read and the step; the wedge recovery and the group
/// tail that follow are outside it (the tail is still inside the caller's wall time).
async fn run_action(action: &Action, session: &CliSession) -> DispatchOutcome {
    // The action's own clock: every envelope it returns — refusals included — is stamped
    // with it (see [`stamp_elapsed`]).
    let started = Instant::now();
    let use_outcome = match preflight(action, session, started).await {
        Ok(outcome) => outcome,
        Err(refused) => return refused,
    };
    // The session exists from here on: chrome-use is handed the lifetime as its cleanup
    // backstop, or the cleanup switched off for a kept one.
    set_group_cleanup_backstop(cli_sessions::backstop_ms(use_outcome.kept));
    CLI_STEP_KILLED_BY_CLOCK.store(false, Ordering::Relaxed);
    // `open` names no tab in its reply, so the tabs the session already had are read before the
    // step to tell the one it creates apart later; every other verb's own tab comes from its
    // reply ([`chrome_groups`]). An unreadable answer only costs the moving — the tab then stays in
    // chrome-use's group. The pre-flight has already refused a step its own arguments reject, so no
    // listing is spent on one.
    let tabs_before = match action {
        Action::Open { .. } if !session.ephemeral => tabs_before_open(&session.name).await,
        _ => None,
    };
    let (mut env, opened_tab) = execute(action, &session.name).await;
    // Read the kill attribution before ANY recovery call — hygiene only: the wedge
    // probe's own steps are probes ([`CliRecovery::Suppressed`]), so they cannot
    // re-attribute this action's ending.
    let killed_by_clock = CLI_STEP_KILLED_BY_CLOCK.load(Ordering::Relaxed);
    // The resolved session (named or defaulted ephemeral) and the call's own wall time ride
    // every envelope (see [`situate`]).
    situate(&mut env, &session.name, started);
    // A process chrome-use left behind still holding a step's output channel is a NOTE
    // whatever the call's own outcome — also on a failure: the call finished, its exit
    // status and output are the answer, and the leftover was started deliberately and is
    // left running. It sits alongside `error` and never changes the kind or the exit
    // code.
    if CHROME_USE_LEFTOVER_PIPES.load(Ordering::Relaxed) {
        surface_leftover_note(&mut env, &session.name);
    }
    // A session that stopped answering is RECOVERED here — the one place the resolved
    // session name is in scope — and the envelope's error text says what was done
    // ([`recover_session_wedge`]). Whether the session is named is the resolution's own
    // answer ([`CliSession::ephemeral`]), not re-derived from the name. It runs before the group
    // tail below: a wedged session answers no listing, so the tail would spend its bound on a
    // daemon the recovery is about to replace.
    recover_session_wedge(&mut env, &session.name, !session.ephemeral, killed_by_clock).await;
    // The tab this step created is taken out of chrome-use's group here, before the caller sees
    // the result, and the groups this session's tabs sit in are collapsed — see [`chrome_groups`]
    // for the whole rule. Best-effort and silent: its trouble never becomes the step's own failure
    // and is reported nowhere. A step whose tab the tail has no way to establish — no
    // tab created, or an `open` baseline the browser would not answer — leaves the collapse to the
    // post-envelope tail ([`run_cli`]), so its answer is not held back by browser work it does not
    // depend on.
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

/// The tabs the session has created before an `open` step — the baseline its reply cannot give.
/// `None` when there is nothing to compare against: the browser would not answer (the step's tab
/// then stays in chrome-use's group) or the helper cannot be resolved at all.
async fn tabs_before_open(session: &str) -> Option<std::collections::BTreeSet<i64>> {
    let cli = cli_path()?;
    chrome_groups::created_tabs(&cli, session).await
}

/// The chrome-use binary a step's group tail drives, when the step's session can be reached at all:
/// a named session whose step ran — not one refused on its environment — after the helper actually
/// spawned. The one place that condition is spelled out; the pre-envelope tab tail and the
/// post-envelope collapse both read it here.
fn tail_cli(env: &OutEnvelope, session: &CliSession) -> Option<std::path::PathBuf> {
    if env.kind == OutKind::Environment
        || !CHROME_USE_SPAWNED.load(Ordering::Relaxed)
        || session.ephemeral
    {
        return None;
    }
    cli_path()
}

/// Every action's own work, in one place: the report/repair verbs [`dispatch`] answers
/// session-unscoped, and the browser actions it wraps in the pre-flight and the tail. `opened` is
/// the tab a verb's reply named as one its page opened — a `click` sets it, every other verb leaves
/// it alone; it is the caller's own local, never process state.
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
/// presence, this call's own record of use, and the pre-action readiness gate. An `Err` is
/// the refusal envelope to return; the usage and missing-helper refusals come before anything
/// is recorded, so such a call never restarts a lifetime, while a readiness refusal ends the
/// use it recorded a moment earlier.
async fn preflight(
    action: &Action,
    session: &CliSession,
    started: Instant,
) -> Result<cli_sessions::UseOutcome, DispatchOutcome> {
    let name = session.name.as_str();
    // A malformed URL or `--expect` selector is a usage error whatever the environment
    // looks like, so it is answered without a readiness probe that could bury the precise
    // usage error under an environment refusal (see [`validate_action_args`]).
    if let Some(mut refused) = validate_action_args(action) {
        situate(&mut refused, name, started);
        return Err(DispatchOutcome::plain(refused));
    }
    // A session's lifetime contract is fixed when it is created: a keep request on a
    // session that already exists without the mark is a usage error, answered before any
    // environment probe.
    let recorded = match cli_sessions::check_keep(name, session.keep) {
        Ok(recorded) => recorded,
        Err(error) => return Err(refusal(action, name, OutKind::Usage, &error, started)),
    };
    // The pre-action gate: no action is dispatched until the connection to the owner's
    // REAL browser is established, so the work can never quietly land in a browser
    // chrome-use launches itself. The probe/cleanup verbs (`status`, `session …`) are
    // answered before this and are deliberately not gated — they are what reports and
    // repairs.
    //
    // A missing helper is its own case, checked BEFORE the gate: the gate's refusal is
    // about a connection it cannot even probe, so running it (and its recovery pass)
    // would answer an absent helper with the long speculative readiness refusal instead of
    // this precise missing-install report — the same envelope `require_cli` builds.
    let Some(path) = cli_path() else {
        return Err(refusal(
            action,
            name,
            OutKind::Environment,
            "chrome-use CLI not found",
            started,
        ));
    };
    // A keep request on a session this record has never seen is a creating call only if
    // chrome-use is not already running it: a session that predates the record (or came
    // from elsewhere) already carries the idle cleanup its daemon started with, and the `0`
    // this call hands its own spawns cannot reach it — the stop-and-restart refusal applies
    // to it as much as to a recorded one.
    if session.keep && !recorded && session_running(&path, name).await {
        return Err(refusal(
            action,
            name,
            OutKind::Usage,
            &cli_sessions::mark_refusal(name),
            started,
        ));
    }
    // This call's use restarts the session's lifetime, applies the keep request and adds
    // this call's in-flight mark — from here no sweep closes the session, including one that
    // runs during the readiness gate below.
    let use_outcome = match cli_sessions::note_use(name, session.keep) {
        Ok(outcome) => outcome,
        Err(error) => return Err(refusal(action, name, OutKind::Usage, &error, started)),
    };
    SESSION_RECORD_CREATED.store(use_outcome.created, Ordering::Relaxed);
    if let Err(why) = ensure_ready_for_actions().await {
        // The call ends without a step, so it ends its use itself: the mark goes, and a
        // record it created for a session that never reached the browser is dropped (the
        // same cleanup `run_cli`'s tail does after a step).
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

/// The facts every action's envelope carries: the resolved session (so a swallowed or
/// silently defaulted `--session` is always observable) and the call's own wall time — a
/// failing call's duration is read straight off it, while `timeout_ms` (the declared/kill
/// bound) stays a separate fact.
fn situate(env: &mut OutEnvelope, name: &str, started: Instant) {
    surface_session(env, name);
    stamp_elapsed(env, started);
}

/// Insert the leftover-helper note as a top-level envelope field on the result —
/// success or failure (the payload is flattened, so a payload key is a top-level
/// stdout key, sitting alongside `error`). The note carries a `session stop` recipe,
/// so it names the session the way a person writes it ([`cli_sessions::display_name`]),
/// exactly like the envelope's own `session` field: `session stop` resolves a bare name
/// into the record's, so the recipe works as written.
fn surface_leftover_note(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        let name = cli_sessions::display_name(name);
        obj.insert("leftover".into(), json!(chrome_leftover_pipe_note(name)));
    }
}

/// Insert the session's name as a top-level envelope field (the payload is flattened, so
/// a payload key is a top-level stdout key): the name a person reads
/// ([`cli_sessions::display_name`]), never the record's own key. The `session` subcommand
/// envelopes carry their own `session` key and skip this — they run session-unscoped.
fn surface_session(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("session".into(), json!(cli_sessions::display_name(name)));
    }
}

/// Stamp one action envelope with how long the call took, from the instant the
/// action's own clock started. Every envelope of an action that RAN carries it —
/// a success, a failure of any kind, and an argument/environment refusal alike —
/// because a call's duration is a fact about the call itself, not something a
/// reader should have to infer or time for itself; the one envelope without it is
/// the usage refusal of a command line that never parsed ([`run_cli`]), where no
/// action ever started. `timeout_ms`, the declared or killed bound, stays a
/// separate fact.
fn stamp_elapsed(env: &mut OutEnvelope, started: Instant) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("elapsed_ms".into(), json!(started.elapsed().as_millis()));
    }
}

// ── Step runner ──────────────────────────────────────────────────

/// The clocks one CLI step runs to. The two-clock rule itself — which clock is
/// declared to chrome-use, and where the product's own kill rides — is
/// [the chrome policy](crate::chrome)'s; each constructor below states which
/// shape a step of this surface takes. A product-side kill reports the product's
/// own bound for the step — its `kill` — as the envelope's `timeout_ms`.
#[derive(Debug, Clone, Copy)]
struct StepClocks {
    /// The two clocks of this step ([`crate::chrome::ChromeCallClocks`]).
    call: ChromeCallClocks,
    /// Whether chrome-use's own relay self-heal may run inside this step.
    recovery: CliRecovery,
}

impl StepClocks {
    /// Assemble a step's clocks from the declared deadline and the recovery
    /// policy bound ONCE for the step — the derived pair plus that same binding,
    /// so a step's kill and the policy it spawns with can never disagree.
    fn of(declared: Option<Duration>, recovery: CliRecovery) -> Self {
        Self {
            call: clocks(declared, recovery),
            recovery,
        }
    }

    /// `wait`/`expect`: the step forwards its declared deadline as `--timeout`,
    /// which chrome-use honours in full, so the declared value IS the
    /// chrome-side clock; its relay self-heal is allowed.
    fn forwarded(deadline: Duration) -> Self {
        Self::of(Some(deadline), CliRecovery::Allowed)
    }

    /// Every other agent-facing verb (click, count, eval, extract, fill, type,
    /// press, upload): the step forwards no `--timeout` chrome-use
    /// honours, so [`CHROME_USE_DECLARED_BUDGET`] — one [`KILL_SLACK`] under
    /// chrome-use's own client tolerance — is the clock it is given.
    /// `own_bound` is the caller's `--timeout`, which [`parse_bound_timeout`]
    /// accepts only at or above that clock (a shorter one is refused there:
    /// chrome-use takes no per-call deadline for these verbs, so nothing could
    /// honour it). A larger bound is one the PRODUCT must really let the call run
    /// to, so it widens the kill alone: chrome-use cannot work past its own
    /// tolerance, and declaring a larger deadline to it would make it run out of
    /// that tolerance and fail as a wedged session instead of reporting its own
    /// reason (see [the chrome policy](crate::chrome)). The reported `timeout_ms`
    /// is then the bound the call really ran to, never one it was killed below.
    /// Its relay self-heal is allowed.
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

    /// The composite operation's OWN step (the `open` navigation): chrome-use
    /// runs it to its own client tolerance — it neither reads nor honours a
    /// `--timeout` for `open`'s navigate — so the clock declared to it is
    /// [`CHROME_USE_DECLARED_BUDGET`] and the kill is
    /// [`crate::chrome::kill_bound`] above it; its relay self-heal is allowed.
    /// The operation's budget is deliberately NOT a kill bound here: a navigation
    /// chrome-use was still working on must be allowed to report its own reason,
    /// so the operation may legitimately outlive `--timeout`.
    fn operation() -> Self {
        Self::of(None, CliRecovery::Allowed)
    }

    /// A step of an operation the product bounds itself: `bound` is the
    /// product-side budget for the step, `chrome_side` the clock declared to
    /// chrome-use. It is a probe in everything but name (see the
    /// [chrome policy](crate::chrome)): it declares `chrome_side` honestly to
    /// chrome-use, suppresses the relay self-heal, and its kill rides
    /// [`KILL_SLACK`] above `bound` ([`kill_bound`]) — so a product bound, never
    /// chrome-use's verdict, is what ends it. Its failure never fails the
    /// operation. Every one of the operation's best-effort sub-steps (the
    /// error-page probe, the post-navigation settle, the content capture) takes this
    /// shape.
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

    /// The product's own probe (`session stop`, `session status`): `bound` IS the
    /// kill and [`probe_clocks`] declares that same bound to chrome-use — capped at
    /// [`CHROME_USE_DECLARED_BUDGET`] for a probe whose own bound sits above it, as
    /// the `session stop` paths do — and its relay self-heal is suppressed: the
    /// documented exception, where the product reports "the tool did not answer"
    /// instead of spending the recovery window.
    fn probe(bound: Duration) -> Self {
        let recovery = CliRecovery::Suppressed;
        Self {
            call: probe_clocks(bound),
            recovery,
        }
    }
}

/// Run one chrome-use step under `clocks` and classify its output. `session`
/// scopes the call via `--session`; `None` leaves the call session-unscoped
/// (`session stop` names its session via the positional instead). `input` pipes
/// a stdin payload (only `fill --stdin` uses one).
async fn spawn_step(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    clocks: StepClocks,
    input: Option<&[u8]>,
) -> StepOutcome {
    // A session-scoped spawn attempt may materialize the ephemeral session
    // (a vanished-binary race still sets this; the close attempt is then a
    // harmless best-effort no-op).
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
            // The product's own clock ended this step. That is wedge evidence
            // only when the step's outcome IS the caller's verdict
            // ([`CliRecovery::Allowed`]) and its kill rode above the clock
            // chrome-use itself was working to: every step declares that clock
            // to chrome-use (as `AGENT_BROWSER_DEFAULT_TIMEOUT`), so a kill above
            // it means the tool did not answer its own deadline. A step the
            // product bounds itself (probes and the operation's best-effort
            // sub-steps) is [`CliRecovery::Suppressed`] and can never attribute
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
            // The child's exit decides the step; a process it left behind holding
            // the output channel is recorded here and reported as a NOTE on the
            // envelope, never as a failure (see [`chrome_leftover_pipe_note`]).
            // OR-ed, not stored: a composite operation runs several steps, and one
            // step's clean finish must not erase an earlier step's leftover.
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

/// Classify one chrome-use output (the `CliRun::Output` payload) into a step
/// outcome. Pure so tests pin the mapping. `expect` is the one chrome-use
/// action whose `--json` envelope can succeed on a non-zero exit — a failed
/// assertion arrives as `success:true` with exit 1, the verdict riding in
/// `data` — so envelope-success is trusted over the exit code there and
/// nowhere else. `truncated` is the product's own flag
/// ([`crate::chrome::spawn::CliOutput::truncated`]): whenever there is no parsed
/// envelope — whatever the exit status — an answer the product cut off names
/// itself rather than reading as chrome-use's malformed output. The parse is the
/// shared tolerant one ([`crate::chrome::contract::parse_first`]), so trailing
/// bytes a leftover process wrote after the envelope are ignored.
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
                // The stderr fallback can also carry a timeout phrasing plus the
                // canned hint (chrome-use prints diagnostics there when stdout is
                // non-JSON), so it is classified and sanitized too.
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

/// A successful step whose envelope carries chrome-use's browser-replacement
/// note is a plain failure, never a quiet success: the command ran in a browser
/// chrome-use launched ITSELF rather than the owner's real one, so the page state
/// the caller assumed is not there and nothing it read or wrote happened in the
/// owner's session.
///
/// Classified [`OutKind::Environment`] (rc 2 — fix the environment rather than
/// blind-retry), and deliberately NOT the daemon-down path: the daemon and the
/// relay may be healthy, and only the connection to the real browser was lost, so
/// daemon health is left untouched and auto-recovery is not woken.
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

// ── Actions ─────────────────────────────────────────────────────

/// The `out_env` kind contract: a kind emitted for a registered CLI action
/// must be declared in that action's shared-registry kind list, so the help
/// text cannot drift from runtime emission. Unregistered pseudo-actions (the
/// `"usage"` fallback) and non-CLI actions are exempt. Kept a pure predicate
/// so the unit test exercises it in every build profile, not just debug.
fn kind_contract_holds(action: &str, kind: OutKind) -> bool {
    match actions::desc(action) {
        None => true,
        Some(d) => d.cli.as_ref().is_none_or(|c| c.kinds.contains(&kind)),
    }
}

/// Construct an [`OutEnvelope`].
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

/// Resolve the chrome-use CLI path, or build the environment-failure envelope that
/// carries the action's own params plus the reason.
///
/// The standalone `mahbot chrome` CLI deliberately resolves ONLY the product's
/// own copy ([`crate::tools::chrome_daemon::cli_path`], the location the
/// helper's own installer uses) — never a `chrome-use` found on the owner's
/// search path or in a home location, so a foreign or leftover copy is never
/// run. A host whose product-owned install has not happened yet therefore
/// reports `chrome-use CLI not found` even when some other `chrome-use` is on
/// PATH.
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

/// The request fields an action reports when its own envelope cannot be built, as the
/// object a refusal extends with its reason. Taking them as pairs is what makes the
/// payload an object by construction, so a refusal can never lose either half.
fn request(
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> serde_json::Map<String, Value> {
    fields
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// `status` — pure preflight: the readiness report and RAW probes only (never
/// Chrome auto-launch, no environment mutation, no recovery — only the on-demand
/// verbs and the pre-action gate recover).
///
/// It reports what WAS established about the owner's real browser
/// ([`crate::tools::chrome_daemon::Readiness::report`]) rather than a list of raw
/// probes, and states the machine-readable `verdict` it amounts to: `ready`
/// (rc 0), `blocked` (rc 2 — a fact established the connection cannot work), or
/// `not-proven` (rc 0 — nothing could be established either way, and the
/// pre-action gate would still dispatch an action). A CLI that cannot state its
/// version is rc 2 as well. The `chrome_use`, `relay_up`, `chrome_running` and
/// `display` keys stay for scripts that read them — the last three as the raw
/// tri-state facts of the same snapshot the report renders.
async fn status() -> OutEnvelope {
    // One `--version` spawn in the happy path: a parsed version proves the
    // CLI is available; `cli_probe` only re-runs to classify WHY it didn't.
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
    // One bounded, daemon-free readiness snapshot (cached for HEALTH_TTL), every
    // fact tri-state — including the display fact, taken from the snapshot
    // instead of re-probed here.
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
    // The three-way state in one word, so a machine read does not have to infer
    // it from the report text: `ready` is PROVEN, `blocked` means a fact
    // established the connection cannot work, and `not-proven` means nothing was
    // established either way (the gate still dispatches an action there).
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
    // Three states, mirroring the pre-action gate. A CLI that cannot state its
    // version, and a fact that RULES THE CONNECTION OUT, are both environment
    // failures. A snapshot that established nothing wrong — not proven, nothing
    // ruled out — is a reporting gap: the gate would still dispatch an action,
    // so the report's own verdict is the answer and this is not reported as a
    // broken environment (the readiness snapshot cannot name an absent/broken
    // CLI, which is why the version probe's verdict leads there).
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

/// The honest `error` text for a committed-Chrome-error-page network failure:
/// the specific cause when the net error code was extracted, the generic
/// fallback otherwise (never lose the honest generic reason).
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

/// The factual `error` text for a product-side deadline kill; `what` names the
/// step that did not complete. `chrome_side` is the clock that step declared to
/// chrome-use and `kill` the product's own bound that ended it.
///
/// It says plainly that the PRODUCT's own bound ended the call — never that
/// chrome-use timed out. When the kill rode above the declared clock (a step
/// whose outcome is the caller's verdict) it says so, so a truncated call is
/// never mistaken for chrome-use's own verdict; otherwise (a probe, or the
/// operation's own budget ending one of its best-effort sub-steps) it says the
/// bound was the product's own.
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
        // The product bounded this step itself: its own probe bound, or the
        // operation's own budget ending one of its best-effort sub-steps.
        format!(
            "deadline reached after {kill}ms — {what}: that is the product's own bound for the \
             step (chrome-use was given {}ms), so a product-side bound — not chrome-use's \
             verdict — ended the call",
            chrome_side.as_millis()
        )
    }
}

/// Post-navigation settle cap for `open`'s plain path: a best-effort
/// `wait --load networkidle` (what the interactive tool does after its open)
/// so the first step after `open` on heavy SPAs is not racing a still-settling
/// page under daemon contention.
const SETTLE_CAP: Duration = Duration::from_secs(10);

/// Budget reserved for content capture so the settle can never consume the
/// whole remaining operation budget and silently drop the open's snapshot.
const CAPTURE_RESERVE: Duration = Duration::from_millis(2500);

/// Below this a settle spawn is not worth its startup cost — skipped.
const MIN_SETTLE_BUDGET: Duration = Duration::from_secs(1);

/// Settle budget for `open`'s plain path: capped at [`SETTLE_CAP`], must leave
/// [`CAPTURE_RESERVE`] for content capture, skipped (None) when the remainder
/// is too small for both. Pure so tests pin the policy.
fn settle_budget(remaining: Duration) -> Option<Duration> {
    let budget = remaining.saturating_sub(CAPTURE_RESERVE).min(SETTLE_CAP);
    (budget >= MIN_SETTLE_BUDGET).then_some(budget)
}

/// The pure argument checks of `open`, run both by [`preflight`] (before the
/// readiness gate) and by [`open`] itself. The URL must be one this product may
/// navigate to, and the `--expect` selector goes through the shared wait-target
/// policy (so the numeric silent-sleep form is rejected here too) — both checked
/// before navigating, never after. The resolved wait target is the check's own
/// by-product.
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

/// `open` — navigate to `url`, optionally wait for `--expect` (redesign-aware
/// via `--structural`), report the committed final URL and best-effort attach
/// the page content (compact accessibility snapshot). On the plain path (no
/// `--expect`, which already serves as the settle) the navigation is followed
/// by a best-effort network settle, capped at [`SETTLE_CAP`]. `timeout` bounds
/// the waits the operation drives — the error-page probe, the settle /
/// `--expect` wait, and the content capture — NOT the navigation, which runs to
/// the clock mahbot declares to chrome-use: the operation may therefore
/// legitimately outlive `timeout`.
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
    // The operation's own step: neither `timeout` (the budget `--timeout`
    // declared, the CLI's whole-operation default otherwise) nor `total` bounds
    // the NAVIGATION. `--timeout` is no deadline chrome-use honours for `open`'s
    // navigate: chrome-use runs it to the clock mahbot declares to it, and killing it
    // on the operation's budget would end a call chrome-use was still working on
    // — losing chrome-use's own reason for the failure, which is exactly what the
    // product's kill must never do. The navigation's kill therefore rides
    // `kill_bound` above chrome-use's own budget, and the operation may
    // legitimately outlive `--timeout`. `total` is then the operation's own
    // bookkeeping — `timeout` plus the slack reserved for the error-page probe —
    // for that probe, the settle and the content capture; the `--expect` wait
    // declares what remains of `timeout` instead (see below), and every other step
    // of the operation rides [`StepClocks::bounded`].
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
    // One of the operation's best-effort sub-steps (the policy exception in
    // [`crate::chrome`]): the product reserves it up to the 2 s slack so it runs
    // even when the navigation consumed the whole declared deadline, and its kill
    // rides that slack above the reserved bound. Best-effort (inconclusive =
    // pass); skipped when nothing is left. One eval yields both the error-page
    // verdict and the net error token Chrome renders in `div.error-code`
    // (empty until the neterror script runs — the generic message covers it).
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
        // A missed `--expect` after a committed navigation is a suspected
        // redesign when `--structural` was declared, a plain timeout otherwise.
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
        // The wait's declared deadline is what REMAINS of the caller's own
        // --timeout — never the operation's bookkeeping `total`, whose extra
        // slack belongs to the error-page probe. A declaration that crosses
        // chrome-use's own client tolerance is what makes chrome-use run out of
        // tolerance instead of answering, and `parse_forwarded_timeout` already
        // refuses a --timeout at or above it, so the remaining part of one stays
        // below it.
        let wait_budget = timeout.saturating_sub(started.elapsed());
        // No budget left for the wait (or its capture) — emit the same timeout
        // envelope the wait-timeout arm produces, without content. What ran out
        // here is the OPERATION's own budget, spent before the wait step was given
        // any of it, so no deadline was ever declared to chrome-use for that step:
        // the envelope reports the operation's own budget, and says so rather than
        // naming a chrome-use clock that was never given a value.
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
        // The wait forwards its remaining budget as --timeout, which chrome-use
        // honours in full — so that remaining budget IS the step's chrome-side
        // clock — and its failure IS the caller's verdict, so it runs to the
        // forwarded clocks with the relay self-heal allowed: a relay drop during
        // the wait is healed exactly the way it is for any reported step.
        let wait_clocks = StepClocks::forwarded(wait_budget);
        let wargs = wait_args(&target, wait_budget.as_millis());
        let refs: Vec<&str> = wargs.iter().map(String::as_str).collect();
        let waited = spawn_step(&path, &refs, Some(session), wait_clocks, None).await;
        // The page is open regardless of the wait outcome, so the content
        // rides on failure envelopes too: it is exactly what diagnoses a
        // redesign. Same remaining-budget rule as the plain path — slow
        // waits simply exhaust it and the content is omitted.
        let content =
            capture_open_content(&path, session, total.saturating_sub(started.elapsed())).await;
        if let Some(content) = content {
            payload["content"] = json!(content);
        }
        return match waited {
            Ok(_) => out_env("open", true, OutKind::Ok, payload),
            Err(f) if f.kind == OutKind::Timeout => {
                // chrome-use's own verdict reports the deadline it was given; a
                // product kill reports the product's own bound (the kill).
                let reported = if f.message.is_empty() {
                    wait_clocks.call.kill
                } else {
                    wait_clocks.call.chrome_side
                };
                payload["timeout_ms"] = json!(reported.as_millis());
                // The chrome-side timeout message when chrome-use phrased the
                // timeout itself, a factual deadline text on a product kill.
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
    // Best-effort settle: heavy SPAs keep the network busy right after the
    // navigation commits, so the first following step can otherwise start
    // racing a still-settling page. Raw argv — the `wait` action deliberately
    // does not expose `--load`. chrome-use honours no `--timeout` for this form:
    // its clock is the AGENT_BROWSER_DEFAULT_TIMEOUT this step declares (the
    // remaining operation budget), and the product's own bound on the step is the
    // same number — one of the operation's best-effort sub-steps, so its kill
    // rides [`KILL_SLACK`] above that bound and the operation still stays within
    // its budget plus that slack. The result is discarded — a settle timeout
    // never downgrades a committed navigation.
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
    // Content capture is best-effort and bounded: it runs on the budget the
    // navigation (+ error probe + settle) left over, and a failure, exhaustion,
    // or content-free page simply omits `content` — a successful navigation is
    // never downgraded. The step declares the product's clock for a verb it
    // forwards no `--timeout` to ([`CHROME_USE_DECLARED_BUDGET`]) while the
    // product's own bound on it is the remaining operation budget — the
    // operation's last best-effort sub-step, so its kill rides [`KILL_SLACK`]
    // above that bound.
    let mut payload = json!({ "url": final_url });
    let budget = total.saturating_sub(started.elapsed());
    if let Some(content) = capture_open_content(&path, session, budget).await {
        payload["content"] = json!(content);
    }
    out_env("open", true, OutKind::Ok, payload)
}

/// Best-effort compact page snapshot for the `open` envelope — the same
/// content form the interactive chrome tool surfaces after an open. One of the
/// operation's bounded sub-steps ([`StepClocks::bounded`]): skipped
/// under 500 ms of budget (mirroring the error-page probe); a failed step,
/// an unrecognized response shape, or a content-free page yields no content.
/// Non-empty content is byte-truncated so the single-line JSON envelope stays
/// valid: `TOOL_OUTPUT_BUDGET_BYTES` triggers the head/tail sandwich and the
/// omission marker it inserts is additional to that trigger.
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

/// Run the count-eval shim for `selector` — shared by the `count` action and
/// the `extract` honest-empty gate. The eval shim is a step of the action that
/// called it, so it runs to that action's clocks.
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

/// `wait` — bounded wait for a safe [`WaitTarget`] (the numeric sleep form is
/// rejected at parse time). The requested `--timeout` IS chrome-use's deadline
/// (its wait forms honour it in full), so its honest timeout error surfaces at
/// the declared clock; the product's kill rides the relay-recovery window +
/// [`crate::chrome::KILL_SLACK`] above it (see [`StepClocks::forwarded`]).
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

/// Map a parsed expect verdict to the envelope: pass → rc 0; deadline
/// expiration → kind timeout; a plain false (only reachable if a future
/// chrome-use stops folding it into timedOut) → kind error. Pure so tests
/// pin the mapping.
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

/// `expect` — assert a condition with a bounded wait. The envelope (not the
/// exit code) is authoritative: a failed assertion arrives as
/// `success:true` with the verdict in `data`, which [`expect_outcome`]
/// parses (see [`spawn_step`]).
async fn expect(cond: &ExpectCond, timeout: Duration, session: &str) -> OutEnvelope {
    let condition = describe(cond);
    let base = request([("condition", json!(condition))]);
    let path = match require_cli("expect", base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Same rule as wait: the forwarded `--timeout` is chrome-use's own clock,
    // and the product's kill rides the relay-recovery window + slack above it.
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

/// `eval` — run JS and emit the unwrapped result as a JSON value (number,
/// string, object, or null).
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

/// `extract` — schema-driven rows extraction, gated by a count so an empty
/// region is reported honestly (chrome-use's rows-mode `extract` returns
/// phantom rows when the rows selector matches 0).
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
    // Honest-empty gate: when the rows selector matches 0, report empty without
    // invoking chrome-use's phantom-row `extract`.
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

    // `--schema-file` is the installed chrome-use 1.5.101's own documented
    // form (`extract --help`); rows-mode extract answers
    // `data = {extracted: [...], count, origin, meta}` and single-object mode
    // with the bare object; extract_output tolerates a plain array too.
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

/// `click` — click a selector, forwarding `--if-present` verbatim when set
/// (chrome-use itself treats an `--if-present` miss as a no-op success).
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
            // A page the click opens is reported by chrome-use (its reply names the tab); the
            // step's tail takes it out of chrome-use's group ([`chrome_groups`]), which is why the
            // id rides back with the envelope rather than into its result.
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

/// Read this process's stdin to EOF for `fill --stdin`, capped so an
/// oversized/accidental stream is a usage error, not a memory event.
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

/// The fill/type/press success handling, pure for tests: a success envelope
/// carrying chrome-use's degraded-success `warning` ([`chrome_use_warning`]) is
/// classified as kind error (rc 1) with the warning surfaced — the action may not
/// have taken effect, so it must never exit 0. A clean success emits kind ok with
/// the action params plus chrome-use's `data`.
///
/// The browser-replacement note is never one of these: [`classify_step_output`]
/// already turned such an envelope into an Environment failure before it could
/// reach here (see [`self_launched_browser_cli_error`]).
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

/// `fill` — clear + verified fill. Exactly one text source (inline text,
/// --file passthrough, or mahbot-piped --stdin).
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
    // Leading-dash text rides after the `--` shield; action flags must come
    // BEFORE it (everything after is a verbatim value).
    argv.extend(text_value_argv(text));
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(resp) => text_input_ok_envelope(action, &Value::Object(base.clone()), resp),
        Err(f) => f.envelope(action, Value::Object(base), clocks),
    }
}

/// `press` — press a key at the focused element, optionally trying to focus a
/// `--selector` first (a no-op for non-focusable elements) and holding `--hold`
/// ms before release.
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

/// The delivery verdict a failed attach carries: a kill at the step's own clock
/// never learned what reached the page, and a failure chrome-use did report may
/// have streamed part of a file before it gave up — so neither ever claims the
/// files are on the input.
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

/// An attach that never reached the browser, reported against the same payload a
/// successful one carries.
fn upload_refusal(selector: &str, files: &[String], kind: OutKind, error: &str) -> OutEnvelope {
    let mut payload = upload_base(selector, files);
    payload.insert("error".into(), json!(error));
    out_env("upload", false, kind, Value::Object(payload))
}

/// An attach's envelope from chrome-use's answer: the success carries its
/// `uploaded`/`attached` counts, the absolute files that were read, and its
/// `warning` verbatim when it could not confirm the input still holds them — a
/// delivered attach only while something was uploaded, and never a claim about the
/// page's own validation, which nothing here can see. A stated zero upload is the
/// failure envelope: nothing reached the page.
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
    // A stated zero is the opposite of delivery: nothing reached the page, so the
    // attach is a failure whatever the browser's exit said.
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

/// `upload` — attach local files to the page's file input, the label that pairs
/// with it, or the container holding it.
///
/// Paths resolve against the directory this command was run in and are handed to
/// chrome-use as absolute files, which is also what the result reports. A success
/// carries chrome-use's own `uploaded`/`attached` counts and, verbatim, the
/// warning it raises when it could not confirm the input still holds the files —
/// that warning is a degraded success, never this action's failure, and it is
/// never a claim about the page's own validation, which nothing here can see.
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

/// Ask the page the one-file question the files module puts to a several-file
/// attach, and hand back its verdict. The probe is the product's own bounded
/// question, so it runs to [`DEFAULT_STEP_TIMEOUT`] rather than a step's clock.
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

/// A screenshot's reserved file: removed on every path that is not a reported
/// capture — a spawn failure, a corrupt write, and an interrupted call, whose
/// future is dropped mid-step. [`ReservedPicture::commit`] is what hands the file
/// to the caller.
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

/// `screenshot` — capture the visible page into a produced PNG.
///
/// The destination is the caller's own directory ([`crate::cli::resolve_output_dir`]):
/// it is resolved here, where the command ran, and the picture is written under a
/// name nothing occupies yet — a produced file never overwrites one. The result
/// reports the absolute path that was written, in the platform's own spelling
/// (never the verbatim form), so the caller can open it as it is.
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
    // The claim IS the collision check, and chrome-use writes over the reserved
    // file it is handed, so a name already present can never be taken.
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
    // The reservation is removed on every path that does not become a reported
    // capture — including a call interrupted mid-step, whose future is dropped here.
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
            // chrome-use warns when the pixels did not come from the page the
            // caller thinks it captured; the file is still written, so the capture
            // stands and the warning is repeated rather than swallowed.
            if let Some(warning) = chrome_use_warning(&resp) {
                payload.insert("warning".into(), warning);
            }
            picture.commit();
            out_env(action, true, OutKind::Ok, Value::Object(payload))
        }
        Err(f) => f.envelope(action, json!({}), clocks),
    }
}

/// `session stop` — name-gated stop of a CLI session (protects the interactive
/// tool's `agent-tab-*` sessions unless `--force`). The envelope echoes the name the
/// owner wrote ([`cli_sessions::display_name`]), which is also what `session list` shows.
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
    // A product probe (the probe exception in [`StepClocks::probe`], see the
    // policy in [`crate::chrome`]), not agent work: bounded by the product itself
    // with chrome-use's relay self-heal suppressed. The bound is the shared
    // [`SESSION_STOP_TIMEOUT`] — a product bound, not a declared deadline
    // (chrome-use forwards no `--timeout` to `session stop`).
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

/// Whether chrome-use is running this session right now: the same daemon-free
/// `session list` read `session status` starts with. `false` when the list cannot be
/// read — an unknown answer must not turn a creating call into a refusal.
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

/// `session list` — the sessions this CLI created: name, keep mark, and how long
/// since each was last used. The listing itself reads the local record only, never
/// the browser.
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

/// `session status <name>` — opt-in liveness probe for a named session. A
/// daemon-free `session list` preflight (never creates a session or spawns a
/// daemon) reports a stopped session as `empty` (rc 0); a listed session is
/// probed with a real bounded `get url`. Any timeout on that probe is wedge
/// evidence (the probe reads the current URL, it never navigates), so it
/// re-classifies as Environment (rc 2), says plainly that the session stopped
/// answering, and names the recovery. This verb is a diagnostic and stops
/// nothing; the paths that ACT on the browser — the action verbs and the watchdog
/// — recover such a session themselves
/// ([`recover_unresponsive_session`]), so its text also says the next action
/// does it automatically. An honest non-wedge cause passes through unnamed.
async fn session_status(name: &str) -> OutEnvelope {
    let target = resolve_session_target(name);
    // What the envelopes echo: the name the owner wrote, as `session list` shows it.
    let shown = cli_sessions::display_name(&target);
    let path = match require_cli("session", request([("session", json!(shown))])) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // A diagnostic is still a later call with the same name: it restarts the lifetime,
    // and it must spawn the session's daemon with the same cleanup backstop its mark asks
    // for. A session this CLI did not create is left to chrome-use's own defaults.
    if let Some(kept) = cli_sessions::touch(&target) {
        set_group_cleanup_backstop(cli_sessions::backstop_ms(kept));
    }
    // Both steps are the product's own probes — the probe exception: bounded by
    // the product, with chrome-use's relay self-heal suppressed
    // ([`StepClocks::probe`]).
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
            // Any timeout on the probe is wedge evidence (it reads the
            // current URL, it never navigates) — re-classify as Environment
            // with chrome-use's own signature phrasing, exactly like its
            // session-unresponsive classification. The probe deliberately does
            // NOT auto-recover (it is the diagnostic an owner runs to see the
            // state), so its text names the manual flow instead — the one place
            // the automatic recovery never ran. Other failures pass through with
            // their honest cause (not every failure is a wedge).
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
            // A protected-namespace wedge needs `--force` to stop — make the
            // recovery flow directly actionable for that edge. Only wedges
            // get the note (an honest non-wedge cause must not carry stop
            // guidance), and only protected ones (an ordinary session's stop
            // needs no --force).
            append_protected_stop_note(&mut f.message, &target);
            f.envelope("session", json!({ "session": shown }), probe)
        }
    }
}

/// Append the protected-namespace `--force` note to a probe-failure message
/// that reads as a session wedge on a protected (`agent-tab-*` /
/// `link-enricher-*`) target: the recovery flow names
/// `session stop <name>`, which is only actionable with `--force` here. Pure
/// so tests pin both gates (protected membership, wedge-shaped message).
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

/// Tolerance-first membership check over a `session list` envelope: only an
/// explicit failure verdict rules the target out as a hard miss; a payload
/// with no verdict key still gets its names extracted. Handles both chrome-use
/// shapes — latest `{"ok":true,"sessions":[{"name":..},..]}` (the top-level
/// `sessions` key lands in `extra`) and legacy `{"data":{"sessions":["name",..]}}`.
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

/// Best-effort close of an ephemeral session AFTER the action envelope is
/// emitted — never changes the action's exit code; its diagnostics, including
/// the report of a helper it leaves behind, go to stderr. A successful close
/// forgets the session's record; a failed one leaves it for the lifetime sweep.
async fn close_ephemeral(name: &str) {
    // Every line below is read by a person, so the session is named the way its owner
    // would name it, not by the record's own key.
    let display = cli_sessions::display_name(name);
    let Some(path) = cli_path() else {
        eprintln!(
            "mahbot chrome: could not close ephemeral session '{display}' (chrome-use CLI not found)"
        );
        return;
    };
    // A lifecycle close, not agent work: the shared product probe
    // ([`run_cli_bounded_at`]) bounds it with its own product clock and suppresses
    // the relay self-heal, exactly like the CLI's other probes.
    match run_cli_bounded_at(
        &path,
        &["session", "stop", name],
        None,
        SESSION_STOP_TIMEOUT,
    )
    .await
    {
        CliRun::Output(out) => {
            // A helper the close itself leaves holding the call's output channel
            // is recorded here for the caller ([`run_cli`]) to report — the
            // action envelope went out before this ran, so it cannot carry it.
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

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    // The clock arithmetic has one home; tests here derive their expectations
    // from it instead of repeating its literals.
    use crate::chrome::kill_bound;

    /// Command-line arguments for a test.
    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().copied().map(String::from).collect()
    }

    /// [`parse_invocation`] for a test that only cares about the action (the keep
    /// request has its own test).
    fn parse_action(args: &[String]) -> Result<Invocation, String> {
        parse_invocation(args).map(|(invocation, _)| invocation)
    }

    /// The rows `help` prints under `Flags:` — `(spelling, description)`, in order.
    /// An action whose help has no flags block yields no rows.
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

    /// The `Usage:` line of a rendered action help, without its
    /// `mahbot chrome <action>` prefix.
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

    /// The action words the top-level help advertises, in printed order.
    ///
    /// An action's line is indented two spaces; its purpose line, four more, so
    /// only the shallower lines name actions.
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

    /// The session-wide pair — handled outside the common flag path
    /// (`extract_global_flags`), so it has its own guard below rather than a shape
    /// probe.
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

    #[test]
    fn resolve_session_named_prefixing_is_idempotent() {
        let (name, ephemeral) = resolve_session(Some("foo"));
        assert_eq!(name, "mahbot-chrome-foo");
        assert!(!ephemeral);

        let (name, ephemeral) = resolve_session(Some("mahbot-chrome-foo"));
        assert_eq!(name, "mahbot-chrome-foo");
        assert!(!ephemeral);
    }

    #[test]
    fn resolve_session_ephemeral_uses_cli_prefix() {
        let (name, ephemeral) = resolve_session(None);
        assert!(name.starts_with(CLI_EPHEMERAL_PREFIX));
        assert!(ephemeral);
    }

    /// The alphabet is chrome-use's own for `session stop`
    /// (`validation::is_valid_session_name`: alphanumerics, '-' and '_'). A name
    /// with a '.' parses and opens fine, but chrome-use then refuses to stop it,
    /// so the session's daemon and tabs stay behind and neither the hand recipe
    /// nor the automatic recovery can clear it — the name has to be refused here,
    /// where the caller can still choose another one.
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
        // The arithmetic has one home ([`crate::chrome::clocks`] /
        // [`crate::chrome::kill_bound`], pinned there); these assertions are about
        // which shape the CLI takes, so they derive their expectations from it
        // rather than repeating its literals.
        let chrome_own = clocks(None, CliRecovery::Allowed);

        // The operation's own step (`open`'s navigation): chrome-use runs it to
        // its own client tolerance, and the operation's budget does NOT bound
        // the kill — a navigation chrome-use is still working on must be free to
        // report its own reason.
        let navigation = StepClocks::operation();
        assert_eq!(navigation.call.chrome_side, CHROME_USE_DECLARED_BUDGET);
        assert_eq!(
            navigation.call.kill,
            kill_bound(CHROME_USE_DECLARED_BUDGET, CliRecovery::Allowed)
        );

        // A verb chrome-use accepts no `--timeout` for: with no user bound the kill
        // is the one the declared clock derives, and a user bound IS the bound the
        // call really runs to — the parse refuses anything below the declared clock
        // (pinned at that layer), so the kill is never shorter than the declaration.
        let no_bound = StepClocks::tool_clock(None);
        assert_eq!(no_bound.call.chrome_side, CHROME_USE_DECLARED_BUDGET);
        assert_eq!(no_bound.call.kill, chrome_own.kill);
        let at_the_clock = StepClocks::tool_clock(Some(CHROME_USE_DECLARED_BUDGET));
        assert_eq!(at_the_clock.call.kill, chrome_own.kill);

        // A LONGER one (`eval --timeout 300`) is a bound the product must really
        // let the call run to, so it widens the kill — and the kill alone: the
        // clock DECLARED to chrome-use stays one kill margin under its own client
        // tolerance, which nothing can raise without making chrome-use fail as a
        // wedged session. The product's own kill is what the envelope reports as
        // `timeout_ms`.
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

        // A bounded step of the operation: a probe in everything but name — it
        // suppresses the relay self-heal and its kill rests [`KILL_SLACK`] above
        // the product's own bound, never chrome-use's clock.
        let probe = StepClocks::bounded(CHROME_USE_DECLARED_BUDGET, Duration::from_secs(2));
        assert_eq!(probe.call.kill, Duration::from_secs(2) + KILL_SLACK);
        assert!(matches!(probe.recovery, CliRecovery::Suppressed));
        let settle = StepClocks::bounded(Duration::from_secs(6), Duration::from_secs(6));
        assert_eq!(settle.call.chrome_side, Duration::from_secs(6));
        assert_eq!(settle.call.kill, Duration::from_secs(6) + KILL_SLACK);
        // A bound above chrome-use's own clock is still the PRODUCT's bound: the
        // step is cut off at it, not at chrome-use's clock.
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
        // A bare name is prefixed like `--session`; both namespace names pass
        // through as-is (status may probe protected sessions read-only).
        assert_eq!(resolve_session_target("docs"), "mahbot-chrome-docs");
        assert_eq!(
            resolve_session_target("mahbot-chrome-foo"),
            "mahbot-chrome-foo"
        );
        assert_eq!(resolve_session_target("agent-tab-1"), "agent-tab-1");
        assert_eq!(resolve_session_target("link-enricher-7"), "link-enricher-7");
    }

    #[test]
    fn session_status_parses_and_rejects() {
        // status parses into SessionStatus.
        let inv = parse_action(&["session".into(), "status".into(), "docs".into()])
            .expect("session status parses");
        match inv.action {
            Action::SessionStatus { name } => assert_eq!(name, "docs"),
            _ => panic!("expected SessionStatus"),
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
        assert_eq!(err, "--session is not valid for session stop/status/list");

        // `--force` is `stop`'s flag alone (the declaration the parser reads): the other
        // subcommands accept no flags.
        let allowed: Vec<&CliFlag> = declared_flags("session").iter().collect();
        let err = parse_session(
            &["status".into(), "docs".into(), "--force".into()],
            &allowed,
        )
        .err()
        .expect("--force must be rejected for session status");
        assert_eq!(err, "unknown flag --force");

        // Extra positionals are rejected for status.
        assert!(parse_session(&["status".into(), "a".into(), "b".into()], &allowed).is_err());

        // An unknown sub names both verbs.
        let err = parse_action(&["session".into(), "destroy".into(), "docs".into()])
            .err()
            .expect("unknown sub must be rejected");
        assert!(
            err.contains(
                "unknown session subcommand 'destroy' (expected 'stop', 'status' or 'list')"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn session_listed_handles_both_shapes() {
        // Latest shape: the top-level `sessions` array of objects (lands in
        // `extra`), listed by name.
        let latest: ChromeResponse = serde_json::from_value(json!({
            "ok": true, "sessions": [{ "name": "mahbot-chrome-docs" }]
        }))
        .expect("deserialize latest shape");
        assert!(session_listed(&latest, "mahbot-chrome-docs"));
        assert!(!session_listed(&latest, "mahbot-chrome-other"));

        // Legacy shape: `data.sessions` array of strings.
        let legacy: ChromeResponse = serde_json::from_value(json!({
            "success": true, "data": { "sessions": ["mahbot-chrome-docs"] }
        }))
        .expect("deserialize legacy shape");
        assert!(session_listed(&legacy, "mahbot-chrome-docs"));

        // An explicit failure verdict rules the target out.
        let failed: ChromeResponse =
            serde_json::from_value(json!({ "success": false })).expect("deserialize failure");
        assert!(!session_listed(&failed, "mahbot-chrome-docs"));

        // No sessions key at all → not listed.
        let no_sessions: ChromeResponse =
            serde_json::from_value(json!({ "success": true, "data": {} }))
                .expect("deserialize empty payload");
        assert!(!session_listed(&no_sessions, "mahbot-chrome-docs"));
    }

    #[test]
    fn protected_stop_note_targets_only_protected_wedges() {
        // A wedge-shaped failure on a protected target gets the note — with
        // the full actionable command, so no stitching with the shared hint.
        let mut msg = "session unresponsive: no response within 45s".to_string();
        append_protected_stop_note(&mut msg, "agent-tab-1");
        assert!(
            msg.ends_with("(protected namespace: stopping 'agent-tab-1' requires `session stop agent-tab-1 --force`)"),
            "note missing: {msg}"
        );

        // An ordinary session's stop needs no --force — no note.
        let mut msg = "session unresponsive: no response within 45s".to_string();
        append_protected_stop_note(&mut msg, "mahbot-chrome-docs");
        assert_eq!(msg, "session unresponsive: no response within 45s");

        // A non-wedge failure on a protected target keeps its honest cause.
        let mut msg = "element not found".to_string();
        append_protected_stop_note(&mut msg, "agent-tab-1");
        assert_eq!(msg, "element not found");
    }

    #[test]
    fn envelope_layers_accurate_remediation() {
        // The envelope itself does not carry the session-wedge text — that is
        // applied after dispatch, where the session name is known (see
        // `recover_session_wedge`). A wedge message passes through untouched.
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "session unresponsive: no response within 45s".to_string(),
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            StepClocks::probe(SESSION_STOP_TIMEOUT),
        );
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some("session unresponsive: no response within 45s")
        );

        // A non-wedge message is returned untouched too.
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "relay isn't connected".to_string(),
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            StepClocks::probe(SESSION_STOP_TIMEOUT),
        );
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some("relay isn't connected")
        );

        // An orphaned-tab error still picks up its unreachable-tab guidance here:
        // it is the one remediation the envelope owns, because it is the one state
        // nothing recovers (the tab itself is the product's to close, never the
        // agent's to touch).
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "the tab this session was driving can no longer be resolved".to_string(),
        }
        .envelope(
            "open",
            json!({ "url": "https://x" }),
            StepClocks::operation(),
        );
        let err = env
            .payload
            .get("error")
            .and_then(Value::as_str)
            .expect("error present");
        assert!(
            err.contains("the product closes a session's tabs itself"),
            "got: {err}"
        );

        // A Timeout deadline kill of a product probe reports the probe's own
        // bound, and says it is a product-side bound rather than chrome-use's
        // verdict.
        let clocks = StepClocks::probe(SESSION_PROBE_TIMEOUT);
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            clocks,
        );
        let expected = deadline_error(
            clocks.call.chrome_side,
            clocks.call.kill,
            "the step did not complete",
        );
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some(expected.as_str())
        );
        assert_eq!(env.payload["timeout_ms"], 20_000);
        assert!(expected.contains("the product's own bound for the step"));
        assert!(!expected.contains("rode above"));
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
        // chrome-use's own session-unresponsive classification (Environment) is
        // the one wedge signal that needs no confirmation: chrome-use diagnosed
        // it rather than being cut off.
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

        // A Timeout chrome-use itself reported — what a failed `expect` /
        // `open --expect` / `wait` produces normally — is NOT a wedge: stopping
        // that session would only lose its tabs.
        let verdict = env(OutKind::Timeout, "Wait timed out after 15000ms");
        assert_eq!(wedge_action(&verdict, true, false), WedgeAction::Leave);

        // A Timeout the PRODUCT's own clock ended is only a candidate: chrome-use
        // was cut off before it could classify anything, so the session's
        // liveness is unknown and the bounded probe has to confirm it.
        let killed = env(OutKind::Timeout, "deadline reached after 92000ms");
        assert_eq!(wedge_action(&killed, true, true), WedgeAction::Probe);

        // `open --expect --structural` labels a deadline expiration Redesign
        // instead of Timeout, so a product-clock ending keys the same way.
        let redesign = env(
            OutKind::Redesign,
            "possible DOM redesign or structural change",
        );
        assert_eq!(wedge_action(&redesign, true, true), WedgeAction::Probe);
        assert_eq!(wedge_action(&redesign, true, false), WedgeAction::Leave);

        // Other kinds and other Environment messages are no evidence either, and
        // an ephemeral (unnamed for recovery) session is never touched.
        let relay = env(OutKind::Environment, "relay isn't connected");
        assert_eq!(wedge_action(&relay, true, true), WedgeAction::Leave);
        let missing = env(OutKind::Error, "element not found");
        assert_eq!(wedge_action(&missing, true, false), WedgeAction::Leave);
        let net = env(OutKind::Network, "net::ERR_CONNECTION_REFUSED");
        assert_eq!(wedge_action(&net, true, true), WedgeAction::Leave);
        assert_eq!(wedge_action(&killed, false, true), WedgeAction::Leave);
    }

    /// The end of the keying: an envelope the action left for
    /// [`WedgeAction::Leave`] reaches the caller untouched — no session stop, no
    /// annotation.
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

    #[test]
    fn wedge_recovery_note_reports_what_was_done_or_the_manual_flow() {
        // The automatic recovery stopped the session: its own summary, and no
        // manual flow.
        let stopped = wedge_recovery_note(SessionRecovery::Stopped);
        assert_eq!(stopped, SessionRecovery::Stopped.summary());
        assert!(!stopped.contains("recover with"), "got: {stopped}");
        // The stop was issued but never confirmed, or could not be started: the
        // summary itself carries that, and the manual flow is named — the only
        // place it is.
        for recovery in [SessionRecovery::Unanswered, SessionRecovery::NotStarted] {
            let note = wedge_recovery_note(recovery);
            assert!(note.starts_with(recovery.summary()), "got: {note}");
            assert!(note.contains(SESSION_RECOVERY_FLOW), "got: {note}");
            assert!(!note.contains("could not run"), "got: {note}");
        }
    }

    #[test]
    fn append_error_note_extends_the_error_text_only() {
        let timeout_error = "deadline reached after 92000ms — the step did not complete";
        let mut env = out_env(
            "open",
            false,
            OutKind::Timeout,
            json!({ "url": "https://x", "error": timeout_error }),
        );
        append_error_note(&mut env, "what was done");
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some(format!("{timeout_error} — what was done").as_str())
        );
        // Other payload keys and an error-less payload are untouched.
        let mut env = out_env("open", false, OutKind::Ok, json!({ "url": "https://x" }));
        append_error_note(&mut env, "what was done");
        assert!(env.payload.get("error").is_none());
    }

    /// Direct pin of the spawn-level classification, including the crux of the
    /// expect contract: a failed assertion arrives as `success:true` with
    /// exit 1 and the verdict in `data`.
    #[test]
    fn classify_step_output_pins_envelope_over_exit_code() {
        let envelope = |body: Value| serde_json::to_vec(&body).expect("serialize envelope");
        let outcome = |r: &StepOutcome| match r {
            Ok(resp) => Ok(resp.data.clone()),
            Err(f) => Err((f.kind, f.message.clone())),
        };

        // expect pass=false: success:true + exit 1 → the verdict is returned.
        let resp = classify_step_output(
            true,
            false,
            Some(1),
            false,
            &envelope(
                json!({"success": true, "data": {"pass": false, "actual": 3, "timedOut": true}}),
            ),
            "",
        )
        .expect("expect verdict must survive the non-zero exit");
        assert_eq!(
            resp.data
                .as_ref()
                .and_then(expect_outcome)
                .map(|o| (o.pass, o.timed_out, o.actual)),
            Some((false, true, Some(json!(3))))
        );

        // …but only for expect: the same payload on another action is a
        // generic failure (fallback message, no envelope error).
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            &envelope(json!({"success": true, "data": {"pass": false}})),
            "",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Error, "chrome-use exited with code 1".into()))
        );

        // expect un-evaluable (no browser): success:false + exit 2 → environment.
        let r = classify_step_output(
            true,
            false,
            Some(2),
            false,
            &envelope(
                json!({"success": false, "error": "Browser not launched", "code": "browser_not_launched"}),
            ),
            "",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Environment, "Browser not launched".into()))
        );

        // Honest timeout classification on a failed wait.
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            &envelope(json!({"success": false, "error": "Wait timed out after 15000ms"})),
            "",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Timeout, "Wait timed out after 15000ms".into()))
        );

        // Happy path: exit 0 + success envelope.
        assert!(
            outcome(&classify_step_output(
                false,
                false,
                Some(0),
                true,
                &envelope(json!({"success": true, "data": {}})),
                ""
            ))
            .is_ok()
        );

        // Unparseable stdout: zero exit → non-JSON error; non-zero → stderr fallback.
        let r = classify_step_output(false, false, Some(0), true, b"garbage", "");
        assert_eq!(
            outcome(&r),
            Err((OutKind::Error, "chrome-use returned non-JSON output".into()))
        );
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            b"garbage",
            "relay is not connected",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Environment, "relay is not connected".into()))
        );
    }

    /// The canned stale-relay hint is stripped from Timeout-classified
    /// failures only — envelope `error` text and the stderr fallback alike —
    /// while non-Timeout messages keep it verbatim.
    #[test]
    fn classify_step_output_strips_canned_relay_hint() {
        let envelope = |body: Value| serde_json::to_vec(&body).expect("serialize envelope");
        let outcome = |r: &StepOutcome| match r {
            Ok(resp) => Ok(resp.data.clone()),
            Err(f) => Err((f.kind, f.message.clone())),
        };
        let hint = "Hint: the session's browser connection is unresponsive (likely a stale \
                    relay/service-worker mid-session). Reconnect with `connect`, or close the \
                    session and reopen it.";

        // Timeout envelope: hint stripped, honest prefix kept.
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            &envelope(
                json!({"success": false, "error": format!("Wait timed out after 15000ms. {hint}")}),
            ),
            "",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Timeout, "Wait timed out after 15000ms".into()))
        );

        // A non-timeout envelope carrying the hint keeps it verbatim (kind Error,
        // not stripped — only Timeout-classified messages are rewritten).
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            &envelope(json!({"success": false, "error": format!("some other failure. {hint}")})),
            "",
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Error, format!("some other failure. {hint}")))
        );

        // The stderr fallback is classified and sanitized too: a timeout
        // phrasing with the canned hint in stderr → hint stripped.
        let r = classify_step_output(
            false,
            false,
            Some(1),
            false,
            b"garbage",
            &format!("Wait timed out after 15000ms. {hint}"),
        );
        assert_eq!(
            outcome(&r),
            Err((OutKind::Timeout, "Wait timed out after 15000ms".into()))
        );
    }

    /// A successful envelope carrying chrome-use's browser-replacement note is a
    /// failure, never a quiet success — and it is deliberately NOT the
    /// daemon-down path (the daemon is fine; the connection to the real browser
    /// was lost). The benign degraded-success warnings keep today's behaviour.
    #[test]
    fn classify_step_output_fails_a_self_launched_browser() {
        let envelope = |body: Value| serde_json::to_vec(&body).expect("serialize envelope");
        let note = "This session's previous browser is gone (its browser connection was dead) \
                    and a fresh one was launched for this command";
        let r = classify_step_output(
            false,
            false,
            Some(0),
            true,
            &envelope(json!({"success": true, "warning": note})),
            "",
        );
        let Err(f) = r else {
            panic!("a browser chrome-use started itself must not classify as success")
        };
        assert_eq!(f.kind, OutKind::Environment);
        assert_eq!(f.message, self_launched_browser_cli_error(note));

        // The expect-style success path (envelope trusted over a non-zero exit)
        // applies the same rule.
        let r = classify_step_output(
            true,
            false,
            Some(1),
            false,
            &envelope(json!({"success": true, "warning": note, "data": {"pass": true}})),
            "",
        );
        let Err(f) = r else {
            panic!("a browser chrome-use started itself must not classify as success")
        };
        assert_eq!(f.kind, OutKind::Environment);

        // A degraded-success warning is NOT browser replacement: the step still
        // succeeds and the action-specific envelope judges it.
        let r = classify_step_output(
            false,
            false,
            Some(0),
            true,
            &envelope(
                json!({"success": true, "warning": "no key listeners on the focused element"}),
            ),
            "",
        );
        assert!(r.is_ok());
    }

    /// An unparseable answer the product itself cut off names the truncation,
    /// while a genuinely malformed one keeps the generic text — a product
    /// truncation must never read as chrome-use's malformed output, on a
    /// non-zero exit too, where the stderr fallback would otherwise claim
    /// chrome-use's own failure.
    #[test]
    fn classify_step_output_names_a_truncated_answer() {
        let outcome = |r: &StepOutcome| match r {
            Ok(resp) => Ok(resp.data.clone()),
            Err(f) => Err((f.kind, f.message.clone())),
        };
        let r = classify_step_output(false, true, Some(0), true, b"garbage", "");
        assert_eq!(outcome(&r), Err((OutKind::Error, truncated_output_error())));
        let r = classify_step_output(false, false, Some(0), true, b"garbage", "");
        assert_eq!(
            outcome(&r),
            Err((OutKind::Error, "chrome-use returned non-JSON output".into()))
        );
        // A non-zero exit with an unparsed, truncated answer names the
        // truncation too, instead of the stderr fallback.
        let r = classify_step_output(false, true, Some(1), false, b"garbage", "some stderr");
        assert_eq!(outcome(&r), Err((OutKind::Error, truncated_output_error())));
        let r = classify_step_output(false, false, Some(1), false, b"garbage", "some stderr");
        assert_eq!(outcome(&r), Err((OutKind::Error, "some stderr".into())));
    }

    #[test]
    fn network_failure_error_names_code_or_falls_back() {
        assert_eq!(
            network_failure_error(Some("ERR_NAME_NOT_RESOLVED")),
            "Chrome rendered an error page — ERR_NAME_NOT_RESOLVED (DNS resolution failure)"
        );
        assert_eq!(
            network_failure_error(None),
            "Chrome rendered an error page (site unreachable or refused)"
        );
    }

    #[test]
    fn step_failure_envelope_reports_timeout_vs_error() {
        // A declared (forwarded) deadline kill: the envelope reports the
        // product's own bound for the step (its kill) as `timeout_ms`, and the
        // error names that bound and the clock chrome-use was working to.
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

        // A Network failure with no recognizable net error token gets no
        // error_code.
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

    /// The `--timeout` of a verb that DECLARES its value to chrome-use
    /// (`wait`/`expect`, and `open` through its `--expect` wait) is capped below
    /// chrome-use's own client tolerance ([`CHROME_USE_OWN_BUDGET`]). A deadline at
    /// or above the tolerance makes chrome-use run OUT of tolerance instead of
    /// answering, so its session-unresponsive classification — which stops the
    /// session and loses its open tabs — replaces the honest "condition was not
    /// met". The rule is enforced at parse time: a too-long deadline is refused with
    /// a usage error instead of accepted and then silently mis-reported.
    ///
    /// A verb that forwards nothing is refused the OTHER way: chrome-use takes no
    /// per-call deadline for it, so nothing can honour a bound below the clock
    /// mahbot declares to chrome-use — such a request is refused rather than
    /// accepted and discarded, while a bound at or above it (`eval --timeout 300`)
    /// is one the call really runs to.
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
            // The tolerance refusal itself, not another rejection of the line: it
            // names both the value asked for and the tolerance it crossed.
            assert!(
                err.contains("45") && err.contains("client tolerance"),
                "{argv:?} must be refused on the value: {err}"
            );
        }

        // A verb chrome-use takes no per-call deadline for: a bound below the clock
        // mahbot declares to chrome-use cannot be honoured, so it is refused — with
        // the value the caller asked for and the shortest one that IS honoured.
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

    #[expect(clippy::too_many_lines)]
    #[test]
    fn parse_happy_paths_support_both_flag_forms() {
        // status takes no args.
        assert!(matches!(
            parse_action(&["status".into()])
                .expect("status parses")
                .action,
            Action::Status
        ));

        // open with `--flag value`.
        let inv = parse_action(&[
            "open".into(),
            "https://example.com".into(),
            "--expect".into(),
            ".btn".into(),
            "--structural".into(),
            "--timeout".into(),
            "5".into(),
        ])
        .expect("open parses");
        match inv.action {
            Action::Open {
                url,
                expect,
                structural,
                timeout,
            } => {
                assert_eq!(url, "https://example.com");
                assert_eq!(expect.as_deref(), Some(".btn"));
                assert!(structural);
                assert_eq!(timeout, Duration::from_secs(5));
            }
            _ => panic!("expected Open"),
        }

        // open with `--flag=value`.
        let inv = parse_action(&[
            "open".into(),
            "https://example.com".into(),
            "--expect=.btn".into(),
            "--timeout=3".into(),
        ])
        .expect("open parses");
        match inv.action {
            Action::Open {
                expect, timeout, ..
            } => {
                assert_eq!(expect.as_deref(), Some(".btn"));
                assert_eq!(timeout, Duration::from_secs(3));
            }
            _ => panic!("expected Open"),
        }

        // open without --timeout defaults to the whole-operation budget (20 s);
        // the navigation itself runs to the clock mahbot declares to chrome-use.
        let inv =
            parse_action(&["open".into(), "https://example.com".into()]).expect("open parses");
        match inv.action {
            Action::Open { timeout, .. } => assert_eq!(timeout, DEFAULT_OPEN_TIMEOUT),
            _ => panic!("expected Open"),
        }

        // count / eval: a user `--timeout` is their own bound on the step (no
        // default), and only a value at or above the clock mahbot declares to
        // chrome-use parses — anything shorter is refused (see
        // `timeout_flag_is_capped_below_chrome_use_client_tolerance`), while
        // wait/expect forward their value as chrome-use's clock.
        for (action, sel) in [("count", ".x"), ("eval", "1+1")] {
            let inv = parse_action(&[action.into(), sel.into(), "--timeout=60".into()])
                .expect("action parses");
            match inv.action {
                Action::Count { bound, .. } | Action::Eval { bound, .. } => {
                    assert_eq!(bound, Some(Duration::from_secs(60)));
                }
                _ => panic!("expected a bounded action"),
            }
        }
        // Without --timeout a forwarding-less verb carries no bound at all:
        // the clock mahbot declares to chrome-use is its clock.
        let inv = parse_action(&["count".into(), ".x".into()]).expect("count parses");
        assert!(matches!(inv.action, Action::Count { bound: None, .. }));
        let inv = parse_action(&["eval".into(), "1+1".into()]).expect("eval parses");
        assert!(matches!(inv.action, Action::Eval { bound: None, .. }));

        // wait / expect default to the condition deadline they declare AND
        // forward as their `--timeout`, well under chrome-use's own client
        // tolerance so the tool's own verdict always fires first.
        let inv = parse_action(&["wait".into(), ".x".into()]).expect("wait parses");
        match inv.action {
            Action::Wait { timeout, .. } => assert_eq!(timeout, DEFAULT_STEP_TIMEOUT),
            _ => panic!("expected Wait"),
        }

        // wait with --url / --text targets.
        let inv = parse_action(&[
            "wait".into(),
            "--url".into(),
            "dashboard".into(),
            "--timeout=9".into(),
        ])
        .expect("wait url parses");
        match inv.action {
            Action::Wait { target, timeout } => {
                assert!(matches!(target, WaitTarget::Url(u) if u == "dashboard"));
                assert_eq!(timeout, Duration::from_secs(9));
            }
            _ => panic!("expected Wait"),
        }
        let inv = parse_action(&["wait".into(), "--text".into(), "Loaded".into()])
            .expect("wait text parses");
        match inv.action {
            Action::Wait { target, timeout } => {
                assert!(matches!(target, WaitTarget::Text(t) if t == "Loaded"));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
            }
            _ => panic!("expected Wait"),
        }

        // expect conditions via the forms grammar.
        let inv = parse_action(&["expect".into(), "#main".into(), "visible".into()])
            .expect("expect state parses");
        match inv.action {
            Action::Expect { cond, timeout } => {
                assert!(matches!(
                    cond,
                    ExpectCond::State { selector, state } if selector == "#main" && state == "visible"
                ));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
            }
            _ => panic!("expected Expect"),
        }
        let inv = parse_action(&[
            "expect".into(),
            "count".into(),
            ".card".into(),
            ">=".into(),
            "3".into(),
            "--timeout=5".into(),
        ])
        .expect("expect count parses");
        match inv.action {
            Action::Expect { cond, timeout } => {
                assert!(matches!(
                    cond,
                    ExpectCond::Count { selector, op, n } if selector == ".card" && op == ">=" && n == 3
                ));
                assert_eq!(timeout, Duration::from_secs(5));
            }
            _ => panic!("expected Expect"),
        }
        let inv = parse_action(&[
            "expect".into(),
            "text".into(),
            "h1".into(),
            "contains".into(),
            "Hello".into(),
            "World".into(),
        ])
        .expect("expect text parses");
        match inv.action {
            Action::Expect { cond, timeout } => {
                assert!(matches!(
                    cond,
                    ExpectCond::Text { selector, predicate, value }
                        if selector == "h1" && predicate == "contains" && value == "Hello World"
                ));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
            }
            _ => panic!("expected Expect"),
        }
        let inv = parse_action(&[
            "expect".into(),
            "url".into(),
            "contains".into(),
            "dashboard".into(),
        ])
        .expect("expect url parses");
        match inv.action {
            Action::Expect { cond, timeout } => {
                assert!(matches!(
                    cond,
                    ExpectCond::Url { predicate, pattern }
                        if predicate == "contains" && pattern == "dashboard"
                ));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
            }
            _ => panic!("expected Expect"),
        }

        // extract requires --schema-file.
        let inv = parse_action(&[
            "extract".into(),
            "--schema-file=rows.json".into(),
            "--limit=3".into(),
        ])
        .expect("extract parses");
        match inv.action {
            Action::Extract {
                schema_file,
                limit,
                bound,
            } => {
                assert_eq!(schema_file, "rows.json");
                assert_eq!(limit, Some(3));
                assert_eq!(bound, None);
            }
            _ => panic!("expected Extract"),
        }

        // click with --if-present.
        let inv = parse_action(&["click".into(), ".btn".into(), "--if-present".into()])
            .expect("click parses");
        match inv.action {
            Action::Click {
                selector,
                if_present,
                ..
            } => {
                assert_eq!(selector, ".btn");
                assert!(if_present);
            }
            _ => panic!("expected Click"),
        }

        // global --session anywhere, including before the action.
        let inv = parse_action(&[
            "--session".into(),
            "run-1".into(),
            "count".into(),
            ".x".into(),
        ])
        .expect("session parses");
        assert_eq!(inv.session.as_deref(), Some("run-1"));
        assert!(matches!(inv.action, Action::Count { .. }));

        // session stop with --force.
        let inv = parse_action(&[
            "session".into(),
            "stop".into(),
            "foo".into(),
            "--force".into(),
        ])
        .expect("session stop parses");
        match inv.action {
            Action::SessionStop { name, force } => {
                assert_eq!(name, "foo");
                assert!(force);
            }
            _ => panic!("expected SessionStop"),
        }
    }

    #[test]
    fn parse_text_input_happy_paths() {
        // fill inline multi-word text.
        let inv = parse_action(&["fill".into(), "#q".into(), "hello".into(), "world".into()])
            .expect("fill inline parses");
        match inv.action {
            Action::Fill {
                selector, source, ..
            } => {
                assert_eq!(selector, "#q");
                assert!(matches!(source, TextInput::Inline(t) if t == "hello world"));
            }
            _ => panic!("expected Fill"),
        }

        // fill with `--file=<path>`.
        let inv = parse_action(&["fill".into(), "#q".into(), "--file=post.md".into()])
            .expect("fill --file parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::File(p) if p == "post.md"));
            }
            _ => panic!("expected Fill"),
        }

        // fill with a separate `--file <path>` value.
        let inv = parse_action(&[
            "fill".into(),
            "#q".into(),
            "--file".into(),
            "post.md".into(),
        ])
        .expect("fill --file value parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::File(p) if p == "post.md"));
            }
            _ => panic!("expected Fill"),
        }

        // fill --stdin.
        let inv = parse_action(&["fill".into(), "#q".into(), "--stdin".into()])
            .expect("fill --stdin parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Stdin));
            }
            _ => panic!("expected Fill"),
        }

        // type with and without --key-events.
        let inv = parse_action(&["type".into(), "#q".into(), "hi".into()]).expect("type parses");
        match inv.action {
            Action::Type {
                selector,
                text,
                key_events,
                ..
            } => {
                assert_eq!(selector, "#q");
                assert_eq!(text, "hi");
                assert!(!key_events);
            }
            _ => panic!("expected Type"),
        }
        let inv = parse_action(&[
            "type".into(),
            "#q".into(),
            "hi".into(),
            "--key-events".into(),
        ])
        .expect("type --key-events parses");
        match inv.action {
            Action::Type { key_events, .. } => assert!(key_events),
            _ => panic!("expected Type"),
        }

        // press with selector, hold and timeout.
        let inv = parse_action(&[
            "press".into(),
            "Enter".into(),
            "--selector".into(),
            "#t".into(),
            "--hold".into(),
            "50".into(),
            "--timeout".into(),
            "60".into(),
        ])
        .expect("press parses");
        match inv.action {
            Action::Press {
                key,
                selector,
                hold,
                bound,
            } => {
                assert_eq!(key, "Enter");
                assert_eq!(selector.as_deref(), Some("#t"));
                assert_eq!(hold, Some(50));
                assert_eq!(bound, Some(Duration::from_secs(60)));
            }
            _ => panic!("expected Press"),
        }
    }

    /// `--` stays a true end-of-options terminator: every token after it is
    /// verbatim text, even one that names a flag. (Single-dash text no longer
    /// needs the shield — see [`fill_type_accept_single_dash_text`] — but the
    /// terminator behavior itself is unchanged.)
    #[test]
    fn parse_dash_separator_takes_text_verbatim() {
        let inv = parse_action(&["fill".into(), "#q".into(), "--".into(), "--foo".into()])
            .expect("fill -- separator parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Inline(t) if t == "--foo"));
            }
            _ => panic!("expected Fill"),
        }

        let inv = parse_action(&[
            "type".into(),
            "#q".into(),
            "--".into(),
            "--session".into(),
            "sneaky".into(),
            "rest".into(),
        ])
        .expect("type with --session-looking text parses");
        match inv.action {
            Action::Type { selector, text, .. } => {
                assert_eq!(selector, "#q");
                assert_eq!(text, "--session sneaky rest");
            }
            _ => panic!("expected Type"),
        }
    }

    /// Single-dash text is accepted naturally in fill/type (they have no
    /// single-dash flags, so any `-token` is literal text), while other
    /// actions keep rejecting it as an unknown flag.
    #[test]
    fn fill_type_accept_single_dash_text() {
        let inv = parse_action(&["fill".into(), "#q".into(), "-tail".into()]).expect("fill parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Inline(t) if t == "-tail"));
            }
            _ => panic!("expected Fill"),
        }

        // A `--session` after the text still binds globally.
        let inv = parse_action(&[
            "fill".into(),
            "#q".into(),
            "-tail".into(),
            "--session".into(),
            "abc".into(),
        ])
        .expect("fill with trailing --session parses");
        assert_eq!(inv.session.as_deref(), Some("abc"));
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Inline(t) if t == "-tail"));
            }
            _ => panic!("expected Fill"),
        }

        let inv =
            parse_action(&["type".into(), "#q".into(), "-dashprobe".into()]).expect("type parses");
        match inv.action {
            Action::Type { text, .. } => assert_eq!(text, "-dashprobe"),
            _ => panic!("expected Type"),
        }

        // Other actions still reject single-dash tokens as unknown flags, and
        // fill/type still reject unknown double-dash flags.
        assert!(parse_action(&["count".into(), ".x".into(), "-bogus".into()]).is_err());
        assert!(parse_action(&["open".into(), "https://x".into(), "-bogus".into()]).is_err());
        assert!(parse_action(&["fill".into(), "#q".into(), "--bogus".into()]).is_err());
    }

    /// The resolved session lands as a top-level stdout field, under the name a person
    /// reads (the record's namespace prefix is not part of it).
    #[test]
    fn surface_session_adds_top_level_field() {
        let mut env = out_env("fill", true, OutKind::Ok, json!({ "selector": "#q" }));
        surface_session(&mut env, "mahbot-chrome-abc");
        assert_eq!(env.payload["session"], json!("abc"));
        let wire: Value = serde_json::from_str(&env.to_json()).expect("wire json");
        assert_eq!(wire["session"], json!("abc"));
        assert_eq!(wire["kind"], json!("ok"));
    }

    /// The inline-text argv shield: a leading-dash value rides after `--` so
    /// chrome-use's arg preprocessor forwards it verbatim.
    #[test]
    fn text_value_argv_shields_leading_dash_text() {
        assert_eq!(text_value_argv("hello"), vec!["hello"]);
        assert_eq!(text_value_argv("--foo"), vec!["--", "--foo"]);
        assert_eq!(text_value_argv("-5"), vec!["--", "-5"]);
        assert_eq!(text_value_argv(""), vec![""]);
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

    /// Usage rejections for the text-input actions.
    #[test]
    fn parse_rejects_text_input_usage_errors() {
        // fill: exactly one text source.
        assert!(parse_action(&["fill".into(), "#q".into()]).is_err()); // none
        assert!(
            parse_action(&[
                "fill".into(),
                "#q".into(),
                "a".into(),
                "--file".into(),
                "f".into()
            ])
            .is_err()
        ); // inline + --file
        assert!(parse_action(&["fill".into(), "#q".into(), "a".into(), "--stdin".into()]).is_err()); // inline + --stdin
        assert!(
            parse_action(&[
                "fill".into(),
                "#q".into(),
                "--stdin".into(),
                "--file".into(),
                "f".into()
            ])
            .is_err()
        ); // --stdin + --file
        assert!(parse_action(&["type".into(), "#q".into()]).is_err()); // missing text
        assert!(parse_action(&["press".into(), "Enter".into(), "extra".into()]).is_err()); // extra positional
        assert!(
            parse_action(&[
                "press".into(),
                "Enter".into(),
                "--hold".into(),
                "abc".into()
            ])
            .is_err()
        ); // non-numeric hold
        assert!(parse_action(&["press".into(), "Enter".into(), "--selector".into()]).is_err()); // missing value
    }

    #[test]
    fn text_input_ok_envelope_clean_success() {
        let base = json!({ "selector": "#q" });
        let resp = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "value": "filled" })),
            ..ChromeResponse::default()
        };
        let env = text_input_ok_envelope("fill", &base, resp);
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
        assert_eq!(env.payload["data"]["value"], json!("filled"));
        assert!(env.payload.get("warning").is_none());
    }

    /// chrome-use emits `readBack` on EVERY readable `type` success and
    /// `keyListeners` on every listener-dependent `press` — they are not the
    /// degraded signal. Without chrome-use's `warning` field these are clean
    /// successes (rc 0).
    #[test]
    fn text_input_ok_envelope_read_back_and_key_listeners_without_warning_are_ok() {
        let typed = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "typed": "hi", "readBack": "hi" })),
            ..ChromeResponse::default()
        };
        let env = text_input_ok_envelope("type", &json!({ "selector": "#q", "text": "hi" }), typed);
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);

        let press = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "key": "ArrowDown", "keyListeners": 3 })),
            ..ChromeResponse::default()
        };
        let env = text_input_ok_envelope("press", &json!({ "key": "ArrowDown" }), press);
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
    }

    #[test]
    fn text_input_ok_envelope_warning_in_data_is_error() {
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
        assert!(!env.ok);
        assert_eq!(env.kind, OutKind::Error);
        assert_eq!(
            env.payload["warning"],
            json!("the field does not contain what was typed")
        );
    }

    #[test]
    fn text_input_ok_envelope_warning_at_top_level_is_error() {
        let base = json!({ "key": "Enter" });
        let resp = ChromeResponse {
            success: Some(true),
            data: Some(json!({})),
            extra: serde_json::Map::from_iter([("warning".to_string(), json!("no key listeners"))]),
            ..ChromeResponse::default()
        };
        let env = text_input_ok_envelope("press", &base, resp);
        assert!(!env.ok);
        assert_eq!(env.kind, OutKind::Error);
        assert_eq!(env.payload["warning"], json!("no key listeners"));
    }

    fn upload_payload() -> (Vec<String>, Vec<std::path::PathBuf>) {
        (
            vec!["a.txt".to_string()],
            vec![std::path::PathBuf::from("/tmp/a.txt")],
        )
    }

    #[test]
    fn upload_envelope_reports_the_counts_and_the_absolute_files() {
        let (files, resolved) = upload_payload();
        let kept = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "uploaded": 1, "attached": 1, "selector": "#f" })),
            ..ChromeResponse::default()
        };
        let env = upload_ok_envelope("#f", &files, &resolved, &kept, false);
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
        assert_eq!(env.payload["uploaded"], json!(1));
        assert_eq!(env.payload["attached"], json!(1));
        assert_eq!(env.payload["absolute_files"], json!(["/tmp/a.txt"]));
        assert!(env.payload.get("warning").is_none());
        assert!(env.payload.get("note").is_none());
    }

    #[test]
    fn upload_envelope_fails_a_zero_delivery() {
        let (files, resolved) = upload_payload();
        let nothing = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "uploaded": 0, "attached": 0, "selector": "#f" })),
            ..ChromeResponse::default()
        };
        let env = upload_ok_envelope("#f", &files, &resolved, &nothing, false);
        assert!(!env.ok);
        assert_eq!(env.kind, OutKind::Error);
        assert!(
            env.payload["error"]
                .as_str()
                .unwrap()
                .contains("nothing reached")
        );
    }

    #[test]
    fn upload_envelope_notes_a_skipped_one_file_check() {
        let (files, resolved) = upload_payload();
        let kept = ChromeResponse {
            success: Some(true),
            data: Some(json!({ "uploaded": 1, "attached": 1, "selector": "#f" })),
            ..ChromeResponse::default()
        };
        let env = upload_ok_envelope("#f", &files, &resolved, &kept, true);
        assert_eq!(
            env.payload["note"],
            json!(crate::chrome::files::unchecked_several_files_note())
        );
    }

    #[test]
    fn upload_envelope_keeps_a_consumed_input_a_success_with_its_warning() {
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
        assert!(env.ok);
        assert_eq!(env.kind, OutKind::Ok);
        assert_eq!(env.payload["attached"], json!(0));
        assert_eq!(
            env.payload["warning"],
            json!("upload events were delivered, but the file input is now empty")
        );
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
        // outright, and a bare `--keep` on one is refused for the action itself rather
        // than told to pass a flag the action forbids.
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
                // The printed placeholder is the shape a reader is promised; the
                // declaration's `takes_value` must agree with it, and the parser must
                // enforce that shape through its own shape error alone.
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
            // A documented example is a valid invocation, so it proves acceptance
            // without this guard inventing positionals of its own.
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
                // The parsed action's name is also the declaration key the
                // session-wide decision and the envelope read.
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
        // Description prose is deliberately NOT scanned; `session`'s and `wait`'s
        // details mention flags of other actions or of none, and this guard passes.
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
        // A `--` after the action word ends help lookup: what follows it is text,
        // while the unshielded form stays help.
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

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
//! environment, don't blind-retry), 3 usage error. Every step runs to the two
//! clocks the [chrome policy](crate::chrome) defines — chrome-use's clock,
//! declared to it, and the product's own kill above that — as [`StepClocks`]
//! derives them for this surface. The wait action never exposes `--load`; its one
//! raw-argv use is `open`'s internal best-effort post-navigation settle.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::chrome::actions;
use crate::chrome::contract::{
    ChromeResponse, ERROR_PAGE_PROBE_JS, EXPECT_TIMEOUT_NOTE, ErrorPageProbe, ExpectOutcome,
    OutEnvelope, OutKind, chrome_leftover_pipe_note, chrome_use_warning, classify_call_failure,
    eval_count, eval_result, expect_outcome, extract_net_error_code, extract_output,
    extract_snapshot_text, is_session_unresponsive_error, is_unreachable_tab_error,
    net_error_phrase, parse_error_page_probe, parse_first, sanitize_timeout_message,
    self_launched_browser_error, self_launched_browser_note, truncated_output_error,
    unreachable_tab_message, with_condition_timeout_note,
};
use crate::chrome::forms::{
    ExpectCond, ExtractGate, WaitTarget, count_eval_js, describe, expect_args, extract_gate,
    parse_count_op, parse_predicate, parse_state, text_value_argv, validate_extract_getters,
    wait_args, wait_target,
};
use crate::chrome::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::chrome::{
    CHROME_USE_DECLARED_BUDGET, CHROME_USE_OWN_BUDGET, CLI_EPHEMERAL_PREFIX, CLI_SESSION_PREFIX,
    ChromeCallClocks, CliRecovery, DEFAULT_OPEN_TIMEOUT, DEFAULT_STEP_TIMEOUT, KILL_SLACK,
    SESSION_STOP_TIMEOUT, clocks, is_blank_page_url, kill_bound, probe_clocks, validate_url,
};
use crate::tools::chrome_daemon::{
    CliStatus, SessionRecovery, cli_path, cli_probe, cli_version, ensure_ready_for_actions,
    readiness, recover_unresponsive_session,
};
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
        "  --session <name>   use/name a session — letters, digits, '-' or '_' only (not valid for status / session subcommands)\n\n",
    );
    out.push_str("Output (stdout, one JSON line):\n");
    out.push_str(
        "  {\"schema\":1,\"action\":\"...\",\"ok\":true|false,\"kind\":\"...\",...payload}\n\n",
    );
    out.push_str("Exit codes:\n");
    out.push_str("  0  ok / empty (including a legitimately empty region)\n");
    out.push_str("  1  site/data step failure (timeout, network, redesign, not-found, error)\n");
    out.push_str("  2  environment failure (no chrome-use CLI/relay/Chrome/display)\n");
    out.push_str("  3  usage error\n\n");
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

    let mut flag_rows: Vec<(&str, &str)> = cli.flags.to_vec();
    if cli.session {
        flag_rows.push((
            "--session <name>",
            "use/name a session — letters, digits, '-' or '_' only (not valid for status / session subcommands)",
        ));
    }
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
/// and any token after it is `-h`/`--help`. Flags/session values before the
/// action word are skipped. Returns the action's registry name.
fn action_help_request(args: &[String]) -> Option<&'static str> {
    let (i, word) = first_positional(args)?;
    let d = actions::desc(word)?;
    d.cli
        .as_ref()
        .is_some_and(|_| args[i + 1..].iter().any(|t| t == "-h" || t == "--help"))
        .then_some(d.name)
}

/// A session the CLI resolved for an action that runs in one.
struct CliSession {
    name: String,
    ephemeral: bool,
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
/// (see [`StepClocks`]).
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
    SessionStop {
        name: String,
        force: bool,
    },
    SessionStatus {
        name: String,
    },
}

impl Action {
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
            Action::SessionStop { .. } | Action::SessionStatus { .. } => "session",
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
/// record; `dispatch` turns a step's leftover into a note on the envelope
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
/// never attribute. [`dispatch`] resets the mark before the action and reads it
/// right after the action returns; a later step of a composite operation cannot
/// erase an earlier decisive step's mark, and the wedge probe's own steps cannot
/// leak into it.
static CLI_STEP_KILLED_BY_CLOCK: AtomicBool = AtomicBool::new(false);

/// `mahbot chrome` CLI entry — returns the process exit code.
pub async fn run_cli(args: &[String]) -> i32 {
    // Re-enterable pub API: clear the previous call's spawn bookkeeping.
    CHROME_USE_SPAWNED.store(false, Ordering::Relaxed);
    CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
    CLI_STEP_KILLED_BY_CLOCK.store(false, Ordering::Relaxed);
    if let Some(a) = args.first()
        && (a == "-h" || a == "--help")
    {
        print!("{}", top_help());
        return 0;
    }
    // A `-h`/`--help` after the action word is per-action help — anywhere in
    // the tail, not just immediately after it (so `open <url> --help` and
    // `--session x open -h` work too). A `-h` before any action word falls
    // through to top-level help / normal parsing: no valid invocation starts
    // with a `-h`-looking token, parse_flags rejects those as usage errors, so
    // per-action help only turns would-be usage errors into help.
    if let Some(action) = action_help_request(args) {
        print!("{}", action_help(action));
        return 0;
    }

    let invocation = match parse_invocation(args) {
        Ok(inv) => inv,
        Err(msg) => {
            let action = recognized_action(args);
            eprintln!("mahbot chrome: {msg}");
            eprintln!("run 'mahbot chrome --help' for usage.");
            out_env(&action, false, OutKind::Usage, json!({ "error": msg })).emit();
            return 3;
        }
    };

    let (envelope, session) = dispatch(&invocation).await;
    envelope.emit();
    if let Some(s) = session.as_ref().filter(|s| s.ephemeral)
        && CHROME_USE_SPAWNED.load(Ordering::Relaxed)
    {
        // The close runs after the envelope is emitted, so a helper it leaves
        // behind cannot be noted in it — it is reported here, like every other
        // diagnostic of that close.
        CHROME_USE_LEFTOVER_PIPES.store(false, Ordering::Relaxed);
        close_ephemeral(&s.name).await;
        if CHROME_USE_LEFTOVER_PIPES.load(Ordering::Relaxed) {
            eprintln!("mahbot chrome: {}", chrome_leftover_pipe_note(&s.name));
        }
    }
    envelope.kind.exit_code()
}

// ── Parsing ──────────────────────────────────────────────────────

/// Flag sets for one action: `(value flags, boolean flags)`.
type FlagSet = (&'static [&'static str], &'static [&'static str]);

/// Per-action accepted flags — the single source for both the parser and the
/// CliHelp lockstep test. `(action word, value flags, boolean flags)`.
const ACTION_FLAGS: &[(&str, &[&str], &[&str])] = &[
    ("status", &[], &[]),
    ("open", &["expect", "timeout"], &["structural"]),
    ("count", &["timeout"], &[]),
    ("wait", &["timeout", "url", "text"], &[]),
    ("expect", &["timeout"], &[]),
    ("eval", &["timeout"], &[]),
    ("extract", &["schema-file", "limit", "timeout"], &[]),
    ("click", &["timeout"], &["if-present"]),
    ("fill", &["file", "timeout"], &["stdin"]),
    ("type", &["timeout"], &["key-events"]),
    ("press", &["selector", "hold", "timeout"], &[]),
    ("session", &[], &["force"]),
];

/// Accepted flags for `word`: `(value flags, boolean flags)`.
fn action_flags(word: &str) -> FlagSet {
    ACTION_FLAGS
        .iter()
        .find(|(name, ..)| *name == word)
        .map_or((&[], &[]), |(_, v, b)| (*v, *b))
}

fn parse_invocation(args: &[String]) -> Result<Invocation, String> {
    let (session, remaining) = extract_global_session(args)?;
    let action_word = remaining
        .first()
        .ok_or_else(|| "no action given".to_string())?;
    let rest = &remaining[1..];
    let allowed = action_flags(action_word);
    match action_word.as_str() {
        "status" => parse_status(session.as_deref(), rest),
        "open" => parse_open(session.as_deref(), rest, allowed),
        "count" => parse_count(session.as_deref(), rest, allowed),
        "wait" => parse_wait(session.as_deref(), rest, allowed),
        "expect" => parse_expect(session.as_deref(), rest, allowed),
        "eval" => parse_eval(session.as_deref(), rest, allowed),
        "extract" => parse_extract(session.as_deref(), rest, allowed),
        "click" => parse_click(session.as_deref(), rest, allowed),
        "fill" => parse_fill(session.as_deref(), rest, allowed),
        "type" => parse_type(session.as_deref(), rest, allowed),
        "press" => parse_press(session.as_deref(), rest, allowed),
        "session" => parse_session(session.as_deref(), rest, allowed),
        other => Err(format!("unknown action '{other}'")),
    }
}

/// Pull the global `--session` flag out of anywhere before the `--`
/// end-of-options marker, returning the session value (last occurrence wins)
/// and the remaining tokens (everything from `--` on kept verbatim).
fn extract_global_session(args: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let mut session: Option<String> = None;
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
        } else {
            remaining.push(a.clone());
            i += 1;
        }
    }
    Ok((session, remaining))
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
/// `--flag value` and `--flag=value`. `value_flags` name flags that take a
/// value; `bool_flags` name boolean flags.
fn parse_flags(
    args: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Result<(Vec<String>, Flags), String> {
    parse_flags_impl(args, value_flags, bool_flags, false)
}

/// [`parse_flags`] for the fill/type text actions: a single-dash token is
/// taken as literal positional text (these actions have no single-dash
/// flags, so `fill '#q' -tail` fills the text `-tail` naturally instead of
/// demanding the `--` shield). Unknown `--flags` are still rejected.
fn parse_flags_text(
    args: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Result<(Vec<String>, Flags), String> {
    parse_flags_impl(args, value_flags, bool_flags, true)
}

fn parse_flags_impl(
    args: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
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
            if value_flags.contains(&name) {
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
                i += 1;
            } else if bool_flags.contains(&name) {
                if eq_value.is_some() {
                    return Err(format!("flag --{name} does not take a value"));
                }
                flags.bools.insert(name.to_string());
                i += 1;
            } else {
                return Err(format!("unknown flag --{name}"));
            }
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

fn parse_status(session: Option<&str>, rest: &[String]) -> Result<Invocation, String> {
    if session.is_some() {
        return Err("--session is not valid for status".to_string());
    }
    if !rest.is_empty() {
        return Err("unexpected arguments for status".to_string());
    }
    Ok(Invocation {
        action: Action::Status,
        session: None,
    })
}

fn parse_open(
    session: Option<&str>,
    rest: &[String],
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags_text(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags_text(rest, value_flags, bool_flags)?;
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
    allowed: FlagSet,
) -> Result<Invocation, String> {
    let (value_flags, bool_flags) = allowed;
    let (positionals, flags) = parse_flags(rest, value_flags, bool_flags)?;
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

fn parse_session(
    session: Option<&str>,
    rest: &[String],
    allowed: FlagSet,
) -> Result<Invocation, String> {
    if session.is_some() {
        return Err("--session is not valid for session stop/status".to_string());
    }
    let sub = rest
        .first()
        .map(String::as_str)
        .ok_or_else(|| "usage: session stop <name> | session status <name>".to_string())?;
    match sub {
        "stop" => {
            let (value_flags, bool_flags) = allowed;
            let (positionals, flags) = parse_flags(&rest[1..], value_flags, bool_flags)?;
            let name = take_positional(&positionals, 0, "name")?;
            validate_session_name(&name)?;
            reject_extra_positionals(&positionals, 1)?;
            Ok(Invocation {
                action: Action::SessionStop {
                    name,
                    force: flags.has("force"),
                },
                session: None,
            })
        }
        "status" => {
            let (value_flags, bool_flags) = allowed;
            let (positionals, flags) = parse_flags(&rest[1..], value_flags, bool_flags)?;
            if flags.has("force") {
                return Err("--force is not valid for session status".to_string());
            }
            let name = take_positional(&positionals, 0, "name")?;
            validate_session_name(&name)?;
            reject_extra_positionals(&positionals, 1)?;
            Ok(Invocation {
                action: Action::SessionStatus { name },
                session: None,
            })
        }
        other => Err(format!(
            "unknown session subcommand '{other}' (expected 'stop' or 'status')"
        )),
    }
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
/// deadline for (count, eval, extract, click, fill, type, press). Nothing can make
/// chrome-use give up earlier on those, so a bound below the clock the product
/// declares to chrome-use ([`CHROME_USE_DECLARED_BUDGET`]) CANNOT be honoured: it
/// is refused as a usage error rather than accepted and silently discarded. A
/// bound at or above that clock is accepted — the call is never cut off below it,
/// and above the clock it widens the product's own kill
/// ([`StepClocks::tool_clock`]) — and `None` (no flag) leaves the call at the
/// declared clock.
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
/// The flag's own value is validated at extraction ([`extract_global_session`]),
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

/// The pure argument checks the pre-action arm runs BEFORE the readiness gate:
/// the usage errors an action can answer without probing anything, so they are
/// never buried under an environment refusal. Only checks that are pure and
/// already exist inside the action live here — `open`'s URL and `--expect`
/// validation, which `open` re-runs for real.
fn validate_action_args(action: &Action) -> Option<OutEnvelope> {
    let Action::Open { url, expect, .. } = action else {
        return None;
    };
    open_args(url, expect.as_deref()).err()
}

async fn dispatch(invocation: &Invocation) -> (OutEnvelope, Option<CliSession>) {
    match &invocation.action {
        Action::Status => (stamped(status()).await, None),
        Action::SessionStop { name, force } => (stamped(session_stop(name, *force)).await, None),
        Action::SessionStatus { name } => (stamped(session_status(name)).await, None),
        action => {
            // The action's own clock: every envelope this arm returns — refusals
            // included — is stamped with it (see [`stamp_elapsed`]).
            let started = Instant::now();
            // The session this action runs in, resolved first so the argument
            // refusal below carries it like every other action envelope.
            let (name, ephemeral) = resolve_session(invocation.session.as_deref());
            // Pure argument checks come next: a malformed URL or `--expect`
            // selector is a usage error whatever the environment looks like, so
            // it is answered as one, without a readiness probe that could bury
            // the precise usage error under an environment refusal (see
            // [`validate_action_args`]).
            if let Some(mut refusal) = validate_action_args(action) {
                surface_session(&mut refusal, &name);
                stamp_elapsed(&mut refusal, started);
                return (refusal, None);
            }
            // The pre-action gate: no action is dispatched until the connection
            // to the owner's REAL browser is established, so the work can never
            // quietly land in a browser chrome-use launches itself. The
            // probe/cleanup verbs (`status`, `session …`) run before this arm and
            // are deliberately not gated — they are what reports and repairs.
            //
            // A missing helper is its own case, checked BEFORE the gate: the
            // gate's refusal is about a connection it cannot even probe, so
            // running it (and its recovery pass) would answer an absent helper
            // with the long speculative readiness refusal instead of this
            // precise missing-install report — the same envelope `require_cli`
            // builds.
            if cli_path().is_none() {
                let error = "chrome-use CLI not found";
                eprintln!("mahbot chrome: {error}");
                let mut env = env_failure(action.name(), json!({}), error);
                // As in the readiness refusal below, the resolved session is
                // surfaced like every other action envelope's.
                surface_session(&mut env, &name);
                stamp_elapsed(&mut env, started);
                return (env, None);
            }
            if let Err(refusal) = ensure_ready_for_actions().await {
                eprintln!("mahbot chrome: {refusal}");
                let mut env = out_env(
                    action.name(),
                    false,
                    OutKind::Environment,
                    json!({ "error": refusal }),
                );
                // The refused action's session is surfaced like every other
                // action envelope's, so a swallowed or defaulted `--session` is
                // observable here too.
                surface_session(&mut env, &name);
                stamp_elapsed(&mut env, started);
                return (env, None);
            }
            CLI_STEP_KILLED_BY_CLOCK.store(false, Ordering::Relaxed);
            let mut env = match action {
                Action::Open {
                    url,
                    expect,
                    structural,
                    timeout,
                } => open(url, expect.as_deref(), *structural, *timeout, &name).await,
                Action::Count { selector, bound } => count(selector, *bound, &name).await,
                Action::Wait { target, timeout } => wait(target, *timeout, &name).await,
                Action::Expect { cond, timeout } => expect(cond, *timeout, &name).await,
                Action::Eval { js, bound } => eval(js, *bound, &name).await,
                Action::Extract {
                    schema_file,
                    limit,
                    bound,
                } => extract(schema_file, *limit, *bound, &name).await,
                Action::Click {
                    selector,
                    if_present,
                    bound,
                } => click(selector, *if_present, *bound, &name).await,
                Action::Fill {
                    selector,
                    source,
                    bound,
                } => fill(selector, source, *bound, &name).await,
                Action::Type {
                    selector,
                    text,
                    key_events,
                    bound,
                } => r#type(selector, text, *key_events, *bound, &name).await,
                Action::Press {
                    key,
                    selector,
                    hold,
                    bound,
                } => press(key, selector.as_deref(), *hold, *bound, &name).await,
                Action::Status | Action::SessionStop { .. } | Action::SessionStatus { .. } => {
                    unreachable!("handled by the outer match")
                }
            };
            // Read the kill attribution before ANY recovery call — hygiene
            // only: the wedge probe's own steps are probes
            // ([`CliRecovery::Suppressed`]), so they cannot re-attribute this
            // action's ending.
            let killed_by_clock = CLI_STEP_KILLED_BY_CLOCK.load(Ordering::Relaxed);
            // Surface the resolved session (named or defaulted ephemeral) so
            // a swallowed or silently defaulted `--session` is always
            // observable in the envelope.
            surface_session(&mut env, &name);
            // The call's own wall time rides every envelope this arm returns — a
            // failing call's duration is read straight off it — while
            // `timeout_ms` (the declared/kill bound) stays a separate fact.
            stamp_elapsed(&mut env, started);
            // A process chrome-use left behind still holding a step's output
            // channel is a NOTE whatever the call's own outcome — also on a
            // failure: the call finished, its exit status and output are the
            // answer, and the leftover was started deliberately and is left
            // running. It sits alongside `error` and never changes the kind or
            // the exit code.
            if CHROME_USE_LEFTOVER_PIPES.load(Ordering::Relaxed) {
                surface_leftover_note(&mut env, &name);
            }
            // A session that stopped answering is RECOVERED here — the one place
            // the resolved session name is in scope — and the envelope's error
            // text says what was done ([`recover_session_wedge`]). `named` is the
            // resolution's own answer, not re-derived from the name.
            recover_session_wedge(&mut env, &name, !ephemeral, killed_by_clock).await;
            (env, Some(CliSession { name, ephemeral }))
        }
    }
}

/// Insert the leftover-helper note as a top-level envelope field on the result —
/// success or failure (the payload is flattened, so a payload key is a top-level
/// stdout key, sitting alongside `error`). The note names the chrome-use session
/// the call drove, so it is built from the resolved session name
/// ([`chrome_leftover_pipe_note`]).
fn surface_leftover_note(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("leftover".into(), json!(chrome_leftover_pipe_note(name)));
    }
}

/// Insert the resolved session name as a top-level envelope field (the
/// payload is flattened, so a payload key is a top-level stdout key). The
/// `session` subcommand envelopes carry their own `session` key and skip
/// this — they run session-unscoped.
fn surface_session(env: &mut OutEnvelope, name: &str) {
    if let Some(obj) = env.payload.as_object_mut() {
        obj.insert("session".into(), json!(name));
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

/// [`stamp_elapsed`] around one of the diagnostic/control verbs, whose calls the
/// dispatch arm does not clock itself.
async fn stamped<F: std::future::Future<Output = OutEnvelope>>(f: F) -> OutEnvelope {
    let started = Instant::now();
    let mut env = f.await;
    stamp_elapsed(&mut env, started);
    env
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
    /// press): the step forwards no `--timeout` chrome-use
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

    /// The product's own probe (`session stop`, `session status`, the ephemeral
    /// close): `bound` IS the kill and [`probe_clocks`] declares that same bound to
    /// chrome-use — capped at [`CHROME_USE_DECLARED_BUDGET`] for a probe whose own
    /// bound sits above it, as the `session stop` paths do — and its relay self-heal
    /// is suppressed: the documented exception, where the product reports "the tool
    /// did not answer" instead of spending the recovery window.
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
        timeout: CliTimeout::Bounded(clocks.call.kill),
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
        message: self_launched_browser_error(&note),
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

/// Construct an environment-failure envelope from the action params.
#[must_use]
fn env_failure(action: &str, params: Value, error: &str) -> OutEnvelope {
    let mut obj = match params {
        Value::Object(m) => m,
        other => {
            let mut m = serde_json::Map::new();
            m.insert("error".into(), other);
            m
        }
    };
    obj.insert("error".into(), json!(error));
    out_env(action, false, OutKind::Environment, Value::Object(obj))
}

/// Resolve the chrome-use CLI path, or build the environment-failure envelope.
///
/// The standalone `mahbot chrome` CLI deliberately resolves ONLY the product's
/// own copy ([`crate::tools::chrome_daemon::cli_path`], the location the
/// helper's own installer uses) — never a `chrome-use` found on the owner's
/// search path or in a home location, so a foreign or leftover copy is never
/// run. A host whose product-owned install has not happened yet therefore
/// reports `chrome-use CLI not found` even when some other `chrome-use` is on
/// PATH.
fn require_cli(action: &str, params: Value) -> Result<std::path::PathBuf, OutEnvelope> {
    cli_path().ok_or_else(|| env_failure(action, params, "chrome-use CLI not found"))
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

/// The pure argument checks of `open`, run both by [`dispatch`] (before the
/// readiness gate) and by [`open`] itself. The URL must be one this product may
/// navigate to, and the `--expect` selector goes through the shared wait-target
/// policy (so the numeric silent-sleep form is rejected here too) — both checked
/// before navigating, never after. The resolved wait target is the check's own
/// by-product.
fn open_args(url: &str, expect: Option<&str>) -> Result<Option<WaitTarget>, OutEnvelope> {
    if let Err(e) = validate_url(url) {
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
    let path = match require_cli("open", json!({ "url": url })) {
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
/// Non-empty content is byte-capped so the single-line JSON envelope stays
/// valid and within the tool-output budget.
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
    let path = match require_cli("count", json!({ "selector": selector })) {
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
    let base = json!({ "target": target.describe() });
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
            f.envelope("wait", base, clocks)
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
    let base = json!({ "condition": condition });
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
            f.envelope("expect", base, clocks)
        }
    }
}

/// `eval` — run JS and emit the unwrapped result as a JSON value (number,
/// string, object, or null).
async fn eval(js: &str, bound: Option<Duration>, session: &str) -> OutEnvelope {
    let path = match require_cli("eval", json!({ "js": js })) {
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

    let path = match require_cli("extract", json!({})) {
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
) -> OutEnvelope {
    let path = match require_cli("click", json!({ "selector": selector })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut args = vec!["click".to_string(), selector.to_string()];
    if if_present {
        args.push("--if-present".to_string());
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let clocks = StepClocks::tool_clock(bound);
    match spawn_step(&path, &refs, Some(session), clocks, None).await {
        Ok(_) => out_env("click", true, OutKind::Ok, json!({ "selector": selector })),
        Err(f) => f.envelope("click", json!({ "selector": selector }), clocks),
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
/// reach here (see [`self_launched_browser_error`]).
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
    let base = json!({ "selector": selector });
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
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, clocks),
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
    let base = json!({ "selector": selector, "text": text });
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
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, clocks),
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
    let base = Value::Object(base_obj);
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
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, clocks),
    }
}

/// `session stop` — name-gated stop of a CLI session (protects the interactive
/// tool's `agent-tab-*` sessions unless `--force`).
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
    let path = match require_cli("session", json!({ "session": target })) {
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
        Ok(_) => out_env("session", true, OutKind::Ok, json!({ "session": target })),
        Err(f) => f.envelope("session", json!({ "session": target }), clocks),
    }
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
    let path = match require_cli("session", json!({ "session": target })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Both steps are the product's own probes — the probe exception: bounded by
    // the product, with chrome-use's relay self-heal suppressed
    // ([`StepClocks::probe`]).
    let enumerate = StepClocks::probe(SESSION_LIST_TIMEOUT);
    let list = match spawn_step(&path, &["session", "list"], None, enumerate, None).await {
        Ok(resp) => resp,
        Err(f) => {
            return f.envelope(
                "session",
                json!({ "session": target, "stage": "enumerate" }),
                enumerate,
            );
        }
    };
    if !session_listed(&list, &target) {
        return out_env(
            "session",
            true,
            OutKind::Empty,
            json!({ "session": target, "detail": "session is not running — nothing to probe" }),
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
                json!({ "session": target, "status": "usable", "url": url }),
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
            f.envelope("session", json!({ "session": target }), probe)
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
/// the report of a helper it leaves behind, go to stderr.
async fn close_ephemeral(name: &str) {
    let Some(path) = cli_path() else {
        eprintln!(
            "mahbot chrome: could not close ephemeral session '{name}' (chrome-use CLI not found)"
        );
        return;
    };
    // A lifecycle close, not agent work (the probe exception in
    // [`StepClocks::probe`]): the product bounds it itself, declares chrome-use's
    // clock honestly, and suppresses the relay self-heal — derived through the
    // one probe shape so it cannot drift from the CLI's other probes.
    let clocks = StepClocks::probe(SESSION_STOP_TIMEOUT);
    match spawn_cli(CliSpawn {
        path: &path,
        args: &["session", "stop", name],
        session: None,
        json: true,
        capture_stderr: false,
        cancel_kills: true,
        timeout: CliTimeout::Bounded(clocks.call.kill),
        input: None,
        chrome_side: clocks.call.chrome_side,
        recovery: clocks.recovery,
    })
    .await
    {
        CliRun::Output(out) => {
            // A helper the close itself leaves holding the call's output channel
            // is recorded here for the caller ([`run_cli`]) to report — the
            // action envelope went out before this ran, so it cannot carry it.
            CHROME_USE_LEFTOVER_PIPES.fetch_or(out.leftover_pipes, Ordering::Relaxed);
            if !out.status.success() {
                eprintln!("mahbot chrome: failed to close ephemeral session '{name}'");
            }
        }
        CliRun::SpawnFailure => {
            eprintln!(
                "mahbot chrome: could not spawn chrome-use to close ephemeral session '{name}'"
            );
        }
        CliRun::TimedOut => {
            eprintln!("mahbot chrome: timed out closing ephemeral session '{name}'");
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
            parse_invocation(&[
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
        let inv = parse_invocation(&["session".into(), "status".into(), "docs".into()])
            .expect("session status parses");
        match inv.action {
            Action::SessionStatus { name } => assert_eq!(name, "docs"),
            _ => panic!("expected SessionStatus"),
        }

        // The global --session flag is rejected for session subcommands.
        let err = parse_invocation(&[
            "session".into(),
            "status".into(),
            "docs".into(),
            "--session".into(),
            "x".into(),
        ])
        .err()
        .expect("--session must be rejected for session subcommands");
        assert_eq!(err, "--session is not valid for session stop/status");

        // --force is rejected for status.
        let err = parse_session(
            None,
            &["status".into(), "docs".into(), "--force".into()],
            (&[], &["force"]),
        )
        .err()
        .expect("--force must be rejected for session status");
        assert_eq!(err, "--force is not valid for session status");

        // Extra positionals are rejected for status.
        assert!(
            parse_session(
                None,
                &["status".into(), "a".into(), "b".into()],
                (&[], &["force"]),
            )
            .is_err()
        );

        // An unknown sub names both verbs.
        let err = parse_invocation(&["session".into(), "destroy".into(), "docs".into()])
            .err()
            .expect("unknown sub must be rejected");
        assert!(
            err.contains("unknown session subcommand 'destroy' (expected 'stop' or 'status')"),
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
        assert_eq!(f.message, self_launched_browser_error(note));

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
        let err = parse_invocation(&["wait".into(), "#x".into(), "--timeout".into(), "45".into()])
            .err()
            .expect("a deadline at the tolerance must be refused");
        assert!(
            err.contains("45") && err.contains("client tolerance"),
            "err must be the tolerance refusal, naming the value: {err}"
        );

        assert!(
            parse_invocation(&["wait".into(), "#x".into(), "--timeout".into(), "44".into(),])
                .is_ok(),
            "one second below the tolerance must parse"
        );

        for args in [
            &["expect".into(), "#x".into(), "visible".into()][..],
            &["open".into(), "https://example.com".into()][..],
        ] {
            let mut argv = args.to_vec();
            argv.extend(["--timeout".into(), "45".into()]);
            let err = parse_invocation(&argv)
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
        ] {
            let mut argv = action.to_vec();
            argv.extend(["--timeout".into(), "5".into()]);
            let err = parse_invocation(&argv)
                .err()
                .unwrap_or_else(|| panic!("{argv:?} must refuse an unhonourable bound"));
            assert!(
                err.contains("--timeout 5s") && err.contains(&declared),
                "{argv:?} must name the refused value and the shortest honoured one: {err}"
            );
        }
        // At the declared clock it parses, and it is the bound the call runs to.
        let at_the_clock = parse_invocation(&[
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

        let wide = parse_invocation(&[
            "eval".into(),
            "1+1".into(),
            "--timeout".into(),
            "300".into(),
        ])
        .expect("a verb that forwards no deadline takes any bound");
        match wide.action {
            Action::Eval { bound, .. } => {
                assert_eq!(bound, Some(Duration::from_secs(300)));
            }
            _ => panic!("expected Eval with the user's bound"),
        }
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn parse_happy_paths_support_both_flag_forms() {
        // status takes no args.
        assert!(matches!(
            parse_invocation(&["status".into()])
                .expect("status parses")
                .action,
            Action::Status
        ));

        // open with `--flag value`.
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&[
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
            parse_invocation(&["open".into(), "https://example.com".into()]).expect("open parses");
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
            let inv = parse_invocation(&[action.into(), sel.into(), "--timeout=60".into()])
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
        let inv = parse_invocation(&["count".into(), ".x".into()]).expect("count parses");
        assert!(matches!(inv.action, Action::Count { bound: None, .. }));
        let inv = parse_invocation(&["eval".into(), "1+1".into()]).expect("eval parses");
        assert!(matches!(inv.action, Action::Eval { bound: None, .. }));

        // wait / expect default to the condition deadline they declare AND
        // forward as their `--timeout`, well under chrome-use's own client
        // tolerance so the tool's own verdict always fires first.
        let inv = parse_invocation(&["wait".into(), ".x".into()]).expect("wait parses");
        match inv.action {
            Action::Wait { timeout, .. } => assert_eq!(timeout, DEFAULT_STEP_TIMEOUT),
            _ => panic!("expected Wait"),
        }

        // wait with --url / --text targets.
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&["wait".into(), "--text".into(), "Loaded".into()])
            .expect("wait text parses");
        match inv.action {
            Action::Wait { target, timeout } => {
                assert!(matches!(target, WaitTarget::Text(t) if t == "Loaded"));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
            }
            _ => panic!("expected Wait"),
        }

        // expect conditions via the forms grammar.
        let inv = parse_invocation(&["expect".into(), "#main".into(), "visible".into()])
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&["click".into(), ".btn".into(), "--if-present".into()])
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
        let inv = parse_invocation(&[
            "--session".into(),
            "run-1".into(),
            "count".into(),
            ".x".into(),
        ])
        .expect("session parses");
        assert_eq!(inv.session.as_deref(), Some("run-1"));
        assert!(matches!(inv.action, Action::Count { .. }));

        // session stop with --force.
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&["fill".into(), "#q".into(), "hello".into(), "world".into()])
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
        let inv = parse_invocation(&["fill".into(), "#q".into(), "--file=post.md".into()])
            .expect("fill --file parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::File(p) if p == "post.md"));
            }
            _ => panic!("expected Fill"),
        }

        // fill with a separate `--file <path>` value.
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&["fill".into(), "#q".into(), "--stdin".into()])
            .expect("fill --stdin parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Stdin));
            }
            _ => panic!("expected Fill"),
        }

        // type with and without --key-events.
        let inv =
            parse_invocation(&["type".into(), "#q".into(), "hi".into()]).expect("type parses");
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&[
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
        let inv = parse_invocation(&["fill".into(), "#q".into(), "--".into(), "--foo".into()])
            .expect("fill -- separator parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Inline(t) if t == "--foo"));
            }
            _ => panic!("expected Fill"),
        }

        let inv = parse_invocation(&[
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
        let inv =
            parse_invocation(&["fill".into(), "#q".into(), "-tail".into()]).expect("fill parses");
        match inv.action {
            Action::Fill { source, .. } => {
                assert!(matches!(source, TextInput::Inline(t) if t == "-tail"));
            }
            _ => panic!("expected Fill"),
        }

        // A `--session` after the text still binds globally.
        let inv = parse_invocation(&[
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

        let inv = parse_invocation(&["type".into(), "#q".into(), "-dashprobe".into()])
            .expect("type parses");
        match inv.action {
            Action::Type { text, .. } => assert_eq!(text, "-dashprobe"),
            _ => panic!("expected Type"),
        }

        // Other actions still reject single-dash tokens as unknown flags, and
        // fill/type still reject unknown double-dash flags.
        assert!(parse_invocation(&["count".into(), ".x".into(), "-bogus".into()]).is_err());
        assert!(parse_invocation(&["open".into(), "https://x".into(), "-bogus".into()]).is_err());
        assert!(parse_invocation(&["fill".into(), "#q".into(), "--bogus".into()]).is_err());
    }

    /// The resolved session lands as a top-level stdout field.
    #[test]
    fn surface_session_adds_top_level_field() {
        let mut env = out_env("fill", true, OutKind::Ok, json!({ "selector": "#q" }));
        surface_session(&mut env, "mahbot-chrome-abc");
        assert_eq!(env.payload["session"], json!("mahbot-chrome-abc"));
        let wire: Value = serde_json::from_str(&env.to_json()).expect("wire json");
        assert_eq!(wire["session"], json!("mahbot-chrome-abc"));
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
        assert!(parse_invocation(&["bogus".into()]).is_err());
        assert!(parse_invocation(&["open".into()]).is_err()); // missing url
        assert!(parse_invocation(&["open".into(), "https://x".into(), "--bogus".into()]).is_err());
        assert!(parse_invocation(&["count".into()]).is_err()); // missing selector
        assert!(parse_invocation(&["extract".into()]).is_err()); // missing --schema-file
        assert!(parse_invocation(&["status".into(), "--session".into(), "x".into()]).is_err());
        assert!(parse_invocation(&["session".into(), "stop".into()]).is_err()); // missing name
        assert!(
            parse_invocation(&[
                "open".into(),
                "https://x".into(),
                "--timeout".into(),
                "abc".into()
            ])
            .is_err()
        );
        assert!(
            parse_invocation(&["open".into(), "https://x".into(), "--timeout=0".into()]).is_err()
        );
        assert!(parse_invocation(&["session".into(), "destroy".into(), "x".into()]).is_err());
        // wait must reject the numeric silent-sleep form and mixed targets.
        assert!(parse_invocation(&["wait".into(), "5000".into()]).is_err());
        assert!(
            parse_invocation(&[
                "wait".into(),
                "--url".into(),
                "a".into(),
                "--text".into(),
                "b".into()
            ])
            .is_err()
        );
        assert!(
            parse_invocation(&["wait".into(), "--url".into(), "a".into(), ".x".into()]).is_err()
        );
        // wait rejects extra positionals (previously silently ignored).
        assert!(parse_invocation(&["wait".into(), "#main".into(), "extra".into()]).is_err());
        // expect rejects missing / off-allowlist conditions.
        assert!(parse_invocation(&["expect".into()]).is_err());
        assert!(parse_invocation(&["expect".into(), "#a".into(), "gone".into()]).is_err());
        assert!(
            parse_invocation(&[
                "expect".into(),
                "count".into(),
                ".c".into(),
                "~=".into(),
                "2".into()
            ])
            .is_err()
        );
        assert!(
            parse_invocation(&[
                "expect".into(),
                "count".into(),
                ".c".into(),
                "==".into(),
                "x".into()
            ])
            .is_err()
        );
        assert!(
            parse_invocation(&[
                "expect".into(),
                "text".into(),
                "h1".into(),
                "starts".into(),
                "x".into()
            ])
            .is_err()
        );
        assert!(
            parse_invocation(&["expect".into(), "count".into(), ".c".into(), "==".into()]).is_err()
        );
    }

    /// Usage rejections for the text-input actions.
    #[test]
    fn parse_rejects_text_input_usage_errors() {
        // fill: exactly one text source.
        assert!(parse_invocation(&["fill".into(), "#q".into()]).is_err()); // none
        assert!(
            parse_invocation(&[
                "fill".into(),
                "#q".into(),
                "a".into(),
                "--file".into(),
                "f".into()
            ])
            .is_err()
        ); // inline + --file
        assert!(
            parse_invocation(&["fill".into(), "#q".into(), "a".into(), "--stdin".into()]).is_err()
        ); // inline + --stdin
        assert!(
            parse_invocation(&[
                "fill".into(),
                "#q".into(),
                "--stdin".into(),
                "--file".into(),
                "f".into()
            ])
            .is_err()
        ); // --stdin + --file
        assert!(parse_invocation(&["type".into(), "#q".into()]).is_err()); // missing text
        assert!(parse_invocation(&["press".into(), "Enter".into(), "extra".into()]).is_err()); // extra positional
        assert!(
            parse_invocation(&[
                "press".into(),
                "Enter".into(),
                "--hold".into(),
                "abc".into()
            ])
            .is_err()
        ); // non-numeric hold
        assert!(parse_invocation(&["press".into(), "Enter".into(), "--selector".into()]).is_err()); // missing value
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
    fn action_flags_match_the_cli_help_flag_sets() {
        // ACTION_FLAGS must cover exactly the CLI action words.
        let mut table: Vec<&str> = ACTION_FLAGS.iter().map(|(name, ..)| *name).collect();
        table.sort_unstable();
        let mut cli: Vec<&str> = actions::ACTIONS
            .iter()
            .filter(|a| a.cli.is_some())
            .map(|a| a.name)
            .collect();
        cli.sort_unstable();
        assert_eq!(table, cli, "ACTION_FLAGS must cover the CLI action words");

        for &(word, value_flags, bool_flags) in ACTION_FLAGS {
            let help = actions::desc(word)
                .and_then(|d| d.cli.as_ref())
                .expect("ACTION_FLAGS word should have a CliHelp");
            // Machine name of each declared flag: `--expect <sel>` → `expect`,
            // `--structural` → `structural`.
            let mut expected: Vec<&str> = help
                .flags
                .iter()
                .map(|(flag, _)| {
                    flag.strip_prefix("--")
                        .unwrap_or(flag)
                        .split_whitespace()
                        .next()
                        .unwrap_or(flag)
                })
                .collect();
            let mut actual: Vec<&str> = value_flags
                .iter()
                .chain(bool_flags.iter())
                .copied()
                .collect();
            if help.session {
                // The global `--session` flag is rendered for and accepted by
                // every session-enabled action, so it is on both sides.
                expected.push("session");
                actual.push("session");
            }
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(
                actual, expected,
                "flag set mismatch for action {word}: ACTION_FLAGS vs CliHelp"
            );
        }
    }

    #[test]
    fn action_help_request_intercepts_per_action_help() {
        let arg = |s: &[&str]| s.iter().copied().map(String::from).collect::<Vec<_>>();
        assert_eq!(action_help_request(&arg(&["open", "-h"])), Some("open"));
        assert_eq!(
            action_help_request(&arg(&["open", "https://x", "--help"])),
            Some("open")
        );
        assert_eq!(
            action_help_request(&arg(&["--session", "docs", "open", "-h"])),
            Some("open")
        );
        assert_eq!(action_help_request(&arg(&["open"])), None);
        assert_eq!(action_help_request(&arg(&["snapshot", "-h"])), None);
        assert_eq!(action_help_request(&arg(&["frobnicate", "-h"])), None);
        assert_eq!(action_help_request(&arg(&["--timeout", "5"])), None);
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

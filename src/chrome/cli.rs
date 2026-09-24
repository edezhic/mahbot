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
//! environment, don't blind-retry), 3 usage error. Every step timeout is
//! enforced MAHBOT-side via [`CliTimeout::Bounded`]. The wait action never
//! exposes `--load`; its one raw-argv use is `open`'s internal best-effort
//! post-navigation settle.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::chrome::actions;
use crate::chrome::contract::{
    ChromeResponse, ERROR_PAGE_PROBE_JS, EXPECT_TIMEOUT_NOTE, ErrorPageProbe, ExpectOutcome,
    OutEnvelope, OutKind, classify_call_failure, eval_count, eval_result, expect_outcome,
    extract_net_error_code, extract_output, extract_snapshot_text, is_session_unresponsive_error,
    is_unreachable_tab_error, net_error_phrase, parse_error_page_probe, sanitize_timeout_message,
    unreachable_tab_message, with_condition_timeout_note,
};
use crate::chrome::forms::{
    ExpectCond, ExtractGate, WaitTarget, count_eval_js, describe, expect_args, extract_gate,
    parse_count_op, parse_predicate, parse_state, text_value_argv, validate_extract_getters,
    wait_args, wait_target,
};
use crate::chrome::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::chrome::{
    CLI_EPHEMERAL_PREFIX, CLI_SESSION_PREFIX, DEADLINE_SLACK, DEFAULT_OPEN_TIMEOUT,
    is_blank_page_url, validate_url,
};
use crate::tools::chrome_daemon::{
    CliStatus, chrome_running, cli_path, cli_probe, cli_version, display_available, relay_up,
};
use crate::util::{TOOL_OUTPUT_BUDGET_BYTES, truncate_sandwich};
use serde_json::{Value, json};

/// Default step timeout (8 s) — every step uses it unless `--timeout` is given.
/// `open` defaults higher: [`DEFAULT_OPEN_TIMEOUT`] (whole-operation budget).
const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(8);

/// `session stop` bound: one stop waits out the session daemon's shutdown grace
/// (8 s on the installed CLI) and then reconnects to reclaim the tabs the session
/// created under chrome-use's own 20 s — the ≈28 s end to end the live-verified
/// behaviours in [`crate::tools::chrome_daemon`] state. The 8 s default step bound
/// would cut that off with a misleading timeout, which describes the commoner case —
/// a stop cut off while it was still legitimately reclaiming its tabs — not just the
/// narrower race in which it had already succeeded. The bound is therefore the 60 s
/// the ended-run release gives one attempt (`chrome_release`'s
/// `RELEASE_ATTEMPT_TIMEOUT`) — a bit over twice the worst case, and the same number,
/// so the two cannot drift apart. At 25 s it fell ~3 s short of that worst case.
const SESSION_STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// `session status` probe bound: a real command against the named session, so
/// chrome-use's own session-unresponsive classification has room to fire
/// (its AGENT_BROWSER_DEFAULT_TIMEOUT is 15s) instead of the CLI preempting it.
/// Opt-in — never part of the default per-command path.
const SESSION_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// The recovery flow for a wedged named session — the single source for both
/// hint surfaces ([`with_session_wedge_hint`] and
/// [`named_session_timeout_hint`]) so a wording change only happens here.
/// References only verbs that exist in the mahbot surface.
const SESSION_RECOVERY_FLOW: &str = "`mahbot chrome session stop <name>`, then re-run the action with \
     `--session <name>` to re-create it (cookies persist in the profile; open \
     tabs do not)";

/// The session-wedge remediation appended by [`StepFailure::envelope`] to an
/// Environment-classified chrome-use message that reads as a session wedge
/// (matched by [`is_session_unresponsive_error`]); any other kind or message
/// is returned unchanged. Gated on `named`: the `<name>` recovery verbs are
/// only actionable for a session the agent chose (the interactive tool runs
/// its own per-run sessions with daemon auto-recovery, so the CLI-only verbs
/// would mislead there anyway).
fn with_session_wedge_hint(kind: OutKind, message: &str, named: bool) -> String {
    if named && kind == OutKind::Environment && is_session_unresponsive_error(message) {
        format!(
            "{message} — wedged: recover with {SESSION_RECOVERY_FLOW}. \
             Probe first with `mahbot chrome session status <name>` if unsure."
        )
    } else {
        message.to_string()
    }
}

/// Appended to Timeout failures on a NAMED session: the CLI's own deadline
/// preempts chrome-use's session-unresponsive diagnostic, so a wedged session
/// surfaces as a generic per-command timeout. Factual hint only — no
/// automatic recovery (a slow site would thrash). Shares the recovery flow
/// with [`with_session_wedge_hint`] so the two hint surfaces cannot drift.
fn named_session_timeout_hint() -> String {
    format!(
        " — a wedged session can also time out on every command; probe it with \
         `mahbot chrome session status <name>`, or recover with {SESSION_RECOVERY_FLOW}."
    )
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
        "  --session <name>   use/name a session (not valid for status / session subcommands)\n\n",
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
            "use/name a session (not valid for status / session subcommands)",
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
        timeout: Duration,
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
        timeout: Duration,
    },
    Extract {
        schema_file: String,
        limit: Option<usize>,
        timeout: Duration,
    },
    Click {
        selector: String,
        if_present: bool,
        timeout: Duration,
    },
    Fill {
        selector: String,
        source: TextInput,
        timeout: Duration,
    },
    Type {
        selector: String,
        text: String,
        key_events: bool,
        timeout: Duration,
    },
    Press {
        key: String,
        selector: Option<String>,
        hold: Option<u64>,
        timeout: Duration,
    },
    SessionStop {
        name: String,
        force: bool,
    },
    SessionStatus {
        name: String,
    },
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
/// (empty for a mahbot-side deadline kill). `named` marks a step that ran in
/// a named (non-ephemeral) session — it gates the session-wedge remediation
/// hint, whose `session status/stop <name>` verbs are only actionable for a
/// session the agent chose.
#[derive(Debug)]
struct StepFailure {
    kind: OutKind,
    message: String,
    named: bool,
}

impl StepFailure {
    /// The failure envelope for `action`, layering the failure detail onto
    /// `base` params: a deadline kill reports `timeout_ms` plus a factual
    /// `error`; any other failure reports the chrome-use `error` text with
    /// the accurate remediation layered on (unreachable-tab guidance, or the
    /// session-wedge hint for named sessions), and Network failures carry the
    /// extracted net error token as `error_code` when the message contains
    /// one.
    fn envelope(self, action: &str, base: Value, timeout: Duration) -> OutEnvelope {
        let mut obj = match base {
            Value::Object(m) => m,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("error".into(), other);
                m
            }
        };
        if self.kind == OutKind::Timeout && self.message.is_empty() {
            obj.insert("timeout_ms".into(), json!(timeout.as_millis()));
            obj.insert(
                "error".into(),
                json!(deadline_error(timeout, "the step did not complete")),
            );
        } else {
            let message = if is_unreachable_tab_error(&self.message) {
                unreachable_tab_message(&self.message)
            } else {
                with_session_wedge_hint(self.kind, &self.message, self.named)
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

/// `mahbot chrome` CLI entry — returns the process exit code.
pub async fn run_cli(args: &[String]) -> i32 {
    // Re-enterable pub API: clear the previous call's spawn bookkeeping.
    CHROME_USE_SPAWNED.store(false, Ordering::Relaxed);
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
        close_ephemeral(&s.name).await;
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
            session = Some(val.clone());
            i += 1;
        } else if let Some(val) = a.strip_prefix("--session=") {
            if val.is_empty() {
                return Err("missing value for --session".to_string());
            }
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
            timeout: parse_timeout_flag(&flags, DEFAULT_OPEN_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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
            timeout: parse_timeout_flag(&flags, DEFAULT_STEP_TIMEOUT)?,
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

fn parse_timeout_flag(flags: &Flags, default: Duration) -> Result<Duration, String> {
    match flags.value("timeout") {
        Some(v) => {
            let secs: u64 = v
                .parse()
                .map_err(|_| format!("--timeout must be an integer: {v}"))?;
            if secs < 1 {
                return Err("--timeout must be at least 1 second".to_string());
            }
            Ok(Duration::from_secs(secs))
        }
        None => Ok(default),
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

/// Resolve a session name for a session-word subcommand. A bare name is
/// treated as a CLI session name and prefixed (consistent with `--session`);
/// CLI-namespace and protected-namespace names pass through as-is (status may
/// probe protected sessions read-only; only stop gates them behind --force).
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

async fn dispatch(invocation: &Invocation) -> (OutEnvelope, Option<CliSession>) {
    match &invocation.action {
        Action::Status => (status().await, None),
        Action::SessionStop { name, force } => (session_stop(name, *force).await, None),
        Action::SessionStatus { name } => (session_status(name).await, None),
        action => {
            let (name, ephemeral) = resolve_session(invocation.session.as_deref());
            let started = Instant::now();
            let mut env = match action {
                Action::Open {
                    url,
                    expect,
                    structural,
                    timeout,
                } => open(url, expect.as_deref(), *structural, *timeout, &name).await,
                Action::Count { selector, timeout } => count(selector, *timeout, &name).await,
                Action::Wait { target, timeout } => wait(target, *timeout, &name).await,
                Action::Expect { cond, timeout } => expect(cond, *timeout, &name).await,
                Action::Eval { js, timeout } => eval(js, *timeout, &name).await,
                Action::Extract {
                    schema_file,
                    limit,
                    timeout,
                } => extract(schema_file, *limit, *timeout, &name).await,
                Action::Click {
                    selector,
                    if_present,
                    timeout,
                } => click(selector, *if_present, *timeout, &name).await,
                Action::Fill {
                    selector,
                    source,
                    timeout,
                } => fill(selector, source, *timeout, &name).await,
                Action::Type {
                    selector,
                    text,
                    key_events,
                    timeout,
                } => r#type(selector, text, *key_events, *timeout, &name).await,
                Action::Press {
                    key,
                    selector,
                    hold,
                    timeout,
                } => press(key, selector.as_deref(), *hold, *timeout, &name).await,
                Action::Status | Action::SessionStop { .. } | Action::SessionStatus { .. } => {
                    unreachable!("handled by the outer match")
                }
            };
            // Surface the resolved session (named or defaulted ephemeral) so
            // a swallowed or silently defaulted `--session` is always
            // observable in the envelope.
            surface_session(&mut env, &name);
            // Deadline-expiration kinds carry the observed wall time alongside
            // the declared `timeout_ms` (redesign is open's `--structural`
            // deadline-expiration label).
            if matches!(env.kind, OutKind::Timeout | OutKind::Redesign)
                && let Some(obj) = env.payload.as_object_mut()
            {
                obj.insert("elapsed_ms".into(), json!(started.elapsed().as_millis()));
            }
            append_named_session_timeout_hint(&mut env, step_named(Some(&name)));
            (env, Some(CliSession { name, ephemeral }))
        }
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

/// Append the named-session timeout hint to a Timeout envelope. `named` is
/// [`step_named`] on the resolved session name — the same predicate that
/// gates the wedge hint in [`with_session_wedge_hint`], so the two surfaces
/// agree on every input. The CLI's own deadline preempts chrome-use's
/// session-unresponsive diagnostic, so a wedged named session otherwise
/// surfaces as a bare timeout — the factual hint leaves it recoverable (rc
/// stays 1, hint only; no automatic recovery, a slow site would thrash).
fn append_named_session_timeout_hint(env: &mut OutEnvelope, named: bool) {
    if env.kind == OutKind::Timeout
        && named
        && let Some(obj) = env.payload.as_object_mut()
        && let Some(err) = obj.get("error").and_then(Value::as_str).map(str::to_string)
    {
        let hint = named_session_timeout_hint();
        obj.insert("error".into(), json!(format!("{err}{hint}")));
    }
}

// ── Step runner ──────────────────────────────────────────────────

/// Run one chrome-use step, bounded by `timeout`. `session` scopes the call
/// via `--session`; `None` leaves the call session-unscoped (`session stop`
/// names its session via the positional instead). `input` pipes a stdin
/// payload (only `fill --stdin` uses one). `chrome_deadline` sets the
/// chrome-use-side deadline for verbs without a `--timeout` flag (i.e.
/// `open`); verbs that forward `--timeout` pass `None`.
async fn spawn_step(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    timeout: Duration,
    input: Option<&[u8]>,
    chrome_deadline: Option<Duration>,
) -> StepOutcome {
    // A session-scoped spawn attempt may materialize the ephemeral session
    // (a vanished-binary race still sets this; the close attempt is then a
    // harmless best-effort no-op).
    if session.is_some() {
        CHROME_USE_SPAWNED.store(true, Ordering::Relaxed);
    }
    let failed = |kind: OutKind, message: String| {
        Err(StepFailure {
            kind,
            message,
            named: false,
        })
    };
    let mut outcome = match spawn_cli(CliSpawn {
        path,
        args,
        session,
        json: true,
        capture_stderr: true,
        timeout: CliTimeout::Bounded(timeout),
        cancel_kills: true,
        input: input.map(<[u8]>::to_vec),
        chrome_deadline,
    })
    .await
    {
        CliRun::SpawnFailure => failed(
            OutKind::Environment,
            "chrome-use CLI could not be spawned/found".to_string(),
        ),
        CliRun::TimedOut => Err(StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
            named: false,
        }),
        CliRun::Output(out) => classify_step_output(
            args.first() == Some(&"expect"),
            out.status.code(),
            out.status.success(),
            &out.stdout,
            String::from_utf8_lossy(&out.stderr).trim(),
        ),
    };
    if let Err(f) = &mut outcome {
        f.named = step_named(session);
    }
    outcome
}

/// Whether a spawned step ran in a named (non-ephemeral) session.
fn step_named(session: Option<&str>) -> bool {
    session.is_some_and(|s| !s.starts_with(CLI_EPHEMERAL_PREFIX))
}

/// Classify one chrome-use output (the `CliRun::Output` payload) into a step
/// outcome. Pure so tests pin the mapping. `expect` is the one chrome-use
/// action whose `--json` envelope can succeed on a non-zero exit — a failed
/// assertion arrives as `success:true` with exit 1, the verdict riding in
/// `data` — so envelope-success is trusted over the exit code there and
/// nowhere else.
fn classify_step_output(
    expect_style: bool,
    exit_code: Option<i32>,
    status_success: bool,
    stdout: &[u8],
    stderr: &str,
) -> StepOutcome {
    let failed = |kind: OutKind, message: String| {
        Err(StepFailure {
            kind,
            message,
            named: false,
        })
    };
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
    let parsed: Option<ChromeResponse> = serde_json::from_slice(stdout).ok();
    if !status_success {
        if expect_style && parsed.as_ref().is_some_and(ChromeResponse::is_success) {
            return Ok(parsed.expect("success checked"));
        }
        return if let Some(resp) = parsed {
            classified(resp)
        } else {
            // The stderr fallback can also carry a timeout phrasing plus the
            // canned hint (chrome-use prints diagnostics there when stdout is
            // non-JSON), so it is classified and sanitized too.
            let msg = fallback();
            let kind = classify_call_failure(None, &msg);
            failed(kind, sanitize_timeout_message(kind, &msg))
        };
    }
    match parsed {
        Some(resp) if resp.is_success() => Ok(resp),
        Some(resp) => classified(resp),
        None => failed(
            OutKind::Error,
            if stderr.is_empty() {
                "chrome-use returned non-JSON output".to_string()
            } else {
                stderr.to_string()
            },
        ),
    }
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

/// `status` — pure preflight: RAW probes only (never `evaluate_health`, never
/// Chrome auto-launch, no environment mutation).
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
    let relay_up = relay_up().await;
    let chrome_running = chrome_running().await;
    let display = display_available();

    let mut failures: Vec<String> = Vec::new();
    if !version_ok {
        failures.push(format!("chrome-use CLI: {chrome_use}"));
    }
    match relay_up {
        Some(true) => {}
        Some(false) => failures.push("extension relay is down".to_string()),
        None => failures.push("extension relay is unknown".to_string()),
    }
    match chrome_running {
        Some(true) => {}
        Some(false) => failures.push("Chrome is not running".to_string()),
        None => failures.push("Chrome running is unknown".to_string()),
    }
    if !display {
        failures.push("no usable display".to_string());
    }

    let mut payload = serde_json::Map::new();
    payload.insert("chrome_use".into(), json!(chrome_use));
    payload.insert("relay_up".into(), json!(relay_up));
    payload.insert("chrome_running".into(), json!(chrome_running));
    payload.insert("display".into(), json!(display));

    if failures.is_empty() {
        return out_env("status", true, OutKind::Ok, Value::Object(payload));
    }
    let error = failures.join("; ");
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

/// The factual `error` text for a mahbot-side deadline kill; `what` names the
/// step that did not complete.
fn deadline_error(timeout: Duration, what: &str) -> String {
    format!("deadline reached after {}ms — {what}", timeout.as_millis())
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

/// `open` — navigate to `url`, optionally wait for `--expect` (redesign-aware
/// via `--structural`), report the committed final URL and best-effort attach
/// the page content (compact accessibility snapshot). On the plain path (no
/// `--expect`, which already serves as the settle) the navigation is followed
/// by a best-effort network settle, capped at [`SETTLE_CAP`]. `timeout` bounds
/// the whole operation (navigation + error-page probe + settle / `--expect`
/// wait + content capture); total wall time stays within `timeout +
/// DEADLINE_SLACK`.
#[expect(clippy::too_many_lines)]
async fn open(
    url: &str,
    expect: Option<&str>,
    structural: bool,
    timeout: Duration,
    session: &str,
) -> OutEnvelope {
    if let Err(e) = validate_url(url) {
        return out_env(
            "open",
            false,
            OutKind::Usage,
            json!({ "url": url, "error": e.to_string() }),
        );
    }
    // The `--expect` selector goes through the shared wait-target policy (so
    // the numeric silent-sleep form is rejected here too) — validated before
    // navigating, never after.
    let wait_for = match expect
        .map(|sel| wait_target(Some(sel), None, None))
        .transpose()
    {
        Ok(t) => t,
        Err(e) => {
            return out_env(
                "open",
                false,
                OutKind::Usage,
                json!({ "url": url, "error": e }),
            );
        }
    };
    let path = match require_cli("open", json!({ "url": url })) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let started = Instant::now();
    let total = timeout + DEADLINE_SLACK;
    // chrome-use's open verb has no `--timeout` flag and uses the
    // AGENT_BROWSER_DEFAULT_TIMEOUT env default (15 s today), so the override
    // gives heavy SPAs the full declared budget as the chrome-side deadline;
    // the mahbot-side kill rides DEADLINE_SLACK above it.
    let resp = match spawn_step(
        &path,
        &["open", url],
        Some(session),
        total,
        None,
        Some(timeout),
    )
    .await
    {
        Ok(resp) => resp,
        Err(f) => return f.envelope("open", json!({ "url": url }), timeout),
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
    // The error-page probe is reserved up to the 2 s slack so it runs even
    // when the navigation consumed the whole declared deadline; total wall
    // time stays within declared + slack. Best-effort (inconclusive = pass);
    // skipped when nothing is left. One eval yields both the error-page
    // verdict and the net error token Chrome renders in `div.error-code`
    // (empty until the neterror script runs — the generic message covers it).
    let probe_budget = total.saturating_sub(started.elapsed()).min(DEADLINE_SLACK);
    if probe_budget >= Duration::from_millis(500)
        && let Ok(probe) = spawn_step(
            &path,
            &["eval", ERROR_PAGE_PROBE_JS],
            Some(session),
            probe_budget,
            None,
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
        let wait_budget = total.saturating_sub(started.elapsed());
        // No budget left for the wait (or its capture) — emit the same timeout
        // envelope the wait-timeout arm produces, without content.
        if wait_budget < Duration::from_millis(250) {
            payload["timeout_ms"] = json!(timeout.as_millis());
            payload["error"] = json!(deadline_error(timeout, "the target wait never started"));
            payload["hint"] = json!(hint);
            return out_env("open", false, timeout_kind, payload);
        }
        // The wait's chrome-side deadline equals its remaining budget (the
        // whole-operation ceiling), forwarded via --timeout; the mahbot bound
        // matches it — whichever fires, the envelope below is identical.
        let wargs = wait_args(&target, wait_budget.as_millis());
        let refs: Vec<&str> = wargs.iter().map(String::as_str).collect();
        let waited = spawn_step(&path, &refs, Some(session), wait_budget, None, None).await;
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
                payload["timeout_ms"] = json!(timeout.as_millis());
                // The chrome-side timeout message when chrome-use phrased the
                // timeout itself, a factual deadline text on a mahbot kill.
                if f.message.is_empty() {
                    payload["error"] = json!(deadline_error(timeout, "the target never appeared"));
                } else {
                    payload["error"] = json!(f.message);
                }
                payload["hint"] = json!(hint);
                out_env("open", false, timeout_kind, payload)
            }
            Err(f) => f.envelope("open", payload, timeout),
        };
    }
    // Best-effort settle: heavy SPAs keep the network busy right after the
    // navigation commits, so the first following step can otherwise hit its
    // 8s default under contention. Raw argv — the `wait` action deliberately
    // does not expose `--load`. chrome-use ignores `--timeout` for this form
    // (its chrome-side deadline is the seeded 15s AGENT_BROWSER_DEFAULT_TIMEOUT,
    // 25s only as chrome-use's own fallback), so the real cap is the
    // mahbot-side bound below. The result is discarded — a settle timeout
    // never downgrades a committed navigation.
    if let Some(budget) = settle_budget(total.saturating_sub(started.elapsed())) {
        let _ = spawn_step(
            &path,
            &["wait", "--load", "networkidle"],
            Some(session),
            budget,
            None,
            None,
        )
        .await;
    }
    // Content capture is best-effort and bounded: it runs on the budget the
    // navigation (+ error probe + settle) left over, and a failure, exhaustion,
    // or content-free page simply omits `content` — a successful navigation is
    // never downgraded.
    let mut payload = json!({ "url": final_url });
    let budget = total.saturating_sub(started.elapsed());
    if let Some(content) = capture_open_content(&path, session, budget).await {
        payload["content"] = json!(content);
    }
    out_env("open", true, OutKind::Ok, payload)
}

/// Best-effort compact page snapshot for the `open` envelope — the same
/// content form the interactive chrome tool surfaces after an open. Skipped
/// under 500 ms of budget (mirroring the error-page probe); a failed step,
/// an unrecognized response shape, or a content-free page yields no content.
/// Non-empty content is byte-capped so the single-line JSON envelope stays
/// valid and within the tool-output budget.
async fn capture_open_content(path: &Path, session: &str, budget: Duration) -> Option<String> {
    if budget < Duration::from_millis(500) {
        return None;
    }
    let resp = spawn_step(path, &["snapshot", "-c"], Some(session), budget, None, None)
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
/// the `extract` honest-empty gate.
async fn count_via_eval(
    path: &Path,
    selector: &str,
    session: &str,
    timeout: Duration,
) -> Result<u64, StepFailure> {
    let js = count_eval_js(selector);
    let args = ["eval".to_string(), js];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let resp = spawn_step(path, &refs, Some(session), timeout, None, None).await?;
    eval_count(&resp).ok_or(StepFailure {
        kind: OutKind::Error,
        message: "count eval returned a non-numeric result".to_string(),
        named: false,
    })
}

/// `count` — eval shim over `querySelectorAll` (chrome-use has no `count` verb).
async fn count(selector: &str, timeout: Duration, session: &str) -> OutEnvelope {
    let path = match require_cli("count", json!({ "selector": selector })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match count_via_eval(&path, selector, session, timeout).await {
        Ok(n) => {
            let kind = if n == 0 { OutKind::Empty } else { OutKind::Ok };
            out_env(
                "count",
                true,
                kind,
                json!({ "selector": selector, "count": n }),
            )
        }
        Err(f) => f.envelope("count", json!({ "selector": selector }), timeout),
    }
}

/// `wait` — bounded wait for a safe [`WaitTarget`] (the numeric sleep form is
/// rejected at parse time). The requested `--timeout` IS chrome-use's deadline
/// (its wait forms honor it), so its honest timeout error surfaces at the
/// declared deadline; mahbot kills only once the deadline is actually
/// exceeded (see [`DEADLINE_SLACK`]).
async fn wait(target: &WaitTarget, timeout: Duration, session: &str) -> OutEnvelope {
    let base = json!({ "target": target.describe() });
    let path = match require_cli("wait", base.clone()) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let args = wait_args(target, timeout.as_millis());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(
        &path,
        &refs,
        Some(session),
        timeout + DEADLINE_SLACK,
        None,
        None,
    )
    .await
    {
        Ok(_) => out_env(
            "wait",
            true,
            OutKind::Ok,
            json!({ "target": target.describe() }),
        ),
        Err(mut f) => {
            with_condition_timeout_note("wait", f.kind, &mut f.message);
            f.envelope("wait", base, timeout)
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
    // Same rule as wait: the forwarded `--timeout` is chrome-use's own
    // deadline; the mahbot-side kill rides DEADLINE_SLACK above it.
    let args = expect_args(cond, timeout.as_millis());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(
        &path,
        &refs,
        Some(session),
        timeout + DEADLINE_SLACK,
        None,
        None,
    )
    .await
    {
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
            f.envelope("expect", base, timeout)
        }
    }
}

/// `eval` — run JS and emit the unwrapped result as a JSON value (number,
/// string, object, or null).
async fn eval(js: &str, timeout: Duration, session: &str) -> OutEnvelope {
    let path = match require_cli("eval", json!({ "js": js })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match spawn_step(&path, &["eval", js], Some(session), timeout, None, None).await {
        Ok(resp) => {
            let result = eval_result(&resp).cloned().unwrap_or(Value::Null);
            out_env("eval", true, OutKind::Ok, json!({ "result": result }))
        }
        Err(f) => f.envelope("eval", json!({ "js": js }), timeout),
    }
}

/// `extract` — schema-driven rows extraction, gated by a count so an empty
/// region is reported honestly (chrome-use's rows-mode `extract` returns
/// phantom rows when the rows selector matches 0).
async fn extract(
    schema_file: &str,
    limit: Option<usize>,
    timeout: Duration,
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

    // Honest-empty gate: when the rows selector matches 0, report empty without
    // invoking chrome-use's phantom-row `extract`.
    if let Some(rows_sel) = schema.get("rows").and_then(Value::as_str) {
        let count = count_via_eval(&path, rows_sel, session, timeout).await;
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
            ExtractGate::Fail(f) => return f.envelope("extract", json!({}), timeout),
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
    match spawn_step(&path, &refs, Some(session), timeout, None, None).await {
        Ok(resp) => out_env(
            "extract",
            true,
            OutKind::Ok,
            extract_output(resp.data.as_ref().unwrap_or(&Value::Null), limit),
        ),
        Err(f) => f.envelope("extract", json!({}), timeout),
    }
}

/// `click` — click a selector, forwarding `--if-present` verbatim when set
/// (chrome-use itself treats an `--if-present` miss as a no-op success).
async fn click(selector: &str, if_present: bool, timeout: Duration, session: &str) -> OutEnvelope {
    let path = match require_cli("click", json!({ "selector": selector })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut args = vec!["click".to_string(), selector.to_string()];
    if if_present {
        args.push("--if-present".to_string());
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), timeout, None, None).await {
        Ok(_) => out_env("click", true, OutKind::Ok, json!({ "selector": selector })),
        Err(f) => f.envelope("click", json!({ "selector": selector }), timeout),
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
/// carrying chrome-use's degraded-success `warning`
/// ([`crate::chrome::contract::chrome_use_warning`]) is classified as kind
/// error (rc 1) with the warning surfaced — the action may not have taken
/// effect, so it must never exit 0. A clean success emits kind ok with the
/// action params plus chrome-use's `data`.
fn text_input_ok_envelope(action: &str, base: &Value, resp: ChromeResponse) -> OutEnvelope {
    let warning = crate::chrome::contract::chrome_use_warning(&resp);
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
async fn fill(selector: &str, source: &TextInput, timeout: Duration, session: &str) -> OutEnvelope {
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
    match spawn_step(&path, &refs, Some(session), timeout, input.as_deref(), None).await {
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, timeout),
    }
}

/// `type` — character-by-character typing (appends, does not clear).
async fn r#type(
    selector: &str,
    text: &str,
    key_events: bool,
    timeout: Duration,
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
    match spawn_step(&path, &refs, Some(session), timeout, None, None).await {
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, timeout),
    }
}

/// `press` — press a key at the focused element, optionally trying to focus a
/// `--selector` first (a no-op for non-focusable elements) and holding `--hold`
/// ms before release.
async fn press(
    key: &str,
    selector: Option<&str>,
    hold: Option<u64>,
    timeout: Duration,
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
    match spawn_step(&path, &refs, Some(session), timeout, None, None).await {
        Ok(resp) => text_input_ok_envelope(action, &base, resp),
        Err(f) => f.envelope(action, base, timeout),
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
    match spawn_step(
        &path,
        &["session", "stop", &target],
        None,
        SESSION_STOP_TIMEOUT,
        None,
        None,
    )
    .await
    {
        Ok(_) => out_env("session", true, OutKind::Ok, json!({ "session": target })),
        Err(f) => f.envelope(
            "session",
            json!({ "session": target }),
            SESSION_STOP_TIMEOUT,
        ),
    }
}

/// `session status <name>` — opt-in liveness probe for a named session. A
/// daemon-free `session list` preflight (never creates a session or spawns a
/// daemon) reports a stopped session as `empty` (rc 0); a listed session is
/// probed with a real bounded `get url`. Any timeout on that probe is wedge
/// evidence (the probe reads the current URL, it never navigates), so it
/// re-classifies as Environment (rc 2) and picks up the wedge hint via
/// [`StepFailure::envelope`].
async fn session_status(name: &str) -> OutEnvelope {
    let target = resolve_session_target(name);
    let path = match require_cli("session", json!({ "session": target })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let list = match spawn_step(
        &path,
        &["session", "list"],
        None,
        DEFAULT_STEP_TIMEOUT,
        None,
        None,
    )
    .await
    {
        Ok(resp) => resp,
        Err(f) => {
            return f.envelope(
                "session",
                json!({ "session": target, "stage": "enumerate" }),
                DEFAULT_STEP_TIMEOUT,
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
    match spawn_step(
        &path,
        &["get", "url"],
        Some(&target),
        SESSION_PROBE_TIMEOUT,
        None,
        None,
    )
    .await
    {
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
            // with chrome-use's signature phrasing so the wedge hint attaches
            // in `StepFailure::envelope`, exactly like chrome-use's own
            // session-unresponsive classification. Namedness rides
            // [`step_named`] (a literal `mahbot-chrome-ephemeral-*` argument
            // is not a name the agent can recover through). Other failures
            // pass through with their honest cause (not every failure is a
            // wedge).
            if f.kind == OutKind::Timeout {
                f = StepFailure {
                    kind: OutKind::Environment,
                    message: format!(
                        "session unresponsive: no answer to the liveness probe (get url) within {}s",
                        SESSION_PROBE_TIMEOUT.as_secs()
                    ),
                    named: step_named(Some(&target)),
                };
            }
            // A protected-namespace wedge needs `--force` to stop — make the
            // recovery hint directly actionable for that edge. Only wedges
            // get the note (an honest non-wedge cause must not carry stop
            // guidance), and only protected ones (an ordinary session's stop
            // needs no --force).
            append_protected_stop_note(&mut f.message, &target);
            f.envelope(
                "session",
                json!({ "session": target }),
                SESSION_PROBE_TIMEOUT,
            )
        }
    }
}

/// Append the protected-namespace `--force` note to a probe-failure message
/// that reads as a session wedge on a protected (`agent-tab-*` /
/// `link-enricher-*`) target: the shared wedge hint says
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
/// emitted — never changes the action's exit code; diagnostics only on stderr.
async fn close_ephemeral(name: &str) {
    let Some(path) = cli_path() else {
        eprintln!(
            "mahbot chrome: could not close ephemeral session '{name}' (chrome-use CLI not found)"
        );
        return;
    };
    match spawn_cli(CliSpawn {
        path: &path,
        args: &["session", "stop", name],
        session: None,
        json: true,
        capture_stderr: false,
        cancel_kills: true,
        timeout: CliTimeout::Bounded(DEFAULT_STEP_TIMEOUT),
        input: None,
        chrome_deadline: None,
    })
    .await
    {
        CliRun::Output(out) if out.status.success() => {}
        CliRun::Output(_) => {
            eprintln!("mahbot chrome: failed to close ephemeral session '{name}'");
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
    fn with_session_wedge_hint_appends_only_for_named_environment_matches() {
        let message = "CDP session is unresponsive after attaching (Connection reset).";
        let wedge = format!(
            "{message} — wedged: recover with {SESSION_RECOVERY_FLOW}. \
             Probe first with `mahbot chrome session status <name>` if unsure."
        );
        // Named + Environment + matcher hit → hint appended.
        assert_eq!(
            with_session_wedge_hint(OutKind::Environment, message, true),
            wedge
        );
        // Ephemeral session → unchanged (the `<name>` verbs are unactionable).
        assert_eq!(
            with_session_wedge_hint(OutKind::Environment, message, false),
            message
        );
        // Timeout + matcher hit → unchanged (the matcher only fires for
        // Environment-classified messages).
        assert_eq!(
            with_session_wedge_hint(OutKind::Timeout, message, true),
            message
        );
        // Environment + plain text → unchanged.
        assert_eq!(
            with_session_wedge_hint(OutKind::Environment, "some other failure", true),
            "some other failure"
        );
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
        // An Environment-classified session wedge on a named session picks up
        // the wedge hint.
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "session unresponsive: no response within 45s".to_string(),
            named: true,
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            DEFAULT_STEP_TIMEOUT,
        );
        let err = env
            .payload
            .get("error")
            .and_then(Value::as_str)
            .expect("error present");
        assert!(err.contains(&with_session_wedge_hint(
            OutKind::Environment,
            "session unresponsive: no response within 45s",
            true
        )));

        // The same wedge on an ephemeral session is returned untouched (the
        // `<name>` recovery verbs are unactionable there).
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "session unresponsive: no response within 45s".to_string(),
            named: false,
        }
        .envelope("open", json!({ "url": "https://x" }), DEFAULT_STEP_TIMEOUT);
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some("session unresponsive: no response within 45s")
        );

        // A non-wedge message is returned untouched (no hint).
        let env = StepFailure {
            kind: OutKind::Environment,
            message: "relay isn't connected".to_string(),
            named: true,
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            DEFAULT_STEP_TIMEOUT,
        );
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some("relay isn't connected")
        );

        // A Timeout deadline kill still reports the factual deadline error.
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
            named: true,
        }
        .envelope(
            "session",
            json!({ "session": "mahbot-chrome-docs" }),
            Duration::from_secs(8),
        );
        let expected = deadline_error(Duration::from_secs(8), "the step did not complete");
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some(expected.as_str())
        );
        assert!(env.payload.get("timeout_ms").is_some());
    }

    #[test]
    fn append_named_session_timeout_hint_only_applies_to_named_timeouts() {
        let timeout_error = "deadline reached after 8000ms — the step did not complete";
        // A named Timeout envelope gets the hint appended to its error.
        let mut env = out_env(
            "open",
            false,
            OutKind::Timeout,
            json!({ "url": "https://x", "error": timeout_error }),
        );
        append_named_session_timeout_hint(&mut env, true);
        let err = env
            .payload
            .get("error")
            .and_then(Value::as_str)
            .expect("error present");
        assert!(
            err.ends_with(&named_session_timeout_hint()),
            "named timeout hint missing: {err}"
        );

        // An unnamed (ephemeral) session Timeout is untouched.
        let mut env = out_env(
            "open",
            false,
            OutKind::Timeout,
            json!({ "url": "https://x", "error": timeout_error }),
        );
        append_named_session_timeout_hint(&mut env, false);
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some(timeout_error)
        );

        // A named non-Timeout envelope is untouched.
        let mut env = out_env(
            "open",
            false,
            OutKind::Error,
            json!({ "url": "https://x", "error": "some error" }),
        );
        append_named_session_timeout_hint(&mut env, true);
        assert_eq!(
            env.payload.get("error").and_then(Value::as_str),
            Some("some error")
        );
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
                Some(0),
                true,
                &envelope(json!({"success": true, "data": {}})),
                ""
            ))
            .is_ok()
        );

        // Unparseable stdout: zero exit → non-JSON error; non-zero → stderr fallback.
        let r = classify_step_output(false, Some(0), true, b"garbage", "");
        assert_eq!(
            outcome(&r),
            Err((OutKind::Error, "chrome-use returned non-JSON output".into()))
        );
        let r = classify_step_output(false, Some(1), false, b"garbage", "relay is not connected");
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
        let timeout = Duration::from_secs(8);
        // A deadline kill now reports timeout_ms AND a factual error.
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
            named: false,
        }
        .envelope("wait", json!({ "selector": ".x" }), timeout);
        assert_eq!(env.payload["timeout_ms"], 8000);
        assert_eq!(
            env.payload["error"],
            "deadline reached after 8000ms — the step did not complete"
        );

        // Any other failure reports the chrome-use error text, no timeout_ms.
        let env = StepFailure {
            kind: OutKind::Network,
            message: "net::ERR_NAME_NOT_RESOLVED".into(),
            named: false,
        }
        .envelope("open", json!({ "url": "https://x" }), timeout);
        assert_eq!(env.payload["error"], "net::ERR_NAME_NOT_RESOLVED");
        assert_eq!(env.payload["error_code"], "ERR_NAME_NOT_RESOLVED");
        assert!(env.payload.get("timeout_ms").is_none());

        // A Network failure with no recognizable net error token gets no
        // error_code.
        let env = StepFailure {
            kind: OutKind::Network,
            message: "connection refused by peer".into(),
            named: false,
        }
        .envelope("open", json!({ "url": "https://x" }), timeout);
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

        // open without --timeout defaults to the whole-operation budget (20 s),
        // NOT the per-step 8 s.
        let inv =
            parse_invocation(&["open".into(), "https://example.com".into()]).expect("open parses");
        match inv.action {
            Action::Open { timeout, .. } => assert_eq!(timeout, DEFAULT_OPEN_TIMEOUT),
            _ => panic!("expected Open"),
        }

        // count / wait / eval.
        for (action, sel) in [("count", ".x"), ("wait", ".x"), ("eval", "1+1")] {
            let inv = parse_invocation(&[action.into(), sel.into(), "--timeout=7".into()])
                .expect("action parses");
            match inv.action {
                Action::Count { timeout, .. }
                | Action::Wait { timeout, .. }
                | Action::Eval { timeout, .. } => assert_eq!(timeout, Duration::from_secs(7)),
                _ => panic!("expected a timed action"),
            }
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
                timeout,
            } => {
                assert_eq!(schema_file, "rows.json");
                assert_eq!(limit, Some(3));
                assert_eq!(timeout, DEFAULT_STEP_TIMEOUT);
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
            "9".into(),
        ])
        .expect("press parses");
        match inv.action {
            Action::Press {
                key,
                selector,
                hold,
                timeout,
            } => {
                assert_eq!(key, "Enter");
                assert_eq!(selector.as_deref(), Some("#t"));
                assert_eq!(hold, Some(50));
                assert_eq!(timeout, Duration::from_secs(9));
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

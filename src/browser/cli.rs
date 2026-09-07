//! `mahbot browser` — headless browser automation CLI over the shared core.
//!
//! Dispatched from `main()` before temp-root/lock init (runs alongside the
//! daemon), so it must not rely on the `/tmp/mahbot` temp root or initialized
//! config; chrome-use binary resolution falls back to PATH/home locations.
//! Sessions are namespaced `mahbot-browser-*` so they can never collide with
//! the interactive tool's `agent-tab-*`.
//!
//! stdout is exactly ONE line of JSON (the [`OutEnvelope`]); stderr carries
//! human-readable diagnostics. Exit codes follow the contract: 0 success,
//! 1 site/data step failure (schedulable), 2 environment failure (fix the
//! environment, don't blind-retry), 3 usage error. Every step timeout is
//! enforced MAHBOT-side via [`CliTimeout::Bounded`]; `networkidle` is never a
//! default here.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::browser::actions;
use crate::browser::contract::{BrowserResponse, OutEnvelope, OutKind, extract_snapshot_text};
use crate::browser::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::browser::{
    CLI_EPHEMERAL_PREFIX, CLI_SESSION_PREFIX, escape_js_single_quoted, is_blank_page_url,
    validate_url,
};
use crate::tools::browser_daemon::{
    CliStatus, chrome_running, cli_path, cli_probe, cli_version, display_available,
    is_daemon_unavailable_code, is_daemon_unavailable_error, is_relay_unavailable_error, relay_up,
};
use serde_json::{Value, json};

/// Default step timeout (8 s) — every step uses it unless `--timeout` is given.
const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(8);

/// Classify a chrome-use step failure into an [`OutKind`]. Environment
/// signatures win because rc 2 must not be masked by a page-level wrapper
/// text: a relay/daemon/Chrome-launch problem is a CLI environment failure
/// even when the CLI wraps the message in a "page failed" envelope.
fn classify_call_failure(code: Option<&str>, error: &str) -> OutKind {
    if is_daemon_unavailable_error(error)
        || is_daemon_unavailable_code(code)
        || is_relay_unavailable_error(error)
    {
        return OutKind::Environment;
    }
    if error.contains("net::ERR_") || code == Some("connection_failed") {
        return OutKind::Network;
    }
    // chrome-use's internal action timeout (AGENT_BROWSER_DEFAULT_TIMEOUT
    // fires before the mahbot-side bound when the user passes --timeout
    // above 15 s). Matched by its specific phrasings ("Wait timed out after
    // Nms", "waitFor timed out: …") rather than a bare "timed out" substring,
    // so an unrecognized connect-timeout message still classifies as Error —
    // same exit code (1), honest kind label.
    let lower = error.to_ascii_lowercase();
    if code == Some("timeout")
        || lower.contains("timed out after")
        || lower.contains("waitfor timed out")
    {
        return OutKind::Timeout;
    }
    if code == Some("element_not_found") || lower.contains("element not found") {
        return OutKind::NotFound;
    }
    OutKind::Error
}

/// Top-level `mahbot browser -h` — rendered from the shared action registry.
#[must_use]
fn top_help() -> String {
    let mut out = String::from("mahbot browser — headless browser automation CLI\n\n");
    out.push_str("Usage:\n");
    out.push_str("  mahbot browser <action> [options]\n\n");
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
        "  --session <name>   use/name a session (not valid for status / session stop)\n\n",
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
    out.push_str("  mahbot browser -h            this help\n");
    out.push_str("  mahbot browser <action> -h   per-action help (flags, kinds, examples)\n");
    out
}

/// Per-action `mahbot browser <action> -h`.
#[must_use]
fn action_help(name: &str) -> String {
    let Some(d) = actions::desc(name) else {
        return String::new();
    };
    let Some(cli) = d.cli.as_ref() else {
        return String::new();
    };

    let mut out = String::new();
    let _ = write!(out, "mahbot browser {} — {}\n\n", d.name, d.purpose);
    out.push_str("Usage:\n");
    if cli.syntax.is_empty() {
        let _ = writeln!(out, "  mahbot browser {}", d.name);
    } else {
        let _ = writeln!(out, "  mahbot browser {} {}", d.name, cli.syntax);
    }

    let mut flag_rows: Vec<(&str, &str)> = cli.flags.to_vec();
    if cli.session {
        flag_rows.push((
            "--session <name>",
            "use/name a session (not valid for status / session stop)",
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

/// One parsed `mahbot browser` action.
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
        selector: String,
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
    SessionStop {
        name: String,
        force: bool,
    },
}

/// A failed chrome-use step: the classified [`OutKind`] plus the error text
/// (empty for a mahbot-side deadline kill).
struct StepFailure {
    kind: OutKind,
    message: String,
}

impl StepFailure {
    /// The failure envelope for `action`, layering the failure detail onto
    /// `base` params: a deadline kill reports `timeout_ms`, any other failure
    /// reports the chrome-use `error` text.
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
        } else {
            obj.insert("error".into(), json!(self.message));
        }
        out_env(action, false, self.kind, Value::Object(obj))
    }
}

/// Outcome of one spawned chrome-use step.
type StepOutcome = Result<BrowserResponse, StepFailure>;

/// Whether any session-scoped chrome-use spawn has been attempted this
/// process — an ephemeral session can only exist after such a spawn, so
/// cleanup is skipped (no stderr noise) when every failure happened before
/// any spawn attempt. Never read before `run_cli` resets it.
static CHROME_USE_SPAWNED: AtomicBool = AtomicBool::new(false);

/// `mahbot browser` CLI entry — returns the process exit code.
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
            eprintln!("mahbot browser: {msg}");
            eprintln!("run 'mahbot browser --help' for usage.");
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
    ("wait", &["timeout"], &[]),
    ("eval", &["timeout"], &[]),
    ("extract", &["schema-file", "limit", "timeout"], &[]),
    ("click", &["timeout"], &["if-present"]),
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
        "eval" => parse_eval(session.as_deref(), rest, allowed),
        "extract" => parse_extract(session.as_deref(), rest, allowed),
        "click" => parse_click(session.as_deref(), rest, allowed),
        "session" => parse_session_stop(session.as_deref(), rest, allowed),
        other => Err(format!("unknown action '{other}'")),
    }
}

/// Pull the global `--session` flag out of anywhere in the args, returning the
/// session value (last occurrence wins) and the remaining tokens.
fn extract_global_session(args: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let mut session: Option<String> = None;
    let mut remaining = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
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
    let mut positionals = Vec::new();
    let mut flags = Flags::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
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
        if a.starts_with('-') && a.len() > 1 {
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
            timeout: parse_timeout_flag(&flags)?,
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
            timeout: parse_timeout_flag(&flags)?,
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
    let selector = take_positional(&positionals, 0, "selector")?;
    reject_extra_positionals(&positionals, 1)?;
    Ok(Invocation {
        action: Action::Wait {
            selector,
            timeout: parse_timeout_flag(&flags)?,
        },
        session: session.map(String::from),
    })
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
            timeout: parse_timeout_flag(&flags)?,
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
            timeout: parse_timeout_flag(&flags)?,
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
            timeout: parse_timeout_flag(&flags)?,
        },
        session: session.map(String::from),
    })
}

fn parse_session_stop(
    session: Option<&str>,
    rest: &[String],
    allowed: FlagSet,
) -> Result<Invocation, String> {
    if session.is_some() {
        return Err("--session is not valid for session stop".to_string());
    }
    let sub = rest
        .first()
        .map(String::as_str)
        .ok_or_else(|| "usage: session stop <name>".to_string())?;
    if sub != "stop" {
        return Err(format!(
            "unknown session subcommand '{sub}' (expected 'stop')"
        ));
    }
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

fn parse_timeout_flag(flags: &Flags) -> Result<Duration, String> {
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
        None => Ok(DEFAULT_STEP_TIMEOUT),
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
/// becomes `mahbot-browser-<name>` (idempotent for an already-prefixed name);
/// no flag yields an ephemeral `mahbot-browser-ephemeral-<suffix>` session.
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

/// Resolve a `session stop <name>` target. A bare name is treated as a CLI
/// session name and prefixed (consistent with `--session`); a name in the CLI
/// namespace passes through; a protected-namespace name is refused unless
/// `force` — it is then used as-is (that is the point of forcing).
fn resolve_stop_target(name: &str, force: bool) -> Result<String, String> {
    if name.starts_with(CLI_SESSION_PREFIX) {
        return Ok(name.to_string());
    }
    let protected = PROTECTED_SESSION_PREFIXES
        .iter()
        .any(|p| name.starts_with(p));
    if protected {
        if force {
            return Ok(name.to_string());
        }
        return Err(format!(
            "{name} is not a mahbot-browser session; pass --force to stop it anyway"
        ));
    }
    Ok(format!("{CLI_SESSION_PREFIX}{name}"))
}

// ── Dispatch ─────────────────────────────────────────────────────

async fn dispatch(invocation: &Invocation) -> (OutEnvelope, Option<CliSession>) {
    match &invocation.action {
        Action::Status => (status().await, None),
        Action::SessionStop { name, force } => (session_stop(name, *force).await, None),
        action => {
            let (name, ephemeral) = resolve_session(invocation.session.as_deref());
            let env = match action {
                Action::Open {
                    url,
                    expect,
                    structural,
                    timeout,
                } => open(url, expect.as_deref(), *structural, *timeout, &name).await,
                Action::Count { selector, timeout } => count(selector, *timeout, &name).await,
                Action::Wait { selector, timeout } => wait(selector, *timeout, &name).await,
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
                Action::Status | Action::SessionStop { .. } => {
                    unreachable!("handled by the outer match")
                }
            };
            (env, Some(CliSession { name, ephemeral }))
        }
    }
}

// ── Step runner ──────────────────────────────────────────────────

/// SAFE wait form only: `wait <selector>`, selector first. There is
/// deliberately NO timeout parameter — chrome-use has no documented
/// `--timeout` flag for waits (only `--download` mode), so the deadline is
/// enforced by bounding the spawned process. The trap form
/// `wait --timeout N <sel>` is unconstructible through this kernel.
fn wait_selector_args(selector: &str) -> Vec<String> {
    vec!["wait".into(), selector.into()]
}

/// Build eval JS that returns the count of elements matching `selector`.
fn count_eval_js(selector: &str) -> String {
    format!(
        "document.querySelectorAll('{}').length",
        escape_js_single_quoted(selector)
    )
}

/// Run one chrome-use step, bounded by `timeout`. `session` scopes the call
/// via `--session`; `None` leaves the call session-unscoped (`session stop`
/// names its session via the positional instead).
async fn spawn_step(
    path: &Path,
    args: &[&str],
    session: Option<&str>,
    timeout: Duration,
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
        timeout: CliTimeout::Bounded(timeout),
        cancel_kills: true,
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
        }),
        CliRun::Output(out) => {
            let code = out.status.code();
            let stderr_owned = String::from_utf8_lossy(&out.stderr).trim().to_string();
            // Fallback message for a non-success exit / unparseable stdout.
            let fallback = || fallback_step_message(code, &stderr_owned);
            let classified = |resp: BrowserResponse| {
                let code = resp.code.clone();
                let msg = resp
                    .error
                    .filter(|e| !e.trim().is_empty())
                    .unwrap_or_else(fallback);
                failed(classify_call_failure(code.as_deref(), &msg), msg)
            };
            if !out.status.success() {
                return match serde_json::from_slice::<BrowserResponse>(&out.stdout) {
                    Ok(resp) => classified(resp),
                    Err(_) => failed(classify_call_failure(None, &fallback()), fallback()),
                };
            }
            match serde_json::from_slice::<BrowserResponse>(&out.stdout) {
                Ok(resp) if resp.is_success() => Ok(resp),
                Ok(resp) => classified(resp),
                Err(_) => failed(
                    OutKind::Error,
                    if stderr_owned.is_empty() {
                        "chrome-use returned non-JSON output".to_string()
                    } else {
                        stderr_owned
                    },
                ),
            }
        }
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
    eprintln!("mahbot browser: {error}");
    payload.insert("error".into(), json!(error));
    out_env(
        "status",
        false,
        OutKind::Environment,
        Value::Object(payload),
    )
}

/// `open` — navigate to `url`, optionally wait for `--expect` (redesign-aware
/// via `--structural`) and report the committed final URL.
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
    let path = match require_cli("open", json!({ "url": url })) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let navigation_started = Instant::now();
    let resp = match spawn_step(&path, &["open", url], Some(session), timeout).await {
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
                "url": final_url,
                "error": "navigation never committed — tab still on about:blank"
            }),
        );
    }
    // The navigation step consumed part of the user's `--timeout`; the
    // chrome-error probe is best-effort (inconclusive = pass) so it runs on
    // the REMAINING budget instead of a second full step — total wall time
    // stays bounded by `timeout`. Skipped when nothing is left.
    let probe_budget = timeout.saturating_sub(navigation_started.elapsed());
    let probe_js = "location.protocol === 'chrome-error:'";
    if probe_budget >= Duration::from_millis(500)
        && let Ok(probe) = spawn_step(&path, &["eval", probe_js], Some(session), probe_budget).await
    {
        // The probe result is a JS boolean; chrome-use may deliver it as JSON
        // `true` or the text "true".
        let result = eval_result(&probe);
        let err_page = result.and_then(Value::as_bool) == Some(true)
            || result
                .and_then(extract_snapshot_text)
                .is_some_and(|t| t.trim().eq_ignore_ascii_case("true"));
        if err_page {
            return out_env(
                "open",
                false,
                OutKind::Network,
                json!({
                    "url": final_url,
                    "error": "Chrome rendered an error page (site unreachable or refused)"
                }),
            );
        }
    }
    if let Some(sel) = expect {
        let wargs = wait_selector_args(sel);
        let refs: Vec<&str> = wargs.iter().map(String::as_str).collect();
        return match spawn_step(&path, &refs, Some(session), timeout).await {
            Ok(_) => out_env(
                "open",
                true,
                OutKind::Ok,
                json!({ "url": final_url, "selector": sel }),
            ),
            Err(f) if f.kind == OutKind::Timeout => {
                let (kind, hint) = if structural {
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
                out_env(
                    "open",
                    false,
                    kind,
                    json!({
                        "url": final_url,
                        "selector": sel,
                        "timeout_ms": timeout.as_millis(),
                        "hint": hint
                    }),
                )
            }
            Err(f) => f.envelope(
                "open",
                json!({ "url": final_url, "selector": sel }),
                timeout,
            ),
        };
    }
    out_env("open", true, OutKind::Ok, json!({ "url": final_url }))
}

/// Unwrap the actual JS result from an eval response. chrome-use wraps
/// `eval` output as `data = {origin, result}` — an object carrying BOTH keys
/// is treated as that wrapper; every other shape (a bare value, a plain
/// string, a page result that merely has a `result` field) passes through.
fn eval_result(resp: &BrowserResponse) -> Option<&Value> {
    resp.data
        .as_ref()
        .map(|d| match (d.get("result"), d.get("origin")) {
            (Some(result), Some(_)) => result,
            _ => d,
        })
}

/// Extract a non-negative element count from a count-eval response — the eval
/// result arrives as a JSON number or its text form.
fn eval_count(resp: &BrowserResponse) -> Option<u64> {
    let data = eval_result(resp)?;
    data.as_u64()
        .or_else(|| data.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
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
    let resp = spawn_step(path, &refs, Some(session), timeout).await?;
    eval_count(&resp).ok_or(StepFailure {
        kind: OutKind::Error,
        message: "count eval returned a non-numeric result".to_string(),
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

/// `wait` — `wait <selector>` bounded by the step deadline (never a forwarded
/// `--timeout`; the trap form is unconstructible).
async fn wait(selector: &str, timeout: Duration, session: &str) -> OutEnvelope {
    let path = match require_cli("wait", json!({ "selector": selector })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let wargs = wait_selector_args(selector);
    let refs: Vec<&str> = wargs.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), timeout).await {
        Ok(_) => out_env("wait", true, OutKind::Ok, json!({ "selector": selector })),
        Err(f) => f.envelope("wait", json!({ "selector": selector }), timeout),
    }
}

/// `eval` — run JS and emit the unwrapped result as a JSON value (number,
/// string, object, or null).
async fn eval(js: &str, timeout: Duration, session: &str) -> OutEnvelope {
    let path = match require_cli("eval", json!({ "js": js })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match spawn_step(&path, &["eval", js], Some(session), timeout).await {
        Ok(resp) => {
            let result = eval_result(&resp).cloned().unwrap_or(Value::Null);
            out_env("eval", true, OutKind::Ok, json!({ "result": result }))
        }
        Err(f) => f.envelope("eval", json!({ "js": js }), timeout),
    }
}

/// Trim `rows` to `limit` (mahbot-side) while reporting the honest `total`.
/// Returns `(trimmed, total, returned)`.
fn trim_extract_rows(rows: Vec<Value>, limit: Option<usize>) -> (Vec<Value>, usize, usize) {
    let total = rows.len();
    let returned = limit.map_or(total, |l| l.min(total));
    let trimmed: Vec<Value> = rows.into_iter().take(returned).collect();
    (trimmed, total, returned)
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

    let path = match require_cli("extract", json!({})) {
        Ok(p) => p,
        Err(e) => return e,
    };

    // Honest-empty gate: when the rows selector matches 0, report empty without
    // invoking chrome-use's phantom-row `extract`.
    if let Some(rows_sel) = schema.get("rows").and_then(Value::as_str) {
        match count_via_eval(&path, rows_sel, session, timeout).await {
            Ok(0) => {
                return out_env(
                    "extract",
                    true,
                    OutKind::Empty,
                    json!({ "data": [], "total": 0 }),
                );
            }
            Ok(_) => {}
            Err(f) => return f.envelope("extract", json!({}), timeout),
        }
    }

    // `--schema-file` is the installed chrome-use 1.5.101's own documented
    // form (`extract --help`); chrome-use 1.5.101 answers rows-mode
    // `extract --json` with `data = {rows: [...], meta: {...}}` and
    // single-object mode with the bare object; tolerate a plain array too.
    let args = [
        "extract".to_string(),
        "--schema-file".to_string(),
        schema_file.to_string(),
    ];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match spawn_step(&path, &refs, Some(session), timeout).await {
        Ok(resp) => {
            let rows: Vec<Value> = match resp.data.as_ref() {
                Some(v) if v.get("rows").is_some_and(Value::is_array) => {
                    v["rows"].as_array().expect("rows checked").clone()
                }
                Some(Value::Array(a)) => a.clone(),
                Some(other) => vec![other.clone()],
                None => Vec::new(),
            };
            let (trimmed, total, returned) = trim_extract_rows(rows, limit);
            let mut payload = json!({ "data": trimmed, "total": total });
            if limit.is_some() {
                payload["returned"] = json!(returned);
            }
            out_env("extract", true, OutKind::Ok, payload)
        }
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
    match spawn_step(&path, &refs, Some(session), timeout).await {
        Ok(_) => out_env("click", true, OutKind::Ok, json!({ "selector": selector })),
        Err(f) => f.envelope("click", json!({ "selector": selector }), timeout),
    }
}

/// `session stop` — name-gated stop of a CLI session (protects the interactive
/// tool's `agent-tab-*` sessions unless `--force`).
async fn session_stop(name: &str, force: bool) -> OutEnvelope {
    let target = match resolve_stop_target(name, force) {
        Ok(t) => t,
        Err(error) => {
            eprintln!("mahbot browser: {error}");
            return out_env(
                "session",
                false,
                OutKind::Usage,
                json!({ "session": name, "error": error }),
            );
        }
    };
    let path = match require_cli("session", json!({ "session": name })) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match spawn_step(
        &path,
        &["session", "stop", &target],
        None,
        DEFAULT_STEP_TIMEOUT,
    )
    .await
    {
        Ok(_) => out_env("session", true, OutKind::Ok, json!({ "session": target })),
        Err(f) => f.envelope(
            "session",
            json!({ "session": target }),
            DEFAULT_STEP_TIMEOUT,
        ),
    }
}

/// Best-effort close of an ephemeral session AFTER the action envelope is
/// emitted — never changes the action's exit code; diagnostics only on stderr.
async fn close_ephemeral(name: &str) {
    let Some(path) = cli_path() else {
        eprintln!(
            "mahbot browser: could not close ephemeral session '{name}' (chrome-use CLI not found)"
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
    })
    .await
    {
        CliRun::Output(out) if out.status.success() => {}
        CliRun::Output(_) => {
            eprintln!("mahbot browser: failed to close ephemeral session '{name}'");
        }
        CliRun::SpawnFailure => {
            eprintln!(
                "mahbot browser: could not spawn chrome-use to close ephemeral session '{name}'"
            );
        }
        CliRun::TimedOut => {
            eprintln!("mahbot browser: timed out closing ephemeral session '{name}'");
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_selector_args_never_builds_trap_form() {
        assert_eq!(
            wait_selector_args(".btn"),
            vec!["wait".to_string(), ".btn".to_string()]
        );
        // A selector that looks like a timeout flag stays a single positional.
        assert_eq!(
            wait_selector_args("--timeout 5 .btn"),
            vec!["wait".to_string(), "--timeout 5 .btn".to_string()]
        );
    }
    #[test]
    fn eval_count_unwraps_chrome_use_result_envelope() {
        let resp = |data: Value| BrowserResponse {
            success: Some(true),
            ok: None,
            data: Some(data),
            error: None,
            code: None,
            retryable: None,
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
        assert_eq!(name, "mahbot-browser-foo");
        assert!(!ephemeral);

        let (name, ephemeral) = resolve_session(Some("mahbot-browser-foo"));
        assert_eq!(name, "mahbot-browser-foo");
        assert!(!ephemeral);
    }

    #[test]
    fn resolve_session_ephemeral_uses_cli_prefix() {
        let (name, ephemeral) = resolve_session(None);
        assert!(name.starts_with(CLI_EPHEMERAL_PREFIX));
        assert!(ephemeral);
    }

    #[test]
    fn session_stop_gating_resolves_and_protects() {
        // CLI-namespace names pass through; bare names are prefixed.
        assert_eq!(
            resolve_stop_target("mahbot-browser-foo", false).unwrap(),
            "mahbot-browser-foo"
        );
        assert_eq!(
            resolve_stop_target("foo", false).unwrap(),
            "mahbot-browser-foo"
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
    fn classify_call_failure_mapping() {
        // Environment signatures win (rc 2 must not be masked by page text).
        assert_eq!(
            classify_call_failure(None, "daemon may be busy or unresponsive"),
            OutKind::Environment
        );
        assert_eq!(
            classify_call_failure(Some("browser_not_launched"), "whatever"),
            OutKind::Environment
        );
        assert_eq!(
            classify_call_failure(None, "relay isn't connected"),
            OutKind::Environment
        );
        // Site-level network failures.
        assert_eq!(
            classify_call_failure(None, "net::ERR_NAME_NOT_RESOLVED"),
            OutKind::Network
        );
        assert_eq!(
            classify_call_failure(Some("connection_failed"), "connection refused"),
            OutKind::Network
        );
        // chrome-use's internal action timeout (reachable when --timeout
        // exceeds its 15 s AGENT_BROWSER_DEFAULT_TIMEOUT) — kind timeout, rc 1.
        assert_eq!(
            classify_call_failure(None, "Wait timed out after 15000ms"),
            OutKind::Timeout
        );
        assert_eq!(
            classify_call_failure(None, "waitFor timed out: .submit"),
            OutKind::Timeout
        );
        // Not-found selectors.
        assert_eq!(
            classify_call_failure(Some("element_not_found"), "whatever"),
            OutKind::NotFound
        );
        assert_eq!(
            classify_call_failure(None, "Element not found"),
            OutKind::NotFound
        );
        // Everything else is a generic step failure.
        assert_eq!(
            classify_call_failure(None, "some random issue"),
            OutKind::Error
        );
        assert_eq!(
            classify_call_failure(Some("other_code"), "some issue"),
            OutKind::Error
        );
    }

    #[test]
    fn step_failure_envelope_reports_timeout_vs_error() {
        let timeout = Duration::from_secs(8);
        let env = StepFailure {
            kind: OutKind::Timeout,
            message: String::new(),
        }
        .envelope("wait", json!({ "selector": ".x" }), timeout);
        assert!(env.payload.get("timeout_ms").is_some());
        assert!(env.payload.get("error").is_none());

        let env = StepFailure {
            kind: OutKind::Network,
            message: "net::ERR_NAME_NOT_RESOLVED".into(),
        }
        .envelope("open", json!({ "url": "https://x" }), timeout);
        assert_eq!(env.payload["error"], "net::ERR_NAME_NOT_RESOLVED");
        assert!(env.payload.get("timeout_ms").is_none());
    }

    #[test]
    fn extract_limit_trim_is_honest() {
        let rows: Vec<Value> = (0..5).map(|i| json!({ "i": i })).collect();
        let (trimmed, total, returned) = trim_extract_rows(rows.clone(), Some(2));
        assert_eq!(total, 5);
        assert_eq!(returned, 2);
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[0], json!({ "i": 0 }));

        let (trimmed, total, returned) = trim_extract_rows(rows, None);
        assert_eq!(total, 5);
        assert_eq!(returned, 5);
        assert_eq!(trimmed.len(), 5);
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
        // ACTION_FLAGS must cover exactly the 8 CLI action words.
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

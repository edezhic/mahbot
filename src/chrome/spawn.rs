//! The single chrome-use CLI spawn path.
//!
//! A wedged daemon hangs inside the CLI's own ~152 s retry loop, so every
//! health/watchdog/sweep call is deadline-bounded (the shutdown close path is
//! instead bounded by its outer total-budget timeout); the interactive tool
//! bounds its dispatch itself per-call (open = 20s declared + 2s slack,
//! wait 10+2s, expect 20+2s, default 8s) via `CliTimeout::Bounded`, and on a timeout runs a bounded
//! health evaluation — failing fast with daemon guidance when the daemon is
//! down or wedged (a second consecutive hang on a session-daemon probe), since
//! the mahbot-side bound cuts off the CLI's own wedge signature. One helper,
//! [`spawn_cli`], plus the per-call timeout and cancellation policies.

use std::path::Path;
use std::time::Duration;
use tokio::process::Command;

/// Per-call timeout policy for a chrome-use CLI call.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CliTimeout {
    /// No per-call bound — only the shutdown close path uses it (its own outer
    /// total-budget timeout bounds the whole cleanup sequence; see module doc).
    Unbounded,
    /// Kill the child after the deadline (`kill_on_drop` makes dropping the
    /// read future fatal).
    Bounded(Duration),
}

/// Everything one chrome-use CLI invocation needs. The binary path is
/// resolved by the caller so call-site-specific missing-CLI handling (and,
/// for the tool, session tracking after the path resolves) stays at the call
/// site. Argument ORDER is the caller's business: the helper folds
/// `--json` / `--session <s>` in last, or before a caller `--` marker (see
/// [`build_argv`]).
pub(crate) struct CliSpawn<'a> {
    pub(crate) path: &'a Path,
    pub(crate) args: &'a [&'a str],
    pub(crate) session: Option<&'a str>,
    /// Add `--json` and pipe stdout (JSON envelopes are read from stdout).
    /// Without it stdout is nulled — non-JSON callers judge by exit status.
    pub(crate) json: bool,
    /// Pipe stderr (needed when the caller reads failure details from it);
    /// otherwise null it.
    pub(crate) capture_stderr: bool,
    /// Kill the child when the awaiting future is dropped (task cancellation
    /// or a `Bounded` timeout) instead of letting it run to completion. Every
    /// caller kills — once the interactive tool's dispatch became bounded,
    /// letting a timed-out call's chrome-use child keep retrying in the
    /// background has no upside.
    pub(crate) cancel_kills: bool,
    /// Optional stdin payload (e.g. `fill --stdin`). When set, stdin is piped
    /// and the payload is written concurrently with output collection.
    pub(crate) input: Option<Vec<u8>>,
    pub(crate) timeout: CliTimeout,
    /// Overrides the `AGENT_BROWSER_DEFAULT_TIMEOUT` env default for this one
    /// invocation — used where the chrome-use verb has no `--timeout` flag,
    /// i.e. `open`.
    pub(crate) chrome_deadline: Option<Duration>,
}

pub(crate) enum CliRun {
    Output(std::process::Output),
    /// chrome-use binary not found (caller resolved `path` from
    /// `chrome_daemon::cli_path()`, so this is a vanished-binary race).
    SpawnFailure,
    /// Bounded call exceeded its deadline (child killed).
    TimedOut,
}

/// Set HOME, `CHROMIUM_FLAGS`, and default timeout env vars on the command
/// so that the Chromium spawned by chrome-use works in service/docker
/// environments.
pub(crate) fn ensure_chrome_env(cmd: &mut Command) {
    if std::env::var_os("HOME").is_none() {
        cmd.env("HOME", "/tmp");
    }
    // Suppress Chromium's "--enable-crashes-dialog" and GPU-related flags
    // that cause issues in headless/service environments.
    if std::env::var_os("CHROMIUM_FLAGS").is_none() {
        cmd.env(
            "CHROMIUM_FLAGS",
            "--no-first-run --no-default-browser-check --disable-gpu",
        );
    }
    // Default 15-second timeout for all chrome-use actions (including
    // `wait --text` which would otherwise block much longer).
    cmd.env("AGENT_BROWSER_DEFAULT_TIMEOUT", "15000");
    // 5-minute idle timeout — the chrome-use daemon still stops after 5 idle
    // minutes, but chrome-use ≥1.5.101 PRESERVES external Chrome tabs on idle
    // (only an explicit close/session stop cleans them up), so mahbot closes
    // agent-opened sessions explicitly at run end.
    cmd.env("AGENT_BROWSER_IDLE_TIMEOUT_MS", "300000");
    // Enable human-like interaction speed for bot-detection avoidance.
    // chrome-use supports the same env vars as agent-browser for backward
    // compatibility.
    cmd.env("AGENT_BROWSER_HUMANIZE", "human");
    // Keep the upgrade-available banner out of every command's stderr.
    cmd.env("CHROME_USE_NO_UPDATE_CHECK", "1");
    cmd.env("AGENT_BROWSER_NO_UPDATE_CHECK", "1");
    // The watchdog owns recovery. Without these, a chrome command issued while
    // the relay is down makes the CLI kill session daemons / the native host
    // and wait up to 45s for a relay revive — racing the watchdog's own
    // cause-aware recovery and turning a health check into a 45s stall.
    cmd.env("AGENT_BROWSER_NO_AUTO_RECONNECT", "1");
    cmd.env("AGENT_BROWSER_RELAY_REVIVE_SECS", "0");
}

/// Override the chrome-use-side deadline ([`CliSpawn::chrome_deadline`]) on
/// `cmd`, which [`ensure_chrome_env`] already seeded with the
/// `AGENT_BROWSER_DEFAULT_TIMEOUT` default. Only `Some` overrides.
pub(crate) fn apply_chrome_deadline(cmd: &mut Command, deadline: Option<Duration>) {
    if let Some(d) = deadline {
        cmd.env("AGENT_BROWSER_DEFAULT_TIMEOUT", d.as_millis().to_string());
    }
}

/// Caller args with the global `--json` / `--session <s>` flags folded in.
/// The flags go before a `--` end-of-options marker when the caller args
/// contain one (a fill/type leading-dash text shield): chrome-use's arg
/// preprocessor drops everything after `--`, so appended flags would be
/// swallowed into the text value (verified against chrome-use 1.5.111).
/// Without a `--` the flags go last.
fn build_argv(args: &[&str], json: bool, session: Option<&str>) -> Vec<String> {
    let mut global: Vec<String> = Vec::new();
    if json {
        global.push("--json".to_string());
    }
    if let Some(s) = session {
        global.extend(["--session".to_string(), s.to_string()]);
    }
    if global.is_empty() {
        return args.iter().map(|a| (*a).to_string()).collect();
    }
    let insert_at = args.iter().position(|a| *a == "--").unwrap_or(args.len());
    let mut folded: Vec<String> = args[..insert_at].iter().map(|a| (*a).to_string()).collect();
    folded.extend(global);
    folded.extend(args[insert_at..].iter().map(|a| (*a).to_string()));
    folded
}

/// Spawn a chrome-use CLI invocation per [`CliSpawn`]: apply
/// [`ensure_chrome_env`], the caller's args with the global `--json` /
/// `--session <s>` flags folded in, pipe stdout and stderr per the `json` /
/// `capture_stderr` flags, and kill-on-drop. A bounded call that hits its
/// deadline reports [`CliRun::TimedOut`]; any spawn IO error is folded into
/// [`CliRun::SpawnFailure`].
pub(crate) async fn spawn_cli(spec: CliSpawn<'_>) -> CliRun {
    let mut cmd = Command::new(spec.path);
    ensure_chrome_env(&mut cmd);
    apply_chrome_deadline(&mut cmd, spec.chrome_deadline);
    cmd.args(build_argv(spec.args, spec.json, spec.session));
    if spec.json {
        cmd.stdout(std::process::Stdio::piped());
    } else {
        cmd.stdout(std::process::Stdio::null());
    }
    if spec.capture_stderr {
        cmd.stderr(std::process::Stdio::piped());
    } else {
        cmd.stderr(std::process::Stdio::null());
    }
    cmd.kill_on_drop(spec.cancel_kills);

    // When a stdin payload is set, pipe stdin and write the payload
    // concurrently with output collection — a full stdin write must not
    // deadlock against a full stdout pipe. Without a payload, `output()`
    // (stdin nulled) is the simple path.
    if spec.input.is_some() {
        cmd.stdin(std::process::Stdio::piped());
    }
    let run = async {
        let Some(data) = spec.input else {
            return cmd.output().await;
        };
        let mut child = cmd.spawn()?;
        let mut stdin = child.stdin.take().expect("stdin piped when input is set");
        let writer = tokio::task::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(&data).await;
            let _ = stdin.shutdown().await;
        });
        let out = child.wait_with_output().await;
        let _ = writer.await; // normally already done; EPIPE if the child never read
        out
    };

    match spec.timeout {
        CliTimeout::Unbounded => match run.await {
            Ok(out) => CliRun::Output(out),
            Err(_) => CliRun::SpawnFailure,
        },
        CliTimeout::Bounded(d) => match tokio::time::timeout(d, run).await {
            Err(_) => CliRun::TimedOut,
            Ok(Err(_)) => CliRun::SpawnFailure,
            Ok(Ok(out)) => CliRun::Output(out),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::set_env_var;
    use tokio::process::Command;

    /// Read an env var explicitly set on the command by `ensure_chrome_env`.
    fn cmd_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.as_std()
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new(key))
            .and_then(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
    }

    #[test]
    fn build_argv_folds_global_flags_before_dash_shield() {
        assert_eq!(
            build_argv(
                &["fill", "#q", "--", "-tail"],
                true,
                Some("mahbot-chrome-abc")
            ),
            [
                "fill",
                "#q",
                "--json",
                "--session",
                "mahbot-chrome-abc",
                "--",
                "-tail"
            ]
        );
        assert_eq!(
            build_argv(&["open", "https://x"], true, Some("s")),
            ["open", "https://x", "--json", "--session", "s"]
        );
        assert_eq!(
            build_argv(&["session", "list"], false, None),
            ["session", "list"]
        );
        assert_eq!(
            build_argv(&["session", "list"], false, Some("s")),
            ["session", "list", "--session", "s"]
        );
    }

    #[test]
    fn ensure_chrome_env_defaults_home_only_when_missing() {
        {
            let _guard = set_env_var("HOME", None);
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME").as_deref(), Some("/tmp"));
        }
        {
            let _guard = set_env_var("HOME", Some("/home/user"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME"), None);
        }
    }

    #[test]
    fn ensure_chrome_env_defaults_chromium_flags_only_when_missing() {
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", None);
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(
                cmd_env(&cmd, "CHROMIUM_FLAGS").as_deref(),
                Some("--no-first-run --no-default-browser-check --disable-gpu")
            );
        }
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", Some("--headless"));
            let mut cmd = Command::new("true");
            ensure_chrome_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "CHROMIUM_FLAGS"), None);
        }
    }

    #[test]
    fn ensure_chrome_env_sets_fixed_env_vars() {
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT").as_deref(),
            Some("15000")
        );
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_IDLE_TIMEOUT_MS").as_deref(),
            Some("300000")
        );
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_NO_AUTO_RECONNECT").as_deref(),
            Some("1")
        );
    }

    #[test]
    fn chrome_deadline_override_replaces_env_default() {
        // The override replaces the seeded 15000 ms default for this one call.
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        apply_chrome_deadline(&mut cmd, Some(Duration::from_secs(18)));
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT").as_deref(),
            Some("18000")
        );

        // None keeps the seeded default untouched.
        let mut cmd = Command::new("true");
        ensure_chrome_env(&mut cmd);
        apply_chrome_deadline(&mut cmd, None);
        assert_eq!(
            cmd_env(&cmd, "AGENT_BROWSER_DEFAULT_TIMEOUT").as_deref(),
            Some("15000")
        );
    }
}

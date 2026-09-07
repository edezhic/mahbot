//! The single chrome-use CLI spawn path.
//!
//! A wedged daemon hangs inside the CLI's own ~152 s retry loop, so every
//! health/watchdog/sweep call is deadline-bounded (the shutdown close path is
//! instead bounded by its outer total-budget timeout); the interactive tool
//! bounds its dispatch itself per-call (open 15s, networkidle wait 10s,
//! default 8s) via `CliTimeout::Bounded`, and on a timeout runs a bounded
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
/// site. Argument ORDER is the caller's business: the helper only appends
/// `--json` and `--session <s>` after the caller's args.
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
    pub(crate) timeout: CliTimeout,
}

pub(crate) enum CliRun {
    Output(std::process::Output),
    /// chrome-use binary not found (caller resolved `path` from
    /// `browser_daemon::cli_path()`, so this is a vanished-binary race).
    SpawnFailure,
    /// Bounded call exceeded its deadline (child killed).
    TimedOut,
}

/// Set HOME, `CHROMIUM_FLAGS`, and default timeout env vars on the command
/// so that the Chromium spawned by chrome-use works in service/docker
/// environments.
pub(crate) fn ensure_browser_env(cmd: &mut Command) {
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
    // The watchdog owns recovery. Without these, a browser command issued while
    // the relay is down makes the CLI kill session daemons / the native host
    // and wait up to 45s for a relay revive — racing the watchdog's own
    // cause-aware recovery and turning a health check into a 45s stall.
    cmd.env("AGENT_BROWSER_NO_AUTO_RECONNECT", "1");
    cmd.env("AGENT_BROWSER_RELAY_REVIVE_SECS", "0");
}

/// Spawn a chrome-use CLI invocation per [`CliSpawn`]: apply
/// [`ensure_browser_env`], the caller's args, the optional `--json` / `--session
/// <s>` suffixes, pipe stdout and stderr per the `json` / `capture_stderr`
/// flags, and kill-on-drop. A bounded call that hits its deadline reports
/// [`CliRun::TimedOut`]; any spawn IO error is folded into
/// [`CliRun::SpawnFailure`].
pub(crate) async fn spawn_cli(spec: CliSpawn<'_>) -> CliRun {
    let mut cmd = Command::new(spec.path);
    ensure_browser_env(&mut cmd);
    cmd.args(spec.args);
    if spec.json {
        cmd.arg("--json").stdout(std::process::Stdio::piped());
    } else {
        cmd.stdout(std::process::Stdio::null());
    }
    if let Some(session) = spec.session {
        cmd.args(["--session", session]);
    }
    if spec.capture_stderr {
        cmd.stderr(std::process::Stdio::piped());
    } else {
        cmd.stderr(std::process::Stdio::null());
    }
    cmd.kill_on_drop(spec.cancel_kills);

    match spec.timeout {
        CliTimeout::Unbounded => match cmd.output().await {
            Ok(out) => CliRun::Output(out),
            Err(_) => CliRun::SpawnFailure,
        },
        CliTimeout::Bounded(d) => match tokio::time::timeout(d, cmd.output()).await {
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

    /// Read an env var explicitly set on the command by `ensure_browser_env`.
    fn cmd_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.as_std()
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new(key))
            .and_then(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
    }

    #[test]
    fn ensure_browser_env_defaults_home_only_when_missing() {
        {
            let _guard = set_env_var("HOME", None);
            let mut cmd = Command::new("true");
            ensure_browser_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME").as_deref(), Some("/tmp"));
        }
        {
            let _guard = set_env_var("HOME", Some("/home/user"));
            let mut cmd = Command::new("true");
            ensure_browser_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "HOME"), None);
        }
    }

    #[test]
    fn ensure_browser_env_defaults_chromium_flags_only_when_missing() {
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", None);
            let mut cmd = Command::new("true");
            ensure_browser_env(&mut cmd);
            assert_eq!(
                cmd_env(&cmd, "CHROMIUM_FLAGS").as_deref(),
                Some("--no-first-run --no-default-browser-check --disable-gpu")
            );
        }
        {
            let _guard = set_env_var("CHROMIUM_FLAGS", Some("--headless"));
            let mut cmd = Command::new("true");
            ensure_browser_env(&mut cmd);
            assert_eq!(cmd_env(&cmd, "CHROMIUM_FLAGS"), None);
        }
    }

    #[test]
    fn ensure_browser_env_sets_fixed_env_vars() {
        let mut cmd = Command::new("true");
        ensure_browser_env(&mut cmd);
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
}

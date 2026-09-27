//! Shared chrome automation core: the chrome-use CLI spawn mechanics and the
//! tolerant envelope contract that every chrome frontend needs.
//!
//! Frontends own everything user-facing: the interactive `chrome` tool
//! ([`crate::tools::chrome`]) owns its LLM-facing framing and error texts,
//! and the `mahbot chrome` CLI ([`cli`]) owns its stdout/exit-code emission.
//! The per-action descriptor registry ([`actions`]) is the ONE shared source
//! of LLM-facing action descriptions, rendered by both frontends (the tool's
//! parameter schema and the CLI help); the shared chrome-failure remediation
//! notes ([`contract`]) are the one other agent-facing text owned here. It
//! also holds the chrome-use CLI spawning
//! ([`spawn`]), the tolerant `--json` response envelope parser ([`contract`]),
//! and the session-name/URL rules the tool and CLI share.
//!
//! ## The two clocks
//!
//! Every call the product bounds runs to a clock chrome-use itself works to, and
//! the product's own kill rides above it — save for the product's OWN bounded extra
//! work below (a probe, or a composite operation's best-effort sub-step), which the
//! product ends itself, and for the tool's shutdown sweep (session list/close),
//! which carries no product kill at all. That chrome-side clock is DECLARED to
//! chrome-use up front (the `AGENT_BROWSER_DEFAULT_TIMEOUT` it is spawned with):
//! the verb's own forwarded `--timeout` for the verbs chrome-use honours one on
//! (`wait`, `expect`), and [`CHROME_USE_DECLARED_BUDGET`] for every verb the
//! product forwards no deadline to (`open`'s navigate included, which chrome-use
//! does not bound by the forwarded env value) — save for a probe, which declares a
//! bound of its own ([`probe_clocks`]). No declaration may reach chrome-use's
//! own client tolerance ([`CHROME_USE_OWN_BUDGET`]) — the clock the tool works
//! to for a call it was given no deadline for: declared at the tolerance, it
//! runs out of tolerance instead of reporting, and its session-unresponsive
//! verdict (which stops the session and loses its tabs) replaces its honest one.
//! [`CHROME_USE_DECLARED_BUDGET`] therefore sits one [`KILL_SLACK`] UNDER that
//! tolerance, and the product's kill then rides [`KILL_SLACK`] above the
//! declaration plus whatever relay self-heal the call may run: [`clocks`] builds
//! that pair for a step that declares only its own clock, and a step whose kill
//! must ride above a WIDER bound instead (a caller's `--timeout`, a product-side
//! budget) is widened through [`kill_bound`] without re-adding the margin by hand.
//! Declaring first is what makes chrome-use's own honest verdict (a structured
//! timeout, with chrome-use's own reason) the one that normally reaches the
//! caller: on every step whose outcome IS the caller's verdict, chrome-use must be
//! the clock that gives up first, and a mahbot kill there means it did not answer
//! at all.
//!
//! One deliberate exception covers both cases where the product must bound its
//! OWN work instead of letting chrome-use run to its own clock — probes in
//! everything but name:
//!
//! - The product's own probes (`session stop`, `session status`, the CLI's
//!   ephemeral close, and the health/watchdog calls plus the tab sweep they drive):
//!   the product's bound IS the kill, and [`probe_clocks`] declares that same bound
//!   to chrome-use — capped at [`CHROME_USE_DECLARED_BUDGET`] for a probe whose own
//!   bound sits above it. chrome-use's relay self-heal is suppressed
//!   ([`CliRecovery::Suppressed`]), so an unanswered probe reports "the tool did
//!   not answer" rather than spending the recovery window.
//! - A composite operation's best-effort sub-steps (`open`'s error-page probe,
//!   post-navigation settle and content capture): also
//!   [`CliRecovery::Suppressed`], with a kill resting [`KILL_SLACK`] above the
//!   bound the product declares for them. Their failure never fails the
//!   operation — their result only refines or is discarded — so they are bounded
//!   by the product exactly the way a probe is. The product-side bound of such a
//!   sub-step may sit below chrome-use's own clock; that is the point, and a
//!   product bound — not chrome-use's verdict — is what ends it.
//!
//! So the two named exceptions are really one rule: the product's OWN bounded
//! extra work. Every step the caller's own action drives — the navigation included,
//! and `open`'s `--expect` wait — runs to the clocks below, never to a product-side
//! bound: a call must never fail because the product cut it off. The probes above
//! are the product's own liveness checks rather than the caller's work, so a
//! `session stop` / `session status` the caller ran by hand still takes the product
//! bound above.
//!
//! [`KILL_SLACK`] is the product's margin wherever a kill rides above the clock
//! a step runs to: above the declared chrome-side clock for a step the caller's
//! action drives, and above the product's own bound for a best-effort sub-step,
//! where that bound — not the declared clock — is what the
//! kill follows. A probe takes NO margin: [`probe_clocks`] makes its bound its
//! kill.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;
use std::time::Duration;

use ipnet::IpNet;
use url::Host;

pub(crate) mod actions;
pub(crate) mod cli;
pub(crate) mod contract;
pub(crate) mod forms;
pub(crate) mod spawn;

/// Name prefix of every session the `mahbot chrome` CLI creates (named and
/// ephemeral). The interactive tool keeps `agent-tab-*`; link enrichment keeps
/// `link-enricher-*`.
pub(crate) const CLI_SESSION_PREFIX: &str = "mahbot-chrome-";

/// The margin by which the product's own kill rides above the clock a step runs
/// to: that clock is the chrome-side deadline declared to chrome-use
/// ([`CHROME_USE_DECLARED_BUDGET`] for every verb mahbot forwards no `--timeout`
/// to, the forwarded deadline otherwise), or the product's own bound for a
/// best-effort sub-step. It is what lets a child that
/// answers at the very end of its own deadline be reaped rather than killed; the
/// same margin is what holds every declaration one step below chrome-use's own
/// client tolerance, which is how [`CHROME_USE_DECLARED_BUDGET`] is derived. It is
/// the reap allowance only: the child's bytes are collected in their own bound,
/// one [`spawn`] drain window covering both pipes, separate from this margin.
pub(crate) const KILL_SLACK: Duration = Duration::from_secs(2);

/// chrome-use's own CLIENT TOLERANCE: its 30 s CDP command cap plus its 15 s
/// socket-read margin — the clock it works to for a call it was given no
/// deadline for. Never a declaration the product makes: a declaration AT the
/// tolerance makes chrome-use run out of that tolerance instead of reporting, so
/// its session-unresponsive verdict (which stops the session) replaces its honest
/// one. Every declaration the product makes must stay strictly under it — this
/// is what a forwarded `--timeout` is refused against (see
/// [`crate::chrome::cli`]) — and the declaration for a verb the product forwards
/// no `--timeout` to is [`CHROME_USE_DECLARED_BUDGET`]: the longest declaration
/// mahbot itself makes (a verb that does forward a `--timeout` declares the
/// caller's own value), with a probe's own bound at or below it ([`probe_clocks`]).
pub(crate) const CHROME_USE_OWN_BUDGET: Duration = Duration::from_secs(45);

/// What the product DECLARES to chrome-use as the chrome-side deadline for a
/// verb it forwards no `--timeout` to: [`CHROME_USE_OWN_BUDGET`] — the tool's own
/// client tolerance — less one [`KILL_SLACK`], so the tool always gives up on its
/// own declared deadline and reports its own verdict before the product's kill
/// rides in.
pub(crate) const CHROME_USE_DECLARED_BUDGET: Duration =
    CHROME_USE_OWN_BUDGET.saturating_sub(KILL_SLACK);

/// One `session stop`'s bound: the session daemon's 8 s shutdown grace, then a
/// reconnect that reclaims the session's tabs under chrome-use's own 20 s — the
/// ≈28 s end to end the live-verified behaviours of [`crate::tools::chrome_daemon`]
/// state. About twice that, so the estimate drifting with a chrome-use release
/// cannot charge a working stop as a failure. Shared by the CLI's stop verbs and
/// the ended-run release ([`crate::tools::chrome_release`]) so a working stop is
/// never cut off early on one path and not the other.
pub(crate) const SESSION_STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// The relay self-heal chrome-use may run inside a call whose relay connection
/// is gone: it kills the stale daemon, re-registers the native host, launches
/// the owner's real Chrome when it is not running, and polls for up to its own
/// `AGENT_BROWSER_RELAY_REVIVE_SECS` — 45 s by default, pinned to this number
/// for a call that allows the self-heal, so the tool's clock and the product's
/// kill ([`kill_bound`]) agree on the same recovery window.
pub(crate) const RELAY_RECOVERY_BUDGET: Duration = Duration::from_secs(45);

/// Whether a chrome-use call may run chrome-use's own relay self-heal.
///
/// [`Allowed`](Self::Allowed) is the policy for every call that carries out
/// work for the agent — the interactive tool's actions and the `mahbot chrome`
/// CLI's steps — because that work must survive a relay hiccup exactly the way
/// chrome-use intends (kill the stale daemon, re-register the native host,
/// relaunch the owner's Chrome, retry). [`Suppressed`](Self::Suppressed) is
/// ONLY for the product's own bounded extra work — the snapshot/probe calls
/// (`status`, `extension status`, `agent-browser` daemon probes, session
/// close/stop and the release path) and a composite operation's best-effort
/// sub-steps, which are probes in everything but name (see the module doc's
/// exception).
#[derive(Debug, Clone, Copy)]
pub(crate) enum CliRecovery {
    /// The call may run chrome-use's relay self-heal.
    Allowed,
    /// The call must not run chrome-use's relay self-heal.
    Suppressed,
}

impl CliRecovery {
    /// The recovery window the product's kill has to ride above under this
    /// policy.
    #[must_use]
    pub(crate) fn budget(self) -> Duration {
        match self {
            Self::Allowed => RELAY_RECOVERY_BUDGET,
            Self::Suppressed => Duration::ZERO,
        }
    }
}

/// The ONE derivation of the product's kill for a chrome-use call: the clock the
/// call runs to — the chrome-side deadline declared to chrome-use
/// ([`CHROME_USE_DECLARED_BUDGET`] for every verb mahbot forwards no `--timeout`
/// to), or a bound the product sets itself for a best-effort sub-step — plus
/// whatever relay recovery the call is allowed to run, plus [`KILL_SLACK`]. A probe
/// takes no margin at all ([`probe_clocks`]); see the module doc for both cases.
/// Saturating, so an absurd caller-supplied bound cannot overflow.
#[must_use]
pub(crate) fn kill_bound(chrome_side: Duration, recovery: CliRecovery) -> Duration {
    chrome_side
        .saturating_add(recovery.budget())
        .saturating_add(KILL_SLACK)
}

/// The two clocks of one chrome-use call, derived together by [`clocks`]: the
/// deadline DECLARED to chrome-use and the product's own kill above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChromeCallClocks {
    /// The deadline this call declares to chrome-use (its
    /// `AGENT_BROWSER_DEFAULT_TIMEOUT`) — the clock chrome-use works to.
    pub(crate) chrome_side: Duration,
    /// The product's own kill for this call.
    pub(crate) kill: Duration,
}

/// Derive the two clocks of one chrome-use call from the command-level deadline
/// the caller declares for the verb: `Some(d)` means the verb honours a deadline
/// of `d` that the product forwards (`--timeout`), and that deadline IS the
/// chrome-side clock; `None` means the verb has no command-level deadline at all
/// — chrome-use neither accepts nor honours a `--timeout` for it — so the clock
/// declared to chrome-use is [`CHROME_USE_DECLARED_BUDGET`], one [`KILL_SLACK`]
/// under the tolerance the tool itself works to. The kill is [`kill_bound`] above
/// whichever clock applies.
#[must_use]
pub(crate) fn clocks(declared: Option<Duration>, recovery: CliRecovery) -> ChromeCallClocks {
    let chrome_side = declared.unwrap_or(CHROME_USE_DECLARED_BUDGET);
    ChromeCallClocks {
        chrome_side,
        kill: kill_bound(chrome_side, recovery),
    }
}

/// The clocks of one of the product's own probes: the product's own `bound` IS the
/// kill, so a probe that does not answer is reported as such instead of spending the
/// recovery window, and that same bound is what is DECLARED to chrome-use — capped
/// at [`CHROME_USE_DECLARED_BUDGET`], the longest declaration mahbot makes for a verb
/// it forwards no `--timeout` to (see [`CHROME_USE_OWN_BUDGET`]). A deliberately
/// short liveness probe therefore declares the bound it really runs to, never a
/// longer one it would never let run; a probe whose own bound sits above the cap
/// (the `session stop` paths) declares the cap. The ONE derivation of a probe's
/// clocks, shared by the tool's probes and the CLI's.
#[must_use]
pub(crate) fn probe_clocks(bound: Duration) -> ChromeCallClocks {
    ChromeCallClocks {
        chrome_side: bound.min(CHROME_USE_DECLARED_BUDGET),
        kill: bound,
    }
}

/// Default `open` operation budget (20 s), kept as the CLI's `open --timeout`
/// default: it bounds the waits `open` drives — the error-page probe, the
/// post-navigation settle / `--expect` wait, and the content capture. It is NOT
/// a kill bound on the navigation, and not chrome-use's `open` deadline either:
/// chrome-use neither forwards nor honours a timeout for `open`'s navigate, which
/// runs to the clock the product declares for a verb it forwards no `--timeout`
/// to ([`CHROME_USE_DECLARED_BUDGET`]) — so `open` may legitimately outlive this
/// budget, and a navigation chrome-use is still working on is never cut off
/// mid-report.
pub(crate) const DEFAULT_OPEN_TIMEOUT: Duration = Duration::from_secs(20);

/// Default condition deadline for `wait`/`expect` (8 s), the CLI's default when
/// the caller gives no `--timeout` ([`crate::chrome::cli`] forwards it as the
/// verb's `--timeout`); the tool's own `wait`/`expect` deadlines sit in the same
/// range. Deliberately far below [`CHROME_USE_OWN_BUDGET`]: a forward declaring
/// exactly chrome-use's own client tolerance makes chrome-use run out of that
/// tolerance instead of answering, so its session-unresponsive classification —
/// which costs the session its tabs — replaces its honest "condition was not
/// met" verdict. A condition deadline must always leave the tool room to answer.
pub(crate) const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(8);

/// Prefix of per-invocation ephemeral CLI sessions — the ONLY CLI prefix the
/// daemon-side sweep may close (orphan protection after crashes); named
/// `mahbot-chrome-<name>` sessions are never swept (cookie persistence is a
/// feature).
pub(crate) const CLI_EPHEMERAL_PREFIX: &str = "mahbot-chrome-ephemeral-";

/// IP networks [`validate_url`] refuses to navigate to — the unambiguous
/// metadata/link-local targets only: link-local (incl. the 169.254.169.254
/// cloud-metadata endpoint), the 0.0.0.0/8 source-address space, the IPv6
/// loopback, and all IPv4-mapped IPv6. Loopback (127.0.0.1) and RFC1918 are
/// deliberately NOT blocked — driving the user's real browser for local dev is
/// a legitimate scenario, and deny-lists are inherently bypass-prone.
static DENIED_IP_NETS: LazyLock<[IpNet; 4]> = LazyLock::new(|| {
    [
        IpNet::new(IpAddr::V4(Ipv4Addr::new(169, 254, 0, 0)), 16)
            .expect("169.254.0.0/16 is a valid IPv4 net"),
        IpNet::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8).expect("0.0.0.0/8 is a valid IPv4 net"),
        IpNet::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 128).expect("::1/128 is a valid IPv6 net"),
        IpNet::new(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0, 0)), 96)
            .expect("::ffff:0:0/96 is a valid IPv6 net"),
    ]
});

/// Validate a URL is structurally safe to navigate to.
///
/// The SSRF guard is deliberately narrow: it blocks only the unambiguous
/// metadata/link-local targets (see [`DENIED_IP_NETS`]), leaving loopback and
/// RFC1918 reachable. A TLS-intercepting captive portal presenting a fake
/// certificate still passes this structural check — that is caught at runtime
/// by the chrome-error page detection, not here.
pub(crate) fn validate_url(url: &str) -> anyhow::Result<()> {
    let url = url.trim();

    if url.is_empty() {
        anyhow::bail!("URL cannot be empty");
    }

    // Block file:// case-insensitively — it bypasses SSRF controls.
    if url.to_ascii_lowercase().starts_with("file://") {
        anyhow::bail!("file:// URLs are not allowed in chrome automation");
    }

    // url::Url lowercases scheme+host, so "HTTP://EXAMPLE.COM" passes and
    // scheme-less strings ("example.com") stay rejected. Parse error or a
    // non-http(s) scheme is the same structural rejection.
    let parsed = url::Url::parse(url)
        .map_err(|_| anyhow::anyhow!("Only http:// and https:// URLs are allowed"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("Only http:// and https:// URLs are allowed");
    }

    let Some(host) = parsed.host() else {
        return Ok(());
    };

    let ip = match host {
        Host::Domain(domain) => {
            if domain.eq_ignore_ascii_case("metadata.google.internal") {
                anyhow::bail!(
                    "Refused to navigate to {url}: host {host} is a denied link-local/metadata address (SSRF guard)"
                );
            }
            None
        }
        Host::Ipv4(addr) => Some(IpAddr::V4(addr)),
        Host::Ipv6(addr) => Some(IpAddr::V6(addr)),
    };
    if ip.is_some_and(|ip| DENIED_IP_NETS.iter().any(|net| net.contains(&ip))) {
        anyhow::bail!(
            "Refused to navigate to {url}: host {host} is a denied link-local/metadata address (SSRF guard)"
        );
    }

    Ok(())
}

/// A navigation that ends here never committed — the tab stayed on the scratch
/// `about:blank` page (relay broken, navigation blocked, etc.). Keyed on the
/// final URL only, so legitimately content-free pages (image URLs, PDFs, canvas
/// shells) are not false-flagged by having zero extracted text.
pub(crate) fn is_blank_page_url(url: &str) -> bool {
    let url = url.trim();
    url.is_empty() || url.starts_with("about:blank")
}

/// chrome-use's scratch error-page URL (live-verified on 1.5.101 and
/// reconfirmed on the installed 1.5.106; the installed copy is replaced with the
/// newest release on every product start): DNS/refused/unsafe-port navigations
/// return
/// rc=0 with success:true and commit to `chrome-error://chromewebdata/`
/// (prefix match — the trailing slash exists; no false positives on 404/5xx
/// or real captive-portal pages). The only false positive is a
/// TLS-intercepting portal with a fake certificate — accepted trade-off.
pub(crate) fn is_chrome_error_page(url: &str) -> bool {
    url.trim().starts_with("chrome-error://")
}

/// Escape a string for embedding in a single-quoted JavaScript literal.
/// Shared by the CLI's count/eval shims and the interactive tool's
/// `innerText` eval.
#[must_use]
pub(crate) fn escape_js_single_quoted(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_validation_accepts_real_and_local_targets() {
        for url in [
            "https://example.com",
            "HTTP://EXAMPLE.COM",
            "https://EXAMPLE.com/path",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://192.168.1.1",
            "http://10.0.0.5",
            // Canonicalizes to 127.0.0.1 — loopback is not blocked.
            "http://0x7f000001/",
        ] {
            assert!(validate_url(url).is_ok(), "expected {url:?} to be accepted");
        }
    }

    #[test]
    fn clocks_declare_the_verb_clock_and_ride_the_kill_above_it() {
        // No declared deadline: the product declares one kill margin under
        // chrome-use's own client tolerance (so the tool always gives up with its
        // own verdict first). The kill is the hand-computed expected value
        // CHROME_USE_DECLARED_BUDGET + RELAY_RECOVERY_BUDGET + KILL_SLACK, not a
        // derivation from `clocks`.
        assert_eq!(
            clocks(None, CliRecovery::Allowed),
            ChromeCallClocks {
                chrome_side: CHROME_USE_DECLARED_BUDGET,
                kill: Duration::from_secs(90),
            }
        );
        // A declared deadline the verb forwards IS the chrome-side clock.
        assert_eq!(
            clocks(Some(Duration::from_secs(10)), CliRecovery::Allowed).chrome_side,
            Duration::from_secs(10)
        );
        assert!(
            clocks(Some(Duration::from_secs(10)), CliRecovery::Allowed).kill
                > Duration::from_secs(10)
        );
        // A suppressed step keeps the same declared clock but spends no
        // recovery window on the kill — hand-computed as
        // CHROME_USE_DECLARED_BUDGET + KILL_SLACK.
        assert_eq!(
            clocks(None, CliRecovery::Suppressed),
            ChromeCallClocks {
                chrome_side: CHROME_USE_DECLARED_BUDGET,
                kill: Duration::from_secs(45),
            }
        );
        // A probe's kill is the product's own bound, and that SAME bound is what
        // it declares to chrome-use — never a longer one it would never let run —
        // up to the cap, the longest declaration mahbot makes for a verb it
        // forwards no `--timeout` to.
        assert_eq!(
            probe_clocks(Duration::from_secs(8)),
            ChromeCallClocks {
                chrome_side: Duration::from_secs(8),
                kill: Duration::from_secs(8),
            }
        );
        // A probe whose own bound sits above the cap declares the cap.
        assert_eq!(
            probe_clocks(Duration::from_secs(60)),
            ChromeCallClocks {
                chrome_side: CHROME_USE_DECLARED_BUDGET,
                kill: Duration::from_secs(60),
            }
        );
    }

    #[test]
    fn url_validation_rejects_unsafe_targets() {
        let err = validate_url("").unwrap_err();
        assert!(err.to_string().contains("URL cannot be empty"));

        for url in ["ftp://example.com", "example.com"] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string()
                    .contains("Only http:// and https:// URLs are allowed"),
                "unexpected error for {url:?}: {err}"
            );
        }

        for url in ["FILE:///etc/passwd", "file:///etc/passwd"] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string().contains("not allowed"),
                "unexpected error for {url:?}: {err}"
            );
        }

        for url in [
            "http://169.254.169.254/",
            "http://0.0.0.0/",
            "http://0.1.2.3/",
            "http://[::1]/",
            "http://[::ffff:169.254.169.254]/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string().contains("SSRF guard"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    #[test]
    fn blank_page_url_predicate() {
        // Scratch/about pages that never committed are blank-page failures.
        assert!(is_blank_page_url("about:blank"));
        assert!(is_blank_page_url("about:blank#blocked"));
        assert!(is_blank_page_url(""));
        // Real pages — even ones that render no text — are NOT blank failures
        // (image URLs, PDFs, canvas shells).
        assert!(!is_blank_page_url("https://example.com/image.png"));
        assert!(!is_blank_page_url("https://example.com/file.pdf"));
        assert!(!is_blank_page_url("about:srcdoc"));
    }

    #[test]
    fn chrome_error_page_predicate() {
        assert!(is_chrome_error_page("chrome-error://chromewebdata/"));
        assert!(!is_chrome_error_page("https://example.com/"));
    }
}

//! Shared chrome automation core: the chrome-use CLI spawn mechanics and the tolerant envelope
//! contract every chrome frontend needs.
//!
//! Frontends own everything user-facing: the interactive `chrome` tool
//! ([`crate::tools::chrome`]) owns its LLM-facing framing and error texts, and the `mahbot chrome`
//! CLI ([`cli`]) its stdout/exit-code emission. The per-action descriptor registry ([`actions`]) is
//! the ONE shared source of LLM-facing action descriptions, rendered by both frontends, and the
//! chrome-failure remediation notes ([`contract`]) are the other agent-facing text owned here. The
//! chrome-use spawn ([`spawn`]), the tolerant `--json` envelope parser, the session-name/URL rules
//! and the CLI's own session record ([`cli_sessions`]) live here too.
//!
//! ## The two clocks
//!
//! Chrome-use bounds each call it runs to a clock it works to, and the product's own kill rides
//! above it — except for the product's OWN bounded extra work (a probe, or a composite operation's
//! best-effort sub-step), which the product ends itself. That chrome-side clock is DECLARED to
//! chrome-use up front (`AGENT_BROWSER_DEFAULT_TIMEOUT`): the verb's forwarded `--timeout` for the
//! verbs chrome-use honours one on (`wait`, `expect`), and [`CHROME_USE_DECLARED_BUDGET`] for every
//! verb the product forwards no deadline to — except a probe, which declares its own bound
//! ([`probe_clocks`]). No declaration may reach chrome-use's client tolerance
//! ([`CHROME_USE_OWN_BUDGET`]): declared at the tolerance, chrome-use runs out of tolerance instead
//! of reporting, and its session-unresponsive verdict (which stops the session and loses its tabs)
//! replaces its honest one. [`CHROME_USE_DECLARED_BUDGET`] therefore sits one [`KILL_SLACK`] under
//! that tolerance, and the product's kill rides [`KILL_SLACK`] above the declaration plus whatever
//! relay self-heal the call may run: [`clocks`] builds that pair, and a step whose kill must ride
//! above a WIDER bound (a caller's `--timeout`, a product-side budget) is widened through
//! [`kill_bound`] without re-adding the margin by hand. Declaring first is what keeps chrome-use's
//! own honest structured timeout the verdict that normally reaches the caller: on every step whose
//! outcome IS the caller's verdict, chrome-use must give up first, and a mahbot kill there means it
//! did not answer at all.
//!
//! One deliberate exception covers both cases where the product must bound its OWN work — probes in
//! everything but name. The product's probes (`session stop`, `session status`, the CLI's ephemeral
//! close, the health/watchdog calls and their tab sweep) take the product's bound AS the kill, and
//! [`probe_clocks`] declares that same bound to chrome-use, capped at
//! [`CHROME_USE_DECLARED_BUDGET`]; relay self-heal is suppressed ([`CliRecovery::Suppressed`]) so
//! an unanswered probe reports "the tool did not answer" instead of spending the recovery window.
//! A composite operation's best-effort sub-steps (`open`'s error-page probe, the post-navigation
//! settle and content capture) are also suppressed, with a kill resting [`KILL_SLACK`] above the
//! bound the product declares for them; their failure never fails the operation, so they are
//! bounded exactly like a probe — that product bound, not chrome-use's verdict, ends them.
//!
//! So the two named exceptions are one rule: the product's OWN bounded extra work. Every step the
//! caller's action drives — the navigation included, and `open`'s `--expect` wait — runs to the
//! clocks here, never to a product-side bound, so a call never fails because the product cut it
//! off. [`KILL_SLACK`] is the margin wherever a kill rides above the clock a step runs to, and a
//! probe takes no margin at all: [`probe_clocks`] makes its bound its kill.

use std::time::Duration;

pub(crate) mod actions;
pub(crate) mod cli;
pub(crate) mod cli_sessions;
pub(crate) mod contract;
pub(crate) mod files;
pub(crate) mod forms;
pub(crate) mod spawn;

/// Name prefix of every session the `mahbot chrome` CLI creates (named and ephemeral). The
/// interactive tool keeps `agent-tab-*`; link enrichment keeps `link-enricher-*`.
pub(crate) const CLI_SESSION_PREFIX: &str = "mahbot-chrome-";

/// The margin by which the product's own kill rides above the clock a step runs to: the declared
/// chrome-side deadline ([`CHROME_USE_DECLARED_BUDGET`] for a verb mahbot forwards no `--timeout`
/// to, the forwarded deadline otherwise), or the product's own bound for a best-effort sub-step. It
/// lets a child answering at the very end of its deadline be reaped rather than killed, and it
/// holds every declaration one step below chrome-use's client tolerance. Reap allowance only: the
/// child's bytes are collected in their own [`spawn`] drain window.
pub(crate) const KILL_SLACK: Duration = Duration::from_secs(2);

/// chrome-use's own CLIENT TOLERANCE: its 30 s CDP command cap plus a 15 s socket-read margin — the
/// clock it works to for a call given no deadline. Never a declaration the product makes: a
/// declaration AT the tolerance makes chrome-use run out of tolerance instead of reporting, so its
/// session-unresponsive verdict (which stops the session) replaces its honest one. Every
/// declaration must stay strictly under it — that is what a forwarded `--timeout` is refused
/// against (see [`crate::chrome::cli`]) — and the declaration for a verb with no `--timeout` is
/// [`CHROME_USE_DECLARED_BUDGET`], with a probe's bound at or below it ([`probe_clocks`]).
pub(crate) const CHROME_USE_OWN_BUDGET: Duration = Duration::from_secs(45);

/// What the product DECLARES as the chrome-side deadline for a verb it forwards no `--timeout` to:
/// [`CHROME_USE_OWN_BUDGET`] less one [`KILL_SLACK`], so the tool always gives up on its own
/// declared deadline and reports its own verdict before the product's kill rides in.
pub(crate) const CHROME_USE_DECLARED_BUDGET: Duration =
    CHROME_USE_OWN_BUDGET.saturating_sub(KILL_SLACK);

/// One `session stop`'s bound: the session daemon's 8 s shutdown grace, then a reconnect reclaiming
/// the session's tabs under chrome-use's own 20 s — the ≈28 s end to end the live-verified
/// behaviours of [`crate::tools::chrome_daemon`] state. About twice that, so an estimate drifting
/// with a chrome-use release cannot charge a working stop as a failure. Shared by the CLI's stop
/// verbs and by [`crate::tools::chrome_release`]'s `ATTEMPT_TIMEOUT`, which holds it alongside a
/// reserve for each phase of browser reads: a stop the pass can no longer hold in full is not
/// spawned at all, so its names stay retries for a later pass rather than being concluded by reads.
pub(crate) const SESSION_STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// The relay self-heal chrome-use may run inside a call whose relay connection is gone: it kills
/// the stale daemon, re-registers the native host, launches the owner's real Chrome when needed,
/// and polls for up to its own `AGENT_BROWSER_RELAY_REVIVE_SECS` (45 s by default, pinned to this
/// number for a call that allows the self-heal) so the tool's clock and the product's kill
/// ([`kill_bound`]) agree on one recovery window.
pub(crate) const RELAY_RECOVERY_BUDGET: Duration = Duration::from_secs(45);

/// Whether a chrome-use call may run chrome-use's own relay self-heal.
///
/// [`Allowed`](Self::Allowed) for every call that carries out the agent's work — the tool's actions
/// and the CLI's steps — which must survive a relay hiccup the way chrome-use intends.
/// [`Suppressed`](Self::Suppressed) is ONLY for the product's own bounded extra work: the
/// snapshot/probe calls (`status`, `extension status`, `agent-browser` daemon probes, session
/// close/stop and the release path) and a composite operation's best-effort sub-steps.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CliRecovery {
    /// The call may run chrome-use's relay self-heal.
    Allowed,
    /// The call must not run chrome-use's relay self-heal.
    Suppressed,
}

impl CliRecovery {
    /// The recovery window the product's kill has to ride above under this policy.
    #[must_use]
    pub(crate) fn budget(self) -> Duration {
        match self {
            Self::Allowed => RELAY_RECOVERY_BUDGET,
            Self::Suppressed => Duration::ZERO,
        }
    }
}

/// The ONE derivation of the product's kill for a chrome-use call: the clock the call runs to (the
/// declared chrome-side deadline, or a product-side bound for a best-effort sub-step) plus whatever
/// relay recovery it may run plus [`KILL_SLACK`]. A probe takes no margin ([`probe_clocks`]).
/// Saturating, so an absurd caller-supplied bound cannot overflow.
#[must_use]
pub(crate) fn kill_bound(chrome_side: Duration, recovery: CliRecovery) -> Duration {
    chrome_side
        .saturating_add(recovery.budget())
        .saturating_add(KILL_SLACK)
}

/// The two clocks of one chrome-use call, derived together by [`clocks`]: the deadline DECLARED to
/// chrome-use and the product's own kill above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChromeCallClocks {
    /// The deadline declared to chrome-use (its `AGENT_BROWSER_DEFAULT_TIMEOUT`).
    pub(crate) chrome_side: Duration,
    /// The product's own kill for this call.
    pub(crate) kill: Duration,
}

/// Derive the two clocks of one chrome-use call from the deadline the caller declares for the verb:
/// `Some(d)` means the verb honours a forwarded `--timeout` of `d`, and that deadline IS the
/// chrome-side clock; `None` means the verb has no command-level deadline, so the declared clock is
/// [`CHROME_USE_DECLARED_BUDGET`], one [`KILL_SLACK`] under the tool's own tolerance. The kill is
/// [`kill_bound`] above whichever clock applies.
#[must_use]
pub(crate) fn clocks(declared: Option<Duration>, recovery: CliRecovery) -> ChromeCallClocks {
    let chrome_side = declared.unwrap_or(CHROME_USE_DECLARED_BUDGET);
    ChromeCallClocks {
        chrome_side,
        kill: kill_bound(chrome_side, recovery),
    }
}

/// The clocks of one of the product's own probes: the product's `bound` IS the kill, so a probe
/// that does not answer is reported as such instead of spending the recovery window, and that same
/// bound is DECLARED to chrome-use, capped at [`CHROME_USE_DECLARED_BUDGET`]. A deliberately short
/// liveness probe therefore declares the bound it really runs to; one whose bound sits above the
/// cap (the `session stop` paths) declares the cap.
#[must_use]
pub(crate) fn probe_clocks(bound: Duration) -> ChromeCallClocks {
    ChromeCallClocks {
        chrome_side: bound.min(CHROME_USE_DECLARED_BUDGET),
        kill: bound,
    }
}

/// Default `open` operation budget (20 s), kept as the CLI's `open --timeout` default: it bounds
/// the waits `open` drives — the error-page probe, the post-navigation settle / `--expect` wait,
/// and the content capture. NOT a kill bound on the navigation, and not chrome-use's `open`
/// deadline either: the navigate runs to [`CHROME_USE_DECLARED_BUDGET`], so `open` may legitimately
/// outlive this budget and a navigation still in progress is never cut off mid-report.
pub(crate) const DEFAULT_OPEN_TIMEOUT: Duration = Duration::from_secs(20);

/// Default condition deadline for `wait`/`expect` (8 s), the CLI's default when the caller gives no
/// `--timeout`; the tool's own deadlines sit in the same range. Deliberately far below
/// [`CHROME_USE_OWN_BUDGET`]: a forward declaring exactly chrome-use's client tolerance makes it
/// run out of tolerance instead of answering, replacing its honest "condition was not met" verdict
/// — which costs the session its tabs.
pub(crate) const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(8);

/// Prefix of per-invocation ephemeral CLI sessions — the ONLY CLI prefix the daemon-side start-up
/// sweep may close (orphan protection after crashes), and only for names [`cli_sessions`] does not
/// hold. Named `mahbot-chrome-<name>` sessions are never swept (cookie persistence is a feature).
pub(crate) const CLI_EPHEMERAL_PREFIX: &str = "mahbot-chrome-ephemeral-";

/// A navigation that ends here never committed — the tab stayed on the scratch `about:blank` page
/// (relay broken, navigation blocked). Keyed on the final URL only, so legitimately content-free
/// pages (image URLs, PDFs, canvas shells) are not false-flagged.
pub(crate) fn is_blank_page_url(url: &str) -> bool {
    let url = url.trim();
    url.is_empty() || url.starts_with("about:blank")
}

/// chrome-use's scratch error-page URL (live-verified on 1.5.101, reconfirmed on 1.5.106):
/// DNS/refused/unsafe-port navigations return rc=0 with success:true and commit to
/// `chrome-error://chromewebdata/` (a prefix match — no false positives on 404/5xx or captive
/// portals). The only false positive is a TLS-intercepting portal with a fake certificate — an
/// accepted trade-off.
pub(crate) fn is_chrome_error_page(url: &str) -> bool {
    url.trim().starts_with("chrome-error://")
}
/// How a navigation ended, read off the answer's committed URL: the tab never left its scratch
/// page, or Chrome committed to its own error page. `None` when the answer carries no committed
/// URL.
pub(crate) enum NavigationFailure {
    Blank,
    ErrorPage,
}
/// [`NavigationFailure`] of one `open` answer — the one reading both the
/// interactive tool and a rasterize capture refuse a navigation by.
pub(crate) fn navigation_failure(
    response: &crate::chrome::contract::ChromeResponse,
) -> Option<NavigationFailure> {
    let committed = response.data.as_ref()?.get("url")?.as_str()?;
    if is_blank_page_url(committed) {
        Some(NavigationFailure::Blank)
    } else if is_chrome_error_page(committed) {
        Some(NavigationFailure::ErrorPage)
    } else {
        None
    }
}
/// The net error code an error page's probe named, when it named one.
pub(crate) fn error_page_code(probe: &crate::chrome::contract::ChromeResponse) -> Option<String> {
    crate::chrome::contract::parse_error_page_probe(probe)
        .filter(|probe| probe.is_error_page)
        .and_then(|probe| probe.code)
}

/// Escape a string for a single-quoted JavaScript literal, shared by the CLI's count/eval shims and
/// the tool's `innerText` eval: a backslash is doubled before the apostrophe it precedes is
/// escaped, so text like `it\'s` can neither end the literal nor survive as an escape.
#[must_use]
pub(crate) fn escape_js_single_quoted(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A blank scratch page and a chrome error page are both recognised as page
    /// failures, while a real page is not.
    #[test]
    fn page_failure_predicates() {
        {
            let case = "blank page url predicate";
            // Scratch/about pages that never committed are blank-page failures.
            assert!(
                is_blank_page_url("about:blank"),
                "case {case}: a scratch page is a blank failure"
            );
            assert!(
                is_blank_page_url("about:blank#blocked"),
                "case {case}: a blocked scratch page is a blank failure"
            );
            assert!(
                is_blank_page_url(""),
                "case {case}: an empty URL is a blank failure"
            );
            // Real pages — even ones that render no text — are NOT blank failures
            // (image URLs, PDFs, canvas shells).
            assert!(
                !is_blank_page_url("https://example.com/image.png"),
                "case {case}: an image URL is a real page"
            );
            assert!(
                !is_blank_page_url("https://example.com/file.pdf"),
                "case {case}: a PDF is a real page"
            );
            assert!(
                !is_blank_page_url("about:srcdoc"),
                "case {case}: about:srcdoc is a committed page"
            );
        }

        {
            let case = "chrome error page predicate";
            assert!(
                is_chrome_error_page("chrome-error://chromewebdata/"),
                "case {case}: Chrome's own error page is an error page"
            );
            assert!(
                !is_chrome_error_page("https://example.com/"),
                "case {case}: a real page is not an error page"
            );
        }
    }

    /// A selector is escaped for a single-quoted JS literal: a backslash is
    /// doubled before the apostrophe it precedes is escaped, so text like
    /// `it\'s` can neither end the literal nor survive as an escape.
    #[test]
    fn escape_js_single_quoted_escapes_backslashes_and_quotes() {
        assert_eq!(
            escape_js_single_quoted("it's"),
            "it\\'s",
            "a quote is escaped"
        );
        assert_eq!(
            escape_js_single_quoted("a\\b"),
            "a\\\\b",
            "a backslash is doubled"
        );
        assert_eq!(
            escape_js_single_quoted("a\\'b"),
            "a\\\\\\'b",
            "the backslash before a quote is escaped too"
        );
    }
}

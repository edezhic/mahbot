//! Shared chrome automation core: the chrome-use CLI spawn mechanics and the
//! tolerant envelope contract that every chrome frontend needs.
//!
//! Frontends own everything user-facing: the interactive `chrome` tool
//! ([`crate::tools::chrome`]) owns its LLM-facing framing and error texts,
//! and the `mahbot chrome` CLI ([`cli`]) owns its stdout/exit-code emission.
//! The per-action descriptor registry ([`actions`]) is the ONE shared source
//! of LLM-facing action descriptions, rendered by both frontends (the tool's
//! parameter schema and the CLI help); outside that registry this module still
//! holds no other LLM-facing text. It also holds the chrome-use CLI spawning
//! ([`spawn`]), the tolerant `--json` response envelope parser ([`contract`]),
//! and the session-name/URL rules the tool and CLI share.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

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

/// chrome-use's scratch error-page URL (v1.5.101, live-verified): DNS/refused/
/// unsafe-port navigations return rc=0 with success:true and commit to
/// `chrome-error://chromewebdata/` (prefix match — the trailing slash exists;
/// no false positives on 404/5xx or real captive-portal pages). The only
/// false positive is a TLS-intercepting portal with a fake certificate —
/// accepted trade-off.
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

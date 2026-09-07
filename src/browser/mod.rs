//! Shared browser automation core: the chrome-use CLI spawn mechanics and the
//! tolerant envelope contract that every browser frontend needs.
//!
//! Frontends own everything user-facing: the interactive `browser` tool
//! ([`crate::tools::browser`]) owns its LLM-facing framing and error texts,
//! and the `mahbot browser` CLI ([`cli`]) owns its stdout/exit-code emission.
//! This module stays free of any LLM-facing text — it holds the chrome-use CLI
//! spawning ([`spawn`]), the tolerant `--json` response envelope parser
//! ([`contract`]), and the session-name/URL rules the tool and CLI share.

use crate::util::is_http_url;

pub(crate) mod cli;
pub(crate) mod contract;
pub(crate) mod spawn;

/// Name prefix of every session the `mahbot browser` CLI creates (named and
/// ephemeral). The interactive tool keeps `agent-tab-*`; link enrichment keeps
/// `link-enricher-*`.
pub(crate) const CLI_SESSION_PREFIX: &str = "mahbot-browser-";

/// Prefix of per-invocation ephemeral CLI sessions — the ONLY CLI prefix the
/// daemon-side sweep may close (orphan protection after crashes); named
/// `mahbot-browser-<name>` sessions are never swept (cookie persistence is a
/// feature).
pub(crate) const CLI_EPHEMERAL_PREFIX: &str = "mahbot-browser-ephemeral-";

/// Validate a URL is structurally safe to navigate to.
pub(crate) fn validate_url(url: &str) -> anyhow::Result<()> {
    let url = url.trim();

    if url.is_empty() {
        anyhow::bail!("URL cannot be empty");
    }

    // Block file:// — bypasses SSRF controls.
    if url.starts_with("file://") {
        anyhow::bail!("file:// URLs are not allowed in browser automation");
    }

    if !is_http_url(url) {
        anyhow::bail!("Only http:// and https:// URLs are allowed");
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
    fn url_validation_accepts_all_domains() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("https://docs.example.com").is_ok());
        assert!(validate_url("https://other.com").is_ok());
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
}

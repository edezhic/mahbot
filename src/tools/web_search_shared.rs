//! Shared caching infrastructure for web search tools.
//!
//! [`WebSearchTool`](super::web_search::WebSearchTool) uses this to avoid
//! duplicating the identical cache, argument-validation, and HTTP
//! error-handling logic across its backends.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

/// A cached search result with full text content.
#[derive(Debug, Clone)]
pub(crate) struct CachedResult {
    pub(crate) title: String,
    pub(crate) url: String,
    pub(crate) text: String,
}

/// In-memory cache shared between web search backends.
///
/// Holds search results that can later be expanded by ID.  Each agent
/// session gets its own cache, so entries are freed automatically when
/// the agent session ends.
pub(crate) struct WebSearchCache {
    cache: RwLock<HashMap<u64, CachedResult>>,
    next_id: AtomicU64,
}

impl WebSearchCache {
    pub(crate) fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Store a result and return its numeric expand ID.
    pub(crate) async fn cache_result(&self, title: String, url: String, text: String) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache
            .write()
            .await
            .insert(id, CachedResult { title, url, text });
        id
    }

    /// Retrieve the full text for a cached result by expand ID.
    pub(crate) async fn expand_result(&self, id: u64) -> anyhow::Result<String> {
        let cache_guard = self.cache.read().await;

        let cached = cache_guard.get(&id).ok_or_else(|| {
            anyhow::anyhow!(
                "No cached result with expand id: {id}. This id does not exist \
                 in the current session's cache (it may be from a previous search \
                 in a different context). Use `web_search` with `query` parameter \
                 to run a new search first."
            )
        })?;

        let text = if cached.text.is_empty() {
            "No full text available for this result.".to_string()
        } else {
            cached.text.clone()
        };

        Ok(format!(
            "# {}\n\n**URL:** {}\n\n{}",
            cached.title, cached.url, text
        ))
    }

    /// Return the current counter value for expand-ID range checking.
    pub(crate) fn next_counter(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }
}

// ── Shared HTTP error handling ─────────────────────────────────────────

/// Check a web search API response for HTTP errors and extract structured
/// JSON error messages when available.
///
/// Both Firecrawl and Exa return structured JSON errors — this reads the
/// response body, tries to extract an `"error"` field, and bails with a
/// descriptive message.  Returns the response on success (2xx) so the
/// caller can continue processing it.
pub(crate) async fn check_search_error(
    response: reqwest::Response,
) -> anyhow::Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = crate::util::http::read_error_body(response, "search").await;

    // APIs return structured JSON errors — try to extract for a better message.
    if let Ok(err_resp) = serde_json::from_str::<serde_json::Value>(&body)
        && let Some(error_msg) = err_resp.get("error").and_then(|e| e.as_str())
    {
        anyhow::bail!("search failed: {error_msg}");
    }

    anyhow::bail!("search failed with status {status}: {body}");
}

impl WebSearchCache {
    /// Parse and validate `web_search` tool arguments.
    ///
    /// Returns `(query, Option<expand_id>)` where `query` is the trimmed query
    /// string (empty if expanding).  Exactly one of `expand_id` is `Some` or
    /// `query` is non-empty.
    pub(crate) fn validate_execute_args<'a>(
        &self,
        args: &'a Value,
    ) -> anyhow::Result<(&'a str, Option<u64>)> {
        let has_query = args.get("query").and_then(|q| q.as_str()).is_some();
        let has_expand = args.get("expand").and_then(Value::as_u64).is_some();

        match (has_query, has_expand) {
            (true, true) => {
                anyhow::bail!(
                    "The web_search tool accepts either `query` (to run a new search) \
                     or `expand` (to expand a cached result), but not both at the same \
                     time. Please use one or the other."
                );
            }
            (false, false) => {
                anyhow::bail!(
                    "Missing required argument. Provide either `query` (to search \
                     the web) or `expand` (to expand a previous result by id)."
                );
            }
            (true, false) => {
                let query = super::get_opt_str(args, "query")
                    .map(str::trim)
                    .ok_or_else(|| anyhow::anyhow!("Invalid `query` value"))?;

                if query.len() <= 2 {
                    anyhow::bail!(
                        "Search query must be at least 3 characters long. \
                         Got: \"{query}\" ({} chars)",
                        query.len()
                    );
                }

                Ok((query, None))
            }
            (false, true) => {
                let id = super::get_opt_u64(args, "expand")
                    .ok_or_else(|| anyhow::anyhow!("Invalid `expand` value"))?;

                if id >= self.next_counter() {
                    anyhow::bail!(
                        "Invalid expand id: {id}. No search has been run yet in this \
                         session, or the id exceeds any generated value. Use `web_search` \
                         with `query` parameter to run a new search first.",
                    );
                }

                // query is empty, expand_id is Some
                Ok(("", Some(id)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_no_args() {
        let cache = WebSearchCache::new();
        let args = json!({});
        let result = cache.validate_execute_args(&args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("either `query`"),
            "Error should mention both args"
        );
    }

    #[test]
    fn test_validate_both_args() {
        let cache = WebSearchCache::new();
        let args = json!({"query": "test", "expand": 1});
        let result = cache.validate_execute_args(&args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not both"),
            "Error should mention not both"
        );
    }

    #[test]
    fn test_validate_query_too_short() {
        let cache = WebSearchCache::new();
        let args = json!({"query": "ab"});
        let result = cache.validate_execute_args(&args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("at least 3"),
            "Error should mention min length"
        );
    }

    #[test]
    fn test_validate_expand_invalid_future_id() {
        let cache = WebSearchCache::new();
        // Cache counter starts at 1, so any id >= 1 without a prior search fails
        let args = json!({"expand": 999_999_999});
        let result = cache.validate_execute_args(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid expand id"),
            "Error should contain 'Invalid expand id': {err}"
        );
    }

    #[tokio::test]
    async fn test_cache_and_expand_roundtrip() {
        let cache = WebSearchCache::new();
        let id = cache
            .cache_result(
                "Test Title".to_string(),
                "https://example.com".to_string(),
                "Full text content".to_string(),
            )
            .await;
        assert_eq!(id, 1, "first cached result should get id 1");

        let id2 = cache
            .cache_result(
                "Second".to_string(),
                "https://example.org".to_string(),
                "More content".to_string(),
            )
            .await;
        assert_eq!(id2, 2, "second cached result should get id 2");

        let expanded = cache.expand_result(1).await.unwrap();
        assert!(expanded.contains("Test Title"));
        assert!(expanded.contains("https://example.com"));
        assert!(expanded.contains("Full text content"));

        let expanded2 = cache.expand_result(2).await.unwrap();
        assert!(expanded2.contains("Second"));
        assert!(expanded2.contains("More content"));
    }

    #[tokio::test]
    async fn test_expand_missing_id() {
        let cache = WebSearchCache::new();
        let result = cache.expand_result(42).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("No cached result"),
            "Error should mention no cached result"
        );
    }
}

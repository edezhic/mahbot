use crate::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

/// A cached search result with full text content.
#[derive(Debug, Clone)]
struct CachedResult {
    title: String,
    url: String,
    text: String,
}

// ── Exa API types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ContentsOptions {
    text: bool,
    highlights: bool,
}

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    #[serde(rename = "useAutoprompt")]
    use_autoprompt: bool,
    #[serde(rename = "numResults")]
    num_results: usize,
    contents: ContentsOptions,
}

#[derive(Deserialize, Debug)]
struct SearchResult {
    title: Option<String>,
    url: String,
    text: Option<String>,
    highlights: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

// ── Tool ────────────────────────────────────────────────────────────────────────

/// Web search tool backed by the Exa API.
///
/// Uses `api.exa.ai` for search. Results are cached per-instance in-memory and
/// can be expanded via the `expand` parameter to retrieve full text content.
/// Each agent session gets its own tool instance, so caches are automatically
/// freed when the agent session ends.
pub struct WebSearchTool {
    exa_key: String,
    cache: RwLock<HashMap<u64, CachedResult>>,
    next_id: AtomicU64,
}

impl WebSearchTool {
    #[must_use]
    pub fn new(exa_key: String) -> Self {
        Self {
            exa_key,
            cache: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    async fn exa_search(&self, query: &str) -> anyhow::Result<String> {
        let res = crate::util::http::media_http_client()
            .post("https://api.exa.ai/search")
            .header("x-api-key", &self.exa_key)
            .json(&SearchRequest {
                query: query.to_string(),
                use_autoprompt: true,
                num_results: 10,
                contents: ContentsOptions {
                    text: true,
                    highlights: true,
                },
            })
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_else(|e| {
                tracing::warn!(?e, "Failed to read response body");
                String::new()
            });

            // Exa returns HTML block pages (Cloudflare, etc.) for 403 and
            // potentially other non-2xx statuses — replace the HTML dump
            // with a clean message to avoid polluting the LLM context.
            if status == reqwest::StatusCode::FORBIDDEN {
                anyhow::bail!("search provider blocked the request for unspecified reasons");
            }

            anyhow::bail!("Exa search failed with status {status}: {body}");
        }

        let search_resp = res.json::<SearchResponse>().await?;

        if search_resp.results.is_empty() {
            return Ok(format!("No results found for: {query}"));
        }

        let mut cache_guard = self.cache.write().await;

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Search results for: {query}"));

        for (i, result) in search_resp.results.iter().enumerate() {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let title = result.title.as_deref().unwrap_or("(no title)").to_string();

            let highlights = result
                .highlights
                .as_ref()
                .map(|h| h.join(" … "))
                .unwrap_or_default();

            // Cache with text for later expansion
            cache_guard.insert(
                id,
                CachedResult {
                    title: title.clone(),
                    url: result.url.clone(),
                    text: result.text.as_deref().unwrap_or_default().to_string(),
                },
            );

            lines.push(format!("{}. [{title}]({url})", i + 1, url = result.url));
            if !highlights.is_empty() {
                lines.push(format!("   > {highlights}"));
            }
            lines.push(format!("   `expand` id: `{id}`"));
        }

        Ok(lines.join("\n"))
    }

    async fn expand_result(&self, id: u64) -> anyhow::Result<String> {
        let cache_guard = self.cache.read().await;

        let cached = cache_guard
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("No cached result with expand id: {id}. This id does not exist in the current session's cache (it may be from a previous search in a different context). Use `web_search` with `query` parameter to run a new search first."))?;

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
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query"
                },
                "expand": {
                    "type": "integer",
                    "description": "The expand id of a previous search result to retrieve full cached content. Provide this to expand a result instead of running a new search."
                }
            },
            "oneOf": [
                {"required": ["query"]},
                {"required": ["expand"]}
            ]
        })
    }

    /// Bypass default truncation to preserve full search results and
    /// expanded article content. Web search results (especially expanded
    /// articles) can easily exceed 5K chars; truncation would lose
    /// critical information the LLM needs to read in its entirety.
    fn format_output(&self, output: &str) -> String {
        output.to_string()
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // external API query, no workspace mutations
    }

    async fn execute(
        &self,
        _ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let has_query = args.get("query").and_then(|q| q.as_str()).is_some();
        let has_expand = args
            .get("expand")
            .and_then(serde_json::Value::as_u64)
            .is_some();

        match (has_query, has_expand) {
            (true, true) => {
                anyhow::bail!(
                    "The web_search tool accepts either `query` (to run a new search) or `expand` (to expand a cached result), but not both at the same time. Please use one or the other."
                );
            }
            (false, false) => {
                anyhow::bail!(
                    "Missing required argument. Provide either `query` (to search the web) or `expand` (to expand a previous result by id)."
                );
            }
            (true, false) => {
                let query = super::get_opt_str(&args, "query")
                    .map(str::trim)
                    .ok_or_else(|| anyhow::anyhow!("Invalid `query` value"))?;

                if query.len() <= 2 {
                    anyhow::bail!(
                        "Search query must be at least 3 characters long. Got: \"{query}\" ({} chars)",
                        query.len()
                    );
                }

                tracing::info!("Searching web for: {}", query);
                let output = self.exa_search(query).await?;
                Ok(output)
            }
            (false, true) => {
                let id = super::get_opt_u64(&args, "expand")
                    .ok_or_else(|| anyhow::anyhow!("Invalid `expand` value"))?;

                let counter = self.next_id.load(Ordering::Relaxed);
                if id >= counter {
                    anyhow::bail!(
                        "Invalid expand id: {id}. No search has been run yet in this session, or the id exceeds any generated value. Use `web_search` with `query` parameter to run a new search first.",
                    );
                }

                tracing::info!("Expanding result id: {}", id);
                let output = self.expand_result(id).await?;
                Ok(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_ws;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_execute_no_args() {
        let tool = WebSearchTool::new("test_key".to_string());
        let result = tool.execute(&test_ws(PathBuf::new()), json!({})).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("either `query`"),
            "Error should mention both args"
        );
    }

    #[tokio::test]
    async fn test_execute_both_args() {
        let tool = WebSearchTool::new("test_key".to_string());
        let result = tool
            .execute(
                &test_ws(PathBuf::new()),
                json!({"query": "test", "expand": 1}),
            )
            .await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not both"),
            "Error should mention not both"
        );
    }

    #[tokio::test]
    async fn test_execute_query_too_short() {
        let tool = WebSearchTool::new("test_key".to_string());
        let result = tool
            .execute(&test_ws(PathBuf::new()), json!({"query": "ab"}))
            .await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("at least 3"),
            "Error should mention min length"
        );
    }

    #[tokio::test]
    async fn test_execute_expand_invalid_future_id() {
        // Each tool instance starts with counter=1, so any id >= 1 without a prior
        // search will fail the range check.
        let tool = WebSearchTool::new("test_key".to_string());
        let result = tool
            .execute(&test_ws(PathBuf::new()), json!({"expand": 999_999_999}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid expand id") || err.contains("expand"),
            "Error should mention invalid expand id: {err}"
        );
    }

    #[test]
    fn test_parameters_schema() {
        let tool = WebSearchTool::new("test_key".to_string());
        let schema = tool.parameters_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema.get("oneOf").is_some());
        assert!(schema["properties"]["query"]["type"] == "string");
        assert!(schema["properties"]["expand"]["type"] == "integer");
    }
}

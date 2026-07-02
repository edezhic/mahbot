use crate::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::web_search_shared::{WebSearchCache, check_search_error};

// ── Backend selection ───────────────────────────────────────────────────────────

/// Backend provider for web search.
pub enum WebSearchBackend {
    /// Firecrawl API (`api.firecrawl.dev/v2/search`).
    Firecrawl { key: String },
    /// Exa API (`api.exa.ai/search`).
    Exa { key: String },
}

// ── Firecrawl API types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ScrapeOptions {
    formats: Vec<String>,
    #[serde(rename = "onlyMainContent")]
    only_main_content: bool,
}

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    limit: usize,
    #[serde(rename = "scrapeOptions")]
    scrape_options: ScrapeOptions,
}

#[derive(Deserialize, Debug)]
struct WebResult {
    title: Option<String>,
    url: String,
    markdown: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize, Debug)]
struct SearchData {
    web: Vec<WebResult>,
}

#[derive(Deserialize, Debug)]
struct SearchResponse {
    success: bool,
    data: SearchData,
}

// ── Exa API types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ExaContentsOptions {
    text: bool,
}

#[derive(Serialize)]
struct ExaSearchBody {
    query: String,
    #[serde(rename = "numResults")]
    num_results: usize,
    contents: ExaContentsOptions,
}

#[derive(Deserialize, Debug)]
struct ExaResult {
    title: Option<String>,
    url: String,
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ExaSearchResponse {
    results: Vec<ExaResult>,
}

// ── Tool ────────────────────────────────────────────────────────────────────────

/// Web search tool backed by a configurable backend (Firecrawl or Exa).
///
/// Dispatches to the appropriate API based on [`WebSearchBackend`]. Results are
/// cached per-instance in-memory and can be expanded via the `expand` parameter
/// to retrieve full text content.  Each agent session gets its own tool instance,
/// so caches are automatically freed when the agent session ends.
pub struct WebSearchTool {
    backend: WebSearchBackend,
    cache: WebSearchCache,
}

impl WebSearchTool {
    #[must_use]
    pub fn new(backend: WebSearchBackend) -> Self {
        Self {
            backend,
            cache: WebSearchCache::new(),
        }
    }

    async fn firecrawl_search(&self, query: &str) -> anyhow::Result<String> {
        let WebSearchBackend::Firecrawl { key } = &self.backend else {
            unreachable!("firecrawl_search called for non-Firecrawl backend");
        };

        let res = crate::util::http::media_http_client()
            .post("https://api.firecrawl.dev/v2/search")
            .header("Authorization", format!("Bearer {key}"))
            .json(&SearchRequest {
                query: query.to_string(),
                limit: 10,
                scrape_options: ScrapeOptions {
                    formats: vec!["markdown".to_string()],
                    only_main_content: true,
                },
            })
            .send()
            .await?;

        let res = check_search_error(res).await?;

        let search_resp = res.json::<SearchResponse>().await?;

        if !search_resp.success || search_resp.data.web.is_empty() {
            return Ok(format!("No results found for: {query}"));
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Search results for: {query}"));

        for (i, result) in search_resp.data.web.iter().enumerate() {
            let title = result.title.as_deref().unwrap_or("(no title)").to_string();
            let description = result.description.as_deref().unwrap_or("");

            let id = self
                .cache
                .cache_result(
                    title.clone(),
                    result.url.clone(),
                    result.markdown.clone().unwrap_or_default(),
                )
                .await;

            lines.push(format!("{}. [{title}]({url})", i + 1, url = result.url));
            if !description.is_empty() {
                lines.push(format!("   > {description}"));
            }
            lines.push(format!("   `expand` id: `{id}`"));
        }

        Ok(lines.join("\n"))
    }

    async fn exa_search(&self, query: &str) -> anyhow::Result<String> {
        let WebSearchBackend::Exa { key } = &self.backend else {
            unreachable!("exa_search called for non-Exa backend");
        };

        let res = crate::util::http::media_http_client()
            .post("https://api.exa.ai/search")
            .header("x-api-key", key)
            .json(&ExaSearchBody {
                query: query.to_string(),
                num_results: 10,
                contents: ExaContentsOptions { text: true },
            })
            .send()
            .await?;

        let res = check_search_error(res).await?;

        let search_resp = res.json::<ExaSearchResponse>().await?;

        if search_resp.results.is_empty() {
            return Ok(format!("No results found for: {query}"));
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Search results for: {query}"));

        for (i, result) in search_resp.results.iter().enumerate() {
            let title = result.title.as_deref().unwrap_or("(no title)").to_string();

            let id = self
                .cache
                .cache_result(
                    title.clone(),
                    result.url.clone(),
                    result.text.clone().unwrap_or_default(),
                )
                .await;

            lines.push(format!("{}. [{title}]({url})", i + 1, url = result.url));
            lines.push(format!("   `expand` id: `{id}`"));
        }

        Ok(lines.join("\n"))
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

    /// Preserve full search results — don't truncate with default 5K limit.
    fn format_output(&self, output: &str) -> String {
        output.to_string()
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        false // web search has no workspace side effects
    }

    async fn execute(
        &self,
        _ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let (query, expand_id) = self.cache.validate_execute_args(&args)?;

        if let Some(id) = expand_id {
            tracing::info!("Expanding result id: {}", id);
            return self.cache.expand_result(id).await;
        }

        match &self.backend {
            WebSearchBackend::Firecrawl { .. } => {
                tracing::info!("Searching web (Firecrawl) for: {}", query);
                self.firecrawl_search(query).await
            }
            WebSearchBackend::Exa { .. } => {
                tracing::info!("Searching web (Exa) for: {}", query);
                self.exa_search(query).await
            }
        }
    }
}

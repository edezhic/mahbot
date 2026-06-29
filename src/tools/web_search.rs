use crate::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::web_search_shared::{WebSearchCache, check_search_error};

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

// ── Tool ────────────────────────────────────────────────────────────────────────

/// Web search tool backed by the Firecrawl API.
///
/// Uses `api.firecrawl.dev/v2/search` for search. Results are cached
/// per-instance in-memory and can be expanded via the `expand` parameter
/// to retrieve full text content.  Each agent session gets its own tool
/// instance, so caches are automatically freed when the agent session ends.
pub struct WebSearchTool {
    firecrawl_key: String,
    cache: WebSearchCache,
}

impl WebSearchTool {
    #[must_use]
    pub fn new(firecrawl_key: String) -> Self {
        Self {
            firecrawl_key,
            cache: WebSearchCache::new(),
        }
    }

    async fn firecrawl_search(&self, query: &str) -> anyhow::Result<String> {
        let res = crate::util::http::media_http_client()
            .post("https://api.firecrawl.dev/v2/search")
            .header("Authorization", format!("Bearer {}", &self.firecrawl_key))
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

        // When changing result formatting here, also check
        // `src/tools/exa_search.rs` — the Exa loop follows the same pattern
        // but differs in fields (no `description`, uses `text` vs `markdown`).
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
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        WebSearchCache::parameters_schema()
    }

    fn format_output(&self, output: &str) -> String {
        WebSearchCache::format_output(output)
    }

    fn side_effects(&self, _args: &serde_json::Value) -> bool {
        WebSearchCache::side_effects()
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

        tracing::info!("Searching web (Firecrawl) for: {}", query);
        self.firecrawl_search(query).await
    }
}

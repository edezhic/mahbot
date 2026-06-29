use crate::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::web_search_shared::{WebSearchCache, check_search_error};

// ── Exa API types ─────────────────────────────────────────────────────────

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

// ── Tool ──────────────────────────────────────────────────────────────────

/// Web search tool backed by the Exa API.
///
/// Uses `api.exa.ai/search` for search. Results are cached
/// per-instance in-memory and can be expanded via the `expand` parameter
/// to retrieve full text content.  Each agent session gets its own tool
/// instance, so caches are automatically freed when the agent session ends.
pub struct ExaSearchTool {
    exa_key: String,
    cache: WebSearchCache,
}

impl ExaSearchTool {
    #[must_use]
    pub fn new(exa_key: String) -> Self {
        Self {
            exa_key,
            cache: WebSearchCache::new(),
        }
    }

    async fn exa_search(&self, query: &str) -> anyhow::Result<String> {
        let res = crate::util::http::media_http_client()
            .post("https://api.exa.ai/search")
            .header("x-api-key", &self.exa_key)
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

        // When changing result formatting here, also check
        // `src/tools/web_search.rs` — the Firecrawl loop follows the same
        // pattern but includes a `description` field and uses `markdown`.
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
impl Tool for ExaSearchTool {
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

        tracing::info!("Searching web (Exa) for: {}", query);
        self.exa_search(query).await
    }
}

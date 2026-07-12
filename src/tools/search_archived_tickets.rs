//! Search archived tickets via hybrid FTS + semantic vector search.
//!
//! Available only to the Manager role. Searches the `is_archived = 1` subset
//! of the tickets table. Returns up to 10 matches as `[phase] id: title`
//! lines, sorted by RRF score. The Manager then uses [`crate::tools::ticket::GetTicketTool`] on the
//! returned IDs for full details.

use crate::Workspace;
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;
use tracing::debug;

/// Search archived tickets only, by keyword + semantic similarity.
///
/// Uses the board's FTS index (`BoardStore::search_archived_by_fts`) and local
/// embeddings (`embedder::embed_query()`) with RRF merge (`vector::hybrid_merge`).
pub struct SearchArchivedTicketsTool;

#[async_trait]
impl Tool for SearchArchivedTicketsTool {
    fn name(&self) -> &'static str {
        "search_archived_tickets"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "query": {
                    "type": "string",
                    "description": "Search query — matches ticket titles via keyword + semantic search"
                }
            }),
            &["query"],
        )
    }

    fn side_effects(&self) -> bool {
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> Result<String> {
        let query = super::get_str(&args, "query")?;
        let board = crate::board::store();

        let (fts_results, vector_results) = tokio::join!(
            Self::fts_search(board, query),
            Self::vector_search(board, query),
        );
        let fts_results = fts_results?;
        let vector_results = vector_results?;

        let merged = crate::vector::hybrid_merge(vector_results, fts_results);

        let top_ids: Vec<String> = merged.into_iter().take(10).map(|r| r.id).collect();

        if top_ids.is_empty() {
            return Ok("No archived tickets found matching the query.".to_string());
        }

        Self::format_results(board, &top_ids).await
    }
}

impl SearchArchivedTicketsTool {
    /// FTS keyword search. Delegates to [`crate::board::BoardStore::search_archived_by_fts`]
    /// which sanitizes the query internally. A graceful failure returns an
    /// empty vec (fall through to vector search).
    async fn fts_search(
        board: &crate::board::BoardStore,
        query: &str,
    ) -> Result<Vec<(String, f32)>> {
        let results = board.search_archived_by_fts(query, 50).await?;
        Ok(results
            .into_iter()
            .map(|(id, score)| {
                #[allow(clippy::cast_possible_truncation)]
                (id, score as f32)
            })
            .collect())
    }

    /// Semantic vector search: embed the query, then rank all archived ticket
    /// embeddings by cosine similarity.
    async fn vector_search(
        board: &crate::board::BoardStore,
        query: &str,
    ) -> Result<Vec<(String, f32)>> {
        let Some(q_emb) = crate::embedder::embed_query(query) else {
            debug!("Embedding model failed — skipping vector search");
            return Ok(Vec::new());
        };

        let candidates = board.list_archived_with_embeddings().await?;

        if candidates.is_empty() {
            debug!("All archived tickets have NULL embeddings — vector search skipped");
            return Ok(Vec::new());
        }

        let mut scored: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|(id, emb)| {
                let sim = crate::vector::cosine_similarity(&q_emb, &emb);
                (id, sim)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(50);
        Ok(scored)
    }

    /// Format the top-N results for display.
    ///
    /// Fetches id/title/phase via [`crate::board::BoardStore::get_tickets_by_ids`]
    /// and formats them in rank order. At N ≤ 10 a linear scan suffices.
    async fn format_results(
        board: &crate::board::BoardStore,
        top_ids: &[String],
    ) -> Result<String> {
        let tickets = board
            .get_tickets_by_ids(top_ids, crate::board::LoadComments::No)
            .await?;

        let mut output = String::from("Search results (top 10, highest score first):\n");
        for id in top_ids {
            if let Some(ticket) = tickets.iter().find(|t| t.id == *id) {
                let _ = writeln!(output, "  [{}] {}: {}", ticket.phase, id, ticket.title);
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::board::BoardStore;
    use crate::open_test_store;
    use crate::util::test::make_ticket;

    /// Test that the tool-layer fts_search wrapper correctly converts
    /// f64 scores from the board to f32.
    #[tokio::test]
    async fn test_fts_search_returns_floats() {
        let (store, _tmp) = open_test_store!(BoardStore, "board");
        // Create and archive a ticket so FTS has something to find
        let ws = crate::workspace::test_ws("ws");
        let id = make_ticket(
            &store,
            &ws,
            "UniqueSearchableTitleXYZ",
            crate::board::TicketPhase::Done,
        )
        .await;
        store.set_archived(&id).await.expect("archive");

        let results = super::SearchArchivedTicketsTool::fts_search(&store, "UniqueSearchable")
            .await
            .expect("fts_search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        // f64→f32 conversion should succeed without precision loss for scores
        assert!(results[0].1.is_finite(), "score should be a finite f32");
    }

    /// Test that format_results formats real ticket data correctly.
    #[tokio::test]
    async fn test_format_results_formats_correctly() {
        let (store, _tmp) = open_test_store!(BoardStore, "board");
        let ws = crate::workspace::test_ws("ws");
        let id_a = make_ticket(&store, &ws, "Alpha ticket", crate::board::TicketPhase::Done).await;
        store.set_archived(&id_a).await.expect("archive");
        let id_b = make_ticket(
            &store,
            &ws,
            "Beta ticket",
            crate::board::TicketPhase::Cancelled,
        )
        .await;
        store.set_archived(&id_b).await.expect("archive");

        let ids = vec![id_a.clone(), id_b.clone()];
        let result = super::SearchArchivedTicketsTool::format_results(&store, &ids)
            .await
            .expect("format_results");
        let expected_header = "Search results (top 10, highest score first):\n";
        assert!(result.starts_with(expected_header));
        assert!(
            result.contains(&format!("[done] {id_a}: Alpha ticket")),
            "should contain first ticket line"
        );
        assert!(
            result.contains(&format!("[cancelled] {id_b}: Beta ticket")),
            "should contain second ticket line"
        );
    }
}

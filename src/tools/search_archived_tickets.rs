//! Search archived tickets by text in their titles.
//!
//! Available only to the Manager role. Searches the whole installation's
//! `is_archived = 1` subset of the tickets table — every workspace. Returns at
//! most [`RESULT_LIMIT`] matches as `[workspace] [phase] id: title` lines,
//! best-scoring first. The Manager then uses
//! [`crate::tools::ticket::GetTicketTool`] on the returned IDs for full details.

use crate::Workspace;
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write;

/// Matches returned for one query — the one number the search and its answer
/// header both use, so the reported count can never overstate what came back.
const RESULT_LIMIT: usize = 10;

/// Search archived tickets only, by words in their titles.
///
/// Uses the board's FTS index (`BoardStore::search_archived_by_fts`).
/// Installation-wide — the search spans every workspace, and each result names
/// its workspace so a hit can be read wherever it was filed.
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
                    "description": "Search query — matched against ticket titles"
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
        let board = crate::pipeline::board::store();

        let top_ids = board.search_archived_by_fts(query, RESULT_LIMIT).await?;
        if top_ids.is_empty() {
            return Ok("No archived tickets found matching the query.".to_string());
        }

        Self::format_results(board, &top_ids).await
    }
}

impl SearchArchivedTicketsTool {
    /// Format the results in rank order, each line naming the workspace the hit
    /// was filed in.
    ///
    /// Fetches id/title/phase/workspace via
    /// [`crate::pipeline::board::BoardStore::get_tickets_by_ids`]. At N ≤ 10 a
    /// linear scan suffices.
    async fn format_results(
        board: &crate::pipeline::board::BoardStore,
        top_ids: &[String],
    ) -> Result<String> {
        let tickets = board
            .get_tickets_by_ids(top_ids, crate::pipeline::board::LoadComments::No)
            .await?;

        let mut output = format!(
            "Search results (highest score first, {} of at most {RESULT_LIMIT}):\n",
            top_ids.len()
        );
        for id in top_ids {
            if let Some(ticket) = tickets.iter().find(|t| t.id == *id) {
                let _ = writeln!(
                    output,
                    "  [{}] [{}] {}: {}",
                    ticket.workspace_name, ticket.phase, id, ticket.title
                );
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::open_test_store;
    use crate::pipeline::board::{BoardStore, TicketPhase};
    use crate::util::test::make_ticket;

    /// Archive search must rank by relevance and must never let live tickets
    /// consume the limit: the archive predicate is applied outside the ranked
    /// scope, and that scope is unbounded. The fixtures pin both halves — the
    /// archived titles are seeded weakest-first, so results in insertion (scan)
    /// order cannot match, and the live titles outrank every archived one, so a
    /// limit applied before the archive filter would return nothing. The GUI's
    /// `search_by_fts` is the known-good reference ranking.
    #[tokio::test]
    async fn archived_search_ranks_by_score_and_is_not_crowded_out_by_live_tickets() {
        let (store, _tmp) = open_test_store!(BoardStore, "board");
        let ws = crate::workspace::test_ws("ws");

        let mut archived_ids = Vec::new();
        for title in ["migration one", "migration"] {
            let id = make_ticket(&store, &ws, title, TicketPhase::Done).await;
            store.set_archived(&id).await.expect("archive");
            archived_ids.push(id);
        }
        for title in ["migration migration migration", "migration migration"] {
            make_ticket(&store, &ws, title, TicketPhase::Backlog).await;
        }

        let ranked = store
            .search_by_fts("migration", 50)
            .await
            .expect("gui search");
        assert!(
            !archived_ids.contains(&ranked[0].id),
            "a live ticket must outrank every archived hit or the limit cannot bite"
        );
        let expected: Vec<String> = ranked
            .into_iter()
            .map(|ticket| ticket.id)
            .filter(|id| archived_ids.contains(id))
            .collect();

        let found = store
            .search_archived_by_fts("migration", 2)
            .await
            .expect("archive search");
        assert_eq!(found, expected, "archive hits must keep the FTS order");
        assert_ne!(
            found, archived_ids,
            "results must not come back in insertion (scan) order"
        );
    }
}

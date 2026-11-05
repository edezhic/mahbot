//! `Session` — the single source of truth for conversation persistence, owning the in-memory
//! history of the current turn.
//!
//! A turn: `Session::default()`; [`init`](Session::init) loads the persisted history and the token
//! length; the turn's messages land via [`append_turn_message`](Session::append_turn_message) after
//! the universal resume-completion step; the agent loop pushes and persists during tool rounds; and
//! [`finalize`](Session::finalize) persists the final assistant response.
//!
//! An empty unpersisted tail at finalize is a no-op (everything is already durable) and is surfaced
//! to the caller to log.

use std::fmt::Write;
use std::sync::Arc;

use anyhow::Result;
use arc_swap::ArcSwap;
use futures_util::future::join_all;

use crate::pipeline::board::BOARD;
use crate::prompt::{build_workspace_context, format_ticket_block, load_prompt, substitute};
use crate::runtime_events::RuntimeEvent;
use crate::workspace::{WORKSPACES, truncate_workspace_notes};

use crate::agent::skills;
use crate::automations::Automation;
use crate::custom_role::{CUSTOM_ROLES, CustomRole};
use crate::tools::active_models::{ModelKind, ModelSnapshot};
use crate::{ChatMessage, ChatRole, Role, Workspace};

use super::TranscriptSnapshot;

/// Coordinates session persistence and prompt building across a single agent turn.
#[derive(Default)]
pub(crate) struct Session {
    /// In-memory history for the current turn: system prompt, conversation and the latest message.
    history: Vec<ChatMessage>,
    /// Leading `history` entries already persisted: `history[..persisted_len]` mirrors the DB
    /// contiguously and the rest is exactly the unpersisted tail (failed appends, tool traffic
    /// awaiting commit, the final answer). Every successful persist advances this prefix over the
    /// span it wrote.
    persisted_len: usize,
    /// Real provider-reported session length (input + output tokens of the most recent successful
    /// agent call), loaded at [`init`](Self::init) and updated after every agent call. `None` for
    /// sessions that never recorded a value (approved no-backfill semantics) and for sessions whose
    /// value the last compaction cleared.
    token_length: Option<u64>,
    /// Pictures the conversation holds — [`crate::session::image_cap`]'s gate, so the common
    /// at-or-below-cap request costs no history walk. Kept in step with `history` at every mutation
    /// and recorded in `session_metadata.image_count`; established from the loaded history on first
    /// use (`None` there = never counted).
    image_count: usize,
    /// Shared lock-free holder this session publishes live transcript snapshots into; `None` when
    /// the session is not attached to a running agent (tests, detached sessions). Set by
    /// [`attach_transcript`](Self::attach_transcript) from `Agent::new`.
    transcript: Option<Arc<ArcSwap<TranscriptSnapshot>>>,
    /// The launch this session runs under: a custom role's authorial definition plus resolved
    /// branch address, or a worker branch's address. `Some` for a branch session, `None` for
    /// built-in roles.
    pub(crate) launch: Option<crate::Launch>,
}

/// Outcome of a [`Session::finalize`] call. The guard is deliberately logging-free: whether an
/// empty unpersisted tail is expected (a drain cut after the last commit) or anomalous depends on
/// shutdown state this module does not track, so the caller logs with full attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalizeOutcome {
    /// The unpersisted tail was flushed and the persisted prefix advanced.
    Flushed,
    /// Nothing was unpersisted — everything committed is already durable; the caller decides
    /// whether the no-op is expected (drain) or a genuine anomaly.
    NoUnpersistedTail,
}

/// Outcome of [`Session::rewrite_last_user_message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RewriteOutcome {
    /// The message was durably rewritten and the in-memory entry swapped.
    Rewritten,
    /// The most recent User-role message is in the unpersisted tail, where a positional UPDATE
    /// would hit a different row — a conservative no-op that leaves the caller its original error
    /// path.
    UnpersistedTailNoop,
}

/// The unresolved tool calls of the session's LAST tool-call frame plus the frame-level context the
/// sleep-delivery guard needs on resume: whether the frame carried non-empty raw text (never
/// delivered in a tool round) and how many calls the round bundled.
pub(crate) struct PendingToolFrame {
    pub calls: Vec<crate::ToolCall>,
    pub has_text: bool,
    pub call_count: usize,
}

/// Render the fixed system block a custom role's prompt opens with: this launch's folder and the
/// system. Nothing else — the role itself is the administrator's authorial text
/// ([`crate::Launch::prompt`]), which follows as the second system part. A launch in a different
/// folder re-renders it (see [`Session::append_turn_message`]).
fn custom_role_block(ws: &Workspace) -> String {
    substitute(
        &load_prompt("context/custom_role.md"),
        &[
            ("{{workspace}}", &ws.as_path().display().to_string()),
            ("{{operating_system}}", std::env::consts::OS),
        ],
    )
}

impl Session {
    // Begin a turn: load the persisted history and the real session length, record the persisted
    // prefix and publish the initial transcript snapshot. The turn's user message — and the
    // first-turn system prompt caching that goes with it — comes separately via
    // [`append_turn_message`](Self::append_turn_message), after the orchestrator's
    // resume-completion step has settled any dangling tool calls.

    pub(crate) async fn init(&mut self, agent_id: &str) -> Result<()> {
        self.history = crate::session::store().load(agent_id).await;

        // Load the real session length so `maybe_summarize` and the Running Agents card see it from
        // the start of the turn; `None` (never recorded) is treated as below the summarization
        // threshold.
        self.token_length = crate::session::store().get_token_length(agent_id).await;

        // The recorded picture count gates the request cap; a session that predates the column (or
        // whose count a truncate cleared) is counted here once, so the per-round check stays a
        // comparison.
        // The count is established from a strict read: `load` above fails open to an empty history,
        // and recording that as the count would let every later over-cap request through.
        self.image_count = match crate::session::store().get_image_count(agent_id).await {
            Some(count) => count,
            None => match crate::session::store().load_checked(agent_id).await {
                Ok(history) => {
                    let count = crate::session::image_cap::count(&history);
                    // Record only for a session that has rows: the write would otherwise create a
                    // `session_metadata` row for an empty session (and resurrect one `/clear`
                    // removed). The first append records the count with its rows.
                    if !history.is_empty()
                        && let Err(e) = crate::session::store()
                            .set_image_count(agent_id, count)
                            .await
                    {
                        // Fail open: the in-memory count is still right for this turn, and the next
                        // init retries the record.
                        tracing::warn!(
                            agent_id = %agent_id,
                            error = %e,
                            "Failed to record the session's picture count"
                        );
                    }
                    count
                }
                // Unrecorded: the next init establishes it rather than trusting a partial read.
                Err(e) => {
                    tracing::warn!(
                        agent_id = %agent_id,
                        error = %e,
                        "Failed to read the history for the session's picture count"
                    );
                    crate::session::image_cap::count(&self.history)
                }
            },
        };

        // All loaded history above is persisted — record the prefix so finalize appends only new
        // output.
        self.persisted_len = self.history.len();

        // Publish the initial snapshot so the Running Agents card shows the loaded state from the
        // start.
        self.publish_transcript();

        Ok(())
    }

    /// Append a turn's messages and session context into the loaded session. A turn answers one
    /// message, or several when an urgent message cut the previous turn: each lands as its own user
    /// message in arrival order. The per-turn blocks — personal-workspace report, model-change
    /// line, activity line — ride the FIRST message only, so a turn carrying several never repeats
    /// them.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn append_turn_messages(
        &mut self,
        agent_id: &str,
        messages: &[&str],
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        channel: &str,
        user_name: &str,
        round_ts: Option<&str>,
        is_admin: bool,
    ) -> Result<()> {
        let present: Vec<&str> = messages
            .iter()
            .copied()
            .filter(|message| !message.is_empty())
            .collect();
        for (at, message) in present.iter().enumerate() {
            self.append_turn_message(
                agent_id,
                message,
                ws,
                role,
                ticket,
                channel,
                user_name,
                round_ts,
                is_admin,
                at == 0,
            )
            .await?;
        }
        Ok(())
    }

    /// Append this turn's user message and session context. The append half of [`init`](Self::init)
    /// lives here so the orchestrator can run the resume-completion step in between: the session
    /// loads WITHOUT the new message, the dangling tool calls are settled after their frame, and
    /// only then does the message land. An empty `msg` is a no-op. `turn_first` says whether this
    /// message is its turn's first (the per-turn blocks ride only that one); `round_ts` pins its
    /// timestamp, so parallel members write byte-identical first messages, and `None` stamps now.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn append_turn_message(
        &mut self,
        agent_id: &str,
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        channel: &str,
        user_name: &str,
        round_ts: Option<&str>,
        is_admin: bool,
        turn_first: bool,
    ) -> Result<()> {
        if !msg.is_empty() {
            let is_new = self.history.is_empty();

            // The one composition point for both branches below: the activity line at the very
            // head, then the personal-workspace report, then a later Assistant turn's machine
            // change-info (a first turn's system prompt holds the models instead). The
            // session-start `<workspace-structure>` block cannot stand in for the report: it is a
            // shape — folder names with counts, never a path — so it cannot say what the user just
            // touched. Every block here is per-turn: on a turn carrying several messages they ride
            // the first one only.
            let (content, new_snapshot) =
                if turn_first && !is_new && matches!(role, Role::Assistant) {
                    Self::prepend_model_change(agent_id, msg, user_name).await
                } else {
                    (msg.to_string(), None)
                };
            let content = if turn_first {
                crate::session::personal_changes::prepend_changes(&content, ws, role).await
            } else {
                content
            };
            let (content, activity) = if turn_first {
                super::activity::prepend(&content, agent_id, role, is_admin).await
            } else {
                (content, None)
            };

            if is_new {
                let (msgs, snapshot) = Self::build_turn_messages(
                    &content,
                    ws,
                    role,
                    ticket,
                    round_ts,
                    is_admin,
                    user_name,
                    self.launch.as_ref(),
                )
                .await;

                // Batch-write all messages and session context atomically. A launch session's
                // leading context block is `msgs` minus the one turn message, and its length is
                // recorded in the SAME transaction so a later folder change rebuilds the block from
                // what was WRITTEN, not from live workspace state.
                let meta =
                    super::BatchMetadata::context(channel, user_name, &ws.name, role.as_str())
                        .block_len(self.launch.as_ref().map(|_| msgs.len() - 1));
                crate::session::store()
                    .batch_append_with_context(agent_id, &msgs, meta)
                    .await?;
                self.take_persisted(msgs);

                if matches!(role, Role::Assistant) {
                    Self::persist_active_models_snapshot(agent_id, &snapshot).await;
                }
            } else {
                // A launch's folder is named on EVERY use, so the cached opening context block must
                // follow the folder the branch now works in — a custom role's or a worker branch's
                // (the block is the session's leading system run); the conversation after it is
                // never touched. The recorded folder is read BEFORE the append below rewrites it:
                // it is what tells a change from a repeat use of the same folder.
                if self.launch.is_some()
                    && let Some(previous) = crate::session::store()
                        .session_workspace_name(agent_id)
                        .await
                    && previous != ws.name
                {
                    // Refreshed for EVERY folder change, including one whose previous folder no
                    // longer resolves: the block's length was recorded when it was written, so
                    // nothing about the old folder is needed.
                    self.refresh_context_block(agent_id, ws, role, ticket, is_admin, user_name)
                        .await?;
                    // The launch landed in a different folder than the one recorded, so the engine
                    // the branch held for that folder is now unreferenced — release it (the new
                    // folder's engine is created on demand).
                    crate::workspace_index::release_path_named_engine(&previous);
                }
                // Caching: the system prompt is NOT rebuilt on every turn; the session DB caches
                // the full first-turn message set. The rebuild paths are the folder change above
                // and `apply_summary` below.
                let user_msg = crate::session::user_msg_with_ts(&content, round_ts);
                // Append the user message and update session context atomically.
                crate::session::store()
                    .append_with_context(
                        agent_id,
                        &user_msg,
                        super::BatchMetadata::context(channel, user_name, &ws.name, role.as_str()),
                    )
                    .await?;
                // Refresh the change-detection baseline only after the message carrying the
                // change-info is persisted — otherwise a failed append would lose the change-info
                // while advancing the baseline, so a retry could never communicate the switch.
                // Residual window: a successful append with a failed baseline write leaves a stale
                // baseline, so the next message re-fires the same change-info — benign and
                // self-healing.
                if let Some(snapshot) = new_snapshot {
                    Self::persist_active_models_snapshot(agent_id, &snapshot).await;
                }
                self.take_persisted(std::iter::once(user_msg));
            }
            // Recorded only once the append landed, so a failed write cannot silently swallow the
            // change.
            if let Some(location) = activity
                && let Err(e) = crate::session::store()
                    .set_user_activity(agent_id, location.as_str())
                    .await
            {
                tracing::warn!(
                    agent_id = %agent_id,
                    error = %e,
                    "Failed to record the user-activity reference"
                );
            }
            // A branch session records its kind and address — a custom role's name and the
            // identifier it continues, or a worker branch's `None` role name and its branch name —
            // on EVERY appending turn, so the dead-session poller can resume the branch as the same
            // kind without parsing the agent ID. Outside the `is_new` branch, so one failed
            // first-turn write cannot leave the branch permanently unrecorded.
            if let Some(launch) = &self.launch
                && let Err(e) = crate::session::store()
                    .set_branch(
                        agent_id,
                        (launch.role == Role::Custom).then_some(launch.name.as_str()),
                        launch.identifier.as_deref(),
                        launch.reply_to.as_deref(),
                    )
                    .await
            {
                tracing::warn!(
                    agent_id = %agent_id,
                    error = %e,
                    "Failed to record branch session metadata"
                );
            }
            // The message is durably written above — advance the persisted prefix and refresh the
            // transcript.
            self.persisted_len = self.history.len();
            self.publish_transcript();
        }
        Ok(())
    }

    /// Rebuild a launch session's opening context block for the folder the branch now works in: the
    /// block is the session's leading system run and its length is recorded in the same transaction
    /// whenever it is written. It is measured by that recorded length — never by re-rendering the
    /// previous folder, whose length follows live state (skills on disk, workspace context,
    /// automations) and could drift. A session predating the record is measured from the history
    /// itself.
    async fn refresh_context_block(
        &mut self,
        agent_id: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        is_admin: bool,
        user_name: &str,
    ) -> Result<()> {
        let previous_block_len = crate::session::store()
            .context_block_len(agent_id)
            .await
            .unwrap_or_else(|| {
                // Recorded before the column existed: the block is the leading run of system
                // messages.
                self.history
                    .iter()
                    .take_while(|message| message.role == ChatRole::System)
                    .count()
            })
            // Written with the block it describes, so it always fits; clamped so the slice cannot
            // panic.
            .min(self.history.len());
        let (fresh, _) = Self::build_context_messages(
            ws,
            role,
            ticket,
            is_admin,
            user_name,
            self.launch.as_ref(),
        )
        .await;
        let fresh_len = fresh.len();
        let mut rebuilt = fresh;
        rebuilt.extend_from_slice(&self.history[previous_block_len..]);
        // The rewritten block and its length land in one transaction.
        crate::session::store()
            .replace_messages(agent_id, &rebuilt, Some(fresh_len), false)
            .await?;
        self.history = rebuilt;
        self.persisted_len = self.history.len();
        self.publish_transcript();
        Ok(())
    }

    // Attach the shared lock-free holder this session publishes live transcript snapshots into.
    // Called by `Agent::new` after registering the holder in the global accessor; must run before
    // the first mutation.

    pub(crate) fn attach_transcript(&mut self, holder: Arc<ArcSwap<TranscriptSnapshot>>) {
        self.transcript = Some(holder);
    }

    /// Publish a fresh read-only snapshot of the current history and token count. Copy-on-write:
    /// one history clone per publish. No-op when no holder is attached (detached sessions, tests).
    fn publish_transcript(&self) {
        if let Some(holder) = &self.transcript {
            let snapshot = TranscriptSnapshot {
                history: self.history.clone(),
                token_count: self.token_length,
            };
            holder.store(Arc::new(snapshot));
            // Live transcript content changed → the Running Agents cards and footer need a
            // re-render.
            crate::runtime_events::publish(RuntimeEvent::Registries);
        }
    }

    // Read-only access to the in-memory history (for LLM calls).
    #[must_use]
    pub(crate) fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Real provider-reported session length (see the field doc). `None` when no usage-bearing
    /// agent call ever recorded a value, or the last compaction cleared it.
    #[must_use]
    pub(crate) fn token_length(&self) -> Option<u64> {
        self.token_length
    }

    /// Pictures the conversation holds (see the field doc).
    #[must_use]
    pub(crate) fn image_count(&self) -> usize {
        self.image_count
    }

    /// Update the in-memory length after a successful agent call. The durable write is the caller's
    /// ([`crate::Agent::record_session_usage`]).
    pub(crate) fn set_token_length(&mut self, token_length: Option<u64>) {
        self.token_length = token_length;
        self.publish_transcript();
    }

    /// Take persisted messages into the live history, keeping the picture count in step.
    fn take_persisted(&mut self, messages: impl IntoIterator<Item = ChatMessage>) {
        let from = self.history.len();
        self.history.extend(messages);
        self.image_count += crate::session::image_cap::count(&self.history[from..]);
    }

    /// Append an assistant message (final response or intermediate tool-call) to in-memory history
    /// only.
    pub(crate) fn push_assistant(&mut self, content: String) {
        self.history.push(ChatMessage::assistant(content));
        self.publish_transcript();
    }

    /// Persist any unpersisted tail plus `messages` in one transaction, then deliver `messages` to
    /// memory and advance `persisted_len` over the whole persisted span. On failure nothing is
    /// delivered (the caller aborts or delivers unpersisted). `messages` must not already be in the
    /// unpersisted tail.
    pub(crate) async fn persist_messages(
        &mut self,
        agent_id: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        debug_assert!(self.persisted_len <= self.history.len());
        let mut batch =
            Vec::with_capacity(self.history.len() - self.persisted_len + messages.len());
        batch.extend_from_slice(&self.history[self.persisted_len..]);
        batch.extend_from_slice(messages);
        crate::session::store()
            .batch_append(agent_id, &batch)
            .await?;
        self.persisted_len += batch.len();
        // The store's increment counted the flushed tail plus `messages`; the tail was counted here
        // when it was pushed, so only `messages` is new.
        self.take_persisted(messages.iter().cloned());
        self.publish_transcript();
        Ok(())
    }

    /// The calls of the session's LAST tool-call frame whose results are missing — the universal
    /// resume-completion input. Walks back to the last assistant tool-call frame, then forward over
    /// the Tool-role entries it resolved. `None` when nothing is missing or no (decodable) frame
    /// exists.
    pub(crate) fn pending_tool_frame(&self) -> Option<PendingToolFrame> {
        let frame_idx = self.history.iter().rposition(super::is_tool_call_frame)?;
        let Some(super::DecodedNativeHistoryMessage::Assistant {
            content,
            tool_calls: Some(calls),
            ..
        }) = super::decode_native_history_message(&self.history[frame_idx])
        else {
            return None;
        };
        let has_text = content.as_deref().is_some_and(|t| !t.trim().is_empty());
        let call_count = calls.len();
        // Walk forward over Tool-role entries (stop at the first non-tool).
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::default();
        for msg in &self.history[frame_idx + 1..] {
            if msg.role == ChatRole::Tool {
                if let Some(super::DecodedNativeHistoryMessage::ToolResult {
                    tool_call_id, ..
                }) = super::decode_native_history_message(msg)
                {
                    completed.insert(tool_call_id);
                }
            } else {
                break;
            }
        }
        let pending: Vec<crate::ToolCall> = calls
            .into_iter()
            .filter(|c| !completed.contains(&c.id))
            .collect();
        if pending.is_empty() {
            None
        } else {
            Some(PendingToolFrame {
                calls: pending,
                has_text,
                call_count,
            })
        }
    }

    /// Test-only thin wrapper: the store's self-transactional settle, mirrored into memory.
    /// Production calls
    /// [`SessionStore::settle_tool_results_tx`](super::SessionStore::settle_tool_results_tx) inside
    /// the agent's atomic settle+terminalize transaction and mirrors only after the commit.
    #[cfg(test)]
    pub(crate) async fn settle_tool_results(
        &mut self,
        agent_id: &str,
        results: &[(String, String)],
        follow_up: &[ChatMessage],
    ) -> Result<()> {
        let settled = crate::session::store()
            .settle_tool_results(agent_id, results, follow_up)
            .await?;
        self.mirror_settled(settled)?;
        Ok(())
    }

    /// Mirror a store settle's returned sequence into memory: truncate after the tool-call frame
    /// and replay the final sequence verbatim.
    pub(crate) fn mirror_settled(&mut self, settled: Vec<super::SettledRow>) -> Result<()> {
        let Some(frame_idx) = self.history.iter().rposition(super::is_tool_call_frame) else {
            debug_assert!(
                false,
                "settle_tool_results called without a tool-call frame in history"
            );
            return Ok(());
        };
        // The rows after the frame are dropped and replaced by the replayed sequence: take their
        // pictures out before the truncate, then count each replayed row as it lands.
        let dropped = crate::session::image_cap::count(&self.history[frame_idx + 1..]);
        self.history.truncate(frame_idx + 1);
        self.image_count = self.image_count.saturating_sub(dropped);
        for row in settled {
            let role = row.role.parse::<ChatRole>().map_err(|e| {
                anyhow::anyhow!("settled session role {} is invalid: {e}", row.role)
            })?;
            self.take_persisted(std::iter::once(ChatMessage {
                role,
                content: row.content,
            }));
        }
        self.persisted_len = self.history.len();
        self.publish_transcript();
        Ok(())
    }

    /// Deliver `messages` to memory without persisting them (store append failed); the next
    /// successful persist flushes them in order.
    pub(crate) fn push_messages_unpersisted(&mut self, messages: &[ChatMessage]) {
        self.take_persisted(messages.iter().cloned());
        self.publish_transcript();
    }

    /// Durable rewrite of the most recent User-role message: persist the new content to the store
    /// FIRST, then swap the in-memory history entry.
    ///
    /// Ordering invariant: the store write happens before the in-memory swap, so a persist failure
    /// leaves the in-memory history untouched and the caller can fall back to the original error
    /// path without a half-stripped continuation or a swallowed error.
    ///
    /// The store UPDATE targets the last `user`-role row, which corresponds to the most recent
    /// User-role message **only while that message is within the persisted prefix**
    /// (`index < persisted_len`) — `history[..persisted_len]` mirrors the DB contiguously. A message
    /// in the unpersisted tail returns [`RewriteOutcome::UnpersistedTailNoop`] instead of
    /// corrupting an earlier row. The target message is re-derived here (authoritative), not taken
    /// from the caller.
    pub(crate) async fn rewrite_last_user_message(
        &mut self,
        agent_id: &str,
        content: String,
    ) -> Result<RewriteOutcome> {
        debug_assert!(self.persisted_len <= self.history.len());
        let Some(idx) = self.history.iter().rposition(|m| m.role == ChatRole::User) else {
            anyhow::bail!("no user message in history to rewrite");
        };
        if idx >= self.persisted_len {
            return Ok(RewriteOutcome::UnpersistedTailNoop);
        }
        crate::session::store()
            .rewrite_last_user_message(agent_id, &content)
            .await?;
        // The swap changes only this message's pictures (the strip removes them), so the live count
        // moves by that delta; the durable value follows, and a failed write only costs the next
        // init a recount — the strip itself already landed.
        let previous = crate::session::image_cap::count(std::slice::from_ref(&self.history[idx]));
        self.history[idx].content = content;
        let replacement =
            crate::session::image_cap::count(std::slice::from_ref(&self.history[idx]));
        self.image_count = self.image_count.saturating_sub(previous) + replacement;
        if let Err(e) = crate::session::store()
            .set_image_count(agent_id, self.image_count)
            .await
        {
            tracing::warn!(
                agent_id = %agent_id,
                error = %e,
                "Failed to record the stripped message's picture count"
            );
        }
        self.publish_transcript();
        Ok(RewriteOutcome::Rewritten)
    }

    /// Replace the in-memory and persisted history with a compacted version: a fresh system prompt
    /// (via `build_context_messages`), the `summary_text`, the latest [`super::RETENTION_PER_SIDE`]
    /// user messages and assistant answers from the pre-compaction history (tool traffic excluded),
    /// plus every message from `turn_start` on. Before persisting, the messages this compaction
    /// deletes are dumped via [`super::compaction_dump`] and the path is appended to the summary
    /// message so the agent can re-read the dropped content; a dump failure is fail-open. Producing
    /// the summary text is [`crate::Agent::summarize`]'s job. KV-cache preservation: the Agent calls
    /// this with byte-identical parameters (model, reasoning effort, tools, routing) so the provider
    /// reuses the cached prefix. On a persist failure the full in-memory history is kept and the
    /// next turn reloads from the DB.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn apply_summary(
        &mut self,
        agent_id: &str,
        summary_text: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        is_admin: bool,
        user_name: &str,
        turn_start: usize,
    ) {
        // Build the fresh system prompt (it may have changed since session start).
        let (mut compacted, snapshot) = Self::build_context_messages(
            ws,
            role,
            ticket,
            is_admin,
            user_name,
            self.launch.as_ref(),
        )
        .await;
        // The leading context block is exactly `compacted` here; a launch session records its
        // length.
        let context_block_len = compacted.len();

        // Append the summary, then the retained latest turns in chronological order. The newest
        // user message is already among them — no separate re-append here.
        let prefix = load_prompt("context/summary_prefix.md");
        let retained = crate::session::select_retention_window(&self.history, turn_start);
        // Dump what this compaction deletes BEFORE persisting, so the path rides in the persisted
        // summary; a persist failure orphans the dump (accepted: its lifecycle is in
        // `compaction_dump`).
        let dump_path = super::compaction_dump::dump_deleted_messages(
            agent_id,
            &self.history,
            &retained,
            role,
            user_name,
        )
        .await;
        let mut summary = format!("{prefix}{summary_text}");
        if let Some(path) = &dump_path {
            summary.push_str("\n\n");
            let rendered_path = path.display().to_string();
            summary.push_str(&substitute(
                &load_prompt("context/summary_spill.md"),
                &[("{{path}}", rendered_path.as_str())],
            ));
        }
        compacted.push(ChatMessage::system(summary));
        compacted.extend(retained);

        // Persist the compacted history; only on success does the in-memory history follow. On
        // failure the full in-memory history stays — the next turn reloads from the DB and retries.
        if let Err(e) = crate::session::store()
            .replace_messages(
                agent_id,
                &compacted,
                self.launch.as_ref().map(|_| context_block_len),
                // A compaction: the recorded length clears in the same transaction.
                true,
            )
            .await
        {
            tracing::error!(
                agent_id = %agent_id,
                error = %e,
                "Failed to persist compacted session after summarization"
            );
            return;
        }
        self.history = compacted;
        self.persisted_len = self.history.len();
        // The store SET the same absolute count from the rows it wrote.
        self.image_count = crate::session::image_cap::count(&self.history);
        // The store cleared the length in the same transaction; the product shows no count until
        // the reply.
        self.token_length = None;
        self.publish_transcript();
        // Refresh the change-detection baseline when the block was (re)rendered: it is the newest
        // authoritative state, and a stale snapshot would fire a spurious change-info on the first
        // turn after every compaction. Sections that did not re-render (catalog down) keep their
        // old baseline so a mid-session switch of that model still surfaces as a change line. When
        // no block rendered at all the baseline is left untouched for the same reason (accepted
        // fail-open gap: the block is only re-rendered here).
        let block_rendered = snapshot.image.is_some() || snapshot.video.is_some();
        if matches!(role, Role::Assistant) && block_rendered {
            let merged = Self::merge_active_models_snapshot(agent_id, &snapshot).await;
            Self::persist_active_models_snapshot(agent_id, &merged).await;
        }
    }

    // Flush any unpersisted history tail to the store and advance the persisted prefix — the final
    // assistant answer, or just delivered comments on aborted turns. No-op on an empty tail (a
    // drain cut after the last tool-group commit, or a turn with no output), so a retained answer
    // is never duplicated; the no-op is surfaced as [`FinalizeOutcome::NoUnpersistedTail`] for the
    // caller to log with attribution. Skipped by the agent-level cancel path.

    pub(crate) async fn finalize(&mut self, agent_id: &str) -> Result<FinalizeOutcome> {
        debug_assert!(self.persisted_len <= self.history.len());
        if self.persisted_len == self.history.len() {
            return Ok(FinalizeOutcome::NoUnpersistedTail);
        }
        self.persist_messages(agent_id, &[]).await?;
        Ok(FinalizeOutcome::Flushed)
    }

    /// Build fresh system prompt + ticket context for the current turn: the system-level messages
    /// plus the model snapshot rendered in the `<active-models-opts>` block (`Default` when none was
    /// injected).
    ///
    /// Message order, which the cached prefix depends on: role description, onboarding guide (the
    /// admin's Assistant while onboarding), active-models opts (Assistant), workspace boilerplate
    /// (with the per-role frame; `<workspace-context>` carries `MEMORY.md` only for the account's own
    /// Assistant), skills, automations, personal files, workspaces, custom tools, custom roles, board
    /// context (Manager), ticket block.
    ///
    /// Caching contract: called directly by [`Session::apply_summary`] when rebuilding context after
    /// compaction, and wrapped by [`Self::build_turn_messages`] to add the per-turn user message. The
    /// Assistant block is re-emitted on compaction, so long sessions never lose the capability info.
    ///
    /// Custom roles: a [`Role::Custom`] session receives exactly two system parts and nothing else —
    /// the fixed [`custom_role_block`] first and the launch's authorial text second. Every
    /// built-in-role block is deliberately absent, because the administrator's own words are the
    /// whole definition.
    async fn build_context_messages(
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        is_admin: bool,
        user_name: &str,
        launch: Option<&crate::Launch>,
    ) -> (Vec<ChatMessage>, ModelSnapshot) {
        // A `Role::Custom` session ALWAYS opens with the two-part custom prompt. A launch without
        // one (a mis-routed envelope, an old row, a bug) renders the fixed block with an empty
        // authorial part instead of falling through to the built-in path, which would
        // `load_prompt("role/custom.md")` and panic (there is no such asset). The message router
        // already refuses a launch-less custom-role job.
        if role == &Role::Custom {
            return (
                vec![
                    ChatMessage::system(custom_role_block(ws)),
                    ChatMessage::system(launch.map_or("", |launch| launch.prompt.as_str())),
                ],
                ModelSnapshot::default(),
            );
        }
        let (stored_context, board_context) = tokio::join!(
            lookup_workspace_context(ws, role),
            build_board_context(ws, role),
        );

        let workspace_context = if let Some(ctx) = stored_context.as_deref() {
            ctx.to_owned()
        } else {
            let personal_memory = carries_personal_memory(ws, *role, user_name);
            build_workspace_context(ws.as_path(), personal_memory).await
        };

        let workspace_context = {
            let mut ctx = String::new();

            // Auto-discovered workspace context block.
            ctx.push_str(&crate::prompt::wrap_workspace_context(&workspace_context));

            // User-curated manual notes block (survives rediscover). NOTE: changes to `ws.notes` do
            // not reach an in-flight session until a compaction rebuilds the prompt — the
            // documented caching contract.
            if !ws.notes.trim().is_empty() {
                let notes = truncate_workspace_notes(&ws.notes);
                let _ = write!(ctx, "\n<user-notes>\n{notes}\n</user-notes>\n");
            }

            ctx
        };

        let workspace_boilerplate = substitute(
            &load_prompt("context/workspace.md"),
            &[
                (
                    "{{path_frame}}",
                    &load_prompt(workspace_frame_key(*role, is_admin)),
                ),
                ("{{operating_system}}", std::env::consts::OS),
                ("{{workspace}}", &ws.as_path().display().to_string()),
                ("{{workspace_context}}", &workspace_context),
                (
                    "{{system_locale}}",
                    &sys_locale::get_locale().unwrap_or_else(|| "unknown".to_string()),
                ),
            ],
        );

        let role_description = role.role_description_for(is_admin);
        let skills = skills::load_skills(ws).await;

        let mut msgs = Vec::with_capacity(7);
        msgs.push(ChatMessage::system(&role_description));

        // Onboarding guide (the admin's Assistant only, while onboarding is live): it disappears
        // once the state is `Finished`. The greeting kickoff persists `Finished` BEFORE the
        // pipeline builds the greeting's session (many async hops against one config_kv upsert), so
        // the state check alone would miss the guide — the kickoff also arms a process-local flag
        // that this build consumes once. After that the state can only move `Init` → `Finished`.
        // Guests receive no onboarding.
        if matches!(role, Role::Assistant) && is_admin {
            let live = crate::config::CONFIG.onboarding_stage()
                != crate::config::OnboardingState::Finished;
            if live || crate::onboarding::take_greeting_guide_pending() {
                msgs.push(ChatMessage::system(crate::prompt::load_prompt(
                    "role/onboarding.md",
                )));
            }
        }

        // Assistant sessions carry the <active-models-opts> block: the active
        // image/video models' parameter envelope, rendered from the live
        // catalogs plus static per-model video-edit nuances. Fail-open — no
        // block when nothing renders (catalogs unavailable, no nuances).
        let mut snapshot = ModelSnapshot::default();
        if matches!(role, Role::Assistant)
            && let Some((block, rendered)) =
                crate::tools::active_models::render_block(user_name).await
        {
            msgs.push(ChatMessage::system(block));
            snapshot = rendered;
        }

        msgs.push(ChatMessage::system(&workspace_boilerplate));
        if !skills.is_empty() {
            msgs.push(ChatMessage::system(skills::skills_to_prompt(&skills, ws)));
        }
        // Assistant sessions carry the user's automation snapshot, the personal
        // folder's shape (its immediate child folders with their file counts —
        // never a path), the managed workspace list (the admin's Assistant
        // only), the custom-tool catalogue and the custom-role listing (the
        // admin's Assistant only) — the same snapshot-at-session-start contract
        // as the board block below.
        if matches!(role, Role::Assistant) {
            let ((automations, workspaces, personal_files, custom_roles), custom_tools) = tokio::join!(
                fetch_assistant_context(user_name, is_admin),
                crate::tools::custom::context_block(user_name, is_admin),
            );
            for block in assistant_context_blocks(
                is_admin,
                &automations,
                &workspaces,
                personal_files.as_deref(),
                &custom_roles,
                &custom_tools,
            ) {
                msgs.push(ChatMessage::system(&block));
            }
        }
        if let Some(board_context) = board_context {
            msgs.push(ChatMessage::system(&board_context));
        }
        if let Some(t) = ticket {
            msgs.push(ChatMessage::system(format_ticket_block(t)));
        }
        (msgs, snapshot)
    }

    /// Build the fresh system prompt, ticket context and user message for the current turn: the
    /// system messages above plus the user message, and the rendered active-models snapshot
    /// (Assistant only). Caching contract: the output is persisted on the FIRST turn (via
    /// `batch_append_with_context` in [`Self::append_turn_message`]) and loaded from the DB
    /// afterwards — this is only called for new sessions. System prompts are cached; user messages
    /// carry the timestamp block, round-pinned for parallel-round first messages and fresh
    /// otherwise.
    #[expect(clippy::too_many_arguments)]
    async fn build_turn_messages(
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::pipeline::board::Ticket>,
        round_ts: Option<&str>,
        is_admin: bool,
        user_name: &str,
        launch: Option<&crate::Launch>,
    ) -> (Vec<ChatMessage>, ModelSnapshot) {
        let (mut msgs, snapshot) =
            Self::build_context_messages(ws, role, ticket, is_admin, user_name, launch).await;
        msgs.push(crate::session::user_msg_with_ts(msg, round_ts));
        (msgs, snapshot)
    }

    // Persist the `<active-models-opts>` baseline after the system prompt (re)build. Only the model
    // ids actually rendered are recorded (fail-open: a section absent from the block gets no
    // baseline, so a later switch of it never fires a change-info). An empty snapshot clears any
    // baseline — at first turn nothing rendered, so there is nothing to compare against. At
    // compaction the caller merges in the preserved baseline for sections that did not re-render
    // and only calls this when it did.
    async fn persist_active_models_snapshot(agent_id: &str, snapshot: &ModelSnapshot) {
        let json =
            (snapshot.image.is_some() || snapshot.video.is_some()).then(|| snapshot.to_json());
        if let Err(e) = crate::session::store()
            .set_active_models(agent_id, json.as_deref())
            .await
        {
            tracing::warn!(agent_id = %agent_id, error = %e, "Failed to persist active-models snapshot");
        }
    }

    /// Extend a freshly rendered snapshot with the preserved baseline for sections that did not
    /// render, so a mid-session switch of that model can still surface as a change line.
    async fn merge_active_models_snapshot(
        agent_id: &str,
        rendered: &ModelSnapshot,
    ) -> ModelSnapshot {
        let previous = crate::session::store()
            .get_active_models(agent_id)
            .await
            .and_then(|json| ModelSnapshot::from_json(&json));
        ModelSnapshot {
            image: rendered
                .image
                .clone()
                .or_else(|| previous.as_ref().and_then(|p| p.image.clone())),
            video: rendered
                .video
                .clone()
                .or_else(|| previous.as_ref().and_then(|p| p.video.clone())),
        }
    }

    /// Detect a mid-session image/video model change against the persisted baseline, driven by the
    /// user's per-user model picks. On a change, returns the message with a change-info block
    /// prepended plus the snapshot to persist (the caller persists it only after the message is
    /// stored); otherwise the message unchanged and no snapshot.
    async fn prepend_model_change(
        agent_id: &str,
        msg: &str,
        user_name: &str,
    ) -> (String, Option<ModelSnapshot>) {
        let Some(previous) = crate::session::store()
            .get_active_models(agent_id)
            .await
            .and_then(|json| ModelSnapshot::from_json(&json))
        else {
            return (msg.to_string(), None);
        };
        let current = ModelSnapshot::from_user(user_name).await;

        // Advance only the changed model(s): a model whose section never rendered must not enter
        // the baseline through an unrelated change, or a later switch of it would fire for an id
        // never shown.
        let mut advanced = previous.clone();
        let mut changed = Vec::new();
        if let (Some(old), Some(new)) = (previous.image.as_deref(), current.image.as_deref())
            && old != new
        {
            advanced.image = current.image.clone();
            changed.push((ModelKind::Image, old, new));
        }
        if let (Some(old), Some(new)) = (previous.video.as_deref(), current.video.as_deref())
            && old != new
        {
            advanced.video = current.video.clone();
            changed.push((ModelKind::Video, old, new));
        }
        if changed.is_empty() {
            return (msg.to_string(), None);
        }

        // Both catalog fetches run concurrently (each internally timeout-bounded), so a dual switch
        // costs one bounded wall-clock duration, not two sequential ones.
        let blocks = join_all(
            changed
                .into_iter()
                .map(|(kind, old, new)| Self::model_change_block(kind, old, new)),
        )
        .await;

        // The change-info is the newest authoritative state for the changed model(s): the caller
        // refreshes the baseline once the message is persisted, so the next message does not
        // re-fire it.
        (format!("{}\n\n{msg}", blocks.join("\n")), Some(advanced))
    }

    /// Build a change-info block for one model: "old → new" plus the new model's capabilities from
    /// the catalog (or its static edit nuances when the catalog is unavailable); a change-only line
    /// otherwise.
    async fn model_change_block(kind: ModelKind, old: &str, new: &str) -> String {
        match crate::tools::active_models::render_section(kind, new).await {
            Some(section) => format!(
                "<model-change>Active {} model changed from {} to {}.\nNew model capabilities:\n{section}</model-change>",
                kind.label(),
                old,
                new
            ),
            None => format!(
                "<model-change>Active {} model changed from {} to {}.</model-change>",
                kind.label(),
                old,
                new
            ),
        }
    }
}

/// Build a board-state context block for the Manager role: active tickets for the workspace, in the
/// same style as the `list_tickets` tool. `None` for other roles or when there are no active
/// tickets.
async fn build_board_context(ws: &Workspace, role: &Role) -> Option<String> {
    if !matches!(role, Role::Manager) {
        return None;
    }
    let board = BOARD.get()?;
    let tickets = board.list_all_tickets(Some(&ws.name), None).await.ok()?;
    let active: Vec<_> = tickets
        .into_iter()
        .filter(|t| !t.phase.is_unblocking())
        .collect();
    if active.is_empty() {
        return None;
    }
    let count = active.len();
    let mut output = format!(
        "<workspace-board>\nTickets in {} ({count} active):\n",
        ws.name
    );
    for t in &active {
        let _ = writeln!(output, "{}", t.short_display());
    }
    output.push_str("</workspace-board>");
    Some(output)
}

/// Upper char bound for a workspace's collapsed discovery-summary line.
const MAX_WORKSPACE_SUMMARY_CHARS: usize = 1000;

/// Fetch the data for the Assistant's context blocks. Fail-open per store: an uninitialized store
/// or a read error yields empty data, so the block is omitted — the same contract as
/// `build_board_context`. The automation fetch, the workspace fetch, the personal-folder walk and
/// the custom-role listing run concurrently, and the per-workspace context reads are batched via
/// `join_all`.
async fn fetch_assistant_context(
    user_name: &str,
    is_admin: bool,
) -> (
    Vec<Automation>,
    Vec<(Workspace, Option<String>)>,
    Option<String>,
    Vec<CustomRole>,
) {
    let automations = async {
        if crate::automations::AUTOMATIONS.get().is_none() {
            return Vec::new();
        }
        crate::automations::list_user_automations(user_name)
            .await
            .unwrap_or_default()
    };
    let workspaces = async {
        if !is_admin {
            return Vec::new();
        }
        let Some(workspaces) = WORKSPACES.get() else {
            return Vec::new();
        };
        let Ok(all) = workspaces.list().await else {
            return Vec::new();
        };
        // Defensive: personal workspaces are never managed rows, but never
        // let one leak into the block (or pay a context read for it).
        let managed: Vec<Workspace> = all
            .into_iter()
            .filter(|w| !w.name.starts_with("personal:"))
            .collect();
        let summaries = join_all(
            managed
                .iter()
                .map(|w| workspaces.get_general_context(&w.name)),
        )
        .await
        .into_iter()
        .map(|res| res.ok().flatten())
        .collect::<Vec<_>>();
        managed.into_iter().zip(summaries).collect()
    };
    let custom_roles = async {
        if !is_admin {
            return Vec::new();
        }
        // The store may not be up yet mid-boot: a missing store omits the
        // block instead of panicking (fail-open, like the sibling blocks).
        if CUSTOM_ROLES.get().is_none() {
            return Vec::new();
        }
        crate::custom_role::list().await.unwrap_or_default()
    };
    tokio::join!(
        automations,
        workspaces,
        fetch_personal_files(user_name),
        custom_roles
    )
}

/// Whether the file-derived workspace context carries the account's personal memory (`MEMORY.md`):
/// only the Assistant of the account whose personal workspace it is. No other role receives it, and
/// a `MEMORY.md` in a managed workspace is nobody's continuity record.
fn carries_personal_memory(ws: &Workspace, role: Role, user_name: &str) -> bool {
    matches!(role, Role::Assistant)
        && crate::users::is_valid_personal_user_name(user_name)
        && crate::users::personal_user_name(&ws.name) == Some(user_name)
}

/// The workspace-frame asset for a role: the admin's Assistant is not confined to the workspace, so
/// it gets the frame that says so (and names what stays closed); every other role keeps the
/// workspace frame. `context/workspace.md` carries the rest of the boilerplate either way.
fn workspace_frame_key(role: Role, is_admin: bool) -> &'static str {
    if matches!(role, Role::Assistant) && is_admin {
        "context/workspace_frame_admin.md"
    } else {
        "context/workspace_frame.md"
    }
}

/// Compose the Assistant-only system-prompt blocks from pre-fetched data. Pure and fail-open: a
/// block is omitted when its content is empty, and the workspace list and custom-role listing are
/// the admin's Assistant's only. Order: automations → personal files → workspaces → custom tools →
/// custom roles. The custom-tool block always renders (the catalogue has a brief "nothing
/// available" form); the custom-role block is omitted when no roles exist. The caller gates on
/// `role == Assistant` before fetching.
fn assistant_context_blocks(
    is_admin: bool,
    automations: &[Automation],
    workspaces: &[(Workspace, Option<String>)],
    personal_files: Option<&str>,
    custom_roles: &[CustomRole],
    custom_tools: &str,
) -> Vec<String> {
    let mut blocks = Vec::new();
    if let Some(lines) = render_automation_lines(automations) {
        blocks.push(substitute(
            &load_prompt("context/automations.md"),
            &[("{{automations}}", &lines)],
        ));
    }
    if let Some(lines) = personal_files {
        blocks.push(substitute(
            &load_prompt("context/personal_files.md"),
            &[("{{folders}}", lines)],
        ));
    }
    if is_admin && let Some(lines) = render_workspace_lines(workspaces) {
        blocks.push(substitute(
            &load_prompt("context/workspaces.md"),
            &[("{{workspaces}}", &lines)],
        ));
    }
    if !custom_tools.is_empty() {
        blocks.push(custom_tools.to_string());
    }
    if is_admin && let Some(lines) = render_custom_role_lines(custom_roles) {
        blocks.push(substitute(
            &load_prompt("context/custom_roles.md"),
            &[("{{roles}}", &lines)],
        ));
    }
    blocks
}

/// Render one line per automation — the `list_automations` shape plus the interval for periodic
/// ones. `None` when there are no automations.
fn render_automation_lines(automations: &[Automation]) -> Option<String> {
    if automations.is_empty() {
        return None;
    }
    let mut out = String::new();
    for automation in automations {
        let fire = crate::automations::format_fire_time(&automation.next_fire_at)
            .unwrap_or_else(|_| automation.next_fire_at.clone());
        let _ = write!(out, "- {}, {}", automation.id, automation.kind);
        if let Some(interval) = automation.interval_seconds {
            let _ = write!(out, " (every {interval} seconds)");
        }
        let _ = write!(out, ", {}, next fire: {}", automation.text, fire);
        if let Some(trigger) = &automation.trigger {
            let _ = write!(out, ", trigger: {}", trigger.render());
        }
        if let Some(target) = automation.role_target() {
            let _ = write!(out, ", {target}");
        }
        out.push('\n');
    }
    Some(out.trim_end().to_string())
}

/// Render one entry per managed workspace — name, status, path, plus a one-line collapsed discovery
/// summary when one exists. A workspace without a summary is still listed. Personal rows are
/// filtered upstream in [`fetch_assistant_context`]. `None` when nothing remains.
fn render_workspace_lines(workspaces: &[(Workspace, Option<String>)]) -> Option<String> {
    let mut out = String::new();
    for (ws, summary) in workspaces {
        let _ = writeln!(out, "- {} ({}): {}", ws.name, ws.status, ws.path);
        if let Some(summary) = summary.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let _ = writeln!(
                out,
                "  {}",
                crate::util::truncate(
                    &summary.split_whitespace().collect::<Vec<_>>().join(" "),
                    MAX_WORKSPACE_SUMMARY_CHARS,
                )
            );
        }
    }
    (!out.is_empty()).then(|| out.trim_end().to_string())
}

/// Upper char bound for a role's collapsed prompt opening: only the opening is quoted, the full
/// prompt being read at launch time.
const MAX_ROLE_EXCERPT_CHARS: usize = 500;

/// Render one entry per custom role: name, behaviour token and the administrator's chosen tool set
/// on the first line, the opening of the authored prompt on the second. `None` when no roles are
/// defined — unlike the custom-tool catalogue, this block is omitted rather than rendered empty.
fn render_custom_role_lines(roles: &[CustomRole]) -> Option<String> {
    if roles.is_empty() {
        return None;
    }
    let mut out = String::new();
    for role in roles {
        let tools = if role.tools.is_empty() {
            "none".to_string()
        } else {
            crate::tools::grants::describe_all(&role.tools).join(", ")
        };
        let _ = writeln!(
            out,
            "- {} (behaviour: {}, tools: {tools})",
            role.name,
            role.behavior.as_str(),
        );
        let _ = writeln!(out, "  {}", role_prompt_excerpt(&role.prompt));
    }
    Some(out.trim_end().to_string())
}

/// The opening of a role's authored prompt: whitespace collapsed to single spaces,
/// credential-scrubbed like the other admin-authored text that enters a prompt, then cut to
/// [`MAX_ROLE_EXCERPT_CHARS`] on a char boundary (a role's text may be Russian) with an ellipsis.
/// Never a model-written summary, never the rendered launch prompt.
fn role_prompt_excerpt(prompt: &str) -> String {
    let collapsed = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    crate::util::truncate(
        &crate::util::scrub_credentials(&collapsed),
        MAX_ROLE_EXCERPT_CHARS,
    )
}

/// The session-start `<workspace-structure>` picture for the user's personal folder: its immediate
/// child folders with the number of files beneath each plus the files sitting directly in it — the
/// material the change report watches (see [`crate::session::personal_changes`]), rendered without
/// a single path. `None` when the name cannot address a folder, the folder does not exist, it shows
/// nothing of the user's, or it cannot be walked (fail-open). The walk runs on the blocking pool.
async fn fetch_personal_files(user_name: &str) -> Option<String> {
    if !crate::users::is_valid_personal_user_name(user_name) {
        return None;
    }
    let root = crate::users::personal_workspace_path(user_name);
    tokio::task::spawn_blocking(move || {
        crate::session::personal_changes::personal_folder_picture(&root)
    })
    .await
    .ok()
    .flatten()
}

/// Try to find a stored workspace context for the given workspace and role.
async fn lookup_workspace_context(ws: &Workspace, role: &Role) -> Option<String> {
    let workspaces = crate::workspace::store();
    workspaces.get_context(&ws.name, role.as_str()).await.ok()?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The persisted automation lines the assistant sees: one statement of the timing per call, the
    /// tool trigger's arguments verbatim (scrubbing was deliberately removed), and the custom-role
    /// branch address.
    #[test]
    fn automation_lines_show_trigger_and_interval() {
        let one_shot = Automation {
            id: "a1".to_string(),
            session_id: "s1".to_string(),
            user_name: "u".to_string(),
            kind: "one-shot".to_string(),
            text: "hello".to_string(),
            trigger: None,
            role: None,
            identifier: None,
            workspace: None,
            interval_seconds: None,
            next_fire_at: "2026-01-01T00:00:00+00:00".to_string(),
            unfinished_attempts: 0,
        };
        let periodic = Automation {
            id: "a2".to_string(),
            session_id: "s1".to_string(),
            user_name: "u".to_string(),
            kind: "periodic".to_string(),
            text: "check".to_string(),
            trigger: Some(crate::automations::StoredTrigger::Tool(
                crate::automations::Trigger {
                    tool: "weather".to_string(),
                    args: serde_json::Map::from_iter([
                        ("city".to_string(), serde_json::json!("Minsk")),
                        (
                            "api_key".to_string(),
                            serde_json::json!("supersecretvalue123"),
                        ),
                    ]),
                },
            )),
            role: None,
            identifier: None,
            workspace: None,
            interval_seconds: Some(3600),
            next_fire_at: "2026-01-01T00:00:00+00:00".to_string(),
            unfinished_attempts: 0,
        };
        let role = Automation {
            id: "a3".to_string(),
            session_id: "s1".to_string(),
            user_name: "u".to_string(),
            kind: "one-shot".to_string(),
            text: "wake the role".to_string(),
            trigger: None,
            role: Some("analyst".to_string()),
            identifier: Some("daily".to_string()),
            workspace: Some("my-workspace".to_string()),
            interval_seconds: None,
            next_fire_at: "2026-01-01T00:00:00+00:00".to_string(),
            unfinished_attempts: 0,
        };

        let lines =
            render_automation_lines(&[one_shot, periodic, role]).expect("automations render");
        assert!(lines.contains("every 3600 seconds"));
        assert!(lines.contains("trigger: weather "));
        assert!(lines.contains("\"city\":\"Minsk\""));
        // Scrubbing was deliberately removed: a credential-shaped argument reaches the prompt
        // verbatim.
        assert!(lines.contains("\"api_key\":\"supersecretvalue123\""));
        assert!(lines.contains("next fire:"));
        assert!(
            lines.contains(
                "addressed to custom role \"analyst\" (branch \"daily\") in my-workspace"
            ),
            "got: {lines}"
        );

        assert!(render_automation_lines(&[]).is_none());
    }

    /// Mid-session change detection: switching image AND video models against a persisted baseline
    /// fires one change-info per model (change-only lines — both catalog caches are seeded
    /// fail-open, so no network I/O), returns the new snapshot for the caller to persist after the
    /// message append, and does not re-fire once the baseline is advanced. A missing baseline is a
    /// no-op.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn prepend_model_change_detects_switch_and_refreshes_baseline() {
        let user = "test_assistant_change_user";
        crate::util::test::init_test_stores().await;
        // Empty-catalog seeds: both lookups return a fresh empty catalog with no network fetch, so
        // the change path exercises the change-only fallback hermetically. Unlike a `None` seed
        // (negative-cache residue), the caches stay Fresh — no cross-test coupling.
        crate::tools::media_catalog::image::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::image::ImageCatalog::default(),
        )));
        crate::tools::media_catalog::video::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::video::VideoCatalog::default(),
        )));
        let agent_id = "test_assistant_change";

        crate::users::store()
            .set_image_gen_model(user, "model-a")
            .await
            .expect("set image model");
        crate::users::store()
            .set_video_model(user, "video-a")
            .await
            .expect("set video model");
        let baseline = ModelSnapshot::from_user(user).await;
        crate::session::store()
            .set_active_models(agent_id, Some(&baseline.to_json()))
            .await
            .expect("baseline persisted");

        // Switch both models mid-session → one change-info per model.
        crate::users::store()
            .set_image_gen_model(user, "model-b")
            .await
            .expect("set image model");
        crate::users::store()
            .set_video_model(user, "video-b")
            .await
            .expect("set video model");
        let (out, new_snapshot) = Session::prepend_model_change(agent_id, "hello", user).await;
        assert!(
            out.starts_with(
                "<model-change>Active image model changed from model-a to model-b.</model-change>\n\
                 <model-change>Active video model changed from video-a to video-b.</model-change>"
            ),
            "unexpected change-info: {out:?}"
        );
        assert!(out.ends_with("\n\nhello"));
        let new_snapshot = new_snapshot.expect("snapshot returned for persistence");

        // Baseline refreshed (as the caller does after a successful append) → the same switches do
        // not re-fire.
        crate::session::store()
            .set_active_models(agent_id, Some(&new_snapshot.to_json()))
            .await
            .expect("baseline advanced");
        let (out, snapshot) = Session::prepend_model_change(agent_id, "again", user).await;
        assert_eq!(out, "again");
        assert!(snapshot.is_none());

        // No persisted baseline at all → nothing to compare against, no line.
        let (out, snapshot) =
            Session::prepend_model_change("test_assistant_no_baseline", "hello", "no_such_user")
                .await;
        assert_eq!(out, "hello");
        assert!(snapshot.is_none());
    }

    /// Partial-block session (one catalog down at start): switching the model whose section DID
    /// render must not mint a baseline for the never-rendered model — a later switch of it stays
    /// silent.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn prepend_model_change_preserves_absent_sections() {
        let user = "test_assistant_partial_user";
        crate::util::test::init_test_stores().await;
        crate::tools::media_catalog::image::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::image::ImageCatalog::default(),
        )));
        crate::tools::media_catalog::video::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::video::VideoCatalog::default(),
        )));
        let agent_id = "test_assistant_partial";

        // Only the video section rendered at session start — the image id is NOT in the baseline.
        crate::users::store()
            .set_video_model(user, "video-a")
            .await
            .expect("set video model");
        let baseline = ModelSnapshot {
            image: None,
            video: Some("video-a".into()),
        };
        crate::session::store()
            .set_active_models(agent_id, Some(&baseline.to_json()))
            .await
            .expect("baseline persisted");

        // Switch the video model → change-info for video; the snapshot advances only video.
        crate::users::store()
            .set_video_model(user, "video-b")
            .await
            .expect("set video model");
        let (out, new_snapshot) = Session::prepend_model_change(agent_id, "hello", user).await;
        assert!(
            out.starts_with("<model-change>Active video model changed from video-a to video-b.")
        );
        let new_snapshot = new_snapshot.expect("snapshot returned for persistence");
        assert_eq!(new_snapshot.image, None);
        assert_eq!(new_snapshot.video.as_deref(), Some("video-b"));
        crate::session::store()
            .set_active_models(agent_id, Some(&new_snapshot.to_json()))
            .await
            .expect("baseline advanced");

        // A later image switch stays silent — the image id was never rendered.
        crate::users::store()
            .set_image_gen_model(user, "image-b")
            .await
            .expect("set image model");
        let (out, snapshot) = Session::prepend_model_change(agent_id, "again", user).await;
        assert_eq!(out, "again");
        assert!(snapshot.is_none());
    }

    /// Compaction merge: a rendered snapshot is extended with the preserved baseline for sections
    /// that did not re-render, so a mid-session switch of that model can still surface as a change
    /// line.
    #[tokio::test]
    async fn merge_active_models_snapshot_preserves_unrendered_sections() {
        crate::util::test::init_test_stores().await;
        let previous = ModelSnapshot {
            image: Some("model-a".into()),
            video: Some("video-a".into()),
        };
        crate::session::store()
            .set_active_models("test_assistant_merge", Some(&previous.to_json()))
            .await
            .expect("baseline persisted");

        // Compaction re-rendered only the image section (video catalog down).
        let rendered = ModelSnapshot {
            image: Some("model-b".into()),
            video: None,
        };
        let merged = Session::merge_active_models_snapshot("test_assistant_merge", &rendered).await;
        assert_eq!(
            merged,
            ModelSnapshot {
                image: Some("model-b".into()),
                video: Some("video-a".into()),
            }
        );

        // No prior baseline → the merged snapshot is exactly the rendered one.
        let merged =
            Session::merge_active_models_snapshot("test_assistant_no_merge_baseline", &rendered)
                .await;
        assert_eq!(merged, rendered);
    }

    #[tokio::test]
    async fn rewrite_last_user_message_persists_then_swaps() {
        crate::util::test::init_test_stores().await;
        let agent_id = "test_rewrite_ok";
        // The in-memory history must mirror the store (persisted prefix).
        let seed = vec![
            ChatMessage::system("role"),
            ChatMessage::user("[IMAGE:/tmp/a.png] first"),
            ChatMessage::user("[IMAGE:/tmp/b.png] second"),
        ];
        crate::session::store()
            .batch_append(agent_id, &seed)
            .await
            .unwrap();
        let mut session = Session::default();
        session.history = seed;
        session.persisted_len = session.history.len();

        let outcome = session
            .rewrite_last_user_message(agent_id, "rewritten".to_string())
            .await
            .unwrap();
        assert_eq!(outcome, RewriteOutcome::Rewritten);

        // In-memory swap happened AFTER the durable write.
        assert_eq!(session.history[2].content, "rewritten");
        assert_eq!(
            session.history[1].content, "[IMAGE:/tmp/a.png] first",
            "earlier user row untouched"
        );

        // Durable: the store row reflects the rewrite.
        let msgs = crate::session::store().load(agent_id).await;
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].content, "rewritten");
    }

    #[tokio::test]
    async fn rewrite_last_user_message_unpersisted_tail_is_conservative_noop() {
        crate::util::test::init_test_stores().await;
        let agent_id = "test_rewrite_noop";
        // Seed the DB with one persisted user row.
        crate::session::store()
            .batch_append(
                agent_id,
                &[ChatMessage::user("[IMAGE:/tmp/a.png] persisted")],
            )
            .await
            .unwrap();

        // The most recent user message lives in the UNPERSISTED tail (index 1 >= persisted_len 1):
        // a positional UPDATE would rewrite the persisted row instead — conservative no-op.
        let mut session = Session {
            history: vec![
                ChatMessage::user("[IMAGE:/tmp/a.png] persisted"),
                ChatMessage::user("[IMAGE:/tmp/b.png] unpersisted"),
            ],
            persisted_len: 1,
            ..Default::default()
        };

        let outcome = session
            .rewrite_last_user_message(agent_id, "rewritten".to_string())
            .await
            .unwrap();
        assert_eq!(outcome, RewriteOutcome::UnpersistedTailNoop);
        assert_eq!(
            session.history[1].content, "[IMAGE:/tmp/b.png] unpersisted",
            "in-memory history untouched"
        );

        // The durable row is untouched.
        let msgs = crate::session::store().load(agent_id).await;
        assert_eq!(msgs[0].content, "[IMAGE:/tmp/a.png] persisted");
    }

    #[tokio::test]
    async fn rewrite_last_user_message_persist_failure_leaves_history_untouched() {
        crate::util::test::init_test_stores().await;
        let agent_id = "test_rewrite_persist_fail";
        // Seed a DIFFERENT agent's row so the table is non-empty: the positional UPDATE for
        // `agent_id` targets only that agent's last `user` row — none exists, so the store reports
        // 0 rows and errors.
        crate::session::store()
            .batch_append(
                "other_agent",
                &[ChatMessage::user("[IMAGE:/tmp/a.png] other")],
            )
            .await
            .unwrap();

        let mut session = Session {
            history: vec![
                ChatMessage::system("role"),
                ChatMessage::user("[IMAGE:/tmp/a.png] mine"),
            ],
            persisted_len: 2, // idx 1 < persisted_len → the guard passes
            ..Default::default()
        };

        let err = session
            .rewrite_last_user_message(agent_id, "rewritten".to_string())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no user message row"),
            "store error propagates: {err}"
        );
        assert_eq!(
            session.history[1].content, "[IMAGE:/tmp/a.png] mine",
            "in-memory history untouched on persist failure (ordering invariant)"
        );
    }

    // Helper: the assistant frame JSON for an analyze tool call.

    fn analyze_frame(call_id: &str) -> String {
        crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: call_id.to_string(),
                name: "analyze".to_string(),
                arguments: serde_json::json!({"message": "analyze task"}),
            }],
            None,
        )
        .to_string()
    }

    /// (a) `pending_tool_frame` returns the session's LAST tool-call frame's
    /// result-less calls. No frame → None; a frame with calls and no results →
    /// all of them; one result present → only the missing one remains.
    #[test]
    fn pending_tool_frame_none_and_dangling_and_partial() {
        assert!(Session::default().pending_tool_frame().is_none());

        // Frame with two calls, no results → both dangling.
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[
                crate::ToolCall {
                    id: "call_a".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "a"}),
                },
                crate::ToolCall {
                    id: "call_b".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "b"}),
                },
            ],
            None,
        )
        .to_string();
        let mut session = Session {
            history: vec![ChatMessage::assistant(frame)],
            persisted_len: 1,
            ..Default::default()
        };
        let pending = session
            .pending_tool_frame()
            .expect("dangling calls present")
            .calls;
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, "call_a");
        assert_eq!(pending[1].id, "call_b");

        // One result present → only the missing one remains.
        session
            .history
            .push(ChatMessage::tool_result("call_a", "result a"));
        let pending = session
            .pending_tool_frame()
            .expect("still one dangling")
            .calls;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "call_b");

        // All results present → None (nothing dangling).
        session
            .history
            .push(ChatMessage::tool_result("call_b", "result b"));
        assert!(session.pending_tool_frame().is_none());
    }

    /// (b) `settle_tool_results` inserts the settled result directly after the dangling frame,
    /// bumps `message_count` by the result count, and mirrors the insert in memory with
    /// `persisted_len` == `history.len()`.
    #[tokio::test]
    async fn settle_tool_results_inserts_result_after_frame() {
        crate::util::test::init_test_stores().await;
        let agent_id = "settle_tool_results_test";
        let frame = analyze_frame("call_analyze_1");

        let mut session = Session::default();
        session
            .persist_messages(
                agent_id,
                &[ChatMessage::user("first"), ChatMessage::assistant(frame)],
            )
            .await
            .unwrap();
        assert_eq!(session.persisted_len, 2, "both seeded rows persisted");

        let conn = &crate::session::store().conn;
        let count_before: i64 = conn
            .query_optional(
                "SELECT message_count FROM session_metadata WHERE agent_id = ?1",
                crate::db::params![agent_id],
                |r| r.get::<i64>(0),
            )
            .await
            .unwrap()
            .expect("metadata row exists");

        session
            .settle_tool_results(
                agent_id,
                &[("call_analyze_1".to_string(), "the result".to_string())],
                &[],
            )
            .await
            .unwrap();

        // DB rows by id: user, assistant(frame), tool(result).
        let rows = conn
            .query(
                "SELECT id, role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
                crate::db::params![agent_id],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 3, "one result row inserted after the frame");
        let roles: Vec<String> = rows.iter().map(|r| r.get::<String>(1).unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
        assert_eq!(
            rows[2].get::<String>(2).unwrap(),
            serde_json::to_string(&crate::ToolResultPayload {
                tool_call_id: "call_analyze_1".to_string(),
                content: "the result".to_string(),
            })
            .unwrap(),
            "the tool row carries the settled result with the original call id"
        );

        // message_count delta == 1 (the result row).
        let count_after: i64 = conn
            .query_optional(
                "SELECT message_count FROM session_metadata WHERE agent_id = ?1",
                crate::db::params![agent_id],
                |r| r.get::<i64>(0),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count_after - count_before, 1);

        // In-memory history mirrors the insert; the persisted prefix covers it.
        let roles: Vec<crate::ChatRole> = session.history().iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![ChatRole::User, ChatRole::Assistant, ChatRole::Tool]
        );
        assert_eq!(
            session.history()[2].content,
            serde_json::to_string(&crate::ToolResultPayload {
                tool_call_id: "call_analyze_1".to_string(),
                content: "the result".to_string(),
            })
            .unwrap()
        );
        assert_eq!(session.persisted_len, session.history().len());
    }

    /// (b2) `settle_tool_results` preserves the frame's tool-call ORDER when a partial result set
    /// is settled: with `call_b`'s row already committed, settling `call_a` leaves the DB order
    /// frame, tool(call_a), tool(call_b) — the recovered result keeps the frame's order and the
    /// captured sibling stays put.
    #[tokio::test]
    async fn settle_tool_results_preserves_frame_call_order() {
        crate::util::test::init_test_stores().await;
        let agent_id = "settle_tool_results_order_test";
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[
                crate::ToolCall {
                    id: "call_a".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "a"}),
                },
                crate::ToolCall {
                    id: "call_b".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "b"}),
                },
            ],
            None,
        )
        .to_string();

        let mut session = Session::default();
        session
            .persist_messages(
                agent_id,
                &[
                    ChatMessage::user("first"),
                    ChatMessage::assistant(frame),
                    ChatMessage::tool_result("call_b", "result b"),
                ],
            )
            .await
            .unwrap();

        // Settle only call_a; call_b's row is the captured sibling.
        session
            .settle_tool_results(
                agent_id,
                &[("call_a".to_string(), "result a".to_string())],
                &[],
            )
            .await
            .unwrap();

        // DB rows by id: user, assistant(frame), tool(call_a), tool(call_b) — the frame's call
        // order, and the captured sibling verbatim.
        let conn = &crate::session::store().conn;
        let rows = conn
            .query(
                "SELECT id, role, content FROM sessions WHERE agent_id = ?1 ORDER BY id",
                crate::db::params![agent_id],
            )
            .await
            .unwrap();
        assert_eq!(
            rows.len(),
            4,
            "frame + captured sibling + one recovered result"
        );
        let roles: Vec<String> = rows.iter().map(|r| r.get::<String>(1).unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "tool"]);
        let mut ids: Vec<String> = Vec::new();
        for row in &rows[2..] {
            let payload: crate::ToolResultPayload =
                serde_json::from_str(&row.get::<String>(2).unwrap()).unwrap();
            ids.push(payload.tool_call_id);
        }
        assert_eq!(ids, vec!["call_a", "call_b"], "frame call order preserved");
        let b_payload: crate::ToolResultPayload =
            serde_json::from_str(&rows[3].get::<String>(2).unwrap()).unwrap();
        assert_eq!(b_payload.content, "result b", "captured sibling verbatim");
    }

    /// Pure gating of the Assistant context blocks: a guest's Assistant gets automations,
    /// workspace-structure and custom tools (the workspace list and custom-role listing are the
    /// admin's only); the admin's gets automations → personal files → workspaces plus the
    /// custom-tool block last; empty data omits the conditional blocks while the custom-tool block
    /// is always present and the custom-role block is omitted when no roles exist. Role gating
    /// lives at the call site.
    #[test]
    fn assistant_blocks_gating() {
        let automation = Automation {
            id: "a1".to_string(),
            session_id: "s1".to_string(),
            user_name: "u".to_string(),
            kind: "one-shot".to_string(),
            text: "hello".to_string(),
            trigger: None,
            role: None,
            identifier: None,
            workspace: None,
            interval_seconds: None,
            next_fire_at: "2026-01-01T00:00:00+00:00".to_string(),
            unfinished_attempts: 0,
        };
        let workspaces = vec![(
            crate::workspace::test_ws_named("/tmp/proj", "proj"),
            Some("summary".to_string()),
        )];
        let files = Some("- album/ — 2 files\n2 files directly in the folder");
        let custom = "<custom-tools>\n- weather\n</custom-tools>";

        // Guest's Assistant: automations + personal files + custom tools, no workspace list.
        let blocks = assistant_context_blocks(
            false,
            std::slice::from_ref(&automation),
            &workspaces,
            files,
            &[],
            custom,
        );
        assert_eq!(blocks.len(), 3);
        assert!(blocks[0].contains("<user-automations>"));
        assert!(blocks[1].contains("<workspace-structure>"));
        assert!(!blocks[1].contains("<managed-workspaces>"));
        assert_eq!(blocks[2], custom);

        // The admin's Assistant: the workspace list joins them, and the custom-tool block stays
        // last.
        let blocks = assistant_context_blocks(
            true,
            std::slice::from_ref(&automation),
            &workspaces,
            files,
            &[],
            custom,
        );
        assert_eq!(blocks.len(), 4);
        assert!(blocks[0].contains("<user-automations>"));
        assert!(blocks[1].contains("<workspace-structure>"));
        assert!(blocks[2].contains("<managed-workspaces>"));
        assert_eq!(blocks[3], custom);

        // Empty data omits the conditional blocks but never the custom tools.
        assert_eq!(
            assistant_context_blocks(true, &[], &[], None, &[], custom),
            vec![custom.to_string()]
        );
    }

    /// Path-safety guard: blank or synthetic user names (path separators, dot segments) never
    /// resolve to a workspace — the block is omitted instead of picturing the userspaces root or
    /// another user's folders.
    #[tokio::test]
    async fn personal_files_fetch_rejects_synthetic_names() {
        for name in ["", "  ", ".", "..", "a/b", "a\\b"] {
            assert!(fetch_personal_files(name).await.is_none(), "name: {name}");
        }
    }

    /// Through the real session-start path: the block is built from the `userspaces/<name>` folder
    /// the Assistant is pinned to, and a folder with nothing of the user's yields no block at all.
    #[tokio::test]
    async fn personal_files_fetch_pictures_the_users_own_folder() {
        let folder = crate::users::personal_workspace_path("manager_picture_user");
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(folder.join("album/deep")).expect("mkdir");
        std::fs::write(folder.join("album/deep/a.txt"), "one").expect("write");
        std::fs::write(folder.join("top.txt"), "one").expect("write");

        let block = fetch_personal_files("manager_picture_user")
            .await
            .expect("a picture");
        assert_eq!(block, "- album/ — 1 file\n1 file directly in the folder");

        std::fs::remove_dir_all(&folder).expect("rmdir");
        assert!(fetch_personal_files("manager_picture_user").await.is_none());
    }

    /// A custom role's system prompt is exactly two parts — the fixed folder/system block and the
    /// administrator's authorial text verbatim — in that order, whatever folder the launch runs in.
    #[tokio::test]
    async fn custom_role_prompt_is_the_fixed_block_plus_the_authorial_text() {
        let ws = crate::workspace::test_ws_named("/data/proj", "/data/proj");
        let launch = crate::Launch {
            role: crate::Role::Custom,
            name: "briefer".to_string(),
            identifier: Some("week2".to_string()),
            prompt: "You write the morning brief.\nNothing else.".to_string(),
            tools: vec!["read".to_string()],
            agent_id: "role_briefer:week2".to_string(),
            reply_to: None,
        };
        let (messages, _) = Session::build_context_messages(
            &ws,
            &crate::Role::Custom,
            None,
            false,
            "admin",
            Some(&launch),
        )
        .await;

        assert_eq!(messages.len(), 2, "got: {messages:?}");
        assert_eq!(messages[0].content, custom_role_block(&ws));
        assert!(messages[0].content.contains("/data/proj"));
        assert_eq!(messages[1].content, launch.prompt);
    }

    /// A worker branch's folder is named on every use, so its cached opening context block follows
    /// the folder it now works in: the turn in folder B rebuilds the leading block from B and
    /// leaves the conversation rows untouched. The block is measured by the length RECORDED when it
    /// was written, not by a fresh render of the previous folder: A's skills appear on disk after
    /// the block was cached, so re-rendering A would slice the conversation one message too far and
    /// drop its first row.
    #[tokio::test]
    async fn worker_branch_context_block_follows_the_folder() {
        // A branch turn: appends one message as the launch's own role.
        async fn turn(session: &mut Session, agent_id: &str, msg: &str, ws: &Workspace) {
            session
                .append_turn_message(
                    agent_id,
                    msg,
                    ws,
                    &crate::Role::Worker,
                    None,
                    "gui",
                    "tester",
                    None,
                    false,
                    true,
                )
                .await
                .unwrap();
        }
        // The message count of the block this folder renders right now.
        async fn block_len(ws: &Workspace, launch: &crate::Launch) -> usize {
            Session::build_context_messages(
                ws,
                &crate::Role::Worker,
                None,
                false,
                "tester",
                Some(launch),
            )
            .await
            .0
            .len()
        }

        crate::util::test::init_test_stores().await;
        // The recorded folder is the session's own address; a real canonical path (name == path)
        // keeps the recorded name comparable with the folder the next turn names.
        let dir_a = tempfile::TempDir::new().unwrap();
        let dir_b = tempfile::TempDir::new().unwrap();
        let path_a = dir_a
            .path()
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let path_b = dir_b
            .path()
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let ws_a = crate::workspace::test_ws_named(&path_a, &path_a);
        let ws_b = crate::workspace::test_ws_named(&path_b, &path_b);

        let agent_id = crate::session::worker_branch_agent_id("feature");
        let launch = crate::Launch {
            role: crate::Role::Worker,
            name: String::new(),
            identifier: Some("feature".to_string()),
            prompt: String::new(),
            tools: Vec::new(),
            agent_id: agent_id.clone(),
            reply_to: None,
        };
        let block_b = block_len(&ws_b, &launch).await;
        let mut session = Session {
            launch: Some(launch),
            ..Default::default()
        };
        session.init(&agent_id).await.unwrap();
        // Two turns in A: the leading block plus pre-existing conversation rows.
        turn(&mut session, &agent_id, "first", &ws_a).await;
        turn(&mut session, &agent_id, "second", &ws_a).await;
        // A's live context grows AFTER the block was cached: a skill appears in the folder, so a
        // fresh render of A now counts one message more than the block really written. The switch
        // below must use the recorded length — a re-render would slice the conversation one row too
        // far and drop its first row.
        let skill = dir_a.path().join("skills").join("demo");
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(skill.join("SKILL.md"), "# Instructions\nDo something.").unwrap();
        assert_eq!(
            block_len(&ws_a, session.launch.as_ref().unwrap()).await,
            block_b + 1,
            "the previous folder's fresh render must differ from the cached block"
        );

        // A turn in B rebuilds the block: A's path is gone from the leading rows, and the earlier
        // conversation rows survive after the block.
        turn(&mut session, &agent_id, "third", &ws_b).await;

        let msgs = crate::session::store().load(&agent_id).await;
        assert_eq!(msgs.len(), block_b + 3, "rebuilt block + three user turns");
        let leading = &msgs[..block_b];
        assert!(
            leading.iter().any(|m| m.content.contains(&path_b)),
            "the leading block must name the folder the branch now works in"
        );
        assert!(
            !leading.iter().any(|m| m.content.contains(&path_a)),
            "the old folder must be gone from the leading block"
        );
        assert!(
            msgs[block_b].content.starts_with("first"),
            "the pre-existing conversation row after the block is untouched"
        );
        assert!(
            msgs[block_b + 1].content.starts_with("second"),
            "the second earlier row is untouched too"
        );
    }

    /// The personal memory (`MEMORY.md`) is the owning account's Assistant-only continuity record:
    /// another role in that folder gets none of it, another account's Assistant gets none of it, a
    /// `MEMORY.md` in a managed workspace reaches nobody, and a non-agent model call carries none
    /// either.
    #[tokio::test]
    #[serial_test::serial(active_models)] // seeds the process-global catalog caches
    async fn personal_memory_reaches_only_the_owning_assistant() {
        const MEMORY: &str = "durable preference: tea over coffee";
        const SHARED: &str = "the user's folder";

        async fn context(ws: &Workspace, role: crate::Role, user_name: &str) -> String {
            Session::build_context_messages(ws, &role, None, false, user_name, None)
                .await
                .0
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        }

        crate::util::test::init_test_stores().await;
        // Empty catalog seeds keep the Assistant block hermetic (no network).
        crate::tools::media_catalog::image::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::image::ImageCatalog::default(),
        )));
        crate::tools::media_catalog::video::seed_cache(Some(std::sync::Arc::new(
            crate::tools::media_catalog::video::VideoCatalog::default(),
        )));

        let user = "memory_gate_owner";
        let personal = crate::users::personal_workspace_struct(user);
        std::fs::create_dir_all(personal.as_path()).unwrap();
        std::fs::write(personal.as_path().join("MEMORY.md"), MEMORY).unwrap();
        std::fs::write(personal.as_path().join("README.md"), SHARED).unwrap();

        // The owner's Assistant keeps the section, unchanged.
        let assistant = context(&personal, crate::Role::Assistant, user).await;
        assert!(assistant.contains("--- MEMORY.md ---"));
        assert!(assistant.contains(MEMORY));

        // Every other role in the same folder renders without it; the rest of the material still
        // renders.
        for role in [
            crate::Role::Manager,
            crate::Role::Engineer,
            crate::Role::Analyst,
            crate::Role::Worker,
            crate::Role::Qa,
            crate::Role::Reviewer,
            crate::Role::Discovery,
            crate::Role::Maintainer,
            crate::Role::Sanitation,
        ] {
            let rendered = context(&personal, role, user).await;
            assert!(
                !rendered.contains(MEMORY),
                "{role:?} in a personal folder received the memory"
            );
            assert!(
                rendered.contains(SHARED),
                "{role:?} lost the rest of the workspace context"
            );
        }

        // Another account's Assistant in that folder does not inherit it.
        let stranger = context(&personal, crate::Role::Assistant, "memory_gate_other").await;
        assert!(!stranger.contains(MEMORY));

        // A managed workspace's `MEMORY.md` is nobody's continuity record, while its other context
        // files still reach the prompt.
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("MEMORY.md"), MEMORY).unwrap();
        std::fs::write(dir.path().join("README.md"), SHARED).unwrap();
        let project =
            crate::workspace::test_ws_named(&dir.path().to_string_lossy(), "memory_gate_project");
        for role in [crate::Role::Assistant, crate::Role::Analyst] {
            let rendered = context(&project, role, user).await;
            assert!(
                !rendered.contains(MEMORY),
                "{role:?} in a managed workspace received its MEMORY.md"
            );
            assert!(rendered.contains(SHARED));
        }

        // A non-agent model call carrying the workspace context carries none.
        let mut messages = Vec::new();
        crate::prompt::prepend_general_context(&mut messages, &personal).await;
        assert!(
            messages.iter().any(|m| m.content.contains(SHARED)),
            "the rest of the workspace context must still render"
        );
        assert!(messages.iter().all(|m| !m.content.contains(MEMORY)));
    }
}

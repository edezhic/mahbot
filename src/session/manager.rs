//! `Session` — single source of truth for conversation persistence.
//!
//! Owns the in-memory history Vec for the current turn. The orchestrator
//! calls [`init`](Session::init) to load/build history, then passes `&mut Session` to the agent loop.
//!
//! ## Usage
//! 1. `Session::default()` at turn start
//! 2. `session.init(agent_id, msg, ws, role, ticket, channel, user_name, round_ts)` — loads history, builds prompt
//!    for new sessions, persists user message, stores history internally
//! 3. Agent loop calls `session.push_assistant()`, `session.persist_messages()`,
//!    `session.push_messages_unpersisted()`, etc. during tool rounds
//! 4. `session.finalize(agent_id)` on success — persists the final assistant response
//!    and reports a [`FinalizeOutcome`]; an empty unpersisted tail is a no-op
//!    (everything already durable) and is surfaced to the caller to log

use std::fmt::Write;

use anyhow::Result;
use futures_util::future::join_all;

use crate::board::BOARD;
use crate::prompt::{build_workspace_context, format_ticket_block, load_prompt, substitute};
use crate::workspace::truncate_workspace_notes;

use crate::skills;
use crate::tools::active_models::{ModelKind, ModelSnapshot};
use crate::{ChatMessage, ChatRole, Role, Workspace};

/// Coordinates session persistence and prompt building across a single
/// agent turn.
#[derive(Default)]
pub struct Session {
    /// In-memory history for the current turn (includes system prompt,
    /// conversation, and the latest user message).
    history: Vec<ChatMessage>,
    /// Number of leading `history` entries already persisted to the store;
    /// `history[..persisted_len]` mirrors the DB contents contiguously and
    /// `history[persisted_len..]` is exactly the unpersisted tail (failed
    /// appends, tool traffic awaiting commit, the final answer). Every
    /// successful persist advances this prefix over the span it wrote.
    persisted_len: usize,
    /// Real provider-reported session length (input + output tokens of the
    /// most recent successful agent LLM call), loaded at [`init`](Self::init)
    /// and updated after every successful agent-purpose LLM call. `None` for
    /// sessions that never recorded a value (new sessions, pre-migration
    /// sessions — approved no-backfill semantics).
    token_length: Option<u64>,
}

/// Outcome of a [`Session::finalize`] call.
///
/// The finalize guard is deliberately logging-free: whether an empty
/// unpersisted tail is expected (graceful drain cut right after the last
/// tool-group commit) or anomalous (LLM failure with no output, iteration
/// limit) depends on shutdown state this module does not track. The caller
/// inspects the outcome and logs with full agent/role/workspace/ticket
/// attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalizeOutcome {
    /// The unpersisted tail (final assistant answer / delivered comments)
    /// was flushed to the store and the persisted prefix advanced.
    Flushed,
    /// Nothing was unpersisted — the turn ended with an empty tail. All
    /// committed state is already durable; the caller decides whether the
    /// no-op is expected (drain) or a genuine anomaly.
    NoUnpersistedTail,
}

/// Outcome of [`Session::rewrite_last_user_message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RewriteOutcome {
    /// The most recent User-role message was durably rewritten and the
    /// in-memory history entry swapped.
    Rewritten,
    /// The most recent User-role message is in the unpersisted tail
    /// (index >= `persisted_len`): the positional store UPDATE would target a
    /// different row, so the rewrite is skipped (conservative no-op — the
    /// caller keeps the original error path).
    UnpersistedTailNoop,
}

impl Session {
    // ── Session lifecycle ──────────────────────────────────────────────

    /// Delete a session. Intended for `/new` command handling.
    pub async fn delete(agent_id: &str) -> String {
        let _ = crate::session::store().delete(agent_id).await;
        "Session cleared. Starting fresh.".to_string()
    }

    /// Begin a turn. Loads or builds history, stores the user message, and
    /// populates internal history for the agent loop.
    ///
    /// ## First-turn caching
    ///
    /// On the first turn (new session), the full system prompt, ticket context,
    /// workspace context, and skills are built via `build_turn_messages` and
    /// cached in the session DB. On subsequent turns, only the new user
    /// message is appended — the system prompt is **not** rebuilt. This avoids
    /// regenerating workspace state and ticket data on every turn.
    ///
    /// If the system prompt logic changes, existing sessions will not pick up
    /// the change until summarization fires or the user runs `/new`
    /// ([`Session::delete`]).
    ///
    /// ## Summarization note
    ///
    /// Summarization is **not** handled here. It is run separately by
    /// `Agent::work` when the conversation exceeds the token budget.
    /// See [`crate::session`] for the summarization constants and helpers.
    ///
    /// `round_ts` pins the timestamp of the appended user message (one value
    /// per round for byte-identical first messages across parallel members);
    /// `None` stamps now.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn init(
        &mut self,
        agent_id: &str,
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
        channel: &str,
        user_name: &str,
        round_ts: Option<&str>,
    ) -> Result<()> {
        // Load existing history from DB
        let mut history = crate::session::store().load(agent_id).await;

        // Empty messages carry no semantic content — skip append in all paths.
        // Recovery retries pass an empty message to re-trigger the agent
        // against the existing session history without adding a new turn.
        if !msg.is_empty() {
            let is_new = history.is_empty();

            if is_new {
                let (msgs, snapshot) =
                    Self::build_turn_messages(msg, ws, role, ticket, round_ts).await;

                // Batch-write all messages + session context atomically.
                crate::session::store()
                    .batch_append_with_context(
                        agent_id,
                        &msgs,
                        channel,
                        user_name,
                        &ws.name,
                        role.as_str(),
                    )
                    .await?;
                history.extend(msgs);

                if matches!(role, Role::Artist) {
                    Self::persist_active_models_snapshot(agent_id, &snapshot).await;
                }
            } else {
                // Caching: system prompt is NOT rebuilt on subsequent turns
                // (see doc comment above). The session DB caches the full
                // first-turn message set. The only rebuild path is
                // `apply_summary` below.
                let (content, new_snapshot) = if matches!(role, Role::Artist) {
                    Self::prepend_model_change(agent_id, msg).await
                } else {
                    (msg.to_string(), None)
                };
                let user_msg = crate::session::user_msg_with_ts(&content, round_ts);
                // Append the user message and update session context atomically.
                crate::session::store()
                    .append_with_context(
                        agent_id,
                        &user_msg,
                        channel,
                        user_name,
                        &ws.name,
                        role.as_str(),
                    )
                    .await?;
                // Refresh the change-detection baseline only after the message
                // (carrying the change-info) is persisted — otherwise a failed
                // append would lose the change-info while advancing the
                // baseline, so a retry could never communicate the switch.
                // Residual window: an append that succeeds while the baseline
                // write fails leaves a stale baseline, so the next message
                // re-fires the same change-info — benign and self-healing.
                if let Some(snapshot) = new_snapshot {
                    Self::persist_active_models_snapshot(agent_id, &snapshot).await;
                }
                history.push(user_msg);
            }
        }
        self.history = history;

        // Load the real provider-reported session length (last successful
        // agent LLM call's input + output tokens) so `maybe_summarize` and
        // the Running Agents card see it from the start of the turn. `None`
        // for sessions that never recorded a value — treated as below the
        // summarization threshold.
        self.token_length = crate::session::store().get_token_length(agent_id).await;

        // Session context (channel, user_name, workspace_name, role) was
        // already persisted atomically alongside the messages above —
        // no separate write needed.

        // All loaded/appended history above is persisted — record the prefix
        // so finalize only appends genuinely new assistant output.
        self.persisted_len = self.history.len();

        Ok(())
    }

    // ── History access (for agent loop) ────────────────────────────────

    /// Read-only access to the in-memory history (for LLM calls).
    #[must_use]
    pub(crate) fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Real provider-reported session length (see the field doc), loaded at
    /// [`init`](Self::init). `None` when no successful usage-bearing agent
    /// call ever recorded a value.
    #[must_use]
    pub(crate) fn token_length(&self) -> Option<u64> {
        self.token_length
    }

    /// Update the in-memory session length after a successful agent LLM call.
    /// The durable store write and the observational registry mirror are the
    /// caller's responsibility ([`crate::Agent::record_session_usage`]).
    pub(crate) fn set_token_length(&mut self, token_length: Option<u64>) {
        self.token_length = token_length;
    }

    /// Append an assistant message (final response or intermediate tool-call).
    /// Operates on in-memory history only, not the session store.
    pub(crate) fn push_assistant(&mut self, content: String) {
        self.history.push(ChatMessage::assistant(content));
    }

    /// Persist any unpersisted history tail plus `messages` in one
    /// transaction, then deliver `messages` to in-memory history and advance
    /// `persisted_len` over the whole persisted span. On failure nothing is
    /// delivered — the caller aborts or delivers unpersisted via
    /// [`Session::push_messages_unpersisted`].
    ///
    /// `messages` must not already be part of the unpersisted tail — they
    /// would be written twice.
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
        self.history.extend_from_slice(messages);
        Ok(())
    }

    /// Deliver `messages` to in-memory history without persisting them
    /// (store append failed). They remain in the unpersisted tail; the next
    /// successful persist flushes them in order.
    pub(crate) fn push_messages_unpersisted(&mut self, messages: &[ChatMessage]) {
        self.history.extend_from_slice(messages);
    }

    /// Durable rewrite of the most recent User-role message: persist the new
    /// content to the store FIRST, then swap the in-memory history entry.
    ///
    /// Ordering invariant: the store write happens before the in-memory swap,
    /// so a persist failure leaves the in-memory history untouched and the
    /// caller can fall back to the original error path without a half-stripped
    /// continuation or a swallowed error.
    ///
    /// The store UPDATE targets the last `user`-role row, which corresponds to
    /// the most recent User-role message **only while that message is within
    /// the persisted prefix** (`index < persisted_len`) — `history[..persisted_len]`
    /// mirrors the DB contiguously. A message in the unpersisted tail returns
    /// [`RewriteOutcome::UnpersistedTailNoop`] instead of corrupting an earlier
    /// row. The target message is re-derived here (authoritative), not taken
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
        self.history[idx].content = content;
        Ok(RewriteOutcome::Rewritten)
    }

    // ── Summarization — apply summary produced by Agent ─────────────────

    /// Replace the in-memory and persisted history with a compacted version
    /// containing a fresh system prompt (via `build_context_messages`), the
    /// given `summary_text`, and the latest [`crate::session::RETENTION_PER_SIDE`]
    /// user messages + assistant answers from the pre-compaction history
    /// (tool traffic excluded — see [`crate::session::select_retention_window`]).
    ///
    /// The LLM call to produce the summary text is the responsibility of
    /// [`crate::Agent::summarize`] — this method
    /// handles only the history rebuild and persistence.
    ///
    /// KV-cache preservation: called by Agent after producing a summary
    /// with byte-identical parameters (model, reasoning_effort, tools,
    /// provider routing) so the provider can reuse the cached prefix.
    ///
    /// On persist failure, the full in-memory history is preserved — the
    /// next turn reloads from DB and retries (self-healing).
    pub(crate) async fn apply_summary(
        &mut self,
        agent_id: &str,
        summary_text: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
    ) {
        // Build fresh system prompt (may have changed since session start).
        let (mut compacted, snapshot) = Self::build_context_messages(ws, role, ticket).await;

        // Append conversation summary, then the retained latest turns in
        // chronological order. The in-flight user message is already among
        // them (newest user message) — no separate re-append here.
        let prefix = load_prompt("context/summary_prefix.md");
        compacted.push(ChatMessage::system(format!("{prefix}{summary_text}")));
        compacted.extend(crate::session::select_retention_window(&self.history));

        // Persist compacted history (system prompt + summary + retained window).
        // On success, update in-memory history to match. On failure, keep the
        // full in-memory history — next turn reloads from DB and retries.
        if let Err(e) = crate::session::store()
            .replace_messages(agent_id, &compacted)
            .await
        {
            tracing::error!(
                agent_id = %agent_id,
                error = %e,
                "Failed to persist compacted session after summarization"
            );
        } else {
            self.history = compacted;
            self.persisted_len = self.history.len();
            // Refresh the change-detection baseline when the block was
            // (re)rendered: it is the newest authoritative state — a stale
            // snapshot would fire a spurious change-info on the first turn
            // after every compaction. Sections that did not re-render
            // (catalog down) keep their old baseline so a mid-session switch
            // of that model can still surface as a change line. When no block
            // rendered at all the baseline is left untouched for the same
            // reason — the block itself is dropped from the system prompt
            // until the next compaction or /new (accepted fail-open gap: the
            // block is only re-rendered through this path).
            let block_rendered = snapshot.image.is_some() || snapshot.video.is_some();
            if matches!(role, Role::Artist) && block_rendered {
                let merged = Self::merge_active_models_snapshot(agent_id, &snapshot).await;
                Self::persist_active_models_snapshot(agent_id, &merged).await;
            }
        }
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    /// Flush any unpersisted history tail to the session store and advance
    /// the persisted prefix — the final assistant answer, or just delivered
    /// comments on aborted turns. No-op on an empty tail (e.g. a drain cut
    /// right after the last tool-group commit, or a turn with no output), so
    /// a retained assistant answer is never duplicated. The no-op is surfaced
    /// via [`FinalizeOutcome::NoUnpersistedTail`] for the caller to log with
    /// attribution. Skipped by the agent-level cancel path (see
    /// `finalize_session`).
    pub(crate) async fn finalize(&mut self, agent_id: &str) -> Result<FinalizeOutcome> {
        debug_assert!(self.persisted_len <= self.history.len());
        if self.persisted_len == self.history.len() {
            return Ok(FinalizeOutcome::NoUnpersistedTail);
        }
        self.persist_messages(agent_id, &[]).await?;
        Ok(FinalizeOutcome::Flushed)
    }

    // ── Internals ────────────────────────────────────────────────────

    /// Build fresh system prompt + ticket context for the current turn.
    /// Returns the system-level messages plus the model snapshot rendered in
    /// the `<active-models-opts>` block (Artist only; `Default` when no block
    /// was injected). Messages appear in this order:
    ///
    /// ```text
    /// role_description       — from src/prompt/role/{role}.md (always)
    /// active_models_opts     — Artist only, when the catalogs are available
    /// workspace boilerplate  — from src/prompt/context/workspace.md, substituted (always)
    /// skills                 — if any skills exist in the workspace
    /// board_context          — Manager role only, when active tickets exist
    /// ticket_block           — when a ticket is assigned to this session
    /// ```
    ///
    /// # Caching contract
    /// This function is called directly by
    /// [`Session::apply_summary`] when rebuilding context after compaction.
    /// [`Self::build_turn_messages`] wraps it to add the per-turn user message.
    /// The Artist block is re-emitted on compaction, so long sessions never
    /// lose the capability info.
    async fn build_context_messages(
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
    ) -> (Vec<ChatMessage>, ModelSnapshot) {
        let (stored_context, board_context) = tokio::join!(
            lookup_workspace_context(ws, role),
            build_board_context(ws, role),
        );

        let workspace_context = match stored_context.as_deref() {
            Some(ctx) => ctx.to_owned(),
            None => build_workspace_context(ws.as_path()).await,
        };

        let workspace_context = {
            let mut ctx = String::new();

            // Auto-discovered workspace context block.
            ctx.push_str(&crate::prompt::wrap_workspace_context(&workspace_context));

            // User-curated manual notes block (survives rediscover).
            // NOTE: changes to ws.notes do not affect in-flight agent sessions
            // until summarization compacts the history — identical to all other
            // system prompt changes (documented caching contract).
            if !ws.notes.trim().is_empty() {
                let notes = truncate_workspace_notes(&ws.notes);
                let _ = write!(ctx, "\n<user-notes>\n{notes}\n</user-notes>\n");
            }

            ctx
        };

        let workspace_boilerplate = substitute(
            &load_prompt("context/workspace.md"),
            &[
                ("{{operating_system}}", std::env::consts::OS),
                ("{{workspace}}", &ws.as_path().display().to_string()),
                ("{{workspace_context}}", &workspace_context),
            ],
        );

        let role_description = role.role_description();
        let skills = skills::load_skills(ws).await;

        let mut msgs = Vec::with_capacity(6);
        msgs.push(ChatMessage::system(&role_description));

        // Artist sessions carry the <active-models-opts> block: the active
        // image/video models' parameter envelope, rendered from the live
        // catalogs plus static per-model video-edit nuances. Fail-open — no
        // block when nothing renders (catalogs unavailable, no nuances).
        let mut snapshot = ModelSnapshot::default();
        if matches!(role, Role::Artist)
            && let Some((block, rendered)) = crate::tools::active_models::render_block().await
        {
            msgs.push(ChatMessage::system(block));
            snapshot = rendered;
        }

        msgs.push(ChatMessage::system(&workspace_boilerplate));
        if !skills.is_empty() {
            msgs.push(ChatMessage::system(skills::skills_to_prompt(&skills, ws)));
        }
        if let Some(board_context) = board_context {
            msgs.push(ChatMessage::system(&board_context));
        }
        if let Some(t) = ticket {
            msgs.push(ChatMessage::system(format_ticket_block(t)));
        }
        (msgs, snapshot)
    }

    /// Build fresh system prompt + ticket context + user message for the
    /// current turn.
    /// Returns messages: [role_description, active_models_opts?, workspace_boilerplate,
    /// skills?, board_context?, ticket_block?, user_msg] plus the rendered
    /// active-models snapshot (Artist only; `Default` when no block injected).
    ///
    /// # Caching contract
    /// The output of this function is intended to be persisted to the session
    /// DB on the **first** turn (via `batch_append` in [`Session::init`]) and
    /// loaded from there on subsequent turns. This function is NOT called
    /// every turn — only on new sessions.
    ///
    /// System prompts are cached; user messages carry the timestamp block —
    /// round-pinned for parallel-round first messages, fresh for everything
    /// else (see [`crate::session::render_timestamp`]).
    async fn build_turn_messages(
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
        round_ts: Option<&str>,
    ) -> (Vec<ChatMessage>, ModelSnapshot) {
        let (mut msgs, snapshot) = Self::build_context_messages(ws, role, ticket).await;
        msgs.push(crate::session::user_msg_with_ts(msg, round_ts));
        (msgs, snapshot)
    }

    // ── Active-models snapshot (Artist mid-session change detection) ──

    /// Persist the `<active-models-opts>` baseline after the system prompt
    /// (re)build. Only the model ids actually rendered in the block are
    /// recorded (fail-open: a section absent from the block means no baseline
    /// for that model, so a later switch of it never fires a change-info).
    /// An empty snapshot clears any baseline — used at first turn when no
    /// block rendered (nothing to compare against). At compaction the caller
    /// merges in the preserved baseline for sections that did not re-render
    /// and only calls this when the block (re)rendered.
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

    /// Extend a freshly rendered snapshot with the preserved baseline for
    /// sections that did not render (catalog down), so a mid-session switch
    /// of that model can still surface as a change line.
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

    /// Detect a mid-session image/video model change against the persisted
    /// baseline. When one occurred, returns the message with a change-info
    /// block prepended plus the new snapshot to persist (the caller persists
    /// it only after the message is stored); otherwise the message is
    /// returned unchanged with no snapshot.
    async fn prepend_model_change(agent_id: &str, msg: &str) -> (String, Option<ModelSnapshot>) {
        let Some(previous) = crate::session::store()
            .get_active_models(agent_id)
            .await
            .and_then(|json| ModelSnapshot::from_json(&json))
        else {
            return (msg.to_string(), None);
        };
        let current = ModelSnapshot::from_config();

        // Advance only the changed model(s) from the persisted baseline — a
        // model whose section never rendered (partial-block fail-open session)
        // must not enter the baseline through an unrelated change, or a later
        // switch of it would fire a change-info for an id never shown to the
        // LLM.
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

        // Both catalog fetches run concurrently under join_all (each
        // timeout-bounded internally), so a dual image+video switch adds two
        // concurrent fetches — one bounded wall-clock duration, not two
        // sequential ones.
        let blocks = join_all(
            changed
                .into_iter()
                .map(|(kind, old, new)| Self::model_change_block(kind, old, new)),
        )
        .await;

        // The change-info is the newest authoritative state for the changed
        // model(s) — the caller refreshes the baseline once the message is
        // persisted, so the next message does not re-fire the same change.
        (format!("{}\n\n{msg}", blocks.join("\n")), Some(advanced))
    }

    /// Build a change-info block for one model: "old → new" plus the new
    /// model's capabilities from the catalog (or its static edit nuances when
    /// the catalog is unavailable); a change-only line when neither renders.
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

/// Build a board-state context block for the Manager role.
///
/// Fetches all tickets for the workspace, filters out unblocking phases,
/// and formats them in the same style as the `list_tickets` tool.
/// Returns `None` for non-Manager roles, or when there are no active tickets.
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

/// Try to find a stored workspace context for the given workspace and role.
async fn lookup_workspace_context(ws: &Workspace, role: &Role) -> Option<String> {
    let workspaces = crate::workspace::store();
    workspaces.get_context(&ws.name, role.as_str()).await.ok()?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Panic-safe guard restoring the config model fields mutated by the test.
    struct ModelConfigGuard {
        image: String,
        video: String,
    }

    impl Drop for ModelConfigGuard {
        fn drop(&mut self) {
            let _ = crate::config::CONFIG.set_string_field("image_gen_model", &self.image);
            let _ = crate::config::CONFIG.set_string_field("video_model", &self.video);
        }
    }

    /// Mid-session change detection: switching image AND video models against
    /// a persisted baseline fires one change-info per model (change-only
    /// lines — both shared catalog caches are seeded fail-open, so the change
    /// path performs no network I/O), returns the new snapshot for the caller
    /// to persist after the message append, and does not re-fire once the
    /// baseline is advanced. A missing baseline is a no-op.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn prepend_model_change_detects_switch_and_refreshes_baseline() {
        crate::util::test::init_test_stores().await;
        // Empty-catalog seeds: both catalog lookups return a fresh (empty)
        // catalog with no network fetch, so render_section finds no model and
        // the change path exercises the change-only fallback hermetically.
        // Unlike a None seed (1-min negative-cache Backoff residue), the
        // caches stay Fresh — no cross-test coupling.
        crate::tools::image_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::image_catalog::ImageCatalog::default(),
        )));
        crate::tools::video_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::video_catalog::VideoCatalog::default(),
        )));
        let agent_id = "test_artist_change";

        let _guard = ModelConfigGuard {
            image: crate::config::CONFIG.image_gen_model(),
            video: crate::config::CONFIG.video_model(),
        };
        let _ = crate::config::CONFIG.set_string_field("image_gen_model", "model-a");
        let _ = crate::config::CONFIG.set_string_field("video_model", "video-a");
        let baseline = ModelSnapshot::from_config();
        crate::session::store()
            .set_active_models(agent_id, Some(&baseline.to_json()))
            .await
            .expect("baseline persisted");

        // Switch both models mid-session → one change-info per model.
        let _ = crate::config::CONFIG.set_string_field("image_gen_model", "model-b");
        let _ = crate::config::CONFIG.set_string_field("video_model", "video-b");
        let (out, new_snapshot) = Session::prepend_model_change(agent_id, "hello").await;
        assert!(
            out.starts_with(
                "<model-change>Active image model changed from model-a to model-b.</model-change>\n\
                 <model-change>Active video model changed from video-a to video-b.</model-change>"
            ),
            "unexpected change-info: {out:?}"
        );
        assert!(out.ends_with("\n\nhello"));
        let new_snapshot = new_snapshot.expect("snapshot returned for persistence");

        // Baseline refreshed (as the caller does after a successful append) →
        // the same switches do not re-fire.
        crate::session::store()
            .set_active_models(agent_id, Some(&new_snapshot.to_json()))
            .await
            .expect("baseline advanced");
        let (out, snapshot) = Session::prepend_model_change(agent_id, "again").await;
        assert_eq!(out, "again");
        assert!(snapshot.is_none());
    }

    #[tokio::test]
    async fn prepend_model_change_without_baseline_is_noop() {
        crate::util::test::init_test_stores().await;
        let (out, snapshot) =
            Session::prepend_model_change("test_artist_no_baseline", "hello").await;
        assert_eq!(out, "hello");
        assert!(snapshot.is_none());
    }

    /// Partial-block session (one catalog down at start): switching the model
    /// whose section DID render must not mint a baseline for the never-rendered
    /// model — a later switch of it stays silent.
    #[tokio::test]
    #[serial_test::serial(active_models)]
    async fn prepend_model_change_preserves_absent_sections() {
        crate::util::test::init_test_stores().await;
        crate::tools::image_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::image_catalog::ImageCatalog::default(),
        )));
        crate::tools::video_catalog::seed_cache(Some(std::sync::Arc::new(
            crate::tools::video_catalog::VideoCatalog::default(),
        )));
        let agent_id = "test_artist_partial";

        let _guard = ModelConfigGuard {
            image: crate::config::CONFIG.image_gen_model(),
            video: crate::config::CONFIG.video_model(),
        };
        // Session started with only the video section rendered (image catalog
        // was down at session start) — the image id is NOT in the baseline.
        let _ = crate::config::CONFIG.set_string_field("video_model", "video-a");
        let baseline = ModelSnapshot {
            image: None,
            video: Some("video-a".into()),
        };
        crate::session::store()
            .set_active_models(agent_id, Some(&baseline.to_json()))
            .await
            .expect("baseline persisted");

        // Switch the video model → change-info for video; the returned
        // snapshot advances only video, leaving the absent image baseline
        // untouched.
        let _ = crate::config::CONFIG.set_string_field("video_model", "video-b");
        let (out, new_snapshot) = Session::prepend_model_change(agent_id, "hello").await;
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
        let _ = crate::config::CONFIG.set_string_field("image_gen_model", "image-b");
        let (out, snapshot) = Session::prepend_model_change(agent_id, "again").await;
        assert_eq!(out, "again");
        assert!(snapshot.is_none());
    }

    /// Compaction merge: a rendered snapshot is extended with the preserved
    /// baseline for sections that did not re-render (catalog down), so a
    /// mid-session switch of that model can still surface as a change line.
    #[tokio::test]
    async fn merge_active_models_snapshot_preserves_unrendered_sections() {
        crate::util::test::init_test_stores().await;
        let previous = ModelSnapshot {
            image: Some("model-a".into()),
            video: Some("video-a".into()),
        };
        crate::session::store()
            .set_active_models("test_artist_merge", Some(&previous.to_json()))
            .await
            .expect("baseline persisted");

        // Compaction re-rendered only the image section (video catalog down).
        let rendered = ModelSnapshot {
            image: Some("model-b".into()),
            video: None,
        };
        let merged = Session::merge_active_models_snapshot("test_artist_merge", &rendered).await;
        assert_eq!(
            merged,
            ModelSnapshot {
                image: Some("model-b".into()),
                video: Some("video-a".into()),
            }
        );

        // No prior baseline → the merged snapshot is exactly the rendered one.
        let merged =
            Session::merge_active_models_snapshot("test_artist_no_merge_baseline", &rendered).await;
        assert_eq!(merged, rendered);
    }

    // ── rewrite_last_user_message (input-image-rejection strip durability) ─

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

        // The most recent user message lives in the UNPERSISTED tail
        // (index 1 >= persisted_len 1): a positional UPDATE would rewrite the
        // persisted row instead — conservative no-op.
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

        // Durable row untouched.
        let msgs = crate::session::store().load(agent_id).await;
        assert_eq!(msgs[0].content, "[IMAGE:/tmp/a.png] persisted");
    }

    #[tokio::test]
    async fn rewrite_last_user_message_no_user_message_errors() {
        crate::util::test::init_test_stores().await;
        let mut session = Session {
            history: vec![ChatMessage::system("role")],
            persisted_len: 1,
            ..Default::default()
        };
        let err = session
            .rewrite_last_user_message("test_rewrite_err", "x".to_string())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no user message"), "{err}");
    }

    #[tokio::test]
    async fn rewrite_last_user_message_persist_failure_leaves_history_untouched() {
        crate::util::test::init_test_stores().await;
        let agent_id = "test_rewrite_persist_fail";
        // Seed a DIFFERENT agent's row so the sessions table is non-empty:
        // the positional UPDATE for `agent_id` targets only that agent's last
        // `user` row — none exists, so the store reports 0 rows affected and
        // errors.
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
}

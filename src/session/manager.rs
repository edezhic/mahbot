//! `Session` — single source of truth for conversation persistence.
//!
//! Owns the in-memory history Vec for the current turn. The orchestrator
//! calls [`init`](Session::init) to load/build history, then passes `&mut Session` to the agent loop.
//!
//! ## Usage
//! 1. `Session::default()` at turn start
//! 2. `session.init(agent_id, msg, ws, role, ticket, channel, user_name)` — loads history, builds prompt
//!    for new sessions, persists user message, stores history internally
//! 3. Agent loop calls `session.push_assistant()`, `session.push_messages()`, etc. during tool rounds
//! 4. `session.finalize(agent_id)` on success — persists the final assistant response

use std::fmt::Write;

use anyhow::Result;

use crate::board::BOARD;
use crate::prompt::{build_workspace_context, format_ticket_block, load_prompt, substitute};

use crate::skills;
use crate::{ChatMessage, ChatRole, Role, Workspace};

/// Coordinates session persistence and prompt building across a single
/// agent turn.
#[derive(Default)]
pub struct Session {
    /// In-memory history for the current turn (includes system prompt,
    /// conversation, and the latest user message).
    history: Vec<ChatMessage>,
    /// Number of leading `history` entries already persisted to the store.
    /// [`Session::finalize`] only appends beyond this prefix, so a trailing
    /// assistant answer retained by compaction is never duplicated.
    persisted_len: usize,
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn init(
        &mut self,
        agent_id: &str,
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
        channel: &str,
        user_name: &str,
    ) -> Result<()> {
        // Load existing history from DB
        let mut history = crate::session::store().load(agent_id).await;

        // Empty messages carry no semantic content — skip append in all paths.
        // Recovery retries pass an empty message to re-trigger the agent
        // against the existing session history without adding a new turn.
        if !msg.is_empty() {
            let is_new = history.is_empty();

            if is_new {
                let msgs = Self::build_turn_messages(msg, ws, role, ticket).await;

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
            } else {
                // Caching: system prompt is NOT rebuilt on subsequent turns
                // (see doc comment above). The session DB caches the full
                // first-turn message set. The only rebuild path is
                // `apply_summary` below.
                let user_msg = crate::session::user_msg_with_datetime(msg);
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
                history.push(user_msg);
            }
        }
        self.history = history;

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

    /// Append an assistant message (final response or intermediate tool-call).
    /// Operates on in-memory history only, not the session store.
    pub(crate) fn push_assistant(&mut self, content: String) {
        self.history.push(ChatMessage::assistant(content));
    }

    /// Bulk-append messages to in-memory history.
    /// Operates on in-memory history only, not the session store.
    pub(crate) fn push_messages(&mut self, messages: &[ChatMessage]) {
        self.persisted_len += messages.len();
        self.history.extend_from_slice(messages);
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
        let mut compacted = Self::build_context_messages(ws, role, ticket).await;

        // Append conversation summary, then the retained latest turns in
        // chronological order. The in-flight user message is already among
        // them (newest user message) — no separate re-append here.
        let prefix = load_prompt("summary_prefix.md");
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
        }
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    /// Persist the final assistant message (last assistant entry in history)
    /// to the session store.
    ///
    /// Only appends beyond the already-persisted prefix — after compaction
    /// the retained window may end with a persisted assistant answer, which
    /// must not be duplicated.
    pub(crate) async fn finalize(&self, agent_id: &str) -> Result<()> {
        let Some(final_msg) = self
            .history
            .get(self.persisted_len..)
            .and_then(|tail| tail.last())
            .filter(|m| m.role == ChatRole::Assistant)
        else {
            tracing::warn!("finalize called but no new assistant message in history");
            return Ok(());
        };
        crate::session::store().append(agent_id, final_msg).await?;
        Ok(())
    }

    // ── Internals ────────────────────────────────────────────────────

    /// Build fresh system prompt + ticket context for the current turn.
    /// Returns system-level messages in this order:
    ///
    /// ```text
    /// role_description       — from src/prompt/role/{role}.md (always)
    /// workspace boilerplate  — from src/prompt/workspace.md, substituted (always)
    /// skills                 — if any skills exist in the workspace
    /// board_context          — Manager role only, when active tickets exist
    /// ticket_block           — when a ticket is assigned to this session
    /// ```
    ///
    /// # Caching contract
    /// This function is called directly by
    /// [`Session::apply_summary`] when rebuilding context after compaction.
    /// [`Self::build_turn_messages`] wraps it to add the per-turn user message.
    async fn build_context_messages(
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
    ) -> Vec<ChatMessage> {
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
                // Safe UTF-8 char-level truncation — must not use byte slicing
                // which would panic on multi-byte characters at the boundary.
                let notes: String = ws.notes.chars().take(4000).collect();
                let _ = write!(ctx, "\n<user-notes>\n{notes}\n</user-notes>\n");
            }

            ctx
        };

        let workspace_boilerplate = substitute(
            &load_prompt("workspace.md"),
            &[
                ("{{operating_system}}", std::env::consts::OS),
                ("{{workspace}}", &ws.as_path().display().to_string()),
                ("{{workspace_context}}", &workspace_context),
            ],
        );

        let role_description = role.role_description();
        let skills = skills::load_skills(ws).await;

        let mut msgs = Vec::with_capacity(5);
        msgs.push(ChatMessage::system(&role_description));
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
        msgs
    }

    /// Build fresh system prompt + ticket context + user message for the
    /// current turn.
    /// Returns messages: [role_description, workspace_boilerplate, skills?,
    /// board_context?, ticket_block?, user_msg].
    ///
    /// # Caching contract
    /// The output of this function is intended to be persisted to the session
    /// DB on the **first** turn (via `batch_append` in [`Session::init`]) and
    /// loaded from there on subsequent turns. This function is NOT called
    /// every turn — only on new sessions.
    ///
    /// System prompts are cached; user messages carry per-turn data
    /// (e.g., datetime) that is freshly generated at each call site.
    async fn build_turn_messages(
        msg: &str,
        ws: &Workspace,
        role: &Role,
        ticket: Option<&crate::board::Ticket>,
    ) -> Vec<ChatMessage> {
        let mut msgs = Self::build_context_messages(ws, role, ticket).await;
        msgs.push(crate::session::user_msg_with_datetime(msg));
        msgs
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

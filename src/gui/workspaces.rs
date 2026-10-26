//! Workspaces dashboard page.

use crate::Workspace;
use crate::workspace::truncate_workspace_notes;

use iced::Task;
use iced::widget::markdown;

use std::collections::HashMap;

use super::common::SingleLineEditorState;
use super::editor_widget::EditorAction;

#[derive(Debug, Clone)]
#[expect(private_interfaces)] // ContextKind is deliberately pub(crate) (see gui/mod.rs Message)
pub enum WorkspacesMessage {
    DeleteWorkspace(String),
    ConfirmDelete(String),
    CancelDelete,
    /// A deletion finished: the project removed and the admin's active project
    /// afterwards (`None` = the personal folder), as the deletion's own
    /// transaction left it.
    DeleteResult(Result<(String, Option<String>), String>),
    Reanalyze(String),
    ReanalyzeResult(Result<(), String>),

    /// User clicked a role icon to view per-agent context (read-only markdown).
    ViewContext(String, String), // workspace_name, role

    /// User clicked the general-context icon to view the workspace's
    /// non-role context (read-only markdown).
    ViewGeneralContext(String), // workspace_name

    /// Async fetch of workspace context completed.
    ContextViewed(String, ContextKind, Result<Option<String>, String>), // ws_name, kind, result

    /// Markdown link clicked in the context view.
    LinkClicked(String),

    /// Show diagnostics modal for a workspace.
    ShowDiagnostics(String),

    /// A diagnostics command field was edited: (workspace_name, field_index, new_value).
    /// Field index corresponds to the order in [`crate::DiagnosticsCommands::commands`].
    DiagnosticsFieldEdited(String, usize, EditorAction),
    /// Save diagnostics commands for a workspace.
    SaveDiagnostics(String),
    /// Async result of saving diagnostics.
    DiagnosticsSaved(String, Result<(), String>),
    /// Re-discover diagnostics commands for a workspace (from scratch).
    RediscoverDiagnostics(String),
    /// Async result of re-discovering diagnostics.
    RediscoverDiagnosticsResult(String, Result<(), String>),

    /// Dismiss modals/panels (Escape key or Close button).
    Escape,

    /// Request toast notification.
    Toast(super::ToastMessage),

    // ── Workspace rules popup ────────────────────────────────────
    /// Open the rules popup for a workspace. The popup's backdrop covers the
    /// page, so it is the only surface reachable until it closes.
    OpenRules(String),
    /// Rules editor content changed.
    RulesEdited(super::editor_widget::EditorAction),
    /// Save the rules buffer to the store.
    SaveRules,
    /// Async result of saving rules: (workspace, result).
    RulesSaved(String, Result<(), String>),
    /// Close the rules popup, throwing the buffer away — its explicit Discard.
    RulesDiscard,
    /// Escape or a click outside the popup: close it only when nothing is
    /// unsaved.
    RulesDismiss,
    /// Close the read-only contexts popup.
    ContextsClose,
}

/// Which context entry the read-only context panel is showing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContextKind {
    /// Per-role discovery context (`workspace_contexts.role = <role>`).
    Role(String),
    /// General workspace context (the `role IS NULL` row).
    General,
}

pub struct WorkspacesState {
    pub(crate) workspaces: Vec<Workspace>,
    pub(crate) load_state: super::common::AsyncLoadState,
    pub(crate) delete_target: Option<String>,

    /// Read-only context view modal: (workspace_name, context kind, parsed_markdown_items).
    /// `None` while the modal is not open, `Some` with `None` items while loading.
    pub(crate) context_view: Option<(String, ContextKind, Option<Vec<markdown::Item>>)>,
    pub(crate) context_view_error: Option<String>,

    /// Diagnostics modal: workspace name being viewed/edited.
    pub(crate) diagnostics_modal: Option<String>,
    /// Edit buffers for the 7 diagnostics command fields (when modal is open).
    /// Keyed by workspace name. Each entry is a 7-element array corresponding
    /// to the order in [`crate::DiagnosticsCommands::commands`].
    pub(crate) diagnostics_edit_buffers:
        HashMap<String, [SingleLineEditorState; crate::DiagnosticsCommands::COMMAND_COUNT]>,
    /// Whether a diagnostics save or rediscover operation is in progress.
    pub(crate) diagnostics_busy: bool,
    /// Last save error for diagnostics (resets on modal open).
    pub(crate) diagnostics_error: Option<String>,

    // ── Rules popup ──────────────────────────────────────────────
    /// The open rules popup, or nothing — one at a time, so its buffer and undo
    /// stack live here rather than in a per-workspace map.
    pub(crate) rules_editor: Option<RulesEditor>,
}

/// The focus id of the rules popup's editor — one popup at a time, so one id.
pub(crate) const RULES_EDITOR_ID: &str = "workspace_rules";

/// The open rules popup: the workspace it edits, the text as the store holds
/// it, the unsaved buffer, and the last save failure shown in the popup.
pub(crate) struct RulesEditor {
    pub(crate) workspace: String,
    /// What the store had when the popup opened — the yardstick for "unsaved".
    saved: String,
    pub(crate) buffer: super::editor_widget::EditorBuffer,
    pub(crate) undo: super::common::UndoStack,
    /// A failed save, reported inside the popup; the buffer stays as it was so
    /// a retry needs no retyping.
    pub(crate) error: Option<String>,
}

impl RulesEditor {
    fn new(workspace: String, saved: String) -> Self {
        let buffer = super::editor_widget::EditorBuffer::with_text(
            &saved,
            Some(super::highlight::HighlightLanguage::Markdown),
        );
        // Opened over existing rules to add to them: the caret goes to the end
        // so the first keystroke appends instead of rewriting the first line.
        buffer.move_to_end();
        Self {
            workspace,
            saved,
            buffer,
            undo: super::common::UndoStack::new(),
            error: None,
        }
    }

    /// Whether the buffer holds anything the store does not.
    fn is_dirty(&self) -> bool {
        self.buffer.text() != self.saved
    }
}

impl WorkspacesState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            delete_target: None,
            context_view: None,
            context_view_error: None,
            diagnostics_modal: None,
            diagnostics_edit_buffers: HashMap::new(),
            diagnostics_busy: false,
            diagnostics_error: None,
            rules_editor: None,
        }
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: WorkspacesMessage) -> Task<WorkspacesMessage> {
        // Allow match_same_arms: separate error variants that happen to share the
        // same error-handling body after initial processing (e.g. logging variant
        // info). Narrowing per-arm would duplicate the handler across variants.
        #[expect(clippy::match_same_arms)]
        match msg {
            WorkspacesMessage::DeleteWorkspace(name) => {
                self.delete_target = Some(name);
                Task::none()
            }
            WorkspacesMessage::ConfirmDelete(name) => {
                self.delete_target = None;
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .delete(&name)
                            .await
                            .map(|moved_to| (name, moved_to))
                            .map_err(|e| e.to_string())
                    },
                    WorkspacesMessage::DeleteResult,
                )
            }
            WorkspacesMessage::CancelDelete => {
                self.delete_target = None;
                Task::none()
            }
            // Successful mutations are refreshed by the Dashboard's shared-map
            // reload (see process_settings_message) — no local store read. The
            // Dashboard also reports the deletion.
            WorkspacesMessage::DeleteResult(Ok(_)) => {
                self.load_state.clear_error();
                Task::none()
            }
            WorkspacesMessage::DeleteResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::Reanalyze(name) => Task::perform(
                async move {
                    crate::workspace::store()
                        .rediscover(&name)
                        .await
                        .map_err(|e| e.to_string())
                },
                WorkspacesMessage::ReanalyzeResult,
            ),
            WorkspacesMessage::ReanalyzeResult(Ok(())) => Task::none(),
            WorkspacesMessage::ReanalyzeResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::ViewContext(ws_name, role) => {
                self.context_view = Some((ws_name.clone(), ContextKind::Role(role.clone()), None));
                self.context_view_error = None;
                let ws_name2 = ws_name.clone();
                let role2 = role.clone();
                Task::perform(
                    async move {
                        let content = crate::workspace::store()
                            .get_context(&ws_name, &role)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok::<_, String>((ws_name, role, content))
                    },
                    move |res| match res {
                        Ok((name, role, content)) => WorkspacesMessage::ContextViewed(
                            name,
                            ContextKind::Role(role),
                            Ok(content),
                        ),
                        Err(e) => WorkspacesMessage::ContextViewed(
                            ws_name2,
                            ContextKind::Role(role2),
                            Err(e),
                        ),
                    },
                )
            }
            WorkspacesMessage::ViewGeneralContext(ws_name) => {
                self.context_view = Some((ws_name.clone(), ContextKind::General, None));
                self.context_view_error = None;
                let ws_name2 = ws_name.clone();
                Task::perform(
                    async move {
                        let content = crate::workspace::store()
                            .get_general_context(&ws_name)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok::<_, String>((ws_name, content))
                    },
                    move |res| match res {
                        Ok((name, content)) => WorkspacesMessage::ContextViewed(
                            name,
                            ContextKind::General,
                            Ok(content),
                        ),
                        Err(e) => {
                            WorkspacesMessage::ContextViewed(ws_name2, ContextKind::General, Err(e))
                        }
                    },
                )
            }
            WorkspacesMessage::ContextViewed(ws_name, kind, result) => {
                // A fetch that lands after the popup was dismissed has nowhere
                // to go: reopening it would undo the dismissal.
                if self.context_view.is_none() {
                    return Task::none();
                }
                // No context set yet — an empty item list is the empty state.
                let items = match &result {
                    Ok(Some(content)) => markdown::parse(content).collect(),
                    Ok(None) | Err(_) => Vec::new(),
                };
                self.context_view_error = result.err();
                self.context_view = Some((ws_name, kind, Some(items)));
                Task::none()
            }
            WorkspacesMessage::LinkClicked(_url) => {
                // Handled by the Dashboard (mod.rs) which intercepts this
                // variant to call open_url() before forwarding to update().
                Task::none()
            }
            WorkspacesMessage::ShowDiagnostics(name) => {
                self.diagnostics_modal = Some(name.clone());
                self.diagnostics_busy = false;
                self.diagnostics_error = None;

                // Populate edit buffers from current diagnostics. An absent or
                // empty blob is legitimately empty; a non-empty blob that
                // fails to parse must surface instead of rendering as the
                // "(skipped)" placeholders.
                let raw = self
                    .workspaces
                    .iter()
                    .find(|w| w.name == name)
                    .and_then(|w| w.diagnostics.clone());
                let commands = match raw.as_deref().filter(|json| !json.trim().is_empty()) {
                    Some(json) => match serde_json::from_str::<crate::DiagnosticsCommands>(json) {
                        Ok(cmds) => Some(cmds),
                        Err(e) => {
                            self.diagnostics_error =
                                Some(format!("failed to parse saved diagnostics: {e}"));
                            None
                        }
                    },
                    None => None,
                };
                let fields = commands.map_or(
                    std::array::from_fn(|_| SingleLineEditorState::new("")),
                    |cmds| {
                        std::array::from_fn(|i| {
                            cmds.commands()[i].1.map_or_else(
                                || SingleLineEditorState::new(""),
                                SingleLineEditorState::new,
                            )
                        })
                    },
                );
                self.diagnostics_edit_buffers.insert(name, fields);
                Task::none()
            }
            WorkspacesMessage::DiagnosticsFieldEdited(name, idx, action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                if let Some(buffers) = self.diagnostics_edit_buffers.get_mut(&name) {
                    if idx < crate::DiagnosticsCommands::COMMAND_COUNT {
                        buffers[idx].apply_action(action);
                    }
                }
                Task::none()
            }
            WorkspacesMessage::SaveDiagnostics(name) => {
                self.diagnostics_busy = true;
                self.diagnostics_error = None;

                // Build DiagnosticsCommands from edit buffers using the
                // canonical from_buffers method.
                let buffers: [String; crate::DiagnosticsCommands::COMMAND_COUNT] = match self
                    .diagnostics_edit_buffers
                    .get(&name)
                {
                    Some(buffers) => std::array::from_fn(|i| buffers[i].text()),
                    None => [const { String::new() }; crate::DiagnosticsCommands::COMMAND_COUNT],
                };

                let cmds = crate::DiagnosticsCommands::from_buffers(&buffers);

                let name_clone = name.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .set_diagnostics(&name_clone, &cmds)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| WorkspacesMessage::DiagnosticsSaved(name, result),
                )
            }
            WorkspacesMessage::DiagnosticsSaved(name, Ok(())) => {
                self.diagnostics_busy = false;
                self.diagnostics_edit_buffers.remove(&name);
                self.diagnostics_modal = None;
                Task::none()
            }
            WorkspacesMessage::DiagnosticsSaved(_name, Err(e)) => {
                self.diagnostics_busy = false;
                self.diagnostics_error = Some(e.clone());
                // Keep modal open so user can retry
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::RediscoverDiagnostics(name) => {
                self.diagnostics_busy = true;
                self.diagnostics_error = None;
                let name_for_task = name.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .rediscover_diagnostics(&name_for_task)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| WorkspacesMessage::RediscoverDiagnosticsResult(name, result),
                )
            }
            WorkspacesMessage::RediscoverDiagnosticsResult(name, Ok(())) => {
                self.diagnostics_busy = false;
                self.diagnostics_modal = None;
                self.diagnostics_edit_buffers.remove(&name);
                Task::none()
            }
            WorkspacesMessage::RediscoverDiagnosticsResult(_name, Err(e)) => {
                self.diagnostics_busy = false;
                self.diagnostics_error = Some(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }

            // ── Rules popup ──────────────────────────────────────
            WorkspacesMessage::OpenRules(workspace) => {
                let saved = self
                    .workspaces
                    .iter()
                    .find(|w| w.name == workspace)
                    .map_or_else(String::new, |w| w.notes.clone());
                self.rules_editor = Some(RulesEditor::new(workspace, saved));
                // The popup is the thing that was just asked for: the keyboard
                // goes there, so the first keystroke lands without a click.
                iced::widget::operation::focus(iced::widget::Id::from(RULES_EDITOR_ID))
            }
            WorkspacesMessage::RulesEdited(action) => {
                let Some(rules) = self.rules_editor.as_mut() else {
                    return Task::none();
                };
                // The shared undo-aware path, so Cmd+Z / Cmd+Shift+Z work here
                // like in every other editor.
                super::common::apply_editor_action(&mut rules.buffer, &mut rules.undo, action);
                // The store cuts at the cap; the buffer must not grow past it,
                // or the counter would promise what a save throws away.
                let current = rules.buffer.text();
                let truncated = truncate_workspace_notes(&current);
                if truncated.len() < current.len() {
                    rules.buffer.set_text(&truncated);
                    rules.buffer.move_to_end();
                }
                Task::none()
            }
            WorkspacesMessage::SaveRules => {
                let Some(rules) = self.rules_editor.as_mut() else {
                    return Task::none();
                };
                rules.error = None;
                let name = rules.workspace.clone();
                let text = rules.buffer.text();
                let name_clone = name.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .set_notes(&name_clone, &text)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| WorkspacesMessage::RulesSaved(name, result),
                )
            }
            WorkspacesMessage::RulesSaved(workspace, Ok(())) => {
                if self
                    .rules_editor
                    .as_ref()
                    .is_some_and(|rules| rules.workspace == workspace)
                {
                    self.rules_editor = None;
                }
                Task::none()
            }
            WorkspacesMessage::RulesSaved(workspace, Err(e)) => {
                if let Some(rules) = self.rules_editor.as_mut()
                    && rules.workspace == workspace
                {
                    rules.error = Some(e);
                }
                Task::none()
            }
            WorkspacesMessage::RulesDiscard => {
                self.rules_editor = None;
                Task::none()
            }
            // Escape and a click outside are not ways out of unsaved text: only
            // Save and the explicit Discard are.
            WorkspacesMessage::RulesDismiss => self.dismiss_rules(),
            WorkspacesMessage::ContextsClose => {
                self.context_view = None;
                self.context_view_error = None;
                Task::none()
            }

            // One dismissal closes one surface — the topmost that is open — and
            // leaves the others alone. Unsaved rules survive it.
            WorkspacesMessage::Escape => {
                if self.rules_editor.is_some() {
                    self.dismiss_rules()
                } else if self.context_view.is_some() {
                    self.context_view = None;
                    self.context_view_error = None;
                    Task::none()
                } else if self.diagnostics_modal.is_some() {
                    self.diagnostics_modal = None;
                    self.diagnostics_edit_buffers.clear();
                    self.diagnostics_busy = false;
                    self.diagnostics_error = None;
                    Task::none()
                } else {
                    self.delete_target = None;
                    Task::none()
                }
            }
            WorkspacesMessage::Toast(_) => Task::none(),
        }
    }

    /// Close the rules popup unless it holds unsaved text — the one place that
    /// decides what "unsaved" means for a dismissal. A refused dismissal puts
    /// the focus back in the editor: a click outside has just taken it away.
    fn dismiss_rules(&mut self) -> Task<WorkspacesMessage> {
        let Some(rules) = self.rules_editor.as_ref() else {
            return Task::none();
        };
        if rules.is_dirty() {
            iced::widget::operation::focus(iced::widget::Id::from(RULES_EDITOR_ID))
        } else {
            self.rules_editor = None;
            Task::none()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_diagnostics(diagnostics: Option<String>) -> WorkspacesState {
        let mut state = WorkspacesState::new();
        state.workspaces.push(Workspace {
            name: "ws".to_string(),
            diagnostics,
            ..Workspace::default()
        });
        state
    }

    #[test]
    fn show_diagnostics_surfaces_unparsable_blob() {
        let mut state = state_with_diagnostics(Some("{not json".to_string()));
        let _ = state.update(WorkspacesMessage::ShowDiagnostics("ws".to_string()));
        assert!(state.diagnostics_error.is_some());
    }

    #[test]
    fn show_diagnostics_treats_absent_or_empty_blob_as_empty() {
        for diagnostics in [None, Some(String::new()), Some("  ".to_string())] {
            let mut state = state_with_diagnostics(diagnostics);
            let _ = state.update(WorkspacesMessage::ShowDiagnostics("ws".to_string()));
            assert!(state.diagnostics_error.is_none());
        }
    }

    #[test]
    fn unsaved_rules_survive_escape_and_only_discard_drops_them() {
        let mut state = state_with_diagnostics(None);
        state.workspaces[0].notes = "stored rules".to_string();

        let _ = state.update(WorkspacesMessage::OpenRules("ws".to_string()));
        let _ = state.update(WorkspacesMessage::RulesEdited(EditorAction::Insert('!')));

        let _ = state.update(WorkspacesMessage::Escape);
        assert!(
            state.rules_editor.is_some(),
            "escape must not throw unsaved rules away"
        );
        let _ = state.update(WorkspacesMessage::RulesDismiss);
        assert!(
            state.rules_editor.is_some(),
            "a click outside must not throw unsaved rules away"
        );

        let _ = state.update(WorkspacesMessage::RulesDiscard);
        assert!(state.rules_editor.is_none(), "discard is the way out");

        // A popup the user has not touched closes on the same dismissals.
        let _ = state.update(WorkspacesMessage::OpenRules("ws".to_string()));
        let _ = state.update(WorkspacesMessage::Escape);
        assert!(
            state.rules_editor.is_none(),
            "a clean popup closes on escape"
        );
    }
}

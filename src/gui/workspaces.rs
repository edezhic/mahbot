//! Workspaces dashboard page.

use crate::Workspace;
use crate::workspace::truncate_workspace_notes;

use iced::Task;
use iced::widget::markdown;

use std::collections::{HashMap, HashSet};

use super::common::SingleLineEditorState;
use super::editor_widget::EditorAction;

/// Format the time until the next maintainer run, if applicable.
///
/// Returns `None` when maintenance is disabled.
#[must_use]
pub(crate) fn next_maintenance_label(ws: &Workspace) -> Option<String> {
    if !ws.maintenance_enabled {
        return None;
    }
    let Some(ref last_str) = ws.maintainer_last_run_at else {
        return Some("Next maintenance: pending".to_string());
    };
    let last_time = match crate::db::parse_utc_timestamp(last_str) {
        Ok(dt) => dt,
        Err(e) => {
            tracing::warn!(maintainer_last_run_at = %last_str, error = %e, "Failed to parse maintainer_last_run_at in workspace label, showing 'pending'");
            return Some("Next maintenance: pending".to_string());
        }
    };
    let now = chrono::Utc::now();
    let next_run = last_time
        + chrono::Duration::minutes(
            ws.maintainer_debounce_mins
                .clamp(0, Workspace::MAX_MAINTAINER_DEBOUNCE_MINS),
        );
    let remaining = next_run - now;
    let mins = remaining.num_minutes();
    if mins <= 0 {
        Some("Next maintenance: due now".to_string())
    } else {
        let hours = (mins / 60).cast_unsigned();
        let minutes = (mins % 60).cast_unsigned();
        if hours > 0 {
            Some(format!("Next maintenance in {hours}h {minutes}min"))
        } else {
            Some(format!("Next maintenance in {minutes} min"))
        }
    }
}

#[derive(Debug, Clone)]
#[expect(private_interfaces)] // ContextKind is deliberately pub(crate) (see gui/mod.rs Message)
pub enum WorkspacesMessage {
    DeleteWorkspace(String),
    ConfirmDelete(String),
    CancelDelete,
    DeleteResult(Result<(), String>),
    Reanalyze(String),
    ReanalyzeResult(Result<(), String>),
    ToggleMaintainer(String, bool),
    ToggleResult(Result<(), String>),
    TogglePaused(String, bool),
    PauseResult(String, bool, Result<(), String>), // name, now_paused, result

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

    // ── User notes editor ────────────────────────────────────────
    /// Toggle the notes editor for a workspace.
    ToggleNotes(String),
    /// Notes editor content changed.
    NotesEdited(String, super::editor_widget::EditorAction),
    /// Save notes to DB.
    SaveNotes(String),
    /// Async result of saving notes.
    NotesSaved(String, Result<(), String>),
    /// Discard notes edits and close editor.
    NotesCancel(String),
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
    pub(crate) deleting: bool,

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

    // ── User notes editor ────────────────────────────────────────
    /// Open notes editors per workspace (keyed by workspace name).
    pub(crate) notes_editor_content: HashMap<String, super::editor_widget::EditorBuffer>,
    /// Per-workspace undo stacks for the notes editors, kept in lockstep with
    /// [`notes_editor_content`](Self::notes_editor_content). Each open editors
    /// owns its own undo/redo, matching the per-field undo requirement.
    pub(crate) notes_undo: HashMap<String, super::common::UndoStack>,
    /// Which workspaces have their notes editor expanded.
    pub(crate) notes_open: HashSet<String>,
}

impl WorkspacesState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            delete_target: None,
            deleting: false,
            context_view: None,
            context_view_error: None,
            diagnostics_modal: None,
            diagnostics_edit_buffers: HashMap::new(),
            diagnostics_busy: false,
            diagnostics_error: None,
            notes_editor_content: HashMap::new(),
            notes_undo: HashMap::new(),
            notes_open: HashSet::new(),
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
                self.deleting = true;
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .delete(&name)
                            .await
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
            // reload (see process_settings_message) — no local store read.
            WorkspacesMessage::DeleteResult(Ok(())) => {
                self.deleting = false;
                self.load_state.clear_error();
                Task::none()
            }
            WorkspacesMessage::DeleteResult(Err(e)) => {
                self.deleting = false;
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
            WorkspacesMessage::ToggleMaintainer(name, enabled) => Task::perform(
                async move {
                    crate::workspace::store()
                        .set_maintenance_enabled(&name, enabled)
                        .await
                        .map_err(|e| e.to_string())
                },
                WorkspacesMessage::ToggleResult,
            ),
            WorkspacesMessage::ToggleResult(Ok(())) => Task::none(),
            WorkspacesMessage::ToggleResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::TogglePaused(name, paused) => {
                let name2 = name.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .set_paused(&name, paused)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| WorkspacesMessage::PauseResult(name2, paused, result),
                )
            }
            WorkspacesMessage::PauseResult(name, paused, Ok(())) => Task::done(
                WorkspacesMessage::Toast(super::ToastMessage::SuccessMsg(format!(
                    "Pipeline {} for {name}",
                    if paused { "paused" } else { "resumed" }
                ))),
            ),
            WorkspacesMessage::PauseResult(_, _, Err(e)) => {
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
            WorkspacesMessage::ContextViewed(ws_name, kind, Ok(Some(content))) => {
                let md_items: Vec<markdown::Item> = markdown::parse(&content).collect();
                self.context_view = Some((ws_name, kind, Some(md_items)));
                self.context_view_error = None;
                Task::none()
            }
            WorkspacesMessage::ContextViewed(ws_name, kind, Ok(None)) => {
                // No context set yet — show empty state with empty items
                self.context_view = Some((ws_name, kind, Some(Vec::new())));
                self.context_view_error = None;
                Task::none()
            }
            WorkspacesMessage::ContextViewed(ws_name, kind, Err(e)) => {
                self.context_view = Some((ws_name, kind, Some(Vec::new())));
                self.context_view_error = Some(e);
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

                // Populate edit buffers from current diagnostics (or leave empty).
                let fields = self
                    .workspaces
                    .iter()
                    .find(|w| w.name == name)
                    .and_then(|w| w.diagnostics.as_deref())
                    .and_then(|json| serde_json::from_str::<crate::DiagnosticsCommands>(json).ok())
                    .map_or(
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

            // ── User notes editor ────────────────────────────────
            WorkspacesMessage::ToggleNotes(name) => {
                if self.notes_open.contains(&name) {
                    // Close: discard editor state
                    self.notes_open.remove(&name);
                    self.notes_editor_content.remove(&name);
                    self.notes_undo.remove(&name);
                } else {
                    // Open: initialize editor from current workspace's notes
                    let notes = self
                        .workspaces
                        .iter()
                        .find(|w| w.name == name)
                        .map_or("", |w| w.notes.as_str());
                    self.notes_open.insert(name.clone());
                    self.notes_editor_content.insert(
                        name.clone(),
                        super::editor_widget::EditorBuffer::with_text(
                            notes,
                            Some(super::highlight::HighlightLanguage::Markdown),
                        ),
                    );
                    self.notes_undo
                        .insert(name, super::common::UndoStack::new());
                }
                Task::none()
            }
            WorkspacesMessage::NotesEdited(name, action) => {
                let name_for_entry = name.clone();
                let content = self
                    .notes_editor_content
                    .entry(name_for_entry)
                    .or_insert_with(|| {
                        self.workspaces.iter().find(|w| w.name == name).map_or_else(
                            || {
                                super::editor_widget::EditorBuffer::with_text(
                                    "",
                                    Some(super::highlight::HighlightLanguage::Markdown),
                                )
                            },
                            |w| {
                                super::editor_widget::EditorBuffer::with_text(
                                    &w.notes,
                                    Some(super::highlight::HighlightLanguage::Markdown),
                                )
                            },
                        )
                    });
                // Route through the shared undo-aware path so Cmd+Z / Cmd+Shift+Z
                // undo/redo in the notes editor instead of being a no-op.
                let undo = self
                    .notes_undo
                    .entry(name.clone())
                    .or_insert_with(super::common::UndoStack::new);
                super::common::apply_editor_action(content, undo, action);
                // Enforce cap at the UI level
                let current = content.text().clone();
                let truncated = truncate_workspace_notes(&current);
                if truncated.len() < current.len() {
                    content.set_text(&truncated);
                }
                Task::none()
            }
            WorkspacesMessage::SaveNotes(name) => {
                let notes = self
                    .notes_editor_content
                    .get(&name)
                    .map(|c| c.text().clone())
                    .unwrap_or_default();
                let name_clone = name.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .set_notes(&name_clone, &notes)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| WorkspacesMessage::NotesSaved(name, result),
                )
            }
            WorkspacesMessage::NotesSaved(name, Ok(())) => {
                self.notes_open.remove(&name);
                self.notes_editor_content.remove(&name);
                self.notes_undo.remove(&name);
                Task::none()
            }
            WorkspacesMessage::NotesSaved(_name, Err(e)) => {
                self.load_state.fail(e.clone());
                // Keep editor open so user can retry
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::NotesCancel(name) => {
                self.notes_open.remove(&name);
                self.notes_editor_content.remove(&name);
                self.notes_undo.remove(&name);
                Task::none()
            }

            WorkspacesMessage::Escape => {
                self.delete_target = None;
                self.context_view = None;
                self.context_view_error = None;
                self.diagnostics_modal = None;
                self.diagnostics_edit_buffers.clear();
                self.diagnostics_busy = false;
                self.diagnostics_error = None;
                self.notes_open.clear();
                self.notes_editor_content.clear();
                self.notes_undo.clear();
                Task::none()
            }
            WorkspacesMessage::Toast(_) => Task::none(),
        }
    }
}

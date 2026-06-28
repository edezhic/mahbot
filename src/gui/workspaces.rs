//! Workspaces dashboard page.

#![allow(clippy::match_same_arms)]

use crate::Workspace;

use iced::Task;
use iced::widget::markdown;

/// Format the time until the next maintainer run, if applicable.
///
/// Returns `None` when maintenance is disabled.
#[must_use]
pub(crate) fn next_maintenance_label(ws: &Workspace) -> Option<String> {
    if !ws.maintenance {
        return None;
    }
    let Some(ref last_str) = ws.maintainer_last_run_at else {
        return Some("Next maintenance: pending".to_string());
    };
    let last_time = match crate::turso::parse_utc_timestamp(last_str) {
        Ok(dt) => dt,
        Err(e) => {
            tracing::warn!(maintainer_last_run_at = %last_str, error = %e, "Failed to parse maintainer_last_run_at in workspace label, showing 'pending'");
            return Some("Next maintenance: pending".to_string());
        }
    };
    let now = chrono::Utc::now();
    let next_run = last_time + chrono::Duration::minutes(ws.maintainer_debounce_mins.clamp(0, 240));
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
pub enum WorkspacesMessage {
    Refreshed(Vec<Workspace>),
    RefreshError(String),
    DeleteWorkspace(String),
    ConfirmDelete(String),
    CancelDelete,
    DeleteResult(Result<(), String>),
    Reanalyze(String),
    ReanalyzeResult(Result<(), String>),
    ToggleMaintainer(String, bool),
    ToggleResult(Result<(), String>),

    /// User clicked a role icon to view per-agent context (read-only markdown).
    ViewContext(String, String), // workspace_name, role

    /// Async fetch of workspace context completed.
    ContextViewed(String, String, Result<Option<String>, String>), // ws_name, role, result

    /// Markdown link clicked in the context view.
    LinkClicked(String),

    /// Right-click context menu on a row.
    ContextMenu(usize),

    /// Show diagnostics modal for a workspace.
    ShowDiagnostics(String),

    /// Dismiss modals/panels (Escape key or Close button).
    Escape,

    /// Request toast notification.
    Toast(super::ToastMessage),
}

pub struct WorkspacesState {
    pub(crate) workspaces: Vec<Workspace>,
    pub(crate) load_state: super::common::AsyncLoadState,
    pub(crate) delete_target: Option<String>,
    pub(crate) deleting: bool,

    /// Read-only context view modal: (workspace_name, role, parsed_markdown_items).
    /// `None` while the modal is not open, `Some` with `None` items while loading.
    pub(crate) context_view: Option<(String, String, Option<Vec<markdown::Item>>)>,
    pub(crate) context_view_error: Option<String>,

    /// Right-click context menu target row index.
    pub(crate) context_row: Option<usize>,
    /// Diagnostics modal: workspace name being viewed.
    pub(crate) diagnostics_modal: Option<String>,
}

impl WorkspacesState {
    pub const fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            delete_target: None,
            deleting: false,
            context_view: None,
            context_view_error: None,
            context_row: None,
            diagnostics_modal: None,
        }
    }

    pub fn refresh(&self) -> Task<WorkspacesMessage> {
        Task::perform(
            async {
                crate::workspace::store()
                    .list()
                    .await
                    .map_err(|e| e.to_string())
            },
            |res| match res {
                Ok(ws_list) => WorkspacesMessage::Refreshed(ws_list),
                Err(e) => WorkspacesMessage::RefreshError(e),
            },
        )
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: WorkspacesMessage) -> Task<WorkspacesMessage> {
        match msg {
            WorkspacesMessage::Refreshed(ws_list) => {
                self.workspaces = ws_list;
                self.load_state.finish_loading();
                Task::none()
            }
            WorkspacesMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            WorkspacesMessage::DeleteWorkspace(name) => {
                self.context_row = None;
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
            WorkspacesMessage::DeleteResult(Ok(())) => {
                self.deleting = false;
                self.load_state.clear_error();
                Task::batch([
                    self.refresh(),
                    Task::done(WorkspacesMessage::Toast(super::ToastMessage::Deleted)),
                ])
            }
            WorkspacesMessage::DeleteResult(Err(e)) => {
                self.deleting = false;
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::Reanalyze(name) => {
                self.context_row = None;
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .rediscover(&name)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    WorkspacesMessage::ReanalyzeResult,
                )
            }
            WorkspacesMessage::ReanalyzeResult(Ok(())) => Task::batch([
                self.refresh(),
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::SuccessMsg(
                    "Re-analysis started".into(),
                ))),
            ]),
            WorkspacesMessage::ReanalyzeResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::ToggleMaintainer(name, enabled) => Task::perform(
                async move {
                    crate::workspace::store()
                        .set_maintenance(&name, enabled)
                        .await
                        .map_err(|e| e.to_string())
                },
                WorkspacesMessage::ToggleResult,
            ),
            WorkspacesMessage::ToggleResult(Ok(())) => Task::batch([
                self.refresh(),
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Saved)),
            ]),
            WorkspacesMessage::ToggleResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(WorkspacesMessage::Toast(super::ToastMessage::Error(e)))
            }
            WorkspacesMessage::ViewContext(ws_name, role) => {
                self.context_view = Some((ws_name.clone(), role.clone(), None));
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
                        Ok((name, role, content)) => {
                            WorkspacesMessage::ContextViewed(name, role, Ok(content))
                        }
                        Err(e) => WorkspacesMessage::ContextViewed(ws_name2, role2, Err(e)),
                    },
                )
            }
            WorkspacesMessage::ContextViewed(ws_name, role, Ok(Some(content))) => {
                let md_items: Vec<markdown::Item> = markdown::parse(&content).collect();
                self.context_view = Some((ws_name, role, Some(md_items)));
                self.context_view_error = None;
                Task::none()
            }
            WorkspacesMessage::ContextViewed(ws_name, role, Ok(None)) => {
                // No context set yet — show empty state with empty items
                self.context_view = Some((ws_name, role, Some(Vec::new())));
                self.context_view_error = None;
                Task::none()
            }
            WorkspacesMessage::ContextViewed(ws_name, role, Err(e)) => {
                self.context_view = Some((ws_name, role, Some(Vec::new())));
                self.context_view_error = Some(e);
                Task::none()
            }
            WorkspacesMessage::LinkClicked(_url) => {
                // Handled by the Dashboard (mod.rs) which intercepts this
                // variant to call open_url() before forwarding to update().
                Task::none()
            }
            WorkspacesMessage::ContextMenu(idx) => {
                self.context_row = Some(idx);
                Task::none()
            }
            WorkspacesMessage::ShowDiagnostics(name) => {
                self.context_row = None;
                self.diagnostics_modal = Some(name);
                Task::none()
            }
            WorkspacesMessage::Escape => {
                self.delete_target = None;
                self.context_view = None;
                self.context_view_error = None;
                self.context_row = None;
                self.diagnostics_modal = None;
                Task::none()
            }
            WorkspacesMessage::Toast(_) => Task::none(),
        }
    }
}

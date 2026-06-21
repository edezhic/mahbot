//! Workspaces dashboard page.

use crate::Workspace;

use iced::widget::{
    Column, Row, Space, button, column, container, markdown, mouse_area, row, scrollable, text,
    text_input, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::time::Duration;

use super::theme;
use super::widgets;

/// Format the time until the next maintainer run, if applicable.
///
/// Returns `None` when maintenance is disabled.
#[must_use]
fn next_maintenance_label(ws: &Workspace) -> Option<String> {
    if !ws.maintenance {
        return None;
    }
    let Some(ref last_str) = ws.maintainer_last_run_at else {
        return Some("Next maintenance: pending".to_string());
    };
    let last_time = match chrono::DateTime::parse_from_rfc3339(last_str) {
        Ok(dt) => dt,
        Err(e) => {
            tracing::warn!(maintainer_last_run_at = %last_str, error = %e, "Failed to parse maintainer_last_run_at in workspace label, showing 'pending'");
            return Some("Next maintenance: pending".to_string());
        }
    };
    let now = chrono::Utc::now();
    let next_run = last_time.with_timezone(&chrono::Utc)
        + chrono::Duration::minutes(ws.maintainer_debounce_mins.clamp(0, 240));
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
    AddNameInput(String),
    AddPathInput(String),
    SubmitAdd,
    AddResult(Result<Workspace, String>),
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
    workspaces: Vec<Workspace>,
    error: Option<String>,
    pub(crate) loading: bool,
    /// Whether at least one refresh has completed (prevents "Loading..." flicker
    /// on empty datasets when auto-poll Ticks).
    has_loaded: bool,
    add_name: String,
    add_path: String,
    adding: bool,
    delete_target: Option<String>,
    deleting: bool,

    /// Read-only context view modal: (workspace_name, role, parsed_markdown_items).
    /// `None` while the modal is not open, `Some` with `None` items while loading.
    context_view: Option<(String, String, Option<Vec<markdown::Item>>)>,
    context_view_error: Option<String>,

    /// Right-click context menu target row index.
    context_row: Option<usize>,
    /// Diagnostics modal: workspace name being viewed.
    diagnostics_modal: Option<String>,
}

impl WorkspacesState {
    pub fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            error: None,
            loading: false,
            has_loaded: false,
            add_name: String::new(),
            add_path: String::new(),
            adding: false,
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

    pub fn update(&mut self, msg: WorkspacesMessage) -> Task<WorkspacesMessage> {
        match msg {
            WorkspacesMessage::Refreshed(ws_list) => {
                self.workspaces = ws_list;
                self.loading = false;
                self.has_loaded = true;
                Task::none()
            }
            WorkspacesMessage::RefreshError(e) => {
                self.error = Some(e);
                self.loading = false;
                Task::none()
            }
            WorkspacesMessage::AddNameInput(v) => {
                self.add_name = v;
                Task::none()
            }
            WorkspacesMessage::AddPathInput(v) => {
                self.add_path = v;
                Task::none()
            }
            WorkspacesMessage::SubmitAdd => {
                if self.add_name.is_empty() || self.add_path.is_empty() {
                    self.error = Some("Name and path required".into());
                    return Task::none();
                }
                self.adding = true;
                let name = self.add_name.clone();
                let path = self.add_path.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .add(&name, &path)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    WorkspacesMessage::AddResult,
                )
            }
            WorkspacesMessage::AddResult(Ok(_ws)) => {
                self.adding = false;
                self.add_name.clear();
                self.add_path.clear();
                self.error = None;
                self.refresh()
            }
            WorkspacesMessage::AddResult(Err(e)) => {
                self.adding = false;
                self.error = Some(e);
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
                self.error = None;
                Task::batch([
                    self.refresh(),
                    Task::done(WorkspacesMessage::Toast(super::ToastMessage::Deleted)),
                ])
            }
            WorkspacesMessage::DeleteResult(Err(e)) => {
                self.deleting = false;
                self.error = Some(e.clone());
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
                self.error = Some(e.clone());
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
                self.error = Some(e.clone());
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

    pub fn view(&self) -> Element<'_, WorkspacesMessage> {
        let mut content = column![];

        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(12));
        }

        if self.loading && !self.has_loaded {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.workspaces.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::folder_open::<iced::Theme, iced::Renderer>(),
                "No workspaces configured",
            ));
        } else {
            let mut rows = Column::new().spacing(4);
            for (row_index, ws) in self.workspaces.iter().enumerate() {
                let (status_color, status_bg) = theme::workspace_status_color(&ws.status);
                let maintainer_on = ws.maintenance;

                let delete_btn = if self.delete_target == Some(ws.name.clone()) {
                    row![
                        text("Delete?").size(12).color(theme::STATUS_ERROR),
                        Space::new().width(4),
                        button(text("Yes").size(11).color(theme::STATUS_ERROR))
                            .style(theme::button_danger)
                            .on_press(WorkspacesMessage::ConfirmDelete(ws.name.clone())),
                        Space::new().width(4),
                        button(text("No").size(11))
                            .style(theme::button_secondary)
                            .on_press(WorkspacesMessage::CancelDelete),
                    ]
                } else {
                    row![
                        tooltip(
                            button(
                                lucide::x::<iced::Theme, iced::Renderer>()
                                    .size(18)
                                    .color(theme::STATUS_ERROR),
                            )
                            .style(theme::button_text)
                            .on_press(WorkspacesMessage::DeleteWorkspace(ws.name.clone())),
                            "Delete",
                            tooltip::Position::Top,
                        )
                        .delay(Duration::from_millis(400)),
                    ]
                };

                let ws_row = container(
                    column![
                        row![
                            text(&ws.name)
                                .size(14)
                                .color(theme::TEXT_PRIMARY)
                                .width(Length::Fixed(140.0)),
                            container(text(&ws.status).size(11).color(status_color),)
                                .padding([2, 8])
                                .style(move |_theme: &iced::Theme| container::Style {
                                    background: Some(iced::Background::Color(status_bg)),
                                    border: iced::Border {
                                        radius: 4.0.into(),
                                        ..iced::Border::default()
                                    },
                                    ..container::Style::default()
                                }),
                            Space::new().width(8),
                            text(&ws.path)
                                .size(12)
                                .color(theme::TEXT_MUTED)
                                .width(Length::Fixed(200.0)),
                            Space::new().width(4),
                            // Per-agent context edit buttons (icon only, color-coded)
                            {
                                let mut role_btns = Row::new().spacing(2);
                                for role in <crate::Role as strum::IntoEnumIterator>::iter()
                                    .filter(|r| crate::role::role_info(r).has_discovery)
                                {
                                    let name = role.as_str();
                                    let (color, _bg) = theme::role_badge_color_for(&role);
                                    role_btns = role_btns.push(
                                        button(theme::role_icon(&role).size(11).color(color))
                                            .style(theme::button_text)
                                            .on_press(WorkspacesMessage::ViewContext(
                                                ws.name.clone(),
                                                name.to_string(),
                                            )),
                                    );
                                }
                                role_btns
                            },
                            {
                                // Diagnostics button (icon-only, with tooltip)
                                let diag_color = if ws.diagnostics_updated_at.is_some() {
                                    theme::ACCENT
                                } else {
                                    theme::TEXT_MUTED
                                };
                                tooltip(
                                    button(
                                        lucide::list_checks::<iced::Theme, iced::Renderer>()
                                            .size(11)
                                            .color(diag_color),
                                    )
                                    .style(theme::button_text)
                                    .on_press(WorkspacesMessage::ShowDiagnostics(ws.name.clone())),
                                    if ws.diagnostics_updated_at.is_some() {
                                        "Diagnostics"
                                    } else {
                                        "Diagnostics (not yet discovered)"
                                    },
                                    tooltip::Position::Top,
                                )
                                .delay(Duration::from_millis(400))
                            },
                            Space::new().width(Length::Fill),
                            button(
                                lucide::wrench::<iced::Theme, iced::Renderer>()
                                    .size(11)
                                    .color(if maintainer_on {
                                        theme::ACCENT
                                    } else {
                                        theme::TEXT_MUTED
                                    }),
                            )
                            .style(theme::button_text)
                            .on_press(
                                WorkspacesMessage::ToggleMaintainer(
                                    ws.name.clone(),
                                    !maintainer_on,
                                )
                            ),
                            Space::new().width(4),
                            button(row![
                                lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                                    .size(11)
                                    .color(theme::TEXT_MUTED),
                                Space::new().width(4),
                                text("Re-analyze").size(11),
                            ])
                            .style(theme::button_text)
                            .on_press(WorkspacesMessage::Reanalyze(ws.name.clone())),
                            Space::new().width(4),
                            delete_btn,
                        ]
                        .align_y(Alignment::Center),
                        {
                            // Second line: next maintenance time
                            if let Some(label) = next_maintenance_label(ws) {
                                column![text(label).size(11).color(theme::TEXT_MUTED),]
                            } else {
                                column![]
                            }
                        },
                    ]
                    .spacing(4),
                )
                .padding(8)
                .style(|_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_SURFACE)),
                    border: iced::Border {
                        radius: 4.0.into(),
                        width: 1.0,
                        color: theme::BORDER,
                    },
                    ..container::Style::default()
                });

                // Wrap with mouse_area for right-click context menu
                let row_with_ctx =
                    mouse_area(ws_row).on_right_press(WorkspacesMessage::ContextMenu(row_index));

                rows = rows.push(row_with_ctx);

                // Render context menu action buttons below the row
                if self.context_row == Some(row_index) {
                    let ctx_actions = container(
                        row![
                            button(text("Re-analyze").size(11))
                                .style(theme::button_text)
                                .on_press(WorkspacesMessage::Reanalyze(ws.name.clone())),
                            Space::new().width(4),
                            button(
                                lucide::x::<iced::Theme, iced::Renderer>()
                                    .size(18)
                                    .color(theme::STATUS_ERROR),
                            )
                            .style(theme::button_text)
                            .on_press(WorkspacesMessage::DeleteWorkspace(ws.name.clone())),
                        ]
                        .spacing(4)
                        .padding([2, 8]),
                    )
                    .style(|_t: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                        border: iced::Border {
                            radius: 4.0.into(),
                            width: 1.0,
                            color: theme::BORDER_STRONG,
                        },
                        ..container::Style::default()
                    });
                    rows = rows.push(ctx_actions);
                }
            }
            content = content.push(
                scrollable(rows)
                    .height(Length::Fill)
                    .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
                    .style(theme::scrollbar_style),
            );
        }

        // Add workspace form
        content = content.push(Space::new().height(16));
        content = content.push(
            text("Add Workspace")
                .size(16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
        );
        content = content.push(Space::new().height(8));

        let add_form = row![
            text_input("name", &self.add_name)
                .on_input(WorkspacesMessage::AddNameInput)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(160.0)),
            Space::new().width(8),
            text_input("path", &self.add_path)
                .on_input(WorkspacesMessage::AddPathInput)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(300.0)),
            Space::new().width(8),
            button(text(if self.adding { "Adding..." } else { "Add" }).size(13))
                .style(theme::button_primary)
                .on_press_maybe(
                    if self.adding || self.add_name.is_empty() || self.add_path.is_empty() {
                        None
                    } else {
                        Some(WorkspacesMessage::SubmitAdd)
                    },
                ),
        ]
        .align_y(Alignment::Center);

        content = content.push(add_form);

        let base_elem: Element<_> = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            })
            .into();

        // Context view overlay — read-only markdown
        if let Some((ref _ws_name, ref role, ref md_items_opt)) = self.context_view {
            let title = format!("Context for {role}");

            let body: Element<'_, WorkspacesMessage> = match md_items_opt {
                None => {
                    // Loading state
                    container(text("Loading...").size(13).color(theme::TEXT_MUTED))
                        .width(Length::Fill)
                        .into()
                }
                Some(items) => {
                    let mut view_col = column![];

                    // Error banner
                    if let Some(ref err) = self.context_view_error {
                        view_col = view_col.push(
                            container(text(err).size(12).color(theme::STATUS_ERROR))
                                .padding(8)
                                .style(|_theme: &iced::Theme| container::Style {
                                    background: Some(iced::Background::Color(
                                        iced::Color::from_rgba(1.0, 0.267, 0.4, 0.08),
                                    )),
                                    border: iced::Border {
                                        radius: 4.0.into(),
                                        ..iced::Border::default()
                                    },
                                    ..container::Style::default()
                                }),
                        );
                        view_col = view_col.push(Space::new().height(8));
                    }

                    if items.is_empty() {
                        // No context set
                        view_col = view_col
                            .push(text("Not yet discovered").size(13).color(theme::TEXT_MUTED));
                    } else {
                        // Render markdown in a scrollable container
                        let md: Element<'_, WorkspacesMessage> =
                            iced::widget::markdown::view(items, theme::markdown_settings())
                                .map(WorkspacesMessage::LinkClicked);
                        view_col = view_col.push(
                            container(scrollable(md).direction(scrollable::Direction::Vertical(
                                theme::thin_scrollbar(),
                            )))
                            .padding(4)
                            .height(Length::Fixed(400.0))
                            .style(|_theme: &iced::Theme| container::Style {
                                background: Some(iced::Background::Color(theme::BG_BASE)),
                                border: iced::Border {
                                    radius: 4.0.into(),
                                    width: 1.0,
                                    color: theme::BORDER,
                                },
                                ..Default::default()
                            }),
                        );
                    }

                    view_col = view_col.push(Space::new().height(12));
                    view_col = view_col.push(
                        row![
                            Space::new().width(Length::Fill),
                            button(text("Close").size(13))
                                .style(theme::button_secondary)
                                .on_press(WorkspacesMessage::Escape),
                        ]
                        .align_y(Alignment::Center),
                    );
                    view_col.spacing(4).into()
                }
            };

            let view_container = container(
                column![
                    text(title).size(16).color(theme::TEXT_PRIMARY),
                    Space::new().height(12),
                    body,
                ]
                .spacing(8)
                .width(Length::Fill),
            )
            .width(700)
            .padding(16)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                border: iced::Border {
                    radius: 8.0.into(),
                    width: 1.0,
                    color: theme::BORDER_STRONG,
                },
                ..container::Style::default()
            });

            let centered = container(view_container)
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center);

            iced::widget::stack([base_elem, centered.into()]).into()
        } else if let Some(ref diag_ws_name) = self.diagnostics_modal {
            // Diagnostics modal — read-only table
            let ws = self.workspaces.iter().find(|w| &w.name == diag_ws_name);

            let diag_rows: Element<'_, WorkspacesMessage> =
                match ws.and_then(|w| w.diagnostics.as_deref()) {
                    Some(diag_json) => {
                        match serde_json::from_str::<crate::DiagnosticsCommands>(diag_json) {
                            Ok(cmds) => {
                                let fields: Vec<(&str, String)> = cmds
                                    .commands()
                                    .iter()
                                    .map(|(label, cmd)| {
                                        (*label, cmd.unwrap_or("Not found").to_string())
                                    })
                                    .collect();
                                let mut rows_col = Column::new().spacing(6);
                                for (label, cmd) in &fields {
                                    rows_col = rows_col.push(
                                        row![
                                            text(*label)
                                                .size(12)
                                                .color(theme::TEXT_MUTED)
                                                .width(Length::Fixed(120.0)),
                                            text(cmd.clone())
                                                .size(12)
                                                .color(theme::TEXT_PRIMARY)
                                                .font(iced::Font::MONOSPACE),
                                        ]
                                        .spacing(8),
                                    );
                                }
                                rows_col.into()
                            }
                            Err(_) => text("Failed to parse diagnostics data")
                                .size(12)
                                .color(theme::STATUS_ERROR)
                                .into(),
                        }
                    }
                    None => text("Not yet discovered")
                        .size(13)
                        .color(theme::TEXT_MUTED)
                        .into(),
                };

            let modal_title = format!("Diagnostics: {diag_ws_name}");
            let diag_body = column![
                text(modal_title).size(16).color(theme::TEXT_PRIMARY),
                Space::new().height(16),
                diag_rows,
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Close").size(13))
                        .style(theme::button_secondary)
                        .on_press(WorkspacesMessage::Escape),
                ],
            ]
            .spacing(8)
            .width(Length::Fill);

            let diag_container =
                container(diag_body)
                    .width(500)
                    .padding(24)
                    .style(|_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                        border: iced::Border {
                            radius: 8.0.into(),
                            width: 1.0,
                            color: theme::BORDER_STRONG,
                        },
                        ..container::Style::default()
                    });

            let centered = container(diag_container)
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center);

            iced::widget::stack([base_elem, centered.into()]).into()
        } else {
            base_elem
        }
    }
}

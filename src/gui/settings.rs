//! Settings page — dynamic configuration editor.
//!
//! Reads the current config snapshot from [`crate::config::CONFIG`],
//! presents editable fields organised in sections, and saves changes
//! via [`crate::config::save_and_reload`].
//!
//! Also manages workspaces and users (formerly separate pages), with
//! modal dialogs for add operations.

#![allow(clippy::from_iter_instead_of_collect)]

use crate::Role;
use crate::Workspace;
use crate::config::{CONFIG, ConfigData, ModelRouting, RoleConfig};

use iced::widget::{
    Column, Row, Space, button, column, container, mouse_area, pick_list, row, scrollable, stack,
    text, text_input, toggler, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::time::Duration;

use super::theme;
use super::users;
use super::widgets;
use super::workspaces;

// ── Messages ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    /// Field edits
    ProviderKey(String),
    ProviderEndpoint(String),
    ImageTranscriptionModel(String),
    AudioTranscriptionModel(String),
    TranscriptionProvider(String),
    AudioTranscriptionProvider(String),
    ImageGenModel(String),
    /// Newline-separated list of available image generation models.
    ImageGenModels(String),
    VideoGenModel(String),
    /// Newline-separated list of available video generation models.
    VideoGenModels(String),
    ExaKey(String),
    TelegramToken(String),
    /// Per-role model edits
    RoleModel {
        role: String,
        model: String,
    },
    RoleReasoning {
        role: String,
        effort: String,
    },
    /// Per-model provider routing edits
    ModelRoutingOrder {
        model: String,
        order: String,
    },
    ModelRoutingAllowFallbacks {
        model: String,
        allow: bool,
    },
    /// Actions
    Save,
    SaveResult(Result<(), String>),
    /// Toggle password visibility
    ToggleShowKey,
    ToggleShowExa,
    ToggleShowTelegram,
    // ── Workspace management (sub-messages) ─────────────────────
    /// Wrapped workspace message.
    WorkspaceMsg(workspaces::WorkspacesMessage),
    /// Toggle the add-workspace modal.
    ToggleAddWorkspaceModal,
    /// Add-workspace modal fields.
    AddWorkspaceName(String),
    AddWorkspacePath(String),
    /// Submit the add-workspace modal.
    SubmitAddWorkspace,
    /// Result of workspace add.
    AddWorkspaceResult(Result<Workspace, String>),
    // ── User management (sub-messages) ──────────────────────────
    /// Wrapped user message.
    UserMsg(users::UsersMessage),
    /// Toggle the add-user modal.
    ToggleAddUserModal,
    /// Add-user modal fields.
    AddUserSender(String),
    AddUserPermissions(String),
    /// Submit the add-user modal.
    SubmitAddUser,
    /// Result of user add.
    AddUserResult(Result<(), String>),
    /// Escape key pressed (dismisses modal if open).
    Escape,
}

// ── State ────────────────────────────────────────────────────────

const REASONING_EFFORT_OPTIONS: &[&str] = &["off", "xhigh", "high", "medium", "low", "minimal"];

pub struct SettingsState {
    /// Current editable snapshot, loaded from CONFIG each refresh.
    config: ConfigData,
    /// Whether a save is in progress.
    saving: bool,
    /// Last error message from save.
    error: Option<String>,
    /// Password visibility toggles.
    show_provider_key: bool,
    show_exa_key: bool,
    show_telegram_token: bool,

    // ── Workspace management state ──────────────────────────────
    pub(crate) workspaces_state: workspaces::WorkspacesState,
    /// Whether the add-workspace modal is visible.
    show_add_workspace_modal: bool,
    /// Name field in the add-workspace modal.
    add_workspace_name: String,
    /// Path field in the add-workspace modal.
    add_workspace_path: String,
    /// Whether the add-workspace operation is in flight.
    add_workspace_adding: bool,

    // ── User management state ───────────────────────────────────
    pub(crate) users_state: users::UsersState,
    /// Whether the add-user modal is visible.
    show_add_user_modal: bool,
    /// Name field in the add-user modal.
    add_user_sender: String,
    /// Permissions field in the add-user modal.
    add_user_permissions: String,
    /// Whether the add-user operation is in flight.
    add_user_adding: bool,
}

impl SettingsState {
    pub fn new() -> Self {
        Self {
            config: CONFIG.snapshot(),
            saving: false,
            error: None,
            show_provider_key: false,
            show_exa_key: false,
            show_telegram_token: false,
            workspaces_state: workspaces::WorkspacesState::new(),
            users_state: users::UsersState::new(),
            show_add_workspace_modal: false,
            add_workspace_name: String::new(),
            add_workspace_path: String::new(),
            add_workspace_adding: false,
            show_add_user_modal: false,
            add_user_sender: String::new(),
            add_user_permissions: String::new(),
            add_user_adding: false,
        }
    }

    /// Reload the editable snapshot from the current CONFIG.
    pub fn refresh(&mut self) {
        self.config = CONFIG.snapshot();
        self.error = None;
    }

    /// Whether a modal is currently open (for Escape key routing).
    pub fn is_modal_open(&self) -> bool {
        self.show_add_workspace_modal
            || self.show_add_user_modal
            || self.workspaces_state.context_view.is_some()
            || self.workspaces_state.diagnostics_modal.is_some()
            || self.users_state.delete_target.is_some()
            || self.users_state.bind_target.is_some()
    }

    pub fn update(&mut self, msg: SettingsMessage) -> Task<SettingsMessage> {
        match msg {
            // ── Config field edits ─────────────────────────────
            SettingsMessage::ProviderKey(v) => {
                self.config.provider_key = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::ProviderEndpoint(v) => {
                self.config.provider_endpoint = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::ImageTranscriptionModel(v) => {
                self.config.image_transcription_model = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::AudioTranscriptionModel(v) => {
                self.config.audio_transcription_model = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::TranscriptionProvider(v) => {
                self.config.transcription_provider = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::AudioTranscriptionProvider(v) => {
                self.config.audio_transcription_provider = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::ImageGenModel(v) => {
                self.config.image_gen_model = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::ImageGenModels(v) => {
                self.config.image_gen_models = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::VideoGenModel(v) => {
                self.config.video_gen_model = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::VideoGenModels(v) => {
                self.config.video_gen_models = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::ExaKey(v) => {
                self.config.exa_key = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::TelegramToken(v) => {
                self.config.telegram_bot_token = Some(v).filter(|s| !s.is_empty());
                Task::none()
            }
            SettingsMessage::RoleModel { role, model } => {
                let model_opt = Some(model).filter(|s| !s.is_empty());
                if let Some(existing) = self
                    .config
                    .per_role_configs
                    .iter_mut()
                    .find(|rc| rc.role == role)
                {
                    existing.model = model_opt;
                } else {
                    self.config.per_role_configs.push(RoleConfig {
                        role,
                        model: model_opt,
                        reasoning_effort: None,
                    });
                }
                Task::none()
            }
            SettingsMessage::RoleReasoning { role, effort } => {
                let effort_opt = if effort == "off" {
                    None
                } else {
                    Some(effort).filter(|s| !s.is_empty())
                };
                if let Some(existing) = self
                    .config
                    .per_role_configs
                    .iter_mut()
                    .find(|rc| rc.role == role)
                {
                    existing.reasoning_effort = effort_opt;
                } else {
                    self.config.per_role_configs.push(RoleConfig {
                        role,
                        model: None,
                        reasoning_effort: effort_opt,
                    });
                }
                Task::none()
            }
            SettingsMessage::ModelRoutingOrder { model, order } => {
                let order_opt = Some(order).filter(|s| !s.is_empty());
                if let Some(existing) = self
                    .config
                    .model_routings
                    .iter_mut()
                    .find(|mr| mr.model == model)
                {
                    existing.provider_order = order_opt;
                } else {
                    self.config.model_routings.push(ModelRouting {
                        model,
                        provider_order: order_opt,
                        allow_fallbacks: None,
                    });
                }
                Task::none()
            }
            SettingsMessage::ModelRoutingAllowFallbacks { model, allow } => {
                if let Some(existing) = self
                    .config
                    .model_routings
                    .iter_mut()
                    .find(|mr| mr.model == model)
                {
                    existing.allow_fallbacks = Some(allow);
                } else {
                    self.config.model_routings.push(ModelRouting {
                        model,
                        provider_order: None,
                        allow_fallbacks: Some(allow),
                    });
                }
                Task::none()
            }
            SettingsMessage::ToggleShowKey => {
                self.show_provider_key = !self.show_provider_key;
                Task::none()
            }
            SettingsMessage::ToggleShowExa => {
                self.show_exa_key = !self.show_exa_key;
                Task::none()
            }
            SettingsMessage::ToggleShowTelegram => {
                self.show_telegram_token = !self.show_telegram_token;
                Task::none()
            }
            SettingsMessage::Save => {
                self.saving = true;
                self.error = None;
                let config = self.config.clone();
                Task::perform(
                    async move {
                        crate::config::save_and_reload(&config)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    SettingsMessage::SaveResult,
                )
            }
            SettingsMessage::SaveResult(Ok(())) => {
                self.saving = false;
                self.refresh();
                Task::none()
            }
            SettingsMessage::SaveResult(Err(e)) => {
                self.saving = false;
                self.error = Some(e);
                Task::none()
            }

            // ── Workspace messages ──────────────────────────────
            SettingsMessage::WorkspaceMsg(msg) => self
                .workspaces_state
                .update(msg)
                .map(SettingsMessage::WorkspaceMsg),

            SettingsMessage::ToggleAddWorkspaceModal => {
                self.show_add_workspace_modal = !self.show_add_workspace_modal;
                if !self.show_add_workspace_modal {
                    // Reset form when closing
                    self.add_workspace_name.clear();
                    self.add_workspace_path.clear();
                    self.add_workspace_adding = false;
                }
                Task::none()
            }
            SettingsMessage::AddWorkspaceName(v) => {
                self.add_workspace_name = v;
                Task::none()
            }
            SettingsMessage::AddWorkspacePath(v) => {
                self.add_workspace_path = v;
                Task::none()
            }
            SettingsMessage::SubmitAddWorkspace => {
                if self.add_workspace_name.is_empty() || self.add_workspace_path.is_empty() {
                    return Task::none();
                }
                self.add_workspace_adding = true;
                let name = self.add_workspace_name.clone();
                let path = self.add_workspace_path.clone();
                Task::perform(
                    async move {
                        crate::workspace::store()
                            .add(&name, &path)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    SettingsMessage::AddWorkspaceResult,
                )
            }
            SettingsMessage::AddWorkspaceResult(Ok(_ws)) => {
                self.add_workspace_adding = false;
                self.show_add_workspace_modal = false;
                self.add_workspace_name.clear();
                self.add_workspace_path.clear();
                Task::batch([
                    self.workspaces_state
                        .refresh()
                        .map(SettingsMessage::WorkspaceMsg),
                    Task::done(SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::Toast(super::ToastMessage::Created),
                    )),
                ])
            }
            SettingsMessage::AddWorkspaceResult(Err(e)) => {
                self.add_workspace_adding = false;
                Task::done(SettingsMessage::WorkspaceMsg(
                    workspaces::WorkspacesMessage::Toast(super::ToastMessage::Error(e)),
                ))
            }

            // ── User messages ───────────────────────────────────
            SettingsMessage::UserMsg(msg) => {
                self.users_state.update(msg).map(SettingsMessage::UserMsg)
            }

            SettingsMessage::ToggleAddUserModal => {
                self.show_add_user_modal = !self.show_add_user_modal;
                if !self.show_add_user_modal {
                    // Reset form when closing
                    self.add_user_sender.clear();
                    self.add_user_permissions.clear();
                    self.add_user_adding = false;
                }
                Task::none()
            }
            SettingsMessage::AddUserSender(v) => {
                self.add_user_sender = v;
                Task::none()
            }
            SettingsMessage::AddUserPermissions(v) => {
                self.add_user_permissions = v;
                Task::none()
            }
            SettingsMessage::SubmitAddUser => {
                if self.add_user_sender.is_empty() {
                    return Task::none();
                }
                self.add_user_adding = true;
                let sender = self.add_user_sender.clone();
                let permissions = if self.add_user_permissions.is_empty() {
                    None
                } else {
                    Some(self.add_user_permissions.clone())
                };
                Task::perform(
                    async move {
                        let store = users::user_store()?;
                        store
                            .add_user(&sender, permissions.as_deref())
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    },
                    SettingsMessage::AddUserResult,
                )
            }
            SettingsMessage::AddUserResult(Ok(())) => {
                self.add_user_adding = false;
                self.show_add_user_modal = false;
                self.add_user_sender.clear();
                self.add_user_permissions.clear();
                Task::batch([
                    self.users_state.refresh().map(SettingsMessage::UserMsg),
                    Task::done(SettingsMessage::UserMsg(users::UsersMessage::Toast(
                        super::ToastMessage::Created,
                    ))),
                ])
            }
            SettingsMessage::AddUserResult(Err(e)) => {
                self.add_user_adding = false;
                Task::done(SettingsMessage::UserMsg(users::UsersMessage::Toast(
                    super::ToastMessage::Error(e),
                )))
            }

            SettingsMessage::Escape => {
                if self.show_add_workspace_modal {
                    self.show_add_workspace_modal = false;
                    self.add_workspace_name.clear();
                    self.add_workspace_path.clear();
                    self.add_workspace_adding = false;
                } else if self.show_add_user_modal {
                    self.show_add_user_modal = false;
                    self.add_user_sender.clear();
                    self.add_user_permissions.clear();
                    self.add_user_adding = false;
                } else {
                    return Task::batch([
                        self.workspaces_state
                            .update(workspaces::WorkspacesMessage::Escape)
                            .map(SettingsMessage::WorkspaceMsg),
                        self.users_state
                            .update(users::UsersMessage::Escape)
                            .map(SettingsMessage::UserMsg),
                    ]);
                }
                Task::none()
            }
        }
    }

    // ── View ─────────────────────────────────────────────────────

    pub fn view(&self, active_user: Option<&str>) -> Element<'_, SettingsMessage> {
        // Workspace management section (top)
        let ws_section = self.workspaces_section();

        // User management section (second)
        let us_section = self.users_section(active_user);

        // Existing config sections
        let config_sections = column![
            self.provider_section(),
            Space::new().height(16),
            self.models_section(),
            Space::new().height(16),
            self.reasoning_section(),
            Space::new().height(16),
            self.routing_section(),
            Space::new().height(16),
            self.transcription_section(),
            Space::new().height(16),
            self.generation_section(),
            Space::new().height(16),
            self.integrations_section(),
        ];

        let mut content = column![
            ws_section,
            Space::new().height(16),
            us_section,
            Space::new().height(16),
            config_sections,
        ];

        if let Some(ref err) = self.error {
            content = content.push(Space::new().height(8));
            content = content.push(container(text(err).color(theme::STATUS_ERROR)).padding(8));
        }

        let scroll = scrollable(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
            .style(theme::scrollbar_style);

        // Floating save button near bottom-right
        let save_btn = container(save_button(self.saving))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::End)
            .align_y(Alignment::End)
            .padding(iced::Padding::default().right(20.0).bottom(20.0));

        // Modal overlay (rendered above everything else)
        let modal = self.render_modal_overlay();

        // Stack order: [scroll content, floating save button, modal overlay]
        // so the save button doesn't appear above the modal backdrop.
        let body = stack([scroll.into(), save_btn.into(), modal]);

        container(body)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            })
            .into()
    }

    // ── Workspace management section ──────────────────────────

    /// Render the workspaces section for the Settings page. No inner
    /// scrollable — rows expand the outer Settings scrollable naturally.
    fn workspaces_section(&self) -> Element<'_, SettingsMessage> {
        let ws = &self.workspaces_state;

        let mut rows = Column::new().spacing(4);

        if let Some(ref err) = ws.error {
            rows = rows.push(widgets::error_banner(err));
            rows = rows.push(Space::new().height(8));
        }

        if ws.loading && !ws.has_loaded {
            rows = rows.push(text("Loading...").size(13).color(theme::TEXT_MUTED));
        } else if ws.workspaces.is_empty() {
            rows = rows.push(
                text("No workspaces configured. Add one below.")
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        } else {
            for (row_index, ws_item) in ws.workspaces.iter().enumerate() {
                let (status_color, status_bg) = theme::workspace_status_color(&ws_item.status);
                let maintainer_on = ws_item.maintenance;

                let delete_btn = if ws.delete_target == Some(ws_item.name.clone()) {
                    row![
                        text("Delete?").size(12).color(theme::STATUS_ERROR),
                        Space::new().width(4),
                        button(text("Yes").size(11).color(theme::STATUS_ERROR))
                            .style(theme::button_danger)
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::ConfirmDelete(ws_item.name.clone(),),
                            )),
                        Space::new().width(4),
                        button(text("No").size(11))
                            .style(theme::button_secondary)
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::CancelDelete,
                            )),
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
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::DeleteWorkspace(
                                    ws_item.name.clone(),
                                ),
                            )),
                            "Delete",
                            tooltip::Position::Top,
                        )
                        .delay(Duration::from_millis(400)),
                    ]
                };

                let ws_row = container(
                    column![
                        row![
                            text(&ws_item.name)
                                .size(14)
                                .color(theme::TEXT_PRIMARY)
                                .width(Length::Fixed(140.0)),
                            container(text(&ws_item.status).size(11).color(status_color),)
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
                            text(&ws_item.path)
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
                                            .on_press(SettingsMessage::WorkspaceMsg(
                                                workspaces::WorkspacesMessage::ViewContext(
                                                    ws_item.name.clone(),
                                                    name.to_string(),
                                                ),
                                            )),
                                    );
                                }
                                role_btns
                            },
                            Space::new().width(Length::Fill),
                            // Maintainer toggle
                            button(
                                column![
                                    text("Maint").size(8).color(theme::TEXT_MUTED),
                                    text(if maintainer_on { "ON" } else { "OFF" })
                                        .size(9)
                                        .color(if maintainer_on {
                                            theme::ACCENT
                                        } else {
                                            theme::TEXT_MUTED
                                        },),
                                ]
                                .spacing(0)
                                .align_x(Alignment::Center),
                            )
                            .style(theme::button_text)
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::ToggleMaintainer(
                                    ws_item.name.clone(),
                                    !maintainer_on,
                                ),
                            )),
                            Space::new().width(4),
                            button(row![
                                lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                                    .size(11)
                                    .color(theme::TEXT_MUTED),
                                Space::new().width(4),
                                text("Re-analyze").size(11),
                            ])
                            .style(theme::button_text)
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::Reanalyze(ws_item.name.clone()),
                            )),
                            Space::new().width(4),
                            delete_btn,
                        ]
                        .align_y(Alignment::Center),
                        {
                            // Second line: next maintenance time
                            if let Some(label) = super::workspaces::next_maintenance_label(ws_item)
                            {
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
                    mouse_area(ws_row).on_right_press(SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::ContextMenu(row_index),
                    ));

                rows = rows.push(row_with_ctx);

                // Render context menu action buttons below the row
                if ws.context_row == Some(row_index) {
                    let ctx_actions = container(
                        row![
                            button(text("Re-analyze").size(11))
                                .style(theme::button_text)
                                .on_press(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::Reanalyze(ws_item.name.clone(),),
                                )),
                            Space::new().width(4),
                            button(
                                lucide::x::<iced::Theme, iced::Renderer>()
                                    .size(18)
                                    .color(theme::STATUS_ERROR),
                            )
                            .style(theme::button_text)
                            .on_press(SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::DeleteWorkspace(
                                    ws_item.name.clone(),
                                ),
                            )),
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
        }

        // "Add Workspace" button
        let add_btn = row![
            Space::new().width(Length::Fill),
            button(row![
                lucide::plus::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::ACCENT),
                Space::new().width(4),
                text("Add Workspace").size(13),
            ])
            .style(theme::button_primary)
            .on_press(SettingsMessage::ToggleAddWorkspaceModal),
        ];

        let mut section_content = column![rows, Space::new().height(8), add_btn];

        // Context view overlay — read-only markdown (inline in section)
        if let Some((ref _ws_name, ref role, ref md_items_opt)) = ws.context_view {
            section_content = section_content.push(Space::new().height(16));

            let title = format!("Context for {role}");

            let body: Element<'_, SettingsMessage> = match md_items_opt {
                None => container(text("Loading...").size(13).color(theme::TEXT_MUTED))
                    .width(Length::Fill)
                    .into(),
                Some(items) => {
                    let mut view_col = column![];

                    if let Some(ref err) = ws.context_view_error {
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
                        view_col = view_col
                            .push(text("Not yet discovered").size(13).color(theme::TEXT_MUTED));
                    } else {
                        let md: Element<'_, SettingsMessage> =
                            iced::widget::markdown::view(items, theme::markdown_settings()).map(
                                |url| {
                                    SettingsMessage::WorkspaceMsg(
                                        workspaces::WorkspacesMessage::LinkClicked(url),
                                    )
                                },
                            );
                        view_col = view_col.push(
                            container(scrollable(md).direction(scrollable::Direction::Vertical(
                                theme::thin_scrollbar(),
                            )))
                            .padding(4)
                            .height(Length::Fixed(300.0))
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
                                .on_press(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::Escape,
                                )),
                        ]
                        .align_y(Alignment::Center),
                    );
                    view_col.spacing(4).into()
                }
            };

            let view_container = container(
                column![
                    text(title).size(16).color(theme::TEXT_PRIMARY),
                    Space::new().height(8),
                    body,
                ]
                .padding(16),
            )
            .width(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                border: iced::Border {
                    radius: 8.0.into(),
                    width: 1.0,
                    color: theme::BORDER_STRONG,
                },
                ..container::Style::default()
            });

            section_content = section_content.push(view_container);
        }

        section("Workspaces", section_content)
    }

    /// Render the users section for the Settings page.
    fn users_section(&self, active_user: Option<&str>) -> Element<'_, SettingsMessage> {
        let us = &self.users_state;

        let mut rows = Column::new().spacing(4);

        if let Some(ref err) = us.error {
            rows = rows.push(widgets::error_banner(err));
            rows = rows.push(Space::new().height(8));
        }

        if us.loading && !us.has_loaded {
            rows = rows.push(text("Loading...").size(13).color(theme::TEXT_MUTED));
        } else if us.users.is_empty() {
            rows = rows.push(
                text("No users configured. Add one below.")
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        } else {
            for user in &us.users {
                let is_admin = user.name == "admin";
                let is_active = active_user == Some(user.name.as_str());

                // Switch-user icon column: clickable when not the active user
                let switch_icon: Element<'_, SettingsMessage> = if is_active {
                    container(
                        lucide::user_check::<iced::Theme, iced::Renderer>()
                            .size(18)
                            .color(theme::ACCENT),
                    )
                    .width(Length::Fixed(28.0))
                    .align_x(iced::alignment::Horizontal::Center)
                    .into()
                } else {
                    container(
                        button(
                            lucide::log_in::<iced::Theme, iced::Renderer>()
                                .size(18)
                                .color(theme::TEXT_MUTED),
                        )
                        .style(theme::button_text)
                        .padding(0)
                        .on_press(SettingsMessage::UserMsg(
                            users::UsersMessage::SwitchUser(user.name.clone()),
                        )),
                    )
                    .width(Length::Fixed(28.0))
                    .align_x(iced::alignment::Horizontal::Center)
                    .into()
                };

                let delete_btn = if is_admin {
                    row![]
                } else if Some(&user.name) == us.delete_target.as_ref() {
                    row![
                        text("Delete?").size(12).color(theme::STATUS_ERROR),
                        Space::new().width(4),
                        button(text("Yes").size(11).color(theme::STATUS_ERROR))
                            .style(theme::button_danger)
                            .on_press(SettingsMessage::UserMsg(
                                users::UsersMessage::ConfirmDelete(user.name.clone()),
                            )),
                        Space::new().width(4),
                        button(text("No").size(11))
                            .style(theme::button_secondary)
                            .on_press(SettingsMessage::UserMsg(users::UsersMessage::CancelDelete,)),
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
                            .on_press(SettingsMessage::UserMsg(users::UsersMessage::DeleteUser(
                                user.name.clone()
                            ),)),
                            "Delete",
                            tooltip::Position::Top,
                        )
                        .delay(Duration::from_millis(400)),
                    ]
                };

                let user_row = container(
                    column![
                        row![
                            switch_icon,
                            Space::new().width(8),
                            {
                                let user_elem: Element<'_, SettingsMessage> = if let Some(p) =
                                    user.permissions.as_deref().filter(|p| !p.is_empty())
                                {
                                    row![
                                        text(&user.name).size(14).color(theme::TEXT_PRIMARY),
                                        text(p).size(12).color(theme::TEXT_MUTED),
                                    ]
                                    .spacing(4)
                                    .align_y(Alignment::Center)
                                    .into()
                                } else {
                                    text(&user.name).size(14).color(theme::TEXT_PRIMARY).into()
                                };
                                container(user_elem).width(Length::FillPortion(3))
                            },
                            // Workspace dropdown for inline editing
                            {
                                let ws_value = user.selected_workspace.as_deref().unwrap_or("");
                                let ws_selected = us
                                    .workspace_options
                                    .iter()
                                    .find(|o| o.value == ws_value)
                                    .cloned();
                                pick_list(us.workspace_options.as_slice(), ws_selected, |opt| {
                                    SettingsMessage::UserMsg(users::UsersMessage::UpdateWorkspace(
                                        user.name.clone(),
                                        opt.value,
                                    ))
                                })
                                .style(widgets::pick_list_style)
                                .padding([4, 8])
                                .width(Length::Fixed(140.0))
                            },
                            Space::new().width(6),
                            // Role dropdown for inline editing
                            {
                                let role_selected = user
                                    .selected_role
                                    .as_ref()
                                    .and_then(|name| {
                                        us.role_options.iter().find(|o| o.value == *name)
                                    })
                                    .cloned();
                                pick_list(us.role_options.as_slice(), role_selected, |opt| {
                                    SettingsMessage::UserMsg(users::UsersMessage::UpdateRole(
                                        user.name.clone(),
                                        opt.value,
                                    ))
                                })
                                .style(widgets::pick_list_style)
                                .padding([4, 8])
                                .width(Length::Fixed(120.0))
                            },
                            Space::new().width(Length::FillPortion(7)),
                            delete_btn,
                        ]
                        .align_y(Alignment::Center),
                        // Second row: Telegram channel binding
                        {
                            let telegram_binding =
                                user.channels.iter().find(|c| c.channel == "telegram");
                            if us.bind_target.as_deref() == Some(&user.name) {
                                // Inline binding input open
                                let mut row_elements: Vec<Element<'_, SettingsMessage>> = vec![
                                    text("Telegram:")
                                        .size(12)
                                        .color(theme::TEXT_SECONDARY)
                                        .into(),
                                    Space::new().width(8).into(),
                                    text_input("@username", &us.bind_input)
                                        .on_input(|v| {
                                            SettingsMessage::UserMsg(
                                                users::UsersMessage::BindInputChanged(v),
                                            )
                                        })
                                        .style(widgets::text_input_style)
                                        .size(13)
                                        .padding([2, 6])
                                        .width(Length::Fixed(180.0))
                                        .into(),
                                    Space::new().width(8).into(),
                                ];
                                row_elements.push(
                                    button(
                                        text(if us.binding { "Binding..." } else { "Bind" })
                                            .size(11),
                                    )
                                    .style(theme::button_primary)
                                    .on_press_maybe(if us.bind_input.is_empty() {
                                        None
                                    } else {
                                        Some(SettingsMessage::UserMsg(
                                            users::UsersMessage::SubmitBind(user.name.clone()),
                                        ))
                                    })
                                    .into(),
                                );
                                row_elements.push(
                                    button(text("Cancel").size(11))
                                        .style(theme::button_secondary)
                                        .on_press(SettingsMessage::UserMsg(
                                            users::UsersMessage::CloseBindInput,
                                        ))
                                        .into(),
                                );
                                Row::with_children(row_elements)
                                    .spacing(4)
                                    .align_y(Alignment::Center)
                            } else if let Some(binding) = telegram_binding {
                                // Already bound — show channel info and unbind button
                                let display = binding.identifier.as_str();
                                row![
                                    Space::new().width(26),
                                    lucide::link::<iced::Theme, iced::Renderer>()
                                        .size(11)
                                        .color(theme::ACCENT),
                                    Space::new().width(6),
                                    text("Telegram:").size(12).color(theme::TEXT_MUTED),
                                    Space::new().width(6),
                                    text(display).size(12).color(theme::TEXT_SECONDARY),
                                    Space::new().width(4),
                                    if us.binding {
                                        let e: Element<'_, SettingsMessage> = text("Unbinding...")
                                            .size(11)
                                            .color(theme::TEXT_MUTED)
                                            .into();
                                        e
                                    } else {
                                        button(
                                            lucide::x::<iced::Theme, iced::Renderer>()
                                                .size(11)
                                                .color(theme::TEXT_MUTED),
                                        )
                                        .style(theme::button_text)
                                        .on_press(SettingsMessage::UserMsg(
                                            users::UsersMessage::UnbindChannel(
                                                user.name.clone(),
                                                binding.identifier.clone(),
                                            ),
                                        ))
                                        .into()
                                    },
                                ]
                                .align_y(Alignment::Center)
                            } else {
                                // No Telegram binding — show bind button
                                row![
                                    Space::new().width(26),
                                    lucide::link::<iced::Theme, iced::Renderer>()
                                        .size(11)
                                        .color(theme::TEXT_MUTED),
                                    Space::new().width(6),
                                    text("Not bound").size(12).color(theme::TEXT_MUTED),
                                    Space::new().width(6),
                                    button(row![
                                        lucide::plus::<iced::Theme, iced::Renderer>()
                                            .size(11)
                                            .color(theme::ACCENT),
                                        Space::new().width(3),
                                        text("Bind Telegram").size(11),
                                    ])
                                    .style(theme::button_primary)
                                    .on_press(
                                        SettingsMessage::UserMsg(
                                            users::UsersMessage::OpenBindInput(user.name.clone()),
                                        )
                                    ),
                                ]
                                .align_y(Alignment::Center)
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

                rows = rows.push(user_row);
            }
        }

        // "Add User" button
        let add_btn = row![
            Space::new().width(Length::Fill),
            button(row![
                lucide::plus::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::ACCENT),
                Space::new().width(4),
                text("Add User").size(13),
            ])
            .style(theme::button_primary)
            .on_press(SettingsMessage::ToggleAddUserModal),
        ];

        section("Users", column![rows, Space::new().height(8), add_btn])
    }

    /// Render the add-workspace or add-user modal overlay. Returns a
    /// type-stable placeholder when no modal is open.
    fn render_modal_overlay(&self) -> Element<'_, SettingsMessage> {
        if self.show_add_workspace_modal {
            let dialog = self.add_workspace_dialog();
            modal_with_backdrop(dialog, SettingsMessage::ToggleAddWorkspaceModal)
        } else if self.show_add_user_modal {
            let dialog = self.add_user_dialog();
            modal_with_backdrop(dialog, SettingsMessage::ToggleAddUserModal)
        } else if let Some(ref diag_ws_name) = self.workspaces_state.diagnostics_modal {
            let dialog = self.diagnostics_dialog(diag_ws_name);
            modal_with_backdrop(
                dialog,
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Escape),
            )
        } else {
            // Keep Stack widget type stable
            iced::widget::stack([container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()])
            .into()
        }
    }

    /// Build the add-workspace modal dialog content.
    fn add_workspace_dialog(&self) -> Element<'_, SettingsMessage> {
        container(
            column![
                text("Add Workspace")
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().height(16),
                field_row(
                    "Name",
                    text_input("workspace name", &self.add_workspace_name)
                        .on_input(SettingsMessage::AddWorkspaceName)
                        .style(widgets::text_input_style)
                        .width(Length::Fixed(250.0))
                        .into(),
                    None,
                ),
                Space::new().height(8),
                field_row(
                    "Path",
                    text_input("/path/to/workspace", &self.add_workspace_path)
                        .on_input(SettingsMessage::AddWorkspacePath)
                        .style(widgets::text_input_style)
                        .width(Length::Fixed(250.0))
                        .into(),
                    None,
                ),
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::ToggleAddWorkspaceModal),
                    Space::new().width(8),
                    button(
                        text(if self.add_workspace_adding {
                            "Adding..."
                        } else {
                            "Add"
                        })
                        .size(13),
                    )
                    .style(theme::button_primary)
                    .on_press_maybe(
                        if self.add_workspace_adding
                            || self.add_workspace_name.is_empty()
                            || self.add_workspace_path.is_empty()
                        {
                            None
                        } else {
                            Some(SettingsMessage::SubmitAddWorkspace)
                        }
                    ),
                ]
                .align_y(Alignment::Center),
            ]
            .padding(24),
        )
        .width(Length::Fixed(420.0))
        .style(dialog_container_style)
        .into()
    }

    /// Build the add-user modal dialog content.
    fn add_user_dialog(&self) -> Element<'_, SettingsMessage> {
        container(
            column![
                text("Add User")
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().height(16),
                field_row(
                    "Name",
                    text_input("user name", &self.add_user_sender)
                        .on_input(SettingsMessage::AddUserSender)
                        .style(widgets::text_input_style)
                        .width(Length::Fixed(250.0))
                        .into(),
                    None,
                ),
                Space::new().height(8),
                field_row(
                    "Permissions",
                    text_input("optional", &self.add_user_permissions)
                        .on_input(SettingsMessage::AddUserPermissions)
                        .style(widgets::text_input_style)
                        .width(Length::Fixed(250.0))
                        .into(),
                    None,
                ),
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::ToggleAddUserModal),
                    Space::new().width(8),
                    button(
                        text(if self.add_user_adding {
                            "Adding..."
                        } else {
                            "Add"
                        })
                        .size(13),
                    )
                    .style(theme::button_primary)
                    .on_press_maybe(
                        if self.add_user_adding || self.add_user_sender.is_empty() {
                            None
                        } else {
                            Some(SettingsMessage::SubmitAddUser)
                        }
                    ),
                ]
                .align_y(Alignment::Center),
            ]
            .padding(24),
        )
        .width(Length::Fixed(420.0))
        .style(dialog_container_style)
        .into()
    }

    /// Build the diagnostics modal dialog content for the given workspace.
    fn diagnostics_dialog(&self, diag_ws_name: &str) -> Element<'_, SettingsMessage> {
        let ws = self
            .workspaces_state
            .workspaces
            .iter()
            .find(|w| w.name == diag_ws_name);

        let diag_rows: Element<'_, SettingsMessage> = match ws
            .and_then(|w| w.diagnostics.as_deref())
        {
            Some(diag_json) => {
                match serde_json::from_str::<crate::DiagnosticsCommands>(diag_json) {
                    Ok(cmds) => {
                        let fields: Vec<(&str, String)> = cmds
                            .commands()
                            .iter()
                            .map(|(label, cmd)| (*label, cmd.unwrap_or("Not found").to_string()))
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
        container(
            column![
                text(modal_title).size(16).color(theme::TEXT_PRIMARY),
                Space::new().height(16),
                diag_rows,
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Close").size(13))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::WorkspaceMsg(
                            workspaces::WorkspacesMessage::Escape,
                        )),
                ],
            ]
            .spacing(8)
            .width(Length::Fill)
            .padding(24),
        )
        .width(500)
        .style(dialog_container_style)
        .into()
    }
    // ── Section helpers ──────────────────────────────────────────

    fn provider_section(&self) -> Element<'_, SettingsMessage> {
        section(
            "Provider",
            column![
                field_row(
                    "API Key",
                    password_input(
                        "sk-...",
                        self.config.provider_key.as_deref().unwrap_or_default(),
                        self.show_provider_key,
                        SettingsMessage::ProviderKey,
                        SettingsMessage::ToggleShowKey,
                    ),
                    None,
                ),
                field_row(
                    "Endpoint",
                    text_input(
                        "https://openrouter.ai/api/v1",
                        self.config.provider_endpoint.as_deref().unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::ProviderEndpoint)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
            ],
        )
    }

    fn models_section(&self) -> Element<'_, SettingsMessage> {
        let rows = Role::all_roles().into_iter().map(|role| {
            let key: &str = role.into();
            let info = crate::role::role_info(&role);
            let label = info.display_label;
            let default = info.default_model;
            let current = self
                .config
                .per_role_configs
                .iter()
                .find(|rc| rc.role == key)
                .and_then(|rc| rc.model.clone())
                .unwrap_or_default();
            field_row(
                label,
                text_input(default, &current)
                    .on_input(move |v| SettingsMessage::RoleModel {
                        role: key.to_string(),
                        model: v,
                    })
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                Some(default),
            )
        });
        section("Models (per-role)", Column::from_iter(rows))
    }

    fn reasoning_section(&self) -> Element<'_, SettingsMessage> {
        let rows = Role::all_roles().into_iter().map(|role| {
            let key: &str = role.into();
            let info = crate::role::role_info(&role);
            let label = info.display_label;
            let default = info.default_reasoning_effort;
            let current = self
                .config
                .per_role_configs
                .iter()
                .find(|rc| rc.role == key)
                .and_then(|rc| rc.reasoning_effort.clone())
                .unwrap_or_else(|| default.to_string());
            let effort_buttons = Row::from_iter(REASONING_EFFORT_OPTIONS.iter().map(move |&opt| {
                let is_active = if opt == "off" {
                    current.is_empty()
                } else {
                    current == opt
                };
                let mut btn = button(text(opt).size(11)).padding(2);
                if is_active {
                    btn = btn.style(theme::button_primary);
                } else {
                    btn = btn.style(theme::button_secondary);
                }
                btn = btn.on_press(SettingsMessage::RoleReasoning {
                    role: key.to_string(),
                    effort: opt.to_string(),
                });
                row![
                    {
                        let btn_elem: Element<_> = btn.into();
                        btn_elem
                    },
                    Space::new().width(4),
                ]
                .into()
            }));
            field_row(label, effort_buttons.into(), None)
        });
        section("Reasoning Effort (per-role)", Column::from_iter(rows))
    }

    fn transcription_section(&self) -> Element<'_, SettingsMessage> {
        section(
            "Transcription",
            column![
                field_row(
                    "Image Model",
                    text_input(
                        "qwen/qwen3.6-plus",
                        self.config
                            .image_transcription_model
                            .as_deref()
                            .unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::ImageTranscriptionModel)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Audio Model",
                    text_input(
                        "xiaomi/mimo-v2.5",
                        self.config
                            .audio_transcription_model
                            .as_deref()
                            .unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::AudioTranscriptionModel)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Transcription Provider",
                    text_input(
                        "",
                        self.config
                            .transcription_provider
                            .as_deref()
                            .unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::TranscriptionProvider)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Audio Provider",
                    text_input(
                        "",
                        self.config
                            .audio_transcription_provider
                            .as_deref()
                            .unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::AudioTranscriptionProvider)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
            ],
        )
    }

    fn generation_section(&self) -> Element<'_, SettingsMessage> {
        section(
            "Generation",
            column![
                field_row(
                    "Image Gen Model (active)",
                    text_input(
                        "google/gemini-3.1-flash-image-preview",
                        self.config.image_gen_model.as_deref().unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::ImageGenModel)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Image Gen Models (one per line)",
                    text_input(
                        "one model per line",
                        self.config.image_gen_models.as_deref().unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::ImageGenModels)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(350.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Video Gen Model (active)",
                    text_input(
                        "google/veo-3.1-lite",
                        self.config.video_gen_model.as_deref().unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::VideoGenModel)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(250.0))
                    .into(),
                    None,
                ),
                field_row(
                    "Video Gen Models (one per line)",
                    text_input(
                        "one model per line",
                        self.config.video_gen_models.as_deref().unwrap_or_default(),
                    )
                    .on_input(SettingsMessage::VideoGenModels)
                    .style(super::widgets::text_input_style)
                    .width(Length::Fixed(350.0))
                    .into(),
                    None,
                ),
            ],
        )
    }

    fn integrations_section(&self) -> Element<'_, SettingsMessage> {
        section(
            "Integrations",
            column![
                field_row(
                    "Exa API Key",
                    password_input(
                        "sk-...",
                        self.config.exa_key.as_deref().unwrap_or_default(),
                        self.show_exa_key,
                        SettingsMessage::ExaKey,
                        SettingsMessage::ToggleShowExa,
                    ),
                    None,
                ),
                field_row(
                    "Telegram Bot Token",
                    password_input(
                        "123:abc",
                        self.config
                            .telegram_bot_token
                            .as_deref()
                            .unwrap_or_default(),
                        self.show_telegram_token,
                        SettingsMessage::TelegramToken,
                        SettingsMessage::ToggleShowTelegram,
                    ),
                    Some("Applied automatically on save"),
                ),
            ],
        )
    }

    fn routing_section(&self) -> Element<'_, SettingsMessage> {
        // Collect all unique models that should appear in the routing section:
        // 1. Every role's effective model (override from per_role_configs → hardcoded default)
        // 2. Every model with a saved routing entry (preserves orphaned entries)
        let mut model_names: Vec<String> = Vec::new();
        for role in Role::all_roles() {
            let role_key: &str = role.into();
            let model = self
                .config
                .per_role_configs
                .iter()
                .find(|rc| rc.role == role_key)
                .and_then(|rc| rc.model.clone().filter(|m| !m.is_empty()))
                .unwrap_or_else(|| crate::role::role_info(&role).default_model.to_string());
            if !model_names.contains(&model) {
                model_names.push(model);
            }
        }
        for mr in &self.config.model_routings {
            if !model_names.contains(&mr.model) {
                model_names.push(mr.model.clone());
            }
        }
        model_names.sort();

        let mut rows: Vec<Element<'_, SettingsMessage>> = Vec::new();
        for model_name in &model_names {
            // Look up the current routing entry for this model
            let current = self
                .config
                .model_routings
                .iter()
                .find(|mr| mr.model == *model_name)
                .map_or(
                    ModelRouting {
                        model: model_name.clone(),
                        provider_order: None,
                        allow_fallbacks: None,
                    },
                    Clone::clone,
                );
            let current_order = current.provider_order;
            let current_allow = current.allow_fallbacks;

            let display_name = model_name.clone();
            let order_model = model_name.clone();
            let allow_model = model_name.clone();
            let order_input = text_input("DeepSeek", &current_order.unwrap_or_default())
                .on_input(move |v| SettingsMessage::ModelRoutingOrder {
                    model: order_model.clone(),
                    order: v,
                })
                .style(super::widgets::text_input_style)
                .width(Length::Fixed(250.0));

            let allow_toggle = toggler(current_allow.unwrap_or(false)).on_toggle(move |b| {
                SettingsMessage::ModelRoutingAllowFallbacks {
                    model: allow_model.clone(),
                    allow: b,
                }
            });

            rows.push(
                column![
                    // Model name label (read-only)
                    text(display_name)
                        .font(iced::Font::MONOSPACE)
                        .size(13)
                        .color(theme::TEXT_SECONDARY),
                    Space::new().height(4),
                    field_row(
                        "Provider Order",
                        order_input.into(),
                        Some("Comma-separated provider slugs"),
                    ),
                    field_row("Allow Fallbacks", allow_toggle.into(), None,),
                ]
                .spacing(2)
                .into(),
            );
        }

        // No empty-state needed — defaults from Role::all_roles() always
        // populate the list.

        section("Provider Routing (per-model)", Column::from_iter(rows))
    }
}

// ── Shared widgets ───────────────────────────────────────────────

/// Section heading with a divider line.
fn section<'a>(
    title: &'static str,
    content: Column<'a, SettingsMessage>,
) -> Element<'a, SettingsMessage> {
    column![
        text(title)
            .font(iced::Font::MONOSPACE)
            .size(16)
            .color(theme::ACCENT),
        Space::new().height(4),
        content.spacing(4),
    ]
    .spacing(2)
    .into()
}

/// Label on the left, input on the right, optional hint below.
fn field_row<'a>(
    label: &'static str,
    input: Element<'a, SettingsMessage>,
    hint: Option<&'static str>,
) -> Element<'a, SettingsMessage> {
    let mut row_widget = row![
        text(label).size(13).width(Length::Fixed(180.0)),
        Space::new().width(8),
        input,
    ]
    .align_y(Alignment::Center);

    if let Some(h) = hint {
        row_widget = row_widget.push(Space::new().width(8));
        row_widget = row_widget.push(text(h).size(10).color(theme::TEXT_SECONDARY));
    }

    row_widget.into()
}

/// Password input — masked by default, eye button toggles visibility.
fn password_input<'a>(
    placeholder: &str,
    value: &str,
    show: bool,
    on_input: fn(String) -> SettingsMessage,
    on_toggle: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    let input: Element<_> = text_input(placeholder, value)
        .secure(!show)
        .on_input(on_input)
        .style(super::widgets::text_input_style)
        .width(Length::Fixed(250.0))
        .into();

    let eye_text: Element<_> = if show {
        text("×").size(14.0).into()
    } else {
        text("👁").size(14.0).into()
    };

    row![
        input,
        Space::new().width(4),
        button(eye_text)
            .padding(2)
            .style(theme::button_secondary)
            .on_press(on_toggle),
    ]
    .align_y(Alignment::Center)
    .into()
}

/// Save button — disabled while saving.
fn save_button<'a>(saving: bool) -> Element<'a, SettingsMessage> {
    let content = if saving {
        row![text("⟳").size(14.0), Space::new().width(4), text("Saving…"),]
    } else {
        row![text("✓").size(14.0), Space::new().width(4), text("Save"),]
    };

    let mut btn = button(content).padding(6);
    if saving {
        btn = btn.style(theme::button_secondary);
    } else {
        btn = btn
            .style(theme::button_primary)
            .on_press(SettingsMessage::Save);
    }
    btn.into()
}

/// Wrap a dialog element inside a semi-transparent backdrop, centered on screen.
/// Uses `Stack` to overlay the dialog above the backdrop.
fn modal_with_backdrop(
    dialog: Element<'_, SettingsMessage>,
    on_backdrop_click: SettingsMessage,
) -> Element<'_, SettingsMessage> {
    let backdrop = mouse_area(
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.5,
                ))),
                ..container::Style::default()
            }),
    )
    .on_press(on_backdrop_click);

    let centered = container(dialog)
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Alignment::Center)
        .align_y(Alignment::Center);

    iced::widget::stack([backdrop.into(), centered.into()]).into()
}

/// Style for modal dialog containers: elevated background, rounded border.
fn dialog_container_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(iced::Background::Color(theme::BG_ELEVATED)),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: theme::BORDER_STRONG,
        },
        ..container::Style::default()
    }
}

#[cfg(test)]
mod tests {
    /// Every string config field from [`crate::config::STRING_CONFIG_KEYS`] must have a
    /// corresponding [`SettingsMessage`] variant used in the settings editor view.
    ///
    /// This is a manually-maintained mapping that catches newly-added fields that were
    /// added to `STRING_CONFIG_KEYS` but forgot in the GUI.
    #[test]
    fn settings_message_variants_match_config_fields() {
        // Manual mapping: SettingsMessage variant → config key
        let expected_pairs: &[(&str, &str)] = &[
            ("provider_key", "ProviderKey"),
            ("provider_endpoint", "ProviderEndpoint"),
            ("image_transcription_model", "ImageTranscriptionModel"),
            ("audio_transcription_model", "AudioTranscriptionModel"),
            ("transcription_provider", "TranscriptionProvider"),
            ("audio_transcription_provider", "AudioTranscriptionProvider"),
            ("image_gen_model", "ImageGenModel"),
            ("image_gen_models", "ImageGenModels"),
            ("video_gen_model", "VideoGenModel"),
            ("video_gen_models", "VideoGenModels"),
            ("exa_key", "ExaKey"),
            ("telegram_bot_token", "TelegramToken"),
        ];

        // Count must match
        assert_eq!(
            expected_pairs.len(),
            crate::config::STRING_CONFIG_KEYS.len(),
            "expected_pairs count must match STRING_CONFIG_KEYS count — \
             add or remove entries when config fields change"
        );

        // Verify every config key has a corresponding entry
        for &(config_key, variant_name) in expected_pairs {
            assert!(
                crate::config::STRING_CONFIG_KEYS.contains(&config_key),
                "config key '{config_key}' (mapped to SettingsMessage::{variant_name}) \
                 not found in STRING_CONFIG_KEYS — update STRING_CONFIG_KEYS, \
                 string_fields(), and set_string_field()"
            );
        }

        // Verify every STRING_CONFIG_KEYS entry has a SettingsMessage mapping
        for &config_key in crate::config::STRING_CONFIG_KEYS {
            let found = expected_pairs.iter().any(|(k, _)| *k == config_key);
            assert!(
                found,
                "config key '{config_key}' has no SettingsMessage mapping — \
                 add an entry to expected_pairs"
            );
        }
    }
}

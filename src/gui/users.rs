//! Users dashboard page — manage user preferences.

use crate::users::{FieldUpdate, UserRecord, UserStorage};

use iced::widget::{
    Column, Row, Space, button, column, container, pick_list, row, scrollable, text, text_input,
    tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::time::Duration;

use super::theme;
use super::widgets;

/// De-duplicated access to the global user store.
fn user_store() -> Result<&'static UserStorage, String> {
    crate::users::USER_STORE
        .get()
        .ok_or_else(|| "User store not initialized".to_string())
}

#[derive(Debug, Clone)]
pub enum UsersMessage {
    Refreshed(Vec<UserRecord>, Vec<super::widgets::PickOption>),
    RefreshError(String),
    AddSender(String),
    AddPermissions(String),
    SubmitAdd,
    AddResult(Result<(), String>),
    UpdateRole(String, String),
    UpdateWorkspace(String, String),
    UpdatePermissions(String, String),
    UpdateResult(Result<(), String>),
    DeleteUser(String),
    ConfirmDelete(String),
    CancelDelete,
    DeleteResult(Result<(), String>, String),

    /// Switch active user to this one (icon button on users page).
    SwitchUser(String),

    /// Open the inline Telegram binding input for a user.
    OpenBindInput(String),
    /// Close the inline binding input.
    CloseBindInput,
    /// Inline binding text input changed.
    BindInputChanged(String),
    /// Confirm binding the entered Telegram username to the target user.
    SubmitBind(String),
    /// Unbind a Telegram channel from a user.
    UnbindChannel(String, String),
    /// Result of a bind/unbind operation.
    BindResult(Result<(), String>, String),

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// Request toast notification.
    Toast(super::ToastMessage),
}

pub struct UsersState {
    users: Vec<UserRecord>,
    error: Option<String>,
    pub(crate) loading: bool,
    /// Whether at least one refresh has completed (prevents "Loading..." flicker
    /// on empty datasets when auto-poll Ticks).
    has_loaded: bool,

    // Add form
    add_sender: String,
    add_permissions: String,
    adding: bool,

    // Dropdown options (populated on refresh)
    workspace_options: Vec<super::widgets::PickOption>,
    role_options: Vec<super::widgets::PickOption>,

    // Delete confirmation
    delete_target: Option<String>,
    deleting: bool,

    // Telegram binding inline input (single-target, like delete_target)
    bind_target: Option<String>,
    bind_input: String,
    bind_error: Option<String>,
    binding: bool,
}

impl UsersState {
    pub fn new() -> Self {
        Self {
            users: Vec::new(),
            error: None,
            loading: false,
            has_loaded: false,
            add_sender: String::new(),
            add_permissions: String::new(),
            adding: false,
            workspace_options: Vec::new(),
            role_options: Vec::new(),
            delete_target: None,
            deleting: false,
            bind_target: None,
            bind_input: String::new(),
            bind_error: None,
            binding: false,
        }
    }

    pub fn refresh(&self) -> Task<UsersMessage> {
        Task::perform(
            async {
                let store = user_store()?;
                let users = store.list_users().await.map_err(|e| e.to_string())?;

                // Also load workspace options — prepend "Personal" for NULL.
                let mut ws_options = Vec::new();
                ws_options.push(super::widgets::PickOption {
                    value: String::new(),
                    label: "Personal".to_string(),
                });
                if let Ok(ws_list) = crate::workspace::store().list().await {
                    for ws in ws_list {
                        let display = ws.display_name();
                        ws_options.push(super::widgets::PickOption {
                            value: ws.name,
                            label: display,
                        });
                    }
                }

                Ok::<_, String>((users, ws_options))
            },
            |res| match res {
                Ok((users, ws_options)) => UsersMessage::Refreshed(users, ws_options),
                Err(e) => UsersMessage::RefreshError(e),
            },
        )
    }

    pub fn update(&mut self, msg: UsersMessage) -> Task<UsersMessage> {
        match msg {
            UsersMessage::Refreshed(users, ws_options) => {
                self.users = users;
                self.loading = false;
                self.has_loaded = true;

                self.workspace_options = ws_options;

                // Build role options from Role::iter()
                self.role_options = <crate::Role as strum::IntoEnumIterator>::iter()
                    .map(|r| {
                        let name = r.to_string();
                        super::widgets::PickOption {
                            value: name.clone(),
                            label: name,
                        }
                    })
                    .collect();

                Task::none()
            }
            UsersMessage::RefreshError(e) => {
                self.error = Some(e);
                self.loading = false;
                Task::none()
            }
            UsersMessage::AddSender(v) => {
                self.add_sender = v;
                Task::none()
            }
            UsersMessage::AddPermissions(v) => {
                self.add_permissions = v;
                Task::none()
            }
            UsersMessage::SubmitAdd => {
                if self.add_sender.is_empty() {
                    self.error = Some("Sender name required".into());
                    return Task::none();
                }
                self.adding = true;
                let sender = self.add_sender.clone();
                let permissions = if self.add_permissions.is_empty() {
                    None
                } else {
                    Some(self.add_permissions.clone())
                };
                Task::perform(
                    async move {
                        let store = user_store()?;
                        store
                            .add_user(&sender, permissions.as_deref())
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    },
                    move |res: Result<(), String>| UsersMessage::AddResult(res),
                )
            }
            UsersMessage::AddResult(Ok(())) => {
                self.adding = false;
                self.add_sender.clear();
                self.add_permissions.clear();
                self.error = None;
                Task::batch([
                    self.refresh(),
                    Task::done(UsersMessage::Toast(super::ToastMessage::Created)),
                ])
            }
            UsersMessage::AddResult(Err(e)) => {
                self.adding = false;
                self.error = Some(e.clone());
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
            UsersMessage::UpdateRole(sender, role) => Task::perform(
                async move {
                    let store = user_store()?;
                    let val = if role.is_empty() {
                        FieldUpdate::Clear
                    } else {
                        FieldUpdate::Set(&role)
                    };
                    store
                        .update_user(&sender, val, FieldUpdate::Unchanged, FieldUpdate::Unchanged)
                        .await
                        .map_err(|e| e.to_string())
                },
                UsersMessage::UpdateResult,
            ),
            UsersMessage::UpdateWorkspace(sender, ws) => Task::perform(
                async move {
                    let store = user_store()?;
                    let val = if ws.is_empty() {
                        FieldUpdate::Clear
                    } else {
                        FieldUpdate::Set(&ws)
                    };
                    store
                        .update_user(&sender, FieldUpdate::Unchanged, val, FieldUpdate::Unchanged)
                        .await
                        .map_err(|e| e.to_string())
                },
                UsersMessage::UpdateResult,
            ),
            UsersMessage::UpdatePermissions(sender, perms) => Task::perform(
                async move {
                    let store = user_store()?;
                    let val = if perms.is_empty() {
                        FieldUpdate::Clear
                    } else {
                        FieldUpdate::Set(&perms)
                    };
                    store
                        .update_user(&sender, FieldUpdate::Unchanged, FieldUpdate::Unchanged, val)
                        .await
                        .map_err(|e| e.to_string())
                },
                UsersMessage::UpdateResult,
            ),
            UsersMessage::UpdateResult(Ok(())) => Task::batch([
                self.refresh(),
                Task::done(UsersMessage::Toast(super::ToastMessage::Saved)),
            ]),
            UsersMessage::UpdateResult(Err(e)) => {
                self.error = Some(e.clone());
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
            UsersMessage::DeleteUser(sender) => {
                self.delete_target = Some(sender);
                Task::none()
            }
            UsersMessage::ConfirmDelete(sender) => {
                self.delete_target = None;
                self.deleting = true;
                let s = sender.clone();
                let s_clone = s.clone();
                Task::perform(
                    async move {
                        let store = user_store()?;
                        store.delete_user(&s).await.map_err(|e| e.to_string())
                    },
                    move |res| UsersMessage::DeleteResult(res, s_clone),
                )
            }
            UsersMessage::CancelDelete | UsersMessage::Escape => {
                self.delete_target = None;
                self.bind_target = None;
                self.bind_input.clear();
                self.bind_error = None;
                Task::none()
            }
            UsersMessage::DeleteResult(Ok(()), _deleted_user) => {
                self.deleting = false;
                self.error = None;
                Task::batch([
                    self.refresh(),
                    Task::done(UsersMessage::Toast(super::ToastMessage::Deleted)),
                ])
            }
            UsersMessage::DeleteResult(Err(e), _deleted_user) => {
                self.deleting = false;
                self.error = Some(e.clone());
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
            UsersMessage::Toast(_) => Task::none(),
            UsersMessage::SwitchUser(_) => {
                // Intercepted by Dashboard — no-op in UsersState.
                Task::none()
            }
            UsersMessage::OpenBindInput(user_name) => {
                self.bind_target = Some(user_name);
                self.bind_input.clear();
                self.bind_error = None;
                // Also cancel any pending delete confirmation (mutual exclusion).
                self.delete_target = None;
                Task::none()
            }
            UsersMessage::CloseBindInput => {
                self.bind_target = None;
                self.bind_input.clear();
                self.bind_error = None;
                Task::none()
            }
            UsersMessage::BindInputChanged(v) => {
                self.bind_input = v;
                self.bind_error = None;
                Task::none()
            }
            UsersMessage::SubmitBind(user_name) => {
                if self.bind_input.trim().is_empty() {
                    self.bind_error = Some("Telegram username required".into());
                    return Task::none();
                }
                self.binding = true;
                self.bind_error = None;
                // Strip leading @ if the admin typed it.
                let mut identifier = self.bind_input.trim().to_string();
                if let Some(stripped) = identifier.strip_prefix('@') {
                    identifier = stripped.to_string();
                }
                let user_clone = user_name.clone();
                Task::perform(
                    async move {
                        let store = user_store()?;
                        store
                            .bind_channel(&user_clone, "telegram", &identifier)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |res| UsersMessage::BindResult(res, user_name.clone()),
                )
            }
            UsersMessage::UnbindChannel(user_name, identifier) => {
                self.binding = true;
                self.bind_error = None;
                let user_clone = user_name.clone();
                Task::perform(
                    async move {
                        let store = user_store()?;
                        store
                            .unbind_channel(&user_clone, "telegram", &identifier)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |res| UsersMessage::BindResult(res, user_name.clone()),
                )
            }
            UsersMessage::BindResult(Ok(()), _user_name) => {
                self.binding = false;
                self.bind_target = None;
                self.bind_input.clear();
                self.bind_error = None;
                Task::batch([
                    self.refresh(),
                    Task::done(UsersMessage::Toast(super::ToastMessage::Saved)),
                ])
            }
            UsersMessage::BindResult(Err(e), user_name) => {
                self.binding = false;
                if self.bind_target.as_deref() == Some(&user_name) {
                    self.bind_error = Some(format!("Failed to bind: {e}"));
                } else {
                    self.error = Some(format!("Failed to unbind: {e}"));
                }
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
        }
    }

    pub fn view(&self, active_user: Option<&str>) -> Element<'_, UsersMessage> {
        let mut content = column![];

        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(12));
        }

        if self.loading && !self.has_loaded {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.users.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::user::<iced::Theme, iced::Renderer>(),
                "No users",
            ));
        } else {
            let mut rows = Column::new().spacing(4);
            for user in &self.users {
                let is_admin = user.name == "admin";
                let is_active = active_user == Some(user.name.as_str());

                // Switch-user icon column
                let switch_icon: Element<'_, UsersMessage> = if is_active {
                    // Static indicator — not clickable.
                    container(
                        lucide::user_check::<iced::Theme, iced::Renderer>()
                            .size(18)
                            .color(theme::ACCENT),
                    )
                    .width(Length::Fixed(28.0))
                    .align_x(iced::alignment::Horizontal::Center)
                    .into()
                } else {
                    // Clickable icon-button to switch to this user.
                    container(
                        button(
                            lucide::log_in::<iced::Theme, iced::Renderer>()
                                .size(18)
                                .color(theme::TEXT_MUTED),
                        )
                        .style(theme::button_text)
                        .padding(0)
                        .on_press(UsersMessage::SwitchUser(user.name.clone())),
                    )
                    .width(Length::Fixed(28.0))
                    .align_x(iced::alignment::Horizontal::Center)
                    .into()
                };

                let delete_btn = if is_admin {
                    // Admin cannot be deleted — no delete button.
                    row![]
                } else if Some(&user.name) == self.delete_target.as_ref() {
                    row![
                        text("Delete?").size(12).color(theme::STATUS_ERROR),
                        Space::new().width(4),
                        button(text("Yes").size(11).color(theme::STATUS_ERROR))
                            .style(theme::button_danger)
                            .on_press(UsersMessage::ConfirmDelete(user.name.clone())),
                        Space::new().width(4),
                        button(text("No").size(11))
                            .style(theme::button_secondary)
                            .on_press(UsersMessage::CancelDelete),
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
                            .on_press(UsersMessage::DeleteUser(user.name.clone())),
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
                                let user_elem: Element<'_, UsersMessage> = if let Some(p) =
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
                                let ws_selected = self
                                    .workspace_options
                                    .iter()
                                    .find(|o| o.value == ws_value)
                                    .cloned();
                                pick_list(self.workspace_options.as_slice(), ws_selected, |opt| {
                                    UsersMessage::UpdateWorkspace(user.name.clone(), opt.value)
                                })
                                .style(super::widgets::pick_list_style)
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
                                        self.role_options.iter().find(|o| o.value == *name)
                                    })
                                    .cloned();
                                pick_list(self.role_options.as_slice(), role_selected, |opt| {
                                    UsersMessage::UpdateRole(user.name.clone(), opt.value)
                                })
                                .style(super::widgets::pick_list_style)
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
                            if self.bind_target.as_deref() == Some(&user.name) {
                                // Inline binding input open
                                let mut row_elements: Vec<Element<'_, UsersMessage>> = vec![
                                    text("Telegram:")
                                        .size(12)
                                        .color(theme::TEXT_SECONDARY)
                                        .into(),
                                    Space::new().width(8).into(),
                                    text_input("@username", &self.bind_input)
                                        .on_input(UsersMessage::BindInputChanged)
                                        .style(super::widgets::text_input_style)
                                        .size(13)
                                        .padding([2, 6])
                                        .width(Length::Fixed(180.0))
                                        .into(),
                                    Space::new().width(8).into(),
                                ];
                                if let Some(ref err) = self.bind_error {
                                    row_elements.push(
                                        text(err.as_str())
                                            .size(11)
                                            .color(theme::STATUS_ERROR)
                                            .into(),
                                    );
                                    row_elements.push(Space::new().width(6).into());
                                }
                                let submit_label = if self.binding { "Binding..." } else { "Bind" };
                                row_elements.push(
                                    button(row![
                                        lucide::check::<iced::Theme, iced::Renderer>()
                                            .size(13)
                                            .color(theme::ACCENT),
                                        Space::new().width(3),
                                        text(submit_label).size(12),
                                    ])
                                    .style(theme::button_primary)
                                    .on_press_maybe(
                                        if self.binding || self.bind_input.trim().is_empty() {
                                            None
                                        } else {
                                            Some(UsersMessage::SubmitBind(user.name.clone()))
                                        },
                                    )
                                    .into(),
                                );
                                row_elements.push(Space::new().width(4).into());
                                row_elements.push(
                                    button(
                                        lucide::x::<iced::Theme, iced::Renderer>()
                                            .size(14)
                                            .color(theme::TEXT_MUTED),
                                    )
                                    .style(theme::button_text)
                                    .on_press(UsersMessage::CloseBindInput)
                                    .into(),
                                );
                                Row::from_vec(row_elements)
                            } else if let Some(binding) = telegram_binding {
                                // Bound — show @username with unbind button
                                let display = format!("@{}", binding.identifier);
                                row![
                                    Space::new().width(26),
                                    lucide::link::<iced::Theme, iced::Renderer>()
                                        .size(11)
                                        .color(theme::TEXT_MUTED),
                                    Space::new().width(6),
                                    text(display).size(12).color(theme::TEXT_SECONDARY),
                                    Space::new().width(4),
                                    if self.binding {
                                        let e: Element<'_, UsersMessage> = text("Unbinding...")
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
                                        .on_press(UsersMessage::UnbindChannel(
                                            user.name.clone(),
                                            binding.identifier.clone(),
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
                                    .on_press(UsersMessage::OpenBindInput(user.name.clone())),
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
            content = content.push(
                scrollable(rows)
                    .height(Length::Fill)
                    .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
                    .style(theme::scrollbar_style),
            );
        }

        // Add user form
        content = content.push(Space::new().height(16));
        content = content.push(
            text("Add User")
                .size(16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
        );
        content = content.push(Space::new().height(8));

        let add_form = row![
            text("Name:")
                .size(13)
                .color(theme::TEXT_SECONDARY)
                .width(Length::Fixed(80.0)),
            text_input("user name", &self.add_sender)
                .on_input(UsersMessage::AddSender)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(180.0)),
            Space::new().width(12),
            text("Perms:")
                .size(13)
                .color(theme::TEXT_SECONDARY)
                .width(Length::Fixed(80.0)),
            text_input("optional", &self.add_permissions)
                .on_input(UsersMessage::AddPermissions)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(180.0)),
            Space::new().width(Length::Fill),
            button(row![
                lucide::plus::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::ACCENT),
                Space::new().width(4),
                text(if self.adding { "Adding..." } else { "Add" }).size(13),
            ])
            .style(theme::button_primary)
            .on_press_maybe(if self.adding || self.add_sender.is_empty() {
                None
            } else {
                Some(UsersMessage::SubmitAdd)
            }),
        ]
        .align_y(Alignment::Center);

        content = content.push(add_form);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            })
            .into()
    }
}

//! Users dashboard page — manage user preferences.

use crate::users::{FieldUpdate, UserRecord, UserStorage};

use iced::Task;

/// De-duplicated access to the global user store.
pub(crate) fn user_store() -> Result<&'static UserStorage, String> {
    crate::users::USER_STORE
        .get()
        .ok_or_else(|| "User store not initialized".to_string())
}

#[derive(Debug, Clone)]
pub enum UsersMessage {
    Refreshed(Vec<UserRecord>, Vec<super::widgets::PickOption>),
    RefreshError(String),
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
    pub(crate) users: Vec<UserRecord>,
    pub(crate) load_state: super::common::AsyncLoadState,

    // Dropdown options (populated on refresh)
    pub(crate) workspace_options: Vec<super::widgets::PickOption>,
    pub(crate) role_options: Vec<super::widgets::PickOption>,

    // Delete confirmation
    pub(crate) delete_target: Option<String>,
    pub(crate) deleting: bool,

    // Telegram binding inline input (single-target, like delete_target)
    pub(crate) bind_target: Option<String>,
    pub(crate) bind_input: String,
    pub(crate) bind_error: Option<String>,
    pub(crate) binding: bool,
}

impl UsersState {
    pub const fn new() -> Self {
        Self {
            users: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
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
                self.load_state.finish_loading();

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
                self.load_state.fail(e);
                Task::none()
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
                self.load_state.error = Some(e.clone());
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
                self.load_state.error = None;
                Task::batch([
                    self.refresh(),
                    Task::done(UsersMessage::Toast(super::ToastMessage::Deleted)),
                ])
            }
            UsersMessage::DeleteResult(Err(e), _deleted_user) => {
                self.deleting = false;
                self.load_state.error = Some(e.clone());
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
                    self.load_state.error = Some(format!("Failed to unbind: {e}"));
                }
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
        }
    }
}

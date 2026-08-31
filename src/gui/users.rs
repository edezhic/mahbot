//! Users dashboard page — manage user preferences.

use crate::Role;
use crate::users::{FieldUpdate, UserRecord, UserStore};

use std::collections::HashMap;

use strum::IntoEnumIterator;

use iced::Task;

use super::common::SingleLineEditorState;
use super::editor_widget::EditorAction;

/// De-duplicated access to the global user store.
pub(crate) fn user_store() -> Result<&'static UserStore, String> {
    crate::users::USER_STORE
        .get()
        .ok_or_else(|| "User store not initialized".to_string())
}

/// Run a single-field `update_user`, mapping an empty value to
/// [`FieldUpdate::Clear`]. `is_role` selects which column is updated.
pub(crate) async fn update_user_field(
    sender: String,
    value: String,
    is_role: bool,
) -> Result<(), String> {
    let store = user_store()?;
    let val = if value.is_empty() {
        FieldUpdate::Clear
    } else {
        FieldUpdate::Set(&value)
    };
    let (role, ws) = if is_role {
        (val, FieldUpdate::Unchanged)
    } else {
        (FieldUpdate::Unchanged, val)
    };
    store
        .update_user(&sender, role, ws, FieldUpdate::Unchanged)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Debug, Clone)]
pub enum UsersMessage {
    Refreshed(Vec<UserRecord>),
    RefreshError(String),
    UpdateRole(String, String),
    UpdateWorkspace(String, String),
    UpdateResult(Result<(), String>),
    /// Result of an active-role change via the users-table picker.
    RoleUpdateResult(Result<(), String>),
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
    BindInputChanged(EditorAction),
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
    /// Per-user active-role picker options, restricted to each user's pool.
    pub(crate) active_role_options: HashMap<String, Vec<super::widgets::PickOption>>,

    // Delete confirmation
    pub(crate) delete_target: Option<String>,
    pub(crate) deleting: bool,

    // Telegram binding inline input (single-target, like delete_target)
    pub(crate) bind_target: Option<String>,
    pub(crate) bind_input: SingleLineEditorState,
    pub(crate) bind_error: Option<String>,
    pub(crate) binding: bool,
}

impl UsersState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            users: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            workspace_options: Vec::new(),
            role_options: Vec::new(),
            active_role_options: HashMap::new(),
            delete_target: None,
            deleting: false,
            bind_target: None,
            bind_input: SingleLineEditorState::new(""),
            bind_error: None,
            binding: false,
        }
    }

    #[allow(clippy::unused_self)]
    pub fn refresh(&self) -> Task<UsersMessage> {
        Task::perform(
            async {
                let store = user_store()?;
                let users = store.list_users().await.map_err(|e| e.to_string())?;
                Ok::<_, String>(users)
            },
            |res| match res {
                Ok(users) => UsersMessage::Refreshed(users),
                Err(e) => UsersMessage::RefreshError(e),
            },
        )
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: UsersMessage) -> Task<UsersMessage> {
        match msg {
            UsersMessage::Refreshed(users) => {
                self.users = users;

                // workspace_options is synchronized from the dashboard's shared
                // workspace map (see gui/mod.rs) rather than read here.

                // Build role options from Role::iter()
                self.role_options = Role::iter()
                    .map(|r| {
                        let name = r.to_string();
                        super::widgets::PickOption {
                            value: name.clone(),
                            label: name,
                        }
                    })
                    .collect();

                // Per-user active-role options, restricted to each user's pool.
                self.active_role_options = self
                    .users
                    .iter()
                    .map(|u| {
                        let options = u
                            .roles
                            .iter()
                            .filter_map(|name| {
                                self.role_options.iter().find(|o| o.value == *name).cloned()
                            })
                            .collect();
                        (u.name.clone(), options)
                    })
                    .collect();

                Task::none()
            }
            UsersMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            UsersMessage::UpdateRole(sender, role) => Task::perform(
                async move { update_user_field(sender, role, true).await },
                UsersMessage::RoleUpdateResult,
            ),
            UsersMessage::UpdateWorkspace(sender, ws) => Task::perform(
                async move { update_user_field(sender, ws, false).await },
                UsersMessage::UpdateResult,
            ),
            UsersMessage::UpdateResult(Ok(())) | UsersMessage::RoleUpdateResult(Ok(())) => {
                self.refresh()
            }
            UsersMessage::UpdateResult(Err(e)) | UsersMessage::RoleUpdateResult(Err(e)) => {
                self.load_state.fail(e.clone());
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
            UsersMessage::DeleteUser(sender) => {
                self.delete_target = Some(sender);
                Task::none()
            }
            UsersMessage::ConfirmDelete(sender) => {
                self.delete_target = None;
                self.deleting = true;
                let s = sender;
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
                self.load_state.clear_error();
                self.refresh()
            }
            UsersMessage::DeleteResult(Err(e), _deleted_user) => {
                self.deleting = false;
                self.load_state.fail(e.clone());
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
            UsersMessage::BindInputChanged(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.bind_input.apply_action(action);
                self.bind_error = None;
                Task::none()
            }
            UsersMessage::SubmitBind(user_name) => {
                if self.bind_input.text().trim().is_empty() {
                    self.bind_error = Some("Telegram username required".into());
                    return Task::none();
                }
                self.binding = true;
                self.bind_error = None;
                // Strip leading @ if the admin typed it.
                let mut identifier = self.bind_input.text().trim().to_string();
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
                    move |res| UsersMessage::BindResult(res, user_name),
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
                    move |res| UsersMessage::BindResult(res, user_name),
                )
            }
            UsersMessage::BindResult(Ok(()), _user_name) => {
                self.binding = false;
                self.bind_target = None;
                self.bind_input.clear();
                self.bind_error = None;
                self.refresh()
            }
            UsersMessage::BindResult(Err(e), user_name) => {
                self.binding = false;
                if self.bind_target.as_deref() == Some(&user_name) {
                    self.bind_error = Some(format!("Failed to bind: {e}"));
                } else {
                    self.load_state.fail(format!("Failed to unbind: {e}"));
                }
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
        }
    }
}

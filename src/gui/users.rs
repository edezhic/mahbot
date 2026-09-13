//! Settings Users section state — user list, delete confirmation, and Telegram
//! channel binding.

use crate::users::{FieldUpdate, UserRecordEntry, UserStore};

use iced::Task;

use super::common::SingleLineEditorState;
use super::editor_widget::EditorAction;

/// De-duplicated access to the global user store.
pub(crate) fn user_store() -> Result<&'static UserStore, String> {
    crate::users::USER_STORE
        .get()
        .ok_or_else(|| "User store not initialized".to_string())
}

/// Run a single-field `update_user`, mapping an empty value (or a `personal:{user}`
/// workspace name) to [`FieldUpdate::Clear`] — the personal workspace is stored
/// as NULL and computed on the fly. Updates the workspace column.
pub(crate) async fn update_user_field(sender: String, workspace: String) -> Result<(), String> {
    let store = user_store()?;
    // Empty and personal-workspace values both mean "no shared workspace
    // selected" → NULL.
    let val = if workspace.is_empty() || crate::users::is_personal_workspace(&workspace) {
        FieldUpdate::Clear
    } else {
        FieldUpdate::Set(&workspace)
    };
    store
        .update_user(&sender, FieldUpdate::Unchanged, val, FieldUpdate::Unchanged)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Debug, Clone)]
pub enum UsersMessage {
    Refreshed(Vec<UserRecordEntry>),
    RefreshError(String),
    DeleteUser(String),
    ConfirmDelete(String),
    CancelDelete,
    DeleteResult(Result<(), String>),

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
    pub(crate) users: Vec<UserRecordEntry>,
    pub(crate) load_state: super::common::AsyncLoadState,

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
                // A successful read replaces the previous failure.
                self.load_state.clear_error();

                Task::none()
            }
            UsersMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            UsersMessage::DeleteUser(sender) => {
                self.delete_target = Some(sender);
                Task::none()
            }
            UsersMessage::ConfirmDelete(sender) => {
                self.delete_target = None;
                self.deleting = true;
                Task::perform(
                    async move {
                        let store = user_store()?;
                        store.delete_user(&sender).await.map_err(|e| e.to_string())
                    },
                    UsersMessage::DeleteResult,
                )
            }
            UsersMessage::CancelDelete | UsersMessage::Escape => {
                self.delete_target = None;
                self.bind_target = None;
                self.bind_input.clear();
                self.bind_error = None;
                Task::none()
            }
            UsersMessage::DeleteResult(Ok(())) => {
                self.deleting = false;
                self.load_state.clear_error();
                self.refresh()
            }
            UsersMessage::DeleteResult(Err(e)) => {
                self.deleting = false;
                self.load_state.fail(e.clone());
                Task::done(UsersMessage::Toast(super::ToastMessage::Error(e)))
            }
            UsersMessage::Toast(_) => Task::none(),
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
                self.binding = true;
                self.bind_error = None;
                let user_clone = user_name.clone();
                let input = self.bind_input.text().clone();
                Task::perform(
                    async move {
                        let store = user_store()?;
                        let identifier = store
                            .validate_telegram_bind(&user_clone, &input)
                            .await
                            .map_err(|e| e.to_string())?;
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

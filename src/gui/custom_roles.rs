//! Custom roles area state: the admin-defined agent records — the list, the
//! create/edit window's state and the pending deletion. The Settings page
//! renders it (see [`super::settings`]).
//!
//! The list only ever changes through the admin's own actions, so it is read on
//! navigation and after every mutation instead of being polled.

use crate::custom_role::{self, CustomRole, CustomRoleBehavior};

use iced::Task;

use super::editor_widget::{EditorAction, EditorBuffer};

#[derive(Debug, Clone)]
pub(crate) enum CustomRolesMessage {
    /// Read the list from the store.
    Refresh,
    /// A read finished. `Err` keeps the last loaded list on screen.
    Refreshed(Result<Vec<CustomRole>, String>),
    /// Open the window on an empty record.
    OpenCreate,
    /// Open the window on the named record.
    OpenEdit(String),
    /// Close the create/edit window.
    CloseEditor,
    /// A shared-editor action on the name field.
    Name(EditorAction),
    /// A shared-editor action on the prompt editor.
    Prompt(EditorAction),
    /// Pick what the role does when it is launched without a branch identifier.
    SetBehavior(CustomRoleBehavior),
    /// Mark or unmark one tool of the catalogue.
    ToggleTool(String),
    /// Mark or unmark one KIND of a catalog tool that has a kind vocabulary.
    ToggleToolKind { tool: String, kind: String },
    /// Submit the open window: create the record, or save the edited one.
    SubmitEditor,
    /// Result of a submit. An error keeps the window open with it.
    SubmitResult(Result<(), String>),
    /// Open the delete confirmation for the named record.
    DeleteRequest(String),
    /// Delete the record the confirmation is open for.
    ConfirmDelete,
    /// Close the delete confirmation without deleting.
    DismissDelete,
    /// Result of a delete. An error is shown by the section's banner.
    DeleteResult(Result<(), String>),
}

pub(crate) struct CustomRolesState {
    pub(crate) roles: Vec<CustomRole>,
    pub(crate) loaded: bool,
    pub(crate) error: Option<String>,
    /// The open create/edit window; `None` when closed.
    pub(crate) editor: Option<RoleEditor>,
    /// The role name a delete confirmation is open for; `None` when closed.
    pub(crate) delete_target: Option<String>,
}

pub(crate) struct RoleEditor {
    /// `Some(name)` when editing an existing record (the name then cannot be
    /// changed), `None` when creating.
    pub(crate) original: Option<String>,
    pub(crate) name: EditorBuffer,
    pub(crate) prompt: EditorBuffer,
    pub(crate) behavior: CustomRoleBehavior,
    /// The chosen tool names (the picker's marks).
    pub(crate) tools: Vec<String>,
    /// The tool catalogue the picker offers, captured when the window opened —
    /// it constructs the tool objects and reads CONFIG, so it must not be
    /// rebuilt on every render pass.
    pub(crate) catalog: Vec<&'static str>,
    pub(crate) in_flight: bool,
    pub(crate) error: Option<String>,
}

impl RoleEditor {
    /// A window opened on the given record (or on an empty one for a create).
    fn new(
        original: Option<String>,
        name: &str,
        prompt: &str,
        behavior: CustomRoleBehavior,
        tools: Vec<String>,
    ) -> Self {
        // A role name is one line: newlines never enter the field.
        let name_buffer = EditorBuffer::with_text(name, None);
        name_buffer.set_single_line(true);
        let catalog = crate::Role::custom_role_tool_catalog();
        // The stored names are kept as they are: a name this installation cannot
        // serve right now stays marked — the launch omits it and the record keeps
        // it, so a mere open-and-save never drops it. The picker renders it too,
        // marked and labelled unavailable, so it can still be removed deliberately.
        Self {
            original,
            name: name_buffer,
            prompt: EditorBuffer::with_text(prompt, None),
            behavior,
            tools,
            catalog,
            in_flight: false,
            error: None,
        }
    }

    /// The kinds of `tool` the record currently holds, in interface order. A
    /// bare entry holds the tool's narrowest kind, so the picker marks what the
    /// role can actually call. The read is the same lenient one the launch path
    /// uses ([`crate::tools::grants::parse_lossy`]): one unparsable entry drops
    /// itself alone instead of unmarking every kind.
    pub(crate) fn held_kinds(&self, tool: &str) -> Vec<String> {
        crate::tools::grants::parse_lossy(&self.tools)
            .into_iter()
            .find(|g| g.tool == tool)
            .map(|grant| crate::tools::grants::held_kinds(&grant))
            .unwrap_or_default()
    }

    /// Mark or unmark one kind of a tool with a kind vocabulary — a thin call to
    /// [`crate::tools::grants::toggle`], which owns the grammar, the interface
    /// order and the drop of an entry with its last kind.
    fn toggle_kind(&mut self, tool: &str, kind: &str) {
        self.tools = crate::tools::grants::toggle(&self.tools, tool, kind);
    }
}

impl CustomRolesState {
    pub(crate) const fn new() -> Self {
        Self {
            roles: Vec::new(),
            loaded: false,
            error: None,
            editor: None,
            delete_target: None,
        }
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn refresh(&self) -> Task<CustomRolesMessage> {
        Task::perform(custom_role::list(), |res| {
            CustomRolesMessage::Refreshed(res.map_err(|e| e.to_string()))
        })
    }

    /// Dismiss the open window and the delete confirmation (Escape).
    pub(crate) fn close_dialogs(&mut self) {
        self.editor = None;
        self.delete_target = None;
    }

    #[expect(clippy::too_many_lines)]
    pub(crate) fn update(&mut self, msg: CustomRolesMessage) -> Task<CustomRolesMessage> {
        match msg {
            CustomRolesMessage::Refresh => self.refresh(),
            CustomRolesMessage::Refreshed(Ok(roles)) => {
                self.roles = roles;
                self.loaded = true;
                // A successful read replaces the previous failure.
                self.error = None;
                Task::none()
            }
            CustomRolesMessage::Refreshed(Err(e)) => {
                self.error = Some(e);
                Task::none()
            }
            CustomRolesMessage::OpenCreate => {
                self.editor = Some(RoleEditor::new(
                    None,
                    "",
                    "",
                    CustomRoleBehavior::Continuous,
                    Vec::new(),
                ));
                Task::none()
            }
            CustomRolesMessage::OpenEdit(name) => {
                // The list is the source the window is filled from: a name it
                // no longer holds opens nothing.
                if let Some(role) = self.roles.iter().find(|r| r.name == name) {
                    self.editor = Some(RoleEditor::new(
                        Some(role.name.clone()),
                        &role.name,
                        &role.prompt,
                        role.behavior,
                        role.tools.clone(),
                    ));
                }
                Task::none()
            }
            CustomRolesMessage::CloseEditor => {
                self.editor = None;
                Task::none()
            }
            CustomRolesMessage::Name(action) => {
                if let Some(editor) = &self.editor {
                    // The window's buffers are their own state (it is
                    // discarded on close), so an action applies directly.
                    editor.name.perform_action(action);
                }
                Task::none()
            }
            CustomRolesMessage::Prompt(action) => {
                if let Some(editor) = &self.editor {
                    editor.prompt.perform_action(action);
                }
                Task::none()
            }
            CustomRolesMessage::SetBehavior(behavior) => {
                if let Some(editor) = &mut self.editor {
                    editor.behavior = behavior;
                }
                Task::none()
            }
            CustomRolesMessage::ToggleTool(name) => {
                if let Some(editor) = &mut self.editor {
                    // An entry is identified by its tool name: a stored entry
                    // this installation cannot serve may still carry kinds, and
                    // unticking it must remove it rather than add a second entry.
                    match editor
                        .tools
                        .iter()
                        .position(|entry| crate::tools::grants::tool_name(entry) == name.as_str())
                    {
                        Some(at) => {
                            editor.tools.remove(at);
                        }
                        None => editor.tools.push(name),
                    }
                }
                Task::none()
            }
            CustomRolesMessage::ToggleToolKind { tool, kind } => {
                if let Some(editor) = &mut self.editor {
                    editor.toggle_kind(&tool, &kind);
                }
                Task::none()
            }
            CustomRolesMessage::SubmitEditor => {
                let Some(editor) = &mut self.editor else {
                    return Task::none();
                };
                // A double submit never goes out.
                if editor.in_flight {
                    return Task::none();
                }
                // Nothing is validated here: the record's own validators are
                // the authority, and their errors land in the window.
                let name = editor.name.text().trim().to_string();
                let prompt = editor.prompt.text();
                let tools = editor.tools.clone();
                let behavior = editor.behavior;
                // An edit addresses the record by the name it was opened on —
                // the window edits a role, it never renames one.
                let original = editor.original.clone();
                editor.in_flight = true;
                Task::perform(
                    async move {
                        let saved = match &original {
                            Some(target) => {
                                custom_role::update(
                                    target,
                                    Some(&prompt),
                                    Some(&tools),
                                    Some(behavior),
                                )
                                .await
                            }
                            None => custom_role::create(&name, &prompt, &tools, behavior).await,
                        };
                        saved.map_err(|e| e.to_string())
                    },
                    CustomRolesMessage::SubmitResult,
                )
            }
            CustomRolesMessage::SubmitResult(Ok(())) => {
                self.editor = None;
                self.error = None;
                // The list the window was opened from is re-read, so the saved
                // record shows what the store now holds.
                Task::done(CustomRolesMessage::Refresh)
            }
            CustomRolesMessage::SubmitResult(Err(e)) => {
                // The window stays open on what it holds, with the reason.
                if let Some(editor) = &mut self.editor {
                    editor.in_flight = false;
                    editor.error = Some(e);
                }
                Task::none()
            }
            CustomRolesMessage::DeleteRequest(name) => {
                self.delete_target = Some(name);
                Task::none()
            }
            CustomRolesMessage::ConfirmDelete => {
                let Some(name) = self.delete_target.take() else {
                    return Task::none();
                };
                Task::perform(
                    async move { custom_role::delete(&name).await.map_err(|e| e.to_string()) },
                    CustomRolesMessage::DeleteResult,
                )
            }
            CustomRolesMessage::DismissDelete => {
                self.delete_target = None;
                Task::none()
            }
            CustomRolesMessage::DeleteResult(Ok(())) => {
                self.error = None;
                // The card leaves through the re-read.
                Task::done(CustomRolesMessage::Refresh)
            }
            CustomRolesMessage::DeleteResult(Err(e)) => {
                // The record is still there — the banner says why it stayed.
                self.error = Some(e);
                Task::none()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn role(name: &str) -> CustomRole {
        CustomRole {
            name: name.to_string(),
            prompt: format!("prompt of {name}"),
            tools: vec!["web_search".to_string()],
            behavior: CustomRoleBehavior::OneOff,
        }
    }

    fn loaded_state() -> CustomRolesState {
        let mut state = CustomRolesState::new();
        let _ = state.update(CustomRolesMessage::Refreshed(Ok(vec![role("analyst")])));
        state
    }

    /// A state loaded with one role whose tool marks are `tools`.
    fn state_with_tools(name: &str, tools: &[&str]) -> CustomRolesState {
        let mut state = CustomRolesState::new();
        let mut role = role(name);
        role.tools = tools.iter().map(|s| (*s).to_string()).collect();
        let _ = state.update(CustomRolesMessage::Refreshed(Ok(vec![role])));
        state
    }

    /// The kind picker reads and writes the stored grant grammar: a bare
    /// `analyze` shows the kind it implies, a kind toggles on and off, the entry
    /// drops when its last kind goes, and a plain `ToggleTool` on the bare entry
    /// still behaves as it always has.
    #[test]
    fn kind_marks_track_the_stored_entries() {
        let mut state = state_with_tools("analyst", &["analyze", "web_search"]);
        let _ = state.update(CustomRolesMessage::OpenEdit("analyst".to_string()));

        let editor = state.editor.as_ref().expect("the window is open");
        assert_eq!(editor.held_kinds("analyze"), vec!["quick".to_string()]);
        assert!(
            editor.held_kinds("web_search").is_empty(),
            "a kindless tool holds no kinds"
        );
        assert!(
            editor.held_kinds("read").is_empty(),
            "an unheld tool holds no kinds"
        );

        // Marking the second kind spells the entry out.
        let _ = state.update(CustomRolesMessage::ToggleToolKind {
            tool: "analyze".to_string(),
            kind: "deep".to_string(),
        });
        assert_eq!(
            state.editor.as_ref().unwrap().tools,
            vec!["analyze:quick,deep".to_string(), "web_search".to_string()]
        );

        // Unmarking the implied kind leaves the one still held.
        let _ = state.update(CustomRolesMessage::ToggleToolKind {
            tool: "analyze".to_string(),
            kind: "quick".to_string(),
        });
        assert_eq!(
            state.editor.as_ref().unwrap().tools,
            vec!["analyze:deep".to_string(), "web_search".to_string()]
        );
        assert_eq!(
            state.editor.as_ref().unwrap().held_kinds("analyze"),
            vec!["deep".to_string()]
        );

        // The last kind leaving drops the entry from the record.
        let _ = state.update(CustomRolesMessage::ToggleToolKind {
            tool: "analyze".to_string(),
            kind: "deep".to_string(),
        });
        assert_eq!(
            state.editor.as_ref().unwrap().tools,
            vec!["web_search".to_string()]
        );

        // `ToggleTool` on the (now absent) bare entry behaves as today: it adds
        // the name plainly, and the implied kind shows again.
        let _ = state.update(CustomRolesMessage::ToggleTool("analyze".to_string()));
        assert_eq!(
            state.editor.as_ref().unwrap().tools,
            vec!["web_search".to_string(), "analyze".to_string()]
        );
        assert_eq!(
            state.editor.as_ref().unwrap().held_kinds("analyze"),
            vec!["quick".to_string()]
        );
        let _ = state.update(CustomRolesMessage::ToggleTool("analyze".to_string()));
        assert_eq!(
            state.editor.as_ref().unwrap().tools,
            vec!["web_search".to_string()]
        );
    }

    /// The window is filled from the record the list holds — including the
    /// behaviour and the marks — and a mark toggles in and out.
    #[test]
    fn editing_fills_the_window_and_marks_toggle() {
        let mut state = loaded_state();
        let _ = state.update(CustomRolesMessage::OpenEdit("analyst".to_string()));

        let editor = state.editor.as_ref().expect("the window is open");
        assert_eq!(editor.original.as_deref(), Some("analyst"));
        assert_eq!(editor.name.text(), "analyst");
        assert_eq!(editor.prompt.text(), "prompt of analyst");
        assert_eq!(editor.behavior, CustomRoleBehavior::OneOff);
        assert_eq!(editor.tools, vec!["web_search".to_string()]);

        let _ = state.update(CustomRolesMessage::ToggleTool("web_search".to_string()));
        let _ = state.update(CustomRolesMessage::ToggleTool("read".to_string()));
        assert_eq!(
            state.editor.as_ref().expect("the window is open").tools,
            vec!["read".to_string()]
        );

        let _ = state.update(CustomRolesMessage::CloseEditor);
        assert!(state.editor.is_none());
    }

    /// A submit in flight is never doubled, and an error keeps the window open
    /// with its own line.
    #[test]
    fn a_submit_in_flight_is_not_doubled_and_an_error_keeps_the_window() {
        let mut state = loaded_state();
        let _ = state.update(CustomRolesMessage::OpenCreate);
        state.editor.as_mut().expect("the window is open").in_flight = true;
        let _ = state.update(CustomRolesMessage::SubmitEditor);
        assert!(state.editor.as_ref().expect("the window is open").in_flight);

        let _ = state.update(CustomRolesMessage::SubmitResult(Err("nope".to_string())));
        let editor = state.editor.as_ref().expect("the window stays open");
        assert!(!editor.in_flight);
        assert_eq!(editor.error.as_deref(), Some("nope"));
    }
}

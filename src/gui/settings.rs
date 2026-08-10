//! Settings page — dynamic configuration editor.
//!
//! Reads the current config snapshot from [`crate::config::CONFIG`],
//! presents editable fields organised in sections, and saves changes
//! via [`crate::config::save_and_reload`].
//!
//! Also manages workspaces and users (formerly separate pages), with
//! modal dialogs for add operations.

#![expect(clippy::from_iter_instead_of_collect)]

use crate::Role;
use crate::Workspace;
use crate::config::{CONFIG, ConfigData, ModelRouting, RoleConfig};
use crate::workspace::MAX_WORKSPACE_NOTES_CHARS;
use strum::{EnumCount, IntoEnumIterator};

use iced::widget::{
    Checkbox, Column, Row, Space, button, column, container, mouse_area, pick_list, row,
    scrollable, slider, stack, text, text_editor, text_input, toggler, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::collections::{BTreeSet, HashSet};
use std::time::Duration;

use super::theme;
use super::users;
use super::widget_helpers;
use super::widgets;
use super::workspaces;

// ── Shared helpers ────────────────────────────────────────────────

/// Parse a newline-separated model list into a vector of non-empty model names.
///
/// Delegates to [`crate::config::parse_newline_list`] — the shared implementation
/// used by both the config typed accessors and the Settings GUI.
fn parse_models(raw: Option<&str>) -> Vec<String> {
    raw.map_or_else(Vec::new, crate::config::parse_newline_list)
}

/// Add a model from an input buffer to a model list, preventing duplicates.
/// Clears the input buffer after the operation.
fn add_model_to_list(input: &mut String, list: &mut Option<String>) {
    let model = input.trim().to_string();
    if !model.is_empty() {
        let mut models = parse_models(list.as_deref());
        if !models.contains(&model) {
            models.push(model);
            *list = Some(models.join("\n"));
        }
        input.clear();
    }
}

/// Remove a model from a list. If the removed model was the active model,
/// resets the active model to the first remaining entry (or clears it).
fn remove_model_from_list(model: &str, list: &mut Option<String>, active: &mut Option<String>) {
    let mut models = parse_models(list.as_deref());
    models.retain(|m| m != model);
    *list = if models.is_empty() {
        None
    } else {
        Some(models.join("\n"))
    };
    if active.as_deref() == Some(model) {
        *active = models.first().cloned();
    }
}

/// Render a model picker with a list of model entries, active indicator,
/// remove buttons per entry, and an add-model row (text input + "Add" button).
///
/// The active model is always merged into the rendered list (it may have been
/// set independently of the list, e.g. via Telegram), so the active indicator
/// is unambiguous even when the list omits it; merged entries render without
/// a remove button. Accepts a `target` to build the correct parameterized
/// `SettingsMessage::ModelPicker` values internally.
#[allow(clippy::too_many_lines)]
fn model_picker_list<'a>(
    target: ModelPickerTarget,
    models_field: Option<&'a str>,
    active_field: Option<&'a str>,
    add_input: &'a str,
    add_placeholder: &'static str,
) -> Element<'a, SettingsMessage> {
    let on_add_input = move |v| SettingsMessage::ModelPicker {
        target,
        action: ModelPickerAction::AddInput(v),
    };
    let on_add = SettingsMessage::ModelPicker {
        target,
        action: ModelPickerAction::AddModel,
    };
    let on_remove = move |m| SettingsMessage::ModelPicker {
        target,
        action: ModelPickerAction::RemoveModel(m),
    };
    let on_set_active = move |m| SettingsMessage::ModelPicker {
        target,
        action: ModelPickerAction::SetActive(m),
    };
    let mut models = parse_models(models_field);
    let original_models = models.clone();
    let active = active_field;

    // Always merge the active model into the rendered list so the active
    // indicator is unambiguous even when the list omits it (the active model
    // may have been set independently of the list, e.g. via Telegram).
    if !models.iter().any(|m| Some(m.as_str()) == active)
        && let Some(active_model) = active
    {
        models.push(active_model.to_string());
    }

    let items: Vec<Element<'a, SettingsMessage>> = if models.is_empty() {
        vec![
            text("No models configured yet.")
                .size(12)
                .color(theme::TEXT_SECONDARY)
                .into(),
        ]
    } else {
        models
            .iter()
            .map(|model| {
                let is_active = Some(model.as_str()) == active;
                // A merged display-only entry (active model absent from the
                // list) must not offer removal — it is not in the list to
                // remove, and removing it would silently repoint the active.
                let is_merged = is_active && !original_models.iter().any(|m| m == model);
                let indicator = if is_active {
                    lucide::circle_check::<iced::Theme, iced::Renderer>()
                        .size(12)
                        .color(theme::BG_BASE)
                } else {
                    lucide::circle::<iced::Theme, iced::Renderer>()
                        .size(12)
                        .color(theme::TEXT_SECONDARY)
                };
                let mut model_btn = button(
                    row![
                        indicator,
                        Space::new().width(4),
                        text(model.clone()).size(12),
                    ]
                    .align_y(Alignment::Center),
                )
                .padding(4);
                if is_active {
                    model_btn = model_btn.style(theme::button_primary);
                } else {
                    model_btn = model_btn.style(theme::button_secondary);
                }
                model_btn = model_btn.on_press(on_set_active(model.clone()));

                let mut entry = row![model_btn];
                if !is_merged {
                    let remove_btn = button(text("×").size(12))
                        .padding(2)
                        .style(theme::button_text_danger)
                        .on_press(on_remove(model.clone()));
                    entry = entry.push(Space::new().width(4)).push(remove_btn);
                }
                entry.align_y(Alignment::Center).into()
            })
            .collect()
    };

    let add_row = row![
        text_input(add_placeholder, add_input)
            .on_input(on_add_input)
            .style(super::widgets::text_input_style)
            .width(Length::Fixed(450.0)),
        Space::new().width(4),
        button(text("Add").size(11))
            .padding(4)
            .style(theme::button_primary)
            .on_press(on_add),
    ]
    .align_y(Alignment::Center);

    column![
        Column::from_iter(items).spacing(2),
        Space::new().height(4),
        add_row,
    ]
    .into()
}

// ── Messages ─────────────────────────────────────────────────────

/// Which model picker is being operated on.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, EnumCount)]
pub enum ModelPickerTarget {
    ImageGen,
    Video,
}

impl ModelPickerTarget {
    fn idx(self) -> usize {
        match self {
            ModelPickerTarget::ImageGen => 0,
            ModelPickerTarget::Video => 1,
        }
    }
}

/// Action performed on a model picker.
#[derive(Debug, Clone)]
pub enum ModelPickerAction {
    AddInput(String),
    AddModel,
    RemoveModel(String),
    SetActive(String),
}

/// Map a `ModelPickerTarget` to the corresponding `(models_list, active_model)` fields
/// in `ConfigData`.
fn picker_config_fields<'a>(
    t: &'a ModelPickerTarget,
    config: &'a mut ConfigData,
) -> (&'a mut Option<String>, &'a mut Option<String>) {
    match t {
        ModelPickerTarget::ImageGen => (&mut config.image_gen_models, &mut config.image_gen_model),
        ModelPickerTarget::Video => (&mut config.video_models, &mut config.video_model),
    }
}

/// Which password field the visibility toggle applies to.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PasswordTarget {
    ProviderKey,
    FirecrawlKey,
    ExaKey,
    TelegramToken,
}

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    /// Generic editable config field identified by its snake_case key
    /// (matches the keys in [`crate::config::ConfigData::set_string_field`]).
    ConfigField {
        key: &'static str,
        value: String,
    },
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
    /// Toggle password visibility for a specific field.
    TogglePasswordVisibility(PasswordTarget),
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
    /// Toggle a role checkbox in the add-user modal (index into [`Role::iter`]).
    AddUserRoleToggle(usize),
    /// Submit the add-user modal.
    SubmitAddUser,
    /// Result of user add.
    AddUserResult(Result<(), String>),
    /// Escape key pressed (dismisses modal if open).
    Escape,
    // ── Model picker messages ─────────────────────────────
    /// Operations on a model picker (add/remove/set-active model).
    ModelPicker {
        target: ModelPickerTarget,
        action: ModelPickerAction,
    },
    /// Result of an async model-picker add: validated against the image-models
    /// catalog before the model is appended to the list.
    ModelPickerAddResult {
        target: ModelPickerTarget,
        model: String,
        ok: Result<(), String>,
    },
    // ── Voice assistant messages ──────────────────────────
    /// Toggle voice assistant on/off (immediately activates/deactivates the pipeline).
    VoiceToggle(bool),
    /// Result of async DB persistence after a voice toggle.
    /// The `u64` is a generation counter used to detect stale results
    /// from rapid toggling — if it doesn't match `SettingsState::voice_toggle_gen`,
    /// the result is ignored as stale.
    VoiceToggleResult(u64, Result<(), String>),
    /// Start enrollment session for wake word.
    StartVoiceEnrollment,
    /// Cancel enrollment session.
    CancelVoiceEnrollment,
    /// Retry loading voice models after a [`VoiceStatus::ModelError`].
    RetryVoiceModels,
    /// User typed in the wake word phrase text input.
    WakeWordPhraseInput(String),
    // ── TTS messages ─────────────────────────────────────
    /// Toggle TTS on/off (persisted to config DB).
    TtsToggle(bool),
    /// Result of async DB persistence after a TTS toggle.
    /// The `u64` is a generation counter used to detect stale results
    /// from rapid toggling — if it doesn't match `SettingsState::tts_toggle_gen`,
    /// the result is ignored as stale.
    TtsToggleResult(u64, Result<(), String>),
    /// Retry TTS model download after a permanent failure.
    TtsRetryModels,
    /// Test TTS by speaking a test phrase aloud.
    TtsTest,
    /// Request a toast notification from the dashboard.
    Toast(super::ToastMessage),
}

// ── State ────────────────────────────────────────────────────────

const REASONING_EFFORT_OPTIONS: &[&str] = &["xhigh", "high", "medium", "low", "minimal"];

pub struct SettingsState {
    /// Current editable snapshot, loaded from CONFIG each refresh.
    config: ConfigData,
    /// Whether a save is in progress.
    saving: bool,
    /// Last error message from save.
    error: Option<String>,
    /// Which password fields are currently visible.
    password_visible: HashSet<PasswordTarget>,

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
    /// Role checkboxes in the add-user modal, indexed by [`Role::iter`].
    add_user_roles: Vec<bool>,
    /// Whether the add-user operation is in flight.
    add_user_adding: bool,

    // ── Model picker state ────────────────────────────────
    /// Text input buffers for model pickers, indexed by [`ModelPickerTarget::idx`].
    model_picker_inputs: [String; ModelPickerTarget::COUNT],

    // ── Voice assistant state ─────────────────────────────
    /// Generation counter for voice toggle operations.
    /// Incremented before each `VoiceToggle`; the expected value is
    /// passed through to `VoiceToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    voice_toggle_gen: u64,
    /// Transient text input for the wake word phrase.
    /// Not persisted — passed to [`VoiceCommand::StartEnrollment`] on click.
    wake_word_phrase_input: String,

    // ── TTS state ─────────────────────────────────────────
    /// Generation counter for TTS toggle operations.
    /// Incremented before each `TtsToggle`; the expected value is
    /// passed through to `TtsToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    tts_toggle_gen: u64,
}

/// Sync the voice assistant pipeline state with `CONFIG.voice_enabled()`.
/// Called both from the immediate `VoiceToggle` handler and from `SaveResult`
/// (after a full Save where the config may have changed).
fn sync_voice_state(enabled: bool) {
    if enabled {
        crate::audio::voice::set_enabled(true);
        crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::StartListening);
    } else {
        crate::audio::voice::set_enabled(false);
        crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::StopListening);
    }
}

impl SettingsState {
    pub fn new() -> Self {
        Self {
            config: CONFIG.snapshot(),
            saving: false,
            error: None,
            password_visible: HashSet::new(),
            workspaces_state: workspaces::WorkspacesState::new(),
            users_state: users::UsersState::new(),
            show_add_workspace_modal: false,
            add_workspace_name: String::new(),
            add_workspace_path: String::new(),
            add_workspace_adding: false,
            show_add_user_modal: false,
            add_user_sender: String::new(),
            add_user_permissions: String::new(),
            add_user_roles: Vec::new(),
            add_user_adding: false,
            model_picker_inputs: [const { String::new() }; ModelPickerTarget::COUNT],
            voice_toggle_gen: 0,
            wake_word_phrase_input: String::new(),
            tts_toggle_gen: 0,
        }
    }

    /// Reload the editable snapshot from the current CONFIG.
    pub fn refresh(&mut self) {
        self.config = CONFIG.snapshot();
        self.error = None;
    }

    /// Close the add-workspace modal and reset all form fields.
    fn close_add_workspace_modal(&mut self) {
        self.show_add_workspace_modal = false;
        self.add_workspace_name.clear();
        self.add_workspace_path.clear();
        self.add_workspace_adding = false;
    }

    /// Close the add-user modal and reset all form fields.
    fn close_add_user_modal(&mut self) {
        self.show_add_user_modal = false;
        self.add_user_sender.clear();
        self.add_user_permissions.clear();
        self.add_user_roles.clear();
        self.add_user_adding = false;
    }

    /// Look up the role config for a given role key.
    fn role_config_for(&self, key: &str) -> Option<&RoleConfig> {
        self.config
            .per_role_configs
            .iter()
            .find(|rc| rc.role == key)
    }

    /// Mirror a toggle in both config snapshots: `"true"`/`""` (the empty
    /// string keeps the [non_empty] accessor collapsing to None = disabled),
    /// plus the global CONFIG so refresh() can't revert the change.
    fn set_toggle(&mut self, key: &str, enabled: bool) {
        let val = if enabled { "true" } else { "" };
        let _ = self.config.set_string_field(key, val);
        let _ = crate::config::CONFIG.set_string_field(key, val);
    }

    /// Shared voice/TTS toggle arm: mirror the new state, fire the per-toggle
    /// side effect, bump the generation counter, then persist to the DB.
    fn run_toggle(
        &mut self,
        key: &'static str,
        enabled: bool,
        bump_gen: fn(&mut SettingsState) -> u64,
        make_result: fn(u64, Result<(), String>) -> SettingsMessage,
        on_toggle: fn(bool),
    ) -> Task<SettingsMessage> {
        self.set_toggle(key, enabled);
        on_toggle(enabled);
        // Bump generation so stale results from a previous toggle are ignored.
        let toggle_gen = bump_gen(self);

        // Persist async; delete the key on disable so it's absent (None on reload).
        Task::perform(
            async move {
                let store = crate::config_db::store();
                if enabled {
                    store.set_kv(key, "true").await.map_err(|e| e.to_string())
                } else {
                    store.delete_kv(key).await.map_err(|e| e.to_string())
                }
            },
            move |result| make_result(toggle_gen, result),
        )
    }

    /// Shared voice/TTS result arm: ignore stale generations; on DB error
    /// revert both config snapshots and the pipeline via `on_revert`.
    fn handle_toggle_result(
        &mut self,
        key: &str,
        g: u64,
        result: Result<(), String>,
        get_gen: fn(&SettingsState) -> u64,
        get_enabled: fn(&ConfigData) -> &Option<String>,
        on_revert: fn(bool),
    ) -> Task<SettingsMessage> {
        // Ignore results from a superseded toggle (user toggled again mid-write).
        if g != get_gen(self) {
            return Task::none();
        }
        match result {
            Ok(()) => Task::none(),
            Err(e) => {
                self.error = Some(e);
                // Revert both snapshots so the toggle isn't shown enabled but lost on restart.
                let current_enabled = get_enabled(&self.config).as_deref() == Some("true");
                let target_state = !current_enabled;
                self.set_toggle(key, target_state);
                on_revert(target_state);
                Task::none()
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: SettingsMessage) -> Task<SettingsMessage> {
        match msg {
            // ── Config field edits ─────────────────────────────
            SettingsMessage::ConfigField { key, value } => {
                let _ = self.config.set_string_field(key, &value);
                Task::none()
            }
            SettingsMessage::RoleModel { role, model } => {
                let model_opt = Some(model).filter(|s| !s.is_empty());
                RoleConfig::upsert(&mut self.config.per_role_configs, role, |c| {
                    c.model = model_opt;
                });
                Task::none()
            }
            SettingsMessage::RoleReasoning { role, effort } => {
                let effort_opt = Some(effort).filter(|s| !s.is_empty());
                RoleConfig::upsert(&mut self.config.per_role_configs, role, |c| {
                    c.reasoning_effort = effort_opt;
                });
                Task::none()
            }
            SettingsMessage::ModelRoutingOrder { model, order } => {
                let order_opt = Some(order).filter(|s| !s.is_empty());
                ModelRouting::upsert(&mut self.config.model_routings, model, |mr| {
                    mr.provider_order = order_opt;
                });
                Task::none()
            }
            SettingsMessage::ModelRoutingAllowFallbacks { model, allow } => {
                ModelRouting::upsert(&mut self.config.model_routings, model, |mr| {
                    mr.allow_fallbacks = Some(allow);
                });
                Task::none()
            }
            SettingsMessage::TogglePasswordVisibility(target) => {
                if self.password_visible.contains(&target) {
                    self.password_visible.remove(&target);
                } else {
                    self.password_visible.insert(target);
                }
                Task::none()
            }
            SettingsMessage::Save => {
                self.saving = true;
                self.error = None;
                let config = self.config.clone();
                // NOTE: wake_word_templates is intentionally NOT preserved
                // here — save_and_reload skips it, leaving the voice pipeline
                // (persist_model_state) as the sole owner of that key.
                // This avoids the dual-writer race entirely.
                Task::perform(
                    async move {
                        crate::config::save_and_reload(config)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    SettingsMessage::SaveResult,
                )
            }
            SettingsMessage::SaveResult(Ok(())) => {
                self.saving = false;
                self.refresh();

                // Sync voice assistant state with config
                sync_voice_state(crate::config::CONFIG.voice_enabled().as_deref() == Some("true"));

                Task::none()
            }
            SettingsMessage::SaveResult(Err(e)) => {
                self.saving = false;
                self.error = Some(e);
                Task::none()
            }

            // ── Voice assistant ─────────────────────────────────
            SettingsMessage::VoiceToggle(enabled) => self.run_toggle(
                "voice_enabled",
                enabled,
                |s| {
                    s.voice_toggle_gen += 1;
                    s.voice_toggle_gen
                },
                SettingsMessage::VoiceToggleResult,
                sync_voice_state,
            ),
            SettingsMessage::VoiceToggleResult(g, result) => self.handle_toggle_result(
                "voice_enabled",
                g,
                result,
                |s| s.voice_toggle_gen,
                |c| &c.voice_enabled,
                sync_voice_state,
            ),
            SettingsMessage::TtsToggle(enabled) => self.run_toggle(
                "tts_enabled",
                enabled,
                |s| {
                    s.tts_toggle_gen += 1;
                    s.tts_toggle_gen
                },
                SettingsMessage::TtsToggleResult,
                |enabled| {
                    // Toggle ON with uncached models triggers download (handles
                    // ModelState::Failed retries too, matching voice's auto-retry).
                    if enabled && !crate::audio::tts::try_load_cached() {
                        crate::audio::tts::spawn_or_retry_download();
                    }
                },
            ),
            SettingsMessage::TtsToggleResult(g, result) => self.handle_toggle_result(
                "tts_enabled",
                g,
                result,
                |s| s.tts_toggle_gen,
                |c| &c.tts_enabled,
                |_| {},
            ),
            SettingsMessage::TtsRetryModels => {
                let _ = crate::audio::tts::retry_download();
                Task::none()
            }
            SettingsMessage::TtsTest => {
                if !crate::audio::tts::audio_output_ready() {
                    return Task::done(SettingsMessage::Toast(super::ToastMessage::Warning(
                        "Audio output device not available".to_string(),
                    )));
                }
                crate::audio::tts::speak("This is a test of the text to speech system.");
                Task::none()
            }
            SettingsMessage::StartVoiceEnrollment => {
                let phrase = self.wake_word_phrase_input.clone();
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StartEnrollment(phrase),
                );
                Task::none()
            }
            SettingsMessage::WakeWordPhraseInput(v) => {
                self.wake_word_phrase_input = v;
                Task::none()
            }
            SettingsMessage::CancelVoiceEnrollment => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::CancelEnrollment,
                );
                Task::none()
            }
            SettingsMessage::RetryVoiceModels => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::RetryModelLoading,
                );
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
                    self.close_add_workspace_modal();
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
                self.close_add_workspace_modal();
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
                if self.show_add_user_modal {
                    // Fresh checkbox state: all unchecked.
                    self.add_user_roles = Role::iter().map(|_| false).collect();
                } else {
                    self.close_add_user_modal();
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
            SettingsMessage::AddUserRoleToggle(idx) => {
                if let Some(checked) = self.add_user_roles.get_mut(idx) {
                    *checked = !*checked;
                }
                Task::none()
            }
            SettingsMessage::SubmitAddUser => {
                if self.add_user_sender.is_empty() {
                    return Task::none();
                }
                let roles: Vec<Role> = Role::iter()
                    .zip(self.add_user_roles.iter())
                    .filter(|(_, checked)| **checked)
                    .map(|(r, _)| r)
                    .collect();
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
                            .add_user(&sender, permissions.as_deref(), &roles)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    },
                    SettingsMessage::AddUserResult,
                )
            }
            SettingsMessage::AddUserResult(Ok(())) => {
                self.close_add_user_modal();
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

            // ── Model picker messages ─────────────────────────
            SettingsMessage::ModelPicker { target, action } => {
                match (target, action) {
                    (t, ModelPickerAction::AddInput(v)) => {
                        self.model_picker_inputs[t.idx()] = v;
                        Task::none()
                    }
                    (t, ModelPickerAction::AddModel) => match t {
                        // Image-gen additions are validated against the catalog
                        // (fail-open when it is unavailable) before the list is
                        // mutated; the input buffer is kept on rejection so the
                        // user can correct it. Validation uses the endpoint being
                        // drafted — the committed global may differ when the user
                        // changed the endpoint in this unsaved session.
                        ModelPickerTarget::ImageGen => {
                            let model = self.model_picker_inputs[t.idx()].trim().to_string();
                            if model.is_empty() {
                                return Task::none();
                            }
                            let endpoint = self
                                .config
                                .provider_endpoint
                                .as_deref()
                                .unwrap_or(crate::config::DEFAULT_PROVIDER_ENDPOINT)
                                .to_string();
                            Task::perform(
                                async move {
                                    let ok = crate::tools::image_catalog::
                                        validate_image_model_for_endpoint(&model, &endpoint)
                                        .await
                                        .map_err(|e| e.to_string());
                                    (t, model, ok)
                                },
                                |(target, model, ok)| SettingsMessage::ModelPickerAddResult {
                                    target,
                                    model,
                                    ok,
                                },
                            )
                        }
                        ModelPickerTarget::Video => {
                            let (models, _active) = picker_config_fields(&t, &mut self.config);
                            add_model_to_list(&mut self.model_picker_inputs[t.idx()], models);
                            Task::none()
                        }
                    },
                    (t, ModelPickerAction::RemoveModel(model)) => {
                        let (models, active) = picker_config_fields(&t, &mut self.config);
                        remove_model_from_list(&model, models, active);
                        Task::none()
                    }
                    (t, ModelPickerAction::SetActive(model)) => {
                        let (_models, active) = picker_config_fields(&t, &mut self.config);
                        *active = Some(model);
                        Task::none()
                    }
                }
            }

            SettingsMessage::ModelPickerAddResult { target, model, ok } => match ok {
                Ok(()) => {
                    // Append the validated model directly — never route through
                    // the input buffer, which may hold text the user typed while
                    // the catalog validation was in flight.
                    let (models, _active) = picker_config_fields(&target, &mut self.config);
                    let mut list = parse_models(models.as_deref());
                    if !list.contains(&model) {
                        list.push(model.clone());
                        *models = Some(list.join("\n"));
                    }
                    // Clear the input only when it still holds the added model
                    // (modulo whitespace); anything the user typed meanwhile
                    // must survive.
                    if self.model_picker_inputs[target.idx()].trim() == model {
                        self.model_picker_inputs[target.idx()].clear();
                    }
                    Task::done(SettingsMessage::Toast(super::ToastMessage::Saved))
                }
                Err(e) => Task::done(SettingsMessage::Toast(super::ToastMessage::Error(format!(
                    "Model `{model}` rejected: {e}"
                )))),
            },

            SettingsMessage::Toast(_) => {
                // Toast messages are intercepted by Dashboard::as_toast()
                // before dispatch — this arm should never be reached.
                Task::none()
            }

            SettingsMessage::Escape => {
                if self.show_add_workspace_modal {
                    self.close_add_workspace_modal();
                } else if self.show_add_user_modal {
                    self.close_add_user_modal();
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
            self.voice_section(),
            Space::new().height(16),
            self.tts_section(),
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
            .direction(theme::vertical_scrollbar())
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
            .style(theme::base_container_style)
            .into()
    }

    // ── Workspace management section ──────────────────────────

    /// Render the workspaces section for the Settings page. No inner
    /// scrollable — rows expand the outer Settings scrollable naturally.
    #[allow(clippy::too_many_lines)]
    fn workspaces_section(&self) -> Element<'_, SettingsMessage> {
        let ws = &self.workspaces_state;

        let mut rows = Column::new().spacing(4);

        rows = widgets::push_error_banner(rows, ws.load_state.error());

        if ws.load_state.loading() && !ws.load_state.has_loaded() {
            rows = rows.push(widgets::loading_text());
        } else if ws.workspaces.is_empty() {
            rows = rows.push(
                text("No workspaces configured. Add one below.")
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        } else {
            for (row_index, ws_item) in ws.workspaces.iter().enumerate() {
                let (status_color, status_bg) = theme::workspace_status_color(ws_item.status);
                let maintainer_on = ws_item.maintenance_enabled;

                let delete_btn = delete_confirm_button(
                    Some(&ws_item.name) == ws.delete_target.as_ref(),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::ConfirmDelete(
                        ws_item.name.clone(),
                    )),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::CancelDelete),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::DeleteWorkspace(
                        ws_item.name.clone(),
                    )),
                );

                let ws_row = container(
                    column![
                        row![
                            // Name column (FillPortion: 15)
                            container(text(&ws_item.name).size(14).color(theme::TEXT_PRIMARY))
                                .width(Length::FillPortion(15))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center),
                            // Status column (FillPortion: 10)
                            container(
                                container(
                                    text(ws_item.status.as_ref()).size(11).color(status_color)
                                )
                                .padding([2, 8])
                                .style(theme::pill_style(status_bg)),
                            )
                            .width(Length::FillPortion(10))
                            .align_x(Alignment::Start)
                            .align_y(Alignment::Center),
                            // Path column (FillPortion: 35)
                            container(text(&ws_item.path).size(12).color(theme::TEXT_MUTED))
                                .width(Length::FillPortion(35))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center),
                            // Agent icons column (FillPortion: 25)
                            {
                                let mut role_btns = Row::new().spacing(2);
                                for role in
                                    Role::iter().filter(|r| crate::role::role_info(r).has_discovery)
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
                                container(role_btns)
                                    .width(Length::FillPortion(25))
                                    .align_x(Alignment::Start)
                                    .align_y(Alignment::Center)
                            },
                            // Actions column (FillPortion: 15)
                            container(
                                row![
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
                                    .on_press(
                                        SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::ToggleMaintainer(
                                                ws_item.name.clone(),
                                                !maintainer_on,
                                            ),
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
                                    .on_press(
                                        SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::Reanalyze(
                                                ws_item.name.clone()
                                            ),
                                        )
                                    ),
                                    Space::new().width(4),
                                    {
                                        let is_open = ws.notes_open.contains(&ws_item.name);
                                        button(
                                            text(if is_open { "Notes ✓" } else { "Notes" })
                                                .size(11)
                                                .color(theme::TEXT_MUTED),
                                        )
                                        .style(theme::button_text)
                                        .on_press(
                                            SettingsMessage::WorkspaceMsg(
                                                workspaces::WorkspacesMessage::ToggleNotes(
                                                    ws_item.name.clone(),
                                                ),
                                            ),
                                        )
                                    },
                                    Space::new().width(4),
                                    button(text("Diag").size(11).color(theme::TEXT_MUTED),)
                                        .style(theme::button_text)
                                        .on_press(SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::ShowDiagnostics(
                                                ws_item.name.clone(),
                                            ),
                                        )),
                                    Space::new().width(4),
                                    delete_btn,
                                ]
                                .align_y(Alignment::Center)
                            )
                            .width(Length::FillPortion(15))
                            .align_x(Alignment::End)
                            .align_y(Alignment::Center),
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
                .style(theme::surface_card_style);

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
                            button(text("Diag").size(11))
                                .style(theme::button_text)
                                .on_press(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::ShowDiagnostics(
                                        ws_item.name.clone(),
                                    ),
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
                    .style(|_| {
                        theme::container_style(theme::BG_ELEVATED, 4.0, 1.0, theme::BORDER_STRONG)
                    });
                    rows = rows.push(ctx_actions);
                }

                // ── Inline notes editor ──────────────────────────────────
                if ws.notes_open.contains(&ws_item.name) {
                    // Content is guaranteed to exist in the HashMap because
                    // ToggleNotes always inserts before adding to notes_open.
                    let content = ws
                        .notes_editor_content
                        .get(&ws_item.name)
                        .expect("notes editor content must exist when notes_open contains name");
                    let char_count = content.text().chars().count();
                    let over_limit = char_count > MAX_WORKSPACE_NOTES_CHARS;

                    let editor = text_editor(content)
                        .on_action(move |action| {
                            SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::NotesEdited(
                                    ws_item.name.clone(),
                                    action,
                                ),
                            )
                        })
                        .placeholder(
                            format!("Add manual context notes for all agents… (max {MAX_WORKSPACE_NOTES_CHARS} characters)"),
                        )
                        .min_height(100.0)
                        .max_height(300.0);

                    let char_counter = text(if over_limit {
                        format!("{char_count}/{MAX_WORKSPACE_NOTES_CHARS} — please trim")
                    } else {
                        format!("{char_count}/{MAX_WORKSPACE_NOTES_CHARS}")
                    })
                    .size(11)
                    .color(if over_limit {
                        theme::STATUS_ERROR
                    } else {
                        theme::TEXT_MUTED
                    });

                    let save_btn = button(text("Save Notes").size(12)).style(theme::button_primary);
                    // Only enable Save when under the character limit
                    let save_btn = if over_limit {
                        save_btn
                    } else {
                        save_btn.on_press(SettingsMessage::WorkspaceMsg(
                            workspaces::WorkspacesMessage::SaveNotes(ws_item.name.clone()),
                        ))
                    };

                    let cancel_btn = button(text("Cancel").size(12))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::WorkspaceMsg(
                            workspaces::WorkspacesMessage::NotesCancel(ws_item.name.clone()),
                        ));

                    let notes_section = container(
                        column![
                            editor,
                            Space::new().height(4),
                            row![
                                char_counter,
                                Space::new().width(Length::Fill),
                                save_btn,
                                Space::new().width(4),
                                cancel_btn,
                            ]
                            .align_y(Alignment::Center),
                        ]
                        .spacing(4),
                    )
                    .padding([4, 8])
                    .style(theme::pill_style(theme::BG_ELEVATED));

                    rows = rows.push(notes_section);
                }
            }
        }

        // Inline "+" button in the section header
        let plus_btn: Element<'_, SettingsMessage> = button(
            lucide::plus::<iced::Theme, iced::Renderer>()
                .size(16)
                .color(theme::ACCENT),
        )
        .style(theme::button_text)
        .on_press(SettingsMessage::ToggleAddWorkspaceModal)
        .into();

        let mut section_content = column![rows];

        // Context view overlay — read-only markdown (inline in section)
        if let Some((ref _ws_name, ref role, ref md_items_opt)) = ws.context_view {
            section_content = section_content.push(Space::new().height(16));

            let title = format!("Context for {role}");

            let body: Element<'_, SettingsMessage> = match md_items_opt {
                None => container(widgets::loading_text())
                    .width(Length::Fill)
                    .into(),
                Some(items) => {
                    let mut view_col = column![];

                    if let Some(ref err) = ws.context_view_error {
                        view_col = view_col.push(widgets::error_banner(err));
                        view_col = view_col.push(Space::new().height(8));
                    }

                    if items.is_empty() {
                        view_col = view_col
                            .push(text("Not yet discovered").size(13).color(theme::TEXT_MUTED));
                    } else {
                        let md: Element<'_, SettingsMessage> =
                            iced_selection::markdown::view(items, theme::markdown_settings()).map(
                                |url| {
                                    SettingsMessage::WorkspaceMsg(
                                        workspaces::WorkspacesMessage::LinkClicked(url),
                                    )
                                },
                            );
                        view_col = view_col.push(
                            container(scrollable(md).direction(theme::vertical_scrollbar()))
                                .padding(4)
                                .height(Length::Fixed(300.0))
                                .style(|_| {
                                    theme::container_style(theme::BG_BASE, 4.0, 1.0, theme::BORDER)
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
            .style(theme::dialog_container_style);

            section_content = section_content.push(view_container);
        }

        section_with_header_action("Workspaces", plus_btn, section_content)
    }

    /// Render the users section for the Settings page.
    #[allow(clippy::too_many_lines)]
    fn users_section(&self, active_user: Option<&str>) -> Element<'_, SettingsMessage> {
        let us = &self.users_state;

        let mut rows = Column::new().spacing(4);

        rows = widgets::push_error_banner(rows, us.load_state.error());

        if us.load_state.loading() && !us.load_state.has_loaded() {
            rows = rows.push(widgets::loading_text());
        } else if us.users.is_empty() {
            rows = rows.push(
                text("No users configured. Add one below.")
                    .size(12)
                    .color(theme::TEXT_MUTED),
            );
        } else {
            for user in &us.users {
                let is_admin = user.is_admin();
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
                    row![].into()
                } else {
                    delete_confirm_button(
                        Some(&user.name) == us.delete_target.as_ref(),
                        SettingsMessage::UserMsg(users::UsersMessage::ConfirmDelete(
                            user.name.clone(),
                        )),
                        SettingsMessage::UserMsg(users::UsersMessage::CancelDelete),
                        SettingsMessage::UserMsg(users::UsersMessage::DeleteUser(
                            user.name.clone(),
                        )),
                    )
                };

                let user_row = container(
                    column![
                        row![
                            // Name + permissions column (FillPortion: 20)
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
                                container(user_elem)
                                    .width(Length::FillPortion(20))
                                    .align_x(Alignment::Start)
                                    .align_y(Alignment::Center)
                            },
                            // Workspace column (FillPortion: 20)
                            {
                                let ws_value = user.selected_workspace.as_deref().unwrap_or("");
                                let ws_selected = us
                                    .workspace_options
                                    .iter()
                                    .find(|o| o.value == ws_value)
                                    .cloned();
                                container(
                                    pick_list(
                                        us.workspace_options.as_slice(),
                                        ws_selected,
                                        |opt| {
                                            SettingsMessage::UserMsg(
                                                users::UsersMessage::UpdateWorkspace(
                                                    user.name.clone(),
                                                    opt.value,
                                                ),
                                            )
                                        },
                                    )
                                    .style(widgets::pick_list_style)
                                    .padding([4, 8])
                                    .width(Length::Fixed(200.0)),
                                )
                                .width(Length::FillPortion(20))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center)
                            },
                            // Role column (FillPortion: 15) — active role
                            // picker (pool-restricted) + pool editor button
                            {
                                let role_picker: Element<'_, SettingsMessage> =
                                    match us.active_role_options.get(&user.name) {
                                        Some(options) if !options.is_empty() => {
                                            let role_selected = user
                                                .selected_role
                                                .as_ref()
                                                .and_then(|name| {
                                                    options.iter().find(|o| o.value == *name)
                                                })
                                                .cloned()
                                                .or_else(|| {
                                                    // No (or stale) stored selection —
                                                    // mirror resolve_active_role's default:
                                                    // Analyst when in the pool, else the
                                                    // first pool role.
                                                    options
                                                        .iter()
                                                        .find(|o| o.value == "analyst")
                                                        .or_else(|| options.first())
                                                        .cloned()
                                                });
                                            pick_list(options.as_slice(), role_selected, |opt| {
                                                SettingsMessage::UserMsg(
                                                    users::UsersMessage::UpdateRole(
                                                        user.name.clone(),
                                                        opt.value,
                                                    ),
                                                )
                                            })
                                            .style(widgets::pick_list_style)
                                            .padding([4, 8])
                                            .width(Length::Fixed(150.0))
                                            .into()
                                        }
                                        _ => text("none").size(12).color(theme::TEXT_MUTED).into(),
                                    };
                                container(
                                    row![
                                        role_picker,
                                        Space::new().width(4),
                                        button(
                                            lucide::pencil_line::<iced::Theme, iced::Renderer>()
                                                .size(15)
                                                .color(theme::TEXT_MUTED),
                                        )
                                        .style(theme::button_text)
                                        .padding(2)
                                        .on_press(
                                            SettingsMessage::UserMsg(
                                                users::UsersMessage::OpenPoolEdit(
                                                    user.name.clone()
                                                ),
                                            )
                                        ),
                                    ]
                                    .align_y(Alignment::Center),
                                )
                                .width(Length::FillPortion(15))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center)
                            },
                            // Actions column (FillPortion: 50) — switch icon + delete
                            container({
                                let mut actions = Row::new().align_y(Alignment::Center);
                                actions = actions.push(switch_icon);
                                if !is_admin {
                                    actions = actions.push(Space::new().width(8));
                                    actions = actions.push(delete_btn);
                                }
                                actions
                            })
                            .width(Length::FillPortion(50))
                            .align_x(Alignment::End)
                            .align_y(Alignment::Center),
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
                                        .width(Length::Fixed(270.0))
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
                .style(theme::surface_card_style);

                rows = rows.push(user_row);
            }
        }

        // Inline "+" button in the section header
        let plus_btn: Element<'_, SettingsMessage> = button(
            lucide::plus::<iced::Theme, iced::Renderer>()
                .size(16)
                .color(theme::ACCENT),
        )
        .style(theme::button_text)
        .on_press(SettingsMessage::ToggleAddUserModal)
        .into();

        section_with_header_action("Users", plus_btn, column![rows])
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
        } else if let Some(ref pool_user) = self.users_state.pool_edit_target {
            let dialog = self.pool_edit_dialog(pool_user);
            modal_with_backdrop(
                dialog,
                SettingsMessage::UserMsg(users::UsersMessage::ClosePoolEdit),
            )
        } else if let Some(ref diag_ws_name) = self.workspaces_state.diagnostics_modal {
            let dialog = self.diagnostics_dialog(diag_ws_name);
            modal_with_backdrop(
                dialog,
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Escape),
            )
        } else {
            // Keep Stack widget type stable
            iced::widget::stack([widget_helpers::empty_stack_placeholder()]).into()
        }
    }

    /// Build the role-pool editor modal for a user.
    fn pool_edit_dialog(&self, user_name: &str) -> Element<'_, SettingsMessage> {
        let checked = &self.users_state.pool_edit_checked;
        let mut col = Column::new().padding(24);
        col = col.push(
            text(format!("Edit roles — {user_name}"))
                .size(16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
        );
        col = col.push(Space::new().height(16));
        col = col.push(
            text("Unchecking every role stops agent answers until a role is assigned.")
                .size(12)
                .color(theme::TEXT_MUTED),
        );
        col = col.push(Space::new().height(8));

        let role_checks: Vec<Element<'_, SettingsMessage>> = Role::iter()
            .enumerate()
            .map(|(i, role)| {
                let is_checked = checked.get(i).copied().unwrap_or(false);
                let label = crate::role::role_info(&role).display_label;
                Checkbox::new(is_checked)
                    .label(label)
                    .on_toggle(move |_| {
                        SettingsMessage::UserMsg(users::UsersMessage::TogglePoolRole(i))
                    })
                    .into()
            })
            .collect();
        col = col.push(Row::with_children(role_checks).spacing(6).wrap());

        col = col.push(Space::new().height(16));
        col = col.push(
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .style(theme::button_secondary)
                    .on_press(SettingsMessage::UserMsg(users::UsersMessage::ClosePoolEdit)),
                Space::new().width(8),
                button(text("Save").size(13))
                    .style(theme::button_primary)
                    .on_press(SettingsMessage::UserMsg(
                        users::UsersMessage::SubmitPoolEdit(user_name.to_string(),)
                    )),
            ]
            .align_y(Alignment::Center),
        );

        container(col)
            .width(Length::Fixed(620.0))
            .style(theme::dialog_container_style)
            .into()
    }

    /// Build the add-workspace modal dialog content.
    fn add_workspace_dialog(&self) -> Element<'_, SettingsMessage> {
        modal_dialog(
            "Add Workspace",
            &[
                DialogField {
                    label: "Name",
                    placeholder: "workspace name",
                    value: &self.add_workspace_name,
                    on_input: SettingsMessage::AddWorkspaceName,
                },
                DialogField {
                    label: "Path",
                    placeholder: "/path/to/workspace",
                    value: &self.add_workspace_path,
                    on_input: SettingsMessage::AddWorkspacePath,
                },
            ],
            self.add_workspace_adding,
            !self.add_workspace_name.is_empty() && !self.add_workspace_path.is_empty(),
            SettingsMessage::ToggleAddWorkspaceModal,
            SettingsMessage::SubmitAddWorkspace,
        )
    }

    /// Build the add-user modal dialog content.
    fn add_user_dialog(&self) -> Element<'_, SettingsMessage> {
        let any_role_checked = self.add_user_roles.iter().any(|c| *c);
        let mut col = Column::new().padding(24);

        col = col.push(
            text("Add User")
                .size(16)
                .color(theme::TEXT_PRIMARY)
                .font(theme::FONT_BOLD),
        );
        col = col.push(Space::new().height(16));
        col = col.push(field_row(
            "Name",
            text_input("user name", &self.add_user_sender)
                .on_input(SettingsMessage::AddUserSender)
                .style(widgets::text_input_style)
                .width(Length::Fixed(375.0))
                .into(),
            None,
        ));
        col = col.push(Space::new().height(8));
        col = col.push(field_row(
            "Permissions",
            text_input("optional", &self.add_user_permissions)
                .on_input(SettingsMessage::AddUserPermissions)
                .style(widgets::text_input_style)
                .width(Length::Fixed(375.0))
                .into(),
            None,
        ));

        // Role pool checkboxes — at least one must be checked.
        col = col.push(Space::new().height(16));
        col = col.push(
            text("Allowed roles (at least one required)")
                .size(12)
                .color(theme::TEXT_MUTED),
        );
        let role_checks: Vec<Element<'_, SettingsMessage>> = Role::iter()
            .enumerate()
            .map(|(i, role)| {
                let checked = self.add_user_roles.get(i).copied().unwrap_or(false);
                let label = crate::role::role_info(&role).display_label;
                Checkbox::new(checked)
                    .label(label)
                    .on_toggle(move |_| SettingsMessage::AddUserRoleToggle(i))
                    .into()
            })
            .collect();
        col = col.push(Row::with_children(role_checks).spacing(6).wrap());
        col = col.push(Space::new().height(8));

        col = col.push(Space::new().height(16));
        col = col.push(
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
                    if self.add_user_adding || self.add_user_sender.is_empty() || !any_role_checked
                    {
                        None
                    } else {
                        Some(SettingsMessage::SubmitAddUser)
                    }
                ),
            ]
            .align_y(Alignment::Center),
        );

        container(col)
            .width(Length::Fixed(620.0))
            .style(theme::dialog_container_style)
            .into()
    }

    /// Build the diagnostics modal dialog content for the given workspace.
    #[allow(clippy::too_many_lines)]
    fn diagnostics_dialog(&self, diag_ws_name: &str) -> Element<'_, SettingsMessage> {
        let ws_name = diag_ws_name.to_string();
        let ws_state = &self.workspaces_state;

        let is_busy = ws_state.diagnostics_busy;
        let error = ws_state.diagnostics_error.as_deref();

        // Get the edit buffers — if modal is open they should exist.
        let buffers: [String; crate::DiagnosticsCommands::COMMAND_COUNT] = ws_state
            .diagnostics_edit_buffers
            .get(&ws_name)
            .cloned()
            .unwrap_or([const { String::new() }; crate::DiagnosticsCommands::COMMAND_COUNT]);

        // Use static labels from DiagnosticsCommands to avoid duplicating
        // the label-to-field mapping in two places.
        let labels = &crate::DiagnosticsCommands::COMMAND_LABELS;

        let mut rows_col = Column::new().spacing(8);

        // Error banner
        if let Some(err) = error {
            rows_col = rows_col.push(widgets::error_banner(err));
        }

        for (i, label) in labels.iter().enumerate() {
            let value = &buffers[i];
            rows_col = rows_col.push(
                row![
                    text(*label)
                        .size(12)
                        .color(theme::TEXT_MUTED)
                        .width(Length::Fixed(120.0))
                        .align_y(Alignment::Center),
                    text_input("(skipped)", value)
                        .size(12)
                        .font(iced::Font::MONOSPACE)
                        .on_input({
                            let name = ws_name.clone();
                            move |v| {
                                SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::DiagnosticsFieldEdited(
                                        name.clone(),
                                        i,
                                        v,
                                    ),
                                )
                            }
                        })
                        .width(Length::Fill),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            );
        }

        rows_col = rows_col.push(Space::new().height(8));

        // Action buttons row: [Re-discover] [Save] [Cancel]
        rows_col = rows_col.push(
            row![
                button(row![
                    lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                        .size(12)
                        .color(theme::TEXT_MUTED),
                    Space::new().width(4),
                    text("Re-discover").size(12),
                ])
                .style(theme::button_text)
                .on_press(SettingsMessage::WorkspaceMsg(
                    workspaces::WorkspacesMessage::RediscoverDiagnostics(ws_name.clone(),),
                )),
                Space::new().width(Length::Fill),
                button(
                    text(if is_busy { "Working…" } else { "Save" })
                        .size(12)
                        .color(if is_busy {
                            theme::TEXT_MUTED
                        } else {
                            theme::ACCENT
                        }),
                )
                .style(theme::button_text)
                .on_press_maybe(if is_busy {
                    None
                } else {
                    Some(SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::SaveDiagnostics(ws_name.clone()),
                    ))
                }),
                Space::new().width(8),
                button(text("Cancel").size(12).color(theme::TEXT_MUTED))
                    .style(theme::button_text)
                    .on_press(SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::Escape,
                    )),
            ]
            .align_y(Alignment::Center),
        );

        let modal_title = format!("Diagnostics: {ws_name}");
        container(
            column![
                text(modal_title).size(16).color(theme::TEXT_PRIMARY),
                Space::new().height(16),
                rows_col,
            ]
            .spacing(8)
            .width(Length::Fill)
            .padding(24),
        )
        .width(620)
        .style(theme::dialog_container_style)
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
                        "sk-or-v1-...",
                        self.config.provider_key.as_deref().unwrap_or_default(),
                        self.password_visible.contains(&PasswordTarget::ProviderKey),
                        |v| SettingsMessage::ConfigField {
                            key: "provider_key",
                            value: v
                        },
                        SettingsMessage::TogglePasswordVisibility(PasswordTarget::ProviderKey),
                    ),
                    None,
                ),
                config_text_input(
                    "Endpoint",
                    "https://openrouter.ai/api/v1",
                    self.config.provider_endpoint.as_deref().unwrap_or_default(),
                    "provider_endpoint",
                ),
            ],
        )
    }

    fn models_section(&self) -> Element<'_, SettingsMessage> {
        let rows = Role::iter().map(|role| {
            let key: &str = role.into();
            let info = crate::role::role_info(&role);
            let label = info.display_label;
            let default = info.default_model;
            let current = self
                .role_config_for(key)
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
                    .width(Length::Fixed(375.0))
                    .into(),
                Some(default),
            )
        });
        section("Models (per-role)", Column::from_iter(rows))
    }

    fn reasoning_section(&self) -> Element<'_, SettingsMessage> {
        let rows = Role::iter().map(|role| {
            let key: &str = role.into();
            let info = crate::role::role_info(&role);
            let label = info.display_label;
            let default = info.default_reasoning_effort;
            let current = self
                .role_config_for(key)
                .and_then(|rc| rc.reasoning_effort.clone())
                .unwrap_or_else(|| default.to_string());
            let effort_buttons = Row::from_iter(REASONING_EFFORT_OPTIONS.iter().map(move |&opt| {
                let is_active = current == opt;
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
        let local_enabled = self.config.audio_transcription_use_local.as_deref() != Some("false");

        section(
            "Transcription",
            column![
                field_row(
                    "Local Transcription",
                    toggler(local_enabled)
                        .on_toggle(move |b| SettingsMessage::ConfigField {
                            key: "audio_transcription_use_local",
                            value: if b {
                                String::new()
                            } else {
                                "false".to_string()
                            },
                        })
                        .into(),
                    Some("Use local Qwen3-ASR (offline) — audio never leaves the machine"),
                ),
                Space::new().height(8),
                config_text_input(
                    "Media Model",
                    "qwen/qwen3.6-plus",
                    self.config
                        .media_transcription_model
                        .as_deref()
                        .unwrap_or_default(),
                    "media_transcription_model",
                ),
                config_text_input(
                    "Media Provider",
                    "",
                    self.config
                        .media_transcription_provider
                        .as_deref()
                        .unwrap_or_default(),
                    "media_transcription_provider",
                ),
            ],
        )
    }

    // ── Model picker view helper ───────────────────────────────

    fn generation_section(&self) -> Element<'_, SettingsMessage> {
        section(
            "Generation",
            column![
                text("Image Generation")
                    .size(13)
                    .font(iced::Font::MONOSPACE)
                    .color(theme::ACCENT),
                Space::new().height(2),
                model_picker_list(
                    ModelPickerTarget::ImageGen,
                    self.config.image_gen_models.as_deref(),
                    self.config.image_gen_model.as_deref(),
                    self.model_picker_inputs[ModelPickerTarget::ImageGen.idx()].as_str(),
                    "model name (e.g. google/gemini-...)",
                ),
                Space::new().height(12),
                text("Video Generation")
                    .size(13)
                    .font(iced::Font::MONOSPACE)
                    .color(theme::ACCENT),
                Space::new().height(2),
                model_picker_list(
                    ModelPickerTarget::Video,
                    self.config.video_models.as_deref(),
                    self.config.video_model.as_deref(),
                    self.model_picker_inputs[ModelPickerTarget::Video.idx()].as_str(),
                    "model name (e.g. minimax/hailuo-3)",
                ),
            ],
        )
    }

    fn integrations_section(&self) -> Element<'_, SettingsMessage> {
        // ── Web search provider pick list ──────────────────────────
        // Three options: Auto (None), Firecrawl, Exa
        let current_display = match self.config.web_search_provider.as_deref() {
            Some("firecrawl") => "Firecrawl",
            Some("exa") => "Exa",
            _ => "Auto",
        };
        let pick_options: &[&str] = &["Auto", "Firecrawl", "Exa"];
        let pick_list = pick_list(pick_options, Some(current_display), |v| {
            let value = match v {
                "Firecrawl" => "firecrawl".to_string(),
                "Exa" => "exa".to_string(),
                _ => String::new(), // "Auto" → empty → None
            };
            SettingsMessage::ConfigField {
                key: "web_search_provider",
                value,
            }
        })
        .text_size(13)
        .style(super::widgets::pick_list_style)
        .width(Length::Fixed(180.0));

        let provider_row = field_row("Web Search Provider", pick_list.into(), None);

        section(
            "Integrations",
            column![
                provider_row,
                field_row(
                    "Firecrawl API Key",
                    password_input(
                        "fc-...",
                        self.config.firecrawl_key.as_deref().unwrap_or_default(),
                        self.password_visible
                            .contains(&PasswordTarget::FirecrawlKey),
                        |v| SettingsMessage::ConfigField {
                            key: "firecrawl_key",
                            value: v
                        },
                        SettingsMessage::TogglePasswordVisibility(PasswordTarget::FirecrawlKey),
                    ),
                    None,
                ),
                field_row(
                    "Exa API Key",
                    password_input(
                        "exa-...",
                        self.config.exa_key.as_deref().unwrap_or_default(),
                        self.password_visible.contains(&PasswordTarget::ExaKey),
                        |v| SettingsMessage::ConfigField {
                            key: "exa_key",
                            value: v
                        },
                        SettingsMessage::TogglePasswordVisibility(PasswordTarget::ExaKey),
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
                        self.password_visible
                            .contains(&PasswordTarget::TelegramToken),
                        |v| SettingsMessage::ConfigField {
                            key: "telegram_bot_token",
                            value: v
                        },
                        SettingsMessage::TogglePasswordVisibility(PasswordTarget::TelegramToken),
                    ),
                    Some("Applied automatically on save"),
                ),
            ],
        )
    }

    /// Voice Assistant section — enable/disable, enrollment, status.
    #[allow(clippy::too_many_lines)]
    fn voice_section(&self) -> Element<'_, SettingsMessage> {
        use iced::widget::Text;

        let voice_enabled = self.config.voice_enabled.as_deref() == Some("true");
        let status = crate::audio::voice::get_status();
        let templates = crate::audio::voice::get_classifier_weights();
        let has_templates = templates.is_some();

        let status_text: Element<'_, SettingsMessage> = match status.clone() {
            crate::audio::voice::VoiceStatus::Disabled => Text::new("Disabled").into(),
            crate::audio::voice::VoiceStatus::LoadingModels => Text::new("Loading models…").into(),
            crate::audio::voice::VoiceStatus::ModelError => {
                // Show retry button inline so the user can trigger recovery
                // without toggling voice off/on.
                let retry_btn = iced::widget::button(Text::new("   Retry   ").size(13))
                    .on_press(SettingsMessage::RetryVoiceModels)
                    .style(theme::button_danger)
                    .padding(4);
                iced::widget::row![
                    Text::new("Model error").size(14),
                    iced::widget::Space::new().width(8),
                    retry_btn,
                ]
                .align_y(iced::Alignment::Center)
                .into()
            }
            crate::audio::voice::VoiceStatus::Listening => {
                Text::new("Listening for wake word").into()
            }
            crate::audio::voice::VoiceStatus::Recording => Text::new("Recording command").into(),
            crate::audio::voice::VoiceStatus::RecordingManual => {
                Text::new("Recording voice message").into()
            }
            crate::audio::voice::VoiceStatus::Transcribing => Text::new("Transcribing…").into(),
            crate::audio::voice::VoiceStatus::MicPermissionDenied => {
                Text::new("Microphone permission denied").into()
            }
            crate::audio::voice::VoiceStatus::MicDisconnected => {
                Text::new("Microphone disconnected").into()
            }
            crate::audio::voice::VoiceStatus::Enrolling {
                sample,
                total,
                duration_ms,
                quality,
            } => {
                let remaining = total.saturating_sub(sample);
                let mut lines: Vec<String> = Vec::new();

                // Main status line with sample count and duration hint
                let duration_hint = if duration_ms > 0 {
                    if duration_ms >= crate::audio::voice::ENROLLMENT_QUALITY_DURATION_MAX_MS {
                        format!(
                            " — captured {}.{}s ✅",
                            duration_ms / 1000,
                            (duration_ms % 1000) / 100
                        )
                    } else if duration_ms >= crate::audio::voice::ENROLLMENT_QUALITY_DURATION_MIN_MS
                    {
                        format!(
                            " — captured {}.{}s 📝",
                            duration_ms / 1000,
                            (duration_ms % 1000) / 100
                        )
                    } else {
                        format!(
                            " — captured {}.{}s ⚠ too short",
                            duration_ms / 1000,
                            (duration_ms % 1000) / 100
                        )
                    }
                } else {
                    String::new()
                };

                if remaining > 0 {
                    if remaining == 1 {
                        lines.push(format!(
                            "Sample {sample}/{total}{duration_hint} — 1 more time."
                        ));
                    } else {
                        lines.push(format!(
                            "Sample {sample}/{total}{duration_hint} — {remaining} more times."
                        ));
                    }

                    // Enrollment prompt for multi-position guidance.
                    // Show a prompt as long as there's at least one more
                    // utterance to record (including before the first one).
                    if sample < total {
                        // `sample` is the count of completed samples; the prompt for
                        // the NEXT utterance is at index `sample` (0-based).
                        let prompt = crate::audio::voice::enrollment_prompt_for_sample(sample);
                        lines.push(format!("📢 {prompt}"));
                    }

                    // Quality feedback for the most recent sample
                    if let Some(ref q) = quality {
                        let quality_line = format!("{} (score: {:.2})", q.level.label(), q.score);
                        lines.push(quality_line);

                        if q.clipping_detected {
                            lines.push(
                                "⚠️ Clipping detected — your microphone gain may be too high"
                                    .to_string(),
                            );
                        }
                        if q.snr_db.is_finite() && q.snr_db < 10.0 {
                            lines.push(
                                "⚠️ Low signal-to-noise ratio — try speaking closer to the mic"
                                    .to_string(),
                            );
                        }
                    }
                } else {
                    lines.push("Processing…".to_string());
                }

                let text = lines.join("\n");
                Text::new(text).into()
            }
            crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. } => {
                Text::new("Listening…").into()
            }
            crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. } => {
                Text::new("Keep silent to confirm…").into()
            }
            crate::audio::voice::VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs,
                wall_clock_elapsed,
            } => {
                let pct = (accumulated_secs * 100)
                    .checked_div(target_secs)
                    .unwrap_or(0);
                Text::new(format!(
                    "Collecting negative samples… {accumulated_secs}s/{target_secs}s \
                     ({pct}%) elapsed {wall_clock_elapsed}s"
                ))
                .into()
            }
            crate::audio::voice::VoiceStatus::Enrolled => Text::new("Enrolled").into(),
            crate::audio::voice::VoiceStatus::Error(msg) => Text::new(msg).into(),
        };

        // Enrollment progress UI (shown during active enrollment)
        let enrollment_ui: Option<Element<'_, SettingsMessage>> = match status {
            crate::audio::voice::VoiceStatus::Enrolling { .. }
            | crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. }
            | crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
            | crate::audio::voice::VoiceStatus::EnrollingNegatives { .. } => {
                let cancel_btn: Element<'_, SettingsMessage> = container(
                    button(Text::new("Cancel").size(13))
                        .on_press(SettingsMessage::CancelVoiceEnrollment)
                        .style(theme::button_danger)
                        .padding(6),
                )
                .into();
                Some(
                    Column::new()
                        .push(iced::widget::Space::new().height(8))
                        .push(cancel_btn)
                        .into(),
                )
            }
            _ => None,
        };

        let enroll_btn: Element<'_, SettingsMessage> = if voice_enabled {
            container(
                iced::widget::button(Text::new("Enroll Wake Word").size(13))
                    .on_press(SettingsMessage::StartVoiceEnrollment)
                    .style(theme::button_primary)
                    .padding(6),
            )
            .into()
        } else {
            container(Text::new("")).into()
        };

        let wake_word_row = if voice_enabled {
            if let Some(phrase) = crate::audio::voice::get_enrolled_phrase() {
                field_row("Wake Word", Text::new(phrase).size(13).into(), None)
            } else if has_templates {
                // Legacy model loaded (pre-PersistedModel format) — no phrase known
                field_row(
                    "Wake Word",
                    Text::new("Legacy model (phrase unknown)").size(13).into(),
                    Some("Re-enroll to set a custom wake word phrase"),
                )
            } else {
                field_row(
                    "Wake Word",
                    Text::new("Enroll a wake word to get started")
                        .size(13)
                        .into(),
                    None,
                )
            }
        } else {
            iced::widget::Space::new().height(0).into()
        };

        // Text input for the wake word phrase (before enrollment).
        let is_enrolling = matches!(
            status,
            crate::audio::voice::VoiceStatus::Enrolling { .. }
                | crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::EnrollingNegatives { .. }
        );
        let phrase_input = if voice_enabled && !is_enrolling {
            let input = text_input("mahbot", &self.wake_word_phrase_input)
                .on_input(SettingsMessage::WakeWordPhraseInput)
                .style(super::widgets::text_input_style)
                .width(Length::Fixed(250.0));
            field_row("Wake Word Phrase", input.into(), None)
        } else {
            iced::widget::Space::new().height(0).into()
        };

        let mut column = Column::new()
            .push(field_row(
                "Enable Voice",
                iced::widget::toggler(voice_enabled)
                    .on_toggle(SettingsMessage::VoiceToggle)
                    .into(),
                Some("Hands-free voice commands with wake word detection"),
            ))
            .push(iced::widget::Space::new().height(8))
            .push(field_row("Status", status_text, None))
            .push(wake_word_row)
            .push(phrase_input)
            .push(enroll_btn);

        // Adaptive threshold k slider
        let current_k: f32 = self
            .config
            .adaptive_k
            .as_deref()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(2.5);
        let k_slider = slider(1.0_f32..=4.0_f32, current_k, move |v| {
            SettingsMessage::ConfigField {
                key: "adaptive_k",
                value: format!("{v:.1}"),
            }
        })
        .step(0.1_f32)
        .width(Length::Fixed(200.0));
        column = column.push(iced::widget::Space::new().height(8));
        column = column.push(field_row(
            "Adaptive K",
            iced::widget::row![
                k_slider,
                iced::widget::Space::new().width(8),
                text(format!("{current_k:.1}")).size(13),
            ]
            .align_y(iced::Alignment::Center)
            .into(),
            Some("Detection sensitivity (1.0–4.0, default 2.5)"),
        ));

        if let Some(ui) = enrollment_ui {
            column = column.push(ui);
        }

        section("Voice Assistant", column)
    }

    // ── TTS section ──────────────────────────────────────────

    fn tts_section(&self) -> Element<'_, SettingsMessage> {
        use iced::widget::Text;

        let tts_enabled = self.config.tts_enabled.as_deref() == Some("true");
        let ready = crate::audio::tts::models_ready();
        let failed = crate::audio::tts::download_failed();
        let audio_ok = crate::audio::tts::audio_output_ready();

        let status_text: Element<'_, SettingsMessage> = if !tts_enabled {
            Text::new("Disabled").into()
        } else if ready {
            if audio_ok {
                Text::new("Ready").into()
            } else {
                Text::new("No audio output device")
                    .color(theme::STATUS_ERROR)
                    .into()
            }
        } else if failed {
            // Show retry button inline so the user can trigger recovery
            // without toggling TTS off/on.
            let retry_btn = iced::widget::button(Text::new("   Retry   ").size(13))
                .on_press(SettingsMessage::TtsRetryModels)
                .style(theme::button_danger)
                .padding(4);
            iced::widget::row![
                Text::new("Model download failed").size(14),
                iced::widget::Space::new().width(8),
                retry_btn,
            ]
            .align_y(iced::Alignment::Center)
            .into()
        } else {
            Text::new("Downloading models…").into()
        };

        // Test button: enabled only when TTS is enabled and models loaded.
        // On click, audio output readiness is checked separately and shown
        // as a toast if unavailable — the button itself stays clickable so
        // the user gets immediate feedback about the problem.
        let test_btn = iced::widget::button(Text::new("   Test TTS   ").size(13))
            .style(theme::button_primary)
            .padding(4)
            .on_press_maybe(if tts_enabled && ready {
                Some(SettingsMessage::TtsTest)
            } else {
                None
            });

        let column = Column::new()
            .push(field_row(
                "Enable TTS",
                iced::widget::toggler(tts_enabled)
                    .on_toggle(SettingsMessage::TtsToggle)
                    .into(),
                Some("Text-to-speech for agent responses"),
            ))
            .push(iced::widget::Space::new().height(8))
            .push(field_row("Status", status_text, None))
            .push(iced::widget::Space::new().height(8))
            .push(field_row("", test_btn.into(), None));

        section("TTS", column)
    }

    fn routing_section(&self) -> Element<'_, SettingsMessage> {
        // Collect all unique models that should appear in the routing section:
        // 1. Every role's effective model (override from per_role_configs → hardcoded default)
        // 2. Every model with a saved routing entry (preserves orphaned entries)
        let mut model_names: BTreeSet<String> = BTreeSet::new();
        for role in Role::iter() {
            let role_key: &str = role.into();
            let model = self
                .role_config_for(role_key)
                .and_then(|rc| rc.model.clone().filter(|m| !m.is_empty()))
                .unwrap_or_else(|| crate::role::role_info(&role).default_model.to_string());
            model_names.insert(model);
        }
        for mr in &self.config.model_routings {
            model_names.insert(mr.model.clone());
        }

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
                .width(Length::Fixed(375.0));

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

        // No empty-state needed — defaults from Role::iter() always
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
    section_impl(title, None, content)
}

/// Section heading with an action button inline in the header row.
fn section_with_header_action<'a>(
    title: &'static str,
    action: Element<'a, SettingsMessage>,
    content: Column<'a, SettingsMessage>,
) -> Element<'a, SettingsMessage> {
    section_impl(title, Some(action), content)
}

/// Shared implementation: renders a section header (plain text or text +
/// right-aligned action), a spacer, and the content column.
fn section_impl<'a>(
    title: &'static str,
    action: Option<Element<'a, SettingsMessage>>,
    content: Column<'a, SettingsMessage>,
) -> Element<'a, SettingsMessage> {
    let styled_title = text(title)
        .font(iced::Font::MONOSPACE)
        .size(16)
        .color(theme::ACCENT);

    let header: Element<'a, SettingsMessage> = match action {
        Some(btn) => row![styled_title, Space::new().width(Length::Fill), btn,]
            .align_y(Alignment::Center)
            .into(),
        None => styled_title.into(),
    };

    column![header, Space::new().height(4), content.spacing(4)]
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
    on_input: impl Fn(String) -> SettingsMessage + 'a,
    on_toggle: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    let input: Element<_> = text_input(placeholder, value)
        .secure(!show)
        .on_input(on_input)
        .style(super::widgets::text_input_style)
        .width(Length::Fixed(375.0))
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

/// Delete-confirm button — shows a trash icon (with tooltip) or a
/// "Delete? Yes / No" confirmation prompt when the item is the delete target.
fn delete_confirm_button<'a>(
    is_delete_target: bool,
    on_confirm: SettingsMessage,
    on_cancel: SettingsMessage,
    on_delete: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    if is_delete_target {
        row![
            text("Delete?").size(12).color(theme::STATUS_ERROR),
            Space::new().width(4),
            button(text("Yes").size(11).color(theme::STATUS_ERROR))
                .style(theme::button_danger)
                .on_press(on_confirm),
            Space::new().width(4),
            button(text("No").size(11))
                .style(theme::button_secondary)
                .on_press(on_cancel),
        ]
        .into()
    } else {
        row![
            tooltip(
                button(
                    lucide::x::<iced::Theme, iced::Renderer>()
                        .size(18)
                        .color(theme::STATUS_ERROR),
                )
                .style(theme::button_text)
                .on_press(on_delete),
                "Delete",
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style)
            .delay(Duration::from_millis(400)),
        ]
        .into()
    }
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

/// Config text input — label on left, styled text input on right.
fn config_text_input<'a>(
    label: &'static str,
    placeholder: &str,
    value: &str,
    config_key: &'static str,
) -> Element<'a, SettingsMessage> {
    field_row(
        label,
        text_input(placeholder, value)
            .on_input(move |v| SettingsMessage::ConfigField {
                key: config_key,
                value: v,
            })
            .style(super::widgets::text_input_style)
            .width(Length::Fixed(375.0))
            .into(),
        None,
    )
}

/// Configuration for a single text field in [`modal_dialog`].
struct DialogField<'a> {
    label: &'static str,
    placeholder: &'static str,
    value: &'a str,
    /// Function pointer for the text-input `on_input` handler.
    ///
    /// Uses `fn(String) -> SettingsMessage` (function pointer) rather than
    /// `impl Fn(String) -> SettingsMessage + 'a` to keep the struct simple,
    /// avoid boxing, and rely on monomorphization at the callsite. This works
    /// because all current callers pass enum tuple-variant constructors (e.g.
    /// [`SettingsMessage::AddWorkspaceName`]), which coerce to function pointers.
    ///
    /// If a future caller needs to capture state in the closure, this field
    /// must be changed to `Box<dyn Fn(String) -> SettingsMessage + 'a>`.
    on_input: fn(String) -> SettingsMessage,
}

/// Build a reusable add-item modal dialog with text fields, a Cancel button,
/// and a conditionally-enabled Submit (Add) button.
///
/// Layout: title, spacer(16), field rows (8 px between), spacer(16), Cancel / Add buttons.
fn modal_dialog<'a>(
    title: &'static str,
    fields: &[DialogField<'a>],
    adding: bool,
    submit_enabled: bool,
    on_cancel: SettingsMessage,
    on_submit: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    let mut col = Column::new().padding(24);

    col = col.push(
        text(title)
            .size(16)
            .color(theme::TEXT_PRIMARY)
            .font(theme::FONT_BOLD),
    );

    if !fields.is_empty() {
        col = col.push(Space::new().height(16));
    }

    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            col = col.push(Space::new().height(8));
        }
        col = col.push(field_row(
            field.label,
            text_input(field.placeholder, field.value)
                .on_input(field.on_input)
                .style(widgets::text_input_style)
                .width(Length::Fixed(375.0))
                .into(),
            None,
        ));
    }

    if !fields.is_empty() {
        col = col.push(Space::new().height(16));
    }

    col = col.push(
        row![
            Space::new().width(Length::Fill),
            button(text("Cancel").size(13))
                .style(theme::button_secondary)
                .on_press(on_cancel),
            Space::new().width(8),
            button(text(if adding { "Adding..." } else { "Add" }).size(13),)
                .style(theme::button_primary)
                .on_press_maybe(if adding || !submit_enabled {
                    None
                } else {
                    Some(on_submit)
                }),
        ]
        .align_y(Alignment::Center),
    );

    container(col)
        .width(Length::Fixed(620.0))
        .style(theme::dialog_container_style)
        .into()
}

/// Wrap a dialog element inside a semi-transparent backdrop, centered on screen.
/// Uses the shared backdrop helper for consistent overlay behavior.
fn modal_with_backdrop(
    dialog: Element<'_, SettingsMessage>,
    on_backdrop_click: SettingsMessage,
) -> Element<'_, SettingsMessage> {
    widget_helpers::modal_backdrop(dialog, on_backdrop_click, 0.5)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_models ─────────────────────────────────────────

    #[test]
    fn parse_models_cases() {
        struct Case {
            name: &'static str,
            input: Option<&'static str>,
            expected: &'static [&'static str],
        }

        let cases = [
            Case {
                name: "None returns empty",
                input: None,
                expected: &[],
            },
            Case {
                name: "empty string returns empty",
                input: Some(""),
                expected: &[],
            },
            Case {
                name: "single line",
                input: Some("google/gemini-3.1-flash-image-preview"),
                expected: &["google/gemini-3.1-flash-image-preview"],
            },
            Case {
                name: "multiple lines",
                input: Some("model-a\nmodel-b\nmodel-c"),
                expected: &["model-a", "model-b", "model-c"],
            },
            Case {
                name: "trims whitespace",
                input: Some("  model-a  \n  model-b  "),
                expected: &["model-a", "model-b"],
            },
            Case {
                name: "skips empty lines",
                input: Some("model-a\n\n\nmodel-b"),
                expected: &["model-a", "model-b"],
            },
            Case {
                name: "skips whitespace-only lines",
                input: Some("model-a\n   \nmodel-b"),
                expected: &["model-a", "model-b"],
            },
        ];

        for case in &cases {
            let result = parse_models(case.input);
            let expected: Vec<String> = case.expected.iter().map(ToString::to_string).collect();
            assert_eq!(result, expected, "case: {}", case.name);
        }
    }

    // ── add_model_to_list ────────────────────────────────────

    #[test]
    fn add_model_to_list_cases() {
        struct Case {
            name: &'static str,
            input: &'static str,
            initial_list: Option<&'static str>,
            expected_list: Option<&'static str>,
            expect_input_cleared: bool,
        }

        let cases = [
            Case {
                name: "empty input does nothing",
                input: "",
                initial_list: None,
                expected_list: None,
                expect_input_cleared: false,
            },
            Case {
                name: "whitespace input does nothing",
                input: "  ",
                initial_list: None,
                expected_list: None,
                expect_input_cleared: false,
            },
            Case {
                name: "adds to empty list",
                input: "model-a",
                initial_list: None,
                expected_list: Some("model-a"),
                expect_input_cleared: true,
            },
            Case {
                name: "adds to existing list",
                input: "model-c",
                initial_list: Some("model-a\nmodel-b"),
                expected_list: Some("model-a\nmodel-b\nmodel-c"),
                expect_input_cleared: true,
            },
            Case {
                name: "skips duplicates",
                input: "model-a",
                initial_list: Some("model-a\nmodel-b"),
                expected_list: Some("model-a\nmodel-b"),
                expect_input_cleared: true,
            },
            Case {
                name: "trims input",
                input: "  model-a  ",
                initial_list: Some("model-b"),
                expected_list: Some("model-b\nmodel-a"),
                expect_input_cleared: true,
            },
        ];

        for case in &cases {
            let mut input = case.input.to_string();
            let mut list = case.initial_list.map(String::from);
            add_model_to_list(&mut input, &mut list);
            assert_eq!(
                list,
                case.expected_list.map(String::from),
                "case: {} — list mismatch",
                case.name
            );
            if case.expect_input_cleared {
                assert!(
                    input.is_empty(),
                    "case: {} — input buffer should be cleared",
                    case.name
                );
            } else {
                assert_eq!(
                    input, case.input,
                    "case: {} — input should remain unchanged",
                    case.name
                );
            }
        }
    }

    // ── remove_model_from_list ───────────────────────────────

    #[test]
    fn remove_model_from_list_cases() {
        struct Case {
            name: &'static str,
            model: &'static str,
            initial_list: Option<&'static str>,
            initial_active: Option<&'static str>,
            expected_list: Option<&'static str>,
            expected_active: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "removes and updates active",
                model: "model-b",
                initial_list: Some("model-a\nmodel-b\nmodel-c"),
                initial_active: Some("model-b"),
                expected_list: Some("model-a\nmodel-c"),
                expected_active: Some("model-a"),
            },
            Case {
                name: "non-active removal keeps active",
                model: "model-b",
                initial_list: Some("model-a\nmodel-b\nmodel-c"),
                initial_active: Some("model-a"),
                expected_list: Some("model-a\nmodel-c"),
                expected_active: Some("model-a"),
            },
            Case {
                name: "last entry clears active",
                model: "model-a",
                initial_list: Some("model-a"),
                initial_active: Some("model-a"),
                expected_list: None,
                expected_active: None,
            },
            Case {
                name: "not found no change",
                model: "model-c",
                initial_list: Some("model-a\nmodel-b"),
                initial_active: Some("model-a"),
                expected_list: Some("model-a\nmodel-b"),
                expected_active: Some("model-a"),
            },
            Case {
                name: "empty list with matching active clears active",
                model: "model-a",
                initial_list: None,
                initial_active: Some("model-a"),
                expected_list: None,
                expected_active: None,
            },
        ];

        for case in &cases {
            let mut list = case.initial_list.map(String::from);
            let mut active = case.initial_active.map(String::from);
            remove_model_from_list(case.model, &mut list, &mut active);
            assert_eq!(
                list,
                case.expected_list.map(String::from),
                "case: {} — list mismatch",
                case.name
            );
            assert_eq!(
                active,
                case.expected_active.map(String::from),
                "case: {} — active mismatch",
                case.name
            );
        }
    }

    // ── Toggle generation counter & rollback (shared) ──────────────

    /// Shared helper for toggle generation-counter and rollback tests.
    /// Parameterised by message constructors and field accessors so the
    /// same 7-scenario sequence (toggle ON, stale result, DB error revert,
    /// re-toggle, successful persist, stale-old-gen, toggle OFF + revert)
    /// is exercised exactly once per toggle without code duplication.
    fn assert_toggle_gen_counter_and_rollback(
        toggle_on: impl Fn(bool) -> SettingsMessage,
        toggle_result: impl Fn(u64, Result<(), String>) -> SettingsMessage,
        get_enabled: impl Fn(&ConfigData) -> &Option<String>,
        get_gen: impl Fn(&SettingsState) -> u64,
        setup: impl Fn(&mut SettingsState),
    ) {
        let mut state = SettingsState::new();
        setup(&mut state);

        // ── Initial state ──
        assert_eq!(get_gen(&state), 0, "initial gen is 0");
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            None,
            "starts disabled"
        );
        assert!(state.error.is_none(), "no error initially");

        // ── Toggle ON ──
        let _task = state.update(toggle_on(true));
        assert_eq!(get_gen(&state), 1, "gen incremented after toggle ON");
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some("true"),
            "enabled set to Some(\"true\") after toggle ON"
        );

        // ── Stale result from previous generation must be ignored ──
        let _task = state.update(toggle_result(0, Err("stale result".into())));
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some("true"),
            "stale ToggleResult with Err must NOT revert the state"
        );
        assert_eq!(get_gen(&state), 1, "gen unchanged by stale result");

        // ── Correct generation + DB error → rollback to disabled ──
        let _task = state.update(toggle_result(1, Err("db write failed".into())));
        assert!(
            get_enabled(&state.config).as_deref() != Some("true"),
            "errant ToggleResult must revert enabled away from Some(\"true\")"
        );
        assert_eq!(
            state.error.as_deref(),
            Some("db write failed"),
            "error message set after failed toggle"
        );
        assert_eq!(get_gen(&state), 1, "gen unchanged after rollback");

        // ── Toggle ON again, succeed this time ──
        state.error = None; // clear previous error
        let _task = state.update(toggle_on(true));
        assert_eq!(get_gen(&state), 2, "gen incremented on second toggle");
        let _task = state.update(toggle_result(2, Ok(())));
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some("true"),
            "successful ToggleResult must keep enabled state"
        );
        assert!(state.error.is_none(), "no error after successful toggle");

        // ── Stale result from old generation must also be ignored ──
        let _task = state.update(toggle_result(1, Err("stale from old gen".into())));
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some("true"),
            "stale ToggleResult from gen=1 must NOT revert state when current gen=2"
        );
        assert!(state.error.is_none(), "stale result must NOT set error");

        // ── Toggle OFF with DB error → rollback back to enabled ──
        let _task = state.update(toggle_on(false));
        assert_eq!(get_gen(&state), 3, "gen incremented on toggle OFF");
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some(""),
            "enabled set to Some(\"\") after toggle OFF"
        );
        let _task = state.update(toggle_result(3, Err("db delete failed".into())));
        assert_eq!(
            get_enabled(&state.config).as_deref(),
            Some("true"),
            "errant ToggleResult(false) must revert back to enabled"
        );
        assert_eq!(
            state.error.as_deref(),
            Some("db delete failed"),
            "error set after failed disable toggle"
        );
    }

    #[test]
    fn voice_toggle_generation_counter_and_rollback() {
        // The update handler calls sync_voice_state which accesses voice pipeline
        // globals.  Initialise the pipeline state (no-op if already initialised
        // by another test — OnceCell::set only succeeds once).
        assert_toggle_gen_counter_and_rollback(
            SettingsMessage::VoiceToggle,
            SettingsMessage::VoiceToggleResult,
            |c| &c.voice_enabled,
            |s| s.voice_toggle_gen,
            |s| {
                // Clear any pre-existing voice_enabled from the snapshot to avoid
                // test isolation issues (other tests may have set it in global CONFIG).
                let _ = s.config.set_string_field("voice_enabled", "");
                s.config.normalize();
                let _ = crate::audio::voice::init_global();
            },
        );
    }

    #[test]
    fn tts_toggle_generation_counter_and_rollback() {
        // Set TTS state to READY so that toggling ON does not trigger
        // spawn_or_retry_download() (which requires a Tokio runtime).  The
        // model download logic is tested separately — this test focuses on
        // the generation-counter and rollback behaviour.
        assert_toggle_gen_counter_and_rollback(
            SettingsMessage::TtsToggle,
            SettingsMessage::TtsToggleResult,
            |c| &c.tts_enabled,
            |s| s.tts_toggle_gen,
            |s| {
                // Clear any pre-existing tts_enabled from the snapshot to avoid
                // test isolation issues (other tests may have set it in global CONFIG).
                let _ = s.config.set_string_field("tts_enabled", "");
                s.config.normalize();
                crate::audio::tts::test_set_state(2); // ModelState::Ready
            },
        );
    }
}

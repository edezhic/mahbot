//! Settings page — dynamic configuration editor.
//!
//! Reads the current config snapshot from [`crate::config::CONFIG`],
//! presents editable fields organised in sections, and persists every change
//! immediately when the value settles (debounced text inputs, immediate
//! toggles/pickers) via the per-field persistence
//! functions in [`crate::config`].
//!
//! Also manages workspaces and users (formerly separate pages), with
//! modal dialogs for add operations.

use crate::Role;
use crate::Workspace;
use crate::config::{
    CONFIG, CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, CONFIG_KEY_EXA_KEY, CONFIG_KEY_FIRECRAWL_KEY,
    CONFIG_KEY_MANAGER_MODEL, CONFIG_KEY_PROVIDER_ENDPOINT, CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
    CONFIG_KEY_PROVIDER_KEY, CONFIG_KEY_TELEGRAM_BOT_TOKEN, CONFIG_KEY_TTS_ENABLED,
    CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL, CONFIG_KEY_VOICE_ENABLED, CONFIG_KEY_WEB_SEARCH_PROVIDER,
    CONFIG_KEY_WORKER_MODEL, ConfigData, ModelRouting,
};
use crate::workspace::MAX_WORKSPACE_NOTES_CHARS;
use strum::{EnumCount, IntoEnumIterator};

use iced::widget::{
    Column, Id, Row, Space, button, column, container, pick_list, row, scrollable, stack, text,
    toggler, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::collections::{BTreeSet, HashMap, HashSet};

use super::common::SingleLineEditorState;
use super::editor_widget::EditorAction;
use super::menus::{ContextMenu, MenuItem};
use super::theme;
use super::users;
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
fn add_model_to_list(input: &mut SingleLineEditorState, list: &mut Option<String>) {
    let model = input.text().trim().to_string();
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
fn model_picker_list<'a>(
    target: ModelPickerTarget,
    models_field: Option<&'a str>,
    active_field: Option<&'a str>,
    add_input: &'a SingleLineEditorState,
    add_placeholder: &'static str,
    error: Option<&'a str>,
) -> Element<'a, SettingsMessage> {
    let on_add_input = move |action| SettingsMessage::ModelPicker {
        target,
        action: ModelPickerAction::AddInput(action),
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
        widgets::single_line_editor(
            &add_input.buffer,
            add_placeholder,
            false, // Enter adds nothing here; the Add button drives the mutation.
            Length::Fixed(450.0),
            Some(Id::from(format!("model_picker:{}", target.idx()))),
            on_add_input,
        ),
        Space::new().width(4),
        button(text("Add").size(11))
            .padding(4)
            .style(theme::button_primary)
            .on_press(on_add),
    ]
    .align_y(Alignment::Center);

    let mut col = column![
        Column::from_iter(items).spacing(2),
        Space::new().height(4),
        add_row,
    ];

    if let Some(err) = error {
        col = col.push(Space::new().height(2));
        col = col.push(inline_error(err, 0.0));
    }

    col.into()
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
    AddInput(EditorAction),
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
    EndpointKey,
}

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    /// Generic editable config field identified by its snake_case key
    /// (matches the keys in [`crate::config::ConfigData::set_string_field`]).
    /// Stages the value in the editable snapshot; text inputs also arm a
    /// debounced settle, immediate controls (toggles / pick lists) persist
    /// right away, and the value is settled via
    /// [`ConfigFieldSettleNow`](Self::ConfigFieldSettleNow).
    ConfigField {
        key: &'static str,
        value: String,
    },
    /// A config field value has settled and should be persisted now.
    ///
    /// `field` is the canonical field id (`config:<key>`,
    /// `routing_order:<model>`). `generation` is the
    /// generation counter captured when the value was staged; a settle whose
    /// generation no longer matches the current counter is stale and dropped,
    /// so per-keystroke and out-of-order writes are impossible.
    ConfigFieldSettled {
        field: String,
        value: String,
        generation: u64,
    },
    /// Settle the staged value of a field immediately (Enter on a text input)
    /// instead of waiting for the debounce timer.
    ConfigFieldSettleNow {
        field: String,
    },
    /// Result of an async per-field persist. `Ok` carries a
    /// [`crate::config::PersistOutcome`]: the canonical persisted value
    /// (trimmed, `None` collapsed to `""`) plus an optional non-fatal warning
    /// (e.g. an unreachable custom endpoint saved anyway). The handler
    /// applies the value only when the generation is still current (a stale
    /// result is dropped).
    ConfigFieldSaveResult {
        field: String,
        generation: u64,
        result: Result<crate::config::PersistOutcome, String>,
    },
    /// Toggle the custom chat-completions endpoint section.
    /// ON only reveals the endpoint fields — nothing is staged, settled, or
    /// persisted (an endpoint becomes active when the user settles a
    /// genuinely non-default URL). OFF reverts to OpenRouter: closes the
    /// section and settles the endpoint field with `""`, so the persist
    /// path clears both persisted endpoint rows.
    CustomEndpointToggle(bool),
    /// Per-model provider routing edits
    ModelRoutingOrder {
        model: String,
        action: EditorAction,
    },
    /// A shared-editor action on a stateless config text field. The field's
    /// buffer lives in [`SettingsState::field_editors`] keyed by
    /// `config:<key>`; the handler applies the action to it and stages the
    /// resulting text as a [`ConfigField`], settling immediately on
    /// [`EditorAction::Submit`].
    ConfigFieldAction {
        key: &'static str,
        action: EditorAction,
    },
    /// A shared-editor action on a masked config (password) field. Mirrors
    /// [`ConfigFieldAction`], with [`password_field_editor`](super::widgets::password_field_editor)
    /// rendering and the eye toggle handled separately.
    PasswordFieldAction {
        key: &'static str,
        action: EditorAction,
    },
    /// Toggle password visibility for a specific field.
    TogglePasswordVisibility(PasswordTarget),
    // ── Workspace management (sub-messages) ─────────────────────
    /// Wrapped workspace message.
    WorkspaceMsg(workspaces::WorkspacesMessage),
    /// Toggle the add-workspace modal.
    ToggleAddWorkspaceModal,
    /// Add-workspace modal fields.
    AddWorkspaceName(EditorAction),
    AddWorkspacePath(EditorAction),
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
    AddUserSender(EditorAction),
    /// Select the default agent in the add-user modal (index into
    /// `[Role::Assistant, Role::Artist]`).
    AddUserDefaultRole(usize),
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
    // ── Transcription messages ────────────────────────────
    /// Toggle local transcription on/off. Turning it OFF also turns Wake Word
    /// Detection off (they share the loaded ASR model — the cascade persists
    /// `voice_enabled` away and stops the pipeline). Toggling ON kicks the
    /// model load/download in the background (auto-activates when ready).
    TranscriptionToggle(bool),
    /// Result of the async transcription-toggle persistence. `voice_was_enabled`
    /// carries the pre-toggle wake-word state so a failed persist can roll both
    /// keys (and the pipeline) back. `generation` guards against stale results
    /// from rapid toggling (mirrors [`Self::VoiceToggleResult`]).
    TranscriptionToggleResult {
        generation: u64,
        voice_was_enabled: bool,
        result: Result<(), String>,
    },
    /// Retry the local ASR model load/download after a terminal failure
    /// (re-toggling alone cannot recover — a dedicated action is required).
    RetryTranscription,
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
    WakeWordPhraseInput(EditorAction),
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

/// Debounce delay for text inputs: a change is persisted only when the value
/// has settled (typing paused this long), or immediately on Enter.
const SETTLE_MS: u64 = 700;

/// Config keys rendered as free-text inputs — persisted only after the value
/// settles (debounce / Enter), never per keystroke.
const TEXT_INPUT_KEYS: &[&str] = &[
    CONFIG_KEY_PROVIDER_KEY,
    CONFIG_KEY_PROVIDER_ENDPOINT,
    CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
    CONFIG_KEY_FIRECRAWL_KEY,
    CONFIG_KEY_EXA_KEY,
    CONFIG_KEY_TELEGRAM_BOT_TOKEN,
    CONFIG_KEY_MANAGER_MODEL,
    CONFIG_KEY_WORKER_MODEL,
    CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL,
];

/// Config keys rendered as discrete controls (toggles / pick lists) that
/// persist immediately on change. `audio_transcription_use_local` is not a
/// generic ConfigField key — it is a dedicated
/// [`SettingsMessage::TranscriptionToggle`] toggle with its own transactional
/// persist path.
const IMMEDIATE_KEYS: &[&str] = &[CONFIG_KEY_WEB_SEARCH_PROVIDER];

pub struct SettingsState {
    /// Current editable snapshot, loaded from CONFIG each refresh.
    config: ConfigData,
    /// Per-field generation counters for settled autosaves, keyed by the
    /// canonical field id (see [`SettingsMessage::ConfigFieldSettled`]).
    ///
    /// Bumped every time a field's value is staged (keystroke, toggle,
    /// picker action). A settle or persist result whose generation no longer
    /// matches is stale and dropped, so out-of-order async writes can never
    /// land after a newer edit.
    field_gen: HashMap<String, u64>,
    /// Fields with a persist task currently in flight. Only one persist per
    /// field runs at a time, so the last settle always lands last in the DB —
    /// generation counters alone protect the UI snapshot, not write ordering
    /// (two concurrent persists for the same field could otherwise complete
    /// out of order and the older value would land last).
    in_flight_persists: HashSet<String>,
    /// Values queued for fields whose persist is in flight: the user's latest
    /// edit for that field (value plus the generation at which it was
    /// queued), flushed when the in-flight persist completes. The flush
    /// carries the queued generation — if a newer edit was staged in the
    /// meantime, the flushed persist's result is dropped by the stale-result
    /// check instead of clobbering the newer staged value.
    pending_persists: HashMap<String, (String, u64)>,
    /// Per-field inline errors from rejected settles (invalid values are
    /// never persisted — the error is shown next to the offending control).
    field_errors: HashMap<String, String>,
    /// Inline warning from the last custom-endpoint save (non-fatal — e.g.
    /// an unreachable endpoint saved anyway). Only the
    /// endpoint persist arm can produce a warning, so a single slot
    /// suffices — a keyed map here would be single-key dead state.
    endpoint_warning: Option<String>,
    /// Last error message rendered in the bottom banner — voice/TTS toggle
    /// failures and failed custom-endpoint saves.
    error: Option<String>,
    /// Per-field presentation/undo state for stateless config fields rendered
    /// through the shared single-line editor. Keyed by the canonical field id
    /// (`config:<key>`, `routing_order:<model>`), so each field owns its own
    /// buffer and undo stack across renders while `view(&self)` stays immutable.
    /// Entries are (re)populated from the config snapshot on [`Self::refresh`];
    /// update handlers insert lazily via [`Self::field_editor_mut`].
    field_editors: HashMap<String, SingleLineEditorState>,
    /// Which password fields are currently visible.
    password_visible: HashSet<PasswordTarget>,
    /// Whether the custom-endpoint section was revealed by the user this
    /// session (UI-only). An active custom endpoint keeps the section open
    /// across renders without this flag. Distinct from
    /// [`Self::custom_endpoint_active_ui`]: revealing the fields alone does
    /// not configure a custom endpoint (nothing is persisted on reveal).
    /// A failed endpoint save closes the section (the runtime stayed on
    /// OpenRouter), so the toggle never shows ON while nothing is configured.
    custom_revealed: bool,

    // ── Workspace management state ──────────────────────────────
    pub(crate) workspaces_state: workspaces::WorkspacesState,
    /// Whether the add-workspace modal is visible.
    show_add_workspace_modal: bool,
    /// Name field in the add-workspace modal.
    add_workspace_name: SingleLineEditorState,
    /// Path field in the add-workspace modal.
    add_workspace_path: SingleLineEditorState,
    /// Whether the add-workspace operation is in flight.
    add_workspace_adding: bool,

    // ── User management state ───────────────────────────────────
    pub(crate) users_state: users::UsersState,
    /// Whether the add-user modal is visible.
    show_add_user_modal: bool,
    /// Name field in the add-user modal.
    add_user_sender: SingleLineEditorState,
    /// Default agent for the new user, as an index into
    /// `[Role::Assistant, Role::Artist]` (0 = Assistant).
    add_user_default: usize,
    /// Whether the add-user operation is in flight.
    add_user_adding: bool,

    // ── Model picker state ────────────────────────────────
    /// Text input buffers for model pickers, indexed by [`ModelPickerTarget::idx`].
    model_picker_inputs: [SingleLineEditorState; ModelPickerTarget::COUNT],

    // ── Voice assistant state ─────────────────────────────
    /// Generation counter for voice toggle operations.
    /// Incremented before each `VoiceToggle`; the expected value is
    /// passed through to `VoiceToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    voice_toggle_gen: u64,
    /// Generation counter for transcription toggle operations.
    /// Incremented before each `TranscriptionToggle`; the expected value is
    /// passed through to `TranscriptionToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    transcription_toggle_gen: u64,
    /// Transient text input for the wake word phrase.
    /// Not persisted — passed to [`VoiceCommand::StartEnrollment`] on click.
    wake_word_phrase_input: SingleLineEditorState,

    // ── TTS state ─────────────────────────────────────────
    /// Generation counter for TTS toggle operations.
    /// Incremented before each `TtsToggle`; the expected value is
    /// passed through to `TtsToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    tts_toggle_gen: u64,
}

/// Sync the voice assistant pipeline state with `CONFIG.voice_enabled()`.
/// Called from the immediate `VoiceToggle` handler after the toggle persists.
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
            field_gen: HashMap::new(),
            in_flight_persists: HashSet::new(),
            pending_persists: HashMap::new(),
            field_errors: HashMap::new(),
            field_editors: Self::initial_field_editors(&CONFIG.snapshot()),
            endpoint_warning: None,
            error: None,
            password_visible: HashSet::new(),
            custom_revealed: false,
            workspaces_state: workspaces::WorkspacesState::new(),
            users_state: users::UsersState::new(),
            show_add_workspace_modal: false,
            add_workspace_name: SingleLineEditorState::new(""),
            add_workspace_path: SingleLineEditorState::new(""),
            add_workspace_adding: false,
            show_add_user_modal: false,
            add_user_sender: SingleLineEditorState::new(""),
            add_user_default: 0,
            add_user_adding: false,
            model_picker_inputs: std::array::from_fn(|_| SingleLineEditorState::new("")),
            voice_toggle_gen: 0,
            transcription_toggle_gen: 0,
            wake_word_phrase_input: SingleLineEditorState::new(""),
            tts_toggle_gen: 0,
        }
    }

    /// Reload the editable snapshot from the current CONFIG.
    ///
    /// Inline errors are cleared (the page is being shown fresh). Pending
    /// debounced settles are NOT invalidated — a value typed before
    /// navigating away must still be persisted when its settle fires.
    pub fn refresh(&mut self) {
        self.config = CONFIG.snapshot();
        self.error = None;
        self.field_errors.clear();
        self.endpoint_warning = None;
        self.resync_field_editors();
    }

    /// Pre-populate the lazy field editors from a config snapshot: every
    /// `config:<key>` in [`TEXT_INPUT_KEYS`] plus a `routing_order:<model>`
    /// entry for the two routable slots (manager + worker), the set reflected
    /// by [`Self::routing_section`]. Kept in sync on [`Self::refresh`] via
    /// [`Self::resync_field_editors`].
    fn initial_field_editors(config: &ConfigData) -> HashMap<String, SingleLineEditorState> {
        let mut map = HashMap::new();
        for key in TEXT_INPUT_KEYS {
            let field = format!("config:{key}");
            let value = config.get_string_field(key).unwrap_or_default().to_string();
            map.insert(field, SingleLineEditorState::new(&value));
        }
        let mut models = BTreeSet::new();
        models.insert(crate::config::resolve_or(
            config.manager_model.clone(),
            crate::config::DEFAULT_MANAGER_MODEL,
        ));
        models.insert(crate::config::resolve_or(
            config.worker_model.clone(),
            crate::config::DEFAULT_WORKER_MODEL,
        ));
        for model in models {
            let field = format!("routing_order:{model}");
            let value = config
                .model_routings
                .iter()
                .find(|mr| mr.model == model)
                .and_then(|mr| mr.provider_order.clone())
                .unwrap_or_default();
            map.insert(field, SingleLineEditorState::new(&value));
        }
        map
    }

    /// Re-create/re-sync the stateless field editors from the current config
    /// snapshot. Existing editors get `set_text` (so an in-flight external
    /// refresh is applied); entries for fields not yet rendered are inserted
    /// with the current value. Routing editors cover the two routable slots
    /// (manager + worker), the set reflected by [`Self::routing_section`].
    fn resync_field_editors(&mut self) {
        for key in TEXT_INPUT_KEYS {
            let field = format!("config:{key}");
            let value = self
                .config
                .get_string_field(key)
                .unwrap_or_default()
                .to_string();
            self.field_editors
                .entry(field)
                .and_modify(|e| e.set_text(&value))
                .or_insert_with(|| SingleLineEditorState::new(&value));
        }
        let mut models = BTreeSet::new();
        models.insert(crate::config::resolve_or(
            self.config.manager_model.clone(),
            crate::config::DEFAULT_MANAGER_MODEL,
        ));
        models.insert(crate::config::resolve_or(
            self.config.worker_model.clone(),
            crate::config::DEFAULT_WORKER_MODEL,
        ));
        for model in models {
            let field = format!("routing_order:{model}");
            let value = self
                .config
                .model_routings
                .iter()
                .find(|mr| mr.model == model)
                .and_then(|mr| mr.provider_order.clone())
                .unwrap_or_default();
            self.field_editors
                .entry(field)
                .and_modify(|e| e.set_text(&value))
                .or_insert_with(|| SingleLineEditorState::new(&value));
        }
    }

    /// Push the canonical value for a single field id into its editor (if the
    /// editor has been created). Called after an external config reload or a
    /// completed persist so the rendered value reflects the source of truth.
    fn resync_field_editor(&mut self, field: &str) {
        if let Some(value) = self.staged_value(field) {
            if let Some(editor) = self.field_editors.get_mut(field) {
                editor.set_text(&value);
            }
        }
    }

    /// Borrow the per-field editor for a rendered stateless config field.
    /// Entries are guaranteed to exist via [`Self::resync_field_editors`]
    /// (config keys) / the routing pre-population, so this never panics in
    /// practice.
    fn field_editor(&self, key: &str) -> &SingleLineEditorState {
        self.field_editors
            .get(key)
            .expect("field editor populated on refresh")
    }

    /// Mutable variant of [`Self::field_editor`] used by update handlers. The
    /// editor is inserted from `initial` if the field was never rendered (and
    /// thus not yet pre-populated).
    fn field_editor_mut(&mut self, key: &str, initial: &str) -> &mut SingleLineEditorState {
        self.field_editors
            .entry(key.to_string())
            .or_insert_with(|| SingleLineEditorState::new(initial))
    }

    /// Apply an [`EditorAction`] to a stateless `config:<key>` field's editor
    /// and stage the resulting text. Non-submit edits arm the debounced settle
    /// via [`ConfigField`]; [`EditorAction::Submit`] settles immediately (the
    /// staged value is already current in the editable snapshot).
    fn handle_field_action(
        &mut self,
        key: &'static str,
        action: EditorAction,
    ) -> Task<SettingsMessage> {
        if let Some(task) = super::common::focus_navigation_task(&action) {
            return task;
        }
        let field = format!("config:{key}");
        self.field_errors.remove(&field);
        let submit = matches!(action, EditorAction::Submit);
        let changes_text = action.changes_text();
        let value = {
            let initial = self.staged_value(&field).unwrap_or_default();
            let editor = self.field_editor_mut(&field, &initial);
            editor.apply_action(action);
            editor.text()
        };
        if submit {
            self.settle_now(&field, value)
        } else if changes_text {
            self.update(SettingsMessage::ConfigField { key, value })
        } else {
            Task::none()
        }
    }

    /// Whether a genuinely non-default custom chat endpoint is staged or
    /// persisted in the editable snapshot (normalized — trivial variants of
    /// the default OpenRouter URL never count). Single predicate for the
    /// OpenRouter-key highlight, the toggle state, and the Provider Routing
    /// annotation, so UI state can never diverge from the
    /// normalized runtime endpoint.
    fn custom_endpoint_active_ui(&self) -> bool {
        crate::config::is_custom_endpoint(&crate::config::effective_chat_endpoint(&self.config))
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
        self.add_user_default = 0;
        self.add_user_adding = false;
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

    // ── Per-field autosave helpers ─────────────────────────────

    /// Bump the generation counter for a field and return the new value.
    fn bump_gen(&mut self, field: &str) -> u64 {
        let g = self.field_gen.entry(field.to_string()).or_insert(0);
        *g = g.wrapping_add(1);
        *g
    }

    /// Read the currently staged value of a canonical field id from the
    /// editable snapshot (used by Enter/submit settles).
    ///
    /// Only text-style fields reach this path: `ConfigFieldSettleNow` is
    /// dispatched by the Enter/submit handlers, and discrete controls settle
    /// with their value passed directly to [`Self::settle_now`] instead.
    fn staged_value(&self, field: &str) -> Option<String> {
        if let Some(key) = field.strip_prefix("config:") {
            return self.config.get_string_field(key).map(String::from);
        }
        if let Some(model) = field.strip_prefix("routing_order:") {
            return self
                .config
                .model_routings
                .iter()
                .find(|mr| mr.model == model)
                .and_then(|mr| mr.provider_order.clone());
        }
        None
    }

    /// Schedule an immediate (delay-0) settle for a field.
    fn settle_now(&mut self, field: &str, value: String) -> Task<SettingsMessage> {
        let generation = self.bump_gen(field);
        let f = field.to_string();
        Task::perform(super::widgets::debounce_sleep(0, generation), move |g| {
            SettingsMessage::ConfigFieldSettled {
                field: f,
                value,
                generation: g,
            }
        })
    }

    /// Spawn the async persist for a settled field. The generation captured
    /// at settle time rides along so a stale persist result can be dropped.
    /// Callers pass the generation at which the value settled (or, for a
    /// pending flush, the generation at which the value was queued) — never
    /// the current one, so a flushed value can never carry a newer
    /// generation than the edit it represents.
    /// The persist layer returns the canonical value (trimmed, `None`
    /// collapsed to `""`), which is surfaced through the result's `Ok` so
    /// the display snapshot re-syncs to exactly what was written.
    fn spawn_field_persist(field: String, value: String, generation: u64) -> Task<SettingsMessage> {
        let f_async = field.clone();
        let v_async = value;
        Task::perform(
            async move {
                persist_settled_field(&f_async, &v_async)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |result| SettingsMessage::ConfigFieldSaveResult {
                field,
                generation,
                result,
            },
        )
    }

    /// Re-sync a single persisted column of a routing row into the editable
    /// snapshot.
    ///
    /// `value` is the canonical persisted value for the field's own column;
    /// when `None` (a failed persist rolling back a discrete control) the
    /// column is read back from CONFIG, the source of truth. When the persist
    /// deleted the row, the row is dropped from the snapshot.
    fn sync_field(&mut self, field: &str, value: Option<&str>) {
        if let Some(model) = field.strip_prefix("routing_order:") {
            let order = value.and_then(crate::config::trimmed_or_none).or_else(|| {
                crate::config::CONFIG
                    .model_routing_by_key(model)
                    .and_then(|mr| mr.provider_order)
            });
            ModelRouting::upsert(&mut self.config.model_routings, model, |row| {
                row.provider_order = order;
            });
            self.drop_routing_row_if_cleared(model);
            return;
        }
        if let Some(key) = field.strip_prefix("config:") {
            let value = value.map(str::to_owned).or_else(|| {
                crate::config::CONFIG
                    .snapshot()
                    .get_string_field(key)
                    .map(String::from)
            });
            let _ = self
                .config
                .set_string_field(key, value.as_deref().unwrap_or_default());
        }
    }

    /// Drop a routing row from the editable snapshot when the persist deleted
    /// it from the DB/CONFIG.
    fn drop_routing_row_if_cleared(&mut self, model: &str) {
        if crate::config::CONFIG.model_routing_by_key(model).is_none() {
            self.config
                .model_routings
                .retain(|mr| mr.model != model || mr.provider_order.is_some());
        }
    }

    /// Revert a field's staged snapshot value to the last persisted value
    /// (used when a persist fails and the control is a discrete state —
    /// toggle / picker — rather than free text).
    fn revert_field(&mut self, field: &str) {
        self.sync_field(field, None);
        self.resync_field_editor(field);
    }

    /// Apply a canonical persisted value back into the editable snapshot
    /// (a settled text field may have been replaced by a refresh while the
    /// persist was in flight — the persisted value is the user's last edit
    /// and must win).
    fn apply_persisted_value(&mut self, field: &str, value: &str) {
        self.sync_field(field, Some(value));
        self.resync_field_editor(field);
    }

    /// Whether a failed persist should roll the control back to the last
    /// persisted value. Discrete-state controls (toggles, pick lists, and the
    /// model pickers' optimistic active/list markers) revert; free-text inputs
    /// keep the typed value so the user can correct it.
    ///
    /// `config:provider_endpoint` is treated as discrete because it defines the
    /// chat provider mode (custom vs default): a failed save (e.g. the
    /// toggle-off cascade) must not leave the UI showing a mode the runtime
    /// isn't in — the field reverts and the toggle re-derives from the
    /// persisted value.
    fn field_reverts_on_error(field: &str) -> bool {
        matches!(
            field,
            "config:web_search_provider"
                | "config:image_gen_model"
                | "config:image_gen_models"
                | "config:video_model"
                | "config:video_models"
                | "config:provider_endpoint"
        )
    }

    /// Persist a model-picker's list field immediately (discrete action —
    /// add/remove a model). Clears the picker's inline error and settles now.
    fn persist_picker_list(&mut self, target: ModelPickerTarget) -> Task<SettingsMessage> {
        let (field, value) = match target {
            ModelPickerTarget::ImageGen => (
                "config:image_gen_models",
                self.config.image_gen_models.clone().unwrap_or_default(),
            ),
            ModelPickerTarget::Video => (
                "config:video_models",
                self.config.video_models.clone().unwrap_or_default(),
            ),
        };
        self.field_errors.remove(field);
        self.settle_now(field, value)
    }

    /// Persist a model-picker's active-model field immediately. Clears the
    /// picker's inline error and settles now.
    fn persist_picker_active(&mut self, target: ModelPickerTarget) -> Task<SettingsMessage> {
        let (field, value) = match target {
            ModelPickerTarget::ImageGen => (
                "config:image_gen_model",
                self.config.image_gen_model.clone().unwrap_or_default(),
            ),
            ModelPickerTarget::Video => (
                "config:video_model",
                self.config.video_model.clone().unwrap_or_default(),
            ),
        };
        self.field_errors.remove(field);
        self.settle_now(field, value)
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: SettingsMessage) -> Task<SettingsMessage> {
        match msg {
            // ── Config field edits ─────────────────────────────
            SettingsMessage::ConfigField { key, value } => {
                let _ = self.config.set_string_field(key, &value);
                let field = format!("config:{key}");
                self.field_errors.remove(&field);
                if TEXT_INPUT_KEYS.contains(&key) {
                    // Text input: persist only after the value settles
                    // (debounce), never per keystroke.
                    let generation = self.bump_gen(&field);
                    let f = field.clone();
                    Task::perform(
                        super::widgets::debounce_sleep(SETTLE_MS, generation),
                        move |g| SettingsMessage::ConfigFieldSettled {
                            field: f,
                            value,
                            generation: g,
                        },
                    )
                } else if IMMEDIATE_KEYS.contains(&key) {
                    // Toggles / pick lists are discrete: persist right away.
                    self.settle_now(&field, value)
                } else {
                    // No settle path for this key: the value would stage
                    // forever and never persist — exactly the silent-edit-loss
                    // this design removes. Every rendered config field must be
                    // classified in TEXT_INPUT_KEYS or IMMEDIATE_KEYS.
                    tracing::warn!(
                        key,
                        "settings: unclassified config field staged but not persisted"
                    );
                    Task::none()
                }
            }
            SettingsMessage::ConfigFieldSettleNow { field } => {
                let value = self.staged_value(&field).unwrap_or_default();
                self.settle_now(&field, value)
            }
            SettingsMessage::ConfigFieldSettled {
                field,
                value,
                generation,
            } => {
                // Drop settles superseded by a newer edit for the same field
                // (the user typed/toggled again while the timer was running).
                if generation != self.field_gen.get(&field).copied().unwrap_or(0) {
                    return Task::none();
                }
                // Serialize persists per field: if one is already in flight,
                // queue the newest value instead of spawning a second persist
                // — two concurrent persists for the same field could complete
                // out of order and the older value would land last in the DB
                // while the UI shows the newer one.
                if self.in_flight_persists.contains(&field) {
                    self.pending_persists.insert(field, (value, generation));
                    return Task::none();
                }
                self.in_flight_persists.insert(field.clone());
                Self::spawn_field_persist(field, value, generation)
            }
            SettingsMessage::ConfigFieldSaveResult {
                field,
                generation,
                result,
            } => {
                // A completed persist frees the field. If a newer settle
                // queued a pending value while this persist was in flight,
                // persist it now — the pending value is the user's latest edit
                // and must win regardless of this result's age.
                let pending = if self.in_flight_persists.remove(&field) {
                    self.pending_persists.remove(&field)
                } else {
                    None
                };
                if let Some((pending_value, queued_generation)) = pending {
                    self.in_flight_persists.insert(field.clone());
                    // Flush with the generation at which the value was queued
                    // — NOT the current one: if a newer edit was staged since,
                    // the flushed persist's result is dropped as stale and can
                    // never clobber the newer staged value, while the value
                    // itself still lands in the DB.
                    return Self::spawn_field_persist(field, pending_value, queued_generation);
                }
                // Ignore results from a superseded persist.
                if generation != self.field_gen.get(&field).copied().unwrap_or(0) {
                    return Task::none();
                }
                match result {
                    Ok(outcome) => {
                        self.field_errors.remove(&field);
                        self.apply_persisted_value(&field, &outcome.value);
                        // Only the endpoint persist arm can produce a warning —
                        // a key-field result must never touch the endpoint
                        // warning slot.
                        if field == "config:provider_endpoint" {
                            self.endpoint_warning = outcome.warning;
                        }
                        Task::none()
                    }
                    Err(e) => {
                        self.field_errors.insert(field.clone(), e.clone());
                        if field == "config:provider_endpoint" {
                            self.endpoint_warning = None;
                        }
                        if Self::field_reverts_on_error(&field) {
                            // Discrete-state controls (toggles/pickers) roll
                            // back to the last persisted value; free text
                            // keeps the typed value for correction.
                            self.revert_field(&field);
                            // A failed endpoint save means no configuration change — close
                            // the revealed section so the toggle cannot show ON while the
                            // runtime stays on OpenRouter. The error stays visible in the
                            // bottom banner (the inline row hides with the section) and
                            // re-appears inline if the user reveals the section again.
                            if field == "config:provider_endpoint" {
                                self.custom_revealed = false;
                                self.error = Some(e);
                                // Toggle-off clears both endpoint fields as one discrete
                                // action — a failed endpoint persist must restore the key
                                // field too, so the UI cannot show a half-reverted custom
                                // setup (endpoint defaulted, key cleared).
                                self.revert_field("config:provider_endpoint_key");
                            }
                        }
                        Task::none()
                    }
                }
            }
            SettingsMessage::CustomEndpointToggle(true) => {
                // ON: reveal the custom-endpoint fields. Nothing is persisted
                // here — the endpoint becomes active only when the user
                // settles a genuinely non-default URL
                // ([`Self::custom_endpoint_active_ui`]), so a reveal alone
                // never leaves a spurious persisted row, and the toggle state
                // never diverges from the normalized runtime endpoint.
                self.custom_revealed = true;
                self.field_errors
                    .remove(&format!("config:{CONFIG_KEY_PROVIDER_ENDPOINT}"));
                Task::none()
            }
            SettingsMessage::CustomEndpointToggle(false) => {
                // OFF: revert to OpenRouter and clear both persisted rows. Bump
                // generation for both fields (drops pending settles/results),
                // remove pending persists, clear both from the editable
                // snapshot, then settle the endpoint field with `""` — the
                // persist path cascades the provider_endpoint_key row deletion.
                self.custom_revealed = false;
                let endpoint_field = format!("config:{CONFIG_KEY_PROVIDER_ENDPOINT}");
                let key_field = format!("config:{CONFIG_KEY_PROVIDER_ENDPOINT_KEY}");
                self.bump_gen(&endpoint_field);
                self.bump_gen(&key_field);
                self.pending_persists.remove(&endpoint_field);
                self.pending_persists.remove(&key_field);
                self.config.provider_endpoint = None;
                self.config.provider_endpoint_key = None;
                self.field_errors.remove(&endpoint_field);
                self.field_errors.remove(&key_field);
                self.endpoint_warning = None;
                self.resync_field_editor(&endpoint_field);
                self.resync_field_editor(&key_field);
                self.settle_now(&endpoint_field, String::new())
            }
            SettingsMessage::ModelRoutingOrder { model, action } => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                let field = format!("routing_order:{model}");
                self.field_errors.remove(&field);
                let submit = matches!(action, EditorAction::Submit);
                let changes_text = action.changes_text();
                let order = {
                    let initial = self.staged_value(&field).unwrap_or_default();
                    let editor = self.field_editor_mut(&field, &initial);
                    editor.apply_action(action);
                    editor.text()
                };
                if changes_text || submit {
                    let order_opt = Some(order.clone()).filter(|s| !s.is_empty());
                    ModelRouting::upsert(&mut self.config.model_routings, model, |mr| {
                        mr.provider_order = order_opt;
                    });
                }
                if submit {
                    self.settle_now(&field, order)
                } else if changes_text {
                    let generation = self.bump_gen(&field);
                    let f = field.clone();
                    Task::perform(
                        super::widgets::debounce_sleep(SETTLE_MS, generation),
                        move |g| SettingsMessage::ConfigFieldSettled {
                            field: f,
                            value: order,
                            generation: g,
                        },
                    )
                } else {
                    Task::none()
                }
            }
            SettingsMessage::ConfigFieldAction { key, action }
            | SettingsMessage::PasswordFieldAction { key, action } => {
                self.handle_field_action(key, action)
            }
            SettingsMessage::TogglePasswordVisibility(target) => {
                if self.password_visible.contains(&target) {
                    self.password_visible.remove(&target);
                } else {
                    self.password_visible.insert(target);
                }
                Task::none()
            }

            // ── Transcription ───────────────────────────────────
            SettingsMessage::TranscriptionToggle(enabled) => {
                self.transcription_toggle_gen += 1;
                let generation = self.transcription_toggle_gen;
                let voice_was_enabled = self.config.voice_enabled.as_deref() == Some("true");
                // Mirror into both snapshots (enabled → "" = absent row;
                // disabled → "false"), matching the persisted row semantics.
                let transcription_value = if enabled { "" } else { "false" };
                let _ = self.config.set_string_field(
                    CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
                    transcription_value,
                );
                let _ = crate::config::CONFIG.set_string_field(
                    CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
                    transcription_value,
                );
                // Turning transcription OFF also turns Wake Word Detection OFF
                // (shared ASR model): stop the pipeline now.
                if !enabled && voice_was_enabled {
                    let _ = self.config.set_string_field(CONFIG_KEY_VOICE_ENABLED, "");
                    let _ = crate::config::CONFIG.set_string_field(CONFIG_KEY_VOICE_ENABLED, "");
                    sync_voice_state(false);
                }
                // Toggle ON: kick the model load/download in the background —
                // load from cache first, otherwise download with retries. A
                // terminal failure is recovered via retry_init (same path as
                // the inline Retry button); any other state falls through to
                // spawn_background_init, which claims Uninit→Loading with the
                // same panic→Failed guard the boot path uses (a raw spawn
                // here could strand the transcriber in Loading on a panic).
                if enabled && !crate::audio::local_transcriber::is_loaded() {
                    if !crate::audio::local_transcriber::retry_init() {
                        crate::audio::local_transcriber::spawn_background_init();
                    }
                }
                Task::perform(
                    async move {
                        let store = crate::config_db::store();
                        store
                            .set_transcription_toggle(enabled, !enabled && voice_was_enabled)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| SettingsMessage::TranscriptionToggleResult {
                        generation,
                        voice_was_enabled,
                        result,
                    },
                )
            }
            SettingsMessage::TranscriptionToggleResult {
                generation,
                voice_was_enabled,
                result,
            } => {
                // Ignore results from a superseded toggle (rapid re-toggle).
                if generation != self.transcription_toggle_gen {
                    return Task::none();
                }
                match result {
                    Ok(()) => Task::none(),
                    Err(e) => {
                        self.error = Some(e);
                        // Revert the transcription snapshot to the opposite
                        // staged state (binary toggle).
                        let currently_off =
                            self.config.audio_transcription_use_local.as_deref() == Some("false");
                        let restore = if currently_off { "" } else { "false" };
                        let _ = self
                            .config
                            .set_string_field(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, restore);
                        let _ = crate::config::CONFIG
                            .set_string_field(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, restore);
                        // Restore the wake-word state the cascade disabled.
                        if voice_was_enabled {
                            let _ = self
                                .config
                                .set_string_field(CONFIG_KEY_VOICE_ENABLED, "true");
                            let _ = crate::config::CONFIG
                                .set_string_field(CONFIG_KEY_VOICE_ENABLED, "true");
                            sync_voice_state(true);
                        }
                        Task::none()
                    }
                }
            }
            SettingsMessage::RetryTranscription => {
                let _ = crate::audio::local_transcriber::retry_init();
                Task::none()
            }

            // ── Voice assistant ─────────────────────────────────
            SettingsMessage::VoiceToggle(enabled) => self.run_toggle(
                CONFIG_KEY_VOICE_ENABLED,
                enabled,
                |s| {
                    s.voice_toggle_gen += 1;
                    s.voice_toggle_gen
                },
                SettingsMessage::VoiceToggleResult,
                sync_voice_state,
            ),
            SettingsMessage::VoiceToggleResult(g, result) => self.handle_toggle_result(
                CONFIG_KEY_VOICE_ENABLED,
                g,
                result,
                |s| s.voice_toggle_gen,
                |c| &c.voice_enabled,
                sync_voice_state,
            ),
            SettingsMessage::TtsToggle(enabled) => self.run_toggle(
                CONFIG_KEY_TTS_ENABLED,
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
                CONFIG_KEY_TTS_ENABLED,
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
                let phrase = self.wake_word_phrase_input.text();
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StartEnrollment(phrase),
                );
                Task::none()
            }
            SettingsMessage::WakeWordPhraseInput(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.wake_word_phrase_input.apply_action(action);
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
            SettingsMessage::AddWorkspaceName(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.add_workspace_name.apply_action(action);
                Task::none()
            }
            SettingsMessage::AddWorkspacePath(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.add_workspace_path.apply_action(action);
                Task::none()
            }
            SettingsMessage::SubmitAddWorkspace => {
                if self.add_workspace_name.text().is_empty()
                    || self.add_workspace_path.text().is_empty()
                {
                    return Task::none();
                }
                self.add_workspace_adding = true;
                let name = self.add_workspace_name.text();
                let path = self.add_workspace_path.text();
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
                self.workspaces_state
                    .refresh()
                    .map(SettingsMessage::WorkspaceMsg)
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
                    // Fresh default-agent selection.
                    self.add_user_default = 0;
                } else {
                    self.close_add_user_modal();
                }
                Task::none()
            }
            SettingsMessage::AddUserSender(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.add_user_sender.apply_action(action);
                Task::none()
            }
            SettingsMessage::AddUserDefaultRole(idx) => {
                if idx < [Role::Assistant, Role::Artist].len() {
                    self.add_user_default = idx;
                }
                Task::none()
            }
            SettingsMessage::SubmitAddUser => {
                if self.add_user_sender.text().is_empty() {
                    return Task::none();
                }
                // The permission-derived role pool no longer stores per-user
                // roles; the manual Settings bypass picks a single default agent
                // from the hard-coded {Assistant, Artist} pool.
                let default_role = [Role::Assistant, Role::Artist][self.add_user_default];
                self.add_user_adding = true;
                let sender = self.add_user_sender.text();
                Task::perform(
                    async move {
                        let store = users::user_store()?;
                        // Reject a duplicate name here too: the Settings bypass has no
                        // way to complete a leftover unbound row, so a duplicate would
                        // silently no-op (INSERT OR IGNORE) while reporting success.
                        if store
                            .user_exists(&sender)
                            .await
                            .map_err(|e| e.to_string())?
                        {
                            return Err(format!("A user named '{sender}' already exists"));
                        }
                        store
                            .add_user(&sender, None, default_role)
                            .await
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    },
                    SettingsMessage::AddUserResult,
                )
            }
            SettingsMessage::AddUserResult(Ok(())) => {
                self.close_add_user_modal();
                self.users_state.refresh().map(SettingsMessage::UserMsg)
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
                    (t, ModelPickerAction::AddInput(action)) => {
                        if let Some(task) = super::common::focus_navigation_task(&action) {
                            return task;
                        }
                        self.model_picker_inputs[t.idx()].apply_action(action);
                        Task::none()
                    }
                    (t, ModelPickerAction::AddModel) => match t {
                        // Image-gen additions are validated against the catalog
                        // (fail-open when it is unavailable) before the list is
                        // mutated; the input buffer is kept on rejection so the
                        // user can correct it. Image models always run on
                        // OpenRouter, so validation always runs
                        // against the default endpoint — a custom chat endpoint
                        // must not gate media model additions.
                        ModelPickerTarget::ImageGen => {
                            let model = self.model_picker_inputs[t.idx()].text().trim().to_string();
                            if model.is_empty() {
                                return Task::none();
                            }
                            let endpoint = crate::config::DEFAULT_PROVIDER_ENDPOINT.to_string();
                            Task::perform(
                                async move {
                                    let ok = crate::tools::media_catalog::image::
                                        validate_image_model_for_endpoint(&model, &endpoint)
                                        .await
                                        .map_err(|e| e.to_string());
                                    (model, ok)
                                },
                                |(model, ok)| SettingsMessage::ModelPickerAddResult {
                                    target: ModelPickerTarget::ImageGen,
                                    model,
                                    ok,
                                },
                            )
                        }
                        ModelPickerTarget::Video => {
                            let (models, _active) = picker_config_fields(&t, &mut self.config);
                            add_model_to_list(&mut self.model_picker_inputs[t.idx()], models);
                            self.persist_picker_list(t)
                        }
                    },
                    (t, ModelPickerAction::RemoveModel(model)) => {
                        let (models, active) = picker_config_fields(&t, &mut self.config);
                        remove_model_from_list(&model, models, active);
                        // The active model may have been reset by the removal —
                        // persist both the list and the active model.
                        Task::batch([self.persist_picker_list(t), self.persist_picker_active(t)])
                    }
                    (t, ModelPickerAction::SetActive(model)) => {
                        let (_models, active) = picker_config_fields(&t, &mut self.config);
                        *active = Some(model.clone());
                        // Persist immediately. For the image target the persist
                        // validates the model against the endpoint-keyed catalog
                        // and fails without writing on rejection (the optimistic
                        // active marker is then reverted inline).
                        let key = match t {
                            ModelPickerTarget::ImageGen => "config:image_gen_model",
                            ModelPickerTarget::Video => "config:video_model",
                        };
                        let field = key.to_string();
                        self.field_errors.remove(&field);
                        self.settle_now(&field, model)
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
                    if self.model_picker_inputs[target.idx()].text().trim() == model {
                        self.model_picker_inputs[target.idx()].clear();
                    }
                    // Persist silently — no success toast.
                    self.persist_picker_list(target)
                }
                Err(e) => {
                    // Inline error on the picker — no toast (silent success /
                    // inline rejection for config validation).
                    let key = match target {
                        ModelPickerTarget::ImageGen => "config:image_gen_models",
                        ModelPickerTarget::Video => "config:video_models",
                    };
                    self.field_errors
                        .insert(key.to_string(), format!("Model `{model}` rejected: {e}"));
                    Task::none()
                }
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
            self.integrations_section(),
            Space::new().height(16),
            self.audio_section(),
            Space::new().height(16),
            self.models_section(),
            Space::new().height(16),
            self.generation_section(),
            Space::new().height(16),
            self.routing_section(),
            Space::new().height(16),
            Self::about_section(),
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

        // Modal overlay (rendered above everything else)
        let modal = self.render_modal_overlay();

        // Stack order: [scroll content, modal overlay] — every change on the
        // page persists automatically, so there is no floating Save button.
        let body = stack([scroll.into(), modal]);

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
    #[expect(clippy::too_many_lines)]
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
            for ws_item in &ws.workspaces {
                let (status_color, status_bg) = theme::workspace_status_color(ws_item.status);
                let maintainer_on = ws_item.maintenance_enabled;

                let ws_row = container(
                    column![
                        row![
                            // Name column (FillPortion: 15)
                            container(text(&ws_item.name).size(14).color(theme::TEXT_PRIMARY))
                                .width(Length::FillPortion(15))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center),
                            // Status column (FillPortion: 10)
                            container(widgets::badge_pill(
                                ws_item.status.to_string(),
                                (status_bg, status_color),
                                11,
                                [2, 8],
                            ))
                            .width(Length::FillPortion(10))
                            .align_x(Alignment::Start)
                            .align_y(Alignment::Center),
                            // Path column (FillPortion: 35)
                            container(text(&ws_item.path).size(12).color(theme::TEXT_MUTED))
                                .width(Length::FillPortion(35))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center),
                            // Left cluster column (FillPortion: 28) — per-role
                            // context icons, general context, Diag, Notes.
                            {
                                let mut left = Row::new().spacing(4).align_y(Alignment::Center);
                                let mut role_btns = Row::new().spacing(2);
                                for role in Role::iter()
                                    .filter(|r| crate::agent::role::role_info(r).has_discovery)
                                {
                                    let name = role.as_str();
                                    let (color, _bg) = theme::role_badge_color_for(&role);
                                    role_btns = role_btns.push(
                                        button(theme::role_icon(&role).size(11).color(color))
                                            .style(theme::button_text)
                                            // Halve the default 10px horizontal
                                            // padding so adjacent role icons sit
                                            // closer; vertical stays at 5px.
                                            .padding([5.0, 5.0])
                                            .on_press(SettingsMessage::WorkspaceMsg(
                                                workspaces::WorkspacesMessage::ViewContext(
                                                    ws_item.name.clone(),
                                                    name.to_string(),
                                                ),
                                            )),
                                    );
                                }
                                left = left.push(role_btns);
                                left = left.push(
                                    button(
                                        theme::general_context_icon().size(11).color(theme::ACCENT),
                                    )
                                    .style(theme::button_text)
                                    .on_press(
                                        SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::ViewGeneralContext(
                                                ws_item.name.clone(),
                                            ),
                                        ),
                                    ),
                                );
                                left = left.push(
                                    button(text("Diag").size(11).color(theme::TEXT_MUTED))
                                        .style(theme::button_text)
                                        .on_press(SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::ShowDiagnostics(
                                                ws_item.name.clone(),
                                            ),
                                        )),
                                );
                                left = left.push(
                                    button(text("Notes").size(11).color(theme::TEXT_MUTED))
                                        .style(theme::button_text)
                                        .on_press(SettingsMessage::WorkspaceMsg(
                                            workspaces::WorkspacesMessage::ToggleNotes(
                                                ws_item.name.clone(),
                                            ),
                                        )),
                                );
                                container(left)
                                    .width(Length::FillPortion(28))
                                    .align_x(Alignment::Start)
                                    .align_y(Alignment::Center)
                            },
                            // Right column (FillPortion: 12) — Maintainer toggle only.
                            container(widgets::icon_tooltip_button(
                                widgets::maint_badge(maintainer_on),
                                if maintainer_on {
                                    "stop maintenance"
                                } else {
                                    "start maintenance"
                                },
                                Some(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::ToggleMaintainer(
                                        ws_item.name.clone(),
                                        !maintainer_on,
                                    ),
                                )),
                                button::DEFAULT_PADDING,
                                theme::button_text,
                                tooltip::Position::Top,
                            ),)
                            .width(Length::FillPortion(12))
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

                // Right-click context menu (Re-analyze / Delete). The card's
                // own buttons still work: ContextMenu forwards all events to
                // the underlay first; only right-clicks open the menu.
                let ws_row: Element<'_, SettingsMessage> = ContextMenu::new(
                    ws_row,
                    vec![
                        MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::refresh_cw,
                            "Re-analyze".into(),
                            SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::Reanalyze(ws_item.name.clone()),
                            ),
                        ),
                        MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::trash,
                            "Delete".into(),
                            SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::DeleteWorkspace(
                                    ws_item.name.clone(),
                                ),
                            ),
                        ),
                    ],
                )
                .into();
                rows = rows.push(ws_row);

                // Inline two-step delete confirmation — armed by the context menu's
                // "Delete" item (the menu closes after firing); lives below the card,
                // reusing the delete_confirm_button machinery.
                if ws.delete_target.as_ref() == Some(&ws_item.name) {
                    let confirm = delete_confirm_button(
                        SettingsMessage::WorkspaceMsg(
                            workspaces::WorkspacesMessage::ConfirmDelete(ws_item.name.clone()),
                        ),
                        SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::CancelDelete),
                    );
                    rows = rows.push(
                        container(confirm)
                            .padding([4, 8])
                            .style(theme::pill_style(theme::BG_ELEVATED)),
                    );
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

                    let editor_widget = super::editor_widget::EditorWidget::new(content)
                        .show_gutter(false)
                        .code_mode(false)
                        .enter(super::editor_widget::EnterBehavior::Newline)
                        .id(iced::widget::Id::from(format!("workspace_notes:{}", ws_item.name)))
                        .placeholder(
                            format!("Add manual context notes for all agents… (max {MAX_WORKSPACE_NOTES_CHARS} characters)"),
                        )
                        .min_height(100.0)
                        .max_height(300.0)
                        .padding(5.0);
                    let editor: Element<'_, SettingsMessage> =
                        container(iced::Element::new(editor_widget).map(move |action| {
                            SettingsMessage::WorkspaceMsg(
                                workspaces::WorkspacesMessage::NotesEdited(
                                    ws_item.name.clone(),
                                    action,
                                ),
                            )
                        }))
                        .width(Length::Fill)
                        .height(Length::Shrink)
                        .style(|_theme| {
                            theme::container_style(theme::BG_ELEVATED, 8.0, 1.0, theme::ACCENT)
                        })
                        .into();

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
        if let Some((ref _ws_name, ref kind, ref md_items_opt)) = ws.context_view {
            section_content = section_content.push(Space::new().height(16));

            let title = match kind {
                workspaces::ContextKind::Role(role) => format!("Context for {role}"),
                workspaces::ContextKind::General => "General context".to_string(),
            };

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
    #[expect(clippy::too_many_lines)]
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
                    container(widgets::icon_tooltip_button(
                        lucide::log_in::<iced::Theme, iced::Renderer>()
                            .size(18)
                            .color(theme::TEXT_MUTED),
                        "Switch active user",
                        Some(SettingsMessage::UserMsg(users::UsersMessage::SwitchUser(
                            user.name.clone(),
                        ))),
                        0,
                        theme::button_text,
                        tooltip::Position::Top,
                    ))
                    .width(Length::Fixed(28.0))
                    .align_x(iced::alignment::Horizontal::Center)
                    .into()
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
                                    tooltip(
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
                                        text("Active workspace").size(11),
                                        tooltip::Position::Top,
                                    )
                                    .style(theme::tooltip_style),
                                )
                                .width(Length::FillPortion(20))
                                .align_x(Alignment::Start)
                                .align_y(Alignment::Center)
                            },
                            // Role column (FillPortion: 15) — active role
                            // picker (permission-derived pool)
                            {
                                let role_picker: Element<'_, SettingsMessage> = match us
                                    .active_role_options
                                    .get(&user.name)
                                {
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
                                                // mirror resolve_active_role's default
                                                // (the first pool role).
                                                options.first().cloned()
                                            });
                                        tooltip(
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
                                            .width(Length::Fixed(150.0)),
                                            text("Active role").size(11),
                                            tooltip::Position::Top,
                                        )
                                        .style(theme::tooltip_style)
                                        .into()
                                    }
                                    _ => text("none").size(12).color(theme::TEXT_MUTED).into(),
                                };
                                container(role_picker)
                                    .width(Length::FillPortion(15))
                                    .align_x(Alignment::Start)
                                    .align_y(Alignment::Center)
                            },
                            // Actions column (FillPortion: 12) — switch icon
                            container({
                                let mut actions = Row::new().align_y(Alignment::Center);
                                actions = actions.push(switch_icon);
                                actions
                            })
                            .width(Length::FillPortion(12))
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
                                    widgets::single_line_editor(
                                        &us.bind_input.buffer,
                                        "@username",
                                        false,
                                        Length::Fixed(270.0),
                                        Some(Id::from(format!("bind_input:{}", user.name))),
                                        |action| {
                                            SettingsMessage::UserMsg(
                                                users::UsersMessage::BindInputChanged(action),
                                            )
                                        },
                                    ),
                                    Space::new().width(8).into(),
                                ];
                                row_elements.push(
                                    button(
                                        text(if us.binding { "Binding..." } else { "Bind" })
                                            .size(11),
                                    )
                                    .style(theme::button_primary)
                                    .on_press_maybe(if us.bind_input.text().trim().is_empty() {
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
                                        widgets::icon_tooltip_button(
                                            lucide::x::<iced::Theme, iced::Renderer>()
                                                .size(11)
                                                .color(theme::TEXT_MUTED),
                                            "Unlink Telegram",
                                            Some(SettingsMessage::UserMsg(
                                                users::UsersMessage::UnbindChannel(
                                                    user.name.clone(),
                                                    binding.identifier.clone(),
                                                ),
                                            )),
                                            button::DEFAULT_PADDING,
                                            theme::button_text,
                                            tooltip::Position::Top,
                                        )
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

                // Right-click context menu (Delete). The card's own controls
                // still work: ContextMenu forwards all events to the underlay
                // first; only right-clicks open the menu. Admins are exempt —
                // no context menu at all (an empty menu would render a hollow
                // box).
                let user_row: Element<'_, SettingsMessage> = if is_admin {
                    user_row.into()
                } else {
                    ContextMenu::new(
                        user_row,
                        vec![MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::trash,
                            "Delete".into(),
                            SettingsMessage::UserMsg(users::UsersMessage::DeleteUser(
                                user.name.clone(),
                            )),
                        )],
                    )
                    .into()
                };
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
            widgets::modal_backdrop(dialog, SettingsMessage::ToggleAddWorkspaceModal, 0.5)
        } else if self.show_add_user_modal {
            let dialog = self.add_user_dialog();
            widgets::modal_backdrop(dialog, SettingsMessage::ToggleAddUserModal, 0.5)
        } else if let Some(ref del_user) = self.users_state.delete_target {
            let dialog = Self::user_delete_dialog(del_user);
            widgets::modal_backdrop(
                dialog,
                SettingsMessage::UserMsg(users::UsersMessage::CancelDelete),
                0.5,
            )
        } else if let Some(ref diag_ws_name) = self.workspaces_state.diagnostics_modal {
            let dialog = self.diagnostics_dialog(diag_ws_name);
            widgets::modal_backdrop(
                dialog,
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Escape),
                0.5,
            )
        } else {
            // Keep Stack widget type stable
            iced::widget::stack([widgets::empty_stack_placeholder()]).into()
        }
    }

    /// Build the user-deletion confirmation modal.
    ///
    /// The text truthfully states what [`users::UserStore::delete_user`]
    /// does — removes the user row and all channel bindings (access cut
    /// immediately) — and what it preserves (sessions, chat history,
    /// userspace files).
    fn user_delete_dialog(user_name: &str) -> Element<'_, SettingsMessage> {
        container(
            column![
                text(format!("Delete user {user_name}?"))
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().height(12),
                text(
                    "This permanently removes the user and all their \
                     channel bindings (including Telegram) — their access is \
                     cut immediately. Their sessions, chat history, and \
                     userspace files are preserved.",
                )
                .size(13)
                .color(theme::TEXT_SECONDARY),
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Keep user").size(13))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::UserMsg(users::UsersMessage::CancelDelete)),
                    Space::new().width(8),
                    button(text("Delete").size(13))
                        .style(theme::button_danger)
                        .on_press(SettingsMessage::UserMsg(
                            users::UsersMessage::ConfirmDelete(user_name.to_string()),
                        )),
                ]
                .align_y(Alignment::Center),
            ]
            .width(Length::Fill),
        )
        .width(Length::Fixed(460.0))
        .padding(24)
        .style(theme::dialog_container_style)
        .into()
    }

    /// Build the add-workspace modal dialog content.
    fn add_workspace_dialog(&self) -> Element<'_, SettingsMessage> {
        modal_dialog(
            "Add Workspace".to_string(),
            &[
                DialogField {
                    label: "Name",
                    placeholder: "workspace name",
                    value: &self.add_workspace_name.buffer,
                    id: "add_workspace_name",
                    on_input: SettingsMessage::AddWorkspaceName,
                },
                DialogField {
                    label: "Path",
                    placeholder: "/path/to/workspace",
                    value: &self.add_workspace_path.buffer,
                    id: "add_workspace_path",
                    on_input: SettingsMessage::AddWorkspacePath,
                },
            ],
            None,
            "Add",
            self.add_workspace_adding,
            !self.add_workspace_name.text().is_empty()
                && !self.add_workspace_path.text().is_empty(),
            SettingsMessage::ToggleAddWorkspaceModal,
            SettingsMessage::SubmitAddWorkspace,
        )
    }

    /// Build the add-user modal dialog content.
    fn add_user_dialog(&self) -> Element<'_, SettingsMessage> {
        modal_dialog(
            "Add User".to_string(),
            &[DialogField {
                label: "Name",
                placeholder: "user name",
                value: &self.add_user_sender.buffer,
                id: "add_user_sender",
                on_input: SettingsMessage::AddUserSender,
            }],
            Some(
                column![
                    text("Default agent").size(12).color(theme::TEXT_MUTED),
                    pick_list(
                        vec![Role::Assistant, Role::Artist],
                        Some([Role::Assistant, Role::Artist][self.add_user_default]),
                        |r| match r {
                            Role::Artist => SettingsMessage::AddUserDefaultRole(1),
                            _ => SettingsMessage::AddUserDefaultRole(0),
                        },
                    )
                    .style(super::widgets::pick_list_style)
                    .padding([4, 8]),
                    Space::new().height(8),
                ]
                .into(),
            ),
            "Add",
            self.add_user_adding,
            !self.add_user_sender.text().is_empty(),
            SettingsMessage::ToggleAddUserModal,
            SettingsMessage::SubmitAddUser,
        )
    }

    /// Build the diagnostics modal dialog content for the given workspace.
    fn diagnostics_dialog(&self, diag_ws_name: &str) -> Element<'_, SettingsMessage> {
        let ws_name = diag_ws_name.to_string();
        let ws_state = &self.workspaces_state;

        let is_busy = ws_state.diagnostics_busy;
        let error = ws_state.diagnostics_error.as_deref();

        // Get the edit buffers — if modal is open they should exist (they are
        // inserted by ShowDiagnostics before the modal is flagged open).
        let buffers: &[SingleLineEditorState; crate::DiagnosticsCommands::COMMAND_COUNT] = ws_state
            .diagnostics_edit_buffers
            .get(&ws_name)
            .expect("diagnostics edit buffers inserted when the modal opens");

        // Use static labels from DiagnosticsCommands to avoid duplicating
        // the label-to-field mapping in two places.
        let labels = &crate::DiagnosticsCommands::COMMAND_LABELS;

        let mut rows_col = Column::new().spacing(8);

        // Error banner
        if let Some(err) = error {
            rows_col = rows_col.push(widgets::error_banner(err));
        }

        for (i, label) in labels.iter().enumerate() {
            rows_col = rows_col.push(
                row![
                    text(*label)
                        .size(12)
                        .color(theme::TEXT_MUTED)
                        .width(Length::Fixed(120.0))
                        .align_y(Alignment::Center),
                    widgets::single_line_editor(
                        &buffers[i].buffer,
                        "(skipped)",
                        false,
                        Length::Fill,
                        Some(Id::from(format!("diagnostics:{ws_name}:{i}"))),
                        {
                            let name = ws_name.clone();
                            move |action| {
                                SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::DiagnosticsFieldEdited(
                                        name.clone(),
                                        i,
                                        action,
                                    ),
                                )
                            }
                        },
                    ),
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
    // ── Config-field builders ────────────────────────────────────
    //
    // Collapse the repeated ~20-line block (a `field_row`/`field_row_with_error`
    // wrapping a shared single-line editor that emits `ConfigFieldAction` /
    // `PasswordFieldAction` on input, styled + fixed-width, with the
    // `field_errors` lookup) into a single helper per input kind. Behavior-
    // preserving: the field id is always `config:<key>` derived from the
    // `CONFIG_KEY_*` const, which is `stringify!` of the snake_case field name —
    // no per-call literal ids.

    /// A single inline-editable text config field row (with error placement).
    fn config_text_field<'a>(
        &'a self,
        label: &'static str,
        placeholder: &'static str,
        key: &'static str,
        hint: Option<&'static str>,
    ) -> Element<'a, SettingsMessage> {
        let field = format!("config:{key}");
        let error = self.field_errors.get(&field).map(String::as_str);
        field_row_with_error(
            label,
            widgets::single_line_editor(
                &self.field_editor(&field).buffer,
                placeholder,
                true,
                Length::Fixed(375.0),
                Some(Id::from(field.clone())),
                move |action| SettingsMessage::ConfigFieldAction { key, action },
            ),
            hint,
            error,
        )
    }

    /// A single maskable password config field row (with password highlighting).
    fn config_password_field<'a>(
        &'a self,
        label: &'static str,
        target: PasswordTarget,
        placeholder: &'static str,
        key: &'static str,
        hint: Option<&'static str>,
        highlight: bool,
    ) -> Element<'a, SettingsMessage> {
        let field = format!("config:{key}");
        let error = self.field_errors.get(&field).map(String::as_str);
        field_row_with_error(
            label,
            widgets::password_field_editor(
                &self.field_editor(&field).buffer,
                placeholder,
                self.password_visible.contains(&target),
                Length::Fixed(375.0),
                highlight,
                Some(Id::from(field.clone())),
                move |action| SettingsMessage::PasswordFieldAction { key, action },
                SettingsMessage::TogglePasswordVisibility(target),
            ),
            hint,
            error,
        )
    }

    // ── Section helpers ──────────────────────────────────────────

    fn provider_section(&self) -> Element<'_, SettingsMessage> {
        // The API key field is highlighted until a valid (trimmed non-empty)
        // key is present. Computed per-render from the editable snapshot, so
        // the highlight clears on the first typed character and re-arms when
        // the field is cleared to empty/whitespace. With a genuinely custom
        // endpoint active the OpenRouter key is only needed for
        // media, so a keyless custom endpoint user's key field is not
        // highlighted.
        let custom_active = self.custom_endpoint_active_ui();
        let api_key_unset = !custom_active
            && self
                .config
                .provider_key
                .as_deref()
                .is_none_or(|key| key.trim().is_empty());
        // The custom-endpoint section is open when a custom endpoint is active
        // OR the user revealed it this session. The toggle reflects this —
        // OFF means default mode with the section closed, so the toggle state
        // tracks the section's visibility and never implies a custom endpoint
        // is configured when only the normalized predicate says otherwise.
        let custom_section_open = custom_active || self.custom_revealed;

        let mut rows: Vec<Element<'_, SettingsMessage>> = vec![
            self.config_password_field(
                "OpenRouter key",
                PasswordTarget::ProviderKey,
                "sk-or-v1-...",
                CONFIG_KEY_PROVIDER_KEY,
                None,
                api_key_unset,
            ),
            field_row(
                "Custom endpoint",
                toggler(custom_section_open)
                    .on_toggle(SettingsMessage::CustomEndpointToggle)
                    .into(),
                Some("custom chat-completions server (e.g. llama.cpp or vLLM)"),
            ),
        ];

        if custom_section_open {
            // Endpoint URL — free text, settled on Enter / debounce. An
            // unreachable endpoint still saves (with an inline warning).
            let mut endpoint_row = self.config_text_field(
                "Endpoint URL",
                "https://openrouter.ai/api/v1",
                CONFIG_KEY_PROVIDER_ENDPOINT,
                None,
            );
            if let Some(w) = self.endpoint_warning.as_ref() {
                endpoint_row = column![endpoint_row, inline_warning(w, 188.0)]
                    .spacing(2)
                    .into();
            }
            rows.push(endpoint_row);

            // Endpoint key (optional) — only ever sent to the custom endpoint.
            rows.push(self.config_password_field(
                "Endpoint key (optional)",
                PasswordTarget::EndpointKey,
                "Leave empty for keyless servers",
                CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
                None,
                false,
            ));
        }

        section("Provider", Column::with_children(rows).spacing(4))
    }

    fn models_section(&self) -> Element<'_, SettingsMessage> {
        let manager_row = self.config_text_field(
            "Manager",
            crate::config::DEFAULT_MANAGER_MODEL,
            CONFIG_KEY_MANAGER_MODEL,
            Some("Manager, Assistant, Discovery, Engineer, Support"),
        );
        let worker_row = self.config_text_field(
            "Worker",
            crate::config::DEFAULT_WORKER_MODEL,
            CONFIG_KEY_WORKER_MODEL,
            Some("Artist, Analyst, Coder, QA, Reviewer, Maintainer, Sanitation"),
        );
        let video_transcription_row = self.config_text_field(
            "Video Transcription",
            crate::config::DEFAULT_VIDEO_TRANSCRIPTION_MODEL,
            CONFIG_KEY_VIDEO_TRANSCRIPTION_MODEL,
            None,
        );
        section(
            "Models",
            column![manager_row, worker_row, video_transcription_row].spacing(4),
        )
    }

    // ── Audio section ──────────────────────────
    //
    // One 'Audio' section with exactly three rows — Transcription, Wake Word
    // Detection, Text to Speech — each with the toggle and an inline status in
    // the same row. The wake-word enrollment UI (phrase input, Enroll button,
    // enrolled-phrase display, multi-line progress/Cancel) sits below the
    // three rows, unchanged.

    #[expect(clippy::too_many_lines)]
    fn audio_section(&self) -> Element<'_, SettingsMessage> {
        use iced::widget::Text;

        let transcription_enabled =
            self.config.audio_transcription_use_local.as_deref() != Some("false");
        let voice_enabled = self.config.voice_enabled.as_deref() == Some("true");
        let status = crate::audio::voice::get_status();
        let has_enrollment = crate::audio::voice::get_enrollment().is_some();
        let is_enrolling = matches!(
            status,
            crate::audio::voice::VoiceStatus::Enrolling { .. }
                | crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::EnrollingNegatives { .. }
        );

        // ── Row 1: Transcription (toggle + inline status + Retry) ──
        let transcription_status: Element<'_, SettingsMessage> = if !transcription_enabled {
            Text::new("Disabled")
                .size(13)
                .color(theme::TEXT_SECONDARY)
                .into()
        } else if crate::audio::local_transcriber::is_loaded() {
            Text::new("Ready").size(13).into()
        } else if crate::audio::local_transcriber::is_failed() {
            let retry_btn = button(Text::new("   Retry   ").size(13))
                .on_press(SettingsMessage::RetryTranscription)
                .style(theme::button_danger)
                .padding(4);
            row![
                Text::new("Download failed").size(14),
                Space::new().width(8),
                retry_btn,
            ]
            .align_y(Alignment::Center)
            .into()
        } else {
            Text::new("Downloading…").size(13).into()
        };
        let transcription_row = field_row(
            "Transcription",
            row![
                toggler(transcription_enabled).on_toggle(SettingsMessage::TranscriptionToggle),
                Space::new().width(12),
                transcription_status,
            ]
            .align_y(Alignment::Center)
            .into(),
            Some("using local CPU-optimized Qwen3-ASR"),
        );

        // ── Row 2: Wake Word Detection (toggle gated on Transcription + live status) ──
        // Live status: re-rendered every second by the dashboard tick, so the
        // pipeline state (listening / enrolling / model error …) is current.
        let wake_status: Element<'_, SettingsMessage> = match status.clone() {
            crate::audio::voice::VoiceStatus::Disabled => Text::new("Disabled")
                .size(13)
                .color(theme::TEXT_SECONDARY)
                .into(),
            crate::audio::voice::VoiceStatus::LoadingModels => {
                Text::new("Loading models…").size(13).into()
            }
            crate::audio::voice::VoiceStatus::ModelError => {
                let retry_btn = button(Text::new("   Retry   ").size(13))
                    .on_press(SettingsMessage::RetryVoiceModels)
                    .style(theme::button_danger)
                    .padding(4);
                row![
                    Text::new("Model error").size(14),
                    Space::new().width(8),
                    retry_btn,
                ]
                .align_y(Alignment::Center)
                .into()
            }
            crate::audio::voice::VoiceStatus::Listening => {
                Text::new("Listening for wake word").size(13).into()
            }
            crate::audio::voice::VoiceStatus::Recording => {
                Text::new("Recording command").size(13).into()
            }
            crate::audio::voice::VoiceStatus::RecordingManual => {
                Text::new("Recording voice message").size(13).into()
            }
            crate::audio::voice::VoiceStatus::Transcribing => {
                Text::new("Transcribing…").size(13).into()
            }
            crate::audio::voice::VoiceStatus::MicPermissionDenied => {
                Text::new("Microphone permission denied").size(13).into()
            }
            crate::audio::voice::VoiceStatus::MicDisconnected => {
                Text::new("Microphone disconnected").size(13).into()
            }
            crate::audio::voice::VoiceStatus::Enrolling { .. }
            | crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. }
            | crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
            | crate::audio::voice::VoiceStatus::EnrollingNegatives { .. } => {
                Text::new("Enrolling…").size(13).into()
            }
            crate::audio::voice::VoiceStatus::Enrolled => Text::new("Enrolled").size(13).into(),
            crate::audio::voice::VoiceStatus::Error(msg) => Text::new(msg).size(13).into(),
        };
        // Gated: wake word can only be enabled while Transcription is ON (they
        // share the loaded ASR model). Turning Transcription OFF cascades it off.
        let wake_row = field_row(
            "Wake Word Detection",
            row![
                toggler(voice_enabled).on_toggle_maybe(if transcription_enabled {
                    Some(SettingsMessage::VoiceToggle)
                } else {
                    None
                }),
                Space::new().width(12),
                wake_status,
            ]
            .align_y(Alignment::Center)
            .into(),
            Some(if transcription_enabled {
                "Hands-free voice commands with wake word detection"
            } else {
                "Requires Transcription to be enabled"
            }),
        );

        // ── Row 3: Text to Speech (toggle + inline status + Test in one row) ──
        let tts_enabled = self.config.tts_enabled.as_deref() == Some("true");
        let tts_ready = crate::audio::tts::models_ready();
        let tts_failed = crate::audio::tts::download_failed();
        let audio_ok = crate::audio::tts::audio_output_ready();
        let tts_status: Element<'_, SettingsMessage> = if !tts_enabled {
            Text::new("Disabled")
                .size(13)
                .color(theme::TEXT_SECONDARY)
                .into()
        } else if tts_ready {
            if audio_ok {
                Text::new("Ready").size(13).into()
            } else {
                Text::new("No audio output device")
                    .size(13)
                    .color(theme::STATUS_ERROR)
                    .into()
            }
        } else if tts_failed {
            let retry_btn = button(Text::new("   Retry   ").size(13))
                .on_press(SettingsMessage::TtsRetryModels)
                .style(theme::button_danger)
                .padding(4);
            row![
                Text::new("Model download failed").size(14),
                Space::new().width(8),
                retry_btn,
            ]
            .align_y(Alignment::Center)
            .into()
        } else {
            Text::new("Downloading models…").size(13).into()
        };
        let test_btn = button(Text::new("   Test TTS   ").size(13))
            .style(theme::button_primary)
            .padding(4)
            .on_press_maybe(if tts_enabled && tts_ready {
                Some(SettingsMessage::TtsTest)
            } else {
                None
            });
        let tts_row = field_row(
            "Text to Speech",
            row![
                toggler(tts_enabled).on_toggle(SettingsMessage::TtsToggle),
                Space::new().width(12),
                tts_status,
                Space::new().width(12),
                test_btn,
            ]
            .align_y(Alignment::Center)
            .into(),
            Some("Text-to-speech for agent responses"),
        );

        // ── Wake-word enrollment UI (below the three rows, unchanged) ──
        // Enrolled-phrase display / phrase input / Enroll button (when voice
        // is on and transcription allows it — the push site below applies the
        // same guard, so the row collapses to nothing when either is off).
        let wake_word_row = if let Some(phrase) = crate::audio::voice::get_enrolled_phrase() {
            field_row("Wake Word", Text::new(phrase).size(13).into(), None)
        } else if has_enrollment {
            // V2 enrollment present but phrase unavailable (shouldn't
            // happen) — surface it so the user can re-enroll.
            field_row(
                "Wake Word",
                Text::new("Enrollment found — re-enroll to replace it")
                    .size(13)
                    .into(),
                None,
            )
        } else {
            field_row(
                "Wake Word",
                Text::new("Enroll a wake word to get started")
                    .size(13)
                    .into(),
                None,
            )
        };

        // Text input for the wake word phrase (before enrollment).
        let phrase_input = if voice_enabled && transcription_enabled && !is_enrolling {
            let input = widgets::single_line_editor(
                &self.wake_word_phrase_input.buffer,
                "mahbot",
                false,
                Length::Fixed(250.0),
                Some(Id::new("wake_word_phrase")),
                SettingsMessage::WakeWordPhraseInput,
            );
            field_row("Wake Word Phrase", input, None)
        } else {
            Space::new().height(0).into()
        };

        let enroll_btn: Element<'_, SettingsMessage> = if voice_enabled && transcription_enabled {
            container(
                button(Text::new("Enroll Wake Word").size(13))
                    .on_press(SettingsMessage::StartVoiceEnrollment)
                    .style(theme::button_primary)
                    .padding(6),
            )
            .into()
        } else {
            container(Text::new("")).into()
        };

        // Multi-line enrollment progress + Cancel (shown during active
        // enrollment; mirrors the previous status-row rendering).
        let enrollment_ui: Option<Element<'_, SettingsMessage>> = match status {
            crate::audio::voice::VoiceStatus::Enrolling {
                sample,
                total,
                duration_ms,
                quality,
            } => {
                let remaining = total.saturating_sub(sample);
                let mut lines: Vec<String> = Vec::new();

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

                    if sample < total {
                        let prompt = crate::audio::voice::enrollment_prompt_for_sample(sample);
                        lines.push(format!("📢 {prompt}"));
                    }

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

                let cancel_btn: Element<'_, SettingsMessage> = container(
                    button(Text::new("Cancel").size(13))
                        .on_press(SettingsMessage::CancelVoiceEnrollment)
                        .style(theme::button_danger)
                        .padding(6),
                )
                .into();
                Some(
                    Column::new()
                        .push(Space::new().height(8))
                        .push(Text::new(lines.join("\n")).size(13))
                        .push(Space::new().height(8))
                        .push(cancel_btn)
                        .into(),
                )
            }
            crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. } => {
                let cancel_btn: Element<'_, SettingsMessage> = container(
                    button(Text::new("Cancel").size(13))
                        .on_press(SettingsMessage::CancelVoiceEnrollment)
                        .style(theme::button_danger)
                        .padding(6),
                )
                .into();
                Some(
                    Column::new()
                        .push(Space::new().height(8))
                        .push(Text::new("Listening…").size(13))
                        .push(Space::new().height(8))
                        .push(cancel_btn)
                        .into(),
                )
            }
            crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. } => {
                let cancel_btn: Element<'_, SettingsMessage> = container(
                    button(Text::new("Cancel").size(13))
                        .on_press(SettingsMessage::CancelVoiceEnrollment)
                        .style(theme::button_danger)
                        .padding(6),
                )
                .into();
                Some(
                    Column::new()
                        .push(Space::new().height(8))
                        .push(Text::new("Keep silent to confirm…").size(13))
                        .push(Space::new().height(8))
                        .push(cancel_btn)
                        .into(),
                )
            }
            crate::audio::voice::VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs,
                wall_clock_elapsed,
            } => {
                let pct = (accumulated_secs * 100)
                    .checked_div(target_secs)
                    .unwrap_or(0);
                let cancel_btn: Element<'_, SettingsMessage> = container(
                    button(Text::new("Cancel").size(13))
                        .on_press(SettingsMessage::CancelVoiceEnrollment)
                        .style(theme::button_danger)
                        .padding(6),
                )
                .into();
                Some(
                    Column::new()
                        .push(Space::new().height(8))
                        .push(
                            Text::new(format!(
                                "Collecting negative samples… {accumulated_secs}s/{target_secs}s \
                                 ({pct}%) elapsed {wall_clock_elapsed}s"
                            ))
                            .size(13),
                        )
                        .push(Space::new().height(8))
                        .push(cancel_btn)
                        .into(),
                )
            }
            _ => None,
        };

        let mut column = Column::new()
            .push(transcription_row)
            .push(wake_row)
            .push(tts_row);
        if voice_enabled && transcription_enabled {
            column = column.push(Space::new().height(8));
            column = column.push(wake_word_row);
            column = column.push(phrase_input);
            column = column.push(enroll_btn);
        }
        if let Some(ui) = enrollment_ui {
            column = column.push(ui);
        }

        section("Audio", column)
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
                    &self.model_picker_inputs[ModelPickerTarget::ImageGen.idx()],
                    "model name (e.g. google/gemini-...)",
                    self.field_errors
                        .get("config:image_gen_models")
                        .or_else(|| self.field_errors.get("config:image_gen_model"))
                        .map(String::as_str),
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
                    &self.model_picker_inputs[ModelPickerTarget::Video.idx()],
                    "model name (e.g. minimax/hailuo-3)",
                    self.field_errors
                        .get("config:video_models")
                        .or_else(|| self.field_errors.get("config:video_model"))
                        .map(String::as_str),
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
                key: CONFIG_KEY_WEB_SEARCH_PROVIDER,
                value,
            }
        })
        .text_size(13)
        .style(super::widgets::pick_list_style)
        .width(Length::Fixed(180.0));

        let provider_row = field_row_with_error(
            "Web Search Provider",
            pick_list.into(),
            None,
            self.field_errors
                .get("config:web_search_provider")
                .map(String::as_str),
        );

        section(
            "Integrations",
            column![
                provider_row,
                self.config_password_field(
                    "Firecrawl API Key",
                    PasswordTarget::FirecrawlKey,
                    "fc-...",
                    CONFIG_KEY_FIRECRAWL_KEY,
                    None,
                    false,
                ),
                self.config_password_field(
                    "Exa API Key",
                    PasswordTarget::ExaKey,
                    "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    CONFIG_KEY_EXA_KEY,
                    None,
                    false,
                ),
                self.config_password_field(
                    "Telegram Bot Token",
                    PasswordTarget::TelegramToken,
                    "123:abc",
                    CONFIG_KEY_TELEGRAM_BOT_TOKEN,
                    None,
                    false,
                ),
            ],
        )
    }

    /// About section — embedded version and install/update mode.
    fn about_section() -> Element<'static, SettingsMessage> {
        // Bind the classification once per render — `update_mode()` performs
        // filesystem probes and must not run twice per frame.
        let mode = crate::self_update::update_mode();
        let mode_label = match mode {
            crate::self_update::UpdateMode::LocalCheckout => "local checkout",
            crate::self_update::UpdateMode::Registry => "crates.io install",
        };
        let mode_note = match mode {
            crate::self_update::UpdateMode::LocalCheckout => {
                "Self-update builds from the local source checkout."
            }
            crate::self_update::UpdateMode::Registry => {
                "Self-update checks crates.io and installs the latest version."
            }
        };
        section(
            "About",
            column![
                row![
                    text(format!("MahBot v{}", crate::self_update::VERSION))
                        .size(13)
                        .color(theme::TEXT_MUTED),
                ],
                row![
                    text(format!("Install mode: {mode_label}"))
                        .size(11)
                        .color(theme::TEXT_FAINT)
                ],
                row![text(mode_note).size(11).color(theme::TEXT_FAINT),],
            ],
        )
    }

    fn routing_section(&self) -> Element<'_, SettingsMessage> {
        // The two routable model slots (manager and worker) — saved routing
        // rows for models outside these slots are not rendered (they are inert
        // orphans). The video-transcription model never consults routing.
        let mut model_names: BTreeSet<String> = BTreeSet::new();
        model_names.insert(crate::config::resolve_or(
            self.config.manager_model.clone(),
            crate::config::DEFAULT_MANAGER_MODEL,
        ));
        model_names.insert(crate::config::resolve_or(
            self.config.worker_model.clone(),
            crate::config::DEFAULT_WORKER_MODEL,
        ));

        let mut rows: Vec<Element<'_, SettingsMessage>> = Vec::new();
        // With a custom chat endpoint staged, provider routing is a no-op —
        // surface that so the section isn't misleading (req 8).
        if self.custom_endpoint_active_ui() {
            rows.push(
                text(
                    "Provider routing only applies to OpenRouter — the custom endpoint ignores it.",
                )
                .size(10)
                .color(theme::STATUS_WARNING)
                .into(),
            );
        }
        for model_name in &model_names {
            let display_name = model_name.clone();
            let order_model = model_name.clone();
            let order_field = format!("routing_order:{model_name}");
            let order_submit = SettingsMessage::ConfigFieldSettleNow {
                field: order_field.clone(),
            };
            let order_error = self.field_errors.get(&order_field).map(String::as_str);
            // Placeholder is a hint only: "DeepSeek" shows for
            // models whose ID starts with the lowercase `deepseek/` prefix
            // (OpenRouter model IDs use lowercase vendor prefixes), and no
            // placeholder for every other model. An empty field always means
            // auto-routing — the hint can show while routing is auto.
            let placeholder = if model_name.starts_with("deepseek/") {
                "DeepSeek"
            } else {
                ""
            };
            let order_input: Element<'_, SettingsMessage> = widgets::single_line_editor(
                &self.field_editor(&order_field).buffer,
                placeholder,
                true,
                Length::Fixed(375.0),
                Some(Id::from(order_field.clone())),
                move |action| {
                    if matches!(action, EditorAction::Submit) {
                        order_submit.clone()
                    } else {
                        SettingsMessage::ModelRoutingOrder {
                            model: order_model.clone(),
                            action,
                        }
                    }
                },
            );

            let row = column![
                // Model name label (read-only)
                text(display_name)
                    .font(iced::Font::MONOSPACE)
                    .size(13)
                    .color(theme::TEXT_SECONDARY),
                Space::new().height(4),
                order_input,
            ]
            .spacing(2);

            rows.push(if let Some(err) = order_error {
                column![row, inline_error(err, 0.0)].spacing(2).into()
            } else {
                row.into()
            });
        }

        section("Provider Routing", Column::from_iter(rows))
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
    field_row_with_error(label, input, hint, None)
}

/// The inline error label rendered under a control: small, error-colored,
/// indented `left_pad` px to align with the input column.
fn inline_error(err: &str, left_pad: f32) -> Element<'_, SettingsMessage> {
    container(text(err).size(10).color(theme::STATUS_ERROR))
        .padding(iced::Padding::default().left(left_pad))
        .into()
}

/// The inline warning label rendered under a control: small, warning-colored,
/// indented `left_pad` px to align with the input column. Non-fatal — the
/// value was still saved (e.g. an unreachable custom endpoint).
fn inline_warning(msg: &str, left_pad: f32) -> Element<'_, SettingsMessage> {
    container(text(msg).size(10).color(theme::STATUS_WARNING))
        .padding(iced::Padding::default().left(left_pad))
        .into()
}

/// Like [`field_row`], with an optional inline error rendered under the
/// input (aligned with the input column) in the error color.
fn field_row_with_error<'a>(
    label: &'static str,
    input: Element<'a, SettingsMessage>,
    hint: Option<&'static str>,
    error: Option<&'a str>,
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

    let row_elem: Element<'a, SettingsMessage> = row_widget.into();
    if let Some(err) = error {
        column![row_elem, inline_error(err, 188.0),]
            .spacing(2)
            .into()
    } else {
        row_elem
    }
}

/// Delete confirmation prompt — the inline "Delete? Yes / No" row shown
/// below a workspace card when the row is the delete target.
fn delete_confirm_button<'a>(
    on_confirm: SettingsMessage,
    on_cancel: SettingsMessage,
) -> Element<'a, SettingsMessage> {
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
}

/// Dispatch a settled field value to its per-field persistence function.
///
/// Field ids (see [`SettingsMessage::ConfigFieldSettled`]):
/// - `config:<key>` — string config fields
/// - `routing_order:<model>` — per-model rows
///
/// Returns the canonical persisted value.
async fn persist_settled_field(
    field: &str,
    value: &str,
) -> anyhow::Result<crate::config::PersistOutcome> {
    if let Some(key) = field.strip_prefix("config:") {
        return crate::config::persist_settled_string_field(key, value).await;
    }
    if let Some(model) = field.strip_prefix("routing_order:") {
        let value = crate::config::persist_settled_routing_order(model, value).await?;
        return Ok(crate::config::PersistOutcome {
            value,
            warning: None,
        });
    }
    anyhow::bail!("unknown settings field: {field}");
}

/// Configuration for a single text field in [`modal_dialog`].
struct DialogField<'a> {
    label: &'static str,
    placeholder: &'static str,
    value: &'a super::editor_widget::EditorBuffer,
    /// Stable widget id for the field (unique across fields on the page).
    id: &'static str,
    /// Function pointer for the shared-editor `on_action` handler.
    ///
    /// Uses `fn(EditorAction) -> SettingsMessage` (function pointer) rather
    /// than `impl Fn(EditorAction) -> SettingsMessage + 'a` to keep the struct
    /// simple, avoid boxing, and rely on monomorphization at the callsite. This
    /// works because all current callers pass enum tuple-variant constructors
    /// (e.g. [`SettingsMessage::AddWorkspaceName`]), which coerce to function
    /// pointers.
    ///
    /// If a future caller needs to capture state in the closure, this field
    /// must be changed to `Box<dyn Fn(EditorAction) -> SettingsMessage + 'a>`.
    on_input: fn(EditorAction) -> SettingsMessage,
}

/// Build a reusable modal dialog: title, optional field rows, optional `middle`
/// content, and a Cancel / submit footer. The submit button shows `Adding...`
/// while in flight and is otherwise disabled unless `submit_enabled`.
///
/// Layout: title, spacer(16), field rows (8 px between), spacer(16),
/// middle (if any), spacer(16), footer.
#[expect(clippy::too_many_arguments)]
fn modal_dialog<'a>(
    title: String,
    fields: &[DialogField<'a>],
    middle: Option<Element<'a, SettingsMessage>>,
    submit_label: &'static str,
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

    if !fields.is_empty() || middle.is_some() {
        col = col.push(Space::new().height(16));
    }

    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            col = col.push(Space::new().height(8));
        }
        col = col.push(field_row(
            field.label,
            widgets::single_line_editor(
                field.value,
                field.placeholder,
                true,
                Length::Fixed(375.0),
                Some(Id::new(field.id)),
                field.on_input,
            ),
            None,
        ));
    }

    if !fields.is_empty() {
        col = col.push(Space::new().height(16));
    }

    if let Some(m) = middle {
        col = col.push(m);
        col = col.push(Space::new().height(16));
    }

    col = col.push(
        row![
            Space::new().width(Length::Fill),
            button(text("Cancel").size(13))
                .style(theme::button_secondary)
                .on_press(on_cancel),
            Space::new().width(8),
            button(text(if adding { "Adding..." } else { submit_label }).size(13))
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
            let mut input = SingleLineEditorState::new(case.input);
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
                    input.text().is_empty(),
                    "case: {} — input buffer should be cleared",
                    case.name
                );
            } else {
                assert_eq!(
                    input.text(),
                    case.input,
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
    #[serial_test::serial(voice)] // touches the process-global audio::voice pipeline via init_global()
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

    // ── Per-field autosave: settle generations ──────────────────────
    //
    // The safety requirement "no per-keystroke writes, no out-of-order async
    // writes" is enforced by the per-field generation counter: a settle whose
    // generation no longer matches the current counter is dropped, and so is
    // a persist result from a superseded settle. These tests pin that guard
    // (they never await the spawned tasks — dropped tasks never run, so no
    // store is needed; state transitions are the observable surface).

    #[test]
    fn text_field_keystrokes_arm_debounced_settles_with_generations() {
        let mut state = SettingsState::new();

        // Keystroke stages the value and arms a debounced settle (gen 1).
        let _task = state.update(SettingsMessage::ConfigField {
            key: "exa_key",
            value: "key-a".into(),
        });
        assert_eq!(
            state.field_gen.get("config:exa_key").copied(),
            Some(1),
            "first keystroke bumps gen to 1"
        );
        assert_eq!(
            state.config.exa_key.as_deref(),
            Some("key-a"),
            "value staged in the editable snapshot"
        );

        // Continued typing bumps the generation and re-stages the value.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "exa_key",
            value: "key-ab".into(),
        });
        assert_eq!(
            state.field_gen.get("config:exa_key").copied(),
            Some(2),
            "second keystroke bumps gen to 2"
        );
    }

    #[test]
    fn stale_settle_is_dropped_never_staging_the_old_value() {
        let mut state = SettingsState::new();
        let _ = state.update(SettingsMessage::ConfigField {
            key: "exa_key",
            value: "key-a".into(),
        });
        let _ = state.update(SettingsMessage::ConfigField {
            key: "exa_key",
            value: "key-ab".into(),
        });

        // The stale settle (gen 1) is dropped: no persist is spawned, the
        // staged value does not regress, and the generation is not consumed.
        let _task = state.update(SettingsMessage::ConfigFieldSettled {
            field: "config:exa_key".into(),
            value: "key-a".into(),
            generation: 1,
        });
        assert_eq!(
            state.field_gen.get("config:exa_key").copied(),
            Some(2),
            "stale settle must not consume the generation"
        );
        assert_eq!(
            state.config.exa_key.as_deref(),
            Some("key-ab"),
            "stale settle must not stage the stale value"
        );

        // A stale RESULT is dropped too — its error must not surface.
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:exa_key".into(),
            generation: 1,
            result: Err("stale failure".into()),
        });
        assert!(
            !state.field_errors.contains_key("config:exa_key"),
            "stale result must not surface its error"
        );
    }

    #[test]
    fn stale_result_does_not_apply_stale_value() {
        let mut state = SettingsState::new();
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "model-a".into(),
        });
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "model-b".into(),
        });

        // A stale SUCCESS result (gen 1) must not overwrite the staged value.
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:manager_model".into(),
            generation: 1,
            result: Ok(crate::config::PersistOutcome {
                value: "model-a".into(),
                warning: None,
            }),
        });
        assert_eq!(
            state.config.manager_model.as_deref(),
            Some("model-b"),
            "stale success must not overwrite the newer staged value"
        );
    }

    #[test]
    fn in_flight_persist_queues_newer_settle_and_flushes_on_result() {
        let mut state = SettingsState::new();

        // First settle spawns a persist (field marked in flight).
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: "exa".into(),
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettled {
            field: "config:web_search_provider".into(),
            value: "exa".into(),
            generation: 1,
        });
        assert!(
            state
                .in_flight_persists
                .contains("config:web_search_provider"),
            "fresh settle marks the field in flight"
        );

        // A second toggle while the first persist runs must not spawn a
        // second persist — it queues the newest value instead, so the last
        // settle always lands last in the DB.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: String::new(), // toggled back to Auto
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettled {
            field: "config:web_search_provider".into(),
            value: String::new(),
            generation: 2,
        });
        assert_eq!(
            state.pending_persists.get("config:web_search_provider"),
            Some(&(String::new(), 2)),
            "newer settle queued with the generation it was queued at"
        );

        // The in-flight persist's (stale) result frees the field and flushes
        // the pending value into a fresh persist — the latest edit wins.
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:web_search_provider".into(),
            generation: 1,
            result: Ok(crate::config::PersistOutcome {
                value: "exa".into(),
                warning: None,
            }),
        });
        assert!(
            !state
                .pending_persists
                .contains_key("config:web_search_provider"),
            "pending value flushed"
        );
        assert!(
            state
                .in_flight_persists
                .contains("config:web_search_provider"),
            "flushed persist re-marks the field in flight"
        );
    }

    #[test]
    fn stale_pending_flush_never_clobbers_a_newer_staged_edit() {
        let mut state = SettingsState::new();

        // Settle #1 spawns a persist (field in flight, gen 1).
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: "exa".into(),
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettled {
            field: "config:web_search_provider".into(),
            value: "exa".into(),
            generation: 1,
        });

        // Settle #2 while the first persist runs queues a pending value.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: String::new(), // toggled back to Auto
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettled {
            field: "config:web_search_provider".into(),
            value: String::new(),
            generation: 2,
        });
        assert_eq!(
            state.pending_persists.get("config:web_search_provider"),
            Some(&(String::new(), 2)),
            "pending value queued with its own generation"
        );

        // The user stages a THIRD edit (gen 3) before the first persist
        // completes — its debounce is armed and the snapshot holds it.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: "exa".into(),
        });
        assert_eq!(
            state.field_gen.get("config:web_search_provider").copied(),
            Some(3),
            "third edit bumped the generation"
        );

        // The first persist's result frees the field and flushes the pending
        // value with ITS OWN generation (2) — never the current one (3).
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:web_search_provider".into(),
            generation: 1,
            result: Ok(crate::config::PersistOutcome {
                value: "exa".into(),
                warning: None,
            }),
        });
        assert!(
            !state
                .pending_persists
                .contains_key("config:web_search_provider"),
            "pending value flushed"
        );
        assert!(
            state
                .in_flight_persists
                .contains("config:web_search_provider"),
            "flush re-marks the field in flight"
        );

        // The flushed persist's result carries gen 2 — stale against the
        // newer staged edit (gen 3) — so it must NOT overwrite the snapshot.
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:web_search_provider".into(),
            generation: 2,
            result: Ok(crate::config::PersistOutcome {
                value: String::new(), // the flushed (older) value's canonical
                warning: None,
            }),
        });
        assert_eq!(
            state.config.web_search_provider.as_deref(),
            Some("exa"),
            "stale flushed result must not clobber the newer staged value"
        );
        assert!(
            !state
                .field_errors
                .contains_key("config:web_search_provider"),
            "stale result must not surface anything either"
        );
    }

    #[test]
    fn settle_now_uses_the_staged_value_on_enter() {
        let mut state = SettingsState::new();

        // Enter on a text field settles the staged value immediately.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "model-c".into(),
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettleNow {
            field: "config:manager_model".into(),
        });
        assert_eq!(
            state.field_gen.get("config:manager_model").copied(),
            Some(2),
            "Enter settles the staged value with a fresh generation"
        );

        // Same for a second text field: the staged value is read from the
        // editable snapshot at settle time.
        let _task = state.update(SettingsMessage::ConfigField {
            key: "worker_model",
            value: "worker-d".into(),
        });
        let _task = state.update(SettingsMessage::ConfigFieldSettleNow {
            field: "config:worker_model".into(),
        });
        assert_eq!(
            state.field_gen.get("config:worker_model").copied(),
            Some(2),
            "Enter settles the second text field's staged value too"
        );
    }

    #[test]
    fn immediate_toggle_arms_a_zero_delay_settle() {
        let mut state = SettingsState::new();
        let _task = state.update(SettingsMessage::ConfigField {
            key: "web_search_provider",
            value: "firecrawl".into(),
        });
        assert_eq!(
            state.field_gen.get("config:web_search_provider").copied(),
            Some(1),
            "immediate toggle arms a zero-delay settle"
        );
    }

    #[test]
    fn model_slot_keystroke_arms_debounced_settle() {
        let mut state = SettingsState::new();

        // Model slots are text inputs → debounced settle (700 ms).
        let _task = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "test-model".to_string(),
        });
        assert_eq!(
            state.field_gen.get("config:manager_model").copied(),
            Some(1),
            "model slot keystroke arms a debounced settle"
        );
        assert_eq!(
            state.config.manager_model.as_deref(),
            Some("test-model"),
            "model slot staged in the editable snapshot"
        );
    }

    /// Every config key rendered by the Settings page must be a real config
    /// field: `TEXT_INPUT_KEYS ∪ IMMEDIATE_KEYS ⊆ ConfigData::string_fields()`.
    /// A key outside the vocabulary would stage forever or persist an
    /// orphaned row with no visible error — this cheap guard catches a
    /// typo'd or removed field the moment the array drifts.
    #[test]
    fn settings_config_keys_are_real_config_fields() {
        let known: Vec<&'static str> = ConfigData::STRUCT_FIELDS_DEFAULT
            .string_fields()
            .iter()
            .map(|(k, _)| *k)
            .collect();
        for key in TEXT_INPUT_KEYS.iter().chain(IMMEDIATE_KEYS.iter()) {
            assert!(
                known.contains(key),
                "settings key '{key}' is not a config field — TEXT_INPUT_KEYS / IMMEDIATE_KEYS drifted"
            );
        }
    }

    /// Custom-endpoint toggle state machine: ON reveals the
    /// endpoint fields only — nothing is staged, persisted, or settle-armed;
    /// OFF closes the section and clears both endpoint fields from the
    /// editable snapshot (the async persist path then removes the persisted
    /// rows). A failed endpoint save re-closes the revealed section. The
    /// returned Tasks are dropped — the async persist never runs here
    /// (matching the other settings tests).
    #[test]
    fn custom_endpoint_toggle_state_machine() {
        let mut state = SettingsState::new();
        assert!(
            state.config.provider_endpoint.is_none(),
            "precondition: no custom endpoint staged"
        );
        assert!(
            !state.custom_revealed,
            "precondition: custom section closed"
        );

        // ON: reveal only — nothing persisted, no settle armed, the editable
        // snapshot is untouched (an endpoint becomes active only once the
        // user settles a genuinely non-default URL).
        let _task = state.update(SettingsMessage::CustomEndpointToggle(true));
        assert!(
            state.custom_revealed,
            "ON reveals the custom-endpoint fields"
        );
        assert!(
            state.config.provider_endpoint.is_none(),
            "ON must not stage or persist an endpoint — a reveal alone must not configure one"
        );
        assert!(
            state.field_gen.get("config:provider_endpoint").is_none(),
            "ON must not arm a settle — no spurious persisted row"
        );

        // OFF: revert to OpenRouter, close the section, clear both fields,
        // drop pending settles.
        let _task = state.update(SettingsMessage::CustomEndpointToggle(false));
        assert!(!state.custom_revealed, "OFF closes the section");
        assert!(
            state.config.provider_endpoint.is_none(),
            "OFF clears the endpoint field"
        );
        assert!(
            state.config.provider_endpoint_key.is_none(),
            "OFF clears the endpoint key field"
        );
        assert!(
            state.field_gen.contains_key("config:provider_endpoint_key"),
            "OFF bumps the key field's generation (drops pending settles/results)"
        );

        // A failed endpoint save re-closes the revealed section: the toggle
        // must not show ON while the runtime stays on OpenRouter, and the
        // error surfaces in the bottom banner (the inline row hides with the
        // section).
        let _task = state.update(SettingsMessage::CustomEndpointToggle(true));
        assert!(state.custom_revealed, "re-reveal opens the section");
        let endpoint_generation = state
            .field_gen
            .get("config:provider_endpoint")
            .copied()
            .unwrap_or(0);
        let _task = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:provider_endpoint".into(),
            generation: endpoint_generation,
            result: Err("invalid endpoint URL".into()),
        });
        assert!(
            !state.custom_revealed,
            "failed endpoint save closes the revealed section"
        );
        assert!(
            state.error.is_some(),
            "failed endpoint save surfaces in the bottom banner"
        );
    }
}

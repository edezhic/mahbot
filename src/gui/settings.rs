//! Settings page — dynamic configuration editor.
//!
//! Reads the current config snapshot from [`crate::config::CONFIG`],
//! presents editable fields organised in sections, and persists every change
//! immediately when the value settles (debounced text inputs, immediate
//! toggles/pickers) via the per-field persistence
//! functions in [`crate::config`].
//!
//! Also manages workspaces, users, the custom-tools list and the admin-defined
//! custom roles (formerly separate pages), with modal dialogs for the add
//! operations.

use crate::Role;
use crate::Workspace;
use crate::channels::telegram_local::{self, Health, Mode};
use crate::config::{
    CONFIG, CONFIG_KEY_AUTOUPDATE, CONFIG_KEY_EXA_KEY, CONFIG_KEY_FIRECRAWL_KEY,
    CONFIG_KEY_MANAGER_MODEL, CONFIG_KEY_PROVIDER_ENDPOINT, CONFIG_KEY_PROVIDER_ENDPOINT_KEY,
    CONFIG_KEY_PROVIDER_KEY, CONFIG_KEY_TELEGRAM_API_HASH, CONFIG_KEY_TELEGRAM_API_ID,
    CONFIG_KEY_TELEGRAM_BOT_TOKEN, CONFIG_KEY_WEB_SEARCH_PROVIDER, CONFIG_KEY_WORKER_MODEL,
    ConfigData, ModelRouting,
};
#[cfg(target_os = "macos")]
use crate::config::{
    CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, CONFIG_KEY_TTS_ENABLED, CONFIG_KEY_VOICE_ENABLED,
};
use crate::custom_role::{CustomRole, CustomRoleBehavior};
use crate::tools::custom::ToolListing;
use crate::tools::media_catalog::image::validate_image_model;
use crate::workspace::MAX_WORKSPACE_NOTES_CHARS;
use strum::{EnumCount, IntoEnumIterator};

#[cfg(target_os = "macos")]
use iced::widget::Text;
use iced::widget::text::IntoFragment;
use iced::widget::{
    Column, Id, Row, Space, button, checkbox, column, container, image, pick_list, responsive, row,
    text, toggler, tooltip,
};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use std::collections::{BTreeSet, HashMap, HashSet};

use super::common::SingleLineEditorState;
use super::custom_roles;
use super::custom_tools;
use super::dialog;
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

/// Shared body of the single-line editor update arms: route
/// focus-navigation actions out, otherwise apply the edit to the buffer.
fn apply_simple_editor_action(
    editor: &mut SingleLineEditorState,
    action: EditorAction,
) -> Task<SettingsMessage> {
    if let Some(task) = super::common::focus_navigation_task(&action) {
        return task;
    }
    editor.apply_action(action);
    Task::none()
}

/// Remove a model from a list.
fn remove_model_from_list(model: &str, list: &mut Option<String>) {
    let mut models = parse_models(list.as_deref());
    models.retain(|m| m != model);
    *list = if models.is_empty() {
        None
    } else {
        Some(models.join("\n"))
    };
}

/// Build one model-picker entry row: the model name and the shared compact gray
/// [`widgets::compact_x_button`] removal button. Both the row and the label take
/// the width they are given — the name wraps instead of pushing the button out
/// of the popup, and a shrink-width row would measure the name against the whole
/// row and leave the button nothing.
fn model_entry_row(target: ModelPickerTarget, model: String) -> Element<'static, SettingsMessage> {
    let mut entry = row![
        text(model.clone())
            .size(theme::TEXT_12)
            .width(Length::Fill)
            .wrapping(iced::widget::text::Wrapping::WordOrGlyph)
    ];
    entry = entry
        .push(Space::new().width(theme::SPACE_4))
        .push(widgets::compact_x_button(
            theme::TEXT_SECONDARY,
            theme::SPACE_6,
            Some(SettingsMessage::ModelPicker {
                target,
                action: ModelPickerAction::RemoveModel(model),
            }),
            None,
        ));
    entry.align_y(Alignment::Center).width(Length::Fill).into()
}

/// Render a model picker with a list of model entries, remove buttons per
/// entry, and an add-model row (text input + "Add" button). The current active
/// selection is per-user and not shown here. Accepts a `target` to build the
/// correct parameterized `SettingsMessage::ModelPicker` values internally.
fn model_picker_list<'a>(
    target: ModelPickerTarget,
    models_field: Option<&'a str>,
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
    let models = parse_models(models_field);

    let items: Vec<Element<'a, SettingsMessage>> = if models.is_empty() {
        vec![
            text("No models configured yet.")
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
        ]
    } else {
        models
            .iter()
            .map(|model| model_entry_row(target, model.clone()))
            .collect()
    };

    let add_row = row![
        widgets::single_line_editor(
            &add_input.buffer,
            add_placeholder,
            false, // Enter adds nothing here; the Add button drives the mutation.
            Length::Fill,
            Some(Id::from(format!("model_picker:{}", target.idx()))),
            on_add_input,
        ),
        Space::new().width(theme::SPACE_4),
        button(text("Add").size(theme::TEXT_11))
            .padding(theme::PAD_4)
            .style(theme::button_secondary)
            .on_press(on_add),
    ]
    .align_y(Alignment::Center);

    let mut col = column![
        Column::from_iter(items).spacing(theme::SPACE_2),
        Space::new().height(4),
        add_row,
    ];

    if let Some(err) = error {
        col = col.push(Space::new().height(2));
        col = col.push(inline_error(err, 0.0));
    }

    col.into()
}

// ── Telegram Bot API mode helpers ────────────────────────────────

/// A mode in the words the one-line block shows it — short enough to sit beside
/// its controls in a half-width settings column.
fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Cloud => "Cloud",
        Mode::Local => "Local server",
    }
}

/// The one line the mode block carries, or `None` when it must not be drawn at
/// all: a local server that is installed and answering has nothing to report, so a
/// quiet bot and a busy one look exactly alike. Installing, starting, a switch in
/// flight and every failure do have something to say, the last one through its
/// reason.
fn telegram_line(status: &telegram_local::Status) -> Option<String> {
    let mode = mode_name(status.serving);
    let switching = status.target != status.serving;
    // Healthy is silence: installed, answering, nothing under way. A switch in
    // flight is not healthy, even once it reached the side it is heading to.
    if status.serving == Mode::Local
        && !switching
        && !status.migrating
        && status.health == Health::Running
        && status.last_error.is_none()
    {
        return None;
    }
    let state = if switching {
        // Where a pending switch is taking the bot is the new information; the
        // mode just named is what serves until it gets there.
        Some(format!("switching to {}…", mode_name(status.target)))
    } else if status.last_error.is_some() {
        // A failure is the whole message: no state word beside it, which would
        // misdescribe a server that answers but refuses the bot — or one whose
        // listener never came back.
        None
    } else if status.serving == Mode::Cloud {
        Some("served by Telegram".to_string())
    } else if status.health == Health::Down {
        Some("not running".to_string())
    } else {
        // Installing or starting, or a switch that already cut over and has not
        // finished bringing the bot up on this side.
        Some("installing or starting…".to_string())
    };
    let mut line = match state {
        Some(state) => format!("{mode} · {state}"),
        None => mode.to_string(),
    };
    if let Some(reason) = status.last_error.as_deref() {
        // The reason lives in the status snapshot, not in the page state. When
        // the switch is blocked this is the journal's cause, which survives
        // restarts.
        line.push_str(" — ");
        line.push_str(reason);
    }
    Some(line)
}

/// Whether the way back to the cloud is offered: only a failure of the local
/// server earns it, never a bot that has merely been quiet.
fn telegram_escape(status: &telegram_local::Status) -> bool {
    status.target == Mode::Local && status.last_error.is_some()
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
}

/// Map a `ModelPickerTarget` to the corresponding model-list field in
/// `ConfigData`.
fn picker_list_field<'a>(
    t: &'a ModelPickerTarget,
    config: &'a mut ConfigData,
) -> &'a mut Option<String> {
    match t {
        ModelPickerTarget::ImageGen => &mut config.image_gen_models,
        ModelPickerTarget::Video => &mut config.video_models,
    }
}

/// Map a `ModelPickerTarget` to its `config:…` settings field key — the
/// persisted-form counterpart of [`picker_list_field`].
fn picker_list_key(t: ModelPickerTarget) -> &'static str {
    match t {
        ModelPickerTarget::ImageGen => CONFIG_FIELD_IMAGE_GEN_MODELS,
        ModelPickerTarget::Video => CONFIG_FIELD_VIDEO_MODELS,
    }
}

/// Whether `field` is one the Models settings popup owns — the two model slots
/// and the two model lists. A save that fails for one of them reopens the popup,
/// so its inline error is never hidden behind a closed window.
fn is_model_setting_field(field: &str) -> bool {
    matches!(
        field,
        CONFIG_FIELD_IMAGE_GEN_MODELS | CONFIG_FIELD_VIDEO_MODELS
    ) || matches!(
        field.strip_prefix("config:"),
        Some(CONFIG_KEY_MANAGER_MODEL | CONFIG_KEY_WORKER_MODEL)
    )
}

/// Which password field the visibility toggle applies to.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PasswordTarget {
    ProviderKey,
    FirecrawlKey,
    ExaKey,
    TelegramToken,
    TelegramApiId,
    TelegramApiHash,
    EndpointKey,
}

#[derive(Debug, Clone)]
#[expect(private_interfaces)] // CustomToolsMsg wraps pub(crate) CustomToolsMessage
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
    // ── Application ─────────────────────────────────────────
    /// The Autolaunch switch: apply the decision to the system's own
    /// registration.
    AutolaunchToggle(bool),
    /// The system's answer to a read of its login registration, tagged with the
    /// click count it was taken under (`None` when the read itself failed, which
    /// is unknown rather than off).
    AutolaunchRead {
        epoch: u64,
        state: Option<bool>,
    },
    /// One apply's result: the state the system reports afterwards (`None` when it
    /// could not tell), or the failure.
    AutolaunchApplied(Result<Option<bool>, String>),
    // ── Telegram Bot API mode messages ──────────────────────────
    /// Open the api_id/api_hash dialog, re-seeded from CONFIG.
    OpenTelegramCredentials,
    /// Close the api_id/api_hash dialog without switching.
    CloseTelegramCredentials,
    /// A shared-editor action on the dialog's api_id field.
    TelegramApiIdInput(EditorAction),
    /// A shared-editor action on the dialog's api_hash field.
    TelegramApiHashInput(EditorAction),
    /// Save the pair, then ask for the switch to the local server. Refused
    /// while a save or switch started here is still in flight.
    SubmitTelegramCredentials,
    /// Open the confirmation that gives up the local server and its data.
    RequestSwitchToCloud,
    /// Close that confirmation without switching.
    DismissSwitchToCloud,
    /// The confirmation was accepted: ask for the switch back to the cloud.
    ConfirmSwitchToCloud,
    /// Re-attempt the switch that is pending. The direction is read from the
    /// journal at the moment of the click, so a retry can only repeat the switch
    /// the owner already asked for — never start the opposite one.
    RetrySwitch,
    /// Result of a save-and-switch this page started. `target` names which
    /// switch it was, so a failure is reported where it happened.
    TelegramSwitchResult {
        target: Mode,
        result: Result<(), String>,
    },
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
    /// Submit the add-user modal.
    SubmitAddUser,
    /// Result of user add.
    AddUserResult(Result<crate::users::AddUserOutcome, String>),
    // ── Custom tools area (sub-messages) ────────────────────────
    /// Wrapped custom-tools area message.
    CustomToolsMsg(custom_tools::CustomToolsMessage),
    // ── Custom roles area (sub-messages) ────────────────────────
    /// Wrapped custom-roles area message.
    CustomRolesMsg(custom_roles::CustomRolesMessage),
    /// Escape key pressed (dismisses modal if open).
    Escape,
    // ── Model settings popup ───────────────────────────────
    /// Open the Models settings popup from the Provider section.
    OpenModelsPopup,
    /// Close the Models settings popup. Closing is not a cancel: text still
    /// inside its settle window keeps saving.
    CloseModelsPopup,
    // ── Model picker messages ─────────────────────────────
    /// Operations on a model picker (add/remove list models).
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
    // ── Transcription messages ────────────────────────────
    /// Toggle local transcription on/off. Turning it OFF also turns Wake Word
    /// Detection off (they share the loaded ASR model — the cascade persists
    /// `voice_enabled` away and stops the pipeline). Toggling ON kicks the
    /// model load/download in the background (auto-activates when ready).
    #[cfg(target_os = "macos")]
    TranscriptionToggle(bool),
    /// Result of the async transcription-toggle persistence. `voice_was_enabled`
    /// carries the pre-toggle wake-word state so a failed persist can roll both
    /// keys (and the pipeline) back. `generation` guards against stale results
    /// from rapid toggling (mirrors [`Self::VoiceToggleResult`]).
    #[cfg(target_os = "macos")]
    TranscriptionToggleResult {
        generation: u64,
        voice_was_enabled: bool,
        result: Result<(), String>,
    },
    /// Retry the local ASR model load/download after a terminal failure
    /// (re-toggling alone cannot recover — a dedicated action is required).
    #[cfg(target_os = "macos")]
    RetryTranscription,
    // ── Voice assistant messages ──────────────────────────
    /// Toggle voice assistant on/off (immediately activates/deactivates the pipeline).
    #[cfg(target_os = "macos")]
    VoiceToggle(bool),
    /// Result of async DB persistence after a voice toggle.
    /// The `u64` is a generation counter used to detect stale results
    /// from rapid toggling — if it doesn't match `SettingsState::voice_toggle_gen`,
    /// the result is ignored as stale.
    #[cfg(target_os = "macos")]
    VoiceToggleResult(u64, Result<(), String>),
    /// Start enrollment session for wake word.
    #[cfg(target_os = "macos")]
    StartVoiceEnrollment,
    /// Cancel enrollment session.
    #[cfg(target_os = "macos")]
    CancelVoiceEnrollment,
    /// Retry loading voice models after a [`VoiceStatus::ModelError`].
    #[cfg(target_os = "macos")]
    RetryVoiceModels,
    // ── TTS messages ─────────────────────────────────────
    /// Toggle TTS on/off (persisted to config DB).
    #[cfg(target_os = "macos")]
    TtsToggle(bool),
    /// Result of async DB persistence after a TTS toggle.
    /// The `u64` is a generation counter used to detect stale results
    /// from rapid toggling — if it doesn't match `SettingsState::tts_toggle_gen`,
    /// the result is ignored as stale.
    #[cfg(target_os = "macos")]
    TtsToggleResult(u64, Result<(), String>),
    /// Retry TTS model download after a permanent failure.
    #[cfg(target_os = "macos")]
    TtsRetryModels,
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
];

/// Config keys rendered as discrete controls (toggles / pick lists) that
/// persist immediately on change. `audio_transcription_use_local` is not a
/// generic ConfigField key — it is a dedicated
/// [`SettingsMessage::TranscriptionToggle`] toggle with its own transactional
/// persist path (macOS only, like the rest of the audio settings).
const IMMEDIATE_KEYS: &[&str] = &[CONFIG_KEY_WEB_SEARCH_PROVIDER, CONFIG_KEY_AUTOUPDATE];

/// Field id of the custom chat-endpoint URL (the `config:`-prefixed form of
/// [`CONFIG_KEY_PROVIDER_ENDPOINT`]).
const CONFIG_FIELD_PROVIDER_ENDPOINT: &str = "config:provider_endpoint";
/// Field id of the custom chat-endpoint API key (the `config:`-prefixed form
/// of [`CONFIG_KEY_PROVIDER_ENDPOINT_KEY`]).
const CONFIG_FIELD_PROVIDER_ENDPOINT_KEY: &str = "config:provider_endpoint_key";
/// Field id of the web-search provider pick list.
const CONFIG_FIELD_WEB_SEARCH_PROVIDER: &str = "config:web_search_provider";
/// Field id of the image-generation model picker list.
const CONFIG_FIELD_IMAGE_GEN_MODELS: &str = "config:image_gen_models";
/// Field id of the video-generation model picker list.
const CONFIG_FIELD_VIDEO_MODELS: &str = "config:video_models";
/// Field id of the Autoupdate switch.
const CONFIG_FIELD_AUTOUPDATE: &str = "config:autoupdate";

/// The two routable model slots (manager + worker), resolved to their
/// trimmed/default model names. This is the set of models represented by
/// [`SettingsState::routing_section`] — saved routing rows for models outside
/// these slots are not rendered (they are inert orphans).
fn routing_slots(config: &ConfigData) -> BTreeSet<String> {
    let mut models = BTreeSet::new();
    models.insert(crate::config::resolve_or(
        config.manager_model.clone(),
        crate::config::DEFAULT_MANAGER_MODEL,
    ));
    models.insert(crate::config::resolve_or(
        config.worker_model.clone(),
        crate::config::DEFAULT_WORKER_MODEL,
    ));
    models
}

/// A model's saved provider routing order from a config snapshot.
fn routing_order_of(config: &ConfigData, model: &str) -> Option<String> {
    config
        .model_routings
        .iter()
        .find(|mr| mr.model == model)
        .and_then(|mr| mr.provider_order.clone())
}

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
    /// Last error message rendered in the bottom banner — failed custom-endpoint
    /// saves, failed Autolaunch registration changes, plus voice/TTS toggle
    /// failures on macOS.
    error: Option<String>,
    /// Per-field presentation/undo state for stateless config fields rendered
    /// through the shared single-line editor. Keyed by the canonical field id
    /// (`config:<key>`, `routing_order:<model>`), so each field owns its own
    /// buffer and undo stack across renders while `view(&self)` stays immutable.
    /// Entries are seeded from the config snapshot at construction and
    /// re-synced on [`Self::refresh`]; update handlers insert lazily via
    /// [`Self::field_editor_mut`].
    field_editors: HashMap<String, SingleLineEditorState>,
    /// Shared empty editor used as a never-panicking render fallback by
    /// [`Self::field_editor`]. Config keys are always pre-populated and routing
    /// rows are seeded at every manager/worker model mutation, so this is a
    /// pure safety net: a current model always has a routing row before render.
    empty_field_editor: SingleLineEditorState,
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
    /// Whether the Models settings popup is open. Held, not derived, because a
    /// save that fails after the owner closed the popup reopens it with its
    /// inline error — the failure is never hidden behind a closed window.
    models_popup_open: bool,

    // ── Application state ───────────────────────────────────────
    /// What the system reports about starting the product at login — the
    /// Autolaunch switch's shown state, not the config decision. `None` while it
    /// is not known: before the first answer, from the start of every read, and
    /// again after a failed apply.
    autolaunch: Option<bool>,
    /// Whether an apply is running. One at a time, and the switch is inert
    /// meanwhile: a click made then is lost, which is intended — it keeps one
    /// apply from overtaking another.
    autolaunch_applying: bool,
    /// Bumped by every Autolaunch click. A read carries the count it was taken
    /// under, so an answer that was already on its way when the owner clicked is
    /// discarded instead of painting the state the click replaced.
    autolaunch_epoch: u64,

    // ── Telegram Bot API mode state ─────────────────────────────
    /// Whether the api_id/api_hash dialog is open.
    telegram_credentials_open: bool,
    /// api_id field of the credentials dialog. Seeded from CONFIG when the
    /// dialog opens; its own buffer so an abandoned edit is not staged into the
    /// config snapshot before submit.
    telegram_api_id_input: SingleLineEditorState,
    /// api_hash field of the credentials dialog — rendered masked, like every
    /// other secret on this page (masking is render-only).
    telegram_api_hash_input: SingleLineEditorState,
    /// Inline error inside the credentials dialog: a rejected save, or the
    /// switch request refusing to start. Never carries the values themselves.
    telegram_credentials_error: Option<String>,
    /// Whether the switch-back-to-the-cloud confirmation is open.
    confirm_switch_to_cloud: bool,
    /// A save-and-switch this page started is in flight. Kept apart from the
    /// feature's own `migrating` flag because it exists from the click until
    /// the request returns — before anything is journaled and after a request
    /// that refused to start — so a double submit cannot start two switches.
    telegram_switch_busy: bool,

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
    /// Whether the add-user operation is in flight.
    add_user_adding: bool,

    // ── Custom tools area state ─────────────────────────────────
    /// The read-only list of custom tool files (see [`custom_tools`]).
    pub(crate) custom_tools_state: custom_tools::CustomToolsState,

    // ── Custom roles area state ─────────────────────────────────
    /// The admin-defined custom roles, their create/edit window and their
    /// deletion (see [`custom_roles`]).
    pub(crate) custom_roles_state: custom_roles::CustomRolesState,

    // ── Model picker state ────────────────────────────────
    /// Text input buffers for model pickers, indexed by [`ModelPickerTarget::idx`].
    model_picker_inputs: [SingleLineEditorState; ModelPickerTarget::COUNT],

    // ── Voice assistant state ─────────────────────────────
    /// Generation counter for voice toggle operations.
    /// Incremented before each `VoiceToggle`; the expected value is
    /// passed through to `VoiceToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    #[cfg(target_os = "macos")]
    voice_toggle_gen: u64,
    /// Generation counter for transcription toggle operations.
    /// Incremented before each `TranscriptionToggle`; the expected value is
    /// passed through to `TranscriptionToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    #[cfg(target_os = "macos")]
    transcription_toggle_gen: u64,

    // ── TTS state ─────────────────────────────────────────
    /// Generation counter for TTS toggle operations.
    /// Incremented before each `TtsToggle`; the expected value is
    /// passed through to `TtsToggleResult` so stale results from
    /// earlier toggles are detected and ignored.
    #[cfg(target_os = "macos")]
    tts_toggle_gen: u64,
}

/// Sync the voice assistant pipeline state with `CONFIG.voice_enabled()`.
/// Called from the immediate `VoiceToggle` handler after the toggle persists.
#[cfg(target_os = "macos")]
fn sync_voice_state(enabled: bool) {
    if enabled {
        crate::audio::voice::set_enabled(true);
        crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::StartListening);
    } else {
        crate::audio::voice::set_enabled(false);
        crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::StopListening);
    }
}

/// The custom-role prompt editor's own resizing range: it grows with its text up
/// to the maximum, and gives that height back down to the minimum when the window
/// runs out of room for the tools below it.
const CUSTOM_ROLE_PROMPT_MIN_HEIGHT: f32 = 120.0;
const CUSTOM_ROLE_PROMPT_MAX_HEIGHT: f32 = 280.0;

/// Space the tools section keeps under the prompt editor: the branching field,
/// the tool list's label and a few of its rows. The editor shrinks to its own
/// minimum before this does.
const CUSTOM_ROLE_TOOLS_MIN_HEIGHT: f32 = 146.0;

/// The behaviours the custom-role window's picker offers, in the order it shows
/// them. The labels it shows are [`CustomRoleBehavior`]'s `Display`; the stored
/// value stays `as_str()`.
const BEHAVIORS: [CustomRoleBehavior; 2] =
    [CustomRoleBehavior::Continuous, CustomRoleBehavior::OneOff];

impl SettingsState {
    pub fn new() -> Self {
        // The field editors are seeded from `self.config` (see
        // `resync_field_editors`), so the editable config and the editors always
        // reflect one snapshot: two independent CONFIG reads could race a
        // concurrent write and seed routing editors for a model set that does
        // not match `config`, leaving the current model to render the empty
        // fallback instead of its saved routing order.
        let mut state = Self {
            config: CONFIG.snapshot(),
            field_gen: HashMap::new(),
            in_flight_persists: HashSet::new(),
            pending_persists: HashMap::new(),
            field_errors: HashMap::new(),
            field_editors: HashMap::new(),
            empty_field_editor: SingleLineEditorState::new(""),
            endpoint_warning: None,
            error: None,
            password_visible: HashSet::new(),
            custom_revealed: false,
            models_popup_open: false,
            autolaunch: None,
            autolaunch_applying: false,
            autolaunch_epoch: 0,
            telegram_credentials_open: false,
            telegram_api_id_input: SingleLineEditorState::new(""),
            telegram_api_hash_input: SingleLineEditorState::new(""),
            telegram_credentials_error: None,
            confirm_switch_to_cloud: false,
            telegram_switch_busy: false,
            workspaces_state: workspaces::WorkspacesState::new(),
            users_state: users::UsersState::new(),
            show_add_workspace_modal: false,
            add_workspace_name: SingleLineEditorState::new(""),
            add_workspace_path: SingleLineEditorState::new(""),
            add_workspace_adding: false,
            show_add_user_modal: false,
            add_user_sender: SingleLineEditorState::new(""),
            add_user_adding: false,
            custom_tools_state: custom_tools::CustomToolsState::new(),
            custom_roles_state: custom_roles::CustomRolesState::new(),
            model_picker_inputs: std::array::from_fn(|_| SingleLineEditorState::new("")),
            #[cfg(target_os = "macos")]
            voice_toggle_gen: 0,
            #[cfg(target_os = "macos")]
            transcription_toggle_gen: 0,
            #[cfg(target_os = "macos")]
            tts_toggle_gen: 0,
        };
        state.resync_field_editors();
        state
    }

    /// Reload the editable snapshot from the current CONFIG and read what the
    /// system reports about Autolaunch.
    ///
    /// Inline errors are cleared (the page is being shown fresh). Pending
    /// debounced settles are NOT invalidated — a value typed before
    /// navigating away must still be persisted when its settle fires.
    ///
    /// Re-entered on every page entry, so the switch reports the system as it
    /// is now — including a disable the owner made in the system's own
    /// settings. The read runs off the update thread, so it is started rather
    /// than awaited.
    pub fn refresh(&mut self) -> Task<SettingsMessage> {
        self.config = CONFIG.snapshot();
        self.error = None;
        self.field_errors.clear();
        self.endpoint_warning = None;
        self.resync_field_editors();
        // A copy built from sources never registers, so there is nothing to
        // read and no subprocess to pay for. A running apply answers for
        // itself, so a read now could only report a stale state.
        if crate::self_update::is_published_copy() && !self.autolaunch_applying {
            return self.read_system_autolaunch();
        }
        Task::none()
    }

    /// Start a read of the system's registration, tagged with the click count it
    /// was taken under so a click made while it was in flight supersedes it. What
    /// the system last said is not what it says now, so the state is unknown again
    /// until the answer is in.
    fn read_system_autolaunch(&mut self) -> Task<SettingsMessage> {
        self.autolaunch = None;
        let epoch = self.autolaunch_epoch;
        Task::perform(read_autolaunch(), move |state| {
            SettingsMessage::AutolaunchRead { epoch, state }
        })
    }

    /// Re-sync the config text editors from the current config snapshot.
    /// Called from [`Self::new`] (initial seeding) and [`Self::refresh`] (page
    /// navigation). A config editor is rewritten only when the snapshot value
    /// actually differs from what it already shows — an identical-value
    /// rewrite would reset its caret and undo stack. Routing editors are
    /// seeded insert-if-absent (see [`Self::seed_routing_editors`]) so
    /// existing routing rows keep their undo stack / caret across refreshes.
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
                .and_modify(|e| {
                    if e.text() != value {
                        e.set_text(&value);
                    }
                })
                .or_insert_with(|| SingleLineEditorState::new(&value));
        }
        self.seed_routing_editors();
    }

    /// Ensure a routing-order editor exists for the two routable slots
    /// (manager + worker), the set reflected by [`Self::routing_section`].
    ///
    /// Insert-if-absent: an existing routing editor's undo stack and caret are
    /// never reset, so a model that already has a routing row keeps its editor
    /// state across refreshes and model-value changes. Values are read from the
    /// editable snapshot's model routings. Called from every window that can
    /// change the manager/worker model (staging, persist result, refresh).
    fn seed_routing_editors(&mut self) {
        for model in routing_slots(&self.config) {
            let field = format!("routing_order:{model}");
            if self.field_editors.contains_key(&field) {
                continue;
            }
            let value = routing_order_of(&self.config, &model).unwrap_or_default();
            self.field_editors
                .insert(field, SingleLineEditorState::new(&value));
        }
    }

    /// Push the canonical value for a single field id into its editor (if the
    /// editor has been created). Called after a completed persist (via
    /// [`Self::sync_persisted`]) and from the endpoint toggle-off cascade so
    /// the rendered value reflects the source of truth.
    ///
    /// The rewrite is skipped when the editor already shows the canonical
    /// value: a settled persist echoes back exactly what the user typed, and
    /// rewriting identical text would snap the caret to (0,0) (cursor reset
    /// lives in `EditorBuffer::set_text`) and drop the undo stack (cleared in
    /// `SingleLineEditorState::set_text`). A genuinely different canonical
    /// value (e.g. whitespace trimmed on save) still rewrites the editor and
    /// moves the caret — accepted, the text changed.
    fn resync_field_editor(&mut self, field: &str) {
        if let Some(value) = self.staged_value(field) {
            if let Some(editor) = self.field_editors.get_mut(field) {
                if editor.text() == value {
                    return;
                }
                editor.set_text(&value);
            }
        }
    }

    /// Borrow the per-field editor for a rendered stateless config field.
    ///
    /// Config keys are always pre-populated and routing rows are seeded at every
    /// manager/worker model mutation, so the empty fallback is a pure safety net
    /// that can never abort a render.
    fn field_editor(&self, key: &str) -> &SingleLineEditorState {
        self.field_editors
            .get(key)
            .unwrap_or(&self.empty_field_editor)
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
    /// OpenRouter-key highlight, the toggle state, and the visibility of
    /// the OpenRouter Provider Routing section, so UI state can never
    /// diverge from the normalized runtime endpoint.
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
        self.add_user_adding = false;
    }

    /// Mirror a config value into both snapshots: the editable `self.config`
    /// and the global `CONFIG`, so a refresh() can't revert the change.
    #[cfg(target_os = "macos")]
    fn set_both(&mut self, key: &str, value: &str) {
        let _ = self.config.set_string_field(key, value);
        let _ = crate::config::CONFIG.set_string_field(key, value);
    }

    /// Mirror a toggle in both config snapshots: `"true"`/`""` (the empty
    /// string keeps the `non_empty` accessor collapsing to None = disabled).
    #[cfg(target_os = "macos")]
    fn set_toggle(&mut self, key: &str, enabled: bool) {
        self.set_both(key, if enabled { "true" } else { "" });
    }

    /// Shared voice/TTS toggle arm: mirror the new state, fire the per-toggle
    /// side effect, bump the generation counter, then persist to the DB.
    #[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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
            return routing_order_of(&self.config, model);
        }
        None
    }

    /// Schedule a settle for a field: bump its generation, then emit
    /// `ConfigFieldSettled` after `delay_ms`. The bumped generation is captured
    /// into the settle message so a stale settle is dropped by its handler.
    fn schedule_settle(
        &mut self,
        field: &str,
        value: String,
        delay_ms: u64,
    ) -> Task<SettingsMessage> {
        let generation = self.bump_gen(field);
        let f = field.to_string();
        Task::perform(
            super::widgets::debounce_sleep(delay_ms, generation),
            move |g| SettingsMessage::ConfigFieldSettled {
                field: f,
                value,
                generation: g,
            },
        )
    }

    /// Schedule an immediate (delay-0) settle for a field.
    fn settle_now(&mut self, field: &str, value: String) -> Task<SettingsMessage> {
        self.schedule_settle(field, value, 0)
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
            // A persist/revert of a manager/worker model updates the routing
            // key set; seed the routing editor for the applied (trimmed) model
            // so the routing rows render without a missing-editor panic.
            if matches!(key, CONFIG_KEY_MANAGER_MODEL | CONFIG_KEY_WORKER_MODEL) {
                self.seed_routing_editors();
            }
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

    /// Sync a field's staged snapshot and editor with the last persisted value:
    /// `None` reverts the staged value (a failed persist on a discrete
    /// toggle/picker control), `Some(value)` applies a canonical persisted value
    /// (a settled text field may have been replaced by a refresh while the
    /// persist was in flight — the persisted value is the user's last edit and
    /// must win).
    fn sync_persisted(&mut self, field: &str, value: Option<&str>) {
        self.sync_field(field, value);
        self.resync_field_editor(field);
    }

    /// Whether a failed persist should roll the control back to the last
    /// persisted value. Discrete-state controls (toggles, pick lists, and the
    /// model pickers' optimistic list markers) revert; free-text inputs
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
            CONFIG_FIELD_WEB_SEARCH_PROVIDER
                | CONFIG_FIELD_IMAGE_GEN_MODELS
                | CONFIG_FIELD_VIDEO_MODELS
                | CONFIG_FIELD_PROVIDER_ENDPOINT
                | CONFIG_FIELD_AUTOUPDATE
        )
    }

    /// Persist a model-picker's list field immediately (discrete action —
    /// add/remove a model). Clears the picker's inline error and settles now.
    fn persist_picker_list(&mut self, target: ModelPickerTarget) -> Task<SettingsMessage> {
        let field = picker_list_key(target);
        let value = picker_list_field(&target, &mut self.config)
            .clone()
            .unwrap_or_default();
        self.field_errors.remove(field);
        self.settle_now(field, value)
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: SettingsMessage) -> Task<SettingsMessage> {
        match msg {
            // ── Config field edits ─────────────────────────────
            SettingsMessage::ConfigField { key, value } => {
                let _ = self.config.set_string_field(key, &value);
                // A manager/worker model change re-keys the routing rows, so
                // seed the routing editor for the new (resolved/trimmed) model
                // before the next render — a mid-keystroke partial name must
                // render its routing row without a missing-editor panic.
                if matches!(key, CONFIG_KEY_MANAGER_MODEL | CONFIG_KEY_WORKER_MODEL) {
                    self.seed_routing_editors();
                }
                let field = format!("config:{key}");
                self.field_errors.remove(&field);
                if TEXT_INPUT_KEYS.contains(&key) {
                    // Text input: persist only after the value settles
                    // (debounce), never per keystroke.
                    self.schedule_settle(&field, value, SETTLE_MS)
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
                        self.sync_persisted(&field, Some(&outcome.value));
                        // Switching Autoupdate back on must restore today's
                        // behaviour now, not up to ten minutes from now: wake
                        // the availability loop for a tick. The stored spelling is
                        // free, so it is read by the same rule as everywhere else.
                        if field == CONFIG_FIELD_AUTOUPDATE
                            && crate::config::flag_on(Some(outcome.value.as_str()))
                        {
                            crate::self_update::nudge_update_tick();
                        }
                        // The endpoint arm's warning has a slot of its own, and it
                        // is the only arm that carries one: a post-write failure
                        // that leaves the save standing is logged by the persist
                        // layer instead.
                        if field == CONFIG_FIELD_PROVIDER_ENDPOINT {
                            self.endpoint_warning = outcome.warning;
                        }
                        Task::none()
                    }
                    Err(e) => {
                        self.field_errors.insert(field.clone(), e.clone());
                        if field == CONFIG_FIELD_PROVIDER_ENDPOINT {
                            self.endpoint_warning = None;
                        }
                        // The model settings live in a popup: a save that fails
                        // there reopens it, so the inline error is never hidden
                        // behind a closed window (closing is not a cancel).
                        if is_model_setting_field(&field) {
                            self.models_popup_open = true;
                        }
                        if Self::field_reverts_on_error(&field) {
                            // Discrete-state controls (toggles/pickers) roll
                            // back to the last persisted value; free text
                            // keeps the typed value for correction.
                            self.sync_persisted(&field, None);
                            // A failed endpoint save means no configuration change — close
                            // the revealed section so the toggle cannot show ON while the
                            // runtime stays on OpenRouter. The error stays visible in the
                            // bottom banner (the inline row hides with the section) and
                            // re-appears inline if the user reveals the section again.
                            if field == CONFIG_FIELD_PROVIDER_ENDPOINT {
                                self.custom_revealed = false;
                                self.error = Some(e);
                                // Toggle-off clears both endpoint fields as one discrete
                                // action — a failed endpoint persist must restore the key
                                // field too, so the UI cannot show a half-reverted custom
                                // setup (endpoint defaulted, key cleared).
                                self.sync_persisted(CONFIG_FIELD_PROVIDER_ENDPOINT_KEY, None);
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
                self.field_errors.remove(CONFIG_FIELD_PROVIDER_ENDPOINT);
                Task::none()
            }
            SettingsMessage::CustomEndpointToggle(false) => {
                // OFF: revert to OpenRouter and clear both persisted rows. Bump
                // generation for both fields (drops pending settles/results),
                // remove pending persists, clear both from the editable
                // snapshot, then settle the endpoint field with `""` — the
                // persist path cascades the provider_endpoint_key row deletion.
                self.custom_revealed = false;
                let endpoint_field = CONFIG_FIELD_PROVIDER_ENDPOINT;
                let key_field = CONFIG_FIELD_PROVIDER_ENDPOINT_KEY;
                self.bump_gen(endpoint_field);
                self.bump_gen(key_field);
                self.pending_persists.remove(endpoint_field);
                self.pending_persists.remove(key_field);
                self.config.provider_endpoint = None;
                self.config.provider_endpoint_key = None;
                self.field_errors.remove(endpoint_field);
                self.field_errors.remove(key_field);
                self.endpoint_warning = None;
                self.resync_field_editor(endpoint_field);
                self.resync_field_editor(key_field);
                self.settle_now(endpoint_field, String::new())
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
                    self.schedule_settle(&field, order, SETTLE_MS)
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

            // ── Application ────────────────────────────────────
            SettingsMessage::AutolaunchToggle(enabled) => {
                // The switch is inert while an apply runs, so a click that still
                // reaches here is lost — intended: applies cannot overtake each
                // other.
                if self.autolaunch_applying {
                    return Task::none();
                }
                // Optimistic: the switch follows the click until the system
                // answers, and this click supersedes any read already on its way.
                self.autolaunch_epoch += 1;
                self.autolaunch_applying = true;
                self.autolaunch = Some(enabled);
                Task::perform(
                    save_and_apply_autolaunch(enabled),
                    SettingsMessage::AutolaunchApplied,
                )
            }
            SettingsMessage::AutolaunchRead { epoch, state } => {
                // Only the answer to the newest read counts: one taken before the
                // owner's last click would paint the state that click already
                // replaced. A read that could not run leaves the state unknown,
                // which is not "off".
                if epoch == self.autolaunch_epoch {
                    self.autolaunch = state;
                }
                Task::none()
            }
            SettingsMessage::AutolaunchApplied(result) => {
                self.autolaunch_applying = false;
                match result {
                    Ok(state) => {
                        self.autolaunch = state;
                        Task::none()
                    }
                    Err(e) => {
                        self.error = Some(e);
                        // The failed write leaves the system's state unknown; read
                        // it rather than asserting the state it may not have.
                        self.read_system_autolaunch()
                    }
                }
            }

            // ── Telegram Bot API mode ──────────────────────────
            SettingsMessage::OpenTelegramCredentials => {
                // Seed from what is on file, so a correction starts from the
                // pair already saved — this dialog is the only place it is ever
                // shown, and always masked. An empty pair simply seeds empty.
                self.telegram_api_id_input
                    .set_text(CONFIG.telegram_api_id().as_deref().unwrap_or_default());
                self.telegram_api_hash_input
                    .set_text(CONFIG.telegram_api_hash().as_deref().unwrap_or_default());
                self.telegram_credentials_error = None;
                self.telegram_credentials_open = true;
                Task::none()
            }
            SettingsMessage::CloseTelegramCredentials => {
                self.telegram_credentials_open = false;
                Task::none()
            }
            SettingsMessage::TelegramApiIdInput(action) => {
                apply_simple_editor_action(&mut self.telegram_api_id_input, action)
            }
            SettingsMessage::TelegramApiHashInput(action) => {
                apply_simple_editor_action(&mut self.telegram_api_hash_input, action)
            }
            SettingsMessage::SubmitTelegramCredentials => {
                // One switch at a time: the request behind this button journals
                // a migration, and a second one would interleave the two.
                if self.telegram_switch_busy {
                    return Task::none();
                }
                let api_id = self.telegram_api_id_input.text();
                let api_hash = self.telegram_api_hash_input.text();
                let (id_error, hash_error) =
                    telegram_local::credential_problems(&api_id, &api_hash);
                if id_error.is_some() || hash_error.is_some() {
                    // Both messages are already inline under their fields; the
                    // dialog stays open and nothing is saved or started.
                    return Task::none();
                }
                self.telegram_switch_busy = true;
                self.telegram_credentials_error = None;
                Task::perform(
                    save_credentials_and_switch_local(api_id, api_hash),
                    |result| SettingsMessage::TelegramSwitchResult {
                        target: Mode::Local,
                        result,
                    },
                )
            }
            SettingsMessage::RequestSwitchToCloud => {
                if !self.telegram_switch_busy {
                    self.confirm_switch_to_cloud = true;
                }
                Task::none()
            }
            SettingsMessage::DismissSwitchToCloud => {
                self.confirm_switch_to_cloud = false;
                Task::none()
            }
            SettingsMessage::ConfirmSwitchToCloud => {
                self.confirm_switch_to_cloud = false;
                if self.telegram_switch_busy {
                    return Task::none();
                }
                self.telegram_switch_busy = true;
                Task::perform(
                    async {
                        telegram_local::request_switch(Mode::Cloud)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    |result| SettingsMessage::TelegramSwitchResult {
                        target: Mode::Cloud,
                        result,
                    },
                )
            }
            SettingsMessage::RetrySwitch => {
                if self.telegram_switch_busy {
                    return Task::none();
                }
                // The pending direction comes from the journal, so the retry
                // repeats what was asked for — a blocked revert stays a revert.
                let target = telegram_local::status().target;
                self.telegram_switch_busy = true;
                Task::perform(
                    async move {
                        telegram_local::request_switch(target)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| SettingsMessage::TelegramSwitchResult { target, result },
                )
            }
            SettingsMessage::TelegramSwitchResult { target, result } => {
                self.telegram_switch_busy = false;
                match (target, result) {
                    // The local switch is journaled and runs in the background
                    // from here; the dialog's job is done.
                    (Mode::Local, Ok(())) => {
                        self.telegram_credentials_open = false;
                        self.telegram_credentials_error = None;
                        self.telegram_api_id_input.clear();
                        self.telegram_api_hash_input.clear();
                    }
                    // The pair is saved but the switch did not start: reopen the
                    // dialog if the owner dismissed it while the request ran,
                    // so the reason is never hidden behind a closed window.
                    (Mode::Local, Err(e)) => {
                        self.telegram_credentials_open = true;
                        self.telegram_credentials_error = Some(e);
                    }
                    // The back-to-cloud switch reports on the page banner: its
                    // confirmation is gone by the time the answer lands.
                    (Mode::Cloud, Ok(())) => {}
                    (Mode::Cloud, Err(e)) => {
                        self.error = Some(format!("The bot could not return to the cloud: {e}"));
                    }
                }
                Task::none()
            }

            // ── Transcription ───────────────────────────────────
            #[cfg(target_os = "macos")]
            SettingsMessage::TranscriptionToggle(enabled) => {
                self.transcription_toggle_gen += 1;
                let generation = self.transcription_toggle_gen;
                let voice_was_enabled = self.config.voice_enabled.as_deref() == Some("true");
                // enabled → "" = absent row; disabled → "false", matching the
                // persisted row semantics.
                let transcription_value = if enabled { "" } else { "false" };
                self.set_both(
                    CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL,
                    transcription_value,
                );
                // Turning transcription OFF also turns Wake Word Detection OFF
                // (shared ASR model): stop the pipeline now.
                if !enabled && voice_was_enabled {
                    self.set_both(CONFIG_KEY_VOICE_ENABLED, "");
                    sync_voice_state(false);
                }
                // Toggle ON: kick the model load/download in the background now
                // (otherwise the recogniser loads on the first voice
                // operation) — load from cache first, otherwise download with
                // retries. A terminal failure is recovered via retry_init (same
                // path as the inline Retry button); any other state falls
                // through to spawn_background_init, which claims Uninit→Loading
                // with the same panic→Failed guard every background load uses
                // (a raw spawn here could strand the transcriber in Loading on
                // a panic).
                if enabled && !crate::audio::local_transcriber::is_loaded() {
                    if !crate::audio::local_transcriber::retry_init() {
                        crate::audio::local_transcriber::spawn_background_init();
                    }
                }
                // Wake the voice pipeline so it re-reads config: the 1s
                // periodic timer is parked while transcription is disabled, and
                // a runtime re-enable must re-resolve status / auto-start here.
                // Cheap no-op wake for the enable path too.
                crate::audio::voice::send_command(crate::audio::voice::VoiceCommand::ConfigChanged);
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
            #[cfg(target_os = "macos")]
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
                        self.set_both(CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL, restore);
                        // Restore the wake-word state the cascade disabled.
                        if voice_was_enabled {
                            self.set_both(CONFIG_KEY_VOICE_ENABLED, "true");
                            sync_voice_state(true);
                        }
                        Task::none()
                    }
                }
            }
            #[cfg(target_os = "macos")]
            SettingsMessage::RetryTranscription => {
                let _ = crate::audio::local_transcriber::retry_init();
                Task::none()
            }

            // ── Voice assistant ─────────────────────────────────
            #[cfg(target_os = "macos")]
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
            #[cfg(target_os = "macos")]
            SettingsMessage::VoiceToggleResult(g, result) => self.handle_toggle_result(
                CONFIG_KEY_VOICE_ENABLED,
                g,
                result,
                |s| s.voice_toggle_gen,
                |c| &c.voice_enabled,
                sync_voice_state,
            ),
            #[cfg(target_os = "macos")]
            SettingsMessage::TtsToggle(enabled) => self.run_toggle(
                CONFIG_KEY_TTS_ENABLED,
                enabled,
                |s| {
                    s.tts_toggle_gen += 1;
                    s.tts_toggle_gen
                },
                SettingsMessage::TtsToggleResult,
                |enabled| {
                    if enabled {
                        // Open the OS audio output device so playback works immediately without a restart.
                        let _ = crate::audio::tts::ensure_audio_output();
                    }
                    // Toggle ON with uncached models triggers download (handles
                    // ModelState::Failed retries too, matching voice's auto-retry).
                    if enabled && !crate::audio::tts::try_load_cached() {
                        crate::audio::tts::spawn_or_retry_download();
                    }
                },
            ),
            #[cfg(target_os = "macos")]
            SettingsMessage::TtsToggleResult(g, result) => self.handle_toggle_result(
                CONFIG_KEY_TTS_ENABLED,
                g,
                result,
                |s| s.tts_toggle_gen,
                |c| &c.tts_enabled,
                |_| {},
            ),
            #[cfg(target_os = "macos")]
            SettingsMessage::TtsRetryModels => {
                let _ = crate::audio::tts::retry_download();
                Task::none()
            }
            #[cfg(target_os = "macos")]
            SettingsMessage::StartVoiceEnrollment => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StartEnrollment,
                );
                Task::none()
            }
            #[cfg(target_os = "macos")]
            SettingsMessage::CancelVoiceEnrollment => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::CancelEnrollment,
                );
                Task::none()
            }
            #[cfg(target_os = "macos")]
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
                apply_simple_editor_action(&mut self.add_workspace_name, action)
            }
            SettingsMessage::AddWorkspacePath(action) => {
                apply_simple_editor_action(&mut self.add_workspace_path, action)
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
                    // Adding activates the new workspace: the Dashboard
                    // turns this result into the active one (see
                    // `process_settings_message`), so the addition is all
                    // this task does.
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
                // The list refresh is handled by the Dashboard's shared-map
                // reload on this same result (see process_settings_message).
                Task::none()
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
                    self.close_add_user_modal();
                }
                Task::none()
            }
            SettingsMessage::AddUserSender(action) => {
                apply_simple_editor_action(&mut self.add_user_sender, action)
            }
            SettingsMessage::SubmitAddUser => {
                if self.add_user_sender.text().is_empty() {
                    return Task::none();
                }
                self.add_user_adding = true;
                let sender = self.add_user_sender.text();
                Task::perform(
                    async move {
                        let store = users::user_store()?;
                        // The dialog already refuses every name it can judge;
                        // what is left here is the fallback for a race or a
                        // store error. The Settings bypass has no way to
                        // complete a leftover unbound row, so a duplicate would
                        // silently no-op (INSERT OR IGNORE) while reporting
                        // success. A name that only matches by case or Unicode
                        // composition passes this and is refused by `add_user`,
                        // with the same message.
                        crate::users::validate_new_user_name(&sender).map_err(|e| e.to_string())?;
                        if store
                            .user_exists(&sender)
                            .await
                            .map_err(|e| e.to_string())?
                        {
                            return Err(crate::users::name_taken_message(&sender));
                        }
                        let outcome = store.add_user(&sender).await.map_err(|e| e.to_string())?;
                        Ok(outcome)
                    },
                    SettingsMessage::AddUserResult,
                )
            }
            SettingsMessage::AddUserResult(Ok(outcome)) => {
                // The account is added either way; a personal workspace folder
                // that was not created must not read as a whole success, so it
                // is told as a warning beside the refreshed list.
                let name = self.add_user_sender.text();
                self.close_add_user_modal();
                let refresh = self.users_state.refresh().map(SettingsMessage::UserMsg);
                match outcome {
                    crate::users::AddUserOutcome::Complete => refresh,
                    crate::users::AddUserOutcome::WithoutPersonalWorkspace => Task::batch([
                        refresh,
                        Task::done(SettingsMessage::UserMsg(users::UsersMessage::Toast(
                            super::ToastMessage::Warning(
                                crate::users::missing_personal_workspace_note(&name),
                            ),
                        ))),
                    ]),
                }
            }
            SettingsMessage::AddUserResult(Err(e)) => {
                self.add_user_adding = false;
                Task::done(SettingsMessage::UserMsg(users::UsersMessage::Toast(
                    super::ToastMessage::Error(e),
                )))
            }

            // ── Custom tools area ───────────────────────────────
            SettingsMessage::CustomToolsMsg(msg) => self
                .custom_tools_state
                .update(msg)
                .map(SettingsMessage::CustomToolsMsg),

            // ── Custom roles area ───────────────────────────────
            // The list is only ever changed by the admin's own actions, so the
            // area re-reads it after every mutation and never polls.
            SettingsMessage::CustomRolesMsg(msg) => self
                .custom_roles_state
                .update(msg)
                .map(SettingsMessage::CustomRolesMsg),

            // ── Model picker messages ─────────────────────────
            SettingsMessage::ModelPicker { target, action } => {
                match (target, action) {
                    (t, ModelPickerAction::AddInput(action)) => {
                        apply_simple_editor_action(&mut self.model_picker_inputs[t.idx()], action)
                    }
                    (t, ModelPickerAction::AddModel) => match t {
                        // Image-gen additions are validated against the catalog
                        // (fail-open when it is unavailable) before the list is
                        // mutated; the input buffer is kept on rejection so the
                        // user can correct it.
                        ModelPickerTarget::ImageGen => {
                            let model = self.model_picker_inputs[t.idx()].text().trim().to_string();
                            if model.is_empty() {
                                return Task::none();
                            }
                            Task::perform(
                                async move {
                                    let ok = validate_image_model(&model)
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
                            let models = picker_list_field(&t, &mut self.config);
                            add_model_to_list(&mut self.model_picker_inputs[t.idx()], models);
                            self.persist_picker_list(t)
                        }
                    },
                    (t, ModelPickerAction::RemoveModel(model)) => {
                        let models = picker_list_field(&t, &mut self.config);
                        remove_model_from_list(&model, models);
                        self.persist_picker_list(t)
                    }
                }
            }

            SettingsMessage::ModelPickerAddResult { target, model, ok } => match ok {
                Ok(()) => {
                    // Append the validated model directly — never route through
                    // the input buffer, which may hold text the user typed while
                    // the catalog validation was in flight.
                    let models = picker_list_field(&target, &mut self.config);
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
                    // inline rejection for config validation). The popup it
                    // belongs to reopens if it was closed meanwhile.
                    let key = picker_list_key(target);
                    self.field_errors
                        .insert(key.to_string(), format!("Model `{model}` rejected: {e}"));
                    self.models_popup_open = true;
                    Task::none()
                }
            },

            SettingsMessage::OpenModelsPopup => {
                self.models_popup_open = true;
                Task::none()
            }
            SettingsMessage::CloseModelsPopup => {
                self.models_popup_open = false;
                Task::none()
            }

            SettingsMessage::Escape => {
                if self.show_add_workspace_modal {
                    self.close_add_workspace_modal();
                } else if self.show_add_user_modal {
                    self.close_add_user_modal();
                } else if self.telegram_credentials_open {
                    self.telegram_credentials_open = false;
                } else if self.confirm_switch_to_cloud {
                    self.confirm_switch_to_cloud = false;
                } else if self.models_popup_open {
                    self.models_popup_open = false;
                } else {
                    // The last tier, matching render_modal_overlay: the
                    // custom-tools details window, the custom-roles editor and
                    // delete dialogs, the workspace rules and contexts popups,
                    // and the delete-confirm and diagnostics modals are all
                    // dismissed here (each only if it is the open one).
                    self.custom_tools_state.close_details();
                    self.custom_roles_state.close_dialogs();
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

    pub fn view<'a>(
        &'a self,
        workspace_map_error: Option<&'a str>,
    ) -> Element<'a, SettingsMessage> {
        // Workspace management section (right column).
        let ws_section = self.workspaces_section(workspace_map_error);

        // User management section (right column, below workspaces)
        let us_section = self.users_section();

        // Custom tools area (right column, below users)
        let tools_section = self.custom_tools_section();

        // Custom roles area (right column, below the custom tools)
        let roles_section = self.custom_roles_section();

        // Existing config sections (left column). The Application section leads
        // when this copy offers either of its switches.
        let mut config_sections = Column::new();
        if let Some(application) = self.application_section() {
            config_sections = config_sections
                .push(application)
                .push(Space::new().height(16));
        }
        let config_sections = config_sections
            .push(self.provider_section())
            .push(Space::new().height(16))
            .push(self.integrations_section())
            .push(Space::new().height(16));
        // With a custom chat endpoint active, provider routing is a no-op —
        // hide the section entirely.
        let config_sections = if !self.custom_endpoint_active_ui() {
            config_sections
                .push(self.routing_section())
                .push(Space::new().height(16))
        } else {
            config_sections
        };
        // Audio section — macOS only; off macOS no section and no spacer is
        // pushed, so nothing is left dangling between Provider Routing and
        // About.
        #[cfg(target_os = "macos")]
        let config_sections = config_sections
            .push(self.audio_section())
            .push(Space::new().height(16));
        let config_sections = config_sections.push(Self::about_section());

        // Two strict 50/50 columns sharing ONE outer vertical scroll (`vscroll`
        // below). Both columns are `FillPortion(1)` inside a single row, so the
        // taller column naturally defines the page length; column separation
        // comes only from the right column's internal left padding (no spacer
        // element).
        let left_col = config_sections;
        let right_col = column![
            ws_section,
            Space::new().height(16),
            us_section,
            Space::new().height(16),
            tools_section,
            Space::new().height(16),
            roles_section,
        ];

        let mut content = column![];

        // Page-level error banner — a full-width band above both columns.
        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner_fill(err));
            content = content.push(Space::new().height(8));
        }

        content = content.push(row![
            container(left_col).width(Length::FillPortion(1)),
            container(right_col)
                .width(Length::FillPortion(1))
                .padding(iced::Padding {
                    left: theme::PAD_16,
                    ..iced::Padding::ZERO
                }),
        ]);

        let scroll = widgets::vscroll(content);

        // The dialog layer sits above the padded page, so its backdrop dims the
        // whole page area. Every change here persists automatically — hence no
        // floating Save button.
        widgets::page_with_dialog(scroll, self.render_modal_overlay())
    }

    // ── Workspace management section ──────────────────────────

    /// Render the workspaces section for the Settings page. No inner
    /// scrollable — rows expand the outer Settings scrollable naturally.
    ///
    /// `map_error` is the Dashboard's workspace-map read failure, passed in so
    /// the footer picker and this list render one state bit rather than two
    /// copies of it.
    #[expect(clippy::too_many_lines)]
    fn workspaces_section<'a>(
        &'a self,
        map_error: Option<&'a str>,
    ) -> Element<'a, SettingsMessage> {
        let ws = &self.workspaces_state;

        let mut rows = Column::new().spacing(theme::SPACE_4);
        rows = widgets::push_error_banner(rows, ws.load_state.error());
        rows = widgets::push_error_banner(rows, map_error);

        if ws.workspaces.is_empty() {
            // A failed map read is not an empty installation — the banner above
            // stands in for the "add one below" empty state.
            if map_error.is_none() {
                rows = rows.push(
                    text("No workspaces configured. Add one below.")
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_MUTED),
                );
            }
        } else {
            for ws_item in &ws.workspaces {
                let ws_row = container(
                    column![
                        // Line 1 — name, status pill, and path on one row.
                        // Maintainer/pause toggles live in the footer only.
                        row![
                            text(&ws_item.name)
                                .size(theme::TEXT_14)
                                .color(theme::TEXT_PRIMARY),
                            widgets::tooltip_hint(
                                widgets::badge_pill(
                                    ws_item.status.to_string(),
                                    theme::workspace_status_color(ws_item.status),
                                    widgets::PILL_MODAL,
                                ),
                                "Status",
                            ),
                            // The path hugs the row's right edge, clipped at the
                            // left when it is too long for the column: the
                            // paragraph's own right alignment keeps both cases
                            // true (`Wrapping::None` keeps it on one line at any
                            // width — iced text has no ellipsis).
                            container(
                                text(&ws_item.path)
                                    .size(theme::TEXT_12)
                                    .color(theme::TEXT_SECONDARY)
                                    .wrapping(text::Wrapping::None)
                                    .width(Length::Fill)
                                    .align_x(text::Alignment::Right),
                            )
                            .width(Length::Fill)
                            .clip(true)
                            .align_y(Alignment::Center),
                        ]
                        .spacing(theme::SPACE_10)
                        .align_y(Alignment::Center),
                        // Line 2 — per-role context icons, general context,
                        // Diag, Rules.
                        {
                            let mut left = Row::new()
                                .spacing(theme::SPACE_4)
                                .align_y(Alignment::Center);
                            let mut role_btns = Row::new().spacing(theme::SPACE_2);
                            for role in Role::iter()
                                .filter(|r| crate::agent::role::role_info(r).has_discovery)
                            {
                                let name = role.as_str();
                                let (color, _bg) = theme::role_badge_color_for(&role);
                                role_btns = role_btns.push(widgets::icon_tooltip_button(
                                    theme::role_icon(&role).size(theme::TEXT_14).color(color),
                                    role.display_label(),
                                    Some(SettingsMessage::WorkspaceMsg(
                                        workspaces::WorkspacesMessage::ViewContext(
                                            ws_item.name.clone(),
                                            name.to_string(),
                                        ),
                                    )),
                                    // Halve the default 10px horizontal
                                    // padding so adjacent role icons sit
                                    // closer; vertical stays at 5px.
                                    [theme::PAD_5, theme::PAD_5],
                                    theme::button_text,
                                    tooltip::Position::Top,
                                ));
                            }
                            left = left.push(role_btns);
                            left = left.push(widgets::icon_tooltip_button(
                                theme::general_context_icon()
                                    .size(theme::TEXT_14)
                                    .color(theme::ACCENT),
                                "General context",
                                Some(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::ViewGeneralContext(
                                        ws_item.name.clone(),
                                    ),
                                )),
                                button::DEFAULT_PADDING,
                                theme::button_text,
                                tooltip::Position::Top,
                            ));
                            left = left.push(widgets::icon_tooltip_button(
                                theme::diagnostics_icon()
                                    .size(theme::TEXT_14)
                                    .color(theme::TEXT_PRIMARY),
                                "Diagnostics commands",
                                Some(SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::ShowDiagnostics(
                                        ws_item.name.clone(),
                                    ),
                                )),
                                button::DEFAULT_PADDING,
                                theme::button_text,
                                tooltip::Position::Top,
                            ));
                            left = left.push(
                                button(text("Rules").size(theme::TEXT_14).color(theme::TEXT_MUTED))
                                    .padding(button::DEFAULT_PADDING)
                                    .style(theme::button_text)
                                    .on_press(SettingsMessage::WorkspaceMsg(
                                        workspaces::WorkspacesMessage::OpenRules(
                                            ws_item.name.clone(),
                                        ),
                                    )),
                            );
                            left
                        },
                    ]
                    .spacing(theme::SPACE_6),
                )
                .padding(theme::PAD_8)
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
                            .padding([theme::PAD_4, theme::PAD_8])
                            .style(theme::pill_style(theme::BG_ELEVATED)),
                    );
                }
            }
        }

        let plus_btn =
            section_plus_button("Add workspace", SettingsMessage::ToggleAddWorkspaceModal);

        let section_content = column![rows];

        section_with_header_action("Workspaces", plus_btn, section_content)
    }

    /// Render the users section for the Settings page.
    #[expect(clippy::too_many_lines)]
    fn users_section(&self) -> Element<'_, SettingsMessage> {
        let us = &self.users_state;

        let mut rows = Column::new().spacing(theme::SPACE_4);

        rows = widgets::push_error_banner(rows, us.load_state.error());

        if us.users.is_empty() {
            // A failed read is not an installation with no users: the banner
            // above is the surface for it, not this empty state.
            if us.load_state.error().is_none() {
                rows = rows.push(
                    text("No users configured. Add one below.")
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_MUTED),
                );
            }
        } else {
            for entry in &us.users {
                let user = &entry.record;
                let is_admin = crate::users::is_admin_name(&user.name);

                let telegram_binding = user.channels.iter().find(|c| c.channel == "telegram");
                let telegram_elem: Element<'_, SettingsMessage> =
                    if let Some(err) = &entry.channels_error {
                        // The channel read failed — surface it instead of the
                        // "Bind Telegram" affordance, which would falsely imply
                        // the user has no binding.
                        widgets::error_banner(err)
                    } else if us.bind_target.as_deref() == Some(&user.name) {
                        // Inline binding input open
                        let mut row_elements: Vec<Element<'_, SettingsMessage>> = vec![
                            text("Telegram:")
                                .size(theme::TEXT_12)
                                .color(theme::TEXT_SECONDARY)
                                .into(),
                            Space::new().width(theme::SPACE_8).into(),
                            widgets::single_line_editor(
                                &us.bind_input.buffer,
                                "@username or id",
                                false,
                                // Fills the left segment so the editor spans
                                // the row; the fixed-size label and
                                // Bind/Cancel buttons floor it.
                                Length::Fill,
                                Some(Id::from(format!("bind_input:{}", user.name))),
                                |action| {
                                    SettingsMessage::UserMsg(users::UsersMessage::BindInputChanged(
                                        action,
                                    ))
                                },
                            ),
                            Space::new().width(theme::SPACE_8).into(),
                        ];
                        row_elements.push(
                            button(
                                text(if us.binding { "Binding..." } else { "Bind" })
                                    .size(theme::TEXT_11),
                            )
                            .style(theme::button_secondary)
                            .on_press_maybe(if us.bind_input.text().trim().is_empty() {
                                None
                            } else {
                                Some(SettingsMessage::UserMsg(users::UsersMessage::SubmitBind(
                                    user.name.clone(),
                                )))
                            })
                            .into(),
                        );
                        row_elements.push(
                            button(text("Cancel").size(theme::TEXT_11))
                                .style(theme::button_secondary)
                                .on_press(SettingsMessage::UserMsg(
                                    users::UsersMessage::CloseBindInput,
                                ))
                                .into(),
                        );
                        Row::with_children(row_elements)
                            .spacing(theme::SPACE_4)
                            .align_y(Alignment::Center)
                            .into()
                    } else if let Some(binding) = telegram_binding {
                        // Bound — a passive label: a muted send icon, then the
                        // identifier as stored (a nickname as is, a numeric
                        // binding as `id <digits>`), then the unbind control.
                        let display = crate::users::settings_binding_label(&binding.identifier);
                        row![
                            lucide::send::<iced::Theme, iced::Renderer>()
                                .size(theme::TEXT_11)
                                .color(theme::TEXT_MUTED),
                            Space::new().width(theme::SPACE_6),
                            text(display)
                                .size(theme::TEXT_12)
                                .color(theme::TEXT_SECONDARY),
                            Space::new().width(theme::SPACE_4),
                            if us.binding {
                                let e: Element<'_, SettingsMessage> = text("Unbinding...")
                                    .size(theme::TEXT_11)
                                    .color(theme::TEXT_MUTED)
                                    .into();
                                e
                            } else {
                                widgets::compact_x_button(
                                    theme::TEXT_SECONDARY,
                                    button::DEFAULT_PADDING,
                                    Some(SettingsMessage::UserMsg(
                                        users::UsersMessage::UnbindChannel(
                                            user.name.clone(),
                                            binding.identifier.clone(),
                                        ),
                                    )),
                                    Some("Unlink Telegram"),
                                )
                            },
                        ]
                        .align_y(Alignment::Center)
                        .into()
                    } else {
                        // No Telegram binding — show bind button
                        button(
                            row![
                                lucide::link::<iced::Theme, iced::Renderer>()
                                    .size(theme::TEXT_11)
                                    .color(theme::BG_BASE),
                                Space::new().width(theme::SPACE_3),
                                text("Bind Telegram").size(theme::TEXT_11),
                            ]
                            .align_y(Alignment::Center),
                        )
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::UserMsg(
                            users::UsersMessage::OpenBindInput(user.name.clone()),
                        ))
                        .into()
                    };

                let segment = row![
                    text(&user.name)
                        .size(theme::TEXT_14)
                        .color(theme::TEXT_PRIMARY),
                    telegram_elem,
                ]
                .spacing(theme::SPACE_8)
                .align_y(Alignment::Center);

                let mut card_body = Column::new().spacing(theme::SPACE_2).push(segment);
                // The admin holds the whole tool catalogue implicitly, so a
                // grant line on the card would misstate what they can call; a
                // user granted nothing gets no line at all.
                if !is_admin && !user.granted_tools.is_empty() {
                    card_body = card_body.push(
                        text(format!(
                            "Custom tools: {}",
                            crate::users::format_grants(&user.granted_tools)
                        ))
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_SECONDARY),
                    );
                }

                let user_row = container(card_body)
                    .width(Length::Fill)
                    .align_x(Alignment::Start)
                    .align_y(Alignment::Center)
                    .padding(theme::PAD_8)
                    .style(theme::surface_card_style);

                // Right-click context menu (Delete). The card's own controls
                // still work: ContextMenu forwards all events to the underlay
                // first; only right-clicks open the menu. The admin is exempt
                // — no context menu at all (an empty menu would render a hollow
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

        let plus_btn = section_plus_button("Add user", SettingsMessage::ToggleAddUserModal);

        section_with_header_action("Users", plus_btn, column![rows])
    }

    /// Render the custom tools area for the Settings page.
    fn custom_tools_section(&self) -> Element<'_, SettingsMessage> {
        let tools = &self.custom_tools_state;

        let mut rows = Column::new().spacing(theme::SPACE_4);
        rows = widgets::push_error_banner(rows, tools.list.error());

        // The empty line stands in only for a successfully read, genuinely
        // empty folder: a failed read keeps its banner and its last list, and
        // before the first read lands nothing is shown for the empty case.
        if tools.list.entries().is_empty() && tools.list.loaded() && tools.list.error().is_none() {
            rows = rows.push(
                text("No custom tools.")
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_MUTED),
            );
        }

        for listing in tools.list.entries() {
            rows = rows.push(Self::custom_tool_card(listing));
        }

        section("Custom Tools", rows)
    }

    /// One custom tool as the area lists it: a small card carrying the tool's
    /// name, plus a warning glyph when its file cannot be read or parsed. The
    /// card itself is the click target — it carries no separate button or link.
    fn custom_tool_card(listing: &ToolListing) -> Element<'_, SettingsMessage> {
        let mut label = Row::new()
            .spacing(theme::SPACE_4)
            .align_y(Alignment::Center)
            .push(
                text(listing.name())
                    .size(theme::TEXT_13)
                    .font(theme::JETBRAINS_MONO)
                    .color(theme::TEXT_PRIMARY),
            );
        if matches!(listing, ToolListing::Broken(_)) {
            label = label.push(
                lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                    .size(theme::TEXT_14)
                    .color(theme::STATUS_WARNING),
            );
        }

        container(
            button(label)
                .width(Length::Fill)
                .padding(theme::PAD_6)
                .style(theme::button_text)
                .on_press(SettingsMessage::CustomToolsMsg(
                    custom_tools::CustomToolsMessage::OpenDetails(listing.name().to_string()),
                )),
        )
        .width(Length::Fill)
        .style(theme::surface_card_style)
        .into()
    }

    /// The read-only details window for one tool: its name, its description and
    /// every parameter (name, type, required/optional, description) — or, for a
    /// file that cannot be read or parsed, the "not usable" statement in place
    /// of them. Nothing is actioned here but closing the window.
    fn custom_tool_dialog(tool: &ToolListing) -> Element<'_, SettingsMessage> {
        let mut body = Column::new().spacing(theme::SPACE_6);
        match tool {
            ToolListing::Broken(_) => {
                body = body.push(
                    text("Not usable — the file cannot be read or its header cannot be parsed")
                        .size(theme::TEXT_13)
                        .color(theme::STATUS_WARNING),
                );
            }
            ToolListing::Usable(entry) => {
                body = body.push(
                    text(&entry.description)
                        .size(theme::TEXT_13)
                        .color(theme::TEXT_SECONDARY)
                        .width(Length::Fill),
                );
                for param in &entry.params {
                    body = body.push(
                        row![
                            text(&param.name)
                                .size(theme::TEXT_12)
                                .font(theme::JETBRAINS_MONO)
                                .color(theme::TEXT_PRIMARY),
                            widgets::badge_pill(
                                param.ty.as_str().to_string(),
                                (theme::TEXT_SECONDARY, theme::HOVER),
                                widgets::PILL_COMPACT,
                            ),
                            widgets::badge_pill(
                                if param.required {
                                    "required"
                                } else {
                                    "optional"
                                }
                                .to_string(),
                                (theme::TEXT_SECONDARY, theme::HOVER),
                                widgets::PILL_COMPACT,
                            ),
                            text(&param.description)
                                .size(theme::TEXT_12)
                                .color(theme::TEXT_SECONDARY)
                                .width(Length::Fill),
                        ]
                        .spacing(theme::SPACE_6)
                        .align_y(Alignment::Center),
                    );
                }
            }
        }

        dialog::dialog_shell(
            column![
                dialog::dialog_title_row(
                    dialog::dialog_title(tool.name()),
                    SettingsMessage::CustomToolsMsg(custom_tools::CustomToolsMessage::CloseDetails),
                ),
                Space::new().height(16),
                // Sized to the details and capped, so a long description or
                // many parameters stay readable by scrolling inside the window.
                container(widgets::vscroll_sized(body, Length::Fill, Length::Shrink))
                    .max_height(320.0),
            ]
            .spacing(theme::SPACE_8)
            .width(Length::Fill),
            620.0,
            24.0,
        )
        .into()
    }

    /// Render the custom roles area for the Settings page: the "Add role"
    /// header action, the list's error banner and one card per record.
    fn custom_roles_section(&self) -> Element<'_, SettingsMessage> {
        let roles = &self.custom_roles_state;

        let mut rows = Column::new().spacing(theme::SPACE_4);
        rows = widgets::push_error_banner(rows, roles.error.as_deref());

        // The empty line stands in only for a successfully read, genuinely
        // empty list: a failed read keeps its banner instead.
        if roles.roles.is_empty() && roles.loaded && roles.error.is_none() {
            rows = rows.push(
                text("No custom roles.")
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_MUTED),
            );
        }

        for role in &roles.roles {
            rows = rows.push(Self::custom_role_card(role));
        }

        section_with_header_action(
            "Custom roles",
            section_plus_button(
                "Add role",
                SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::OpenCreate),
            ),
            rows,
        )
    }

    /// One custom role as the area lists it: its name, its behaviour's label and
    /// how many tools it holds, with the two actions that act on it. The
    /// prompt is deliberately absent — it is the whole definition of a role and
    /// belongs in the editor, not on a card.
    fn custom_role_card(role: &CustomRole) -> Element<'_, SettingsMessage> {
        let card_body = Column::new()
            .spacing(theme::SPACE_2)
            .push(
                text(&role.name)
                    .size(theme::TEXT_13)
                    .font(theme::JETBRAINS_MONO)
                    .color(theme::TEXT_PRIMARY),
            )
            .push(
                row![
                    text(role.behavior.to_string())
                        .size(theme::TEXT_11)
                        .color(theme::TEXT_SECONDARY),
                    text(format!("{} tools", role.tools.len()))
                        .size(theme::TEXT_11)
                        .color(theme::TEXT_SECONDARY),
                ]
                .spacing(theme::SPACE_10),
            );

        container(
            row![
                card_body.width(Length::Fill),
                button(text("Edit").size(theme::TEXT_11))
                    .style(theme::button_secondary)
                    .on_press(SettingsMessage::CustomRolesMsg(
                        custom_roles::CustomRolesMessage::OpenEdit(role.name.clone()),
                    )),
                button(text("Delete").size(theme::TEXT_11))
                    .style(theme::button_danger)
                    .on_press(SettingsMessage::CustomRolesMsg(
                        custom_roles::CustomRolesMessage::DeleteRequest(role.name.clone()),
                    )),
            ]
            .spacing(theme::SPACE_8)
            .align_y(Alignment::Center),
        )
        .width(Length::Fill)
        .padding(theme::PAD_8)
        .style(theme::surface_card_style)
        .into()
    }

    /// The custom-role create/edit window: the name field, the prompt editor,
    /// the branching picker, the tool picker and the window's own error line.
    fn custom_role_editor_window(
        editor: &custom_roles::RoleEditor,
    ) -> Element<'_, SettingsMessage> {
        let is_new = editor.original.is_none();
        let mut middle = Column::new().spacing(theme::SPACE_8);

        // Editing never renames: the name is shown as a static row and the
        // submit targets the record by the name it was opened on.
        if let Some(original) = &editor.original {
            middle = middle.push(field_row(
                "Name",
                text(original.clone()).size(theme::TEXT_13).into(),
                None,
            ));
        }

        // The body gets the room between the fields and the footer, so the list runs
        // down to the submit button; `Shrink` keeps the dialog at the body's own
        // height instead of stretching it. The reported size still has the footer in it.
        middle = middle.push(
            responsive(move |size| Self::custom_role_body(editor, size.height))
                .height(Length::Shrink),
        );

        if let Some(err) = &editor.error {
            middle = middle.push(text(err).size(theme::TEXT_12).color(theme::STATUS_ERROR));
        }

        let name_field = [DialogField {
            label: "Name",
            placeholder: "role name",
            value: &editor.name,
            id: "custom_role_name",
            on_input: |action| {
                SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::Name(action))
            },
        }];
        let fields: &[DialogField] = if is_new { &name_field } else { &[] };

        modal_dialog(
            if is_new {
                "New custom role"
            } else {
                "Edit custom role"
            }
            .to_string(),
            fields,
            Some(middle.into()),
            if is_new { "Add" } else { "Save" },
            editor
                .in_flight
                .then_some(if is_new { BUSY_ADDING } else { BUSY_SAVING }),
            !editor.name.text().is_empty() && !editor.prompt.text().is_empty() && !editor.in_flight,
            SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::CloseEditor),
            SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::SubmitEditor),
        )
    }

    /// One line of `size`px text as iced's `text` lays it out.
    fn text_line_height(size: f32) -> f32 {
        iced::advanced::text::LineHeight::default()
            .to_absolute(iced::Pixels(size))
            .0
    }

    /// What [`modal_dialog`] lays out under its middle slot: the 16px gap and the
    /// footer's button row, a 13px label plus the button's own vertical padding.
    fn dialog_footer_height() -> f32 {
        theme::SPACE_16
            + Self::text_line_height(theme::TEXT_13)
            + 2.0 * iced::widget::button::DEFAULT_PADDING.y()
    }

    /// The room the window's error line takes under the body when a failed save
    /// shows one: the 8px gap and two lines of its 12px text, since save errors
    /// are sentences that wrap at the dialog's width.
    fn custom_role_error_height() -> f32 {
        theme::SPACE_8 + 2.0 * Self::text_line_height(theme::TEXT_12)
    }

    /// The custom-role window's body, no taller than its content and never past
    /// the room `reported` measures: the prompt editor, the branching picker and
    /// the tool list, sized so the list runs down to the footer button and
    /// scrolls inside itself when its rows do not fit.
    fn custom_role_body(
        editor: &custom_roles::RoleEditor,
        reported: f32,
    ) -> Element<'_, SettingsMessage> {
        let error = if editor.error.is_some() {
            Self::custom_role_error_height()
        } else {
            0.0
        };
        let available = (reported - Self::dialog_footer_height() - error).max(0.0);
        let prompt_max = (available - CUSTOM_ROLE_TOOLS_MIN_HEIGHT)
            .clamp(CUSTOM_ROLE_PROMPT_MIN_HEIGHT, CUSTOM_ROLE_PROMPT_MAX_HEIGHT);

        let mut body = Column::new()
            .spacing(theme::SPACE_8)
            .push(Self::custom_role_prompt(editor, prompt_max));

        // How the role branches when a launch names no branch.
        let behavior_pick = pick_list(BEHAVIORS, Some(editor.behavior), |behavior| {
            SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::SetBehavior(behavior))
        })
        .text_size(theme::TEXT_13)
        .style(theme::pick_list_style)
        .menu_style(theme::pick_list_menu_style)
        .width(Length::Fill);
        body = body.push(field_row("Default branching", behavior_pick.into(), None));

        body = body.push(Self::custom_role_tool_picker(editor));

        container(widgets::scroll_h_inset(body))
            .max_height(available)
            .into()
    }

    /// The prompt editor: the administrator's instructions, the whole definition
    /// of the role (the same multi-line editor frame the rules popup uses),
    /// grown to `max_height`.
    fn custom_role_prompt(
        editor: &custom_roles::RoleEditor,
        max_height: f32,
    ) -> Element<'_, SettingsMessage> {
        let prompt_editor = super::editor_widget::EditorWidget::new(&editor.prompt)
            .show_gutter(false)
            .code_mode(false)
            .enter(super::editor_widget::EnterBehavior::Newline)
            .id(iced::widget::Id::from("custom_role_prompt"))
            .keyboard_passthrough(super::is_dashboard_shortcut)
            .placeholder("The instructions the agents of this role must always follow")
            .min_height(CUSTOM_ROLE_PROMPT_MIN_HEIGHT)
            .max_height(max_height)
            .padding(theme::PAD_5);
        container(iced::Element::new(prompt_editor).map(|action| {
            SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::Prompt(action))
        }))
        .width(Length::Fill)
        .height(Length::Shrink)
        .style(theme::elevated_card_style)
        .into()
    }

    /// The custom-role tool list under its label: the window's catalogue plus any
    /// stored name this installation cannot serve right now, marked and labelled
    /// as unavailable. A catalog tool that has a kind vocabulary
    /// ([`crate::tools::grants::kinds_of`]) renders one checkbox per
    /// kind — the tool is granted iff at least one kind is checked — while a
    /// kindless tool keeps its single checkbox. The record's validators re-check
    /// a newly marked entry on save, so "validated while marking" is structural —
    /// what can be marked is what a role may be given; an unserviceable name is
    /// still offered because the launch omits it and the record keeps it, so it
    /// must be visible to be removable.
    fn custom_role_tool_picker(editor: &custom_roles::RoleEditor) -> Element<'_, SettingsMessage> {
        let mut tool_rows = Column::new().spacing(theme::SPACE_2);

        let unavailable = editor
            .tools
            .iter()
            .map(|entry| crate::tools::grants::tool_name(entry))
            .filter(|tool| !editor.catalog.contains(tool))
            .map(str::to_string)
            .collect::<Vec<String>>();
        let catalog = editor.catalog.clone();
        let rows = catalog
            .iter()
            .map(|name| ((*name).to_string(), false))
            .chain(unavailable.into_iter().map(|name| (name, true)));
        for (label, is_unavailable) in rows {
            let shown = if is_unavailable {
                format!("{label} (not available here)")
            } else {
                label.clone()
            };
            match (!is_unavailable)
                .then(|| crate::tools::grants::kinds_of(&label))
                .flatten()
            {
                // A tool with a kind vocabulary: one checkbox per kind, the tool
                // granted iff at least one kind is checked.
                Some(vocab) => {
                    let held = editor.held_kinds(&label);
                    tool_rows = tool_rows.push(
                        text(shown)
                            .size(theme::TEXT_12)
                            .color(theme::TEXT_SECONDARY),
                    );
                    for kind in vocab {
                        let kind = (*kind).to_string();
                        let on = held.iter().any(|k| k == &kind);
                        tool_rows = tool_rows.push(
                            checkbox(on)
                                .label(format!("    {kind}"))
                                .text_size(theme::TEXT_12)
                                .style(theme::checkbox_style)
                                .on_toggle({
                                    let label = label.clone();
                                    let kind = kind.clone();
                                    move |_| {
                                        SettingsMessage::CustomRolesMsg(
                                            custom_roles::CustomRolesMessage::ToggleToolKind {
                                                tool: label.clone(),
                                                kind: kind.clone(),
                                            },
                                        )
                                    }
                                }),
                        );
                    }
                }
                // A kindless tool (or an unavailable stored name): one checkbox.
                None => {
                    tool_rows = tool_rows.push(
                        checkbox(editor.tools.iter().any(|tool| tool == &label))
                            .label(shown)
                            .text_size(theme::TEXT_12)
                            .style(theme::checkbox_style)
                            .on_toggle(move |_| {
                                SettingsMessage::CustomRolesMsg(
                                    custom_roles::CustomRolesMessage::ToggleTool(label.clone()),
                                )
                            }),
                    );
                }
            }
        }
        column![
            text("Tools")
                .size(theme::TEXT_13)
                .color(theme::TEXT_SECONDARY),
            widgets::vscroll(tool_rows),
        ]
        .spacing(theme::SPACE_4)
        .into()
    }

    /// The custom-role delete confirmation: what the record's removal takes
    /// with it.
    fn custom_role_delete_dialog(name: &str) -> Element<'_, SettingsMessage> {
        dialog::confirm_dialog(
            dialog::dialog_title(format!("Delete custom role {name}?")),
            dialog::dialog_body([
                "This removes the role's record. Automations addressed to it can no longer wake \
                 anyone, and the role's continued branches can never be resumed.",
            ]),
            [
                dialog::DialogAction::secondary(
                    "Keep",
                    SettingsMessage::CustomRolesMsg(
                        custom_roles::CustomRolesMessage::DismissDelete,
                    ),
                ),
                dialog::DialogAction::danger(
                    "Delete",
                    SettingsMessage::CustomRolesMsg(
                        custom_roles::CustomRolesMessage::ConfirmDelete,
                    ),
                ),
            ],
        )
        .into()
    }

    /// The overlay the custom roles area owns — the create/edit window, the
    /// delete confirmation, or nothing. The settings page's overlay chain takes
    /// `Option`s, so the closed state is `None` rather than a placeholder.
    fn custom_roles_modal(&self) -> Option<Element<'_, SettingsMessage>> {
        if let Some(editor) = &self.custom_roles_state.editor {
            let dialog = Self::custom_role_editor_window(editor);
            return Some(widgets::modal_backdrop(
                dialog,
                SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::CloseEditor),
                0.5,
            ));
        }
        if let Some(name) = self.custom_roles_state.delete_target.as_deref() {
            let dialog = Self::custom_role_delete_dialog(name);
            return Some(widgets::modal_backdrop(
                dialog,
                SettingsMessage::CustomRolesMsg(custom_roles::CustomRolesMessage::DismissDelete),
                0.5,
            ));
        }
        None
    }

    /// The workspace rules popup: the editor, its counter and Save, and a
    /// failed save reported in place. The corner cross discards the buffer; its
    /// backdrop cannot dismiss it while there is unsaved text — see
    /// `WorkspacesMessage::RulesDismiss`.
    fn rules_dialog(rules: &workspaces::RulesEditor) -> Element<'_, SettingsMessage> {
        let char_count = rules.buffer.text().chars().count();
        let editor: Element<'_, SettingsMessage> = container(
            iced::Element::new(
                super::editor_widget::EditorWidget::new(&rules.buffer)
                    .show_gutter(false)
                    .code_mode(false)
                    .enter(super::editor_widget::EnterBehavior::Newline)
                    .id(Id::from(workspaces::RULES_EDITOR_ID))
                    .keyboard_passthrough(super::is_dashboard_shortcut)
                    .placeholder(format!(
                        "Add rules for every agent in this workspace… (max \
                         {MAX_WORKSPACE_NOTES_CHARS} characters)"
                    ))
                    .min_height(160.0)
                    .max_height(320.0)
                    .padding(theme::PAD_5),
            )
            .map(|action| {
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::RulesEdited(action))
            }),
        )
        .width(Length::Fill)
        .height(Length::Shrink)
        .style(theme::elevated_card_style)
        .into();

        let mut body = column![editor].spacing(theme::SPACE_4);
        if let Some(err) = rules.error.as_deref() {
            body = body.push(widgets::error_banner(err));
        }
        body = body.push(
            row![
                text(format!("{char_count}/{MAX_WORKSPACE_NOTES_CHARS}"))
                    .size(theme::TEXT_11)
                    .color(theme::TEXT_SECONDARY),
                Space::new().width(Length::Fill),
                button(text("Save Rules").size(theme::TEXT_12))
                    .style(theme::button_secondary)
                    .on_press(SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::SaveRules,
                    )),
            ]
            .align_y(Alignment::Center),
        );

        dialog::dialog_shell(
            column![
                dialog::dialog_title_row(
                    dialog::dialog_title(format!("Rules: {}", rules.workspace)),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::RulesDiscard),
                ),
                Space::new().height(16),
                body,
            ]
            .spacing(theme::SPACE_4)
            .width(Length::Fill),
            620.0,
            24.0,
        )
        .into()
    }

    /// The read-only workspace context popup: one role's discovered context or
    /// the workspace's general context, as markdown.
    fn contexts_dialog<'a>(
        context: &'a (
            String,
            workspaces::ContextKind,
            Option<Vec<iced::widget::markdown::Item>>,
        ),
        error: Option<&'a str>,
    ) -> Element<'a, SettingsMessage> {
        let (ws_name, kind, md_items) = context;

        let scope = match kind {
            workspaces::ContextKind::Role(role) => format!("{role} context"),
            workspaces::ContextKind::General => "General context".to_string(),
        };

        let body: Element<'_, SettingsMessage> = match md_items {
            None => container(widgets::loading_text())
                .width(Length::Fill)
                .into(),
            Some(items) => {
                let mut view_col = column![];

                if let Some(err) = error {
                    view_col = view_col.push(widgets::error_banner(err));
                    view_col = view_col.push(Space::new().height(8));
                }

                if items.is_empty() {
                    view_col = view_col.push(
                        text("Not yet discovered")
                            .size(theme::TEXT_13)
                            .color(theme::TEXT_MUTED),
                    );
                } else {
                    let md: Element<'_, SettingsMessage> =
                        super::selectable::markdown::view(items, theme::markdown_settings()).map(
                            |url| {
                                SettingsMessage::WorkspaceMsg(
                                    workspaces::WorkspacesMessage::LinkClicked(url),
                                )
                            },
                        );
                    view_col = view_col.push(
                        container(widgets::vscroll_sized(md, Length::Fill, Length::Shrink))
                            .padding([theme::PAD_4, 0.0])
                            .height(Length::Fixed(300.0))
                            .style(|_| {
                                theme::container_style(theme::BG_BASE, 4.0, 1.0, theme::BORDER)
                            }),
                    );
                }

                view_col.spacing(theme::SPACE_4).into()
            }
        };

        dialog::dialog_shell(
            column![
                dialog::dialog_title_row(
                    dialog::dialog_title(format!("{scope}: {ws_name}")),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::ContextsClose),
                ),
                Space::new().height(16),
                body,
            ]
            .spacing(theme::SPACE_4)
            .width(Length::Fill),
            620.0,
            24.0,
        )
        .into()
    }

    /// Render the Settings modal overlay: whichever of the add-workspace,
    /// add-user, user delete-confirm, workspace diagnostics, workspace rules,
    /// workspace contexts, custom-tool details, custom-role editor /
    /// delete-confirm and Models settings windows is open, or a type-stable
    /// placeholder when none is.
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
        } else if let Some(rules) = self.workspaces_state.rules_editor.as_ref() {
            widgets::modal_backdrop(
                Self::rules_dialog(rules),
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::RulesDismiss),
                0.5,
            )
        } else if let Some(context) = self.workspaces_state.context_view.as_ref() {
            widgets::modal_backdrop(
                Self::contexts_dialog(context, self.workspaces_state.context_view_error.as_deref()),
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::ContextsClose),
                0.5,
            )
        } else if let Some(ref diag_ws_name) = self.workspaces_state.diagnostics_modal {
            let dialog = self.diagnostics_dialog(diag_ws_name);
            widgets::modal_backdrop(
                dialog,
                SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Escape),
                0.5,
            )
        } else if let Some(tool) = self.custom_tools_state.details() {
            let dialog = Self::custom_tool_dialog(tool);
            widgets::modal_backdrop(
                dialog,
                SettingsMessage::CustomToolsMsg(custom_tools::CustomToolsMessage::CloseDetails),
                0.5,
            )
        } else if let Some(roles_modal) = self.custom_roles_modal() {
            roles_modal
        } else if self.models_popup_open {
            let dialog = self.models_dialog();
            widgets::modal_backdrop(dialog, SettingsMessage::CloseModelsPopup, 0.5)
        } else if self.telegram_credentials_open {
            let dialog = self.telegram_credentials_dialog();
            widgets::modal_backdrop(dialog, SettingsMessage::CloseTelegramCredentials, 0.5)
        } else if self.confirm_switch_to_cloud {
            let dialog = Self::switch_to_cloud_dialog();
            widgets::modal_backdrop(dialog, SettingsMessage::DismissSwitchToCloud, 0.5)
        } else {
            // Keep Stack widget type stable
            widgets::no_dialog()
        }
    }

    /// Build the user-deletion confirmation modal.
    ///
    /// The text truthfully states what [`users::UserStore::delete_user`]
    /// does — removes the user row and all channel bindings (access cut
    /// immediately) — and what it preserves (sessions, chat history,
    /// userspace files).
    fn user_delete_dialog(user_name: &str) -> Element<'_, SettingsMessage> {
        dialog::confirm_dialog(
            dialog::dialog_title(format!("Delete user {user_name}?")),
            dialog::dialog_body(["This permanently removes the user and all their \
                 channel bindings (including Telegram) — their access is cut \
                 immediately. Their sessions, chat history, and userspace \
                 files are preserved."]),
            [
                dialog::DialogAction::secondary(
                    "Keep user",
                    SettingsMessage::UserMsg(users::UsersMessage::CancelDelete),
                ),
                dialog::DialogAction::danger(
                    "Delete",
                    SettingsMessage::UserMsg(users::UsersMessage::ConfirmDelete(
                        user_name.to_string(),
                    )),
                ),
            ],
        )
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
            self.add_workspace_adding.then_some(BUSY_ADDING),
            !self.add_workspace_name.text().is_empty()
                && !self.add_workspace_path.text().is_empty(),
            SettingsMessage::ToggleAddWorkspaceModal,
            SettingsMessage::SubmitAddWorkspace,
        )
    }

    /// Build the add-user modal dialog. The account name becomes that account's
    /// personal-workspace folder, so a name the product cannot use is refused
    /// right under the field, in one line — the reserved slot it always occupies
    /// keeps the dialog from jumping as the line comes and goes.
    fn add_user_dialog(&self) -> Element<'_, SettingsMessage> {
        let name = self.add_user_sender.text();
        let problem = if name.is_empty() {
            None
        } else {
            crate::users::new_account_name_problem(
                &name,
                self.users_state
                    .users
                    .iter()
                    .map(|entry| entry.record.name.as_str()),
            )
        };
        let submit_enabled = !name.is_empty() && problem.is_none();
        // The slot is always filled: a lone space lays out at exactly the height
        // of a one-line message, so nothing moves as the red line comes and goes.
        let line = problem.unwrap_or_else(|| " ".to_string());
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
                text(line)
                    .size(theme::TEXT_11)
                    .color(theme::STATUS_ERROR)
                    .into(),
            ),
            "Add",
            self.add_user_adding.then_some(BUSY_ADDING),
            submit_enabled,
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

        let mut rows_col = Column::new().spacing(theme::SPACE_8);

        // Error banner
        if let Some(err) = error {
            rows_col = rows_col.push(widgets::error_banner(err));
        }

        for (i, label) in labels.iter().enumerate() {
            rows_col = rows_col.push(
                row![
                    text(*label)
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_SECONDARY)
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
                .spacing(theme::SPACE_8)
                .align_y(Alignment::Center),
            );
        }

        rows_col = rows_col.push(Space::new().height(8));

        // Action row: [Re-discover] [Save]
        rows_col = rows_col.push(
            row![
                button(row![
                    lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_MUTED),
                    Space::new().width(theme::SPACE_4),
                    text("Re-discover").size(theme::TEXT_12),
                ])
                .style(theme::button_text)
                .on_press(SettingsMessage::WorkspaceMsg(
                    workspaces::WorkspacesMessage::RediscoverDiagnostics(ws_name.clone(),),
                )),
                Space::new().width(Length::Fill),
                button(
                    text(if is_busy { "Working…" } else { "Save" })
                        .size(theme::TEXT_12)
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
            ]
            .align_y(Alignment::Center),
        );

        let modal_title = format!("Diagnostics: {ws_name}");
        dialog::dialog_shell(
            column![
                dialog::dialog_title_row(
                    dialog::dialog_title(modal_title),
                    SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Escape),
                ),
                Space::new().height(16),
                rows_col,
            ]
            .spacing(theme::SPACE_8)
            .width(Length::Fill),
            620.0,
            24.0,
        )
        .into()
    }
    // ── Config-field builders ────────────────────────────────────
    //
    // The field id is always `config:<key>` derived from the `CONFIG_KEY_*`
    // const — `stringify!` of the snake_case field name — no per-call
    // literal ids.

    /// A single inline-editable text config field row (with error placement).
    fn config_text_field<'a>(
        &'a self,
        label: &'static str,
        placeholder: &'static str,
        key: &'static str,
    ) -> Element<'a, SettingsMessage> {
        let field = format!("config:{key}");
        let error = self.field_errors.get(&field).map(String::as_str);
        field_row_with_error(
            label,
            self.config_text_editor(placeholder, key),
            None,
            error,
        )
    }

    /// The editor of a plain-text config field, without the label row — for a
    /// field whose call site lays it out itself.
    fn config_text_editor<'a>(
        &'a self,
        placeholder: &'static str,
        key: &'static str,
    ) -> Element<'a, SettingsMessage> {
        let field = format!("config:{key}");
        widgets::single_line_editor(
            &self.field_editor(&field).buffer,
            placeholder,
            true,
            Length::Fill,
            Some(Id::from(field.clone())),
            move |action| SettingsMessage::ConfigFieldAction { key, action },
        )
    }

    /// A model-slot config field: the field takes the whole column width and
    /// the group's role list is its hint ABOVE it, at the field's left edge.
    fn config_model_slot_field<'a>(
        &'a self,
        placeholder: &'static str,
        key: &'static str,
        role_hint: &'static str,
    ) -> Element<'a, SettingsMessage> {
        let field = format!("config:{key}");
        let error = self.field_errors.get(&field).map(String::as_str);
        let mut col = column![
            inline_label(role_hint, 0.0, theme::TEXT_SECONDARY),
            self.config_text_editor(placeholder, key),
        ]
        .spacing(theme::SPACE_2);
        if let Some(err) = error {
            col = col.push(inline_error(err, 0.0));
        }
        col.into()
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
                Length::Fill,
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

    /// The Application section: what the product does for itself — start with the
    /// login, and update by itself. Each row is offered only where it can do
    /// something: only a published copy is the login target, and only a copy that
    /// downloads its own releases has an update policy to switch. No row at all
    /// means no section.
    fn application_section(&self) -> Option<Element<'_, SettingsMessage>> {
        let mut rows: Vec<Element<'_, SettingsMessage>> = Vec::new();
        if crate::self_update::is_published_copy() {
            // The switch reports the system and nothing else, so a state that is
            // not known yet (or again, after a failed apply or read) says so
            // instead of asserting "off".
            let control: Element<'_, SettingsMessage> = match self.autolaunch {
                Some(state) => {
                    let switch = toggler(state).style(theme::toggler_style);
                    // Inert while an apply runs: a click now would be lost.
                    let switch = if self.autolaunch_applying {
                        switch
                    } else {
                        switch.on_toggle(SettingsMessage::AutolaunchToggle)
                    };
                    switch.into()
                }
                None => text("Checking…")
                    .size(theme::TEXT_13)
                    .color(theme::TEXT_FAINT)
                    .into(),
            };
            rows.push(field_row(
                "Autolaunch",
                control,
                Some("Starts MahBot when you log in"),
            ));
        }
        if crate::self_update::update_mode() == crate::self_update::UpdateMode::Downloaded {
            let autoupdate = crate::config::flag_on(self.config.autoupdate.as_deref());
            rows.push(field_row(
                "Autoupdate",
                toggler(autoupdate)
                    .on_toggle(|on| SettingsMessage::ConfigField {
                        key: CONFIG_KEY_AUTOUPDATE,
                        value: if on { "true" } else { "false" }.to_string(),
                    })
                    .style(theme::toggler_style)
                    .into(),
                Some("Installs newer versions by itself"),
            ));
        }
        if rows.is_empty() {
            return None;
        }
        Some(section("Application", Column::from_iter(rows)))
    }

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
                    .style(theme::toggler_style)
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
            );
            if let Some(w) = self.endpoint_warning.as_ref() {
                endpoint_row = column![endpoint_row, inline_warning(w, FIELD_ROW_INLINE_PAD)]
                    .spacing(theme::SPACE_2)
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

        section(
            "Provider",
            Column::with_children(rows)
                .spacing(theme::SPACE_4)
                .push(Space::new().height(theme::SPACE_4))
                // The model slots and lists live only in the popup this opens.
                .push(
                    button(text("Models settings").size(theme::TEXT_12))
                        .style(theme::button_secondary)
                        .on_press(SettingsMessage::OpenModelsPopup),
                ),
        )
    }

    /// The model settings: the two model slots and the two model lists, with
    /// their add/remove controls and inline errors. Rendered only inside the
    /// Models settings popup — the page keeps no copy of them.
    fn models_body(&self) -> Column<'_, SettingsMessage> {
        column![
            self.config_model_slot_field(
                crate::config::DEFAULT_MANAGER_MODEL,
                CONFIG_KEY_MANAGER_MODEL,
                "Manager, Assistant, Discovery, Engineer, Maintainer",
            ),
            self.config_model_slot_field(
                crate::config::DEFAULT_WORKER_MODEL,
                CONFIG_KEY_WORKER_MODEL,
                "Analyst, Worker, QA, Reviewer, Sanitation",
            ),
            Space::new().height(12),
            widgets::section_subheading("Image Generation"),
            Space::new().height(2),
            model_picker_list(
                ModelPickerTarget::ImageGen,
                self.config.image_gen_models.as_deref(),
                &self.model_picker_inputs[ModelPickerTarget::ImageGen.idx()],
                "model name (e.g. google/gemini-...)",
                self.field_errors
                    .get(CONFIG_FIELD_IMAGE_GEN_MODELS)
                    .map(String::as_str),
            ),
            Space::new().height(12),
            widgets::section_subheading("Video Generation"),
            Space::new().height(2),
            model_picker_list(
                ModelPickerTarget::Video,
                self.config.video_models.as_deref(),
                &self.model_picker_inputs[ModelPickerTarget::Video.idx()],
                "model name (e.g. minimax/hailuo-3)",
                self.field_errors
                    .get(CONFIG_FIELD_VIDEO_MODELS)
                    .map(String::as_str),
            ),
        ]
        .spacing(theme::SPACE_4)
    }

    /// The Models settings popup: the model body, kept within the room the
    /// window leaves under the heading — minus a margin, so the bubble never
    /// spans the window — and scrolled inside itself past that, so the unbounded
    /// lists can never grow it past the window.
    fn models_dialog(&self) -> Element<'_, SettingsMessage> {
        let body = responsive(|size| {
            container(widgets::vscroll_sized(
                self.models_body(),
                Length::Fill,
                Length::Shrink,
            ))
            .max_height((size.height - 2.0 * theme::PAD_16).max(0.0))
            .into()
        })
        .height(Length::Shrink);
        dialog::dialog_shell(
            column![
                dialog::dialog_title_row(
                    dialog::dialog_title("Models settings"),
                    SettingsMessage::CloseModelsPopup,
                ),
                Space::new().height(16),
                body,
            ]
            .width(Length::Fill),
            620.0,
            24.0,
        )
        .into()
    }

    // ── Audio section (macOS only) ───────────────
    //
    // One 'Audio' section with exactly three rows — Transcription, Wake Word
    // Detection, Text to Speech — each with the toggle and an inline status in
    // the same row, and its explanation as a hover hint over the whole row. The
    // wake-word enrollment UI (Enroll button, multi-line progress/Cancel) sits
    // below the three rows. The section does not exist off macOS, where there
    // is no audio subsystem at all.

    #[cfg(target_os = "macos")]
    #[expect(clippy::too_many_lines)]
    fn audio_section(&self) -> Element<'_, SettingsMessage> {
        let transcription_enabled =
            self.config.audio_transcription_use_local.as_deref() != Some("false");
        let voice_enabled = self.config.voice_enabled.as_deref() == Some("true");
        let status = crate::audio::voice::get_status();

        // ── Row 1: Transcription (toggle + inline status + Retry) ──
        let transcription_status: Element<'_, SettingsMessage> = if !transcription_enabled {
            disabled_status()
        } else if crate::audio::local_transcriber::is_loaded() {
            status_label("Ready")
        } else if crate::audio::local_transcriber::is_failed() {
            retry_status_row("Download failed", SettingsMessage::RetryTranscription)
        } else if crate::audio::local_transcriber::is_loading() {
            status_label("Loading…")
        } else {
            status_label("Loads on first use")
        };
        let transcription_row = row_hint(
            field_row(
                "Transcription",
                toggle_status_row(
                    toggler(transcription_enabled)
                        .on_toggle(SettingsMessage::TranscriptionToggle)
                        .style(theme::toggler_style),
                    transcription_status,
                ),
                None,
            ),
            "Voice message transcription",
        );

        // ── Row 2: Wake Word Detection (toggle gated on Transcription + live status) ──
        // Live status: voice status changes are broadcast as runtime-change
        // events, so the rendered pipeline state (listening / enrolling /
        // model error …) is current after each coalesced event.
        // Transcription and the wake word share the loaded ASR model, so with it
        // off the status says why the toggle is dead: a hover hint alone cannot
        // explain a disabled control.
        let wake_status: Element<'_, SettingsMessage> = if !transcription_enabled {
            status_label("Requires Transcription to be enabled")
        } else {
            match status.clone() {
                crate::audio::voice::VoiceStatus::Disabled => disabled_status(),
                crate::audio::voice::VoiceStatus::LoadingModels => status_label("Loading models…"),
                crate::audio::voice::VoiceStatus::ModelError => {
                    retry_status_row("Model error", SettingsMessage::RetryVoiceModels)
                }
                crate::audio::voice::VoiceStatus::Listening => {
                    status_label("Listening for wake word")
                }
                crate::audio::voice::VoiceStatus::RecordingWakeWord => {
                    status_label("Recording command")
                }
                crate::audio::voice::VoiceStatus::RecordingButton => {
                    status_label("Recording voice message")
                }
                crate::audio::voice::VoiceStatus::Transcribing => status_label("Transcribing…"),
                crate::audio::voice::VoiceStatus::MicPermissionDenied => {
                    status_label("Microphone permission denied")
                }
                crate::audio::voice::VoiceStatus::MicDisconnected => {
                    status_label("Microphone disconnected")
                }
                crate::audio::voice::VoiceStatus::Enrolling { .. }
                | crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. }
                | crate::audio::voice::VoiceStatus::EnrollingNegatives { .. } => {
                    status_label("Enrolling…")
                }
                crate::audio::voice::VoiceStatus::Enrolled => status_label("Enrolled"),
                crate::audio::voice::VoiceStatus::Error(msg) => status_label(msg),
            }
        };
        // Gated: wake word can only be enabled while Transcription is ON (they
        // share the loaded ASR model). Turning Transcription OFF cascades it off.
        let wake_row = row_hint(
            field_row(
                "Wake Word Detection",
                toggle_status_row(
                    toggler(voice_enabled)
                        .on_toggle_maybe(if transcription_enabled {
                            Some(SettingsMessage::VoiceToggle)
                        } else {
                            None
                        })
                        .style(theme::toggler_style),
                    wake_status,
                ),
                None,
            ),
            "Listen for the command",
        );

        // ── Row 3: Text to Speech (toggle + inline status in one row) ──
        let tts_enabled = self.config.tts_enabled.as_deref() == Some("true");
        let tts_ready = crate::audio::tts::models_ready();
        let tts_failed = crate::audio::tts::download_failed();
        let audio_ok = crate::audio::tts::audio_output_ready();
        let tts_status: Element<'_, SettingsMessage> = if !tts_enabled {
            disabled_status()
        } else if tts_ready {
            if audio_ok {
                status_label("Ready")
            } else {
                Text::new("No audio output device")
                    .size(theme::TEXT_13)
                    .color(theme::STATUS_ERROR)
                    .into()
            }
        } else if tts_failed {
            retry_status_row("Model download failed", SettingsMessage::TtsRetryModels)
        } else {
            status_label("Downloading models…")
        };
        let tts_row = row_hint(
            field_row(
                "Text to Speech",
                toggle_status_row(
                    toggler(tts_enabled)
                        .on_toggle(SettingsMessage::TtsToggle)
                        .style(theme::toggler_style),
                    tts_status,
                ),
                None,
            ),
            "Let assistant talk",
        );

        // ── Wake-word enrollment controls (below the three rows) ──
        let enroll_btn: Element<'_, SettingsMessage> = if voice_enabled && transcription_enabled {
            container(
                button(Text::new("Enroll Wake Word").size(theme::TEXT_13))
                    .on_press(SettingsMessage::StartVoiceEnrollment)
                    .style(theme::button_secondary)
                    .padding(theme::PAD_6),
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

                Some(enrollment_status_column(lines.join("\n")))
            }
            crate::audio::voice::VoiceStatus::ListeningDuringEnrollment { .. } => {
                Some(enrollment_status_column("Listening…".to_string()))
            }
            crate::audio::voice::VoiceStatus::WaitingForSilenceDuringEnrollment { .. } => Some(
                enrollment_status_column("Keep silent to confirm…".to_string()),
            ),
            crate::audio::voice::VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs,
                wall_clock_elapsed,
            } => {
                let pct = (accumulated_secs * 100)
                    .checked_div(target_secs)
                    .unwrap_or(0);
                Some(enrollment_status_column(format!(
                    "Collecting negative samples… {accumulated_secs}s/{target_secs}s \
                     ({pct}%) elapsed {wall_clock_elapsed}s"
                )))
            }
            _ => None,
        };

        let mut column = Column::new()
            .push(transcription_row)
            .push(wake_row)
            .push(tts_row);
        if voice_enabled && transcription_enabled {
            column = column.push(Space::new().height(8));
            column = column.push(enroll_btn);
        }
        if let Some(ui) = enrollment_ui {
            column = column.push(ui);
        }

        section("Audio", column)
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
        .text_size(theme::TEXT_13)
        .style(theme::pick_list_style)
        .menu_style(theme::pick_list_menu_style)
        .width(Length::Fill);

        let provider_row = field_row_with_error(
            "Web Search Provider",
            pick_list.into(),
            None,
            self.field_errors
                .get(CONFIG_FIELD_WEB_SEARCH_PROVIDER)
                .map(String::as_str),
        );

        let mut integrations = column![
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
        ];
        if let Some(area) = self.telegram_local_area() {
            integrations = integrations.push(area);
        }
        section("Integrations", integrations)
    }

    /// The Telegram Bot API mode area of the Integrations section: one line
    /// carrying the mode serving the bot, what it is doing, and the controls this
    /// state allows — or nothing at all: a healthy local server has nothing to
    /// report, and without a bot token there is no bot to serve.
    ///
    /// The state is read at render time: every change of the feature publishes
    /// [`crate::runtime_events::RuntimeEvent::TelegramLocal`], which the
    /// dashboard turns into a repaint, so there is no poll loop here.
    fn telegram_local_area(&self) -> Option<Element<'_, SettingsMessage>> {
        CONFIG.telegram_bot_token()?;
        let status = telegram_local::status();
        // A switch actually in flight, or a request this page just started:
        // `busy` disables the controls that would start a second switch. A switch
        // waiting for the owner (blocked with a cause) is not busy by this
        // definition.
        let busy = self.telegram_switch_busy || (status.migrating && status.blocked.is_none());

        // Silence is the whole report of a working local setup: not even an empty
        // row, which would leave a gap in the section.
        let state = telegram_line(&status)?;

        let mut summary = text(state).size(theme::TEXT_13).width(Length::Fill);
        if status.last_error.is_some() {
            summary = summary.color(theme::STATUS_ERROR);
        }

        let mut line = Row::new()
            .spacing(theme::SPACE_8)
            .align_y(Alignment::Center)
            .width(Length::Fill)
            .push(summary);
        if busy {
            // The switch reports progress through the same line, so the dot alone
            // is enough: the text beside it already says where the bot is going.
            line = line.push(super::activity_dot());
        }
        Some(line.push(self.telegram_line_controls(&status, busy)).into())
    }

    /// The controls carried by the mode line. Every one of them stays on that
    /// line, and each is rendered only in the states it resolves: the way into
    /// the local mode, the repair a blocked switch needs, and the way back to the
    /// cloud.
    fn telegram_line_controls(
        &self,
        status: &telegram_local::Status,
        busy: bool,
    ) -> Element<'static, SettingsMessage> {
        // Telegram refused the pasted pair — the switch blocks on the same test,
        // and only that outcome is fixed by entering the pair again. Only a switch
        // towards the local server can block on the pair: the dialog's submit is
        // what asks for that mode, so it would reverse any other direction.
        let rejected = status.target == Mode::Local
            && status
                .last_error
                .as_deref()
                .is_some_and(telegram_local::credentials_rejected);
        let control = |label: &'static str, on_press| {
            button(text(label).size(theme::TEXT_12))
                .style(theme::button_secondary)
                .on_press_maybe(on_press)
        };
        let credentials_again = || {
            control(
                "Enter api_id/api_hash again",
                Some(SettingsMessage::OpenTelegramCredentials),
            )
        };

        let mut controls = Row::new()
            .spacing(theme::SPACE_8)
            .align_y(Alignment::Center);

        // The way in: nothing else in the product offers it, so it is always on
        // the cloud line — disabled rather than removed while a switch is under
        // way. A switch heading local carries its own retry instead of a second
        // control that would ask for the same thing.
        if status.serving == Mode::Cloud && !status.migrating {
            controls = controls.push(control(
                "Use a local Bot API server",
                (!busy).then_some(SettingsMessage::OpenTelegramCredentials),
            ));
        }
        // The repair: a refused pair is re-entered through the same dialog, whose
        // submit resumes the switch; every other cause is re-run as the direction
        // the journal already holds.
        if !busy && status.migrating {
            controls = controls.push(if rejected {
                credentials_again()
            } else {
                control(
                    match status.target {
                        Mode::Local => "Try switching again",
                        Mode::Cloud => "Try switching back again",
                    },
                    Some(SettingsMessage::RetrySwitch),
                )
            });
        } else if !busy && status.serving == Mode::Local && (rejected || !status.credentials_set) {
            // Settled on the local server with a pair that cannot work: the same
            // dialog is the way on.
            controls = controls.push(credentials_again());
        }
        // The way out is offered for a failure of the local server and nothing
        // else: a working setup needs no way back, and neither does a switch that
        // has not failed, however long it runs.
        if telegram_escape(status) {
            controls = controls.push(control(
                "Return to the cloud",
                (!self.telegram_switch_busy).then_some(SettingsMessage::RequestSwitchToCloud),
            ));
        }
        controls.into()
    }

    /// Build the api_id/api_hash dialog.
    ///
    /// The body is passed as [`modal_dialog`]'s `middle` because each field
    /// carries a hint and its validator's message, which the shared field rows
    /// do not — [`credentials_dialog_field`] builds them.
    fn telegram_credentials_dialog(&self) -> Element<'_, SettingsMessage> {
        let api_id = self.telegram_api_id_input.text();
        let api_hash = self.telegram_api_hash_input.text();
        let (id_error, hash_error) = telegram_local::credential_problems(&api_id, &api_hash);
        let busy = self.telegram_switch_busy;
        // The shape check only gates the button: nothing is downloaded or
        // started until the owner submits.
        let ready = !busy && id_error.is_none() && hash_error.is_none();

        // Both halves of the pair are secrets and are rendered masked: this
        // dialog is the only place either is ever shown, and never unmasked
        // without the eye toggle.
        let id_input = widgets::password_field_editor(
            &self.telegram_api_id_input.buffer,
            "1234567",
            self.password_visible
                .contains(&PasswordTarget::TelegramApiId),
            Length::Fixed(375.0),
            false,
            Some(Id::new("telegram_api_id")),
            SettingsMessage::TelegramApiIdInput,
            SettingsMessage::TogglePasswordVisibility(PasswordTarget::TelegramApiId),
        );
        let hash_input = widgets::password_field_editor(
            &self.telegram_api_hash_input.buffer,
            "0123456789abcdef0123456789abcdef",
            self.password_visible
                .contains(&PasswordTarget::TelegramApiHash),
            Length::Fixed(375.0),
            false,
            Some(Id::new("telegram_api_hash")),
            SettingsMessage::TelegramApiHashInput,
            SettingsMessage::TogglePasswordVisibility(PasswordTarget::TelegramApiHash),
        );

        let mut body = Column::new()
            .push(dialog::dialog_body([
                "MahBot can serve the bot from its own Bot API server instead of Telegram's \
                 cloud API: up to 2000 MB can be sent, and incoming files are no longer \
                 refused at 20 MB.",
                "Paste the api_id and api_hash of a Telegram application from \
                 my.telegram.org. They are stored with the product's settings and are only \
                 ever shown masked.",
            ]))
            .push(Space::new().height(16))
            .push(credentials_dialog_field(
                "api_id",
                id_input,
                "the number shown on my.telegram.org",
                id_error,
            ))
            .push(Space::new().height(theme::SPACE_8))
            .push(credentials_dialog_field(
                "api_hash",
                hash_input,
                "the 32-character value next to it",
                hash_error,
            ));

        if let Some(error) = self.telegram_credentials_error.as_deref() {
            body = body
                .push(Space::new().height(theme::SPACE_8))
                .push(inline_error(error, 0.0));
        }

        modal_dialog(
            "Local Telegram Bot API server".to_string(),
            &[],
            Some(body.into()),
            "Switch to the local server",
            busy.then_some(BUSY_SWITCHING),
            ready,
            SettingsMessage::CloseTelegramCredentials,
            SettingsMessage::SubmitTelegramCredentials,
        )
    }

    /// The confirmation for giving up the local server — the one action here
    /// that deletes something.
    fn switch_to_cloud_dialog() -> Element<'static, SettingsMessage> {
        dialog::confirm_dialog(
            dialog::dialog_title("Switch back to the cloud Bot API?"),
            dialog::dialog_body([
                "The bot returns to Telegram's cloud API, and the local Bot API server is \
                 removed together with everything it holds: the installed server, its state \
                 directory and the bot's login there. Files are limited by the cloud again — \
                 20 MB to the bot, 50 MB from it.",
            ]),
            [
                dialog::DialogAction::secondary(
                    "Keep the local server",
                    SettingsMessage::DismissSwitchToCloud,
                ),
                dialog::DialogAction::danger("Switch back", SettingsMessage::ConfirmSwitchToCloud),
            ],
        )
        .into()
    }

    /// About block — version and how this copy of it was obtained, on one line.
    fn about_section() -> Element<'static, SettingsMessage> {
        // Bound once, so the classification is asked for one value per render.
        let mode = crate::self_update::update_mode();
        let origin = match mode {
            crate::self_update::UpdateMode::Downloaded => "Downloaded copy",
            crate::self_update::UpdateMode::SourceTree => "Copy built from sources",
        };
        row![
            image(crate::app_icon::widget_image()).width(20).height(20),
            text(format!("MahBot v{}", crate::self_update::VERSION))
                .size(theme::TEXT_13)
                .color(theme::TEXT_FAINT),
            text(origin).size(theme::TEXT_11).color(theme::TEXT_FAINT)
        ]
        .spacing(theme::SPACE_8)
        .align_y(Alignment::Center)
        .into()
    }

    fn routing_section(&self) -> Element<'_, SettingsMessage> {
        // The two routable model slots (manager and worker) — saved routing
        // rows for models outside these slots are not rendered (they are inert
        // orphans).
        let model_names = routing_slots(&self.config);

        let mut rows: Vec<Element<'_, SettingsMessage>> = Vec::new();
        for (i, model_name) in model_names.iter().enumerate() {
            // `section_impl` clamps the content spacing to 4px, so a 4px
            // spacer row between pairs yields the 12px group gap while each
            // label sits tight (2px) above its own input.
            if i > 0 {
                rows.push(Space::new().height(4).into());
            }
            let display_name = model_name.clone();
            let order_model = model_name.clone();
            let order_field = format!("routing_order:{model_name}");
            let order_submit = SettingsMessage::ConfigFieldSettleNow {
                field: order_field.clone(),
            };
            let order_error = self.field_errors.get(&order_field).map(String::as_str);
            // Placeholder is the provider an empty (unset/cleared) field falls
            // back to — the same value the request will carry.
            let placeholder = crate::config::default_model_routing(model_name).unwrap_or("");
            let order_input: Element<'_, SettingsMessage> = widgets::single_line_editor(
                &self.field_editor(&order_field).buffer,
                placeholder,
                true,
                Length::Fill,
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
                // Model name label (read-only) — the same face as the image and
                // video model rows.
                text(display_name).size(theme::TEXT_12),
                order_input,
            ]
            .spacing(theme::SPACE_2);

            rows.push(if let Some(err) = order_error {
                column![row, inline_error(err, 0.0)]
                    .spacing(theme::SPACE_2)
                    .into()
            } else {
                row.into()
            });
        }

        section("OpenRouter Provider Routing", Column::from_iter(rows))
    }
}

// ── Shared widgets ───────────────────────────────────────────────

/// Section heading (secondary-colored monospace title above the content).
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
    let styled_title = widgets::section_heading(title);

    let header: Element<'a, SettingsMessage> = match action {
        Some(btn) => row![styled_title, Space::new().width(Length::Fill), btn,]
            .align_y(Alignment::Center)
            .into(),
        None => styled_title.into(),
    };

    column![
        header,
        Space::new().height(4),
        content.spacing(theme::SPACE_4)
    ]
    .spacing(theme::SPACE_2)
    .into()
}

/// Left pad aligning inline messages/hints with a field row's input column:
/// the fixed 180px label width plus the 8px row gap.
const FIELD_ROW_INLINE_PAD: f32 = 188.0;

/// Label on the left, input on the right, optional hint below.
fn field_row<'a>(
    label: &'static str,
    input: Element<'a, SettingsMessage>,
    hint: Option<&'static str>,
) -> Element<'a, SettingsMessage> {
    field_row_with_error(label, input, hint, None)
}

/// Small inline status label in the given color, indented `left_pad` px to
/// align with the input column — shared body of [`inline_error`],
/// [`inline_warning`] and the model-slot role-list hint above the input.
fn inline_label<'a>(
    msg: impl IntoFragment<'a>,
    left_pad: f32,
    color: iced::Color,
) -> Element<'a, SettingsMessage> {
    container(text(msg).size(theme::TEXT_10).color(color))
        .padding(iced::Padding::default().left(left_pad))
        .into()
}

/// Plain-text inline status cell at the default text size — shared body of
/// the audio-row status arms.
#[cfg(target_os = "macos")]
fn status_label<'a>(msg: impl IntoFragment<'a>) -> Element<'a, SettingsMessage> {
    Text::new(msg).size(theme::TEXT_13).into()
}

/// The inline error label rendered under a control, in the error color. Takes a
/// borrowed or an owned message: a field's shape problem is computed at render
/// time and cannot be borrowed from the page state.
fn inline_error<'a>(err: impl IntoFragment<'a>, left_pad: f32) -> Element<'a, SettingsMessage> {
    inline_label(err, left_pad, theme::STATUS_ERROR)
}

/// The inline warning label rendered under a control, in the warning color.
/// Non-fatal — the value was still saved (e.g. an unreachable custom endpoint).
fn inline_warning(msg: &str, left_pad: f32) -> Element<'_, SettingsMessage> {
    inline_label(msg, left_pad, theme::STATUS_WARNING)
}

/// The gray "Disabled" inline status label shared by the audio rows.
#[cfg(target_os = "macos")]
fn disabled_status() -> Element<'static, SettingsMessage> {
    text("Disabled")
        .size(theme::TEXT_13)
        .color(theme::TEXT_SECONDARY)
        .into()
}

/// A hover hint over the whole strip of a settings row: the row is stretched to
/// the column so the hint answers the title, the toggle, the status and the empty
/// tail alike, unlike a hint snug around the row's own content.
#[cfg(target_os = "macos")]
fn row_hint<'a>(
    row: Element<'a, SettingsMessage>,
    hint: &'static str,
) -> Element<'a, SettingsMessage> {
    widgets::tooltip_hint(container(row).width(Length::Fill), hint)
}

/// Toggle + inline status side by side — the shared frame of the three
/// audio rows (Transcription / Wake Word / Text to Speech).
#[cfg(target_os = "macos")]
fn toggle_status_row<'a>(
    toggler: impl Into<Element<'a, SettingsMessage>>,
    status: impl Into<Element<'a, SettingsMessage>>,
) -> Element<'a, SettingsMessage> {
    row![
        toggler.into(),
        Space::new().width(theme::SPACE_12),
        status.into()
    ]
    .align_y(Alignment::Center)
    .into()
}

/// Inline "status label + Retry" row for a failed download / model load.
#[cfg(target_os = "macos")]
fn retry_status_row(
    label: &'static str,
    on_retry: SettingsMessage,
) -> Element<'static, SettingsMessage> {
    row![
        text(label).size(theme::TEXT_14),
        Space::new().width(theme::SPACE_8),
        button(text("   Retry   ").size(theme::TEXT_13))
            .on_press(on_retry)
            .style(theme::button_danger)
            .padding(theme::PAD_4),
    ]
    .align_y(Alignment::Center)
    .into()
}

/// Accent "+" header button shared by the Workspaces and Users section
/// headers.
fn section_plus_button(
    tooltip_text: &'static str,
    on_press: SettingsMessage,
) -> Element<'static, SettingsMessage> {
    widgets::icon_tooltip_button(
        lucide::plus::<iced::Theme, iced::Renderer>()
            .size(theme::TEXT_16)
            .color(theme::ACCENT),
        tooltip_text,
        Some(on_press),
        button::DEFAULT_PADDING,
        theme::button_text,
        tooltip::Position::Top,
    )
}

/// The Cancel button closing each live-enrollment progress column.
#[cfg(target_os = "macos")]
fn cancel_enrollment_button() -> Element<'static, SettingsMessage> {
    container(
        button(text("Cancel").size(theme::TEXT_13))
            .on_press(SettingsMessage::CancelVoiceEnrollment)
            .style(theme::button_danger)
            .padding(theme::PAD_6),
    )
    .into()
}

/// Status text framed by vertical spacers above the Cancel button — the
/// shared skeleton of the four active-enrollment status columns.
#[cfg(target_os = "macos")]
fn enrollment_status_column(status: String) -> Element<'static, SettingsMessage> {
    Column::new()
        .push(Space::new().height(8))
        .push(Text::new(status).size(theme::TEXT_13))
        .push(Space::new().height(8))
        .push(cancel_enrollment_button())
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
        text(label).size(theme::TEXT_13).width(Length::Fixed(180.0)),
        Space::new().width(theme::SPACE_8),
        input,
    ]
    .align_y(Alignment::Center);

    if let Some(h) = hint {
        row_widget = row_widget.push(Space::new().width(theme::SPACE_8));
        row_widget = row_widget.push(text(h).size(theme::TEXT_10).color(theme::TEXT_SECONDARY));
    }

    let row_elem: Element<'a, SettingsMessage> = row_widget.into();
    if let Some(err) = error {
        column![row_elem, inline_error(err, FIELD_ROW_INLINE_PAD),]
            .spacing(theme::SPACE_2)
            .into()
    } else {
        row_elem
    }
}

/// One field of the api_id/api_hash dialog: the shared label/input row, the
/// hint under it, and the validator's message about this field below that.
fn credentials_dialog_field<'a>(
    label: &'static str,
    input: Element<'a, SettingsMessage>,
    hint: &'static str,
    error: Option<String>,
) -> Element<'a, SettingsMessage> {
    let mut col = column![
        field_row(label, input, None),
        inline_label(hint, FIELD_ROW_INLINE_PAD, theme::TEXT_SECONDARY),
    ]
    .spacing(theme::SPACE_2);
    if let Some(err) = error {
        let line: Element<'a, SettingsMessage> = inline_error(err, FIELD_ROW_INLINE_PAD);
        col = col.push(line);
    }
    col.into()
}

/// Delete confirmation prompt — the inline "Delete? Yes / No" row shown
/// below a workspace card when the row is the delete target.
fn delete_confirm_button<'a>(
    on_confirm: SettingsMessage,
    on_cancel: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    row![
        text("Delete?")
            .size(theme::TEXT_12)
            .color(theme::STATUS_ERROR),
        Space::new().width(theme::SPACE_4),
        button(text("Yes").size(theme::TEXT_11).color(theme::STATUS_ERROR))
            .style(theme::button_danger)
            .on_press(on_confirm),
        Space::new().width(theme::SPACE_4),
        button(text("No").size(theme::TEXT_11))
            .style(theme::button_secondary)
            .on_press(on_cancel),
    ]
    .into()
}

/// Read the system's registration off the update thread. `None` when the read
/// itself failed — unknown, which is not the same as "off".
async fn read_autolaunch() -> Option<bool> {
    tokio::task::spawn_blocking(crate::autostart::registered)
        .await
        .ok()
        .flatten()
}

/// Persist the owner's Autolaunch decision and apply it to the system's own
/// registration in one go: the decision first, so a registration failure never
/// loses the choice. Returns what the system reports afterwards.
async fn save_and_apply_autolaunch(enabled: bool) -> Result<Option<bool>, String> {
    let value = if enabled { "true" } else { "false" };
    crate::config::persist_settled_string_field(crate::config::CONFIG_KEY_AUTOLAUNCH, value)
        .await
        .map_err(|e| format!("The Autolaunch setting could not be saved: {e}"))?;
    tokio::task::spawn_blocking(move || crate::autostart::set_enabled(enabled))
        .await
        .map_err(|e| format!("The login registration could not be changed: {e}"))?
        .map_err(|e| format!("The login registration could not be changed: {e:#}"))
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

/// Save the pair the owner pasted, then ask for the switch to the local server.
///
/// api_id is written first and a failed save stops before the request, so a
/// switch never starts against a half-saved pair; [`telegram_local`] itself
/// re-checks the shape of both before it journals anything. The error names the
/// field that failed and never carries a value.
async fn save_credentials_and_switch_local(api_id: String, api_hash: String) -> Result<(), String> {
    // Saved as typed (trimmed): telegram_local::credentials is the one place that
    // normalises what launches, so the pair has a single spelling there.
    let api_id = api_id.trim();
    let api_hash = api_hash.trim();
    let previous_id = crate::config::CONFIG.telegram_api_id();
    crate::config::persist_settled_string_field(CONFIG_KEY_TELEGRAM_API_ID, api_id)
        .await
        .map_err(|e| format!("api_id could not be saved: {e}"))?;
    if let Err(e) =
        crate::config::persist_settled_string_field(CONFIG_KEY_TELEGRAM_API_HASH, api_hash).await
    {
        // Half a pair is worse than none: the new id beside the old hash would be
        // fingerprinted and launched as if it were the pair that was checked, so the
        // id is put back before the error reaches the dialog.
        let _ = crate::config::persist_settled_string_field(
            CONFIG_KEY_TELEGRAM_API_ID,
            previous_id.as_deref().unwrap_or_default(),
        )
        .await;
        return Err(format!("api_hash could not be saved: {e}"));
    }
    telegram_local::request_switch(Mode::Local)
        .await
        .map_err(|e| e.to_string())
}

/// The progress labels a dialog's submit button shows while its work is in flight:
/// one spelling per family, so the button cannot drift from what is happening.
const BUSY_ADDING: &str = "Adding…";
const BUSY_SAVING: &str = "Saving…";
const BUSY_SWITCHING: &str = "Switching…";

/// Configuration for a single text field in [`modal_dialog`].
struct DialogField<'a> {
    label: &'static str,
    placeholder: &'static str,
    value: &'a super::editor_widget::EditorBuffer,
    /// Stable widget id for the field (unique across fields on the page).
    id: &'static str,
    /// Function pointer for the shared-editor `on_action` handler.
    ///
    /// The fields of one dialog sit in a single `&[DialogField]`, so the
    /// handler type cannot be a type parameter; a `fn` pointer keeps the slice
    /// homogeneous without boxing. Message constructors and non-capturing
    /// closures both coerce.
    on_input: fn(EditorAction) -> SettingsMessage,
}

/// Build a reusable modal dialog: its heading with the corner close cross,
/// optional field rows, optional `middle` content, and a submit footer. `busy`
/// is the label the submit button shows while the action is in flight (`None`
/// when it is idle); the button is disabled while `busy` is set or
/// `submit_enabled` is false. The cross dismisses the dialog with `on_dismiss`.
///
/// Layout: heading, spacer(16), field rows (8 px between), spacer(16),
/// middle (if any), spacer(16), footer.
#[expect(clippy::too_many_arguments)]
fn modal_dialog<'a>(
    title: String,
    fields: &[DialogField<'a>],
    middle: Option<Element<'a, SettingsMessage>>,
    submit_label: &'static str,
    busy: Option<&'static str>,
    submit_enabled: bool,
    on_dismiss: SettingsMessage,
    on_submit: SettingsMessage,
) -> Element<'a, SettingsMessage> {
    let mut col = Column::new();

    col = col.push(dialog::dialog_title_row(
        dialog::dialog_title(title),
        on_dismiss,
    ));

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

    let submit_btn = button(text(busy.unwrap_or(submit_label)).size(theme::TEXT_13))
        .style(theme::button_secondary)
        .on_press_maybe(if busy.is_some() || !submit_enabled {
            None
        } else {
            Some(on_submit)
        });
    col = col.push(dialog::dialog_footer_row([submit_btn.into()]));

    dialog::dialog_shell(col, 620.0, 24.0).into()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Model list editing: parse_models / add_model_to_list / remove_model_from_list ──

    /// Parsing trims every line and skips blank or whitespace-only ones.
    #[test]
    fn parse_models_cases() {
        let cases: &[(&str, Option<&str>, &[&str])] = &[
            ("None returns empty", None, &[]),
            ("empty string returns empty", Some(""), &[]),
            (
                "single line",
                Some("google/gemini-3.1-flash-image-preview"),
                &["google/gemini-3.1-flash-image-preview"],
            ),
            ("multiple lines", Some("m1\nm2\nm3"), &["m1", "m2", "m3"]),
            ("trims whitespace", Some("  m1  \n  m2  "), &["m1", "m2"]),
            ("skips empty lines", Some("m1\n\n\nm2"), &["m1", "m2"]),
            (
                "skips whitespace-only lines",
                Some("m1\n   \nm2"),
                &["m1", "m2"],
            ),
        ];
        for &(name, raw, expected) in cases {
            let expected: Vec<String> = expected.iter().map(ToString::to_string).collect();
            assert_eq!(parse_models(raw), expected, "case {name}");
        }
    }

    /// Adding trims its input and clears the buffer — a blank input does nothing at all —
    /// and a duplicate never lands twice.
    #[test]
    fn add_model_to_list_cases() {
        let cases: &[(&str, &str, Option<&str>, Option<&str>)] = &[
            ("empty input does nothing", "", None, None),
            ("whitespace input does nothing", "  ", None, None),
            ("adds to empty list", "model-a", None, Some("model-a")),
            (
                "adds to existing list",
                "model-c",
                Some("model-a\nmodel-b"),
                Some("model-a\nmodel-b\nmodel-c"),
            ),
            (
                "skips duplicates",
                "model-a",
                Some("model-a\nmodel-b"),
                Some("model-a\nmodel-b"),
            ),
            (
                "trims input",
                "  model-a  ",
                Some("model-b"),
                Some("model-b\nmodel-a"),
            ),
        ];
        for &(name, raw, initial, expected) in cases {
            let mut input = SingleLineEditorState::new(raw);
            let mut list = initial.map(String::from);
            add_model_to_list(&mut input, &mut list);
            assert_eq!(list, expected.map(String::from), "case {name}: list");
            if raw.trim().is_empty() {
                assert_eq!(input.text(), raw, "case {name}: input kept as typed");
            } else {
                assert!(input.text().is_empty(), "case {name}: input cleared");
            }
        }
    }

    /// Removing drops the matching entry; an emptied list becomes `None`.
    #[test]
    fn remove_model_from_list_cases() {
        let cases: &[(&str, &str, Option<&str>, Option<&str>)] = &[
            (
                "removes matching model",
                "model-b",
                Some("model-a\nmodel-b\nmodel-c"),
                Some("model-a\nmodel-c"),
            ),
            (
                "non-matching removal keeps list",
                "model-d",
                Some("model-a\nmodel-b\nmodel-c"),
                Some("model-a\nmodel-b\nmodel-c"),
            ),
            ("last entry clears list", "model-a", Some("model-a"), None),
            ("empty list stays empty", "model-a", None, None),
        ];
        for &(name, model, initial, expected) in cases {
            let mut list = initial.map(String::from);
            remove_model_from_list(model, &mut list);
            assert_eq!(list, expected.map(String::from), "case {name}: list");
        }
    }

    // ── Toggle generation counter & rollback (shared) ──────────────
    // macOS-only: both toggles drive the audio pipeline.

    /// Parameterised by message constructors and field accessors so the
    /// same 7-scenario sequence (toggle ON, stale result, DB error revert,
    /// re-toggle, successful persist, stale-old-gen, toggle OFF + revert)
    /// is exercised exactly once per toggle without code duplication.
    #[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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

    // ── Application: the Autolaunch switch's shown state ─────────────
    //
    // The switch reports the system's answer and nothing else: it never asserts a
    // state it does not know, and an answer to a read the owner has already
    // clicked past is dropped instead of painting the state the click replaced.
    // State transitions only — the spawned tasks are dropped, never awaited, so no
    // store or launchctl is touched.

    #[test]
    fn the_autolaunch_switch_only_ever_shows_what_the_system_answered() {
        let mut state = SettingsState::new();
        assert_eq!(state.autolaunch, None, "nothing is known before the answer");

        let _ = state.update(SettingsMessage::AutolaunchRead {
            epoch: 0,
            state: Some(true),
        });
        assert_eq!(state.autolaunch, Some(true));

        // A click supersedes a read already on its way: that answer is the state
        // the click replaced, so it must not paint over it. The switch is inert
        // while the apply runs.
        let _ = state.update(SettingsMessage::AutolaunchToggle(false));
        assert_eq!(state.autolaunch, Some(false));
        assert!(state.autolaunch_applying);
        let _ = state.update(SettingsMessage::AutolaunchRead {
            epoch: 0,
            state: Some(true),
        });
        assert_eq!(
            state.autolaunch,
            Some(false),
            "a superseded read is dropped"
        );

        let _ = state.update(SettingsMessage::AutolaunchApplied(Ok(Some(false))));
        assert_eq!(state.autolaunch, Some(false));
        assert!(!state.autolaunch_applying);

        // A read that could not run is unknown, never "off".
        let epoch = state.autolaunch_epoch;
        let _ = state.update(SettingsMessage::AutolaunchRead { epoch, state: None });
        assert_eq!(state.autolaunch, None);
        assert!(state.error.is_none(), "a failed read is not an error");

        // A failed apply is unknown too, not the state it was asked for.
        let _ = state.update(SettingsMessage::AutolaunchToggle(true));
        let _ = state.update(SettingsMessage::AutolaunchApplied(Err("denied".into())));
        assert_eq!(state.autolaunch, None);
        assert_eq!(state.error.as_deref(), Some("denied"));
    }

    // ── Per-field autosave ────────────────────────────────────────
    //
    // A dropped task never runs; no store is needed, so the state is the surface.

    /// One per-field autosave case: the field, the steps on it, and what it must show.
    struct AutosaveCase {
        name: &'static str,
        key: &'static str,
        steps: &'static [Act],
        see: See,
    }

    /// One step applied to the case's own field before it is inspected.
    enum Act {
        /// A value staged in the field: a keystroke, or an immediate control's pick.
        Typed(&'static str),
        /// The field's settle fires at this generation.
        Settled(&'static str, u64),
        /// A persist result lands for this generation: `Ok` echoes the value.
        Saved(&'static str, u64),
        /// A persist failure lands for this generation.
        Failed(u64),
        /// Enter: the staged value settles at once.
        Enter,
        /// A value parked in the field's editor, the caret at its end.
        Edit(&'static str),
        /// One editor action on the field: a keystroke or an undo.
        Editor(EditorAction),
    }

    /// What the case's field must show once its steps have run.
    enum See {
        /// Generation, staged value, in flight, pending value, surfaced error.
        Field(
            Option<u64>,
            Option<&'static str>,
            bool,
            Option<(&'static str, u64)>,
            Option<&'static str>,
        ),
        /// The field's editor: its text and, where the case pins it, the caret.
        Editor(&'static str, Option<usize>),
    }

    /// One settle generation guards every field's autosave: only the current one may stage a
    /// value or land a persist, and a stale result never lands.
    #[expect(clippy::too_many_lines)] // one row per autosave transition
    #[test]
    fn per_field_autosave_cases() {
        use Act::{Edit, Editor, Enter, Failed, Saved, Settled, Typed};
        let cases = [
            AutosaveCase {
                name: "a text field keystroke arms a debounced settle",
                key: "exa_key",
                steps: &[Typed("key-a"), Typed("key-ab")],
                see: See::Field(Some(2), Some("key-ab"), false, None, None),
            },
            AutosaveCase {
                name: "a stale settle is dropped, never staging the old value",
                key: "exa_key",
                steps: &[
                    Typed("key-a"),
                    Typed("key-ab"),
                    Settled("key-a", 1),
                    Failed(1),
                ],
                see: See::Field(Some(2), Some("key-ab"), false, None, None),
            },
            AutosaveCase {
                name: "a stale result does not apply the stale value",
                key: "manager_model",
                steps: &[Typed("model-a"), Typed("model-b"), Saved("model-a", 1)],
                see: See::Field(Some(2), Some("model-b"), false, None, None),
            },
            AutosaveCase {
                name: "Enter settles the staged value at once",
                key: "manager_model",
                steps: &[Typed("model-c"), Enter],
                see: See::Field(Some(2), Some("model-c"), false, None, None),
            },
            AutosaveCase {
                name: "Enter settles the second text field's staged value too",
                key: "worker_model",
                steps: &[Typed("worker-d"), Enter],
                see: See::Field(Some(2), Some("worker-d"), false, None, None),
            },
            AutosaveCase {
                name: "an immediate toggle arms a zero-delay settle",
                key: "web_search_provider",
                steps: &[Typed("firecrawl")],
                see: See::Field(Some(1), Some("firecrawl"), false, None, None),
            },
            AutosaveCase {
                name: "a model slot keystroke arms a debounced settle",
                key: "manager_model",
                steps: &[Typed("test-model")],
                see: See::Field(Some(1), Some("test-model"), false, None, None),
            },
            AutosaveCase {
                name: "an in-flight persist queues a newer settle and flushes on result",
                key: "web_search_provider",
                steps: &[
                    Typed("exa"),
                    Settled("exa", 1),
                    Typed(""),
                    Settled("", 2),
                    Saved("exa", 1),
                ],
                see: See::Field(Some(2), Some(""), true, None, None),
            },
            AutosaveCase {
                name: "a stale pending flush never clobbers a newer staged edit",
                key: "web_search_provider",
                steps: &[
                    Typed("exa"),
                    Settled("exa", 1),
                    Typed(""),
                    Settled("", 2),
                    Typed("exa"),
                    Saved("exa", 1),
                    Saved("", 2),
                ],
                see: See::Field(Some(3), Some("exa"), false, None, None),
            },
            AutosaveCase {
                name: "an identical echo preserves the caret",
                key: "manager_model",
                steps: &[
                    Edit("deepseek/deepseek-chat"),
                    Editor(EditorAction::Insert('2')),
                    Typed("deepseek/deepseek-chat2"),
                    Saved("deepseek/deepseek-chat2", 1),
                ],
                see: See::Editor("deepseek/deepseek-chat2", Some(23)),
            },
            AutosaveCase {
                name: "an identical echo keeps the undo stack",
                key: "manager_model",
                steps: &[
                    Edit("deepseek/deepseek-chat"),
                    Editor(EditorAction::Insert('2')),
                    Typed("deepseek/deepseek-chat2"),
                    Saved("deepseek/deepseek-chat2", 1),
                    Editor(EditorAction::Undo),
                ],
                see: See::Editor("deepseek/deepseek-chat", None),
            },
            AutosaveCase {
                name: "a different canonical value rewrites the editor",
                key: "manager_model",
                steps: &[Typed("openai/gpt-4o"), Saved("openai/gpt-4o", 1)],
                see: See::Editor("openai/gpt-4o", None),
            },
        ];
        for case in cases {
            let AutosaveCase {
                name,
                key,
                steps,
                see,
            } = case;
            let mut state = SettingsState::new();
            let field = format!("config:{key}");
            for step in steps {
                let message = match step {
                    Typed(value) => SettingsMessage::ConfigField {
                        key,
                        value: value.to_string(),
                    },
                    Settled(value, generation) => SettingsMessage::ConfigFieldSettled {
                        field: field.clone(),
                        value: value.to_string(),
                        generation: *generation,
                    },
                    Saved(value, generation) => SettingsMessage::ConfigFieldSaveResult {
                        field: field.clone(),
                        generation: *generation,
                        result: Ok(crate::config::PersistOutcome {
                            value: value.to_string(),
                            warning: None,
                        }),
                    },
                    Failed(generation) => SettingsMessage::ConfigFieldSaveResult {
                        field: field.clone(),
                        generation: *generation,
                        result: Err("stale failure".into()),
                    },
                    Enter => SettingsMessage::ConfigFieldSettleNow {
                        field: field.clone(),
                    },
                    Edit(text) => {
                        let editor = SingleLineEditorState::new(text);
                        editor.buffer.move_to(0, text.chars().count());
                        state.field_editors.insert(field.clone(), editor);
                        continue;
                    }
                    Editor(action) => {
                        let editor = state.field_editors.get_mut(&field).unwrap();
                        editor.apply_action(action.clone());
                        continue;
                    }
                };
                drop(state.update(message));
            }
            match see {
                See::Field(generation, value, in_flight, queued, error) => {
                    let bumped = state.field_gen.get(&field).copied();
                    assert_eq!(bumped, generation, "case {name}: generation");
                    let staged = state.config.get_string_field(key);
                    assert_eq!(staged, value, "case {name}: staged value");
                    let flying = state.in_flight_persists.contains(&field);
                    assert_eq!(flying, in_flight, "case {name}: in flight");
                    let pending = state.pending_persists.get(&field);
                    let pending = pending.map(|(value, bumped)| (value.as_str(), *bumped));
                    assert_eq!(pending, queued, "case {name}: pending");
                    let failure = state.field_errors.get(&field).map(String::as_str);
                    assert_eq!(failure, error, "case {name}: error");
                }
                See::Editor(text, column) => {
                    let editor = state.field_editor(&field);
                    assert_eq!(editor.text(), text, "case {name}: text");
                    if let Some(column) = column {
                        let caret = editor.buffer.cursor().column;
                        assert_eq!(caret, column, "case {name}: caret");
                    }
                }
            }
        }
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

    /// Regression test for the Settings-page crash when editing the Manager /
    /// Worker model.
    ///
    /// The routing-row editor state is keyed by the current manager/worker
    /// model, but that editor was only created for the models present when the
    /// page loaded. Typing a different model re-keyed [`SettingsState::routing_section`]
    /// to a model with no editor, and the render-time lookup panicked. The fix
    /// seeds routing editors (insert-if-absent) at every manager/worker model
    /// mutation window and makes the render lookup panic-free.
    ///
    /// Exercises the routing render path directly (no voice/audio pipeline) by
    /// calling `routing_section()` and dropping the returned Element before
    /// mutating state again.
    #[expect(clippy::too_many_lines)] // long but linear state-transition fixture
    #[test]
    fn routing_rows_never_panic_when_manager_or_worker_model_edits() {
        let mut state = SettingsState::new();
        // Deterministic base snapshot (independent of the live CONFIG's
        // manager/worker/routings), re-seeded into the editor map below.
        state.config = ConfigData {
            manager_model: Some("deepseek/base".into()),
            worker_model: Some("deepseek/base-worker".into()),
            model_routings: vec![ModelRouting {
                model: "saved-routing-model".into(),
                provider_order: Some("SavedOrder".into()),
            }],
            ..ConfigData::STRUCT_FIELDS_DEFAULT
        };
        state.field_editors = HashMap::new();
        state.resync_field_editors();

        // ── Mid-edit window: a raw/partial model name must render its routing
        // row (an empty field resolves to the model's default provider)
        // instead of panicking on a missing editor.
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "custom-model".into(),
        });
        assert_eq!(
            state
                .field_editors
                .get("routing_order:custom-model")
                .map(super::super::common::SingleLineEditorState::text),
            Some(String::new()),
            "mid-edit: routing row seeded for the newly-typed model"
        );
        let _ = state.routing_section(); // must not panic

        // ── Saved routing order is displayed, not replaced by empty.
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "saved-routing-model".into(),
        });
        assert_eq!(
            state
                .field_editors
                .get("routing_order:saved-routing-model")
                .map(super::super::common::SingleLineEditorState::text),
            Some("SavedOrder".into()),
            "a model with a saved routing order shows that order"
        );
        let _ = state.routing_section(); // must not panic

        // ── Post-persist window: the injectable persist result applies the
        // canonical value and the existing routing editor keeps its text/undo.
        let manager_gen = state
            .field_gen
            .get("config:manager_model")
            .copied()
            .unwrap_or(0);
        let _ = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "config:manager_model".into(),
            generation: manager_gen,
            result: Ok(crate::config::PersistOutcome {
                value: "saved-routing-model".into(),
                warning: None,
            }),
        });
        assert_eq!(
            state.config.manager_model.as_deref(),
            Some("saved-routing-model"),
            "persist applied the canonical model value"
        );
        assert_eq!(
            state
                .field_editors
                .get("routing_order:saved-routing-model")
                .map(super::super::common::SingleLineEditorState::text),
            Some("SavedOrder".into()),
            "post-persist: the routing editor is not reset to empty"
        );
        let _ = state.routing_section(); // must not panic

        // A routing-order persist whose value already matches the editor must
        // NOT rewrite the editor (which would drop its undo stack / caret).
        let _ = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: "routing_order:saved-routing-model".into(),
            generation: 0,
            result: Ok(crate::config::PersistOutcome {
                value: "SavedOrder".into(),
                warning: None,
            }),
        });
        assert_eq!(
            state
                .field_editors
                .get("routing_order:saved-routing-model")
                .map(super::super::common::SingleLineEditorState::text),
            Some("SavedOrder".into()),
            "routing-order persist with identical value preserves the editor (undo/caret kept)"
        );
        let _ = state.routing_section(); // must not panic

        // ── Clear-to-empty coercion: whitespace falls back to the default
        // manager model, whose routing row renders without a panic.
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "   ".into(),
        });
        let default_field = format!("routing_order:{}", crate::config::DEFAULT_MANAGER_MODEL);
        assert!(
            state.field_editors.contains_key(&default_field),
            "clear-to-empty coerces to the default manager model; its routing row is seeded"
        );
        let _ = state.routing_section(); // must not panic

        // ── Worker model edit seeds its own routing row; manager==worker
        // collapses into a single routing row (no duplicates).
        let _ = state.update(SettingsMessage::ConfigField {
            key: "worker_model",
            value: "worker-custom".into(),
        });
        assert!(
            state
                .field_editors
                .contains_key("routing_order:worker-custom"),
            "worker model edit seeds its routing row"
        );
        let _ = state.update(SettingsMessage::ConfigField {
            key: "manager_model",
            value: "common-model".into(),
        });
        let _ = state.update(SettingsMessage::ConfigField {
            key: "worker_model",
            value: "common-model".into(),
        });
        let deduped = state
            .field_editors
            .keys()
            .filter(|k| k.starts_with("routing_order:common-model"))
            .count();
        assert_eq!(deduped, 1, "manager==worker dedups to a single routing row");
        let _ = state.routing_section(); // must not panic
    }

    /// Closing the Models settings popup is not a cancel, but a model save that
    /// fails afterwards reopens it — its inline error is never hidden behind a
    /// closed window. A failure for any other field leaves the popup closed.
    #[test]
    fn a_failed_model_save_reopens_the_closed_models_popup() {
        let mut state = SettingsState::new();
        let close = |state: &mut SettingsState| {
            let _ = state.update(SettingsMessage::CloseModelsPopup);
            assert!(!state.models_popup_open, "the popup is closed");
        };

        let _ = state.update(SettingsMessage::OpenModelsPopup);
        assert!(
            state.models_popup_open,
            "the Provider section's button opens the popup"
        );

        close(&mut state);
        let _ = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: format!("config:{CONFIG_KEY_FIRECRAWL_KEY}"),
            generation: 0,
            result: Err("refused".into()),
        });
        assert!(
            !state.models_popup_open,
            "a failure outside the model settings leaves the popup closed"
        );

        let _ = state.update(SettingsMessage::ConfigFieldSaveResult {
            field: format!("config:{CONFIG_KEY_WORKER_MODEL}"),
            generation: 0,
            result: Err("refused".into()),
        });
        assert!(
            state.models_popup_open,
            "a failed model-slot save reopens the popup"
        );
        assert_eq!(
            state
                .field_errors
                .get(&format!("config:{CONFIG_KEY_WORKER_MODEL}"))
                .map(String::as_str),
            Some("refused"),
            "the reopened popup carries the inline error"
        );
        let _ = state.models_dialog(); // must not panic

        close(&mut state);
        let _ = state.update(SettingsMessage::ModelPickerAddResult {
            target: ModelPickerTarget::ImageGen,
            model: "vendor/text-only".into(),
            ok: Err("no image output".into()),
        });
        assert!(
            state.models_popup_open,
            "a rejected added model reopens the popup"
        );
    }

    /// The toggle only reveals the custom endpoint; what is staged and gated on it follows the
    /// snapshot, and a failed save closes the section again.
    #[test]
    fn custom_endpoint_cases() {
        let cases: &[(&str, Option<&str>, Option<&str>)] = &[
            (
                "the toggle reveals, closes and re-closes on a failed save",
                None,
                None,
            ),
            (
                "the routing section hides while a custom endpoint is staged",
                Some("http://localhost:11434/v1"),
                None,
            ),
            (
                "OFF clears a staged endpoint key as well as the URL",
                Some("http://localhost:11434/v1"),
                Some("sk-local"),
            ),
        ];
        let ep = CONFIG_FIELD_PROVIDER_ENDPOINT;
        let kp = CONFIG_FIELD_PROVIDER_ENDPOINT_KEY;
        let url_key = CONFIG_KEY_PROVIDER_ENDPOINT;
        let key_key = CONFIG_KEY_PROVIDER_ENDPOINT_KEY;
        for &(name, staged, staged_key) in cases {
            let mut state = SettingsState::new();
            assert_eq!(state.config.provider_endpoint, None, "case {name}: unset");
            assert!(!state.custom_revealed, "case {name}: closed");
            assert!(!state.custom_endpoint_active_ui(), "case {name}: routing");

            // ON reveals only: nothing persisted, no settle armed, the snapshot untouched.
            let _task = state.update(SettingsMessage::CustomEndpointToggle(true));
            assert!(state.custom_revealed, "case {name}: ON reveals");
            assert_eq!(state.config.provider_endpoint, None, "case {name}: ON");
            assert!(!state.field_gen.contains_key(ep), "case {name}: no settle");

            if let Some(url) = staged {
                let _ = state.update(SettingsMessage::ConfigField {
                    key: url_key,
                    value: url.into(),
                });
                assert!(state.custom_endpoint_active_ui(), "case {name}: staged");
            }
            if let Some(key) = staged_key {
                let _ = state.update(SettingsMessage::ConfigField {
                    key: key_key,
                    value: key.into(),
                });
                assert_eq!(
                    state.config.provider_endpoint_key.as_deref(),
                    Some(key),
                    "case {name}: key staged"
                );
            }

            // OFF reverts to OpenRouter: the section closes, both fields clear, pending settles drop.
            let _task = state.update(SettingsMessage::CustomEndpointToggle(false));
            assert!(!state.custom_revealed, "case {name}: OFF closes");
            assert_eq!(state.config.provider_endpoint, None, "case {name}: cleared");
            assert_eq!(
                state.config.provider_endpoint_key, None,
                "case {name}: key cleared"
            );
            assert!(state.field_gen.contains_key(kp), "case {name}: armed");
            assert!(!state.custom_endpoint_active_ui(), "case {name}: back");

            // A failed endpoint save re-closes the section and surfaces in the bottom banner.
            let _task = state.update(SettingsMessage::CustomEndpointToggle(true));
            assert!(state.custom_revealed, "case {name}: re-revealed");
            let generation = state.field_gen.get(ep).copied().unwrap_or(0);
            let result = Err("invalid endpoint URL".into());
            let save = SettingsMessage::ConfigFieldSaveResult {
                field: ep.into(),
                generation,
                result,
            };
            let _task = state.update(save);
            assert!(!state.custom_revealed, "case {name}: failure closes");
            assert!(state.error.is_some(), "case {name}: failure surfaces");
        }
    }

    /// A status for the mode-line tests: every field the line reads, and nothing
    /// that would need a running supervisor.
    fn telegram_status(
        serving: Mode,
        target: Mode,
        health: Health,
        migrating: bool,
        error: Option<&str>,
    ) -> telegram_local::Status {
        telegram_local::Status {
            target,
            serving,
            health,
            last_error: error.map(str::to_string),
            port: None,
            migrating,
            blocked: None,
            credentials_set: true,
        }
    }

    // (name, stored mode, health mode, health, never-configured, error, warning)
    #[expect(clippy::type_complexity)]
    const MODE_LINES: &[(&str, Mode, Mode, Health, bool, Option<&str>, Option<&str>)] = &[
        (
            "installed and answering: nothing at all",
            Mode::Local,
            Mode::Local,
            Health::Running,
            false,
            None,
            None,
        ),
        (
            "a switch that already cut over is still progress",
            Mode::Local,
            Mode::Local,
            Health::Running,
            true,
            None,
            Some("Local server · installing or starting…"),
        ),
        (
            "not yet serving is progress",
            Mode::Local,
            Mode::Local,
            Health::Starting,
            false,
            None,
            Some("Local server · installing or starting…"),
        ),
        (
            "not running is said",
            Mode::Local,
            Mode::Local,
            Health::Down,
            false,
            None,
            Some("Local server · not running"),
        ),
        (
            "a failure is the reason, with no state word beside it",
            Mode::Local,
            Mode::Local,
            Health::Down,
            false,
            Some("the local server refused the bot: Unauthorized"),
            Some("Local server — the local server refused the bot: Unauthorized"),
        ),
        (
            "a switch in flight reports where it is going",
            Mode::Cloud,
            Mode::Local,
            Health::Starting,
            true,
            None,
            Some("Cloud · switching to Local server…"),
        ),
        (
            "a switch back reports where it is going",
            Mode::Local,
            Mode::Cloud,
            Health::Running,
            true,
            None,
            Some("Local server · switching to Cloud…"),
        ),
        (
            "a failure that stops a switch keeps its reason",
            Mode::Cloud,
            Mode::Local,
            Health::Starting,
            true,
            Some("the install failed"),
            Some("Cloud · switching to Local server… — the install failed"),
        ),
        (
            "the cloud keeps its own line and its way in",
            Mode::Cloud,
            Mode::Cloud,
            Health::Cloud,
            false,
            None,
            Some("Cloud · served by Telegram"),
        ),
        (
            "a cloud failure is its reason alone",
            Mode::Cloud,
            Mode::Cloud,
            Health::Cloud,
            true,
            Some("the cloud Bot API has not taken the bot back yet"),
            Some("Cloud — the cloud Bot API has not taken the bot back yet"),
        ),
    ];

    /// The mode line: silence while it works, progress while it does not, the reason on a failure.
    #[test]
    fn telegram_mode_line_cases() {
        for &(name, serving, target, health, migrating, error, expected) in MODE_LINES {
            let line = telegram_line(&telegram_status(serving, target, health, migrating, error));
            assert_eq!(line.as_deref(), expected, "case {name}");
        }

        // (name, serving, target, health, migrating, error, expected escape).
        #[expect(clippy::type_complexity)] // case table
        let escapes: &[(&str, Mode, Mode, Health, bool, Option<&str>, bool)] = &[
            (
                "a working local server earns nothing",
                Mode::Local,
                Mode::Local,
                Health::Running,
                false,
                None,
                false,
            ),
            (
                "a local failure earns the way back",
                Mode::Local,
                Mode::Local,
                Health::Down,
                false,
                Some("the install failed"),
                true,
            ),
            (
                "a cloud failure earns nothing",
                Mode::Cloud,
                Mode::Cloud,
                Health::Cloud,
                true,
                Some("the cloud Bot API has not taken the bot back yet"),
                false,
            ),
        ];
        for &(name, serving, target, health, migrating, error, expected) in escapes {
            let status = telegram_status(serving, target, health, migrating, error);
            assert_eq!(telegram_escape(&status), expected, "case {name}");
        }
    }
}

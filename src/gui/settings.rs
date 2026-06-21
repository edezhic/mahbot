//! Settings page — dynamic configuration editor.
//!
//! Reads the current config snapshot from [`crate::config::CONFIG`],
//! presents editable fields organised in sections, and saves changes
//! via [`crate::config::save_and_reload`].

#![allow(clippy::from_iter_instead_of_collect)]

use crate::Role;
use crate::config::{CONFIG, ConfigData, ModelRouting, RoleConfig};

use iced::widget::{
    Column, Row, Space, button, column, container, row, scrollable, stack, text, text_input,
    toggler,
};
use iced::{Alignment, Element, Length, Task};

use super::theme;

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
    VideoGenModel(String),
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
        }
    }

    /// Reload the editable snapshot from the current CONFIG.
    pub fn refresh(&mut self) {
        self.config = CONFIG.snapshot();
        self.error = None;
    }

    pub fn update(&mut self, msg: SettingsMessage) -> Task<SettingsMessage> {
        match msg {
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
            SettingsMessage::VideoGenModel(v) => {
                self.config.video_gen_model = Some(v).filter(|s| !s.is_empty());
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
        }
    }

    // ── View ─────────────────────────────────────────────────────

    pub fn view(&self) -> Element<'_, SettingsMessage> {
        let sections = column![
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

        let mut content = column![sections];

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

        let body = stack([scroll.into(), save_btn.into()]);

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
                    "Image Gen Model",
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
                    "Video Gen Model",
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
            ("video_gen_model", "VideoGenModel"),
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

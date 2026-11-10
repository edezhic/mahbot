//! Dashboard theme — defines the Flexoki dark color palette, ticket phase colors, role badge
//! colors, log level colors, and workspace status colors for the native Iced GUI.

use iced::Background;
use iced::Color;
use iced::border;
use iced::widget::{checkbox, container, pick_list, scrollable, toggler};

use crate::WorkspaceStatus;
use crate::pipeline::board::{TicketLabel, TicketPhase};

use iced_fonts::lucide;
use std::sync::atomic::AtomicBool;

pub const BG_BASE: Color = Color::from_rgb(0.063, 0.059, 0.059); // #100f0f
pub const BG_SURFACE: Color = Color::from_rgb(0.110, 0.106, 0.102); // #1c1b1a
pub const BG_ELEVATED: Color = Color::from_rgb(0.157, 0.153, 0.149); // #282726

pub const BORDER: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.08);
pub const BORDER_STRONG: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.12);

pub const TEXT_PRIMARY: Color = Color::from_rgb(0.808, 0.804, 0.765); // #cecdc3
pub const TEXT_SECONDARY: Color = Color::from_rgb(0.529, 0.522, 0.502); // #878580
pub const TEXT_MUTED: Color = Color::from_rgb(0.341, 0.337, 0.325); // #575653
pub const TEXT_FAINT: Color = Color::from_rgb(0.204, 0.200, 0.192); // #343331

pub const ACCENT: Color = Color::from_rgb(0.227, 0.663, 0.624); // #3aa99f
pub const ACCENT_LIGHT: Color = Color::from_rgb(0.357, 0.749, 0.710); // #5bbfb5
pub const ACCENT_DIM: Color = Color::from_rgba(0.227, 0.663, 0.624, 0.3);

// Find match highlights (Flexoki amber #D0A215): the current match at higher alpha for
// prominence, the others at low alpha as a background tint. Both are distinct from the
// ACCENT_DIM teal used for selection.
pub const FIND_MATCH_DIM: Color = Color::from_rgba(0.816, 0.635, 0.082, 0.25);
pub const FIND_MATCH_CURRENT: Color = Color::from_rgba(0.816, 0.635, 0.082, 0.45);

/// Bracket matching highlight: subtle teal background (ACCENT_LIGHT at 35%).
pub const BRACKET_MATCH: Color = Color::from_rgba(0.357, 0.749, 0.710, 0.35);

pub const HOVER: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.05);
pub const HOVER_STRONG: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.08);

pub const STATUS_SUCCESS: Color = Color::from_rgb(0.0, 0.902, 0.541); // #00e68a
pub const STATUS_WARNING: Color = Color::from_rgb(1.0, 0.667, 0.0); // #ffaa00
pub const STATUS_ERROR: Color = Color::from_rgb(1.0, 0.267, 0.4); // #ff4466

pub const DIFF_ADDED_TINT: Color =
    Color::from_rgba(STATUS_SUCCESS.r, STATUS_SUCCESS.g, STATUS_SUCCESS.b, 0.10);
pub const DIFF_REMOVED_TINT: Color =
    Color::from_rgba(STATUS_ERROR.r, STATUS_ERROR.g, STATUS_ERROR.b, 0.10);
pub const DIFF_FILE_HEADER_BG: Color =
    Color::from_rgba(STATUS_WARNING.r, STATUS_WARNING.g, STATUS_WARNING.b, 0.06);

#[must_use]
pub fn log_level_color(level: &str) -> (Color, Color) {
    match level.to_uppercase().as_str() {
        "ERROR" => (STATUS_ERROR, Color::from_rgba(0.937, 0.267, 0.267, 0.08)),
        "WARN" => (STATUS_WARNING, Color::from_rgba(1.0, 0.667, 0.0, 0.08)),
        "INFO" => (
            Color::from_rgb(0.219, 0.741, 0.973),
            Color::from_rgba(0.219, 0.741, 0.973, 0.08),
        ),
        "DEBUG" => (
            Color::from_rgb(0.655, 0.545, 0.980),
            Color::from_rgba(0.655, 0.545, 0.980, 0.08),
        ),
        "TRACE" => (TEXT_MUTED, Color::from_rgba(0.5, 0.5, 0.5, 0.08)),
        _ => (TEXT_MUTED, HOVER),
    }
}

/// Role badge colors from [`crate::agent::role::role_info()`] — the single source of truth, so
/// a role the palette does not name cannot silently fall through to black.
const fn badge_bg(fg: Color) -> Color {
    Color::from_rgba(fg.r, fg.g, fg.b, 0.1)
}

/// Returns the badge (foreground, background) color for a role name.
///
/// Accepts canonical names, derivative names carrying a trailing `_<digits>` suffix (stripped by
/// [`crate::channels::reply::strip_agent_role_suffix`], the same rule the reply author label
/// uses) and the joint-comment stage roles, including the retired stages' labels historical
/// comments still carry. Unknown strings — LLM API roles included — fall back to a muted grey.
#[must_use]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub const fn role_badge_color_for(role: &crate::Role) -> (Color, Color) {
    let info = crate::agent::role::role_info(role);
    let (r, g, b) = info.badge_fg;
    let fg = Color::from_rgb(r, g, b);
    (fg, badge_bg(fg))
}

/// Returns the badge (foreground, background) color for a role name string.
///
/// Accepts canonical names, derivative names with a trailing `_<digits>` suffix, and the
/// joint-comment stage roles ("Analysis"/"Verification", plus the retired stages' labels that
/// historical comments carry). The suffix is stripped by
/// [`crate::channels::reply::strip_agent_role_suffix`], so a derivative name renders as its base
/// role; unknown strings — LLM API roles included — fall back to a muted grey.
///
/// The background member is always the foreground at 0.1 alpha ([`badge_bg`]), the fallback
/// included, so consumers can use the second member directly.
#[must_use]
pub fn role_badge_color(role: &str) -> (Color, Color) {
    // Stage-name comment roles, resolved through the shared inverse mapping so the badge cannot
    // drift from verdict::stage_role.
    if let Some(r) = crate::pipeline::verdict::stage_role(role) {
        return role_badge_color_for(&r);
    }

    // One path for canonical and derivative names: the shared suffix rule returns its input
    // unchanged when there is no numeric suffix.
    if let Ok(r) = crate::channels::reply::strip_agent_role_suffix(role).parse::<crate::Role>() {
        return role_badge_color_for(&r);
    }

    // The background member must stay `badge_bg(fg)`: consumers render it directly.
    (TEXT_MUTED, badge_bg(TEXT_MUTED))
}

/// Returns the Lucide icon widget for a given agent role; callers apply `.size()`, `.color()`
/// and `.into()` to style it.
#[must_use]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn role_icon(role: &crate::Role) -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    match role {
        crate::Role::Manager => lucide::folder_kanban(),
        crate::Role::Engineer => lucide::code(),
        crate::Role::Analyst => lucide::search(),
        crate::Role::Worker => lucide::hammer(),
        crate::Role::Qa => lucide::gavel(),
        crate::Role::Maintainer => lucide::cog(),
        crate::Role::Discovery => lucide::folder_search_two(),
        crate::Role::Reviewer => lucide::file_check(),
        crate::Role::Sanitation => lucide::spray_can(),
        crate::Role::Assistant => lucide::bot(),
        crate::Role::Custom => lucide::puzzle(),
    }
}

/// Returns the Lucide icon widget for the workspace GENERAL context (the discovery-produced
/// summary for non-role LLM calls).
#[must_use]
pub fn general_context_icon() -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    lucide::book_open_text()
}

#[must_use]
pub fn diagnostics_icon() -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    lucide::stethoscope()
}

/// The chat-style role icon and its badge foreground color for a ticket-comment author label, or
/// `None` when the label must render as plain muted text.
/// Only the exact author values the pipeline writes today resolve: the joint-verdict stage names
/// (via [`crate::pipeline::verdict::stage_role`], which also carries the retired stages' labels),
/// the canonical agent names and "diagnostics". Legacy or unexpected values intentionally return
/// `None`.
#[must_use]
pub fn comment_author_icon(
    author: &str,
) -> Option<(
    iced::widget::Text<'static, iced::Theme, iced::Renderer>,
    Color,
)> {
    let icon_for = |role: crate::Role| {
        let (fg, _) = role_badge_color_for(&role);
        (role_icon(&role), fg)
    };

    if let Some(role) = crate::pipeline::verdict::stage_role(author) {
        return Some(icon_for(role));
    }
    match author {
        "engineer" => Some(icon_for(crate::Role::Engineer)),
        "manager" => Some(icon_for(crate::Role::Manager)),
        "sanitation" => Some(icon_for(crate::Role::Sanitation)),
        "assistant" => Some(icon_for(crate::Role::Assistant)),
        "diagnostics" => Some((diagnostics_icon(), TEXT_PRIMARY)),
        _ => None,
    }
}

/// Dashboard default font (JetBrains Mono), registered via `.default_font()` in `main.rs`.
pub const JETBRAINS_MONO: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Normal,
    stretch: iced::font::Stretch::Normal,
    style: iced::font::Style::Normal,
};

pub const FONT_BOLD: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Bold,
    ..iced::Font::DEFAULT
};

pub const FONT_REGULAR: iced::Font = JETBRAINS_MONO;

/// Italic variant of JetBrains Mono (narration text).
pub const FONT_ITALIC: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    style: iced::font::Style::Italic,
    ..iced::Font::DEFAULT
};

// f32 is the only type iced 0.14 accepts for both `Into<Pixels>` and `Into<Padding>`, so the
// whole scale is f32. Deliberately shared beyond body text: icons and the editor/diff buffer font
// sit on the same rungs, so a change here moves those too.
pub const TEXT_10: f32 = 10.0;
pub const TEXT_11: f32 = 11.0;
pub const TEXT_12: f32 = 12.0;
pub const TEXT_13: f32 = 13.0;
pub const TEXT_14: f32 = 14.0;
pub const TEXT_16: f32 = 16.0;
pub const TEXT_24: f32 = 24.0;

/// Glyph scale of the chat input panel: every icon it draws takes its size through
/// [`composer_icon`], so one knob enlarges them all by the same factor. Text, padding, spacing
/// and alignment stay on the shared rungs.
pub const COMPOSER_ICON_SCALE: f32 = 1.25;

/// `size` at [`COMPOSER_ICON_SCALE`] — the composer's copy of a `TEXT_*` rung.
#[must_use]
pub const fn composer_icon(size: f32) -> f32 {
    size * COMPOSER_ICON_SCALE
}

pub const SPACE_1: f32 = 1.0;
pub const SPACE_2: f32 = 2.0;
pub const SPACE_3: f32 = 3.0;
pub const SPACE_4: f32 = 4.0;
pub const SPACE_6: f32 = 6.0;
pub const SPACE_8: f32 = 8.0;
pub const SPACE_10: f32 = 10.0;
pub const SPACE_12: f32 = 12.0;
pub const SPACE_16: f32 = 16.0;
pub const SPACE_20: f32 = 20.0;

pub const PAD_1: f32 = 1.0;
pub const PAD_2: f32 = 2.0;
pub const PAD_3: f32 = 3.0;
pub const PAD_4: f32 = 4.0;
pub const PAD_5: f32 = 5.0;
pub const PAD_6: f32 = 6.0;
pub const PAD_8: f32 = 8.0;
pub const PAD_10: f32 = 10.0;
pub const PAD_12: f32 = 12.0;
pub const PAD_14: f32 = 14.0;
pub const PAD_16: f32 = 16.0;
pub const PAD_18: f32 = 18.0;
pub const PAD_24: f32 = 24.0;
/// Page-level base-container padding (the de-facto 24px inset shared by every page).
pub const PAGE_PADDING: f32 = PAD_24;

/// Body text size (px) of the transcript markdown renderer. The sessions-page collapse
/// measurement measures body elements at this size, so a font-size change cannot silently drift
/// the measured wrap count from the actual render.
pub const MARKDOWN_TEXT_SIZE: f32 = TEXT_13;

/// Narration text size (px) — the italic line on Running Agents cards and on the transcript.
pub const NARRATION_TEXT_SIZE: f32 = TEXT_14;

/// Markdown rendering settings consistent with the Flexoki dark theme.
#[must_use]
pub fn markdown_settings() -> iced::widget::markdown::Settings {
    markdown_settings_with(FONT_REGULAR, MARKDOWN_TEXT_SIZE)
}

/// Markdown rendering settings for narration bodies: italic face at [`NARRATION_TEXT_SIZE`].
#[must_use]
pub fn narration_markdown_settings() -> iced::widget::markdown::Settings {
    markdown_settings_with(FONT_ITALIC, NARRATION_TEXT_SIZE)
}

/// Markdown rendering settings with an explicit body face and size. Fenced code blocks always
/// stay in the regular face while body and inline code take `font`.
#[must_use]
fn markdown_settings_with(font: iced::Font, text_size: f32) -> iced::widget::markdown::Settings {
    let style = iced::widget::markdown::Style {
        font,
        inline_code_highlight: iced::widget::markdown::Highlight {
            background: iced::Color::from_rgba(0.0, 0.0, 0.0, 0.15).into(),
            border: iced::border::rounded(4),
        },
        inline_code_padding: iced::padding::left(1).right(1),
        inline_code_color: TEXT_PRIMARY,
        inline_code_font: font,
        code_block_font: FONT_REGULAR,
        link_color: ACCENT,
    };
    iced::widget::markdown::Settings::with_text_size(text_size, style)
}

// All TicketPhase variants exhaustively matched — no catch-all.

#[must_use]
pub const fn ticket_phase_color(phase: TicketPhase) -> (Color, Color) {
    use TicketPhase::{
        Analysis, Backlog, Cancelled, Done, Failed, InDevelopment, InSanitation, Planning, Queued,
        Verification,
    };
    match phase {
        // Early phases — cool/muted, neutral
        Backlog => (
            Color::from_rgb(0.808, 0.804, 0.765),
            Color::from_rgb(0.176, 0.176, 0.176),
        ),
        Planning => (
            Color::from_rgb(0.902, 0.863, 0.784),
            Color::from_rgb(0.263, 0.243, 0.114),
        ),
        Analysis => (
            Color::from_rgb(0.784, 0.863, 0.949),
            Color::from_rgb(0.114, 0.216, 0.310),
        ),
        // Queued — olive gateway (Manager→Engineer)
        Queued => (
            Color::from_rgb(0.902, 0.863, 0.784),
            Color::from_rgb(0.263, 0.224, 0.114),
        ),
        // Active phases — warm
        InDevelopment => (
            Color::from_rgb(0.941, 0.878, 0.784),
            Color::from_rgb(0.380, 0.216, 0.078),
        ),
        // Sanitation phases — neutral gray
        InSanitation => (
            Color::from_rgb(0.788, 0.788, 0.788),
            Color::from_rgb(0.310, 0.310, 0.310),
        ),
        // Verification — the merged agentic code review + functional check
        Verification => (
            Color::from_rgb(0.816, 0.816, 0.933),
            Color::from_rgb(0.184, 0.216, 0.380),
        ),
        // Unblocking phases — distinct
        Done => (
            Color::from_rgb(0.753, 0.816, 0.753),
            Color::from_rgb(0.114, 0.176, 0.114),
        ),
        Cancelled => (
            Color::from_rgb(0.690, 0.690, 0.690),
            Color::from_rgb(0.145, 0.145, 0.145),
        ),
        Failed => (
            Color::from_rgb(0.878, 0.753, 0.753),
            Color::from_rgb(0.310, 0.114, 0.114),
        ),
    }
}

// Priority uses the same muted-tone approach as phase badges: lighter readable text on darker
// backgrounds. P0 red (failed-like), P1 orange, P2 yellow, P3/P4+ green (passed/done-like).

#[must_use]
pub fn ticket_priority_color(priority: i64) -> (Color, Color) {
    match priority {
        0 => (
            Color::from_rgb(0.878, 0.753, 0.753),
            Color::from_rgb(0.310, 0.114, 0.114),
        ),
        1 => (
            Color::from_rgb(0.941, 0.878, 0.784),
            Color::from_rgb(0.380, 0.216, 0.078),
        ),
        2 => (
            Color::from_rgb(0.902, 0.863, 0.784),
            Color::from_rgb(0.310, 0.224, 0.102),
        ),
        3 => (
            Color::from_rgb(0.784, 0.902, 0.816),
            Color::from_rgb(0.176, 0.310, 0.208),
        ),
        _ => (
            Color::from_rgb(0.753, 0.816, 0.753),
            Color::from_rgb(0.114, 0.176, 0.114),
        ),
    }
}

#[must_use]
pub const fn ticket_label_color(label: TicketLabel) -> (Color, Color) {
    match label {
        TicketLabel::Feat => (
            Color::from_rgb(0.706, 0.878, 0.847),
            Color::from_rgb(0.078, 0.239, 0.224),
        ),
        TicketLabel::Fix => (
            Color::from_rgb(0.902, 0.780, 0.812),
            Color::from_rgb(0.310, 0.153, 0.184),
        ),
        TicketLabel::Maint => (
            Color::from_rgb(0.741, 0.741, 0.741),
            Color::from_rgb(0.227, 0.227, 0.227),
        ),
    }
}

/// One-shot guard to log timestamp parse failure only once.
static TIMESTAMP_PARSE_WARNED: AtomicBool = AtomicBool::new(false);

/// Format an ISO 8601 timestamp into a human-readable absolute form in local time ("Jun 5,
/// 21:54"). Storage stays UTC; only the displayed wall-clock is converted. On parse failure the
/// first 16 characters are returned.
#[must_use]
pub fn format_timestamp(ts: &str) -> String {
    if let Ok(dt) = crate::db::parse_utc_timestamp(ts) {
        dt.with_timezone(&chrono::Local)
            .format("%b %-d, %H:%M")
            .to_string()
    } else {
        if !TIMESTAMP_PARSE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(timestamp = %ts, "Failed to parse timestamp, falling back to truncated string");
        }
        crate::util::truncate_bytes(ts, 16).to_string()
    }
}

/// The local-time HH:MM:SS portion of an ISO 8601 timestamp. Char-boundary safe; on parse
/// failure the raw slice is kept, so malformed inputs degrade to their stored wall-clock.
pub fn format_hhmmss(ts: &str) -> String {
    if let Ok(dt) = crate::db::parse_utc_timestamp(ts) {
        dt.with_timezone(&chrono::Local)
            .format("%H:%M:%S")
            .to_string()
    } else if ts.len() > 19 {
        ts[ts.floor_char_boundary(11)..ts.floor_char_boundary(19)].to_string()
    } else {
        ts.to_string()
    }
}

/// A relative timestamp label for a chat message against the supplied local `now`: today as
/// HH:MM, yesterday as "yesterday", older as "X days ago". Clock skew is treated as today and an
/// invalid timestamp renders as an empty string.
#[must_use]
pub fn format_relative_time(ts: &str, now: chrono::DateTime<chrono::Local>) -> String {
    let Ok(dt) = crate::db::parse_utc_timestamp(ts) else {
        return String::new();
    };
    let local_dt = dt.with_timezone(&chrono::Local);
    let days = (now.date_naive() - local_dt.date_naive()).num_days();
    if days <= 0 {
        local_dt.format("%H:%M").to_string()
    } else if days == 1 {
        "yesterday".to_string()
    } else {
        format!("{days} days ago")
    }
}

/// Compact session-length format: raw below 1,000, one decimal in the k-range, one decimal
/// at/above one million. Values that would round up past a unit boundary render in the next unit
/// (999_999 → "1.0M").
#[expect(clippy::cast_precision_loss)] // token counts are far below f64's 2^53 exact range
pub(crate) fn format_compact_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 999_950 {
        // k-range stops below values that would round to "1000.0k".
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else if tokens < 999_950_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else {
        format!("{:.1}G", tokens as f64 / 1_000_000_000.0)
    }
}

#[must_use]
pub fn workspace_status_color(status: WorkspaceStatus) -> (Color, Color) {
    match status {
        WorkspaceStatus::Ready => (
            Color::from_rgb(0.133, 0.773, 0.369),
            badge_bg(Color::from_rgb(0.133, 0.773, 0.369)),
        ),
        WorkspaceStatus::Analyzing => (
            Color::from_rgb(0.851, 0.557, 0.0),
            badge_bg(Color::from_rgb(0.851, 0.557, 0.0)),
        ),
        WorkspaceStatus::Failed => (
            Color::from_rgb(0.957, 0.247, 0.369),
            badge_bg(Color::from_rgb(0.957, 0.247, 0.369)),
        ),
        WorkspaceStatus::Pending => (
            Color::from_rgb(0.631, 0.631, 0.631),
            badge_bg(Color::from_rgb(0.631, 0.631, 0.631)),
        ),
    }
}

/// Log entry fade‑in duration (ms).
pub const ANIM_LOG_FADE_MS: u64 = 100;

/// A [`scrollable::Scrollbar`] with thin 6px dimensions, used by every scrollable widget for a
/// consistent appearance.
#[must_use]
pub fn thin_scrollbar() -> scrollable::Scrollbar {
    scrollable::Scrollbar::new().width(6).scroller_width(6)
}

/// A vertical [`scrollable::Direction`] with the thin scrollbar — prefer this over spelling out
/// `scrollable::Direction::Vertical(theme::thin_scrollbar())` at every call site.
#[must_use]
pub fn vertical_scrollbar() -> scrollable::Direction {
    scrollable::Direction::Vertical(thin_scrollbar())
}

/// Returns a vertical [`scrollable::Direction`] that scrolls without drawing a scrollbar — for a
/// container too narrow to spend its edge on a rail.
#[must_use]
pub fn hidden_vertical_scrollbar() -> scrollable::Direction {
    scrollable::Direction::Vertical(scrollable::Scrollbar::hidden())
}

/// A horizontal [`scrollable::Direction`] with the thin scrollbar, embedded (zero spacing) so
/// the rail occupies layout space: iced reserves it only while the content overflows.
#[must_use]
pub fn horizontal_scrollbar() -> scrollable::Direction {
    scrollable::Direction::Horizontal(thin_scrollbar().spacing(0.0))
}

/// Standard scrollbar style for the dark Flexoki theme: [`TEXT_PRIMARY`] as the scroller base,
/// at 0.4 active, 0.6 hovered and 0.8 dragged. The rail is transparent and both borders are
/// rounded at 2px.
#[must_use]
pub fn scrollbar_style(theme: &iced::Theme, status: scrollable::Status) -> scrollable::Style {
    let base = scrollable::default(theme, status);

    let opacity = match status {
        scrollable::Status::Active { .. } => 0.4,
        scrollable::Status::Hovered { .. } => 0.6,
        scrollable::Status::Dragged { .. } => 0.8,
    };

    let rail = scrollable::Rail {
        background: None,
        border: border::rounded(2),
        scroller: scrollable::Scroller {
            background: Background::Color(TEXT_PRIMARY.scale_alpha(opacity)),
            border: border::rounded(2),
        },
    };

    scrollable::Style {
        vertical_rail: rail,
        horizontal_rail: rail,
        ..base
    }
}

/// Transparent button with no background, for icon-only buttons embedded in bars and toolbars.
#[must_use]
pub fn button_transparent(
    _: &iced::Theme,
    _status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    iced::widget::button::Style {
        background: None,
        ..Default::default()
    }
}

/// Transparent-background button style with a hover/press highlight — the shared factory for the
/// composer's icon controls and dropdown menu items.
#[must_use = "button style factory returns a style closure"]
fn transparent_button_style(
    hover: Color,
    pressed: Color,
    radius: f32,
    disabled: bool,
) -> impl Fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style {
    move |_: &iced::Theme, status| {
        let bg = if disabled {
            Color::TRANSPARENT
        } else {
            match status {
                iced::widget::button::Status::Hovered => hover,
                iced::widget::button::Status::Pressed => pressed,
                _ => Color::TRANSPARENT,
            }
        };
        iced::widget::button::Style {
            background: Some(Background::Color(bg)),
            border: iced::Border {
                radius: radius.into(),
                width: 0.0,
                color: Color::TRANSPARENT,
            },
            ..Default::default()
        }
    }
}

/// Icon-only button with a subtle hover/press background. `disabled: true` suppresses the
/// highlight.
#[must_use = "button style factory returns a style closure"]
pub fn icon_button_style(
    disabled: bool,
) -> impl Fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style {
    transparent_button_style(HOVER_STRONG, ACCENT_DIM, 6.0, disabled)
}

/// Label color for editor/shell tab bars: [`ACCENT`] on the active tab, [`TEXT_SECONDARY`] on
/// the others.
#[must_use]
pub fn tab_label_color(is_active: bool) -> Color {
    if is_active { ACCENT } else { TEXT_SECONDARY }
}

/// Tab-button style: active tabs use [`BG_ELEVATED`], hovered [`HOVER`], otherwise [`BG_SURFACE`].
pub fn tab_button_style(
    is_active: bool,
) -> impl Fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style {
    move |_: &iced::Theme, status| {
        let bg = if is_active {
            BG_ELEVATED
        } else if status == iced::widget::button::Status::Hovered {
            HOVER
        } else {
            BG_SURFACE
        };
        iced::widget::button::Style {
            background: Some(Background::Color(bg)),
            border: iced::Border {
                radius: 0.0.into(),
                width: 0.0,
                color: Color::TRANSPARENT,
            },
            ..Default::default()
        }
    }
}

/// Danger button (Delete, Purge, Clear). Uses Flexoki error red.
#[must_use]
pub fn button_danger(
    _: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let base = match status {
        iced::widget::button::Status::Active => Color::from_rgba(1.0, 0.267, 0.4, 0.15),
        iced::widget::button::Status::Hovered => Color::from_rgba(1.0, 0.267, 0.4, 0.25),
        iced::widget::button::Status::Pressed => Color::from_rgba(1.0, 0.267, 0.4, 0.35),
        iced::widget::button::Status::Disabled => Color::TRANSPARENT,
    };
    let text = match status {
        iced::widget::button::Status::Disabled => TEXT_MUTED,
        _ => STATUS_ERROR,
    };
    iced::widget::button::Style {
        background: Some(iced::Background::Color(base)),
        text_color: text,
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: Color::from_rgba(1.0, 0.267, 0.4, 0.2),
        },
        ..iced::widget::button::Style::default()
    }
}

/// Secondary/neutral button (Cancel, Close). Uses Flexoki surface tones.
#[must_use]
pub fn button_secondary(
    _: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let bg = match status {
        iced::widget::button::Status::Active => HOVER,
        iced::widget::button::Status::Hovered => HOVER_STRONG,
        iced::widget::button::Status::Pressed => Color::from_rgba(0.808, 0.804, 0.765, 0.12),
        iced::widget::button::Status::Disabled => Color::TRANSPARENT,
    };
    let text = match status {
        iced::widget::button::Status::Disabled => TEXT_MUTED,
        _ => TEXT_PRIMARY,
    };
    iced::widget::button::Style {
        background: Some(iced::Background::Color(bg)),
        text_color: text,
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: BORDER,
        },
        ..iced::widget::button::Style::default()
    }
}

/// Shared text-button factory: hover/pressed background plus the enabled text color.
fn button_text_style(
    status: iced::widget::button::Status,
    enabled_color: Color,
) -> iced::widget::button::Style {
    let bg = match status {
        iced::widget::button::Status::Hovered => HOVER,
        iced::widget::button::Status::Pressed => HOVER_STRONG,
        _ => Color::TRANSPARENT,
    };
    let text = match status {
        iced::widget::button::Status::Disabled => TEXT_MUTED,
        _ => enabled_color,
    };
    iced::widget::button::Style {
        background: Some(iced::Background::Color(bg)),
        text_color: text,
        border: iced::Border {
            radius: 4.0.into(),
            width: 0.0,
            color: Color::TRANSPARENT,
        },
        ..iced::widget::button::Style::default()
    }
}

/// Text-only danger button (Cancel in modals, delete triggers).
#[must_use]
pub fn button_text_danger(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    button_text_style(status, STATUS_ERROR)
}

/// Text-only button (sidebar nav items, inline actions). Minimal Flexoki styling.
#[must_use]
pub fn button_text(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    button_text_style(status, TEXT_PRIMARY)
}

#[must_use]
pub fn pick_list_style(_theme: &iced::Theme, _status: pick_list::Status) -> pick_list::Style {
    pick_list::Style {
        text_color: TEXT_PRIMARY,
        placeholder_color: TEXT_MUTED,
        handle_color: TEXT_MUTED,
        background: Background::Color(BG_ELEVATED),
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: BORDER_STRONG,
        },
    }
}

/// Flexoki-dark style for the expanded menu of a [`fn@pick_list`], matching the ContextMenu
/// overlay look. The menu's internal overflow scrollbar is styled by iced's default catalog.
#[must_use]
pub fn pick_list_menu_style(_theme: &iced::Theme) -> iced::overlay::menu::Style {
    iced::overlay::menu::Style {
        background: Background::Color(BG_ELEVATED),
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: BORDER_STRONG,
        },
        text_color: TEXT_SECONDARY,
        selected_text_color: TEXT_PRIMARY,
        selected_background: Background::Color(HOVER),
        shadow: iced::Shadow::default(),
    }
}

#[must_use]
pub fn toggler_style(_theme: &iced::Theme, status: toggler::Status) -> toggler::Style {
    let (track, knob, toggled) = match status {
        toggler::Status::Active { is_toggled: true } => (ACCENT, BG_BASE, true),
        toggler::Status::Active { is_toggled: false } => (BG_ELEVATED, TEXT_MUTED, false),
        toggler::Status::Hovered { is_toggled: true } => (ACCENT_LIGHT, BG_BASE, true),
        toggler::Status::Hovered { is_toggled: false } => (BG_ELEVATED, TEXT_SECONDARY, false),
        toggler::Status::Disabled { is_toggled: true } => (ACCENT_DIM, BG_BASE, true),
        toggler::Status::Disabled { is_toggled: false } => (BG_SURFACE, TEXT_FAINT, false),
    };
    let (border_width, border_color) = if toggled {
        (0.0, Color::TRANSPARENT)
    } else {
        (1.0, BORDER_STRONG)
    };
    toggler::Style {
        background: Background::Color(track),
        background_border_width: border_width,
        background_border_color: border_color,
        foreground: Background::Color(knob),
        foreground_border_width: 0.0,
        foreground_border_color: Color::TRANSPARENT,
        text_color: None,
        border_radius: None,
        padding_ratio: 0.1,
    }
}

#[must_use]
pub fn checkbox_style(_theme: &iced::Theme, status: checkbox::Status) -> checkbox::Style {
    let (background, icon, border) = match status {
        checkbox::Status::Active { is_checked: true }
        | checkbox::Status::Hovered { is_checked: true } => (ACCENT, BG_BASE, None),
        checkbox::Status::Active { is_checked: false } => (BG_ELEVATED, TEXT_MUTED, Some(BORDER)),
        checkbox::Status::Hovered { is_checked: false } => {
            (BG_ELEVATED, TEXT_SECONDARY, Some(BORDER_STRONG))
        }
        checkbox::Status::Disabled { is_checked: true } => (ACCENT_DIM, BG_BASE, None),
        checkbox::Status::Disabled { is_checked: false } => (BG_SURFACE, TEXT_FAINT, Some(BORDER)),
    };
    // A marked box is filled outright; an empty one is drawn by its border.
    let (border_color, border_width) = match border {
        Some(color) => (color, 1.0),
        None => (Color::TRANSPARENT, 0.0),
    };
    checkbox::Style {
        background: Background::Color(background),
        icon_color: icon,
        border: iced::Border {
            color: border_color,
            width: border_width,
            radius: 4.0.into(),
        },
        text_color: Some(TEXT_PRIMARY),
    }
}

/// Shared container style factory: background fill plus border parameters. The wrappers below
/// delegate here so the variants stay in one place.
pub(crate) fn container_style(
    bg: Color,
    radius: f32,
    width: f32,
    border_color: Color,
) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg)),
        border: iced::Border {
            radius: radius.into(),
            width,
            color: border_color,
        },
        ..container::Style::default()
    }
}

/// Style for tooltip containers: elevated background with subtle rounded corners and a hairline
/// border, so a tooltip stays readable whatever is underneath it.
#[must_use]
pub fn tooltip_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 6.0, 1.0, BORDER_STRONG)
}

/// Style for a control drawn over arbitrary content (the picture-tile remove cross): a dark
/// translucent scrim under the glyph, legible on a bright picture and distinct on a dark one.
#[must_use]
pub fn scrim_style(_theme: &iced::Theme) -> container::Style {
    container_style(
        Color::from_rgba(0.0, 0.0, 0.0, 0.6),
        4.0,
        0.0,
        Color::TRANSPARENT,
    )
}

/// Style for chat message bubbles and the typing indicator: shared padding/radius/border, with
/// the background parameterized and `text_color` optional.
pub fn bubble_style(
    bg: Color,
    text_color: Option<Color>,
) -> impl Fn(&iced::Theme) -> container::Style {
    move |_theme: &iced::Theme| container::Style {
        background: Some(iced::Background::Color(bg)),
        text_color,
        border: iced::Border {
            radius: 8.0.into(),
            width: 0.0,
            color: iced::Color::TRANSPARENT,
        },
        ..container::Style::default()
    }
}

/// Style for bar containers (find/replace, go-to-line): flat elevated background, no radius.
#[must_use]
pub fn container_bar(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for surface cards: 1px border, 4px corners — ticket sections, comment cards, log rows.
#[must_use]
pub fn surface_card_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_SURFACE, 4.0, 1.0, BORDER)
}

/// Style for elevated cards: 1px border, 4px corners — board cards and session round cards.
#[must_use]
pub fn elevated_card_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 4.0, 1.0, BORDER)
}

/// Style for modal dialog containers: elevated background, 8px corners, a strong border.
#[must_use]
pub fn dialog_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 8.0, 1.0, BORDER_STRONG)
}

/// Style for the base page background: the BG_BASE fill with no border.
#[must_use]
pub fn base_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_BASE, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for surface-only containers with no border — sidebar panels, tab bars, filter bars.
#[must_use]
pub fn surface_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_SURFACE, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for badge pills: a rounded container with the given background and no border.
pub fn pill_style(bg: Color) -> impl Fn(&iced::Theme) -> container::Style {
    move |_theme: &iced::Theme| container_style(bg, 4.0, 0.0, Color::TRANSPARENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;
    use strum::IntoEnumIterator;

    /// `role_badge_color` resolves labels exactly as `role_badge_color_for` resolves roles.
    #[test]
    fn role_badge_color_cases() {
        for role in Role::iter() {
            let name = role.as_str();
            let expected = role_badge_color_for(&role);
            assert_eq!(role_badge_color(name), expected, "case canonical {name}");
            for suffix in ["_1", "_42", "_"] {
                assert_eq!(
                    role_badge_color(&format!("{name}{suffix}")),
                    expected,
                    "case derivative {name}{suffix}"
                );
            }
        }

        let muted = (TEXT_MUTED, badge_bg(TEXT_MUTED));
        let analyst = role_badge_color_for(&Role::Analyst);
        let cases: &[(&str, &str, (Color, Color))] = &[
            ("case-insensitive upper", "ANALYST", analyst),
            ("case-insensitive title", "Analyst", analyst),
            ("case-insensitive derivative", "ANALYST_1", analyst),
            ("non-numeric suffix analyst_final", "analyst_final", muted),
            ("non-numeric suffix worker_abc", "worker_abc", muted),
            ("llm api role user", "user", muted),
            ("llm api role system", "system", muted),
            ("llm api role tool", "tool", muted),
            ("empty", "", muted),
            ("garbage", "garbage", muted),
            ("unknown role", "unknown_role", muted),
            ("bare suffix only", "_1", muted),
        ];
        for &(name, input, expected) in cases {
            assert_eq!(role_badge_color(input), expected, "case {name}: {input:?}");
        }
    }

    /// Compact counts stay raw below 1k and carry one decimal at every k/M/G edge.
    #[test]
    fn compact_token_format_edges() {
        // Raw below 1,000; one decimal in the k-range and at/above 1M. Values rounding up past
        // a unit boundary render in the next unit.
        let cases: &[(&str, u64, &str)] = &[
            ("zero", 0, "0"),
            ("hundreds", 500, "500"),
            ("last raw value", 999, "999"),
            ("first k", 1_000, "1.0k"),
            ("k with one decimal", 12_300, "12.3k"),
            ("two hundred k", 200_000, "200.0k"),
            ("last k below the rounding edge", 999_949, "999.9k"),
            ("rounds up to M", 999_999, "1.0M"),
            ("first M", 1_000_000, "1.0M"),
            ("M with one decimal", 1_234_567, "1.2M"),
            ("last M below the rounding edge", 999_949_999, "999.9M"),
            ("rounds up to G", 999_950_000, "1.0G"),
            ("first G", 1_000_000_000, "1.0G"),
        ];
        for &(name, tokens, expected) in cases {
            assert_eq!(
                format_compact_tokens(tokens),
                expected,
                "case {name}: {tokens} tokens"
            );
        }
    }

    /// Locks every pre-consolidation style value so parameterization cannot silently change
    /// rendering.
    #[expect(clippy::too_many_lines)] // exhaustive style-lock fixture
    #[test]
    fn style_factory_values_unchanged() {
        let theme = iced::Theme::Dark;
        let bg = |c: iced::Color| Some(iced::Background::Color(c));
        let border = |r: f32, w: f32, c: iced::Color| iced::Border {
            radius: r.into(),
            width: w,
            color: c,
        };

        assert_eq!(
            tooltip_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_ELEVATED),
                border: border(6.0, 1.0, BORDER_STRONG),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            container_bar(&theme),
            iced::widget::container::Style {
                background: bg(BG_ELEVATED),
                border: border(0.0, 0.0, iced::Color::TRANSPARENT),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            surface_card_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_SURFACE),
                border: border(4.0, 1.0, BORDER),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            elevated_card_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_ELEVATED),
                border: border(4.0, 1.0, BORDER),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            pill_style(BG_ELEVATED)(&theme),
            iced::widget::container::Style {
                background: bg(BG_ELEVATED),
                border: border(4.0, 0.0, iced::Color::TRANSPARENT),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            dialog_container_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_ELEVATED),
                border: border(8.0, 1.0, BORDER_STRONG),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            base_container_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_BASE),
                ..iced::widget::container::Style::default()
            }
        );
        assert_eq!(
            surface_container_style(&theme),
            iced::widget::container::Style {
                background: bg(BG_SURFACE),
                ..iced::widget::container::Style::default()
            }
        );

        for status in [
            iced::widget::button::Status::Active,
            iced::widget::button::Status::Hovered,
            iced::widget::button::Status::Pressed,
            iced::widget::button::Status::Disabled,
        ] {
            let expected = |enabled: iced::Color| iced::widget::button::Style {
                background: bg(match status {
                    iced::widget::button::Status::Hovered => HOVER,
                    iced::widget::button::Status::Pressed => HOVER_STRONG,
                    _ => iced::Color::TRANSPARENT,
                }),
                text_color: if status == iced::widget::button::Status::Disabled {
                    TEXT_MUTED
                } else {
                    enabled
                },
                border: border(4.0, 0.0, iced::Color::TRANSPARENT),
                ..iced::widget::button::Style::default()
            };
            assert_eq!(button_text(&theme, status), expected(TEXT_PRIMARY));
            assert_eq!(button_text_danger(&theme, status), expected(STATUS_ERROR));
        }

        // pick_list field style is status-independent.
        assert_eq!(
            pick_list_style(&theme, pick_list::Status::Active),
            pick_list::Style {
                text_color: TEXT_PRIMARY,
                placeholder_color: TEXT_MUTED,
                handle_color: TEXT_MUTED,
                background: Background::Color(BG_ELEVATED),
                border: border(4.0, 1.0, BORDER_STRONG),
            }
        );
        assert_eq!(
            pick_list_menu_style(&theme),
            iced::overlay::menu::Style {
                background: Background::Color(BG_ELEVATED),
                border: border(4.0, 1.0, BORDER_STRONG),
                text_color: TEXT_SECONDARY,
                selected_text_color: TEXT_PRIMARY,
                selected_background: Background::Color(HOVER),
                shadow: iced::Shadow::default(),
            }
        );

        for (status, toggled, track, knob, bg_border) in [
            (
                toggler::Status::Active { is_toggled: true },
                true,
                ACCENT,
                BG_BASE,
                Color::TRANSPARENT,
            ),
            (
                toggler::Status::Active { is_toggled: false },
                false,
                BG_ELEVATED,
                TEXT_MUTED,
                BORDER_STRONG,
            ),
            (
                toggler::Status::Hovered { is_toggled: true },
                true,
                ACCENT_LIGHT,
                BG_BASE,
                Color::TRANSPARENT,
            ),
            (
                toggler::Status::Hovered { is_toggled: false },
                false,
                BG_ELEVATED,
                TEXT_SECONDARY,
                BORDER_STRONG,
            ),
            (
                toggler::Status::Disabled { is_toggled: true },
                true,
                ACCENT_DIM,
                BG_BASE,
                Color::TRANSPARENT,
            ),
            (
                toggler::Status::Disabled { is_toggled: false },
                false,
                BG_SURFACE,
                TEXT_FAINT,
                BORDER_STRONG,
            ),
        ] {
            let expected = toggler::Style {
                background: Background::Color(track),
                background_border_width: if toggled { 0.0 } else { 1.0 },
                background_border_color: bg_border,
                foreground: Background::Color(knob),
                foreground_border_width: 0.0,
                foreground_border_color: Color::TRANSPARENT,
                text_color: None,
                border_radius: None,
                padding_ratio: 0.1,
            };
            assert_eq!(toggler_style(&theme, status), expected);
        }
    }

    #[test]
    fn joint_comment_stage_roles_get_their_role_colors() {
        // The pipeline writes comments with the STAGE NAME as the comment role, and they must
        // render with the role color rather than the muted-grey fallback.
        assert_eq!(
            role_badge_color("Analysis"),
            role_badge_color_for(&crate::Role::Analyst)
        );
        assert_eq!(
            role_badge_color("Verification"),
            role_badge_color_for(&crate::Role::Qa)
        );
        assert_eq!(
            role_badge_color("Review"),
            role_badge_color_for(&crate::Role::Reviewer)
        );
        assert_eq!(
            role_badge_color("QA"),
            role_badge_color_for(&crate::Role::Qa)
        );
    }

    /// The badge tuple's background member must always be the foreground at 0.1 alpha
    /// ([`badge_bg`]).
    #[test]
    fn badge_background_is_alpha_scaled_foreground() {
        for role in Role::iter() {
            let (fg, bg) = role_badge_color_for(&role);
            assert_eq!(bg, badge_bg(fg), "canonical role {role:?}");
        }
        // Unknown/fallback strings, LLM API roles, and stage-name comment roles
        // all flow through the same tuple invariant.
        for name in [
            "analyst_final",
            "worker_abc",
            "user",
            "assistant",
            "system",
            "tool",
            "Analysis",
            "Verification",
            "Review",
            "QA",
            "analyst_3",
            "",
            "garbage",
            "unknown_role",
            "_1",
        ] {
            let (fg, bg) = role_badge_color(name);
            assert_eq!(bg, badge_bg(fg), "fallback/stage/derivative role {name:?}");
        }
    }

    #[test]
    fn relative_time_formats_against_the_local_wall_clock() {
        use chrono::TimeZone;

        let now = chrono::Local
            .with_ymd_and_hms(2026, 8, 28, 12, 0, 0)
            .single()
            .unwrap();
        let off = *now.offset();
        // (name, stamp date, expected label); fixed dates, local offset taken
        // from the fixed `now`, so no wall-clock or DST drift can enter.
        let cases: &[(&str, &str, &str)] = &[
            ("earlier today", "2026-08-28", "09:15"),
            ("yesterday", "2026-08-27", "yesterday"),
            ("three days ago", "2026-08-25", "3 days ago"),
            // A future instant is still "today": only the clock is shown.
            ("future today", "2026-08-29", "09:15"),
            ("not a timestamp", "", ""),
        ];
        for &(name, date, expected) in cases {
            let stamp = if date.is_empty() {
                "not-a-timestamp".to_string()
            } else {
                format!("{date}T09:15:00{off}")
            };
            let actual = format_relative_time(&stamp, now);
            assert_eq!(
                actual, expected,
                "case {name} ({stamp}): expected {expected:?}, got {actual:?}"
            );
        }
    }
}

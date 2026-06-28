//! Dashboard theme — ports color constants from web/src/index.css (Flexoki dark theme),
//! web/src/roleConfig.ts (10 role badge colors), Board.tsx STATUS_STYLE (16 ticket status
//! colors), Logs.tsx level colors, and Workspaces.tsx status colors.

use iced::Background;
use iced::Color;
use iced::border;
use iced::widget::{container, scrollable};

use crate::board::TicketPhase;

use iced_fonts::lucide;
use std::sync::atomic::AtomicBool;

// ── Flexoki dark palette ─────────────────────────────────────────

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

// ── Find match highlight colors (Flexoki amber) ──────────────────
//
// Non-current matches: amber/gold at low alpha (background tint).
// Current match: same amber at higher alpha for prominence.
// Both are visually distinct from ACCENT_DIM (teal) used for selection.
//
// Flexoki amber: #D0A215 = rgb(0.816, 0.635, 0.082)
pub const FIND_MATCH_DIM: Color = Color::from_rgba(0.816, 0.635, 0.082, 0.25);
pub const FIND_MATCH_CURRENT: Color = Color::from_rgba(0.816, 0.635, 0.082, 0.45);

/// Bracket matching highlight: subtle teal background (ACCENT_LIGHT at 35%).
pub const BRACKET_MATCH: Color = Color::from_rgba(0.357, 0.749, 0.710, 0.35);

pub const HOVER: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.05);
pub const HOVER_STRONG: Color = Color::from_rgba(0.808, 0.804, 0.765, 0.08);

/// Semi-transparent black used for modal backdrops that capture clicks.
/// Shared by board.rs, settings.rs, mod.rs, and editor.rs.
pub const BACKDROP_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.5);

// ── Status colors ─────────────────────────────────────────────────

pub const STATUS_SUCCESS: Color = Color::from_rgb(0.0, 0.902, 0.541); // #00e68a
pub const STATUS_WARNING: Color = Color::from_rgb(1.0, 0.667, 0.0); // #ffaa00
pub const STATUS_ERROR: Color = Color::from_rgb(1.0, 0.267, 0.4); // #ff4466

// ── Log level colors (from Logs.tsx) ─────────────────────────────

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

// ── Role badge colors (from roleConfig.ts) ───────────────────────

/// Returns the badge (foreground, background) color for a given [`crate::Role`].
///
/// Reads from [`crate::role::role_info()`] and converts the RGB tuple to
/// [`iced::Color`] — this avoids duplicating color data in an exhaustive match.
/// Adding a new [`crate::Role`] variant requires updating the `role_info()`
/// match (`badge_fg` field); the compiler will not catch a missing field here
/// (it defaults from `BASE_ROLE_INFO`), but the `badge_colors_set` test in
/// `role.rs` guards against silent black fallthrough.
#[must_use]
pub const fn role_badge_color_for(role: &crate::Role) -> (Color, Color) {
    let info = crate::role::role_info(role);
    let (r, g, b) = info.badge_fg;
    (Color::from_rgb(r, g, b), Color::from_rgba(r, g, b, 0.1))
}

/// Returns the badge (foreground, background) color for a role name string.
///
/// Accepts canonical names (e.g. `"analyst"`) and derivative names with a
/// numeric suffix (e.g. `"analyst_1"`, `"analyst_2"`). Unknown strings
/// (including LLM API roles like `"user"`, `"assistant"`, `"system"`, `"tool"`)
/// fall back to a muted grey.
///
/// Delegates to [`role_badge_color_for`] after resolving the string, which
/// reads colors from [`crate::role::role_info()`] as the single source of truth.
#[must_use]
pub fn role_badge_color(role: &str) -> (Color, Color) {
    // Try exact match first (handles canonical names like "analyst")
    if let Ok(r) = role.parse::<crate::Role>() {
        return role_badge_color_for(&r);
    }

    // Try stripping a trailing `_<digits>` suffix (handles "analyst_1" etc.)
    if let Some(idx) = role.rfind('_')
        && idx + 1 < role.len()
        && role.as_bytes()[idx + 1..].iter().all(u8::is_ascii_digit)
    {
        let stripped = &role[..idx];
        if let Ok(r) = stripped.parse::<crate::Role>() {
            return role_badge_color_for(&r);
        }
    }

    (TEXT_MUTED, HOVER)
}

// ── Role icon mapping (shared between sidebar and workspaces) ──────

/// Returns the Lucide icon widget for a given agent role.
///
/// Callers apply `.size()`, `.color()`, and `.into()` to style the icon
/// for their specific context.
#[must_use]
pub fn role_icon(role: &crate::Role) -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    match role {
        crate::Role::Manager => lucide::bot(),
        crate::Role::Engineer => lucide::wrench(),
        crate::Role::Analyst => lucide::scan_search(),
        crate::Role::Coder => lucide::code(),
        crate::Role::Qa => lucide::gavel(),
        crate::Role::Maintainer => lucide::cog(),
        crate::Role::Discovery => lucide::search(),
        crate::Role::Artist => lucide::palette(),
        crate::Role::Reviewer => lucide::file_check(),
        crate::Role::Sanitation => lucide::spray_can(),
    }
}

/// Bold weight variant of JetBrains Mono (the dashboard default font).
pub const FONT_BOLD: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Bold,
    ..iced::Font::DEFAULT
};

/// Regular weight variant of JetBrains Mono.
pub const FONT_REGULAR: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    ..iced::Font::DEFAULT
};

/// Markdown rendering settings consistent with the Flexoki dark theme.
#[must_use]
pub fn markdown_settings() -> iced::widget::markdown::Settings {
    let style = iced::widget::markdown::Style {
        font: FONT_REGULAR,
        inline_code_highlight: iced::widget::markdown::Highlight {
            background: iced::Color::from_rgba(0.0, 0.0, 0.0, 0.15).into(),
            border: iced::border::rounded(4),
        },
        inline_code_padding: iced::padding::left(1).right(1),
        inline_code_color: TEXT_PRIMARY,
        inline_code_font: FONT_REGULAR,
        code_block_font: FONT_REGULAR,
        link_color: ACCENT,
    };
    iced::widget::markdown::Settings::with_text_size(13, style)
}

// ── Ticket status badge colors (from Board.tsx STATUS_STYLE) ─────
// 15 TicketPhase variants, exhaustively matched — no catch-all.

pub const fn ticket_status_color(phase: TicketPhase) -> (Color, Color) {
    use TicketPhase::{
        Analysis, Backlog, Cancelled, DiagnosticsDone, Done, Failed, InDevelopment, InDiagnostics,
        InQa, InReview, InSanitation, Planning, QaPassed, ReadyForDevelopment, Reviewed,
        SanitationPassed,
    };
    match phase {
        // Early phases — cool/muted, neutral
        Backlog => (
            Color::from_rgb(0.176, 0.176, 0.176),
            Color::from_rgb(0.808, 0.804, 0.765),
        ),
        Planning => (
            Color::from_rgb(0.263, 0.243, 0.114),
            Color::from_rgb(0.902, 0.863, 0.784),
        ),
        Analysis => (
            Color::from_rgb(0.114, 0.216, 0.310),
            Color::from_rgb(0.784, 0.863, 0.949),
        ),
        // Ready — olive gateway (Manager→Engineer)
        ReadyForDevelopment => (
            Color::from_rgb(0.263, 0.224, 0.114),
            Color::from_rgb(0.902, 0.863, 0.784),
        ),
        // Active phases — warm
        InDevelopment => (
            Color::from_rgb(0.380, 0.216, 0.078),
            Color::from_rgb(0.941, 0.878, 0.784),
        ),
        // Diagnostic phases — amber/teal
        InDiagnostics => (
            Color::from_rgb(0.310, 0.224, 0.102),
            Color::from_rgb(0.902, 0.863, 0.784),
        ),
        DiagnosticsDone => (
            Color::from_rgb(0.161, 0.235, 0.224),
            Color::from_rgb(0.784, 0.902, 0.871),
        ),
        // Sanitation phases — neutral gray
        InSanitation => (
            Color::from_rgb(0.310, 0.310, 0.310),
            Color::from_rgb(0.788, 0.788, 0.788),
        ),
        SanitationPassed => (
            Color::from_rgb(0.247, 0.247, 0.247),
            Color::from_rgb(0.863, 0.863, 0.863),
        ),
        // Review & QA
        InReview => (
            Color::from_rgb(0.184, 0.216, 0.380),
            Color::from_rgb(0.816, 0.816, 0.933),
        ),
        Reviewed => (
            Color::from_rgb(0.224, 0.208, 0.322),
            Color::from_rgb(0.878, 0.816, 0.933),
        ),
        InQa => (
            Color::from_rgb(0.216, 0.184, 0.380),
            Color::from_rgb(0.816, 0.816, 0.933),
        ),
        QaPassed => (
            Color::from_rgb(0.176, 0.310, 0.208),
            Color::from_rgb(0.784, 0.902, 0.816),
        ),
        // Unblocking phases — distinct
        Done => (
            Color::from_rgb(0.114, 0.176, 0.114),
            Color::from_rgb(0.753, 0.816, 0.753),
        ),
        Cancelled => (
            Color::from_rgb(0.145, 0.145, 0.145),
            Color::from_rgb(0.690, 0.690, 0.690),
        ),
        Failed => (
            Color::from_rgb(0.310, 0.114, 0.114),
            Color::from_rgb(0.878, 0.753, 0.753),
        ),
    }
}

/// One-shot guard to log timestamp parse failure only once.
static TIMESTAMP_PARSE_WARNED: AtomicBool = AtomicBool::new(false);

/// Format an ISO 8601 timestamp string into a human-readable absolute form.
/// Output style: "Jun 5, 21:54" — no microseconds, no raw timezone suffixes.
/// If parsing fails, returns the first 16 characters as a fallback.
#[must_use]
pub fn format_timestamp(ts: &str) -> String {
    if let Ok(dt) = crate::turso::parse_utc_timestamp(ts) {
        dt.format("%b %-d, %H:%M").to_string()
    } else {
        if !TIMESTAMP_PARSE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(timestamp = %ts, "Failed to parse timestamp, falling back to truncated string");
        }
        if ts.len() > 16 {
            ts[..ts.floor_char_boundary(16)].to_string()
        } else {
            ts.to_string()
        }
    }
}

// ── Workspace status colors (from Workspaces.tsx) ────────────────

pub fn workspace_status_color(status: &str) -> (Color, Color) {
    match status {
        "ready" => (
            Color::from_rgb(0.133, 0.773, 0.369),
            Color::from_rgba(0.133, 0.773, 0.369, 0.1),
        ),
        "analyzing" => (
            Color::from_rgb(0.851, 0.557, 0.0),
            Color::from_rgba(0.851, 0.557, 0.0, 0.1),
        ),
        "failed" => (
            Color::from_rgb(0.957, 0.247, 0.369),
            Color::from_rgba(0.957, 0.247, 0.369, 0.1),
        ),
        _ => (
            Color::from_rgb(0.631, 0.631, 0.631),
            Color::from_rgba(0.631, 0.631, 0.631, 0.1),
        ),
    }
}

// ── Animation timing constants ────────────────────────────────────

/// Log entry fade‑in duration (ms).
pub const ANIM_LOG_FADE_MS: u64 = 100;
/// Sort‑indicator rotation duration (ms).
pub const ANIM_SORT_MS: u64 = 200;
/// Selected row background transition (ms).
pub const ANIM_SELECTED_MS: u64 = 150;

// ── Shared scrollbar helpers ─────────────────────────────────────

/// Returns a [`scrollable::Scrollbar`] with thin 6px dimensions for
/// both the rail and scroller widths. Used across all scrollable
/// widgets in the dashboard for a consistent appearance.
#[must_use]
pub fn thin_scrollbar() -> scrollable::Scrollbar {
    scrollable::Scrollbar::new().width(6).scroller_width(6)
}

/// Returns a vertical [`scrollable::Direction`] with the thin scrollbar.
///
/// Convenience wrapper around [`thin_scrollbar`] — prefer this over
/// spelling out `scrollable::Direction::Vertical(theme::thin_scrollbar())`
/// at every call site.
#[must_use]
pub fn vertical_scrollbar() -> scrollable::Direction {
    scrollable::Direction::Vertical(thin_scrollbar())
}

/// Returns a horizontal [`scrollable::Direction`] with the thin scrollbar.
///
/// Convenience wrapper around [`thin_scrollbar`] — prefer this over
/// spelling out `scrollable::Direction::Horizontal(theme::thin_scrollbar())`
/// at every call site.
#[must_use]
pub fn horizontal_scrollbar() -> scrollable::Direction {
    scrollable::Direction::Horizontal(thin_scrollbar())
}

/// Standard scrollbar style for the dark Flexoki theme.
///
/// Uses [`TEXT_PRIMARY`] as the scroller base color, varying opacity:
/// * Active: 0.4
/// * Hovered: 0.6
/// * Dragged: 0.8
///
/// The rail background is transparent, both rail and scroller borders
/// are rounded (2px radius). Other fields (`container`, `gap`,
/// `auto_scroll`) are inherited from [`scrollable::default`].
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

// ── Button theme helpers ──────────────────────────────────────────

/// Transparent button with no background. Useful for icon-only buttons
/// embedded in bars, toolbars, and tab close buttons.
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

/// Primary action button (Save, Submit, Confirm). Uses Flexoki accent green.
pub fn button_primary(
    _: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let base = match status {
        iced::widget::button::Status::Active => ACCENT,
        iced::widget::button::Status::Hovered => ACCENT_LIGHT,
        iced::widget::button::Status::Pressed => Color::from_rgb(0.165, 0.482, 0.455),
        iced::widget::button::Status::Disabled => TEXT_FAINT,
    };
    let text = match status {
        iced::widget::button::Status::Disabled => TEXT_MUTED,
        _ => BG_BASE,
    };
    iced::widget::button::Style {
        background: Some(iced::Background::Color(base)),
        text_color: text,
        border: iced::Border {
            radius: 4.0.into(),
            width: 0.0,
            color: Color::TRANSPARENT,
        },
        ..iced::widget::button::Style::default()
    }
}

/// Danger button (Delete, Purge, Clear). Uses Flexoki error red.
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

/// Text-only danger button (Cancel in modals, delete triggers). Like [`button_text`]
/// but with red text. No colored background, subtle hover highlight, red text only.
pub fn button_text_danger(
    _: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let bg = match status {
        iced::widget::button::Status::Hovered => HOVER,
        iced::widget::button::Status::Pressed => HOVER_STRONG,
        _ => Color::TRANSPARENT,
    };
    let text = match status {
        iced::widget::button::Status::Disabled => TEXT_MUTED,
        _ => STATUS_ERROR,
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

/// Text-only button (sidebar nav items, inline actions). Minimal Flexoki styling.
pub fn button_text(
    _: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let bg = match status {
        iced::widget::button::Status::Hovered => HOVER,
        iced::widget::button::Status::Pressed => HOVER_STRONG,
        _ => Color::TRANSPARENT,
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
            width: 0.0,
            color: Color::TRANSPARENT,
        },
        ..iced::widget::button::Style::default()
    }
}

// ── Tooltip container style ──────────────────────────────────────

/// Style for tooltip containers: elevated background with subtle rounded
/// corners and a hairline border, matching the `dialog_container_style`
/// convention used for modal dialogs.
///
/// Applying this via `.style(theme::tooltip_style)` gives every tooltip a
/// dark/neutral fill that stays readable regardless of what content is
/// underneath it.
#[must_use]
pub fn tooltip_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(iced::Background::Color(BG_ELEVATED)),
        border: iced::Border {
            radius: 6.0.into(),
            width: 1.0,
            color: BORDER_STRONG,
        },
        ..container::Style::default()
    }
}

/// Style for chat message bubbles and the typing indicator.
///
/// Shared padding/radius/border across all bubbles.  Background is
/// parameterized (user vs agent messages); `text_color` is optional —
/// message bubbles set it to `TEXT_PRIMARY`, while the typing indicator
/// leaves it inherited (the inner `text()` widget sets its own color).
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

/// Style for bar containers (find/replace bar, go-to-line bar).
/// Flat elevated background with zero-radius border.
#[must_use]
pub fn container_bar(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(iced::Background::Color(BG_ELEVATED)),
        border: iced::Border {
            radius: 0.0.into(),
            width: 0.0,
            color: iced::Color::TRANSPARENT,
        },
        ..container::Style::default()
    }
}

/// Style for surface cards: surface background with a 1px border and
/// 4px rounded corners. Used for ticket detail sections, comment cards,
/// log entries, and session transcript messages.
#[must_use]
pub fn surface_card_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(BG_SURFACE)),
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: BORDER,
        },
        ..container::Style::default()
    }
}

/// Style for modal dialog containers: elevated background, 8px rounded
/// corners, and a strong border. Shared by all modal overlays across the
/// dashboard (board detail, settings dialogs, editor overlays, diff/branch
/// modals, etc.).
#[must_use]
pub fn dialog_container_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(BG_ELEVATED)),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: BORDER_STRONG,
        },
        ..container::Style::default()
    }
}

/// Style for the base page background: just the BG_BASE fill with no border.
/// Used as the outermost container on most pages (home, sessions, logs).
#[must_use]
pub fn base_container_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(BG_BASE)),
        ..container::Style::default()
    }
}

/// Style for surface-only containers: surface background with no border.
/// Used for sidebar panels, tab bars, and filter bars.
#[must_use]
pub fn surface_container_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(BG_SURFACE)),
        ..container::Style::default()
    }
}

/// Style for role badge pills: a pill-shaped container with a translucent
/// version of the role's color and 4px rounded corners.
#[must_use]
pub fn role_badge_pill_style(_theme: &iced::Theme, color: Color) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(
            color.r, color.g, color.b, 0.1,
        ))),
        border: iced::Border {
            radius: 4.0.into(),
            ..iced::Border::default()
        },
        ..container::Style::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_match() {
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
            let name = role.as_str();
            assert_eq!(
                role_badge_color_for(&role),
                role_badge_color(name),
                "role_badge_color_for and role_badge_color must agree for {name}"
            );
        }
    }

    #[test]
    fn derivative_analyst_names_get_correct_color() {
        let analyst_color = role_badge_color_for(&crate::Role::Analyst);
        assert_eq!(role_badge_color("analyst_1"), analyst_color);
        assert_eq!(role_badge_color("analyst_2"), analyst_color);
        assert_eq!(role_badge_color("analyst_3"), analyst_color);
    }

    #[test]
    fn derivative_other_role_names_get_correct_color() {
        for role in <crate::Role as strum::IntoEnumIterator>::iter() {
            let name = role.as_str();
            let expected = role_badge_color_for(&role);
            assert_eq!(
                role_badge_color(&format!("{name}_1")),
                expected,
                "derivative {name}_1 should match role_badge_color_for"
            );
            assert_eq!(
                role_badge_color(&format!("{name}_42")),
                expected,
                "derivative {name}_42 should match role_badge_color_for"
            );
        }
    }

    #[test]
    fn non_numeric_suffix_is_unknown() {
        assert_eq!(role_badge_color("analyst_final"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("coder_abc"), (TEXT_MUTED, HOVER));
    }

    #[test]
    fn llm_api_roles_are_unknown() {
        assert_eq!(role_badge_color("user"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("assistant"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("system"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("tool"), (TEXT_MUTED, HOVER));
    }

    #[test]
    fn empty_and_garbage_are_unknown() {
        assert_eq!(role_badge_color(""), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("garbage"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("unknown_role"), (TEXT_MUTED, HOVER));
        assert_eq!(role_badge_color("_1"), (TEXT_MUTED, HOVER));
    }

    #[test]
    fn case_insensitive_parse() {
        let analyst_color = role_badge_color_for(&crate::Role::Analyst);
        assert_eq!(role_badge_color("ANALYST"), analyst_color);
        assert_eq!(role_badge_color("Analyst"), analyst_color);
        assert_eq!(role_badge_color("ANALYST_1"), analyst_color);
    }
}

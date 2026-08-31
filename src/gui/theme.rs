//! Dashboard theme — defines the Flexoki dark color palette, ticket phase colors, role badge
//! colors, log level colors, and workspace status colors for the native Iced GUI.

use iced::Background;
use iced::Color;
use iced::border;
use iced::widget::{container, pick_list, scrollable, toggler};

use crate::WorkspaceStatus;
use crate::pipeline::board::TicketPhase;

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

// ── Status colors ─────────────────────────────────────────────────

pub const STATUS_SUCCESS: Color = Color::from_rgb(0.0, 0.902, 0.541); // #00e68a
pub const STATUS_WARNING: Color = Color::from_rgb(1.0, 0.667, 0.0); // #ffaa00
pub const STATUS_ERROR: Color = Color::from_rgb(1.0, 0.267, 0.4); // #ff4466

// ── Diff widget tints (derived from STATUS_* palette) ─────────────

pub const DIFF_ADDED_TINT: Color =
    Color::from_rgba(STATUS_SUCCESS.r, STATUS_SUCCESS.g, STATUS_SUCCESS.b, 0.10);
pub const DIFF_REMOVED_TINT: Color =
    Color::from_rgba(STATUS_ERROR.r, STATUS_ERROR.g, STATUS_ERROR.b, 0.10);
pub const DIFF_FILE_HEADER_BG: Color =
    Color::from_rgba(STATUS_WARNING.r, STATUS_WARNING.g, STATUS_WARNING.b, 0.06);

// ── Log level colors ──────────────────────────────────────────────

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

// ── Role badge colors (from roleConfig.ts) ───────────────────────

/// Translucent pill background for a badge foreground color: the foreground
/// at 0.1 alpha. Single source of the badge-background math — the second
/// member of every [`role_badge_color_for`] / [`role_badge_color`] tuple is
/// `badge_bg(fg)` (including the unknown-role fallback), as is the background
/// member of [`workspace_status_color`]. Consumers (role pill, logs span,
/// workspace status pill) use that member directly, so rendering cannot
/// drift from this alpha.
const fn badge_bg(fg: Color) -> Color {
    Color::from_rgba(fg.r, fg.g, fg.b, 0.1)
}

/// Returns the badge (foreground, background) color for a given [`crate::Role`].
///
/// Reads from [`crate::agent::role::role_info()`] and converts the RGB tuple to
/// [`iced::Color`] — this avoids duplicating color data in an exhaustive match.
/// Adding a new [`crate::Role`] variant requires updating the `role_info()`
/// match (`badge_fg` field); the compiler will not catch a missing field here
/// (it defaults from `BASE_ROLE_INFO`), but the `badge_colors_set` test in
/// `role.rs` guards against silent black fallthrough.
///
/// The background member is always the foreground at 0.1 alpha ([`badge_bg`]);
/// consumers that render the pill background use it directly.
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
/// Accepts canonical names (e.g. `"analyst"`), derivative names with a
/// numeric suffix (e.g. `"analyst_1"`, `"analyst_2"`), and the joint-comment
/// stage roles ("Analysis"/"Review"/"QA" — the comment role is the stage
/// name, per the joint-verdict pipeline). Unknown strings (including LLM API
/// roles like `"user"`, `"assistant"`, `"system"`, `"tool"`) fall back to a
/// muted grey.
///
/// The background member is always the foreground at 0.1 alpha ([`badge_bg`]) —
/// including the fallback, so consuming the second member directly (role pill,
/// logs span) renders the same alpha-scaled background for unknown roles as
/// for canonical ones.
///
/// Delegates to [`role_badge_color_for`] after resolving the string, which
/// reads colors from [`crate::agent::role::role_info()`] as the single source of truth.
#[must_use]
pub fn role_badge_color(role: &str) -> (Color, Color) {
    // Stage-name comment roles from the joint-verdict pipeline ("Analysis"/
    // "Review"/"QA" — the comment role is the stage name). Resolved via the
    // shared inverse mapping so it can't drift from verdict::stage_name.
    if let Some(r) = crate::pipeline::verdict::stage_role(role) {
        return role_badge_color_for(&r);
    }

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

    // The background member must stay `badge_bg(fg)` (not an independent
    // constant): consumers render it directly, so this preserves the
    // pre-consolidation rendering for unknown roles.
    (TEXT_MUTED, badge_bg(TEXT_MUTED))
}

// ── Role icon mapping (shared between sidebar and workspaces) ──────

/// Returns the Lucide icon widget for a given agent role.
///
/// Callers apply `.size()`, `.color()`, and `.into()` to style the icon
/// for their specific context.
#[must_use]
#[allow(clippy::trivially_copy_pass_by_ref)]
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
        crate::Role::Assistant => lucide::message_square(),
        crate::Role::Support => lucide::life_buoy(),
    }
}

/// Returns the Lucide icon widget for the workspace GENERAL context
/// (the discovery-produced summary for non-role LLM calls, shown in the
/// Settings → Workspaces row).
#[must_use]
pub fn general_context_icon() -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    lucide::book_open_text()
}

/// Returns the Lucide icon widget for the diagnostics pipeline role
/// (a plain `"diagnostics"` string, not a [`crate::Role`] variant).
#[must_use]
pub fn diagnostics_icon() -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    lucide::stethoscope()
}

/// Returns the chat-style role icon and its badge foreground color for a
/// ticket-comment author label, or `None` when the label must render as
/// plain muted text.
///
/// Only the exact author values the pipeline writes today resolve to an
/// icon: the joint-verdict stage names ("Analysis"/"Review"/"QA", resolved
/// via the shared [`crate::pipeline::verdict::stage_role`] inverse mapping),
/// the canonical agent names "engineer"/"manager"/"sanitation", and
/// "diagnostics" (stethoscope, [`TEXT_PRIMARY`] tint). Legacy or unexpected
/// values (e.g. old suffixed labels like "analyst_1", "system",
/// "user:{name}") intentionally return `None`.
///
/// Delegates to [`role_badge_color_for`] for the foreground color, and to
/// [`role_icon`] for the glyph.
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
        "diagnostics" => Some((diagnostics_icon(), TEXT_PRIMARY)),
        _ => None,
    }
}

/// Dashboard default font (JetBrains Mono). Registered via `.default_font()`
/// in `main.rs`; fonts embedded via `.font()` calls there.
pub const JETBRAINS_MONO: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Normal,
    stretch: iced::font::Stretch::Normal,
    style: iced::font::Style::Normal,
};

/// Bold weight variant of JetBrains Mono (the dashboard default font).
pub const FONT_BOLD: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Bold,
    ..iced::Font::DEFAULT
};

/// Regular weight variant of JetBrains Mono.
pub const FONT_REGULAR: iced::Font = JETBRAINS_MONO;

/// Italic variant of JetBrains Mono (narration text).
pub const FONT_ITALIC: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    style: iced::font::Style::Italic,
    ..iced::Font::DEFAULT
};

/// Body text size (px) of the transcript markdown renderer
/// (`markdown_settings`). The sessions-page collapse measurement measures
/// body elements at this size (narration at [`NARRATION_TEXT_SIZE`]), so a
/// theme font-size change cannot silently drift the measured wrap count away
/// from the actual render.
pub const MARKDOWN_TEXT_SIZE: f32 = 13.0;

/// Narration text size (px) — the italic narration line on Running Agents
/// cards and the narration body on the Sessions transcript. The
/// sessions-page collapse measurement measures narration at this size.
pub const NARRATION_TEXT_SIZE: f32 = 14.0;

/// Markdown rendering settings consistent with the Flexoki dark theme.
#[must_use]
pub fn markdown_settings() -> iced::widget::markdown::Settings {
    markdown_settings_with(FONT_REGULAR, MARKDOWN_TEXT_SIZE)
}

/// Markdown rendering settings for narration bodies: italic face at
/// [`NARRATION_TEXT_SIZE`].
#[must_use]
pub fn narration_markdown_settings() -> iced::widget::markdown::Settings {
    markdown_settings_with(FONT_ITALIC, NARRATION_TEXT_SIZE)
}

/// Markdown rendering settings with an explicit body face and text size.
/// Fenced code blocks always stay in the regular face (never italic) while
/// the body and inline code take the passed `font`.
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

// ── Ticket phase colors ───────────────────────────────────────────
// All TicketPhase variants exhaustively matched — no catch-all.

/// Returns a `(foreground, background)` tuple — the badge-pill convention
/// shared with `widgets::badge_pill`. The first member is the readable text
/// color and the second the darker pill background.
#[must_use]
pub const fn ticket_phase_color(phase: TicketPhase) -> (Color, Color) {
    use TicketPhase::{
        Analysis, Backlog, Cancelled, Done, Failed, InDevelopment, InDiagnostics, InQa, InReview,
        InSanitation, Planning, Queued,
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
        // Diagnostic phases — amber/teal
        InDiagnostics => (
            Color::from_rgb(0.902, 0.863, 0.784),
            Color::from_rgb(0.310, 0.224, 0.102),
        ),
        // Sanitation phases — neutral gray
        InSanitation => (
            Color::from_rgb(0.788, 0.788, 0.788),
            Color::from_rgb(0.310, 0.310, 0.310),
        ),
        // Review & QA
        InReview => (
            Color::from_rgb(0.816, 0.816, 0.933),
            Color::from_rgb(0.184, 0.216, 0.380),
        ),
        InQa => (
            Color::from_rgb(0.816, 0.816, 0.933),
            Color::from_rgb(0.216, 0.184, 0.380),
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

// ── Ticket priority chip colors (Flexoki muted palette) ──────────
//
// Priority uses the same muted-tone approach as phase badges:
// lighter, readable text on darker backgrounds (a dark theme).
//
// P0 (urgent):     red     — failed-like
// P1 (high):       orange  — in-development-like
// P2 (medium):     yellow  — in-diagnostics-like
// P3 (low):        green   — qa-passed-like
// P4+ (lowest):    green   — done-like

/// Returns a `(foreground, background)` tuple — the badge-pill convention
/// shared with `widgets::badge_pill`. The first member is the readable text
/// color and the second the darker pill background.
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

/// One-shot guard to log timestamp parse failure only once.
static TIMESTAMP_PARSE_WARNED: AtomicBool = AtomicBool::new(false);

/// Format an ISO 8601 timestamp string into a human-readable absolute form in
/// the machine's local timezone. Output style: "Jun 5, 21:54" — no
/// microseconds, no raw timezone suffixes. Storage stays UTC; only the
/// displayed wall-clock is converted from the parsed instant to local.
/// If parsing fails, returns the first 16 characters as a fallback.
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

/// Render the local-time HH:MM:SS portion of an ISO 8601 timestamp, or the
/// full string when shorter than 20 chars. Char-boundary safe (mirrors
/// `format_timestamp`'s hardening). On parse failure the raw `ts[11..19]`
/// slice (or the whole string when short) is kept unchanged, so malformed
/// inputs degrade to their stored (UTC) wall-clock rather than being dropped.
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

/// Render a relative timestamp label for a chat message, computed against
/// the supplied local `now`. Today's messages render as local HH:MM,
/// yesterday as "yesterday", and older messages as "X days ago" (calendar-day
/// difference). Future timestamps / clock skew (negative day difference) are
/// treated as today and render as HH:MM; invalid timestamps render as an
/// empty string.
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

/// Compact session-length format: raw below 1,000 ("500"), one decimal in the
/// k-range ("12.3k" for 12300), one decimal at/above one million ("1.2M") —
/// sessions routinely exceed 200K tokens with context windows up to 1M.
/// Values that would round up past a unit boundary are rendered in the next
/// unit (999_999 → "1.0M", never "1000.0k"). Shared by the Running Agents
/// card and the Sessions page session cards.
#[expect(clippy::cast_precision_loss)] // token counts are far below f64's 2^53 exact range
pub(crate) fn format_compact_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 999_950 {
        // k-range stops below values that would round to "1000.0k"
        // (999_950 / 1000 = 999.95).
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else if tokens < 999_950_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else {
        format!("{:.1}G", tokens as f64 / 1_000_000_000.0)
    }
}

// ── Workspace status colors ───────────────────────────────────────

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

// ── Animation timing constants ────────────────────────────────────

/// Log entry fade‑in duration (ms).
pub const ANIM_LOG_FADE_MS: u64 = 100;
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

/// Transparent-background button style with a hover/press highlight. Shared
/// factory for the chat composer controls (role/mic, send button) and dropdown
/// menu items — the two differ only in highlight colors and corner radius.
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

/// Icon-only button with a subtle hover/press background (chat composer
/// role/mic controls, send button). Pass `disabled: true` to suppress the
/// highlight (used for the greyed send button on empty input).
#[must_use = "button style factory returns a style closure"]
pub fn icon_button_style(
    disabled: bool,
) -> impl Fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style {
    transparent_button_style(HOVER_STRONG, ACCENT_DIM, 6.0, disabled)
}

/// Tab-button style for editor/shell tab bars: active tabs use [`BG_ELEVATED`],
/// hovered [`HOVER`], otherwise [`BG_SURFACE`], with a zero-radius border.
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

/// Shared text-button factory: hover/pressed background plus an enabled
/// text color that differs between the normal and danger variants.
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

/// Text-only danger button (Cancel in modals, delete triggers). Like [`button_text`]
/// but with red text. No colored background, subtle hover highlight, red text only.
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

/// Flexoki-dark themed style for [`fn@pick_list`] widgets.
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

/// Flexoki-dark themed style for the expanded menu of [`fn@pick_list`] widgets,
/// matching the ContextMenu/RoleMenu overlay look (BG_ELEVATED fill, radius-4
/// hairline border, TEXT_SECONDARY items highlighted to TEXT_PRIMARY on hover,
/// no shadow). The menu's internal overflow scrollbar is styled by iced's
/// default scrollable catalog and is not reachable through this API.
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

/// Flexoki-dark themed style for [`fn@toggler`] widgets.
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

// ── Container styles ─────────────────────────────────────────────

/// Shared container style factory: background fill plus border parameters.
/// Public wrappers below delegate here so the variants stay in one place.
/// Callers with a one-off combination may use this directly.
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

/// Style for tooltip containers: elevated background with subtle rounded
/// corners and a hairline border, matching the `dialog_container_style`
/// convention used for modal dialogs.
///
/// Applying this via `.style(theme::tooltip_style)` gives every tooltip a
/// dark/neutral fill that stays readable regardless of what content is
/// underneath it.
#[must_use]
pub fn tooltip_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 6.0, 1.0, BORDER_STRONG)
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
    container_style(BG_ELEVATED, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for surface cards: surface background with a 1px border and
/// 4px rounded corners. Used for ticket detail sections, comment cards,
/// log entries, and session transcript messages.
#[must_use]
pub fn surface_card_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_SURFACE, 4.0, 1.0, BORDER)
}

/// Style for elevated cards: elevated background with a 1px border and
/// 4px rounded corners. Used for ticket board cards and session round cards.
#[must_use]
pub fn elevated_card_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 4.0, 1.0, BORDER)
}

/// Style for modal dialog containers: elevated background, 8px rounded
/// corners, and a strong border. Shared by all modal overlays across the
/// dashboard (board detail, settings dialogs, editor overlays, diff/branch
/// modals, etc.).
#[must_use]
pub fn dialog_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_ELEVATED, 8.0, 1.0, BORDER_STRONG)
}

/// Style for the base page background: just the BG_BASE fill with no border.
/// Used as the outermost container on most pages (home, sessions, logs).
#[must_use]
pub fn base_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_BASE, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for surface-only containers: surface background with no border.
/// Used for sidebar panels, tab bars, and filter bars.
#[must_use]
pub fn surface_container_style(_theme: &iced::Theme) -> container::Style {
    container_style(BG_SURFACE, 0.0, 0.0, Color::TRANSPARENT)
}

/// Style for badge pills: a rounded container with the given background and
/// no border (canonical 4px radius). Used for log level tags, tool badges,
/// workspace status pills, and notes panels.
pub fn pill_style(bg: Color) -> impl Fn(&iced::Theme) -> container::Style {
    move |_theme: &iced::Theme| container_style(bg, 4.0, 0.0, Color::TRANSPARENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;
    use strum::IntoEnumIterator;

    #[test]
    fn canonical_names_match() {
        for role in Role::iter() {
            let name = role.as_str();
            assert_eq!(
                role_badge_color_for(&role),
                role_badge_color(name),
                "role_badge_color_for and role_badge_color must agree for {name}"
            );
        }
    }

    #[test]
    fn derivative_other_role_names_get_correct_color() {
        for role in Role::iter() {
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
        assert_eq!(
            role_badge_color("analyst_final"),
            (TEXT_MUTED, badge_bg(TEXT_MUTED))
        );
        assert_eq!(
            role_badge_color("coder_abc"),
            (TEXT_MUTED, badge_bg(TEXT_MUTED))
        );
    }

    #[test]
    fn compact_token_format_edges() {
        // Raw below 1,000; one decimal in the k-range; one decimal at/above
        // 1M (sessions routinely exceed 200K with context windows up to 1M).
        // Values rounding up past a unit boundary render in the next unit.
        assert_eq!(format_compact_tokens(0), "0");
        assert_eq!(format_compact_tokens(500), "500");
        assert_eq!(format_compact_tokens(999), "999");
        assert_eq!(format_compact_tokens(1_000), "1.0k");
        assert_eq!(format_compact_tokens(12_300), "12.3k");
        assert_eq!(format_compact_tokens(200_000), "200.0k");
        assert_eq!(format_compact_tokens(999_949), "999.9k");
        assert_eq!(format_compact_tokens(999_999), "1.0M");
        assert_eq!(format_compact_tokens(1_000_000), "1.0M");
        assert_eq!(format_compact_tokens(1_234_567), "1.2M");
        assert_eq!(format_compact_tokens(999_949_999), "999.9M");
        assert_eq!(format_compact_tokens(999_950_000), "1.0G");
        assert_eq!(format_compact_tokens(1_000_000_000), "1.0G");
    }

    #[test]
    fn joint_comment_stage_roles_get_their_role_colors() {
        // The joint-verdict pipeline writes comments with the STAGE NAME as the
        // comment role ("Analysis"/"Review"/"QA") — they must render with the
        // corresponding role color, not the muted-grey fallback.
        assert_eq!(
            role_badge_color("Analysis"),
            role_badge_color_for(&crate::Role::Analyst)
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

    #[test]
    fn llm_api_roles_are_unknown() {
        assert_eq!(role_badge_color("user"), (TEXT_MUTED, badge_bg(TEXT_MUTED)));
        assert_eq!(
            role_badge_color("system"),
            (TEXT_MUTED, badge_bg(TEXT_MUTED))
        );
        assert_eq!(role_badge_color("tool"), (TEXT_MUTED, badge_bg(TEXT_MUTED)));
    }

    #[test]
    fn empty_and_garbage_are_unknown() {
        assert_eq!(role_badge_color(""), (TEXT_MUTED, badge_bg(TEXT_MUTED)));
        assert_eq!(
            role_badge_color("garbage"),
            (TEXT_MUTED, badge_bg(TEXT_MUTED))
        );
        assert_eq!(
            role_badge_color("unknown_role"),
            (TEXT_MUTED, badge_bg(TEXT_MUTED))
        );
        assert_eq!(role_badge_color("_1"), (TEXT_MUTED, badge_bg(TEXT_MUTED)));
    }

    #[test]
    fn case_insensitive_parse() {
        let analyst_color = role_badge_color_for(&crate::Role::Analyst);
        assert_eq!(role_badge_color("ANALYST"), analyst_color);
        assert_eq!(role_badge_color("Analyst"), analyst_color);
        assert_eq!(role_badge_color("ANALYST_1"), analyst_color);
    }

    /// The badge tuple's background member must always be the foreground at
    /// 0.1 alpha ([`badge_bg`]). The role pill (`pill_style(colors.1)`) and the
    /// logs span consume the second member directly, so any divergence — e.g.
    /// an independent constant like the old `(TEXT_MUTED, HOVER)` fallback —
    /// would silently change the rendered fallback background.
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
            "coder_abc",
            "user",
            "assistant",
            "system",
            "tool",
            "Analysis",
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

    /// Locks the pre-consolidation values of the container factories, the
    /// button_text/button_text_danger pair, the pick_list field/menu styles,
    /// and the toggler style so parameterization cannot silently change any
    /// rendered style.
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
    fn relative_time_today_yesterday_days_ago() {
        use chrono::TimeZone;
        let now = chrono::Local
            .with_ymd_and_hms(2026, 8, 28, 12, 0, 0)
            .single()
            .unwrap();
        let off = now.offset().clone();
        assert_eq!(
            format_relative_time(&format!("2026-08-28T09:15:00{off}"), now),
            "09:15"
        );
        assert_eq!(
            format_relative_time(&format!("2026-08-27T09:15:00{off}"), now),
            "yesterday"
        );
        assert_eq!(
            format_relative_time(&format!("2026-08-25T09:15:00{off}"), now),
            "3 days ago"
        );
    }

    #[test]
    fn relative_time_future_treated_as_today() {
        use chrono::TimeZone;
        let now = chrono::Local
            .with_ymd_and_hms(2026, 8, 28, 12, 0, 0)
            .single()
            .unwrap();
        let off = now.offset().clone();
        assert_eq!(
            format_relative_time(&format!("2026-08-29T09:15:00{off}"), now),
            "09:15"
        );
    }

    #[test]
    fn relative_time_invalid_returns_empty() {
        assert_eq!(
            format_relative_time("not-a-timestamp", chrono::Local::now()),
            ""
        );
    }

    #[test]
    fn comment_author_icons_resolve_exact_pipeline_labels_only() {
        let (analysis_fg, _) = role_badge_color_for(&crate::Role::Analyst);
        assert_eq!(
            comment_author_icon("Analysis").map(|(_, color)| color),
            Some(analysis_fg)
        );

        for author in [
            "Analysis",
            "Review",
            "QA",
            "engineer",
            "manager",
            "sanitation",
            "diagnostics",
        ] {
            assert!(
                comment_author_icon(author).is_some(),
                "{author} should resolve to an icon"
            );
        }
        for author in ["system", "user:admin", "analyst_1", "coder", ""] {
            assert!(
                comment_author_icon(author).is_none(),
                "{author} must render as plain muted text"
            );
        }
    }
}

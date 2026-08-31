//! Shared dashboard widgets: styled pick_list, PickOption type, FileTree state struct
//! and build_tree_panel for shared file-tree panel rendering.

use super::theme;
use iced::advanced::layout::{self, Layout};
use iced::advanced::overlay;
use iced::advanced::renderer;
use iced::advanced::widget::Operation;
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{Clipboard, Shell, Widget};
use iced::mouse;
use iced::widget::{
    self, Column, Row, Space, button, column, container, mouse_area, pick_list, row, scrollable,
    stack, text, tooltip,
};
use iced::{
    Alignment, Color, Element, Event, Length, Padding, Point, Rectangle, Size, Task, Vector,
};
use iced_fonts::lucide;
use iced_selection;
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

/// An option for [`fn@pick_list`] with separate value and display label.
///
/// Equality is determined by `value` only — two `PickOption`s with the same
/// `value` are considered equal regardless of label. This lets [`fn@pick_list`]
/// highlight the correct option even when the selected value is constructed
/// independently of the options list.
#[derive(Debug, Clone)]
pub struct PickOption {
    pub value: String,
    pub label: String,
}

impl PartialEq for PickOption {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for PickOption {}

impl std::fmt::Display for PickOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// Flexoki-dark themed style for [`fn@pick_list`] widgets.
#[must_use]
pub fn pick_list_style(_theme: &iced::Theme, _status: pick_list::Status) -> pick_list::Style {
    pick_list::Style {
        text_color: theme::TEXT_PRIMARY,
        placeholder_color: theme::TEXT_MUTED,
        handle_color: theme::TEXT_MUTED,
        background: iced::Background::Color(theme::BG_ELEVATED),
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: theme::BORDER_STRONG,
        },
    }
}

// ── Shared single-line editor field ──────────────────────────────────

/// A single shared-editor field frame matching the app's `text_input` visual
/// language (`BG_ELEVATED` fill, radius 4, hairline border).
pub(crate) fn editor_field_style(border_color: Color) -> impl Fn(&iced::Theme) -> container::Style {
    move |_theme: &iced::Theme| theme::container_style(theme::BG_ELEVATED, 4.0, 1.0, border_color)
}

/// Render a single-line shared editor field, replacing iced `text_input`.
///
/// `submit_on_enter` selects the bare-Enter behavior: `true` submits the
/// field's action, `false` leaves Enter a no-op. The buffer is owned by the
/// caller; the caller applies [`super::editor_widget::EditorAction`] via
/// [`super::common::apply_editor_action`] and reads `buffer.text()`.
///
/// `id` makes the field click-to-focus and gates keyboard processing on focus.
/// Tab in a single-line field emits a focus action that the page maps to
/// `operation::focus_next()` / `focus_previous()` (the code editor keeps
/// Tab→Indent itself). Pass `None` when the field does not need its own focus
/// id.
#[must_use]
pub fn single_line_editor<'a, M: 'a>(
    buffer: &'a super::editor_widget::EditorBuffer,
    placeholder: &'a str,
    submit_on_enter: bool,
    width: Length,
    id: Option<iced::widget::Id>,
    on_action: impl Fn(super::editor_widget::EditorAction) -> M + 'a,
) -> Element<'a, M> {
    let mut editor = super::editor_widget::EditorWidget::new(buffer)
        .single_line(true)
        .show_gutter(false)
        .code_mode(false)
        .enter(if submit_on_enter {
            super::editor_widget::EnterBehavior::Submit
        } else {
            super::editor_widget::EnterBehavior::Newline
        })
        .placeholder(placeholder)
        .padding(5.0)
        .background(Some(theme::BG_ELEVATED));
    if let Some(id) = id {
        editor = editor.id(id);
    }
    let element = iced::Element::new(editor).map(on_action);
    container(element)
        .width(width)
        .style(editor_field_style(theme::BORDER_STRONG))
        .into()
}

/// Render a masked single-line shared editor field with a lucide show/hide
/// toggle (the mismatched glyph button is replaced by `eye`/`eye_off`).
///
/// Copy always returns the plaintext value; masking only affects rendering.
/// `id` makes the field click-to-focus and gates keyboard processing on focus.
/// Tab emits a focus action the page maps to `focus_next` / `focus_previous`
/// (single-line fields never indent).
#[must_use]
#[expect(clippy::too_many_arguments)]
pub fn password_field_editor<'a, M: Clone + 'a>(
    buffer: &'a super::editor_widget::EditorBuffer,
    placeholder: &'a str,
    show: bool,
    width: Length,
    highlight: bool,
    id: Option<iced::widget::Id>,
    on_action: impl Fn(super::editor_widget::EditorAction) -> M + 'a,
    on_toggle: M,
) -> Element<'a, M> {
    let border_color = if highlight {
        theme::ACCENT
    } else {
        theme::BORDER_STRONG
    };
    let mut editor = super::editor_widget::EditorWidget::new(buffer)
        .single_line(true)
        .masked(!show)
        .show_gutter(false)
        .code_mode(false)
        .enter(super::editor_widget::EnterBehavior::Submit)
        .placeholder(placeholder)
        .padding(5.0)
        .background(Some(theme::BG_ELEVATED));
    if let Some(id) = id {
        editor = editor.id(id);
    }
    let element = iced::Element::new(editor).map(on_action);
    let field = container(element)
        .width(width)
        .style(editor_field_style(border_color));

    let eye_icon: Element<'_, M> = if show {
        lucide::eye_off::<iced::Theme, iced::Renderer>()
            .size(14)
            .color(theme::TEXT_MUTED)
            .into()
    } else {
        lucide::eye::<iced::Theme, iced::Renderer>()
            .size(14)
            .color(theme::TEXT_MUTED)
            .into()
    };
    let toggle = button(eye_icon)
        .padding(2)
        .style(theme::button_secondary)
        .on_press(on_toggle);

    row![field, Space::new().width(4), toggle]
        .align_y(Alignment::Center)
        .into()
}

/// Render a styled error banner for dashboard panels.
#[must_use]
pub fn error_banner<'a, Message: 'a>(err: &'a str) -> Element<'a, Message> {
    container(text(err).size(13).color(theme::STATUS_ERROR))
        .padding(8)
        .style(theme::pill_style(theme::STATUS_ERROR.scale_alpha(0.08)))
        .into()
}

/// Standardized "Loading..." placeholder label for load-state scaffolding.
#[must_use]
pub fn loading_text<'a, Message: 'a>() -> Element<'a, Message> {
    text("Loading...").size(14).color(theme::TEXT_MUTED).into()
}

/// Push an [`error_banner`] plus trailing 8px spacer onto `col` when `err` is present.
#[must_use]
pub fn push_error_banner<'a, Message: 'a>(
    mut col: Column<'a, Message>,
    err: Option<&'a str>,
) -> Column<'a, Message> {
    if let Some(err) = err {
        col = col.push(error_banner(err));
        col = col.push(Space::new().height(8));
    }
    col
}

/// Render a centered empty-state placeholder with a lucide icon and label.
#[must_use]
pub fn empty_state_placeholder<'a, Message: 'a>(
    icon: iced::widget::Text<'a, iced::Theme, iced::Renderer>,
    label: &'a str,
) -> Element<'a, Message> {
    container(
        column![
            icon.size(48).color(theme::TEXT_MUTED),
            text(label).size(14).color(theme::TEXT_MUTED),
        ]
        .spacing(12)
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .center_x(Length::Fill)
    .center_y(Length::Fill)
    .into()
}

/// Badge pill with an opaque background; `colors` is a `(background, text)` tuple.
#[must_use]
pub fn badge_pill<'a, Message: 'a>(
    label: String,
    colors: (Color, Color),
    text_size: u32,
    padding: [u16; 2],
) -> Element<'a, Message> {
    container(text(label).size(text_size).color(colors.1))
        .padding(padding)
        .style(theme::pill_style(colors.0))
        .into()
}

/// Role badge pill: a container with the role name, caller-specified padding,
/// and the translucent role-colored pill background (canonical 4px radius).
///
/// Takes the role as an owned `String` (not `&str`) because the sessions
/// transcript renders move a loop-local String into an Element that outlives
/// the iteration — a borrowed parameter would not compile there; call sites
/// that only have a borrow pay a trivial `.clone()`.
///
/// `colors` is the `(foreground, background)` tuple from
/// [`theme::role_badge_color`] / [`theme::role_badge_color_for`]; the
/// background member (always the foreground at 0.1 alpha — that math lives
/// in exactly one place, `theme::badge_bg`) feeds [`theme::pill_style`].
///
/// `padding` is the container padding `[vertical, horizontal]`, passed
/// through to the container's padding builder; the board comment rows use
/// the enlarged `[2, 12]` so the role stands out while scrolling, while the
/// sessions transcript and tool-failure metadata rows keep the compact
/// `[1, 6]`.
///
/// `selectable` chooses between plain [`text`] and [`selectable_text`]: both
/// arms coerce into `Element` via `.into()`, but plain `text` is cheaper (no
/// selection machinery) while `selectable_text` lets the role name be
/// selected/copied from the UI. The sessions transcript uses selectable text
/// so a whole line can be copied in one drag; the board comment rows and
/// tool-failure metadata rows use plain text.
#[must_use]
pub fn role_badge<'a, Message: 'a>(
    role: String,
    colors: (Color, Color),
    text_size: u32,
    padding: [u16; 2],
    selectable: bool,
) -> Element<'a, Message> {
    let label: Element<'a, Message> = if selectable {
        selectable_text(role, colors.0).size(text_size).into()
    } else {
        text(role).size(text_size).color(colors.0).into()
    };
    container(label)
        .padding(padding)
        .style(theme::pill_style(colors.1))
        .into()
}

/// Maintainer icon + ON/OFF badge shared by the sidebar Maintainer toggle and
/// the Settings workspace-row Maintainer toggle; the wrapping toggle button
/// stays with each caller.
#[must_use]
pub fn maint_badge<'a, Message: 'a>(enabled: bool) -> Column<'a, Message> {
    column![
        theme::role_icon(&crate::Role::Maintainer)
            .size(11)
            .color(if enabled {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            }),
        text(if enabled { "ON" } else { "OFF" })
            .size(9)
            .color(if enabled {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            }),
    ]
    .spacing(0)
    .align_x(Alignment::Center)
}

/// Icon size of the board's ticket-card action/archive buttons.
pub const ACTION_ICON_SIZE: f32 = 16.0;

/// Shared icon/action button wrapped in a tooltip.
///
/// Replicates the `tooltip(button(icon).style(..).padding(..).on_press_maybe(..),
/// text(..).size(11), pos)` idiom plus `theme::tooltip_style`. `padding` must
/// mirror the original site exactly — pass [`iced::widget::button::DEFAULT_PADDING`]
/// where there was no explicit `.padding()`. Full-width / container-wrapped
/// variants keep their bespoke construction.
#[must_use]
pub fn icon_tooltip_button<'a, Message>(
    icon: impl Into<Element<'a, Message>>,
    tooltip_text: impl Into<Cow<'a, str>>,
    on_press: Option<Message>,
    padding: impl Into<Padding>,
    style: impl Fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style + 'a,
    position: tooltip::Position,
) -> Element<'a, Message>
where
    Message: Clone + 'a,
{
    tooltip(
        button(icon)
            .style(style)
            .padding(padding)
            .on_press_maybe(on_press),
        text(tooltip_text.into()).size(11),
        position,
    )
    .style(theme::tooltip_style)
    .into()
}

/// Invisible, non-interactive, zero-width replica of an icon button:
/// an [`iced_fonts::lucide`] icon at [`ACTION_ICON_SIZE`] inside a
/// default-padded [`button`], clipped to zero width by its container.
///
/// Push it into an action-icon row whose real buttons are all
/// conditionally absent (e.g. archived ticket cards): it contributes no
/// width and paints nothing, but pins the row's height to whatever the
/// real buttons produce, so the surrounding row never collapses.
#[must_use]
pub fn ghost_icon_button<Message: 'static + Clone>() -> Element<'static, Message> {
    container(
        button(lucide::circle_check::<iced::Theme, iced::Renderer>().size(ACTION_ICON_SIZE))
            .padding(button::DEFAULT_PADDING),
    )
    .width(0)
    .clip(true)
    .into()
}

/// Create a selectable text widget with the given color.
///
/// Accepts both borrowed (`&str`) and owned (`String`) text content.
pub fn selectable_text<'a>(
    content: impl iced_selection::text::IntoFragment<'a>,
    color: Color,
) -> iced_selection::text::Text<'a, iced::Theme, iced::Renderer> {
    iced_selection::text::Text::new(content).style(move |_theme| iced_selection::text::Style {
        color: Some(color),
        selection: theme::ACCENT_DIM,
    })
}

/// Close button for editor/shell tab bars: a 12px lucide X colored by tab
/// active state (secondary vs faint text).
#[must_use]
pub fn tab_close_button<'a, Message: Clone + 'a>(
    is_active: bool,
    on_press: Message,
) -> widget::Button<'a, Message> {
    widget::button(
        lucide::x::<iced::Theme, iced::Renderer>()
            .size(12)
            .color(if is_active {
                theme::TEXT_SECONDARY
            } else {
                theme::TEXT_FAINT
            }),
    )
    .on_press(on_press)
    .style(theme::button_transparent)
    .padding(0)
}

/// Wrap a tab strip in the shared scrollable + surface-container chrome.
/// `scroll_id` is optional — the editor passes one for scroll-to-active-tab.
/// `on_scroll` is optional — the editor passes a closure tracking the
/// [`scrollable::Viewport`] so its reveal logic can decide whether the
/// active tab is visible; the shell passes `None` (its strip never needs
/// programmatic reveal).
#[must_use]
pub fn tab_scrollable<'a, Message: 'a>(
    tab_buttons: Vec<Element<'a, Message>>,
    scroll_id: Option<widget::Id>,
    on_scroll: Option<impl Fn(scrollable::Viewport) -> Message + 'a>,
) -> Element<'a, Message> {
    let mut sc = scrollable(row(tab_buttons).spacing(0).width(Length::Fill))
        .direction(theme::horizontal_scrollbar())
        .style(theme::scrollbar_style)
        .width(Length::Fill)
        .height(Length::Shrink);
    if let Some(id) = scroll_id {
        sc = sc.id(id);
    }
    if let Some(on_scroll) = on_scroll {
        sc = sc.on_scroll(on_scroll);
    }
    container(sc)
        .style(theme::surface_container_style)
        .width(Length::Fill)
        .into()
}

/// Unified chat vertical rhythm: the composer's horizontal inset, reused
/// as the Home inter-bubble gap, the Home composer's bottom padding, and
/// the Sessions transcript inter-entry spacing.
pub const CHAT_VERTICAL_RHYTHM: f32 = 8.0;

/// Options for [`chat_composer`] that differ between the Home and Board
/// pages. Bundled so the shared signature does not grow with page-specific
/// knobs.
pub struct ChatComposerOptions<'a, M> {
    /// A send is in flight — button disabled.
    pub sending: bool,
    /// Editor min/max heights in px.
    pub min_height: f32,
    pub max_height: f32,
    /// Action toolbar controls rendered alongside the send button inside the
    /// input surface (Home role/mic buttons; empty for the plain Board
    /// composer).
    pub controls: Vec<Element<'a, M>>,
    /// Grey the send button while the input is empty/whitespace-only.
    /// Home enables this (empty-input affordance); Board keeps its legacy
    /// always-active look.
    pub grey_on_empty: bool,
    /// Tooltip text for the send button — surface-specific wording
    /// ("send text message" on Home, "send comment" on the Board ticket
    /// modal). Shown on hover even while the button is disabled.
    pub send_tooltip: &'a str,
    /// Optional focus id for the composer editor, making it click-to-focus
    /// and gating keyboard processing on focus so keystrokes don't leak when
    /// it is not focused.
    pub id: Option<iced::widget::Id>,
}

/// Shared chat composer: a full-width bubble-styled input surface that
/// callers inset to keep small side margins, holding the multi-line prose
/// editor above a right-aligned action toolbar.
/// `on_action`/`send_msg` parameterize the page's messages; callers supply
/// the placeholder and a [`ChatComposerOptions`] bundle (editor min/max
/// heights, the sending flag, optional controls, and whether the send button
/// greys on empty input).
#[must_use]
pub fn chat_composer<'a, M: Clone + 'a>(
    content: &'a super::editor_widget::EditorBuffer,
    on_action: impl Fn(super::editor_widget::EditorAction) -> M + 'a,
    send_msg: M,
    placeholder: &'a str,
    options: ChatComposerOptions<'a, M>,
) -> Element<'a, M> {
    let send_msg_btn = send_msg.clone();

    let mut editor = super::editor_widget::EditorWidget::new(content)
        .show_gutter(false)
        .code_mode(false)
        .enter(super::editor_widget::EnterBehavior::Submit)
        .placeholder(placeholder)
        .min_height(options.min_height)
        .max_height(options.max_height)
        .padding(5.0)
        .background(Some(theme::BG_ELEVATED));
    if let Some(id) = options.id {
        editor = editor.id(id);
    }
    let input_editor: Element<'a, M> =
        container(iced::Element::new(editor).map(move |action| match action {
            super::editor_widget::EditorAction::Submit => send_msg.clone(),
            other => on_action(other),
        }))
        .width(Length::Fill)
        .height(Length::Shrink)
        .into();

    // Whitespace-only input counts as empty (greys the send button).
    // The emptiness check is gated on grey_on_empty so Board (legacy look)
    // never allocates content.text() per frame.
    let send_disabled =
        options.sending || (options.grey_on_empty && content.text().trim().is_empty());
    let send_btn = icon_tooltip_button(
        lucide::send::<iced::Theme, iced::Renderer>()
            .size(14)
            .color(if send_disabled {
                theme::TEXT_MUTED
            } else {
                theme::ACCENT
            }),
        options.send_tooltip,
        if send_disabled {
            None
        } else {
            Some(send_msg_btn)
        },
        4,
        theme::icon_button_style(send_disabled),
        tooltip::Position::Top,
    );

    // Right-aligned action toolbar inside the input surface: controls (Home
    // role/mic; empty for the plain Board composer) then the send button.
    let mut toolbar = Row::new().spacing(6).align_y(Alignment::Center);
    toolbar = toolbar.push(Space::new().width(Length::Fill));
    for c in options.controls {
        toolbar = toolbar.push(c);
    }
    toolbar = toolbar.push(send_btn);

    // Bubble-styled input surface: editor above, action icons below-right,
    // inside one container matching the user message bubble form, stretched to
    // the full width of the chat pane. Callers inset the bubble (Home's black
    // composer strip) to keep small side margins; Board's modal already pads it.
    container(column![input_editor, toolbar].spacing(6))
        .padding(10)
        .style(theme::bubble_style(
            theme::BG_ELEVATED,
            Some(theme::TEXT_PRIMARY),
        ))
        .width(Length::Fill)
        .into()
}

/// Render formatted diff stats (+X/−Y) matching ticket card style.
///
/// Returns a [`Row`] showing only non-zero sides with a `/` separator.
/// Returns an empty [`Row`] when both `added` and `removed` are zero.
///
/// Callers typically wrap this in a styled [`button()`] with an appropriate
/// action message.
#[must_use]
pub fn diff_stats_row<'a, Message: 'a>(added: i64, removed: i64, size: f32) -> Row<'a, Message> {
    let mut parts: Vec<Element<'a, Message>> = Vec::new();
    if added > 0 {
        parts.push(
            text(format!("+{added}"))
                .size(size)
                .color(theme::STATUS_SUCCESS)
                .into(),
        );
    }
    if added > 0 && removed > 0 {
        parts.push(text("/").size(size).color(theme::TEXT_MUTED).into());
    }
    if removed > 0 {
        parts.push(
            text(format!("\u{2212}{removed}"))
                .size(size)
                .color(theme::STATUS_ERROR)
                .into(),
        );
    }
    Row::with_children(parts)
        .spacing(0)
        .align_y(Alignment::Center)
}

/// Build the git footer diff-stats content: `+X / -Y` (non-zero sides) plus
/// an optional "N files" indicator for oversized/binary untracked files.
///
/// Returns `None` when there is nothing to show (all stats zero). The
/// file-count is shown even when both line counts are zero so oversized/binary
/// untracked files are never silently dropped from the footer.
#[must_use]
pub fn git_footer_stats<'a, Message: 'a>(
    added: i64,
    removed: i64,
    huge_binary_file_count: usize,
    size: f32,
) -> Option<Row<'a, Message>> {
    if added == 0 && removed == 0 && huge_binary_file_count == 0 {
        return None;
    }
    let mut parts: Vec<Element<'a, Message>> = Vec::new();
    if added != 0 || removed != 0 {
        parts.push(diff_stats_row::<Message>(added, removed, size).into());
    }
    if huge_binary_file_count > 0 {
        parts.push(
            lucide::files::<iced::Theme, iced::Renderer>()
                .size(size)
                .color(theme::TEXT_MUTED)
                .into(),
        );
        parts.push(
            text(format!("{huge_binary_file_count} files"))
                .size(size)
                .color(theme::TEXT_MUTED)
                .into(),
        );
    }
    Some(
        Row::with_children(parts)
            .spacing(3)
            .align_y(Alignment::Center),
    )
}

// ── Debounce helpers ───────────────────────────────────────────────

/// Spawn a sleep task that returns `generation` after `ms` milliseconds.
///
/// Used by `DebounceState` to implement
/// debounced refresh: increment a generation counter, spawn this task
/// with the new generation, and check the returned generation against
/// the current counter in the response handler.
pub async fn debounce_sleep(ms: u64, generation: u64) -> u64 {
    tokio::time::sleep(Duration::from_millis(ms)).await;
    generation
}

// ── File tree ───────────────────────────────────────────────────────

/// A node in a shared file-tree sidebar.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Display name (directory or file name component).
    pub name: String,
    /// Full relative path from workspace/repo root.
    pub full_path: String,
    /// Whether this is a directory node.
    pub is_dir: bool,
    /// Children (only populated for expanded directory nodes).
    pub children: Vec<TreeNode>,
    /// Error message if this entry couldn't be inspected (broken symlink, etc.).
    pub error: Option<String>,
    /// Whether gitignore machinery marked this node ignored — rendered dimmed
    /// (never hidden). Only the editor sets it true.
    pub ignored: bool,
}

/// Direction for navigating the file tree (arrow-key vertical movement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeNavDirection {
    Up,
    Down,
}

/// Shared file-tree state used by both the editor and diff dashboard pages.
pub struct FileTree {
    /// The hierarchical tree nodes.
    pub nodes: Vec<TreeNode>,
    /// Which directories are expanded (by `full_path`).
    pub expanded_dirs: HashSet<String>,
    /// Whether keyboard focus is in the file tree.
    pub tree_focused: bool,
    /// Index into `visible_tree_nodes` of the focused entry.
    pub tree_focus_index: usize,
    /// Flattened visible tree entries: (full_path, is_dir).
    pub visible_tree_nodes: Vec<(String, bool)>,
    /// Scrollable ID for the tree panel (for scroll-into-view).
    pub tree_scroll_id: iced::widget::Id,
    /// Current vertical scroll offset of the tree panel viewport.
    /// Updated via `on_scroll` on the scrollable widget.
    pub scroll_y: f32,
    /// Visible height of the tree panel viewport.
    /// `None` until the first scroll event fires, at which point it becomes
    /// `Some(viewport_h)`. When `None`, [`scroll_to_tree_focus`] with
    /// [`ScrollMode::ScrollIntoView`] falls back to [`ScrollMode::SnapToTop`].
    pub viewport_h: Option<f32>,
}

impl FileTree {
    /// Create a new empty `FileTree` with the given scrollable ID.
    #[must_use]
    pub fn new(scroll_id: iced::widget::Id) -> Self {
        Self {
            nodes: Vec::new(),
            expanded_dirs: HashSet::new(),
            tree_focused: false,
            tree_focus_index: 0,
            visible_tree_nodes: Vec::new(),
            tree_scroll_id: scroll_id,
            scroll_y: 0.0,
            viewport_h: None,
        }
    }

    /// Rebuild the flattened list of visible tree nodes for keyboard navigation.
    pub fn rebuild_visible(&mut self) {
        self.visible_tree_nodes.clear();
        Self::flatten_tree_nodes(
            &self.nodes,
            &self.expanded_dirs,
            &mut self.visible_tree_nodes,
        );
        if self.visible_tree_nodes.is_empty() {
            self.tree_focus_index = 0;
        } else {
            self.tree_focus_index = self.tree_focus_index.min(self.visible_tree_nodes.len() - 1);
        }
    }

    /// Move focus up one visible node. Returns `true` if focus moved.
    ///
    /// No-op when the tree is not focused, or when already at the top.
    #[must_use]
    pub fn nav_up(&mut self) -> bool {
        if self.tree_focused && self.tree_focus_index > 0 {
            self.tree_focus_index -= 1;
            true
        } else {
            false
        }
    }

    /// Move focus down one visible node. Returns `true` if focus moved.
    ///
    /// No-op when the tree is not focused, or when already at the bottom.
    #[must_use]
    pub fn nav_down(&mut self) -> bool {
        if self.tree_focused && self.tree_focus_index + 1 < self.visible_tree_nodes.len() {
            self.tree_focus_index += 1;
            true
        } else {
            false
        }
    }

    /// Recursively flatten tree nodes, respecting expanded state.
    fn flatten_tree_nodes(
        nodes: &[TreeNode],
        expanded: &HashSet<String>,
        out: &mut Vec<(String, bool)>,
    ) {
        for node in nodes {
            out.push((node.full_path.clone(), node.is_dir));
            if node.is_dir && expanded.contains(&node.full_path) && !node.children.is_empty() {
                Self::flatten_tree_nodes(&node.children, expanded, out);
            }
        }
    }

    /// Sort tree nodes: directories first, then case-insensitive alphabetical.
    /// Applied recursively so subdirectory children are also sorted.
    pub fn sort_nodes(nodes: &mut [TreeNode]) {
        nodes.sort_by(|a, b| {
            if a.is_dir != b.is_dir {
                return b.is_dir.cmp(&a.is_dir);
            }
            a.name.to_lowercase().cmp(&b.name.to_lowercase())
        });
        for node in nodes {
            Self::sort_nodes(&mut node.children);
        }
    }

    /// Set the focus index to the visible-tree position of `path`, if found.
    ///
    /// Returns the found position, or [`None`] if `path` is not in the visible tree.
    /// The caller can use the returned position for additional logic (e.g. advancing
    /// focus past a directory to its first child).
    pub fn focus_path(&mut self, path: &str) -> Option<usize> {
        let pos = self
            .visible_tree_nodes
            .iter()
            .position(|(p, _)| p == path)?;
        self.tree_focus_index = pos;
        Some(pos)
    }

    /// Expand a directory and move keyboard focus to its first child.
    ///
    /// Caller must have already inserted `path` into [`expanded_dirs`](Self::expanded_dirs)
    /// and updated [`nodes`](Self::nodes). This method rebuilds the visible tree, locates
    /// the directory in the new flattened list via [`Self::focus_path`], advances focus
    /// to the entry immediately after it (the first child), and returns a scroll-into-view
    /// task.
    ///
    /// Returns [`Task::none()`] if the directory is no longer in the visible tree or has
    /// no children — focus stays on the directory itself in that case.
    pub fn expand_dir_and_focus_first_child<Message: 'static>(
        &mut self,
        path: &str,
    ) -> Task<Message> {
        debug_assert!(
            self.expanded_dirs.contains(path),
            "expand_dir_and_focus_first_child: path must be in expanded_dirs before calling"
        );
        self.rebuild_visible();
        if let Some(dir_idx) = self.focus_path(path) {
            if dir_idx + 1 < self.visible_tree_nodes.len() {
                self.tree_focus_index = dir_idx + 1;
                return scroll_to_tree_focus(self, ScrollMode::SnapToTop);
            }
        }
        Task::none()
    }

    /// Collapse an expanded directory and keep keyboard focus on it.
    ///
    /// Caller must have already removed `path` from [`expanded_dirs`](Self::expanded_dirs)
    /// and updated [`nodes`](Self::nodes). This method rebuilds the visible tree,
    /// re-focuses the now-collapsed directory via [`Self::focus_path`], and returns a
    /// scroll-into-view task.
    ///
    /// Returns [`Task::none()`] if the directory is no longer in the visible tree —
    /// focus is left at whatever position it ended up at after rebuilding.
    pub fn collapse_dir_and_keep_focus<Message: 'static>(&mut self, path: &str) -> Task<Message> {
        debug_assert!(
            !self.expanded_dirs.contains(path),
            "collapse_dir_and_keep_focus: path must have been removed from expanded_dirs \
             before calling"
        );
        self.rebuild_visible();
        if self.focus_path(path).is_some() {
            return scroll_to_tree_focus(self, ScrollMode::SnapToTop);
        }
        Task::none()
    }

    /// Move focus to the parent of the focused node (ArrowLeft on a collapsed
    /// directory or file). Returns a snap-to-top scroll task, or [`Task::none()`]
    /// when the focused item has no parent in the visible tree.
    pub fn focus_parent<Message: 'static>(&mut self) -> Task<Message> {
        match self.focused_parent_path() {
            Some(ref p) if self.focus_path(p).is_some() => {
                scroll_to_tree_focus(self, ScrollMode::SnapToTop)
            }
            // Root-level item has no parent — no-op.
            _ => Task::none(),
        }
    }

    /// Move focus to the row after `idx` (the first child of an expanded
    /// directory), if it exists. Returns a snap-to-top scroll task.
    ///
    /// `idx` is the already-clamped focused index from [`Self::focused_tree_node`];
    /// keeping it a parameter preserves the caller's bounds-check view even if a
    /// tree rebuild re-clamped `tree_focus_index`.
    pub fn focus_next_row<Message: 'static>(&mut self, idx: usize) -> Task<Message> {
        if idx + 1 < self.visible_tree_nodes.len() {
            self.tree_focus_index = idx + 1;
            scroll_to_tree_focus(self, ScrollMode::SnapToTop)
        } else {
            Task::none()
        }
    }

    /// Move focus one visible node in `direction` and scroll it into view.
    ///
    /// Returns [`Task::none()`] when the tree is not focused or focus is already
    /// at the boundary.
    pub fn nav_and_scroll<Message: 'static>(
        &mut self,
        direction: TreeNavDirection,
    ) -> Task<Message> {
        let moved = match direction {
            TreeNavDirection::Up => self.nav_up(),
            TreeNavDirection::Down => self.nav_down(),
        };
        if moved {
            scroll_to_tree_focus(self, ScrollMode::ScrollIntoView)
        } else {
            Task::none()
        }
    }

    /// Return the focused visible tree node, if the tree has focus and is non-empty.
    ///
    /// Returns `None` when the tree is not focused or there are no visible nodes.
    /// Otherwise returns `(clamped_index, path, is_dir)` where `clamped_index` is
    /// `tree_focus_index` clamped to `visible_tree_nodes.len() - 1`. The clamped
    /// index is returned (rather than the raw `tree_focus_index`) so callers can
    /// safely use it for subsequent adjacency checks (e.g. `idx + 1` bounds check
    /// in `TreeNavRight`).
    #[must_use]
    pub fn focused_tree_node(&self) -> Option<(usize, String, bool)> {
        if !self.tree_focused || self.visible_tree_nodes.is_empty() {
            return None;
        }
        let idx = self.tree_focus_index.min(self.visible_tree_nodes.len() - 1);
        let path = self.visible_tree_nodes[idx].0.clone();
        let is_dir = self.visible_tree_nodes[idx].1;
        Some((idx, path, is_dir))
    }

    /// Returns `true` when the focused node is a directory and is currently expanded.
    ///
    /// This is a read-only inspection helper that centralises the common
    /// `is_dir && expanded_dirs.contains(path)` check that appears in tree-navigation
    /// keyboard handlers.  Returns `false` when the tree is not focused, empty, or
    /// the focused node is a file or a collapsed directory.
    #[must_use]
    pub fn focused_is_expanded_dir(&self) -> bool {
        self.focused_tree_node()
            .is_some_and(|(_, ref path, is_dir)| is_dir && self.expanded_dirs.contains(path))
    }

    /// Returns the parent path of the focused node, or [`None`] for root-level items.
    ///
    /// Computes the parent by calling [`std::path::Path::parent`] on the focused
    /// node's full path.  Returns [`None`] when the tree is not focused, empty, or
    /// the focused node is already at the root (no parent).
    ///
    /// This is a read-only helper that replaces the repeated
    /// `Path::new(&path).parent().map(|p| p.to_string_lossy().to_string())`
    /// pattern in tree-navigation keyboard handlers.
    #[must_use]
    pub fn focused_parent_path(&self) -> Option<String> {
        let (_idx, path, _is_dir) = self.focused_tree_node()?;
        let parent = Path::new(&path).parent()?;
        let parent_str = parent.to_string_lossy().to_string();
        if parent_str.is_empty() {
            None
        } else {
            Some(parent_str)
        }
    }
}

/// Font size for file tree item labels and connector guides.
pub const TREE_FONT_SIZE: f32 = 14.0;

/// Icon size for directory nodes in the file tree (slightly larger than
/// [`TREE_FONT_SIZE`] to compensate for lucide icons appearing smaller
/// at the same nominal point size).
pub const TREE_ICON_SIZE: f32 = 15.0;

/// Minimum width of the auto-sizing file-tree panel — the tree never gets
/// narrower than this (matches the previous fixed 260px width).
pub const TREE_MIN_WIDTH: f32 = 260.0;

/// Maximum width cap of the auto-sizing file-tree panel. The cap guarantees
/// sibling panels (editor content, diff content) keep a sane minimum width
/// at the smallest supported window size.
pub const TREE_MAX_WIDTH: f32 = 400.0;

/// Right-side room reserved in the auto-sized panel width so the 6px overlay
/// scrollbar (see [`super::theme::thin_scrollbar`]) never covers the widest
/// visible row's content: 6px scrollbar + 4px breathing room.
pub const TREE_SCROLLBAR_ALLOWANCE: f32 = 10.0;

/// Horizontal padding of every tree row (`.padding([0, 8])` in the editor
/// and diff row renderers). The panel width computation adds both sides.
pub const TREE_ROW_H_PADDING: f32 = 8.0;

/// Glyph advance of JetBrains Mono as a fraction of em. Every glyph the tree
/// rows render — ASCII, box-drawing (`│ ├ └`), `⚠`, `…`, digits — measures
/// exactly 0.6em in both the Regular and Bold faces (verified from the TTFs
/// in `src/gui/`), so `chars × size × 0.6` is an exact width, not an
/// estimate. The dashboard default font is JetBrains Mono
/// (see [`super::JETBRAINS_MONO`]), so `text()` widgets without an explicit
/// font (diff ± counts, `[⚠]` suffixes) use it too.
pub const JETBRAINS_MONO_ADVANCE: f32 = 0.6;

/// Glyph advance of the lucide icon font as a fraction of em (verified from
/// the lucide TTF). Every icon in the tree measures exactly 1.0em, so an
/// icon rendered via `.size(s)` is `s` px wide.
pub const LUCIDE_ADVANCE: f32 = 1.0;

/// Width in px of `chars` glyphs of JetBrains Mono at `size` px.
///
/// Exact for every glyph the tree renders — see [`JETBRAINS_MONO_ADVANCE`].
#[must_use]
#[expect(clippy::cast_precision_loss)] // usize glyph count → px width
pub fn mono_text_width(chars: usize, size: f32) -> f32 {
    chars as f32 * size * JETBRAINS_MONO_ADVANCE
}

/// Natural content width of a rendered tree row, excluding the row's
/// horizontal padding (added by [`tree_panel_width`]).
///
/// Replicates the exact row composition in the editor and diff renderers:
/// `guide_chars` box-drawing guide glyphs (14px) + a lucide icon at
/// `icon_size` px + a 4px gap + the name label at `name_size` px, an optional
/// `name_suffix` segment preceded by another 4px gap (editor error file rows
/// render `[⚠]` at 11px), and — for diff file rows — the ± change counts at
/// 10px followed by a 6px trailing gap.
///
/// `counts` is `Some((add, remove))` for diff file rows (either string may be
/// empty; the 6px trailing gap is part of the row either way) and `None` for
/// every other row type. When both strings are non-empty they render as
/// `"{add} {remove}"` — one separating space at 10px.
///
/// JetBrains Mono is monospace with a single advance for all weights, so the
/// bold name of selected rows and the regular name of unselected rows measure
/// identically.
#[must_use]
pub fn tree_row_natural_width(
    guide_chars: usize,
    icon_size: f32,
    name: &str,
    name_size: f32,
    name_suffix: Option<(&str, f32)>,
    counts: Option<(&str, &str)>,
) -> f32 {
    let mut w = mono_text_width(guide_chars, TREE_FONT_SIZE)
        + icon_size * LUCIDE_ADVANCE
        + 4.0
        + mono_text_width(name.chars().count(), name_size);
    if let Some((suffix, size)) = name_suffix {
        w += 4.0 + mono_text_width(suffix.chars().count(), size);
    }
    if let Some((add, rem)) = counts {
        let counts_chars = if add.is_empty() {
            rem.chars().count()
        } else if rem.is_empty() {
            add.chars().count()
        } else {
            add.chars().count() + 1 + rem.chars().count()
        };
        w += mono_text_width(counts_chars, 10.0) + 6.0;
    }
    w
}

/// Collect the natural content width of every rendered tree row, in render
/// order (one entry per row — the same DFS order as the recursive node
/// renderers, which may nest multiple rows per root element).
///
/// Mirrors the render walk: a directory row is followed by its children's
/// rows when the directory is expanded. `row_width` computes one row's
/// width from the node and its nesting `depth` (0 = root).
pub fn collect_tree_row_widths(
    nodes: &[TreeNode],
    expanded: &HashSet<String>,
    row_width: impl Fn(&TreeNode, usize) -> f32,
) -> Vec<f32> {
    fn walk(
        nodes: &[TreeNode],
        expanded: &HashSet<String>,
        row_width: &impl Fn(&TreeNode, usize) -> f32,
        depth: usize,
        out: &mut Vec<f32>,
    ) {
        for node in nodes {
            out.push(row_width(node, depth));
            if node.is_dir && expanded.contains(&node.full_path) {
                walk(&node.children, expanded, row_width, depth + 1, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(nodes, expanded, &row_width, 0, &mut out);
    out
}

/// Compute the auto-sized width of the tree panel from the natural content
/// width of every rendered row (in render order — see
/// [`collect_tree_row_widths`]).
///
/// Only rows currently visible in the viewport are measured. The visible
/// range is derived from [`FileTree::scroll_y`] / [`FileTree::viewport_h`]
/// with [`ESTIMATED_TREE_ROW_HEIGHT`] row spacing, extended by one row on
/// each side to absorb height-estimate drift (over-measuring is safe; the
/// result is clamped either way).
///
/// Before the first scroll event — and for trees whose content fits without
/// scrolling, where Iced never fires `on_scroll` (see
/// [`FileTree::viewport_h`]) — `viewport_h` is `None` and all rows are
/// measured: the documented fallback.
///
/// The result is the widest measured row's natural content width plus the
/// row's horizontal padding on both sides and
/// [`TREE_SCROLLBAR_ALLOWANCE`], clamped to
/// [`TREE_MIN_WIDTH`]..=[`TREE_MAX_WIDTH`]. A tree whose rows all fit at the
/// minimum width stays at [`TREE_MIN_WIDTH`].
#[must_use]
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // viewport px → row index
pub fn tree_panel_width(file_tree: &FileTree, row_widths: &[f32]) -> f32 {
    let widest = match file_tree.viewport_h {
        Some(viewport_h) if viewport_h > 0.0 => {
            let row_h = ESTIMATED_TREE_ROW_HEIGHT;
            let first = (file_tree.scroll_y / row_h).floor().max(0.0) as usize;
            let last = ((file_tree.scroll_y + viewport_h) / row_h).ceil() as usize + 1;
            row_widths
                .iter()
                .enumerate()
                .skip(first.saturating_sub(1))
                .take(last - first + 2)
                .map(|(_, w)| *w)
                .fold(0.0f32, f32::max)
        }
        // Viewport unknown — measure everything (first frame / non-scrolling
        // tree, where every row is visible anyway).
        _ => row_widths.iter().copied().fold(0.0f32, f32::max),
    };
    (widest + 2.0 * TREE_ROW_H_PADDING + TREE_SCROLLBAR_ALLOWANCE)
        .clamp(TREE_MIN_WIDTH, TREE_MAX_WIDTH)
}

/// Controls whether [`scroll_to_tree_focus`] snaps to the focused row or
/// uses viewport-aware scroll-into-view logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollMode {
    /// Scroll so that the focused row is at the top of the viewport.
    SnapToTop,
    /// Only scroll when the focused row is outside the visible viewport.
    /// Requires [`FileTree::viewport_h`] to be `Some`; falls back to
    /// [`SnapToTop`](ScrollMode::SnapToTop) when unknown.
    ScrollIntoView,
}

/// Estimated height per tree row for scroll-into-view on keyboard navigation.
/// Derived from [`TREE_FONT_SIZE`] × Iced's default relative line height (1.3)
/// for a close approximation of actual rendered row height. File entries are
/// ~18.2 px; directory entries (with [`TREE_ICON_SIZE`] 15 pt icons) are
/// ~19.5 px.
///
/// This constant is used directly by [`scroll_to_tree_focus`] to compute
/// row positions for keyboard-navigation scroll-into-view logic.
pub const ESTIMATED_TREE_ROW_HEIGHT: f32 = TREE_FONT_SIZE * 1.3;

/// Scroll the tree panel to bring the focused row into view.
///
/// Behaviour depends on [`ScrollMode`]:
///
/// * [`SnapToTop`](ScrollMode::SnapToTop): absolute offset to
///   `tree_focus_index * ESTIMATED_TREE_ROW_HEIGHT`.
/// * [`ScrollIntoView`](ScrollMode::ScrollIntoView): only scrolls when the
///   focused row is not fully visible — for rows above the viewport the
///   row is brought to the top, for rows below the viewport the view
///   advances by one row height. Falls back to [`ScrollMode::SnapToTop`] when the
///   viewport height is unknown ([`FileTree::viewport_h`] is `None`).
///
/// Row height is approximated by [`ESTIMATED_TREE_ROW_HEIGHT`], derived
/// from [`TREE_FONT_SIZE`] × Iced's default relative line height (1.3).
///
/// This method updates [`FileTree::scroll_y`] directly so that consecutive
/// calls during the same frame see an accurate scroll offset even before the
/// `on_scroll` callback fires.
#[expect(clippy::cast_precision_loss)]
pub fn scroll_to_tree_focus<Message: 'static>(
    file_tree: &mut FileTree,
    mode: ScrollMode,
) -> Task<Message> {
    if file_tree.visible_tree_nodes.is_empty() {
        return Task::none();
    }

    let focus_y = file_tree.tree_focus_index as f32 * ESTIMATED_TREE_ROW_HEIGHT;

    match mode {
        ScrollMode::SnapToTop => absolute_scroll_to(file_tree, focus_y),
        ScrollMode::ScrollIntoView => match file_tree.viewport_h {
            None => {
                // Viewport size unknown — fall back to snap-to-top.
                absolute_scroll_to(file_tree, focus_y)
            }
            Some(viewport_h) => {
                // A row is considered "above viewport" when the bottom edge
                // of the row is above the viewport top. This avoids redundant
                // scrolling when a row is partially visible at the top edge
                // after non-row-aligned mouse-wheel scrolling.
                let row_bottom = focus_y + ESTIMATED_TREE_ROW_HEIGHT;
                let viewport_bottom = file_tree.scroll_y + viewport_h;

                if row_bottom <= file_tree.scroll_y {
                    // Focus is above the visible area — bring it to the top.
                    absolute_scroll_to(file_tree, focus_y)
                } else if focus_y >= viewport_bottom {
                    // Focus is below the visible area — advance by one row
                    // and update scroll_y directly so the next key event
                    // sees accurate state even before on_scroll fires.
                    file_tree.scroll_y = (file_tree.scroll_y + ESTIMATED_TREE_ROW_HEIGHT).max(0.0);
                    iced::widget::operation::scroll_by(
                        file_tree.tree_scroll_id.clone(),
                        iced::widget::operation::AbsoluteOffset {
                            x: 0.0,
                            y: ESTIMATED_TREE_ROW_HEIGHT,
                        },
                    )
                } else {
                    // Row is within the viewport (fully or partially visible).
                    // Partially-visible rows at the bottom edge
                    // (focus_y < viewport_bottom but row_bottom > viewport_bottom)
                    // are intentionally not scrolled — only rows whose top edge
                    // is entirely outside the viewport trigger a scroll.
                    Task::none()
                }
            }
        },
    }
}

/// Helper: absolute scroll to `y` offset and update [`FileTree::scroll_y`].
fn absolute_scroll_to<Message: 'static>(file_tree: &mut FileTree, y: f32) -> Task<Message> {
    // Best-guess update of the tracked scroll offset so that subsequent
    // ScrollIntoView checks within the same frame use a plausible value.
    file_tree.scroll_y = y.max(0.0);
    iced::widget::operation::scroll_to(
        file_tree.tree_scroll_id.clone(),
        iced::widget::operation::AbsoluteOffset { x: 0.0, y },
    )
}

/// Build a file-tree panel widget.
///
/// Renders a scrollable, auto-width column wrapping the pre-built
/// `tree_rows` elements. The panel width adapts to the widest currently
/// visible row (see [`tree_panel_width`]); a focus border is applied when
/// `file_tree.tree_focused` is true.
///
/// `row_widths` holds the natural content width of every rendered row in
/// render order — one entry per row, in the same DFS order as the recursive
/// node renderers (which may nest multiple rows per root element). See
/// [`collect_tree_row_widths`].
///
/// `on_scroll` is attached to the inner [`widget::scrollable()`] via
/// `on_scroll` and fires whenever the viewport changes
/// (scrollbar drag, mouse wheel, programmatic scroll). The caller should
/// produce a message that updates [`FileTree::scroll_y`] and
/// [`FileTree::viewport_h`] from the [`iced::widget::scrollable::Viewport`] data.
pub fn build_tree_panel<'a, Message: 'a>(
    file_tree: &'a FileTree,
    tree_rows: Vec<Element<'a, Message>>,
    row_widths: &[f32],
    on_scroll: impl Fn(scrollable::Viewport) -> Message + 'a,
) -> Element<'a, Message> {
    let panel_width = tree_panel_width(file_tree, row_widths);

    let tree_body = widget::scrollable(column(tree_rows).spacing(0))
        .id(file_tree.tree_scroll_id.clone())
        .on_scroll(on_scroll)
        .width(Length::Fill)
        .height(Length::Fill)
        .direction(theme::vertical_scrollbar())
        .style(theme::scrollbar_style);

    let tree_inner: Element<'_, Message> = container(tree_body)
        .width(Length::Fixed(panel_width))
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .into();

    if file_tree.tree_focused {
        container(tree_inner)
            .style(|_t: &iced::Theme| container::Style {
                border: iced::Border {
                    color: theme::ACCENT_LIGHT,
                    width: 2.0,
                    radius: 0.0.into(),
                },
                ..Default::default()
            })
            .into()
    } else {
        tree_inner
    }
}

// ── Tree node helpers ──────────────────────────────────────────────

/// Build the guide-line prefix string for a tree node.
///
/// Returns box-drawing characters that visually connect tree siblings:
///
/// | Character | Meaning |
/// |---|---|
/// | `│` | Vertical continuation — the ancestor at this depth has more siblings below |
/// | `├` | T-junction — this node has at least one more sibling after it |
/// | `└` | Corner — this node is the last child of its parent |
/// | ` `  | No continuation at this ancestor level |
///
/// Each depth level uses exactly two characters (guide char + one space), so
/// the total visual width per level closely matches the existing 14 px indent.
///
/// `ancestor_mask` has bit `d` set iff the ancestor at depth `d` has more
/// siblings after it (requiring a vertical continuation line at that column).
/// `depth` is the current nesting depth (0 = root, which gets no prefix).
/// `is_last` is true when this node is the last child of its parent.
///
/// # Panics
///
/// Panics in debug builds when `depth >= 64` (the u64 bitmask would overflow).
#[must_use]
pub fn tree_guide_prefix(ancestor_mask: u64, depth: usize, is_last: bool) -> String {
    debug_assert!(
        depth < 64,
        "tree_guide_prefix: depth {depth} exceeds u64 bit limit (max 63)"
    );
    let mut s = String::new();
    for d in 0..depth.saturating_sub(1) {
        if ancestor_mask & (1u64 << d) != 0 {
            s.push('│');
        } else {
            s.push(' ');
        }
        s.push(' ');
    }
    if depth > 0 {
        if is_last {
            s.push('└');
        } else {
            s.push('├');
        }
        s.push(' ');
    }
    s
}

/// Recursively render children of a tree node, computing the correct
/// continuation mask and `is_last` state for each child.
///
/// `render_node` is called for each child with `(child, depth+1, child_mask, child_is_last)`.
/// The returned elements share the lifetime `'a` of the tree nodes.
/// Returns a `Vec` of child elements, one per child, in order.
///
/// This exists to avoid duplicating the child-iteration + mask-computation
/// logic across the two render paths (editor and diff file trees).
pub fn render_tree_children<'a, Message>(
    children: &'a [TreeNode],
    depth: usize,
    ancestor_mask: u64,
    is_last: bool,
    render_node: impl Fn(&'a TreeNode, usize, u64, bool) -> Element<'a, Message>,
) -> Vec<Element<'a, Message>> {
    let child_count = children.len();
    let cont_bit = if !is_last { 1u64 << depth } else { 0u64 };
    let child_mask = ancestor_mask | cont_bit;
    children
        .iter()
        .enumerate()
        .map(|(i, child)| {
            let child_is_last = i == child_count - 1;
            render_node(child, depth + 1, child_mask, child_is_last)
        })
        .collect()
}

/// Dispatch a file-tree node to its dir or file renderer.
///
/// Shared by the editor and diff file trees so the `is_dir` branching lives
/// in one place; exactly one of the two closures is invoked.
pub fn render_tree_node<'a, Message>(
    is_dir: bool,
    render_dir: impl FnOnce() -> Element<'a, Message>,
    render_file: impl FnOnce() -> Element<'a, Message>,
) -> Element<'a, Message> {
    if is_dir { render_dir() } else { render_file() }
}

/// Check whether a tree node at the given path is currently focused
/// in the file tree's keyboard navigation.
#[must_use]
pub fn tree_node_focused(tree: &FileTree, node_path: &str) -> bool {
    tree.tree_focused
        && tree.tree_focus_index < tree.visible_tree_nodes.len()
        && tree.visible_tree_nodes[tree.tree_focus_index].0 == node_path
}

/// Return a button style closure for tree node entries.
/// When `is_highlighted` is true, uses [`theme::HOVER_STRONG`]; otherwise
/// hover gets [`theme::HOVER`], and default is transparent.
fn tree_node_button_style(
    is_highlighted: bool,
) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    move |_t: &iced::Theme, status| {
        let bg = if is_highlighted {
            theme::HOVER_STRONG
        } else if status == button::Status::Hovered {
            theme::HOVER
        } else {
            iced::Color::TRANSPARENT
        };
        button::Style {
            background: Some(iced::Background::Color(bg)),
            ..Default::default()
        }
    }
}

/// Build a tree-node button from a content row, highlight state, and
/// optional press message. Uses `tree_node_button_style` internally
/// and spans full width.
///
/// This returns only the button element — callers that need context menus
/// (e.g., the editor page) must wrap the result themselves.
pub fn tree_node_button<'a, Message: Clone + 'a>(
    content: impl Into<Element<'a, Message>>,
    is_highlighted: bool,
    on_press: Option<Message>,
) -> Element<'a, Message> {
    let mut btn = widget::button(content)
        .style(tree_node_button_style(is_highlighted))
        .width(Length::Fill)
        .padding(Padding::ZERO);
    if let Some(msg) = on_press {
        btn = btn.on_press(msg);
    }
    btn.into()
}

/// Build the semi-transparent click-to-dismiss backdrop layer shared by all
/// modal overlays. The backdrop is a full-screen [`mouse_area`] that emits
/// `on_backdrop` when clicked.
///
/// Extracted from [`modal_backdrop`] so the 80%-width diff overlay can
/// compose the same backdrop while claiming only its dialog rectangle.
pub(super) fn modal_backdrop_layer<'a, Message: 'a + Clone>(
    on_backdrop: Message,
    opacity: f32,
) -> Element<'a, Message> {
    mouse_area(
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    0.0, 0.0, 0.0, opacity,
                ))),
                ..container::Style::default()
            }),
    )
    .on_press(on_backdrop)
    .into()
}

/// Claim clicks over a dialog box so they do not fall through to the modal
/// backdrop, while leaving the surrounding margin clickable.
///
/// The [`mouse_area`] reports [`iced::mouse::Interaction::Idle`] when the cursor
/// is over the dialog and the content is non-interactive. That non-`None`
/// interaction levitates the cursor in the parent `Stack`, so the backdrop sees
/// a levitating cursor and bails instead of closing the modal. Interactive inner
/// elements still capture their own events first, and the surrounding margin
/// still reaches the backdrop.
pub(super) fn dialog_click_guard<'a, Message: 'a + Clone>(
    content: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    mouse_area(content)
        .interaction(iced::mouse::Interaction::Idle)
        .into()
}

/// Wrap dialog content in a centered modal overlay with a semi-transparent
/// backdrop that closes on click.
///
/// This is the shared helper for all modal backdrop patterns across the
/// dashboard. It creates a stack with a click-to-dismiss backdrop and a
/// centered container for the dialog content, wrapped in [`dialog_click_guard`]
/// so clicks on non-interactive interior areas do not fall through to the
/// backdrop and close the modal.
///
/// This helper does **not** apply `dialog_container_style` or padding —
/// callers are responsible for styling their content as needed before
/// passing it in.
///
/// # Content contract
/// `content` must be a bounded dialog bubble (not `Length::Fill` on the outer
/// dimension). [`dialog_click_guard`] claims the content's whole rectangle, so
/// full-window content would claim the screen and make the backdrop
/// unreachable — silently breaking outside-click. The 80%-width diff
/// overlay composes the backdrop directly via [`modal_backdrop_layer`] for
/// exactly this reason.
///
/// # Parameters
/// - `content`: The dialog body to overlay. Should already be styled
///   (container, padding, etc.) by the caller as needed.
/// - `on_backdrop`: Message to emit when the backdrop is clicked.
/// - `opacity`: Opacity of the backdrop (e.g., `0.5` for standard
///   semi-transparent black, `0.4` for lighter).
pub fn modal_backdrop<'a, Message: 'a + Clone>(
    content: impl Into<Element<'a, Message>>,
    on_backdrop: Message,
    opacity: f32,
) -> Element<'a, Message> {
    let backdrop = modal_backdrop_layer(on_backdrop, opacity);

    let centered = container(dialog_click_guard(content))
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    stack([backdrop, centered.into()]).into()
}

/// Zero-size placeholder that keeps an overlay slot's widget type stable
/// across show/hide transitions. Iced destroys widget state (scroll
/// positions, open popovers) when the widget tree tag changes between
/// frames, so the closed state must return the identical bare Container.
/// Callers whose open state is a Stack (board.rs, settings.rs modal
/// overlays) must wrap this in `iced::widget::stack([...])` themselves.
/// Known pre-existing quirks, not fixed here: git.rs's closed-branch is
/// unreachable (`view()` is gated on the modal being open), and the mod.rs
/// diff/branch overlay slots use a bare-Container placeholder against
/// Stack open states (type mismatch predates this helper).
pub fn empty_stack_placeholder<'a, Message: 'a>() -> Element<'a, Message> {
    container(text(""))
        .width(Length::Shrink)
        .height(Length::Shrink)
        .into()
}

/// Wrap `content` so a completed click (press+release without a selection
/// drag) inside it publishes `on_click`, while selection gestures publish
/// `on_cancel`: the second click of a multi-click sequence (a word/line
/// selection), or a drag starting inside the content. `on_cancel` lets a
/// consumer that defers its toggle past [`DOUBLE_CLICK_WINDOW`] drop it, so
/// selection never fights a pending toggle; a third+ click publishes
/// nothing. A click the child answers with a message of its own — e.g. a
/// markdown link click — publishes `on_cancel` only when a click sequence
/// was already in progress.
#[must_use]
pub fn click_to_toggle<'a, Message: Clone + 'a>(
    content: impl Into<Element<'a, Message>>,
    on_click: Message,
    on_cancel: Message,
) -> Element<'a, Message> {
    ClickToggle {
        content: content.into(),
        on_click,
        on_cancel,
    }
    .into()
}

/// Outcome of a press+release pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClickKind {
    Click,
    Drag,
    MultiClick,
}

/// Max cursor travel (px) between press and release for the pair to count as
/// a click rather than a drag selection. Matches iced_selection's 6 px click
/// slop so both widgets classify a sloppy press the same way.
const CLICK_DRAG_MAX_DISTANCE: f32 = 6.0;
/// Max gap between consecutive clicks for the second+ click to be a
/// double/triple click (word/line selection, not a toggle).
const DOUBLE_CLICK_WINDOW: Duration = Duration::from_millis(300);

/// Classify a press+release pair by cursor travel and time since the last click.
///
/// The [`DOUBLE_CLICK_WINDOW`] boundary is inclusive: a gap of exactly 300 ms
/// still counts as a multi-click.
fn classify_click(distance: f32, since_last_click: Option<Duration>) -> ClickKind {
    if distance > CLICK_DRAG_MAX_DISTANCE {
        return ClickKind::Drag;
    }
    if since_last_click.is_some_and(|elapsed| elapsed <= DOUBLE_CLICK_WINDOW) {
        return ClickKind::MultiClick;
    }
    ClickKind::Click
}

/// Per-widget state persisted across frames via iced's `Tree`.
#[derive(Default, Clone)]
struct ClickToggleState {
    press: Option<Point>,
    last_click: Option<Instant>,
    /// Completed (non-drag) clicks ending at this widget, reset by a drag.
    click_count: u32,
    /// Whether the in-progress drag gesture already published a cancel.
    drag_canceled: bool,
}

struct ClickToggle<'a, Message, Theme = iced::Theme, Renderer = iced::Renderer> {
    content: Element<'a, Message, Theme, Renderer>,
    on_click: Message,
    on_cancel: Message,
}

impl<'a, Message, Theme, Renderer> From<ClickToggle<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a + Clone,
    Theme: 'a,
    Renderer: 'a + renderer::Renderer,
{
    fn from(widget: ClickToggle<'a, Message, Theme, Renderer>) -> Self {
        Self::new(widget)
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for ClickToggle<'_, Message, Theme, Renderer>
where
    Renderer: renderer::Renderer,
    Message: Clone,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<ClickToggleState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(ClickToggleState::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        // Run the child with its own message buffer so we can tell whether the
        // child produced a message of its own for this event (e.g. a markdown
        // link click publishing `LinkClicked` on release) — such a click
        // already carries meaning and must not ALSO toggle the block. Merging
        // back preserves the child's capture/redraw/layout requests.
        let mut child_messages = Vec::new();
        let mut child_shell = Shell::new(&mut child_messages);
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            &mut child_shell,
            viewport,
        );
        let child_published = !child_shell.is_empty();
        shell.merge(child_shell, std::convert::identity);

        self.handle_mouse(tree, event, layout, cursor, shell, child_published);
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let content_interaction = self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        );

        match (content_interaction, cursor.is_over(layout.bounds())) {
            (mouse::Interaction::None, true) => mouse::Interaction::Pointer,
            _ => content_interaction,
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        renderer_style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            renderer_style,
            layout,
            cursor,
            viewport,
        );
    }

    fn overlay<'a>(
        &'a mut self,
        tree: &'a mut Tree,
        layout: Layout<'a>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'a, Message, Theme, Renderer>> {
        self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<Message: Clone, Theme, Renderer> ClickToggle<'_, Message, Theme, Renderer> {
    fn handle_mouse(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        shell: &mut Shell<'_, Message>,
        child_published: bool,
    ) {
        let state: &mut ClickToggleState = tree.state.downcast_mut();

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if cursor.is_over(layout.bounds())
                    && let Some(position) = cursor.position()
                {
                    state.press = Some(position);
                    state.drag_canceled = false;
                }
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                // A drag starting inside the content is a selection gesture:
                // cancel a deferred toggle before it could commit
                // mid-selection.
                if let (Some(press_position), Some(position)) = (state.press, cursor.position())
                    && position.distance(press_position) > CLICK_DRAG_MAX_DISTANCE
                {
                    self.publish_cancel_once(state, shell);
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let Some(press_position) = state.press.take() else {
                    return;
                };
                let Some(release_position) = cursor.position() else {
                    return;
                };

                let kind = classify_click(
                    release_position.distance(press_position),
                    state.last_click.map(|t| t.elapsed()),
                );

                // A drag means selection, not clicking: reset the multi-click
                // sequence and cancel any deferred toggle.
                if kind == ClickKind::Drag {
                    state.last_click = None;
                    state.click_count = 0;
                    self.publish_cancel_once(state, shell);
                    return;
                }
                // A click the child answered itself (e.g. a markdown link) is
                // not part of our toggle sequence — but it still ends a
                // pending toggle from an earlier click.
                if child_published {
                    if state.click_count > 0 {
                        shell.publish(self.on_cancel.clone());
                    }
                    state.last_click = None;
                    state.click_count = 0;
                    return;
                }

                // `kind` is Click or MultiClick here (Drag returned above).
                state.click_count = if kind == ClickKind::Click {
                    1
                } else {
                    state.click_count + 1
                };
                state.last_click = Some(Instant::now());

                match state.click_count {
                    1 => shell.publish(self.on_click.clone()),
                    // Second click of a multi-click: a word/line selection.
                    2 => shell.publish(self.on_cancel.clone()),
                    // Triple+ click: word/line selection owns it.
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Publish `on_cancel` once per drag gesture (a deferred toggle must not
    /// commit mid-selection).
    fn publish_cancel_once(&self, state: &mut ClickToggleState, shell: &mut Shell<'_, Message>) {
        if !state.drag_canceled {
            state.drag_canceled = true;
            shell.publish(self.on_cancel.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_footer_stats_gating() {
        // Nothing to show when all stats are zero.
        assert!(git_footer_stats::<()>(0, 0, 0, 15.0).is_none());
        // A binary/huge untracked count alone still shows (not silently dropped).
        assert!(git_footer_stats::<()>(0, 0, 3, 15.0).is_some());
        // Non-zero line counts show regardless of the file-count.
        assert!(git_footer_stats::<()>(1, 0, 0, 15.0).is_some());
        assert!(git_footer_stats::<()>(0, 1, 2, 15.0).is_some());
    }

    /// Helper to create a FileTree with known visible_tree_nodes for testing.
    fn make_tree(nodes: Vec<(&str, bool)>) -> FileTree {
        let mut tree = FileTree::new(iced::widget::Id::new("test"));
        tree.visible_tree_nodes = nodes
            .into_iter()
            .map(|(p, is_dir)| (p.to_string(), is_dir))
            .collect();
        tree
    }

    #[test]
    #[expect(clippy::type_complexity)]
    fn focus_path_cases() {
        // (name, nodes, path, expected return, expected focus index)
        #[rustfmt::skip]
        let cases: &[(&str, &[(&str, bool)], &str, Option<usize>, usize)] = &[
            ("found", &[("src", true), ("src/main.rs", false), ("Cargo.toml", false)], "src/main.rs", Some(1), 1),
            ("empty_tree", &[], "anything", None, 0),
            ("first_node", &[("src", true), ("src/main.rs", false)], "src", Some(0), 0),
        ];
        for &(name, nodes, path, expected, expected_index) in cases {
            let mut tree = make_tree(nodes.to_vec());
            assert_eq!(tree.focus_path(path), expected, "case: {name}");
            assert_eq!(
                tree.tree_focus_index, expected_index,
                "case: {name} (index)"
            );
        }
    }

    #[test]
    fn focus_path_not_found() {
        let mut tree = make_tree(vec![("src", true), ("Cargo.toml", false)]);
        tree.tree_focus_index = 42;
        assert_eq!(tree.focus_path("nonexistent"), None);
        assert_eq!(tree.tree_focus_index, 42);
    }

    #[test]
    fn focus_path_updates_index_no_residual() {
        let mut tree = make_tree(vec![("a", false), ("b", false), ("c", false)]);
        // Focus on "c", then re-focus on "a" — should end up at index 0.
        tree.focus_path("c");
        assert_eq!(tree.tree_focus_index, 2);
        tree.focus_path("a");
        assert_eq!(tree.tree_focus_index, 0);
    }

    #[test]
    #[expect(clippy::type_complexity)]
    fn focused_tree_node_cases() {
        // (name, nodes, focused, focus index, expected node)
        #[rustfmt::skip]
        let cases: &[(&str, &[(&str, bool)], bool, usize, Option<(usize, &str, bool)>)] = &[
            // Tree is not focused (default).
            ("not_focused", &[("src", true), ("src/main.rs", false)], false, 0, None),
            ("empty_visible_nodes", &[], true, 0, None),
            // Set index beyond bounds — method should clamp.
            ("clamps_index", &[("a", false), ("b", false)], true, 10, Some((1, "b", false))),
            ("returns_correct_node", &[("src", true), ("src/main.rs", false), ("Cargo.toml", false)], true, 1, Some((1, "src/main.rs", false))),
            ("returns_directory", &[("src", true), ("src/main.rs", false)], true, 0, Some((0, "src", true))),
        ];
        for &(name, nodes, focused, focus_index, expected) in cases {
            let mut tree = make_tree(nodes.to_vec());
            tree.tree_focused = focused;
            tree.tree_focus_index = focus_index;
            assert_eq!(
                tree.focused_tree_node(),
                expected.map(|(i, p, d)| (i, p.to_string(), d)),
                "case: {name}"
            );
        }
    }

    // ── nav_up / nav_down tests ──────────────────────────────────────

    #[test]
    fn nav_cases() {
        // (name, direction, focused, start index, expected moved, expected index)
        let cases: &[(&str, &str, bool, usize, bool, usize)] = &[
            ("up_at_top_clamped", "up", true, 0, false, 0),
            ("down_at_bottom_clamped", "down", true, 1, false, 1),
            ("up_moves_focus", "up", true, 1, true, 0),
            ("down_moves_focus", "down", true, 0, true, 1),
            ("ignored_when_not_focused_up", "up", false, 0, false, 0),
            ("ignored_when_not_focused_down", "down", false, 0, false, 0),
        ];
        for &(name, dir, focused, start, expected_moved, expected_index) in cases {
            let mut tree = make_tree(vec![("a", false), ("b", false)]);
            tree.tree_focused = focused;
            tree.tree_focus_index = start;
            let moved = if dir == "up" {
                tree.nav_up()
            } else {
                tree.nav_down()
            };
            assert_eq!(moved, expected_moved, "case: {name}");
            assert_eq!(
                tree.tree_focus_index, expected_index,
                "case: {name} (index)"
            );
        }
    }

    // ── rebuild_visible clamping tests ────────────────────────────────

    #[test]
    fn rebuild_visible_clamps_high_focus_index() {
        let mut tree = FileTree::new(iced::widget::Id::new("test"));
        tree.nodes = vec![
            TreeNode {
                name: "a".into(),
                full_path: "a".into(),
                is_dir: false,
                children: vec![],
                error: None,
                ignored: false,
            },
            TreeNode {
                name: "b".into(),
                full_path: "b".into(),
                is_dir: false,
                children: vec![],
                error: None,
                ignored: false,
            },
            TreeNode {
                name: "c".into(),
                full_path: "c".into(),
                is_dir: false,
                children: vec![],
                error: None,
                ignored: false,
            },
        ];
        tree.rebuild_visible();
        assert_eq!(tree.visible_tree_nodes.len(), 3);
        tree.tree_focus_index = 999;
        tree.rebuild_visible();
        assert_eq!(tree.tree_focus_index, 2);
    }

    #[test]
    fn rebuild_visible_empty_tree_resets_focus_index() {
        let mut tree = make_tree(vec![("a", false)]);
        tree.tree_focus_index = 0;
        tree.nodes.clear();
        tree.expanded_dirs.clear();
        tree.rebuild_visible();
        assert!(tree.visible_tree_nodes.is_empty());
        assert_eq!(tree.tree_focus_index, 0);
    }

    // ── focused_is_expanded_dir tests ────────────────────────────────

    #[test]
    #[expect(clippy::type_complexity)]
    fn focused_is_expanded_dir_cases() {
        // (name, nodes, focused, expanded dir, expected)
        #[rustfmt::skip]
        let cases: &[(&str, &[(&str, bool)], bool, Option<&str>, bool)] = &[
            // Tree is not focused.
            ("not_focused", &[("src", true)], false, None, false),
            ("empty_tree", &[], true, None, false),
            ("file", &[("main.rs", false)], true, None, false),
            // "src" is a directory but not in expanded_dirs.
            ("collapsed_directory", &[("src", true)], true, None, false),
            ("expanded_directory", &[("src", true)], true, Some("src"), true),
        ];
        for &(name, nodes, focused, expanded, expected) in cases {
            let mut tree = make_tree(nodes.to_vec());
            tree.tree_focused = focused;
            if let Some(dir) = expanded {
                tree.expanded_dirs.insert(dir.into());
            }
            assert_eq!(tree.focused_is_expanded_dir(), expected, "case: {name}");
        }
    }

    // ── focused_parent_path tests ────────────────────────────────────

    #[test]
    fn focused_parent_path_cases() {
        // (name, nodes, focused, expected)
        #[rustfmt::skip]
        #[expect(clippy::type_complexity)] // focused_parent_path case table
        let cases: &[(&str, &[(&str, bool)], bool, Option<&str>)] = &[
            ("not_focused", &[("src/main.rs", false)], false, None),
            ("empty_tree", &[], true, None),
            // Root-level item — no parent.
            ("root_item", &[("src", true)], true, None),
            ("nested", &[("src/main.rs", false)], true, Some("src")),
            ("deep_nested", &[("a/b/c/file.rs", false)], true, Some("a/b/c")),
        ];
        for &(name, nodes, focused, expected) in cases {
            let mut tree = make_tree(nodes.to_vec());
            tree.tree_focused = focused;
            assert_eq!(
                tree.focused_parent_path(),
                expected.map(str::to_string),
                "case: {name}"
            );
        }
    }

    // ── tree_guide_prefix tests ────────────────────────────────────────────

    /// A single test case for [`tree_guide_prefix`].
    struct GuidePrefixCase {
        /// Human-readable name for failure diagnostics.
        name: &'static str,
        /// Which ancestor depths have continuation markers.
        mask: u64,
        /// Depth of the current node.
        depth: usize,
        /// Whether this is the last child at its depth.
        is_last: bool,
        /// Expected guide prefix string.
        expected: &'static str,
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn tree_guide_prefix_cases() {
        let cases = [
            // Root-level nodes have no guide lines regardless of mask or is_last.
            GuidePrefixCase {
                name: "root, mask=0, not last",
                mask: 0,
                depth: 0,
                is_last: false,
                expected: "",
            },
            GuidePrefixCase {
                name: "root, mask=0, last",
                mask: 0,
                depth: 0,
                is_last: true,
                expected: "",
            },
            GuidePrefixCase {
                name: "root, mask=all, not last",
                mask: 0b_1111,
                depth: 0,
                is_last: false,
                expected: "",
            },
            // Depth 1, no ancestor continuation.
            GuidePrefixCase {
                name: "depth 1, mask=0, not last",
                mask: 0,
                depth: 1,
                is_last: false,
                expected: "├ ",
            },
            GuidePrefixCase {
                name: "depth 1, mask=0, last",
                mask: 0,
                depth: 1,
                is_last: true,
                expected: "└ ",
            },
            // Depth 1, ancestor at depth 0 continues.
            GuidePrefixCase {
                name: "depth 1, mask=0b01, not last",
                mask: 0b_01,
                depth: 1,
                is_last: false,
                expected: "├ ",
            },
            GuidePrefixCase {
                name: "depth 1, mask=0b01, last",
                mask: 0b_01,
                depth: 1,
                is_last: true,
                expected: "└ ",
            },
            // Depth 2: ancestor depth 0 continues, depth 1 does not.
            GuidePrefixCase {
                name: "depth 2, mask=0b01, not last",
                mask: 0b_01,
                depth: 2,
                is_last: false,
                expected: "│ ├ ",
            },
            // Depth 2: both ancestors continue.
            GuidePrefixCase {
                name: "depth 2, mask=0b11, not last",
                mask: 0b_11,
                depth: 2,
                is_last: false,
                expected: "│ ├ ",
            },
            GuidePrefixCase {
                name: "depth 2, mask=0b11, last",
                mask: 0b_11,
                depth: 2,
                is_last: true,
                expected: "│ └ ",
            },
            // Depth 2: neither ancestor continues.
            GuidePrefixCase {
                name: "depth 2, mask=0, not last",
                mask: 0,
                depth: 2,
                is_last: false,
                expected: "  ├ ",
            },
            GuidePrefixCase {
                name: "depth 2, mask=0, last",
                mask: 0,
                depth: 2,
                is_last: true,
                expected: "  └ ",
            },
            // Depth 5: ancestors at 0,1,3 continue; 2 does not.
            GuidePrefixCase {
                name: "depth 5, mask=0b1011, not last",
                mask: 0b_1011,
                depth: 5,
                is_last: false,
                expected: "│ │   │ ├ ",
            },
            GuidePrefixCase {
                name: "depth 5, mask=0b1011, last",
                mask: 0b_1011,
                depth: 5,
                is_last: true,
                expected: "│ │   │ └ ",
            },
            // Bits beyond depth should be ignored.
            GuidePrefixCase {
                name: "high bits, mask=0x100, not last",
                mask: 0b1_0000_0000,
                depth: 1,
                is_last: false,
                expected: "├ ",
            },
            GuidePrefixCase {
                name: "high bits, mask=0x100, last",
                mask: 0b1_0000_0000,
                depth: 1,
                is_last: true,
                expected: "└ ",
            },
        ];

        for case in &cases {
            assert_eq!(
                tree_guide_prefix(case.mask, case.depth, case.is_last),
                case.expected,
                "case '{}' failed",
                case.name
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceeds u64 bit limit")]
    fn guide_prefix_depth_overflow_debug() {
        // debug_assert fires at depth >= 64 in debug builds.
        let _ = tree_guide_prefix(0, 64, false);
    }

    // ── Auto-sizing tree panel width tests ───────────────────────────
    //
    // These are pure-arithmetic tests of the geometry constants (0.6em mono
    // advance, 1.0em lucide, 4px gaps, 10px counts, 6px trailing gap) — they
    // do not touch the global font system, so the expected values are exact.

    #[test]
    fn mono_text_width_uses_06em_advance() {
        assert!(close(mono_text_width(0, 14.0), 0.0));
        assert!(close(mono_text_width(1, 14.0), 8.4));
        assert!(close(mono_text_width(10, 14.0), 84.0));
        // "binary" count label at 10px.
        assert!(close(mono_text_width(6, 10.0), 36.0));
    }

    #[test]
    fn tree_row_natural_width_cases() {
        struct Case {
            label: &'static str,
            guide_chars: usize,
            icon_size: f32,
            name: &'static str,
            name_suffix: Option<(&'static str, f32)>,
            counts: Option<(&'static str, &'static str)>,
            expected: f32,
        }
        #[rustfmt::skip] // compact one-line cases, matching the file's other case tables
        let cases = [
            // Depth 1 (guide 2 chars) + 14px icon + 4px gap + 11-char name.
            Case { label: "plain_file_row", guide_chars: 2, icon_size: TREE_FONT_SIZE, name: "src/main.rs", name_suffix: None, counts: None, expected: 127.2 },
            // Root dir row: 15px icon + 4px gap + "src  Loading…" (13 glyphs).
            Case { label: "dir_loading_suffix", guide_chars: 0, icon_size: TREE_ICON_SIZE, name: "src  Loading…", name_suffix: None, counts: None, expected: 128.2 },
            // Error file row: name + 4px gap + "[⚠]" at 11px.
            Case { label: "error_file_suffix", guide_chars: 0, icon_size: TREE_FONT_SIZE, name: "broken.txt", name_suffix: Some(("[⚠]", 11.0)), counts: None, expected: 125.8 },
            // Diff file row: name + "+123 -45" at 10px + 6px trailing gap.
            Case { label: "diff_counts", guide_chars: 0, icon_size: TREE_FONT_SIZE, name: "lib.rs", name_suffix: None, counts: Some(("+123", "-45")), expected: 122.4 },
            // 8-char name + "binary" at 10px + 6px trailing gap.
            Case { label: "diff_binary_count", guide_chars: 0, icon_size: TREE_FONT_SIZE, name: "data.bin", name_suffix: None, counts: Some(("binary", "")), expected: 127.2 },
        ];
        for case in cases {
            let w = tree_row_natural_width(
                case.guide_chars,
                case.icon_size,
                case.name,
                TREE_FONT_SIZE,
                case.name_suffix,
                case.counts,
            );
            assert!(
                close(w, case.expected),
                "case {}: expected {}, got {}",
                case.label,
                case.expected,
                w
            );
        }
    }

    /// Epsilon-equality for the pure-arithmetic width tests (0.001px is far
    /// below any visible difference; the formulas use `f32` accumulation).
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.001
    }

    /// Build a FileTree with a known viewport for panel-width tests.
    fn tree_with_panel_viewport(scroll_y: f32, viewport_h: Option<f32>) -> FileTree {
        let mut tree = FileTree::new(iced::widget::Id::new("width_test"));
        tree.scroll_y = scroll_y;
        tree.viewport_h = viewport_h;
        tree
    }

    #[test]
    fn tree_panel_width_clamps_to_minimum() {
        // Short rows → widest 68.4 + 26 = 94.4 < 260 → stays at minimum.
        let tree = tree_with_panel_viewport(0.0, Some(400.0));
        let widths = vec![68.4, 100.0, 50.0];
        assert!(close(tree_panel_width(&tree, &widths), TREE_MIN_WIDTH));
    }

    #[test]
    fn tree_panel_width_clamps_to_maximum() {
        // 50-char name → natural 438 → 464 → clamped to the 400px cap.
        let tree = tree_with_panel_viewport(0.0, Some(400.0));
        let long_name = "x".repeat(50);
        let widths = vec![tree_row_natural_width(
            0,
            TREE_FONT_SIZE,
            &long_name,
            TREE_FONT_SIZE,
            None,
            None,
        )];
        assert!(close(tree_panel_width(&tree, &widths), TREE_MAX_WIDTH));
    }

    #[test]
    fn tree_panel_width_scales_with_widest_row() {
        // 27-char name → 261.6 natural → 287.6 panel (within the bounds).
        let tree = tree_with_panel_viewport(0.0, Some(400.0));
        let wide = tree_row_natural_width(
            2,
            TREE_FONT_SIZE,
            "some_really_long_file_name.rs",
            TREE_FONT_SIZE,
            None,
            None,
        );
        let widths = vec![68.4, wide, 50.0];
        assert!(close(
            tree_panel_width(&tree, &widths),
            wide + 2.0 * TREE_ROW_H_PADDING + TREE_SCROLLBAR_ALLOWANCE
        ));
    }

    #[test]
    fn tree_panel_width_measures_all_rows_without_viewport() {
        // viewport_h None (first frame / non-scrolling fallback): all rows count.
        let tree = tree_with_panel_viewport(0.0, None);
        let long_name = "y".repeat(40);
        let wide =
            tree_row_natural_width(0, TREE_FONT_SIZE, &long_name, TREE_FONT_SIZE, None, None);
        let widths = vec![50.0, wide, 60.0];
        assert!(close(
            tree_panel_width(&tree, &widths),
            (wide + 2.0 * TREE_ROW_H_PADDING + TREE_SCROLLBAR_ALLOWANCE)
                .clamp(TREE_MIN_WIDTH, TREE_MAX_WIDTH)
        ));
    }

    #[test]
    fn tree_panel_width_filters_out_of_viewport_rows() {
        // Viewport 200px tall at scroll 0 → rows ~0..14 measured.
        let tree = tree_with_panel_viewport(0.0, Some(200.0));
        let mut widths = vec![68.4; 50];
        widths[40] = 500.0; // below the fold → must not inflate the width
        assert!(close(tree_panel_width(&tree, &widths), TREE_MIN_WIDTH));
        widths[2] = 500.0; // visible → inflates to the cap
        assert!(close(tree_panel_width(&tree, &widths), TREE_MAX_WIDTH));
    }

    #[test]
    fn tree_panel_width_scrolled_viewport_measures_mid_rows() {
        // scroll_y 400, viewport 200 → rows ~20..35 measured.
        let tree = tree_with_panel_viewport(400.0, Some(200.0));
        let mut widths = vec![68.4; 60];
        widths[10] = 500.0; // above the fold → ignored
        assert!(close(tree_panel_width(&tree, &widths), TREE_MIN_WIDTH));
        widths[30] = 500.0; // visible → inflates to the cap
        assert!(close(tree_panel_width(&tree, &widths), TREE_MAX_WIDTH));
    }

    #[test]
    fn tree_panel_width_empty_tree_stays_at_minimum() {
        let tree = tree_with_panel_viewport(0.0, None);
        assert!(close(tree_panel_width(&tree, &[]), TREE_MIN_WIDTH));
    }

    #[test]
    #[expect(clippy::cast_precision_loss)] // test-only encoding closure (depth, name length)
    fn collect_tree_row_widths_mirrors_render_order() {
        // src (dir, expanded) → src/main.rs, src/lib.rs; Cargo.toml at root.
        let mut tree = FileTree::new(iced::widget::Id::new("w"));
        tree.nodes = vec![
            TreeNode {
                name: "src".into(),
                full_path: "src".into(),
                is_dir: true,
                children: vec![
                    TreeNode {
                        name: "main.rs".into(),
                        full_path: "src/main.rs".into(),
                        is_dir: false,
                        children: vec![],
                        error: None,
                        ignored: false,
                    },
                    TreeNode {
                        name: "lib.rs".into(),
                        full_path: "src/lib.rs".into(),
                        is_dir: false,
                        children: vec![],
                        error: None,
                        ignored: false,
                    },
                ],
                error: None,
                ignored: false,
            },
            TreeNode {
                name: "Cargo.toml".into(),
                full_path: "Cargo.toml".into(),
                is_dir: false,
                children: vec![],
                error: None,
                ignored: false,
            },
        ];
        tree.expanded_dirs.insert("src".to_string());

        // Closure encodes depth and name length so the order is observable.
        let widths = collect_tree_row_widths(&tree.nodes, &tree.expanded_dirs, |node, depth| {
            depth as f32 * 10.0 + node.name.chars().count() as f32
        });
        assert_eq!(widths, vec![3.0, 17.0, 16.0, 10.0]);

        // Collapsed src → only the two root rows remain.
        tree.expanded_dirs.clear();
        let widths = collect_tree_row_widths(&tree.nodes, &tree.expanded_dirs, |node, depth| {
            depth as f32 * 10.0 + node.name.chars().count() as f32
        });
        assert_eq!(widths, vec![3.0, 10.0]);
    }

    // ── expand_dir_and_focus_first_child / collapse_dir_and_keep_focus tests ──

    /// Build a `FileTree` with a `src/` directory containing `lib.rs` and `main.rs`.
    /// The returned tree has `nodes` populated (pre-sorted) and `visible_tree_nodes`
    /// initially empty. Callers expand/collapse `"src"` as needed and call the helpers.
    fn tree_with_src_dir() -> FileTree {
        let mut tree = FileTree::new(iced::widget::Id::new("test"));
        tree.nodes = vec![TreeNode {
            name: "src".into(),
            full_path: "src".into(),
            is_dir: true,
            children: vec![
                TreeNode {
                    name: "lib.rs".into(),
                    full_path: "src/lib.rs".into(),
                    is_dir: false,
                    children: vec![],
                    error: None,
                    ignored: false,
                },
                TreeNode {
                    name: "main.rs".into(),
                    full_path: "src/main.rs".into(),
                    is_dir: false,
                    children: vec![],
                    error: None,
                    ignored: false,
                },
            ],
            error: None,
            ignored: false,
        }];
        tree
    }

    #[test]
    fn expand_dir_advances_to_first_child() {
        let mut tree = tree_with_src_dir();
        tree.expanded_dirs.insert("src".into());
        // No visible nodes yet — rebuild is part of the helper.
        assert!(tree.visible_tree_nodes.is_empty());

        let _task = tree.expand_dir_and_focus_first_child::<()>("src");

        // Rebuilt visible tree: src, src/lib.rs, src/main.rs
        assert_eq!(tree.visible_tree_nodes.len(), 3);
        assert_eq!(tree.visible_tree_nodes[0].0, "src");
        assert_eq!(tree.visible_tree_nodes[1].0, "src/lib.rs");
        assert_eq!(tree.visible_tree_nodes[2].0, "src/main.rs");
        // Focus advances to the first child (right after "src").
        assert_eq!(tree.tree_focus_index, 1);
    }

    #[test]
    fn expand_dir_no_children_stays_on_dir() {
        let mut tree = tree_with_src_dir();
        tree.expanded_dirs.insert("src".into());
        // Remove children so the directory has no expandable content.
        tree.nodes[0].children.clear();

        let _task = tree.expand_dir_and_focus_first_child::<()>("src");

        // Only "src" in the visible tree.
        assert_eq!(tree.visible_tree_nodes.len(), 1);
        assert_eq!(tree.visible_tree_nodes[0].0, "src");
        // Focus stays on "src" because there is no child to advance to.
        assert_eq!(tree.tree_focus_index, 0);
    }

    #[test]
    fn expand_dir_not_in_expanded_dirs_panics_in_debug() {
        let mut tree = tree_with_src_dir();
        // Intentionally NOT inserting into expanded_dirs — the debug_assert
        // should fire. Use a catch_unwind to avoid test failure in release builds.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _task = tree.expand_dir_and_focus_first_child::<()>("src");
        }));
        // In debug builds this panics; in release builds it doesn't.
        #[cfg(debug_assertions)]
        assert!(
            result.is_err(),
            "debug_assert should fire when path not in expanded_dirs"
        );
        #[cfg(not(debug_assertions))]
        assert!(result.is_ok(), "no panic expected in release builds");
    }

    #[test]
    fn collapse_dir_keeps_focus_on_directory() {
        let mut tree = tree_with_src_dir();
        // Pre-expand so collapsing has an effect.
        tree.expanded_dirs.insert("src".into());
        tree.rebuild_visible();
        assert_eq!(tree.visible_tree_nodes.len(), 3); // src, lib.rs, main.rs

        // Now collapse — remove from expanded_dirs and call the helper.
        tree.expanded_dirs.remove("src");
        let _task = tree.collapse_dir_and_keep_focus::<()>("src");

        // Collapsed: only "src" visible.
        assert_eq!(tree.visible_tree_nodes.len(), 1);
        assert_eq!(tree.visible_tree_nodes[0].0, "src");
        // Focus stays on "src".
        assert_eq!(tree.tree_focus_index, 0);
    }

    #[test]
    fn collapse_dir_not_in_visible_tree_still_finds_it() {
        let mut tree = tree_with_src_dir();
        // "src" has been removed from expanded_dirs and visible_tree_nodes is empty.
        // Even without an explicit rebuild_visible first, the helper should
        // rebuild and find "src" since it's still in nodes.
        let _task = tree.collapse_dir_and_keep_focus::<()>("src");

        // After rebuild_visible, "src" should appear (it's in nodes).
        assert_eq!(tree.visible_tree_nodes.len(), 1);
        assert_eq!(tree.visible_tree_nodes[0].0, "src");
        assert_eq!(tree.tree_focus_index, 0);
    }

    #[test]
    fn collapse_dir_still_in_expanded_dirs_panics_in_debug() {
        let mut tree = tree_with_src_dir();
        tree.expanded_dirs.insert("src".into());
        // Call without removing from expanded_dirs first — debug_assert fires.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _task = tree.collapse_dir_and_keep_focus::<()>("src");
        }));
        #[cfg(debug_assertions)]
        assert!(
            result.is_err(),
            "debug_assert should fire when path still in expanded_dirs"
        );
        #[cfg(not(debug_assertions))]
        assert!(result.is_ok(), "no panic expected in release builds");
    }

    // ── scroll_to_tree_focus / ScrollIntoView tests ──────────────────

    /// Helper: build a FileTree with `n` flat file entries, a known viewport
    /// height, and `scroll_y` set to a given offset.
    fn tree_with_viewport(n: usize, scroll_y: f32, viewport_h: Option<f32>) -> FileTree {
        let mut tree = FileTree::new(iced::widget::Id::new("scroll_test"));
        tree.visible_tree_nodes = (0..n).map(|i| (format!("file_{i}.rs"), false)).collect();
        tree.scroll_y = scroll_y;
        tree.viewport_h = viewport_h;
        tree
    }

    #[test]
    fn scroll_to_tree_focus_positions_viewport() {
        struct Case {
            label: &'static str,
            n: usize,
            scroll_y: f32,
            viewport_h: Option<f32>,
            focus: usize,
            mode: ScrollMode,
            expected: f32,
        }
        #[rustfmt::skip] // compact one-line cases, matching the file's other case tables
        let cases = [
            // Viewport y=40..440, focus idx 3 (row y≈54..72) fully within → unchanged.
            Case { label: "fully_visible", n: 30, scroll_y: 40.0, viewport_h: Some(400.0), focus: 3, mode: ScrollMode::ScrollIntoView, expected: 40.0 },
            // Viewport 0..200, focus idx 15 (y≈273) below bottom → advance one row (~18.2).
            Case { label: "below_viewport", n: 30, scroll_y: 0.0, viewport_h: Some(200.0), focus: 15, mode: ScrollMode::ScrollIntoView, expected: 18.2 },
            // Viewport 100..500, focus idx 3 (y≈54.6) above top → scroll_y = focus_y.
            Case { label: "above_viewport", n: 30, scroll_y: 100.0, viewport_h: Some(400.0), focus: 3, mode: ScrollMode::ScrollIntoView, expected: 54.6 },
            // Viewport 50..450, focus idx 2 row bottom ≈54.6 straddles top → unchanged.
            Case { label: "top_edge_partial", n: 30, scroll_y: 50.0, viewport_h: Some(400.0), focus: 2, mode: ScrollMode::ScrollIntoView, expected: 50.0 },
            // viewport_h None → SnapToTop fallback: scroll_y = focus_y ≈ 182.
            Case { label: "no_viewport_snap", n: 30, scroll_y: 10.0, viewport_h: None, focus: 10, mode: ScrollMode::ScrollIntoView, expected: 182.0 },
            // SnapToTop: scroll_y = focus_y (8 * 18.2 ≈ 145.6).
            Case { label: "snap_to_top", n: 30, scroll_y: 0.0, viewport_h: Some(400.0), focus: 8, mode: ScrollMode::SnapToTop, expected: 145.6 },
            // Empty tree: early-returns Task::none(); scroll_y unchanged.
            Case { label: "empty_tree", n: 0, scroll_y: 0.0, viewport_h: Some(400.0), focus: 0, mode: ScrollMode::ScrollIntoView, expected: 0.0 },
        ];
        // Must bind (not discard) because iced::Task is #[must_use].
        for case in cases {
            let mut tree = tree_with_viewport(case.n, case.scroll_y, case.viewport_h);
            tree.tree_focus_index = case.focus;
            let _task = scroll_to_tree_focus::<()>(&mut tree, case.mode);
            assert!(
                (tree.scroll_y - case.expected).abs() < 0.01,
                "case {}: scroll_y expected {}, got {}",
                case.label,
                case.expected,
                tree.scroll_y
            );
        }
    }

    #[test]
    fn classify_click_kinds() {
        // Short press, no previous click.
        assert_eq!(classify_click(0.0, None), ClickKind::Click);
        assert_eq!(classify_click(6.0, None), ClickKind::Click);
        // Big distance is always a drag, regardless of timing.
        assert_eq!(classify_click(6.1, None), ClickKind::Drag);
        assert_eq!(classify_click(100.0, Some(Duration::ZERO)), ClickKind::Drag);
        // Tiny distance within 300ms of the last click.
        assert_eq!(
            classify_click(1.0, Some(Duration::from_millis(299))),
            ClickKind::MultiClick
        );
        // Tiny distance with a 301ms gap.
        assert_eq!(
            classify_click(1.0, Some(Duration::from_millis(301))),
            ClickKind::Click
        );
        // Exactly at the 300ms boundary is an inclusive MultiClick.
        assert_eq!(
            classify_click(1.0, Some(DOUBLE_CLICK_WINDOW)),
            ClickKind::MultiClick
        );
    }
}

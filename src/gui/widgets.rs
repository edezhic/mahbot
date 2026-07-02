//! Shared dashboard widgets: styled pick_list, PickOption type, FileTree state struct
//! and build_tree_panel for shared file-tree panel rendering.

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use iced::widget::{
    self, Row, Space, button, column, container, pick_list, scrollable, text, text_input,
};
use iced::{Alignment, Color, Element, Length, Padding, Task};

use iced_selection;

use super::theme;

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

/// Flexoki-dark themed style for [`fn@text_input`] widgets.
/// Matches [`pick_list_style`] for visual consistency.
pub fn text_input_style(_theme: &iced::Theme, _status: text_input::Status) -> text_input::Style {
    text_input::Style {
        background: iced::Background::Color(theme::BG_ELEVATED),
        border: iced::Border {
            radius: 4.0.into(),
            width: 1.0,
            color: theme::BORDER_STRONG,
        },
        icon: theme::TEXT_MUTED,
        placeholder: theme::TEXT_MUTED,
        value: theme::TEXT_PRIMARY,
        selection: theme::ACCENT,
    }
}

/// Render a styled error banner for dashboard panels.
pub fn error_banner<'a, Message: 'a>(err: &'a str) -> Element<'a, Message> {
    container(text(err).size(13).color(theme::STATUS_ERROR))
        .padding(8)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(iced::Color::from_rgba(
                1.0, 0.267, 0.4, 0.08,
            ))),
            border: iced::Border {
                radius: 4.0.into(),
                ..iced::Border::default()
            },
            ..container::Style::default()
        })
        .into()
}

/// Render a centered empty-state placeholder with a lucide icon and label.
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

/// Create a selectable text widget with the given color.
///
/// Accepts both borrowed (`&str`) and owned (`String`) text content.
pub fn selectable_text<'a>(
    content: impl iced_selection::text::IntoFragment<'a>,
    color: Color,
) -> iced_selection::text::Text<'a, iced::Theme, iced::Renderer> {
    iced_selection::text::Text::new(content).style(move |_theme| iced_selection::text::Style {
        color: Some(color),
        ..Default::default()
    })
}

/// Render formatted diff stats (+X/−Y) matching ticket card style.
///
/// Returns a [`Row`] showing only non-zero sides with a `/` separator.
/// Returns an empty [`Row`] when both `added` and `removed` are zero.
///
/// Callers typically wrap this in a styled [`button()`] with an appropriate
/// action message.
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

// ── Debounce helpers ───────────────────────────────────────────────

/// Spawn a sleep task that returns `generation` after `ms` milliseconds.
///
/// Use with a debounced refresh message to avoid multiple rapid refreshes:
/// increment a generation counter, spawn this task with the new generation,
/// and in the response handler check [`debounce_should_process`].
pub async fn debounce_sleep(ms: u64, generation: u64) -> u64 {
    tokio::time::sleep(Duration::from_millis(ms)).await;
    generation
}

/// Returns `true` if a debounced refresh should proceed.
///
/// Pass the generation from the debounce response, the current generation
/// counter, and the pending flag.  This prevents stale debounce tasks
/// from triggering a refresh after a newer task has been spawned.
#[must_use]
pub const fn debounce_should_process(
    generation: u64,
    current_generation: u64,
    pending: bool,
) -> bool {
    generation == current_generation && pending
}

/// Render a pagination bar with ← Prev / Page X of Y / Next →.
///
/// Returns a zero-height element when `total_pages == 0` so callers can
/// unconditionally push it.  The 8px top spacer is included.
///
/// # Generics
///
/// `Message` must be `Clone` so the `on_prev` / `on_next` values can be
/// passed to both the condition check and the button builder.
pub fn pagination_bar<'a, Message: 'a + Clone>(
    page: usize,
    total_pages: usize,
    on_prev: Message,
    on_next: Message,
) -> Element<'a, Message> {
    if total_pages == 0 {
        return Space::new().height(0).into();
    }

    let prev_button = button(text("← Prev").size(12))
        .style(super::theme::button_text)
        .on_press_maybe(if page > 0 { Some(on_prev) } else { None });

    let next_button = button(text("Next →").size(12))
        .style(super::theme::button_text)
        .on_press_maybe(if page + 1 < total_pages {
            Some(on_next)
        } else {
            None
        });

    let pagination = Row::with_children(vec![
        prev_button.into(),
        Space::new().width(8).into(),
        text(format!("Page {} of {}", page + 1, total_pages))
            .size(12)
            .color(super::theme::TEXT_MUTED)
            .into(),
        Space::new().width(8).into(),
        next_button.into(),
    ])
    .align_y(Alignment::Center);

    column![Space::new().height(8), pagination].into()
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

    /// Clear all tree state (nodes, expanded dirs, visible list, focus).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.expanded_dirs.clear();
        self.tree_focused = false;
        self.tree_focus_index = 0;
        self.visible_tree_nodes.clear();
        self.scroll_y = 0.0;
        self.viewport_h = None;
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
#[allow(clippy::cast_precision_loss)]
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
/// Renders a scrollable, fixed-width column wrapping the pre-built
/// `tree_element` rows. A focus border is applied when `file_tree.tree_focused`
/// is true.
///
/// `on_scroll` is attached to the inner [`widget::scrollable()`] via
/// `on_scroll` and fires whenever the viewport changes
/// (scrollbar drag, mouse wheel, programmatic scroll). The caller should
/// produce a message that updates [`FileTree::scroll_y`] and
/// [`FileTree::viewport_h`] from the [`iced::widget::scrollable::Viewport`] data.
pub fn build_tree_panel<'a, Message: 'a>(
    file_tree: &'a FileTree,
    tree_rows: Vec<Element<'a, Message>>,
    on_scroll: impl Fn(scrollable::Viewport) -> Message + 'a,
) -> Element<'a, Message> {
    let tree_body = widget::scrollable(column(tree_rows).spacing(0))
        .id(file_tree.tree_scroll_id.clone())
        .on_scroll(on_scroll)
        .width(Length::Fill)
        .height(Length::Fill)
        .direction(theme::vertical_scrollbar())
        .style(theme::scrollbar_style);

    let tree_inner: Element<'_, Message> = container(tree_body)
        .width(Length::Fixed(260.0))
        .height(Length::Fill)
        .style(|_t: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_SURFACE)),
            border: iced::Border {
                radius: 0.0.into(),
                width: 0.0,
                color: iced::Color::TRANSPARENT,
            },
            ..Default::default()
        })
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn focus_path_found() {
        let mut tree = make_tree(vec![
            ("src", true),
            ("src/main.rs", false),
            ("Cargo.toml", false),
        ]);
        assert_eq!(tree.focus_path("src/main.rs"), Some(1));
        assert_eq!(tree.tree_focus_index, 1);
    }

    #[test]
    fn focus_path_not_found() {
        let mut tree = make_tree(vec![("src", true), ("Cargo.toml", false)]);
        tree.tree_focus_index = 42;
        assert_eq!(tree.focus_path("nonexistent"), None);
        assert_eq!(tree.tree_focus_index, 42);
    }

    #[test]
    fn focus_path_empty_tree() {
        let mut tree = make_tree(vec![]);
        assert_eq!(tree.focus_path("anything"), None);
        assert_eq!(tree.tree_focus_index, 0);
    }

    #[test]
    fn focus_path_first_node() {
        let mut tree = make_tree(vec![("src", true), ("src/main.rs", false)]);
        assert_eq!(tree.focus_path("src"), Some(0));
        assert_eq!(tree.tree_focus_index, 0);
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
    fn focused_tree_node_not_focused() {
        let tree = make_tree(vec![("src", true), ("src/main.rs", false)]);
        // Tree is not focused (default).
        assert!(tree.focused_tree_node().is_none());
    }

    #[test]
    fn focused_tree_node_empty_visible_nodes() {
        let mut tree = make_tree(vec![]);
        tree.tree_focused = true;
        assert!(tree.focused_tree_node().is_none());
    }

    #[test]
    fn focused_tree_node_clamps_index() {
        let mut tree = make_tree(vec![("a", false), ("b", false)]);
        tree.tree_focused = true;
        // Set index beyond bounds — method should clamp.
        tree.tree_focus_index = 10;
        let (idx, path, is_dir) = tree.focused_tree_node().unwrap();
        assert_eq!(idx, 1);
        assert_eq!(path, "b");
        assert!(!is_dir);
    }

    #[test]
    fn focused_tree_node_returns_correct_node() {
        let mut tree = make_tree(vec![
            ("src", true),
            ("src/main.rs", false),
            ("Cargo.toml", false),
        ]);
        tree.tree_focused = true;
        tree.tree_focus_index = 1;
        let (idx, path, is_dir) = tree.focused_tree_node().unwrap();
        assert_eq!(idx, 1);
        assert_eq!(path, "src/main.rs");
        assert!(!is_dir);
    }

    #[test]
    fn focused_tree_node_returns_directory() {
        let mut tree = make_tree(vec![("src", true), ("src/main.rs", false)]);
        tree.tree_focused = true;
        tree.tree_focus_index = 0;
        let (idx, path, is_dir) = tree.focused_tree_node().unwrap();
        assert_eq!(idx, 0);
        assert_eq!(path, "src");
        assert!(is_dir);
    }

    // ── focused_is_expanded_dir tests ────────────────────────────────

    #[test]
    fn focused_is_expanded_dir_not_focused() {
        let tree = make_tree(vec![("src", true)]);
        // Tree is not focused.
        assert!(!tree.focused_is_expanded_dir());
    }

    #[test]
    fn focused_is_expanded_dir_empty_tree() {
        let mut tree = make_tree(vec![]);
        tree.tree_focused = true;
        assert!(!tree.focused_is_expanded_dir());
    }

    #[test]
    fn focused_is_expanded_dir_file() {
        let mut tree = make_tree(vec![("main.rs", false)]);
        tree.tree_focused = true;
        assert!(!tree.focused_is_expanded_dir());
    }

    #[test]
    fn focused_is_expanded_dir_collapsed_directory() {
        let mut tree = make_tree(vec![("src", true)]);
        tree.tree_focused = true;
        // "src" is a directory but not in expanded_dirs.
        assert!(!tree.focused_is_expanded_dir());
    }

    #[test]
    fn focused_is_expanded_dir_expanded_directory() {
        let mut tree = make_tree(vec![("src", true)]);
        tree.tree_focused = true;
        tree.expanded_dirs.insert("src".into());
        assert!(tree.focused_is_expanded_dir());
    }

    // ── focused_parent_path tests ────────────────────────────────────

    #[test]
    fn focused_parent_path_not_focused() {
        let tree = make_tree(vec![("src/main.rs", false)]);
        assert!(tree.focused_parent_path().is_none());
    }

    #[test]
    fn focused_parent_path_empty_tree() {
        let mut tree = make_tree(vec![]);
        tree.tree_focused = true;
        assert!(tree.focused_parent_path().is_none());
    }

    #[test]
    fn focused_parent_path_root_item() {
        let mut tree = make_tree(vec![("src", true)]);
        tree.tree_focused = true;
        // Root-level item — no parent.
        assert!(tree.focused_parent_path().is_none());
    }

    #[test]
    fn focused_parent_path_nested() {
        let mut tree = make_tree(vec![("src/main.rs", false)]);
        tree.tree_focused = true;
        assert_eq!(tree.focused_parent_path(), Some("src".into()));
    }

    #[test]
    fn focused_parent_path_deep_nested() {
        let mut tree = make_tree(vec![("a/b/c/file.rs", false)]);
        tree.tree_focused = true;
        assert_eq!(tree.focused_parent_path(), Some("a/b/c".into()));
    }

    /// Build the expected guide string from a literal. Makes it easier to
    /// see the box-drawing characters in test output.
    fn g(guide: &str) -> String {
        guide.to_string()
    }

    #[test]
    fn guide_prefix_depth_zero() {
        // Root-level nodes have no guide lines regardless of mask or is_last.
        assert_eq!(tree_guide_prefix(0, 0, false), g(""));
        assert_eq!(tree_guide_prefix(0, 0, true), g(""));
        assert_eq!(tree_guide_prefix(0b_1111, 0, false), g(""));
    }

    #[test]
    fn guide_prefix_depth_one_not_last() {
        // Depth 1, no ancestor continuation, non-last child.
        // Just the connector (no ancestor guide needed — connector replaces
        // the single ancestor slot).
        assert_eq!(tree_guide_prefix(0, 1, false), g("├ "));
    }

    #[test]
    fn guide_prefix_depth_one_last() {
        // Depth 1, no ancestor continuation, last child → └ connector.
        assert_eq!(tree_guide_prefix(0, 1, true), g("└ "));
    }

    #[test]
    fn guide_prefix_depth_one_ancestor_continues_not_last() {
        // Depth 1, ancestor at depth 0 has continuation (bit 0 = 1).
        // Connector replaces the depth-0 ancestor slot, showing just ├.
        assert_eq!(tree_guide_prefix(0b_01, 1, false), g("├ "));
    }

    #[test]
    fn guide_prefix_depth_one_ancestor_continues_last() {
        // Depth 1, ancestor continues, current node last → └
        assert_eq!(tree_guide_prefix(0b_01, 1, true), g("└ "));
    }

    #[test]
    fn guide_prefix_depth_two_mixed_ancestors() {
        // Depth 2: ancestor depth 0 continues, depth 1 does NOT continue.
        // mask = bit 0 set, bit 1 clear.
        // Only the depth-0 ancestor appears as a guide; depth-1 slot is
        // replaced by the connector.
        assert_eq!(tree_guide_prefix(0b_01, 2, false), g("│ ├ "));
    }

    #[test]
    fn guide_prefix_depth_two_all_ancestors_continue_not_last() {
        // Both ancestors continue (mask bits 0 and 1 set).
        // Only depth-0 guide appears; depth-1 replaced by connector.
        assert_eq!(tree_guide_prefix(0b_11, 2, false), g("│ ├ "));
    }

    #[test]
    fn guide_prefix_depth_two_all_ancestors_continue_last() {
        // Both ancestors continue, current node last → "│ └ "
        assert_eq!(tree_guide_prefix(0b_11, 2, true), g("│ └ "));
    }

    #[test]
    fn guide_prefix_depth_two_no_ancestor_continuation() {
        // Depth 2: neither ancestor continues (mask bits 0 and 1 clear).
        // Current node not last → two spaces for depth 0 + ├ connector.
        assert_eq!(tree_guide_prefix(0, 2, false), g("  ├ "));
    }

    #[test]
    fn guide_prefix_depth_two_no_ancestor_continuation_last() {
        // Depth 2: no ancestor continuation, last child → "  └ "
        assert_eq!(tree_guide_prefix(0, 2, true), g("  └ "));
    }

    #[test]
    fn guide_prefix_deep_tree() {
        // Depth 5, ancestors at 0,1,3 continue; 2 does not; 4 is the slot
        // replaced by the connector.
        // mask bits set: 0, 1, 3  →  binary 0b_1011
        // Only 4 ancestor guides (depths 0-3), connector replaces depth 4.
        //            d0 d1 d2 d3
        //            │  │  sp │
        assert_eq!(tree_guide_prefix(0b_1011, 5, false), g("│ │   │ ├ "));
    }

    #[test]
    fn guide_prefix_deep_tree_last() {
        // Same as above but last child → └ instead of ├
        assert_eq!(tree_guide_prefix(0b_1011, 5, true), g("│ │   │ └ "));
    }

    #[test]
    fn guide_prefix_mask_ignores_bits_above_depth() {
        // Bits beyond depth should be ignored. depth=1 with high bits set.
        assert_eq!(tree_guide_prefix(0b1_0000_0000, 1, false), g("├ "));
        assert_eq!(tree_guide_prefix(0b1_0000_0000, 1, true), g("└ "));
    }

    #[test]
    #[should_panic(expected = "exceeds u64 bit limit")]
    fn guide_prefix_depth_overflow_debug() {
        // debug_assert fires at depth >= 64 in debug builds.
        let _ = tree_guide_prefix(0, 64, false);
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
                },
                TreeNode {
                    name: "main.rs".into(),
                    full_path: "src/main.rs".into(),
                    is_dir: false,
                    children: vec![],
                    error: None,
                },
            ],
            error: None,
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
    fn tree_with_viewport(n: usize, scroll_y: f32, viewport_h: f32) -> FileTree {
        let mut tree = FileTree::new(iced::widget::Id::new("scroll_test"));
        tree.visible_tree_nodes = (0..n).map(|i| (format!("file_{i}.rs"), false)).collect();
        tree.scroll_y = scroll_y;
        tree.viewport_h = Some(viewport_h);
        tree
    }

    #[test]
    fn scroll_into_view_row_fully_visible_no_scroll() {
        // Viewport: y=40..440 (400px tall, starting at row index ~2)
        // Focus on index 3. Row spans y=54..72 (3 * 18 = 54).
        // Viewport bottom is 440. Row is fully within 40..440.
        let mut tree = tree_with_viewport(30, 40.0, 400.0);
        tree.tree_focus_index = 3; // 3 * ~18.2 = ~54.6

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // scroll_y unchanged — no scroll needed.
        assert!(
            (tree.scroll_y - 40.0).abs() < 0.01,
            "scroll_y should remain 40"
        );
    }

    #[test]
    fn scroll_into_view_row_below_viewport_advances_one_row() {
        // Viewport: y=0..200 (200px tall)
        // Focus on index 15 (y~273). Row bottom ~291.2.
        // Row is below viewport bottom (200).
        let mut tree = tree_with_viewport(30, 0.0, 200.0);
        tree.tree_focus_index = 15; // 15 * ~18.2 = ~273

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // scroll_y advanced by one row height (~18.2px).
        assert!(
            (tree.scroll_y - 18.2_f32).abs() < 0.01,
            "scroll_y should advance by ~18.2, got {}",
            tree.scroll_y
        );
    }

    #[test]
    fn scroll_into_view_row_above_viewport_brings_to_top() {
        // Viewport: y=100..500 (400px tall)
        // Focus on index 3 (y~54.6). Row bottom ~72.8.
        // Row bottom (~72.8) is above viewport top (100).
        let mut tree = tree_with_viewport(30, 100.0, 400.0);
        tree.tree_focus_index = 3; // 3 * ~18.2 = ~54.6

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // scroll_y set to focus_y (~54.6) — bring row to top.
        assert!(
            (tree.scroll_y - 54.6).abs() < 0.01,
            "scroll_y should be ~54.6, got {}",
            tree.scroll_y
        );
    }

    #[test]
    fn scroll_into_view_partially_visible_at_top_edge_no_scroll() {
        // Viewport: y=50..450 (400px tall)
        // Focus on index 2 (y~36.4). Row bottom ~54.6.
        // Row bottom (~54.6) is below viewport top (50) → partially visible.
        let mut tree = tree_with_viewport(30, 50.0, 400.0);
        tree.tree_focus_index = 2; // 2 * ~18.2 = ~36.4

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // scroll_y unchanged — row is partially visible at top edge.
        assert!(
            (tree.scroll_y - 50.0).abs() < 0.01,
            "scroll_y should remain 50, got {}",
            tree.scroll_y
        );
    }

    #[test]
    fn scroll_into_view_unknown_viewport_falls_back_to_snap() {
        // viewport_h is None — should fall back to SnapToTop.
        let mut tree = tree_with_viewport(30, 10.0, 0.0);
        tree.viewport_h = None;
        tree.tree_focus_index = 10; // 10 * ~18.2 = ~182

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // Falls back to absolute scroll: scroll_y = focus_y ≈ 182.
        assert!(
            (tree.scroll_y - 182.0).abs() < 0.01,
            "scroll_y should snap to ~182, got {}",
            tree.scroll_y
        );
    }

    #[test]
    fn scroll_snap_to_top_sets_scroll_y() {
        let mut tree = tree_with_viewport(30, 0.0, 400.0);
        tree.tree_focus_index = 8; // 8 * ~18.2 = ~145.6

        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::SnapToTop);

        // SnapToTop sets scroll_y to focus_y.
        assert!(
            (tree.scroll_y - 145.6).abs() < 0.01,
            "scroll_y should be ~145.6, got {}",
            tree.scroll_y
        );
    }

    #[test]
    fn scroll_into_view_empty_tree_noop() {
        let mut tree = FileTree::new(iced::widget::Id::new("scroll_test"));
        tree.viewport_h = Some(400.0);

        // Must bind (not discard) because iced::Task is #[must_use].
        let _task = scroll_to_tree_focus::<()>(&mut tree, ScrollMode::ScrollIntoView);

        // The task type is opaque, but we can verify it's not panicking
        // and that scroll_y stays at its default.
        assert!((tree.scroll_y - 0.0).abs() < 0.01);
        // The function returns Task::none() for empty trees.
        // We can't easily inspect Task contents, so we just check no crash.
    }
}

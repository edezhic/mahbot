//! Shared dashboard widgets: styled pick_list, PickOption type, FileTree state struct
//! and build_tree_panel for shared file-tree panel rendering.

use std::collections::HashSet;
use std::time::Duration;

use iced::widget::{self, button, column, container, pick_list, text, text_input};
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
}

/// Estimated height per tree row for scroll-into-view on keyboard navigation.
pub const ESTIMATED_TREE_ROW_HEIGHT: f32 = 16.0;

/// Scroll the tree panel to bring the focused row into view.
#[allow(clippy::cast_precision_loss)]
pub fn scroll_to_tree_focus<Message: 'static>(file_tree: &FileTree) -> Task<Message> {
    if file_tree.visible_tree_nodes.is_empty() {
        return Task::none();
    }
    let offset_y = file_tree.tree_focus_index as f32 * ESTIMATED_TREE_ROW_HEIGHT;
    iced::widget::operation::scroll_to(
        file_tree.tree_scroll_id.clone(),
        iced::widget::operation::AbsoluteOffset {
            x: 0.0,
            y: offset_y,
        },
    )
}

/// Build a file-tree panel widget.
///
/// Renders a scrollable, fixed-width column wrapping the pre-built
/// `tree_element` rows. A focus border is applied when `file_tree.tree_focused`
/// is true.
pub fn build_tree_panel<'a, Message: 'a>(
    file_tree: &'a FileTree,
    tree_rows: Vec<Element<'a, Message>>,
) -> Element<'a, Message> {
    let tree_body = widget::scrollable(column(tree_rows).spacing(0))
        .id(file_tree.tree_scroll_id.clone())
        .width(Length::Fill)
        .height(Length::Fill)
        .direction(widget::scrollable::Direction::Vertical(
            theme::thin_scrollbar(),
        ))
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

    // ── tree_guide_prefix tests ─────────────────────────────────────

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
        assert_eq!(tree_guide_prefix(0b_1_0000_0000, 1, false), g("├ "));
        assert_eq!(tree_guide_prefix(0b_1_0000_0000, 1, true), g("└ "));
    }

    #[test]
    #[should_panic(expected = "exceeds u64 bit limit")]
    fn guide_prefix_depth_overflow_debug() {
        // debug_assert fires at depth >= 64 in debug builds.
        let _ = tree_guide_prefix(0, 64, false);
    }
}

//! Shared rendering components for the agent-session ledger — the tool block,
//! one-line key-value pairs summary, and boundary-aware truncation used by
//! both the Sessions page (full view) and the Running Agents page (compact).

use iced::widget::{Column, Row, container, text, tooltip};
use iced::{Alignment, Element, Length};

use crate::agent::registry::RunningTool;
use crate::gui::theme;
use crate::gui::widgets;

/// Max display length (Unicode chars) of a single argument VALUE in the
/// one-line key-value pairs summary. Hover tooltip always shows full value.
const MAX_TOOL_VALUE_CHARS: usize = 104;

/// Max display length of the whole key-value pairs line (values already
/// per-value truncated). Cut at a pair boundary with "…".
const MAX_TOOL_PAIRS_LINE_CHARS: usize = 208;

/// Max width (px) of hover tooltip content; long values wrap within it.
pub(crate) const MAX_TOOL_TOOLTIP_WIDTH: f32 = 560.0;

/// A word/path-delimiter boundary at which wrapped/truncated text may be
/// cut: whitespace (word wrap) and `/`, `_`, `-` (paths, URLs, identifiers).
fn is_delim(c: char) -> bool {
    c.is_whitespace() || matches!(c, '/' | '_' | '-')
}

/// Truncate `s` to at most `max_chars` Unicode chars, choosing a cut at a word
/// or path/URL delimiter boundary (whitespace, `/`, `_`, `-`) rather than
/// mid-token, and appending "…" when truncated. When no delimiter falls within
/// the cap the cut falls back to a hard char boundary so the limit is always
/// honoured.
pub(crate) fn truncate_at_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let hard_end = s
        .char_indices()
        .nth(max_chars)
        .map(|(byte_idx, _)| byte_idx)
        .expect("char count exceeds max_chars, so a char exists at index max_chars");
    // If the cut lands right before a delimiter, the first `max_chars` chars
    // already end at a complete word/path segment — cut exactly there.
    if s[hard_end..].chars().next().is_some_and(is_delim) {
        return format!("{}…", s[..hard_end].trim_end());
    }
    // Otherwise back up to the last delimiter strictly inside the cap so the
    // cut lands at a wrap boundary (the delimiter is the wrapped segment's
    // final char) instead of mid-word.
    let mut cut = None;
    for (char_pos, (byte_idx, c)) in s.char_indices().enumerate() {
        if char_pos >= max_chars {
            break;
        }
        if is_delim(c) {
            cut = Some(byte_idx + c.len_utf8());
        }
    }
    let end = cut.unwrap_or(hard_end);
    format!("{}…", s[..end].trim_end())
}

/// Single-line display form of `s`: control characters (newlines, tabs)
/// collapsed to spaces, collecting at most `max_chars + 1` chars — all that
/// [`truncate_at_boundary`] can need to honour a `max_chars` cap, so a long
/// source is never fully scanned per render.
fn collapse_control_chars(s: &str, max_chars: usize) -> String {
    s.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(max_chars + 1)
        .collect()
}

/// Single-line display form of a value: control characters (newlines, tabs)
/// collapsed to spaces, then truncated to [`MAX_TOOL_VALUE_CHARS`] chars at a
/// word/path-delimiter boundary with "…" when cut. The bool reports whether
/// the value was char-cut (drives the hover tooltip).
fn value_display(value: &str) -> (String, bool) {
    let single_line = collapse_control_chars(value, MAX_TOOL_VALUE_CHARS);
    let was_char_cut = single_line.chars().count() > MAX_TOOL_VALUE_CHARS;
    let display = truncate_at_boundary(&single_line, MAX_TOOL_VALUE_CHARS);
    (display, was_char_cut)
}

/// One-line args summary plus whether it omits any information (drives the
/// hover tooltip): a per-value char cut, the whole-line cap cut, or — in the
/// compact view — a hidden single argument name. Renders `name: value` pairs,
/// comma-separated, each value collapsed to a single line and truncated at a
/// word/path-delimiter boundary to [`MAX_TOOL_VALUE_CHARS`] chars. The whole
/// line is capped at [`MAX_TOOL_PAIRS_LINE_CHARS`] chars, cut at a pair
/// boundary with "…". In the compact view a single argument renders just the
/// value, omitting the argument name.
fn tool_pairs_summary(tool: &RunningTool, view: ToolBlockView) -> (String, bool) {
    let compact = view == ToolBlockView::Compact;
    if tool.args.is_empty() {
        return (String::new(), false);
    }
    if compact && tool.args.len() == 1 {
        let (display, _) = value_display(&tool.args[0].1);
        return (display, true);
    }
    let mut omits = false;
    let rendered: Vec<String> = tool
        .args
        .iter()
        .map(|(k, v)| {
            let (display, cut) = value_display(v);
            omits |= cut;
            format!("{k}: {display}")
        })
        .collect();
    let joined = rendered.join(", ");
    if joined.chars().count() <= MAX_TOOL_PAIRS_LINE_CHARS {
        return (joined, omits);
    }
    // Over budget: keep whole pairs while they fit (leaving room for the
    // trailing "…"), then mark the cut.
    let mut out = String::new();
    for pair in &rendered {
        let sep = if out.is_empty() { "" } else { ", " };
        let candidate_len = out.chars().count() + sep.chars().count() + pair.chars().count();
        if candidate_len + 1 > MAX_TOOL_PAIRS_LINE_CHARS {
            break;
        }
        out.push_str(sep);
        out.push_str(pair);
    }
    if out.is_empty() {
        // Even the first pair alone does not fit — truncate at a delimiter
        // boundary so the cut lands at a sensible wrap point.
        (
            truncate_at_boundary(&rendered[0], MAX_TOOL_PAIRS_LINE_CHARS),
            true,
        )
    } else {
        (format!("{out}…"), true)
    }
}

/// Render the hover tooltip: the tool name header (bold white) followed by
/// every argument pair on its own line, sorted by FULL value length ascending
/// (stable, so equal-length pairs keep registration order) — the shortest
/// pairs sit at the top and stay visible even when the longest values extend
/// beyond the viewport. Values are full and untruncated, including secrets.
fn tool_tooltip<Message: 'static>(tool: &RunningTool) -> Element<'static, Message> {
    let mut pairs = tool.args.clone();
    pairs.sort_by_key(|(_, v)| v.chars().count());

    let mut content = Column::new().spacing(theme::SPACE_2).push(
        text(tool.name.clone())
            .size(theme::TEXT_11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY),
    );
    for (k, v) in &pairs {
        content = content.push(
            text(format!("{k}: {v}"))
                .size(theme::TEXT_11)
                .font(crate::gui::JETBRAINS_MONO)
                .color(theme::TEXT_SECONDARY)
                .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
        );
    }
    container(content).max_width(MAX_TOOL_TOOLTIP_WIDTH).into()
}

/// Which session surface renders a tool block. `Compact` is the Running
/// Agents card line (plain text, glyph-wrapped so long unbroken values fold; a
/// single-argument tool renders just the value, omitting the argument name).
/// `Full` is the Sessions transcript bubble (selectable text, word/glyph
/// wrapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolBlockView {
    Compact,
    Full,
}

/// One shared tool block: bold tool name + one-line key-value args summary
/// (boundary-truncated), with a hover tooltip showing the tool name and every
/// full unscrubbed argument pair (sorted by value length ascending, shortest
/// at top). Values are RAW and unscrubbed in both views — a deliberate,
/// user-approved decision (local-first GUI; the durable stats logs stay
/// scrubbed).
///
/// The tooltip is only shown when the rendered line omits information — a
/// per-value cut, the whole-line cap cut, or a hidden single argument name in
/// the compact view. Zero-arg tools and fully visible full-view lines get no
/// tooltip.
pub(crate) fn tool_block<Message: 'static>(
    tool: &RunningTool,
    view: ToolBlockView,
) -> Element<'static, Message> {
    let name: Element<'static, Message> = match view {
        ToolBlockView::Full => widgets::selectable_text(tool.name.clone(), theme::TEXT_PRIMARY)
            .size(theme::TEXT_11)
            .font(theme::FONT_BOLD)
            .into(),
        ToolBlockView::Compact => text(tool.name.clone())
            .size(theme::TEXT_11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY)
            .into(),
    };

    let align_y = match view {
        ToolBlockView::Compact => Alignment::Center,
        ToolBlockView::Full => Alignment::Start,
    };
    let mut line = Row::new()
        .spacing(theme::SPACE_4)
        .align_y(align_y)
        .push(name);

    let (args_text, omits) = tool_pairs_summary(tool, view);
    if !args_text.is_empty() {
        let args: Element<'static, Message> = match view {
            ToolBlockView::Full => widgets::selectable_text(args_text, theme::TEXT_SECONDARY)
                .size(theme::TEXT_11)
                .width(Length::Fill)
                .wrapping(iced::widget::text::Wrapping::WordOrGlyph)
                .into(),
            // Wrap at glyph level so a long unbroken value (path, URL) folds
            // instead of overflowing the card edge.
            ToolBlockView::Compact => text(args_text)
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY)
                .wrapping(iced::widget::text::Wrapping::WordOrGlyph)
                .into(),
        };
        line = line.push(args);
    }

    if omits {
        tooltip(line, tool_tooltip(tool), tooltip::Position::Top)
            .gap(theme::SPACE_4)
            .style(theme::tooltip_style)
            .into()
    } else {
        line.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_tool(args: Vec<(String, String)>) -> RunningTool {
        RunningTool {
            name: "read_file".to_string(),
            args,
        }
    }

    /// Every cut happens at the last delimiter at or before the cap, falls back to a hard char cut, and reports omission.
    #[test]
    fn truncation_cases() {
        let boundary: &[(&str, &str, usize, &str)] = &[
            (
                "cut at a word boundary rather than mid-word",
                "abc def ghi",
                5,
                "abc…",
            ),
            (
                "a cap landing exactly before a delimiter keeps the full words",
                "abc def ghi",
                7,
                "abc def…",
            ),
            (
                "paths break at a `/` segment boundary",
                "/Users/egordezic/Desktop/foo.rs",
                26,
                "/Users/egordezic/Desktop/…",
            ),
            (
                "an unbroken token falls back to a hard char cut at the cap",
                "aaaaaaaa",
                4,
                "aaaa…",
            ),
            ("under the cap: unchanged", "short", 10, "short"),
        ];
        for &(name, input, cap, expected) in boundary {
            assert_eq!(
                truncate_at_boundary(input, cap),
                expected,
                "case {name}: truncation"
            );
        }

        let values: Vec<(&str, String, String, bool)> = vec![
            (
                "a path under the cap is unchanged",
                "/Users/egordezic/Desktop/foo.rs".to_string(),
                "/Users/egordezic/Desktop/foo.rs".to_string(),
                false,
            ),
            (
                "control chars collapse to spaces",
                "a\nb".to_string(),
                "a b".to_string(),
                false,
            ),
            (
                "an unbroken long token hard-cuts at MAX_TOOL_VALUE_CHARS",
                "a".repeat(110),
                format!("{}…", "a".repeat(MAX_TOOL_VALUE_CHARS)),
                true,
            ),
            (
                "a value over the cap cuts at a `/` delimiter boundary",
                format!("{}/{}", "a".repeat(60), "b".repeat(60)),
                format!("{}…", "a".repeat(60) + "/"),
                true,
            ),
        ];
        for (name, input, expected, expected_omits) in values {
            assert_eq!(
                value_display(&input),
                (expected, expected_omits),
                "case {name}: value display"
            );
        }

        let short = read_tool(vec![("path".to_string(), "a.rs".to_string())]);
        let full = tool_pairs_summary(&short, ToolBlockView::Full);
        assert_eq!(full, ("path: a.rs".to_string(), false));
        let compact = tool_pairs_summary(&short, ToolBlockView::Compact);
        assert_eq!(compact, ("a.rs".to_string(), true));

        let over_cap = read_tool(vec![("path".to_string(), "a".repeat(120))]);
        let (line, omits) = tool_pairs_summary(&over_cap, ToolBlockView::Full);
        assert!(omits);
        assert!(line.ends_with('…'));

        let empty = read_tool(Vec::new());
        let nothing = (String::new(), false);
        assert_eq!(tool_pairs_summary(&empty, ToolBlockView::Full), nothing);

        let mut args = Vec::new();
        for seg in ["aaa", "bbb", "ccc", "ddd", "eee", "fff"] {
            let path = format!("/Users/egordezic/Desktop/project/{seg}/foo_long.rs");
            args.push((format!("{seg}_key"), path));
        }
        let tool = read_tool(args);
        let (line, omits) = tool_pairs_summary(&tool, ToolBlockView::Full);
        assert!(line.chars().count() <= MAX_TOOL_PAIRS_LINE_CHARS);
        assert!(line.ends_with('…'));
        assert!(line.contains("aaa_key"));
        assert!(!line.contains("fff"), "excluded pair must not appear");
        assert!(omits, "line cut must report omission");

        let long_pair = read_tool(vec![("k".repeat(220), "v".to_string())]);
        let (line, omits) = tool_pairs_summary(&long_pair, ToolBlockView::Full);
        assert_eq!(line, format!("{}…", "k".repeat(MAX_TOOL_PAIRS_LINE_CHARS)));
        assert!(omits, "line cut must report omission");
    }
}

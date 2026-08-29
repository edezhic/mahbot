//! Shared rendering components for the agent-session ledger — the tool block,
//! one-line key-value pairs summary, and boundary-aware truncation used by
//! both the Sessions page (full view) and the Running Agents page (compact).

use iced::widget::{Column, Row, container, text, tooltip};
use iced::{Alignment, Element};

use iced_fonts::lucide;

use crate::agent::registry::RunningTool;
use crate::gui::theme;
use crate::gui::widgets;

/// Max display length (Unicode chars) of a single argument VALUE in the
/// one-line key-value pairs summary. Hover tooltip always shows full value.
const MAX_TOOL_VALUE_CHARS: usize = 52;

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

/// Single-line display form of a value: control characters (newlines, tabs)
/// collapsed to spaces, then truncated to [`MAX_TOOL_VALUE_CHARS`] chars at a
/// word/path-delimiter boundary with "…" when cut.
fn value_display(value: &str) -> String {
    let single_line: String = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    truncate_at_boundary(&single_line, MAX_TOOL_VALUE_CHARS)
}

/// Render the row's key-value pairs line: `name: value` pairs, comma-
/// separated, each value collapsed to a single line and truncated at a
/// word/path-delimiter boundary to [`MAX_TOOL_VALUE_CHARS`] chars. The whole
/// line is capped at [`MAX_TOOL_PAIRS_LINE_CHARS`] chars, cut at a pair
/// boundary with "…".
fn tool_pairs_line(pairs: &[(String, String)]) -> String {
    let rendered: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{k}: {}", value_display(v)))
        .collect();
    let joined = rendered.join(", ");
    if joined.chars().count() <= MAX_TOOL_PAIRS_LINE_CHARS {
        return joined;
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
        truncate_at_boundary(&rendered[0], MAX_TOOL_PAIRS_LINE_CHARS)
    } else {
        format!("{out}…")
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

    let mut content = Column::new().spacing(2).push(
        text(tool.name.clone())
            .size(11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY),
    );
    for (k, v) in &pairs {
        content = content.push(
            text(format!("{k}: {v}"))
                .size(11)
                .font(crate::gui::JETBRAINS_MONO)
                .color(theme::TEXT_SECONDARY)
                .wrapping(iced::widget::text::Wrapping::WordOrGlyph),
        );
    }
    container(content).max_width(MAX_TOOL_TOOLTIP_WIDTH).into()
}

/// One shared tool block: lucide `wrench` glyph + bold tool name + one-line
/// key-value args summary (boundary-truncated), with a hover tooltip showing
/// the tool name and every full unscrubbed argument pair (sorted by value
/// length ascending, shortest at top). Values are RAW and unscrubbed in both
/// views — a deliberate, user-approved decision (local-first GUI; the durable
/// stats logs stay scrubbed).
pub(crate) fn tool_block<Message: 'static>(
    tool: &RunningTool,
    selectable: bool,
) -> Element<'static, Message> {
    let icon: Element<'static, Message> = lucide::wrench::<iced::Theme, iced::Renderer>()
        .size(11)
        .color(theme::TEXT_MUTED)
        .into();

    let name: Element<'static, Message> = if selectable {
        widgets::selectable_text(tool.name.clone(), theme::TEXT_PRIMARY)
            .size(11)
            .font(theme::FONT_BOLD)
            .into()
    } else {
        text(tool.name.clone())
            .size(11)
            .font(theme::FONT_BOLD)
            .color(theme::TEXT_PRIMARY)
            .into()
    };

    let mut line = Row::new().spacing(4).align_y(Alignment::Center).push(icon);
    line = line.push(name);

    if !tool.args.is_empty() {
        let args: Element<'static, Message> = if selectable {
            widgets::selectable_text(tool_pairs_line(&tool.args), theme::TEXT_SECONDARY)
                .size(11)
                .into()
        } else {
            text(tool_pairs_line(&tool.args))
                .size(11)
                .color(theme::TEXT_SECONDARY)
                .into()
        };
        line = line.push(args);
    }

    tooltip(line, tool_tooltip(tool), tooltip::Position::Top)
        .gap(4)
        .style(theme::tooltip_style)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_at_boundary_cuts_at_word_and_path_delimiters() {
        // Cut at a word boundary rather than mid-word.
        assert_eq!(truncate_at_boundary("abc def ghi", 5), "abc…");
        // When the cap lands exactly before a delimiter, keep the full words.
        assert_eq!(truncate_at_boundary("abc def ghi", 7), "abc def…");
        // Paths break at a `/` segment boundary.
        assert_eq!(
            truncate_at_boundary("/Users/egordezic/Desktop/foo.rs", 26),
            "/Users/egordezic/Desktop/…"
        );
        // An unbroken token falls back to a hard char cut at the cap.
        assert_eq!(truncate_at_boundary("aaaaaaaa", 4), "aaaa…");
        // Under the cap: unchanged.
        assert_eq!(truncate_at_boundary("short", 10), "short");
    }

    #[test]
    fn value_display_truncates_at_delimiter_boundary() {
        // A path under the (doubled) cap is unchanged.
        assert_eq!(
            value_display("/Users/egordezic/Desktop/foo.rs"),
            "/Users/egordezic/Desktop/foo.rs"
        );
        // Control chars collapse to spaces.
        assert_eq!(value_display("a\nb"), "a b");
        // Unbroken long token hard-cuts at [`MAX_TOOL_VALUE_CHARS`].
        assert_eq!(
            value_display(&"a".repeat(60)),
            format!("{}…", "a".repeat(MAX_TOOL_VALUE_CHARS))
        );
        // A value over the cap cuts at a `/` delimiter boundary.
        let long_path = format!("{}/{}", "a".repeat(20), "b".repeat(40));
        assert_eq!(
            value_display(&long_path),
            format!("{}…", "a".repeat(20) + "/")
        );
    }

    #[test]
    fn tool_pairs_line_truncates_at_pair_boundary() {
        // Under the line cap: returned verbatim.
        let short = vec![("path".to_string(), "a.rs".to_string())];
        assert_eq!(tool_pairs_line(&short), "path: a.rs");

        // Over the line cap: whole pairs kept while they fit, then a trailing
        // "…" marks the cut; the excluded final pair never appears.
        let pairs = vec![
            (
                "path".to_string(),
                "/Users/egordezic/Desktop/project/aaa/foo_long.rs".to_string(),
            ),
            (
                "offset".to_string(),
                "/Users/egordezic/Desktop/project/bbb/bar_long.rs".to_string(),
            ),
            (
                "limit".to_string(),
                "/Users/egordezic/Desktop/project/ccc/baz_long.rs".to_string(),
            ),
            (
                "query".to_string(),
                "/Users/egordezic/Desktop/project/ddd/qux_long.rs".to_string(),
            ),
            (
                "sort".to_string(),
                "/Users/egordezic/Desktop/project/eee/quux_long.rs".to_string(),
            ),
            (
                "desc".to_string(),
                "/Users/egordezic/Desktop/project/fff/corge_long.rs".to_string(),
            ),
        ];
        let line = tool_pairs_line(&pairs);
        assert!(line.chars().count() <= MAX_TOOL_PAIRS_LINE_CHARS);
        assert!(line.ends_with('…'));
        assert!(line.contains("path"));
        assert!(!line.contains("desc"), "excluded pair must not appear");
    }

    #[test]
    fn tool_pairs_line_fallback_truncates_first_pair_at_boundary() {
        // A single pair longer than the line cap: when the first pair contains
        // no delimiter it is hard-cut exactly at the cap (with "…"), never left
        // overflowing the line.
        let pairs = vec![("k".repeat(220), "v".to_string())];
        let line = tool_pairs_line(&pairs);
        assert_eq!(line, format!("{}…", "k".repeat(MAX_TOOL_PAIRS_LINE_CHARS)));
    }
}

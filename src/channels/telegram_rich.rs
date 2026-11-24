//! A Telegram Bot API 10.1 `rich_message` flattened to plain text.
//!
//! A rich message carries no plain `text` at all: the article lives in `rich_message.blocks` as a
//! tree of typed blocks and rich-text runs, so a reader that only looks at `text` sees an empty
//! message. This module walks that tree in reading order and says what it holds.
//!
//! **Invariant:** nothing emitted here is ever mistaken for a delivered file. A picture, video or
//! document inside an article is described in plain words — `(picture: A cat)` — and never as the
//! product's `[IMAGE: …]` / `[FILE: …]` markers, because a marker downstream means a file that
//! arrived with the message, and an article's picture is not one.
//!
//! A block type this product does not render, and any malformed shape, is skipped silently: a
//! message must never fail to arrive because one block of it was written by a newer Bot API.

use crate::util::none_if_empty;
use serde_json::{Map, Value};

/// How deep a block tree may nest. The tree comes from a peer, so it is never trusted to be
/// shallow: past this a block is skipped, which bounds both the text produced and the recursion.
const MAX_DEPTH: usize = 16;

/// The message's article, flattened to plain text in reading order; `None` when the message
/// carries no `rich_message`, or when the article flattens to nothing but whitespace.
pub(crate) fn article_text(message: &Value) -> Option<String> {
    let blocks = message.get("rich_message")?.get("blocks")?.as_array()?;
    let text = flatten_blocks(blocks, 0);
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// A block list in array order, a blank line between blocks. `depth` is the list nesting level
/// the blocks sit at, which only a list block reads.
fn flatten_blocks(blocks: &[Value], depth: usize) -> String {
    blocks
        .iter()
        .filter_map(|block| flatten_block(block, depth))
        // Only the tail is trimmed: a leading indent is a nested list's nesting, not slack.
        .map(|text| text.trim_end().to_string())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// One block as text; `None` for a type this product does not render, or a block nested past
/// [`MAX_DEPTH`], so the caller skips it.
fn flatten_block(block: &Value, depth: usize) -> Option<String> {
    if depth > MAX_DEPTH {
        return None;
    }
    let fields = block.as_object()?;
    let kind = fields.get("type")?.as_str()?;
    let text = || fields.get("text").map_or_else(String::new, flatten_text);
    let blocks = || {
        fields
            .get("blocks")
            .and_then(Value::as_array)
            .map_or_else(String::new, |blocks| flatten_blocks(blocks, depth + 1))
    };
    Some(match kind {
        // A heading is its own line of plain words: no `#` or `**` is ever emitted.
        "paragraph" | "heading" | "pre" | "footer" => text(),
        // A divider is a visual break and an anchor is a jump target; neither says anything.
        "divider" | "anchor" => String::new(),
        "mathematical_expression" => string_field(fields, "expression"),
        "list" => fields
            .get("items")
            .and_then(Value::as_array)
            .map_or_else(String::new, |items| flatten_list(items, depth)),
        // `blockquote` and `expandable_blockquote` carry their body as blocks, `pullquote` as text;
        // whichever a quotation has is read, so no shape loses its body to the other's spelling.
        "blockquote" | "expandable_blockquote" | "pullquote" => {
            let body = if fields.contains_key("blocks") {
                blocks()
            } else {
                text()
            };
            with_note(&body, credit(fields))
        }
        "collage" | "slideshow" => with_note(&blocks(), caption_text(fields.get("caption"))),
        "table" => table_text(fields),
        "details" => {
            let summary = fields.get("summary").map_or_else(String::new, flatten_text);
            join_blocks(&summary, &blocks())
        }
        "map" => media_line("map", fields.get("caption")),
        "buttons" => buttons_line(fields.get("buttons")),
        "photo" => media_line("picture", fields.get("caption")),
        "video" => media_line("video", fields.get("caption")),
        "animation" => media_line("animation", fields.get("caption")),
        "audio" => media_line("audio", fields.get("caption")),
        "voice_note" => media_line("voice message", fields.get("caption")),
        "document" => media_line("document", fields.get("caption")),
        // `thinking` cannot arrive in a received message, and a block of an unknown type is not
        // ours to guess at.
        _ => return None,
    })
}

/// A block's plain string field, empty when absent.
fn string_field(fields: &Map<String, Value>, name: &str) -> String {
    fields
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The credit line a quotation carries, if any.
fn credit(fields: &Map<String, Value>) -> Option<String> {
    fields.get("credit").and_then(plain)
}

/// Body text with a note on a following line — a quotation's credit or a collage's caption.
fn with_note(body: &str, note: Option<String>) -> String {
    match note {
        Some(note) if body.trim().is_empty() => note,
        Some(note) => format!("{}\n{note}", body.trim_end()),
        None => body.to_string(),
    }
}

/// Two passages of one block — a details summary and its body — a blank line apart.
fn join_blocks(first: &str, second: &str) -> String {
    let (first, second) = (first.trim(), second.trim());
    match (first.is_empty(), second.is_empty()) {
        (true, _) => second.to_string(),
        (false, true) => first.to_string(),
        (false, false) => format!("{first}\n\n{second}"),
    }
}

/// A rich-text run as plain text: a string is itself, an array is its parts run together, an
/// object is its wrapper's text or its own payload.
fn flatten_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts.iter().map(flatten_text).collect(),
        Value::Object(fields) => flatten_text_object(fields),
        _ => String::new(),
    }
}

/// A typed rich-text object. Every wrapper (`bold`, `code`, `spoiler`, …) and every run that
/// carries its address in its own text (`mention`, `hashtag`, `date_time`, …) is just its text;
/// only the runs whose address lives elsewhere, or whose words live elsewhere, need more.
fn flatten_text_object(fields: &Map<String, Value>) -> String {
    let text = || fields.get("text").map_or_else(String::new, flatten_text);
    match fields.get("type").and_then(Value::as_str) {
        Some("url") => match fields.get("url").and_then(Value::as_str) {
            Some(url) if !url.is_empty() => link_text(&text(), url),
            _ => text(),
        },
        Some("custom_emoji") => string_field(fields, "alternative_text"),
        Some("mathematical_expression") => string_field(fields, "expression"),
        // An in-document anchor is a jump target, not an address and not a word.
        Some("anchor") => String::new(),
        Some("button") => fields.get("button").map_or_else(String::new, button_text),
        // An unknown type keeps its text rather than dropping the words of the article.
        _ => text(),
    }
}

/// A link's text with its address kept; a text that adds nothing to the address yields the
/// address alone.
fn link_text(text: &str, url: &str) -> String {
    if text.is_empty() || text == url {
        url.to_string()
    } else {
        format!("{text} ({url})")
    }
}

/// A button as text, with its address when it has one.
fn button_text(button: &Value) -> String {
    let text = button.get("text").map_or_else(String::new, flatten_text);
    match button.get("url").and_then(Value::as_str) {
        Some(url) if !url.is_empty() => link_text(&text, url),
        _ => text,
    }
}

/// A list block, one item per line, each at its nesting level's indent.
fn flatten_list(items: &[Value], depth: usize) -> String {
    items
        .iter()
        .filter_map(|item| flatten_item(item, depth))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One list item: its mark, then its own blocks — the first on the mark's line, the rest on
/// following lines, so a multi-block item never leaves its mark alone.
fn flatten_item(item: &Value, depth: usize) -> Option<String> {
    let fields = item.as_object()?;
    let indent = "  ".repeat(depth);
    let marker = list_marker(fields);
    let blocks: &[Value] = fields
        .get("blocks")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    let mut head: Option<String> = None;
    let mut tail: Vec<String> = Vec::new();
    for block in blocks {
        // A nested list is one level deeper; everything else stays at this item's level.
        let nested = block.get("type").and_then(Value::as_str) == Some("list");
        let Some(text) = flatten_block(block, depth + usize::from(nested)) else {
            continue;
        };
        let text = text.trim_end();
        if text.trim().is_empty() {
            continue;
        }
        if nested {
            tail.push(text.to_string());
            continue;
        }
        let text = text.trim_start().replace('\n', &format!("\n{indent}"));
        match head {
            None => head = Some(format!("{indent}{marker} {text}")),
            Some(_) => tail.push(format!("{indent}{text}")),
        }
    }
    let head = head.unwrap_or_else(|| format!("{indent}{marker}"));
    if tail.is_empty() {
        Some(head)
    } else {
        Some(format!("{head}\n{}", tail.join("\n\n")))
    }
}

/// The mark an item starts with: its checkbox, else the label the server supplied, else a dash.
fn list_marker(fields: &Map<String, Value>) -> String {
    if fields.get("has_checkbox").and_then(Value::as_bool) == Some(true) {
        let checked = fields.get("is_checked").and_then(Value::as_bool) == Some(true);
        return if checked { "[x]" } else { "[ ]" }.to_string();
    }
    match fields.get("label").and_then(Value::as_str).map(str::trim) {
        Some(label) if !label.is_empty() => label.to_string(),
        _ => "-".to_string(),
    }
}

/// A caption's text and credit, trimmed; both absent when the caption is.
fn caption_parts(caption: Option<&Value>) -> (Option<String>, Option<String>) {
    let Some(caption) = caption else {
        return (None, None);
    };
    // A `RichText` caption is the whole caption — a string, an array of runs, or a typed run; only
    // a `RichBlockCaption` object (the one rich-text shape with no `type`) splits into text and
    // credit.
    if !caption.is_object() || caption.get("type").is_some() {
        return (plain(caption), None);
    }
    (
        caption.get("text").and_then(plain),
        caption.get("credit").and_then(plain),
    )
}

/// A caption as one passage: its text, then its credit after an em dash.
fn caption_text(caption: Option<&Value>) -> Option<String> {
    match caption_parts(caption) {
        (Some(text), Some(credit)) => Some(format!("{text} — {credit}")),
        (text, credit) => text.or(credit),
    }
}

/// A rich value flattened and trimmed; `None` when it holds nothing.
fn plain(value: &Value) -> Option<String> {
    none_if_empty(flatten_text(value).trim())
}

/// A media block said in plain words where it stands — never as a delivered-file marker.
fn media_line(kind: &str, caption: Option<&Value>) -> String {
    match caption_parts(caption) {
        (Some(text), Some(credit)) => format!("({kind}: {text} — {credit})"),
        (Some(text), None) => format!("({kind}: {text})"),
        (None, Some(credit)) => format!("({kind} — {credit})"),
        (None, None) => format!("({kind})"),
    }
}

/// A buttons block: every button on one line, separated by a pipe.
fn buttons_line(buttons: Option<&Value>) -> String {
    buttons
        .and_then(Value::as_array)
        .map_or_else(String::new, |buttons| {
            buttons
                .iter()
                .map(button_text)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        })
}

/// A table row by row, each row on its own line, its caption after them.
fn table_text(fields: &Map<String, Value>) -> String {
    let mut lines: Vec<String> = fields
        .get("cells")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().map(table_row).collect())
        .unwrap_or_default();
    if let Some(caption) = caption_text(fields.get("caption")) {
        lines.push(caption);
    }
    lines.join("\n")
}

/// One table row: its cells joined with a pipe.
fn table_row(row: &Value) -> String {
    row.as_array().map_or_else(String::new, |cells| {
        cells
            .iter()
            .map(table_cell)
            .collect::<Vec<_>>()
            .join(" | ")
            .trim_end()
            .to_string()
    })
}

/// A cell's text; a cell without one is invisible and contributes nothing.
fn table_cell(cell: &Value) -> String {
    cell.get("text")
        .map_or_else(String::new, |text| flatten_text(text).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::{MAX_DEPTH, article_text};
    use serde_json::{Value, json};

    /// A message carrying exactly these blocks.
    fn message(blocks: &Value) -> Value {
        json!({ "rich_message": { "blocks": blocks } })
    }

    #[test]
    fn a_paragraph_is_its_text() {
        assert_eq!(
            article_text(&message(
                &json!([{"type": "paragraph", "text": "Hello there"}])
            ))
            .as_deref(),
            Some("Hello there")
        );
    }

    #[test]
    fn a_heading_is_its_own_line_with_no_markup() {
        let blocks = json!([
            {"type": "heading", "text": "Quarterly", "size": 2},
            {"type": "paragraph", "text": "Body"}
        ]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Quarterly\n\nBody")
        );
    }

    #[test]
    fn an_ordered_list_keeps_the_labels_the_server_sent() {
        let blocks = json!([{
            "type": "list",
            "items": [
                {"label": "1.", "blocks": [{"type": "paragraph", "text": "First"}]},
                {"label": "2.", "blocks": [{"type": "paragraph", "text": "Second"}]}
            ]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("1. First\n2. Second")
        );
    }

    #[test]
    fn a_nested_item_keeps_its_parent_on_the_mark_line() {
        let blocks = json!([{
            "type": "list",
            "items": [{
                "label": "•",
                "blocks": [
                    {"type": "paragraph", "text": "Parent"},
                    {"type": "list", "items": [
                        {"label": "•", "blocks": [{"type": "paragraph", "text": "Child"}]},
                        {"label": "•", "blocks": [{"type": "paragraph", "text": "Sibling"}]}
                    ]}
                ]
            }]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("• Parent\n  • Child\n  • Sibling")
        );
    }

    #[test]
    fn an_item_with_two_paragraphs_keeps_its_mark_with_the_first() {
        let blocks = json!([{
            "type": "list",
            "items": [{
                "label": "2.",
                "blocks": [
                    {"type": "paragraph", "text": "First thought"},
                    {"type": "paragraph", "text": "Second thought"}
                ]
            }]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("2. First thought\nSecond thought")
        );
    }

    #[test]
    fn a_task_list_shows_its_check_marks_instead_of_its_labels() {
        let blocks = json!([{
            "type": "list",
            "items": [
                {
                    "label": "•",
                    "has_checkbox": true,
                    "is_checked": true,
                    "blocks": [{"type": "paragraph", "text": "Done"}]
                },
                {
                    "label": "•",
                    "has_checkbox": true,
                    "is_checked": false,
                    "blocks": [{"type": "paragraph", "text": "Todo"}]
                }
            ]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("[x] Done\n[ ] Todo")
        );
    }

    #[test]
    fn a_blockquote_puts_its_credit_on_a_new_line() {
        let blocks = json!([{
            "type": "blockquote",
            "blocks": [{"type": "paragraph", "text": "Quoted"}],
            "credit": "The Author"
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Quoted\nThe Author")
        );
    }

    /// An expandable quotation carries its body as blocks, like a plain one; a pull quotation
    /// carries it as text. Neither may lose the body to the other's spelling.
    #[test]
    fn both_quotation_shapes_keep_their_body_and_credit() {
        let blocks = json!([
            {
                "type": "expandable_blockquote",
                "blocks": [{"type": "paragraph", "text": "Expandable body"}],
                "credit": "The Author"
            },
            {
                "type": "pullquote",
                "text": [{"type": "bold", "text": "Pulled"}, " body"],
                "credit": "The Author"
            }
        ]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Expandable body\nThe Author\n\nPulled body\nThe Author")
        );
    }

    #[test]
    fn a_table_is_rendered_row_by_row_with_its_caption_after() {
        let blocks = json!([{
            "type": "table",
            "cells": [
                [{"text": "Name", "is_header": true}, {"text": "Score"}],
                [{"text": "Ada"}, {"text": "9"}],
                [{"text": "Bob"}, {}]
            ],
            "caption": "Scores"
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Name | Score\nAda | 9\nBob |\nScores")
        );
    }

    #[test]
    fn a_pre_block_is_its_text_and_not_its_language() {
        let blocks = json!([{"type": "pre", "text": "fn main() {}", "language": "rust"}]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("fn main() {}")
        );
    }

    #[test]
    fn details_render_their_summary_then_their_blocks() {
        let blocks = json!([{
            "type": "details",
            "summary": "Why",
            "blocks": [{"type": "paragraph", "text": "Because"}],
            "is_open": false
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Why\n\nBecause")
        );
    }

    #[test]
    fn a_mathematical_expression_block_is_its_latex() {
        let blocks = json!([{"type": "mathematical_expression", "expression": "E = mc^2"}]);
        assert_eq!(article_text(&message(&blocks)).as_deref(), Some("E = mc^2"));
    }

    #[test]
    fn a_photo_says_picture_and_carries_its_caption_and_credit() {
        let photo =
            |caption: Value| message(&json!([{"type": "photo", "photo": [], "caption": caption}]));
        assert_eq!(
            article_text(&photo(json!({"text": "A cat"}))).as_deref(),
            Some("(picture: A cat)")
        );
        assert_eq!(
            article_text(&photo(json!({"text": "A cat", "credit": "The Author"}))).as_deref(),
            Some("(picture: A cat — The Author)")
        );
        assert_eq!(
            article_text(&photo(json!({"credit": "The Author"}))).as_deref(),
            Some("(picture — The Author)")
        );
        assert_eq!(
            article_text(&message(&json!([{"type": "photo", "photo": []}]))).as_deref(),
            Some("(picture)")
        );
    }

    #[test]
    fn every_media_block_says_what_it_is_in_plain_words() {
        let blocks = json!([
            {"type": "document", "document": {}, "caption": {"text": "invoice.pdf"}},
            {"type": "voice_note", "voice_note": {}},
            {"type": "video", "video": {}},
            {"type": "animation", "animation": {}},
            {"type": "audio", "audio": {}},
            {"type": "map", "location": {}}
        ]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some(
                "(document: invoice.pdf)\n\n(voice message)\n\n(video)\n\n(animation)\n\n(audio)\n\n(map)"
            )
        );
    }

    #[test]
    fn a_collage_flattens_its_blocks_then_its_caption_and_credit() {
        let blocks = json!([{
            "type": "collage",
            "blocks": [{"type": "photo"}, {"type": "video"}],
            "caption": {"text": "Trip", "credit": "Ada"}
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("(picture)\n\n(video)\nTrip — Ada")
        );
    }

    /// A table's caption is a bare rich text, which may be an array of runs; it is kept whole.
    #[test]
    fn a_table_caption_given_as_runs_is_kept() {
        let blocks = json!([{
            "type": "table",
            "cells": [[{"text": "Ada"}, {"text": "9"}]],
            "caption": [{"type": "italic", "text": "Scores"}, " of the class"]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Ada | 9\nScores of the class")
        );
    }

    #[test]
    fn a_link_keeps_its_address() {
        let blocks = json!([{
            "type": "paragraph",
            "text": [
                {"type": "bold", "text": "Read "},
                {"type": "url", "text": "the docs", "url": "https://example.com/docs"}
            ]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Read the docs (https://example.com/docs)")
        );
    }

    #[test]
    fn a_link_whose_text_is_its_address_is_the_address_alone() {
        let blocks = json!([{
            "type": "paragraph",
            "text": {"type": "url", "text": "https://example.com", "url": "https://example.com"}
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("https://example.com")
        );
    }

    #[test]
    fn a_custom_emoji_renders_its_alternative_text() {
        let blocks = json!([{
            "type": "paragraph",
            "text": {"type": "custom_emoji", "custom_emoji_id": "5368324170671202286", "alternative_text": "😺"}
        }]);
        assert_eq!(article_text(&message(&blocks)).as_deref(), Some("😺"));
    }

    #[test]
    fn a_buttons_block_joins_its_buttons_with_their_addresses() {
        let blocks = json!([{
            "type": "buttons",
            "buttons": [
                {"text": "Open", "url": "https://example.com"},
                {"text": "Call back", "callback_data": "cb"}
            ]
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Open (https://example.com) | Call back")
        );
    }

    #[test]
    fn a_button_inside_text_keeps_its_address() {
        let blocks = json!([{
            "type": "paragraph",
            "text": {"type": "button", "button": {"text": "Go", "url": "https://go.example"}}
        }]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("Go (https://go.example)")
        );
    }

    #[test]
    fn an_unknown_block_is_skipped_and_the_rest_still_arrives() {
        let blocks = json!([
            {"type": "paragraph", "text": "before"},
            {"type": "hologram", "text": "not ours to guess"},
            42,
            {"type": "paragraph"},
            {"type": "paragraph", "text": "after"}
        ]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some("before\n\nafter")
        );
    }

    #[test]
    fn a_deeply_nested_article_is_bounded() {
        // The tree comes from a peer and is never trusted to be shallow: past `MAX_DEPTH` a block
        // is skipped, so neither the text nor the recursion grows with the nesting.
        let mut block = json!({"type": "paragraph", "text": "bottom"});
        for _ in 0..(MAX_DEPTH * 8) {
            block = json!({"type": "details", "summary": "more", "blocks": [block]});
        }
        let text = article_text(&message(&json!([block]))).expect("the outer blocks still read");
        assert!(
            text.len() < 4096,
            "the cap must bound the text, got {} bytes",
            text.len()
        );
    }

    #[test]
    fn a_message_without_a_rich_message_has_no_article() {
        assert_eq!(
            article_text(&json!({"message_id": 1, "text": "plain"})),
            None
        );
    }

    #[test]
    fn a_whitespace_only_article_has_no_text() {
        assert_eq!(
            article_text(&message(&json!([{"type": "paragraph", "text": "  \n "}]))),
            None
        );
        assert_eq!(article_text(&message(&json!([]))), None);
    }

    #[test]
    fn a_full_report_reads_in_order() {
        let blocks = json!([
            {"type": "heading", "text": "Quarterly report", "size": 1},
            {"type": "paragraph", "text": "Revenue grew."},
            {"type": "list", "items": [
                {"label": "1.", "blocks": [{"type": "paragraph", "text": "Hire two engineers."}]},
                {"label": "2.", "blocks": [{"type": "paragraph", "text": "Ship the beta."}]}
            ]},
            {"type": "heading", "text": "Recommendation", "size": 2},
            {"type": "paragraph", "text": [
                {"type": "bold", "text": "Approve "},
                {"type": "url", "text": "the plan", "url": "https://example.com/plan"}
            ]},
            {"type": "heading", "text": "Criteria", "size": 3},
            {"type": "list", "items": [
                {
                    "has_checkbox": true,
                    "is_checked": true,
                    "blocks": [{"type": "paragraph", "text": "Cost"}]
                },
                {
                    "has_checkbox": true,
                    "is_checked": false,
                    "blocks": [{"type": "paragraph", "text": "Risk"}]
                }
            ]},
            {"type": "photo", "photo": [], "caption": {"text": "A cat"}}
        ]);
        assert_eq!(
            article_text(&message(&blocks)).as_deref(),
            Some(
                "Quarterly report\n\nRevenue grew.\n\n1. Hire two engineers.\n2. Ship the beta.\n\nRecommendation\n\nApprove the plan (https://example.com/plan)\n\nCriteria\n\n[x] Cost\n[ ] Risk\n\n(picture: A cat)"
            )
        );
    }
}

//! One line about what a Running Agents card's agent is doing now.
//!
//! The line follows the card's newest step: it is derived from that step's own
//! text — the intermediate text the model wrote for the reader when it has any,
//! otherwise that step's reasoning — and assembled from whole sentences within
//! the card's character budget, so it never reads as a mid-phrase cut. Markdown
//! syntax is stripped first; characters that belong to code (an underscore or
//! asterisk inside an identifier or expression, a `__dunder__` name) are not
//! markers and stay.

use crate::gui::markdown_breaks;

/// Character budget of the line: one record, ellipsis included. A longer line
/// still wraps, exactly as the text it replaced did.
const MAX_LINE_CHARS: usize = 200;

/// How much of a step's text is read when picking the line's sentences, and how
/// much of its reasoning becomes the hover text: reasoning reaches tens of
/// thousands of characters, and this runs per group on every render.
const MAX_SCAN_CHARS: usize = 2_000;

/// A sentence shorter than this carries nothing to read — a bare
/// acknowledgement or a fragment — and is skipped when the same block holds a
/// longer one.
const MIN_SENTENCE_CHARS: usize = 20;

/// Characters that may close a sentence behind its terminator.
const CLOSERS: [char; 7] = ['"', '\'', '”', '’', '»', ')', ']'];

/// One step's line and the hover text that explains it.
pub(crate) struct Line<'a> {
    pub(crate) text: String,
    /// The step's reasoning, cut to [`MAX_SCAN_CHARS`]: the hover text. A blank
    /// reasoning is no hover text at all.
    pub(crate) hover: Option<&'a str>,
}

/// The line of a step with the given own text: its narration (the intermediate
/// text the model wrote for the reader) when that yields a line, otherwise its
/// reasoning. `None` when the step carries no usable text at all — the caller
/// then keeps the line it already shows.
pub(crate) fn line_for_step<'a>(
    narration: &'a str,
    reasoning: Option<&'a str>,
) -> Option<Line<'a>> {
    // The whole block is never copied into the hover text: the tooltip shows the
    // same prefix the picker read.
    let hover = reasoning
        .map(bounded_prefix)
        .filter(|reasoning| !reasoning.trim().is_empty());
    let text = match pick_line(bounded_prefix(narration)) {
        // A narration that strips to nothing (a rule, an image marker, an
        // emoji) is not text the model wrote for the reader: use the step's
        // reasoning instead of holding an older line.
        Some(text) => text,
        None => pick_line(hover?)?,
    };
    Some(Line { text, hover })
}

/// The line for one block of step text, which the caller has already cut to
/// [`MAX_SCAN_CHARS`]: whole sentences packed up to [`MAX_LINE_CHARS`], a junk
/// sentence (see [`MIN_SENTENCE_CHARS`]) skipped in favour of the next one, and
/// — when no sentence qualifies — the block's last complete sentence. A block
/// with no sentence at all (a heading-like fragment) is shown as it is. Text
/// over the budget is the only text cut, and that cut lands on a word boundary
/// with "…".
fn pick_line(text: &str) -> Option<String> {
    let plain = plain_text(text);
    let (sentences, fragment) = split_sentences(&plain);
    let mut readable = sentences
        .iter()
        .copied()
        .filter(|s| s.chars().count() >= MIN_SENTENCE_CHARS);
    let line = match readable.next() {
        Some(first) if first.chars().count() > MAX_LINE_CHARS => cut_at_word(first),
        Some(first) => {
            let mut line = first.to_string();
            for next in readable {
                if line.chars().count() + 1 + next.chars().count() > MAX_LINE_CHARS {
                    break;
                }
                line.push(' ');
                line.push_str(next);
            }
            line
        }
        None => cut_at_word(sentences.last().copied().unwrap_or(fragment)),
    };
    (!line.is_empty() && line.chars().any(char::is_alphanumeric)).then_some(line)
}

/// The first [`MAX_SCAN_CHARS`] chars of a step's text.
fn bounded_prefix(text: &str) -> &str {
    match text.char_indices().nth(MAX_SCAN_CHARS) {
        Some((cut, _)) => &text[..cut],
        None => text,
    }
}

/// The text of a markdown block with its syntax stripped: heading, emphasis,
/// bullet, list-numbering and backtick markers are dropped by pulldown-cmark's
/// own inline rules (so an underscore inside an identifier or an asterisk in
/// an expression stays text), fenced and indented code blocks are dropped whole
/// (source code is not a status line), and block and line boundaries collapse
/// to single spaces.
fn plain_text(text: &str) -> String {
    use pulldown_cmark::{Event, Tag, TagEnd};
    let mut out = String::with_capacity(text.len());
    let mut boundary = false;
    let mut code_block = false;
    for (event, span) in
        pulldown_cmark::Parser::new_ext(text, markdown_breaks::iced_markdown_options())
            .into_offset_iter()
    {
        match event {
            Event::Start(Tag::CodeBlock(_)) => code_block = true,
            Event::End(TagEnd::CodeBlock) => code_block = false,
            // Code inside a fence is source, not prose.
            _ if code_block => {}
            Event::Text(text) | Event::Code(text) => {
                push_separator(&mut out, &mut boundary);
                out.push_str(&text);
            }
            // pulldown-cmark cannot tell `__init__` from `__strong__`, but the
            // span in the source can: an underscore pair around one unspaced
            // word is a dunder identifier and keeps its markers.
            Event::Start(Tag::Strong) | Event::End(TagEnd::Strong)
                if dunder_markers(&text[span.clone()]) =>
            {
                push_separator(&mut out, &mut boundary);
                out.push_str("__");
            }
            // An inline boundary is not a word break: `**parser**'s` and
            // `**fast**—always` are one word each.
            Event::Start(
                Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link { .. }
                | Tag::Image { .. },
            )
            | Event::End(
                TagEnd::Emphasis
                | TagEnd::Strong
                | TagEnd::Strikethrough
                | TagEnd::Link
                | TagEnd::Image,
            ) => {}
            _ => boundary = true,
        }
    }
    let mut plain = String::with_capacity(out.len());
    for word in out.split(is_separator) {
        let word = word.trim_start_matches(BULLETS);
        if word.is_empty() {
            continue;
        }
        if !plain.is_empty() {
            plain.push(' ');
        }
        plain.push_str(word);
    }
    plain
}

/// Whether a strong-emphasis span's source is `__` around one unspaced word —
/// an identifier (`__init__`, `__pycache__`) rather than a bold pair.
fn dunder_markers(span: &str) -> bool {
    span.strip_prefix("__")
        .and_then(|body| body.strip_suffix("__"))
        .is_some_and(|body| {
            !body.is_empty() && !body.contains(|c: char| c.is_whitespace() || c == '`')
        })
}

/// Start a new word: a pending block or line boundary becomes one space between
/// the text already written and whatever comes next.
fn push_separator(out: &mut String, boundary: &mut bool) {
    if std::mem::take(boundary) && !out.is_empty() {
        out.push(' ');
    }
}

/// Bullet glyphs a writer typed as decoration rather than a markdown list
/// marker, dropped from the head of a word.
const BULLETS: &[char] = &['•', '·', '‣', '∙', '▪', '●', '◦'];

/// Characters the line never shows: whitespace, control characters, and the
/// invisible format characters providers leak (byte-order marks, zero-width
/// spaces, bidi overrides, soft hyphens).
fn is_separator(c: char) -> bool {
    c.is_whitespace()
        || c.is_control()
        || matches!(
            c,
            '\u{00ad}' | '\u{200b}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{feff}'
        )
}

/// Sentence terminators: a full stop, a question or exclamation mark, or an
/// ellipsis.
fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// One-line plain text split into whole sentences (a terminator, any closing
/// quotes/parens, then whitespace or the end) and the trailing fragment that
/// carries no terminator.
fn split_sentences(text: &str) -> (Vec<&str>, &str) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if !is_terminator(chars[i].1) {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < chars.len() && is_terminator(chars[end].1) {
            end += 1;
        }
        while end < chars.len() && CLOSERS.contains(&chars[end].1) {
            end += 1;
        }
        // A terminator only ends a block before whitespace or at the end.
        let ends_block = end == chars.len() || chars[end].1.is_whitespace();
        if ends_block && ends_sentence(&chars, i) {
            let sentence = &text[byte_at(&chars, text, start)..byte_at(&chars, text, end)];
            sentences.push(sentence.trim());
            start = end;
        }
        i = end;
    }
    (sentences, text[byte_at(&chars, text, start)..].trim())
}

/// The byte offset of the `index`-th char of `text`, or the end of the text
/// when the index points past it.
fn byte_at(chars: &[(usize, char)], text: &str, index: usize) -> usize {
    chars.get(index).map_or(text.len(), |&(byte, _)| byte)
}

/// Whether the terminator at `i` really ends a sentence. A `.` inside a dotted
/// abbreviation (`e.g.`, `i.e.`) never does. A `.` behind a token that already
/// holds a dot — the end of a file name such as `running.rs` — does only when
/// the next word starts a new sentence or nothing follows, so `… running.rs
/// and …` keeps running while `… running.rs. Then …` and `… running.rs.` split.
/// A decimal (`.5`) is never a terminator, because a terminator is only
/// considered before whitespace or the end.
fn ends_sentence(chars: &[(usize, char)], i: usize) -> bool {
    if chars[i].1 != '.' {
        return true;
    }
    let mut start = i;
    while start > 0 && !chars[start - 1].1.is_whitespace() {
        start -= 1;
    }
    let token: String = chars[start..i].iter().map(|&(_, c)| c).collect();
    if is_dotted_abbreviation(&token) {
        return false;
    }
    if !token.contains('.') {
        return true;
    }
    // A file name closing the block (`… src/gui/running.rs.`) ends its
    // sentence like any other word; only a following word can tell the two
    // apart, so a terminator with nothing behind it is a boundary.
    let Some(next) = chars[i + 1..]
        .iter()
        .map(|&(_, c)| c)
        .find(|c| !c.is_whitespace() && !CLOSERS.contains(c))
    else {
        return true;
    };
    !next.is_lowercase()
}

/// A token of single letters joined by dots, read as an abbreviation
/// (`e.g`, `i.e`, `u.s`).
fn is_dotted_abbreviation(token: &str) -> bool {
    let mut letters = 0;
    let mut letter_next = true;
    for c in token.chars() {
        match (letter_next, c) {
            (true, _) if c.is_alphabetic() => letters += 1,
            (false, '.') => {}
            _ => return false,
        }
        letter_next = !letter_next;
    }
    letters >= 2 && !letter_next
}

/// A line trimmed to [`MAX_LINE_CHARS`]: unchanged when it fits, otherwise cut
/// at its last word boundary with "…". Whitespace is the only boundary reused
/// here — a `/`, `_` or `-` splits an identifier, which is the broken-word
/// artefact this line exists to avoid. A line with no whitespace at all (one
/// long URL or hash) is char-cut: the budget is a hard ceiling, so no boundary
/// can be honoured there.
fn cut_at_word(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= MAX_LINE_CHARS {
        return text.to_string();
    }
    let room = MAX_LINE_CHARS - 1;
    let mut boundary = None;
    let mut hard = text.len();
    for (index, (byte, c)) in text.char_indices().enumerate() {
        if index == room {
            hard = byte;
            break;
        }
        if c.is_whitespace() {
            boundary = Some(byte);
        }
    }
    let head =
        text[..boundary.unwrap_or(hard)].trim_end_matches([' ', ',', ';', ':', '…', '/', '_', '-']);
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRIFT: &str = "The parser struct is missing the offset field.";
    const FIELD: &str = "The parser needs the field.";
    const EDITING: &str = "Now editing the parser";
    const SPLIT_HEAD: &str = "I edited src/gui/running/mod.rs and left e.g. the parser.";
    const TAIL: &str = "Then I ran 0.5 of the tests.";

    fn line(narration: &str, reasoning: Option<&str>) -> String {
        line_for_step(narration, reasoning)
            .expect("the step carries text")
            .text
    }

    #[test]
    fn lines_pack_whole_sentences_within_the_budget() {
        // (case, narration, reasoning, the line it yields)
        let cases: &[(&str, &str, Option<&str>, &str)] = &[
            (
                "heading",
                "Now the `Mem` struct:",
                None,
                "Now the Mem struct:",
            ),
            (
                "bare acknowledgement skipped",
                "",
                Some("Good. The parser struct is missing the offset field, so the columns drift."),
                "The parser struct is missing the offset field, so the columns drift.",
            ),
            (
                "whole sentences packed",
                "",
                Some(
                    "The parser drifts on every column of the offset table. I will add the missing \
                     offset field to the struct definition. Then the whole test suite runs again to \
                     prove the columns line up. Finally the installer is rebuilt and shipped.",
                ),
                "The parser drifts on every column of the offset table. I will add the missing \
                 offset field to the struct definition. Then the whole test suite runs again to prove \
                 the columns line up.",
            ),
            (
                "file name ends a sentence",
                "",
                Some("All good. See src/gui/running/label.rs."),
                "See src/gui/running/label.rs.",
            ),
        ];
        for &(name, narration, reasoning, expected) in cases {
            assert_eq!(line(narration, reasoning), expected, "case {name}");
        }

        // Exactly the budget passes through; one char over costs the ellipsis.
        let exact = "a".repeat(MAX_LINE_CHARS);
        assert_eq!(line("", Some(&exact)), exact, "case exactly the budget");
        let over = "a".repeat(MAX_LINE_CHARS + 1);
        let clipped = format!("{}…", "a".repeat(MAX_LINE_CHARS - 1));
        assert_eq!(line("", Some(&over)), clipped, "case one over");
        // A long sentence is cut at its last whitespace: no word is broken, and the
        // prefix is verbatim.
        let text = "I'll restructure the installer so the archive is unpacked once and the binaries \
            are proven before the swap, which keeps a failed attempt from leaving a half-installed copy \
            behind, and the previous install stays intact until the new one is proven to run.";
        let cut = line("", Some(text));
        assert!(cut.ends_with('…'));
        assert!(cut.chars().count() <= MAX_LINE_CHARS, "the ellipsis counts");
        let kept = cut.trim_end_matches('…');
        assert!(text.starts_with(kept), "kept prefix verbatim");
        assert!(text[kept.len()..].starts_with(' '), "cut on a space");
    }

    #[test]
    fn markdown_markers_are_stripped_from_the_line() {
        // (case, input, plain text)
        let cases: &[(&str, &str, &str)] = &[
            (
                "identifiers keep markers",
                "Use `snake_case_name` and a * b, not **bold** or _slanted_",
                "Use snake_case_name and a * b, not bold or slanted",
            ),
            (
                "dunder is an identifier",
                "Edit __init__.py and `__pycache__` now",
                "Edit __init__.py and __pycache__ now",
            ),
            (
                "dunder opens a block",
                "# __init__\n\n__pycache__ loads it. Then done.",
                "__init__ __pycache__ loads it. Then done.",
            ),
            ("bold", "__a bold phrase__ here", "a bold phrase here"),
            (
                "boundaries keep words whole",
                "the **parser**'s state and **fast**—always a *b*",
                "the parser's state and fast—always a b",
            ),
        ];
        for &(name, input, expected) in cases {
            assert_eq!(plain_text(input), expected, "case {name}");
        }
    }

    #[test]
    fn block_debris_is_stripped_from_the_text() {
        // (case, input, plain text)
        let cases: &[(&str, &str, &str)] = &[
            (
                "bullet glyphs are debris",
                "• read the table\n• patch the struct\n\nThen done.",
                "read the table patch the struct Then done.",
            ),
            (
                "fenced code is source",
                "```rust\nlet field_1 = value_1 + offset;\n```\nThe parser drifts.",
                "The parser drifts.",
            ),
            (
                "indented code is source",
                "    let field = 1;\n\nThe parser drifts.",
                "The parser drifts.",
            ),
            (
                "heading and list debris",
                "## Next\n\n- **first** item\n- second item\n\n1. third",
                "Next first item second item third",
            ),
            (
                "invisible format chars",
                "The parser drifts.\u{feff} \u{200b}Then the tests.",
                "The parser drifts. Then the tests.",
            ),
            (
                "block boundaries are spaces",
                "# Fixing the parser\n\nThe struct is missing `offset`, so I'll add it.\n\n\
                 - read the table\n- patch the struct\n\nDone.",
                "Fixing the parser The struct is missing offset, so I'll add it. \
                 read the table patch the struct Done.",
            ),
            (
                "punctuation keeps its word",
                "Done.\n\n.env.example lists them.",
                "Done. .env.example lists them.",
            ),
        ];
        for &(name, input, expected) in cases {
            assert_eq!(plain_text(input), expected, "case {name}");
        }
    }

    type NarrationCase<'a> = (
        &'a str,
        &'a str,
        Option<&'a str>,
        Option<&'a str>,
        Option<&'a str>,
    );

    #[test]
    fn narration_leads_and_its_step_reasoning_is_the_hover_text() {
        // (case, narration, reasoning, line text, hover)
        let cases: &[NarrationCase] = &[
            (
                "narration leads",
                EDITING,
                Some(FIELD),
                Some(EDITING),
                Some(FIELD),
            ),
            (
                "stripped narration falls through",
                "---",
                Some(DRIFT),
                Some(DRIFT),
                Some(DRIFT),
            ),
            ("no text at all", "", None, None, None),
            ("whitespace only", "  \n\t", Some("   "), None, None),
        ];
        for &(name, narration, reasoning, text, hover) in cases {
            let step = line_for_step(narration, reasoning);
            let shown = step.as_ref().map(|l| l.text.as_str());
            assert_eq!(shown, text, "case {name}: text");
            assert_eq!(
                step.as_ref().and_then(|l| l.hover),
                hover,
                "case {name}: hover"
            );
        }

        // The hover is the scanned prefix, not the whole block.
        let long = DRIFT.to_string().repeat(1_000);
        let step = line_for_step("", Some(&long)).expect("the step carries text");
        assert!(step.hover.is_some_and(|t| t.len() <= MAX_SCAN_CHARS));

        // Abbreviations, file names and decimals do not split sentences.
        let text = format!("{SPLIT_HEAD} {TAIL}");
        let (sentences, fragment) = split_sentences(&text);
        assert_eq!(sentences, [SPLIT_HEAD, TAIL]);
        assert_eq!(fragment, "");
    }
}

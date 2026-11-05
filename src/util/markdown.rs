//! Markdown classification shared by the chat render and the delivery path: the parser options
//! iced's markdown view runs with, and the regions it draws as source code — where a media marker
//! is text the user reads, not an attachment.

use std::ops::Range;

/// Markdown options iced's `markdown::parse` runs with (iced_widget `src/markdown.rs`,
/// `parse_with`). Every classification the product makes — html escaping, soft breaks, code
/// regions — uses this exact set so it agrees with the real parse; keep in sync if iced's defaults
/// change.
pub(crate) fn iced_markdown_options() -> pulldown_cmark::Options {
    pulldown_cmark::Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS
        | pulldown_cmark::Options::ENABLE_TABLES
        | pulldown_cmark::Options::ENABLE_STRIKETHROUGH
        | pulldown_cmark::Options::ENABLE_TASKLISTS
}

/// Byte ranges of `text` the parser draws as source code: fenced and indented blocks, and inline
/// spans.
#[must_use]
pub(crate) fn code_ranges(text: &str) -> Vec<Range<usize>> {
    let mut code = Vec::new();
    let mut block: Option<usize> = None;
    for (event, range) in
        pulldown_cmark::Parser::new_ext(text, iced_markdown_options()).into_offset_iter()
    {
        match event {
            pulldown_cmark::Event::Code(_) => code.push(range),
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::CodeBlock(_)) => {
                block = Some(range.start);
            }
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                if let Some(start) = block.take() {
                    code.push(start..range.end);
                }
            }
            _ => {}
        }
    }
    code
}

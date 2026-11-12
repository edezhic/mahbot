//! Markdown classification behind the desktop's chat render: the parser options iced's markdown
//! view runs with, and the one pass that finds the regions it draws as source code — where a media
//! marker is text the user reads, not an attachment — and the markdown pictures it draws in place of
//! their syntax.

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

/// What one [`scan`] pass finds for the desktop's render: the code regions, and the markdown
/// pictures it draws as attachments instead of their syntax.
pub(crate) struct MarkdownScan {
    /// Byte ranges drawn as source code: fenced and indented blocks, and inline spans.
    pub code: Vec<Range<usize>>,
    /// Byte ranges of markdown pictures (`![alt](url)`), each with the destination as written.
    pub pictures: Vec<(Range<usize>, String)>,
}

/// One `into_offset_iter` pass over `text` with iced's parser options — the parser the markdown
/// view runs. It is the one classification of the code regions and pictures the desktop's render
/// consumes, and the only parse that render makes.
#[must_use]
pub(crate) fn scan(text: &str) -> MarkdownScan {
    let mut code = Vec::new();
    let mut pictures = Vec::new();
    let mut block: Option<usize> = None;
    let mut picture: Option<(usize, String)> = None;
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
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Image { dest_url, .. }) => {
                picture = Some((range.start, dest_url.into_string()));
            }
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Image) => {
                if let Some((start, dest)) = picture.take() {
                    pictures.push((start..range.end, dest));
                }
            }
            _ => {}
        }
    }
    MarkdownScan { code, pictures }
}

//! The vocabulary and shapes every document reader's output is built from: the
//! block headers and unit marks, the story names and the cell addresses shared by
//! the OOXML readers ([`crate::ooxml`]) and the legacy `.doc`/`.xls`/`.ppt` one
//! ([`crate::legacy`]).
//!
//! A reader's output is assembled from a handful of shapes — a header over
//! indented lines, a label over a run of text, a sheet or slide header carrying
//! its state, a cell address — and a mark or a name. What more than one reader
//! prints is stated once here, so two of them cannot name one thing two ways: the
//! Excel reader and the legacy one both print a sheet block, the Word reader and
//! the legacy one both print the stories a Word file carries, and every reader
//! that writes a spreadsheet cell address writes it through [`column_letters`].
//! The names only the Word reader prints — the story names a `.docx` and a `.doc`
//! reading share, and the block a text box prints as — sit beside them rather than
//! inside one of the two files that print them: the marks the kit's refusals match
//! are stated in `assets/docgen/rules.json`, and the Word reader's own tests pin
//! the list to what a reading shows.
//!
//! The PowerPoint marks — `(title)`, `(hidden slide)`, `(diagram text)`,
//! `(diagram text could not be read)`, `(no text)` — are deliberately *not*
//! here: they live in `assets/docgen/rules.json` because the document kit's
//! refusals name them too (see [`crate::docgen::ppt_marks`]), and `(no text)`
//! in particular must be read from there by whoever prints it. [`slide_header`]
//! composes the slide's number with that same hidden-slide mark, so it reads
//! the mark from the kit rather than spelling it here.

// ── Shared text shapes ──────────────────────────────────────────

/// `header` then one indented line per entry. Every reader builds its blocks
/// from this, so the text shapes stay one implementation.
pub(crate) fn text_lines(header: &str, lines: &[String]) -> String {
    let indented = lines
        .iter()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}\n{indented}")
}

/// [`text_lines`], or `header (marker)` when there is nothing to list — each
/// reader supplies the marker its own empty unit prints.
pub(crate) fn text_block(header: &str, empty_marker: &str, lines: &[String]) -> String {
    if lines.is_empty() {
        return format!("{header} {empty_marker}");
    }
    text_lines(header, lines)
}

/// `Label:` then each line of `text` indented by two spaces — the shape
/// [`text_lines`] gives the other two families, for a caller holding a block of
/// text as one string rather than as lines.
pub(crate) fn labeled_block(label: &str, text: &str) -> String {
    text_lines(&format!("{label}:"), &lines_of(text))
}

/// The lines of `text`, as [`text_lines`] takes them.
pub(crate) fn lines_of(text: &str) -> Vec<String> {
    text.lines().map(str::to_owned).collect()
}

// ── Column addresses ────────────────────────────────────────────

/// The spreadsheet column letters for a zero-based column index (`0` is `A`).
/// Every reader that writes a cell address writes it through this.
pub(crate) fn column_letters(mut index: u32) -> String {
    let mut letters: Vec<char> = Vec::new();
    loop {
        let digit = u32::from(b'A') + index % 26;
        letters.push(char::from_u32(digit).unwrap_or('A'));
        if index < 26 {
            break;
        }
        index = index / 26 - 1;
    }
    letters.reverse();
    letters.into_iter().collect()
}

/// The zero-based column index of a cell reference (`"B12"` -> `1`), or `None`
/// when it does not start with a column letter — the inverse of
/// [`column_letters`], so a reference one side writes is read back by one
/// implementation.
pub(crate) fn column_index(reference: &str) -> Option<u32> {
    let mut index = 0u32;
    let mut seen = false;
    for byte in reference.bytes() {
        if !byte.is_ascii_alphabetic() {
            break;
        }
        // An address past `u32` is clamped rather than worth a panic.
        index = index
            .saturating_mul(26)
            .saturating_add(u32::from(byte.to_ascii_uppercase() - b'A') + 1);
        seen = true;
    }
    // `then_some` would evaluate the subtraction even when no letter was seen,
    // which is a panic on a reference that starts with anything else (`1`, `$A1`).
    seen.then(|| index - 1)
}

// ── Sheet marks ─────────────────────────────────────────────────

/// The mark a sheet with no valued cell prints in place of its lines.
pub(crate) const NO_VALUES: &str = "(no values)";

/// A sheet's block header, marked when the workbook hides the sheet. The state a
/// workbook writes is one of two words, and a producer's own casing is no reason
/// to pass a hidden sheet off as an ordinary one. The legacy `.xls` reader
/// deliberately passes `None`: an old format is read for its text alone and takes
/// none of the newer arms' marks — not a hidden sheet, though the crate's own
/// `Sheet::hidden` is right there and that arm reads the sheet struct anyway — so
/// a legacy sheet is never marked (see [`crate::legacy`]).
pub(crate) fn sheet_header(name: &str, state: Option<&str>) -> String {
    match state.map(str::to_ascii_lowercase).as_deref() {
        Some("veryhidden") => format!("Sheet \"{name}\" (very hidden):"),
        Some("hidden") => format!("Sheet \"{name}\" (hidden):"),
        _ => format!("Sheet \"{name}\":"),
    }
}

// ── Slide marks ─────────────────────────────────────────────────

/// One slide's block header: `Slide <n>:`, plus ` (hidden slide)` when the
/// presentation hides it. The mark is the kit's own ([`crate::docgen::ppt_marks`])
/// and the label the kit's refusals name is the shared template
/// ([`crate::docgen::ppt_slide_labels`]), so the reader and the kit cannot name
/// either two ways.
pub(crate) fn slide_header(number: usize, hidden: bool) -> String {
    let mut header = slide_label(
        &crate::docgen::ppt_slide_labels().slide,
        &number.to_string(),
    );
    if hidden {
        header.push(' ');
        header.push_str(&crate::docgen::ppt_marks().hidden_slide);
    }
    header
}

/// The header of the block a slide's speaker notes print under: `Slide <n>
/// notes:`.
pub(crate) fn slide_notes_header(number: usize) -> String {
    slide_label(
        &crate::docgen::ppt_slide_labels().notes,
        &number.to_string(),
    )
}

/// `template` with its `{n}` replaced by `number` — the one place a slide's own
/// label is built from the shared template, for a reading and for the
/// model-facing descriptions alike. A description passes `<n>`, the way the rest
/// of that text spells a place a real value fills, so the prompt names the label
/// the reader prints without spelling it a second time.
pub(crate) fn slide_label(template: &str, number: &str) -> String {
    template.replace("{n}", number)
}

/// The mark a text box is printed as, by both Word readers: a `[text box]`
/// block in the enclosing flow.
pub(crate) const TEXT_BOX: &str = "[text box]";

// ── Word story names ────────────────────────────────────────────
//
// The names of the Word stories a reader prints. The `.docx` reader numbers one
// block per definition — `Footnote 1:` — and pairs a header or footer with the
// variants that use it — `Header (default):`; the legacy `.doc` reader is handed
// a whole story at once, so it prints the same name with no number.

/// A Word header part's block name.
pub(crate) const WORD_HEADER: &str = "Header";
/// A Word footer part's block name.
pub(crate) const WORD_FOOTER: &str = "Footer";
/// A `w:footnote` definition's block name.
pub(crate) const WORD_FOOTNOTE: &str = "Footnote";
/// A `w:endnote` definition's block name.
pub(crate) const WORD_ENDNOTE: &str = "Endnote";
/// A `w:comment` definition's block name.
pub(crate) const WORD_COMMENT: &str = "Comment";

// The names the legacy `.doc` reader prints. Its own reader hands it a whole
// story at once — every footnote in one, every header and every footer in one —
// so there is nothing to number and the same word is used in the plural; the
// header story holds every header and every footer, so it names both. These are
// the same names the `WORD_*` block above gives one definition at a time, and
// the colon a printed block header carries is added where the block is built.
/// The legacy `.doc` header/footer story's name.
pub(crate) const DOC_HEADERS_FOOTERS: &str = "Headers and footers";
/// The legacy `.doc` footnote story's name.
pub(crate) const DOC_FOOTNOTES: &str = "Footnotes";
/// The legacy `.doc` endnote story's name.
pub(crate) const DOC_ENDNOTES: &str = "Endnotes";
/// The legacy `.doc` comment story's name.
pub(crate) const DOC_COMMENTS: &str = "Comments";

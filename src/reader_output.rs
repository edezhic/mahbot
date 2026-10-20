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
//! What a reading found in a file and did not show is stated here as well
//! ([`Unshown`]): each kind's own words, one line per kind, so every reader's arm
//! and both delivery layers report a loss the same way, and
//! [`unshown_report_examples`] quotes a report line in the model-facing
//! description of a reading.
//! The names only the Word reader prints — the story names a `.docx` and a `.doc`
//! reading share, and the block a text box prints as — sit beside them rather than
//! inside one of the two files that print them: the marks the kit's refusals match
//! are stated in `assets/docgen/rules.json`, and the Word reader's own tests pin
//! the list to what a reading shows. The vocabulary a chart's content prints as —
//! its block header, its field labels, the annotations a value carries and the
//! marks standing in for values — is stated here as well, because all three OOXML
//! readers print it through one module and the model-facing description of a
//! reading names it.
//!
//! The words of the PowerPoint marks — `(title)`, `(hidden slide)`,
//! `(diagram text)`, `(diagram text could not be read)`, `(no text)` — are
//! stated in `assets/docgen/rules.json` rather than here, because the document
//! kit's refusals name them too (see [`crate::docgen::ppt_marks`]), and
//! `(no text)` in particular must be read from there by whoever prints it.
//! [`slide_header`] composes the slide's number with that same hidden-slide mark,
//! and [`ppt_description_substitutions`] binds those marks and the three slide
//! labels to the `{{ppt_…}}` keys the model-facing texts naming a reading's marks
//! spell.

use strum::{EnumCount, EnumIter, IntoEnumIterator};

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

/// The mark a value carries when it is the number the file stores rather than a
/// display of it: the cell's or the chart cache's number format was not read, is
/// not one this reader reproduces for this value, or shows nothing at all for it.
/// A spreadsheet cell and a chart's cached value carry the same one, so one
/// number is not spelled two ways in one answer.
pub(crate) const MARK_STORED: &str = " (stored number, format not shown)";

/// The line a chart sheet prints first — the tab is one chart over the whole
/// sheet rather than a grid of cells — with the chart's own content under it,
/// and the report naming the chart unless its content was described.
pub(crate) const CHART_SHEET: &str = "(chart sheet)";

/// A sheet's block header, marked when the workbook hides the sheet. The state a
/// workbook writes is one of two words, and a producer's own casing is no reason
/// to pass a hidden sheet off as an ordinary one. The legacy `.xls` reader
/// deliberately passes `None`: the old format takes none of the newer arms' marks
/// — not a hidden sheet, though the crate's own `Sheet::hidden` is right there and
/// that arm reads the sheet struct anyway — so a legacy sheet is never marked (see
/// [`crate::legacy`]).
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

/// The header of the block a slide's comments and their replies print under:
/// `Slide <n> comments:`.
pub(crate) fn slide_comments_header(number: usize) -> String {
    slide_label(
        &crate::docgen::ppt_slide_labels().comments,
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

/// The `{{ppt_…}}` pairs a model-facing text spells for a reading's marks: the
/// kit's five marks and the three slide labels, with `<n>` where a description
/// leaves the slide's number to the reader — a reading prints a real number
/// through [`slide_header`], [`slide_notes_header`] and [`slide_comments_header`]
/// instead. The chart keys stay bound in the `read` description alone, which is
/// their only consumer.
pub(crate) fn ppt_description_substitutions() -> Vec<(&'static str, String)> {
    let marks = crate::docgen::ppt_marks();
    let labels = crate::docgen::ppt_slide_labels();
    vec![
        ("{{ppt_title_mark}}", marks.title.clone()),
        ("{{ppt_hidden_slide_mark}}", marks.hidden_slide.clone()),
        ("{{ppt_diagram_text_mark}}", marks.diagram_text.clone()),
        (
            "{{ppt_diagram_text_lost_mark}}",
            marks.diagram_text_lost.clone(),
        ),
        ("{{ppt_no_text_mark}}", marks.no_text.clone()),
        ("{{ppt_slide_label}}", slide_label(&labels.slide, "<n>")),
        ("{{ppt_notes_label}}", slide_label(&labels.notes, "<n>")),
        (
            "{{ppt_comments_label}}",
            slide_label(&labels.comments, "<n>"),
        ),
    ]
}

/// The mark a text box is printed as, by both Word readers: a `[text box]`
/// block in the enclosing flow.
pub(crate) const TEXT_BOX: &str = "[text box]";

/// The mark that follows a Word link's own content: `(link: ADDRESS)`, or
/// `(link: internal)` for an anchor inside the same document. The address and
/// its closing `)` are the reader's; this is the opening the description names.
pub(crate) const DOC_LINK: &str = "(link: ";

// ── Chart marks ─────────────────────────────────────────────────
//
// The vocabulary a chart's content prints as, wherever a reading prints it: the
// three OOXML readers all print it through [`crate::ooxml::chart`], and the
// model-facing description of a reading names it from here (`src/tools/read.rs`),
// so the two cannot spell one mark two ways. A chart's own field labels are
// deliberately not patterns in the document kit's refusals (`docx_marks`):
// `Title:`, `Text:` and `Label:` are ordinary words a document's own text can
// hold. So is the bare `Chart:` of a part that names no kind, which would refuse
// any document's own sentence opening with it. The header a named kind prints,
// the series label (named or numbered), the annotation beside a value and the
// marks standing in for values are patterns.

/// The label a chart's own title prints under.
pub(crate) const CHART_TITLE: &str = "Title:";

/// The label an axis title prints under.
pub(crate) const CHART_TEXT: &str = "Text:";

/// The label a data label's own text prints under.
pub(crate) const CHART_LABEL: &str = "Label:";

/// The mark a series with no values in the chart part carries in place of them.
pub(crate) const CHART_NO_VALUES: &str = "(no values in the chart part)";

/// The mark a series whose values are only in the table the chart attaches
/// carries in place of them: that table is not opened by a reading.
pub(crate) const CHART_VALUES_ATTACHED: &str = "(values only in the attached table)";

/// The mark a chart part that cannot be read prints.
pub(crate) const CHART_UNREADABLE: &str = "(the chart part could not be read)";

/// The mark an extended (`chartEx`) chart part prints: its own schema is not
/// read here.
pub(crate) const CHART_EXTENDED: &str = "(an extended chart kind, not read)";

/// The header a chart's content prints under: `Chart (<kind>):`, or `Chart:` for
/// a part that names no kind. A model-facing description passes `<kind>`, the way
/// it passes `<n>` for a slide's number, so the header is stated once.
pub(crate) fn chart_header(kind: Option<&str>) -> String {
    kind.map_or_else(|| "Chart:".to_owned(), |kind| format!("Chart ({kind}):"))
}

/// The label a series' values print under: `Series "<name>":`, or the series'
/// number when the part gives it no name. A model-facing description passes
/// `<name>`, the way it passes `<n>` for a slide's number, so the label is
/// stated once.
pub(crate) fn chart_series(name: Option<&str>, index: usize) -> String {
    name.map_or_else(
        || format!("Series {}:", index + 1),
        |name| format!("Series \"{name}\":"),
    )
}

/// The annotation a bubble point's size prints beside its value.
pub(crate) fn chart_size(size: &str) -> String {
    format!("(size {size})")
}

/// How much of a bounded list a chart's lines leave out: `3 more value(s)`.
pub(crate) fn chart_excess(count: usize, noun: &str) -> String {
    format!("{count} more {noun}")
}

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

// ── The report of what a reading did not show ───────────────────

/// One kind of content a reading found in a file and did not show. The variants'
/// own order is the report's order: the document's structure first, the media a
/// package embeds after it, and the pages a reading could not deliver last, so two
/// documents holding the same kinds read the same way.
#[derive(Clone, Copy, EnumCount, EnumIter)]
pub(crate) enum UnshownKind {
    /// A chart part the package holds (`charts/chart<N>.xml`, the modern
    /// `chartEx<N>.xml` parts with it), whose content a reading did not describe.
    /// An ordinary chart's content is described where the chart sits; a chart
    /// whose part carries no cached values, cannot be read, or is an extended
    /// `chartEx` kind is what the report names.
    Chart,
    /// A sheet that is one chart over a whole tab rather than a grid of cells,
    /// whose chart the reading did not describe where it sits.
    ChartSheet,
    /// A SmartArt diagram, whose data parts no reading here reaches: the Word and
    /// Excel readers name them by count, and the PowerPoint one reads a slide's
    /// diagram text instead.
    Diagram,
    /// A part a package embeds as an object (an OLE object, an embedded
    /// workbook): what it holds is not read. Counted by the part the reading met,
    /// so an object a package stores as more than one part (an OLE object and the
    /// workbook beside it) is named by each.
    Object,
    /// An embedded vector drawing (EMF, WMF, SVG, PICT) among a package's media or
    /// among an old format's pictures.
    Drawing,
    /// Embedded video among a package's media, or among an old presentation's
    /// objects.
    Video,
    /// Embedded audio among a package's media, or among an old presentation's
    /// objects.
    Audio,
    /// Embedded media of a kind the reading cannot name.
    Media,
    /// A file attached to a PDF.
    Attachment,
    /// A PDF annotation holding media rather than a note in the text (a sound, a
    /// movie, a screen recording).
    MediaAnnotation,
    /// A page a reading could not deliver as an image — one its own page list
    /// does not hold, or one whose raster could not be built or written — whether
    /// or not the page's text was read; the text, where there was any, is
    /// delivered without the page.
    Page,
}

impl UnshownKind {
    /// What the report names this kind by, without the count in front of it.
    const fn what(self) -> &'static str {
        match self {
            Self::Chart => "chart(s)",
            Self::ChartSheet => "chart sheet(s)",
            Self::Diagram => "diagram(s)",
            Self::Object => "embedded object part(s)",
            Self::Drawing => "embedded drawing(s)",
            Self::Video => "embedded video(s)",
            Self::Audio => "embedded audio(s)",
            Self::Media => "embedded media file(s)",
            Self::Attachment => "attached file(s)",
            Self::MediaAnnotation => "media annotation(s)",
            Self::Page => "page(s)",
        }
    }
}

/// What a reading found in the file and did not show, counted per kind: the
/// report a conversion returns beside its text and its notes, so a loss is named
/// rather than left to be read as content that is not there.
///
/// A reader counts only what its own pass met and only what it really left out: a
/// kind at zero prints nothing, so the report never names content the file does
/// not hold. What no reader looks at — a PDF text page's vector drawings, or the
/// parts of a package that hold nothing a model reads (its themes, its document
/// properties) — is deliberately absent, and belongs in the model-facing
/// description of the reading instead: a kind here is a promise the reading
/// keeps. A PDF's own description is the exception: it *is* read (see
/// [`crate::pdf_marks`]), while a Word, Excel or PowerPoint package's document
/// properties are not. A package's styles part is read as far as a paragraph's
/// heading follows from it (see [`crate::ooxml::docx`]), and the rest of what it
/// states is formatting, not content a report could name.
#[derive(Default)]
pub(crate) struct Unshown {
    counts: [usize; UnshownKind::COUNT],
}

impl std::fmt::Debug for Unshown {
    /// The findings, not the counter array: what a reader put in the report is
    /// what a failing test wants to see.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.lines()).finish()
    }
}

impl Unshown {
    /// Note `count` more of one kind. A count of zero is not a finding.
    pub(crate) fn add(&mut self, kind: UnshownKind, count: usize) {
        self.counts[kind as usize] += count;
    }

    /// Fold another reading's findings in: the parts of one document are read by
    /// more than one pass (a PDF's pages by the arm, its annotations by
    /// [`crate::pdf_marks`]), and they report into the document's one report.
    pub(crate) fn merge(&mut self, other: &Self) {
        for (slot, count) in self.counts.iter_mut().zip(other.counts.iter()) {
            *slot += count;
        }
    }

    /// How much of one kind the report names.
    pub(crate) fn count(&self, kind: UnshownKind) -> usize {
        self.counts[kind as usize]
    }

    /// One line per kind that happened — `{count} {what} not shown` — in the
    /// kinds' own order rather than the order a reading met them, so two
    /// documents holding the same kinds read the same way.
    pub(crate) fn lines(&self) -> Vec<String> {
        UnshownKind::iter()
            .map(|kind| (kind, self.counts[kind as usize]))
            .filter(|(_, count)| *count > 0)
            .map(|(kind, count)| format!("{count} {} not shown", kind.what()))
            .collect()
    }
}

/// Two of the report's lines, as the model-facing description of a reading spells
/// them: built by the same code that prints a report, so a kind renamed here
/// cannot leave the description quoting a line no reading prints.
pub(crate) fn unshown_report_examples() -> String {
    let mut unshown = Unshown::default();
    unshown.add(UnshownKind::Chart, 1);
    unshown.add(UnshownKind::Video, 2);
    unshown
        .lines()
        .iter()
        .map(|line| format!("`{line}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

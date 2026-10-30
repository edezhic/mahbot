//! The content of a chart, as text, printed where the chart sits.
//!
//! A chart part carries the values its last rendering drew beside the table it
//! points at: a classic chart (`c:chartSpace`) caches each series' name, its
//! category labels and its numbers in `c:strCache`/`c:numCache`/`c:numLit`, with
//! the number format the value was shown under. This module reads that content —
//! the chart's kind, its title and other text, and one line per series of category
//! and value — so a reading tells a model what a chart says instead of only that
//! one is there. Nothing is drawn, and nothing beside the chart part is opened.
//!
//! # What is not described
//!
//! A chart part that carries only references to its attached table (no caches)
//! says per series that its values are in that table, and a part that cannot be
//! read says so: neither is passed off as a chart with no content, and both keep
//! their place in the package's report ([`crate::reader_output::Unshown`]). The
//! modern extended parts (`chartEx`) have a schema of their own — `cx:` rather
//! than `c:` — which this reading does not parse, so they are that same case.
//!
//! A chart whose values are printed is no longer a loss the report names; the
//! readers exclude every part this module describes from their census. The parts
//! read are the ones that census names — `charts/chart<N>.xml` and the
//! `chartEx<N>.xml` beside them — so a part a producer files under another name is
//! outside the reading and its report alike, as it was before this module.
//!
//! # Invariants
//!
//! - **A value is shown the way the reading shows a spreadsheet cell**: a cached
//!   point keeps the format its cache states, through the same formatter
//!   ([`numfmt`]), and a number under a format that cannot be reproduced is shown
//!   as stored, under the same mark ([`MARK_STORED`]), never invented.
//! - **Bounded rather than document-decided**: the walk keeps at most
//!   [`MAX_SERIES`] series and [`MAX_POINTS`] points of each list — the rest is
//!   counted, not allocated or formatted — and prints at most [`MAX_TEXTS`] of
//!   the chart's other texts, saying how much more it holds rather than cutting
//!   it off silently. At most [`MAX_CHARTS`] of a package's parts are read, and
//!   the ones left out stay in the report.
//! - **The notation the model reads is documented in `src/prompt/tool/read.md`**:
//!   the block's header, its field labels, the marks standing in for values and
//!   the annotations a value carries are stated once in [`crate::reader_output`],
//!   which that description and this module both take them from, and the counts
//!   standing in for what the bounds above leave out are what the description
//!   calls an excess counted.

use super::{
    Relationship, ZipEntry, append_entity, attr, is_chart_part, numfmt, read_zip_entry, rel_id,
    resolve_part, scan_elements,
};
use crate::reader_output::{
    CHART_EXTENDED, CHART_LABEL, CHART_NO_VALUES, CHART_TEXT, CHART_TITLE, CHART_UNREADABLE,
    CHART_VALUES_ATTACHED, MARK_STORED, chart_excess, chart_header, chart_series, chart_size,
};
use crate::util::one_line;
use quick_xml::Reader;
use quick_xml::events::{BytesRef, BytesStart, Event};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::{Read, Seek};
use zip::ZipArchive;

// ── Bounds ──────────────────────────────────────────────────────

/// How many chart parts of one package are read. A package that holds more keeps
/// the rest as a loss its census still names, so the number of parts a crafted
/// package carries cannot decide the reading's cost.
const MAX_CHARTS: usize = 64;
/// How many series of one chart are read and printed.
const MAX_SERIES: usize = 8;
/// How many points of one cached list are read and printed.
const MAX_POINTS: usize = 25;
/// How many of a chart's other texts — an axis title and a data label alike — are
/// printed, the two lists together.
const MAX_TEXTS: usize = 6;
/// How deep a part's elements may nest before it is read as unreadable: the walk
/// keeps one stack entry per level, so a crafted part must not decide its size.
const MAX_DEPTH: usize = 128;

// ── What a reading prints ───────────────────────────────────────

/// One chart part's content, as a reading prints it where the chart sits.
#[derive(Debug)]
pub(crate) enum ChartRead {
    /// The part is a chart this reading read: its lines say what it carries —
    /// its title, its other text and one line per series, ready to print.
    Content {
        /// The chart's kind, as the part states it (`column chart`, `pie chart`,
        /// `stacked line chart`), or `None` when the part names none.
        kind: Option<String>,
        /// The `Title: …`, `Text: …`, `Label: …` and `Series …` lines,
        /// unindented.
        body: Vec<String>,
        /// Whether the part carried the values its lines show. When it did not,
        /// each series' line says where its values are instead of printing
        /// them, and the chart stays in the report because what it says did not
        /// reach the answer.
        values: bool,
    },
    /// The part cannot be read, or is a kind whose own schema this reading does
    /// not parse.
    Unreadable(ChartLoss),
}

/// Why a chart part's content is not described.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChartLoss {
    /// The part is unreadable, cut short, nested past [`MAX_DEPTH`] or carries no
    /// chart element.
    Part,
    /// The part is a modern extended chart, whose `cx:` schema this reading does
    /// not parse.
    Extended,
}

impl ChartLoss {
    /// The mark this loss prints in place of the chart's content.
    const fn mark(self) -> &'static str {
        match self {
            Self::Part => CHART_UNREADABLE,
            Self::Extended => CHART_EXTENDED,
        }
    }
}

impl ChartRead {
    /// Whether the reading printed this chart's values — and so whether the chart
    /// is no longer a loss the package's report names.
    pub(crate) fn described(&self) -> bool {
        matches!(self, Self::Content { values: true, .. })
    }

    /// The chart's lines: a `Chart (<kind>):` header over its content indented by
    /// two spaces — or the one line naming why there is no content.
    pub(crate) fn lines(&self) -> Vec<String> {
        match self {
            Self::Content { kind, body, .. } => {
                let mut lines = vec![chart_header(kind.as_deref())];
                lines.extend(body.iter().map(|line| format!("  {line}")));
                lines
            }
            Self::Unreadable(loss) => vec![format!("{} {}", chart_header(None), loss.mark())],
        }
    }
}

// ── Reading the package's chart parts ───────────────────────────

/// Every chart part under `prefix` — the same parts the package's census counts —
/// read into the content a reading prints where the chart sits. `date1904` is the
/// workbook's date system, used for a part that does not state its own.
pub(crate) fn read_charts<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    date1904: bool,
) -> HashMap<String, ChartRead> {
    let parts: Vec<String> = archive
        .file_names()
        .filter(|name| is_chart_part(name, prefix))
        .take(MAX_CHARTS)
        .map(str::to_owned)
        .collect();
    parts
        .into_iter()
        .map(|part| {
            let read = match read_zip_entry(archive, &part) {
                ZipEntry::Bytes(xml) => read_part(&xml, date1904),
                ZipEntry::Missing | ZipEntry::Unreadable => ChartRead::Unreadable(ChartLoss::Part),
            };
            (part, read)
        })
        .collect()
}

/// The chart parts `xml` names through `rels`, resolved against the part's own
/// directory (`base`) — in the order the part draws them, and never the same part
/// twice. The reference is the `chart` element of a slide, a worksheet's drawing
/// or a Word body, whichever namespace wraps it.
pub(crate) fn chart_parts(xml: &[u8], rels: &[Relationship], base: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for id in scan_elements(xml, b"chart", rel_id).unwrap_or_default() {
        let Some(rel) = rels.iter().find(|rel| rel.id == id) else {
            continue;
        };
        let part = resolve_part(base, &rel.target);
        if !parts.contains(&part) {
            parts.push(part);
        }
    }
    parts
}

/// The content of every chart among `parts` the package holds — printed lines,
/// ready to place where the charts sit — and the parts whose content those lines
/// are, which the package's census leaves out. A caller that does not deliver its
/// block (a slide or a sheet it could not read) records none of them, so no chart
/// leaves both the reading and the report.
pub(crate) fn lines(
    charts: &HashMap<String, ChartRead>,
    parts: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut lines = Vec::new();
    let mut described = Vec::new();
    for part in parts {
        let Some(read) = charts.get(part) else {
            continue;
        };
        if read.described() {
            described.push(part.clone());
        }
        lines.extend(read.lines());
    }
    (lines, described)
}

// ── The part's own walk ─────────────────────────────────────────

/// Read one chart part's content. A part that is not a classic chart — an
/// extended one, or one with no chart space at all — is [`ChartRead::Unreadable`],
/// never a chart with no content.
fn read_part(xml: &[u8], date1904: bool) -> ChartRead {
    let mut parser = Parser::new(date1904);
    match parser.run(xml) {
        Ok(()) => parser.finish(),
        Err(loss) => ChartRead::Unreadable(loss),
    }
}

/// The cached points of one list: a series' values, one category level, or a
/// bubble's sizes.
#[derive(Default)]
struct Cache {
    /// Whether the points are numbers a format applies to (`numCache`/`numLit`).
    numeric: bool,
    /// The number format the cache states, when it states one.
    format: Option<String>,
    /// The points, by the index the part gives each.
    points: BTreeMap<usize, String>,
    /// How many points the cache held past [`MAX_POINTS`]: counted, never kept.
    more: usize,
}

/// One series while its element is open.
#[derive(Default)]
struct Series {
    /// The series' name, when the part caches or states one.
    name: Option<String>,
    /// Its category levels: one cache, or one per level of a multi-level list.
    categories: Vec<Cache>,
    /// Its values.
    values: Vec<Cache>,
    /// A bubble chart's sizes, aligned with the values by index.
    sizes: Vec<Cache>,
}

impl Series {
    fn slot(&mut self, slot: Slot) -> &mut Vec<Cache> {
        match slot {
            Slot::Category => &mut self.categories,
            Slot::Value => &mut self.values,
            Slot::Size => &mut self.sizes,
        }
    }
}

/// What a cached list being read belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Category,
    Value,
    Size,
}

/// A run of text being collected: the chart's title, an axis title or a data
/// label.
struct Capture {
    /// The stack depth the element was pushed at, so its end is recognised.
    depth: usize,
    kind: CaptureKind,
    text: String,
}

/// What a collected run of text is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureKind {
    /// The chart's own title.
    Title,
    /// An axis title — the chart's own text, printed with the title.
    Axis,
    /// A data label's own text.
    Label,
}

/// The chart type element being read and the qualifiers its own children state.
struct KindBuilder {
    depth: usize,
    kind: Kind,
    /// A bar chart's direction (`bar` is horizontal, `col` vertical).
    bar_dir: Option<String>,
    /// A bar, line or area chart's grouping (`stacked`, `percentStacked`).
    grouping: Option<String>,
    /// A pie-of-pie chart's second form (`pie` or `bar`).
    of_pie: Option<String>,
}

impl KindBuilder {
    /// The kind as a reading names it: the grouping leads, then the chart's form.
    fn name(&self) -> String {
        let grouping = match self.grouping.as_deref() {
            Some("stacked") => "stacked ",
            Some("percentStacked") => "100% stacked ",
            _ => "",
        };
        match self.kind {
            Kind::Area => format!("{grouping}area chart"),
            Kind::Area3D => format!("{grouping}3-D area chart"),
            Kind::Bar => {
                let bars = if self.bar_dir.as_deref() == Some("bar") {
                    "bar"
                } else {
                    "column"
                };
                format!("{grouping}{bars} chart")
            }
            Kind::Bar3D => {
                let bars = if self.bar_dir.as_deref() == Some("bar") {
                    "bar"
                } else {
                    "column"
                };
                format!("{grouping}3-D {bars} chart")
            }
            Kind::Bubble => "bubble chart".to_owned(),
            Kind::Doughnut => "doughnut chart".to_owned(),
            Kind::Line => format!("{grouping}line chart"),
            Kind::Line3D => format!("{grouping}3-D line chart"),
            Kind::OfPie if self.of_pie.as_deref() == Some("bar") => "bar-of-pie chart".to_owned(),
            Kind::OfPie => "pie-of-pie chart".to_owned(),
            Kind::Pie => "pie chart".to_owned(),
            Kind::Pie3D => "3-D pie chart".to_owned(),
            Kind::Radar => "radar chart".to_owned(),
            Kind::Scatter => "scatter chart".to_owned(),
            Kind::Stock => "stock chart".to_owned(),
            Kind::Surface => "surface chart".to_owned(),
            Kind::Surface3D => "3-D surface chart".to_owned(),
        }
    }
}

/// The classic chart kinds, each named by the element that declares it.
#[derive(Clone, Copy)]
enum Kind {
    Area,
    Area3D,
    Bar,
    Bar3D,
    Bubble,
    Doughnut,
    Line,
    Line3D,
    OfPie,
    Pie,
    Pie3D,
    Radar,
    Scatter,
    Stock,
    Surface,
    Surface3D,
}

/// [`Kind`] for a plot area element's local name.
fn kind_of(name: &[u8]) -> Option<Kind> {
    Some(match name {
        b"areaChart" => Kind::Area,
        b"area3DChart" => Kind::Area3D,
        b"barChart" => Kind::Bar,
        b"bar3DChart" => Kind::Bar3D,
        b"bubbleChart" => Kind::Bubble,
        b"doughnutChart" => Kind::Doughnut,
        b"lineChart" => Kind::Line,
        b"line3DChart" => Kind::Line3D,
        b"ofPieChart" => Kind::OfPie,
        b"pieChart" => Kind::Pie,
        b"pie3DChart" => Kind::Pie3D,
        b"radarChart" => Kind::Radar,
        b"scatterChart" => Kind::Scatter,
        b"stockChart" => Kind::Stock,
        b"surfaceChart" => Kind::Surface,
        b"surface3DChart" => Kind::Surface3D,
        _ => return None,
    })
}

/// The state of one chart part's walk.
struct Parser {
    /// The workbook's date system, for a part that does not state its own.
    date1904: bool,
    /// The part's own date system, when it states one.
    part_date1904: Option<bool>,
    /// Whether the part's root was a chart space — a part that is no chart at all
    /// is not a chart with no values.
    root: bool,
    /// The open elements' local names, outermost first.
    stack: Vec<Vec<u8>>,
    /// The chart kinds the plot area declares, in part order and each named once:
    /// a kind is collapsed as it is read, so a part repeating one element costs
    /// nothing.
    kinds: Vec<String>,
    /// The chart type element being read.
    kind: Option<KindBuilder>,
    /// The title or other text being read.
    capture: Option<Capture>,
    /// The chart's own title, once read.
    title: Option<String>,
    /// The chart's axis titles, in part order.
    texts: Vec<String>,
    /// The chart's data label texts, in part order.
    labels: Vec<String>,
    /// How many of the chart's texts were past their cap.
    more_texts: usize,
    /// The series whose element is open.
    series: Option<Series>,
    /// The depth the open series element was pushed at.
    series_depth: Option<usize>,
    /// The series completed, in part order — at most [`MAX_SERIES`] of them.
    series_done: Vec<Series>,
    /// How many series the part held past [`MAX_SERIES`]: counted, never kept.
    more_series: usize,
    /// The series slot whose caches are being read, and the depth its element was
    /// pushed at.
    slot: Option<(Slot, usize)>,
    /// The point being read, and the depth its element was pushed at.
    point: Option<(usize, usize)>,
    /// The depth a series' own name element was pushed at.
    tx_depth: Option<usize>,
    /// Whether the part attaches the table its values live in.
    attached: bool,
}

impl Parser {
    fn new(date1904: bool) -> Self {
        Self {
            date1904,
            part_date1904: None,
            root: false,
            stack: Vec::new(),
            kinds: Vec::new(),
            kind: None,
            capture: None,
            title: None,
            texts: Vec::new(),
            labels: Vec::new(),
            more_texts: 0,
            series: None,
            series_depth: None,
            series_done: Vec::new(),
            more_series: 0,
            slot: None,
            point: None,
            tx_depth: None,
            attached: false,
        }
    }

    fn run(&mut self, xml: &[u8]) -> Result<(), ChartLoss> {
        let mut reader = Reader::from_reader(xml);
        let mut buffer = Vec::new();
        loop {
            match reader.read_event_into(&mut buffer) {
                Ok(Event::Start(event)) => {
                    let name = event.local_name().as_ref().to_vec();
                    self.stack.push(name.clone());
                    self.start(&event, &name)?;
                }
                Ok(Event::Empty(event)) => {
                    let name = event.local_name().as_ref().to_vec();
                    self.stack.push(name.clone());
                    self.start(&event, &name)?;
                    self.stack.pop();
                    self.end();
                }
                Ok(Event::Text(event)) => {
                    if let Ok(text) = event.xml10_content() {
                        self.text(&text);
                    }
                }
                Ok(Event::CData(event)) => {
                    if let Ok(text) = event.xml10_content() {
                        self.text(&text);
                    }
                }
                Ok(Event::GeneralRef(reference)) => self.entity(&reference),
                Ok(Event::End(_)) => {
                    self.stack.pop();
                    self.end();
                }
                Ok(Event::Eof) => break,
                Err(_) => return Err(ChartLoss::Part),
                Ok(_) => {}
            }
            buffer.clear();
        }
        // A part that ended inside an element was cut short, and what it did not
        // reach is a loss rather than a chart with nothing else.
        if !self.root || !self.stack.is_empty() {
            return Err(ChartLoss::Part);
        }
        Ok(())
    }

    /// The local name of the element `back` levels up (`2` is the parent of the
    /// innermost open element).
    fn ancestor(&self, back: usize) -> Option<&[u8]> {
        let index = self.stack.len().checked_sub(back)?;
        self.stack.get(index).map(Vec::as_slice)
    }

    fn start(&mut self, event: &BytesStart<'_>, name: &[u8]) -> Result<(), ChartLoss> {
        let depth = self.stack.len();
        if depth > MAX_DEPTH {
            return Err(ChartLoss::Part);
        }
        if depth == 1 {
            if name != b"chartSpace" {
                return Err(ChartLoss::Part);
            }
            // An extended chart's own root: a schema this reading does not parse,
            // said plainly rather than read as a chart with none.
            if event.name().prefix().is_some_and(|p| p.as_ref() == b"cx") {
                return Err(ChartLoss::Extended);
            }
            self.root = true;
        }
        match name {
            b"chartData" => return Err(ChartLoss::Extended),
            b"title" => {
                let kind = if self.ancestor(2) == Some(b"chart") {
                    CaptureKind::Title
                } else {
                    CaptureKind::Axis
                };
                if self.capture.is_none() {
                    self.capture = Some(Capture {
                        depth,
                        kind,
                        text: String::new(),
                    });
                }
            }
            // A paragraph or a line break inside a captured run starts a new
            // line of it: rich text is joined with a space, never glued.
            b"p" | b"br" => self.separate(),
            b"ser" if self.series.is_none() => {
                if self.series_done.len() < MAX_SERIES {
                    self.series = Some(Series::default());
                    self.series_depth = Some(depth);
                } else {
                    self.more_series += 1;
                }
            }
            b"tx" => match self.ancestor(2) {
                Some(b"ser") => self.tx_depth = Some(depth),
                Some(b"dLbl") if self.capture.is_none() => {
                    self.capture = Some(Capture {
                        depth,
                        kind: CaptureKind::Label,
                        text: String::new(),
                    });
                }
                _ => {}
            },
            b"cat" | b"val" | b"xVal" | b"yVal" | b"bubbleSize" => {
                if self.series.is_some() {
                    let slot = match name {
                        b"cat" | b"xVal" => Slot::Category,
                        b"val" | b"yVal" => Slot::Value,
                        _ => Slot::Size,
                    };
                    self.slot = Some((slot, depth));
                }
            }
            b"numCache" | b"strCache" | b"numLit" | b"strLit" => {
                self.open_cache(matches!(name, b"numCache" | b"numLit"));
            }
            // A multi-level category list states one cache per level.
            b"lvl" => self.open_cache(false),
            b"pt" => {
                if self.slot.is_some()
                    && let Some(index) = attr(event, b"idx").and_then(|idx| idx.trim().parse().ok())
                {
                    self.keep_point(index, depth);
                }
            }
            b"externalData" => self.attached = true,
            b"date1904" => {
                self.part_date1904 =
                    Some(matches!(attr(event, b"val").as_deref(), Some("1" | "true")));
            }
            // A chart type's own qualifiers are attributes of its child elements.
            b"barDir" | b"grouping" | b"ofPieType" => {
                if let Some(kind) = self.kind.as_mut() {
                    let value = attr(event, b"val");
                    match name {
                        b"barDir" => kind.bar_dir = value,
                        b"grouping" => kind.grouping = value,
                        _ => kind.of_pie = value,
                    }
                }
            }
            _ => {
                if self.ancestor(2) == Some(b"plotArea")
                    && let Some(kind) = kind_of(name)
                {
                    self.kind = Some(KindBuilder {
                        depth,
                        kind,
                        bar_dir: None,
                        grouping: None,
                        of_pie: None,
                    });
                }
            }
        }
        Ok(())
    }

    /// Open the next cache of the series slot being read; a cache outside one (a
    /// series' own name, a title's) is not a value this reading prints.
    fn open_cache(&mut self, numeric: bool) {
        let Some((slot, _)) = self.slot else {
            return;
        };
        let Some(series) = self.series.as_mut() else {
            return;
        };
        series.slot(slot).push(Cache {
            numeric,
            ..Cache::default()
        });
    }

    /// Keep the point at `index` of the cache now being read — unless the cache
    /// already holds [`MAX_POINTS`] of them, which the cache counts instead of
    /// allocating and formatting. A point past the cap is read but not kept, so
    /// its value reaches nothing.
    fn keep_point(&mut self, index: usize, depth: usize) {
        let Some((slot, _)) = self.slot else {
            return;
        };
        let kept = match self
            .series
            .as_mut()
            .and_then(|series| series.slot(slot).last_mut())
        {
            Some(cache) if cache.points.contains_key(&index) || cache.points.len() < MAX_POINTS => {
                true
            }
            Some(cache) => {
                cache.more += 1;
                false
            }
            None => false,
        };
        if kept {
            self.point = Some((index, depth));
        }
    }

    /// Start a new line of a captured run of text: a rich-text paragraph and a
    /// line break both separate what came before from what follows.
    fn separate(&mut self) {
        if let Some(capture) = self.capture.as_mut() {
            push_break(&mut capture.text);
        } else if self.tx_depth.is_some()
            && let Some(series) = self.series.as_mut()
            && let Some(name) = series.name.as_mut()
        {
            push_break(name);
        }
    }

    /// The string the text at the current position is collected into, if any.
    fn target(&mut self) -> Option<&mut String> {
        let depth = self.stack.len();
        let leaf = self.stack.last()?.as_slice();
        // A title, an axis title or a data label: its rich text runs and its
        // cached string, never the reference it points at.
        if let Some(capture) = self.capture.as_mut()
            && capture.depth < depth
        {
            return matches!(leaf, b"t" | b"v").then_some(&mut capture.text);
        }
        if let Some(tx) = self.tx_depth
            && tx < depth
        {
            if !matches!(leaf, b"t" | b"v") {
                return None;
            }
            let series = self.series.as_mut()?;
            return Some(series.name.get_or_insert_with(String::new));
        }
        if leaf == b"v"
            && let (Some((index, _)), Some((slot, _))) = (self.point, self.slot)
        {
            let series = self.series.as_mut()?;
            let cache = series.slot(slot).last_mut()?;
            return Some(cache.points.entry(index).or_default());
        }
        match leaf {
            b"formatCode" => {
                let (slot, _) = self.slot?;
                let series = self.series.as_mut()?;
                let cache = series.slot(slot).last_mut()?;
                Some(cache.format.get_or_insert_with(String::new))
            }
            _ => None,
        }
    }

    fn text(&mut self, text: &str) {
        if let Some(target) = self.target() {
            target.push_str(text);
        }
    }

    /// File one finished run of text: the title, an axis title or a data label.
    /// The last two are bounded together, so a label per point cannot crowd the
    /// axis titles out or grow the description without limit.
    fn text_read(&mut self, kind: CaptureKind, text: String) {
        if kind == CaptureKind::Title {
            self.title = Some(text);
            return;
        }
        if self.texts.len() + self.labels.len() >= MAX_TEXTS {
            self.more_texts += 1;
            return;
        }
        if kind == CaptureKind::Axis {
            self.texts.push(text);
        } else {
            self.labels.push(text);
        }
    }

    fn entity(&mut self, reference: &BytesRef<'_>) {
        if let Some(target) = self.target() {
            append_entity(target, reference);
        }
    }

    fn end(&mut self) {
        let depth = self.stack.len();
        if self
            .capture
            .as_ref()
            .is_some_and(|capture| capture.depth == depth + 1)
            && let Some(capture) = self.capture.take()
        {
            let text = one_line(capture.text.trim().to_owned());
            if !text.is_empty() {
                self.text_read(capture.kind, text);
            }
        }
        if self.tx_depth == Some(depth + 1) {
            self.tx_depth = None;
        }
        if self
            .slot
            .as_ref()
            .is_some_and(|(_, slot_depth)| *slot_depth == depth + 1)
        {
            self.slot = None;
        }
        if self
            .point
            .as_ref()
            .is_some_and(|(_, point_depth)| *point_depth == depth + 1)
        {
            self.point = None;
        }
        if self
            .kind
            .as_ref()
            .is_some_and(|kind| kind.depth == depth + 1)
            && let Some(kind) = self.kind.take()
        {
            let name = kind.name();
            if !self.kinds.contains(&name) {
                self.kinds.push(name);
            }
        }
        if self.series_depth == Some(depth + 1) {
            self.series_depth = None;
            if let Some(mut series) = self.series.take() {
                series.name = series.name.map(one_line).filter(|name| !name.is_empty());
                self.series_done.push(series);
            }
        }
    }

    /// The part's content: the header's kind and the lines under it — its title,
    /// its other text and one line per series, or the mark standing in for the
    /// values a series does not carry.
    fn finish(self) -> ChartRead {
        let date1904 = self.part_date1904.unwrap_or(self.date1904);
        let kind = (!self.kinds.is_empty()).then(|| self.kinds.join(" + "));
        let mut body = Vec::new();
        if let Some(title) = &self.title {
            body.push(format!("{CHART_TITLE} {title}"));
        }
        for text in &self.texts {
            body.push(format!("{CHART_TEXT} {text}"));
        }
        for label in &self.labels {
            body.push(format!("{CHART_LABEL} {label}"));
        }
        if self.more_texts > 0 {
            body.push(chart_excess(self.more_texts, "text(s)"));
        }
        let mut described = false;
        for (index, series) in self.series_done.iter().enumerate() {
            let (line, values) = series_line(index, series, date1904, self.attached);
            described |= values;
            body.push(line);
        }
        if self.more_series > 0 {
            body.push(chart_excess(self.more_series, "series"));
        }
        if self.series_done.is_empty() {
            // A part naming no series at all says what it does not carry the way a
            // series without values does.
            body.push(note(self.attached).to_owned());
        }
        ChartRead::Content {
            kind,
            body,
            values: described,
        }
    }
}

/// The mark standing in for values a chart part does not carry.
const fn note(attached: bool) -> &'static str {
    if attached {
        CHART_VALUES_ATTACHED
    } else {
        CHART_NO_VALUES
    }
}

/// A break between two runs of a captured text, unless `text` is empty or already
/// ends in one.
fn push_break(text: &mut String) {
    if !text.is_empty() && !text.ends_with(' ') {
        text.push(' ');
    }
}

/// One series' line: its label, then its categories and values — `Jan: 10` where
/// the part carries a category label for the value, the value alone where it does
/// not, and a bubble's size beside its value — or, when the part carries none of
/// its values, the categories it does carry and the mark standing in for them.
/// `index` numbers an unnamed series, and the second value says whether the part
/// carried a value for it at all.
fn series_line(index: usize, series: &Series, date1904: bool, attached: bool) -> (String, bool) {
    let label = chart_series(series.name.as_deref(), index);
    let mut stored = false;
    let Some(values) = series
        .values
        .first()
        .filter(|cache| !cache.points.is_empty())
    else {
        // The labels the missing values would be keyed by are the chart's own
        // content, so the categories the part carries are still printed.
        let mut categories = Vec::new();
        for point_index in series
            .categories
            .first()
            .map(|cache| cache.points.keys())
            .into_iter()
            .flatten()
        {
            let (category, category_stored) =
                category_label(&series.categories, *point_index, date1904);
            stored |= category_stored;
            categories.extend(category);
        }
        let mut line = label;
        if !categories.is_empty() {
            let _ = write!(line, " {}", categories.join(", "));
        }
        let _ = write!(line, " {}", note(attached));
        if stored {
            line.push_str(MARK_STORED);
        }
        return (line, false);
    };
    let mut points = Vec::new();
    for point_index in values.points.keys() {
        let Some((value, value_stored)) = point_text(values, *point_index, date1904) else {
            continue;
        };
        stored |= value_stored;
        let (category, category_stored) =
            category_label(&series.categories, *point_index, date1904);
        stored |= category_stored;
        let mut point = match category {
            Some(category) => format!("{category}: {value}"),
            None => value,
        };
        if let Some((size, size_stored)) = series
            .sizes
            .first()
            .and_then(|cache| point_text(cache, *point_index, date1904))
        {
            stored |= size_stored;
            let _ = write!(point, " {}", chart_size(&size));
        }
        points.push(point);
    }
    let mut line = format!("{label} {}", points.join(", "));
    if values.more > 0 {
        let _ = write!(line, ", {}", chart_excess(values.more, "value(s)"));
    }
    if stored {
        line.push_str(MARK_STORED);
    }
    (line, true)
}

/// One cached point as the reading shows it: a number through [`numfmt`] under
/// the cache's own format — the stored text where that format cannot be
/// reproduced, which the second value says — or the text the part wrote.
fn point_text(cache: &Cache, index: usize, date1904: bool) -> Option<(String, bool)> {
    let stored = cache.points.get(&index)?;
    let written = || one_line(stored.trim().to_owned());
    if !cache.numeric {
        return Some((written(), false));
    }
    let format = cache.format.as_deref();
    if numfmt::is_general(0, format) {
        return Some((written(), false));
    }
    let display = stored
        .trim()
        .parse::<f64>()
        .ok()
        .and_then(|value| numfmt::display(value, 0, format, date1904));
    Some(display.map_or_else(|| (written(), true), |text| (text, false)))
}

/// The label of the point at `index`: the innermost level's own entry for it, with
/// each coarser level's label — the last one at or before the index, since a group
/// label spans the points under it — in front of it (`East / Q1`). A single level
/// is just its own entry, and an empty label is no label. The second value says
/// whether a label is shown as stored.
fn category_label(levels: &[Cache], index: usize, date1904: bool) -> (Option<String>, bool) {
    let Some((innermost, coarse)) = levels.split_first() else {
        return (None, false);
    };
    let mut parts: Vec<String> = Vec::new();
    let mut stored = false;
    for level in coarse.iter().rev() {
        if let Some((group, _)) = level.points.range(..=index).next_back()
            && let Some((text, level_stored)) = point_text(level, *group, date1904)
            && !text.is_empty()
        {
            stored |= level_stored;
            parts.push(text);
        }
    }
    if let Some((text, level_stored)) = point_text(innermost, index, date1904)
        && !text.is_empty()
    {
        stored |= level_stored;
        parts.push(text);
    }
    if parts.is_empty() {
        (None, stored)
    } else {
        (Some(parts.join(" / ")), stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A classic chart part around `plot`; `chart` sits inside `c:chart` ahead of
    /// its plot area, `space` inside `c:chartSpace`.
    fn chart_space(plot: &str, chart: &str, space: &str) -> String {
        format!(
            r#"<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{space}<c:chart>{chart}<c:plotArea>{plot}</c:plotArea></c:chart></c:chartSpace>"#
        )
    }

    /// One `c:ser` of a bar chart: its name, categories and values.
    fn series(name: &str, categories: &[&str], values: &[&str]) -> String {
        let categories = cache("strCache", None, categories);
        let values = cache("numCache", None, values);
        format!(
            r#"<c:ser><c:tx><c:strRef><c:f>Sheet1!$B$1</c:f><c:strCache><c:pt idx="0"><c:v>{name}</c:v></c:pt></c:strCache></c:strRef></c:tx><c:cat>{categories}</c:cat><c:val>{values}</c:val></c:ser>"#
        )
    }

    /// A `c:<kind>` cache over `points`, with `format` when one is given.
    fn cache(kind: &str, format: Option<&str>, points: &[&str]) -> String {
        let format = format.map_or_else(String::new, |format| {
            format!("<c:formatCode>{format}</c:formatCode>")
        });
        let mut cached = String::new();
        for (index, value) in points.iter().enumerate() {
            let _ = write!(cached, r#"<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>"#);
        }
        format!(r"<c:{kind}>{format}{cached}</c:{kind}>")
    }

    /// The lines a chart part is read to.
    fn lines_of(xml: &str) -> Vec<String> {
        read_part(xml.as_bytes(), false).lines()
    }

    /// A chart's content is printed as its kind, its title, its other text and one
    /// line per series of category and value. A rich-text title keeps its
    /// paragraphs apart, and a data label's own text prints under its own label, so
    /// a model can tell it from an axis title.
    #[test]
    fn a_chart_is_described_by_its_kind_title_and_series() {
        let plot = format!(
            r#"<c:barChart><c:barDir val="col"/><c:grouping val="clustered"/><c:ser><c:tx><c:strRef><c:f>Sheet1!$B$1</c:f><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx><c:cat>{}</c:cat><c:val>{}</c:val></c:ser>{}</c:barChart><c:catAx><c:title><c:tx><c:rich><a:p><a:r><a:t>Month</a:t></a:r></a:p></c:rich></c:tx></c:title></c:catAx>"#,
            cache("strCache", None, &["Jan", "Feb"]),
            cache("numCache", Some("General"), &["10", "20"]),
            series("Cost", &["Jan", "Feb"], &["4", "8"]),
        );
        let title = r"<c:title><c:tx><c:rich><a:p><a:r><a:t>Sales by region</a:t></a:r></a:p></c:rich></c:tx></c:title>";
        let labelled = format!(
            r#"<c:barChart><c:barDir val="col"/><c:ser>{}</c:ser><c:dLbls><c:dLbl><c:tx><c:rich><a:p><a:r><a:t>Peak</a:t></a:r></a:p></c:rich></c:tx></c:dLbl></c:dLbls></c:barChart>"#,
            r#"<c:val><c:numCache><c:pt idx="0"><c:v>10</c:v></c:pt></c:numCache></c:val>"#,
        );
        let rich_title = r"<c:title><c:tx><c:rich><a:p><a:r><a:t>Sales</a:t></a:r></a:p><a:p><a:r><a:t>by region</a:t></a:r></a:p></c:rich></c:tx></c:title>";
        for (label, xml, shown) in [
            (
                "a kind, a title, an axis text and two series",
                chart_space(&plot, title, ""),
                &[
                    "Chart (column chart):",
                    "  Title: Sales by region",
                    "  Text: Month",
                    "  Series \"Revenue\": Jan: 10, Feb: 20",
                    "  Series \"Cost\": Jan: 4, Feb: 8",
                ][..],
            ),
            (
                "a rich-text title and a data label kept apart",
                chart_space(&labelled, rich_title, ""),
                &[
                    "Chart (column chart):",
                    "  Title: Sales by region",
                    "  Label: Peak",
                    "  Series 1: 10",
                ][..],
            ),
        ] {
            assert_eq!(lines_of(&xml), shown, "{label}");
        }
    }

    /// The cache's own format decides how a value is shown, so a date reads as a
    /// date; a format the reading cannot reproduce shows the stored number, under
    /// the mark a spreadsheet cell carries, once for the series.
    #[test]
    fn a_cached_value_keeps_its_own_number_format() {
        let dates = cache("numCache", Some("mmm-yy"), &["45200", "45231"]);
        let shares = cache("numCache", Some("0.00%"), &["0.25", "0.5"]);
        let formatted = format!(
            r#"<c:lineChart><c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Sales</c:v></c:pt></c:strCache></c:strRef></c:tx><c:cat>{dates}</c:cat><c:val>{shares}</c:val></c:ser></c:lineChart>"#
        );
        let values = cache("numCache", Some("# ?/?"), &["1.5"]);
        let unreproduced = format!(
            r#"<c:pieChart><c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Share</c:v></c:pt></c:strCache></c:strRef></c:tx><c:val>{values}</c:val></c:ser></c:pieChart>"#
        );
        for (label, xml, shown) in [
            (
                "a date and a percentage",
                chart_space(&formatted, "", ""),
                &[
                    "Chart (line chart):",
                    "  Series \"Sales\": 2023-10-01: 25.00%, 2023-11-01: 50.00%",
                ][..],
            ),
            (
                "a format this reading does not reproduce",
                chart_space(&unreproduced, "", ""),
                &[
                    "Chart (pie chart):",
                    "  Series \"Share\": 1.5 (stored number, format not shown)",
                ][..],
            ),
        ] {
            assert_eq!(lines_of(&xml), shown, "{label}");
        }
    }

    /// A series without a name is numbered, a chart without a plot type prints no
    /// kind, several plot types are named together, and a scatter chart's x values
    /// are the labels its y values are paired with. A multi-level category list
    /// prints its levels root first, a group label stated once spanning every point
    /// under it, and a bubble chart prints each value's size beside it.
    #[test]
    fn kinds_names_and_scatter_are_all_printed() {
        let plot = format!(
            r#"<c:barChart><c:barDir val="bar"/><c:grouping val="stacked"/><c:ser><c:cat>{}</c:cat><c:val>{}</c:val></c:ser></c:barChart><c:scatterChart><c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Points</c:v></c:pt></c:strCache></c:strRef></c:tx><c:xVal>{}</c:xVal><c:yVal>{}</c:yVal></c:ser></c:scatterChart>"#,
            cache("strCache", None, &["A"]),
            cache("numCache", None, &["1"]),
            cache("numCache", None, &["1", "2"]),
            cache("numCache", None, &["10", "20"]),
        );
        // The levels are stated innermost first, as a part writes them.
        let two_levels = format!(
            r"<c:multiLvlStrRef><c:multiLvlStrCache>{}{}</c:multiLvlStrCache></c:multiLvlStrRef>",
            cache("lvl", None, &["Q1", "Q2"]),
            cache("lvl", None, &["East", "West"]),
        );
        let spanning = format!(
            r"<c:multiLvlStrRef><c:multiLvlStrCache>{}{}</c:multiLvlStrCache></c:multiLvlStrRef>",
            cache("lvl", None, &["Q1", "Q2"]),
            cache("lvl", None, &["East"]),
        );
        let sizes = cache("numCache", None, &["5", "7"]);
        let values = cache("numCache", None, &["10", "20"]);
        let bubble = |categories: &str| {
            format!(
                r"<c:bubbleChart><c:ser><c:cat>{categories}</c:cat><c:val>{values}</c:val><c:bubbleSize>{sizes}</c:bubbleSize></c:ser></c:bubbleChart>"
            )
        };
        for (label, xml, shown) in [
            (
                "several kinds, a numbered series and a scatter chart",
                chart_space(&plot, "", ""),
                &[
                    "Chart (stacked bar chart + scatter chart):",
                    "  Series 1: A: 1",
                    "  Series \"Points\": 1: 10, 2: 20",
                ][..],
            ),
            (
                "a chart naming no kind",
                chart_space("", "", ""),
                &["Chart:", "  (no values in the chart part)"][..],
            ),
            (
                "multi-level categories and bubble sizes",
                chart_space(&bubble(&two_levels), "", ""),
                &[
                    "Chart (bubble chart):",
                    "  Series 1: East / Q1: 10 (size 5), West / Q2: 20 (size 7)",
                ][..],
            ),
            (
                "a group label stated once spanning its points",
                chart_space(&bubble(&spanning), "", ""),
                &[
                    "Chart (bubble chart):",
                    "  Series 1: East / Q1: 10 (size 5), East / Q2: 20 (size 7)",
                ][..],
            ),
        ] {
            assert_eq!(lines_of(&xml), shown, "{label}");
        }
    }

    /// A cut chart part is a loss, not a chart whose content happens to end where
    /// the package stopped — and so is a part nested past the depth bound, a walk
    /// that would keep a stack entry per level for a document to decide.
    #[test]
    fn a_cut_part_is_a_loss() {
        let cut = r#"<c:chartSpace xmlns:c="x"><c:chart><c:plotArea><c:barChart><c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v>"#;
        let deep = format!(
            r#"<c:chartSpace xmlns:c="x"><c:chart><c:plotArea><c:barChart>{}{}</c:barChart></c:plotArea></c:chart></c:chartSpace>"#,
            "<c:x>".repeat(MAX_DEPTH),
            "</c:x>".repeat(MAX_DEPTH),
        );
        for (label, part) in [
            ("a part cut short", cut),
            ("a part nested past the bound", deep.as_str()),
        ] {
            assert_eq!(
                read_part(part.as_bytes(), false).lines(),
                ["Chart: (the chart part could not be read)"],
                "{label}"
            );
        }
    }

    /// A chart whose values are only referenced states that for the series that
    /// carries none, keeps the names the part does carry, and one whose table is
    /// attached says where its values are instead of claiming the file holds none.
    #[test]
    fn a_chart_without_cached_values_says_where_its_numbers_are() {
        let plot = r#"<c:barChart><c:barDir val="col"/><c:ser><c:tx><c:strRef><c:f>Sheet1!$B$1</c:f></c:strRef></c:tx><c:val><c:numRef><c:f>Sheet1!$B$2:$B$3</c:f></c:numRef></c:val></c:ser></c:barChart>"#;
        let external = r#"<c:externalData r:id="rId1"/>"#;
        let title = r"<c:title><c:tx><c:rich><a:p><a:r><a:t>Sales</a:t></a:r></a:p></c:rich></c:tx></c:title>";
        let named = r#"<c:barChart><c:barDir val="col"/><c:ser><c:tx><c:strRef><c:f>Sheet1!$B$1</c:f><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx><c:val><c:numRef><c:f>Sheet1!$B$2:$B$3</c:f></c:numRef></c:val></c:ser></c:barChart>"#;
        let labelled = format!(
            r#"<c:barChart><c:barDir val="col"/><c:ser><c:cat>{}</c:cat><c:val><c:numRef><c:f>Sheet1!$B$2:$B$3</c:f></c:numRef></c:val></c:ser></c:barChart>"#,
            cache("strCache", None, &["Jan", "Feb"]),
        );
        let uncached = r"<c:ser><c:tx><c:strRef><c:f>Sheet1!$C$1</c:f></c:strRef></c:tx><c:val><c:numRef><c:f>Sheet1!$C$2:$C$3</c:f></c:numRef></c:val></c:ser>";
        let mixed = format!(
            r#"<c:barChart><c:barDir val="col"/><c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx><c:val><c:numCache><c:pt idx="0"><c:v>10</c:v></c:pt></c:numCache></c:val></c:ser>{uncached}</c:barChart>"#
        );
        for (label, xml, shown) in [
            (
                "a series whose values are only referenced",
                chart_space(plot, "", ""),
                &[
                    "Chart (column chart):",
                    "  Series 1: (no values in the chart part)",
                ][..],
            ),
            (
                "a chart with its table attached",
                chart_space(plot, "", external),
                &[
                    "Chart (column chart):",
                    "  Series 1: (values only in the attached table)",
                ][..],
            ),
            (
                "the title and the name the part does carry",
                chart_space(named, title, external),
                &[
                    "Chart (column chart):",
                    "  Title: Sales",
                    "  Series \"Revenue\": (values only in the attached table)",
                ][..],
            ),
            (
                "the categories the part still caches",
                chart_space(&labelled, "", ""),
                &[
                    "Chart (column chart):",
                    "  Series 1: Jan, Feb (no values in the chart part)",
                ][..],
            ),
            (
                "one cached series beside one that is not",
                chart_space(&mixed, "", ""),
                &[
                    "Chart (column chart):",
                    "  Series \"Revenue\": 10",
                    "  Series 2: (no values in the chart part)",
                ][..],
            ),
        ] {
            assert_eq!(lines_of(&xml), shown, "{label}");
        }
        // A chart whose every series is uncached keeps its place in the report.
        assert!(
            !read_part(chart_space(plot, "", "").as_bytes(), false).described(),
            "a chart with no cached values is still described"
        );
    }

    /// The description is bounded: the series, the points and the other texts past
    /// their caps are counted, never silently dropped.
    #[test]
    fn the_description_is_bounded_and_says_what_it_left_out() {
        let values: Vec<String> = (0..(MAX_POINTS + 3)).map(|i| i.to_string()).collect();
        let values: Vec<&str> = values.iter().map(String::as_str).collect();
        let sers: String = (0..(MAX_SERIES + 2))
            .map(|i| series(&format!("S{i}"), &[], &values))
            .collect();
        let axes = (0..=MAX_TEXTS).fold(String::new(), |mut axes, i| {
            let _ = write!(
                axes,
                r"<c:catAx><c:title><c:tx><c:rich><a:p><a:r><a:t>T{i}</a:t></a:r></a:p></c:rich></c:tx></c:title></c:catAx>"
            );
            axes
        });
        let plot = format!(r"<c:lineChart>{sers}{axes}</c:lineChart>");
        let lines = lines_of(&chart_space(&plot, "", ""));
        assert_eq!(lines[0], "Chart (line chart):", "the kind line");
        assert_eq!(
            lines.len(),
            1 + MAX_TEXTS + 1 + MAX_SERIES + 1,
            "one line per kept text and series, and one per excess"
        );
        assert_eq!(
            lines[MAX_TEXTS + 1],
            format!("  {} more text(s)", 1),
            "the texts past the cap are counted"
        );
        assert_eq!(
            lines.last().map(String::as_str),
            Some("  2 more series"),
            "the series past the cap are counted"
        );
        assert!(
            lines[MAX_TEXTS + 2].ends_with(", 3 more value(s)"),
            "{:?}",
            lines[MAX_TEXTS + 2]
        );
    }

    /// The chart block's header, its series label and every mark and annotation a
    /// chart's lines carry are names the document kit's refusals know, so a find
    /// copied out of a reading that carries one is refused for what it is rather
    /// than searched for as the document's own text.
    #[test]
    fn the_marks_the_kit_refuses_are_the_ones_a_reading_prints() {
        let patterns: Vec<regex::Regex> = crate::docgen::docx_marks()
            .iter()
            .map(|mark| regex::Regex::new(&mark.pattern).expect("a mark's own pattern"))
            .collect();
        for mark in [
            chart_header(Some("column chart")),
            chart_series(Some("Revenue"), 0),
            chart_size("12"),
            chart_excess(3, "value(s)"),
            chart_excess(1, "series"),
            chart_excess(1, "text(s)"),
            MARK_STORED.trim().to_owned(),
            CHART_NO_VALUES.to_owned(),
            CHART_VALUES_ATTACHED.to_owned(),
            CHART_UNREADABLE.to_owned(),
            CHART_EXTENDED.to_owned(),
        ] {
            assert!(
                patterns.iter().any(|pattern| pattern.is_match(&mark)),
                "{mark:?} is a mark a chart reading prints, which no entry of the shared list \
                 names — a mark the kit refuses belongs in assets/docgen/rules.json's docx_marks"
            );
        }
        // The bare header a part naming no kind prints stays out: it is an
        // ordinary word a document's own text opens a sentence with.
        assert!(
            !patterns
                .iter()
                .any(|pattern| pattern.is_match(&chart_header(None))),
            "a bare `Chart:` is a document's own word, never a refusal"
        );
    }

    /// The chart parts a part names are the `chart` elements' relationship
    /// targets, in part order, without repeats, and only the ones the part's own
    /// relationships resolve.
    #[test]
    fn chart_references_resolve_in_part_order() {
        let xml = br#"<p:sld><p:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rId2"/></a:graphicData></a:graphic></p:graphicFrame><p:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rId1"/></a:graphicData></a:graphic></p:graphicFrame><p:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rId2"/></a:graphicData></a:graphic></p:graphicFrame>"#;
        let rels = vec![
            Relationship {
                id: "rId1".to_owned(),
                kind: "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"
                    .to_owned(),
                target: "../charts/chart1.xml".to_owned(),
            },
            Relationship {
                id: "rId2".to_owned(),
                kind: "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"
                    .to_owned(),
                target: "../charts/chart2.xml".to_owned(),
            },
        ];
        assert_eq!(
            chart_parts(xml, &rels, "ppt/slides/"),
            ["ppt/charts/chart2.xml", "ppt/charts/chart1.xml"]
        );
    }
}

//! PhotoCraft (`photocraft-cli`, published as ArtCraft): the vendor's stateless headless image
//! editor, the product's managed copy of it, and the thin layer that drives it. One document is
//! opened, a chain of the vendor's own commands is applied in order, and the result is saved.
//!
//! Nothing is pinned — the copy in use is compared with the vendor's newest release at start-up and
//! installed on first use — so vendor fixes reach users without a product release. The vendor is
//! alpha and guarantees no compatibility for command ids, so a command it no longer knows surfaces
//! as a clear failure here, never as a silent no-op.
//!
//! The vendor's stream shape is what the answers are read from: stdout carries the result and stderr
//! follows it, so a document or a batch's summary comes first and the engine's warnings after it.
//!
//! Every invocation runs through [`run`]; [`drive`] is the one that answers with the vendor's own
//! output and turns a non-zero exit into the failure, while [`batch`] reads the outcome itself
//! because a report naming failed files is still its answer. The vendor runs under the product's
//! existing containment ([`crate::tools::shell::run_internal_program_outcome`]) and memory watchdog,
//! from the install root as its working directory — never the caller's — with no wall-clock ceiling
//! and no cap on canvas size, input size or run time. Every invocation that opens a document carries
//! the vendor's `--approve-high-memory`: its own soft gate on a large PSD/PSB is one cap the product
//! does not add, and the watchdog is the real bound. Paths are not confined: a caller's path is
//! passed on as it stands, and no invocation carries the vendor's in-place mode, so an existing file
//! is never modified in place.
//!
//! Every answer is capped at [`MAX_ANSWER_BYTES`] before the product's own tool-output cut, so what a
//! model receives is the product's own shortened answer rather than a raw omission marker; each
//! builder says what its own answer keeps and what it drops.

use std::path::{Path, PathBuf};

use crate::tools::shell::{ProgramOutcome, Stop, run_internal_program_outcome};

mod install;

/// The most of one vendor answer the product keeps — a registry listing, a batch's report or an
/// inspection: the product's own tool-output budget
/// ([`crate::util::TOOL_OUTPUT_BUDGET_BYTES`]), which is what a model receives. Every builder here
/// keeps its answer inside it, note included, so the product's own cut never re-cuts one.
pub(crate) const MAX_ANSWER_BYTES: usize = crate::util::TOOL_OUTPUT_BUDGET_BYTES;

/// The most a note saying what an answer left out may take: [`capped_lines`], [`capped_rows`] and
/// [`batch_answer`] keep it back out of their budget. [`capped_document`] needs no reserve — it
/// measures the note with the rest of the text it emits.
const NOTE_ROOM: usize = 96;

/// The most of a failed run's output one answer quotes, so a batch's worth of lines cannot flood a
/// model's context. Cut as a head and a tail: a fatal error is the last thing the vendor says.
const MAX_FAILURE_BYTES: usize = 2_000;

/// One step of a chain: the vendor's own command id and its parameters.
pub(crate) struct Step {
    pub(crate) command: String,
    pub(crate) params: Option<serde_json::Value>,
}

/// A batch's answer and whether the vendor failed on any file. The report is the answer either way —
/// the files it wrote are the caller's work — and the flag is what a command-line caller exits
/// non-zero on.
#[derive(Debug)]
pub(crate) struct BatchReport {
    pub(crate) answer: String,
    pub(crate) partial: bool,
}

pub use install::install_on_start;

/// Open `input`, apply `chain` in order, save to `output`. An empty chain is the vendor's
/// `convert`, which is the same open-and-save with no commands. With no input at all, `canvas` is
/// the vendor's `--new` parameters and the document is created from nothing.
///
/// The vendor answers with one compact JSON line per step; the tool that delivers the result cuts
/// them to whole steps, with the room its own marker needs.
pub(crate) async fn run_chain(
    input: Option<&Path>,
    canvas: Option<&serde_json::Value>,
    chain: &[Step],
    output: &Path,
) -> anyhow::Result<String> {
    drive(&chain_args(input, canvas, chain, output)).await
}

/// The document's size, mode, depth and layers, as the vendor reports them, with the container
/// format its own name gives added as one field: the vendor's `info` carries no format field, so the
/// extension is what the product can know, and the answer says where it came from.
pub(crate) async fn inspect(input: &Path) -> anyhow::Result<String> {
    let answer = drive(&inspect_args(input)).await?;
    Ok(inspection_answer(&answer, input))
}

/// The container format a file's own name gives — the one fact the engine's `info` does not report.
fn format_field(input: &Path) -> String {
    match crate::util::extension_of(input) {
        Some(format) => format!("{format} (from the file name)"),
        None => "unknown (the file name has no extension)".to_string(),
    }
}

/// The inspection answer: the engine's document with [`format_field`] added as its `format` field, so
/// the answer is one JSON document, with whatever the engine wrote beside it still after it. An
/// answer carrying no JSON object keeps its text and gains the format after it.
///
/// A document longer than one answer keeps its own fields — the size, mode and depth the caller asked
/// for — and its arrays, its layer tree among them, keep the whole entries that fit with the count of
/// those left out: a layer is a record, so it is never half kept. A report the product's own capture
/// cut in the middle holds no whole document to recover, so the answer says so and keeps the format.
fn inspection_answer(answer: &str, input: &Path) -> String {
    let format = format_field(input);
    let (value, beside) = first_json(answer);
    let Some(serde_json::Value::Object(mut fields)) = value else {
        // The product's own capture cut the vendor's report in the middle, so no whole document can
        // be recovered: the answer says so and keeps the one fact the file name gives.
        if answer.contains(crate::util::TRUNCATED_MIDDLE) {
            return format!("… the inspection was too large to capture whole; format: {format}");
        }
        // No document to keep whole: the engine's own text is cut on whole lines, and the format the
        // product adds is kept whatever the text loses.
        let line = format!("format: {format}");
        let text = capped_lines(
            answer.trim_end(),
            MAX_ANSWER_BYTES.saturating_sub(line.len() + 1),
            "line",
            "",
        );
        return if text.is_empty() {
            line
        } else {
            format!("{text}\n{line}")
        };
    };
    fields.insert("format".to_string(), serde_json::Value::String(format));
    let beside = capped_lines(beside, MAX_ANSWER_BYTES / 4, "line", "");
    let room = MAX_ANSWER_BYTES.saturating_sub(beside.len() + 1);
    let document = capped_document(fields, room);
    if beside.is_empty() {
        document
    } else {
        format!("{document}\n{beside}")
    }
}

/// One inspection document cut to `room` bytes. A document that fits is answered exactly as the
/// engine wrote it; past that, its own fields are kept whole — they are what the caller asked for —
/// and an array keeps the whole entries that fit, with the count of those left out as its last
/// element, so a layer is never half kept. Every choice is made by measuring the text that will be
/// emitted, so no margin is predicted and none can drift from what is written.
///
/// The cheapest array is filled first and each one still to come keeps its floor — the count note
/// saying how much it holds — so a layer tree takes what the small ones (notes, snapshots, warnings)
/// leave instead of starving them. A document whose own fields alone are past one answer, which the
/// vendor's inspection does not produce, keeps the format the product adds and one note counting the
/// rest.
fn capped_document(fields: serde_json::Map<String, serde_json::Value>, room: usize) -> String {
    let total = fields.len();
    // The whole document first: the per-array measurement below keeps every later array at its floor,
    // which is a note larger than a short array's own content, so a document that fits is answered
    // exactly as the engine wrote it.
    if let Ok(text) = serde_json::to_string(&fields)
        && text.len() <= room
    {
        return text;
    }
    let mut facts = serde_json::Map::new();
    let mut arrays: Vec<(String, Vec<serde_json::Value>)> = Vec::new();
    for (name, value) in fields {
        match value {
            serde_json::Value::Array(entries) => arrays.push((name, entries)),
            fact => {
                facts.insert(name, fact);
            }
        }
    }
    if !fits(&facts, room) {
        return crowded_document(&facts, total);
    }
    arrays.sort_by_key(|(_, entries)| json_len(entries));
    for (index, (name, entries)) in arrays.iter().enumerate() {
        let kept = kept_entries(&facts, name, entries, &arrays[index + 1..], room);
        facts.insert(name.clone(), serde_json::Value::Array(kept));
    }
    match serde_json::to_string(&facts) {
        Ok(text) if text.len() <= room => text,
        // The arrays' floors alone do not fit beside the fields: the format and one counted note are
        // what is left to say.
        _ => crowded_document(&facts, total),
    }
}

/// The whole entries of `entries` that fit beside `facts` in `room` bytes, with the count of those
/// left out as the last element. `later` are the arrays still to be filled: each keeps its floor in
/// every measurement, so this array cannot take the room they need to say how much they hold.
fn kept_entries(
    facts: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    entries: &[serde_json::Value],
    later: &[(String, Vec<serde_json::Value>)],
    room: usize,
) -> Vec<serde_json::Value> {
    // The answer only grows as more entries are kept — an entry costs at least three bytes with its
    // separator, while the count note loses at most one — so halving reaches the count that fits in a
    // handful of measurements rather than one per entry.
    let (mut fits_count, mut too_many) = (0, entries.len() + 1);
    while too_many - fits_count > 1 {
        let middle = fits_count + (too_many - fits_count) / 2;
        if measured(facts, name, entries, middle, later, room) {
            fits_count = middle;
        } else {
            too_many = middle;
        }
    }
    kept(entries, fits_count)
}

/// Whether the answer holds `count` of `entries` under `name`, with each array in `later` at its
/// floor, in `room` bytes — measured on the text that will be emitted.
fn measured(
    facts: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    entries: &[serde_json::Value],
    count: usize,
    later: &[(String, Vec<serde_json::Value>)],
    room: usize,
) -> bool {
    let mut candidate = facts.clone();
    candidate.insert(
        name.to_string(),
        serde_json::Value::Array(kept(entries, count)),
    );
    for (name, entries) in later {
        candidate.insert(name.clone(), serde_json::Value::Array(kept(entries, 0)));
    }
    fits(&candidate, room)
}

/// The first `count` of `entries`, with the count of those left out as the last element when any are.
fn kept(entries: &[serde_json::Value], count: usize) -> Vec<serde_json::Value> {
    let mut shown = entries[..count].to_vec();
    if count < entries.len() {
        shown.push(serde_json::Value::String(count_note(
            entries.len() - count,
            entries.len(),
        )));
    }
    shown
}

/// Whether the document takes no more than `room` bytes as one JSON object.
fn fits(document: &serde_json::Map<String, serde_json::Value>, room: usize) -> bool {
    serde_json::to_string(document).is_ok_and(|text| text.len() <= room)
}

/// How long an array is as JSON. Only the order the arrays are filled in depends on it — the cheapest
/// first, so a layer tree cannot starve the small ones — never a budget decision.
fn json_len(entries: &[serde_json::Value]) -> usize {
    serde_json::to_string(entries).map_or(0, |json| json.len())
}

/// What a cut array says after the entries it keeps: how many of them the answer does not show.
fn count_note(left_out: usize, total: usize) -> String {
    format!("… {left_out} of {total} further entries not shown")
}

/// The whole answer for a document that cannot be cut down to fit: the format the product adds and one
/// note counting the members the answer does not show, so a shortened answer is still one JSON
/// document rather than half a value. It answers a document whose own fields alone are past one
/// answer — a shape the vendor's inspection does not produce — and one whose arrays cannot even say
/// how much they hold.
fn crowded_document(facts: &serde_json::Map<String, serde_json::Value>, total: usize) -> String {
    let mut answer = serde_json::Map::new();
    let mut omitted = total;
    if let Some(format) = facts.get("format") {
        answer.insert("format".to_string(), format.clone());
        omitted -= 1;
    }
    answer.insert(
        "omitted".to_string(),
        serde_json::Value::String(format!("… {omitted} further member(s) not shown")),
    );
    serde_json::Value::Object(answer).to_string()
}

/// The first JSON value in an engine answer, and what the engine wrote after it: stderr is
/// concatenated after stdout, so the value comes first and a warning the engine printed follows it,
/// never the other way round.
fn first_json(text: &str) -> (Option<serde_json::Value>, &str) {
    let text = text.trim_start();
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<serde_json::Value>();
    match values.next() {
        Some(Ok(value)) => (Some(value), text[values.byte_offset()..].trim()),
        _ => (None, text.trim()),
    }
}

/// The vendor's whole command registry, optionally narrowed by `filter`. A narrowed answer carries
/// each match's parameter documentation, so a caller gets from an id to a callable step in one
/// question; the unfiltered registry is an index of ids and labels.
pub(crate) async fn listing(filter: Option<&str>) -> anyhow::Result<String> {
    let output = drive(&listing_args(filter)).await?;
    if is_empty_listing(&output) {
        return Ok(no_commands(filter));
    }
    Ok(registry_answer(filter, &output))
}

/// The answer a registry response turns into: under the cap it is the vendor's own text, untouched
/// but for the blank lines it may open with; past it, a narrowed answer drops whole commands and the
/// unfiltered one is cut on a line boundary.
fn registry_answer(filter: Option<&str>, output: &str) -> String {
    let output = output.trim_start();
    // The product's own capture cut the vendor's answer in the middle — a filter matching most of the
    // registry is that large — so no whole command can be recovered from it: the answer says so
    // rather than handing back a JSON document cut in half.
    if output.contains(crate::util::TRUNCATED_MIDDLE) {
        return format!("… the registry answer was too large to capture whole; {NARROWER}");
    }
    if output.len() <= MAX_ANSWER_BYTES {
        return output.to_string();
    }
    match filter {
        Some(_) => capped_rows(output),
        None => capped_listing(output),
    }
}

/// Whether the vendor answered with nothing at all, or with the empty JSON array a filter that
/// matches no command produces.
fn is_empty_listing(output: &str) -> bool {
    let trimmed = output.trim();
    trimmed.is_empty()
        || (trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && trimmed[1..trimmed.len() - 1].trim().is_empty())
}

/// What an empty registry answer says instead of nothing: a filter that matches no command is the
/// ordinary way to get one, and a blank tool result tells a model nothing.
fn no_commands(filter: Option<&str>) -> String {
    match filter {
        Some(filter) => format!("no command matches {filter:?}"),
        None => "the vendor's command registry is empty".to_string(),
    }
}

/// Apply `chain` to every file in the `input` folder, writing into the `output` folder.
///
/// The vendor reports a batch line by line: `ok    <input> -> <output>` on stdout per produced file
/// and `FAIL  <input>: <reason>` on stderr per file it could not do, with `<n> succeeded, <m> failed`
/// as the last line of its stdout when it ran the folder to the end. That summary is what tells a run
/// that ran from one that stopped, so it decides between an answer and a failure here.
pub(crate) async fn batch(
    chain: &[Step],
    input: &Path,
    output: &Path,
    format: Option<&str>,
    quality: Option<u32>,
    tiff_layers: bool,
) -> anyhow::Result<BatchReport> {
    let (_recipe_dir, recipe) = recipe_file(chain)?;
    let argv = batch_args(&recipe, input, output, format, quality, tiff_layers);
    let outcome = run(&argv).await?;
    let written = files_written(output).await;
    batch_outcome(&outcome, &written, output).map_err(anyhow::Error::msg)
}

/// A batch's run turned into its answer. The vendor prints its closing summary only when it ran the
/// folder to the end, so a summary present means the run got there and a non-zero exit only repeats
/// what the summary says. A run that succeeded without one is still an answer — nothing was left out,
/// so the report is whatever the vendor wrote. Without either, the run never got that far and its
/// failure is the answer: the tool then reports how many files it did write before stopping. A run
/// that failed on any file answers with the shortfall flag set, so a command-line caller exits
/// non-zero on the report it still receives.
fn batch_outcome(
    outcome: &ProgramOutcome,
    written: &FilesWritten,
    output_dir: &Path,
) -> Result<BatchReport, String> {
    if outcome.success || outcome.output.lines().any(is_summary) {
        Ok(BatchReport {
            answer: batch_answer(&outcome.output, written, output_dir),
            partial: failed_some(&outcome.output),
        })
    } else {
        Err(failure("batch", outcome))
    }
}

/// A batch's answer. It opens with the product's own sentence about what the batch wrote — the one
/// fact a caller must never lose — and then the vendor's report: the lines naming files it failed on
/// with their reasons, whole per-file lines from the start and from the end, and its closing summary.
/// The count and the summary are kept whatever else goes; the failures are what a caller acts on, so
/// they take their room first and the per-file lines and encoder warnings get what is left. Whatever
/// is dropped goes whole and is counted, which the product's own tool-output cut could never do, since
/// it takes a plain middle out with no count.
fn batch_answer(output: &str, written: &FilesWritten, output_dir: &Path) -> String {
    let text = output.trim();
    let header = written.written_into(output_dir);
    if text.is_empty() {
        return header;
    }
    if header.len() + 1 + text.len() <= MAX_ANSWER_BYTES {
        return format!("{header}\n{text}");
    }
    let lines: Vec<&str> = text.lines().collect();
    let summary: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| is_summary(line))
        .collect();
    let failures: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| is_failure(line))
        .collect();
    let results: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| !is_summary(line) && !is_failure(line))
        .collect();

    // What the header, the summary and the per-file report's own note need, so the whole answer stays
    // under the cap.
    let mut left =
        MAX_ANSWER_BYTES.saturating_sub(header.len() + 1 + lines_bytes(&summary) + NOTE_ROOM);
    // A file the vendor failed on is what a caller acts on, so the failures take their room first —
    // appended after the per-file report, because the summary closes the answer. Their own note is
    // kept back only when some of them are left out.
    let mut kept_failures = lines_within(failures.iter().copied(), left);
    if kept_failures < failures.len() {
        left = left.saturating_sub(NOTE_ROOM);
        kept_failures = lines_within(failures.iter().copied(), left);
    }
    left -= lines_bytes(&failures[..kept_failures]);
    // The rest is the per-file report: whole lines from the start and from the end.
    let side = left / 2;
    let head = lines_within(results.iter().copied(), side);
    let tail = lines_within(results[head..].iter().rev().copied(), side);

    let mut answer = header;
    for line in &results[..head] {
        crate::util::push_line(&mut answer, line);
    }
    let omitted = results.len() - head - tail;
    if omitted > 0 {
        crate::util::push_line(
            &mut answer,
            &format!("… {omitted} further result lines not shown"),
        );
    }
    for line in &results[results.len() - tail..] {
        crate::util::push_line(&mut answer, line);
    }
    if kept_failures < failures.len() {
        crate::util::push_line(
            &mut answer,
            &format!(
                "… {} further failures not shown",
                failures.len() - kept_failures
            ),
        );
    }
    for line in &failures[..kept_failures] {
        crate::util::push_line(&mut answer, line);
    }
    for line in &summary {
        crate::util::push_line(&mut answer, line);
    }
    answer
}

/// The vendor's closing line for a batch it ran to the end — `<n> succeeded, <m> failed`, and any
/// further clause it adds after that (`--stop-on-error` appends `, <k> skipped`) — as its two counts.
/// Read word by word, so punctuation and a trailing clause cannot hide it, while a per-file line or
/// the exit status's own `error:` line cannot be mistaken for it.
fn summary_counts(line: &str) -> Option<(u64, u64)> {
    let words: Vec<&str> = line
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    match words.as_slice() {
        [succeeded, "succeeded", failed, "failed", ..] => {
            Some((succeeded.parse().ok()?, failed.parse().ok()?))
        }
        _ => None,
    }
}

/// Whether a line is the vendor's closing summary: what tells a run that ran from one that stopped.
fn is_summary(line: &str) -> bool {
    summary_counts(line).is_some()
}

/// Whether the vendor failed on any file: a `FAIL` line names one, and the closing summary counts
/// them. Either is enough — a report that says a file failed never reads as a whole success.
fn failed_some(output: &str) -> bool {
    output
        .lines()
        .any(|line| is_failure(line) || summary_counts(line).is_some_and(|(_, failed)| failed > 0))
}

/// A file the vendor could not do, as it names one on stderr: `FAIL  <input>: <reason>`.
fn is_failure(line: &str) -> bool {
    line.starts_with("FAIL ")
}

/// How many bytes `lines` take with their line breaks.
fn lines_bytes(lines: &[&str]) -> usize {
    lines.iter().map(|line| line.len() + 1).sum()
}

/// Count the files in a folder. A batch writes into a folder that is new or empty, so this is what
/// it produced — the one fact its answer opens with — and the tool counts a batch that stopped
/// partway the same way.
pub(crate) async fn files_written(dir: &Path) -> FilesWritten {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return FilesWritten {
            count: 0,
            whole: false,
        };
    };
    let mut written = FilesWritten {
        count: 0,
        whole: true,
    };
    loop {
        match entries.next_entry().await {
            // A folder or a link is not a produced file; one whose type cannot be read is a file
            // this count cannot vouch for.
            Ok(Some(entry)) => match entry.file_type().await {
                Ok(kind) if kind.is_file() => written.count += 1,
                Ok(_) => {}
                Err(_) => written.whole = false,
            },
            Ok(None) => return written,
            Err(_) => {
                written.whole = false;
                return written;
            }
        }
    }
}

/// A folder's file count, and whether it is the count or the floor a folder that could not be read
/// in full leaves: [`Self::written_into`] states both in one sentence, so a short read is never
/// passed off as the count.
pub(crate) struct FilesWritten {
    pub(crate) count: usize,
    pub(crate) whole: bool,
}

impl FilesWritten {
    /// What an answer says a batch wrote into `dir`: the count, the floor it could reach when the
    /// folder could not be read in full, or the folder's own unreadability when not one file could be
    /// counted. One sentence, so no caller has to splice a clause into another.
    pub(crate) fn written_into(&self, dir: &Path) -> String {
        let dir = dir.display();
        match (self.whole, self.count) {
            (true, 0) => format!("no file was written into {dir}"),
            (true, count) => format!("{count} file(s) written to {dir}"),
            (false, 0) => format!("the files in {dir} could not be counted"),
            (false, count) => format!(
                "at least {count} file(s) written to {dir} (the folder could not be read in full)"
            ),
        }
    }
}

/// How many whole lines of `lines` fit in `budget` bytes, counting from the start.
fn lines_within<'a>(lines: impl Iterator<Item = &'a str>, budget: usize) -> usize {
    let mut used = 0;
    let mut count = 0;
    for line in lines {
        used += line.len() + 1;
        if used > budget {
            break;
        }
        count += 1;
    }
    count
}

/// Run the vendor with `argv` and answer the whole outcome. Only a run the product itself ended is a
/// failure here: the program could not be started, or the memory watchdog stopped it. Whether the
/// program's own exit status is an error belongs to the entry point that asked for it.
async fn run(argv: &[String]) -> anyhow::Result<ProgramOutcome> {
    let what = argv.first().map_or("", String::as_str);
    let (_held, cwd, program) = install::hold_for_run().await.map_err(anyhow::Error::msg)?;
    let outcome = run_internal_program_outcome(&cwd, &program, argv, &[], None).await;
    if outcome.stopped() {
        return Err(anyhow::Error::msg(failure(what, &outcome)));
    }
    Ok(outcome)
}

/// Run the vendor with `argv` and answer its output. Every entry point that must succeed goes through
/// here, so the containment, the working directory and the shape of a failure are one decision. The
/// first element is the vendor's own subcommand, which is also what a failure names — every builder
/// returns the whole argv, so no call can lose it.
async fn drive(argv: &[String]) -> anyhow::Result<String> {
    let what = argv.first().map_or("", String::as_str);
    let outcome = run(argv).await?;
    if !outcome.success {
        return Err(anyhow::Error::msg(failure(what, &outcome)));
    }
    Ok(outcome.output)
}

/// Whether a format names TIFF: the one output format whose layers the vendor drops unless asked.
fn is_tiff(format: &str) -> bool {
    format.eq_ignore_ascii_case("tif") || format.eq_ignore_ascii_case("tiff")
}

/// The vendor's own soft gate on a large PSD/PSB import. Passing it is what keeps the product's
/// memory watchdog the only cap on how big an input may be; the vendor still holds its hard budget.
const APPROVE_HIGH_MEMORY: &str = "--approve-high-memory";

/// The argv of one chain over one document, or over a canvas created from nothing: the vendor's
/// `run`, the document — or `--new` and the canvas — one `--cmd` per step with its `--params` when it
/// has any, then the destination. With no steps and a document it is the vendor's `convert`, the same
/// open-and-save with nothing applied; a canvas has no file to open, so it always goes through `run`.
/// Every path is absolute: the vendor resolves a relative one against its working directory, which is
/// the install root.
///
/// A TIFF destination carries `--tiff-layers` on both paths: the vendor flattens a TIFF without it, so
/// leaving it out would silently drop the layers a `.tiff` destination is chosen to keep.
fn chain_args(
    input: Option<&Path>,
    canvas: Option<&serde_json::Value>,
    chain: &[Step],
    output: &Path,
) -> Vec<String> {
    let tiff = crate::util::extension_of(output).is_some_and(|extension| is_tiff(&extension));
    let output_arg = path_arg(output);
    if let (Some(input), None) = (input, canvas)
        && chain.is_empty()
    {
        let mut args = vec!["convert".to_string(), path_arg(input), output_arg];
        args.push(APPROVE_HIGH_MEMORY.to_string());
        if tiff {
            args.push("--tiff-layers".to_string());
        }
        return args;
    }
    let mut args = match (input, canvas) {
        (Some(input), None) => vec!["run".to_string(), path_arg(input)],
        (_, Some(canvas)) => vec!["run".to_string(), "--new".to_string(), canvas.to_string()],
        // The tool refuses a `run` with neither; the vendor's own usage error would answer it.
        (None, None) => vec!["run".to_string()],
    };
    for step in chain {
        args.push("--cmd".to_string());
        args.push(step.command.clone());
        if let Some(params) = &step.params {
            args.push("--params".to_string());
            args.push(params.to_string());
        }
    }
    args.push("--out".to_string());
    args.push(output_arg);
    args.push(APPROVE_HIGH_MEMORY.to_string());
    if tiff {
        args.push("--tiff-layers".to_string());
    }
    args
}

/// The argv of one `info`. `--compact` prints the document as one line instead of a pretty tree: a
/// layered document's tree runs past what one answer can hold, and the answer parses either way.
fn inspect_args(input: &Path) -> Vec<String> {
    vec![
        "info".to_string(),
        path_arg(input),
        "--compact".to_string(),
        APPROVE_HIGH_MEMORY.to_string(),
    ]
}

/// The argv of one `commands`: narrowed by the vendor's own `--filter` when the caller has one, and
/// asking for JSON only then, because that is what adds each match's parameter documentation — the
/// unfiltered registry's JSON is far past what one answer can hold.
fn listing_args(filter: Option<&str>) -> Vec<String> {
    let mut args = vec!["commands".to_string()];
    if let Some(filter) = filter {
        args.push("--filter".to_string());
        args.push(filter.to_string());
        args.push("--json".to_string());
    }
    args
}

/// The argv of one `batch`: the recipe, the two folders, the optional flags that were given, and
/// `--tiff-layers` whenever the output is TIFF — whether the caller set the flag or named the
/// format — because the vendor flattens a TIFF without it. The vendor's `--in-place` is never among
/// them.
fn batch_args(
    recipe: &Path,
    input: &Path,
    output: &Path,
    format: Option<&str>,
    quality: Option<u32>,
    tiff_layers: bool,
) -> Vec<String> {
    let mut args = vec![
        "batch".to_string(),
        "--actions".to_string(),
        path_arg(recipe),
        "--in".to_string(),
        path_arg(input),
        "--out".to_string(),
        path_arg(output),
    ];
    if let Some(format) = format {
        args.push("--format".to_string());
        args.push(format.to_string());
    }
    if let Some(quality) = quality {
        args.push("--quality".to_string());
        args.push(quality.to_string());
    }
    // A batch into TIFF keeps its layers whether the caller set the flag or named the format: the
    // vendor flattens a TIFF without it, and flattening is never what writing TIFF asks for.
    if tiff_layers || format.is_some_and(is_tiff) {
        args.push("--tiff-layers".to_string());
    }
    args.push(APPROVE_HIGH_MEMORY.to_string());
    args
}

/// A path as the vendor's argument parser takes it.
fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The recipe `batch` reads: the chain as the vendor's step list, one object per step, with
/// `params` present only where the step has any. The file sits in the product's pinned temp root
/// and its directory is removed when the call ends, whether it succeeded or not.
fn recipe_file(chain: &[Step]) -> anyhow::Result<(tempfile::TempDir, PathBuf)> {
    use anyhow::Context as _;

    let dir = tempfile::Builder::new()
        .prefix("photocraft-")
        .tempdir_in(crate::temp::shell_tmpdir())
        .context("the batch recipe's temporary directory could not be created")?;
    let path = dir.path().join("recipe.json");
    std::fs::write(&path, recipe_json(chain)).context("the batch recipe could not be written")?;
    Ok((dir, path))
}

/// The recipe's JSON: an array of `{"command": "<id>", "params": {...}}`, `params` present only
/// where the step has any.
fn recipe_json(chain: &[Step]) -> String {
    let steps: Vec<serde_json::Value> = chain
        .iter()
        .map(|step| match &step.params {
            Some(params) => serde_json::json!({"command": step.command, "params": params}),
            None => serde_json::json!({"command": step.command}),
        })
        .collect();
    serde_json::Value::Array(steps).to_string()
}

/// What a capped listing says about reaching the rest.
const NARROWER: &str = "a narrower filter shows the rest";

/// A narrowed registry answer past [`MAX_ANSWER_BYTES`]: the vendor's JSON, one object per matching
/// command, cut at whole commands with a line saying how many were left out — never a document cut
/// in the middle, where a model cannot tell a truncated parameter from a complete one, and never a
/// single command cut in half. The JSON is found inside the answer, because a warning the engine
/// wrote beside it is not part of it. An answer carrying no JSON array is cut on line boundaries.
fn capped_rows(text: &str) -> String {
    let Some(rows) = registry_rows(text) else {
        return capped_listing(text);
    };
    // The note saying how many commands were left out is reserved up front, so the array and the
    // note together never run past one answer.
    let room = MAX_ANSWER_BYTES.saturating_sub(NOTE_ROOM);
    let total = rows.len();
    let mut kept: Vec<String> = Vec::new();
    let mut used = 2; // the enclosing brackets
    for row in &rows {
        let row = row.to_string();
        let separator = if kept.is_empty() { 0 } else { 2 }; // ",\n" between commands
        if used + separator + row.len() > room {
            break;
        }
        used += separator + row.len();
        kept.push(row);
    }
    if kept.is_empty() && !rows.is_empty() {
        // Not one command fits whole: the answer names the count rather than half a command, so a
        // model is never handed a truncated parameter it could mistake for a complete one.
        return format!(
            "[]\n… all {total} matching command(s) not shown: each one's own documentation is longer \
             than a whole answer"
        );
    }
    let answer = format!("[{}]", kept.join(",\n"));
    if kept.len() == total {
        // Every command is there but the vendor's pretty printing is past the cap: the compact
        // array says the same thing and fits.
        return answer;
    }
    let left_out = total - kept.len();
    format!("{answer}\n… {left_out} of {total} matching commands not shown; {NARROWER}")
}

/// The commands in a narrowed answer: the JSON array the vendor printed.
fn registry_rows(text: &str) -> Option<Vec<serde_json::Value>> {
    match first_json(text).0 {
        Some(serde_json::Value::Array(rows)) => Some(rows),
        _ => None,
    }
}

/// The unfiltered registry answer cut on a line boundary: the vendor prints one command per line, so
/// a whole line is a whole command.
fn capped_listing(text: &str) -> String {
    capped_lines(text, MAX_ANSWER_BYTES, "command", NARROWER)
}

/// An answer cut on whole lines: the lines from the start that fit `budget` less the room the note
/// needs, then one line saying what was withheld — `remedy`, when there is a way to reach the rest. A
/// line that does not fit on its own is dropped whole rather than cut inside, `noun` being what one
/// line is, and an answer never opens with the blank lines a program may print before its first line.
/// The cut is byte-bounded, so an answer built here, note included, stays under `budget` and the
/// product's own tool-output truncation never re-cuts it.
pub(crate) fn capped_lines(text: &str, budget: usize, noun: &str, remedy: &str) -> String {
    let text = text.trim_start();
    if text.is_empty() {
        return String::new();
    }
    // The note is reserved out of the budget up front, so an answer built here — kept lines and note
    // together — never runs past what its caller allowed.
    let room = budget.saturating_sub(NOTE_ROOM);
    let total = text.lines().count();
    let mut kept = String::new();
    let mut kept_lines = 0;
    for line in text.split_inclusive('\n') {
        if kept.len() + line.len() > room {
            break;
        }
        kept.push_str(line);
        kept_lines += 1;
    }
    let withheld = total - kept_lines;
    let note = if withheld == 0 {
        String::new()
    } else {
        let plural = if withheld == 1 { "" } else { "s" };
        if remedy.is_empty() {
            format!("… {withheld} further {noun}{plural} not shown")
        } else {
            format!("… {withheld} further {noun}{plural} not shown; {remedy}")
        }
    };
    match (kept_lines, note.is_empty()) {
        (0, _) => note,
        (_, true) => kept.trim_end().to_string(),
        (_, false) => format!("{}\n{note}", kept.trim_end()),
    }
}

/// What a failed run tells a model: what was attempted, how the run ended, and the vendor's own
/// words, bounded — its first lines and its last, because a fatal error is the last thing it says. A
/// run the product stopped is named as such — never as an io error.
fn failure(what: &str, outcome: &ProgramOutcome) -> String {
    let detail = match outcome.stop {
        Some(Stop::Spawn) => "the program could not be started on this machine".to_string(),
        Some(Stop::Memory) => format!(
            "the product's memory watchdog stopped it: {}",
            outcome.stop_reason()
        ),
        // No ceiling is set for this program, so a stop at one cannot arise.
        Some(Stop::Timeout) | None => outcome.detail.trim().to_string(),
    };
    let detail = if detail.is_empty() {
        "no reason was reported".to_string()
    } else {
        detail
    };
    let output =
        crate::util::truncate_sandwich(outcome.output.trim(), MAX_FAILURE_BYTES, "vendor output");
    if output.is_empty() {
        return format!("photocraft {what} failed ({detail})");
    }
    format!("photocraft {what} failed ({detail}): {output}")
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn step(command: &str, params: Option<serde_json::Value>) -> Step {
        Step {
            command: command.to_string(),
            params,
        }
    }

    fn outcome(detail: &str, output: &str, stop: Option<Stop>) -> ProgramOutcome {
        ProgramOutcome {
            success: false,
            detail: detail.to_string(),
            output: output.to_string(),
            has_output: !output.is_empty(),
            stop,
        }
    }

    /// A count an answer states as exact, for the fixtures that never read a folder.
    fn counted(count: usize) -> FilesWritten {
        FilesWritten { count, whole: true }
    }

    #[test]
    fn a_chain_runs_its_steps_in_order_before_the_output() {
        let input = Some(Path::new("/work/plate.psd"));
        let output = Path::new("/work/out.png");
        // No steps is the vendor's `convert`, whose destination is the second positional argument.
        assert_eq!(
            chain_args(input, None, &[], output),
            [
                "convert",
                "/work/plate.psd",
                "/work/out.png",
                "--approve-high-memory"
            ]
        );
        // A step without parameters carries no `--params` at all.
        assert_eq!(
            chain_args(input, None, &[step("autocrop", None)], output),
            [
                "run",
                "/work/plate.psd",
                "--cmd",
                "autocrop",
                "--out",
                "/work/out.png",
                "--approve-high-memory"
            ]
        );
        assert_eq!(
            chain_args(
                input,
                None,
                &[
                    step("resize", Some(json!({"width": 800}))),
                    step("sharpen", None),
                ],
                output
            ),
            [
                "run",
                "/work/plate.psd",
                "--cmd",
                "resize",
                "--params",
                "{\"width\":800}",
                "--cmd",
                "sharpen",
                "--out",
                "/work/out.png",
                "--approve-high-memory"
            ]
        );
    }

    #[test]
    fn a_canvas_created_from_nothing_is_the_vendors_new_document() {
        let canvas = json!({"width": 800, "height": 600});
        let output = Path::new("/work/out.png");
        // No document to open: `convert` has nothing to read, so the canvas goes through `run`.
        assert_eq!(
            chain_args(None, Some(&canvas), &[], output),
            [
                "run",
                "--new",
                "{\"height\":600,\"width\":800}",
                "--out",
                "/work/out.png",
                "--approve-high-memory"
            ]
        );
        // A chain applies to the fresh document like any other.
        let args = chain_args(None, Some(&canvas), &[step("fill", None)], output);
        assert_eq!(
            args[..3],
            ["run", "--new", "{\"height\":600,\"width\":800}"]
        );
        assert_eq!(args[3], "--cmd");
        assert!(args.contains(&"--out".to_string()));
    }

    #[test]
    fn a_tiff_destination_keeps_its_layers_and_every_other_format_is_left_alone() {
        let input = Some(Path::new("/work/plate.psd"));
        let chain = [step("filter.sharpen.smartSharpen", None)];
        // Without the flag the vendor flattens a TIFF, so a `.tiff` destination must carry it —
        // on the open-and-save path as well as the chain path.
        for output in ["/work/out.tiff", "/work/out.TIF"] {
            let args = chain_args(input, None, &chain, Path::new(output));
            assert!(
                args.contains(&"--tiff-layers".to_string()),
                "{output}: {args:?}"
            );
            let converted = chain_args(input, None, &[], Path::new(output));
            assert!(
                converted.contains(&"--tiff-layers".to_string()),
                "{output}: {converted:?}"
            );
        }
        // A PNG, a PSD and a name with no extension are not TIFF and carry nothing extra.
        for output in ["/work/out.png", "/work/out.psd", "/work/out"] {
            assert!(
                !chain_args(input, None, &chain, Path::new(output))
                    .contains(&"--tiff-layers".to_string()),
                "{output}"
            );
        }
    }

    #[test]
    fn inspection_and_the_registry_ask_for_what_the_caller_needs() {
        // `--compact` keeps a layered document's report to one line; `--approve-high-memory` keeps
        // the vendor's own soft gate from refusing a large PSD the product was asked to open.
        assert_eq!(
            inspect_args(Path::new("/work/plate.pcraft")),
            [
                "info",
                "/work/plate.pcraft",
                "--compact",
                "--approve-high-memory"
            ]
        );
        assert_eq!(listing_args(None), ["commands"]);
        // A narrowed answer asks for the JSON, which is what carries each match's parameter docs.
        assert_eq!(
            listing_args(Some("blur")),
            ["commands", "--filter", "blur", "--json"]
        );
    }

    #[test]
    fn batch_carries_every_flag_it_was_given_and_none_it_was_not() {
        let recipe = Path::new("/tmp/recipe.json");
        let input = Path::new("/work/in");
        let output = Path::new("/work/out");
        assert_eq!(
            batch_args(recipe, input, output, None, None, false),
            [
                "batch",
                "--actions",
                "/tmp/recipe.json",
                "--in",
                "/work/in",
                "--out",
                "/work/out",
                "--approve-high-memory"
            ]
        );
        assert_eq!(
            batch_args(recipe, input, output, Some("png"), Some(90), true),
            [
                "batch",
                "--actions",
                "/tmp/recipe.json",
                "--in",
                "/work/in",
                "--out",
                "/work/out",
                "--format",
                "png",
                "--quality",
                "90",
                "--tiff-layers",
                "--approve-high-memory"
            ]
        );
        // A TIFF output keeps its layers whether the flag was set or the format names it.
        for format in ["tiff", "TIF"] {
            let args = batch_args(recipe, input, output, Some(format), None, false);
            assert!(
                args.contains(&"--tiff-layers".to_string()),
                "{format}: {args:?}"
            );
        }
    }

    #[test]
    fn a_long_batch_answer_keeps_its_first_and_last_lines_and_says_how_many_it_left_out() {
        // The vendor's shape: one line per produced file, then its own summary.
        let mut spelled: Vec<String> = (0..400)
            .map(|index| format!("ok    /work/in/photo{index}.heic -> /work/out/photo{index}.png"))
            .collect();
        spelled.push("400 succeeded, 0 failed".to_string());
        let results: Vec<&str> = spelled[..400].iter().map(String::as_str).collect();
        let output = spelled.join("\n");
        assert!(output.len() > MAX_ANSWER_BYTES, "the fixture must not fit");
        let capped = batch_answer(&output, &counted(400), Path::new("/work/out"));
        let kept: Vec<&str> = capped.lines().collect();
        // The product's own count opens the answer: the one fact a caller must never lose.
        assert_eq!(kept[0], "400 file(s) written to /work/out");
        let note = kept[1..]
            .iter()
            .position(|line| line.starts_with('…'))
            .map(|at| at + 1)
            .expect("the cap says what it left out");
        // Whole result lines from the start, whole result lines from the end, and the vendor's own
        // summary after them.
        assert_eq!(kept[1..note], results[..note - 1]);
        let tail = &kept[note + 1..kept.len() - 1];
        assert_eq!(tail, &results[results.len() - tail.len()..]);
        assert!(
            kept[note].contains(&format!(
                "{} further result lines not shown",
                results.len() - (note - 1) - tail.len()
            )),
            "{}",
            kept[note]
        );
        // The vendor's own summary is kept whole, after the report.
        assert_eq!(capped.lines().last(), Some("400 succeeded, 0 failed"));
        // The cap binds before the product's own tool-output cut, which would take the same middle
        // out with no count, and the note is inside it.
        assert!(capped.len() <= MAX_ANSWER_BYTES, "{}", capped.len());
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
        // An answer within the cap is the vendor's own text under the count.
        let short = "ok    /work/in/a.heic -> /work/out/a.png\n1 succeeded, 0 failed";
        assert_eq!(
            batch_answer(short, &counted(1), Path::new("/work/out")),
            format!("1 file(s) written to /work/out\n{short}")
        );
    }

    #[test]
    fn a_batch_answer_keeps_every_failure_and_the_summary_whatever_it_drops() {
        // The vendor's own report: successes and the summary on stdout, then the failures and the
        // encoder's per-file warnings on stderr, which arrives appended. The summary is therefore
        // nowhere near the tail, and the failure sits behind hundreds of per-file lines.
        let mut spelled: Vec<String> = (0..300)
            .map(|index| format!("ok    /work/in/photo{index}.jpg -> /work/out/photo{index}.jpg"))
            .collect();
        spelled.push("300 succeeded, 1 failed".to_string());
        spelled.push("FAIL  /work/in/photo150.jpg: unsupported 12-bit depth".to_string());
        spelled.extend(
            (0..300)
                .map(|index| format!("warning: photo{index}.jpg: no ICC profile, assuming sRGB")),
        );
        let output = spelled.join("\n");
        assert!(output.len() > MAX_ANSWER_BYTES, "the fixture must not fit");
        let capped = batch_answer(&output, &counted(300), Path::new("/work/out"));
        let kept: Vec<&str> = capped.lines().collect();
        assert_eq!(kept[0], "300 file(s) written to /work/out");
        // Both facts survive: which file failed and why, and the vendor's own count.
        assert!(
            capped.contains("FAIL  /work/in/photo150.jpg: unsupported 12-bit depth"),
            "{capped}"
        );
        assert_eq!(capped.lines().last(), Some("300 succeeded, 1 failed"));
        // Whole lines only, and the cap binds before the product's own cut.
        assert!(
            kept[1..]
                .iter()
                .all(|line| spelled.contains(&line.to_string()) || line.starts_with('…')),
            "{capped}"
        );
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_batch_with_more_failures_than_fit_keeps_the_summary_and_counts_the_rest() {
        let mut spelled: Vec<String> = (0..400)
            .map(|index| format!("FAIL  /work/in/photo{index}.heic: unreadable header"))
            .collect();
        spelled.push("0 succeeded, 400 failed".to_string());
        let output = spelled.join("\n");
        let capped = batch_answer(&output, &counted(0), Path::new("/work/out"));
        assert_eq!(
            capped.lines().next(),
            Some("no file was written into /work/out"),
            "the count is the head"
        );
        // Every kept failure line is whole, and the ones that did not fit are counted.
        let note = capped
            .lines()
            .find(|line| line.starts_with("… ") && line.ends_with("further failures not shown"))
            .expect("the cap counts the failures it dropped");
        assert!(capped.contains("FAIL  /work/in/photo0.heic: unreadable header"));
        assert_eq!(capped.lines().last(), Some("0 succeeded, 400 failed"));
        assert!(note.contains("further failures not shown"), "{note}");
        assert!(capped.len() <= MAX_ANSWER_BYTES, "{}", capped.len());
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
        // A batch of nothing but failures fills the answer: the room the failures can use is theirs,
        // not half of it.
        assert!(
            capped.len() + 200 >= MAX_ANSWER_BYTES,
            "{} of {MAX_ANSWER_BYTES}",
            capped.len()
        );
    }

    #[test]
    fn a_batch_answer_never_opens_with_the_blank_line_the_engine_printed_first() {
        let output = "\n\nok    /work/in/a.heic -> /work/out/a.png\n1 succeeded, 0 failed\n";
        let answer = batch_answer(output, &counted(1), Path::new("/work/out"));
        assert_eq!(
            answer.lines().next(),
            Some("1 file(s) written to /work/out")
        );
        assert!(!answer.starts_with('\n'), "{answer:?}");
    }

    #[test]
    fn the_recipe_is_the_vendors_step_list_with_params_only_where_there_are_some() {
        let plain = recipe_json(&[step("autocrop", None), step("sharpen", None)]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&plain).expect("the recipe parses"),
            json!([{"command": "autocrop"}, {"command": "sharpen"}])
        );
        // The vendor's own shape: a step without parameters has no `params` key.
        assert!(!plain.contains("params"), "{plain}");
        let with_params = recipe_json(&[step("resize", Some(json!({"width": 800})))]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&with_params).expect("the recipe parses"),
            json!([{"command": "resize", "params": {"width": 800}}])
        );
    }

    #[test]
    fn a_listing_within_the_cap_comes_back_as_the_vendor_wrote_it() {
        let registry = "blur: gaussian blur\ncrop: cut the canvas\n";
        assert_eq!(registry_answer(None, registry), registry);
        // A narrowed answer under the cap is the vendor's own text too, never re-serialized.
        let narrowed = "[\n  {\n    \"id\": \"image.crop\"\n  }\n]";
        assert_eq!(registry_answer(Some("crop"), narrowed), narrowed);
    }

    #[test]
    fn a_filter_that_matches_nothing_says_so_instead_of_answering_blank() {
        // The vendor answers a filter that matches nothing with an empty array, or with nothing at
        // all; a real match is never mistaken for one.
        for empty in ["[]", " [ ]\n", "", "  \n"] {
            assert!(is_empty_listing(empty), "{empty:?}");
        }
        assert!(!is_empty_listing(r#"[{"id":"image.crop"}]"#));
        let answer = no_commands(Some("nosuchthing"));
        assert!(answer.contains("no command matches"), "{answer}");
        assert!(answer.contains("nosuchthing"), "{answer}");
        assert!(no_commands(None).contains("registry is empty"));
    }

    #[test]
    fn a_listing_past_the_cap_is_cut_on_a_line_boundary_and_says_how_much_it_withheld() {
        let line = "blur: gaussian blur; params: radius (number)\n";
        // The lines that fit one answer less the room its note needs.
        let fits = (MAX_ANSWER_BYTES - NOTE_ROOM) / line.len();
        let withheld = 25;
        let registry = line.repeat(fits + withheld);
        let capped = capped_listing(&registry);
        let mut lines = capped.lines().collect::<Vec<_>>();
        let note = lines.pop().expect("the cap says what it withheld");
        // Every kept line is a whole line of the vendor's own text.
        assert!(lines.iter().all(|kept| *kept == line.trim_end()));
        assert_eq!(lines.len(), fits);
        assert!(
            note.contains(&format!("{withheld} further commands")),
            "{note}"
        );
        assert!(note.contains("a narrower filter shows the rest"), "{note}");
        // The cap binds before the product's own tool-output cut, so a model sees this note
        // rather than that cut's raw omission marker — and the note itself is inside the cap.
        assert!(capped.len() <= MAX_ANSWER_BYTES, "{}", capped.len());
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_registry_answer_the_capture_cut_in_the_middle_says_so_instead_of_half_a_document() {
        // A filter matching most of the registry produces more than one capture holds, so the vendor's
        // own JSON arrives cut in the middle: no whole command can be recovered, and half a document
        // would make a model guess at parameter names.
        let text = format!(
            "[{{\"id\":\"image.crop\"}},\n{}\n[{{\"id\":\"image.trim\"}}]",
            crate::util::TRUNCATED_MIDDLE
        );
        let answer = registry_answer(Some("image"), &text);
        assert_eq!(
            answer,
            format!("… the registry answer was too large to capture whole; {NARROWER}")
        );
        assert!(
            !answer.contains('{'),
            "no fragment of the document: {answer}"
        );
    }

    #[test]
    fn an_inspection_the_capture_cut_in_the_middle_says_so_and_keeps_the_format() {
        // A layered document's report can outgrow the product's own capture, which splices its
        // marker into the middle of the vendor's JSON: no whole document is recoverable, so the
        // answer says so rather than handing back half a value — and keeps the one fact the file
        // name gives.
        let text = format!(
            "{{\"width\":4096,\"mode\":\"Rgb\",\"layers\":[{{\"name\":\"Layer 0\"}},\n{}\n{{\"name\":\"Layer 900\"}}]}}",
            crate::util::TRUNCATED_MIDDLE
        );
        let answer = inspection_answer(&text, Path::new("/work/plate.psd"));
        assert_eq!(
            answer,
            "… the inspection was too large to capture whole; format: psd (from the file name)"
        );
        assert!(!answer.contains('{'), "no fragment: {answer}");
        assert!(!answer.starts_with('\n'), "no blank first line");
        assert_eq!(crate::util::truncate_tool_output(&answer), answer);
    }

    #[test]
    fn a_line_too_long_to_fit_is_dropped_whole_rather_than_cut_inside() {
        // One line can be longer than the cap on its own — an answer carrying no JSON array to cut
        // at whole commands. It is a record, so it goes whole and the answer names the count; what
        // arrives is never half a line a caller could mistake for a complete one.
        let line = format!(
            "[{}]",
            "{\"id\":\"image.crop\",\"params\":\"кадр\"},".repeat(2000)
        );
        let capped = capped_listing(&line);
        assert_eq!(capped, format!("… 1 further command not shown; {NARROWER}"));
        assert!(!capped.starts_with('\n'), "no blank first line");
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
        // Blank lines the engine printed first are not part of the answer either.
        let capped = capped_listing("\n\nblur: gaussian blur\n");
        assert_eq!(capped, "blur: gaussian blur");
    }

    #[test]
    fn the_inspection_answer_names_the_format_its_file_name_gives_as_one_more_field() {
        let answer = inspection_answer(
            r#"{"file":"/work/plate.PSD","mode":"Rgb","depth":8}"#,
            Path::new("/work/plate.PSD"),
        );
        let document: serde_json::Value =
            serde_json::from_str(&answer).expect("the answer is still one JSON document");
        assert_eq!(document["format"], "psd (from the file name)");
        assert_eq!(document["mode"], "Rgb", "the engine's own fields stand");
        // No extension, or an empty one: the answer says it does not know, rather than guessing.
        for name in ["/work/scan", "/work/scan."] {
            let answer = inspection_answer(r#"{"file":"x"}"#, Path::new(name));
            let document: serde_json::Value =
                serde_json::from_str(&answer).expect("still one JSON document");
            assert_eq!(
                document["format"],
                "unknown (the file name has no extension)"
            );
        }
        // An engine answer that is not a JSON object keeps its text and gains the format after it.
        let answer = inspection_answer("mode: Rgb\ndepth: 8\n", Path::new("/work/plate.pcraft"));
        assert_eq!(
            answer,
            "mode: Rgb\ndepth: 8\nformat: pcraft (from the file name)"
        );
        // The engine's stderr is concatenated after its stdout: the document is still one JSON
        // object, and what the engine wrote beside it stays after it.
        let answer = inspection_answer(
            "{\"mode\":\"Rgb\"}\nwarning: a profile could not be embedded",
            Path::new("/work/plate.psd"),
        );
        let (document, beside) = answer
            .split_once('\n')
            .expect("the warning stays after the document");
        assert_eq!(beside, "warning: a profile could not be embedded");
        let document: serde_json::Value =
            serde_json::from_str(document).expect("still one JSON document");
        assert_eq!(document["format"], "psd (from the file name)");
        assert_eq!(document["mode"], "Rgb");
    }

    #[test]
    fn a_layered_inspection_keeps_its_facts_and_counts_the_layers_that_do_not_fit() {
        // The engine prints small arrays beside the layer tree — notes, the selected layers, the
        // snapshots, the warnings — and a real document's report lands just past one answer at some
        // sizes and not others. Wherever the cut falls, the document's own facts and those arrays
        // must survive and the tree must end in a count of what did not fit: never half a layer, and
        // never the format alone.
        for count in 1..=40 {
            for padding in [0, 7, 13, 29, 41] {
                let layers: Vec<serde_json::Value> = (0..count)
                    .map(|index| {
                        // A real layer entry carries its bounds, mask and effects: some 320 bytes.
                        json!({
                            "name": format!("Layer {index}{}", "x".repeat(padding)),
                            "kind": "pixel",
                            "visible": true,
                            "opacity": 100,
                            "blend": "normal",
                            "bounds": {"x": 0, "y": 0, "width": 3000, "height": 2000},
                            "source": format!("/work/plate.psd#layer{index}"),
                            "mask": {"path": format!("/work/mask-{index}.png"), "enabled": true, "density": 100},
                            "effects": {"dropShadow": false, "stroke": false, "bevel": false},
                        })
                    })
                    .collect();
                let document = json!({
                    "file": "/work/plate.psd",
                    "width": 3000,
                    "height": 2000,
                    "mode": "Rgb",
                    "depth": 8,
                    "layers": layers,
                    "notes": [],
                    "selectedLayers": [],
                    "snapshots": [],
                    "warnings": [],
                })
                .to_string();
                if document.len() <= MAX_ANSWER_BYTES {
                    continue;
                }
                let where_ = format!("{count} layers padded by {padding}");
                let answer = inspection_answer(&document, Path::new("/work/plate.psd"));
                assert!(
                    answer.len() <= MAX_ANSWER_BYTES,
                    "{where_}: {}",
                    answer.len()
                );
                assert_eq!(
                    crate::util::truncate_tool_output(&answer),
                    answer,
                    "{where_}"
                );
                let document: serde_json::Value = serde_json::from_str(&answer)
                    .unwrap_or_else(|_| panic!("{where_}: not one JSON document: {answer}"));
                for fact in ["format", "width", "height", "mode", "depth"] {
                    assert!(
                        !document[fact].is_null(),
                        "{where_}: {fact} was lost: {answer}"
                    );
                }
                assert_eq!(document["notes"], json!([]), "{where_}: {answer}");
                let kept = document["layers"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{where_}: {}", document["layers"]));
                // The last entry is the count of what did not fit, and every entry before it is whole.
                let (note, entries) = kept.split_last().expect("{where_}: the tree is not empty");
                assert_eq!(
                    note.as_str(),
                    Some(
                        format!(
                            "… {} of {count} further entries not shown",
                            count - entries.len()
                        )
                        .as_str()
                    ),
                    "{where_}"
                );
                assert_eq!(
                    entries[0]["name"],
                    format!("Layer 0{}", "x".repeat(padding)),
                    "{where_}"
                );
            }
        }
    }

    #[test]
    fn a_layered_inspection_leaves_a_later_array_its_own_count() {
        // Two arrays past one answer: the cheaper one is filled first, and it must leave the other the
        // room to say how much it holds — one array never starves the rest of the answer.
        let warnings: Vec<serde_json::Value> = (0..300)
            .map(|index| json!(format!("layer {index} has no colour profile")))
            .collect();
        let layers: Vec<serde_json::Value> = (0..400)
            .map(|index| {
                json!({
                    "name": format!("Layer {index}"),
                    "kind": "pixel",
                    "visible": true,
                    "bounds": {"x": 0, "y": 0, "width": 3000, "height": 2000},
                })
            })
            .collect();
        let document = json!({
            "file": "/work/plate.psd",
            "width": 3000,
            "height": 2000,
            "mode": "Rgb",
            "depth": 8,
            "layers": layers,
            "warnings": warnings,
        })
        .to_string();
        let answer = inspection_answer(&document, Path::new("/work/plate.psd"));
        assert!(answer.len() <= MAX_ANSWER_BYTES, "{}", answer.len());
        let document: serde_json::Value =
            serde_json::from_str(&answer).expect("still one whole JSON document");
        assert_eq!(document["mode"], "Rgb", "{answer}");
        for name in ["layers", "warnings"] {
            let kept = document[name]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: {}", document[name]));
            assert!(
                kept.last().is_some_and(serde_json::Value::is_string),
                "{name} keeps its count of the rest: {answer}"
            );
        }
    }

    #[test]
    fn a_narrowed_answer_drops_whole_commands_and_says_how_many_it_left_out() {
        let rows: Vec<serde_json::Value> = (0..500)
            .map(|index| json!({"id": format!("image.command{index}"), "params": "x".repeat(40)}))
            .collect();
        let text = serde_json::to_string_pretty(&rows).expect("the answer serializes");
        assert!(text.len() > MAX_ANSWER_BYTES, "the fixture must not fit");
        let capped = capped_rows(&text);
        // The kept part is whole commands — a JSON array that parses, never half a row.
        let body = capped
            .lines()
            .take_while(|line| !line.starts_with('…'))
            .collect::<Vec<_>>()
            .join("\n");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&body).expect("the kept commands are a JSON array");
        assert!(!parsed.is_empty(), "the answer carries what fits");
        assert!(parsed.len() < rows.len());
        assert_eq!(
            parsed,
            rows[..parsed.len()].to_vec(),
            "leading commands only"
        );
        let note = capped
            .lines()
            .last()
            .expect("the cap says what it left out");
        assert!(
            note.contains(&format!(
                "{} of 500 matching commands not shown",
                rows.len() - parsed.len()
            )),
            "{note}"
        );
        assert!(note.contains("a narrower filter shows the rest"), "{note}");
        // The cap binds before the product's own tool-output cut, note included.
        assert!(capped.len() <= MAX_ANSWER_BYTES, "{}", capped.len());
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_narrowed_answer_that_fits_comes_back_as_the_vendor_wrote_it() {
        // Many small commands: pretty-printed past the cap, every one of them fitting compactly.
        let rows: Vec<serde_json::Value> = (0..180)
            .map(|index| json!({"id": format!("image.command{index}")}))
            .collect();
        let pretty = serde_json::to_string_pretty(&rows).expect("the answer serializes");
        let compact = serde_json::to_string(&rows).expect("the answer serializes");
        assert!(
            pretty.len() > MAX_ANSWER_BYTES,
            "the fixture must not fit as printed"
        );
        assert!(
            compact.len() <= MAX_ANSWER_BYTES,
            "the fixture fits compactly"
        );
        let capped = registry_answer(Some("command"), &pretty);
        assert!(!capped.contains('…'), "nothing was left out: {capped}");
        assert_eq!(
            serde_json::from_str::<Vec<serde_json::Value>>(&capped).expect("still a JSON array"),
            rows
        );
    }

    #[test]
    fn a_narrowed_answer_that_is_not_a_json_array_is_still_cut_on_whole_lines() {
        let text = "image.crop: Crop\n".repeat(2000);
        let capped = capped_rows(&text);
        assert!(
            capped
                .lines()
                .all(|line| line == "image.crop: Crop" || line.starts_with('…')),
            "whole lines only"
        );
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_narrowed_answer_finds_its_commands_beside_a_warning_the_engine_wrote() {
        // The engine's stderr is concatenated after its stdout, so a warning must not cost the
        // answer its whole-command cut.
        let rows: Vec<serde_json::Value> = (0..300)
            .map(|index| json!({"id": format!("image.command{index}"), "params": "x".repeat(40)}))
            .collect();
        let text = format!(
            "{}\nwarning: some profile could not be embedded",
            serde_json::to_string_pretty(&rows).expect("the answer serializes")
        );
        let capped = capped_rows(&text);
        let body = capped
            .lines()
            .take_while(|line| !line.starts_with('…'))
            .collect::<Vec<_>>()
            .join("\n");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&body).expect("the kept commands are a JSON array");
        assert!(!parsed.is_empty());
        assert!(parsed.len() < rows.len());
        assert_eq!(parsed, rows[..parsed.len()].to_vec());
    }

    #[test]
    fn a_command_too_long_to_fit_whole_is_dropped_with_a_count_rather_than_cut_mid_entry() {
        // A registry answer is never cut mid-entry: half a command's parameter documentation reads
        // as a complete one. Not one command fits here, so the answer is an empty array and the
        // count — never a fragment.
        let rows = vec![json!({"id": "image.crop", "params": "x".repeat(MAX_ANSWER_BYTES)})];
        let text = serde_json::to_string_pretty(&rows).expect("the answer serializes");
        let capped = capped_rows(&text);
        let mut lines = capped.lines();
        assert_eq!(lines.next(), Some("[]"), "{capped}");
        let note = lines.next().expect("the cap says what it left out");
        assert!(
            note.contains("all 1 matching command(s) not shown"),
            "{note}"
        );
        assert!(note.contains("longer than a whole answer"), "{note}");
        assert_eq!(lines.next(), None, "nothing else is left: {capped}");
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_failed_run_carries_the_exit_detail_and_the_vendors_own_words() {
        let message = failure(
            "run",
            &outcome("exit status 2", "unknown flag --cmdd\n", None),
        );
        assert_eq!(
            message,
            "photocraft run failed (exit status 2): unknown flag --cmdd"
        );
    }

    #[test]
    fn a_failure_with_no_reason_says_so_instead_of_leaving_an_empty_parenthesis() {
        assert_eq!(
            failure("run", &outcome("", "", None)),
            "photocraft run failed (no reason was reported)"
        );
    }

    #[test]
    fn a_run_the_product_never_started_says_so_instead_of_quoting_an_io_error() {
        let message = failure(
            "run",
            &outcome("io error: No such file or directory", "", Some(Stop::Spawn)),
        );
        assert!(message.contains("could not be started"), "{message}");
        assert!(!message.contains("No such file"), "{message}");
    }

    #[test]
    fn the_memory_watchdog_is_named_as_the_product_stopping_the_run() {
        let message = failure(
            "batch",
            &outcome(
                "terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)",
                "",
                Some(Stop::Memory),
            ),
        );
        assert!(message.contains("memory watchdog"), "{message}");
        assert!(message.contains("exceeded memory limit"), "{message}");
    }

    #[test]
    fn the_summary_and_the_failures_are_the_vendors_own_lines_and_nothing_else_is() {
        // The vendor's closing line, with and without its --stop-on-error clause.
        for summary in [
            "0 succeeded, 0 failed",
            "300 succeeded, 1 failed",
            "4 succeeded, 2 failed, 7 skipped",
        ] {
            assert!(is_summary(summary), "{summary}");
        }
        // A per-file line, a failure, the exit status's own line and a warning are none of them a
        // summary — the decision between an answer and a failure rests on this.
        for other in [
            "ok    /in/a.png -> /out/a.png",
            "FAIL  /in/a.png: unreadable header",
            "error: 1 file(s) failed",
            "300 files succeeded, 1 failed",
            "warning: photo1.jpg: no ICC profile",
        ] {
            assert!(!is_summary(other), "{other}");
        }
        assert!(is_failure("FAIL  /in/a.png: unreadable header"));
        assert!(!is_failure("error: 1 file(s) failed"));
        assert!(!is_failure("ok    /in/a.png -> /out/a.png"));
    }

    #[test]
    fn a_batch_that_ran_to_its_summary_is_an_answer_and_one_that_stopped_is_a_failure() {
        // A batch that produced a file, failed on another and exited non-zero: the vendor reached its
        // summary, so the answer is the report — the exit status only repeats what it says.
        let ran = ProgramOutcome {
            success: false,
            detail: "exit status 1".to_string(),
            output: "ok    /in/a.png -> /out/a.png\n1 succeeded, 1 failed\nFAIL  /in/b.png: unreadable header".to_string(),
            has_output: true,
            stop: None,
        };
        let report = batch_outcome(&ran, &counted(1), Path::new("/out"))
            .expect("the run reached its summary");
        let answer = &report.answer;
        assert_eq!(answer.lines().next(), Some("1 file(s) written to /out"));
        assert!(
            answer.contains("FAIL  /in/b.png: unreadable header"),
            "{answer}"
        );
        // Nothing was left out, so the vendor's own report arrives as it wrote it.
        assert!(
            answer.lines().any(|line| line == "1 succeeded, 1 failed"),
            "{answer}"
        );
        // A file failed, so the call is a partial one: the command line exits non-zero on the answer
        // it still receives, and the agent reads the report.
        assert!(report.partial, "{answer}");

        // A run that stopped before its summary — a crash, a stop, a usage error — is the failure,
        // and the caller's own count of what it wrote before stopping is the tool's to add.
        let stopped = ProgramOutcome {
            success: false,
            detail: "exit status 1".to_string(),
            output: "ok    /in/a.png -> /out/a.png\nerror: the disk is full".to_string(),
            has_output: true,
            stop: None,
        };
        let error = batch_outcome(&stopped, &counted(1), Path::new("/out"))
            .expect_err("no summary, no report");
        assert!(
            error.starts_with("photocraft batch failed (exit status 1)"),
            "{error}"
        );
        assert!(error.contains("the disk is full"), "{error}");

        // A run that succeeded is an answer even with no summary line: nothing was left out, and
        // reporting a clean exit as a failure would be a lie about the machine.
        let clean = ProgramOutcome {
            success: true,
            detail: "exit status 0".to_string(),
            output: "ok    /in/a.png -> /out/a.png".to_string(),
            has_output: true,
            stop: None,
        };
        let report = batch_outcome(&clean, &counted(1), Path::new("/out"))
            .expect("a clean run is an answer");
        assert_eq!(
            report.answer.lines().next(),
            Some("1 file(s) written to /out")
        );
        assert!(!report.partial, "a clean run is no shortfall");
    }

    #[test]
    fn a_batch_that_produced_nothing_is_a_partial_answer_not_a_success() {
        // The vendor ran the folder to its summary and failed on every file: the report is the
        // answer — it names each file and the reason — and the call is a partial one, so a
        // command-line caller exits non-zero instead of reading a whole success.
        let report = batch_outcome(
            &outcome("exit status 1", "0 succeeded, 2 failed\nFAIL  /in/a.heic: unreadable header\nFAIL  /in/b.heic: unreadable header", None),
            &counted(0),
            Path::new("/out"),
        )
        .expect("the run reached its summary");
        assert_eq!(
            report.answer.lines().next(),
            Some("no file was written into /out")
        );
        assert!(report.partial, "{}", report.answer);
        // The summary's own count is enough on its own: a failure line the capture lost still
        // cannot make a failed batch read as a success.
        let summary_only = outcome("exit status 1", "3 succeeded, 4 failed", None);
        assert!(failed_some(&summary_only.output));
        assert!(!failed_some(
            "ok    /in/a.png -> /out/a.png\n1 succeeded, 0 failed"
        ));
    }

    #[test]
    fn an_inspection_whose_own_fields_alone_do_not_fit_keeps_the_format_and_counts_the_rest() {
        // The vendor's inspection does not produce a document whose own fields alone are past one
        // answer; when one arrives, the answer keeps the format the product adds and one note counting
        // what it could not show — still one JSON document rather than half a value.
        let mut document = serde_json::Map::new();
        document.insert("mode".to_string(), json!("Rgb"));
        document.insert("depth".to_string(), json!(8));
        for index in 0..8 {
            document.insert(format!("field{index}"), json!("y".repeat(800)));
        }
        let answer = inspection_answer(
            &serde_json::Value::Object(document).to_string(),
            Path::new("/work/plate.psd"),
        );
        assert!(answer.len() <= MAX_ANSWER_BYTES, "{}", answer.len());
        let document: serde_json::Value =
            serde_json::from_str(&answer).expect("still one whole JSON document");
        assert_eq!(document["format"], "psd (from the file name)");
        assert_eq!(document["omitted"], "… 10 further member(s) not shown");
        assert_eq!(crate::util::truncate_tool_output(&answer), answer);
    }

    #[test]
    fn a_count_the_folder_could_not_be_read_for_is_stated_as_the_floor_it_is() {
        // A folder that cannot be read in full leaves the count a lower bound: the answer says so
        // beside the count rather than passing a short read off as the one fact a batch answer must
        // never lose — and the sentence stays one the reader can parse, whatever the count is.
        let short = FilesWritten {
            count: 12,
            whole: false,
        };
        let report = "ok    /work/in/a.png -> /work/out/a.png\n1 succeeded, 0 failed";
        let answer = batch_answer(report, &short, Path::new("/work/out"));
        assert_eq!(
            answer.lines().next(),
            Some("at least 12 file(s) written to /work/out (the folder could not be read in full)")
        );
        // A folder that could not be read at all says that, rather than counting a floor of nothing.
        let unread = batch_answer(
            report,
            &FilesWritten {
                count: 0,
                whole: false,
            },
            Path::new("/work/out"),
        );
        assert_eq!(
            unread.lines().next(),
            Some("the files in /work/out could not be counted")
        );
        assert_eq!(
            counted(12).written_into(Path::new("/work/out")),
            "12 file(s) written to /work/out"
        );
        assert_eq!(
            counted(0).written_into(Path::new("/work/out")),
            "no file was written into /work/out"
        );
    }

    #[test]
    fn a_failure_longer_than_the_whole_per_file_report_is_still_named() {
        // One FAIL line can be longer than all 400 per-file lines together — a deep path and a long
        // reason. It still gets the room it needs, because which file failed is what a caller acts on.
        let mut spelled: Vec<String> = (0..400)
            .map(|index| format!("ok    /work/in/photo{index}.jpg -> /work/out/photo{index}.jpg"))
            .collect();
        let reason = "x".repeat(2_600);
        spelled.push(format!("FAIL  /work/in/deep/photo.jpg: {reason}"));
        spelled.push("400 succeeded, 1 failed".to_string());
        let output = spelled.join("\n");
        let capped = batch_answer(&output, &counted(400), Path::new("/work/out"));
        assert!(
            capped.contains(&format!("FAIL  /work/in/deep/photo.jpg: {reason}")),
            "the failing file is named"
        );
        assert_eq!(capped.lines().last(), Some("400 succeeded, 1 failed"));
        assert!(capped.len() <= MAX_ANSWER_BYTES, "{}", capped.len());
        assert_eq!(crate::util::truncate_tool_output(&capped), capped);
    }

    #[test]
    fn a_wall_of_output_is_bounded_before_it_reaches_a_model() {
        // A fatal error is the last thing the vendor says, so the bound keeps a head AND a tail
        // rather than the first two kilobytes alone.
        let message = failure(
            "batch",
            &outcome(
                "exit status 1",
                &format!("{}broken.png: unreadable\n", "noise line\n".repeat(1000)),
                None,
            ),
        );
        assert!(
            message.chars().count() <= MAX_FAILURE_BYTES + 256,
            "{}",
            message.chars().count()
        );
        assert!(message.ends_with("broken.png: unreadable"), "{message}");
        assert!(
            message.contains("bytes omitted"),
            "the cut says so: {message}"
        );
    }
}

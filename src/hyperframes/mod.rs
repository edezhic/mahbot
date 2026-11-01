//! HyperFrames (`hyperframes`, Apache-2.0): the renderer behind the product's video tool. It seeks
//! a paused HTML timeline in a headless Chrome, frame by frame, and hands the frames to FFmpeg. The
//! package comes from npm and belongs to the product: the managed bun installs it into the
//! product's configuration directory on first use, and the start-time task brings an installed copy
//! to the newest release. Nothing is fetched until the tool is used, and no failure ever removes a
//! copy that already works. Every invocation is that bun, by absolute path, given the vendor's
//! `mjs` entry first. The vendor's npm shim is never run: it honours its `#!/usr/bin/env node`
//! shebang and would start a real Node, which the product never installs. The renderer gets the
//! product's own tools instead of the host's: its Chrome for Testing headless shell ([`browser`],
//! downloaded once on the first render) and the product's ffmpeg/ffprobe pair ([`crate::ffmpeg`]),
//! each by absolute path. Telemetry is off on both channels the vendor honours, the vendor's skill
//! tree is never installed or registered, and its update machinery is left alone since the
//! start-time task keeps a copy current ([`install`]). Only `render` and `lint` are reachable, so
//! no vendor subcommand can reach the host through this module. Rendered pages do reach the network
//! — a CDN script, a web font, a remote image is fetched while the composition loads — and nothing
//! blocks that: it is the page's own traffic. Every run gets the product's install directory as its
//! working directory, never the owner's workspace: bun would load `<cwd>/.env` and
//! `<cwd>/bunfig.toml` from it otherwise, and the vendor reads `<cwd>/.env` itself. A composition
//! is always an absolute path. A render has no wall-clock ceiling: its length is the composition's.
//! The memory watchdog, capture bounds and leftover policy every other run is held to still apply
//! ([`crate::tools::shell`]). A render's verdict is its artifact: a file the product's own ffprobe
//! reads is the result — whatever the exit status, output size or printed envelope — and a run that
//! wrote no such file is a failure carrying the renderer's diagnosis. What the product or the
//! renderer had to say afterwards (a stop, a bad end, a failed cleanup, an unanswerable probe) is
//! passed on as a note beside it, so none of it costs the agent a video.

mod browser;
mod install;

use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::tools::shell::ProgramOutcome;

/// The container a render writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Mp4,
    Webm,
    Mov,
    Gif,
}

impl Format {
    pub(crate) const ALL: [Self; 4] = [Self::Mp4, Self::Webm, Self::Mov, Self::Gif];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|format| format.spelling() == value)
    }

    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::Mov => "mov",
            Self::Gif => "gif",
        }
    }
}

/// Encode quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quality {
    Draft,
    Looks,
    Delivery,
}

impl Quality {
    pub(crate) const ALL: [Self; 3] = [Self::Draft, Self::Looks, Self::Delivery];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|quality| quality.spelling() == value)
    }

    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Looks => "looks",
            Self::Delivery => "delivery",
        }
    }
}

/// One render request, all paths absolute.
pub(crate) struct RenderRequest<'a> {
    /// The project directory handed to the renderer.
    pub project: &'a Path,
    /// The composition inside `project`; `None` renders its `index.html`.
    pub composition: Option<&'a str>,
    /// The caller has already reserved this name with an empty file, so the renderer's
    /// overwrite replaces the reservation and nothing else.
    pub output: &'a Path,
    pub format: Format,
    pub quality: Quality,
    pub fps: Option<&'a str>,
    pub resolution: Option<&'a str>,
}

/// What a run needs: the runtime that starts the vendor, and the extra environment the
/// vendor gets over the product's internal one. A render also keeps its browser selected
/// for as long as it runs ([`browser::Shell`]).
struct Runtime {
    bun: PathBuf,
    extra: Vec<(OsString, OsString)>,
    /// Never read: held so the browser's read guard outlives the run, keeping the cache
    /// from removing the build meanwhile.
    shell: Option<browser::Shell>,
}

/// What every run resolves first: the managed bun, the vendor package, and the variables
/// the vendor gets over the product's reduced environment. A static check stops here — no
/// browser, no encode.
async fn runtime() -> Result<Runtime, String> {
    let bun = crate::tools::bun::bun_binary_path()
        .ok_or_else(|| "the product's managed bun runtime is not installed".to_string())?;
    install::ensure_installed()
        .await
        .map_err(|reason| format!("the renderer could not be installed: {reason}"))?;
    // The vendor honours both spellings of the opt-out; its skill tree is never installed
    // or refreshed by us.
    let extra = vec![
        ("HYPERFRAMES_NO_TELEMETRY".into(), "1".into()),
        ("DO_NOT_TRACK".into(), "1".into()),
        ("HYPERFRAMES_SKIP_SKILLS".into(), "1".into()),
    ];
    Ok(Runtime {
        bun,
        extra,
        shell: None,
    })
}

/// [`runtime`] plus what a render needs: the product's ffmpeg pair and the browser it
/// captures in, resolved in the order a missing piece should be reported.
async fn render_runtime() -> Result<Runtime, String> {
    let mut runtime = runtime().await?;
    let bin = crate::ffmpeg::binaries()
        .await
        .map_err(|reason| format!("the product's own FFmpeg payload is unavailable: {reason}"))?;
    let shell = browser::ensure_shell()
        .await
        .map_err(|unavailable| match unavailable {
            browser::Unavailable::Missing(reason) => {
                format!("the renderer's browser is unavailable: {reason}")
            }
            browser::Unavailable::Unchecked(reason) => {
                format!("the renderer's browser could not be checked: {reason}")
            }
        })?;
    runtime.extra.push((
        "HYPERFRAMES_FFMPEG_PATH".into(),
        bin.ffmpeg.into_os_string(),
    ));
    runtime.extra.push((
        "HYPERFRAMES_FFPROBE_PATH".into(),
        bin.ffprobe.into_os_string(),
    ));
    runtime.extra.push((
        "HYPERFRAMES_BROWSER_PATH".into(),
        shell.binary.clone().into_os_string(),
    ));
    runtime.shell = Some(shell);
    Ok(runtime)
}

/// Scratch space for one download or install: inside the product's directory, so the
/// rename that places the result stays on one filesystem. Removed when the guard drops.
fn scratch(dir: &Path) -> Result<tempfile::TempDir, String> {
    tempfile::Builder::new()
        .prefix(".install-")
        .tempdir_in(dir)
        .map_err(|e| {
            format!(
                "the temporary directory for the download could not be created ({})",
                e.kind()
            )
        })
}

/// The ceiling for one download or install: the package install and the browser archive's
/// fetch share it.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_mins(30);

/// Twice [`DOWNLOAD_TIMEOUT`], so a scratch tree older than this cannot be a download still
/// running.
const STALE_SCRATCH: Duration = DOWNLOAD_TIMEOUT.saturating_mul(2);

/// Remove the scratch trees a killed download left in `dir`, past [`STALE_SCRATCH`] so a
/// running download survives.
fn reclaim_scratch(dir: &Path) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let abandoned = name.starts_with(".install-")
            && entry
                .metadata()
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > STALE_SCRATCH);
        if abandoned {
            crate::util::managed_bin::remove_any(&entry.path());
        }
    }
}

/// Remove the `<something>.old` trees a swap set aside and could not remove itself — on
/// Windows the copy may still be in use. Only a caller whose exclusion serialises every swap
/// creating one may call this: an aside exists between a swap's two renames, where the
/// restore still needs it.
fn reclaim_aside(dir: &Path) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let aside = Path::new(&name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("old"));
        if aside {
            crate::util::managed_bin::remove_any(&entry.path());
        }
    }
}

/// Every invocation carries the product's script guards before the vendor's entry point
/// ([`crate::tools::bun::SCRIPT_GUARDS`]): no dependency install (a missing package must
/// fail rather than be fetched mid-render) and no `.env`. The program is the managed bun,
/// run in the product's install directory (see the module docs). `timeout` is the whole
/// run's ceiling: `None` for a render.
async fn run(
    bun: &Path,
    args: &[String],
    extra: &[(OsString, OsString)],
    timeout: Option<Duration>,
) -> Result<ProgramOutcome, String> {
    let cwd = install::dir()?;
    let guarded: Vec<String> = crate::tools::bun::SCRIPT_GUARDS
        .iter()
        .map(|flag| (*flag).to_string())
        .chain(args.iter().cloned())
        .collect();
    Ok(
        crate::tools::shell::run_internal_program_outcome(&cwd, bun, &guarded, extra, timeout)
            .await,
    )
}

/// What failed, in this module's voice, when a run fails without the renderer saying anything
/// structured, and its sibling for a static check.
const RENDER: &str = "the render";
const CHECK: &str = "the check";

/// The composition as the vendor's argument parser sees it: a relative path, so a file named
/// `-scene.html` cannot be read as a flag. `./` cannot be mistaken for one.
fn composition_argument(composition: &str) -> String {
    if composition.starts_with('-') {
        format!("./{composition}")
    } else {
        composition.to_string()
    }
}

/// A render that produced a file: what the probe made of it, and `notes` — what the agent is
/// owed beside it (a stop, a failure reported after the file was written, or the fact that
/// this machine could not check it). None of them costs the agent the file.
#[derive(Debug)]
pub(crate) struct Rendered {
    /// `None` when the probe could not answer — the agent gets the file and a note says it
    /// went unchecked.
    pub(crate) info: Option<Video>,
    /// One line per note, in [`delivery_notes`] order.
    pub(crate) notes: Option<String>,
}

/// A file the product can play as a video: its duration, and the upright size a render is
/// reported at. A file without a size never becomes one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Video {
    pub(crate) duration: Duration,
    pub(crate) size: (u32, u32),
}

/// What the product's probe made of the file a render left.
enum Artifact {
    /// A player could open it, with the facts read out of it.
    Plays(Video),
    /// It is there and nothing could open it: no video came out of this render.
    Unplayable,
    /// The machine, not the file, could not answer. The file is kept and the reason is the
    /// note beside it.
    Unchecked(String),
}

/// Render one composition into `request.output`. A failed call leaves the file the render was
/// writing behind, so the caller that reserved the name removes it — this layer reports why.
pub(crate) async fn render(request: RenderRequest<'_>) -> Result<Rendered, String> {
    let runtime = render_runtime().await?;
    let (_held, entry) = install::hold_for_run().await?;
    let mut args = vec![
        entry.to_string_lossy().into_owned(),
        "render".to_string(),
        request.project.to_string_lossy().into_owned(),
    ];
    if let Some(composition) = request.composition {
        args.push("-c".to_string());
        args.push(composition_argument(composition));
    }
    args.push("-o".to_string());
    args.push(request.output.to_string_lossy().into_owned());
    args.push("--format".to_string());
    args.push(request.format.spelling().to_string());
    args.push("--quality".to_string());
    args.push(request.quality.spelling().to_string());
    if let Some(fps) = request.fps {
        args.push("--fps".to_string());
        args.push(fps.to_string());
    }
    if let Some(resolution) = request.resolution {
        args.push("--resolution".to_string());
        args.push(resolution.to_string());
    }
    args.push("--json".to_string());

    let outcome = run(&runtime.bun, &args, &runtime.extra, None).await?;
    // The file decides, not the run: it existed before the run started (reserved), so it is a
    // result only if it holds something that plays; the exit status, output size and printed
    // envelope shape the failure instead.
    let artifact = artifact(request.output).await;
    render_verdict(artifact, &outcome, request.output)
}

/// One sentence per way the run ended badly, in [`delivery_notes`] order; a failure the renderer
/// names twice is said once.
fn delivery_notes(outcome: &ProgramOutcome) -> Option<String> {
    let late = late_failure(&outcome.output);
    // What the notes name, so a late complaint repeating it adds nothing.
    let mut named = None;
    let mut notes = Vec::new();
    if outcome.stopped() {
        notes.push(format!(
            "the product stopped the render after the file was written: {}",
            outcome.stop_reason()
        ));
    } else if let Some(failure) = failure_envelope(&outcome.output) {
        let entry = after_write_failure(&failure);
        // One failure, one sentence: a complaint naming the same failure is not repeated, and
        // the fuller telling is kept. An entry that carries a code and no message is the telling
        // only when there is no complaint to stand in its place.
        let fuller = late.as_ref().filter(|late| {
            same_failure(&entry.message, &late.message)
                && words(&late.message).len() > words(&entry.message).len()
        });
        let entry_tells = !entry.message.is_empty() || late.is_none();
        if let Some(late) = fuller {
            named = Some(late.message.clone());
            notes.push(late.sentence.clone());
        } else if entry_tells {
            named = Some(entry.message);
            notes.push(entry.sentence);
        }
    } else if !outcome.success {
        notes.push(format!(
            "the file was written; the renderer's run ended: {}",
            outcome.detail
        ));
    }
    if let Some(late) = late
        && !named.is_some_and(|named| same_failure(&named, &late.message))
    {
        notes.push(late.sentence);
    }
    (!notes.is_empty()).then(|| notes.join("\n"))
}

/// Whether two tellings name the same failure: the shorter one's words are the longer one's
/// own opening or close, whole words only — the vendor prints a post-render failure once as its
/// entry and again in a complaint, and either telling may wrap the same message.
fn same_failure(one: &str, other: &str) -> bool {
    let (one, other) = (words(one), words(other));
    let (shorter, longer) = if one.len() <= other.len() {
        (one, other)
    } else {
        (other, one)
    };
    !shorter.is_empty()
        && (longer.starts_with(shorter.as_slice()) || longer.ends_with(shorter.as_slice()))
}

/// The failure entry the renderer printed after the file was written: its own message, filtered
/// like any other diagnosis, and the one sentence an agent reads for it.
fn after_write_failure(failure: &Envelope) -> PostWrite {
    let message = one_line(&without_remedies(&failure.message));
    let named = failure.headline(&message);
    let named = if named.trim().is_empty() {
        "no diagnosis".to_string()
    } else {
        crate::util::cut_bytes(&named, NOTE_BUDGET_BYTES)
    };
    PostWrite {
        sentence: format!(
            "the file was written; the renderer reported a failure after the write: {named}"
        ),
        message,
    }
}

/// `text` with its line breaks and runs of spaces collapsed into one line.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `text`'s words, lowercased and punctuation aside — the form two tellings are compared in,
/// so a word cut in half never matches.
fn words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// The verdict a finished render earns from its artifact alone: a file that plays is the
/// result, with `notes` for what was said after writing it; anything else is the failure
/// `render_failure` describes, whatever status or output the run came with. A file the probe
/// could not check is kept and noted.
fn render_verdict(
    artifact: Artifact,
    outcome: &ProgramOutcome,
    output: &Path,
) -> Result<Rendered, String> {
    match artifact {
        Artifact::Plays(video) => Ok(Rendered {
            info: Some(video),
            notes: delivery_notes(outcome),
        }),
        Artifact::Unchecked(reason) => {
            let unchecked = format!("this machine could not check the file: {reason}");
            let notes = delivery_notes(outcome);
            Ok(Rendered {
                info: None,
                notes: Some(match notes {
                    Some(note) => format!("{note}\n{unchecked}"),
                    None => unchecked,
                }),
            })
        }
        // Nothing the renderer said matters to a file no player can open: the failure is the
        // diagnosis, and no notes are built.
        Artifact::Unplayable => Err(render_failure(outcome, output)),
    }
}

/// What `output` plays as: a non-empty file the product's ffprobe reads with a video stream. A
/// file it cannot read, or one with no video, is no video; a probe that could not run is the
/// machine's failure to answer.
async fn artifact(output: &Path) -> Artifact {
    let non_empty = tokio::fs::metadata(output)
        .await
        .is_ok_and(|meta| meta.len() > 0);
    if !non_empty {
        return Artifact::Unplayable;
    }
    match crate::ffmpeg::probe(output).await {
        // The machine, not the file: the product's half could not run, or ran out of its bound.
        Err(crate::ffmpeg::FfmpegError::Unavailable(reason)) => Artifact::Unchecked(reason),
        Err(crate::ffmpeg::FfmpegError::TimedOut) => {
            Artifact::Unchecked("the probe did not finish in time".to_string())
        }
        Ok(info) => match info.dimensions {
            Some(size) => Artifact::Plays(Video {
                duration: info.duration,
                size,
            }),
            None => Artifact::Unplayable,
        },
        // Every other answer is a fact about the file — no video stream, nothing decodable.
        Err(_) => Artifact::Unplayable,
    }
}

/// Why a render produced no playable file. The renderer's own failure is the diagnosis where it
/// printed one; without it, the run's verdict and the tail of its output are.
fn render_failure(outcome: &ProgramOutcome, output: &Path) -> String {
    if !outcome.success {
        // A failed run: the vendor's envelope, or — for a run the product stopped — the stop's
        // reason, which no envelope can explain.
        return run_failure(outcome, RENDER, None);
    }
    if let Some(failure) = failure_envelope(&outcome.output) {
        return envelope_failure(&failure, RENDER);
    }
    let mut message = format!(
        "{RENDER} failed: the renderer exited clean but wrote no playable file at {}",
        output.display()
    );
    let tail = without_remedies(&failure_tail(&outcome.output));
    if !tail.is_empty() {
        let _ = write!(message, "\n{tail}");
    }
    crate::util::cut_bytes(&message, MESSAGE_BUDGET_BYTES)
}

/// What a composition's static check found.
pub(crate) struct LintReport {
    pub(crate) errors: usize,
    pub(crate) warnings: usize,
    pub(crate) infos: usize,
    pub(crate) files: usize,
    pub(crate) findings: Vec<Finding>,
}

pub(crate) struct Finding {
    /// `error`, `warning` or `info`.
    pub(crate) severity: String,
    pub(crate) code: String,
    /// The vendor's diagnosis, without the advice its own reader is given
    /// ([`without_remedies`]).
    pub(crate) message: String,
    /// The vendor's fix, filtered the same way.
    pub(crate) fix: Option<String>,
    pub(crate) file: Option<String>,
    pub(crate) line: Option<u64>,
}

/// Check one composition (a file or a project directory) without rendering.
pub(crate) async fn lint(target: &Path) -> Result<LintReport, String> {
    let runtime = runtime().await?;
    let (_held, entry) = install::hold_for_run().await?;
    let args = vec![
        entry.to_string_lossy().into_owned(),
        "lint".to_string(),
        target.to_string_lossy().into_owned(),
        "--json".to_string(),
    ];
    // A check reads and answers, so it is held to the ordinary ceiling — a wedged check must
    // not hold the tool call, or the package's read guard, forever.
    let ceiling = crate::tools::shell::DEFAULT_SHELL_TIMEOUT;
    let outcome = run(&runtime.bun, &args, &runtime.extra, Some(ceiling)).await?;
    // A composition with errors exits non-zero and still prints its report: the exit status is
    // a verdict, not a failed run.
    if let Some(report) = lint_report(&outcome.output) {
        return Ok(report);
    }
    // A run the product stopped has no verdict of its own, so it is reported as the stop it was
    // — the vendor's words, printed before it, cannot explain it.
    if outcome.stopped() {
        return Err(run_failure(&outcome, CHECK, Some(ceiling)));
    }
    // A failure envelope is the diagnosis whatever the exit status: the vendor answers a clean
    // run with one when it could not check anything.
    if let Some(failure) = failure_envelope(&outcome.output) {
        return Err(envelope_failure(&failure, CHECK));
    }
    // A clean exit with nothing readable is neither a clean report nor a failed run, and "the
    // check failed: exit status 0" would contradict itself.
    if outcome.success {
        return Err(format!(
            "{CHECK} exited clean but printed no report this tool could read"
        ));
    }
    Err(run_failure(&outcome, CHECK, Some(ceiling)))
}

/// The vendor's JSON answer out of a run's combined output, where stdout arrives before
/// stderr's progress noise. The answer is one document — a single-line failure envelope or a
/// pretty-printed report — so every line that starts one is tried in order: the first that
/// parses as an object is the answer.
fn json_answer(output: &str) -> Option<serde_json::Value> {
    let mut at = 0;
    for line in output.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with('{') {
            let start = at + (line.len() - trimmed.len());
            let answer = serde_json::Deserializer::from_str(&output[start..])
                .into_iter::<serde_json::Value>()
                .next();
            if let Some(Ok(value)) = answer
                && value.is_object()
            {
                return Some(value);
            }
        }
        at += line.len();
    }
    None
}

/// [`LintReport`] read off the vendor's answer, `None` when the output holds no report.
fn lint_report(output: &str) -> Option<LintReport> {
    let answer = json_answer(output)?;
    let failed = answer.get("ok").and_then(serde_json::Value::as_bool) == Some(false)
        && answer
            .get("filesScanned")
            .and_then(serde_json::Value::as_u64)
            == Some(0);
    if answer.get("error").is_some() || failed {
        return None;
    }
    let is_report = [
        "errorCount",
        "warningCount",
        "infoCount",
        "filesScanned",
        "findings",
    ]
    .iter()
    .any(|key| answer.get(key).is_some());
    if !is_report {
        return None;
    }
    let count = |key: &str| {
        usize::try_from(
            answer
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or_default()
    };
    let findings = answer
        .get("findings")
        .and_then(serde_json::Value::as_array)
        .map(|findings| {
            findings
                .iter()
                .map(|finding| Finding {
                    severity: text(finding, "severity").unwrap_or_default(),
                    code: text(finding, "code").unwrap_or_default(),
                    // Both the message and the fix are written for the vendor's own reader: what
                    // sends that reader to its CLI or to an install is dropped, as a failure's hint
                    // is.
                    message: text(finding, "message")
                        .map(|message| without_remedies(&message))
                        .unwrap_or_default(),
                    fix: text(finding, "fixHint")
                        .map(|fix| without_remedies(&fix))
                        .filter(|fix| !fix.is_empty()),
                    file: text(finding, "file"),
                    line: finding.get("line").and_then(serde_json::Value::as_u64),
                })
                .collect()
        })
        .unwrap_or_default();
    Some(LintReport {
        errors: count("errorCount"),
        warnings: count("warningCount"),
        infos: count("infoCount"),
        files: count("filesScanned"),
        findings,
    })
}

/// One string field of a vendor object.
fn text(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// The vendor's own remedy sentence for a missing piece never reaches the model: the piece is
/// the product's, so the sentence is ours. Only the two classes the envelope words plainly are
/// translated; anything else stands.
fn machine_side_gap(message: &str) -> Option<&'static str> {
    let lower = message.to_ascii_lowercase();
    if lower.contains("encoder unavailable") {
        return Some(
            "the product's own FFmpeg payload has no encoder for this format on this machine, \
             which is a machine-side gap rather than a composition problem",
        );
    }
    if lower.contains("chrome") && (lower.contains("launch") || lower.contains("not found")) {
        return Some(
            "the product's own browser build could not start, which is a machine-side gap \
             rather than a composition problem",
        );
    }
    None
}

/// The vendor's own failure object.
struct Envelope {
    code: Option<String>,
    message: String,
    hint: Option<String>,
}

impl Envelope {
    /// The entry as one line — `<code>: <message>`, the code alone when the message was nothing
    /// but instructions, the message alone without a code — from `message` already filtered.
    fn headline(&self, message: &str) -> String {
        match self.code.as_deref().filter(|code| !code.is_empty()) {
            Some(code) if message.is_empty() => code.to_string(),
            Some(code) => format!("{code}: {message}"),
            None => message.to_string(),
        }
    }
}

/// The `error` object of the vendor's failure envelope, when that is what it printed.
fn failure_envelope(output: &str) -> Option<Envelope> {
    let answer = json_answer(output)?;
    if answer.get("ok").and_then(serde_json::Value::as_bool) != Some(false) {
        return None;
    }
    let error = answer.get("error")?;
    Some(Envelope {
        code: text(error, "code"),
        message: text(error, "message")?,
        hint: text(error, "hint"),
    })
}

/// A run the product stopped leads with the stop's reason — and one stopped by `ceiling` carries
/// the `timeout:` class every product run's stop carries. Otherwise the vendor's diagnosis leads
/// ([`envelope_failure`]), else the runner's verdict plus the tail of the output.
fn run_failure(outcome: &ProgramOutcome, subject: &str, ceiling: Option<Duration>) -> String {
    if let (true, Some(ceiling)) = (outcome.timed_out(), ceiling) {
        return crate::tools::shell::timeout_sentence(subject, ceiling);
    }
    // A stop is the product's own reason for ending the run and the vendor never saw it:
    // whatever it printed before cannot be its verdict on the failure.
    let envelope = (!outcome.stopped())
        .then(|| failure_envelope(&outcome.output))
        .flatten();
    let Some(failure) = envelope else {
        let tail = without_remedies(&failure_tail(&outcome.output));
        let text = if tail.is_empty() {
            format!("{subject} failed: {}", outcome.detail)
        } else {
            format!("{subject} failed: {}\n{tail}", outcome.detail)
        };
        return crate::util::cut_bytes(&text, MESSAGE_BUDGET_BYTES);
    };
    envelope_failure(&failure, subject)
}

/// The vendor's own failure, in the words an agent needs. `subject` is used only where the
/// vendor left nothing to quote.
///
/// `code` and `message` lead, then `hint` — where a throwing composition script, a missing asset
/// or a mismatched resolution is actually explained. Remedy lines are removed
/// ([`without_remedies`]) and a machine-side gap is named in this module's voice and kept whole.
///
/// The result is held to [`MESSAGE_BUDGET_BYTES`], and the hint pays for itself first: it is the
/// one line that says what to do, so the message part gives way first.
fn envelope_failure(failure: &Envelope, subject: &str) -> String {
    // The vendor writes the same prose into the message as into the hint, so both are
    // filtered; a message that was nothing but an instruction leaves the code alone.
    let message = without_remedies(&failure.message);
    let head = failure.headline(&message);
    let diagnosis = format!(
        "{} {}",
        failure.message,
        failure.hint.as_deref().unwrap_or_default()
    );
    let note = machine_side_gap(&diagnosis)
        .map(|note| format!(" — {note}"))
        .unwrap_or_default();
    let hint = failure
        .hint
        .as_deref()
        .filter(|hint| !hint.trim().is_empty())
        .map(without_remedies)
        .filter(|hint| !hint.is_empty())
        .map(|hint| {
            let hint = crate::util::cut_bytes(&hint, HINT_BUDGET_BYTES);
            format!("\n  hint: {hint}")
        })
        .unwrap_or_default();
    let head = crate::util::cut_bytes(
        &head,
        MESSAGE_BUDGET_BYTES.saturating_sub(note.len() + hint.len()),
    );
    let message = format!("{head}{note}{hint}");
    if message.trim().is_empty() {
        // Everything the vendor said was its own remedy, and an empty failure tells the agent
        // nothing: this much is true of any envelope that reaches here.
        return format!("{subject} reported a failure without a diagnosis");
    }
    message
}

/// The tail of a run's output an agent needs: the last few non-empty lines, without the
/// vendor's progress noise or the structured answer it printed (a raw JSON line is not prose).
fn failure_tail(output: &str) -> String {
    const LINES: usize = 8;
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('[') && !line.starts_with("(node:"))
        .filter(|line| !line.starts_with('{'))
        .filter(|line| !line.starts_with(SPURIOUS))
        .collect();
    lines[lines.len().saturating_sub(LINES)..].join("\n")
}

/// What the renderer prints on a composition whose folder has no `index.html` even when it
/// rendered the file it was told to — a line that reads as the cause of a failure it had
/// nothing to do with. This tool resolves the project itself, so the render it asks for never
/// lacks one.
const SPURIOUS: &str = "No composition found in ";

/// A failure the renderer reported after it had already written the file — a step that failed
/// behind a delivered render, or the failure entry printed beside it. The message is kept apart
/// so a failure both shapes name is not said twice ([`delivery_notes`]).
struct PostWrite {
    /// The vendor's message, filtered like any other diagnosis and on one line.
    message: String,
    /// The note an agent reads.
    sentence: String,
}

/// The step the renderer reported failing after it had written the file: its own post-render
/// cleanup, or a late error it survives because the render was delivered. The file is the result
/// either way, so this is a note, never a verdict.
fn late_failure(output: &str) -> Option<PostWrite> {
    /// The vendor's wordings for "after the result was delivered", and where its message
    /// starts in each line.
    const MARKERS: [&str; 2] = [
        "failed (render already succeeded): ",
        "delivered its result): ",
    ];
    let line = output
        .lines()
        .rev()
        .find(|line| MARKERS.iter().any(|marker| line.contains(marker)))?;
    let rest = MARKERS.iter().find_map(|marker| line.split_once(marker))?.1;
    let message = one_line(&without_remedies(rest));
    if message.is_empty() {
        return None;
    }
    Some(PostWrite {
        sentence: format!(
            "the file was written; a step after the render failed: {}",
            crate::util::cut_bytes(&message, NOTE_BUDGET_BYTES)
        ),
        message,
    })
}

/// The separators a clause of the vendor's text ends at; `" ("` opens a parenthetical
/// ([`parenthetical_end`]). The advice may follow any of them, so the split cannot decide
/// whether a diagnosis survives.
const CLAUSE_ENDS: [&str; 7] = [". ", "; ", ", ", " — ", " - ", ": ", " ("];

/// What a dropped clause's junction may leave at either end of what is kept: trimmed only
/// where a clause was actually removed.
const SEPARATORS: [char; 6] = [' ', ';', ',', ':', '—', '-'];

/// The openers of a clause that conditions what follows, so the comma behind it is not a
/// boundary: "If the clip fails to play, run the vendor's doctor" is one piece of advice.
const CONDITIONALS: [&str; 3] = ["if ", "when ", "unless "];

/// The words making `hyperframes <word>` a command line rather than the product's name in a
/// sentence. None is reachable through this tool, so a hint naming one sends the reader to a CLI
/// this product does not expose. The list is the root of the CLI as HyperFrames 0.8.216 published
/// it, so a drift shows up here.
const VENDOR_COMMANDS: [&str; 49] = [
    "add",
    "audio",
    "auth",
    "beats",
    "benchmark",
    "browser",
    "capture",
    "catalog",
    "catch-up",
    "check",
    "clean",
    "cloud",
    "cloudrun",
    "compare",
    "compositions",
    "docs",
    "doctor",
    "events",
    "feedback",
    "figma",
    "grade-compare",
    "history",
    "info",
    "init",
    "inspect",
    "keyframes",
    "lambda",
    "layout",
    "lint",
    "media-treatment",
    "media-use",
    "models",
    "normalize-audio",
    "open",
    "play",
    "present",
    "preview",
    "publish",
    "remove-background",
    "render",
    "skills",
    "snapshot",
    "telemetry",
    "timeline",
    "transcribe",
    "tts",
    "upgrade",
    "usage",
    "validate",
];

/// The vendor's own knobs, printed as a line of their own: `Env: …` / `CLI: …`.
const KNOB_LINES: [&str; 2] = ["env:", "cli:"];

/// The verbs a clause opens with when it instructs the reader to act.
const INSTRUCTIONS: [&str; 6] = [
    "install ",
    "please ",
    "reinstall ",
    "run ",
    "try ",
    "scaffold ",
];

/// The runners that fetch something, wherever in a clause they sit: each is an instruction by
/// itself, never part of a description.
const REMEDIES: [&str; 9] = [
    "apt install",
    "apt-get install",
    "brew install",
    "bun add",
    "bunx ",
    "npm i",
    "npx ",
    "yarn add",
    "--docker",
];

/// `text` without the vendor's instructions, one clause at a time ([`is_advice`]) — a
/// description of what happened is never filtered.
fn without_remedies(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(without_advice)
        .collect::<Vec<_>>()
        .join("\n")
}

/// One line without the instruction clauses, their separators going with them: a hint that says
/// what is wrong and then suggests a container keeps the diagnosis, and one that leads with the
/// suggestion keeps what follows. A boundary is any of [`CLAUSE_ENDS`]; a parenthetical goes only
/// as a whole, because half of one is not a sentence.
///
/// A line left with nothing but instructions is answered whole (`Some(line)` for the mixed case,
/// `None` when every clause was advice). Only a clause actually dropped takes punctuation.
fn without_advice(line: &str) -> Option<String> {
    // A knob line names a setting of the vendor's own and nothing else.
    let beginning = line.trim_start().to_ascii_lowercase();
    if KNOB_LINES.iter().any(|knob| beginning.starts_with(knob)) {
        return None;
    }
    let mut kept = String::new();
    let mut gap = false;
    let mut dropped = false;
    let mut start = 0;
    let mut from = 0;
    while let Some((at, end)) = next_clause_end(line, from) {
        if end == " (" {
            // A parenthetical goes only as a whole: its opener belongs to the sentence around
            // it, so dropping just the advice inside would leave a missing bracket or two words
            // run together. One that is advice is dropped with both brackets.
            match parenthetical_end(line, at) {
                Some(close) if is_advice(&line[at + end.len()..close]) => {
                    dropped = true;
                    if !is_advice(&line[start..at]) {
                        join_kept(&mut kept, &mut gap, &line[start..at]);
                    }
                    gap = true;
                    start = close + 1;
                    from = start;
                }
                // A parenthetical that is not advice is kept whole, boundaries inside it
                // included: what it holds is the vendor's description.
                kept_parenthetical => {
                    from = kept_parenthetical.map_or(at + end.len(), |close| close + 1);
                }
            }
            continue;
        }
        from = at + end.len();
        if (end == ", " && is_conditional(&line[start..at]))
            || (end == ": " && is_cue(&line[start..at]))
        {
            continue;
        }
        if is_advice(&line[start..at]) {
            dropped = true;
            gap = true;
        } else {
            join_kept(&mut kept, &mut gap, &line[start..at + end.len()]);
        }
        start = at + end.len();
    }
    if is_advice(&line[start..]) {
        dropped = true;
        gap = true;
    } else {
        join_kept(&mut kept, &mut gap, &line[start..]);
    }
    if !dropped {
        // Nothing was removed, so nothing is rewritten: the line is the vendor's own.
        return Some(line.to_string());
    }
    // A separator is trimmed only where a dropped clause left it: the closing bracket and
    // punctuation of a kept sentence are the sentence's own, and the front of the line is
    // trimmed by [`join_kept`] when that first junction was a drop's.
    let kept = if gap {
        kept.trim_end_matches(SEPARATORS)
    } else {
        kept.trim()
    };
    if kept.is_empty() {
        // Every clause was an instruction — unless the line only carries one inside a sentence
        // the filter could not split, where the line as printed is the only answer that keeps
        // its diagnosis.
        return (!opens_as_instruction(&opening(line))).then(|| line.to_string());
    }
    // What is left must read as the vendor's own line, not a piece of one: an unbalanced
    // bracket means a clause inside a parenthesis was dropped, so the whole line is the only
    // form that keeps the sentence intact.
    if is_fragment(kept) || !balanced(kept) {
        return Some(line.to_string());
    }
    Some(kept.to_string())
}

/// Append `text` to what a line keeps. A dropped clause takes its junction with it, and the
/// one space the drop took is put back: two kept pieces must never run together. A piece that
/// opens with punctuation needs no space.
fn join_kept(kept: &mut String, gap: &mut bool, text: &str) {
    let text = if *gap {
        text.trim_start_matches(SEPARATORS)
    } else {
        text
    };
    if text.is_empty() {
        return;
    }
    let word_ends = |kept: &str| {
        kept.ends_with(|c: char| c.is_alphanumeric() || matches!(c, ')' | '"' | '\'' | '`'))
    };
    let word_begins = |text: &str| {
        text.starts_with(|c: char| c.is_alphanumeric() || matches!(c, '(' | '"' | '\'' | '`'))
    };
    if *gap && word_ends(kept) && word_begins(text) {
        kept.push(' ');
    }
    kept.push_str(text);
    *gap = false;
}

/// Whether every bracket `text` carries was opened and closed inside it: a `)` without an
/// opener is as damaged as an opener without its closer.
fn balanced(text: &str) -> bool {
    let mut depth = 0_u32;
    for character in text.chars() {
        match character {
            '(' => depth += 1,
            ')' => match depth.checked_sub(1) {
                Some(shallower) => depth = shallower,
                None => return false,
            },
            _ => {}
        }
    }
    depth == 0
}

/// Whether a surviving clause is only part of one, so the whole line is what the agent should
/// see: a conjunction or preposition at its end promises something the dropped clause carried.
fn is_fragment(left: &str) -> bool {
    const CONNECTORS: [&str; 13] = [
        "and", "or", "but", "then", "with", "to", "for", "the", "run", "try", "use", "install",
        "via",
    ];
    left.split_whitespace()
        .next_back()
        .map(|word| word.to_ascii_lowercase().trim_end_matches(',').to_string())
        .is_some_and(|word| CONNECTORS.contains(&word.as_str()))
}

/// Whether a clause is only a cue for what follows — `Run: …`, `Try: …` — so the colon
/// behind it is not a boundary.
fn is_cue(clause: &str) -> bool {
    const CUES: [&str; 5] = ["run", "try", "use", "install", "hint"];
    let clause = clause.trim().to_ascii_lowercase();
    CUES.contains(&clause.as_str())
}

/// Whether a clause conditions what follows rather than stating something of its own
/// ([`CONDITIONALS`]).
fn is_conditional(clause: &str) -> bool {
    let clause = clause.trim_start().to_ascii_lowercase();
    CONDITIONALS.iter().any(|opener| clause.starts_with(opener))
}

/// The first clause boundary at or after `from`, and the separator it is.
fn next_clause_end(line: &str, from: usize) -> Option<(usize, &'static str)> {
    CLAUSE_ENDS
        .iter()
        .filter_map(|end| line[from..].find(end).map(|at| (from + at, *end)))
        .min_by_key(|(at, _)| *at)
}

/// Where the parenthetical opened at `at` closes, `None` when the line never closes it.
fn parenthetical_end(line: &str, at: usize) -> Option<usize> {
    let opened = at + " (".len();
    line[opened..].find(')').map(|close| opened + close)
}

/// Whether one clause is an instruction — it tells the reader to run or install something
/// ([`INSTRUCTIONS`], [`REMEDIES`], or the vendor's command line). A clause that describes what
/// happened is kept, even one mentioning an install or the product's name.
fn is_advice(clause: &str) -> bool {
    let clause = opening(clause);
    opens_as_instruction(&clause) || carries_remedy(&clause)
}

/// Whether text opens as an instruction: the verbs a reader is told to act on, or a vendor
/// command line. Positional, unlike [`carries_remedy`] — a line merely mentioning a command is
/// a description.
fn opens_as_instruction(clause: &str) -> bool {
    INSTRUCTIONS.iter().any(|opener| clause.starts_with(opener))
        || REMEDIES.iter().any(|runner| clause.starts_with(runner))
        || opens_with_command(clause)
}

/// Whether text carries a runner that fetches something. A remedy is a command, so it counts only
/// where its own word ends: `npm i` is an instruction, `npm install` names the vendor's installer.
fn carries_remedy(clause: &str) -> bool {
    REMEDIES.iter().any(|remedy| {
        clause.match_indices(remedy).any(|(at, _)| {
            remedy.ends_with(' ')
                || clause[at + remedy.len()..]
                    .chars()
                    .next()
                    .is_none_or(|next| !next.is_ascii_alphanumeric())
        })
    })
}

/// A clause as an instruction would read it: lowercased, without its punctuation, leading
/// conjunction, or condition — "or scaffold with …" and "If the clip fails, run …" are both
/// instructions to the verb that follows.
fn opening(clause: &str) -> String {
    let clause = clause
        .trim()
        .trim_start_matches(['(', '`', ' '])
        .to_ascii_lowercase();
    let mut clause = strip_connectors(clause);
    // The cue is a label for the instruction behind it, not the instruction's own verb.
    if let Some((cue, instruction)) = clause.split_once(": ")
        && is_cue(cue)
    {
        clause = strip_connectors(instruction.to_string());
    }
    if CONDITIONALS.iter().any(|opener| clause.starts_with(opener))
        && let Some((_, consequence)) = clause.split_once(", ")
    {
        clause = strip_connectors(consequence.to_string());
    }
    clause
}

/// `clause` without the conjunctions it may open with.
fn strip_connectors(mut clause: String) -> String {
    'outer: loop {
        for connector in ["or ", "and ", "then ", "so "] {
            if let Some(rest) = clause.strip_prefix(connector) {
                clause = rest.to_string();
                continue 'outer;
            }
        }
        return clause;
    }
}

/// Whether a clause sends the reader to the vendor's command line: `hyperframes` plus one of
/// its subcommands. Naming the product mid-sentence — "the hyperframes package" — is not
/// naming a command.
fn opens_with_command(clause: &str) -> bool {
    let Some(rest) = clause.strip_prefix("hyperframes ") else {
        return false;
    };
    let word: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    VENDOR_COMMANDS.contains(&word.as_str())
}

/// How much of a failure message an agent is handed: enough to act on, short enough to stay
/// one message.
const MESSAGE_BUDGET_BYTES: usize = 1200;

/// How much of that budget the vendor's `hint` may take: it outranks the message but leaves
/// room for the code and the first line of what went wrong.
const HINT_BUDGET_BYTES: usize = 800;

/// How long the note beside a delivered file may be: one sentence, not a second failure report.
const NOTE_BUDGET_BYTES: usize = 400;

/// The boot task: bring an installed copy to the newest npm release. Nothing is installed when
/// the tool has never been used, and a failure never invalidates the working copy.
pub async fn refresh_on_start() {
    install::refresh_on_start().await;
}

#[cfg(test)]
mod tests {
    use crate::tools::shell::Stop;

    use super::*;

    fn failed(output: &str) -> ProgramOutcome {
        ProgramOutcome {
            success: false,
            detail: "exit status 1".to_string(),
            output: output.to_string(),
            has_output: true,
            stop: None,
        }
    }

    /// A sweep of our own directory must not take a download in flight with it: the window is the
    /// age of the tree. A tree a swap set aside goes at once, once the caller holds the exclusion
    /// its swaps take.
    ///
    /// Unix-only: backdating a directory's mtime needs a `File` on it, which Windows does not open.
    #[cfg(unix)]
    #[test]
    fn a_sweep_takes_the_abandoned_scratch_and_the_aside_and_leaves_the_live_one() {
        let dir = tempfile::tempdir().unwrap();
        for name in [".install-live", ".install-dead", "node_modules.old"] {
            std::fs::create_dir_all(dir.path().join(name)).unwrap();
        }
        let dead = std::fs::File::open(dir.path().join(".install-dead")).unwrap();
        dead.set_times(
            std::fs::FileTimes::new().set_modified(
                std::time::SystemTime::now() - STALE_SCRATCH - Duration::from_mins(1),
            ),
        )
        .unwrap();

        reclaim_scratch(dir.path());
        reclaim_aside(dir.path());

        assert!(
            dir.path().join(".install-live").exists(),
            "a download in flight is not ours to remove"
        );
        assert!(
            !dir.path().join(".install-dead").exists(),
            "an abandoned scratch tree goes"
        );
        assert!(
            !dir.path().join("node_modules.old").exists(),
            "a tree a swap set aside goes with it, whatever it was called"
        );
    }

    #[test]
    fn lint_report_reads_the_pretty_printed_answer() {
        // The vendor pretty-prints a report and then keeps logging to stderr: both arrive in one
        // combined stream, stdout first. The second finding's fix is advice to the vendor's own
        // reader, so it is dropped on the way in.
        let output = "{\n  \"ok\": false,\n  \"errorCount\": 2,\n  \"warningCount\": 1,\n  \
                      \"infoCount\": 0,\n  \"findings\": [\n    {\n      \"code\": \"root_missing_dimensions\",\n      \
                      \"severity\": \"error\",\n      \"message\": \"Root composition is missing data-width.\",\n      \
                      \"fixHint\": \"Set numeric data-width.\",\n      \"file\": \"/w/index.html\",\n      \
                      \"line\": 1\n    },\n    {\n      \"code\": \"viewport_mismatch\",\n      \
                      \"severity\": \"error\",\n      \"message\": \"The viewport does not match.\",\n      \
                      \"fixHint\": \"update the meta viewport to match, or scaffold with `hyperframes init \
                      --resolution portrait`\"\n    }\n  ],\n  \"filesScanned\": 1\n}\n[initSession] noise\n";
        let report = lint_report(output).expect("the report is read");
        assert_eq!(
            (report.errors, report.warnings, report.infos, report.files),
            (2, 1, 0, 1)
        );
        assert_eq!(report.findings.len(), 2);
        assert_eq!(report.findings[0].code, "root_missing_dimensions");
        assert_eq!(report.findings[0].line, Some(1));
        assert_eq!(
            report.findings[0].fix.as_deref(),
            Some("Set numeric data-width.")
        );
        assert_eq!(
            report.findings[1].fix.as_deref(),
            Some("update the meta viewport to match"),
            "the fix stays, the vendor's own command does not"
        );
    }

    #[test]
    fn a_failure_envelope_is_not_read_as_a_clean_report() {
        // What the vendor prints when lint cannot run at all: the report's field names, zeroed,
        // beside an `error`. Reading it as a report would tell the agent "no problems found in 0
        // files" right before a failing render.
        let output = "{\n  \"ok\": false,\n  \"error\": {\"code\": \"runtime_error\", \
                      \"message\": \"No composition found in /w\"},\n  \"findings\": [],\n  \
                      \"errorCount\": 0,\n  \"warningCount\": 0,\n  \"infoCount\": 0,\n  \
                      \"filesScanned\": 0\n}\n";
        assert!(lint_report(output).is_none(), "got: {output}");
        let message = run_failure(&failed(output), "the check", None);
        assert!(
            message.starts_with("runtime_error: No composition found in /w"),
            "got: {message}"
        );
    }

    /// The artifact decides the render in every combination of exit status, output size and
    /// printed envelope: a file that plays is the result, a file no player can open is the
    /// failure the run explains, and a file the probe could not judge is kept with the reason.
    #[test]
    fn the_file_decides_the_render_not_the_run() {
        let unplayable = Path::new("/work/clip.mp4");
        let run = failed("");
        let plays = || {
            Artifact::Plays(Video {
                duration: Duration::from_secs(6),
                size: (1920, 1080),
            })
        };

        let delivered =
            render_verdict(plays(), &run, unplayable).expect("a playable file is a result");
        assert!(
            delivered
                .notes
                .is_some_and(|note| note.contains("exit status 1")),
            "a bad run is noted beside the file it did produce"
        );
        assert_eq!(
            delivered.info.expect("the facts are kept").size,
            (1920, 1080)
        );

        // The vendor's own shape: the file is complete and a later step failed, so the note
        // rides beside the file rather than replacing it.
        let complained = failed(
            "  [hyperframes] Post-render step 'history' failed (render already succeeded): \
             history.json could not be written\n",
        );
        let delivered = render_verdict(plays(), &complained, unplayable).expect("the file stands");
        assert!(
            delivered.notes.unwrap().contains("history.json"),
            "the note names what failed after the write"
        );

        // The machine could not answer: the file is not thrown away, and the note says so even
        // when the renderer had something of its own to say.
        let delivered = render_verdict(
            Artifact::Unchecked("the probe did not finish in time".to_string()),
            &complained,
            unplayable,
        )
        .expect("a file the machine cannot judge is kept");
        assert!(delivered.info.is_none());
        let note = delivered.notes.expect("the note names what happened");
        assert!(note.contains("history.json"), "got: {note}");
        assert!(
            note.contains("could not check the file"),
            "the note says the file went unchecked: {note}"
        );

        assert_eq!(
            render_verdict(Artifact::Unplayable, &run, unplayable).unwrap_err(),
            "the render failed: exit status 1"
        );
    }

    /// A playable file is never thrown away for how the run ended, and every bad end is one
    /// sentence beside it: a product stop, a non-zero exit, or the renderer's own failure entry —
    /// and a failure named twice is said once.
    #[test]
    fn a_delivered_file_carries_how_the_run_ended_badly() {
        let plays = || {
            Artifact::Plays(Video {
                duration: Duration::from_secs(6),
                size: (1280, 720),
            })
        };

        let killed = ProgramOutcome {
            detail: "terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)".to_string(),
            stop: Some(Stop::Memory),
            ..failed("")
        };
        let note = delivery_notes(&killed).expect("the stop is named");
        assert_eq!(
            note,
            "the product stopped the render after the file was written: \
             exceeded memory limit (used ~15.4 GiB of 18.0 GiB)",
            "the stop is named once"
        );

        let crashed = failed("Rendering frame 179 of 180\n");
        let note = delivery_notes(&crashed).expect("the exit status is named");
        assert_eq!(
            note,
            "the file was written; the renderer's run ended: exit status 1"
        );

        let reported = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"history.json \
             could not be written\",\"hint\":\"run hyperframes doctor\"}}\n",
        );
        let note = delivery_notes(&reported).expect("the renderer's failure is named");
        assert_eq!(
            note,
            "the file was written; the renderer reported a failure after the write: \
             runtime_error: history.json could not be written"
        );

        // The same post-render failure in both shapes — its entry and its late complaint — is
        // one failure, one sentence: the entry leads and the complaint adds nothing.
        let twice = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"history.json \
             could not be written\"}}\n  [hyperframes] Post-render step 'history' failed \
             (render already succeeded): history.json could not be written\n",
        );
        let note = delivery_notes(&twice).expect("one sentence names it");
        assert_eq!(note.matches("history.json").count(), 1, "got: {note}");
        assert!(
            note.starts_with("the file was written; the renderer reported a failure"),
            "the entry leads: {note}"
        );

        // A complaint that says more about the same failure is the telling that stands.
        let fuller = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"history.json \
             could not be written\"}}\n  [hyperframes] Post-render step 'history' failed \
             (render already succeeded): history.json could not be written: permission denied \
             on the state folder\n",
        );
        let note = delivery_notes(&fuller).expect("one sentence names it");
        assert_eq!(note.matches("history.json").count(), 1, "got: {note}");
        assert!(
            note.contains("permission denied"),
            "the fuller telling is kept: {note}"
        );

        // The same failure with the vendor's wrapper in one telling and not the other is still
        // one failure, one sentence.
        let wrapped = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"Post-render step \
             'history' failed (render already succeeded): history.json could not be written\"}}\n  \
             [hyperframes] Late step failed (render already succeeded): history.json could not be \
             written\n",
        );
        let note = delivery_notes(&wrapped).expect("one sentence names it");
        assert_eq!(note.matches("history.json").count(), 1, "got: {note}");

        // An entry carrying a code and nothing else (its message was all advice) names nothing
        // for the complaint to repeat: the complaint is the only telling.
        let code_only = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"run hyperframes \
             doctor\"}}\n  [hyperframes] Post-render step 'history' failed (render already \
             succeeded): history.json could not be written\n",
        );
        let note = delivery_notes(&code_only).expect("the complaint names it");
        assert_eq!(note.matches("history.json").count(), 1, "got: {note}");
        assert!(
            !note.contains("runtime_error"),
            "one sentence, not two: {note}"
        );
        // Without a complaint the entry is all there is, code alone included.
        let code_alone = failed(
            "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"run hyperframes \
             doctor\"}}\n",
        );
        assert!(delivery_notes(&code_alone).is_some_and(|note| note.ends_with("runtime_error")));

        // Whatever the note says, the file the probe read is still the result.
        let delivered = render_verdict(plays(), &crashed, Path::new("/work/clip.mp4"))
            .expect("the file stands");
        assert!(delivered.notes.unwrap().contains("exit status 1"));

        // A stop and a post-render complaint can both land on a delivered file: the machine's
        // verdict leads and the renderer's complaint follows.
        let stopped = ProgramOutcome {
            detail: "terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)".to_string(),
            stop: Some(Stop::Memory),
            ..failed(
                "  [hyperframes] Post-render step 'history' failed (render already succeeded): \
                 history.json could not be written\n",
            )
        };
        let note = delivery_notes(&stopped).expect("both facts are kept");
        let stop = note
            .find("the product stopped the render")
            .expect("the stop");
        let complaint = note.find("history.json").expect("the complaint");
        assert!(stop < complaint, "the stop leads: {note}");
    }

    #[test]
    fn a_run_the_product_ended_reports_its_own_stop_reason() {
        // A stopped run has no envelope to explain it, and the tail of the renderer's progress
        // is no substitute for the reason it ended.
        let outcome = ProgramOutcome {
            detail: "terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)".to_string(),
            stop: Some(Stop::Memory),
            ..failed("capturing frame 18 of 30\n")
        };
        assert_eq!(
            run_failure(&outcome, RENDER, None),
            "the render failed: terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)\n\
             capturing frame 18 of 30"
        );

        // An entry printed before the product stopped the run cannot speak for the stop: the
        // machine's reason leads.
        let outcome = ProgramOutcome {
            detail: "terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)".to_string(),
            stop: Some(Stop::Memory),
            ..failed(
                "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"page threw\"}}\n",
            )
        };
        let message = run_failure(&outcome, RENDER, None);
        assert!(
            message.starts_with("the render failed: terminated: exceeded memory limit"),
            "got: {message}"
        );
        assert!(!message.contains("page threw"), "got: {message}");
    }

    /// The lines the filter is judged on, and what it must return for each: the vendor's advice
    /// goes, the diagnosis stays, and neither punctuation nor brackets are damaged.
    const ADVICE_CASES: [(&str, &str); 37] = [
        (
            "Try --docker for containerized rendering. The script threw at line 42.",
            "The script threw at line 42.",
        ),
        (
            "The script threw at line 42. Try --docker for containerized rendering.",
            "The script threw at line 42.",
        ),
        // An instruction to install is advice; a description that mentions an install is a
        // diagnosis and stays.
        (
            "Install a working 64-bit FFmpeg build with all required runtime DLLs.",
            "",
        ),
        ("Please install a working FFmpeg build", ""),
        (
            "the install step of the page threw a TypeError",
            "the install step of the page threw a TypeError",
        ),
        // A word that only looks like an advice marker is not advice.
        (
            "the composition did not run: TypeError",
            "the composition did not run: TypeError",
        ),
        // The vendor's `Run: …` cue and its object are one clause.
        ("Run: npx hyperframes browser ensure", ""),
        // A diagnosis sharing a clause with an instruction keeps the whole clause: the filter
        // never leaves a fragment behind.
        (
            "The script threw at line 42 and then run hyperframes doctor",
            "The script threw at line 42 and then run hyperframes doctor",
        ),
        // A line with nothing to remove comes back as printed: the filter only takes a clause it
        // actually dropped.
        ("The page threw a TypeError;", "The page threw a TypeError;"),
        (
            "The clip ended early (data-duration 2).",
            "The clip ended early (data-duration 2).",
        ),
        // A parenthetical goes only as a whole: an advice one is dropped with both brackets, and
        // one that is not keeps both.
        ("Failed (run hyperframes doctor, then retry)", "Failed"),
        (
            "The page threw (run hyperframes doctor) (exit 1)",
            "The page threw (exit 1)",
        ),
        (
            "The page threw. run hyperframes doctor (exit 1)",
            "The page threw.",
        ),
        (
            "The page threw (run hyperframes doctor, then check the asset).",
            "The page threw.",
        ),
        ("run hyperframes doctor (see docs) then render again", ""),
        // A remedy mixed into a sentence the filter cannot split without damaging it: the line as
        // printed is the only form that keeps the diagnosis.
        (
            "Render failed (exit 1) then run npx hyperframes doctor",
            "Render failed (exit 1) then run npx hyperframes doctor",
        ),
        // A description that merely names an installer is not an instruction.
        (
            "The npm install step failed: EACCES on /usr/lib/node_modules",
            "The npm install step failed: EACCES on /usr/lib/node_modules",
        ),
        // The kept sentence's punctuation is its own: a drop before it never takes the full
        // stop's neighbours.
        (
            "Run npm i. The page threw a TypeError;",
            "The page threw a TypeError;",
        ),
        // A hyphen joins advice as a dash does, and goes with it.
        (
            "Render failed - see docs - run npx hyperframes doctor",
            "Render failed - see docs",
        ),
        // The vendor's knob lines name its settings and nothing else.
        (
            "Env: PRODUCER_PAGE_NAVIGATION_TIMEOUT_MS\nThe page did not load.",
            "The page did not load.",
        ),
        // The boundary separator is not what decides: advice joined by a semicolon, a dash, a
        // colon or a parenthesis leaves the diagnosis behind it.
        (
            "The script threw at line 42; try --docker",
            "The script threw at line 42",
        ),
        (
            "The script threw at line 42 — run hyperframes doctor to check the media",
            "The script threw at line 42",
        ),
        (
            "The script threw at line 42: run hyperframes doctor",
            "The script threw at line 42",
        ),
        (
            "The script threw at line 42 (run hyperframes doctor)",
            "The script threw at line 42",
        ),
        // A closing bracket belongs to the sentence: trimming a dropped clause never takes it,
        // and a parenthetical that stays keeps its own.
        ("Render failed (exit code 1)", "Render failed (exit code 1)"),
        // When the dropped clause is inside a parenthesis whose other clauses stay, no bracket
        // removal leaves the sentence whole: the line as printed is the only safe answer.
        (
            "Render failed (exit 1; run hyperframes doctor)",
            "Render failed (exit 1; run hyperframes doctor)",
        ),
        // The vendor's command line, in the shape its lint fixes use.
        (
            "The composition is blank. hyperframes snapshot shows why",
            "The composition is blank.",
        ),
        (
            "If clip.mp4 fails to play in preview, run hyperframes doctor and confirm \
                 media.autoProxy is not false.",
            "",
        ),
        (
            "update the meta viewport to match, or scaffold with `hyperframes init \
                 --resolution portrait`",
            "update the meta viewport to match",
        ),
        ("bunx hyperframes render .", ""),
        // Naming the product is not naming a command.
        (
            "The hyperframes package is not compatible with this Node version",
            "The hyperframes package is not compatible with this Node version",
        ),
        (
            "Compatibility with hyperframes 0.9 requires a newer runtime",
            "Compatibility with hyperframes 0.9 requires a newer runtime",
        ),
        // A line with nothing to remove comes back byte for byte: the filter rewrites only what
        // it took something out of.
        (
            "  The page threw a TypeError  ",
            "  The page threw a TypeError  ",
        ),
        // The junction a dropped leading clause left goes with it: the comma was the advice's.
        (
            "Install an FFmpeg build with libx264 support (brew install ffmpeg), or render \
             WebM instead: hyperframes render .",
            "or render WebM instead",
        ),
        (
            "(brew install ffmpeg), or the page threw a TypeError",
            "or the page threw a TypeError",
        ),
        // A junction in the middle of the line belongs to the clause that was dropped.
        (
            "The encoder is missing. Install an FFmpeg build (brew install ffmpeg), or render \
             WebM instead: hyperframes render .",
            "The encoder is missing. or render WebM instead",
        ),
        (
            "Failed to read the capture path; Alternatively, install FFmpeg with libx264",
            "Failed to read the capture path; Alternatively",
        ),
    ];

    #[test]
    fn advice_never_takes_the_diagnosis_with_it() {
        for (line, kept) in ADVICE_CASES {
            assert_eq!(without_remedies(line), kept, "for: {line}");
        }
    }

    /// The rule behind the table above, over lines the filter has never seen: a balanced line
    /// keeps its brackets balanced, no kept line carries a word the vendor never wrote, and a line
    /// the filter cannot improve comes back whole. The generator is fixed, so the check is the
    /// same on every run.
    #[test]
    fn the_filter_never_damages_what_it_keeps() {
        const WORDS: [&str; 28] = [
            "run",
            "hyperframes",
            "doctor",
            "npx",
            "install",
            "the",
            "page",
            "threw",
            "exit",
            "Failed",
            "render",
            "Try",
            "see",
            "docs",
            "retry",
            "asset",
            "check",
            "and",
            "if",
            "clip.mp4",
            "preview,",
            "then",
            "again.",
            ";",
            ":",
            "(",
            ")",
            "exit)",
        ];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..500 {
            let mut line = String::new();
            for _ in 0..=next() % 10 {
                let word = WORDS[usize::try_from(next() % WORDS.len() as u64).unwrap_or_default()];
                line.push_str(if line.is_empty() { "" } else { " " });
                line.push_str(word);
            }
            let starting_balanced = balanced(&line);
            for kept in without_remedies(&line).lines() {
                assert!(
                    !starting_balanced || balanced(kept) || kept == line,
                    "unbalanced from {line:?}: {kept:?}"
                );
                for word in kept
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|word| word.len() > 1)
                {
                    assert!(
                        line.contains(word),
                        "invented {word:?} from {line:?}: {kept:?}"
                    );
                }
            }
        }
    }

    /// The budget is spent in the order a reader needs the facts: the machine-side note and the
    /// hint come before the vendor's prose is cut, and the vendor's container suggestion — advice
    /// this product never gives — is dropped however long the message runs.
    #[test]
    fn the_note_and_the_hint_are_paid_for_before_the_prose_is_cut() {
        let prose = "prose ".repeat(400);

        // The machine-side note is what a reader is told to look for.
        let output = format!(
            "{{\"ok\":false,\"error\":{{\"code\":\"runtime_error\",\"message\":\"{prose}\
             MP4 H.264 encoder unavailable\"}}}}\n",
        );
        let message = run_failure(&failed(&output), RENDER, None);
        assert!(message.contains("machine-side gap"), "got: {message}");
        assert!(
            message.len() <= MESSAGE_BUDGET_BYTES + '…'.len_utf8(),
            "got {} bytes",
            message.len()
        );

        // The hint is the line that says what to do.
        let output = format!(
            "{{\"ok\":false,\"error\":{{\"code\":\"runtime_error\",\"message\":\"{prose}\",\
             \"hint\":\"compositions/scene.html:42 — intro is not defined. \
             Try --docker for containerized rendering\"}}}}\n",
        );
        let message = run_failure(&failed(&output), RENDER, None);
        assert!(
            message.starts_with("runtime_error: prose"),
            "the code and message still lead: {message}"
        );
        assert!(
            message.contains("hint: compositions/scene.html:42"),
            "got: {message}"
        );
        assert!(!message.contains("--docker"), "got: {message}");
        assert!(
            message.len() <= MESSAGE_BUDGET_BYTES + '…'.len_utf8(),
            "got {} bytes",
            message.len()
        );
    }
}

//! The `hyperframes` tool: an HTML composition in, an ordinary video file out.
//!
//! HyperFrames seeks a paused HTML timeline in a headless Chrome frame by frame
//! and encodes the frames with FFmpeg. The renderer, its browser and its FFmpeg
//! are the product's own ([`crate::hyperframes`]): local, no cloud, no per-render
//! cost.
//!
//! The model-facing cost is part of the design: one short description and a flat
//! schema are paid every turn, while the long field guide (`instructions`) and
//! its deep references (`reference`) are fetched only when the agent asks.
//!
//! The rules the implementation exists to hold:
//!
//! * **The caller's boundary.** The composition is resolved through the same gate
//!   an ordinary read goes through, at the caller's own [`PathAccess`]; the
//!   output through the write gate.
//! * **The project is the vendor's.** `-c` must stay inside the project folder,
//!   and the vendor finds that folder by walking up to the nearest `index.html`;
//!   [`project_of`] mirrors exactly that, so the two never disagree about what
//!   is rendered.
//! * **Nothing is overwritten.** The output name is claimed by creating it, so two
//!   renders naming the same file cannot both be handed it: the loser takes the
//!   next free `_2`, `_3`, and the answer names the file actually written.
//! * **The declaration is not the guarantee.** A value outside an enum is refused
//!   by name rather than passed to the vendor.
//! * **The artifact decides.** A file the product can play is the result, whatever the
//!   run's exit status, output size or printed envelope; a stop, a failed step after the
//!   write or a probe that could not answer is noted beside the file.
//! * **A file is kept when the machine cannot judge it.** A probe that cannot run leaves
//!   the render in place with a note, where a file the probe reads as unplayable is the
//!   failure.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::json;

use super::path::PathAccess;
use crate::hyperframes::{Finding, Format, LintReport, Quality, RenderRequest, Rendered};
use crate::{Tool, Workspace};

/// The actions the schema declares and the dispatch serves.
const ALL: [&str; 4] = ["instructions", "reference", "render", "inspect"];

/// The topics `reference` returns a document for.
const TOPICS: [&str; 3] = ["authoring", "rendering", "troubleshooting"];

/// The resolutions the vendor captures at, each at a raised device pixel ratio.
const RESOLUTIONS: [&str; 6] = [
    "landscape",
    "portrait",
    "square",
    "landscape-4k",
    "portrait-4k",
    "square-4k",
];

/// What a refused `fps` is told it must be.
const FPS_RULE: &str = "an integer 1-240 or an FFmpeg rational such as 30000/1001";

/// How many findings one `inspect` answer lists before it summarizes the rest.
const MAX_FINDINGS: usize = 12;

/// The `hyperframes` tool. `access` is the caller's [`PathAccess`], fixed when
/// the role's toolset is built: the composition is read under it and the output
/// written under it.
pub(crate) struct HyperframesTool {
    access: PathAccess,
}

impl HyperframesTool {
    /// The tool's own name: what [`Tool::name`] reports and the prompt asset
    /// that describes it.
    const NAME: &'static str = "hyperframes";

    #[must_use]
    pub(crate) const fn new(access: PathAccess) -> Self {
        Self { access }
    }
}

#[async_trait]
impl Tool for HyperframesTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Hidden while the managed runtime is missing, exactly as the document tool is:
    /// both cannot work without it (see [`Tool::is_advertised`] for why the verdict
    /// is kept for an agent's lifetime), and `execute` still refuses honestly if the
    /// runtime vanishes mid-run.
    fn is_advertised(&self) -> bool {
        crate::tools::bun::bun_binary_path().is_some()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let formats: Vec<&str> = Format::ALL.iter().map(|format| format.spelling()).collect();
        let qualities: Vec<&str> = Quality::ALL
            .iter()
            .map(|quality| quality.spelling())
            .collect();
        let mut schema = super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ALL,
                    "description": "`instructions` for the field guide, `reference` for one deep topic, `inspect` to check a composition, `render` to write a video."
                },
                "topic": {
                    "type": "string",
                    "enum": TOPICS,
                    "description": "The reference document to return, for `reference`."
                },
                "composition": {
                    "type": "string",
                    "description": "For `render` and `inspect`: the composition — an `.html` file, or a project folder holding `index.html`."
                },
                "output": {
                    "type": "string",
                    "description": "For `render`: the file to write. Default: a new file in your workspace's `generated/`."
                },
                "format": {
                    "type": "string",
                    "enum": formats,
                    "description": "Default `mp4` (H.264). `webm` is VP9 with alpha, `mov` is ProRes 4444 with alpha, `gif` is a short silent loop."
                },
                "quality": {
                    "type": "string",
                    "enum": qualities,
                    "description": "Default `looks`: `draft` encodes fast, `delivery` is the cleanest."
                },
                "fps": {
                    "type": "string",
                    "description": format!("How many frames a second: {FPS_RULE}. Default: the composition's `data-fps`, else 30.")
                },
                "resolution": {
                    "type": "string",
                    "enum": RESOLUTIONS,
                    "description": "Captured at a higher device pixel ratio; the composition's aspect ratio must match."
                }
            }),
            &["action"],
        );
        schema["additionalProperties"] = json!(false);
        schema
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[VIDEO:")
    }

    /// Instructions, references and checks answer without one.
    fn media_marker_always(&self) -> bool {
        false
    }

    fn preserve_full_output(&self) -> bool {
        // The instruction documents must arrive whole, and this tool's own
        // answers are short by construction — a marker plus a few facts, a
        // bounded finding list — so nothing else grows unbounded.
        true
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        match super::get_str(&args, "action")? {
            "instructions" => Ok(crate::prompt::load_prompt(
                "tool/hyperframes_instructions.md",
            )),
            "reference" => Self::reference(&args),
            "inspect" => self.inspect(ws, &args).await,
            "render" => self.render(ws, &args).await,
            other => anyhow::bail!(
                "usage: unknown action {other:?} — hint: use one of {}",
                ALL.join(", ")
            ),
        }
    }
}

impl HyperframesTool {
    /// `reference` — one deep document, fetched only when the agent asks for it.
    fn reference(args: &serde_json::Value) -> anyhow::Result<String> {
        let topic = super::get_str(args, "topic")?;
        if !TOPICS.contains(&topic) {
            return Err(super::refuse_choice("topic", topic, &TOPICS));
        }
        Ok(crate::prompt::load_prompt(&format!(
            "tool/hyperframes_{topic}.md"
        )))
    }

    /// `inspect` — the vendor's static check, rendered as findings the agent can
    /// act on. A file is checked as the render would see it: the vendor resolves a
    /// file's project the way [`project_of`] does and checks only that composition.
    async fn inspect(&self, ws: &Workspace, args: &serde_json::Value) -> anyhow::Result<String> {
        let requested = super::get_str(args, "composition")?;
        let target = self.resolve_composition(ws, requested).await?;
        let report = crate::hyperframes::lint(&target)
            .await
            .map_err(|reason| anyhow::anyhow!(reason))?;
        Ok(format_report(&report))
    }

    /// `render` — resolve both sides, hand the vendor one request, and answer the
    /// file it wrote.
    async fn render(&self, ws: &Workspace, args: &serde_json::Value) -> anyhow::Result<String> {
        let requested = super::get_str(args, "composition")?;
        let format = format_arg(args)?;
        let quality = quality_arg(args)?;
        let fps = super::get_opt_str_strict(args, "fps")?;
        if let Some(fps) = fps
            && !valid_fps(fps)
        {
            anyhow::bail!("usage: `fps` must be {FPS_RULE} — got \"{fps}\"");
        }
        let resolution = super::get_opt_str_strict(args, "resolution")?;
        if let Some(resolution) = resolution
            && !RESOLUTIONS.contains(&resolution)
        {
            return Err(super::refuse_choice("resolution", resolution, &RESOLUTIONS));
        }

        let target = self.resolve_composition(ws, requested).await?;
        let reserved = self.output_path(ws, args, format).await?;
        let (project, composition) = if target.is_dir() {
            (target, None)
        } else {
            // The project the renderer is given follows the caller's access, not the
            // workspace: a caller allowed to read anywhere renders a composition
            // wherever it lives, scoped to that file's own project.
            let root = match self.access {
                PathAccess::Unrestricted => None,
                _ => Some(ws.as_path()),
            };
            project_of(&target, root, |dir| dir.join("index.html").is_file())
        };

        let rendered = crate::hyperframes::render(RenderRequest {
            project: &project,
            composition: composition.as_deref(),
            output: &reserved.path,
            format,
            quality,
            fps,
            resolution,
        })
        .await;
        // The renderer keeps its state beside the composition; the ignore file goes in
        // once it has made the folder, and keeps a managed repository's `git add -A` off
        // that state.
        keep_state_out_of_git(&project);
        // A failure takes the reservation with it: the empty file a failure that struck
        // before the renderer started left, and any file it wrote that no player could open.
        let rendered = rendered.map_err(|reason| anyhow::anyhow!(reason))?;
        Ok(self
            .rendered_result(&reserved.keep(), format, fps, rendered)
            .await)
    }

    /// The resolved composition: a project folder holding `index.html`, or one
    /// HTML file.
    async fn resolve_composition(
        &self,
        ws: &Workspace,
        requested: &str,
    ) -> anyhow::Result<PathBuf> {
        let target = super::path::resolve_read_target(ws.as_path(), requested, self.access).await?;
        let meta = tokio::fs::metadata(&target)
            .await
            .map_err(|error| anyhow::anyhow!("cannot read {}: {error}", target.display()))?;
        if meta.is_dir() {
            anyhow::ensure!(
                target.join("index.html").is_file(),
                "cannot use {requested}: that folder has no index.html — hint: name the \
                 composition file, or a folder holding index.html"
            );
            return Ok(target);
        }
        anyhow::ensure!(
            meta.is_file(),
            "usage: {} is not a regular file or a directory — hint: name an `.html` composition, \
             or a project folder holding index.html",
            target.display()
        );
        Ok(target)
    }

    /// Where the render writes: the caller's own file, or a fresh name in the caller's
    /// `generated/`. Asking for a clip in a folder that does not exist yet is an ordinary
    /// request, so the folder is created rather than refused. The name itself is claimed before
    /// the render starts (see [`claim`]).
    async fn output_path(
        &self,
        ws: &Workspace,
        args: &serde_json::Value,
        format: Format,
    ) -> anyhow::Result<Reservation> {
        if let Some(requested) = super::get_opt_str_strict(args, "output")? {
            let path =
                super::path::resolve_write_target(ws.as_path(), requested, true, self.access)
                    .await?;
            return Ok(Reservation::of(
                claim(&with_format_extension(path, format)?).await?,
            ));
        }
        let dir = super::output_directory(ws, super::GENERATED_DIR);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|error| anyhow::anyhow!("cannot create {}: {error}", dir.display()))?;
        Ok(Reservation::of(
            claim(&dir.join(format!(
                "rendered_{}.{}",
                crate::util::unix_millis(),
                format.spelling()
            )))
            .await?,
        ))
    }

    /// The tool's answer for a file that was written: the marker, then one
    /// sentence of facts from the probe that already let the render count, then whatever
    /// notes the render earned. A file the probe could not read keeps the marker and the
    /// notes; the facts are best-effort, the marker is not.
    async fn rendered_result(
        &self,
        output: &Path,
        format: Format,
        fps: Option<&str>,
        rendered: Rendered,
    ) -> String {
        let marker = self.format_media_result(output);
        let mut answer = marker;
        if let Some(video) = rendered.info {
            let (width, height) = video.size;
            let mut sentence = format!(
                "Rendered {:.1} s at {width}×{height}",
                video.duration.as_secs_f64()
            );
            if let Some(fps) = fps.filter(|value| plain_integer(value)) {
                let _ = write!(sentence, ", {fps} fps");
            }
            if let Ok(meta) = tokio::fs::metadata(output).await {
                let _ = write!(
                    sentence,
                    ", {}, {}.",
                    format.spelling().to_uppercase(),
                    super::listing::human_readable_size(meta.len())
                );
                let _ = write!(answer, "\n\n{sentence}");
            }
        }
        if let Some(note) = rendered.notes {
            let _ = write!(answer, "\n\n{note}");
        }
        answer
    }
}

/// The renderer's own state folder, `project/.hyperframes`, kept out of git: a managed
/// repository's own `git add -A` would otherwise commit it. Only a folder the renderer made is
/// touched — nothing of ours is invented in a project it never ran in — and nothing else inside
/// it is: the ignore file is created, never replaced (`create_new`), so one already there stands.
fn keep_state_out_of_git(project: &Path) {
    let state = project.join(".hyperframes");
    if !state.is_dir() {
        return;
    }
    let ignore = state.join(".gitignore");
    let Ok(mut file) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&ignore)
    else {
        return;
    };
    if std::io::Write::write_all(&mut file, b"*\n").is_err() {
        // A file we could not fill would be skipped by every later attempt, quietly leaving
        // the state committable: an absent one is planted whole next time.
        drop(file);
        let _ = std::fs::remove_file(&ignore);
    }
}

/// Which file inside which folder the vendor is given. The project is the
/// nearest ancestor of `file` — its own directory first, never above `root` —
/// that `has_index` accepts, else the file's own directory; the composition is
/// the file relative to that project, and `None` when the file IS the project's
/// `index.html`.
///
/// `root` bounds the walk because the vendor writes its `.hyperframes` state into
/// the project folder: a caller's access does not reach above its own workspace,
/// so neither may the project. `None` is the unrestricted caller, whose walk is
/// bounded by the filesystem alone.
fn project_of(
    file: &Path,
    root: Option<&Path>,
    has_index: impl Fn(&Path) -> bool,
) -> (PathBuf, Option<String>) {
    let own_dir = file.parent().unwrap_or(file);
    let mut dir = own_dir;
    let mut project = own_dir;
    while root.is_none_or(|root| dir.starts_with(root)) {
        if has_index(dir) {
            project = dir;
            break;
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    let relative = file
        .strip_prefix(project)
        .map(|path| {
            path.components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/")
        })
        .unwrap_or_default();
    let composition = (!relative.is_empty() && relative != "index.html").then_some(relative);
    (project.to_path_buf(), composition)
}

/// The extension `format` writes. A name carrying another extension is refused; a
/// name with none gains the extension. A directory cannot reach here: the write gate
/// already refuses a non-regular target.
fn with_format_extension(path: PathBuf, format: Format) -> anyhow::Result<PathBuf> {
    let extension = format.spelling();
    match path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(existing) if existing.eq_ignore_ascii_case(extension) => Ok(path),
        Some(existing) if !existing.is_empty() => anyhow::bail!(
            "format {extension} writes a .{extension} file, but the output is named .{existing} \
             — hint: name a .{extension} file or change format"
        ),
        _ => {
            let mut named = path.into_os_string();
            named.push(".");
            named.push(extension);
            Ok(PathBuf::from(named))
        }
    }
}

/// Claim `path` for one render: create it now, or take the next free name beside it.
///
/// The file is created with `create_new` ([`crate::util::create_unique_file`]), so
/// the reservation is the collision check itself: two renders naming the same
/// output cannot both be handed it, and the second gets `_2`, `_3` from the shared
/// suffix scheme. The renderer overwrites the file it is given, so the empty
/// reservation is what it writes into, and holding it is what clears it ([`Reservation`]).
async fn claim(path: &Path) -> anyhow::Result<PathBuf> {
    let (Some(name), Some(dir)) = (
        path.file_name().and_then(std::ffi::OsStr::to_str),
        path.parent(),
    ) else {
        anyhow::bail!(
            "cannot write {}: name a file with a UTF-8 name inside a folder",
            path.display()
        );
    };
    let (file, claimed) = crate::util::create_unique_file(dir, name)
        .await
        .map_err(|error| anyhow::anyhow!("cannot write {}: {error}", path.display()))?;
    drop(file);
    Ok(claimed)
}

/// The output name one render holds. Claiming it creates the empty file the renderer writes
/// into, so the claim is also what clears it: a render that fails, and a call whose future is
/// dropped before the file is the result, both take the file with them, and a retry gets the name
/// the caller asked for rather than `_2`. The removal is best effort — a renderer the product has
/// just ended may still hold the file (Windows), and the drop then leaves it.
struct Reservation {
    path: PathBuf,
}

impl Reservation {
    fn of(path: PathBuf) -> Self {
        Self { path }
    }

    /// The file as the result: taking the path out of the reservation is what keeps it.
    fn keep(mut self) -> PathBuf {
        std::mem::take(&mut self.path)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// `format`'s value, defaulting to MP4.
fn format_arg(args: &serde_json::Value) -> anyhow::Result<Format> {
    match super::get_opt_str_strict(args, "format")? {
        None => Ok(Format::Mp4),
        Some(value) => Format::parse(value).ok_or_else(|| {
            let allowed: Vec<&str> = Format::ALL.iter().map(|format| format.spelling()).collect();
            super::refuse_choice("format", value, &allowed)
        }),
    }
}

/// `quality`'s value, defaulting to `looks`.
fn quality_arg(args: &serde_json::Value) -> anyhow::Result<Quality> {
    match super::get_opt_str_strict(args, "quality")? {
        None => Ok(Quality::Looks),
        Some(value) => Quality::parse(value).ok_or_else(|| {
            let allowed: Vec<&str> = Quality::ALL
                .iter()
                .map(|quality| quality.spelling())
                .collect();
            super::refuse_choice("quality", value, &allowed)
        }),
    }
}

/// An integer 1-240, or a positive rational such as `30000/1001`.
fn valid_fps(value: &str) -> bool {
    if let Ok(fps) = value.parse::<u32>() {
        return (1..=240).contains(&fps);
    }
    match value.split_once('/') {
        Some((numerator, denominator)) => matches!(
            (numerator.parse::<u32>(), denominator.parse::<u32>()),
            (Ok(numerator), Ok(denominator)) if numerator > 0 && denominator > 0
        ),
        None => false,
    }
}

/// Digits only — the frame-rate clause is stated only for a rate that is one.
fn plain_integer(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// The bounded, readable form of a lint report: one counts line, then at most
/// [`MAX_FINDINGS`] findings, errors first.
fn format_report(report: &LintReport) -> String {
    let mut out = counts_line(report);
    let mut findings: Vec<&Finding> = report.findings.iter().collect();
    findings.sort_by_key(|finding| severity_rank(&finding.severity));
    for finding in findings.iter().take(MAX_FINDINGS) {
        out.push('\n');
        out.push_str(&finding_line(finding));
    }
    let hidden = findings.len().saturating_sub(MAX_FINDINGS);
    if hidden > 0 {
        let _ = write!(out, "\n…and {hidden} more");
    }
    out
}

/// `2 errors, 1 warning in 1 file`, with zero counts left out.
fn counts_line(report: &LintReport) -> String {
    let counts = [
        (report.errors, "error"),
        (report.warnings, "warning"),
        (report.infos, "info"),
    ];
    let stated: Vec<String> = counts
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, noun)| format!("{count} {noun}{}", if count == 1 { "" } else { "s" }))
        .collect();
    let files = format!(
        "{} {}",
        report.files,
        if report.files == 1 { "file" } else { "files" }
    );
    if stated.is_empty() {
        format!("no problems found in {files}")
    } else {
        format!("{} in {files}", stated.join(", "))
    }
}

/// One finding: severity, code, location, message, and an indented fix line.
fn finding_line(finding: &Finding) -> String {
    let severity = clean(&finding.severity);
    let code = clean(&finding.code);
    let mut line = if code.is_empty() {
        format!("- {severity}")
    } else {
        format!("- {severity} {code}")
    };
    if let Some(location) = location(finding) {
        let _ = write!(line, " ({location})");
    }
    let message = clean(&finding.message);
    if !message.is_empty() {
        let _ = write!(line, ": {message}");
    }
    if let Some(fix) = finding
        .fix
        .as_deref()
        .map(clean)
        .filter(|fix| !fix.is_empty())
    {
        let _ = write!(line, "\n  fix: {fix}");
    }
    line
}

/// `file:line`, with the file reduced to its own name — the project is known
/// from the call, and a whole path only crowds the answer.
fn location(finding: &Finding) -> Option<String> {
    let file = clean(finding.file.as_deref()?);
    if file.is_empty() {
        return None;
    }
    let name = Path::new(file).file_name().map_or_else(
        || file.to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    Some(match finding.line {
        Some(line) => format!("{name}:{line}"),
        None => name,
    })
}

/// Errors, then warnings, then infos; anything else after them, in its own
/// order.
fn severity_rank(severity: &str) -> u8 {
    match clean(severity) {
        "error" => 0,
        "warning" => 1,
        "info" => 2,
        _ => 3,
    }
}

/// A vendor string without the BOM or whitespace it sometimes arrives with.
fn clean(text: &str) -> &str {
    text.trim_start_matches(|character: char| character.is_whitespace() || character == '\u{feff}')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state folder is the vendor's: the product touches it only once the renderer has made
    /// it, and the one file it puts there is created whole, never replaced.
    #[test]
    fn the_state_folder_is_kept_out_of_git_without_replacing_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();
        let state = project.join(".hyperframes");
        let ignore = state.join(".gitignore");

        keep_state_out_of_git(project);
        assert!(
            !state.exists(),
            "a project the renderer never ran in gets nothing from us"
        );

        std::fs::create_dir_all(&state).unwrap();
        keep_state_out_of_git(project);
        assert_eq!(std::fs::read_to_string(&ignore).unwrap(), "*\n");

        std::fs::write(&ignore, "mine\n").unwrap();
        keep_state_out_of_git(project);
        assert_eq!(
            std::fs::read_to_string(&ignore).unwrap(),
            "mine\n",
            "a file already there stands"
        );
    }

    fn has_index_at(root: &Path) -> impl Fn(&Path) -> bool {
        let root = root.to_path_buf();
        move |dir| dir == root
    }

    #[test]
    fn project_of_finds_the_nearest_index() {
        let project = Path::new("project");
        assert_eq!(
            project_of(
                Path::new("project/scene.html"),
                Some(project),
                has_index_at(project)
            ),
            (project.to_path_buf(), Some("scene.html".to_string()))
        );
        assert_eq!(
            project_of(
                Path::new("project/compositions/intro.html"),
                Some(project),
                has_index_at(project)
            ),
            (
                project.to_path_buf(),
                Some("compositions/intro.html".to_string())
            )
        );
        assert_eq!(
            project_of(
                Path::new("project/index.html"),
                Some(project),
                has_index_at(project)
            ),
            (project.to_path_buf(), None)
        );
        assert_eq!(
            project_of(
                Path::new("loose/clip.html"),
                Some(Path::new("loose")),
                |_| false
            ),
            (PathBuf::from("loose"), Some("clip.html".to_string()))
        );
        // The walk stops at the caller's own root: an index.html above it is not
        // the project, because the project folder is where the vendor writes.
        assert_eq!(
            project_of(
                Path::new("ws/deep/clip.html"),
                Some(Path::new("ws/deep")),
                |dir| dir == Path::new("ws")
            ),
            (PathBuf::from("ws/deep"), Some("clip.html".to_string()))
        );
        // An unrestricted caller's walk reaches the file's own project wherever it
        // lives, outside the workspace included.
        assert_eq!(
            project_of(
                Path::new("elsewhere/compositions/intro.html"),
                None,
                |dir| dir == Path::new("elsewhere")
            ),
            (
                PathBuf::from("elsewhere"),
                Some("compositions/intro.html".to_string())
            )
        );
    }

    #[tokio::test]
    async fn claim_never_hands_out_a_taken_name() {
        let dir = tempfile::tempdir().unwrap();
        let requested = dir.path().join("clip.mp4");
        let first = claim(&requested).await.unwrap();
        assert_eq!(first, requested);
        assert!(first.is_file(), "the reservation is the claim itself");
        let second = claim(&requested).await.unwrap();
        assert_eq!(second, dir.path().join("clip_2.mp4"));
        tokio::fs::write(&second, b"a finished render")
            .await
            .unwrap();
        let third = claim(&requested).await.unwrap();
        assert_eq!(third, dir.path().join("clip_3.mp4"));
        assert_eq!(
            tokio::fs::read(&second).await.unwrap(),
            b"a finished render",
            "an existing render is never touched"
        );
    }

    /// A folder that does not exist yet is created for the render rather than refused, and the
    /// never-overwrite rule stands. The tool's own output resolution is what is exercised: the
    /// write gate is asked to make the folder, and the name is claimed through it.
    #[tokio::test]
    async fn a_named_output_in_a_new_folder_is_created_not_refused() {
        let dir = tempfile::tempdir().unwrap();
        let tool = HyperframesTool {
            access: PathAccess::Workspace,
        };
        let ws = Workspace::from_path(dir.path());
        let args = json!({ "output": "clips/2024/first.mp4" });

        let claimed = tool
            .output_path(&ws, &args, Format::Mp4)
            .await
            .expect("a new folder is created, not refused");
        assert!(dir.path().join("clips/2024").is_dir());
        assert!(
            claimed.path.is_file(),
            "the name is reserved before the render"
        );
        assert_eq!(claimed.path.file_name().unwrap(), "first.mp4");

        let again = tool
            .output_path(&ws, &args, Format::Mp4)
            .await
            .expect("the folder exists now");
        assert_eq!(again.path.file_name().unwrap(), "first_2.mp4");
    }

    /// The reservation is the claim and the cleanup: a call whose future is dropped before the
    /// file is the result leaves no empty file behind, so a retry gets the name it asked for.
    #[tokio::test]
    async fn a_dropped_render_leaves_no_reserved_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let tool = HyperframesTool {
            access: PathAccess::Workspace,
        };
        let ws = Workspace::from_path(dir.path());
        let args = json!({ "output": "clip.mp4" });

        let reserved = tool
            .output_path(&ws, &args, Format::Mp4)
            .await
            .expect("the name is claimed");
        let path = reserved.path.clone();
        assert!(
            path.is_file(),
            "the reservation exists while the render runs"
        );
        drop(reserved);
        assert!(
            !path.exists(),
            "a dropped render takes its reservation with it"
        );

        let retried = tool
            .output_path(&ws, &args, Format::Mp4)
            .await
            .expect("the name is free again");
        assert_eq!(retried.path.file_name().unwrap(), "clip.mp4");
        let kept = retried.keep();
        assert!(kept.is_file(), "a delivered render keeps its file");
    }

    #[test]
    fn output_extension_follows_format() {
        let refused = with_format_extension(PathBuf::from("clip.mp4"), Format::Webm).unwrap_err();
        assert!(
            refused
                .to_string()
                .contains("format webm writes a .webm file"),
            "got: {refused}"
        );
        assert_eq!(
            with_format_extension(PathBuf::from("clip"), Format::Webm).unwrap(),
            PathBuf::from("clip.webm")
        );
        assert_eq!(
            with_format_extension(PathBuf::from("clip.MP4"), Format::Mp4).unwrap(),
            PathBuf::from("clip.MP4")
        );
    }

    /// The on-demand documents are the tool's other half: a renamed asset would
    /// surface as a panic in `load_prompt` on a live call, a guide that never names
    /// an action sends the agent guessing, and a document shaped as a link tree into
    /// the vendor's own references would hand it paths that do not exist here.
    #[test]
    fn every_document_an_action_returns_exists_and_stands_on_its_own() {
        let instructions = crate::prompt::load_prompt("tool/hyperframes_instructions.md");
        // The guide IS what `instructions` returns, so it names the other three.
        for action in ALL.into_iter().filter(|action| *action != "instructions") {
            assert!(
                instructions.contains(action),
                "the field guide never names the `{action}` action"
            );
        }
        let references: Vec<String> = TOPICS
            .iter()
            .map(|topic| crate::prompt::load_prompt(&format!("tool/hyperframes_{topic}.md")))
            .collect();
        for document in std::iter::once(&instructions).chain(references.iter()) {
            assert!(!document.contains("references/"), "no vendor link tree");
        }
    }

    fn finding(severity: &str, code: &str) -> Finding {
        Finding {
            severity: severity.to_string(),
            code: code.to_string(),
            message: format!("{code} message"),
            fix: None,
            file: Some("/project/index.html".to_string()),
            line: Some(1),
        }
    }

    #[test]
    fn report_states_counts_then_orders_and_caps_the_findings() {
        let empty = LintReport {
            errors: 0,
            warnings: 0,
            infos: 0,
            files: 2,
            findings: Vec::new(),
        };
        assert_eq!(format_report(&empty), "no problems found in 2 files");
        let mixed = LintReport {
            errors: 2,
            warnings: 1,
            infos: 0,
            files: 1,
            findings: Vec::new(),
        };
        assert_eq!(counts_line(&mixed), "2 errors, 1 warning in 1 file");

        // Errors first, a location reduced to the file's name, and the fix line kept.
        let mut warning = finding("warning", "unused_asset");
        warning.line = None;
        warning.file = None;
        let mut error = finding("error", "root_missing_dimensions");
        error.fix = Some("\n  Set numeric `data-width` and `data-height`.".to_string());
        let report = LintReport {
            errors: 1,
            warnings: 1,
            infos: 0,
            files: 1,
            findings: vec![warning, error],
        };
        let text = format_report(&report);
        assert!(text.contains(
            "- error root_missing_dimensions (index.html:1): root_missing_dimensions message\n  \
             fix: Set numeric `data-width` and `data-height`."
        ));
        assert!(text.ends_with("- warning unused_asset: unused_asset message"));

        // At most MAX_FINDINGS lines; the rest is one summary line.
        let findings = (0..14).map(|_| finding("info", "note")).collect();
        let report = LintReport {
            errors: 0,
            warnings: 0,
            infos: 14,
            files: 1,
            findings,
        };
        let text = format_report(&report);
        assert_eq!(text.matches("- info note").count(), MAX_FINDINGS);
        assert!(text.ends_with("…and 2 more"), "got: {text}");
    }
}

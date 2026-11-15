//! The `photocraft` tool: an image in, the engine's own commands applied, a new file out.
//!
//! The vendor program, its managed copy and the layer that drives it live in [`crate::photocraft`];
//! this module owns only the model-facing surface — one flat schema, the four actions the command
//! line also exposes, and the argument parsing in front of that layer.
//!
//! It holds no [`super::path::PathAccess`] and re-imposes no boundary on a path. Its only checks are
//! about whether a path is usable: an input must exist, a `run` output is claimed by creating it —
//! so an existing one is refused, never replaced — and a `batch` output folder must be new or empty
//! and no other call may be writing into it. Nothing the user already has is overwritten.
//!
//! A `run` opens one document, or creates a canvas from nothing when the caller gives `canvas`
//! instead of `input`; a fresh output of either keeps the input's own format, and a canvas — having
//! no input format — is PNG.
//!
//! A refused request leaves no file behind. A `run` that fails removes the file it claimed: one
//! document that never became a result. A `batch` that fails partway keeps the files it already wrote
//! and says how many — twelve results out of twenty are the caller's work, not debris — and the
//! output folder it created goes only when nothing was written into it. A batch the vendor ran to its
//! end with failures is the report plus a PARTIAL answer, so a command-line caller exits non-zero on
//! what it still receives; only a run that never finished is a failure. A folder another call claimed
//! first is that call's and is never touched.
//!
//! Advertised even with no copy of the program in place: it is installed on first use, so an absent
//! copy is the normal state and must not hide the tool the way a dead backend hides its own.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::json;

use crate::photocraft::Step;
use crate::tools::path::PathAccess;
use crate::util::UnwrapPoison as _;
use crate::{Tool, Workspace};

/// The actions the schema declares and the dispatch serves.
const ALL: [&str; 4] = ["commands", "inspect", "run", "batch"];

/// The formats a fresh `run` output keeps: every extension the vendor's own capability table names
/// for a format it can WRITE — its raster set (PNG, JPEG, TIFF, WebP, GIF, BMP, TGA, ICO, Netpbm,
/// DDS, QOI, OpenEXR, Radiance HDR, FITS, DPX, JPEG XL, Cineon, Amiga IFF) plus the layered and
/// document formats — so a PSD or a PSB stays layered instead of being flattened into PNG and a rare
/// export is not silently turned into a PNG either. Every other input becomes a PNG: the ones the
/// vendor imports but cannot write (HEIC, Krita, Affinity, Paint.NET, camera raw, PCX, XPM, DICOM),
/// the one its release build leaves out entirely (AVIF: neither read nor written), and a name with no
/// extension at all — so a fresh output never names a destination the save fails on.
const WRITABLE_OUTPUT_EXTENSIONS: [&str; 41] = [
    "png", "apng", "jpg", "jpeg", "jpe", "jfif", "webp", "gif", "bmp", "dib", "tga", "icb", "vda",
    "vst", "ico", "pnm", "pbm", "pgm", "ppm", "pam", "pfm", "dds", "qoi", "exr", "hdr", "fits",
    "fit", "fts", "dpx", "jxl", "cin", "iff", "lbm", "ilbm", "pdf", "tif", "tiff", "psd", "psb",
    "ora", "pcraft",
];

/// The `photocraft` tool: its paths are unrestricted for every caller (see the module docs).
pub(crate) struct PhotocraftTool;

impl PhotocraftTool {
    const NAME: &'static str = "photocraft";
}

#[async_trait]
impl Tool for PhotocraftTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ALL,
                    "description": "`commands` for the engine's registry, `inspect` for one document's size, colour mode, depth, layers and the format its file name gives, `run` to apply a chain to one document, `batch` to apply one chain to a folder."
                },
                "input": {
                    "type": "string",
                    "description": "For `run` and `inspect`: the document to open — with `canvas` it is left out. For `batch`: the folder to process."
                },
                "canvas": {
                    "type": "object",
                    "description": "For `run` instead of `input`: the new document's own parameters, like {\"width\": 800, \"height\": 600}, so the chain starts from a canvas created from nothing; a fresh output for one is PNG."
                },
                "output": {
                    "type": "string",
                    "description": "For `run`: the file to write; an existing one is refused. For `batch`: the folder to write into, which must be new or empty."
                },
                "chain": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string"
                            },
                            "params": {
                                "type": "object"
                            }
                        },
                        "required": ["command"]
                    },
                    "description": "One step per engine command, `{\"command\": \"<id>\", \"params\": {...}}`, applied in order to the one open document for `run`, or to every file for `batch`; `params` is optional."
                },
                "filter": {
                    "type": "string",
                    "description": "Text to narrow the command registry by, for `commands`; a narrowed answer also carries each match's parameter documentation."
                },
                "format": {
                    "type": "string",
                    "description": "The output extension to write, for `batch`, like `jpg` or `png`; with no `chain` it is what a folder-wide conversion writes."
                },
                "quality": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "The lossy encoder quality, 1 to 100, for `batch`."
                },
                "tiff_layers": {
                    "type": "boolean",
                    "description": "Keep layers in TIFF output, for `batch`; a `format` of tif or tiff keeps them anyway."
                }
            }),
            &["action"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        Ok(self.dispatch(ws, args).await?.text)
    }

    /// A command-line caller reads the same answer and, for a batch the vendor failed files on, the
    /// shortfall flag that makes its exit non-zero: the files it wrote are the caller's work, so the
    /// report still arrives whole.
    async fn execute_answer(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<super::ToolAnswer> {
        self.dispatch(ws, args).await
    }

    fn format_media_result(&self, output_path: &Path) -> String {
        // The product's own deliverable-picture set, so a produced picture is delivered as one and
        // every layered or other format is a file — beside the document tool's `[FILE:` marker.
        let picture =
            crate::util::has_extension(output_path, crate::util::DELIVERABLE_IMAGE_EXTENSIONS);
        let prefix = if picture {
            "[IMAGE:"
        } else {
            super::document::FILE_MARKER
        };
        format!("{prefix}{}]", output_path.to_string_lossy())
    }
}

/// The hint a caller gets for the `action` argument, spelled once.
fn one_of() -> String {
    format!("use one of {}", ALL.join(", "))
}

impl PhotocraftTool {
    /// The one dispatch both the agent's `execute` and the command line's `execute_answer` read: only
    /// a batch can be a partial answer.
    async fn dispatch(
        &self,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<super::ToolAnswer> {
        let Some(action) = super::get_opt_str_strict(&args, "action")? else {
            return Err(required("action", &one_of()));
        };
        match action {
            "commands" => {
                let filter = super::get_opt_named_str(&args, "filter")?;
                Ok(super::ToolAnswer::done(
                    crate::photocraft::listing(filter).await?,
                ))
            }
            "inspect" => {
                let input = named(&args, "input", "name the document to inspect")?;
                let input = existing(ws, input, Kind::File).await?;
                Ok(super::ToolAnswer::done(
                    crate::photocraft::inspect(&input).await?,
                ))
            }
            "run" => Ok(super::ToolAnswer::done(self.run(ws, &args).await?)),
            "batch" => {
                let report = self.batch(ws, &args).await?;
                Ok(if report.partial {
                    super::ToolAnswer::partial(report.answer)
                } else {
                    super::ToolAnswer::done(report.answer)
                })
            }
            other => anyhow::bail!("usage: unknown action {other:?} — hint: {}", one_of()),
        }
    }

    /// `run` — open one document or create a canvas from nothing, apply the chain in order, save into
    /// a new file.
    async fn run(&self, ws: &Workspace, args: &serde_json::Value) -> anyhow::Result<String> {
        // The chain first: a malformed one is refused before anything on disk is claimed.
        let chain = parse_chain(args)?;
        // A document to open, or the vendor's own new-document parameters: exactly one of the two.
        let canvas = canvas_arg(args)?;
        let input = match (super::get_opt_named_str(args, "input")?, canvas.is_some()) {
            (Some(input), false) => Some(existing(ws, input, Kind::File).await?),
            (Some(_), true) => anyhow::bail!(
                "usage: name `input` or `canvas`, not both — hint: `input` opens a document and \
                 `canvas` starts from nothing"
            ),
            (None, true) => None,
            (None, false) => {
                return Err(required(
                    "input",
                    "name a document to open, or `canvas` to start from nothing",
                ));
            }
        };
        let output = match super::get_opt_named_str(args, "output")? {
            Some(requested) => {
                let path = resolve_path(ws, requested)?;
                claim(&path).await?;
                path
            }
            None => fresh_output(ws, input.as_deref()).await?,
        };
        let answer =
            match crate::photocraft::run_chain(input.as_deref(), canvas.as_ref(), &chain, &output)
                .await
            {
                Ok(answer) => answer,
                Err(error) => {
                    // The name this call claimed must not be left behind empty or half-written, as if
                    // it were the result.
                    let _ = tokio::fs::remove_file(&output).await;
                    return Err(error);
                }
            };
        let marker = self.format_media_result(&output);
        // The vendor answers with one compact JSON line per step. The marker is appended after the
        // answer, so the answer's own cut leaves room for it — a long chain loses whole steps, with a
        // count, rather than half of one.
        let answer = crate::photocraft::capped_lines(
            &answer,
            crate::photocraft::MAX_ANSWER_BYTES.saturating_sub(marker.len() + 1),
            "step",
            "",
        );
        Ok(if answer.is_empty() {
            marker
        } else {
            format!("{answer}\n{marker}")
        })
    }

    /// `batch` — one chain over every file in a folder, into a new or empty folder.
    async fn batch(
        &self,
        ws: &Workspace,
        args: &serde_json::Value,
    ) -> anyhow::Result<crate::photocraft::BatchReport> {
        let chain = parse_chain(args)?;
        let input = named(args, "input", "name the folder of documents to process")?;
        let input = existing(ws, input, Kind::Dir).await?;
        let output = named(args, "output", "name the folder to write the results into")?;
        let format = super::get_opt_named_str(args, "format")?;
        batch_has_work(&chain, format)?;
        let quality = super::get_opt_u64(args, "quality")?
            .map(|quality| {
                u32::try_from(quality)
                    .ok()
                    .filter(|quality| (1..=100).contains(quality))
                    .ok_or_else(|| anyhow::anyhow!("usage: `quality` must be 1-100, got {quality}"))
            })
            .transpose()?;
        let tiff_layers = super::get_bool(args, "tiff_layers", false)?;
        // Last of all, because it creates the folder: a request refused above must leave nothing
        // behind. The claim comes back with it and is held for the whole batch.
        let (output, created, _claim) = batch_output(ws, output).await?;
        match crate::photocraft::batch(&chain, &input, &output, format, quality, tiff_layers).await
        {
            Ok(report) => Ok(report),
            Err(error) => Err(partway(error, &output, created).await),
        }
    }
}

/// A `batch` must change something: a chain with no steps is a folder-wide conversion when a
/// `format` names the extension to write, and a plain copy of every file under a new name when
/// nothing does.
fn batch_has_work(chain: &[Step], format: Option<&str>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !chain.is_empty() || format.is_some(),
        "usage: `batch` needs a `chain` step or a `format` — hint: with neither, every file would \
         be copied under a new name"
    );
    Ok(())
}

/// The `canvas` argument: the vendor's own parameters for a document created from nothing, absent
/// when the caller opens a file instead. An empty object is refused, because the vendor's `--new`
/// needs a canvas to size and a nameless one is a mistake rather than a default.
fn canvas_arg(args: &serde_json::Value) -> anyhow::Result<Option<serde_json::Value>> {
    const SHAPE: &str = r#"{"width": 800, "height": 600}"#;
    match args.get("canvas") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value @ serde_json::Value::Object(fields)) if !fields.is_empty() => {
            Ok(Some(value.clone()))
        }
        Some(serde_json::Value::Object(_)) => anyhow::bail!(
            "usage: `canvas` is empty — hint: name the new document's parameters, like {SHAPE}"
        ),
        Some(_) => anyhow::bail!(
            "usage: `canvas` must be an object of the new document's parameters — hint: {SHAPE}"
        ),
    }
}

/// A required argument that must name something: absent, null and empty are all refused, the last
/// because read as absent it would silently change what the call asks for.
fn named<'a>(args: &'a serde_json::Value, key: &str, hint: &str) -> anyhow::Result<&'a str> {
    super::get_opt_named_str(args, key)?.ok_or_else(|| required(key, hint))
}

/// The one phrasing of a missing required argument.
fn required(key: &str, hint: &str) -> anyhow::Error {
    anyhow::anyhow!("usage: missing required argument \"{key}\" — hint: {hint}")
}

/// Resolve a caller path the way every tool resolves one — `~/` against the user's home, an
/// absolute path as it is, a relative one against the workspace root — with
/// [`PathAccess::Unrestricted`] holding no boundary on where it may point.
fn resolve_path(ws: &Workspace, requested: &str) -> anyhow::Result<PathBuf> {
    super::path::resolve_tool_path_with_base(requested, ws.as_path(), PathAccess::Unrestricted)
}

/// Claim a caller-named output by creating it: the claim IS the existence check, so an existing
/// file is never replaced and two callers cannot both win the same name. The vendor writes the
/// document into the file it finds at that path.
async fn claim(path: &Path) -> anyhow::Result<()> {
    match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await
    {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => anyhow::bail!(
            "cannot write {}: it already exists — hint: choose another name",
            path.display()
        ),
        Err(error) => anyhow::bail!("cannot create {}: {error}", path.display()),
    }
}

/// What an input path must be.
#[derive(Clone, Copy)]
enum Kind {
    File,
    Dir,
}

impl Kind {
    fn matches(self, meta: &std::fs::Metadata) -> bool {
        match self {
            Self::File => meta.is_file(),
            Self::Dir => meta.is_dir(),
        }
    }

    /// What the path had to be, for the refusal of one that is something else.
    const fn needs(self) -> &'static str {
        match self {
            Self::File => "it is not a regular file — hint: name the document itself, not a folder",
            Self::Dir => "it is not a folder — hint: name the folder to process",
        }
    }

    /// The hint for a path that is not there at all.
    const fn missing(self) -> &'static str {
        match self {
            Self::File => "name an existing document",
            Self::Dir => "name an existing folder",
        }
    }
}

/// An input path that must exist and be what the action needs: the document `run`/`inspect` open,
/// or the folder `batch` processes.
async fn existing(ws: &Workspace, requested: &str, kind: Kind) -> anyhow::Result<PathBuf> {
    let path = resolve_path(ws, requested)?;
    let meta = tokio::fs::metadata(&path).await.map_err(|error| {
        anyhow::anyhow!(
            "cannot open {}: {error} — hint: {}",
            path.display(),
            kind.missing()
        )
    })?;
    anyhow::ensure!(
        kind.matches(&meta),
        "cannot open {}: {}",
        path.display(),
        kind.needs()
    );
    Ok(path)
}

/// A `batch` output folder: created when absent, used when empty, refused when it holds anything —
/// that is what keeps every produced file from landing on one already there. It is made first, since
/// creating it is what settles which call owns an absent name, and then claimed by its canonical
/// path, so two concurrent batches cannot share one however it is spelled. A claim refused leaves the
/// other call's folder alone. The flag says whether this call created it, which is what a failed
/// batch may remove when it wrote nothing.
async fn batch_output(
    ws: &Workspace,
    requested: &str,
) -> anyhow::Result<(PathBuf, bool, FolderClaim)> {
    let path = resolve_path(ws, requested)?;
    let created = prepare_batch_output(&path).await?;
    let claim = FolderClaim::take(&path).await?;
    Ok((path, created, claim))
}

/// Make sure the `batch` output folder is there and empty, and answer whether this call created it.
async fn prepare_batch_output(path: &Path) -> anyhow::Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(meta) => {
            anyhow::ensure!(
                meta.is_dir(),
                "cannot write into {}: it is not a folder — hint: name a folder",
                path.display()
            );
            let mut entries = tokio::fs::read_dir(&path)
                .await
                .map_err(|error| anyhow::anyhow!("cannot read {}: {error}", path.display()))?;
            let occupied = entries
                .next_entry()
                .await
                .map_err(|error| anyhow::anyhow!("cannot read {}: {error}", path.display()))?;
            anyhow::ensure!(
                occupied.is_none(),
                "cannot write into {}: the folder is not empty — hint: choose a new folder",
                path.display()
            );
            Ok(false)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| {
                    anyhow::anyhow!("cannot create {}: {error}", parent.display())
                })?;
            }
            match tokio::fs::create_dir(path).await {
                Ok(()) => Ok(true),
                // Another call claimed the same name between the check and the create: the folder is
                // that call's, and this one must not write into it.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => anyhow::bail!(
                    "cannot create {}: another call claimed it first — hint: choose a new folder",
                    path.display()
                ),
                Err(error) => {
                    anyhow::bail!("cannot create {}: {error}", path.display())
                }
            }
        }
        Err(error) => {
            anyhow::bail!("cannot write into {}: {error}", path.display());
        }
    }
}

/// The batch folders this process is writing into. The vendor names each produced file after its
/// input, so two batches sharing a folder overwrite each other's results — and an existing empty
/// folder cannot be claimed by creating it, so the claim is kept here, where every batch runs.
static CLAIMED_FOLDERS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<PathBuf>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// A batch folder claimed for one call, released when the call ends.
#[derive(Debug)]
struct FolderClaim(PathBuf);

impl FolderClaim {
    /// Claim `dir` for this call, or refuse it as one another call is writing into. The folder is there
    /// by now, so its canonical path is the key: every spelling of it is one claim. A folder that cannot
    /// be resolved falls back to the path as spelled — a narrower claim, never a wrong one.
    async fn take(dir: &Path) -> anyhow::Result<Self> {
        let key = tokio::fs::canonicalize(dir)
            .await
            .unwrap_or_else(|_| dir.to_path_buf());
        let mut claimed = CLAIMED_FOLDERS.lock().unwrap_poison();
        anyhow::ensure!(
            claimed.insert(key.clone()),
            "cannot write into {}: another call is writing into it — hint: choose a new folder",
            dir.display()
        );
        Ok(Self(key))
    }
}

impl Drop for FolderClaim {
    fn drop(&mut self) {
        CLAIMED_FOLDERS.lock().unwrap_poison().remove(&self.0);
    }
}

/// What a failed `batch` answers. The files it already wrote are the caller's work, so they stay and
/// the answer says how many there are; the folder goes only when this call created it and nothing was
/// written into it. Counted and stated exactly as the successful answer states it, and never claiming
/// the batch ran: a failure before the vendor started — a copy that could not be installed, a recipe
/// that could not be written — leaves no file either.
async fn partway(error: anyhow::Error, output: &Path, created: bool) -> anyhow::Error {
    let written = crate::photocraft::files_written(output).await;
    if written.count == 0 && written.whole {
        if created {
            let _ = tokio::fs::remove_dir(output).await;
        }
        return anyhow::anyhow!("{error}\n{}", written.written_into(output));
    }
    anyhow::anyhow!(
        "{error}\nthe batch failed partway: {}",
        written.written_into(output)
    )
}

/// A fresh `run` output in the caller's output directory: the input's own extension, so a layered
/// document stays in its own format instead of being flattened into PNG by default — `png` when the
/// input has none, names a format the list does not keep, or when there is no input at all because
/// the canvas was created from nothing.
///
/// The name is CLAIMED by creating it, not checked and then handed over: two concurrent runs cannot
/// pick the same name, and nothing already there is replaced. The vendor writes the document into
/// the file it finds at that path.
async fn fresh_output(ws: &Workspace, input: Option<&Path>) -> anyhow::Result<PathBuf> {
    let dir = super::output_directory(ws, super::GENERATED_DIR);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|error| anyhow::anyhow!("cannot create {}: {error}", dir.display()))?;
    let extension = input
        .and_then(crate::util::extension_of)
        .filter(|extension| WRITABLE_OUTPUT_EXTENSIONS.contains(&extension.as_str()))
        .unwrap_or_else(|| "png".to_string());
    let name = format!("photocraft_{}.{extension}", crate::util::unix_millis());
    let (file, path) = crate::util::create_unique_file(&dir, &name)
        .await
        .map_err(|error| anyhow::anyhow!("cannot create a name in {}: {error}", dir.display()))?;
    drop(file);
    Ok(path)
}

/// The shape of one chain step, spelled once for every refusal that has to show one.
const STEP_SHAPE: &str = r#"{"command": "<id>", "params": {...}}"#;

/// The chain an argument carries: an array of [`STEP_SHAPE`], an absent or null value being an
/// empty chain.
fn parse_chain(args: &serde_json::Value) -> anyhow::Result<Vec<Step>> {
    let Some(chain) = args.get("chain") else {
        return Ok(Vec::new());
    };
    if chain.is_null() {
        return Ok(Vec::new());
    }
    let Some(steps) = chain.as_array() else {
        anyhow::bail!("usage: `chain` must be an array of steps — hint: {STEP_SHAPE} per step");
    };
    steps.iter().map(parse_step).collect()
}

/// One chain step: a non-empty string `command`, and an optional object `params` (a null one is
/// absent).
fn parse_step(step: &serde_json::Value) -> anyhow::Result<Step> {
    let Some(object) = step.as_object() else {
        anyhow::bail!("usage: each `chain` step must be an object — hint: {STEP_SHAPE}");
    };
    let Some(command) = object.get("command").and_then(serde_json::Value::as_str) else {
        anyhow::bail!("usage: each `chain` step needs a string `command` — hint: {STEP_SHAPE}");
    };
    anyhow::ensure!(
        !command.is_empty(),
        "usage: a `chain` step's `command` must not be empty — hint: name an id from the \
         `commands` action"
    );
    let params = match object.get("params") {
        None | Some(serde_json::Value::Null) => None,
        Some(value @ serde_json::Value::Object(_)) => Some(value.clone()),
        Some(_) => {
            anyhow::bail!("usage: a `chain` step's `params` must be an object — hint: {STEP_SHAPE}")
        }
    };
    Ok(Step {
        command: command.to_string(),
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().expect("test dir");
        let ws = Workspace::from_path(dir.path());
        (dir, ws)
    }

    #[tokio::test]
    async fn a_missing_or_unknown_action_names_the_four_actions() {
        let (_dir, ws) = workspace();
        let tool = PhotocraftTool;
        for args in [json!({}), json!({"action": "nope"})] {
            let error = tool.execute(&ws, args).await.expect_err("refused");
            let text = error.to_string();
            for action in ALL {
                assert!(text.contains(action), "{action} missing from: {text}");
            }
        }
    }

    #[test]
    fn a_chain_is_the_steps_it_carries_and_an_absent_one_is_empty() {
        assert!(parse_chain(&json!({})).expect("absent chain").is_empty());
        assert!(
            parse_chain(&json!({"chain": null}))
                .expect("null chain")
                .is_empty()
        );

        let chain = parse_chain(&json!({
            "chain": [
                {"command": "resize", "params": {"width": 800}},
                {"command": "sharpen"},
                {"command": "autocrop", "params": null}
            ]
        }))
        .expect("a valid chain");
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].command, "resize");
        assert_eq!(chain[0].params, Some(json!({"width": 800})));
        assert_eq!(chain[1].command, "sharpen");
        assert_eq!(chain[1].params, None);
        // A null `params` counts as absent.
        assert_eq!(chain[2].params, None);
    }

    #[test]
    fn a_chain_that_is_not_an_array_of_named_steps_is_refused() {
        assert!(parse_chain(&json!({"chain": "resize"})).is_err());
        let empty = parse_chain(&json!({"chain": [{"command": ""}]}))
            .err()
            .expect("empty command");
        assert!(empty.to_string().contains("must not be empty"), "{empty}");
        let nameless = parse_chain(&json!({"chain": [{"params": {}}]}))
            .err()
            .expect("no command");
        assert!(
            nameless.to_string().contains("needs a string `command`"),
            "{nameless}"
        );
        let bad_params = parse_chain(&json!({"chain": [{"command": "x", "params": 1}]}))
            .err()
            .expect("params is not an object");
        assert!(
            bad_params.to_string().contains("must be an object"),
            "{bad_params}"
        );
    }

    #[test]
    fn a_path_expands_a_tilde_and_resolves_a_relative_one_against_the_workspace() {
        let (_dir, ws) = workspace();
        let home = directories::UserDirs::new()
            .expect("a home directory")
            .home_dir()
            .to_path_buf();
        assert_eq!(
            resolve_path(&ws, "~/x.png").expect("a home path"),
            home.join("x.png")
        );

        let absolute = std::env::temp_dir().join("x.png");
        assert_eq!(
            resolve_path(&ws, absolute.to_str().expect("utf-8 temp path"))
                .expect("an absolute path"),
            absolute
        );

        assert_eq!(
            resolve_path(&ws, "sub/x.png").expect("a workspace-relative path"),
            ws.as_path().join("sub/x.png")
        );
        // `~user` is refused rather than answered with `$HOME/user`, as everywhere else.
        assert!(resolve_path(&ws, "~someone/x.png").is_err());
    }

    #[tokio::test]
    async fn an_existing_run_output_is_refused_and_left_untouched() {
        let (dir, ws) = workspace();
        let input = dir.path().join("in.png");
        std::fs::write(&input, b"x").unwrap();
        let output = dir.path().join("out.png");
        std::fs::write(&output, b"x").unwrap();
        let tool = PhotocraftTool;
        let error = tool
            .execute(
                &ws,
                json!({"action": "run", "input": input.to_str().unwrap(), "output": output.to_str().unwrap()}),
            )
            .await
            .expect_err("an existing output is refused");
        assert!(error.to_string().contains("choose another name"), "{error}");
        // Refused, not replaced: the caller's file is untouched.
        assert_eq!(std::fs::read(&output).unwrap(), b"x");
    }

    #[tokio::test]
    async fn a_fresh_output_keeps_the_formats_the_product_hands_back_and_is_png_for_any_other() {
        let (dir, ws) = workspace();
        let extension =
            |path: &Path| crate::util::extension_of(path).expect("a fresh name carries one");
        // A format the product hands back is kept, so a layered document is not flattened by
        // default and a rare export is not turned into a PNG either.
        for (name, expected) in [
            ("plate.png", "png"),
            ("plate.JPG", "jpg"),
            ("plate.tiff", "tiff"),
            ("plate.psd", "psd"),
            ("plate.psb", "psb"),
            ("plate.ora", "ora"),
            ("plate.pcraft", "pcraft"),
            ("plate.webp", "webp"),
            ("plate.gif", "gif"),
            ("plate.exr", "exr"),
            ("plate.pdf", "pdf"),
            ("plate.qoi", "qoi"),
            ("plate.fits", "fits"),
            ("plate.pgm", "pgm"),
            ("plate.pfm", "pfm"),
            ("plate.dib", "dib"),
            ("plate.icb", "icb"),
            ("plate.cin", "cin"),
            ("plate.ilbm", "ilbm"),
        ] {
            let input = dir.path().join(name);
            std::fs::write(&input, b"x").unwrap();
            let fresh = fresh_output(&ws, Some(&input)).await.expect("a fresh name");
            assert_eq!(extension(&fresh), expected, "{name}");
        }
        // Formats the engine only reads, one its release build leaves out entirely, or no format at
        // all: a PNG, never a destination whose save would always fail.
        for name in [
            "IMG_0042.HEIC",
            "drawing.kra",
            "poster.afdesign",
            "shot.dng",
            "shot.avif",
            "scan.pcx",
            "scan",
        ] {
            let input = dir.path().join(name);
            std::fs::write(&input, b"x").unwrap();
            let fresh = fresh_output(&ws, Some(&input)).await.expect("a fresh name");
            assert_eq!(extension(&fresh), "png", "{name}");
        }
        // It lands in the caller's generated area under the product's own name, already created.
        let input = dir.path().join("plate.png");
        let generated = ws.as_path().join(super::super::GENERATED_DIR);
        let fresh = fresh_output(&ws, Some(&input)).await.expect("a fresh name");
        assert_eq!(fresh.parent(), Some(generated.as_path()));
        assert!(
            fresh
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with("photocraft_")),
            "{}",
            fresh.display()
        );
        assert!(fresh.exists(), "a fresh name is claimed");
    }

    #[tokio::test]
    async fn a_fresh_name_is_never_reused() {
        let (dir, ws) = workspace();
        let input = dir.path().join("in.png");
        std::fs::write(&input, b"x").unwrap();
        let generated = ws.as_path().join(super::super::GENERATED_DIR);
        std::fs::create_dir_all(&generated).unwrap();
        // A name already in the way is stepped over, never handed out twice.
        let taken = generated.join(format!("photocraft_{}.png", crate::util::unix_millis()));
        std::fs::write(&taken, b"x").unwrap();
        let first = fresh_output(&ws, Some(&input)).await.expect("a fresh name");
        let second = fresh_output(&ws, Some(&input)).await.expect("a fresh name");
        assert_ne!(first, taken);
        assert_ne!(first, second);
        // Both are claimed by creating them, so neither can go to another run.
        assert!(
            first.exists(),
            "a fresh name is claimed: {}",
            first.display()
        );
        assert!(
            second.exists(),
            "a fresh name is claimed: {}",
            second.display()
        );
    }

    #[tokio::test]
    async fn a_caller_named_output_is_claimed_by_creating_it_and_a_taken_name_is_refused() {
        let (dir, _ws) = workspace();
        let path = dir.path().join("out.png");
        claim(&path).await.expect("an absent name is claimed");
        assert!(path.exists(), "the claim creates the file it reserved");
        let error = claim(&path).await.expect_err("a taken name is refused");
        let text = error.to_string();
        assert!(text.contains("already exists"), "{text}");
        assert!(text.contains("choose another name"), "{text}");
    }

    #[tokio::test]
    async fn a_refused_chain_claims_nothing_on_disk() {
        let (dir, ws) = workspace();
        let input = dir.path().join("in.png");
        std::fs::write(&input, b"x").unwrap();
        let tool = PhotocraftTool;
        let error = tool
            .execute(
                &ws,
                json!({"action": "run", "input": input.to_str().unwrap(), "chain": "resize"}),
            )
            .await
            .expect_err("a malformed chain is refused");
        assert!(error.to_string().contains("must be an array"), "{error}");
        let generated = ws.as_path().join(super::super::GENERATED_DIR);
        let claimed = std::fs::read_dir(&generated).map_or(0, std::iter::Iterator::count);
        assert_eq!(claimed, 0, "a refused request leaves nothing behind");
    }

    #[tokio::test]
    async fn a_canvas_created_from_nothing_is_asked_for_instead_of_an_input_and_is_never_both() {
        let (dir, ws) = workspace();
        let input = dir.path().join("in.png");
        std::fs::write(&input, b"x").unwrap();
        let tool = PhotocraftTool;
        // Both at once is a contradiction: `input` opens a document, `canvas` starts from nothing.
        let error = tool
            .execute(
                &ws,
                json!({
                    "action": "run",
                    "input": input.to_str().unwrap(),
                    "canvas": {"width": 8, "height": 6},
                }),
            )
            .await
            .expect_err("both is refused");
        assert!(error.to_string().contains("not both"), "{error}");
        // Neither is a request with nothing to open.
        let error = tool
            .execute(&ws, json!({"action": "run"}))
            .await
            .expect_err("neither is refused");
        assert!(error.to_string().contains("canvas"), "{error}");
        // A canvas that names no parameters is refused rather than handed to the vendor blank, and
        // the reason is that it is empty — not that it is the wrong shape.
        let error = tool
            .execute(&ws, json!({"action": "run", "canvas": {}}))
            .await
            .expect_err("an empty canvas is refused");
        assert!(error.to_string().contains("`canvas` is empty"), "{error}");
        // A value of any other shape is the wrong kind of argument.
        let error = tool
            .execute(&ws, json!({"action": "run", "canvas": 800}))
            .await
            .expect_err("a canvas that is not an object is refused");
        assert!(error.to_string().contains("must be an object"), "{error}");
    }

    #[tokio::test]
    async fn a_canvas_has_no_input_format_so_its_fresh_output_is_png() {
        let (_dir, ws) = workspace();
        let fresh = fresh_output(&ws, None).await.expect("a fresh name");
        assert_eq!(crate::util::extension_of(&fresh).as_deref(), Some("png"));
        // And it lands in the caller's generated area, claimed by creating it.
        assert_eq!(
            fresh.parent(),
            Some(ws.as_path().join(super::super::GENERATED_DIR).as_path())
        );
        assert!(fresh.exists(), "a fresh name is claimed");
    }

    #[test]
    fn a_picture_gets_an_image_marker_and_every_other_format_a_file_marker() {
        let tool = PhotocraftTool;
        for name in ["a.png", "a.JPG", "a.jpeg", "a.webp"] {
            assert!(
                tool.format_media_result(Path::new(name))
                    .starts_with("[IMAGE:"),
                "{name}"
            );
        }
        for name in ["a.psd", "a.pcraft", "a.tif", "a.tiff", "a.heic", "a.exr"] {
            assert!(
                tool.format_media_result(Path::new(name))
                    .starts_with("[FILE:"),
                "{name}"
            );
        }
    }

    #[tokio::test]
    async fn a_batch_output_folder_may_be_absent_or_empty_but_never_occupied() {
        let (dir, ws) = workspace();
        let absent = dir.path().join("fresh");
        let (created, made, claim) = batch_output(&ws, absent.to_str().unwrap())
            .await
            .expect("an absent folder is created");
        assert_eq!(created, absent);
        assert!(created.is_dir());
        assert!(made, "a folder this call created is reported as such");
        // The claim travels with the folder, so it is held for as long as the batch writes into it.
        let busy = batch_output(&ws, absent.to_str().unwrap())
            .await
            .expect_err("a folder this call is writing into is refused");
        assert!(
            busy.to_string().contains("another call is writing into it"),
            "{busy}"
        );
        // The refused call leaves the folder alone: it is the call holding the claim that writes into
        // it, so nothing of theirs may be removed.
        assert!(absent.is_dir(), "a refused claim removes nothing");
        drop(claim);

        let empty = dir.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let (used, made, claim) = batch_output(&ws, empty.to_str().unwrap())
            .await
            .expect("empty");
        assert_eq!(used, empty);
        assert!(
            !made,
            "a folder the caller made is not this call's to remove"
        );
        drop(claim);

        std::fs::write(empty.join("already.png"), b"x").unwrap();
        let error = batch_output(&ws, empty.to_str().unwrap())
            .await
            .expect_err("a non-empty folder is refused");
        assert!(error.to_string().contains("not empty"), "{error}");

        // A folder whose own parents are absent too: the parents are made, the folder itself is
        // created — the one step two concurrent calls can race on, and the loser is refused.
        let nested = dir.path().join("a/b/out");
        let (made_dir, made, _claim) = batch_output(&ws, nested.to_str().unwrap())
            .await
            .expect("a nested folder is created with its parents");
        assert!(made_dir.is_dir(), "{}", made_dir.display());
        assert!(made, "a folder this call created is reported as such");
    }

    #[tokio::test]
    async fn a_failed_batch_keeps_what_it_wrote_and_removes_an_empty_folder_it_made() {
        let (dir, _ws) = workspace();
        // Nothing written, and the folder was this call's own: it goes. The answer never claims the
        // batch ran, because a failure before the vendor started leaves no file either.
        let fresh = dir.path().join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let answer = partway(
            anyhow::anyhow!("the vendor program could not be installed"),
            &fresh,
            true,
        )
        .await;
        let text = answer.to_string();
        assert!(text.contains("no file was written into"), "{text}");
        assert!(!text.contains("the batch failed"), "{text}");
        assert!(!fresh.exists(), "an empty folder this call made is removed");

        // Twelve of twenty written: the work stays, and the answer says how much of it there is.
        let part = dir.path().join("part");
        std::fs::create_dir_all(&part).unwrap();
        for name in ["a.png", "b.png", "c.png"] {
            std::fs::write(part.join(name), b"x").unwrap();
        }
        let answer = partway(anyhow::anyhow!("exit status 1"), &part, true).await;
        let text = answer.to_string();
        assert!(text.contains("failed partway"), "{text}");
        assert!(text.contains("3 file(s)"), "{text}");
        assert_eq!(std::fs::read_dir(&part).unwrap().count(), 3);
    }

    #[tokio::test]
    async fn a_folder_another_call_is_writing_into_is_refused() {
        // An existing empty folder cannot be claimed by creating it, so two concurrent batches could
        // both write into it and the vendor's input-derived names would overwrite each other.
        let (dir, _ws) = workspace();
        let path = dir.path().join("out");
        std::fs::create_dir(&path).unwrap();
        let first = FolderClaim::take(&path)
            .await
            .expect("a free folder is claimed");
        let error = FolderClaim::take(&path)
            .await
            .expect_err("a claimed folder is refused");
        assert!(
            error
                .to_string()
                .contains("another call is writing into it"),
            "{error}"
        );
        // One directory spelled two ways is one claim: the key is the folder's canonical path.
        let spelled = path.join("..").join("out");
        let error = FolderClaim::take(&spelled)
            .await
            .expect_err("another spelling of a claimed folder is refused");
        assert!(
            error
                .to_string()
                .contains("another call is writing into it"),
            "{error}"
        );
        drop(first);
        FolderClaim::take(&spelled)
            .await
            .expect("the claim is released with the call");
        // A folder that is not there is claimed under the path as spelled: a narrower key, and the
        // create that follows is what settles an absent name between two calls.
        let absent = dir.path().join("fresh");
        let held = FolderClaim::take(&absent)
            .await
            .expect("an unresolvable folder is still claimed");
        let error = FolderClaim::take(&absent)
            .await
            .expect_err("the same spelling is one claim");
        assert!(
            error
                .to_string()
                .contains("another call is writing into it"),
            "{error}"
        );
        drop(held);
    }

    #[test]
    fn a_batch_needs_a_step_or_a_format_to_convert_into() {
        let steps = [Step {
            command: "image.adjustments.invert".to_string(),
            params: None,
        }];
        assert!(batch_has_work(&steps, None).is_ok());
        // No step is a folder-wide conversion when the format says what to write.
        assert!(batch_has_work(&[], Some("png")).is_ok());
        let error = batch_has_work(&[], None).expect_err("a plain copy is refused");
        assert!(
            error.to_string().contains("copied under a new name"),
            "{error}"
        );
    }
}

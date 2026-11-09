//! Document reads: what the shared converter ([`crate::document`]) extracts from
//! is converted here and delivered two ways from ONE reading — the read tool's
//! (extracted text inlined, or spilled when it does not fit the tool output
//! budget, at most [`crate::document::MAX_READ_PICTURES`] extracted pictures
//! attached as images) and the command line's (the whole text and every raster
//! written into a folder of their own). Detection, conversion, scrubbing and the
//! reader's wording are shared; only the delivery is the caller's own — the read
//! tool attaches a bounded number of pictures and reports the rest, and the
//! command line has no length limit at all.
//!
//! The read tool's artifacts (a page raster, an embedded image, a spilled text
//! body) are per-call intermediates in a uniquely named `read_<nonce>` directory,
//! removed when the run ends like a shell spill. Every access but the
//! workspace-only one writes them into the daemon temp service area
//! ([`crate::tools::shell::agent_temp_dir`]); the workspace-only read, which
//! cannot open temp paths, into its caller's workspace `uploads/` (a re-read
//! re-converts). What a crash leaves behind is reclaimed there too — by the OS temp
//! sweep and the periodic temp cleaner in the first case, by the workspace media sweep
//! in the second, which removes the leftover files once they are a day old but not the
//! empty directory. That `uploads/` parent stays in place (the inbound attachment copy
//! writes it too, so removing it could race that writer), and nothing is written beside
//! the document.

use crate::Workspace;
use crate::document::DocOutcome;
use crate::reader_output::Unshown;
use crate::tools::path::PathAccess;
use crate::tools::{ImagePayload, ImagePayloadSource, ToolOutput};
use anyhow::Context;
use std::path::{Path, PathBuf};

/// Bytes read from the head of the file for the format sniff: every arm of
/// [`crate::document::needs_extraction`] is magic-local, and a CFB container —
/// whose own directory lies further in — is extracted on its magic alone.
const HEAD_SNIFF_BYTES: usize = 16;

/// Answer for a document whose conversion could not start (no temp area, or an
/// unwritable workspace) — the file itself is fine.
const NO_ARTIFACT_DIR_REASON: &str = "could not be converted";

/// The reader's one document decision: `Some(file_name)` when `path` is a
/// regular file the shared converter extracts from — the same content-first
/// detection the inbound attachment path uses — and `None` when the ordinary
/// read applies. `Err` only for a document over the product's size ceiling.
async fn document_target(path: &Path) -> anyhow::Result<Option<String>> {
    // Only a regular file is converted: a directory keeps the ordinary listing
    // read, and a special file is never opened (the converter's read is unbounded).
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Ok(None);
    };
    let Some(head) = peek_head(path).await else {
        return Ok(None);
    };
    // Classified before the size check, so an oversized NON-document falls back
    // to the ordinary 10 MB message.
    if !crate::document::needs_extraction(&head, name) {
        return Ok(None);
    }
    // The conversion cap, not the read tool's own 10 MB text cap (both are
    // stated in the read prompts).
    crate::tools::check_size_within(
        &meta,
        crate::util::FILE_MAX_BYTES,
        "File too large to convert",
    )?;
    Ok(Some(name.to_owned()))
}

/// A converted reading's parts — the [`DocOutcome::Text`] fields as one value.
struct ConvertedDocument {
    text: String,
    images: Vec<PathBuf>,
    notes: Vec<String>,
    unshown: Unshown,
    all_page_text_lost: bool,
    page_text: bool,
}

/// What one conversion attempt produced.
enum Conversion {
    /// A document the converter read.
    Reading(ConvertedDocument),
    /// A container it does not take (an Outlook `.msg`, say, or a file that
    /// changed between the sniff and the conversion): the ordinary read applies.
    Ordinary,
    /// The reader's own short reason for a document it could not read.
    Unreadable(String),
}

/// Convert `path` into `dir` as both entrances deliver it, so detection,
/// conversion and scrubbing are one implementation for both.
async fn convert_reading(path: &Path, name: &str, dir: &Path) -> Conversion {
    match crate::document::convert_document_file(path, name, dir).await {
        DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            all_page_text_lost,
            page_text,
        } => Conversion::Reading(ConvertedDocument {
            // The read tool returns `false` from `should_scrub_output`, and the
            // scrub `read_resolved` does instead is path-based — it sees the
            // container's path, never what was extracted from it — so this text
            // is scrubbed here for both entrances. (The inbound route delivers
            // the same text unscrubbed.)
            text: crate::util::scrub_credentials(text.trim()),
            images,
            notes,
            unshown,
            all_page_text_lost,
            page_text,
        }),
        DocOutcome::Unreadable { reason } => Conversion::Unreadable(reason),
        DocOutcome::Unsupported => Conversion::Ordinary,
    }
}

/// Convert `res` when it is a container the shared converter extracts from — the
/// same detection the inbound path uses, so the two cannot disagree about what a
/// document is. `Ok(None)` when it is not (the ordinary read applies), `Err` only
/// for an oversized container.
pub(super) async fn read_document(
    ws: &Workspace,
    res: &super::read::ResolvedRead,
    access: PathAccess,
) -> anyhow::Result<Option<ToolOutput>> {
    let Some(name) = document_target(&res.path).await? else {
        return Ok(None);
    };

    // No artifact directory, no conversion — a plain answer, not a failed call.
    let dir = match create_artifact_dir(ws, access).await {
        Ok(dir) => dir,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to create a document artifact directory");
            return Ok(Some(plain_answer(res, NO_ARTIFACT_DIR_REASON)));
        }
    };
    crate::tools::shell::record_spill_owner(dir.clone());

    let outcome = match convert_reading(&res.path, &name, &dir).await {
        Conversion::Reading(reading) => Some(compose_output(res, &dir, reading).await),
        Conversion::Unreadable(reason) => Some(plain_answer(res, &reason)),
        // The file changed under us, or the container names no Office document
        // at all: the ordinary read applies.
        Conversion::Ordinary => None,
    };
    // Empty-only removal (`remove_dir` refuses a non-empty directory): an answer
    // pointing at nothing leaves no directory behind, anything it points at keeps
    // one until the run-end hook reclaims it.
    let _ = tokio::fs::remove_dir(&dir).await;
    Ok(outcome)
}

/// Read `args`'s `path` the way the read tool does, for the command line: the
/// same detection, conversion and wording, with the whole text and every produced
/// image written into a folder of their own inside `destination` instead of
/// inlined, attached or spilled.
///
/// `destination` is the caller's `--output-dir`, already resolved to an existing,
/// writable directory by the frontend ([`crate::cli::resolve_output_dir`]) and
/// not judged again here. A failure after the folder is made removes it; a killed
/// process cannot clean up, so what it wrote stays.
pub(crate) async fn read_for_command_line(
    ws: &Workspace,
    args: &serde_json::Value,
    destination: &Path,
) -> anyhow::Result<super::ToolAnswer> {
    let path = super::get_str(args, "path")?;
    // The read tool's own resolution — the live-store refusal included — runs
    // before anything is opened or written.
    let res = super::read::resolve_content_read(ws, path, PathAccess::Unrestricted).await?;

    let display = res.path.display().to_string();
    let fallback = crate::util::name_stem(crate::util::file_name_or_path(&display)).to_string();
    let base =
        crate::tools::document::sanitize_base(super::get_opt_str(args, "file_name"), &fallback);
    let base = if base.is_empty() {
        "document".to_string()
    } else {
        base
    };
    // The reader's own document decision, before the destination is touched: an
    // input over the size ceiling is refused with no folder made.
    let target = document_target(&res.path).await?;
    let folder = crate::util::create_unique_dir(destination, &base)
        .await
        .with_context(|| format!("failed to create a folder in {}", destination.display()))?;

    let answer = write_command_line_reading(&res, &folder, target).await;
    if answer.is_err() {
        let _ = tokio::fs::remove_dir_all(&folder).await;
    }
    answer
}

/// Write one reading's whole delivery into `folder`: the text as
/// `<folder>.txt`, every produced image where the converter already wrote it,
/// and the reader's own report beside them. Nothing here bounds the content; the
/// only thing bounded is the answer's own listing.
async fn write_command_line_reading(
    res: &super::read::ResolvedRead,
    folder: &Path,
    target: Option<String>,
) -> anyhow::Result<super::ToolAnswer> {
    let display = res.path.display().to_string();
    // Whether the reading came from the converter or from the ordinary fallback:
    // only a converted document owes the reader's document notes.
    let (reading, is_document) = match target {
        Some(name) => match convert_reading(&res.path, &name, folder).await {
            Conversion::Reading(reading) => (reading, true),
            // The read tool's own rendering of a document it cannot read: the
            // caller sends it as the refusal, and the folder leaves nothing.
            Conversion::Unreadable(reason) => {
                return Err(anyhow::anyhow!(answer_line(&display, reason)));
            }
            Conversion::Ordinary => (ordinary_reading(res).await?, false),
        },
        None => (ordinary_reading(res).await?, false),
    };

    let mut produced: Vec<PathBuf> = Vec::new();
    if !reading.text.is_empty() {
        let folder_name = folder
            .file_name()
            .expect("a created folder has a name")
            .to_string_lossy();
        let text_path = folder.join(format!("{folder_name}.txt"));
        tokio::fs::write(&text_path, reading.text.as_bytes())
            .await
            .with_context(|| format!("failed to write {}", text_path.display()))?;
        produced.push(text_path);
    }
    produced.extend(reading.images.iter().cloned());

    // Every image is written, so every one was delivered — the basis the
    // reader's no-text sentence names images on. The ordinary read of an empty
    // file produces an empty body, and that empty answer is the reader's own.
    let images_complete = !reading.images.is_empty();
    let mut report = Vec::new();
    if is_document && reading.text.is_empty() && !reading.all_page_text_lost {
        report.push(no_text_line(&display, images_complete));
    }
    report.extend(reading_notes(&display, &reading, images_complete));

    let answer = if produced.is_empty() {
        // Nothing was written: the folder goes away and the reader's own words
        // are the whole answer.
        let _ = tokio::fs::remove_dir_all(folder).await;
        report.join("\n")
    } else {
        let names: Vec<String> = produced
            .iter()
            .map(|path| crate::util::file_name_or_path(&path.display().to_string()).to_string())
            .collect();
        let mut sizes = Vec::with_capacity(produced.len());
        for path in &produced {
            let meta = tokio::fs::metadata(path)
                .await
                .with_context(|| format!("failed to read back {}", path.display()))?;
            sizes.push(meta.len());
        }
        let mut lines = vec![crate::tools::document::describe("read", &names, &sizes)];
        lines.extend(report.iter().cloned());
        lines.extend(
            produced
                .iter()
                .map(|path| crate::tools::document::file_marker(path)),
        );
        let full = lines.join("\n");
        if full.len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES {
            full
        } else {
            // The listing and the description are replaced by the folder once;
            // every file is still written, and the report lines stay.
            let lead = format!(
                "read {} files — they are all in {}",
                produced.len(),
                folder.display()
            );
            std::iter::once(lead)
                .chain(report)
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    // A reading the reader failed on every page of is a PARTIAL read: what it
    // could not read is named in the notes, and the family's exit classes keep a
    // partial operation in the failed class. Anything the reader did read is
    // delivered whole above.
    Ok(if reading.all_page_text_lost {
        super::ToolAnswer::partial(answer)
    } else {
        super::ToolAnswer::done(answer)
    })
}

/// The ordinary read of a file the converter does not take, as the parts a
/// reading would have produced: no notes, and the text is the file's whole
/// line-numbered rendering. An empty file has none, and the reader's answer for it
/// is empty too. A raster keeps the read tool's one-line annotation as its text;
/// the payload the agent's read would attach is the caller's own input file, so
/// nothing is copied beside the reading.
async fn ordinary_reading(res: &super::read::ResolvedRead) -> anyhow::Result<ConvertedDocument> {
    // The read tool's own content read, with no arguments and so in its default mode.
    let text = super::read::read_resolved(res, &serde_json::json!({})).await?;
    Ok(ConvertedDocument {
        // The body is the file's own content wherever it has any — the fact the
        // no-text sentence is gated on, so a whitespace-only body is still text.
        page_text: !text.is_empty(),
        text,
        images: Vec::new(),
        notes: Vec::new(),
        unshown: Unshown::default(),
        all_page_text_lost: false,
    })
}

/// The reader's no-text sentence: the one naming images only when every picture
/// the reading produced was actually delivered.
fn no_text_line(display: &str, images_complete: bool) -> String {
    answer_line(display, crate::document::no_text_note(images_complete))
}

/// The reader's own notes for one reading, in its order: what the reading found
/// and did not show, its own notes, then — when no page produced text — the
/// no-text sentence. Empty when every page's text was lost: the notes already say
/// so, and "no text" would report a reader failure as a document with none.
fn reading_notes(display: &str, reading: &ConvertedDocument, images_complete: bool) -> Vec<String> {
    let mut notes: Vec<String> = reading
        .unshown
        .lines()
        .iter()
        .chain(&reading.notes)
        .map(|body| answer_line(display, body))
        .collect();
    if !reading.page_text && !reading.all_page_text_lost && !reading.text.is_empty() {
        notes.push(no_text_line(display, images_complete));
    }
    notes
}

/// The plain answer naming `reason` for a document this tool cannot read: never
/// a failed call, and never its undecodable raw bytes reported as a success.
fn plain_answer(res: &super::read::ResolvedRead, reason: &str) -> ToolOutput {
    ToolOutput {
        text: crate::tools::with_recovery_note(
            res.recovery_note.as_deref(),
            answer_line(&res.path.display().to_string(), reason),
        ),
        image_payloads: Vec::new(),
        text_is_content: true,
        preserve_text: false,
    }
}

/// One line of a converted-document answer: `[<path>: <what it says>]` — the
/// envelope every note here shares.
fn answer_line(display: &str, body: impl std::fmt::Display) -> String {
    format!("[{display}: {body}]")
}

/// A fresh uniquely named directory for this call's artifacts — its location is
/// the module header's subject.
async fn create_artifact_dir(ws: &Workspace, access: PathAccess) -> anyhow::Result<PathBuf> {
    // The workspace-only read cannot open a temp path, so its artifacts go to a
    // directory it can reach; every other access uses the shared agent temp area.
    let parent = if access == PathAccess::Workspace {
        ws.as_path().join("uploads")
    } else {
        crate::tools::shell::agent_temp_dir()
            .ok_or_else(|| anyhow::anyhow!("no temp directory available for document conversion"))?
    };
    tokio::fs::create_dir_all(&parent)
        .await
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let dir = parent.join(format!("read_{:016x}", rand::random::<u64>()));
    tokio::fs::create_dir(&dir)
        .await
        .with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir)
}

/// The leading bytes of `path` for the format sniff, read in a loop so a short
/// read cannot misclassify a document. `None` leaves the decision to the ordinary read.
async fn peek_head(path: &Path) -> Option<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await.ok()?;
    let mut head = [0u8; HEAD_SNIFF_BYTES];
    let mut filled = 0;
    while filled < HEAD_SNIFF_BYTES {
        match file.read(&mut head[filled..]).await {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(_) => return None,
        }
    }
    Some(head[..filled].to_vec())
}

/// The payloads a reading's pictures became, and how many could not be encoded.
/// An image that cannot be encoded is counted rather than failing the call,
/// because the document itself was read.
async fn attach_images(images: &[PathBuf]) -> (Vec<ImagePayload>, usize) {
    let mut attached = Vec::new();
    let mut unattached = 0usize;
    // The payloads carry no recovery note: a document answer is content, so its
    // text block already opens with that note and repeating it on every image
    // annotation would say the same thing 1+N times.
    for image in images {
        match crate::util::local_image_to_compressed_data_uri_with_meta(image).await {
            // The image is this tool's own artifact — it never opened it as a
            // file — so the annotation labels it a produced image.
            Ok(meta) => attached.push(ImagePayload::from_compressed_meta(
                image,
                meta,
                None,
                ImagePayloadSource::Generated,
            )),
            Err(e) => {
                tracing::warn!(
                    path = %image.display(),
                    error = %e,
                    "Failed to encode an extracted document image"
                );
                unattached += 1;
            }
        }
    }
    (attached, unattached)
}

/// Turn a converted reading into the read tool's result: attach the pictures
/// [`crate::document::MAX_READ_PICTURES`] allows, and say what was delivered and
/// what the reading did not show.
async fn compose_output(
    res: &super::read::ResolvedRead,
    dir: &Path,
    reading: ConvertedDocument,
) -> ToolOutput {
    let display = res.path.display().to_string();
    let (pictures, left_out) = crate::document::attachable_pictures(&reading.images);
    let (attached, unattached) = attach_images(pictures).await;
    // "The content was provided as images" is only true when every picture the
    // reading produced is attached; with some left out, the plain sentence plus
    // the limit's own note say what actually happened.
    let images_complete = !attached.is_empty() && left_out == 0;

    let mut text_block = if reading.text.is_empty() {
        // The reader failed on every page it was asked for: the notes below name
        // those pages, and the "no text" sentences would report a reader failure
        // as a document that has no text.
        if reading.all_page_text_lost {
            String::new()
        } else {
            no_text_line(&display, images_complete)
        }
    } else {
        format!(
            "{}\n\n{}",
            answer_line(&display, "extracted text follows"),
            reading.text
        )
    };
    let mut trailing = reading_notes(&display, &reading, images_complete);
    if left_out > 0 {
        trailing.push(answer_line(
            &display,
            crate::document::left_out_pictures_note(reading.images.len(), left_out),
        ));
    }
    if unattached > 0 {
        // Not the inbound sentence ("could not be read"): there the image
        // pipeline rejected an image it was handed, here the image could not be
        // encoded. The folder is named so it stays reachable.
        trailing.push(answer_line(
            &display,
            format!(
                "{unattached} extracted image(s) could not be attached; they are in {} — read them individually if you need them",
                dir.display()
            ),
        ));
    }

    // The whole answer — text block and recovery note included — must fit the
    // shared tool output budget, or `format_output` truncates it mid-answer. (The
    // per-image annotations the agent loop appends afterwards are outside this
    // budget; they are bounded by the picture limit.) The text is spilled when
    // the answer would not fit.
    let compose = |text_block: &str| {
        let mut lines: Vec<&str> = Vec::with_capacity(1 + trailing.len());
        // Empty only when every page's text failed: the notes are then the whole
        // answer's opening.
        if !text_block.is_empty() {
            lines.push(text_block);
        }
        lines.extend(trailing.iter().map(String::as_str));
        crate::tools::with_recovery_note(res.recovery_note.as_deref(), lines.join("\n"))
    };
    let mut text = compose(&text_block);
    if text.len() > crate::util::TOOL_OUTPUT_BUDGET_BYTES && !reading.text.is_empty() {
        text_block = spill_text(dir, &display, &reading.text).await;
        text = compose(&text_block);
    }

    // The answer is content, never a claim about its images.
    ToolOutput {
        text,
        image_payloads: attached,
        text_is_content: true,
        preserve_text: false,
    }
}

/// Write the full extracted text into the call's artifact directory and return the
/// line pointing at it. A write failure is reported in that line, not as a failed call.
async fn spill_text(dir: &Path, display: &str, text: &str) -> String {
    let path = dir.join(crate::tools::path::format_spill_filename());
    match tokio::fs::write(&path, text.as_bytes()).await {
        Ok(()) => answer_line(
            display,
            crate::document::spilled_text_note(text.chars().count(), &path),
        ),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "Failed to save the extracted document text"
            );
            answer_line(display, "the extracted text could not be saved")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::test_fixtures::multi_page_pdf;
    // The `.doc` package recipe lives with the legacy reader.
    use crate::legacy::test_fixtures::doc_fixture;
    use crate::ooxml::test_fixtures::{DOCX_BODY, zip_fixture};
    use crate::util::test::noisy_png;
    use tempfile::TempDir;

    /// A temp workspace holding `files` (name, bytes) — hold the `TempDir` to
    /// keep it alive.
    fn temp_workspace(files: &[(&str, &[u8])]) -> (TempDir, Workspace) {
        let dir = TempDir::new().expect("tempdir");
        for (name, bytes) in files {
            std::fs::write(dir.path().join(name), bytes).expect("write fixture");
        }
        let ws = crate::workspace::test_ws(dir.path());
        (dir, ws)
    }

    /// Owns one test's spills: the conversion registers its artifact directory
    /// under a unique agent id, and dropping this reclaims it. Without it the
    /// directories land in the shared diagnostics bucket — which the pipeline's
    /// own `cleanup_agent_spills` may drain while this test is still reading
    /// what they hold.
    struct SpillOwner(String);

    impl SpillOwner {
        fn new() -> Self {
            Self(format!("read-document-test-{:016x}", rand::random::<u64>()))
        }

        /// Run `future` as this owner, so everything it writes is reclaimed by
        /// the guard's drop.
        async fn scope<T>(&self, future: impl std::future::Future<Output = T>) -> T {
            crate::agent::CURRENT_TOOL_AGENT_ID
                .scope(Some(self.0.clone()), future)
                .await
        }
    }

    impl Drop for SpillOwner {
        fn drop(&mut self) {
            crate::tools::shell::cleanup_agent_spills(&self.0);
        }
    }

    /// Resolve `name` in `ws` the way the read tool's hook does, then convert it.
    async fn convert(
        owner: &SpillOwner,
        ws: &Workspace,
        name: &str,
        access: PathAccess,
    ) -> Option<ToolOutput> {
        let res = super::super::read::resolve_content_read(ws, name, access)
            .await
            .expect("resolve the fixture");
        owner
            .scope(read_document(ws, &res, access))
            .await
            .expect("read the document")
    }

    /// Read one fixture through the command line's entrance into `destination`.
    async fn command_line_read(
        ws: &Workspace,
        destination: &Path,
        path: &str,
        file_name: Option<&str>,
    ) -> anyhow::Result<crate::tools::ToolAnswer> {
        let mut args = serde_json::json!({
            "action": crate::cli::DOCUMENT_READ,
            "path": path,
        });
        if let Some(file_name) = file_name {
            args["file_name"] = serde_json::Value::String(file_name.to_string());
        }
        read_for_command_line(ws, &args, destination).await
    }

    /// The absolute paths an answer's `[FILE:...]` markers name.
    fn marked_files(answer: &str) -> Vec<PathBuf> {
        answer
            .split(crate::tools::document::FILE_MARKER)
            .skip(1)
            .map(|rest| PathBuf::from(rest.split(']').next().unwrap_or_default()))
            .collect()
    }

    /// The file names inside `folder`, sorted.
    fn folder_names(folder: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(folder)
            .expect("the reading's folder")
            .map(|entry| {
                entry
                    .expect("a folder entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    /// The names inside `folder` whose extension is `extension`, sorted.
    fn files_with_extension(folder: &Path, extension: &str) -> Vec<String> {
        folder_names(folder)
            .into_iter()
            .filter(|name| {
                Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
            })
            .collect()
    }

    /// A one-page PDF with a painted rectangle and no text layer: the page has
    /// to be rasterized, so the read comes back as an image and no text.
    fn raster_only_pdf() -> Vec<u8> {
        multi_page_pdf(&[(
            "/MediaBox [0 0 612 792]",
            b"0.1 0.5 0.9 rg 0 0 612 792 re f",
        )])
    }

    /// A one-page PDF whose page has no MediaBox: the reader panics on it, so its
    /// text is lost while the page itself still parses and renders.
    fn pdf_with_no_media_box() -> Vec<u8> {
        // An empty entry list is what leaves the page without a `/MediaBox`.
        multi_page_pdf(&[("", b"BT /F1 24 Tf 72 700 Td (Page text long enough) Tj ET")])
    }

    /// The path a spilled-text line points at.
    fn spilled_path(text: &str) -> PathBuf {
        let start = text.find("saved to ").expect("a spill line") + "saved to ".len();
        let end = start + text[start..].find(" — read that file").expect("a pointer");
        PathBuf::from(&text[start..end])
    }

    /// A Word package with an empty body — no text, so only its media can
    /// produce anything.
    const EMPTY_DOCX_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body></w:body></w:document>"#;

    /// A text-free Word package and what its reading must say: a document with
    /// neither text nor images says the plain sentence (never "could not be
    /// read" for a document that converted fine), and one whose pictures the
    /// limit cuts short never claims its content was provided as images.
    struct TextFreeCase {
        name: &'static str,
        file: &'static str,
        fixture: Vec<u8>,
        /// Substrings the answer must carry.
        present: &'static [&'static str],
        /// Substrings the answer must not carry.
        absent: &'static [&'static str],
        /// How many of the document's pictures the answer attaches.
        attached: usize,
    }

    #[tokio::test]
    async fn text_free_docx_cases() {
        let owner = SpillOwner::new();
        let cases = vec![
            TextFreeCase {
                name: "no_text_and_no_images",
                file: "empty.docx",
                fixture: zip_fixture(&[("word/document.xml", EMPTY_DOCX_BODY)]),
                present: &["empty.docx: no text could be extracted]"],
                absent: &[],
                attached: 0,
            },
            TextFreeCase {
                name: "pictures_at_the_limit_are_all_attached",
                file: "scanned.docx",
                fixture: docx_with_images(EMPTY_DOCX_BODY, 6),
                present: &[
                    "scanned.docx: no text could be extracted — the content was provided as images]",
                ],
                absent: &["left out by the"],
                attached: 6,
            },
            TextFreeCase {
                name: "pictures_over_the_limit_leave_the_no_text_claim_out",
                file: "long_scan.docx",
                fixture: docx_with_images(EMPTY_DOCX_BODY, 7),
                present: &[
                    "long_scan.docx: no text could be extracted]\n",
                    "7 pictures in the document; 1 left out by the 6-picture limit]",
                ],
                absent: &[crate::document::NO_TEXT_LAYER_NOTE],
                attached: 6,
            },
        ];
        for case in cases {
            let (_dir, ws) = temp_workspace(&[(case.file, &case.fixture)]);
            let out = convert(&owner, &ws, case.file, PathAccess::Allowlisted)
                .await
                .unwrap_or_else(|| panic!("{}: a .docx is a document", case.name));
            assert_eq!(
                out.image_payloads.len(),
                case.attached,
                "{}: the answer attaches {} picture(s): {}",
                case.name,
                case.attached,
                out.text
            );
            for expected in case.present {
                assert!(
                    out.text.contains(expected),
                    "{}: the answer does not say {expected:?}: {}",
                    case.name,
                    out.text
                );
            }
            for unexpected in case.absent {
                assert!(
                    !out.text.contains(unexpected),
                    "{}: the answer claims {unexpected:?}: {}",
                    case.name,
                    out.text
                );
            }
        }
    }

    /// What a reading found in the file and did not show is named in the answer —
    /// the kinds the tool saw but never put in the text — and a reading that left
    /// nothing out says nothing about it.
    #[tokio::test]
    async fn the_answer_names_what_the_reading_did_not_show() {
        let owner = SpillOwner::new();
        let with_media = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            ("word/media/clip.mp4", b"not a picture at all"),
            (
                "word/media/scan.tiff",
                b"a picture this pipeline does not convert",
            ),
            ("word/embeddings/oleObject1.bin", b"an object nothing reads"),
        ]);
        let (_dir, ws) = temp_workspace(&[("clip.docx", &with_media)]);

        let out = convert(&owner, &ws, "clip.docx", PathAccess::Allowlisted)
            .await
            .expect("a .docx is a document");
        // The report leads the notes, one line per kind: the clip is a video,
        // never "an image in a format this pipeline cannot convert".
        assert!(
            out.text.contains("1 embedded video(s) not shown]"),
            "{:?}",
            out.text
        );
        assert!(
            out.text.contains("1 embedded object part(s) not shown]"),
            "{:?}",
            out.text
        );
        assert!(
            out.text
                .contains("skipped 1 embedded image(s) in a format this pipeline cannot convert]"),
            "the picture the pipeline cannot convert keeps its reason: {:?}",
            out.text
        );

        let plain = zip_fixture(&[("word/document.xml", DOCX_BODY)]);
        let (_dir, ws) = temp_workspace(&[("plain.docx", &plain)]);
        let out = convert(&owner, &ws, "plain.docx", PathAccess::Allowlisted)
            .await
            .expect("a .docx is a document");
        assert!(
            !out.text.contains("not shown]"),
            "a reading with nothing left out reports nothing: {:?}",
            out.text
        );
    }

    /// A document whose bytes cannot be read is a plain answer, not a failed
    /// call — and never undecodable bytes reported as a success. Read through a
    /// typo'd path that the fuzzy matcher recovers, so the answer also has to
    /// carry the `[Recovered path: ...]` note every other recovered read has.
    #[tokio::test]
    async fn corrupt_pdf_is_reported_not_failed() {
        crate::util::test::init_test_stores().await;
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("broken.pdf", b"%PDF-1.4\ngarbage")]);

        let out = convert(&owner, &ws, "brokn.pdf", PathAccess::Allowlisted)
            .await
            .expect("a .pdf is a document");
        assert!(out.text.starts_with("[Recovered path: "), "{}", out.text);
        assert!(out.text.contains("could not be parsed"), "{}", out.text);
        assert!(out.image_payloads.is_empty());
    }

    /// A `.docx` whose document body is `body` plus `count` embedded media
    /// entries — the shape the converter treats as delivered images.
    fn docx_with_images(body: &[u8], count: usize) -> Vec<u8> {
        let png = noisy_png(16, 16);
        let names: Vec<String> = (1..=count)
            .map(|page| format!("word/media/page_{page}.png"))
            .collect();
        let mut entries: Vec<(&str, &[u8])> = vec![("word/document.xml", body)];
        entries.extend(names.iter().map(|name| (name.as_str(), png.as_slice())));
        zip_fixture(&entries)
    }

    /// The read tool's hook routes a document through this module and returns
    /// the converted text — the conversion happens once, inside the execution.
    #[tokio::test]
    async fn read_tool_delivers_the_converted_document() {
        use crate::Tool;
        let owner = SpillOwner::new();
        let fixture = zip_fixture(&[("word/document.xml", DOCX_BODY)]);
        let (_dir, ws) = temp_workspace(&[("report.docx", &fixture)]);

        let out = owner
            .scope(
                crate::tools::read::ReadTool::general()
                    .execute_with_payloads(&ws, serde_json::json!({"path": "report.docx"})),
            )
            .await
            .expect("read the document");
        assert!(out.text.contains("extracted text follows"), "{}", out.text);
        assert!(
            out.text.contains("First paragraph\nSecond paragraph"),
            "{}",
            out.text
        );
        assert!(
            !out.text.lines().any(|line| line.starts_with("1: ")),
            "a document is converted, not line-numbered: {}",
            out.text
        );
        assert!(
            !out.text.contains("too long to inline"),
            "short text is inlined, never spilled: {}",
            out.text
        );
        assert!(
            out.image_payloads.is_empty(),
            "the fixture has no media entries"
        );
    }

    /// The restricted (workspace-only) variant cannot open temp paths, so its
    /// artifacts have to land in the caller's own workspace.
    #[tokio::test]
    async fn strict_read_keeps_artifacts_inside_the_workspace() {
        let (dir, ws) = temp_workspace(&[("scan.pdf", &raster_only_pdf())]);

        let out = convert(&SpillOwner::new(), &ws, "scan.pdf", PathAccess::Workspace)
            .await
            .expect("a .pdf is a document");
        let image = out.image_payloads.first().expect("the page is attached");
        let uploads = std::fs::canonicalize(dir.path())
            .expect("canonical workspace")
            .join("uploads");
        assert!(
            crate::util::is_within(Path::new(&image.path), &uploads),
            "a restricted read must place artifacts where it can open them: {}",
            image.path
        );
    }

    /// A long text next to a full set of pictures: the answer still has to fit
    /// the budget it is formatted against, so the text is spilled (whole) rather
    /// than truncated by the formatter with nothing to recover it.
    #[tokio::test]
    async fn long_text_with_many_images_is_spilled_not_truncated() {
        let filler = "lorem ipsum dolor sit amet ".repeat(1_200);
        let body = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>FIRST-MARKER</w:t></w:r></w:p><w:p><w:r><w:t>{filler}</w:t></w:r></w:p><w:p><w:r><w:t>LAST-MARKER</w:t></w:r></w:p></w:body></w:document>"#
        );
        let fixture = docx_with_images(body.as_bytes(), 6);
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("long.docx", &fixture)]);

        let out = convert(&owner, &ws, "long.docx", PathAccess::Allowlisted)
            .await
            .expect("a .docx is a document");
        assert!(
            out.text.len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the answer must fit the budget it is formatted against: {} bytes",
            out.text.len()
        );
        assert!(
            !out.text.contains("FIRST-MARKER"),
            "long extracted text is spilled, not inlined: {}",
            out.text
        );
        let spilled =
            std::fs::read_to_string(spilled_path(&out.text)).expect("read the spill file");
        assert!(
            spilled.contains("FIRST-MARKER") && spilled.contains("LAST-MARKER"),
            "the spill file holds the whole text, head to tail"
        );
        assert_eq!(
            out.image_payloads.len(),
            6,
            "spilling the text never costs the document's pictures: {}",
            out.text
        );
    }

    /// The command line's entrance is the read tool's own reading, delivered to
    /// disk: the whole extracted text and every image the converter produced,
    /// each named by an absolute `[FILE:]` path inside one new folder.
    #[tokio::test]
    async fn command_line_reading_writes_the_readers_text_and_every_image() {
        crate::util::test::init_test_stores().await;
        let fixture = docx_with_images(DOCX_BODY, 3);
        let (_dir, ws) = temp_workspace(&[("report.docx", &fixture)]);
        let destination = TempDir::new().expect("destination");

        let answer = command_line_read(&ws, destination.path(), "report.docx", None)
            .await
            .expect("a document is read");
        assert!(
            !answer.partial,
            "a whole reading is not partial: {}",
            answer.text
        );

        let folder = destination.path().join("report");
        assert!(folder.is_dir(), "the reading writes into its own folder");
        let text = std::fs::read_to_string(folder.join("report.txt"))
            .expect("the reader's text is written");
        assert!(
            text.contains("First paragraph") && text.contains("Second paragraph"),
            "{text}"
        );
        let images = files_with_extension(&folder, "png");
        assert_eq!(images.len(), 3, "every media entry is written");

        let marked = marked_files(&answer.text);
        assert_eq!(
            marked.len(),
            4,
            "the text and three images are named: {}",
            answer.text
        );
        assert!(
            marked
                .iter()
                .all(|path| path.is_absolute() && path.exists() && path.starts_with(&folder)),
            "{marked:?}"
        );

        // A non-document is the ordinary read: its line-numbered rendering is
        // written as the text, never a refusal. An empty file has no rendering,
        // so nothing is written and the answer is the empty one the read tool
        // gives it.
        let (_text_dir, text_ws) =
            temp_workspace(&[("notes.md", b"first\nsecond\n"), ("empty.md", b"")]);
        let text_destination = TempDir::new().expect("destination");
        let answer = command_line_read(&text_ws, text_destination.path(), "notes.md", None)
            .await
            .expect("a text file is read");
        assert!(!answer.partial);
        let rendered = std::fs::read_to_string(text_destination.path().join("notes/notes.txt"))
            .expect("the ordinary read's rendering");
        assert!(
            rendered.contains("1: first") && rendered.contains("2: second"),
            "the fallback is the read tool's line-numbered rendering: {rendered}"
        );

        let empty = command_line_read(&text_ws, text_destination.path(), "empty.md", None)
            .await
            .expect("an empty file is read");
        assert_eq!(empty.text, "", "the read tool's answer for it is empty");
        assert!(
            !text_destination.path().join("empty").exists(),
            "nothing produced, so no folder"
        );
    }

    /// A reading the reader lost every page of is a PARTIAL operation: the pages
    /// it could not read are named in the reader's own words, the page it still
    /// rendered is written, and the family's partial class applies.
    #[tokio::test]
    async fn command_line_reading_that_lost_its_text_is_partial() {
        crate::util::test::init_test_stores().await;
        let (_dir, ws) = temp_workspace(&[("broken.pdf", &pdf_with_no_media_box())]);
        let destination = TempDir::new().expect("destination");

        let answer = command_line_read(&ws, destination.path(), "broken.pdf", None)
            .await
            .expect("the page that still renders is delivered");
        assert!(answer.partial, "{}", answer.text);
        assert!(
            answer
                .text
                .contains("broken.pdf: the text of page 1 could not be read (1 of 1 pages)"),
            "the reader's own wording: {}",
            answer.text
        );
        assert!(
            !answer.text.contains(crate::document::NO_TEXT_NOTE),
            "a reader failure is not a document without text: {}",
            answer.text
        );
        assert_eq!(
            files_with_extension(&destination.path().join("broken"), "jpg"),
            ["page_1.jpg"],
            "the page it could render is still written"
        );
        assert!(
            files_with_extension(&destination.path().join("broken"), "txt").is_empty(),
            "no text was extracted, so no text file"
        );
    }

    /// A document the reader cannot deliver, and the refusal the command line
    /// must give: the reader's own words, and no folder left behind.
    struct CommandLineRefusalCase {
        name: &'static str,
        file: &'static str,
        fixture: Vec<u8>,
        /// Substrings the refusal must carry.
        message: &'static [&'static str],
        /// The folder the reading would have written into.
        folder: &'static str,
    }

    fn refusal_cases() -> Vec<CommandLineRefusalCase> {
        vec![
            CommandLineRefusalCase {
                name: "an_unreadable_document",
                file: "broken.pdf",
                fixture: b"%PDF-1.4\ngarbage".to_vec(),
                message: &["could not be parsed", "broken.pdf"],
                folder: "broken",
            },
            CommandLineRefusalCase {
                name: "a_protected_document",
                file: "secret.doc",
                fixture: doc_fixture("secret", &[], true),
                message: &["password-protected"],
                folder: "secret",
            },
        ]
    }

    #[tokio::test]
    async fn command_line_refusal_cases() {
        crate::util::test::init_test_stores().await;
        for case in refusal_cases() {
            let (_dir, ws) = temp_workspace(&[(case.file, &case.fixture)]);
            let destination = TempDir::new().expect("destination");

            let error = command_line_read(&ws, destination.path(), case.file, None)
                .await
                .err()
                .unwrap_or_else(|| panic!("{}: the reading must be refused", case.name));
            let message = format!("{error:#}");
            for expected in case.message {
                assert!(
                    message.contains(expected),
                    "{}: the refusal does not say {expected:?}: {message}",
                    case.name
                );
            }
            assert!(
                !destination.path().join(case.folder).exists(),
                "{}: a failed reading writes nothing",
                case.name
            );
        }
    }

    /// The command line has no length limit: a long text and a hundred images
    /// are all written, and only the ANSWER is collapsed — to one folder line.
    #[tokio::test]
    async fn command_line_reading_is_uncapped() {
        crate::util::test::init_test_stores().await;
        let filler = "lorem ipsum dolor sit amet ".repeat(400);
        let body = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{filler}</w:t></w:r></w:p></w:body></w:document>"#
        );
        let fixture = docx_with_images(body.as_bytes(), 120);
        let (_dir, ws) = temp_workspace(&[("many.docx", &fixture)]);
        let destination = TempDir::new().expect("destination");

        let answer = command_line_read(&ws, destination.path(), "many.docx", None)
            .await
            .expect("a big document is read");
        assert!(!answer.partial);

        let folder = destination.path().join("many");
        let images = files_with_extension(&folder, "png").len();
        assert_eq!(images, 120, "every image is written, with no cap");
        let text = std::fs::read(folder.join("many.txt")).expect("the text is written");
        assert!(
            text.len() > crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the whole text is written: {} bytes",
            text.len()
        );
        assert!(
            !answer.text.contains("saved to "),
            "the command line never spills: {}",
            answer.text
        );
        assert!(
            !answer.text.contains("too long to inline"),
            "{}",
            answer.text
        );
        assert!(
            answer.text.contains("they are all in"),
            "the listing collapses to the folder: {}",
            answer.text
        );
        assert!(
            answer.text.len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the collapsed answer fits the shared budget: {} bytes",
            answer.text.len()
        );
    }

    /// A reading (or a sequence of them) into one destination and the folders
    /// each must produce: `--file-name` replaces the base name through the
    /// family's own sanitisation, and a repeat gets its numeric suffix while the
    /// first folder is left untouched.
    struct NamingCase {
        name: &'static str,
        file: &'static str,
        /// The `--file-name` flag of each reading, in order.
        reads: &'static [Option<&'static str>],
        /// The folder each reading must write into, and the text file inside it.
        folders: &'static [(&'static str, &'static str)],
    }

    const NAMING_CASES: &[NamingCase] = &[
        NamingCase {
            name: "the_flag_names_the_folder_and_its_text_file",
            file: "report.docx",
            reads: &[Some("dir/summary")],
            // A name carrying a directory is reduced to its last component, like
            // every other name the family writes.
            folders: &[("summary", "summary.txt")],
        },
        NamingCase {
            name: "a_second_reading_gets_the_numeric_suffix",
            file: "2024.report.docx",
            reads: &[None, None],
            // A dotted base keeps its dots: a folder has no extension, so the
            // suffix goes at the end.
            folders: &[
                ("2024.report", "2024.report.txt"),
                ("2024.report_2", "2024.report_2.txt"),
            ],
        },
    ];

    #[tokio::test]
    async fn command_line_naming_cases() {
        crate::util::test::init_test_stores().await;
        for case in NAMING_CASES {
            let fixture = zip_fixture(&[("word/document.xml", DOCX_BODY)]);
            let (_dir, ws) = temp_workspace(&[(case.file, &fixture)]);
            let destination = TempDir::new().expect("destination");

            let mut texts: Vec<String> = Vec::new();
            for (read, (folder, text_file)) in case.reads.iter().zip(case.folders) {
                let answer = command_line_read(&ws, destination.path(), case.file, *read)
                    .await
                    .unwrap_or_else(|e| panic!("{}: a document is read: {e}", case.name));
                assert!(!answer.partial, "{}: {}", case.name, answer.text);
                let folder = destination.path().join(folder);
                assert!(
                    folder.is_dir(),
                    "{}: the reading's folder: {:?}",
                    case.name,
                    folder_names(destination.path())
                );
                texts.push(
                    std::fs::read_to_string(folder.join(text_file))
                        .unwrap_or_else(|e| panic!("{}: {text_file}: {e}", case.name)),
                );
                assert_eq!(
                    folder_names(&folder),
                    [text_file.to_string()],
                    "{}: the folder holds its own text file",
                    case.name
                );
            }
            // Every reading delivered the same text, and the first one is still
            // there to say so.
            for text in &texts {
                assert_eq!(
                    text, &texts[0],
                    "{}: the same reading, delivered again",
                    case.name
                );
            }
        }
    }

    /// The product's live store is refused by the read tool's own resolution,
    /// before anything is opened — and no folder is made for it.
    #[tokio::test]
    async fn command_line_reading_refuses_the_products_live_store_before_reading() {
        crate::util::test::init_test_stores().await;
        let (_dir, ws) = temp_workspace(&[]);
        let destination = TempDir::new().expect("destination");
        let store = crate::db::store_db_path(
            crate::util::test::test_root(),
            crate::db::CONSOLIDATED_DB_NAME,
        );

        let error = command_line_read(&ws, destination.path(), &store.to_string_lossy(), None)
            .await
            .err()
            .expect("the live store is refused");
        let message = format!("{error:#}");
        assert!(message.contains("live databases"), "{message}");
        assert_eq!(
            std::fs::read_dir(destination.path())
                .expect("the destination")
                .count(),
            0,
            "nothing was created"
        );
    }
}

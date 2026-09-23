//! Document reads for the read tool: what the shared converter
//! ([`crate::document`]) extracts from is converted here and delivered the read
//! tool's way — extracted text inlined (or spilled when it does not fit the tool
//! output budget), extracted rasters attached as images.
//!
//! Artifacts (a page raster, an embedded image, a spilled text body) are
//! per-call intermediates in a uniquely named `read_<nonce>` directory, removed
//! when the run ends like a shell spill. The general read writes them into the
//! daemon temp service area ([`crate::tools::shell::agent_temp_dir`]); the
//! restricted (workspace-only) read, which cannot open temp paths, into its
//! caller's workspace `uploads/` (a re-read re-converts). What a crash leaves
//! behind is reclaimed there too — by the OS temp sweep and the periodic temp
//! cleaner in the first case, by the workspace media sweep in the second, which
//! removes the leftover files but not the empty directory. That `uploads/` parent
//! stays in place (the inbound attachment copy writes it too, so removing it
//! could race that writer), and nothing is written beside the document.

use crate::Workspace;
use crate::document::DocOutcome;
use crate::tools::{ImagePayload, ImagePayloadSource, ToolOutput};
use anyhow::Context;
use std::path::{Path, PathBuf};

/// Maximum converted images attached as native images in one call: a many-page
/// scan must not flood the conversation or a provider's per-request image cap.
/// Over it the paths are the delivery — or the folder holding them, when the
/// listing would not fit the answer.
const MAX_INJECTED_IMAGES: usize = 5;

/// Bytes read from the head of the file for the format sniff: every extraction
/// arm of [`crate::document::needs_extraction`] is magic-local.
const HEAD_SNIFF_BYTES: usize = 16;

/// Answer for a document whose conversion could not start (no temp area, or an
/// unwritable workspace) — the file itself is fine.
const NO_ARTIFACT_DIR_REASON: &str = "could not be converted";

/// Convert `res` when it is a container the shared converter extracts from — the
/// same detection the inbound path uses, so the two cannot disagree about what a
/// document is. `Ok(None)` when it is not (the ordinary read applies), `Err` only
/// for an oversized container.
pub(super) async fn read_document(
    ws: &Workspace,
    res: &super::read::ResolvedRead,
    strict: bool,
) -> anyhow::Result<Option<ToolOutput>> {
    // Only a regular file is converted: a directory keeps the ordinary listing
    // read, and a special file is never opened (the converter's read is unbounded).
    let Ok(meta) = tokio::fs::metadata(&res.path).await else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    let Some(name) = res.path.file_name().and_then(|n| n.to_str()) else {
        return Ok(None);
    };
    let Some(head) = peek_head(&res.path).await else {
        return Ok(None);
    };
    // Classified before the size check, so an oversized NON-document falls back
    // to the ordinary 10 MB message.
    if !crate::document::needs_extraction(&head, name) {
        return Ok(None);
    }
    // The inbound product cap, not the read tool's own 10 MB text cap (both
    // are stated in the read prompts).
    crate::tools::check_size_within(
        &meta,
        crate::util::FILE_MAX_BYTES,
        "File too large to convert",
    )?;

    // No artifact directory, no conversion — a plain answer, not a failed call.
    let dir = match create_artifact_dir(ws, strict).await {
        Ok(dir) => dir,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to create a document artifact directory");
            return Ok(Some(plain_answer(res, NO_ARTIFACT_DIR_REASON)));
        }
    };
    crate::tools::shell::record_spill_owner(dir.clone());

    let outcome = match crate::document::convert_document_file(&res.path, name, &dir).await {
        // The file changed under us between the sniff and the conversion: the
        // ordinary read decides what it now is.
        DocOutcome::Unsupported => None,
        DocOutcome::Unreadable { reason } => Some(plain_answer(res, &reason)),
        DocOutcome::Text {
            text,
            images,
            notes,
            all_page_text_lost,
        } => Some(compose_output(res, &dir, text, images, notes, all_page_text_lost).await),
    };
    // Empty-only removal (`remove_dir` refuses a non-empty directory): an answer
    // pointing at nothing leaves no directory behind, anything it points at keeps
    // one until the run-end hook reclaims it.
    let _ = tokio::fs::remove_dir(&dir).await;
    Ok(outcome)
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
    }
}

/// One line of a converted-document answer: `[<path>: <what it says>]` — the
/// envelope every note here shares.
fn answer_line(display: &str, body: impl std::fmt::Display) -> String {
    format!("[{display}: {body}]")
}

/// A fresh uniquely named directory for this call's artifacts — its location is
/// the module header's subject.
async fn create_artifact_dir(ws: &Workspace, strict: bool) -> anyhow::Result<PathBuf> {
    let parent = if strict {
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

/// Turn a converted document's parts into the read tool's result: scrub the text,
/// attach the rasters when there are few enough, and say what was delivered.
async fn compose_output(
    res: &super::read::ResolvedRead,
    dir: &Path,
    text: String,
    images: Vec<PathBuf>,
    notes: Vec<String>,
    all_page_text_lost: bool,
) -> ToolOutput {
    let display = res.path.display().to_string();
    // The read tool returns `false` from `should_scrub_output`, and the scrub it
    // does instead is path-based inside `read_resolved` — it sees the container's
    // path, never what was extracted from it — so this text is scrubbed here,
    // unconditionally: the inbound route delivers the same extracted text
    // unscrubbed.
    let text = crate::util::scrub_credentials(text.trim());

    let over_cap = images.len() > MAX_INJECTED_IMAGES;
    let mut attached = Vec::new();
    let mut unattached = 0usize;
    if !over_cap {
        // The payloads carry no recovery note: a document answer is content, so
        // its text block already opens with that note and repeating it on every
        // page annotation would say the same thing 1+N times.
        for image in &images {
            // An image that cannot be encoded is counted in a note rather than
            // failing the call: the document itself was read.
            match crate::util::local_image_to_compressed_data_uri_with_meta(image).await {
                // The page is this tool's own artifact — it never opened it as a
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
    }

    let note = res.recovery_note.as_deref();

    let mut text_block = if text.is_empty() {
        // The reader failed on every page it was asked for: the notes below name
        // those pages, and the "no text" sentences would report a reader failure
        // as a document that has no text.
        if all_page_text_lost {
            String::new()
        } else if attached.is_empty() {
            // Name the pages as provided only when one actually was attached;
            // with none — no pages produced, all over the cap, or none
            // encodable — the plain sentence is the honest one.
            answer_line(&display, crate::document::NO_TEXT_NOTE)
        } else {
            answer_line(&display, crate::document::NO_TEXT_LAYER_NOTE)
        }
    } else {
        format!(
            "{}\n\n{text}",
            answer_line(&display, "extracted text follows")
        )
    };
    let mut trailing: Vec<String> = notes.iter().map(|n| answer_line(&display, n)).collect();
    if unattached > 0 {
        // Not the inbound sentence ("could not be read"): there the image
        // pipeline rejected an image it was handed, here the raster could not be
        // encoded. The folder is named so the page stays reachable.
        trailing.push(answer_line(
            &display,
            format!(
                "{unattached} extracted image(s) could not be attached; they are in {} — read them individually if you need them",
                dir.display()
            ),
        ));
    }
    // Paths are preferred and stay unenclosed so they can be copied straight into
    // another call; the folder line is the fallback when they would not fit.
    let (full_listing, folder_listing) = if over_cap {
        let head = format!(
            "[{display}: {} extracted image(s) were not attached ({MAX_INJECTED_IMAGES} is the per-call limit);",
            images.len()
        );
        let mut listing = vec![format!("{head} read them individually as images:]")];
        listing.extend(images.iter().map(|image| image.display().to_string()));
        let folder = vec![format!(
            "{head} they are all in {} — list that folder, then read the images you need.]",
            dir.display()
        )];
        (listing, folder)
    } else {
        (Vec::new(), Vec::new())
    };

    // The whole answer — text block and recovery note included — must fit the
    // shared tool output budget, or `format_output` truncates it mid-answer. (The
    // per-image annotations the agent loop appends afterwards are outside this
    // budget; they are bounded by the per-call image cap.) Candidates, in order:
    // the paths (never a partial listing — the formatter would drop paths from
    // its middle), the folder line, then both again with the text spilled.
    let compose = |text_block: &str, listing: &[String]| {
        let mut lines: Vec<&str> = Vec::with_capacity(2 + trailing.len() + listing.len());
        // Empty only when every page's text failed: the notes are then the whole
        // answer's opening.
        if !text_block.is_empty() {
            lines.push(text_block);
        }
        lines.extend(trailing.iter().map(String::as_str));
        lines.extend(listing.iter().map(String::as_str));
        crate::tools::with_recovery_note(note, lines.join("\n"))
    };
    let fits = |text_block: &str, listing: &[String]| {
        compose(text_block, listing).len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES
    };
    let pick_listing = |text_block: &str| {
        if fits(text_block, &full_listing) {
            &full_listing
        } else {
            &folder_listing
        }
    };
    let mut listing = pick_listing(&text_block);
    if !fits(&text_block, listing) && !text.is_empty() {
        text_block = spill_text(dir, &display, &text).await;
        listing = pick_listing(&text_block);
    }

    // The answer is content, never a claim about its images.
    ToolOutput {
        text: compose(&text_block, listing),
        image_payloads: attached,
        text_is_content: true,
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
    use crate::document::test_fixtures::{DOCX_BODY, multi_page_pdf, zip_fixture};
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
        strict: bool,
    ) -> Option<ToolOutput> {
        let res = super::super::read::resolve_content_read(ws, name, strict)
            .await
            .expect("resolve the fixture");
        owner
            .scope(read_document(ws, &res, strict))
            .await
            .expect("read the document")
    }

    /// Real PNG bytes of a `size`×`size` image: an embedded media part the
    /// converter writes through verbatim and the payload encoder can decode.
    fn png_fixture(size: u32) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(size, size, image::Rgb([10, 200, 10]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("encode fixture png");
        bytes.into_inner()
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

    /// The pages are named as attached only when they actually are: a document
    /// with neither text nor images says the inbound sentence plainly (never
    /// "could not be read" for a document that converted fine), and one whose
    /// images are all over the cap points at the folder instead of claiming they
    /// were attached.
    #[tokio::test]
    async fn text_free_documents_name_their_pages_only_when_attached() {
        let empty = zip_fixture(&[("word/document.xml", EMPTY_DOCX_BODY)]);
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("empty.docx", &empty)]);
        let out = convert(&owner, &ws, "empty.docx", false)
            .await
            .expect("a .docx is a document");
        assert_eq!(out.image_payloads.len(), 0);
        assert!(
            out.text.contains("empty.docx: no text could be extracted]"),
            "{:?}",
            out.text
        );

        let with_media = docx_with_images(EMPTY_DOCX_BODY, 6);
        let (_dir, ws) = temp_workspace(&[("scanned.docx", &with_media)]);
        let out = convert(&owner, &ws, "scanned.docx", false)
            .await
            .expect("a .docx is a document");
        assert_eq!(
            out.image_payloads.len(),
            0,
            "over the cap nothing is attached"
        );
        assert!(
            out.text
                .contains("scanned.docx: no text could be extracted]\n"),
            "{:?}",
            out.text
        );
        assert!(
            !out.text.contains(crate::document::NO_TEXT_LAYER_NOTE),
            "pages that were not attached are not called attached: {:?}",
            out.text
        );
    }

    #[tokio::test]
    async fn image_only_pdf_pages_come_back_as_images() {
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("scan.pdf", &raster_only_pdf())]);

        let out = convert(&owner, &ws, "scan.pdf", false)
            .await
            .expect("a .pdf is a document");
        assert_eq!(
            out.image_payloads.len(),
            1,
            "one rasterized page: {}",
            out.text
        );
        assert!(
            out.text
                .contains("no text could be extracted — the pages were provided as images"),
            "{}",
            out.text
        );
        assert!(
            out.text_is_content,
            "the page images supplement, never replace, the answer"
        );
    }

    /// A document whose text could not be read is not one that has no text: the
    /// answer names the pages, and the "no text could be extracted" sentence —
    /// which would report the reader's failure as an absence — is left out.
    #[tokio::test]
    async fn unread_text_is_named_not_reported_as_absent() {
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("broken.pdf", &pdf_with_no_media_box())]);

        let out = convert(&owner, &ws, "broken.pdf", false)
            .await
            .expect("a .pdf is a document");
        assert!(
            out.text
                .contains("broken.pdf: the text of page 1 could not be read (1 of 1 pages)"),
            "{}",
            out.text
        );
        assert!(
            !out.text.contains(crate::document::NO_TEXT_NOTE),
            "a reader failure is not a document without text: {}",
            out.text
        );
        assert_eq!(
            out.image_payloads.len(),
            1,
            "the page that could not be read comes back as an image: {}",
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

        let out = convert(&owner, &ws, "brokn.pdf", false)
            .await
            .expect("a .pdf is a document");
        assert!(out.text.starts_with("[Recovered path: "), "{}", out.text);
        assert!(out.text.contains("could not be parsed"), "{}", out.text);
        assert!(out.image_payloads.is_empty());
    }

    /// A `.docx` whose document body is `body` plus `count` embedded media
    /// entries — the shape the converter treats as delivered images.
    fn docx_with_images(body: &[u8], count: usize) -> Vec<u8> {
        let png = png_fixture(16);
        let names: Vec<String> = (1..=count)
            .map(|page| format!("word/media/page_{page}.png"))
            .collect();
        let mut entries: Vec<(&str, &[u8])> = vec![("word/document.xml", body)];
        entries.extend(names.iter().map(|name| (name.as_str(), png.as_slice())));
        zip_fixture(&entries)
    }

    /// Over the per-call cap nothing is attached, but every produced image is
    /// still delivered — as its path in the call's own directory.
    #[tokio::test]
    async fn images_over_the_cap_are_delivered_as_paths() {
        let fixture = docx_with_images(DOCX_BODY, 6);
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("report.docx", &fixture)]);

        let out = convert(&owner, &ws, "report.docx", false)
            .await
            .expect("a .docx is a document");
        assert!(
            out.image_payloads.is_empty(),
            "nothing may be attached: {}",
            out.text
        );
        assert!(
            out.text
                .contains("6 extracted image(s) were not attached (5 is the per-call limit); read them individually as images:]"),
            "the annotation is one closed envelope: {}",
            out.text
        );
        let listed: Vec<&str> = out
            .text
            .lines()
            .skip_while(|line| !line.contains("were not attached"))
            .skip(1)
            .collect();
        assert_eq!(listed.len(), 6, "one line per produced image: {listed:?}");
        assert!(
            listed
                .iter()
                .all(|path| std::path::Path::new(path).is_absolute()),
            "each image is named by its absolute path: {listed:?}"
        );
    }

    /// A listing too long to fit the budget is replaced by the folder holding
    /// the images: the formatter must never drop paths from the middle of it.
    #[tokio::test]
    async fn a_listing_too_long_to_fit_names_the_folder() {
        // ~60 bytes per path, so this listing cannot fit the 5 KB budget.
        let fixture = docx_with_images(DOCX_BODY, 120);
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("many.docx", &fixture)]);

        let out = convert(&owner, &ws, "many.docx", false)
            .await
            .expect("a .docx is a document");
        assert!(
            out.text.len() <= crate::util::TOOL_OUTPUT_BUDGET_BYTES,
            "the answer must fit the budget it is formatted against: {} bytes",
            out.text.len()
        );
        assert!(
            out.text.contains("they are all in "),
            "the folder is named rather than a truncated listing: {}",
            out.text
        );
    }

    /// A plain text file is not a document: the read tool's ordinary path owns
    /// it, and the two sides must never disagree about what a document is — the
    /// hook's fall-through still returns the line-numbered read, unchanged.
    #[tokio::test]
    async fn plain_text_is_not_a_document() {
        use crate::Tool;
        let owner = SpillOwner::new();
        let (_dir, ws) = temp_workspace(&[("notes.md", b"# Notes\n")]);

        assert!(convert(&owner, &ws, "notes.md", false).await.is_none());

        let out = owner
            .scope(
                crate::tools::read::ReadTool::general()
                    .execute_with_payloads(&ws, serde_json::json!({"path": "notes.md"})),
            )
            .await
            .expect("read the text file");
        assert!(out.text.contains("1: # Notes"), "{}", out.text);
        assert!(out.image_payloads.is_empty());
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
            !out.text.contains("1: "),
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

        let out = convert(&SpillOwner::new(), &ws, "scan.pdf", true)
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

    /// A long text next to more images than the per-call cap: the answer still
    /// has to fit the budget it is formatted against, so the text is spilled
    /// (whole) rather than truncated by the formatter with nothing to recover it.
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

        let out = convert(&owner, &ws, "long.docx", false)
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
            out.text
                .lines()
                .filter(|line| std::path::Path::new(line).extension() == Some("png".as_ref()))
                .count(),
            6,
            "spilling the text frees room for the per-image paths again: {}",
            out.text
        );
    }
}

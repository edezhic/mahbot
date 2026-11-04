//! Channel message enrichment: media markers, link enrichment, file operations and media
//! handling — what transforms a [`ChannelMessage`]'s content before it reaches the agent pipeline.
//!
//! Media markers (`[IMAGE: ...]`, `[AUDIO: ...]`, `[VIDEO: ...]`, `[FILE: ...]`): an inbound
//! local image becomes a native data-URI part carrying the original bytes, re-encoded to a
//! bounded JPEG only when it would exceed the encoded-payload cap; audio is transcribed to
//! text; video is a workspace copy plus its timeline; an inbound document is copied
//! into the workspace and converted to text and its first
//! [`crate::document::MAX_READ_PICTURES`] pictures. Link enrichment prepends webpage
//! summaries for URLs in the message.
//!
//! **Containment invariant**: every local read, copy and delete is scoped to the message's
//! own staging directories ([`ChannelMessage::attachment_dirs`], built by
//! `util::telegram_staging_dir_name`) — one for a single message, several for a merged
//! album. A marker is in scope only when it names a file inside one of those; everything
//! else (another chat's or message's attachment, a hand-typed marker, a message owning no
//! directory) is inert text: IMAGE/AUDIO/VIDEO degrade to a plain annotation and
//! `[FILE:...]` stays verbatim, so nothing is read, transcribed, copied or deleted. The
//! desktop composer stages attachments the same way (`gui::attachment`), so a GUI message is
//! enriched exactly like a Telegram one.
//!
//! Entry points are [`enrich_message`] and [`enrich_links`]; [`has_inbound_temp_marker`] is
//! called by `channels::persist_content` to decide what is persisted to chat history.
//! [`enrich_message`] takes the workspace whose `uploads/` receives saved full-resolution
//! copies (`None` disables copies) — the path that makes a clip's timeline possible.

use crate::ChannelMessage;
use crate::document::{DocOutcome, INLINE_TEXT_MAX_CHARS, convert_document_file};
use crate::tools::chrome::ChromeTool;
use crate::util::media_target::{self, MediaTarget};
use crate::util::{
    MEDIA_MARKER_RE, MediaMarkerKind, file_name_or_path, is_http_url, parse_media_marker,
};
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Matches http:// and https:// URLs, stopping at whitespace, angle brackets or quotes.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"']+"#).expect("URL regex must compile"));

/// Process-global sequence for link-enricher session names: the `link-enricher-` prefix is
/// load-bearing (`is_mahbot_session_name` sweeps only sessions starting with it) and unique
/// suffixes keep concurrent [`enrich_links`] runs from sharing a chrome session.
static LINK_ENRICHER_SEQ: AtomicU64 = AtomicU64::new(0);

/// The audio-transcription icon combo, used as the failure fallback and as the annotation for
/// out-of-scope `[AUDIO:...]` markers that must never be read or deleted.
const AUDIO_ICON: &str = "🔊✍️";

/// A transcribed message as presented: the icon combo, one space, then the text. The one
/// definition of that form — an inbound audio attachment and a voice recording both go
/// through it, so a spoken message reads the same however it arrived.
#[must_use]
pub(crate) fn marked_transcript(text: &str) -> String {
    format!("{AUDIO_ICON} {text}")
}

/// The content to embed for an `[AUDIO:...]` marker: the icon combo followed by the transcription.
///
/// The audio file is a pure intermediate artifact — the caller deletes it regardless of
/// outcome — so the returned string never contains its name or path. On failure just the
/// icon combo is returned.
async fn transcribe_audio_marker(path: &str) -> String {
    let path_buf = std::path::PathBuf::from(path);

    if !crate::audio::voice::is_transcription_disabled() {
        // The 10-minute engine bound: attached audio can be arbitrarily long (voice memos,
        // meeting recordings, etc.).
        match crate::speech::transcribe_file_async(&path_buf, crate::speech::INFERENCE_TIMEOUT)
            .await
        {
            Ok(text) => {
                tracing::debug!("Local audio transcription succeeded");
                let text = text.trim();
                return if text.is_empty() {
                    AUDIO_ICON.to_string()
                } else {
                    marked_transcript(text)
                };
            }
            Err(e) => {
                tracing::warn!(error = %e, "Local audio transcription failed");
            }
        }
    }

    tracing::warn!("Audio transcription unavailable");
    AUDIO_ICON.to_string()
}

/// Whether `path` is above the size automatic processing stops at.
///
/// `bound` is the transport's own limit, and only the Telegram receive path has one: a file above
/// it could only have arrived through the local Bot API server, and receiving such a file means
/// saving it. The bound stays on the conversions that read the whole file into memory — a document,
/// or a recording the recogniser decodes in process. A video is not one of them: its timeline is
/// decoded from the file on disk, a piece at a time, so every inbound video gets one.
async fn too_large_to_process(path: &std::path::Path, bound: u64) -> bool {
    tokio::fs::metadata(path)
        .await
        .is_ok_and(|meta| meta.len() > bound)
}

/// A media file copied into the workspace `uploads/` directory: the
/// `[Saved {label}: path]` annotation for the message and the destination
/// path for agent tool references.
struct SavedMedia {
    annotation: String,
    dest: std::path::PathBuf,
}

/// Copy a media file (image/video) into the workspace `uploads/` directory
/// under a fresh `upload_<millis>.<ext>` name, so the agent can reference it
/// via tool calls. Returns `None` when no uploads dir is available or the copy fails.
async fn save_media_to_workspace(
    media_path: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    label: &str,
    fallback_ext: &str,
) -> Option<SavedMedia> {
    let dir = uploads_dir?;
    let ext = media_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or(fallback_ext);
    let name = format!("upload_{}.{ext}", crate::util::unix_millis());
    let dest = copy_to_uploads(media_path, dir, &name).await?;
    Some(SavedMedia {
        annotation: format!("[Saved {label}: {}]", dest.display()),
        dest,
    })
}

/// Copy `src` into the workspace uploads dir `dir` as `name`, creating the dir
/// and disambiguating the name when a file of that name is already there.
///
/// Fail-open: `None` when the dir cannot be created or the copy fails — the
/// caller then annotates the message without a workspace handle (a document's
/// conversion still runs).
async fn copy_to_uploads(
    src: &std::path::Path,
    dir: &std::path::Path,
    name: &str,
) -> Option<std::path::PathBuf> {
    if let Err(e) = tokio::fs::create_dir_all(dir).await {
        tracing::warn!(
            dir = %dir.display(),
            error = %e,
            "Failed to create the workspace uploads dir"
        );
        return None;
    }
    // The created file *is* the collision check, so a failure after this point
    // would leave an empty or truncated file in the uploads dir with nothing
    // referencing it.
    let (mut dest, path) = match crate::util::create_unique_file(dir, name).await {
        Ok(handle) => handle,
        Err(e) => {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "Failed to create a file in the workspace uploads dir"
            );
            return None;
        }
    };
    let copied = async {
        let mut source = tokio::fs::File::open(src).await?;
        tokio::io::copy(&mut source, &mut dest).await?;
        Ok::<_, std::io::Error>(())
    }
    .await;
    if let Err(e) = copied {
        tracing::warn!(
            path = %src.display(),
            error = %e,
            "Failed to copy a file into the workspace uploads dir"
        );
        let _ = tokio::fs::remove_file(&path).await;
        return None;
    }
    Some(path)
}

/// Outcome of processing an IMAGE marker.
enum ImageAction {
    /// Keep the marker unchanged (e.g. HTTP/HTTPS URL).
    Keep,
    /// Replace the marker with the given text, optionally including an
    /// upload-path annotation for agent tool references. `delete_temp` is set
    /// only when the source file was consumed from this message's inbound
    /// attachments (copied/read) — out-of-scope and missing files are never deleted.
    Replace {
        replacement: String,
        upload_annotation: Option<String>,
        delete_temp: bool,
    },
    /// The target is not a decodable local raster. Each caller phrases its own
    /// note: a user-typed marker reads as an invalid image reference, while an
    /// image a document pass extracted reads as a converter note about the
    /// page (its path is a temp artifact the same pass deletes, so it must
    /// never reach model-visible content). `delete_temp` is set when the file
    /// was in scope (this message's inbound attachments).
    Invalid { delete_temp: bool },
}

impl ImageAction {
    /// Whether the source temp file was consumed from this message's inbound
    /// attachments and may be deleted. `false` for [`Self::Keep`] (an
    /// HTTP/HTTPS URL: no local file) and for a missing or out-of-scope path.
    fn delete_temp(&self) -> bool {
        match self {
            Self::Keep => false,
            Self::Replace { delete_temp, .. } | Self::Invalid { delete_temp } => *delete_temp,
        }
    }
}

/// Produce a data-URI for a confirmed-decodable local raster that is guaranteed
/// under the shared encoded-payload cap the provider accepts, so an image is
/// never silently dropped downstream. The primary encode passes the original
/// bytes through; any over-cap result is re-encoded to a bounded JPEG, and a
/// degenerate over-cap result even after that re-encode fails closed.
async fn bounded_image_data_uri(path: &std::path::Path) -> anyhow::Result<String> {
    match crate::util::local_image_to_data_uri(path).await {
        Ok(uri) if uri.len() <= media_target::MAX_DATA_URI_ENCODED_BYTES => Ok(uri),
        Ok(_) => {
            // Over-cap: fall back to a bounded JPEG re-encode; still over cap means failing
            // closed rather than forwarding an over-cap payload.
            tracing::warn!(path = %path.display(), "image data-URI exceeds the encoded-payload cap — re-encoding to a bounded JPEG");
            let reencoded = crate::util::local_image_to_compressed_data_uri_with_meta(path)
                .await?
                .data_uri;
            if reencoded.len() <= media_target::MAX_DATA_URI_ENCODED_BYTES {
                Ok(reencoded)
            } else {
                Err(anyhow::Error::msg(
                    "image exceeds the compressed data-URI cap",
                ))
            }
        }
        Err(e) => Err(e),
    }
}

/// Handle an IMAGE marker — a data URI, an [`ImageAction::Invalid`] target, or (out of scope)
/// a plain-text annotation; a workspace copy is saved when `uploads_dir` is available. The
/// reported `delete_temp` tells the caller whether the source temp file was consumed from this
/// message's inbound attachments and may be cleaned up.
async fn handle_image(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    staging_dirs: &[String],
) -> ImageAction {
    // Only a well-formed http(s) URL is sent as-is; a malformed one falls through to the
    // classifier and stays inert text, matching the provider's final image gate.
    if media_target::is_valid_remote_url(path) {
        return ImageAction::Keep;
    }

    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Image file not found for enrichment");
        return ImageAction::Invalid { delete_temp: false };
    }

    // Containment: only this message's own inbound attachment may be read, copied or deleted.
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        tracing::warn!(%path, "Image path outside this message's inbound attachments — annotating without copy");
        return ImageAction::Replace {
            replacement: format!("[Image attached: {}]", file_name_or_path(path)),
            upload_annotation: None,
            delete_temp: false,
        };
    }

    // Only a real raster converts. The file is in scope, so it is a pure temp artifact and
    // is cleaned up even though it is never consumed as an image. The classifier needs a real
    // decode (authoritative, not a magic sniff), so it runs off the async worker.
    if !matches!(
        crate::util::with_block_in_place(|| media_target::classify_media_image_target(path)),
        MediaTarget::LocalImage
    ) {
        return ImageAction::Invalid { delete_temp: true };
    }

    // The classifier established a decodable native raster, so the only failure left is an
    // image too large even after the bounded re-encode — never a junk data-URI.
    let data_uri = bounded_image_data_uri(path_obj).await;

    // The workspace copy waits until the image converts, so a dead file leaves no junk upload
    // plus a "[Saved image: ...]" annotation beside the invalid ref.
    let saved = match &data_uri {
        Ok(_) => save_media_to_workspace(path_obj, uploads_dir, "image", "png")
            .await
            .map(|saved| saved.annotation),
        Err(_) => None,
    };
    let data_uri = match data_uri {
        Ok(data_uri) => data_uri,
        Err(e) => {
            tracing::warn!(%path, error = %e, "Failed to convert image to data URI");
            return ImageAction::Invalid { delete_temp: true };
        }
    };

    ImageAction::Replace {
        replacement: format!("[IMAGE:{data_uri}]"),
        upload_annotation: saved,
        delete_temp: true,
    }
}

/// Canonical path of the daemon's Telegram attachment temp root; `None` when the directory
/// does not exist (then nothing can be in scope).
async fn canonical_telegram_files_root() -> Option<std::path::PathBuf> {
    tokio::fs::canonicalize(crate::util::telegram_files_root())
        .await
        .ok()
}

/// A file inside one of the message's own staging directories under the Telegram temp root
/// (see `util::telegram_staging_dir_name`).
///
/// This is the only legitimate source of inbound media, and the narrower of the two inbound
/// containment predicates: [`has_inbound_temp_marker`] only widens which text is replaced in
/// chat history, while this authorizes reading, copying and deleting, so it demands the exact
/// owning directory. Anything else — another chat's or message's attachment, a hand-typed
/// `[FILE:/etc/passwd]` — is inert text and is never read, copied or deleted.
async fn is_inbound_attachment(path: &std::path::Path, staging_dirs: &[String]) -> bool {
    if staging_dirs.is_empty() {
        return false;
    }
    let Some(root) = canonical_telegram_files_root().await else {
        return false;
    };
    // A path that exists must canonicalize inside the root on its own; the parent fallback
    // applies ONLY to a path that is not there at all (an attachment deleted before
    // enrichment, in the per-message directory the download created for it). A symlink
    // planted in a per-message directory but pointing outside the root exists, so the
    // fallback never rescues it and the marker is left inert.
    let candidate = match tokio::fs::canonicalize(path).await {
        Ok(canonical) => canonical,
        Err(_) => match path.parent() {
            Some(parent) => match tokio::fs::canonicalize(parent).await {
                Ok(canonical) => canonical,
                Err(_) => return false,
            },
            None => return false,
        },
    };
    let Ok(relative) = candidate.strip_prefix(&root) else {
        return false;
    };
    // Exact match: a prefix comparison would also accept a sibling directory of the same chat.
    relative
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .is_some_and(|dir| staging_dirs.iter().any(|staging_dir| staging_dir == dir))
}

/// The only shape the receive path generates for a staging directory.
/// [`ChannelMessage::attachment_dirs`] is a public field, and an empty or relative name would
/// make the removal below target the shared Telegram root instead of a directory inside it.
fn is_staging_dir_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(std::path::is_separator)
}

/// The replacement text, whether the source temp file was copied into the workspace uploads
/// (so the agent can feed the clip to the video-edit flow) and can be cleaned up, and the
/// optional timeline annotation. HTTP(S) URLs, missing files and out-of-scope paths degrade to
/// plain annotations; in-scope inbound attachments are always cleaned up.
struct VideoAction {
    replacement: String,
    delete_temp: bool,
    /// The clip's timeline in both forms: the model's carries its pictures as ordinary
    /// `[IMAGE:]` markers, the owner's bubble shows the timeline text alone.
    timeline: Option<Annotation>,
}

impl VideoAction {
    /// No workspace copy, no timeline.
    fn annotation(replacement: String) -> Self {
        Self {
            replacement,
            delete_temp: false,
            timeline: None,
        }
    }
}

async fn handle_video(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    staging_dirs: &[String],
) -> VideoAction {
    if is_http_url(path) {
        return VideoAction::annotation(format!("[Video link: {path}]"));
    }
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Video file not found for enrichment");
        return VideoAction::annotation(format!("[Invalid video reference: {path}]"));
    }
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        tracing::warn!(%path, "Video path outside this message's inbound attachments — annotating without copy");
        return VideoAction::annotation(format!("[Video attached: {}]", file_name_or_path(path)));
    }
    if let Some(saved) = save_media_to_workspace(path_obj, uploads_dir, "video", "mp4").await {
        // Read the persistent workspace copy, never the temp file. A failure becomes the
        // timeline's own sentence, carried beside the `[Saved video: ...]` annotation. Every
        // inbound video gets one, whatever its size: the timeline decodes from the file on disk
        // rather than holding it in memory, and a read has no bound of its own.
        return VideoAction {
            replacement: saved.annotation,
            delete_temp: true,
            timeline: Some(video_timeline(&saved.dest).await),
        };
    }
    // Copy failed (or no uploads dir): annotate without a timeline, but the in-scope temp file is
    // still a pure intermediate and is cleaned up.
    VideoAction {
        replacement: format!("[Video attached: {}]", file_name_or_path(path)),
        delete_temp: true,
        timeline: None,
    }
}

/// The clip's timeline as the message carries it: what the model reads — the timeline text, the
/// line naming the frames each picture holds, then its pictures as ordinary `[IMAGE:]` markers —
/// and what the owner's own bubble shows, which is the same timeline without the pictures. A
/// failure says what happened — a decode failure, a missing sound track and a failed recognition
/// are different sentences.
async fn video_timeline(path: &std::path::Path) -> Annotation {
    match crate::media::read(path, crate::media::Window::Overview).await {
        Ok(reading) => {
            let (model, attached) = crate::media::inline(&reading).await;
            // The bubble attaches no pictures, so it shows the timeline as it stands — and only
            // a reading that really inlined one needs that second form.
            Annotation {
                display: attached.then_some(reading.text),
                model,
            }
        }
        Err(error) => Annotation {
            model: error.unreadable(),
            display: None,
        },
    }
}

/// The annotation to add and whether the source temp file may be deleted.
///
/// Only this message's own inbound attachment is transcribed; a recording above the
/// automatic-processing bound is saved to the workspace instead (only the local Bot API server
/// could have delivered it, and receiving it means keeping it). A marker naming anything else —
/// one typed by the user — is annotated without a transcript and its file is never deleted.
async fn handle_audio(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    staging_dirs: &[String],
    processing_bound: Option<u64>,
) -> (String, bool) {
    if path_obj.is_file() && is_inbound_attachment(path_obj, staging_dirs).await {
        if let Some(bound) = processing_bound
            && too_large_to_process(path_obj, bound).await
        {
            let annotation =
                match save_media_to_workspace(path_obj, uploads_dir, "audio", "mp3").await {
                    Some(saved) => saved.annotation,
                    None => AUDIO_ICON.to_string(),
                };
            return (annotation, true);
        }
        return (transcribe_audio_marker(path).await, true);
    }
    tracing::warn!(
        %path,
        "Audio marker is not an inbound attachment file — annotating without transcription"
    );
    (AUDIO_ICON.to_string(), false)
}

/// An inbound attachment the receive path saved under the daemon's Telegram temp dir. Returns
/// the marker's replacement text, or `None` out of scope: a user-typed `[FILE:...]` is the
/// user's own text, and another chat's or message's attachment is never read, copied or
/// deleted. An in-scope attachment is copied into the workspace uploads dir under the sender's
/// name and converted into text and images, accumulating the copy, the text and the converter's
/// notes into `batch`; a missing attachment, an unrouted message and a failed copy each degrade
/// to a plain annotation. A file above the automatic-processing bound is saved, not read.
async fn handle_file(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    staging_dirs: &[String],
    processing_bound: Option<u64>,
    batch: &mut EnrichmentBatch,
) -> Option<String> {
    // Containment: only this message's own inbound attachment may be read, copied or deleted.
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        // Also the vanished-staging-directory shape: the parent fallback needs that directory
        // to exist, so the path reads as hand-typed and its marker stays verbatim.
        tracing::warn!(%path, "File path outside this message's inbound attachment — leaving the marker untouched");
        return None;
    }
    let name = file_name_or_path(path);
    // A pure intermediate: the workspace copy (or the extracted text) outlives this call, so a
    // real file is queued for deletion; a marker naming the staging directory is never removed.
    if path_obj.is_file() {
        batch.files_to_delete.push(path_obj.to_path_buf());
    } else if path_obj.exists() {
        return Some(format!(
            "[File {name}: the attachment is not a regular file]"
        ));
    } else {
        tracing::warn!(%path, "Inbound attachment missing for enrichment");
        return Some(format!(
            "[File {name}: could not be retrieved — the downloaded attachment is missing]"
        ));
    }
    let Some(uploads_dir) = uploads_dir else {
        // No workspace path: nowhere to copy the file into and no agent to convert it for.
        return Some(format!(
            "[File {name}: received, not saved to the workspace]"
        ));
    };

    // Copy first: the workspace handle is what the agent reads, and the conversion then runs
    // off a file that outlives the temp cleanup.
    let copy = copy_to_uploads(path_obj, uploads_dir, name).await;
    let copied_to_workspace = copy.is_some();
    let replacement = match &copy {
        Some(dest) => format!("[FILE:{}]", dest.display()),
        None => format!("[File {name}: received, not saved to the workspace]"),
    };

    // Above the bound the file could only have come through the local Bot API server, and
    // receiving it means keeping it: converting would read the whole file into memory.
    if let Some(bound) = processing_bound
        && too_large_to_process(path_obj, bound).await
    {
        let saved = if copied_to_workspace {
            " — open the saved file directly"
        } else {
            ""
        };
        batch.annotate(format!(
            "[File {name}: received but not converted (larger than the {} MB the bot reads \
             automatically){saved}]",
            bound.div_ceil(1024 * 1024)
        ));
        return Some(replacement);
    }

    // Extracted images go into the attachment's own per-message directory, which is what the
    // inbound IMAGE containment accepts. A file always has a parent directory.
    let out_dir = path_obj.parent().unwrap_or(path_obj);
    match convert_document_file(path_obj, name, out_dir).await {
        DocOutcome::Text {
            text,
            images,
            notes,
            unshown,
            all_page_text_lost,
            page_text,
        } => {
            // The IMAGE pipeline runs before the annotations are composed, because whether an
            // image was delivered decides how the no-text sentence reads.
            let delivered =
                deliver_document_images(name, &images, uploads_dir, staging_dirs, batch).await;
            // "The content was provided as images" is only true when every picture the reading
            // produced was delivered.
            let images_complete = delivered.kept > 0 && delivered.left_out == 0;
            // The reader failed on every page, so the pages carry no text: the notes below name
            // them and the "no text could be extracted" sentence — which would report a reader
            // failure as an absence — is left out. Anything else the conversion read (a comment,
            // a filled field value) is content and is delivered regardless.
            if !all_page_text_lost || !text.trim().is_empty() {
                batch.annotate(
                    extracted_text_annotation(uploads_dir, name, &text, images_complete).await,
                );
            }
            // The report of what the reading did not show leads the notes, as in the read tool's
            // answer: a kind and its count ahead of the detail of each loss.
            for line in unshown.lines() {
                batch.annotate(format!("[File {name}: {line}]"));
            }
            for note in notes {
                batch.annotate(format!("[File {name}: {note}]"));
            }
            // A description, annotation or form list can carry the text while no page produced
            // any: the reading still says so, naming an image only when one was delivered.
            if !page_text && !all_page_text_lost && !text.trim().is_empty() {
                let note = crate::document::no_text_note(images_complete);
                batch.annotate(format!("[File {name}: {note}]"));
            }
        }
        DocOutcome::Unreadable { reason } => {
            batch.annotate(format!("[File {name}: {reason}]"));
        }
        DocOutcome::Unsupported => {
            // The copy is what the agent would open, so claim it is openable only when it landed
            // in the workspace.
            let saved = if copied_to_workspace {
                " — open the saved file directly"
            } else {
                ""
            };
            batch.annotate(format!(
                "[File {name}: received but not converted (unsupported format){saved}]"
            ));
        }
    }
    Some(replacement)
}

/// Delivered through the ordinary IMAGE pipeline: a native data-URI part plus a workspace copy,
/// so the model sees them and the agent can reopen them. Only the pictures the shared limit
/// allows are delivered; the ones it left out and the ones the pipeline rejected are each
/// counted into one note, like the converter's skipped-media note. The batch effects are
/// applied here.
struct DeliveredImages {
    /// How many images the model actually received.
    kept: usize,
    /// How many pictures the document held beyond the limit.
    left_out: usize,
}

async fn deliver_document_images(
    name: &str,
    images: &[std::path::PathBuf],
    uploads_dir: &std::path::Path,
    staging_dirs: &[String],
    batch: &mut EnrichmentBatch,
) -> DeliveredImages {
    let (pictures, left_out) = crate::document::attachable_pictures(images);
    let mut delivered = DeliveredImages { kept: 0, left_out };
    let mut unreadable = 0usize;
    for image in pictures {
        let image_path = image.to_string_lossy().to_string();
        match handle_image(&image_path, image, Some(uploads_dir), staging_dirs).await {
            ImageAction::Keep => delivered.kept += 1,
            ImageAction::Invalid { delete_temp } => {
                // A temp artifact written by this pass, so the note must not name it — the file
                // is deleted right below.
                unreadable += 1;
                if delete_temp {
                    batch.files_to_delete.push(image.clone());
                }
            }
            ImageAction::Replace {
                replacement,
                upload_annotation,
                delete_temp,
            } => {
                delivered.kept += 1;
                batch.appended_markers.push(replacement);
                if let Some(annotation) = upload_annotation {
                    batch.upload_annotations.push(annotation);
                }
                if delete_temp {
                    batch.files_to_delete.push(image.clone());
                }
            }
        }
    }
    if left_out > 0 {
        // Never delivered, so their only copy is this pass's own temp artifact: consumed here
        // rather than left for the OS temp sweep.
        batch
            .files_to_delete
            .extend(images[pictures.len()..].iter().cloned());
        batch.annotate(format!(
            "[File {name}: {}]",
            crate::document::left_out_pictures_note(images.len(), left_out)
        ));
    }
    if unreadable > 0 {
        batch.annotate(format!(
            "[File {name}: {unreadable} extracted image(s) could not be read]"
        ));
    }
    delivered
}

/// Short text is inlined; longer text is spilled to `<uploads>/<stem>.extracted.txt` and the
/// annotation points at that file, so a book-sized document cannot blow up the message. Empty
/// text (a scanned, image-only document) names the delivered images instead — only when the
/// read delivered all of them, never when the picture limit left some out.
async fn extracted_text_annotation(
    uploads_dir: &std::path::Path,
    name: &str,
    text: &str,
    images_complete: bool,
) -> String {
    let text = text.trim();
    if text.is_empty() {
        return format!(
            "[File {name}: {}]",
            crate::document::no_text_note(images_complete)
        );
    }
    let char_count = text.chars().count();
    if char_count <= INLINE_TEXT_MAX_CHARS {
        return format!("[File {name}: extracted text follows]\n\n{text}");
    }
    let spill_name = format!("{}.extracted.txt", crate::util::name_stem(name));
    let spilled = crate::util::write_unique_file(uploads_dir, &spill_name, text.as_bytes()).await;
    let spill = match spilled {
        Ok(path) => path,
        Err(e) => {
            tracing::warn!(
                dir = %uploads_dir.display(),
                error = %e,
                "Failed to save the extracted document text"
            );
            return format!(
                "[File {name}: the extracted text could not be saved to the workspace]"
            );
        }
    };
    format!(
        "[File {name}: {}]",
        crate::document::spilled_text_note(char_count, &spill)
    )
}

/// One line the enriched message carries ahead of its body.
struct Annotation {
    /// What the model reads: a clip's timeline carries its pictures as ordinary `[IMAGE:]`
    /// markers.
    model: String,
    /// What the owner's own bubble shows instead, when it differs — the same line without the
    /// pictures. Usually it does not, and then nothing is kept for it.
    display: Option<String>,
}

impl Annotation {
    fn model(&self) -> &str {
        &self.model
    }

    fn display(&self) -> &str {
        self.display.as_deref().unwrap_or(&self.model)
    }
}

/// What the finished message gets prepended/appended, and which temp artifacts the pass consumed.
#[derive(Default)]
struct EnrichmentBatch {
    /// Annotation blocks prepended to the finished message.
    annotations: Vec<Annotation>,
    /// `[Saved ...]` upload annotations appended to the body.
    upload_annotations: Vec<String>,
    /// `[IMAGE:data:...]` parts a FILE ingestion extracted, appended after the marker loop
    /// (`captures_iter` already snapshotted the source, so the loop never visits them).
    appended_markers: Vec<String>,
    /// Only ever queued for this message's own inbound attachments, so user-typed markers can
    /// never delete arbitrary local files. The staging directories go afterwards.
    files_to_delete: Vec<std::path::PathBuf>,
}

impl EnrichmentBatch {
    /// One line both forms share.
    fn annotate(&mut self, text: String) {
        self.annotations.push(Annotation {
            model: text,
            display: None,
        });
    }

    /// Delete the queued artifacts and the message's staging directories, then assemble the
    /// enriched content: append the native parts, strip the handled markers, prepend the
    /// annotations. Both forms are returned — the model's, then the owner's bubble's when an
    /// annotation differs between them.
    async fn finish(self, mut body: String, staging_dirs: &[String]) -> (String, Option<String>) {
        // Deletion errors are logged; a file already gone is not an error.
        for file_path in &self.files_to_delete {
            if let Err(e) = tokio::fs::remove_file(file_path).await
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(
                    path = %file_path.display(),
                    error = %e,
                    "Failed to delete temp file after enrichment"
                );
            }
        }

        // `remove_dir` removes only an empty directory, so a staging directory still holding an
        // unconsumed artifact is left to the OS temp sweep; an already-gone directory is not an
        // error.
        let root = crate::util::telegram_files_root();
        for staging_dir in staging_dirs {
            if !is_staging_dir_name(staging_dir) {
                continue;
            }
            let _ = tokio::fs::remove_dir(root.join(staging_dir)).await;
        }

        // Upload path annotations, preceded by the `[IMAGE:data:...]` parts a document produced.
        if !self.appended_markers.is_empty() {
            let _ = write!(body, "\n\n{}", self.appended_markers.join("\n"));
        }
        if !self.upload_annotations.is_empty() {
            let annotation_block = self.upload_annotations.join("\n");
            let _ = write!(body, "\n\n{annotation_block}");
        }

        // Strip only the kinds this pass consumed (AUDIO, VIDEO): IMAGE carries the native parts
        // the model consumes, FILE the workspace handle the agent opens itself.
        let cleaned = MEDIA_MARKER_RE
            .replace_all(&body, |caps: &regex::Captures| {
                if matches!(
                    parse_media_marker(caps).0,
                    MediaMarkerKind::Audio | MediaMarkerKind::Video
                ) {
                    String::new()
                } else {
                    caps.get_match().as_str().to_string()
                }
            })
            .to_string();
        let cleaned = cleaned.trim().to_string();

        // Prepend the accumulated text descriptions: transcribed AUDIO content, VIDEO
        // annotations and a FILE's extracted text. Prepending rather than inlining at the marker
        // keeps marker-shaped text inside a document away from the strip pass above. The owner's
        // form is built only when an annotation differs from it — usually only a clip's does.
        let content = composed(&self.annotations, &cleaned, Annotation::model);
        let display = self
            .annotations
            .iter()
            .any(|annotation| annotation.display.is_some())
            .then(|| composed(&self.annotations, &cleaned, Annotation::display));
        (content, display)
    }
}

/// The message as one form reads it: the annotations, then the body.
fn composed(annotations: &[Annotation], body: &str, pick: fn(&Annotation) -> &str) -> String {
    let prefix = annotations.iter().map(pick).collect::<Vec<_>>().join("\n");
    match (prefix.is_empty(), body.is_empty()) {
        (true, _) => body.to_string(),
        (false, true) => prefix,
        (false, false) => format!("{prefix}\n\n{body}"),
    }
}

/// Process every media marker (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`, `[FILE:...]`) in a
/// single pass:
///
/// | Kind | Behavior |
/// |------|----------|
/// | IMAGE | data URI (original bytes, bounded-JPEG re-encode over the cap only) + in-scope copy |
/// | AUDIO | transcription, or the icon combo when it fails or the file is out of scope |
/// | VIDEO | workspace copy + `[Saved video: path]` + the clip's timeline |
/// | FILE | workspace copy + extracted text and images; verbatim when out of scope |
///
/// The kinds this pass consumed (AUDIO, VIDEO) are then stripped and the annotations prepended;
/// IMAGE (native parts) and FILE (workspace handle) are preserved. Temp files are cleaned up
/// within this message's own directories. `workspace_path` is the workspace whose `uploads/`
/// receives saved media copies; `None` disables copies.
// Marker dispatch hub: per-kind handling lives in the handlers above, keeping this loop flat.
pub async fn enrich_message(msg: &mut ChannelMessage, workspace_path: Option<&std::path::Path>) {
    let mut batch = EnrichmentBatch::default();
    let mut result = msg.content.clone();
    let uploads_dir = workspace_path.map(|p| p.join("uploads"));
    // The staging directories this message's producer created: the Telegram receive path or the
    // desktop composer. Empty when it has no attachment.
    let staging_dirs = msg.attachment_dirs.clone();
    // Automatic processing stops at the carrying transport's bound for the conversions that read
    // the whole file into memory — a document, or a recording the recogniser decodes in process.
    // Only the Telegram side has one: a file above the cloud's download limit can only have come
    // through the local Bot API server, and receiving it means keeping it.
    let processing_bound = (msg.channel == "telegram")
        .then_some(crate::channels::telegram::TELEGRAM_MAX_FILE_DOWNLOAD_BYTES);

    for caps in MEDIA_MARKER_RE.captures_iter(&msg.content) {
        let whole = caps.get_match();
        let (kind, path) = parse_media_marker(&caps);
        let path_obj = std::path::Path::new(path);

        match kind {
            MediaMarkerKind::File => {
                // `None` = an out-of-scope user-typed marker, left verbatim.
                if let Some(replacement) = handle_file(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    &staging_dirs,
                    processing_bound,
                    &mut batch,
                )
                .await
                {
                    result = result.replacen(whole.as_str(), &replacement, 1);
                }
            }
            MediaMarkerKind::Image => {
                let action =
                    handle_image(path, path_obj, uploads_dir.as_deref(), &staging_dirs).await;
                let delete_temp = action.delete_temp();
                match action {
                    ImageAction::Keep => {
                        // HTTP/HTTPS URL — no local file to clean up.
                    }
                    ImageAction::Invalid { .. } => {
                        // Not a decodable raster: name the failed reference.
                        let replacement = format!("[Invalid image reference: {path}]");
                        result = result.replacen(whole.as_str(), &replacement, 1);
                    }
                    ImageAction::Replace {
                        replacement,
                        upload_annotation,
                        ..
                    } => {
                        result = result.replacen(whole.as_str(), &replacement, 1);
                        if let Some(ann) = upload_annotation {
                            batch.upload_annotations.push(ann);
                        }
                    }
                }
                // Local IMAGE temp files are cleaned up only when consumed from this message's
                // inbound attachments (`delete_temp`).
                if delete_temp {
                    batch.files_to_delete.push(path_obj.to_path_buf());
                }
            }
            MediaMarkerKind::Audio => {
                // Only this message's own inbound attachment is transcribed or deleted; an
                // out-of-scope marker degrades to the icon only.
                let (annotation, delete_temp) = handle_audio(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    &staging_dirs,
                    processing_bound,
                )
                .await;
                batch.annotate(annotation);
                if delete_temp {
                    batch.files_to_delete.push(path_obj.to_path_buf());
                }
            }
            MediaMarkerKind::Video => {
                let VideoAction {
                    replacement,
                    delete_temp,
                    timeline,
                } = handle_video(path, path_obj, uploads_dir.as_deref(), &staging_dirs).await;
                result = result.replacen(whole.as_str(), &replacement, 1);
                if let Some(timeline) = timeline {
                    batch.annotations.push(timeline);
                }
                if delete_temp {
                    batch.files_to_delete.push(path_obj.to_path_buf());
                }
            }
        }
    }

    let (content, display) = batch.finish(result, &staging_dirs).await;
    msg.display_content = display;
    msg.content = content;
}

/// A marker naming a path under the daemon's Telegram temp root.
///
/// [`crate::channels::persist_content`] uses it to decide what reaches chat history: those
/// markers name temp files the daemon deletes right after enrichment, so the enriched content
/// is persisted instead. Purely syntactic, and deliberately looser than
/// [`is_inbound_attachment`]: a hand-typed marker naming that root also qualifies, which is
/// safe because this only decides which text is replaced — it never reads, copies or deletes.
#[must_use]
pub(crate) fn has_inbound_temp_marker(content: &str) -> bool {
    let root = crate::util::telegram_files_root();
    MEDIA_MARKER_RE.captures_iter(content).any(|caps| {
        let (_, path) = parse_media_marker(&caps);
        std::path::Path::new(path).starts_with(&root)
    })
}

/// Strips common trailing punctuation (commas, periods, closing brackets, colons, semicolons,
/// exclamation/question marks) that naturally appears around URLs in prose.
fn extract_urls(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for m in URL_RE.find_iter(text) {
        let mut url = m.as_str().to_string();
        // Strip trailing punctuation that isn't part of the actual URL
        while url.ends_with(&[',', '.', ')', ']', '}', ':', ';', '!', '?'][..]) {
            url.pop();
        }
        if seen.insert(url.clone()) {
            result.push(url);
        }
    }
    result
}

/// Prepends link summaries for any URLs found in the text; with none, the message is returned
/// unchanged. Links are fetched concurrently through the shared `ChromeTool` — each URL gets
/// its own isolated session tab, closed after text extraction.
pub async fn enrich_links(content: &str) -> Cow<'_, str> {
    // The cap is in bytes: it is the only bound on page text reaching the model, so it must not
    // grow with a page's character count.
    const MAX_TEXT_BYTES: usize = 5000;
    let urls = extract_urls(content);
    if urls.is_empty() {
        return Cow::Borrowed(content);
    }

    // Gate on the cached (non-probing) daemon advertisement first — the in-memory check
    // short-circuits the `--version` spawn while the daemon is confirmed down. A stale state
    // passes optimistically and the fetch tasks re-discover liveness under the probe timeout.
    if !(crate::tools::chrome_daemon::is_advertised()
        && matches!(
            crate::tools::chrome_daemon::cli_probe().await,
            crate::tools::chrome_daemon::CliStatus::Available
        ))
    {
        tracing::debug!("chrome-use not available, skipping link enrichment");
        return Cow::Borrowed(content);
    }

    let chrome = std::sync::Arc::new(ChromeTool::default());
    let mut tasks = Vec::with_capacity(urls.len());
    for url in &urls {
        let url = url.clone();
        let tab = format!(
            "link-enricher-{}",
            LINK_ENRICHER_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let chrome = std::sync::Arc::clone(&chrome);
        tasks.push(tokio::spawn(async move {
            let result = chrome.fetch_page_text(&url, &tab).await;
            // Close the tab (best-effort) whatever the fetch outcome.
            chrome.close_session(&tab).await;
            (url, result)
        }));
    }

    let mut enrichments: Vec<String> = Vec::new();
    for task in tasks {
        match task.await {
            Ok((url, Ok(body_text))) => {
                if body_text.trim().is_empty() {
                    // A blank page inserts no empty snippet.
                    tracing::debug!(url, "Link enricher: page text is empty, skipping snippet");
                    continue;
                }
                let snippet = crate::util::cut_bytes(&body_text, MAX_TEXT_BYTES);
                enrichments.push(format!("📄 [{url}]\n{snippet}"));
            }
            Ok((url, Err(e))) => {
                tracing::debug!(url, error = %e, "Link enricher: failed to fetch page text");
            }
            Err(e) => {
                tracing::debug!("Link enricher task panicked: {e}");
            }
        }
    }

    if enrichments.is_empty() {
        return Cow::Borrowed(content);
    }

    let prefix = enrichments.join("\n\n");
    Cow::Owned(format!("{prefix}\n\n{content}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;

    /// Both schemes count, a repeat is one URL, and the punctuation or brackets around an
    /// address are not part of it.
    #[test]
    fn extract_urls_applies_its_rules() {
        let cases: &[(&str, &[&str])] = &[
            (
                "Check https://example.com and http://test.org/page for info",
                &["https://example.com", "http://test.org/page"],
            ),
            (
                "Visit https://example.com and https://example.com again",
                &["https://example.com"],
            ),
            (
                "See https://example.com, and https://test.org.",
                &["https://example.com", "https://test.org"],
            ),
            (
                "(https://example.com) and [https://test.org]",
                &["https://example.com", "https://test.org"],
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(extract_urls(input), *expected, "input: {input}");
        }
    }

    /// The chat id every inbound fixture belongs to: the test messages carry it and the fixture
    /// directories are named after it.
    const TEST_CHAT_ID: &str = "1";

    /// Carries [`TEST_CHAT_ID`] but no message id, so it owns no inbound attachment — containment
    /// is per message, and any inbound fixture is out of scope for it.
    fn test_msg(content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: "test".into(),
            reply_target: "test".into(),
            content: content.to_string(),
            display_content: None,
            channel: "test".into(),
            workspace: "test".into(),
            optimistic_id: None,
            callback_query_id: None,
            reply_reference: None,
            chat_id: Some(TEST_CHAT_ID.into()),
            message_id: None,
            attachment_dirs: Vec::new(),
            parts: Vec::new(),
            update_date: None,
            forwarded: false,
            update_ids: Vec::new(),
            dictated: false,
            urgent: false,
        }
    }

    /// Owning the staging directories of `message_ids` — one for a single message, several for
    /// the album shape. `message_id` stays the first, as an album merge keeps the first.
    fn inbound_album_msg(message_ids: &[i64], content: &str) -> ChannelMessage {
        ChannelMessage {
            message_id: message_ids.first().copied(),
            attachment_dirs: message_ids
                .iter()
                .map(|id| crate::util::telegram_staging_dir_name(TEST_CHAT_ID, *id))
                .collect(),
            ..test_msg(content)
        }
    }

    fn inbound_msg(message_id: i64, content: &str) -> ChannelMessage {
        inbound_album_msg(&[message_id], content)
    }

    fn telegram_root() -> std::path::PathBuf {
        crate::util::telegram_files_root()
    }

    /// The containment root must exist before path canonicalization — a missing root makes every
    /// path look out of scope.
    async fn ensure_telegram_files_dir() {
        tokio::fs::create_dir_all(telegram_root()).await.unwrap();
    }

    /// A unique scratch dir under the system temp dir (outside the Telegram temp dir) with a fake
    /// workspace and one arbitrary media file. Returns `(tmp_root, ws_path, arbitrary_file)`.
    async fn out_of_scope_fixture(
        prefix: &str,
        file_name: &str,
        contents: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        ensure_telegram_files_dir().await;
        let tmp_root = crate::util::test::test_root().join(prefix);
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();
        let arbitrary = tmp_root.join(file_name);
        tokio::fs::write(&arbitrary, contents).await.unwrap();
        (tmp_root, ws_path, arbitrary)
    }

    /// Exactly as the receive path delivers one: the file inside its own staging subdirectory of
    /// the Telegram temp root. Each test uses its own message id, which is what the per-message
    /// containment compares against. Returns `(msg_dir, attachment)`.
    async fn telegram_attachment_fixture(
        chat_id: &str,
        message_id: i64,
        file_name: &str,
        bytes: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        ensure_telegram_files_dir().await;
        let msg_dir =
            telegram_root().join(crate::util::telegram_staging_dir_name(chat_id, message_id));
        tokio::fs::create_dir_all(&msg_dir).await.unwrap();
        let attachment = msg_dir.join(file_name);
        tokio::fs::write(&attachment, bytes).await.unwrap();
        (msg_dir, attachment)
    }

    /// In scope for [`inbound_msg`] with the same message id.
    async fn inbound_attachment_fixture(
        message_id: i64,
        file_name: &str,
        bytes: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        telegram_attachment_fixture(TEST_CHAT_ID, message_id, file_name, bytes).await
    }

    /// Plus a workspace to ingest it into. Returns `(tmp_root, ws_path, msg_dir, attachment)`.
    async fn inbound_ingest_fixture(
        prefix: &str,
        message_id: i64,
        file_name: &str,
        bytes: &[u8],
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let tmp_root = crate::util::test::test_root().join(prefix);
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();
        let (msg_dir, attachment) = inbound_attachment_fixture(message_id, file_name, bytes).await;
        (tmp_root, ws_path, msg_dir, attachment)
    }

    /// Generate a real decodable PNG of the given dimensions (solid gradient).
    fn real_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                u8::try_from(x % 256).expect("x % 256 fits in u8"),
                u8::try_from(y % 256).expect("y % 256 fits in u8"),
                128,
            ])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("fixture PNG must encode");
        bytes
    }

    #[tokio::test]
    async fn enrich_image_http_url_passthrough() {
        let mut msg = test_msg("Check this [IMAGE:https://example.com/img.png] out");
        enrich_message(&mut msg, None).await;
        assert_eq!(
            msg.content,
            "Check this [IMAGE:https://example.com/img.png] out"
        );
    }

    /// A user-typed marker is out of scope, so it degrades to the icon combo (no engine in tests).
    #[tokio::test]
    async fn enrich_audio_annotation_and_strip() {
        let mut msg = test_msg("Listen [AUDIO:/tmp/audio_xyz.mp3] to this");
        enrich_message(&mut msg, None).await;
        assert!(
            msg.content.contains(AUDIO_ICON),
            "Audio annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
        assert!(
            !msg.content.contains("audio_xyz"),
            "Audio temp file name must not appear, got: {}",
            msg.content
        );
        assert!(msg.content.contains("Listen"), "Original text preserved");
        assert!(msg.content.contains("to this"), "Original text preserved");
    }

    /// Read into a data URI of the source bytes exactly, with the temp file and its per-message
    /// directory going with the read.
    #[tokio::test]
    async fn enrich_image_valid_file_converts_to_data_uri_and_deletes_temp() {
        // The fixture must be one of this message's inbound attachments to be in scope.
        let source_bytes = real_png(2, 1);
        let (msg_dir, tmp) = inbound_attachment_fixture(7001, "photo.png", &source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7001, &format!("Image: [IMAGE:{path_str}]"));
        enrich_message(&mut msg, None).await;

        assert!(
            msg.content.contains("[IMAGE:data:image/png;base64,"),
            "Expected data URI, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(&path_str),
            "Raw file path must not remain in content"
        );
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
        assert!(!msg_dir.exists(), "Per-message directory must be deleted");

        // The data URI carries the source bytes untouched, at any size.
        let source_bytes = real_png(64, 48);
        let (_msg_dir, tmp) = inbound_attachment_fixture(7006, "art.png", &source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();
        let mut msg = inbound_msg(7006, &format!("Art: [IMAGE:{path_str}]"));
        enrich_message(&mut msg, None).await;
        let expected = format!(
            "[IMAGE:data:image/png;base64,{}]",
            STANDARD.encode(&source_bytes)
        );
        assert!(
            msg.content.contains(&expected),
            "Image data URI must be byte-identical to the source, got: {}",
            msg.content
        );
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_image_with_workspace_creates_upload_annotation() {
        let tmp_root = crate::util::test::test_root().join("test_enrich_ws");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // The fixture must be one of this message's inbound attachments to be in scope.
        let source_bytes = real_png(2, 1);
        let (_msg_dir, tmp_img) =
            inbound_attachment_fixture(7002, "photo.png", &source_bytes).await;
        let img_path_str = tmp_img.to_string_lossy().to_string();

        let mut msg = inbound_msg(7002, &format!("Image: [IMAGE:{img_path_str}]"));
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(msg.content.contains("[IMAGE:data:image/png;base64,"));
        assert!(
            msg.content.contains("[Saved image:"),
            "Upload annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !tmp_img.exists(),
            "Temp file must be deleted after enrichment"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    // The read reaches the process-global ffmpeg install (root and the failed-attempt latch), which
    // the install tests own: this one takes their key so the two cannot interleave.
    #[tokio::test]
    #[serial_test::serial(ffmpeg_releases)]
    async fn enrich_video_with_workspace_copies_and_annotates() {
        let tmp_root = crate::util::test::test_root().join("test_enrich_video_ws");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // Only this message's own inbound attachments are eligible for the copy.
        let mp4_header: &[u8] = &[
            0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D, 0x00, 0x00,
            0x00, 0x00, 0x69, 0x73, 0x6F, 0x6D, 0x69, 0x73, 0x6F, 0x32,
        ];
        let (_msg_dir, tmp_video) = inbound_attachment_fixture(7003, "clip.mp4", mp4_header).await;
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = inbound_msg(7003, &format!("Edit this clip: [VIDEO:{video_path_str}]"));
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        // The annotation points at the workspace uploads copy, so the agent can feed it to
        // video_edit.
        assert!(
            msg.content.contains("[Saved video:"),
            "Video upload annotation must be present, got: {}",
            msg.content
        );
        assert!(
            msg.content
                .contains(&ws_path.join("uploads").display().to_string()),
            "Annotation must point into workspace uploads, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[VIDEO:"),
            "VIDEO marker must be stripped"
        );
        // The fixture's bytes are a bare header, so the read cannot produce a timeline: it is the
        // distinguishable failure sentence instead. Which one depends on the machine — a managed
        // ffmpeg that is present reports the decode, one that is not reports itself — so only the
        // envelope is asserted.
        assert!(
            msg.content.contains("The clip could not be read: "),
            "Timeline failure must be annotated, got: {}",
            msg.content
        );
        // A failed timeline has no pictures, so the two forms are the same.
        assert!(msg.display_content.is_none());
        assert!(
            !tmp_video.exists(),
            "Temp video file must be deleted after enrichment"
        );
        let _ = tokio::fs::remove_file(&tmp_video).await;
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_image_outside_inbound_attachments_annotates_without_read_or_copy() {
        // An injected marker naming an arbitrary readable file must not be read into model
        // context, copied into uploads, or deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_img_outside",
            "secret.png",
            b"top secret image bytes",
        )
        .await;
        let marker = format!("Look at [IMAGE:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content.contains("[Image attached: secret.png]"),
            "Out-of-scope path must degrade to a plain-text annotation, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("data:image"),
            "No data URI may be produced for out-of-scope paths (would read the file)"
        );
        assert!(!msg.content.contains("[Saved image:"));
        assert!(!msg.content.contains("[IMAGE:"));
        assert!(
            arbitrary.exists(),
            "Source file outside the telegram temp dir must not be deleted"
        );
        assert_eq!(
            tokio::fs::read(&arbitrary).await.unwrap(),
            b"top secret image bytes",
            "Source file contents must be unchanged"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_http_url_kept_as_plain_text() {
        let mut msg = test_msg("Edit [VIDEO:https://example.com/clip.mp4] this");
        enrich_message(&mut msg, None).await;
        assert!(
            msg.content
                .contains("[Video link: https://example.com/clip.mp4]"),
            "HTTP video URL must be kept as plain-text reference, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            !crate::util::MEDIA_MARKER_RE.is_match(&msg.content),
            "the product's own note must not read as a media marker: {}",
            msg.content
        );
    }

    #[tokio::test]
    async fn enrich_video_outside_inbound_attachments_annotates_without_copy_or_delete() {
        // An injected marker naming an arbitrary readable file must not be copied into uploads
        // (an exfiltration vector) or deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_video_nonmm",
            "secret.mp4",
            b"top secret video bytes",
        )
        .await;
        let marker = format!("Watch [VIDEO:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content.contains("[Video attached: secret.mp4]"),
            "Out-of-scope path must degrade to a plain-text annotation, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            arbitrary.exists(),
            "Source file outside the telegram temp dir must not be deleted"
        );
        assert_eq!(
            tokio::fs::read(&arbitrary).await.unwrap(),
            b"top secret video bytes",
            "Source file contents must be unchanged"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// A file in this message's own inbound directory that is not a decodable raster must never
    /// fail open to a junk data URI: the classifier rejects a non-image with an image extension,
    /// and the primary encode refuses a PNG whose payload is truncated. Both degrade to an
    /// `[Invalid image reference: …]` note, and the in-scope temp file is cleaned up.
    #[tokio::test]
    async fn enrich_image_that_is_not_a_decodable_raster_never_fails_open_to_a_junk_data_uri() {
        // A non-image file with an image extension.
        let (_msg_dir, tmp) = inbound_attachment_fixture(7007, "photo.png", b"not an image").await;
        let path_str = tmp.to_string_lossy().to_string();
        let mut msg = inbound_msg(7007, &format!("Photo: [IMAGE:{path_str}]"));
        enrich_message(&mut msg, None).await;
        assert!(
            msg.content
                .contains(&format!("[Invalid image reference: {path_str}]")),
            "Non-raster in-scope file must degrade to an invalid reference, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[IMAGE:data:"),
            "No junk data URI may be produced for a non-raster file, got: {}",
            msg.content
        );
        assert!(
            !tmp.exists(),
            "In-scope non-raster temp file must be cleaned up"
        );

        // A truncated PNG: valid magic and IHDR, no decodable payload.
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let truncated = &buf[..buf.len().min(24)];
        let (_msg_dir, tmp) = inbound_attachment_fixture(7008, "photo.png", truncated).await;
        let path_str = tmp.to_string_lossy().to_string();
        let mut msg = inbound_msg(7008, &format!("Photo: [IMAGE:{path_str}]"));
        enrich_message(&mut msg, None).await;
        assert!(
            msg.content
                .contains(&format!("[Invalid image reference: {path_str}]")),
            "Corrupt-but-magic-valid in-scope file must degrade to an invalid reference, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[IMAGE:data:"),
            "No junk data URI may be produced for a corrupt file, got: {}",
            msg.content
        );
        // Defensive cleanup in case an assertion above fails.
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_audio_file_deleted_on_failure() {
        // Only this message's own inbound attachments are eligible for cleanup
        // (user-typed markers must never delete arbitrary files).
        let (_msg_dir, tmp) =
            inbound_attachment_fixture(7010, "voice.mp3", b"fake audio content").await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7010, &format!("Audio: [AUDIO:{path_str}]"));
        enrich_message(&mut msg, None).await;

        // This message's own inbound audio attachment is consumed, so the pure intermediate
        // file is deleted even when transcription fails (there is no engine in tests).
        assert!(
            !tmp.exists(),
            "Audio temp file must be deleted on transcription failure"
        );
        // Defensive cleanup in case the assertion above fails.
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_file_inlines_text_and_copies_to_uploads() {
        let document = "# Title\n\nHello from a document";
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_file_md", 7011, "notes.md", document.as_bytes())
                .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7011, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        // The marker becomes the workspace handle and the text is inlined.
        let copy = ws_path.join("uploads").join("notes.md");
        assert!(
            msg.content.contains(&format!("[FILE:{}]", copy.display())),
            "Marker must point at the workspace copy, got: {}",
            msg.content
        );
        assert!(
            msg.content.contains("Hello from a document"),
            "Extracted document text must be inline, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(&attachment.display().to_string()),
            "The temp path must not survive anywhere, got: {}",
            msg.content
        );
        assert_eq!(
            tokio::fs::read(&copy).await.unwrap(),
            document.as_bytes(),
            "Workspace copy must be byte-identical to the inbound file"
        );
        // Inbound temp file and its whole per-message directory are gone.
        assert!(!attachment.exists(), "Inbound temp file must be deleted");
        assert!(
            !msg_dir.exists(),
            "Inbound per-message directory must be deleted"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// A document the reader failed on for every page it could read, yet which
    /// still produced text: the attachment path must deliver that text. Dropping
    /// it because the reader failed is exactly the loss this covers.
    #[tokio::test]
    async fn enrich_file_keeps_text_of_pages_that_fail_after_producing_it() {
        let (tmp_root, ws_path, _msg_dir, attachment) = inbound_ingest_fixture(
            "test_enrich_file_partial_pdf",
            7015,
            "report.pdf",
            &crate::document::test_fixtures::pdf_with_all_pages_failing_midway(),
        )
        .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7015, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert_eq!(
            msg.content.matches("Kept page text long enough").count(),
            2,
            "every page's text must be inline, got: {}",
            msg.content
        );
        assert!(
            msg.content.contains(
                "[File report.pdf: the text of pages 1, 2 could not be read in full (2 of 2 pages)]"
            ),
            "the pages must be named in a delimited note, got: {}",
            msg.content
        );
        assert_eq!(
            msg.content
                .matches("[IMAGE:data:image/jpeg;base64,")
                .count(),
            2,
            "each page that could not be read in full comes back as an image, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(crate::document::NO_TEXT_NOTE),
            "delivered text must never be reported as a document without one, got: {}",
            msg.content
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// A document the reading found things in but did not show: the attachment
    /// path names the kinds in the same words the read tool's answer does, so an
    /// agent knows what it is working without whichever way the file arrived.
    #[tokio::test]
    async fn enrich_file_names_what_the_reading_did_not_show() {
        use crate::ooxml::test_fixtures::{DOCX_BODY, zip_fixture};
        let package = zip_fixture(&[
            ("word/document.xml", DOCX_BODY),
            ("word/media/clip.mp4", b"not a picture at all"),
        ]);
        let (tmp_root, ws_path, _msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_document_report", 7040, "clip.docx", &package)
                .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7040, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content
                .contains("[File clip.docx: 1 embedded video(s) not shown]"),
            "the report must reach the inbound answer too, got: {}",
            msg.content
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// A document holding more pictures than one read may attach: the attachment
    /// path delivers the first six and says how many the document holds and how
    /// many the limit left out — the same count the read tool's answer carries —
    /// and never claims a partial delivery was its whole content provided as
    /// images.
    #[tokio::test]
    async fn enrich_file_delivers_at_most_six_document_pictures() {
        use crate::ooxml::test_fixtures::zip_fixture;
        const EMPTY_BODY: &[u8] = br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body></w:body></w:document>"#;
        let png = crate::util::test::noisy_png(16, 16);
        let names: Vec<String> = (1..=7)
            .map(|page| format!("word/media/page_{page}.png"))
            .collect();
        let mut entries: Vec<(&str, &[u8])> = vec![("word/document.xml", EMPTY_BODY)];
        entries.extend(names.iter().map(|name| (name.as_str(), png.as_slice())));
        let package = zip_fixture(&entries);
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_document_pictures", 7041, "scan.docx", &package)
                .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7041, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert_eq!(
            msg.content.matches("[IMAGE:data:image/png;base64,").count(),
            6,
            "one read attaches at most six pictures, got: {}",
            msg.content
        );
        assert!(
            msg.content.contains(
                "[File scan.docx: 7 pictures in the document; 1 left out by the 6-picture \
                 limit]"
            ),
            "the answer must count what it left out, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(crate::document::NO_TEXT_LAYER_NOTE),
            "a partial delivery is not its whole content provided as images, got: {}",
            msg.content
        );
        // The pictures left out are this pass's own artifacts: consumed, so nothing lingers
        // beside the (deleted) attachment.
        assert!(
            !msg_dir.exists(),
            "the per-message directory must be emptied and deleted"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// A one-page PDF whose page has no `/MediaBox` — the shape the text reader
    /// panics on, so every page's text is lost — but which carries a highlight
    /// with a Russian comment and author, and one filled form field whose widget
    /// is on that page.
    fn pdf_marks_with_every_page_text_lost() -> Vec<u8> {
        use crate::document::test_fixtures::{PdfFixture, one_page_pdf, pdf_text_string};
        const CONTENT: &[u8] = b"0.1 0.5 0.9 rg 0 0 612 792 re f";
        let mut fixture = PdfFixture::new();
        let highlight = fixture.push(format!(
            "<< /Subtype /Highlight /T {} /Contents {} >>",
            pdf_text_string("Иван Петров"),
            pdf_text_string("проверьте третью фигуру")
        ));
        let field = fixture.push(format!(
            "<< /Subtype /Widget /FT /Tx /T (Name) /V {} /Rect [0 0 0 0] >>",
            pdf_text_string("Иван Петров")
        ));
        let acroform = fixture.push(format!("<< /Fields [{field}] >>"));
        // No `/MediaBox` in the page dictionary: the reader panics on this page.
        one_page_pdf(
            fixture,
            CONTENT,
            &format!("/Annots [{highlight} {field}]"),
            &format!(" /AcroForm {acroform}"),
        )
    }

    /// Every page's text lost, yet the document carries marks: the marks are
    /// content, so the attachment path must deliver them — dropping the only
    /// text the document has is exactly the loss this covers — while the page
    /// itself still comes back as an image.
    #[tokio::test]
    async fn enrich_file_delivers_marks_when_every_page_text_was_lost() {
        let (tmp_root, ws_path, _msg_dir, attachment) = inbound_ingest_fixture(
            "test_enrich_file_marks_only_pdf",
            7023,
            "marked.pdf",
            &pdf_marks_with_every_page_text_lost(),
        )
        .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7023, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content
                .contains("highlight (Иван Петров) on page 1:\n  проверьте третью фигуру"),
            "the annotation must survive the lost page text, got: {}",
            msg.content
        );
        assert!(
            msg.content.contains("Name (text) on page 1: Иван Петров"),
            "the filled field must survive the lost page text, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(crate::document::NO_TEXT_NOTE),
            "delivered marks are text, not a document without one, got: {}",
            msg.content
        );
        assert_eq!(
            msg.content
                .matches("[IMAGE:data:image/jpeg;base64,")
                .count(),
            1,
            "the page whose text was lost still comes back as an image, got: {}",
            msg.content
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_long_text_spills_to_extracted_sidecar() {
        let text = "Lorem ipsum dolor sit amet. ".repeat(300);
        assert!(text.chars().count() > INLINE_TEXT_MAX_CHARS);
        let (tmp_root, ws_path, _msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_file_long", 7012, "long.txt", text.as_bytes())
                .await;
        let marker = format!("Read [FILE:{}]", attachment.display());

        let mut msg = inbound_msg(7012, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        // No inline body — the text goes to a sidecar the message names.
        assert!(
            !msg.content.contains(&text[..200]),
            "Long text must not be inlined, got: {}",
            msg.content
        );
        let uploads_dir = ws_path.join("uploads");
        let mut entries = tokio::fs::read_dir(&uploads_dir).await.unwrap();
        let mut spill = None;
        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry
                .file_name()
                .to_string_lossy()
                .ends_with(".extracted.txt")
            {
                spill = Some(entry.path());
            }
        }
        let spill = spill.expect("an .extracted.txt sidecar must be written");
        // The note is asserted in full, closing bracket included, so a note
        // left undelimited cannot pass unnoticed.
        assert!(
            msg.content.contains(&format!(
                "[File long.txt: the extracted text is too long to inline ({} characters); \
                 the full text was saved to {} — read that file.]",
                text.trim().chars().count(),
                spill.display()
            )),
            "Annotation must be a delimited note naming the sidecar, got: {}",
            msg.content
        );
        assert_eq!(
            tokio::fs::read_to_string(&spill).await.unwrap(),
            text.trim(),
            "Sidecar must carry the full extracted text"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_out_of_scope_marker_left_verbatim() {
        // A `[FILE:...]` marker is also user-typed text: reading, copying, or
        // deleting anything it names would be an exfiltration vector.
        let (tmp_root, ws_path, arbitrary) =
            out_of_scope_fixture("test_enrich_file_outside", "secret.md", b"# top secret").await;
        let marker = format!("Look at [FILE:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert_eq!(
            msg.content, marker,
            "Out-of-scope marker must stay verbatim"
        );
        assert_eq!(
            tokio::fs::read(&arbitrary).await.unwrap(),
            b"# top secret",
            "Source file contents must be unchanged"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for out-of-scope paths"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_other_chat_attachment_left_inert() {
        // Inbound containment is per message: only the directory of the message
        // owning the marker is read, copied and deleted. Another chat's
        // directory (chat `12` is also the off-by-prefix trap for chat `1`) and
        // another message of the same chat are both inert.
        let tmp_root = crate::util::test::test_root().join("test_enrich_file_other_chat");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();
        let (other_msg_dir, other_attachment) =
            telegram_attachment_fixture("2", 7014, "secret.md", b"# other chat").await;
        let (longer_id_dir, longer_id_attachment) =
            telegram_attachment_fixture("12", 7015, "secret.md", b"# chat 12").await;
        let (sibling_msg_dir, sibling_attachment) =
            telegram_attachment_fixture(TEST_CHAT_ID, 7016, "secret.md", b"# sibling").await;
        for attachment in [
            &other_attachment,
            &longer_id_attachment,
            &sibling_attachment,
        ] {
            let marker = format!("Look at [FILE:{}]", attachment.display());
            let mut msg = inbound_msg(7014, &marker);
            enrich_message(&mut msg, Some(ws_path.as_path())).await;
            assert_eq!(msg.content, marker, "Another message's attachment is inert");
        }

        assert_eq!(
            tokio::fs::read(&other_attachment).await.unwrap(),
            b"# other chat",
            "Another chat's attachment must not be read or deleted"
        );
        assert_eq!(
            tokio::fs::read(&longer_id_attachment).await.unwrap(),
            b"# chat 12",
            "A longer chat id's attachment must not be read or deleted"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created for another message's attachment"
        );
        let _ = tokio::fs::remove_dir_all(&other_msg_dir).await;
        let _ = tokio::fs::remove_dir_all(&longer_id_dir).await;
        let _ = tokio::fs::remove_dir_all(&sibling_msg_dir).await;
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_album_members_all_ingested() {
        // An album is merged into one message keeping only the first member's
        // message id, so every member's directory must be carried over or all
        // but the first attachment would be rejected by containment.
        let (tmp_root, ws_path, first_dir, first) =
            inbound_ingest_fixture("test_enrich_file_album", 7017, "first.md", b"# first").await;
        let (second_dir, second) =
            telegram_attachment_fixture(TEST_CHAT_ID, 7018, "second.md", b"# second").await;
        let marker = format!(
            "Album [FILE:{}] and [FILE:{}]",
            first.display(),
            second.display()
        );

        let mut msg = inbound_album_msg(&[7017, 7018], &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        let uploads_dir = ws_path.join("uploads");
        for (name, bytes) in [
            ("first.md", &b"# first"[..]),
            ("second.md", &b"# second"[..]),
        ] {
            let copy = uploads_dir.join(name);
            assert!(
                msg.content.contains(&format!("[FILE:{}]", copy.display())),
                "Marker for {name} must point at the workspace copy, got: {}",
                msg.content
            );
            assert_eq!(
                tokio::fs::read(&copy).await.unwrap(),
                bytes,
                "Workspace copy of {name} must be byte-identical to the inbound file"
            );
        }
        for temp in [&first, &second] {
            assert!(
                !msg.content.contains(&temp.display().to_string()),
                "No raw temp path may survive, got: {}",
                msg.content
            );
        }
        assert!(
            !first_dir.exists() && !second_dir.exists(),
            "Both per-message directories must be deleted"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_vanished_attachment_annotated_and_dir_removed() {
        // The download wrote the file into its staging directory and the file is
        // gone by enrichment: the parent-directory fallback still recognizes the
        // directory as ours, so the marker is annotated and the now-empty
        // directory is cleaned up.
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_file_vanished", 7020, "gone.md", b"# gone").await;
        tokio::fs::remove_file(&attachment).await.unwrap();
        let marker = format!("Read [FILE:{}]", attachment.display());

        let mut msg = inbound_msg(7020, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content.contains(
                "[File gone.md: could not be retrieved — the downloaded attachment is missing]"
            ),
            "A vanished attachment must be annotated, got: {}",
            msg.content
        );
        assert!(
            !msg_dir.exists(),
            "The now-empty staging directory must be deleted"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "A vanished attachment must not be copied into the workspace"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// The staging-directory cleanup is per message, not per marker kind: an
    /// image attachment that is gone by enrichment still leaves no directory behind.
    #[tokio::test]
    async fn enrich_vanished_image_attachment_leaves_no_staging_dir() {
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_image_vanished", 7022, "gone.png", b"png").await;
        tokio::fs::remove_file(&attachment).await.unwrap();
        let marker = format!("Look at [IMAGE:{}]", attachment.display());

        let mut msg = inbound_msg(7022, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert!(
            msg.content.contains("[Invalid image reference:"),
            "A vanished image must be annotated, got: {}",
            msg.content
        );
        assert!(
            !msg_dir.exists(),
            "The empty staging directory must be removed"
        );
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_marker_naming_staging_dir_itself_is_not_removed_and_dir_cleaned_up() {
        // A marker naming the staging directory itself passes containment (it is
        // inside the root) but is not a regular file: it must never be
        // `remove_file`d, and the empty directory must still be cleaned up
        // without a spurious deletion failure.
        let (msg_dir, attachment) =
            inbound_attachment_fixture(7021, "placeholder.md", b"# x").await;
        tokio::fs::remove_file(&attachment).await.unwrap();
        let marker = format!("Look at [FILE:{}]", msg_dir.display());

        let mut msg = inbound_msg(7021, &marker);
        enrich_message(&mut msg, None).await;

        assert!(
            msg.content
                .contains(": the attachment is not a regular file]"),
            "A directory marker must be annotated as a non-file, got: {}",
            msg.content
        );
        assert!(
            !msg_dir.exists(),
            "The empty staging directory must be removed"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn enrich_file_symlink_in_inbound_dir_marker_left_verbatim() {
        // A symlink planted in this message's own per-message directory but
        // pointing outside the temp root exists, so the parent-directory
        // fallback must not rescue it: reading or copying it would exfiltrate
        // the target, and deleting it would remove a path the user placed
        // inside the containment root.
        let (tmp_root, ws_path, outside) =
            out_of_scope_fixture("test_enrich_file_symlink", "secret.md", b"# top secret").await;
        let (msg_dir, _) =
            telegram_attachment_fixture(TEST_CHAT_ID, 7019, "placeholder.bin", b"x").await;
        let link = msg_dir.join("link.md");
        std::os::unix::fs::symlink(&outside, &link).expect("plant the symlink");
        let marker = format!("Look at [FILE:{}]", link.display());

        let mut msg = inbound_msg(7019, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        assert_eq!(msg.content, marker, "A symlinked marker must stay verbatim");
        assert_eq!(
            tokio::fs::read(&outside).await.unwrap(),
            b"# top secret",
            "The symlink target must never be read or deleted"
        );
        assert!(
            tokio::fs::symlink_metadata(&link).await.is_ok(),
            "The symlink itself must not be deleted"
        );
        assert!(
            msg_dir.exists(),
            "The per-message directory holding the symlink must not be deleted"
        );
        assert!(
            !ws_path.join("uploads").exists(),
            "No uploads copy may be created through a symlink"
        );
        let _ = tokio::fs::remove_file(&link).await;
        let _ = tokio::fs::remove_dir_all(&msg_dir).await;
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_file_unsupported_blob_annotates_and_keeps_workspace_copy() {
        // Non-UTF-8 bytes with an unknown extension: no converter applies, but
        // the agent still gets the workspace copy to open itself.
        let bytes: &[u8] = &[0x00, 0x01, 0x02, 0xFF, 0xFE, b'x'];
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_file_blob", 7013, "blob.bin", bytes).await;
        let marker = format!("Attached [FILE:{}]", attachment.display());

        let mut msg = inbound_msg(7013, &marker);
        enrich_message(&mut msg, Some(ws_path.as_path())).await;

        let copy = ws_path.join("uploads").join("blob.bin");
        assert!(
            msg.content.contains(&format!("[FILE:{}]", copy.display())),
            "Marker must point at the workspace copy, got: {}",
            msg.content
        );
        assert!(
            msg.content.contains(
                "[File blob.bin: received but not converted (unsupported format) — open the saved file directly]"
            ),
            "Unsupported format must be annotated, got: {}",
            msg.content
        );
        assert_eq!(
            tokio::fs::read(&copy).await.unwrap(),
            bytes,
            "Workspace copy must be byte-identical to the inbound file"
        );
        assert!(!attachment.exists() && !msg_dir.exists());
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_without_workspace_deletes_in_scope_temp() {
        // No workspace path: the video gets a plain annotation and no timeline, but the in-scope
        // inbound attachment is still a pure intermediate artifact and must be cleaned up.
        let (_msg_dir, tmp_video) = inbound_attachment_fixture(7004, "clip.mp4", b"fake mp4").await;
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = inbound_msg(7004, &format!("Watch [VIDEO:{video_path_str}] this clip"));
        enrich_message(&mut msg, None).await;

        assert!(
            msg.content.contains(&format!(
                "[Video attached: {}]",
                tmp_video.file_name().unwrap().to_string_lossy()
            )),
            "Plain video annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[Saved video:"),
            "No workspace copy may be made without an uploads dir, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
        assert!(
            !tmp_video.exists(),
            "In-scope temp video must be deleted even without a workspace copy"
        );
    }
}

//! Channel message enrichment: media marker processing, link enrichment, file
//! operations, and per-role media handling.
//!
//! This module transforms [`ChannelMessage`] content before it reaches the
//! agent pipeline. It handles:
//! - **Media markers** (`[IMAGE: ...]`, `[AUDIO: ...]`, `[VIDEO: ...]`,
//!   `[FILE: ...]`) → inbound local images become native data-URI parts for
//!   EVERY role (byte-identical for Assistant, bounded-JPEG compressed for all
//!   others); audio is transcribed to text; video handling is workspace copy +
//!   transcription for every role (no role split); an inbound document is
//!   copied into the workspace and converted to text plus images
//! - **Link enrichment** → prepends webpage summaries for URLs in the message
//! - **File operations** → saving media to workspace, cleaning up temporary
//!   files
//!
//! **Containment invariant**: all local file reads, copies, and deletes are
//! scoped to the message's own staging directories — the names in
//! [`ChannelMessage::attachment_dirs`] (built by `util::telegram_staging_dir_name`),
//! one for a single message and several for an album merged into a single one.
//! A marker is in scope only when it names a file inside one of those
//! directories; everything else — another chat's or message's attachment, a
//! hand-typed `[IMAGE:...]` / `[AUDIO:...]` / `[VIDEO:...]` / `[FILE:...]`
//! annotation, or an off-chat (gui/voice) message, which carries no Telegram
//! identity and no attachments — is inert text: IMAGE/AUDIO/VIDEO degrade to a
//! plain annotation and `[FILE:...]` stays verbatim, so it is never read,
//! transcribed, copied into workspace uploads, or deleted.
//!
//! The entry points are [`enrich_message`] and [`enrich_links`], re-exported
//! from [`crate::channels`], and [`has_inbound_temp_marker`], which
//! [`crate::channels::persist_content`] calls directly to decide what content is
//! persisted to chat history. The [`EnrichmentStrategy`] struct carries the
//! per-role knobs: image and video handling are unconditional (native data-URI
//! parts for images, workspace copy + transcription for videos — every role),
//! while image compression is role-dependent.

use crate::ChannelMessage;
use crate::channels::document::{DocOutcome, INLINE_TEXT_MAX_CHARS, convert_document};
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

/// URL regex: matches http:// and https:// URLs, stopping at whitespace, angle
/// brackets, or double-quotes.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"']+"#).expect("URL regex must compile"));

/// Process-global sequence for link-enricher session names. The
/// `link-enricher-` prefix is load-bearing: `is_mahbot_session_name` sweeps
/// only sessions starting with it. Unique suffixes keep concurrent
/// `enrich_links` runs from sharing a chrome session.
static LINK_ENRICHER_SEQ: AtomicU64 = AtomicU64::new(0);

/// The audio-transcription icon combo (sound written into text). Used both as
/// the transcription-failure fallback and as the annotation for out-of-scope
/// `[AUDIO:...]` markers that must never be read or deleted.
const AUDIO_ICON: &str = "🔊✍️";

/// Reason reported for an inbound attachment whose bytes could not be read or
/// whose conversion panicked.
const UNREADABLE_REASON: &str = "could not be read";
/// Bound on concurrently converting inbound documents: each conversion holds the
/// whole attachment plus its decoded rasters, and every message runs on its own
/// task, so a burst would otherwise multiply peak memory.
static DOCUMENT_CONVERSIONS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

/// Transcribe an audio file referenced by a `[AUDIO:...]` marker and return
/// the content to embed in the message: the audio-transcription icon combo
/// (`🔊✍️` — sound written into text) followed by the transcription text.
///
/// The audio file is a pure intermediate artifact — the caller deletes it
/// regardless of outcome (scoped to this message's inbound attachments), so the
/// returned string never contains the file name or path. On failure just the
/// icon combo is returned (no text).
async fn transcribe_audio_marker(path: &str) -> String {
    let path_buf = std::path::PathBuf::from(path);

    // ── Step 1: Try local Qwen3-ASR transcription ────────────────────
    // Default to enabled; only explicitly "false" disables local transcription.
    let use_local = crate::config::CONFIG
        .snapshot()
        .audio_transcription_use_local
        .as_deref()
        != Some("false");

    if use_local {
        match crate::audio::local_transcriber::transcribe_file_async(
            &path_buf,
            // 10-minute timeout for enrichment path — attached audio can be
            // arbitrarily long (voice memos, meeting recordings, etc.).
            crate::audio::local_transcriber::INFERENCE_TIMEOUT,
        )
        .await
        {
            Ok(text) => {
                tracing::debug!("Local audio transcription succeeded");
                let text = text.trim();
                return if text.is_empty() {
                    AUDIO_ICON.to_string()
                } else {
                    format!("{AUDIO_ICON} {text}")
                };
            }
            Err(e) => {
                tracing::warn!(error = %e, "Local audio transcription failed");
            }
        }
    }

    // ── Step 2: Icon-only fallback (no text, no filename) ────────────
    tracing::warn!("Audio transcription unavailable");
    AUDIO_ICON.to_string()
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
/// via tool calls. Returns `None` when no uploads dir is available or the copy
/// fails.
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
    let (mut dest, path) = match create_unique_upload(dir, name).await {
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

/// Per-message media-enrichment behavior, decided at the channel boundary for
/// the routed role. Image handling is unconditional (native data-URI parts for
/// every role) and video handling is unconditional too (workspace copy +
/// transcription for every role); only image compression is role-dependent.
#[derive(Debug, Clone)]
pub struct EnrichmentStrategy {
    /// Workspace uploads dir for saved full-resolution media copies (`None`
    /// disables copies).
    pub workspace_path: Option<std::path::PathBuf>,
    /// Downscale/compress inbound local images to a bounded JPEG before they
    /// enter the session — every role EXCEPT Assistant. Assistant passes
    /// through full-resolution byte-identical.
    pub compress_images: bool,
}

/// Outcome of processing an IMAGE marker.
enum ImageAction {
    /// Keep the marker unchanged (e.g. HTTP/HTTPS URL).
    Keep,
    /// Replace the marker with the given text, optionally including an
    /// upload-path annotation for agent tool references. `delete_temp` is set
    /// only when the source file was consumed from this message's inbound
    /// attachments (copied/read) — out-of-scope and missing files are never
    /// deleted.
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
/// never silently dropped downstream. When `compress` is set the primary encode
/// is a bounded JPEG; a compression failure (the only non-validity error left
/// after the classifier gate) falls back to the original bytes. Any over-cap
/// result is re-encoded to a bounded JPEG, and a degenerate over-cap result even
/// after that re-encode fails closed.
async fn bounded_image_data_uri(path: &std::path::Path, compress: bool) -> anyhow::Result<String> {
    let primary = if compress {
        match crate::util::local_image_to_compressed_data_uri(path).await {
            Ok(uri) => Ok(uri),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "Image compression failed — passing the original bytes through");
                crate::util::local_image_to_data_uri(path).await
            }
        }
    } else {
        crate::util::local_image_to_data_uri(path).await
    };
    match primary {
        Ok(uri) if uri.len() <= media_target::MAX_DATA_URI_ENCODED_BYTES => Ok(uri),
        Ok(_) => {
            // Over-cap: fall back to a bounded JPEG re-encode. On the compress
            // path the primary is the byte-identical fallback (a compress-succeeded
            // JPEG is always under cap), so re-running the compressor may recover
            // the image; if it is still over-cap it fails closed rather than forward
            // an over-cap payload.
            tracing::warn!(path = %path.display(), "image data-URI exceeds the encoded-payload cap — re-encoding to a bounded JPEG");
            let reencoded = crate::util::local_image_to_compressed_data_uri(path).await?;
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

/// Handle an IMAGE marker — convert to a data URI, an [`ImageAction::Invalid`]
/// target, or (for out-of-scope paths) a plain-text annotation. Saves a
/// workspace copy if `uploads_dir` is available. When `compress` is set the
/// data URI is a bounded-JPEG re-encode (non-Assistant roles); otherwise the
/// original bytes pass through byte-identical (Assistant). The returned
/// action's `delete_temp` tells the caller whether the source temp file was
/// consumed from this message's inbound attachments and may be cleaned up.
async fn handle_image(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    compress: bool,
    staging_dirs: &[String],
) -> ImageAction {
    // Only a well-formed http(s) URL is sent as-is; a malformed one falls
    // through to the classifier (which is equally strict) and stays inert text,
    // matching the provider's final image gate.
    if media_target::is_valid_remote_url(path) {
        return ImageAction::Keep;
    }

    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Image file not found for enrichment");
        return ImageAction::Invalid { delete_temp: false };
    }

    // Containment: only this message's own inbound attachment may be read,
    // copied, or deleted.
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        tracing::warn!(%path, "Image path outside this message's inbound attachments — annotating without copy");
        return ImageAction::Replace {
            replacement: format!("[Image: {} attached]", file_name_or_path(path)),
            upload_annotation: None,
            delete_temp: false,
        };
    }

    // Only a real raster may be converted. The file is in-scope (containment
    // passed), so it is a pure temp artifact and is cleaned up even though it is
    // never consumed as an image. The classifier requires a real decode for a
    // local file (authoritative, not a magic sniff), so it is offloaded off the
    // async worker (the cached result makes repeated calls cheap).
    if !matches!(
        crate::util::with_block_in_place(|| media_target::classify_media_image_target(path)),
        MediaTarget::LocalImage
    ) {
        return ImageAction::Invalid { delete_temp: true };
    }

    // Convert to data URI for the API request. The classifier above already
    // established the file is a decodable native raster, so a compression
    // failure is a bounded re-encode issue — not a validity one — and the
    // helper falls back to the original bytes (fail-open) rather than a second
    // decode. It is never a junk data-URI, because the authoritative classifier
    // gate already rejected any corrupt-but-magic-valid file.
    let data_uri = bounded_image_data_uri(path_obj, compress).await;

    // Save a workspace copy only once the image actually converts, so a dead /
    // corrupt file never leaves a junk upload copy plus a "[Saved image: ...]"
    // annotation alongside the invalid ref.
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

/// Canonical path of the daemon's Telegram attachment temp root, which the
/// receive path downloads every inbound attachment into. `None` when the
/// directory does not exist (then nothing can be in scope).
async fn canonical_telegram_files_root() -> Option<std::path::PathBuf> {
    tokio::fs::canonicalize(crate::util::telegram_files_root())
        .await
        .ok()
}

/// Whether `path` names an inbound attachment of the message that owns one of
/// `staging_dirs`: a file inside one of the message's own staging directories
/// under the Telegram temp root (the names recorded on the message by the
/// receive path, see `util::telegram_staging_dir_name`).
///
/// This is the only legitimate source of inbound media, and the narrower of the
/// two inbound containment predicates: [`has_inbound_temp_marker`] only widens
/// which text is replaced in chat history, while this one authorizes reading,
/// copying and deleting, so it demands the exact owning directory. A marker
/// naming anything else — another chat's attachment, another message's
/// attachment, or a hand-typed `[FILE:/etc/passwd]` — is inert text and must
/// never be read, copied into a workspace, or deleted. Off-chat messages
/// (gui/voice) carry no Telegram identity and so own nothing, and neither does a
/// message without an attachment.
async fn is_inbound_attachment(path: &std::path::Path, staging_dirs: &[String]) -> bool {
    if staging_dirs.is_empty() {
        return false;
    }
    let Some(root) = canonical_telegram_files_root().await else {
        return false;
    };
    // A path that exists must canonicalize inside the root on its own; the
    // parent fallback applies ONLY to a path that is not there at all: an
    // attachment deleted before enrichment cannot be canonicalized, and the
    // download path creates a per-message directory for the file it just wrote,
    // so a marker naming a since-vanished file inside such a directory is still
    // ours. A symlink planted in a per-message directory but pointing outside
    // the root exists, so it is never rescued by the fallback — its canonical
    // target is outside and the marker is left inert.
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
    // Exact match on the directory name: a prefix comparison would also accept a
    // sibling directory of the same chat.
    relative
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .is_some_and(|dir| staging_dirs.iter().any(|staging_dir| staging_dir == dir))
}

/// Whether `name` is a single plain path component — the only shape the receive
/// path generates for a staging directory (see
/// [`crate::util::telegram_staging_dir_name`]). [`ChannelMessage::attachment_dirs`]
/// is a public field, and an empty or relative name would make the removal below
/// target the shared Telegram root (or leave it) instead of a directory inside
/// it.
fn is_staging_dir_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(std::path::is_separator)
}

/// Outcome of processing a VIDEO marker: the replacement text, whether the
/// source temp file was copied into the workspace uploads (so the agent can
/// feed the clip to the video-edit flow) and can be cleaned up, and the
/// optional "[Video transcription of <name>]: <text>" annotation
/// (prepended as an annotation block by the caller). HTTP(S) URLs, missing
/// files, and out-of-scope paths degrade to plain annotations; in-scope inbound
/// attachments are always cleaned up (whether or not the workspace copy
/// succeeded).
struct VideoAction {
    replacement: String,
    delete_temp: bool,
    transcription: Option<String>,
}

impl VideoAction {
    /// Plain annotation: no workspace copy, no transcription.
    fn annotation(replacement: String) -> Self {
        Self {
            replacement,
            delete_temp: false,
            transcription: None,
        }
    }
}

async fn handle_video(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    workspace: &str,
    staging_dirs: &[String],
) -> VideoAction {
    if is_http_url(path) {
        return VideoAction::annotation(format!("[Video: {path}]"));
    }
    if !path_obj.exists() || !path_obj.is_file() {
        tracing::warn!(%path, "Video file not found for enrichment");
        return VideoAction::annotation(format!("[Invalid video reference: {path}]"));
    }
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        tracing::warn!(%path, "Video path outside this message's inbound attachments — annotating without copy");
        return VideoAction::annotation(format!("[Video: {} attached]", file_name_or_path(path)));
    }
    if let Some(saved) = save_media_to_workspace(path_obj, uploads_dir, "video", "mp4").await {
        // Transcribe the persistent workspace copy — never the temp file
        // (deleted after a successful copy). The annotation keeps the original
        // Telegram filename; fail-open: any failure degrades to the plain
        // [Saved video: ...] annotation.
        let transcription =
            transcribe_saved_video(&saved.dest, file_name_or_path(path), workspace).await;
        return VideoAction {
            replacement: saved.annotation,
            delete_temp: true,
            transcription,
        };
    }
    // Copy failed (or no uploads dir — e.g. a no-role message): annotate
    // without transcription, but the in-scope temp file is still a pure
    // intermediate artifact and is cleaned up. `is_inbound_attachment` was
    // verified above, so the delete stays inside the containment boundary.
    VideoAction {
        replacement: format!("[Video: {} attached]", file_name_or_path(path)),
        delete_temp: true,
        transcription: None,
    }
}

/// Transcribe a saved workspace video copy for the routed role, returning the
/// "[Video transcription of <name>]: <text>" annotation (using the original
/// source `file_name`). Fail-open: returns `None` (plain annotation) when the
/// transcription fails (unavailable transcriber, unsupported format, upload
/// or model error, timeout, empty output) — the overall timeout lives inside
/// [`transcribe_video_file`](crate::providers::transcribe_video_file),
/// bounding both callers. `workspace` names the workspace for telemetry and
/// the live non-agent call row (the inbound path has no agent card).
async fn transcribe_saved_video(
    path: &std::path::Path,
    file_name: &str,
    workspace: &str,
) -> Option<String> {
    let text = crate::providers::transcribe_video_file(path, Some(workspace)).await?;
    Some(format!("[Video transcription of {file_name}]: {text}"))
}

/// Handle a FILE marker — an inbound attachment the receive path saved under
/// the daemon's Telegram temp dir. Returns the marker's replacement text, or
/// `None` for an out-of-scope path: a user-typed `[FILE:...]` is the user's own
/// text, and another chat's or message's attachment must never be read, copied,
/// or deleted. An in-scope attachment is copied into the workspace uploads dir
/// under the sender's name and converted into text and images for the agent,
/// accumulating the workspace copy, the extracted text and the converter's
/// notes into `batch`; a missing attachment, an unrouted message and a failed
/// copy each degrade to a plain annotation rather than dropping the user's
/// message.
async fn handle_file(
    path: &str,
    path_obj: &std::path::Path,
    uploads_dir: Option<&std::path::Path>,
    compress_images: bool,
    staging_dirs: &[String],
    batch: &mut EnrichmentBatch,
) -> Option<String> {
    // Containment: only this message's own inbound attachment may be read,
    // copied, or deleted.
    if !is_inbound_attachment(path_obj, staging_dirs).await {
        // Also the vanished-staging-directory shape: the parent fallback needs
        // that directory to exist, so the path reads as hand-typed and its marker
        // stays verbatim.
        tracing::warn!(%path, "File path outside this message's inbound attachment — leaving the marker untouched");
        return None;
    }
    let name = file_name_or_path(path);
    // An inbound attachment is a pure intermediate: the workspace copy (or the
    // extracted text) is what outlives this call, so a real file is queued for
    // deletion; a marker naming the staging directory itself is never removed.
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
        // No-role message (broadcast but never routed): there is no workspace
        // to copy into and no agent to convert for.
        return Some(format!(
            "[File {name}: received, not saved to the workspace]"
        ));
    };

    // Copy first — the workspace handle is what the agent reads later, and the
    // conversion then runs off a file that outlives the temp cleanup.
    let copy = copy_to_uploads(path_obj, uploads_dir, name).await;
    let copied_to_workspace = copy.is_some();
    let replacement = match &copy {
        Some(dest) => format!("[FILE:{}]", dest.display()),
        None => format!("[File {name}: received, not saved to the workspace]"),
    };

    match convert_inbound(path_obj, name).await {
        DocOutcome::Text {
            text,
            images,
            notes,
        } => {
            batch.annotations.push(
                extracted_text_annotation(uploads_dir, name, &text, !images.is_empty()).await,
            );
            for note in notes {
                batch.annotations.push(format!("[File {name}: {note}]"));
            }
            // Extracted pages go through the ordinary IMAGE pipeline: a native
            // data-URI part plus a workspace copy, so the model sees them and the
            // agent can reopen them. A page the pipeline rejects is counted into
            // one note, like the converter's own skipped-media note.
            let mut unreadable_pages = 0usize;
            for image in images {
                let image_path = image.to_string_lossy().to_string();
                match handle_image(
                    &image_path,
                    &image,
                    Some(uploads_dir),
                    compress_images,
                    staging_dirs,
                )
                .await
                {
                    ImageAction::Keep => {}
                    ImageAction::Invalid { delete_temp } => {
                        // The page is a temp artifact written by this pass into
                        // the attachment's per-message directory, so the note
                        // must not name it — the file is deleted right below.
                        unreadable_pages += 1;
                        if delete_temp {
                            batch.files_to_delete.push(image);
                        }
                    }
                    ImageAction::Replace {
                        replacement,
                        upload_annotation,
                        delete_temp,
                    } => {
                        batch.appended_markers.push(replacement);
                        if let Some(annotation) = upload_annotation {
                            batch.upload_annotations.push(annotation);
                        }
                        if delete_temp {
                            batch.files_to_delete.push(image);
                        }
                    }
                }
            }
            if unreadable_pages > 0 {
                batch.annotations.push(format!(
                    "[File {name}: {unreadable_pages} extracted page image(s) could not be read]"
                ));
            }
        }
        DocOutcome::Unreadable { reason } => {
            batch.annotations.push(format!("[File {name}: {reason}]"));
        }
        DocOutcome::Unsupported => {
            // The copy is what the agent would open, so only claim it is
            // openable when it actually landed in the workspace.
            let saved = if copied_to_workspace {
                " — open the saved file directly"
            } else {
                ""
            };
            batch.annotations.push(format!(
                "[File {name}: received but not converted (unsupported format){saved}]"
            ));
        }
    }
    Some(replacement)
}

/// Create a NEW empty file `dir`/`file_name`, disambiguating with a `_<n>`
/// counting suffix (see [`crate::util::suffixed_name`]) when a file of that name
/// is already there, and return the open handle with its path.
///
/// The `create_new` open *is* the collision check: two chats frequently send a
/// file under the same name, and a check-then-act probe would let two concurrent
/// enrichments pick the same free name and overwrite each other.
async fn create_unique_upload(
    dir: &std::path::Path,
    file_name: &str,
) -> std::io::Result<(tokio::fs::File, std::path::PathBuf)> {
    let mut counter = 1u32;
    loop {
        let candidate = dir.join(crate::util::suffixed_name(file_name, counter));
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
            .await
        {
            Ok(file) => return Ok((file, candidate)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => counter += 1,
            Err(e) => return Err(e),
        }
    }
}

/// Convert an inbound attachment into text and images, off the async worker.
///
/// `pdf-extract` panics on malformed input, so the conversion runs on a
/// blocking thread and its panic is contained at that boundary: a
/// [`tokio::task::JoinError`] degrades to an unreadable document instead of
/// taking down the enrichment (and the user's message) with it. `out_dir` is
/// the attachment's own per-message directory, so extracted pages stay inside
/// the directory the inbound IMAGE containment accepts.
async fn convert_inbound(path_obj: &std::path::Path, file_name: &str) -> DocOutcome {
    // Held across the read and the conversion, covering the attachment bytes and
    // the conversion's rasters (encoding the extracted pages into data URIs
    // happens later, outside this bound). The semaphore is never closed, so
    // acquisition cannot fail.
    let _permit = DOCUMENT_CONVERSIONS.acquire().await;
    let bytes = match tokio::fs::read(path_obj).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!(
                path = %path_obj.display(),
                error = %e,
                "Failed to read the inbound attachment"
            );
            return DocOutcome::Unreadable {
                reason: UNREADABLE_REASON.to_string(),
            };
        }
    };
    // A file (checked by the caller) always has a parent directory.
    let out_dir = path_obj.parent().unwrap_or(path_obj).to_path_buf();
    let name = file_name.to_string();
    match tokio::task::spawn_blocking(move || convert_document(&bytes, &name, &out_dir)).await {
        Ok(outcome) => outcome,
        Err(e) => {
            tracing::warn!(error = %e, "Document conversion failed");
            DocOutcome::Unreadable {
                reason: UNREADABLE_REASON.to_string(),
            }
        }
    }
}

/// Annotation block for a document's extracted text.
///
/// Short text is inlined; longer text is spilled to
/// `<uploads>/<stem>.extracted.txt` and the annotation points at that file, so
/// a book-sized document cannot blow up the message. Empty text (a scanned,
/// image-only document) names the extracted pages instead.
async fn extracted_text_annotation(
    uploads_dir: &std::path::Path,
    name: &str,
    text: &str,
    has_images: bool,
) -> String {
    let text = text.trim();
    if text.is_empty() {
        return if has_images {
            format!("[File {name}: no text could be extracted — the pages were provided as images]")
        } else {
            format!("[File {name}: no text could be extracted]")
        };
    }
    let char_count = text.chars().count();
    if char_count <= INLINE_TEXT_MAX_CHARS {
        return format!("[File {name}: extracted text follows]\n\n{text}");
    }
    let spilled = async {
        let spill_name = format!("{}.extracted.txt", crate::util::name_stem(name));
        let (mut file, path) = create_unique_upload(uploads_dir, &spill_name).await?;
        match tokio::io::AsyncWriteExt::write_all(&mut file, text.as_bytes()).await {
            Ok(()) => Ok::<_, std::io::Error>(path),
            Err(e) => {
                // The file was created before the write: a partial spill is not
                // a workspace handle, so it goes rather than stay behind the
                // note saying nothing was saved.
                drop(file);
                let _ = tokio::fs::remove_file(&path).await;
                Err(e)
            }
        }
    }
    .await;
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
        "[File {name}: the extracted text is too long to inline ({char_count} characters); the full text was saved to {} — read that file.]",
        spill.display()
    )
}

/// Accumulators shared by the per-kind marker handlers in [`enrich_message`]:
/// what the finished message gets prepended/appended, and which temp artifacts
/// the pass consumed.
#[derive(Default)]
struct EnrichmentBatch {
    /// Annotation blocks prepended to the finished message.
    annotations: Vec<String>,
    /// `[Saved ...]` upload annotations appended to the body.
    upload_annotations: Vec<String>,
    /// `[IMAGE:data:...]` parts a FILE ingestion extracted, appended to the
    /// body after the marker loop (`captures_iter` already snapshotted the
    /// source content, so the loop never visits them).
    appended_markers: Vec<String>,
    /// Temp files to remove — only ever queued for this message's own inbound
    /// attachments, so user-typed markers can never delete arbitrary local
    /// files. The message's staging directories are removed afterwards.
    files_to_delete: Vec<std::path::PathBuf>,
}

impl EnrichmentBatch {
    /// Delete the queued temp artifacts and the message's staging directories,
    /// then assemble the enriched content: append the collected native parts,
    /// strip the handled markers and prepend the annotations.
    async fn finish(self, mut body: String, staging_dirs: &[String]) -> String {
        // Deletion errors are logged (not silently discarded); a file that is
        // already gone is not an error.
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

        // `remove_dir` removes only an empty directory, so a staging directory
        // still holding an unconsumed artifact is left alone until the OS
        // reclaims the temp root; any failure (a directory that is already gone)
        // is not an error.
        let root = crate::util::telegram_files_root();
        for staging_dir in staging_dirs {
            if !is_staging_dir_name(staging_dir) {
                continue;
            }
            let _ = tokio::fs::remove_dir(root.join(staging_dir)).await;
        }

        // Append upload path annotations so the model can reference saved
        // files, preceded by the `[IMAGE:data:...]` parts a document produced.
        if !self.appended_markers.is_empty() {
            let _ = write!(body, "\n\n{}", self.appended_markers.join("\n"));
        }
        if !self.upload_annotations.is_empty() {
            let annotation_block = self.upload_annotations.join("\n");
            let _ = write!(body, "\n\n{annotation_block}");
        }

        // ── Marker stripping and annotation prepending ──
        // Strip only the kinds this pass consumed (AUDIO, VIDEO): IMAGE carries
        // the native parts the routed role's model consumes, FILE is the
        // workspace handle the agent opens itself, and anything else is the
        // user's own words.
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

        // Prepend the accumulated text descriptions: transcribed AUDIO content,
        // VIDEO transcription annotations, and a FILE's extracted document
        // text. Prepending (rather than inlining at the marker) is what keeps
        // marker-shaped text inside a document away from the strip pass above.
        if self.annotations.is_empty() {
            return cleaned;
        }
        let prefix = self.annotations.join("\n");
        if cleaned.is_empty() {
            prefix
        } else {
            format!("{prefix}\n\n{cleaned}")
        }
    }
}

/// Process all media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`,
/// `[FILE:...]`) in a single pass. Each marker kind is handled according to the
/// strategy:
///
/// | Kind | Behavior |
/// |------|----------|
/// | IMAGE | data URI conversion (byte-identical for Assistant, bounded-JPEG compression for every other role) + workspace copy when in scope |
/// | AUDIO | transcription (unchanged for all roles) |
/// | VIDEO | workspace copy + `[Saved video: path]` + transcription (every role) |
/// | FILE | workspace copy under the sender's name + extracted text and pages (every role); out-of-scope markers stay verbatim |
///
/// After processing, the markers this pass consumed (AUDIO, VIDEO) are stripped
/// from the content, every other kind is preserved (IMAGE is the native image
/// parts and FILE the workspace handle the routed role's model consumes), and
/// annotations are prepended. Temp files are cleaned up after processing,
/// scoped to this message's own per-message directories — out-of-scope marker
/// paths are never read, copied, or deleted: IMAGE/AUDIO/VIDEO degrade to plain
/// annotations and an out-of-scope `[FILE:...]` is left verbatim.
// Marker dispatch hub (4 kinds, per-kind handling is extracted into the
// handler functions above, keeping this loop flat on purpose).
pub async fn enrich_message(msg: &mut ChannelMessage, strategy: &EnrichmentStrategy) {
    let mut batch = EnrichmentBatch::default();
    let mut result = msg.content.clone();
    let uploads_dir = strategy.workspace_path.as_ref().map(|p| p.join("uploads"));
    // The staging directories the receive path created for this message; empty
    // for an off-chat (gui/voice) message, which owns no attachments.
    let staging_dirs = msg.attachment_dirs.clone();

    for caps in MEDIA_MARKER_RE.captures_iter(&msg.content) {
        let whole = caps.get_match();
        let (kind, path) = parse_media_marker(&caps);
        let path_obj = std::path::Path::new(path);

        match kind {
            MediaMarkerKind::File => {
                // `None` = an out-of-scope user-typed marker, left verbatim:
                // the strip pass preserves FILE markers, so it survives
                // untouched.
                if let Some(replacement) = handle_file(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    strategy.compress_images,
                    &staging_dirs,
                    &mut batch,
                )
                .await
                {
                    result = result.replacen(whole.as_str(), &replacement, 1);
                }
            }
            MediaMarkerKind::Image => {
                let action = handle_image(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    strategy.compress_images,
                    &staging_dirs,
                )
                .await;
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
                // Local IMAGE temp files are cleaned up only when consumed from
                // this message's inbound attachments (delete_temp).
                if delete_temp {
                    batch.files_to_delete.push(path_obj.to_path_buf());
                }
            }
            MediaMarkerKind::Audio => {
                // Containment: only this message's own inbound attachment is
                // transcribed or deleted; out-of-scope markers and a marker
                // naming a directory degrade to the icon only.
                if path_obj.is_file() && is_inbound_attachment(path_obj, &staging_dirs).await {
                    batch.annotations.push(transcribe_audio_marker(path).await);
                    batch.files_to_delete.push(path_obj.to_path_buf());
                } else {
                    tracing::warn!(%path, "Audio marker is not an inbound attachment file — annotating without transcription");
                    batch.annotations.push(AUDIO_ICON.to_string());
                }
            }
            MediaMarkerKind::Video => {
                let VideoAction {
                    replacement,
                    delete_temp,
                    transcription,
                } = handle_video(
                    path,
                    path_obj,
                    uploads_dir.as_deref(),
                    &msg.workspace,
                    &staging_dirs,
                )
                .await;
                result = result.replacen(whole.as_str(), &replacement, 1);
                if let Some(annotation) = transcription {
                    batch.annotations.push(annotation);
                }
                if delete_temp {
                    batch.files_to_delete.push(path_obj.to_path_buf());
                }
            }
        }
    }

    msg.content = batch.finish(result, &staging_dirs).await;
}

/// Whether `content` carries a media marker naming an inbound Telegram
/// attachment — a path under the daemon's Telegram temp root.
///
/// Used by [`crate::channels::persist_content`] to decide what reaches chat
/// history: those markers name temp files the daemon deletes right after
/// enrichment, so the enriched content (transcription, workspace path, image
/// placeholder) is persisted instead of the original text.
///
/// Purely syntactic and deliberately looser than [`is_inbound_attachment`]: a
/// hand-typed marker naming that root also qualifies, which is safe because
/// this only decides which text is replaced in history — it never reads, copies
/// or deletes anything.
#[must_use]
pub(crate) fn has_inbound_temp_marker(content: &str) -> bool {
    let root = crate::util::telegram_files_root();
    MEDIA_MARKER_RE.captures_iter(content).any(|caps| {
        let (_, path) = parse_media_marker(&caps);
        std::path::Path::new(path).starts_with(&root)
    })
}

/// Extract all unique URLs from message text.
///
/// Strips common trailing punctuation (commas, periods, closing brackets,
/// colons, semicolons, exclamation/question marks) that naturally appears
/// around URLs in prose.
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

/// Enrich a message by prepending link summaries for any URLs found in the text.
///
/// If no URLs are found, the original message is returned unchanged.
/// Links are fetched concurrently using the shared `ChromeTool` — each URL
/// gets its own isolated session tab that is closed after text extraction.
pub async fn enrich_links(content: &str) -> Cow<'_, str> {
    // Truncate very long snippets to keep messages manageable.
    const MAX_TEXT_LEN: usize = 5000;
    let urls = extract_urls(content);
    if urls.is_empty() {
        return Cow::Borrowed(content);
    }

    // Gate on the cached (non-probing) daemon advertisement first — the cheap
    // in-memory check short-circuits the `--version` spawn below while the
    // daemon is confirmed down. A stale/unknown state passes optimistically
    // and the concurrent fetch tasks re-discover liveness (bounded by the
    // probe timeout) without failing the message.
    if !(crate::tools::chrome_daemon::is_advertised()
        && matches!(
            crate::tools::chrome_daemon::cli_probe().await,
            crate::tools::chrome_daemon::CliStatus::Available
        ))
    {
        tracing::debug!("chrome-use not available, skipping link enrichment");
        return Cow::Borrowed(content);
    }

    // Fetch all URLs concurrently.
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
            // Close the tab (best-effort) regardless of fetch outcome.
            chrome.close_session(&tab).await;
            (url, result)
        }));
    }

    let mut enrichments: Vec<String> = Vec::new();
    for task in tasks {
        match task.await {
            Ok((url, Ok(body_text))) => {
                if body_text.trim().is_empty() {
                    // Blank/empty page — don't insert an empty snippet.
                    tracing::debug!(url, "Link enricher: page text is empty, skipping snippet");
                    continue;
                }
                let snippet = if body_text.len() > MAX_TEXT_LEN {
                    format!("{}…", crate::util::truncate_bytes(&body_text, MAX_TEXT_LEN))
                } else {
                    body_text
                };
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

    #[test]
    fn extract_urls_finds_http_and_https() {
        let urls = extract_urls("Check https://example.com and http://test.org/page for info");
        assert_eq!(urls, vec!["https://example.com", "http://test.org/page"]);
    }

    #[test]
    fn extract_urls_deduplicates() {
        let urls = extract_urls("Visit https://example.com and https://example.com again");
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn extract_urls_strips_trailing_punctuation() {
        let urls = extract_urls("See https://example.com, and https://test.org.");
        assert_eq!(urls, vec!["https://example.com", "https://test.org"]);
    }

    #[test]
    fn extract_urls_handles_urls_in_parens() {
        let urls = extract_urls("(https://example.com) and [https://test.org]");
        assert_eq!(urls, vec!["https://example.com", "https://test.org"]);
    }

    #[tokio::test]
    async fn enrich_links_returns_borrowed_when_no_urls() {
        let content = "Hello, this is a plain message without any URLs.";
        let result = enrich_links(content).await;
        // No URLs → should borrow the input, not allocate a new String.
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.as_ref(), content);
    }

    // ── Enrichment strategy tests ─────────────────────────────────────

    /// The chat id every inbound fixture belongs to: the test messages carry it
    /// and the fixture directories are named after it.
    const TEST_CHAT_ID: &str = "1";

    /// Helper: quick ChannelMessage for enrichment tests. It carries
    /// [`TEST_CHAT_ID`] but no message id, so it owns no inbound attachment —
    /// containment is per message, and any inbound fixture is therefore out of
    /// scope for it.
    fn test_msg(content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: "test".into(),
            reply_target: "test".into(),
            content: content.to_string(),
            channel: "test".into(),
            workspace: "test".into(),
            optimistic_id: None,
            callback_query_id: None,
            reply_reference: None,
            chat_id: Some(TEST_CHAT_ID.into()),
            message_id: None,
            attachment_dirs: Vec::new(),
        }
    }

    /// Like [`test_msg`], but owning the staging directories of `message_ids` —
    /// one for a single message, several for the album shape (one merged message
    /// carrying every member's directory). Each entry makes an
    /// [`inbound_attachment_fixture`] of the same id ingestible, and
    /// `message_id` stays the first entry, as an album merge keeps the first
    /// member's id for addressing.
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

    /// A message owning the inbound directory of a single `message_id`.
    fn inbound_msg(message_id: i64, content: &str) -> ChannelMessage {
        inbound_album_msg(&[message_id], content)
    }

    /// The daemon's Telegram attachment temp root.
    fn telegram_root() -> std::path::PathBuf {
        crate::util::telegram_files_root()
    }

    /// Create the daemon's Telegram temp dir. The containment root must exist
    /// before path canonicalization — a missing root makes every path look
    /// out of scope.
    async fn ensure_telegram_files_dir() {
        tokio::fs::create_dir_all(telegram_root()).await.unwrap();
    }

    /// Set up an out-of-scope fixture: a unique scratch dir under the system
    /// temp dir (outside the Telegram temp dir) containing a fake workspace
    /// (`ws_path`) and a single arbitrary media file. Returns
    /// `(tmp_root, ws_path, arbitrary_file)`.
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

    /// Set up an inbound attachment exactly as the receive path delivers one:
    /// the file inside its own staging subdirectory of the Telegram temp root
    /// (see [`crate::util::telegram_staging_dir_name`]). Each test uses its own
    /// message id, which is what the per-message containment compares against.
    /// Returns `(msg_dir, attachment)`.
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

    /// An inbound attachment of [`TEST_CHAT_ID`]: in scope for
    /// [`inbound_msg`] with the same message id.
    async fn inbound_attachment_fixture(
        message_id: i64,
        file_name: &str,
        bytes: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        telegram_attachment_fixture(TEST_CHAT_ID, message_id, file_name, bytes).await
    }

    /// Set up an inbound attachment of [`TEST_CHAT_ID`] plus a workspace to
    /// ingest it into. Returns `(tmp_root, ws_path, msg_dir, attachment)`.
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

    /// Extract the image payload embedded in a `[IMAGE:...]` data URI inside
    /// `content` and decode it, returning the decoded image dimensions.
    fn embedded_image_dimensions(content: &str) -> (u32, u32) {
        use image::GenericImageView;
        let data_uri = content
            .split("[IMAGE:")
            .nth(1)
            .expect("data URI marker must be present")
            .split(']')
            .next()
            .expect("data URI marker must be closed");
        let b64 = data_uri
            .split(',')
            .nth(1)
            .expect("data URI must carry a base64 payload");
        let bytes = STANDARD.decode(b64).expect("data URI base64 must decode");
        let img = image::load_from_memory(&bytes).expect("embedded image must decode");
        img.dimensions()
    }

    #[tokio::test]
    async fn enrich_image_http_url_passthrough() {
        let mut msg = test_msg("Check this [IMAGE:https://example.com/img.png] out");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(
            msg.content,
            "Check this [IMAGE:https://example.com/img.png] out"
        );
    }

    #[tokio::test]
    async fn enrich_image_file_not_found() {
        let mut msg = test_msg("Here is [IMAGE:/tmp/nonexistent_xyz_img.png] an image");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        assert!(
            msg.content
                .contains("[Invalid image reference: /tmp/nonexistent_xyz_img.png]")
        );
    }

    #[tokio::test]
    async fn enrich_audio_annotation_and_strip() {
        let mut msg = test_msg("Listen [AUDIO:/tmp/audio_xyz.mp3] to this");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        // AUDIO marker stripped; annotation prepended (icon-only fallback since
        // no audio transcriber is configured in the test environment)
        assert!(
            msg.content.contains("🔊✍️"),
            "Audio annotation must be present, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
        // No file name may survive in any form
        assert!(
            !msg.content.contains("audio_xyz"),
            "Audio temp file name must not appear, got: {}",
            msg.content
        );
        // The original text is preserved
        assert!(msg.content.contains("Listen"), "Original text preserved");
        assert!(msg.content.contains("to this"), "Original text preserved");
    }

    #[tokio::test]
    async fn enrich_image_valid_file_converts_to_data_uri_and_deletes_temp() {
        // The fixture must be one of this message's inbound attachments to be in
        // scope for reading (data URI) and cleanup.
        let source_bytes = real_png(2, 1);
        let (msg_dir, tmp) = inbound_attachment_fixture(7001, "photo.png", &source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7001, &format!("Image: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Marker replaced with data URI
        assert!(
            msg.content.contains("[IMAGE:data:image/png;base64,"),
            "Expected data URI, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains(&path_str),
            "Raw file path must not remain in content"
        );
        // Temp file and its per-message directory deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
        assert!(!msg_dir.exists(), "Per-message directory must be deleted");
    }

    #[tokio::test]
    async fn enrich_image_with_workspace_creates_upload_annotation() {
        let tmp_root = crate::util::test::test_root().join("test_enrich_ws");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // The fixture must be one of this message's inbound attachments to be in
        // scope for reading (data URI) and cleanup.
        let source_bytes = real_png(2, 1);
        let (_msg_dir, tmp_img) =
            inbound_attachment_fixture(7002, "photo.png", &source_bytes).await;
        let img_path_str = tmp_img.to_string_lossy().to_string();

        let mut msg = inbound_msg(7002, &format!("Image: [IMAGE:{img_path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Data URI present and upload annotation added
        assert!(msg.content.contains("[IMAGE:data:image/png;base64,"));
        assert!(
            msg.content.contains("[Saved image:"),
            "Upload annotation must be present, got: {}",
            msg.content
        );
        // Temp file deleted
        assert!(
            !tmp_img.exists(),
            "Temp file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_with_workspace_copies_and_annotates() {
        let tmp_root = crate::util::test::test_root().join("test_enrich_video_ws");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        // Only one of this message's own inbound attachments is eligible for
        // copy.
        let mp4_header: &[u8] = &[
            0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D, 0x00, 0x00,
            0x00, 0x00, 0x69, 0x73, 0x6F, 0x6D, 0x69, 0x73, 0x6F, 0x32,
        ];
        let (_msg_dir, tmp_video) = inbound_attachment_fixture(7003, "clip.mp4", mp4_header).await;
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = inbound_msg(7003, &format!("Edit this clip: [VIDEO:{video_path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Marker replaced with a [Saved video: ...] annotation pointing at the
        // workspace uploads copy so the agent can feed it to video_edit.
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
        // Temp file deleted after the workspace copy
        assert!(
            !tmp_video.exists(),
            "Temp video file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_file(&tmp_video).await;
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_image_outside_inbound_attachments_annotates_without_read_or_copy() {
        // An injected marker pointing at an arbitrary readable file must not
        // be read into model context (data URI), copied into uploads, or
        // deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_img_outside",
            "secret.png",
            b"top secret image bytes",
        )
        .await;
        let marker = format!("Look at [IMAGE:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Image: secret.png attached]"),
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
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_video_http_url_kept_as_plain_text() {
        let mut msg = test_msg("Edit [VIDEO:https://example.com/clip.mp4] this");
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;
        // HTTP URL video reference is preserved as plain text (no marker strip)
        assert!(
            msg.content
                .contains("[Video: https://example.com/clip.mp4]"),
            "HTTP video URL must be kept as plain-text reference, got: {}",
            msg.content
        );
        assert!(!msg.content.contains("[VIDEO:"));
    }

    #[tokio::test]
    async fn enrich_video_without_workspace_deletes_in_scope_temp() {
        // No workspace path (e.g. a no-role user's message): the video gets a
        // plain annotation, no transcription, but the in-scope inbound
        // attachment is still a pure intermediate artifact and must be cleaned
        // up — a regression guard for the no-role gating in main.rs.
        let (_msg_dir, tmp_video) = inbound_attachment_fixture(7004, "clip.mp4", b"fake mp4").await;
        let video_path_str = tmp_video.to_string_lossy().to_string();

        let mut msg = inbound_msg(7004, &format!("Watch [VIDEO:{video_path_str}] this clip"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains(&format!(
                "[Video: {} attached]",
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

    #[tokio::test]
    async fn enrich_video_outside_inbound_attachments_annotates_without_copy_or_delete() {
        // An injected marker pointing at an arbitrary readable file must not
        // be copied into uploads (exfiltration vector) or deleted.
        let (tmp_root, ws_path, arbitrary) = out_of_scope_fixture(
            "test_enrich_video_nonmm",
            "secret.mp4",
            b"top secret video bytes",
        )
        .await;
        let marker = format!("Watch [VIDEO:{}]", arbitrary.display());

        let mut msg = test_msg(&marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Video: secret.mp4 attached]"),
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
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_non_assistant_image_compressed_to_jpeg_data_uri() {
        // Real 1100x800 PNG: the longest side exceeds the 1024 px cap, so the
        // ingestion-time re-encode must downscale it to a bounded JPEG while
        // the workspace copy stays the full-resolution original.
        let source_bytes = real_png(1100, 800);
        let (_msg_dir, tmp) = inbound_attachment_fixture(7005, "photo.png", &source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();

        let tmp_root = crate::util::test::test_root().join("test_enrich_compress_ws");
        let ws_path = tmp_root.join("myworkspace");
        tokio::fs::create_dir_all(&ws_path).await.unwrap();

        let mut msg = inbound_msg(7005, &format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[IMAGE:data:image/jpeg;base64,"),
            "Compressed JPEG data URI expected, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("data:image/png"),
            "Original PNG data URI must not appear, got: {}",
            msg.content
        );
        let (dw, dh) = embedded_image_dimensions(&msg.content);
        assert!(
            dw.max(dh) <= crate::util::INBOUND_IMAGE_MAX_SIDE,
            "Compressed image longest side {} must be ≤ 1024",
            dw.max(dh)
        );
        // The workspace copy is the full-resolution original, byte-identical.
        let uploads_dir = ws_path.join("uploads");
        let mut entries = tokio::fs::read_dir(&uploads_dir).await.unwrap();
        let entry = entries
            .next_entry()
            .await
            .unwrap()
            .expect("one upload copy");
        let copy_bytes = tokio::fs::read(entry.path()).await.unwrap();
        assert_eq!(
            copy_bytes, source_bytes,
            "Workspace copy must be byte-identical to the source PNG"
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    #[tokio::test]
    async fn enrich_assistant_image_byte_identical_data_uri() {
        let source_bytes = real_png(64, 48);
        let (_msg_dir, tmp) = inbound_attachment_fixture(7006, "art.png", &source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7006, &format!("Art: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        let expected = format!(
            "[IMAGE:data:image/png;base64,{}]",
            STANDARD.encode(&source_bytes)
        );
        assert!(
            msg.content.contains(&expected),
            "Assistant data URI must be byte-identical to the source, got: {}",
            msg.content
        );
        // Temp file deleted
        assert!(
            !tmp.exists(),
            "Temp image file must be deleted after enrichment"
        );
    }

    #[tokio::test]
    async fn enrich_image_in_scope_non_raster_is_rejected_not_junk_data_uri() {
        // A non-image file with an image extension inside this message's inbound
        // attachment directory must NOT fail open to a junk data URI (the
        // fail-open bug). The classifier rejects it, the marker becomes an
        // invalid reference, and the in-scope temp file is cleaned up (it is a
        // pure intermediate).
        let source_bytes = b"not an image";
        let (_msg_dir, tmp) = inbound_attachment_fixture(7007, "photo.png", source_bytes).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7007, &format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // In-scope temp file is a pure intermediate: cleaned up even though it
        // was never consumed as an image (it would otherwise accumulate).
        assert!(
            !tmp.exists(),
            "In-scope non-raster temp file must be cleaned up"
        );
        // Defensive cleanup in case the assertion above fails.
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_image_corrupt_raster_does_not_fail_open_to_junk_data_uri() {
        // A file whose leading bytes sniff as PNG (valid magic + IHDR) but whose
        // payload is truncated passes the structural classifier gate but is NOT
        // decodable. Compression fails, and the fail-open fallback must NOT
        // base64-encode the corrupt bytes into a junk data-URI — it degrades to
        // an invalid reference instead.
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let truncated = &buf[..buf.len().min(24)];
        let (_msg_dir, tmp) = inbound_attachment_fixture(7008, "photo.png", truncated).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7008, &format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: true,
        };
        enrich_message(&mut msg, &strategy).await;

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
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_image_corrupt_raster_byte_identical_does_not_fail_open() {
        // The Assistant (compress=false) path sends the original bytes untouched;
        // a corrupt-but-magic-valid file that passed the structural gate must not
        // be base64-encoded into a junk data URI — it degrades to an invalid
        // reference instead.
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let truncated = &buf[..buf.len().min(24)];
        let (_msg_dir, tmp) = inbound_attachment_fixture(7009, "photo.png", truncated).await;
        let path_str = tmp.to_string_lossy().to_string();

        let mut msg = inbound_msg(7009, &format!("Photo: [IMAGE:{path_str}]"));
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content
                .contains(&format!("[Invalid image reference: {path_str}]")),
            "Corrupt-but-magic-valid in-scope file (byte-identical) must degrade to an invalid reference, got: {}",
            msg.content
        );
        assert!(
            !msg.content.contains("[IMAGE:data:"),
            "No junk data URI may be produced for a corrupt file, got: {}",
            msg.content
        );
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
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // Temp file must be deleted even when transcription fails — the audio
        // file is a pure intermediate artifact (no transcriber in tests).
        assert!(
            !tmp.exists(),
            "Audio temp file must be deleted on transcription failure"
        );
        // Defensive cleanup in case the assertion above fails.
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn enrich_combined_image_preserved_audio_annotated() {
        let msg_content = "Here [IMAGE:https://example.com/img.png] and [AUDIO:/tmp/sound_xyz.mp3]";
        let mut msg = test_msg(msg_content);
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        // IMAGE http URL kept
        assert!(
            msg.content.contains("[IMAGE:https://example.com/img.png]"),
            "IMAGE with http URL must be preserved, got: {}",
            msg.content
        );
        // AUDIO marker stripped, annotation present
        assert!(
            msg.content.contains("🔊✍️"),
            "Audio annotation must be present"
        );
        assert!(
            !msg.content.contains("[AUDIO:"),
            "AUDIO marker must be stripped"
        );
    }

    async fn assert_no_markers_unchanged(strategy: EnrichmentStrategy, content: &str) {
        let mut msg = test_msg(content);
        let original = msg.content.clone();
        enrich_message(&mut msg, &strategy).await;
        assert_eq!(msg.content, original, "No markers = no changes");
    }

    #[tokio::test]
    async fn enrich_no_annotations_when_no_markers() {
        assert_no_markers_unchanged(
            EnrichmentStrategy {
                workspace_path: None,
                compress_images: false,
            },
            "Just a plain message with no markers",
        )
        .await;
    }

    // ── FILE marker tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn enrich_file_inlines_text_and_copies_to_uploads() {
        let document = "# Title\n\nHello from a document";
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_file_md", 7011, "notes.md", document.as_bytes())
                .await;
        let marker = format!("Read [FILE:{}] please", attachment.display());

        let mut msg = inbound_msg(7011, &marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };

        for attachment in [
            &other_attachment,
            &longer_id_attachment,
            &sibling_attachment,
        ] {
            let marker = format!("Look at [FILE:{}]", attachment.display());
            let mut msg = inbound_msg(7014, &marker);
            enrich_message(&mut msg, &strategy).await;
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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }

    /// The staging-directory cleanup is per message, not per marker kind: an
    /// image attachment that is gone by enrichment still leaves no directory
    /// behind.
    #[tokio::test]
    async fn enrich_vanished_image_attachment_leaves_no_staging_dir() {
        let (tmp_root, ws_path, msg_dir, attachment) =
            inbound_ingest_fixture("test_enrich_image_vanished", 7022, "gone.png", b"png").await;
        tokio::fs::remove_file(&attachment).await.unwrap();
        let marker = format!("Look at [IMAGE:{}]", attachment.display());

        let mut msg = inbound_msg(7022, &marker);
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

        assert!(
            msg.content.contains("[Invalid image reference:"),
            "A vanished image must be annotated, got: {}",
            msg.content
        );
        assert!(
            !msg_dir.exists(),
            "The empty staging directory must be removed"
        );
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: None,
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
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
        let strategy = EnrichmentStrategy {
            workspace_path: Some(ws_path.clone()),
            compress_images: false,
        };
        enrich_message(&mut msg, &strategy).await;

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
        // Cleanup
        let _ = tokio::fs::remove_dir_all(&tmp_root).await;
    }
}

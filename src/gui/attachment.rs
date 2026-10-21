//! Files the desktop chat composer attaches to a message.
//!
//! A send stages them the way the Telegram receive path stages inbound media —
//! bytes copied into one directory under [`crate::util::telegram_files_root`] —
//! so the message carries the same `[IMAGE:…]` / `[FILE:…]` markers and
//! `attachment_dirs` entry, and `channels::enrich_message` hands it to the agent
//! with no channel-side special case.

use std::path::{Path, PathBuf};

use iced::widget::{container, image, row, text};
use iced::{Alignment, Element, Length};

use super::theme;

/// Side of the chip's image thumbnail or document glyph, in px.
const CHIP_THUMB: f32 = 28.0;

/// Longest attachment name a chip shows, ellipsis included.
const CHIP_NAME_MAX_CHARS: usize = 24;

// ── Attachment ──────────────────────────────────────────────────────

/// Why a candidate path did not become an [`Attachment`].
#[derive(Debug)]
pub(crate) enum Refusal {
    /// Not a file the user offered — a folder (dropping one is out of scope), a
    /// device, or a name that is not UTF-8. Nothing to report.
    Ignored,
    /// Past the product's size cap; carries the warning to show.
    TooLarge(String),
}

/// One file the composer holds for the next send, before anything is written.
#[derive(Debug, Clone)]
pub struct Attachment {
    /// Display name — the source file's name, or the one a pasted image was
    /// given.
    pub name: String,
    /// The bytes to deliver, read only when the message is sent.
    pub source: Source,
    /// Deliver as `[IMAGE:…]` (a raster the receive path would treat as an
    /// image) rather than `[FILE:…]`.
    pub is_image: bool,
}

/// Where an [`Attachment`]'s bytes come from.
#[derive(Debug, Clone)]
pub enum Source {
    /// A file already on disk.
    File(PathBuf),
    /// Encoded image bytes with no file behind them — a pasted image, which
    /// only the macOS pasteboard can hand over. Held as iced's own byte-backed
    /// handle, so the chip's thumbnail decodes once instead of once per frame.
    #[cfg(target_os = "macos")]
    Bytes(image::Handle),
}

/// Turn dropped or copied paths into attachments, naming the ones the size cap
/// turned away so an intake can tell the user. A folder, a device or a nameless
/// path is skipped silently — there is nothing the user meant to attach.
pub(crate) fn intake(paths: impl IntoIterator<Item = PathBuf>) -> (Vec<Attachment>, Vec<String>) {
    let mut attachments = Vec::new();
    let mut refused = Vec::new();
    for path in paths {
        match Attachment::from_path(&path) {
            Ok(attachment) => attachments.push(attachment),
            Err(Refusal::Ignored) => {}
            Err(Refusal::TooLarge(warning)) => refused.push(warning),
        }
    }
    (attachments, refused)
}

impl Attachment {
    /// An attachment for a regular file. A file can grow between attaching and
    /// sending, so [`stage`] applies the same cap again.
    pub(crate) fn from_path(path: &Path) -> Result<Self, Refusal> {
        let (Some(metadata), Some(name)) = (
            path.metadata().ok().filter(std::fs::Metadata::is_file),
            path.file_name().and_then(std::ffi::OsStr::to_str),
        ) else {
            return Err(Refusal::Ignored);
        };
        if metadata.len() > crate::util::FILE_MAX_BYTES {
            return Err(Refusal::TooLarge(too_large(name)));
        }
        // The receive path's own list decides: an image here must be an image
        // there, or delivery refuses it.
        let is_image =
            crate::util::has_extension(path, crate::channels::telegram::IMAGE_EXTENSIONS);
        Ok(Self {
            name: name.to_string(),
            source: Source::File(path.to_path_buf()),
            is_image,
        })
    }

    /// An attachment for image bytes read from the clipboard; `Err` carries the
    /// warning for an image past the product's size cap.
    #[cfg(target_os = "macos")]
    pub(crate) fn from_image_bytes(name: String, bytes: Vec<u8>) -> Result<Self, String> {
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > crate::util::FILE_MAX_BYTES {
            // The generated file name means nothing to the user; name the paste.
            return Err(too_large("The copied image"));
        }
        Ok(Self {
            name,
            source: Source::Bytes(image::Handle::from_bytes(bytes)),
            is_image: true,
        })
    }

    /// A handle for the chip thumbnail of an image attachment. `None` for a
    /// document, which the chip renders as a glyph.
    pub(crate) fn preview_handle(&self) -> Option<image::Handle> {
        if !self.is_image {
            return None;
        }
        Some(match &self.source {
            Source::File(path) => image::Handle::from_path(path.clone()),
            #[cfg(target_os = "macos")]
            Source::Bytes(handle) => handle.clone(),
        })
    }
}

// ── Staging ─────────────────────────────────────────────────────────

/// One message's staged attachments: the staging directories to record on the
/// message, and the markers that deliver its files.
#[derive(Debug, Clone, Default)]
pub struct Staged {
    pub dirs: Vec<String>,
    pub markers: String,
}

/// Stage every attachment for one send: copy the bytes into a fresh directory
/// under [`crate::util::telegram_files_root`] and build the marker block, one
/// marker per line in attachment order.
pub(crate) async fn stage(attachments: &[Attachment]) -> Result<Staged, String> {
    let root = crate::util::telegram_files_root();
    if let Err(error) = tokio::fs::create_dir_all(&root).await {
        return Err(format!("Could not create the attachment folder: {error}"));
    }
    // The clock names the directory, and the claim keeps two sends apart if it
    // repeats — it also gives this send a directory of its own to clean up.
    let base = crate::util::telegram_staging_dir_name(
        "gui",
        i64::try_from(crate::util::unix_millis()).unwrap_or(i64::MAX),
    );
    let dir = crate::util::create_unique_dir(&root, &base)
        .await
        .map_err(|error| format!("Could not create the attachment folder: {error}"))?;
    let dir_name = dir
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or(base.as_str())
        .to_string();
    let mut markers = Vec::with_capacity(attachments.len());
    for attachment in attachments {
        match write_attachment(&dir, attachment).await {
            Ok(path) => markers.push(format!(
                "[{}:{}]",
                if attachment.is_image { "IMAGE" } else { "FILE" },
                path.display()
            )),
            Err(message) => {
                // Nothing of this send is delivered, so no half-filled staging
                // directory is left behind.
                let _ = tokio::fs::remove_dir_all(&dir).await;
                return Err(message);
            }
        }
    }
    Ok(Staged {
        dirs: vec![dir_name],
        markers: markers.join("\n"),
    })
}

/// Remove the staging directories of a send that never reached the pipeline, so
/// a retry does not leave the first copy behind in the temp root. Asynchronous
/// because a clear can land while a large staging copy is still running.
pub(crate) async fn discard(dirs: &[String]) {
    let root = crate::util::telegram_files_root();
    for dir in dirs {
        let _ = tokio::fs::remove_dir_all(root.join(dir)).await;
    }
}

/// Copy one attachment into `dir`, refusing anything over the product's file
/// cap. The returned path is absolute — it is what the marker names.
#[cfg(target_os = "macos")]
async fn write_attachment(dir: &Path, attachment: &Attachment) -> Result<PathBuf, String> {
    match &attachment.source {
        Source::File(source) => copy_file(dir, attachment, source).await,
        Source::Bytes(image::Handle::Bytes(_, bytes)) => write_bytes(dir, attachment, bytes).await,
        Source::Bytes(_) => Err("a pasted image is held as encoded bytes".to_string()),
    }
}

/// Copy one attachment into `dir`, refusing anything over the product's file
/// cap. The returned path is absolute — it is what the marker names.
#[cfg(not(target_os = "macos"))]
async fn write_attachment(dir: &Path, attachment: &Attachment) -> Result<PathBuf, String> {
    let Source::File(source) = &attachment.source;
    copy_file(dir, attachment, source).await
}

/// Copy `source` into `dir` under the attachment's own name, refusing anything
/// over the product's file cap.
async fn copy_file(dir: &Path, attachment: &Attachment, source: &Path) -> Result<PathBuf, String> {
    let size = tokio::fs::metadata(source)
        .await
        .map_err(|error| format!("Could not read {}: {error}", attachment.name))?
        .len();
    if size > crate::util::FILE_MAX_BYTES {
        return Err(too_large(&attachment.name));
    }
    // The claim creates the leaf, so the copy cannot land on another
    // attachment's file.
    let (file, path) = crate::util::create_unique_file(dir, &leaf_name(&attachment.name))
        .await
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))?;
    drop(file);
    tokio::fs::copy(source, &path)
        .await
        .map(|_| path)
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))
}

/// Write pasted image `bytes` into `dir` under the attachment's own name,
/// refusing anything over the product's file cap.
#[cfg(target_os = "macos")]
async fn write_bytes(dir: &Path, attachment: &Attachment, bytes: &[u8]) -> Result<PathBuf, String> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > crate::util::FILE_MAX_BYTES {
        return Err(too_large(&attachment.name));
    }
    crate::util::write_unique_file(dir, &leaf_name(&attachment.name), bytes)
        .await
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))
}

/// The warning for a file past the product's cap: intake and the send path word
/// it the same way.
fn too_large(name: &str) -> String {
    format!(
        "{name} is too large to attach (maximum {} MB)",
        crate::util::FILE_MAX_BYTES / (1024 * 1024)
    )
}

/// A single plain path component for `raw`: marker- and path-safe, and never a
/// directory step. Two attachments of one name stay apart through the claim
/// itself ([`crate::util::create_unique_file`]'s `_<n>` suffix).
fn leaf_name(raw: &str) -> String {
    let sanitized = crate::util::neutralized_name(raw);
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        "attachment".to_string()
    } else {
        sanitized
    }
}

// ── Chip ────────────────────────────────────────────────────────────

/// The composer chip for one attachment: thumbnail (images) or document glyph,
/// the name, and a removal control that publishes `remove`.
pub(crate) fn chip<'a, M: Clone + 'a>(attachment: &'a Attachment, remove: M) -> Element<'a, M> {
    let thumb: Element<'a, M> = match attachment.preview_handle() {
        Some(handle) => image(handle)
            .width(Length::Fixed(CHIP_THUMB))
            .height(Length::Fixed(CHIP_THUMB))
            .border_radius(4.0)
            .into(),
        None => container(
            iced_fonts::lucide::file_text::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_16)
                .color(theme::TEXT_SECONDARY),
        )
        .width(Length::Fixed(CHIP_THUMB))
        .height(Length::Fixed(CHIP_THUMB))
        .center_x(Length::Fixed(CHIP_THUMB))
        .center_y(Length::Fixed(CHIP_THUMB))
        .into(),
    };
    container(
        row![
            thumb,
            text(crate::util::truncate(&attachment.name, CHIP_NAME_MAX_CHARS))
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY),
            super::widgets::compact_x_button(
                [theme::PAD_1, theme::PAD_4],
                Some(remove),
                Some("remove attachment"),
            ),
        ]
        .spacing(theme::SPACE_6)
        .align_y(Alignment::Center),
    )
    .padding([theme::PAD_2, theme::PAD_6])
    .style(theme::elevated_card_style)
    .into()
}

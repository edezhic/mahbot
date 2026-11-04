//! Files the desktop chat composer attaches to a message.
//!
//! A send stages them the way the Telegram receive path stages inbound media — bytes copied into
//! one directory under [`crate::util::telegram_files_root`] — so the message carries the same
//! `[IMAGE:…]` / `[VIDEO:…]` / `[FILE:…]` markers and `attachment_dirs` entry, and
//! `channels::enrich_message` hands it to the agent with no channel-side special case.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use ::image as image_crate;
use iced::alignment::{Horizontal, Vertical};
use iced::widget::{container, image, row, text};
use iced::{Alignment, ContentFit, Element, Length};

use super::media_markers::{BoundedLruCache, CachedImage, cached_image};
use super::theme;

const CHIP_THUMB: f32 = 28.0;

const CHIP_NAME_MAX_CHARS: usize = 24;

const PREVIEW_HEIGHT: f32 = 72.0;

const PREVIEW_MIN_WIDTH: f32 = 48.0;
const PREVIEW_MAX_WIDTH: f32 = 160.0;

/// Pixels a preview is decoded at per tile pixel: twice the tile keeps it sharp on a HiDPI display
/// while the retained RGBA stays tiny.
const PREVIEW_SCALE: f32 = 2.0;

const PREVIEW_RADIUS: f32 = 4.0;

const MAX_CACHED_PREVIEWS: usize = 64;

/// Why a candidate path did not become an [`Attachment`].
#[derive(Debug)]
pub(crate) enum Refusal {
    /// Not a file the user offered — a folder (dropping one is out of scope), a device, or a name
    /// that is not UTF-8. Nothing to report.
    Ignored,
    TooLarge(String),
}

/// One file the composer holds for the next send, before anything is written.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub name: String,
    /// The bytes to deliver, read only when the message is sent.
    pub source: Source,
    pub kind: Kind,
}

/// What a staged attachment is delivered as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A raster the receive path routes on the `[IMAGE:…]` marker.
    Image,
    /// A clip whose `[VIDEO:…]` marker makes the enrichment pass read its timeline.
    Video,
    /// An opaque workspace file, delivered as `[FILE:…]`.
    File,
}

impl Kind {
    /// The marker the staged message carries for this kind.
    fn marker(self) -> crate::util::MediaMarkerKind {
        match self {
            Self::Image => crate::util::MediaMarkerKind::Image,
            Self::Video => crate::util::MediaMarkerKind::Video,
            Self::File => crate::util::MediaMarkerKind::File,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    File(PathBuf),
    /// Encoded image bytes with no file behind them — a pasted image, which only the macOS
    /// pasteboard can hand over. Held as iced's own byte-backed handle, so the paste keeps one
    /// stable identity for its preview.
    #[cfg(target_os = "macos")]
    Bytes(image::Handle),
}

/// Turn dropped or copied paths into attachments, naming the ones the size cap turned away so an
/// intake can tell the user. A folder, a device or a nameless path is skipped silently — there is
/// nothing the user meant to attach.
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
    /// An attachment for a regular file. A file can grow between attaching and sending, so
    /// [`stage`] applies the same cap again.
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
        // The receive path's own list decides: an image here must be an image there, or delivery
        // refuses it.
        let kind = if crate::util::has_extension(path, crate::channels::telegram::IMAGE_EXTENSIONS)
        {
            Kind::Image
        } else if crate::util::is_video_extension(path) {
            Kind::Video
        } else {
            Kind::File
        };
        Ok(Self {
            name: name.to_string(),
            source: Source::File(path.to_path_buf()),
            kind,
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_image_bytes(name: String, bytes: Vec<u8>) -> Result<Self, String> {
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > crate::util::FILE_MAX_BYTES {
            return Err(too_large("The copied image"));
        }
        Ok(Self {
            name,
            source: Source::Bytes(image::Handle::from_bytes(bytes)),
            kind: Kind::Image,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Staged {
    pub dirs: Vec<String>,
    pub markers: String,
}

pub(crate) async fn stage(attachments: &[Attachment]) -> Result<Staged, String> {
    let root = crate::util::telegram_files_root();
    if let Err(error) = tokio::fs::create_dir_all(&root).await {
        return Err(format!("Could not create the attachment folder: {error}"));
    }
    // The clock names the directory, and the claim keeps two sends apart if it repeats — it also
    // gives this send a directory of its own to clean up.
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
                attachment.kind.marker().token(),
                path.display()
            )),
            Err(message) => {
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

/// Remove the staging directories of a send that never reached the pipeline, so a retry does not
/// leave the first copy behind in the temp root. Asynchronous because a clear can land while a
/// large staging copy is still running.
pub(crate) async fn discard(dirs: &[String]) {
    let root = crate::util::telegram_files_root();
    for dir in dirs {
        let _ = tokio::fs::remove_dir_all(root.join(dir)).await;
    }
}

/// Copy one attachment into `dir`, refusing anything over the product's file cap. The returned path
/// is absolute — it is what the marker names.
#[cfg(target_os = "macos")]
async fn write_attachment(dir: &Path, attachment: &Attachment) -> Result<PathBuf, String> {
    match &attachment.source {
        Source::File(source) => copy_file(dir, attachment, source).await,
        Source::Bytes(image::Handle::Bytes(_, bytes)) => write_bytes(dir, attachment, bytes).await,
        Source::Bytes(_) => Err("a pasted image is held as encoded bytes".to_string()),
    }
}

/// Copy one attachment into `dir`, refusing anything over the product's file cap. The returned path
/// is absolute — it is what the marker names.
#[cfg(not(target_os = "macos"))]
async fn write_attachment(dir: &Path, attachment: &Attachment) -> Result<PathBuf, String> {
    let Source::File(source) = &attachment.source;
    copy_file(dir, attachment, source).await
}

/// Copy `source` into `dir` under the attachment's own name, refusing anything over the product's
/// file cap.
async fn copy_file(dir: &Path, attachment: &Attachment, source: &Path) -> Result<PathBuf, String> {
    let size = tokio::fs::metadata(source)
        .await
        .map_err(|error| format!("Could not read {}: {error}", attachment.name))?
        .len();
    if size > crate::util::FILE_MAX_BYTES {
        return Err(too_large(&attachment.name));
    }
    // The claim creates the leaf, so the copy cannot land on another attachment's file.
    let (file, path) = crate::util::create_unique_file(dir, &leaf_name(&attachment.name))
        .await
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))?;
    drop(file);
    tokio::fs::copy(source, &path)
        .await
        .map(|_| path)
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))
}

/// Write pasted image `bytes` into `dir` under the attachment's own name, refusing anything over
/// the product's file cap.
#[cfg(target_os = "macos")]
async fn write_bytes(dir: &Path, attachment: &Attachment, bytes: &[u8]) -> Result<PathBuf, String> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > crate::util::FILE_MAX_BYTES {
        return Err(too_large(&attachment.name));
    }
    crate::util::write_unique_file(dir, &leaf_name(&attachment.name), bytes)
        .await
        .map_err(|error| format!("Could not attach {}: {error}", attachment.name))
}

fn too_large(name: &str) -> String {
    format!(
        "{name} is too large to attach (maximum {} MB)",
        crate::util::FILE_MAX_BYTES / (1024 * 1024)
    )
}

/// A single plain path component for `raw`: marker- and path-safe, and never a directory step. Two
/// attachments of one name stay apart through the claim itself
/// ([`crate::util::create_unique_file`]'s `_<n>` suffix).
fn leaf_name(raw: &str) -> String {
    let sanitized = crate::util::neutralized_name(raw);
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        "attachment".to_string()
    } else {
        sanitized
    }
}

static FILE_PREVIEWS: LazyLock<Mutex<BoundedLruCache<PathBuf, CachedImage>>> =
    LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_PREVIEWS)));

#[cfg(target_os = "macos")]
static PASTED_PREVIEWS: LazyLock<Mutex<BoundedLruCache<iced::advanced::image::Id, CachedImage>>> =
    LazyLock::new(|| Mutex::new(BoundedLruCache::with_capacity(MAX_CACHED_PREVIEWS)));

/// The tile-sized preview of a picture attachment, decoded and cached the first time its tile is
/// drawn: iced's handle, and the width the tile takes at [`PREVIEW_HEIGHT`]. `None` for a document
/// — and for a picture that is gone or is not a raster this product decodes — which [`chip`] then
/// draws as a file. Never touches [`Attachment::kind`], so what is sent does not depend on whether
/// the picture happens to decode.
#[expect(
    clippy::cast_precision_loss,
    reason = "a tile is a couple of hundred pixels wide"
)]
fn preview(attachment: &Attachment) -> Option<(image::Handle, f32)> {
    if attachment.kind != Kind::Image {
        return None;
    }
    let handle = match &attachment.source {
        Source::File(path) => {
            // The file can move or grow between attaching and sending, so the cap is read from it
            // again here.
            let metadata = std::fs::metadata(path).ok()?;
            if !metadata.is_file() || metadata.len() > crate::util::FILE_MAX_BYTES {
                return None;
            }
            cached_image(&FILE_PREVIEWS, path.as_path(), || {
                std::fs::read(path).ok().and_then(|bytes| decode(&bytes))
            })
        }
        #[cfg(target_os = "macos")]
        Source::Bytes(handle) => {
            let image::Handle::Bytes(_, bytes) = handle else {
                return None;
            };
            cached_image(&PASTED_PREVIEWS, &handle.id(), || decode(bytes))
        }
    }?;
    let width = match &handle {
        image::Handle::Rgba { width, .. } => *width,
        _ => return None,
    };
    Some((handle, width as f32 / PREVIEW_SCALE))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a preview is a few hundred pixels, so its pixel counts fit and its proportions are screen magnitudes"
)]
fn decode(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut picture = crate::util::media_target::decode_raster_image(
        bytes,
        crate::util::media_target::raster_decode_limits(),
    )
    .ok()?;
    if let Some(orientation) = crate::util::exif_orientation(bytes) {
        picture.apply_orientation(orientation);
    }
    let aspect = picture.width() as f32 / picture.height().max(1) as f32;
    let logical_width = (PREVIEW_HEIGHT * aspect).clamp(PREVIEW_MIN_WIDTH, PREVIEW_MAX_WIDTH);
    let width = (logical_width * PREVIEW_SCALE).round() as u32;
    let height = (PREVIEW_HEIGHT * PREVIEW_SCALE) as u32;
    let filled = picture.resize_to_fill(width, height, image_crate::imageops::FilterType::Triangle);
    Some((width, height, filled.into_rgba8().into_raw()))
}

/// The composer chip for one attachment: a picture tile when it can be previewed, and the file chip
/// (document glyph, shortened name, removal control) for a document or a picture that cannot.
/// `remove` publishes the removal.
pub(crate) fn chip<'a, M: Clone + 'a>(attachment: &'a Attachment, remove: M) -> Element<'a, M> {
    match preview(attachment) {
        Some((handle, width)) => picture_tile(handle, width, &attachment.name, remove),
        None => file_chip(&attachment.name, remove),
    }
}

fn picture_tile<'a, M: Clone + 'a>(
    handle: image::Handle,
    width: f32,
    name: &'a str,
    remove: M,
) -> Element<'a, M> {
    let picture = image(handle)
        .width(Length::Fixed(width))
        .height(Length::Fixed(PREVIEW_HEIGHT))
        .content_fit(ContentFit::Fill)
        .border_radius(PREVIEW_RADIUS);
    iced::widget::stack![
        super::widgets::tooltip_hint(picture, name),
        container(
            container(super::widgets::compact_x_button(
                theme::TEXT_PRIMARY,
                theme::composer_icon(theme::TEXT_12),
                [theme::PAD_1, theme::PAD_4],
                Some(remove),
                Some("remove attachment"),
            ))
            .padding(theme::PAD_1)
            .style(theme::scrim_style),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Horizontal::Right)
        .align_y(Vertical::Top)
        .padding(theme::PAD_2),
    ]
    .into()
}

/// The file chip of one attachment: the document glyph, the shortened name and the removal control
/// after it.
fn file_chip<'a, M: Clone + 'a>(name: &'a str, remove: M) -> Element<'a, M> {
    container(
        row![
            container(
                iced_fonts::lucide::file_text::<iced::Theme, iced::Renderer>()
                    .size(theme::composer_icon(theme::TEXT_16))
                    .color(theme::TEXT_SECONDARY),
            )
            .width(Length::Fixed(CHIP_THUMB))
            .height(Length::Fixed(CHIP_THUMB))
            .center_x(Length::Fixed(CHIP_THUMB))
            .center_y(Length::Fixed(CHIP_THUMB)),
            text(crate::util::truncate(name, CHIP_NAME_MAX_CHARS))
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY),
            super::widgets::compact_x_button(
                theme::TEXT_SECONDARY,
                theme::composer_icon(theme::TEXT_12),
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

#[cfg(test)]
mod tests {
    use super::*;
    use image_crate::ImageEncoder as _;

    #[test]
    fn a_preview_decodes_at_its_tile_size() {
        assert_eq!(
            decode(&png(300, 300)).map(|(w, h, _)| (w, h)),
            Some((144, 144))
        );
        assert_eq!(
            decode(&png(300, 100)).map(|(w, h, _)| (w, h)),
            Some((320, 144))
        );
        assert_eq!(
            decode(&png(100, 300)).map(|(w, h, _)| (w, h)),
            Some((96, 144))
        );
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let pixels = vec![0x40; usize::try_from(width * height * 4).unwrap_or(0)];
        let mut png = Vec::new();
        image_crate::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &pixels,
                width,
                height,
                image_crate::ExtendedColorType::Rgba8,
            )
            .expect("an in-memory PNG encodes");
        png
    }
}

//! Clipboard attachments for the desktop chat composer.
//!
//! Iced's clipboard is text-only, so copied files and image data are read
//! through NSPasteboard on macOS — the only way to the rest. Where no
//! attachment can be read (another platform, or off the main thread) the
//! composer states that plainly ([`UNSUPPORTED`]): an empty clipboard cannot be
//! told apart from one whose content this platform cannot read, so the notice
//! names the capability rather than a failed paste. Dropped files keep working
//! there through winit.

use super::attachment::Attachment;

/// Shown by the composer when this platform cannot read attachment content
/// from the clipboard at all.
pub(crate) const UNSUPPORTED: &str = "Reading copied files or images from the clipboard is not available on this platform. Drag the file onto the chat instead.";

/// Whether the clipboard's attachment content can be read at all here.
#[cfg(target_os = "macos")]
pub(crate) fn can_read() -> bool {
    // Read on the main thread only — the one place this app touches AppKit —
    // and say so when a paste arrives anywhere else.
    objc2::MainThreadMarker::new().is_some()
}

/// Whether the clipboard's attachment content can be read at all here.
#[cfg(not(target_os = "macos"))]
pub(crate) const fn can_read() -> bool {
    false
}

/// The clipboard's copied files, in paste order, and the names of the ones the
/// size cap turned away. Both empty where [`can_read`] says no.
#[cfg(target_os = "macos")]
pub(crate) fn read_files() -> (Vec<Attachment>, Vec<String>) {
    if !can_read() {
        return (Vec::new(), Vec::new());
    }
    files(&objc2_app_kit::NSPasteboard::generalPasteboard())
}

/// The clipboard's copied files, in paste order.
#[cfg(not(target_os = "macos"))]
pub(crate) const fn read_files() -> (Vec<Attachment>, Vec<String>) {
    (Vec::new(), Vec::new())
}

/// The clipboard's image data, re-encoded as PNG: `None` when it holds none and
/// the warning for an image past the size cap.
#[cfg(target_os = "macos")]
pub(crate) fn read_image() -> Result<Option<Attachment>, String> {
    if !can_read() {
        return Ok(None);
    }
    image(&objc2_app_kit::NSPasteboard::generalPasteboard())
}

/// The clipboard's image data.
#[cfg(not(target_os = "macos"))]
#[expect(clippy::unnecessary_wraps)] // one signature; only the macOS arm refuses
pub(crate) const fn read_image() -> Result<Option<Attachment>, String> {
    Ok(None)
}

/// Copied file URLs on the pasteboard, in paste order. Non-file URLs are
/// skipped, and [`super::attachment::intake`] decides about the rest.
#[cfg(target_os = "macos")]
fn files(pasteboard: &objc2_app_kit::NSPasteboard) -> (Vec<Attachment>, Vec<String>) {
    use objc2::ClassType;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_app_kit::NSPasteboardURLReadingFileURLsOnlyKey;
    use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSURL};

    let classes = NSArray::from_slice(&[NSURL::class()]);
    // SAFETY: an NSNumber is an AnyObject; the dictionary only carries it as
    // the reading option's value.
    let only_files = unsafe { Retained::cast_unchecked::<AnyObject>(NSNumber::new_bool(true)) };
    // SAFETY: the framework defines this type key before any pasteboard read.
    let only_file_urls_key = unsafe { NSPasteboardURLReadingFileURLsOnlyKey };
    let options = NSDictionary::from_retained_objects(&[only_file_urls_key], &[only_files]);
    // SAFETY: the class array and the options dictionary match what the
    // selector documents, and both outlive the call.
    let Some(objects) =
        (unsafe { pasteboard.readObjectsForClasses_options(&classes, Some(&options)) })
    else {
        return (Vec::new(), Vec::new());
    };
    let paths = objects.to_vec().into_iter().filter_map(|object| {
        let url = object.downcast_ref::<NSURL>()?;
        let path = url.path()?;
        Some(std::path::PathBuf::from(path.to_string()))
    });
    super::attachment::intake(paths)
}

/// The clipboard's image, converted to PNG. Tries the PNG flavour, then TIFF,
/// then a pasteboard-backed `NSImage` — whichever the copying application
/// offered.
#[cfg(target_os = "macos")]
fn image(pasteboard: &objc2_app_kit::NSPasteboard) -> Result<Option<Attachment>, String> {
    use objc2::AnyThread;
    use objc2_app_kit::{NSImage, NSPasteboardTypePNG, NSPasteboardTypeTIFF};

    // SAFETY: the framework defines both type keys before any pasteboard read.
    let (png_type, tiff_type) = unsafe { (NSPasteboardTypePNG, NSPasteboardTypeTIFF) };
    let png = pasteboard
        .dataForType(png_type)
        .map(|data| data.to_vec())
        .or_else(|| {
            let tiff = pasteboard.dataForType(tiff_type)?;
            png_from_tiff(&tiff)
        })
        .or_else(|| {
            let image = NSImage::initWithPasteboard(NSImage::alloc(), pasteboard)?;
            let tiff = image.TIFFRepresentation()?;
            png_from_tiff(&tiff)
        });
    let Some(png) = png.filter(|png| !png.is_empty()) else {
        return Ok(None);
    };
    Attachment::from_image_bytes(format!("clipboard-{}.png", crate::util::unix_millis()), png)
        .map(Some)
}

/// Re-encode TIFF data as PNG through the bitmap representation, which decodes
/// whatever `NSImage` or the pasteboard accepted.
#[cfg(target_os = "macos")]
fn png_from_tiff(data: &objc2_foundation::NSData) -> Option<Vec<u8>> {
    use objc2::AnyThread;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep};
    use objc2_foundation::NSDictionary;

    let rep = NSBitmapImageRep::initWithData(NSBitmapImageRep::alloc(), data)?;
    let properties = NSDictionary::new();
    // SAFETY: the properties dictionary is empty, which the selector accepts as
    // "no representation options".
    let png =
        unsafe { rep.representationUsingType_properties(NSBitmapImageFileType::PNG, &properties) }?;
    Some(png.to_vec())
}

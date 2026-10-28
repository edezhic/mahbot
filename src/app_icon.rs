//! The product's icon — the artwork itself, and every surface outside the
//! window it has to appear on: the file's own resource on Windows, the running
//! application's Dock tile on macOS, the launcher entry on Linux, the tray icon of
//! `crate::tray`, and the dashboard's own renderings.
//!
//! One approved file serves everything this code owns: `assets/mb-icon-256.png`,
//! decoded once and handed out as it is. Where a platform asks for a size the
//! artwork does not have, a copy is scaled and re-encoded for it; the approved
//! file itself is never altered.
//!
//! Windows is the exception, and the reason `assets/mb-icon.ico` exists: there the
//! icon belongs to the *file*, not to the running program — it is embedded by
//! `build.rs` and is what Explorer, the Start menu and the file's own properties
//! read. The one part of it that cannot be settled at build time is attached to
//! the live window at startup.
//!
//! Nothing here is fatal. A platform with no such surface is the same code path as
//! a platform whose surface is already correct: work is skipped, a placement that
//! fails is reported, and no failure is propagated into boot.

#[cfg(target_os = "linux")]
use std::path::Path;
use std::sync::{LazyLock, OnceLock};

/// The owner's approved artwork: 256x256, 16-bit RGBA with a fully transparent
/// background. `assets/mb-icon-1024.png` is the same artwork at the size it was
/// drawn at and is kept beside it as the master; nothing loads it at run time.
const ARTWORK_PNG: &[u8] = include_bytes!("../assets/mb-icon-256.png");

/// The name of the product's window and of its launcher entry, which are the same
/// name on purpose: a desktop environment associates a running window with an
/// entry by matching the name the window announces against the entry's file name
/// (`<name>.desktop`), and nothing is shown for the window when they disagree.
#[cfg(target_os = "linux")]
pub const LINUX_APP_ID: &str = "mahbot";

/// The artwork's pixels, decoded once for the process.
struct Artwork {
    /// 8-bit RGBA, row-major, `width * height * 4` bytes: the layout every consumer
    /// here wants (iced's RGBA handle, an `NSBitmapImageRep`, a PNG re-encode).
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// The decoded artwork. The file is compiled into the binary, so a decode failure
/// is a broken build, not a runtime condition — the same rule the embedded prompts
/// follow.
fn artwork() -> &'static Artwork {
    static ARTWORK: LazyLock<Artwork> = LazyLock::new(|| {
        let decode = image::load_from_memory_with_format(ARTWORK_PNG, image::ImageFormat::Png)
            .expect("the embedded product icon is a PNG");
        let rgba = decode.into_rgba8();
        Artwork {
            width: rgba.width(),
            height: rgba.height(),
            rgba: rgba.into_raw(),
        }
    });
    &ARTWORK
}

/// The icon of the window itself, for [`iced::window::Settings::icon`] — the
/// window's icon on X11; on Windows winit sets it as the *small* icon, and the
/// Windows icon pass then replaces both with the executable's own. Wayland and macOS
/// have no window icon at all and ignore it, which is what "degrade quietly" means
/// here.
#[must_use]
pub fn window_icon() -> Option<iced::window::Icon> {
    let artwork = artwork();
    iced::window::icon::from_rgba(artwork.rgba.clone(), artwork.width, artwork.height)
        .map_err(|e| not_placed("the window icon", &e.to_string()))
        .ok()
}

/// The dashboard's own rendering of the icon, for the surfaces that are text-only
/// without it (the starting screen, the About block).
///
/// The handle is built once and handed out as a clone: a fresh handle carries a
/// fresh texture id and a new one per frame would re-upload the image every frame.
#[must_use]
pub(crate) fn widget_image() -> iced::widget::image::Handle {
    static HANDLE: OnceLock<iced::widget::image::Handle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            let artwork = artwork();
            iced::widget::image::Handle::from_rgba(
                artwork.width,
                artwork.height,
                artwork.rgba.clone(),
            )
        })
        .clone()
}

/// Places the icon on the surfaces the platform keeps outside the window, once per
/// start: the running application's Dock tile (macOS) and the launcher entry plus
/// its icon-theme copies (Linux). A no-op on Windows, where the executable's own
/// resource and the window's own handle carry it, and wherever there is no such
/// surface.
pub fn install_desktop_integration() {
    #[cfg(target_os = "macos")]
    set_dock_icon();
    #[cfg(target_os = "linux")]
    install_launcher_entry();
}

/// The size a macOS status item draws its image at: the menu bar's own height, which is
/// what the system's items beside it occupy.
#[cfg(target_os = "macos")]
const TRAY_STATUS_POINTS: u32 = 18;

/// The same mark at backing-store resolution — what a Retina menu bar draws it from.
#[cfg(target_os = "macos")]
const TRAY_MARK_SIDE: u32 = TRAY_STATUS_POINTS * 2;

/// The artwork's mark at the resolution a Retina menu bar draws, derived once — the
/// histogram, crop and scale in [`mark_pixels`] are the most work anything here does.
#[cfg(target_os = "macos")]
fn tray_mark() -> Option<&'static [u8]> {
    static MARK: LazyLock<Option<Vec<u8>>> = LazyLock::new(|| mark_pixels(TRAY_MARK_SIDE));
    MARK.as_deref()
}

/// The tray icon's own image, for [`crate::tray`]. `None` when it cannot be built, which
/// leaves the tray uncreated rather than drawn blank.
///
/// macOS: the artwork's *mark* — its letters without the dark tile they are drawn on — as
/// a template image, which the system tints like its own menu-bar items; the tile is
/// illegible at that size and would only be a dark square there. This is the 1× copy that
/// lays the status item out, and `tray_status_image` replaces it with a sharp one.
///
/// Windows: the executable's icon resource, the one `build.rs` embedded, at the
/// notification area's small size — the same artwork the taskbar and Alt-Tab show.
#[cfg(any(target_os = "macos", target_os = "windows"))]
#[must_use]
pub(crate) fn tray_icon_image() -> Option<tray_icon::Icon> {
    #[cfg(target_os = "macos")]
    {
        let pixels = downscaled(tray_mark()?, TRAY_MARK_SIDE, TRAY_STATUS_POINTS);
        tray_icon::Icon::from_rgba(pixels, TRAY_STATUS_POINTS, TRAY_STATUS_POINTS).ok()
    }
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSMICON};

        // The shell draws the icon at its small-icon size, which follows the display's
        // scaling; the resource holds that size and the larger ones.
        let size = unsafe { GetSystemMetrics(SM_CXSMICON) };
        let size = u32::try_from(size).unwrap_or(0);
        tray_icon::Icon::from_resource(ICON_RESOURCE_ID, Some((size, size))).ok()
    }
}

/// macOS: the same mark at backing-store resolution, sized to the points the status item
/// draws, as a template image. A menu-bar image is drawn at its point size, so this is
/// what keeps the mark sharp on a Retina display. `None` leaves the 1× copy above in
/// place, which still shows the mark.
#[cfg(target_os = "macos")]
#[must_use]
pub(crate) fn tray_status_image() -> Option<objc2::rc::Retained<objc2_app_kit::NSImage>> {
    let image = square_nsimage(tray_mark()?, TRAY_MARK_SIDE, f64::from(TRAY_STATUS_POINTS))?;
    image.setTemplate(true);
    Some(image)
}

/// A square RGBA image scaled down to `to` pixels a side, from one `from` a side.
#[cfg(target_os = "macos")]
fn downscaled(pixels: &[u8], from: u32, to: u32) -> Vec<u8> {
    let image = image::RgbaImage::from_raw(from, from, pixels.to_vec())
        .expect("a square RGBA image is as long as its side says");
    image::imageops::resize(&image, to, to, image::imageops::FilterType::Triangle).into_raw()
}

/// The artwork's mark as a `side`-pixel square of black pixels whose alpha carries the
/// shape — what a monochrome template image is made of. The artwork's own dominant opaque
/// colour is the tile the mark is drawn on and becomes transparent; what differs from it
/// is the mark, cropped to its own bounds and scaled to fill the square, so it stays
/// legible at the size a status area draws it. `None` when the artwork holds no mark.
#[cfg(target_os = "macos")]
fn mark_pixels(side: u32) -> Option<Vec<u8>> {
    /// Below this distance from the tile a pixel is the tile's own — noise in its flat
    /// colour, not the mark drawn on it.
    const CUTOFF: u8 = 8;

    let artwork = artwork();
    let tile = dominant_colour(artwork);
    let mut pixels = image::RgbaImage::new(artwork.width, artwork.height);
    let mut bounds: Option<(u32, u32, u32, u32)> = None;
    let mut strongest = 0u8;
    for (index, pixel) in artwork.rgba.as_chunks::<4>().0.iter().enumerate() {
        let distance = (0..3)
            .map(|channel| pixel[channel].abs_diff(tile[channel]))
            .max()
            .unwrap_or(0);
        // Only the tile's own pixels can carry the mark: the artwork's transparent
        // background is not part of it, and neither is the tile's anti-aliased edge, which
        // would draw a ring around the letters and stretch the crop over the whole tile.
        let opacity = if pixel[3] == 255 { distance } else { 0 };
        strongest = strongest.max(opacity);
        #[expect(clippy::cast_possible_truncation)]
        let (x, y) = (index as u32 % artwork.width, index as u32 / artwork.width);
        pixels.put_pixel(x, y, image::Rgba([0, 0, 0, opacity]));
        if opacity > CUTOFF {
            let (left, top, right, bottom) = bounds.unwrap_or((x, y, x, y));
            bounds = Some((left.min(x), top.min(y), right.max(x), bottom.max(y)));
        }
    }
    // Nothing above the cutoff means the artwork holds no mark at all.
    let (left, top, right, bottom) = bounds?;

    let mut mark =
        image::imageops::crop_imm(&pixels, left, top, right - left + 1, bottom - top + 1)
            .to_image();
    // Fully opaque where the mark is at its strongest, proportional everywhere else: a
    // template image is drawn in the system's colour, so only the shape it carries matters.
    for pixel in mark.pixels_mut() {
        let alpha = u16::from(pixel.0[3]) * 255 / u16::from(strongest);
        pixel.0[3] = u8::try_from(alpha).unwrap_or(u8::MAX);
    }
    // A one-pixel margin on each side, so the mark does not sit flush against the item's
    // edges the way a cropped image would.
    let inner = side.saturating_sub(2).max(1);
    let scale = f64::from(inner) / f64::from(mark.width().max(mark.height()));
    // A proportional fit, rounded to at least one pixel so a mark of any shape survives.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (width, height) = (
        ((f64::from(mark.width()) * scale).round() as u32).max(1),
        ((f64::from(mark.height()) * scale).round() as u32).max(1),
    );
    let scaled =
        image::imageops::resize(&mark, width, height, image::imageops::FilterType::Lanczos3);
    let mut square = image::RgbaImage::new(side, side);
    image::imageops::overlay(
        &mut square,
        &scaled,
        i64::from((side - width) / 2),
        i64::from((side - height) / 2),
    );
    Some(square.into_raw())
}

/// The colour most of the artwork's opaque pixels have: the tile the mark is drawn on.
/// Read from a histogram of the 4 high bits of each channel — enough to separate a tile
/// from anything drawn on it — and reported as that bucket's middle.
#[cfg(target_os = "macos")]
fn dominant_colour(artwork: &Artwork) -> [u8; 3] {
    const LEVELS: usize = 16;

    let mut counts = vec![0u32; LEVELS * LEVELS * LEVELS];
    for pixel in artwork
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| pixel[3] == 255)
    {
        let bucket = (usize::from(pixel[0] >> 4) << 8)
            | (usize::from(pixel[1] >> 4) << 4)
            | usize::from(pixel[2] >> 4);
        counts[bucket] += 1;
    }
    let bucket = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| **count)
        .map_or(0, |(bucket, _)| bucket);
    let middle = |shift: usize| {
        #[expect(clippy::cast_possible_truncation)]
        let level = ((bucket >> shift) & 0x0F) as u8;
        (level << 4) | 0x08
    };
    [middle(8), middle(4), middle(0)]
}

/// Reports a placement that did not happen, through the boot-diagnostic funnel
/// rather than `tracing` directly: this work runs before the log subscriber exists,
/// and a message written to no subscriber at all would be exactly the silent pass
/// a missing icon must not become.
fn not_placed(surface: &str, reason: &str) {
    crate::boot::boot_diagnostic(format!("{surface} was not placed: {reason}"));
}

/// macOS: shows the artwork in the Dock while this process runs.
///
/// A bare binary has no bundle for macOS to read an icon from, so the running
/// application's own icon image is what the Dock tile displays; it returns to the
/// system's generic icon when the process ends.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSApplication;

    let Some(marker) = MainThreadMarker::new() else {
        not_placed("the Dock icon", "it can only be set from the main thread");
        return;
    };
    let artwork = artwork();
    let Some(icon) = square_nsimage(&artwork.rgba, artwork.width, f64::from(artwork.width)) else {
        not_placed("the Dock icon", "its bitmap could not be allocated");
        return;
    };
    // SAFETY: called on the main thread, which is the thread AppKit's own state
    // belongs to; the image is only read by AppKit after this call.
    unsafe {
        NSApplication::sharedApplication(marker).setApplicationIconImage(Some(&icon));
    }
}

/// An `NSImage` of a `side`-pixel RGBA square, drawn `points` square: the pixels are the
/// backing store, the points what the system lays it out at. Built from pixels rather
/// than from an encoded file, because an image made from encoded data decodes lazily and
/// its bytes would then have to outlive it. `None` when the bitmap cannot be allocated.
#[cfg(target_os = "macos")]
// The casts below are the image's own dimensions, and its points: both are pixel counts
// of artwork this process embedded, so the narrowing the lint warns about is unreachable.
#[expect(clippy::cast_possible_wrap)]
fn square_nsimage(
    pixels: &[u8],
    side: u32,
    points: f64,
) -> Option<objc2::rc::Retained<objc2_app_kit::NSImage>> {
    use objc2::AllocAnyThread;
    use objc2_app_kit::{NSBitmapImageRep, NSDeviceRGBColorSpace, NSImage};
    use objc2_foundation::NSSize;

    let side = side as isize;
    // SAFETY: a null `planes` is the documented request for a representation that
    // allocates its own buffer, and the remaining arguments describe exactly the RGBA
    // layout the pixels are in.
    let bitmap = unsafe {
        NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
            NSBitmapImageRep::alloc(),
            std::ptr::null_mut(),
            side,
            side,
            8,
            4,
            true,
            false,
            NSDeviceRGBColorSpace,
            side * 4,
            32,
        )
    };
    let bitmap = bitmap?;
    // SAFETY: the representation allocated `side * side * 4` bytes for us, which is what
    // the copy fills; both it and the image are released by their own `Retained` values
    // once the caller drops them.
    unsafe {
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), bitmap.bitmapData(), pixels.len());
    }
    bitmap.setSize(NSSize::new(points, points));
    let icon = NSImage::initWithSize(NSImage::alloc(), NSSize::new(points, points));
    icon.addRepresentation(&bitmap);
    Some(icon)
}

/// Linux: writes the launcher entry and the icon-theme copies the desktop reads.
///
/// Both live under `$XDG_DATA_HOME` (`~/.local/share` when it is unset) and are
/// rewritten only when their content differs, so an unchanged start touches
/// nothing. `Exec` names the file this process is running from, and the product
/// relocates itself when it updates — the entry follows on the next start, which
/// is what keeps it correct without the installer ever running again.
#[cfg(target_os = "linux")]
fn install_launcher_entry() {
    let Some(data_home) = directories::BaseDirs::new().map(|base| base.data_dir().to_path_buf())
    else {
        not_placed("the launcher entry", "this account has no data directory");
        return;
    };
    let Ok(executable) = std::env::current_exe() else {
        not_placed("the launcher entry", "the running file has no path");
        return;
    };

    let entry = data_home
        .join("applications")
        .join(format!("{LINUX_APP_ID}.desktop"));
    let content = launcher_entry_text(&executable);
    if let Err(e) = write_if_changed(&entry, content.as_bytes()) {
        not_placed(
            &format!("the launcher entry at {}", entry.display()),
            &e.to_string(),
        );
    }

    for size in [256, 48] {
        let Some(png) = scaled_png(artwork(), size) else {
            continue;
        };
        let icon = data_home
            .join("icons/hicolor")
            .join(format!("{size}x{size}"))
            .join("apps")
            .join(format!("{LINUX_APP_ID}.png"));
        if let Err(e) = write_if_changed(&icon, &png) {
            not_placed(
                &format!("the launcher's icon at {}", icon.display()),
                &e.to_string(),
            );
        }
    }
}

/// The launcher entry's text: the visible name, the running file, the icon's theme
/// name and the name the window announces, which is what a dock or an application
/// switcher matches the running window against.
#[cfg(target_os = "linux")]
fn launcher_entry_text(executable: &Path) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=MahBot\n\
         Comment={description}\n\
         Exec={exec}\n\
         Icon={LINUX_APP_ID}\n\
         Terminal=false\n\
         Categories=Development;\n\
         StartupWMClass={LINUX_APP_ID}\n",
        description = env!("CARGO_PKG_DESCRIPTION"),
        exec = exec_argument(executable),
    )
}

/// A path as an `Exec` argument: quoted, with the characters the desktop entry
/// specification reserves inside a quoted argument escaped and `%` doubled — it
/// introduces a field code there, and a literal one is written `%%`. The running
/// file can sit anywhere, spaces and percentages included.
#[cfg(target_os = "linux")]
fn exec_argument(executable: &Path) -> String {
    let path = executable.to_string_lossy();
    let mut argument = String::with_capacity(path.len() + 2);
    argument.push('"');
    for character in path.chars() {
        match character {
            '"' | '\\' | '$' | '`' => {
                argument.push('\\');
                argument.push(character);
            }
            '%' => argument.push_str("%%"),
            _ => argument.push(character),
        }
    }
    argument.push('"');
    argument
}

/// A copy of the artwork for one of the icon theme's size directories, scaled and
/// re-encoded from the embedded pixels: the theme spec's minimum is 48x48, which an
/// application menu asks for, while 256 is what docks and switchers ask for. `None`
/// when it cannot be encoded, which leaves that one size out.
#[cfg(target_os = "linux")]
fn scaled_png(artwork: &Artwork, size: u32) -> Option<Vec<u8>> {
    let source = image::RgbaImage::from_raw(artwork.width, artwork.height, artwork.rgba.clone())
        .expect("the artwork's pixels match its own dimensions");
    let scaled =
        image::imageops::resize(&source, size, size, image::imageops::FilterType::Lanczos3);
    let mut png = Vec::new();
    let encoded = image::DynamicImage::ImageRgba8(scaled)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png);
    match encoded {
        Ok(()) => Some(png),
        Err(e) => {
            not_placed(
                "the launcher's icon",
                &format!("{size}x{size} could not be encoded: {e}"),
            );
            None
        }
    }
}

/// Writes `content` to `path` unless the file already holds exactly it, staging
/// through a sibling name first so a reader (a desktop shell) never sees a partial
/// file.
#[cfg(target_os = "linux")]
fn write_if_changed(path: &Path, content: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|current| current == content) {
        return Ok(());
    }
    let Some(parent) = path.parent() else {
        return Err(std::io::Error::other("the target has no directory"));
    };
    std::fs::create_dir_all(parent)?;
    let staged = path.with_extension("new");
    std::fs::write(&staged, content)?;
    if let Err(e) = std::fs::rename(&staged, path) {
        let _ = std::fs::remove_file(&staged);
        return Err(e);
    }
    Ok(())
}

/// The icon group `build.rs` embeds is the executable's first one: the window's icons
/// ([`apply_window_icons`]) and the tray's ([`tray_icon_image`]) both read it.
#[cfg(target_os = "windows")]
const ICON_RESOURCE_ID: u16 = 1;

/// Windows: attaches the executable's own icons to the window.
///
/// winit registers the window's class with no icon at all and sets only the
/// *small* one, from [`iced::window::Settings::icon`], so the taskbar, Alt-Tab and
/// the system menu — all of which read the big one — had nothing of their own to
/// show. Both sizes are attached here, to the window and to its class, which every
/// later window of that class inherits.
///
/// `raw_handle` is the window's `HWND`: [`iced::window::raw_id`] answers with
/// winit's own window id, and on Windows that conversion *is* the handle.
#[cfg(target_os = "windows")]
// iced hands the handle back as a `u64`, and winit's own conversion into that
// `u64` is an `isize`-to-`u64` cast — this is that same cast reversed, so both
// lints describe the documented round trip rather than a mistake.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(crate) fn apply_window_icons(raw_handle: u64) {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GCLP_HICON, GCLP_HICONSM, GetSystemMetrics, ICON_BIG, ICON_SMALL, IMAGE_ICON, LR_SHARED,
        LoadImageW, SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, SendMessageW, SetClassLongPtrW,
        WM_SETICON,
    };

    let window = raw_handle as isize;
    // SAFETY: `window` is the live window's own handle on the thread that created
    // it, the resource name is the id the build script wrote into this executable,
    // and a shared icon handle is neither freed nor owned by the caller.
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        let resource = usize::from(ICON_RESOURCE_ID) as *const u16;
        let big = LoadImageW(
            instance,
            resource,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXICON),
            GetSystemMetrics(SM_CYICON),
            LR_SHARED,
        );
        let small = LoadImageW(
            instance,
            resource,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON),
            GetSystemMetrics(SM_CYSMICON),
            LR_SHARED,
        );
        for (icon, kind, class_index, surface) in [
            (big, ICON_BIG, GCLP_HICON, "the taskbar icon"),
            (small, ICON_SMALL, GCLP_HICONSM, "the small icon"),
        ] {
            if icon == 0 {
                // No such icon in this executable: `build.rs` has already said so at
                // build time, and the window keeps the icon iced set on it.
                not_placed(surface, &std::io::Error::last_os_error().to_string());
                continue;
            }
            SendMessageW(window, WM_SETICON, kind as usize, icon);
            SetClassLongPtrW(window, class_index, icon);
        }
    }
}

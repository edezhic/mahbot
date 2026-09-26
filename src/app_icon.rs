//! The product's icon — the artwork itself, and every surface outside the window
//! it has to appear on.
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

/// The icon of the window itself, for [`iced::window::Settings::icon`] — the title
/// bar on Windows and on X11. Wayland and macOS have no window icon at all and
/// ignore it, which is what "degrade quietly" means here.
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
// The casts below are the artwork's own dimensions (a 256-pixel image): the
// 32-bit narrowing the lint warns about cannot be reached by them.
#[expect(clippy::cast_possible_wrap)]
fn set_dock_icon() {
    use objc2::{AnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSBitmapImageRep, NSDeviceRGBColorSpace, NSImage};
    use objc2_foundation::NSSize;

    let Some(marker) = MainThreadMarker::new() else {
        not_placed("the Dock icon", "it can only be set from the main thread");
        return;
    };
    let artwork = artwork();
    let (width, height) = (artwork.width as isize, artwork.height as isize);
    // Built from the pixels rather than the encoded file: an image made from
    // encoded data decodes lazily, so its bytes would have to outlive it.
    // SAFETY: a null `planes` is the documented request for a representation that
    // allocates its own buffer, and the remaining arguments describe exactly the
    // RGBA layout the artwork was decoded into.
    let bitmap = unsafe {
        NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
            NSBitmapImageRep::alloc(),
            std::ptr::null_mut(),
            width,
            height,
            8,
            4,
            true,
            false,
            NSDeviceRGBColorSpace,
            width * 4,
            32,
        )
    };
    let Some(bitmap) = bitmap else {
        not_placed("the Dock icon", "its bitmap could not be allocated");
        return;
    };
    // SAFETY: the representation allocated `height * bytes_per_row` bytes for us,
    // which is what the copy fills; both it and the icon are released by their own
    // `Retained` values when this call returns.
    unsafe {
        std::ptr::copy_nonoverlapping(
            artwork.rgba.as_ptr(),
            bitmap.bitmapData(),
            artwork.rgba.len(),
        );
    }
    let icon = NSImage::initWithSize(
        NSImage::alloc(),
        NSSize::new(f64::from(artwork.width), f64::from(artwork.height)),
    );
    icon.addRepresentation(&bitmap);
    // SAFETY: called on the main thread, which is the thread AppKit's own state
    // belongs to; the image is only read by AppKit after this call.
    unsafe {
        NSApplication::sharedApplication(marker).setApplicationIconImage(Some(&icon));
    }
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

/// Windows: attaches the executable's own icons to the window.
///
/// winit registers the window's class with no icon at all and sets only the
/// *small* one for the window, so the title bar gets the icon from
/// [`iced::window::Settings::icon`] while the taskbar, Alt-Tab and the system menu
/// have nothing of their own to show. The big icon is attached here — to the
/// window and to its class, which every later window of that class inherits.
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

    /// The icon group `build.rs` embeds is the executable's first one.
    const ICON_RESOURCE_ID: usize = 1;

    let window = raw_handle as isize;
    // SAFETY: `window` is the live window's own handle on the thread that created
    // it, the resource name is the id the build script wrote into this executable,
    // and a shared icon handle is neither freed nor owned by the caller.
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        let resource = ICON_RESOURCE_ID as *const u16;
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
            (small, ICON_SMALL, GCLP_HICONSM, "the title-bar icon"),
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

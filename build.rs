//! Build script — puts the product's icon inside the Windows executable.
//!
//! The icon resource is the *file's* own icon: Explorer, the Start menu, the
//! file's properties dialog and the shell's fallback for a window with no icon of
//! its own all read it from the executable. That is the one placement the running
//! program cannot do for itself, which is why it happens here.
//!
//! Every other target builds without a resource compiler and needs none: the
//! step below is a no-op outside a Windows target, and a Windows build whose
//! resource compiler cannot be found still builds — it says so instead of passing
//! silently (see the warning) and ships with the generic icon.

/// The artwork's path inside the package.
const ICON_ICO: &str = "assets/mb-icon.ico";

fn main() {
    // The artwork is read from the package's own directory, never generated.
    println!("cargo::rerun-if-changed={ICON_ICO}");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    if let Err(reason) = embed_icon() {
        println!("cargo::warning=MahBot's icon is missing from this Windows build: {reason}");
    }
}

/// Embeds `assets/mb-icon.ico` (256/128/64/48/32/16 inside one file) as the
/// executable's own icon group.
fn embed_icon() -> Result<(), String> {
    // `winresource` finds `rc.exe` through the SDK's registry entries, which do not
    // carry an ARM64 host's own tools. The release workflow puts this host's
    // compiler on `PATH` (the same step a dependency's resource build needs), so
    // prefer the one that is already there.
    if std::env::var_os("RC_PATH").is_none()
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
        && let Some(rc) = rc_on_path()
    {
        // SAFETY: the build script's own thread is the only one running this early,
        // before `winresource` looks the variable up.
        unsafe { std::env::set_var("RC_PATH", rc) };
    }

    let mut resource = winresource::WindowsResource::new();
    // The name Windows shows for the file in its own properties and dialogs;
    // without it the crate's name (`mahbot`) is used.
    resource.set("ProductName", "MahBot");
    resource.set("FileDescription", "MahBot");
    // An absolute path: the resource compiler is run from a directory of its own
    // choosing (the SDK's, on Windows), and not every implementation searches its
    // include path for the icon file itself.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "CARGO_MANIFEST_DIR is not set".to_string())?;
    resource.set_icon(
        &std::path::Path::new(&manifest)
            .join(ICON_ICO)
            .to_string_lossy(),
    );
    resource.compile().map_err(|e| e.to_string())
}

/// `rc.exe` as this host's `PATH` spells it, when it carries one at all.
#[cfg(windows)]
fn rc_on_path() -> Option<std::path::PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|dir| {
        let rc = dir.join("rc.exe");
        rc.is_file().then_some(rc)
    })
}

/// Only a Windows host ever runs the resource compiler, and only on Windows is
/// there an `rc.exe` to look for.
#[cfg(not(windows))]
fn rc_on_path() -> Option<std::path::PathBuf> {
    None
}

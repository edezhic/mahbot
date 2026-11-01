//! Build script — the two things a Windows build needs that Cargo does not do itself: the product's
//! icon inside the executable, and the Computer tool's structured-exception shim.
//!
//! The icon resource is the *file's* own icon: Explorer, the Start menu and the shell all read it
//! from the executable, the one placement the running program cannot do for itself. A build whose
//! resource compiler cannot be found still builds — it says so (see the warning) and ships with the
//! generic icon.
//!
//! The shim is C because Rust cannot catch a structured exception: without it a fault inside the UI
//! Automation client, an in-process provider or a graphics driver aborts the process. Containment
//! is not optional, so a failure to compile it fails the build rather than degrading.
//!
//! Every other target returns before both steps.

/// The artwork's path inside the package.
const ICON_ICO: &str = "assets/mb-icon.ico";

/// The structured-exception shim's path inside the package.
const SEH_C: &str = "src/tools/computer/windows/seh.c";

fn main() {
    // The artwork is read from the package's own directory, never generated.
    println!("cargo::rerun-if-changed={ICON_ICO}");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    if let Err(reason) = embed_icon() {
        println!("cargo::warning=MahBot's icon is missing from this Windows build: {reason}");
    }
    compile_seh();
}

/// Embeds `assets/mb-icon.ico` as the executable's own icon group.
fn embed_icon() -> Result<(), String> {
    // `winresource` finds `rc.exe` through the SDK's registry entries, which do not carry an ARM64
    // host's own tools. The release workflow puts this host's compiler on `PATH`, so prefer it.
    if std::env::var_os("RC_PATH").is_none()
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
        && let Some(rc) = rc_on_path()
    {
        // SAFETY: the build script's own thread is the only one running this early,
        // before `winresource` looks the variable up.
        unsafe { std::env::set_var("RC_PATH", rc) };
    }

    let mut resource = winresource::WindowsResource::new();
    // The name Windows shows for the file in its properties and dialogs.
    resource.set("ProductName", "MahBot");
    resource.set("FileDescription", "MahBot");
    // An absolute path: the resource compiler runs from a directory of its own choosing, and not
    // every implementation searches its include path for the icon file.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "CARGO_MANIFEST_DIR is not set".to_string())?;
    resource.set_icon(
        &std::path::Path::new(&manifest)
            .join(ICON_ICO)
            .to_string_lossy(),
    );
    resource.compile().map_err(|e| e.to_string())
}

/// Compiles [`SEH_C`] into the Windows target as its own static library. Unlike the icon there is
/// no fallback: without it a fault in foreign code kills the process, so a compiler failure fails
/// the build.
fn compile_seh() {
    println!("cargo::rerun-if-changed={SEH_C}");
    let mut build = cc::Build::new();
    build.file(SEH_C);
    // `__try`/`__except` is an MS extension, declared only when the compiler is told to accept
    // it: clang-based non-MSVC toolchains (zig cc on the cross lane) need it switched on
    // explicitly, and mingw-w64's own GCC has no equivalent. Those toolchains also instrument C
    // with UBSan at the low optimisation Cargo asks for in debug, which would leave this archive
    // depending on a sanitizer runtime the final link does not carry.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") {
        build
            .flag("-fms-extensions")
            .flag("-fno-sanitize=undefined");
    }
    build.compile("mahbot_seh");
}

/// `rc.exe` as this host's `PATH` spells it, when it carries one at all.
#[cfg(windows)]
fn rc_on_path() -> Option<std::path::PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|dir| {
        let rc = dir.join("rc.exe");
        rc.is_file().then_some(rc)
    })
}

/// Only a Windows host ever runs the resource compiler.
#[cfg(not(windows))]
fn rc_on_path() -> Option<std::path::PathBuf> {
    None
}

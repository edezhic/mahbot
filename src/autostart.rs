//! Start at login: the product's own registration with the operating system.
//!
//! The running app is the only writer of the system's startup state — the install
//! scripts and the updater never touch it. Each platform keeps one fixed identity,
//! so a start after an install, an update or a move re-points the same entry
//! rather than adding a second one. The owner's decision to want it is
//! `crate::config`'s (`autolaunch`), and it outlives installs and updates. A
//! disable the owner made in the system's own settings is real state here: it
//! reads as "not registered", and nothing in the product overrides it on its own.

use std::path::Path;

use anyhow::Context as _;

/// Keep the system's registration true to the owner's decision and to the binary
/// that is running. Called at every start; a no-op for a copy built from sources.
pub fn reconcile() {
    if !crate::self_update::is_published_copy() {
        return;
    }
    let outcome = if decision() {
        std::env::current_exe()
            .map_err(anyhow::Error::from)
            .and_then(|exe| platform::ensure(&exe))
    } else {
        platform::remove()
    };
    if let Err(e) = outcome {
        tracing::warn!(error = %e, "the login registration could not be kept true");
    }
}

/// Whether the system reports a registration that will start the product at login
/// — an entry the owner disabled from the system's own settings reads as `false`,
/// and a probe that could not run reads as `None`: unknown, which is not "off".
pub(crate) fn registered() -> Option<bool> {
    platform::registered()
}

/// Apply the owner's choice: remove everything, or (re)write the registration and
/// clear a system-side disable. Returns what the system reports afterwards.
pub(crate) fn set_enabled(enabled: bool) -> anyhow::Result<Option<bool>> {
    if enabled {
        let exe = std::env::current_exe().context("the running program's path is unknown")?;
        platform::enable(&exe)?;
    } else {
        platform::remove()?;
    }
    Ok(platform::registered())
}

/// The owner's decision: absence of the row means on.
fn decision() -> bool {
    crate::config::flag_on(crate::config::CONFIG.autolaunch().as_deref())
}

// ── The formats, shared by the platform arms and testable on any host ──────

/// The one identity the registration has on macOS: the launchd label, and the
/// name of the file that carries it.
#[cfg_attr(not(any(test, target_os = "macos")), expect(dead_code))]
const LABEL: &str = "com.mahbot.autostart";

/// `text` with the characters XML gives a meaning to escaped.
#[cfg_attr(not(any(test, target_os = "macos")), expect(dead_code))]
fn xml_escaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// The launchd agent that starts `exe` at login.
#[cfg_attr(not(any(test, target_os = "macos")), expect(dead_code))]
fn plist_text(exe: &Path) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{label}</string>
	<key>ProgramArguments</key>
	<array>
		<string>{program}</string>
		<string>{login}</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
"#,
        label = LABEL,
        program = xml_escaped(&exe.to_string_lossy()),
        login = crate::cli::LOGIN_START_ARG,
    )
}

/// Whether a `launchctl print-disabled` dump carries a disable for `label`. A
/// line of the form `"<label>" => disabled` is a disable; `=> enabled` or no line
/// at all is not.
#[cfg_attr(not(any(test, target_os = "macos")), expect(dead_code))]
fn launchctl_override_disabled(dump: &str, label: &str) -> bool {
    let quoted = format!("\"{label}\"");
    dump.lines().any(|line| {
        let Some((name, state)) = line.trim().split_once("=>") else {
            return false;
        };
        name.trim() == quoted && state.trim() == "disabled"
    })
}

/// The XDG autostart entry that starts `exe` at login.
///
/// A disabled entry carries both of the keys a desktop honours: `Hidden` is the
/// desktop-agnostic one the specification defines, and `X-GNOME-Autostart-enabled`
/// is GNOME's own.
#[cfg_attr(not(any(test, all(unix, not(target_os = "macos")))), expect(dead_code))]
fn desktop_entry_text(exe: &Path, enabled: bool) -> String {
    let mut text = format!(
        "[Desktop Entry]\nType=Application\nName=MahBot\nComment=Start MahBot at login\nExec={program} {login}\nTerminal=false\nX-GNOME-Autostart-enabled={state}\n",
        program = crate::util::desktop_entry::exec_argument(exe),
        login = crate::cli::LOGIN_START_ARG,
        state = if enabled { "true" } else { "false" },
    );
    if !enabled {
        text.push_str("Hidden=true\n");
    }
    text
}

/// Whether a desktop entry carries the owner's system-side disable. The entry is a
/// key file, so the spelling is free: the key and the value are trimmed, and the
/// values a desktop honours for these keys are read case-insensitively.
#[cfg_attr(not(any(test, all(unix, not(target_os = "macos")))), expect(dead_code))]
fn desktop_disabled(text: &str) -> bool {
    text.lines().any(|line| {
        let Some((key, value)) = line.split_once('=') else {
            return false;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "Hidden" => value.eq_ignore_ascii_case("true") || value == "1",
            "X-GNOME-Autostart-enabled" => value.eq_ignore_ascii_case("false") || value == "0",
            _ => false,
        }
    })
}

/// The `Run` value's data: the program path quoted, because profile names and
/// `%LOCALAPPDATA%` can contain spaces, followed by the login word.
#[cfg_attr(not(any(test, windows)), expect(dead_code))]
fn windows_command(exe: &Path) -> String {
    format!(
        "\"{}\" {}",
        exe.to_string_lossy(),
        crate::cli::LOGIN_START_ARG
    )
}

/// Whether the startup manager's record disables the entry: its first byte is
/// odd when disabled (`0x03`, and `0x07` where the system marks a later change)
/// and even when enabled (`0x02`, `0x06`). No value at all means enabled.
#[cfg_attr(not(any(test, windows)), expect(dead_code))]
fn startup_approved_disabled(record: &[u8]) -> bool {
    record.first().is_some_and(|flag| flag & 1 == 1)
}

/// The startup manager's record for an entry the owner has not disabled: the
/// "enabled" flag followed by a zeroed `FILETIME`. The system stamps a time only
/// when it disables an entry, and the product never disables one — it follows the
/// owner and removes the entry instead.
#[cfg_attr(not(any(test, windows)), expect(dead_code))]
const STARTUP_APPROVED_ENABLED: [u8; 12] = [0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Write `text` to `path` through a temp file in the same directory and a rename,
/// so a login never reads half a file. The mode is set explicitly: launchd refuses
/// an agent its group and the world can write to.
#[cfg(unix)]
fn write_atomic(path: &Path, text: &str) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = path
        .parent()
        .context("the registration's directory is unknown")?;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("{} could not be created", dir.display()))?;
    let tmp = dir.join(format!(".mahbot-{}.tmp", std::process::id()));
    let outcome = (|| {
        std::fs::write(&tmp, text)
            .with_context(|| format!("{} could not be written", tmp.display()))?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))
            .with_context(|| format!("{} could not be given its mode", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("{} could not be put in place", path.display()))
    })();
    if outcome.is_err() {
        // Nothing sweeps this directory, so a failed write must not leave the
        // temp file behind.
        let _ = std::fs::remove_file(&tmp);
    }
    outcome
}

/// Remove `path`, with an entry that is not there counting as removed already.
#[cfg(unix)]
fn remove_file_if_present(path: &Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("{} could not be removed", path.display())),
    }
}

// ── macOS: a launchd agent in the owner's own LaunchAgents ────────────────

#[cfg(target_os = "macos")]
mod platform {
    use std::path::{Path, PathBuf};

    use anyhow::Context as _;

    use super::{LABEL, launchctl_override_disabled, plist_text, remove_file_if_present};

    /// The owner's own uid: the per-user launchd domain and its override database
    /// are keyed by it.
    fn uid() -> u32 {
        // SAFETY: `getuid` takes no arguments and has no preconditions.
        unsafe { libc::getuid() }
    }

    /// The agent's file under `~/Library/LaunchAgents`.
    fn plist_path() -> anyhow::Result<PathBuf> {
        let home = directories::UserDirs::new()
            .context("the home directory is unknown")?
            .home_dir()
            .to_path_buf();
        Ok(home
            .join("Library")
            .join("LaunchAgents")
            .join(format!("{LABEL}.plist")))
    }

    pub(super) fn registered() -> Option<bool> {
        let path = plist_path().ok()?;
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                // An entry whose disable cannot be read is not known to start
                // anything.
                Some(!disabled_by_system()?)
            }
            // Nothing launchd can load there.
            Ok(_) => Some(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(false),
            // A registration that cannot be inspected is not known to start
            // anything.
            Err(_) => None,
        }
    }

    /// Write the agent unless it is already exactly right. The disable lives in
    /// launchd's override database rather than in the file, so a rewrite keeps it.
    pub(super) fn ensure(exe: &Path) -> anyhow::Result<()> {
        let path = plist_path()?;
        let text = plist_text(exe);
        if std::fs::read_to_string(&path).is_ok_and(|current| current == text) {
            return Ok(());
        }
        super::write_atomic(&path, &text)
    }

    pub(super) fn enable(exe: &Path) -> anyhow::Result<()> {
        let path = plist_path()?;
        super::write_atomic(&path, &plist_text(exe))?;
        clear_system_disable();
        // The switch reports what the system holds, so the clear is judged by the
        // database rather than by the command's own exit status. A database that
        // cannot be read at all is not a refusal to allow the login item.
        anyhow::ensure!(
            disabled_by_system() != Some(true),
            "macOS still has the login item disabled — allow MahBot in System Settings › General › Login Items"
        );
        Ok(())
    }

    pub(super) fn remove() -> anyhow::Result<()> {
        remove_file_if_present(&plist_path()?)
    }

    /// Whether launchd's override database disables `LABEL` — the state System
    /// Settings' "Allow in the Background" switch writes for a legacy agent. The
    /// database is per-user, so it is the same list whichever domain is named; the
    /// agent itself loads into the GUI one. `None` when the database cannot be
    /// read, which is not the same as "not disabled".
    fn disabled_by_system() -> Option<bool> {
        let output = std::process::Command::new("/bin/launchctl")
            .args(["print-disabled", &format!("gui/{}", uid())])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| launchctl_override_disabled(&String::from_utf8_lossy(&output.stdout), LABEL))
    }

    /// Clear the database's disable. Best effort: the caller judges the outcome by
    /// what the database then holds.
    fn clear_system_disable() {
        let result = std::process::Command::new("/bin/launchctl")
            .args(["enable", &format!("gui/{}/{LABEL}", uid())])
            .status();
        match result {
            Ok(status) if status.success() => {}
            Ok(status) => tracing::debug!(%status, "launchctl could not clear the login disable"),
            Err(e) => {
                tracing::debug!(error = %e, "launchctl could not run to clear the login disable");
            }
        }
    }
}

// ── Windows: the per-user Run value and the startup manager's record ──────

#[cfg(windows)]
mod platform {
    use std::path::Path;

    use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_CREATE_SUB_KEY, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_BINARY,
        REG_OPTION_NON_VOLATILE, REG_SAM_FLAGS, REG_SZ, REG_VALUE_TYPE, RegCloseKey,
        RegCreateKeyExW, RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    };

    use super::{STARTUP_APPROVED_ENABLED, startup_approved_disabled, windows_command};
    use crate::util::registry::{os_reason, utf16_bytes, utf16_text, wide};

    /// The per-user key the system starts a program from at login.
    const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
    /// The record the system's own startup manager reads and writes; the owner's
    /// disable lives here.
    const APPROVED_KEY: &str =
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run";
    /// The one value name the product uses, on both keys.
    const VALUE: &str = "MahBot";

    pub(super) fn registered() -> Option<bool> {
        match read(RUN_KEY, VALUE) {
            Ok(None) => Some(false),
            // A registration whose disable cannot be read is not known to start
            // anything.
            Ok(Some(_)) => disabled_by_system().ok().map(|disabled| !disabled),
            Err(_) => None,
        }
    }

    /// Point the `Run` value at `exe` when it does not already say so. The
    /// startup-manager record is not touched: that is where the owner's own
    /// disable lives.
    pub(super) fn ensure(exe: &Path) -> anyhow::Result<()> {
        let want = windows_command(exe);
        if read_string(RUN_KEY, VALUE)?.as_deref() != Some(want.as_str()) {
            write(RUN_KEY, VALUE, REG_SZ, &utf16_bytes(&want))?;
        }
        Ok(())
    }

    /// Write the `Run` value and an enabled startup-manager record — re-adding
    /// the value alone does not clear a disable made in Task Manager.
    pub(super) fn enable(exe: &Path) -> anyhow::Result<()> {
        write(RUN_KEY, VALUE, REG_SZ, &utf16_bytes(&windows_command(exe)))?;
        write(APPROVED_KEY, VALUE, REG_BINARY, &STARTUP_APPROVED_ENABLED)
    }

    /// Remove the `Run` value and the startup-manager record, so a later
    /// re-registration is not silently disabled. A value that is not there is
    /// not an error.
    pub(super) fn remove() -> anyhow::Result<()> {
        delete(RUN_KEY, VALUE)?;
        delete(APPROVED_KEY, VALUE)
    }

    fn disabled_by_system() -> anyhow::Result<bool> {
        Ok(read(APPROVED_KEY, VALUE)?.is_some_and(|record| startup_approved_disabled(&record)))
    }

    /// A value's raw bytes, or `None` when the key or the value is not there.
    fn read(key_path: &str, name: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let Some(key) = open(key_path, KEY_QUERY_VALUE)? else {
            return Ok(None);
        };
        let outcome = query(key, name);
        // SAFETY: `key` came from `open` and nothing uses it after this.
        unsafe { RegCloseKey(key) };
        outcome
    }

    fn read_string(key_path: &str, name: &str) -> anyhow::Result<Option<String>> {
        Ok(read(key_path, name)?.map(|data| utf16_text(&data)))
    }

    fn query(key: HKEY, name: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let name = wide(name);
        let mut kind: REG_VALUE_TYPE = 0;
        let mut size: u32 = 0;
        // SAFETY: a size query — no data pointer — as the API documents, with
        // `name` outliving the call and the two out-parameters ours.
        let status = unsafe {
            RegQueryValueExW(
                key,
                name.as_ptr(),
                std::ptr::null(),
                &raw mut kind,
                std::ptr::null_mut(),
                &raw mut size,
            )
        };
        if status == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if status != ERROR_SUCCESS {
            anyhow::bail!(os_reason(
                "a login registration value could not be read",
                status
            ));
        }
        let mut data = vec![0u8; size as usize];
        let mut read = size;
        // SAFETY: `data` holds `size` bytes and `read` tells the API how many of
        // them it may fill; `name` outlives the call.
        let status = unsafe {
            RegQueryValueExW(
                key,
                name.as_ptr(),
                std::ptr::null(),
                &raw mut kind,
                data.as_mut_ptr(),
                &raw mut read,
            )
        };
        if status != ERROR_SUCCESS {
            anyhow::bail!(os_reason(
                "a login registration value could not be read",
                status
            ));
        }
        data.truncate(read as usize);
        Ok(Some(data))
    }

    fn write(key_path: &str, name: &str, kind: REG_VALUE_TYPE, data: &[u8]) -> anyhow::Result<()> {
        let key = create(key_path)?;
        let outcome = store(key, name, kind, data);
        // SAFETY: `key` came from `create` and nothing uses it after this.
        unsafe { RegCloseKey(key) };
        outcome
    }

    fn open(key_path: &str, access: REG_SAM_FLAGS) -> anyhow::Result<Option<HKEY>> {
        let path = wide(key_path);
        let mut key: HKEY = 0;
        // SAFETY: `path` is a NUL-terminated UTF-16 string that outlives the call,
        // and `key` is ours to fill.
        let status =
            unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, path.as_ptr(), 0, access, &raw mut key) };
        match status {
            ERROR_SUCCESS => Ok(Some(key)),
            ERROR_FILE_NOT_FOUND => Ok(None),
            _ => anyhow::bail!(os_reason(
                "the login registration's key could not be opened",
                status
            )),
        }
    }

    fn create(key_path: &str) -> anyhow::Result<HKEY> {
        let path = wide(key_path);
        let mut key: HKEY = 0;
        let mut disposition = 0;
        // SAFETY: `path` is a NUL-terminated UTF-16 string that outlives the call,
        // `key` and `disposition` are ours to fill, and the default security
        // attribute is the process's own.
        let status = unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                path.as_ptr(),
                0,
                std::ptr::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_QUERY_VALUE | KEY_SET_VALUE | KEY_CREATE_SUB_KEY,
                std::ptr::null(),
                &raw mut key,
                &raw mut disposition,
            )
        };
        if status != ERROR_SUCCESS {
            anyhow::bail!(os_reason(
                "the login registration's key could not be opened",
                status
            ));
        }
        Ok(key)
    }

    fn store(key: HKEY, name: &str, kind: REG_VALUE_TYPE, data: &[u8]) -> anyhow::Result<()> {
        let name = wide(name);
        let Ok(size) = u32::try_from(data.len()) else {
            anyhow::bail!("a login registration value is too long to store");
        };
        // SAFETY: `name` and `data` are the value name and data the API reads, and
        // both outlive the call.
        let status = unsafe { RegSetValueExW(key, name.as_ptr(), 0, kind, data.as_ptr(), size) };
        if status != ERROR_SUCCESS {
            anyhow::bail!(os_reason(
                "a login registration value could not be stored",
                status
            ));
        }
        Ok(())
    }

    fn delete(key_path: &str, name: &str) -> anyhow::Result<()> {
        let Some(key) = open(key_path, KEY_SET_VALUE)? else {
            return Ok(());
        };
        let name = wide(name);
        // SAFETY: `name` is a NUL-terminated UTF-16 value name that outlives the
        // call and `key` is open for writing.
        let status = unsafe { RegDeleteValueW(key, name.as_ptr()) };
        // SAFETY: `key` came from `open` and nothing uses it after this.
        unsafe { RegCloseKey(key) };
        match status {
            ERROR_SUCCESS | ERROR_FILE_NOT_FOUND => Ok(()),
            _ => {
                anyhow::bail!(os_reason(
                    "a login registration value could not be removed",
                    status
                ))
            }
        }
    }
}

// ── Linux: an XDG autostart entry in the owner's config directory ─────────

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use std::path::{Path, PathBuf};

    use anyhow::Context as _;

    use super::{desktop_disabled, desktop_entry_text, remove_file_if_present};

    /// `~/.config/autostart/mahbot.desktop`.
    fn entry_path() -> anyhow::Result<PathBuf> {
        let config = directories::BaseDirs::new()
            .context("the config directory is unknown")?
            .config_dir()
            .to_path_buf();
        Ok(config.join("autostart").join("mahbot.desktop"))
    }

    pub(super) fn registered() -> Option<bool> {
        let path = entry_path().ok()?;
        match std::fs::read_to_string(&path) {
            Ok(text) => Some(!desktop_disabled(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(false),
            // An entry that cannot be read is not known to start anything.
            Err(_) => None,
        }
    }

    /// Point the entry at `exe`, preserving a disable the owner made in the
    /// system's own settings.
    pub(super) fn ensure(exe: &Path) -> anyhow::Result<()> {
        let path = entry_path()?;
        let current = std::fs::read_to_string(&path).ok();
        let enabled = current
            .as_deref()
            .is_none_or(|text| !desktop_disabled(text));
        let text = desktop_entry_text(exe, enabled);
        if current.as_deref() == Some(text.as_str()) {
            return Ok(());
        }
        super::write_atomic(&path, &text)
    }

    pub(super) fn enable(exe: &Path) -> anyhow::Result<()> {
        let path = entry_path()?;
        super::write_atomic(&path, &desktop_entry_text(exe, true))
    }

    pub(super) fn remove() -> anyhow::Result<()> {
        remove_file_if_present(&entry_path()?)
    }
}

// ── Tests for the formats, runnable on any host ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plist_names_the_program_the_label_and_the_login_word() {
        let text = plist_text(Path::new("/Applications/A & B/<mahbot>"));
        assert!(text.contains(&format!("<string>{LABEL}</string>")));
        assert!(text.contains("<string>/Applications/A &amp; B/&lt;mahbot&gt;</string>"));
        assert!(text.contains(&format!("<string>{}</string>", crate::cli::LOGIN_START_ARG)));
    }

    #[test]
    fn the_desktop_entry_points_at_the_program_and_reads_a_disable() {
        let text = desktop_entry_text(Path::new("/home/o/.local/bin/mahbot"), true);
        assert!(text.contains("Exec=\"/home/o/.local/bin/mahbot\" __login-start"));
        assert!(text.contains("X-GNOME-Autostart-enabled=true"));
        assert!(!desktop_disabled(&text));

        let disabled = desktop_entry_text(Path::new("/opt/mahbot"), false);
        // The desktop-agnostic disable key, not only GNOME's.
        assert!(disabled.contains("Hidden=true"));
        assert!(desktop_disabled(&disabled));
        assert!(desktop_disabled("[Desktop Entry]\nHidden=true\n"));
        assert!(desktop_disabled("X-GNOME-Autostart-enabled=false\n"));
        // A key file's spelling is free: case, spacing, and the numeric forms.
        assert!(desktop_disabled("Hidden=True\n"));
        assert!(desktop_disabled("Hidden = 1\n"));
        assert!(desktop_disabled("X-GNOME-Autostart-enabled=0\n"));
        // A comment, or an explicit enable, is not a disable.
        assert!(!desktop_disabled("#Hidden=true\n"));
        assert!(!desktop_disabled("X-GNOME-Autostart-enabled=true\n"));

        // A literal `%` in the path is a field code unless it is doubled.
        assert!(
            desktop_entry_text(Path::new("/opt/100%/mahbot"), true)
                .contains("Exec=\"/opt/100%%/mahbot\"")
        );
    }

    #[test]
    fn a_launchctl_dump_reads_one_labels_override() {
        let dump = "\
disabled services = {
\t\"com.mahbot.autostart\" => disabled
\t\"com.apple.other\" => enabled
}
";
        assert!(launchctl_override_disabled(dump, "com.mahbot.autostart"));
        assert!(!launchctl_override_disabled(dump, "com.apple.other"));
        assert!(!launchctl_override_disabled(dump, "com.mahbot.absent"));
    }

    #[test]
    fn the_windows_command_quotes_the_program_path() {
        assert_eq!(
            windows_command(Path::new(r"C:\Program Files\MahBot\mahbot.exe")),
            r#""C:\Program Files\MahBot\mahbot.exe" __login-start"#
        );
    }

    #[test]
    fn the_startup_approved_record_reads_as_the_system_would() {
        // The record this product writes is the enabled one.
        assert!(!startup_approved_disabled(&STARTUP_APPROVED_ENABLED));
        // The flag the system writes when the owner disables the entry there, and
        // the one it stamps on a later change; no record at all means enabled.
        assert!(startup_approved_disabled(&[0x03]));
        assert!(startup_approved_disabled(&[0x07]));
        assert!(!startup_approved_disabled(&[]));
    }

    /// A failed rename must not leave the temp file behind in a directory nothing
    /// sweeps.
    #[cfg(unix)]
    #[test]
    fn a_failed_write_leaves_no_temp_file() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = std::env::temp_dir().join(format!("mahbot-autostart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let path = dir.join("entry");
        write_atomic(&path, "hello").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o644);

        // A target that is a directory makes the rename fail.
        let blocked = dir.join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        assert!(write_atomic(&blocked, "hello").is_err());
        assert!(
            !dir.join(format!(".mahbot-{}.tmp", std::process::id()))
                .exists()
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

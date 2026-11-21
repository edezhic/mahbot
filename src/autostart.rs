//! Start at login: the product's own registration with the operating system.
//!
//! A host gets one registration, and the installer is the one that picks it: a Linux host with
//! no graphical system gets the system unit ([`SYSTEM_UNIT`]), every other host starts the
//! product at login. This module owns the login side of that choice and follows the
//! installer's where it made one — with the unit in place it removes the login entry and never
//! writes one, because the two would start two copies. A unit the owner masked counts as in place,
//! in either form — a link to `/dev/null`, or one that points at nothing, in the unit's own place or
//! only for the running session: the owner said this service must not run, so the product neither
//! registers at login nor starts there.
//!
//! Each platform keeps one fixed identity, so a start after an install, an update or a move
//! re-points the same entry rather than adding a second one. The owner's decision is
//! `crate::config`'s (`autolaunch`), and it outlives installs. A disable made in the system's own
//! settings is real state here — a login entry disabled there, or the installer's unit disabled
//! with `systemctl` — it reads as "not registered", and nothing in the product overrides it on its
//! own.

use std::path::Path;

use anyhow::Context as _;

/// The system unit the installer registers on a Linux host with no graphical system. Root's to
/// write and remove, and the one registration such a host gets.
const SYSTEM_UNIT: &str = "/etc/systemd/system/mahbot.service";

/// The name systemd knows the unit by — the word every command that reaches it carries.
const UNIT_NAME: &str = "mahbot.service";

/// The symlink systemd enables a unit with. The file alone starts nothing: a host whose unit was
/// `systemctl disable`d keeps it and boots the product no more, so this — not the file — is what
/// "registered" means.
const SYSTEM_UNIT_ENABLED: &str = "/etc/systemd/system/multi-user.target.wants/mahbot.service";

/// Where `systemctl mask --runtime` leaves its link: a mask that lives for the running session only,
/// and gone at the next boot.
const SYSTEM_UNIT_RUNTIME: &str = "/run/systemd/system/mahbot.service";

/// Whether the installer left its unit on this host at all — a mask included, since the owner who
/// wrote one said the unit is not to run. It answers which registration the host has (the
/// installer's choice), not whether that registration is enabled ([`system_unit_enabled`]). Always
/// `false` off Linux, where there is no such unit to respect.
pub(crate) fn system_unit_installed() -> bool {
    cfg!(target_os = "linux")
        && unit_installed(Path::new(SYSTEM_UNIT), Path::new(SYSTEM_UNIT_RUNTIME))
}

/// Whether systemd is set to start the installer's unit at boot.
pub(crate) fn system_unit_enabled() -> bool {
    cfg!(target_os = "linux") && Path::new(SYSTEM_UNIT_ENABLED).exists()
}

/// The mask on this host's unit, when the owner put one there — the one state in which the product
/// must not register itself at login at all, whatever its own decision says.
pub(crate) fn system_unit_mask() -> Option<Mask> {
    if cfg!(target_os = "linux") {
        host_mask(Path::new(SYSTEM_UNIT), Path::new(SYSTEM_UNIT_RUNTIME))
    } else {
        None
    }
}

/// What a masked unit means for start-at-login — the one sentence the switch's hint, the read-back
/// and the write's refusal all state.
const MASKED_AT_LOGIN: &str = "nothing registers at login while the mask stays";

/// [`MASKED_AT_LOGIN`] with the command that undoes `mask`, for every place that can name one.
pub(crate) fn masked_at_login(mask: Mask) -> String {
    format!("{MASKED_AT_LOGIN}; run `{}` to undo it", mask.undo())
}

/// Whether the host has the installer's unit at all, over the paths it is read from — so a test can
/// exercise it on files it made instead of on a host's own.
fn unit_installed(unit: &Path, runtime: &Path) -> bool {
    unit.is_file() || host_mask(unit, runtime).is_some()
}

/// The mask the host carries, and what it is made of — each form is undone by a different command.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Mask {
    /// A link to `/dev/null` in the unit's own place, which the service manager knows as a mask.
    DevNull,
    /// The same, under `/run` for the running session only.
    DevNullRuntime,
    /// A link to nothing in the unit's own place: the manager calls it `not-found`, and
    /// `systemctl unmask` leaves it where it is.
    Dangling,
    /// The same, under `/run` for the running session only.
    DanglingRuntime,
}

impl Mask {
    /// The command that removes this mask, as the host can run it.
    pub(crate) fn undo(self) -> String {
        match self {
            Self::DevNull => format!("sudo systemctl unmask {UNIT_NAME}"),
            Self::DevNullRuntime => format!("sudo systemctl unmask --runtime {UNIT_NAME}"),
            Self::Dangling => format!("sudo rm {SYSTEM_UNIT}"),
            Self::DanglingRuntime => format!("sudo rm {SYSTEM_UNIT_RUNTIME}"),
        }
    }
}

/// The mask the host carries, if any: the one in the unit's own place, or — where nothing else is
/// there, since systemd reads that path first — one that lives only for the running session. A unit
/// file, a directory, an alias link and a broken unit are not masks.
fn host_mask(unit: &Path, runtime: &Path) -> Option<Mask> {
    if let Some(mask) = mask_at(unit, false) {
        return Some(mask);
    }
    if nothing_at(unit) {
        return mask_at(runtime, true);
    }
    None
}

/// The mask `path` is — the unit's own place when `runtime` is false, the running session's when it
/// is true — or `None` when it is not one.
fn mask_at(path: &Path, runtime: bool) -> Option<Mask> {
    // A link's target names its own directory when it is relative, as it does for systemd.
    let target = std::fs::read_link(path).ok()?;
    let target = match path.parent() {
        Some(dir) => dir.join(target),
        None => target,
    };
    if std::fs::canonicalize(&target).is_ok_and(|real| real == Path::new("/dev/null")) {
        Some(if runtime {
            Mask::DevNullRuntime
        } else {
            Mask::DevNull
        })
    } else if nothing_at(&target) {
        Some(if runtime {
            Mask::DanglingRuntime
        } else {
            Mask::Dangling
        })
    } else {
        // A link to a unit that is there is an alias, and one that cannot be resolved for another
        // reason — a loop, a directory this process may not read — is a broken unit, not a mask.
        None
    }
}

/// Whether nothing at all is at `path` — not a unit, not a link, not a directory.
fn nothing_at(path: &Path) -> bool {
    matches!(
        std::fs::symlink_metadata(path),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound
    )
}

/// Keep the system's registration true to the installer's choice and to the running binary.
/// Called at every start; a no-op for a copy built from sources.
///
/// With the installer's unit in place — masked included — the login entry is only ever removed:
/// enabled or not, the host's registration is the unit, and writing an entry beside it would start
/// two copies. An entry that was already there when the mask went on may start one copy before this
/// removes it, which is accepted; a masked host is told in the log why start-at-login stops working.
pub fn reconcile() {
    if !crate::self_update::is_published_copy() {
        return;
    }
    if decision()
        && let Some(mask) = system_unit_mask()
    {
        tracing::info!(
            "MahBot's service unit is masked on this host: {}",
            masked_at_login(mask)
        );
    }
    let outcome = if system_unit_installed() || !decision() {
        platform::remove()
    } else {
        std::env::current_exe()
            .map_err(anyhow::Error::from)
            .and_then(|exe| platform::ensure(&exe))
    };
    if let Err(e) = outcome {
        tracing::warn!(error = %e, "the login registration could not be kept true");
    }
}

/// What a read of the login registration answered: the OS's own answer, and the mask that overrides
/// it — `None` for each = unknown, never "off".
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoginRead {
    pub(crate) registered: Option<bool>,
    pub(crate) mask: Option<Mask>,
}

/// Whether the system reports a registration that will start the product at login, read with the
/// mask that overrides it so one look at the host answers both. An entry the owner disabled reads as
/// `false`, a masked unit reads as `false` (nothing starts under a mask) whatever a stale enablement
/// says, and a probe that could not run as `None`.
pub(crate) fn read_login() -> LoginRead {
    let mask = system_unit_mask();
    LoginRead {
        registered: if mask.is_some() {
            Some(false)
        } else if system_unit_enabled() {
            Some(true)
        } else {
            platform::registered()
        },
        mask,
    }
}

/// Apply the owner's choice: remove everything, or (re)write the registration and clear a
/// system-side disable. Returns what the system reports afterwards.
///
/// Where the installer left its unit, the start is the service manager's to set — the switch
/// cannot enable a disabled unit (that needs root) and must not claim it did, so both answers
/// name the command that does it. A masked unit is refused first: no enable or disable takes
/// effect on one, and nothing registers at login while the mask stays.
pub(crate) fn set_enabled(enabled: bool) -> anyhow::Result<Option<bool>> {
    if let Some(mask) = system_unit_mask() {
        anyhow::bail!(
            "MahBot's service unit is masked on this host, so {}",
            masked_at_login(mask)
        );
    }
    if system_unit_installed() {
        anyhow::bail!(
            "MahBot is registered as a system service, and its start is the service manager's to \
             set — run `sudo systemctl {} mahbot.service`",
            if enabled {
                "enable --now"
            } else {
                "disable --now"
            }
        );
    }
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

/// The one identity the registration has on macOS: the launchd label, and the name of the
/// file that carries it.
#[cfg_attr(not(any(test, target_os = "macos")), expect(dead_code))]
const LABEL: &str = "com.mahbot.autostart";

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

/// Whether a `launchctl print-disabled` dump carries a disable for `label`: a line of the form
/// `"<label>" => disabled` is a disable; `=> enabled` or no line is not.
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

/// The XDG autostart entry that starts `exe` at login. A disabled entry carries both keys a
/// desktop honours: `Hidden` (desktop-agnostic) and `X-GNOME-Autostart-enabled`.
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

/// Whether a desktop entry carries the owner's system-side disable. The entry is a key file, so
/// the spelling is free: key and value are trimmed, and the values read case-insensitively.
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

/// Whether the startup manager's record disables the entry: its first byte is odd when disabled
/// (`0x03`, `0x07`) and even when enabled (`0x02`, `0x06`). No value at all means enabled.
#[cfg_attr(not(any(test, windows)), expect(dead_code))]
fn startup_approved_disabled(record: &[u8]) -> bool {
    record.first().is_some_and(|flag| flag & 1 == 1)
}

/// The startup manager's record for an entry the owner has not disabled: the "enabled" flag
/// followed by a zeroed `FILETIME`. The system stamps a time only when it disables an entry,
/// and the product never disables one — it removes it instead.
#[cfg_attr(not(any(test, windows)), expect(dead_code))]
const STARTUP_APPROVED_ENABLED: [u8; 12] = [0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Write `text` to `path` through a temp file in the same directory and a rename, so a login
/// never reads half a file. The mode is set explicitly: launchd refuses an agent its group and
/// the world can write to.
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
        // Nothing sweeps this directory, so a failed write must not leave the temp file behind.
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

#[cfg(target_os = "macos")]
mod platform {
    use std::path::{Path, PathBuf};

    use anyhow::Context as _;

    use super::{LABEL, launchctl_override_disabled, plist_text, remove_file_if_present};

    /// The owner's own uid: the per-user launchd domain and its override database are keyed by it.
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
                // An entry whose disable cannot be read is not known to start anything.
                Some(!disabled_by_system()?)
            }
            // Nothing launchd can load there.
            Ok(_) => Some(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(false),
            // A registration that cannot be inspected is not known to start anything.
            Err(_) => None,
        }
    }

    /// Write the agent unless it is already exactly right. The disable lives in launchd's
    /// override database rather than in the file, so a rewrite keeps it.
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
        // The switch reports what the system holds, so the clear is judged by the database
        // rather than by the exit status. An unreadable database is not a refusal to allow the
        // login item.
        anyhow::ensure!(
            disabled_by_system() != Some(true),
            "macOS still has the login item disabled — allow MahBot in System Settings › General › Login Items"
        );
        Ok(())
    }

    pub(super) fn remove() -> anyhow::Result<()> {
        remove_file_if_present(&plist_path()?)
    }

    /// Whether launchd's override database disables `LABEL` — the state System Settings'
    /// "Allow in the Background" switch writes. The database is per-user, so it is the same
    /// list whichever domain is named. `None` when it cannot be read, which is not "not disabled".
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

    /// Clear the database's disable. Best effort: the caller judges by what the database then
    /// holds.
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
    /// The record the system's own startup manager reads and writes; the owner's disable lives
    /// here.
    const APPROVED_KEY: &str =
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run";
    /// The one value name the product uses, on both keys.
    const VALUE: &str = "MahBot";

    pub(super) fn registered() -> Option<bool> {
        match read(RUN_KEY, VALUE) {
            Ok(None) => Some(false),
            // A registration whose disable cannot be read is not known to start anything.
            Ok(Some(_)) => disabled_by_system().ok().map(|disabled| !disabled),
            Err(_) => None,
        }
    }

    /// Point the `Run` value at `exe` when it does not already say so. The startup-manager
    /// record is not touched: that is where the owner's own disable lives.
    pub(super) fn ensure(exe: &Path) -> anyhow::Result<()> {
        let want = windows_command(exe);
        if read_string(RUN_KEY, VALUE)?.as_deref() != Some(want.as_str()) {
            write(RUN_KEY, VALUE, REG_SZ, &utf16_bytes(&want))?;
        }
        Ok(())
    }

    /// Write the `Run` value and an enabled startup-manager record — re-adding the value alone
    /// does not clear a disable made in Task Manager.
    pub(super) fn enable(exe: &Path) -> anyhow::Result<()> {
        write(RUN_KEY, VALUE, REG_SZ, &utf16_bytes(&windows_command(exe)))?;
        write(APPROVED_KEY, VALUE, REG_BINARY, &STARTUP_APPROVED_ENABLED)
    }

    /// Remove the `Run` value and the startup-manager record, so a later re-registration is not
    /// silently disabled. A value that is not there is not an error.
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
        let mut key: HKEY = std::ptr::null_mut();
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
        let mut key: HKEY = std::ptr::null_mut();
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

    /// Point the entry at `exe`, preserving a disable the owner made in the system's own settings.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The installer's own text, which the tripwires below read: what it writes for a unit, how it
    /// decides the registration and where the login entry lives are all stated there.
    fn install_script() -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh"))
            .expect("install.sh is part of the tree")
    }

    /// The two files that register the product with the system name the program and the login
    /// word, and nothing a parser would misread: the escape each format needs is applied to a
    /// path that needs every one of them.
    #[test]
    fn the_platform_files_name_the_program_and_the_login_word() {
        let plist = plist_text(Path::new("/Applications/A & B/<mahbot>"));
        assert!(plist.contains(&format!("<string>{LABEL}</string>")));
        assert!(plist.contains("<string>/Applications/A &amp; B/&lt;mahbot&gt;</string>"));
        assert!(plist.contains(&format!("<string>{}</string>", crate::cli::LOGIN_START_ARG)));

        let entry = desktop_entry_text(Path::new("/home/o/.local/bin/mahbot"), true);
        assert!(entry.contains("Exec=\"/home/o/.local/bin/mahbot\" __login-start"));
        assert!(entry.contains("X-GNOME-Autostart-enabled=true"));
        assert!(!desktop_disabled(&entry));

        // A literal `%` in the path is a field code unless it is doubled.
        assert!(
            desktop_entry_text(Path::new("/opt/100%/mahbot"), true)
                .contains("Exec=\"/opt/100%%/mahbot\"")
        );

        // The Windows `Run` value's data is the program path quoted — a profile name and
        // `%LOCALAPPDATA%` can contain spaces — and the login word after it.
        assert_eq!(
            windows_command(Path::new(r"C:\Program Files\MahBot\mahbot.exe")),
            r#""C:\Program Files\MahBot\mahbot.exe" __login-start"#
        );
    }

    /// A registration the owner disabled in the system's own settings — the desktop entry's
    /// disable keys, launchd's override dump, and the startup-approved record — reads as
    /// disabled, and a plain enable, a comment or no record does not. Each platform's own
    /// spelling is what the product must read back.
    #[test]
    fn a_startup_entry_the_owner_disabled_reads_as_disabled() {
        let disabled_entry = desktop_entry_text(Path::new("/opt/mahbot"), false);
        // The desktop-agnostic disable key, not only GNOME's.
        assert!(disabled_entry.contains("Hidden=true"));
        assert!(desktop_disabled(&disabled_entry));
        assert!(desktop_disabled("[Desktop Entry]\nHidden=true\n"));
        assert!(desktop_disabled("X-GNOME-Autostart-enabled=false\n"));
        // A key file's spelling is free: case, spacing, and the numeric forms.
        assert!(desktop_disabled("Hidden=True\n"));
        assert!(desktop_disabled("Hidden = 1\n"));
        assert!(desktop_disabled("X-GNOME-Autostart-enabled=0\n"));
        // A comment, or an explicit enable, is not a disable.
        assert!(!desktop_disabled("#Hidden=true\n"));
        assert!(!desktop_disabled("X-GNOME-Autostart-enabled=true\n"));

        let dump = "\
disabled services = {
\t\"com.mahbot.autostart\" => disabled
\t\"com.apple.other\" => enabled
}
";
        assert!(launchctl_override_disabled(dump, "com.mahbot.autostart"));
        assert!(!launchctl_override_disabled(dump, "com.apple.other"));
        assert!(!launchctl_override_disabled(dump, "com.mahbot.absent"));

        // The record this product writes is the enabled one; the flag the system writes on a
        // disable, and the one it stamps on a later change, both read as disabled. No record
        // means enabled.
        assert!(!startup_approved_disabled(&STARTUP_APPROVED_ENABLED));
        assert!(startup_approved_disabled(&[0x03]));
        assert!(startup_approved_disabled(&[0x07]));
        assert!(!startup_approved_disabled(&[]));
    }

    /// The two unit facts answer two different questions, and both are the installer's own: the
    /// file says which registration the host has, the wants symlink says whether systemd starts it.
    /// Both paths are built from the installer's own text, so a change on either side fails here
    /// instead of landing on an operator's host.
    #[test]
    fn the_system_unit_paths_are_the_installers_own() {
        let installer = install_script();
        let assigned = |prefix: &str| -> String {
            installer
                .lines()
                .find_map(|line| line.trim().strip_prefix(prefix))
                .unwrap_or_else(|| panic!("install.sh must set {prefix}"))
                .trim_matches('"')
                .to_string()
        };
        let name = assigned("UNIT_NAME=");
        let path = assigned("UNIT_PATH=").replace("$UNIT_NAME", &name);
        // The unmask commands name the unit by this word, so it must be the installer's own too.
        assert_eq!(
            UNIT_NAME, name,
            "the product must name the unit as the installer does"
        );
        assert_eq!(
            SYSTEM_UNIT, path,
            "the product must look for the unit where the installer writes it"
        );
        // systemd enables a unit through the wants symlink of the target its [Install] section
        // names, which is why the product reads that symlink rather than the unit file.
        let target = assigned("WantedBy=");
        assert_eq!(
            SYSTEM_UNIT_ENABLED,
            format!("/etc/systemd/system/{target}.wants/{name}"),
            "the product must read the wants symlink of the target the unit is wanted by"
        );
        // A mask for the running session only is read from `/run`, which is the path the installer
        // reports to its owner — one location, or the product would look where no mask is.
        assert!(
            installer.contains("UNIT_KEPT_AT=\"/run/systemd/system/$UNIT_NAME\""),
            "the installer must read a runtime mask from the one place systemd writes it"
        );
        assert_eq!(
            SYSTEM_UNIT_RUNTIME,
            format!("/run/systemd/system/{name}"),
            "the product must look for a runtime mask where the installer reads one"
        );
    }

    /// A mask is the owner's decision that the service must not run, and a plain file check reads it
    /// as "no unit here" — so the product must count it as the host's registration. Only a mask
    /// does: a broken unit, a directory at the unit's path and a unit file beside a runtime mask all
    /// keep the treatment they had, and each mask form is undone by the command that reaches it.
    #[cfg(unix)]
    #[test]
    fn a_masked_unit_counts_as_installed_and_nothing_else_does() {
        use std::os::unix::fs::symlink;

        let dir = crate::util::test::test_root().join("autostart-mask");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let unit = dir.join("mahbot.service");
        let runtime = dir.join("run-mahbot.service");

        // Nothing at either path: the unit was removed entirely, so the host has no service.
        assert!(!unit_installed(&unit, &runtime));
        assert!(host_mask(&unit, &runtime).is_none());

        // A unit file is the installer's registration, and no mask.
        std::fs::write(&unit, "[Unit]\n").unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert!(host_mask(&unit, &runtime).is_none());

        // The mask `systemctl mask` writes, where the unit was.
        std::fs::remove_file(&unit).unwrap();
        symlink("/dev/null", &unit).unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert_eq!(
            host_mask(&unit, &runtime).map(Mask::undo).as_deref(),
            Some("sudo systemctl unmask mahbot.service")
        );

        // A link spelled relative to its own directory reaches the same `/dev/null`, and is the same
        // statement.
        std::fs::create_dir_all(dir.join("dev")).unwrap();
        symlink("/dev/null", dir.join("dev").join("null")).unwrap();
        std::fs::remove_file(&unit).unwrap();
        symlink("dev/null", &unit).unwrap();
        assert!(host_mask(&unit, &runtime).is_some());

        // A link whose target is gone is the same statement, and systemd does not know it as a
        // mask: the link is removed as the file it is, not unmasked.
        std::fs::remove_file(&unit).unwrap();
        symlink(dir.join("gone.service"), &unit).unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert_eq!(
            host_mask(&unit, &runtime).map(Mask::undo).as_deref(),
            Some("sudo rm /etc/systemd/system/mahbot.service")
        );

        // A mask for the running session only, with no unit file behind it, is undone by the form
        // that names it.
        std::fs::remove_file(&unit).unwrap();
        symlink("/dev/null", &runtime).unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert_eq!(
            host_mask(&unit, &runtime).map(Mask::undo).as_deref(),
            Some("sudo systemctl unmask --runtime mahbot.service")
        );

        // A unit file beside a runtime mask is the unit systemd loads, so it is not a mask.
        std::fs::write(&unit, "[Unit]\n").unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert!(host_mask(&unit, &runtime).is_none());

        // A link to a unit file that is there is an alias, not a mask.
        std::fs::remove_file(&unit).unwrap();
        std::fs::write(dir.join("real.service"), "[Unit]\n").unwrap();
        symlink(dir.join("real.service"), &unit).unwrap();
        assert!(unit_installed(&unit, &runtime));
        assert!(host_mask(&unit, &runtime).is_none());

        // A directory at the unit's path is neither a unit nor a mask, exactly as before — and it is
        // not made one by a runtime mask beside it.
        std::fs::remove_file(&unit).unwrap();
        std::fs::create_dir(&unit).unwrap();
        assert!(!unit_installed(&unit, &runtime));
        assert!(host_mask(&unit, &runtime).is_none());

        // Both masks at once: systemd reads the unit's own place first, so that is the one named.
        std::fs::remove_dir(&unit).unwrap();
        symlink("/dev/null", &unit).unwrap();
        assert_eq!(
            host_mask(&unit, &runtime).map(Mask::undo).as_deref(),
            Some("sudo systemctl unmask mahbot.service")
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A unit systemd refuses leaves the host registered and unstarted, and the shipped defect —
    /// `WorkingDirectory="$HOME"`, which systemd does not unquote — did exactly that while
    /// `systemctl enable` still succeeded. The stop timeout is the product's own drain cap, so a
    /// drift there would either cut a drain short or hold a stop open long after the work is gone.
    #[test]
    fn the_unit_the_installer_writes_is_one_the_service_manager_will_load() {
        let installer = install_script();
        assert!(
            installer.contains("WorkingDirectory=$HOME"),
            "the unit's working directory must be an unquoted path systemd accepts"
        );
        assert!(
            !installer.contains("WorkingDirectory=\""),
            "a quoted working directory is not an absolute path to systemd, and it refuses the unit"
        );
        assert!(
            installer.contains("TimeoutStopSec=$STOP_TIMEOUT_SECS"),
            "the unit's stop timeout must come from the one variable"
        );
        assert!(
            !installer.contains("TimeoutStopSec=900"),
            "a literal stop timeout would drift from the product's drain cap"
        );
        // The value is the drain cap plus an allowance for the closing steps after it, so the manager
        // never ends a stop before the drain has had the time the product allows it. The allowance is
        // whatever the script names it, as long as there is one.
        assert!(
            installer.contains("STOP_TIMEOUT_SECS=$((DRAIN_CAP_SECS + ")
                && !installer.contains("STOP_TIMEOUT_SECS=$((DRAIN_CAP_SECS + 0))"),
            "the unit's stop timeout must be the product's own drain cap plus a closing allowance"
        );
        let drain_cap = installer
            .lines()
            .find_map(|line| line.trim().strip_prefix("DRAIN_CAP_SECS="))
            .expect("install.sh must set DRAIN_CAP_SECS")
            .trim()
            .parse::<u64>()
            .expect("DRAIN_CAP_SECS is a number of seconds");
        assert_eq!(
            drain_cap,
            crate::jobs::DRAIN_CAP_SECS,
            "the unit's stop timeout must not drift from the product's drain cap"
        );
    }

    /// The default target is not evidence of a graphical login — stock Ubuntu answers
    /// `graphical.target` with no desktop at all — so the installer chooses from the display
    /// manager's own files. A regression to `systemctl get-default`, or a login entry derived
    /// differently from the product's, would put servers back on the wrong registration.
    #[test]
    fn the_registration_is_chosen_from_a_display_manager_not_the_default_target() {
        let installer = install_script();
        assert!(
            installer.contains("display-manager.service"),
            "the installer must read the display manager's own alias"
        );
        assert!(
            installer.contains("MAHBOT_INSTALL_REGISTRATION"),
            "the registration must be overridable explicitly"
        );
        assert!(
            !installer.contains("get-default") && !installer.contains("set-default"),
            "the systemd default target decides nothing about a graphical login"
        );
    }

    /// A message that reports a unit's state prints the service manager's own word and offers a
    /// command only where one changes the answer — a mask removed in the form that removes that very
    /// mask, and a disabled unit enabled. The shipped installer called every state but `enabled` one
    /// that starts nothing at boot, wrong for statically enabled and generator-enabled units.
    ///
    /// A command a message prints carries the same privilege framing as the rest of the installer, so
    /// a host that needs `sudo` is never handed a command it cannot run.
    #[test]
    fn the_installer_reports_unit_states_it_can_act_on_and_frames_every_command() {
        let installer = install_script();
        assert!(
            !installer.contains("nothing would start it at boot")
                && !installer.contains("nothing starts it at boot")
                && !installer.contains("starts nothing at boot"),
            "a state other than 'enabled' is not evidence that nothing starts the unit"
        );
        // Both families of state — `is-enabled` and `ActiveState` — go through one word for "the
        // manager said nothing", or the same output would paraphrase the same fact two ways.
        assert!(
            !installer.contains("'$UNIT_ENABLED_BEFORE'") && !installer.contains("no state for it"),
            "a state is printed through unit_state_word, never paraphrased"
        );
        assert!(
            installer.contains("systemctl unmask --runtime $UNIT_NAME"),
            "a runtime mask is removed by the form that names it"
        );
        assert!(
            installer.contains("did not start or enable it"),
            "a state no command changes gets the neutral sentence, not a remedy"
        );
        assert!(
            installer.contains("${AS_ROOT_SAYS}${AS_ROOT}tee $UNIT_PATH")
                && !installer.contains("remove it as root:")
                && !installer.contains("Save this as $UNIT_PATH as root"),
            "a printed command carries the framing the rest of the installer uses"
        );
        assert!(
            installer.contains("stop $UNIT_NAME, then run this installer again")
                && installer.contains("kill $LEFTOVER, then run this installer again"),
            "a printed command is separated from the sentence after it, not run together"
        );
    }

    /// The product resolves the login entry through `directories::BaseDirs::config_dir` — an absolute
    /// `$XDG_CONFIG_HOME`, else `$HOME/.config` — and the installer must land on the same file, or the
    /// service leg removes an entry that is not the one in use. The rule itself is pinned here; the
    /// two paths are exercised against a real install by `scripts/install-acceptance.sh`.
    #[test]
    fn the_installer_resolves_the_login_entry_the_way_the_product_does() {
        let installer = install_script();
        assert!(
            installer.contains("AUTOSTART_DIR=\"$HOME/.config\"")
                && installer.contains("case \"${XDG_CONFIG_HOME:-}\" in")
                && installer.contains("AUTOSTART_DIR=$XDG_CONFIG_HOME"),
            "the entry's directory must be an absolute XDG_CONFIG_HOME, else $HOME/.config"
        );
        // Every trailing separator goes, and the entry is joined onto what is left: a doubled
        // separator is a different string from the one the product computes.
        assert!(
            installer.contains("while [ \"${AUTOSTART_DIR%/}\" != \"$AUTOSTART_DIR\" ]; do"),
            "a trailing separator must be dropped, as the product's own path join drops it"
        );
        assert!(
            installer.contains("AUTOSTART_ENTRY=\"$AUTOSTART_DIR/autostart/mahbot.desktop\""),
            "the entry must be joined onto the stripped directory, with one separator"
        );
    }

    /// A failed rename must not leave the temp file behind in a directory nothing sweeps.
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

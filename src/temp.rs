//! Accepted limits: junk an older build's hard-coded temp variable left in the machine-wide system
//! temp directory is never reclaimed (it belongs to every user, so it is not a scan root); and the
//! free-space reading is live on Windows for the first time, so the store-shrinking gate can skip a
//! TRUNCATE checkpoint on a nearly full disk and this cadence can reach its daily mode. A caller
//! capped below the absolute floor (a small per-user quota) stays in daily mode — accepted, the
//! schedule is policy.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crate::Workspace;
use crate::util::UnwrapPoison;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use futures_util::FutureExt;

/// The pinned private temp root, set once by [`init_temp_root`].
static TEMP_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The pre-pin OS temp dir (the darwin user temp dir on macOS, the user's own temp directory on
/// Windows), captured before the env pin so the readonly guard and the cleaner keep it as a root of
/// their own: bare macOS `mktemp` ignores `TMPDIR` and lands there.
static LEGACY_TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The temp environment variable names this platform uses, in precedence order — the
/// module-internal source of [`shell_temp_vars`]. Keyed on `windows` rather than `unix` (the axis
/// the root derivation uses): the per-user-temp story is the Windows one, so any other target keeps
/// the historical `TMPDIR`.
#[cfg(windows)]
const TEMP_ENV_VARS: &[&str] = &["TMP", "TEMP"];
#[cfg(not(windows))]
const TEMP_ENV_VARS: &[&str] = &["TMPDIR"];

/// The pinned root path, if [`init_temp_root`] ran.
#[must_use]
fn temp_root() -> Option<&'static Path> {
    TEMP_ROOT.get().map(PathBuf::as_path)
}

/// The pre-pin OS temp dir on EVERY platform if captured: the darwin user temp dir
/// (`/var/folders/.../T`) bare `mktemp` uses on macOS, the user's own temp directory on Windows.
#[must_use]
pub(crate) fn legacy_temp_dir() -> Option<&'static Path> {
    LEGACY_TEMP_DIR.get().map(PathBuf::as_path)
}

/// The temp value handed to shell children and the read-only guard: the pinned private root when
/// available, else the platform's own temp location.
#[must_use]
pub(crate) fn shell_tmpdir() -> String {
    temp_root().map_or_else(temp_baseline, |p| p.to_string_lossy().into_owned())
}

/// [`shell_tmpdir`]'s fallback when no private root is pinned: the historical `"/tmp"` baseline,
/// which is where a unix consumer that ignores `TMPDIR` lands anyway.
#[cfg(unix)]
fn temp_baseline() -> String {
    "/tmp".to_string()
}

/// [`shell_tmpdir`]'s fallback when no private root is pinned: the platform's own temp location.
#[cfg(not(unix))]
fn temp_baseline() -> String {
    std::env::temp_dir().to_string_lossy().into_owned()
}

/// Every [`TEMP_ENV_VARS`] name bound to [`shell_tmpdir`], so environment, guard and the cleaner's
/// own root cannot drift apart.
#[must_use]
pub(crate) fn shell_temp_vars() -> Vec<(String, String)> {
    let value = shell_tmpdir();
    TEMP_ENV_VARS
        .iter()
        .map(|name| ((*name).to_string(), value.clone()))
        .collect()
}

/// Initialize the private temp root and pin the platform's temp environment to it.
///
/// Must run at the very start of startup, BEFORE config and any temp use, and AFTER every dispatch
/// that must not create the root — read-only `debug`, the hidden subcommands (`__grep-engine`,
/// `__env-dump`), the standalone mode CLIs (`bench-openrouter`, `chrome`, `audio`) and `-h`/`-V`.
/// Cross-platform: `/tmp/mahbot` on unix, `<user temp>\mahbot` on Windows.
///
/// Failure modes (fail loudly, never paper over): a non-absolute OS temp dir; the root existing as
/// a non-directory, as a link or reparse point, or as something not ours (squatting — also the
/// second-OS-user guard, the accepted multi-user consequence); and a mode with group/other bits
/// where re-chmod fails — a loose mode on OUR OWN verified path is self-healed to 0700, since
/// bricking startup forever over a chmod race with the umask would be worse.
pub fn init_temp_root() -> anyhow::Result<()> {
    let legacy = std::env::temp_dir();
    let root = private_temp_root(&legacy)?;

    // Exclusive create (`create_dir` fails when the path already exists, which
    // is also how reuse is detected). The restriction is applied explicitly on
    // unix because `create_dir` honors the process umask; on Windows the root
    // inherits the DACL of the per-user temp directory it is created in (the
    // accepted approximation of 0700 — see the module doc).
    match std::fs::create_dir(&root) {
        Ok(()) => {
            #[cfg(unix)]
            restrict_private_root(&root)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => verify_private_root(&root)?,
        Err(e) => anyhow::bail!("temp root {}: create failed: {e}", root.display()),
    }

    let _ = TEMP_ROOT.set(root.clone());
    let _ = LEGACY_TEMP_DIR.set(legacy);

    // Pin the temp environment before any temp use, from the SAME pair the shell
    // children and the read-only guard are handed ([`shell_temp_vars`]): one
    // function is what makes the pin, the children and the guard agree by
    // construction instead of by inspection. SAFETY: single-threaded startup
    // (before the tokio runtime / iced), no concurrent env access.
    unsafe {
        for (name, value) in shell_temp_vars() {
            std::env::set_var(name, value);
        }
    }
    tracing::info!(root = %root.display(), "Pinned daemon temp root");
    Ok(())
}

/// The private temp root this platform derives from the OS temp dir it is handed.
///
/// On unix it is the fixed shared `/tmp/mahbot`, deliberately independent of the
/// OS temp dir (see the module doc): a relative `TMPDIR` is then harmless, and
/// the root — which is also one of the cleaner's scan roots — does not move with
/// the environment. It is fixed — no `-{uid}` suffix, this is a single-user
/// deployment, and the ownership check in `verify_private_root` is what makes a
/// shared path safe (a second OS user's daemon fails loudly at boot instead of
/// sharing the root).
#[cfg(unix)]
#[expect(clippy::unnecessary_wraps)] // one signature and one call site for both platforms
fn private_temp_root(_os_temp: &Path) -> Result<PathBuf> {
    Ok(PathBuf::from("/tmp/mahbot"))
}

/// The private temp root off unix: the platform's temp location joined with
/// `mahbot` — on a normal Windows machine that is the calling user's own temp
/// directory, hence user-only through the DACL it inherits (see the module doc
/// for what that does and does not cover). Never the machine-wide
/// `C:\Windows\Temp` and never under the daemon's storage root, which the cleaner
/// must never scan — except under the SYSTEM account, whose own temp lookup is
/// machine-wide (the stated deviation in the module doc). The OS temp dir must be
/// absolute: joining `mahbot` onto a relative one would derive the root from the
/// process's cwd.
#[cfg(not(unix))]
fn private_temp_root(os_temp: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        os_temp.is_absolute(),
        "OS temp dir {} is not an absolute path — refusing to derive the private temp root from it",
        os_temp.display()
    );
    Ok(os_temp.join("mahbot"))
}

/// Restrict a freshly created private root to its owner. Unix only: the Windows
/// root inherits the DACL of the per-user temp directory it is created in (the
/// accepted approximation — see the module doc).
#[cfg(unix)]
fn restrict_private_root(root: &Path) -> Result<()> {
    std::fs::set_permissions(root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
        .map_err(|e| anyhow::anyhow!("temp root {}: chmod 0700 failed: {e}", root.display()))
}

/// Verify a root that already exists (reuse after a previous boot): a real
/// directory, not a link/reparse point, owned by us — with unix's loose mode
/// self-healed on our own path.
fn verify_private_root(root: &Path) -> Result<()> {
    let meta = std::fs::symlink_metadata(root)
        .map_err(|e| anyhow::anyhow!("temp root {}: stat failed: {e}", root.display()))?;
    // Link check first: on unix `symlink_metadata` never reports a symlink as a
    // directory either, but naming the link is the more precise refusal.
    if is_link_like(&meta) {
        anyhow::bail!(
            "temp root {} is a link or reparse point — refusing to use it",
            root.display()
        );
    }
    if !meta.is_dir() {
        anyhow::bail!(
            "temp root {} exists and is not a directory — refusing to use it",
            root.display()
        );
    }
    anyhow::ensure!(
        is_owned_by_current_user(root)?,
        "temp root {} is not owned by the current user — refusing a squatted path",
        root.display()
    );
    // Off unix the restriction is the inherited DACL, so there is no mode to
    // heal; a loose mode there is simply not a thing this check can see.
    #[cfg(unix)]
    self_heal_loose_mode(root, &meta)?;
    Ok(())
}

/// Whether `meta` describes a link: a symlink on unix, a reparse point on
/// Windows. The Windows arm reads the RAW attribute because a name-surrogate
/// check (symlink-only) would let a junction through — and a junction into
/// someone else's directory would move every constraint here onto that directory.
#[cfg(windows)]
#[must_use]
fn is_link_like(meta: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;
    let reparse_point = windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
    meta.file_attributes() & reparse_point != 0
}

/// Whether `meta` describes a link: a symlink on unix, a reparse point on Windows.
#[cfg(not(windows))]
#[must_use]
fn is_link_like(meta: &std::fs::Metadata) -> bool {
    meta.file_type().is_symlink()
}

/// Whether `root` belongs to the current effective user (the uid check that
/// makes the shared unix path safe).
#[cfg(unix)]
fn is_owned_by_current_user(root: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt as _;
    let meta = std::fs::symlink_metadata(root)
        .map_err(|e| anyhow::anyhow!("temp root {}: stat failed: {e}", root.display()))?;
    Ok(meta.uid() == unsafe { libc::geteuid() })
}

/// Accepted limit: those two SIDs are the *elevated* token's own, so a root first created by an
/// elevated run is owned by `BUILTIN\Administrators` and a later non-elevated run matches neither
/// SID — it refuses loudly and needs `<temp>\mahbot` removed by hand before the daemon can boot
/// again.
#[cfg(windows)]
fn is_owned_by_current_user(root: &Path) -> Result<bool> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::Security::{TokenOwner, TokenUser};

    let owner = file_owner_sid(root)?;
    let token = open_process_token()?;
    let user = token_sid(token, TokenUser);
    let default_owner = token_sid(token, TokenOwner);
    // SAFETY: `token` is the handle `open_process_token` produced and both SID
    // byte vectors were already copied out of it.
    unsafe { CloseHandle(token) };
    if owner == user? {
        return Ok(true);
    }
    Ok(default_owner.is_ok_and(|sid| sid == owner))
}

/// The current process's access token, opened for querying only.
#[cfg(windows)]
fn open_process_token() -> Result<windows_sys::Win32::Foundation::HANDLE> {
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Security::TOKEN_QUERY;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    // A null handle: `HANDLE` is a plain integer in `windows_sys`.
    let mut token: HANDLE = 0;
    // SAFETY: `GetCurrentProcess` is a pseudo-handle (no close needed) and
    // `token` is a distinct local written only by the call.
    let opened = unsafe {
        OpenProcessToken(
            GetCurrentProcess(),
            TOKEN_QUERY,
            std::ptr::addr_of_mut!(token),
        )
    };
    if opened == 0 {
        return Err(anyhow::anyhow!(
            "OpenProcessToken failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(token)
}

/// The SID bytes of a token's `TokenUser`/`TokenOwner` information class, copied out of a two-pass
/// `GetTokenInformation` buffer.
#[cfg(windows)]
fn token_sid(token: windows_sys::Win32::Foundation::HANDLE, class: i32) -> Result<Vec<u8>> {
    use windows_sys::Win32::Foundation::PSID;
    use windows_sys::Win32::Security::GetTokenInformation;

    let mut len = 0_u32;
    // First pass: the call is expected to fail with ERROR_INSUFFICIENT_BUFFER,
    // so its return value is only informational — `len` is what is used.
    // SAFETY: a null buffer with length 0 is the documented size query.
    unsafe {
        GetTokenInformation(
            token,
            class,
            std::ptr::null_mut(),
            0,
            std::ptr::addr_of_mut!(len),
        );
    }
    if len == 0 {
        return Err(anyhow::anyhow!(
            "GetTokenInformation({class}) size query failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut buf = vec![0_u8; len as usize];
    // SAFETY: `buf` is a valid writable allocation of the length just reported.
    let ok = unsafe {
        GetTokenInformation(
            token,
            class,
            buf.as_mut_ptr().cast(),
            len,
            std::ptr::addr_of_mut!(len),
        )
    };
    if ok == 0 {
        return Err(anyhow::anyhow!(
            "GetTokenInformation({class}) failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: both `TokenUser` (`TOKEN_USER`) and `TokenOwner` (`TOKEN_OWNER`)
    // report a structure whose only leading member is the SID pointer, so the
    // buffer's first word is it. It is read unaligned because the byte buffer
    // carries no alignment guarantee, and the SID it points at lives inside
    // `buf`, which outlives the copy.
    let sid = unsafe { buf.as_ptr().cast::<PSID>().read_unaligned() };
    copy_sid(sid)
}

/// The owning SID of `root`, copied out of the file's security descriptor.
#[cfg(windows)]
fn file_owner_sid(root: &Path) -> Result<Vec<u8>> {
    use windows_sys::Win32::Security::{
        GetFileSecurityW, GetSecurityDescriptorOwner, OWNER_SECURITY_INFORMATION,
    };

    let Some(mut wide) = crate::util::wide_path(root) else {
        anyhow::bail!(
            "temp root {}: path contains an interior NUL",
            root.display()
        );
    };
    wide.push(0);
    let mut needed = 0_u32;
    // First pass: expected to fail with ERROR_INSUFFICIENT_BUFFER, only `needed`
    // is used. SAFETY: a null descriptor with length 0 is the documented size query.
    unsafe {
        GetFileSecurityW(
            wide.as_ptr(),
            OWNER_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            0,
            std::ptr::addr_of_mut!(needed),
        );
    }
    if needed == 0 {
        return Err(anyhow::anyhow!(
            "GetFileSecurityW({}) size query failed: {}",
            root.display(),
            std::io::Error::last_os_error()
        ));
    }
    // u64 units so the descriptor is naturally aligned for the API.
    let mut descriptor = vec![0_u64; needed.div_ceil(8) as usize];
    // SAFETY: `descriptor` is a writable allocation of at least `needed` bytes.
    let ok = unsafe {
        GetFileSecurityW(
            wide.as_ptr(),
            OWNER_SECURITY_INFORMATION,
            descriptor.as_mut_ptr().cast(),
            needed,
            std::ptr::addr_of_mut!(needed),
        )
    };
    if ok == 0 {
        return Err(anyhow::anyhow!(
            "GetFileSecurityW({}) failed: {}",
            root.display(),
            std::io::Error::last_os_error()
        ));
    }
    let mut owner = std::ptr::null_mut();
    let mut defaulted = 0;
    // SAFETY: `descriptor` holds the descriptor the call above wrote; `owner`
    // receives a pointer INTO it, so the SID is copied out below while it is
    // still alive.
    let ok = unsafe {
        GetSecurityDescriptorOwner(
            descriptor.as_mut_ptr().cast(),
            std::ptr::addr_of_mut!(owner),
            std::ptr::addr_of_mut!(defaulted),
        )
    };
    if ok == 0 {
        return Err(anyhow::anyhow!(
            "GetSecurityDescriptorOwner({}) failed: {}",
            root.display(),
            std::io::Error::last_os_error()
        ));
    }
    copy_sid(owner)
}

/// Copy a SID's bytes out of the structure that owns them.
#[cfg(windows)]
fn copy_sid(sid: windows_sys::Win32::Foundation::PSID) -> Result<Vec<u8>> {
    use windows_sys::Win32::Security::GetLengthSid;

    if sid.is_null() {
        anyhow::bail!("security descriptor reports no owner SID");
    }
    // SAFETY: `sid` points at a valid SID owned by a buffer that outlives this call.
    let len = unsafe { GetLengthSid(sid) };
    if len == 0 {
        return Err(anyhow::anyhow!(
            "GetLengthSid failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `GetLengthSid` reports the SID's byte length and the SID is not
    // mutated for the duration of the copy.
    Ok(unsafe { std::slice::from_raw_parts(sid.cast::<u8>(), len as usize) }.to_vec())
}

/// The loose-mode self-heal (unix only): a loose mode on OUR OWN path is a previous boot's
/// create-time chmod failure (the umask raced it), NOT a squatter — ownership was already proved.
/// Re-chmod 0700 instead of bricking startup forever.
#[cfg(unix)]
fn self_heal_loose_mode(root: &Path, meta: &std::fs::Metadata) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;
    let mode = meta.mode() & 0o777;
    if mode & 0o077 != 0 {
        std::fs::set_permissions(root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .map_err(|e| {
                anyhow::anyhow!(
                    "temp root {} has group/other permissions (mode {mode:o}) and re-chmod 0700 failed: {e}",
                    root.display()
                )
            })?;
        tracing::warn!(
            root = %root.display(),
            mode = format_args!("{mode:o}"),
            "Temp root had loose permissions — re-chmod 0700 (self-heal, path owned by self)"
        );
    }
    Ok(())
}

/// One root the periodic temp cleaner may act in.
struct CleanupRoot {
    path: PathBuf,
    /// The daemon's own private temp root, as opposed to the user's temp area.
    private: bool,
}

/// The candidate roots for a process whose OS temp dir is `os_temp`: the private root this platform
/// derives from it, that temp dir itself, and (unix only) the two literals a command reaches
/// without consulting the environment — what a bare `mktemp` in a pipeline does.
///
/// One builder for the cleaner's roots and for the roots a process that never pinned the temp
/// environment checks a destination against ([`cleanup_scan_roots_unpinned`]), so the two cannot
/// disagree about where the product's own cleanup may act.
#[must_use]
fn cleanup_root_candidates(os_temp: &Path) -> Vec<CleanupRoot> {
    let mut roots = Vec::new();
    if let Ok(private) = private_temp_root(os_temp) {
        roots.push(CleanupRoot {
            path: private,
            private: true,
        });
    }
    roots.push(CleanupRoot {
        path: os_temp.to_path_buf(),
        private: false,
    });
    #[cfg(unix)]
    roots.extend([
        CleanupRoot {
            path: PathBuf::from("/tmp"),
            private: false,
        },
        CleanupRoot {
            path: PathBuf::from("/private/tmp"),
            private: false,
        },
    ]);
    roots
}

/// Select the scanner's roots: the platform's candidates in scan order — the private root derived
/// from the pre-pin OS temp dir, that temp dir itself, then (unix only) `/tmp` and `/private/tmp` —
/// each resolved through [`cleanup_roots_from`].
#[must_use]
fn cleanup_scan_roots() -> Vec<CleanupRoot> {
    let os_temp = legacy_temp_dir().map_or_else(std::env::temp_dir, Path::to_path_buf);
    cleanup_roots_from(cleanup_root_candidates(&os_temp))
}

/// The places the product's own temp cleanup is responsible for, as a process that NEVER pinned the
/// temp environment sees them — the command line's destination check, which must refuse a directory
/// whose files its cleaner would later sweep.
///
/// The private root is derived the same way here and the pre-pin OS temp dir is this process's own
/// `std::env::temp_dir()` — the lookup the daemon's pre-pin value comes from, so the two agree
/// wherever a launch's environment is the machine's own. A process without the pin cannot see a
/// daemon started with a DIFFERENT temp environment; the candidates themselves are built by the
/// cleaner's own code.
#[must_use]
pub(crate) fn cleanup_scan_roots_unpinned() -> Vec<PathBuf> {
    cleanup_roots_from(cleanup_root_candidates(&std::env::temp_dir()))
        .into_iter()
        .map(|root| root.path)
        .collect()
}

/// Resolve candidates into usable roots: a candidate that does not resolve to an existing directory
/// is dropped, a resolved path keeps the plain spelling the cleaner's shell is given (see
/// [`crate::util::strip_verbatim_prefix`]), and duplicates collapse keeping the FIRST occurrence —
/// so the earliest candidate's private flag wins and scan order is preserved.
#[must_use]
fn cleanup_roots_from(candidates: Vec<CleanupRoot>) -> Vec<CleanupRoot> {
    let mut roots: Vec<CleanupRoot> = Vec::new();
    for candidate in candidates {
        let Ok(resolved) = std::fs::canonicalize(&candidate.path) else {
            continue;
        };
        let path = crate::util::strip_verbatim_prefix(&resolved);
        if !path.is_dir() || roots.iter().any(|root| root.path == path) {
            continue;
        }
        roots.push(CleanupRoot {
            path,
            private: candidate.private,
        });
    }
    roots
}

/// Where a BARE `mktemp -d` (no `-p`/template) actually lands on this platform: on macOS it ignores
/// `TMPDIR` and uses `_CS_DARWIN_USER_TEMP_DIR` (the legacy darwin dir); elsewhere it honors the
/// platform's temp environment. The readonly guard's synthetic mktemp anchor must match this, so
/// `..` chains over it resolve like the real value.
#[must_use]
pub(crate) fn bare_mktemp_landing_root() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(legacy) = legacy_temp_dir() {
            return legacy.to_path_buf();
        }
    }
    std::env::temp_dir()
}

/// How often the cadence loop wakes to check whether the cleanup is due. The due test is wall-clock
/// based, so this only bounds how late a due run may start.
const TEMP_CLEANUP_WAKE: Duration = Duration::from_mins(30);

const GIB: u64 = 1 << 30;

/// Absolute floors under the percentage thresholds (10 GiB / 15 GiB).
const TEMP_CLEANUP_LOW_FREE_FLOOR: u64 = 10 * GIB;
const TEMP_CLEANUP_RECOVERED_FREE_FLOOR: u64 = 15 * GIB;
/// Percentage thresholds of the volume's capacity.
const TEMP_CLEANUP_LOW_FREE_PCT: u64 = 10;
const TEMP_CLEANUP_RECOVERED_FREE_PCT: u64 = 15;

/// `config_kv` key holding the cleanup's last-run start (RFC 3339), owned by this subsystem —
/// preserved by the config orphan purge.
pub(crate) const TEMP_CLEANUP_LAST_RUN_KV_KEY: &str = "temp_cleanup_last_run_at";
/// `config_kv` key holding the sticky hysteresis mode (`"daily"`/`"weekly"`).
pub(crate) const TEMP_CLEANUP_MODE_KV_KEY: &str = "temp_cleanup_mode";

/// In-memory mirror of the latest dispatch (wall clock). The persisted `config_kv` timestamp is the
/// cross-restart record; this mirror keeps a failing config-store write from turning the cadence
/// into a re-dispatch loop within the same lifetime.
static LAST_DISPATCH: std::sync::Mutex<Option<DateTime<Utc>>> = std::sync::Mutex::new(None);

/// Free-space mode of the cleaner, with hysteresis between the two thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupMode {
    /// Free space is tight: at most one cleanup per day.
    Daily,
    /// Free space is comfortable: at most one cleanup per week.
    Weekly,
}

impl CleanupMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Weekly => "weekly",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "daily" => Some(Self::Daily),
            "weekly" => Some(Self::Weekly),
            _ => None,
        }
    }

    fn interval(self) -> ChronoDuration {
        match self {
            Self::Daily => ChronoDuration::days(1),
            Self::Weekly => ChronoDuration::days(7),
        }
    }
}

/// Free space below which the cleaner switches to daily mode.
fn daily_entry_threshold(capacity: u64) -> u64 {
    (capacity.saturating_mul(TEMP_CLEANUP_LOW_FREE_PCT) / 100).max(TEMP_CLEANUP_LOW_FREE_FLOOR)
}

/// Free space above which daily mode is left for weekly mode.
fn weekly_return_threshold(capacity: u64) -> u64 {
    (capacity.saturating_mul(TEMP_CLEANUP_RECOVERED_FREE_PCT) / 100)
        .max(TEMP_CLEANUP_RECOVERED_FREE_FLOOR)
}

/// Apply the hysteresis thresholds to the current mode: below the entry threshold it is always
/// daily, above the (higher) return threshold always weekly, and in between the current mode
/// sticks, so the cadence cannot flap around the line.
fn mode_after_free(current: CleanupMode, free: u64, capacity: u64) -> CleanupMode {
    if free < daily_entry_threshold(capacity) {
        CleanupMode::Daily
    } else if free > weekly_return_threshold(capacity) {
        CleanupMode::Weekly
    } else {
        current
    }
}

/// The sticky mode persisted in `config_kv`; unset, unparseable or a read error falls back to
/// weekly (the documented default where free space is unknown).
async fn stored_cleanup_mode(store: Option<&crate::config_db::ConfigStore>) -> CleanupMode {
    let Some(store) = store else {
        return CleanupMode::Weekly;
    };
    match store.get_kv(TEMP_CLEANUP_MODE_KV_KEY).await {
        Ok(Some(raw)) => CleanupMode::parse(&raw).unwrap_or_else(|| {
            tracing::warn!(value = %raw, "Unrecognized temp cleaner mode — using weekly");
            CleanupMode::Weekly
        }),
        Ok(None) => CleanupMode::Weekly,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read temp cleaner mode — using weekly");
            CleanupMode::Weekly
        }
    }
}

/// Parse a persisted last-run timestamp. `None` (missing or unparseable — fail-open) makes the
/// cleanup due.
fn parse_last_run(raw: &str) -> Option<DateTime<Utc>> {
    match crate::db::parse_utc_timestamp(raw) {
        Ok(last) => Some(last),
        Err(e) => {
            tracing::warn!(value = %raw, error = %e, "Unparseable temp cleaner last run — treating as due");
            None
        }
    }
}

/// Is the cleanup due? No recorded run means due; otherwise the wall-clock elapsed time must cover
/// the mode's interval.
fn temp_cleanup_due(
    last_run_at: Option<DateTime<Utc>>,
    mode: CleanupMode,
    now: DateTime<Utc>,
) -> bool {
    match last_run_at {
        None => true,
        Some(last) => now.signed_duration_since(last) >= mode.interval(),
    }
}

/// The cleaner's dedicated cadence loop, spawned once from the background-task set and decoupled
/// from every other periodic loop so its frequency follows free disk space alone. Panic posture
/// matches the sibling loops: the task-level guard in `spawn_cancellable` logs and ends the loop,
/// while the agent run is guarded in [`run_temp_cleanup_and_finish`] so its jobs row is
/// terminalized on every exit.
pub async fn run_temp_cleanup_loop() {
    loop {
        // Sleep BEFORE the first check: a cleaner run must never compete with startup (store opens,
        // provider warmup, boot recovery). The first tick is far later than
        // `jobs::recover_from_restart`, so a leftover `temp_cleanup` row from a crash is already
        // terminalized; each run terminalizes its own row anyway.
        if !crate::shutdown::sleep_or_shutdown_or_drain(TEMP_CLEANUP_WAKE).await {
            break;
        }
        temp_cleanup_tick().await;
    }
}

/// One cadence tick: refresh the sticky mode from current free space, then dispatch the cleaner
/// when due.
async fn temp_cleanup_tick() {
    let store = crate::config_db::CONFIG_STORE.get();
    let current = stored_cleanup_mode(store).await;
    let mode = match crate::util::with_block_in_place(|| {
        crate::util::disk::free_and_capacity(&std::env::temp_dir())
    }) {
        Some((free, capacity)) => mode_after_free(current, free, capacity),
        None => current,
    };
    if mode != current
        && let Some(store) = store
    {
        if let Err(e) = store.set_kv(TEMP_CLEANUP_MODE_KV_KEY, mode.as_str()).await {
            tracing::warn!(error = %e, "Failed to persist temp cleaner mode");
        }
        tracing::info!(mode = mode.as_str(), "Temp cleaner cadence mode changed");
    }

    let persisted = match store {
        Some(store) => match store.get_kv(TEMP_CLEANUP_LAST_RUN_KV_KEY).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to read temp cleaner last run — treating as due");
                None
            }
        },
        None => None,
    };
    // The persisted record is authoritative across restarts; the in-memory mirror keeps a failed
    // config write from re-dispatching every tick.
    let last_run = [
        persisted.as_deref().and_then(parse_last_run),
        *LAST_DISPATCH.lock().unwrap_poison(),
    ]
    .into_iter()
    .flatten()
    .max();
    if !temp_cleanup_due(last_run, mode, Utc::now()) {
        return;
    }
    // A tick with no target is skipped WITHOUT recording a run, so it does not burn a run on every
    // tick and the next tick retries once a root appears. The skip sits inside the due decision —
    // cadence, hysteresis and retention are unchanged.
    let roots = crate::util::with_block_in_place(cleanup_scan_roots);
    if roots.is_empty() {
        tracing::debug!("Temp cleaner has no existing scan root — skipping this pass");
        return;
    }
    if let Err(e) = dispatch_temp_cleanup(&roots).await {
        tracing::warn!(error = %e, "Temp cleaner dispatch failed");
    }
}

/// Prompt asset for the periodic temp cleaner task.
const TEMP_CLEANUP_PROMPT_KEY: &str = "sanitation/temp_cleanup.md";

/// Render the cleaner's scan-root list: one numbered, fully-qualified path per line with a short
/// role label, sourced from the same values the dispatch and the read-only guard use.
#[must_use]
fn render_scan_roots(roots: &[CleanupRoot]) -> String {
    roots
        .iter()
        .enumerate()
        .map(|(idx, root)| {
            let role = if root.private {
                "the daemon's own private temp root"
            } else {
                "an OS temp area the daemon and its shells use"
            };
            format!("{}. `{}` — {role}", idx + 1, root.path.display())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The cleaner's task prompt: the shared procedure with the platform's scan roots and tool
/// spellings substituted in. Both tool blocks ship on every platform; only this one's is
/// substituted.
#[must_use]
fn render_temp_cleanup_prompt(roots: &[CleanupRoot]) -> String {
    let tools = crate::prompt::load_prompt(if cfg!(windows) {
        "sanitation/temp_cleanup_tools_windows.md"
    } else {
        "sanitation/temp_cleanup_tools_unix.md"
    });
    crate::prompt::substitute(
        &crate::prompt::load_prompt(TEMP_CLEANUP_PROMPT_KEY),
        &[
            ("{{scan_roots}}", &render_scan_roots(roots)),
            ("{{platform_tools}}", tools.trim_end()),
        ],
    )
}

/// Synthetic workspace name for the cleaner: no `workspaces` row, ephemeral like research run
/// roots. It must be explicit — the pinned temp root's last component is "mahbot", which would
/// collide with the managed repo workspace via `Workspace::from_path`.
const TEMP_CLEANUP_WORKSPACE_NAME: &str = "tmp";

/// Dispatch the cleaner and run it to completion (Sanitation role).
///
/// The loop is the only dispatcher and awaits the run, so dispatch is serialized and a crashed run
/// cannot strand the cadence. The last-run timestamp is recorded right after the durable row exists
/// (a failed dispatch leaves the cleanup due, retried next tick) in both the persisted `config_kv`
/// record and the in-memory mirror, and the row is terminalized on every exit path, including a
/// panic inside the agent.
async fn dispatch_temp_cleanup(roots: &[CleanupRoot]) -> Result<()> {
    let conn = &crate::session::store().conn;
    let job_id = crate::generate_id();
    let ws = Workspace::ephemeral_run(TEMP_CLEANUP_WORKSPACE_NAME, &roots[0].path);
    let prompt = render_temp_cleanup_prompt(roots);

    crate::jobs::spawn_job(
        conn,
        &job_id,
        &prompt,
        &ws.name,
        "",
        "",
        crate::Role::Sanitation,
        &[crate::jobs::NewAgent {
            agent_id: crate::research_cleanup::cleanup_agent_id(&job_id),
            kind: crate::jobs::AgentKind::Sanitation,
            idx: None,
            task: prompt.clone(),
        }],
        &crate::jobs::SpawnChild::TempCleanup,
        None,
        None,
        None,
    )
    .await
    .map_err(|e| {
        tracing::error!(job = %job_id, error = %e, "Failed to spawn temp cleaner job");
        e
    })?;

    let now = Utc::now();
    *LAST_DISPATCH.lock().unwrap_poison() = Some(now);
    if let Some(store) = crate::config_db::CONFIG_STORE.get()
        && let Err(e) = store
            .set_kv(TEMP_CLEANUP_LAST_RUN_KV_KEY, &now.to_rfc3339())
            .await
    {
        tracing::warn!(error = %e, "Failed to record temp cleaner run start");
    }

    tracing::info!(job = %job_id, "Temp cleaner dispatched");
    run_temp_cleanup_and_finish(&job_id, &ws, &prompt).await;
    Ok(())
}

/// Run the cleaner agent and terminalize the job row on EVERY exit path, including a panic inside
/// the agent. No folder hold exists for this cleaner (unlike research cleanup): the row is simply
/// deleted when the run finishes, success or failure, so the next due pass re-runs cleanly.
async fn run_temp_cleanup_and_finish(job_id: &str, ws: &Workspace, prompt: &str) {
    let agent_id = crate::research_cleanup::cleanup_agent_id(job_id);
    let run = std::panic::AssertUnwindSafe(crate::agent::run_default_agent(
        &agent_id,
        crate::Role::Sanitation,
        ws,
        prompt,
        false,
        None,
        None,
        None,
    ))
    .catch_unwind()
    .await;
    match run {
        Ok((agent, response)) => {
            let report = response.unwrap_or_else(|| {
                format!(
                    "Temp cleaner FAILED (job {job_id}): {}",
                    agent
                        .failure
                        .clone()
                        .unwrap_or_else(|| "no failure detail".to_string())
                )
            });
            tracing::info!(
                job = %job_id,
                agent = %agent_id,
                "Temp cleaner finished: {}",
                &report
            );
        }
        Err(payload) => tracing::error!(
            job = %job_id,
            agent = %agent_id,
            "Temp cleaner panicked: {}",
            crate::util::panic_message(&*payload)
        ),
    }
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_capture_and_pin_are_consistent() {
        // init_temp_root is a process-global singleton — not run in tests (it would pin the temp
        // env for the whole test process). Just verify the accessors are coherent: unset → no root,
        // no legacy.
        assert!(temp_root().is_none());
        assert!(legacy_temp_dir().is_none());
        // Every name this platform uses carries the one shell value.
        let vars = shell_temp_vars();
        assert_eq!(vars.len(), TEMP_ENV_VARS.len());
        assert!(vars.iter().all(|(_, value)| *value == shell_tmpdir()));
        // The unix baseline is the historical literal, unchanged.
        #[cfg(unix)]
        assert_eq!(shell_tmpdir(), "/tmp");
    }

    #[test]
    fn verify_private_root_refuses_a_non_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("mahbot");
        std::fs::write(&file, b"not a directory").expect("write file");

        let err = verify_private_root(&file).expect_err("a regular file must be refused");
        assert!(err.to_string().contains("not a directory"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn verify_private_root_refuses_a_symlink() {
        let dir = tempfile::tempdir().expect("temp dir");
        let target = dir.path().join("target");
        std::fs::create_dir(&target).expect("create target");
        let link = dir.path().join("mahbot");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        // Refused as a link: `symlink_metadata` never follows it, so the check
        // sees the link itself and the daemon can never be moved out of the temp
        // area by one. (A Windows junction IS a directory by attribute, which is
        // why the reparse-point attribute is what the check reads there.)
        let err = verify_private_root(&link).expect_err("a symlink must be refused");
        assert!(err.to_string().contains("link or reparse point"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn verify_private_root_self_heals_a_loose_mode() {
        use std::os::unix::fs::MetadataExt as _;
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("mahbot");
        std::fs::create_dir(&root).expect("create root");
        std::fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .expect("chmod");

        verify_private_root(&root).expect("a loose mode on our own path is self-healed");
        let mode = std::fs::metadata(&root).expect("stat root").mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn cleanup_roots_from_dedupes_and_drops_absent_candidates() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        // The same directory reached twice — once directly, once through a `.`
        // detour — plus a path that does not exist at all.
        let candidates = |private_first: bool| {
            let direct = CleanupRoot {
                path: root.clone(),
                private: private_first,
            };
            let via_dot = CleanupRoot {
                path: root.join("."),
                private: !private_first,
            };
            let absent = CleanupRoot {
                path: root.join("never-created"),
                private: private_first,
            };
            cleanup_roots_from(vec![direct, via_dot, absent])
        };

        // Deduped to one root, resolved and in the plain spelling the cleaner's
        // shell is handed (canonicalization on Windows yields the platform's
        // verbatim prefix, which the root must not carry), and the FIRST
        // occurrence's private flag wins.
        let roots = candidates(true);
        assert_eq!(roots.len(), 1);
        assert_eq!(
            roots[0].path,
            crate::util::strip_verbatim_prefix(&std::fs::canonicalize(&root).expect("canonical"))
        );
        assert!(roots[0].private);
        let roots = candidates(false);
        assert_eq!(roots.len(), 1);
        assert!(!roots[0].private);

        // A candidates list holding only a never-created path resolves to an
        // empty list — exactly the state in which a tick dispatches nothing.
        let absent = std::env::temp_dir().join(format!("mahbot-absent-{}", crate::generate_id()));
        let roots = cleanup_roots_from(vec![CleanupRoot {
            path: absent,
            private: true,
        }]);
        assert!(roots.is_empty());
    }

    #[test]
    fn mode_after_free_applies_hysteresis() {
        // 50 GiB volume: the 10/15 GiB floors dominate the percentages; 460 GiB
        // volume: the 10%/15% percentages dominate the floors.
        let small = 50 * GIB;
        let big = 460 * GIB;
        assert_eq!(daily_entry_threshold(small), 10 * GIB);
        assert_eq!(weekly_return_threshold(small), 15 * GIB);
        assert_eq!(daily_entry_threshold(big), 46 * GIB);
        assert_eq!(weekly_return_threshold(big), 69 * GIB);

        // Below the entry threshold the mode is always daily; above the return
        // threshold always weekly.
        assert_eq!(
            mode_after_free(CleanupMode::Weekly, 5 * GIB, small),
            CleanupMode::Daily
        );
        assert_eq!(
            mode_after_free(CleanupMode::Daily, 20 * GIB, small),
            CleanupMode::Weekly
        );
        // Between the thresholds the current mode sticks — no flapping.
        assert_eq!(
            mode_after_free(CleanupMode::Daily, 12 * GIB, small),
            CleanupMode::Daily
        );
        assert_eq!(
            mode_after_free(CleanupMode::Weekly, 12 * GIB, small),
            CleanupMode::Weekly
        );
    }

    #[test]
    fn due_uses_wall_clock_and_mode_interval() {
        let now = Utc::now();
        assert!(temp_cleanup_due(None, CleanupMode::Weekly, now));
        let two_days_ago = now - ChronoDuration::days(2);
        assert!(temp_cleanup_due(
            Some(two_days_ago),
            CleanupMode::Daily,
            now
        ));
        assert!(!temp_cleanup_due(
            Some(two_days_ago),
            CleanupMode::Weekly,
            now
        ));
        let eight_days_ago = now - ChronoDuration::days(8);
        assert!(temp_cleanup_due(
            Some(eight_days_ago),
            CleanupMode::Weekly,
            now
        ));
        let recent = now - ChronoDuration::hours(1);
        assert!(!temp_cleanup_due(Some(recent), CleanupMode::Daily, now));
    }
}

//! Managed bun runtime (JS/TS runtime, <https://bun.sh>) — installed in bun's
//! OWN standard user directory (`~/.bun/bin`) and brought to the newest release
//! on every product start — no version comparison, no look at what is already
//! there. Silent: there is no consent flow (agents invoke `bun` through the
//! Shell tool, and the product's own tools spawn it by absolute path).

use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::process::Stdio;
use std::time::Duration;

#[cfg(target_os = "macos")]
use tokio::process::Command;

/// GitHub repo whose releases host the bun runtime (single source of truth).
const BUN_RELEASE_REPO: &str = "oven-sh/bun";

/// Timeout for resolving the latest bun release tag.
const BUN_RELEASE_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout for a bun release download (zips are ~30–90 MB).
const BUN_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Timeout for the macOS AVX2 `sysctl` probe.
#[cfg(target_os = "macos")]
const BUN_SYSCTL_TIMEOUT: Duration = Duration::from_secs(8);

/// The platform-appropriate bun binary name.
const fn bun_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "bun.exe"
    } else {
        "bun"
    }
}

/// The product resolves the runtime itself rather than through the search path:
/// only bun's own directory (`~/.bun/bin`) is probed. A user's Homebrew/npm bun
/// is never managed or updated by us, and the owner's own terminal is made to
/// see the runtime's directory by [`crate::util::owner_path`].
///
/// `None` when the product's own copy is absent or not executable — the callers
/// that launch single-file scripts (the `custom` tool) report that as a
/// mahbot-side fault rather than falling back to an unmanaged interpreter.
pub(crate) fn bun_binary_path() -> Option<PathBuf> {
    crate::util::managed_bin::bun_bin_dir()
        .map(|d| d.join(bun_file_name()))
        .filter(|p| crate::util::is_executable(p))
}

/// Bun release-asset platform tag, e.g. `darwin-x64` or `linux-x64-musl`.
/// `None` on platform/arch combos the vendor does not publish. Mirrors the
/// vendor's own asset naming, Windows on ARM included; the `-baseline` suffix
/// marks an AVX2-less x64 build (only x64 ships baseline builds, so aarch64 has
/// none on any platform).
#[must_use]
fn bun_asset_target(os: &str, arch: &str, musl: bool, avx2: bool) -> Option<String> {
    let base = match (os, arch, musl) {
        ("macos", "x86_64", _) => "darwin-x64",
        ("macos", "aarch64", _) => "darwin-aarch64",
        ("linux", "x86_64", false) => "linux-x64",
        ("linux", "x86_64", true) => "linux-x64-musl",
        ("linux", "aarch64", false) => "linux-aarch64",
        ("linux", "aarch64", true) => "linux-aarch64-musl",
        ("windows", "x86_64", _) => "windows-x64",
        // Windows on ARM: the vendor's own arm64 build (`bun-windows-aarch64.zip`,
        // first published in bun v1.3.10) — there is no `-baseline` variant of it.
        ("windows", "aarch64", _) => "windows-aarch64",
        // Everything else — Windows' half-way `arm64ec` arch included — stays
        // unsupported rather than taking another processor's build.
        _ => return None,
    };
    let asset = if !avx2 && arch == "x86_64" {
        format!("{base}-baseline")
    } else {
        base.to_string()
    };
    Some(asset)
}

/// Asset platform tag for THIS build, or `Err` naming the unsupported
/// platform. Uses the compile-time target triple (`cfg!`) so it is correct
/// regardless of the runtime host.
async fn bun_asset_name() -> Result<String, String> {
    let (os, arch) = crate::util::managed_bin::host_os_arch()?;
    let musl = os == "linux" && crate::util::managed_bin::linux_host_is_musl();
    bun_asset_target(os, arch, musl, host_has_avx2().await)
        .ok_or_else(|| format!("bun has no release asset for {os}-{arch}"))
}

/// Runtime AVX2 probe mirroring bun's install.sh. Returns true ("use the plain
/// asset") for non-x86_64 hosts (no baseline split exists) and for hosts with
/// no probe available (e.g. Windows). On Linux read `/proc/cpuinfo`; on macOS
/// run `sysctl -a`. Probe failure → false (baseline, safe on old CPUs,
/// SIGILL-free).
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux")),
    expect(clippy::unused_async)
)]
async fn host_has_avx2() -> bool {
    if !cfg!(target_arch = "x86_64") {
        return true;
    }
    #[cfg(target_os = "linux")]
    {
        let Ok(cpuinfo) = tokio::fs::read_to_string("/proc/cpuinfo").await else {
            return false;
        };
        cpuinfo_has_avx2(&cpuinfo)
    }
    #[cfg(target_os = "macos")]
    {
        macos_sysctl_has_avx2().await
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        // No AVX2 probe is available (e.g. Windows) — prefer the plain asset
        // rather than depending on upstream's redundant `-baseline` copies.
        true
    }
}

/// The `/proc/cpuinfo` flags line (`flags\t\t: fpu ... avx2 ...`) — true when
/// any flag token is exactly `avx2`.
#[cfg(any(target_os = "linux", test))]
#[must_use]
fn cpuinfo_has_avx2(cpuinfo: &str) -> bool {
    cpuinfo
        .lines()
        .any(|l| l.starts_with("flags") && l.split_whitespace().any(|f| f == "avx2"))
}

/// The `sysctl -a` output (`machdep.cpu.features: ... AVX2 ...`) — mirrors
/// install.sh's `sysctl -a | grep machdep.cpu | grep AVX2` (case-sensitive).
#[cfg(any(target_os = "macos", test))]
#[must_use]
fn sysctl_has_avx2(sysctl_out: &str) -> bool {
    sysctl_out
        .lines()
        .any(|l| l.contains("machdep.cpu") && l.contains("AVX2"))
}

/// The macOS AVX2 probe: bounded `sysctl -a`, kill-on-drop, stdout piped.
#[cfg(target_os = "macos")]
async fn macos_sysctl_has_avx2() -> bool {
    let mut cmd = Command::new("sysctl");
    cmd.arg("-a")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let Ok(out) = tokio::time::timeout(BUN_SYSCTL_TIMEOUT, cmd.output()).await else {
        return false;
    };
    let Ok(out) = out else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    sysctl_has_avx2(&String::from_utf8_lossy(&out.stdout))
}

/// Find the SHA-256 hash for `asset` in a `SHASUMS256.txt` body (one
/// `<64-hex>  <filename>` line per asset, parsed with the shared sidecar
/// parser). `Err` when no line matches — the caller treats this as a soft
/// non-fatal failure (the upstream asset name set is not frozen).
fn shasums256_hash_for(body: &str, asset: &str) -> Result<String, String> {
    for line in body.lines() {
        if let Some((hash, name)) = crate::util::managed_bin::parse_sha256_sidecar(line)
            && name == asset
        {
            return Ok(hash);
        }
    }
    Err(format!("SHASUMS256.txt has no entry for {asset}"))
}

/// Fetch the release's aggregate `SHASUMS256.txt` and extract the hash for
/// `asset` via [`shasums256_hash_for`]. Request failures are classified
/// (path-free and URL-free) through
/// [`crate::util::managed_bin::request_reason`].
async fn shasums256_hash(
    client: &reqwest::Client,
    tag: &str,
    asset: &str,
) -> Result<String, String> {
    let url =
        format!("https://github.com/{BUN_RELEASE_REPO}/releases/download/{tag}/SHASUMS256.txt");
    let response = client.get(&url).send().await.map_err(|e| {
        format!(
            "failed to fetch SHASUMS256.txt: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    if !response.status().is_success() {
        return Err(format!(
            "failed to fetch SHASUMS256.txt: HTTP {}",
            response.status()
        ));
    }
    let body = response.text().await.map_err(|e| {
        format!(
            "failed to read SHASUMS256.txt: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })?;
    shasums256_hash_for(&body, asset)
}

/// Bring the runtime's own directory (`~/.bun/bin`) up to the newest bun
/// release: resolve the tag, download this platform's asset, verify it against
/// the published `SHASUMS256.txt` hash, extract the single binary and place it.
/// Returns the installed path. Every failure message is path-free and URL-free
/// (see [`crate::util::managed_bin::download_reason`]), because it is recorded
/// verbatim.
async fn install_latest() -> Result<PathBuf, String> {
    use crate::util::http::{DownloadSizeCheck, build_download_client, download_verified};

    // The destination is resolved before anything is fetched: an unresolvable home
    // is a reason not to download an archive at all.
    let dest = crate::util::managed_bin::bun_bin_dir()
        .ok_or_else(|| "bun install dir unavailable (home not resolvable)".to_string())?
        .join(bun_file_name());
    let tag =
        crate::util::managed_bin::fetch_latest_tag(BUN_RELEASE_REPO, BUN_RELEASE_TIMEOUT).await?;
    let asset = format!("bun-{}.zip", bun_asset_name().await?);
    let url = format!("https://github.com/{BUN_RELEASE_REPO}/releases/download/{tag}/{asset}");
    let client = build_download_client(BUN_DOWNLOAD_TIMEOUT)
        .map_err(|_| "the download client could not be built".to_string())?;
    let hash = shasums256_hash(&client, &tag, &asset).await?;
    let dir = tempfile::tempdir().map_err(|e| {
        format!(
            "the download's temporary directory could not be created ({})",
            e.kind()
        )
    })?;
    let archive_path = dir.path().join("bun.zip");
    download_verified(
        &client,
        &url,
        &archive_path,
        &hash,
        None,
        DownloadSizeCheck::None,
        |_, _| {},
    )
    .await
    .map_err(|e| {
        format!(
            "the bun release could not be downloaded: {}",
            crate::util::managed_bin::download_reason(&e)
        )
    })?;
    let fresh = crate::util::managed_bin::extract_single_file_zip(
        &archive_path,
        dir.path(),
        bun_file_name(),
    )?;
    crate::util::managed_bin::place_extracted(&fresh, &dest)?;
    Ok(dest)
}

/// Spawned one-shot task: install the product's own bun runtime straight away
/// when it is missing, else bring it to the newest release once the boot has
/// settled — whatever sits in the runtime's own directory (`~/.bun/bin`) is
/// always replaced, with no version comparison and no look at what is already
/// there. The shared shape and its failure levels live in
/// [`crate::util::managed_bin::install_on_start`].
pub async fn run_bun_management() {
    crate::util::managed_bin::install_on_start(
        "bun",
        || bun_binary_path().is_some(),
        install_latest(),
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bun_asset_target_maps_supported_combos() {
        assert_eq!(
            bun_asset_target("macos", "x86_64", false, true).as_deref(),
            Some("darwin-x64")
        );
        assert_eq!(
            bun_asset_target("macos", "aarch64", false, true).as_deref(),
            Some("darwin-aarch64")
        );
        assert_eq!(
            bun_asset_target("linux", "x86_64", false, true).as_deref(),
            Some("linux-x64")
        );
        assert_eq!(
            bun_asset_target("linux", "x86_64", true, true).as_deref(),
            Some("linux-x64-musl")
        );
        assert_eq!(
            bun_asset_target("linux", "aarch64", false, true).as_deref(),
            Some("linux-aarch64")
        );
        assert_eq!(
            bun_asset_target("linux", "aarch64", true, true).as_deref(),
            Some("linux-aarch64-musl")
        );
        assert_eq!(
            bun_asset_target("windows", "x86_64", false, true).as_deref(),
            Some("windows-x64")
        );
        assert_eq!(
            bun_asset_target("windows", "aarch64", false, true).as_deref(),
            Some("windows-aarch64")
        );
        // Unsupported platform/arch combos return None — Windows' half-way
        // `arm64ec` arch included (a distinct `target_arch`, so `host_os_arch()`
        // reports it unsupported and it can never reach the arm64 row above).
        assert_eq!(bun_asset_target("freebsd", "x86_64", false, true), None);
        assert_eq!(bun_asset_target("linux", "arm", false, true), None);
        assert_eq!(bun_asset_target("windows", "arm64ec", false, true), None);
    }

    #[test]
    fn bun_asset_target_appends_baseline_for_x64_without_avx2() {
        assert_eq!(
            bun_asset_target("linux", "x86_64", false, false).as_deref(),
            Some("linux-x64-baseline")
        );
        assert_eq!(
            bun_asset_target("macos", "x86_64", false, false).as_deref(),
            Some("darwin-x64-baseline")
        );
        // aarch64 ships no baseline split on any platform (the AVX2 value is
        // unused there) — Windows on ARM included.
        assert_eq!(
            bun_asset_target("linux", "aarch64", false, false).as_deref(),
            Some("linux-aarch64")
        );
        assert_eq!(
            bun_asset_target("windows", "aarch64", false, false).as_deref(),
            Some("windows-aarch64")
        );
    }

    #[test]
    fn cpuinfo_avx2_flags_line_is_detected() {
        let cpuinfo = "processor : 0\nflags\t\t: fpu vme avx2 sse4_1\n";
        assert!(cpuinfo_has_avx2(cpuinfo));
        let cpuinfo = "processor : 0\nflags\t\t: fpu vme sse4_1\n";
        assert!(!cpuinfo_has_avx2(cpuinfo));
        // The flags line must be a `flags:` line (not `Features:`).
        assert!(!cpuinfo_has_avx2("Features\t\t: avx2\n"));
    }

    #[test]
    fn sysctl_avx2_line_is_detected() {
        let out = "machdep.cpu.features: FPU VME AVX2\n";
        assert!(sysctl_has_avx2(out));
        let out = "machdep.cpu.features: FPU VME\n";
        assert!(!sysctl_has_avx2(out));
        // Case-sensitive AVX2, mirrors install.sh grep.
        assert!(!sysctl_has_avx2("machdep.cpu.features: avx2\n"));
    }

    #[test]
    fn shasums256_line_matching_finds_the_requested_asset() {
        let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let body = format!(
            "{hash}  bun-darwin-aarch64.zip\n\
             1111111111111111111111111111111111111111111111111111111111111111  bun-linux-x64.zip\n\
             2222222222222222222222222222222222222222222222222222222222222222  SHASUMS256.txt\n"
        );
        assert_eq!(
            shasums256_hash_for(&body, "bun-linux-x64.zip").as_deref(),
            Ok("1111111111111111111111111111111111111111111111111111111111111111")
        );
        // A soft failure when the chosen asset has no line (the upstream asset
        // name set is not frozen) — the caller skips and retries on the next start.
        assert_eq!(
            shasums256_hash_for(&body, "bun-linux-x64-baseline.zip")
                .expect_err("missing asset must be an Err"),
            "SHASUMS256.txt has no entry for bun-linux-x64-baseline.zip"
        );
    }
}

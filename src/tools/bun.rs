//! Managed bun runtime (JS/TS runtime, <https://bun.sh>) — silent auto-install
//! at startup and a once-per-boot auto-update, mirroring the chrome-use binary
//! management but with a silent first install (no consent flow: the Assistant
//! role directs agents to run `bun` via the Shell tool, which must resolve it).
//! Installs to the bun-standard user path (`~/.bun/bin`); updates in place.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tracing::{debug, info, warn};

/// GitHub repo whose releases host the bun runtime (single source of truth).
const BUN_RELEASE_REPO: &str = "oven-sh/bun";

/// Prefix on bun release tags (`bun-vX.Y.Z` — NOT just `v`).
const BUN_TAG_PREFIX: &str = "bun-v";

/// Timeout for resolving the latest bun release tag.
const BUN_RELEASE_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout for a bun release download (zips are ~30–90 MB).
const BUN_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Timeout for a bun CLI `--version` probe and the AVX2 sysctl probe.
const BUN_CLI_TIMEOUT: Duration = Duration::from_secs(8);

/// The platform-appropriate bun binary name.
const fn bun_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "bun.exe"
    } else {
        "bun"
    }
}

/// Only the managed bun path (`~/.bun/bin`) is probed; a user's Homebrew/npm
/// bun is never managed or updated by us, and the shell PATH order keeps the
/// user's install ahead.
fn bun_binary_path() -> Option<PathBuf> {
    crate::util::managed_bin::bun_bin_dir()
        .map(|d| d.join(bun_file_name()))
        .filter(|p| crate::util::is_executable(p))
}

/// Bun release-asset platform tag, e.g. `darwin-x64` or `linux-x64-musl`.
/// `None` on platform/arch combos the vendor does not publish. Mirrors bun's
/// official install.sh mapping; the `-baseline` suffix marks an AVX2-less x64
/// build (only x64 ships baseline builds — aarch64 has none).
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
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
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
    let Ok(out) = tokio::time::timeout(BUN_CLI_TIMEOUT, cmd.output()).await else {
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

/// Parse the local bun version from `--version` stdout (bare semver), bounded
/// by [`BUN_CLI_TIMEOUT`]. `None` when the CLI is missing, times out, exits
/// non-zero, or its output is not a parseable semver.
async fn bun_cli_version(path: &Path) -> Option<semver::Version> {
    let mut cmd = Command::new(path);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(BUN_CLI_TIMEOUT, cmd.output())
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    crate::util::managed_bin::parse_tag_version(
        String::from_utf8_lossy(&out.stdout).trim(),
        BUN_TAG_PREFIX,
    )
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
/// `asset` via [`shasums256_hash_for`].
async fn shasums256_hash(
    client: &reqwest::Client,
    tag: &str,
    asset: &str,
) -> Result<String, String> {
    let url =
        format!("https://github.com/{BUN_RELEASE_REPO}/releases/download/{tag}/SHASUMS256.txt");
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("failed to fetch SHASUMS256.txt {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "failed to fetch SHASUMS256.txt {url}: HTTP {}",
            response.status()
        ));
    }
    let body = response
        .text()
        .await
        .map_err(|e| format!("failed to read SHASUMS256.txt {url}: {e}"))?;
    shasums256_hash_for(&body, asset)
}

/// Download the bun release for `tag` for this platform, verify it against the
/// published `SHASUMS256.txt` hash, extract the single binary, and swap it into
/// the bun-standard user dir (`~/.bun/bin`). Returns the installed path.
async fn install_tag(tag: &str) -> Result<PathBuf, String> {
    use crate::util::http::{DownloadSizeCheck, build_download_client, download_verified};

    let asset = format!("bun-{}.zip", bun_asset_name().await?);
    let url = format!("https://github.com/{BUN_RELEASE_REPO}/releases/download/{tag}/{asset}");
    let client = build_download_client(BUN_DOWNLOAD_TIMEOUT)
        .map_err(|e| format!("failed to build download client: {e}"))?;
    let hash = shasums256_hash(&client, tag, &asset).await?;
    let dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;
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
    .map_err(|e| format!("failed to download {url}: {e}"))?;
    let fresh = crate::util::managed_bin::extract_single_file_zip(
        &archive_path,
        dir.path(),
        bun_file_name(),
    )?;
    let dest = crate::util::managed_bin::bun_bin_dir()
        .ok_or_else(|| "bun install dir unavailable (home not resolvable)".to_string())?
        .join(bun_file_name());
    let parent = dest
        .parent()
        .ok_or_else(|| format!("invalid bun install path {}", dest.display()))?;
    fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    crate::util::managed_bin::swap_binary_in_place(&fresh, &dest)?;
    // The swap preserves the old dest's mode, so a pre-existing install that
    // lost its executable bit would survive a reinstall unchanged — force the
    // bit back on after every swap (the fresh copy is 0o755, but the swap may
    // override it from the dest).
    crate::util::managed_bin::set_executable(&dest)?;
    Ok(dest)
}

/// Resolve the latest bun release tag and install it. Returns the installed
/// path.
async fn install_latest() -> Result<PathBuf, String> {
    let tag =
        crate::util::managed_bin::fetch_latest_tag(BUN_RELEASE_REPO, BUN_RELEASE_TIMEOUT).await?;
    install_tag(&tag).await
}

/// Once-per-boot auto-update of an existing bun install. No retry loop (a
/// failed update retries next boot); never first-installs.
async fn run_update_check() {
    let Some(path) = bun_binary_path() else {
        debug!("bun not installed — update check skipped (never first-installs)");
        return;
    };
    // Pre-feature installs and transient rc-write failures are healed by
    // ensuring user-shell visibility on every update check (idempotent,
    // non-fatal) — not just when a swap actually happens.
    crate::util::managed_bin::ensure_rc_path_block();
    let local = bun_cli_version(&path).await;
    let tag =
        match crate::util::managed_bin::fetch_latest_tag(BUN_RELEASE_REPO, BUN_RELEASE_TIMEOUT)
            .await
        {
            Ok(tag) => tag,
            Err(e) => {
                debug!("bun auto-update skipped: release check failed: {e}");
                return;
            }
        };
    let Some(latest) = crate::util::managed_bin::parse_tag_version(&tag, BUN_TAG_PREFIX) else {
        info!("bun auto-update: latest tag '{tag}' is not a semver version; giving up");
        return;
    };
    if let Some(local) = local.as_ref() {
        if local >= &latest {
            debug!("bun is up to date ({local})");
            return;
        }
    } else {
        info!(
            "bun binary at {} reports no usable version — self-healing reinstall",
            path.display()
        );
    }
    match install_tag(&tag).await {
        Ok(dest) => info!("bun auto-updated to {latest} ({})", dest.display()),
        Err(e) => info!(
            "bun auto-update failed: {}",
            crate::util::truncate(&e, 1024)
        ),
    }
}

/// Spawned one-shot task: silent first install at startup (no consent flow),
/// then a delayed once-per-boot auto-update of an existing install. All
/// failures are non-fatal (retried next boot).
pub async fn run_bun_management() {
    if bun_binary_path().is_none() {
        match install_latest().await {
            Ok(path) => {
                info!("bun installed to {}", path.display());
                crate::util::managed_bin::ensure_rc_path_block();
                // The fresh install IS the latest release — no update check.
                return;
            }
            Err(e) => {
                warn!(
                    "bun auto-install failed (non-fatal, retried on next boot): {}",
                    crate::util::truncate(&e, 1024)
                );
                // Still no binary — the delayed update check would be a
                // guaranteed no-op (it never first-installs), so end here.
                return;
            }
        }
    }
    // Existing install: once-per-boot update check, delayed like the
    // chrome-use updater so it never competes with boot.
    tokio::time::sleep(Duration::from_mins(5)).await;
    run_update_check().await;
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
        // Unsupported platform/arch combos return None.
        assert_eq!(bun_asset_target("freebsd", "x86_64", false, true), None);
        assert_eq!(bun_asset_target("linux", "arm", false, true), None);
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
        // aarch64 ships no baseline split (the AVX2 value is unused there).
        assert_eq!(
            bun_asset_target("linux", "aarch64", false, false).as_deref(),
            Some("linux-aarch64")
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
        // name set is not frozen) — the caller skips and retries next boot.
        assert_eq!(
            shasums256_hash_for(&body, "bun-linux-x64-baseline.zip")
                .expect_err("missing asset must be an Err"),
            "SHASUMS256.txt has no entry for bun-linux-x64-baseline.zip"
        );
    }
}

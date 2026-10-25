//! Install and update the Bot API server binary.
//!
//! The binary is the product's own build of `tdlib/telegram-bot-api`, published
//! in a release stream of its own: prereleases tagged `tgapi-v<version>` in the
//! product's repository. Prereleases are what keeps the product's updater
//! untouched — it reads `releases/latest` and `version.txt`, which never point
//! at one — while release assets stay downloadable by a plain HTTP client with
//! no credentials.
//!
//! Everything after the download is the shared managed-binary machinery Bun and
//! chrome-use already use: SHA-256 from the release's `SHA256SUMS`, a streamed
//! verified download, one file extracted out of the archive, an atomic swap into
//! place. The owner never fetches, unpacks or replaces anything by hand. Taking
//! the install away — the way back to the cloud — is the whole server root
//! ([`super::server::remove_state`]), binary included.

use std::path::PathBuf;

use semver::Version;

/// The tag prefix of the server's own release stream.
const TAG_PREFIX: &str = "tgapi-v";

/// Test hook: a releases listing that answers in GitHub's shape instead of the
/// real one. The product's own update channel is a different contract and is
/// never read or written here.
const RELEASES_URL_ENV: &str = "MAHBOT_TGAPI_RELEASES_URL";

/// How long the release listing may take: the archive download itself runs on the
/// client's own bound, which is sized for a tens-of-megabytes build.
const LIST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// The total bound of one release download, matching the shared download client's
/// existing policy for tens-of-megabytes artifacts.
const DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(20);

/// The client every request of this module runs on: the shared large-download
/// client, whose long total bound and deliberate 30 s connect timeout are the
/// policy an artifact of this size needs. The GitHub API is the one endpoint that
/// requires a user agent, so the listing sets it on its own request.
fn client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, String>> =
        std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            crate::util::http::build_download_client(DOWNLOAD_TIMEOUT)
                .map_err(|e| format!("the release client could not be built: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The `User-Agent` every GitHub API request needs: it answers `403` without one.
const GITHUB_USER_AGENT: &str = concat!("mahbot/", env!("CARGO_PKG_VERSION"));

/// The GitHub `owner/repo` release listing: the product's own repository, the same
/// one [`crate::self_update::RELEASE_REPO`] names, so the server stream can never
/// drift onto another host. The page query is added by [`latest`].
fn releases_url() -> String {
    let repo = crate::self_update::RELEASE_REPO
        .trim_start_matches("https://github.com/")
        .trim_end_matches('/');
    format!("https://api.github.com/repos/{repo}/releases")
}

/// The executable's file name on this platform.
const fn file_name() -> &'static str {
    if cfg!(windows) {
        "telegram-bot-api.exe"
    } else {
        "telegram-bot-api"
    }
}

/// Where the binary lives: beside the server's own state, inside the product's
/// configuration directory — never in the daemon's temp root, which is swept.
pub(super) fn binary_path() -> Option<PathBuf> {
    Some(
        crate::config::default_config_dir()
            .ok()?
            .join("telegram-bot-api")
            .join("bin")
            .join(file_name()),
    )
}

/// Whether a usable copy is installed.
pub(super) fn present() -> bool {
    binary_path().is_some_and(|path| crate::util::is_executable(&path))
}

/// One release the stream offers for this host.
pub(super) struct Release {
    tag: String,
    pub(super) version: Version,
    archive_url: String,
    sums_url: Option<String>,
}

/// This host's asset tag (`telegram-bot-api-<version>-<os>-<arch>`), or the
/// platform it is not published for.
///
/// Windows has one artifact: x64. Windows 11 on ARM emulates x64 user-mode
/// processes — file, socket and child-process behaviour included — so the same
/// artifact serves both architectures and there is no unsupported state.
fn asset_platform() -> Result<(&'static str, &'static str), String> {
    let (os, arch) = crate::util::managed_bin::host_os_arch()?;
    let arch = if os == "windows" && arch == "aarch64" {
        "x86_64"
    } else {
        arch
    };
    Ok((os, arch))
}

/// How many releases one listing page holds, and how many pages are read before a
/// missing stream is reported: the server stream shares the listing with the
/// product's own releases, so the newest server build is not on the first page
/// once a page worth of newer releases exists.
const LISTING_PAGE_SIZE: usize = 100;
const LISTING_PAGES: u32 = 10;

/// The newest release on offer for this host.
///
/// Pages are read until one carries a server build. The listing is ordered by
/// creation date, so the newest build is on the earliest page that holds one.
pub(super) async fn latest() -> Result<Release, String> {
    let (os, arch) = asset_platform()?;
    let base = std::env::var(RELEASES_URL_ENV).unwrap_or_else(|_| releases_url());
    for page in 1..=LISTING_PAGES {
        let separator = if base.contains('?') { '&' } else { '?' };
        let listing = fetch_listing(format!(
            "{base}{separator}per_page={LISTING_PAGE_SIZE}&page={page}"
        ))
        .await?;
        if let Some(release) = newest_for(&listing, os, arch) {
            return Ok(release);
        }
        // A short page is the end of the listing: no further page can hold one.
        if listing.as_array().map_or(0, Vec::len) < LISTING_PAGE_SIZE {
            break;
        }
    }
    Err(format!("no server release is published for {os}-{arch}"))
}

/// One page of the release listing.
async fn fetch_listing(url: String) -> Result<serde_json::Value, String> {
    let response = client()?
        .get(&url)
        .header(reqwest::header::USER_AGENT, GITHUB_USER_AGENT)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            format!(
                "the newest server release could not be looked up: {}",
                crate::util::managed_bin::request_reason(&e)
            )
        })?;
    if !response.status().is_success() {
        return Err(format!(
            "the newest server release could not be looked up: HTTP {}",
            response.status()
        ));
    }
    response.json().await.map_err(|e| {
        format!(
            "the release listing could not be read: {}",
            crate::util::managed_bin::request_reason(&e)
        )
    })
}

/// The Bot API version a `tgapi-v*` tag names (`tgapi-v10.3`), read with the
/// same padding the server's own banner needs.
fn parse_version(text: &str) -> Option<Version> {
    crate::util::managed_bin::parse_short_version(text.strip_prefix(TAG_PREFIX)?)
}

/// The newest `tgapi-v*` release in `listing` that carries this host's archive,
/// as the number the artifact reports.
fn newest_for(listing: &serde_json::Value, os: &str, arch: &str) -> Option<Release> {
    let extension = if os == "windows" { "zip" } else { "tar.gz" };
    listing
        .as_array()?
        .iter()
        .filter_map(|release| {
            let tag = release["tag_name"].as_str()?;
            let version = parse_version(tag)?;
            let assets = release["assets"].as_array()?;
            let asset = |suffix: &str| {
                assets.iter().find_map(|asset| {
                    let name = asset["name"].as_str()?;
                    name.ends_with(suffix)
                        .then(|| asset["browser_download_url"].as_str().map(String::from))
                        .flatten()
                })
            };
            let archive_url = asset(&format!("-{os}-{arch}.{extension}"))?;
            Some(Release {
                tag: tag.to_string(),
                version,
                archive_url,
                sums_url: asset("SHA256SUMS"),
            })
        })
        .max_by(|a, b| a.version.cmp(&b.version))
}

/// Download, verify, extract and atomically place `release`'s archive.
/// Returns the release version now installed. Every failure names the operation
/// and a reason with no local path and no URL, because it is recorded and shown
/// verbatim.
pub(super) async fn install(release: Release) -> Result<Version, String> {
    use crate::util::http::{DownloadSizeCheck, download_verified};

    let dest = binary_path().ok_or_else(|| "the install directory is unavailable".to_string())?;
    let file = file_name();
    let client = client()?;
    let hash = match &release.sums_url {
        Some(url) => {
            let body = client
                .get(url)
                .send()
                .await
                .map_err(|e| {
                    format!(
                        "the checksum list could not be fetched: {}",
                        crate::util::managed_bin::request_reason(&e)
                    )
                })?
                .text()
                .await
                .map_err(|e| {
                    format!(
                        "the checksum list could not be read: {}",
                        crate::util::managed_bin::request_reason(&e)
                    )
                })?;
            let asset = release
                .archive_url
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string();
            body.lines()
                .filter_map(crate::util::managed_bin::parse_sha256_sidecar)
                .find(|(_, name)| name == &asset)
                .map(|(hash, _)| hash)
                .ok_or_else(|| "the release publishes no checksum for this artifact".to_string())?
        }
        None => return Err("the release publishes no checksum list".to_string()),
    };

    let work = tempfile::tempdir().map_err(|e| {
        format!(
            "the download's temporary directory could not be created ({})",
            e.kind()
        )
    })?;
    let archive = work
        .path()
        .join(release.archive_url.rsplit('/').next().unwrap_or("server"));
    download_verified(
        client,
        &release.archive_url,
        &archive,
        &hash,
        None,
        DownloadSizeCheck::None,
        |_, _| {},
    )
    .await
    .map_err(|e| {
        format!(
            "the server release could not be downloaded: {}",
            crate::util::managed_bin::download_reason(&e)
        )
    })?;

    let fresh = if archive.extension().and_then(|e| e.to_str()) == Some("zip") {
        crate::util::managed_bin::extract_single_file_zip(&archive, work.path(), file)?
    } else {
        crate::util::managed_bin::extract_single_file_tar_gz(&archive, work.path(), file)?
    };
    // Proven before it replaces a working copy: a checksum-valid artifact can still
    // be unable to run (a missing library, the wrong runtime), and placing it would
    // leave nothing but a copy that fails every attempt — with no version to
    // compare, the update check would never repair it either.
    if super::server::reported_version_at(&fresh).await.is_none() {
        return Err("the downloaded server binary could not run".to_string());
    }
    crate::util::managed_bin::place_extracted(&fresh, &dest)?;
    tracing::info!(
        version = %release.version,
        tag = %release.tag,
        "telegram-bot-api: installed"
    );
    Ok(release.version)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing() -> serde_json::Value {
        serde_json::json!([
            {
                "tag_name": "tgapi-v10.1",
                "assets": [
                    {"name": "telegram-bot-api-10.1-linux-x86_64.tar.gz",
                     "browser_download_url": "https://example.test/10.1/linux"},
                    {"name": "SHA256SUMS", "browser_download_url": "https://example.test/10.1/sums"}
                ]
            },
            {
                "tag_name": "tgapi-v10.3",
                "assets": [
                    {"name": "telegram-bot-api-10.3-linux-x86_64.tar.gz",
                     "browser_download_url": "https://example.test/10.3/linux"},
                    {"name": "SHA256SUMS", "browser_download_url": "https://example.test/10.3/sums"}
                ]
            },
            {
                "tag_name": "tgapi-v10.4",
                "assets": [
                    {"name": "telegram-bot-api-10.4-windows-x86_64.zip",
                     "browser_download_url": "https://example.test/10.4/windows"},
                    {"name": "SHA256SUMS", "browser_download_url": "https://example.test/10.4/sums"}
                ]
            },
            {
                "tag_name": "v0.8.0",
                "assets": [
                    {"name": "mahbot-0.8.0-linux-x86_64.tar.gz",
                     "browser_download_url": "https://example.test/mahbot/linux"}
                ]
            }
        ])
    }

    #[test]
    fn the_newest_release_for_the_host_wins_and_the_product_channel_is_ignored() {
        let release = newest_for(&listing(), "linux", "x86_64").expect("a release");
        assert_eq!(release.version, Version::new(10, 3, 0));
        assert_eq!(release.archive_url, "https://example.test/10.3/linux");
        assert_eq!(
            release.sums_url.as_deref(),
            Some("https://example.test/10.3/sums")
        );
    }

    #[test]
    fn a_platform_without_an_artifact_is_not_served_by_another_one() {
        assert!(newest_for(&listing(), "macos", "aarch64").is_none());
        assert!(newest_for(&serde_json::json!({}), "linux", "x86_64").is_none());
    }
}

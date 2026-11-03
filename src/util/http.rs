//! Shared `reqwest::Client` construction, the SSRF screen for daemon-side fetches of a
//! caller-supplied address ([`validate_url`], [`DENIED_IP_NETS`]), and a redirect-screened GET
//! ([`get_guarded`]).
//!
//! Use this helper instead of building a client from scratch: `reqwest::Client` is meant to be
//! created once and reused — it pools connections, caches DNS and reuses TLS sessions.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

/// Install the ring crypto provider for rustls.
///
/// reqwest 0.13's `rustls-no-provider` feature leaves the process without a default
/// `CryptoProvider`, and building any client without one fails ("No rustls crypto provider is
/// configured"), so every client factory must call this first — including tests and the OpenRouter
/// benchmark, which build clients directly. Idempotent: a process-local `OnceLock` runs
/// `install_default` at most once, and `AlreadyInstalled` is ignored as belt-and-suspenders. Root
/// certificates come from the OS trust store (macOS Keychain) via rustls-platform-verifier; bundled
/// roots are deliberately not used.
pub(crate) fn install_ring_provider() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Shared client for media and provider requests: a 2-minute request timeout and a 10-second
/// connect timeout. [`media_http_client()`] hands it out and names its consumers.
static MEDIA_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// Dedicated client for image-generation POSTs: a 10-minute request timeout, since generations
/// routinely take 92–99 s and load spikes push further. Kept separate from [`MEDIA_HTTP_CLIENT`] so
/// that window serves image generation alone instead of becoming the shared media policy.
static IMAGE_GEN_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// A Bearer Authorization header value built from the configured provider API key, or `None` when
/// no key is configured (missing or empty) — callers must surface that as a clear error rather than
/// send a blank `"Bearer "` header and get an opaque 401. Used by
/// [`crate::tools::image_gen::ImageGenTool`] and, through the provider request helpers below, by
/// the video and transcription paths; a future OpenRouter media tool should reuse it too.
#[must_use]
pub fn bearer_auth_header() -> Option<String> {
    let key = crate::config::CONFIG.provider_key()?;
    if key.is_empty() {
        return None;
    }
    Some(format!("Bearer {key}"))
}

/// Safely read a response body on failure, logging a warning with the context and the error and
/// returning a fallback string.
pub(crate) async fn read_error_body(response: reqwest::Response, context: &str) -> String {
    response.text().await.unwrap_or_else(|e| {
        tracing::warn!(?e, "Failed to read {context} response body");
        "failed to read response body".to_string()
    })
}

/// A non-2xx response's body is consumed, truncated to 500 characters and wrapped in a
/// [`HttpError`](super::error::HttpError) that keeps the status and body as typed fields (via
/// `err.downcast_ref`).
async fn check_response(
    response: reqwest::Response,
    error_context: &str,
) -> anyhow::Result<reqwest::Response> {
    if !response.status().is_success() {
        let mut err = super::error::HttpError::from_response(response, error_context).await;
        // Truncate the body to keep error messages concise.
        err.body = crate::util::truncate(&err.body, 500);
        return Err(anyhow::Error::from(err));
    }
    Ok(response)
}

/// Parse a JSON response body, with a detailed error on failure: `"{error_context} response parse
/// error: {e}\nraw response body (N): {body:.500}"`, including the body length and a preview.
///
/// # Errors
///
/// - Invalid JSON: the formatted error described above.
pub(crate) fn parse_json_response(
    body_text: &str,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    serde_json::from_str(body_text).map_err(|e| {
        anyhow::anyhow!(
            "{error_context} response parse error: {e}\nraw response body ({}): {body_text:.500}",
            body_text.len(),
        )
    })
}

/// Shared request boilerplate for provider API calls: the auth header (from
/// [`bearer_auth_header()`]), the shared client (from [`media_http_client()`]), request
/// construction via `build_request`, the send, and the status check. The Authorization header is
/// injected automatically, so the closure only sets method, URL and optional body.
async fn provider_request(
    error_context: &str,
    build_request: impl FnOnce(&reqwest::Client) -> reqwest::RequestBuilder,
) -> anyhow::Result<reqwest::Response> {
    let auth = bearer_auth_header()
        .ok_or_else(|| anyhow::anyhow!("{error_context}: provider API key is not configured"))?;
    let client = media_http_client();
    let response = build_request(client)
        .header("Authorization", &auth)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} request failed: {e}"))?;
    check_response(response, error_context).await
}

/// POST JSON to a provider endpoint, check the status and parse the JSON response, using
/// [`bearer_auth_header()`] and [`media_http_client()`]. Future media tools with the same
/// POST → status-check → parse pattern should reuse it.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: a [`HttpError`](super::error::HttpError) with the status code and the response
///   body (first 500 chars), via `err.downcast_ref::<HttpError>()`
/// - JSON parse failure: the raw body length and a preview in the message.
pub async fn post_json_to_provider(
    url: &str,
    body: &serde_json::Value,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let response = provider_request(error_context, |client| client.post(url).json(body)).await?;

    let body_text = response
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))?;

    parse_json_response(&body_text, error_context)
}

/// GET JSON from a provider endpoint, check the status and parse the JSON response, using
/// [`bearer_auth_header()`] and [`media_http_client()`]. Future consumers with the same
/// GET → status-check → parse pattern should reuse it.
///
/// # Errors
///
/// - Transport errors: `"{error_context} request failed: {err}"`
/// - Non-2xx status: a [`HttpError`](super::error::HttpError) with the status code and the response
///   body (first 500 chars), via `err.downcast_ref::<HttpError>()`
/// - JSON parse failure: the raw body length and a preview in the message.
pub async fn get_json_from_provider(
    url: &str,
    error_context: &str,
) -> anyhow::Result<serde_json::Value> {
    let response = provider_request(error_context, |client| client.get(url)).await?;

    let body_text = response
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))?;

    parse_json_response(&body_text, error_context)
}

/// GET bytes from a provider endpoint, check the status, and return the raw binary response.
/// `request_timeout` overrides the client's 2-minute cap per request — the async-video tools pass
/// the remaining job window so slow downloads are not cut short.
pub async fn get_bytes_from_provider(
    url: &str,
    error_context: &str,
    request_timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    let response = provider_request(error_context, |client| {
        client.get(url).timeout(request_timeout)
    })
    .await?;

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow::anyhow!("{error_context} failed to read response body: {e}"))
}

/// The shared media HTTP client, initialised on first call with a 2-minute request timeout and a
/// 10-second connect timeout.
///
/// Direct users are [`WebSearchTool`](crate::tools::web_search::WebSearchTool) and
/// `util::upload_bridge`; everything else reaches it through [`post_json_to_provider`],
/// [`get_json_from_provider`] and [`get_bytes_from_provider`]. The 2-minute cap is a default, not a
/// guarantee: the video download passes the remaining job window, the poll is raced against it, and
/// the catalog cache bounds its own fetch at 10 seconds. Image generation uses its own client
/// ([`image_gen_http_client`]); another timeout policy should build through [`build_http_client`]
/// and hold the client in a `OnceLock`.
#[must_use]
pub fn media_http_client() -> &'static reqwest::Client {
    MEDIA_HTTP_CLIENT.get_or_init(|| build_http_client(Duration::from_mins(2)))
}

/// The dedicated image-generation client: a 10-minute request timeout and a 10-second connect
/// timeout. Used exclusively by [`crate::tools::image_gen::ImageGenTool`]'s POST — slow generations
/// routinely take 92–99 s, so the shared 2-minute media client would cut valid in-flight requests
/// short. The tool does its own retry/error handling on top (see `image_gen.rs`).
#[must_use]
pub fn image_gen_http_client() -> &'static reqwest::Client {
    IMAGE_GEN_HTTP_CLIENT.get_or_init(|| build_http_client(Duration::from_mins(10)))
}

/// Panics if `reqwest::Client::builder()` fails (typically a TLS initialization failure). TLS
/// failure is non-recoverable — a broken system TLS stack breaks everything — so the process should
/// stop immediately rather than misbehave at runtime.
#[must_use]
pub fn build_http_client(timeout: Duration) -> reqwest::Client {
    build_http_client_with(timeout, reqwest::redirect::Policy::default())
}

/// Build an HTTP client that treats a redirect as the response: the caller checked the one address
/// it supplied, and a hop the server picks would escape that check (the `web_search` picture probe
/// and [`get_guarded`] are the users).
/// # Panics
///
/// Same as [`build_http_client`].
#[must_use]
pub fn build_no_redirect_http_client(timeout: Duration) -> reqwest::Client {
    build_http_client_with(timeout, reqwest::redirect::Policy::none())
}

fn build_http_client_with(
    timeout: Duration,
    redirect: reqwest::redirect::Policy,
) -> reqwest::Client {
    install_ring_provider();
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(10))
        .redirect(redirect)
        .build()
        .expect("Failed to build HTTP client (rustls TLS initialization failure)")
}

/// IP networks [`validate_url`] refuses — the unambiguous metadata/link-local
/// targets only: link-local (incl. the 169.254.169.254 cloud-metadata endpoint),
/// the 0.0.0.0/8 source-address space, the IPv6 loopback, and all IPv4-mapped
/// IPv6. Loopback (127.0.0.1) and RFC1918 are deliberately NOT blocked: this is a
/// local-first product whose agents already drive the owner's own browser and
/// render pages from their own machine and LAN, so a file hosted there is a
/// legitimate target — including for the delivery paths that fetch one.
static DENIED_IP_NETS: std::sync::LazyLock<[ipnet::IpNet; 4]> = std::sync::LazyLock::new(|| {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    [
        ipnet::IpNet::new(IpAddr::V4(Ipv4Addr::new(169, 254, 0, 0)), 16)
            .expect("169.254.0.0/16 is a valid IPv4 net"),
        ipnet::IpNet::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8)
            .expect("0.0.0.0/8 is a valid IPv4 net"),
        ipnet::IpNet::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 128)
            .expect("::1/128 is a valid IPv6 net"),
        ipnet::IpNet::new(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0, 0)), 96)
            .expect("::ffff:0:0/96 is a valid IPv6 net"),
    ]
});

/// Validate an address is structurally safe for the daemon to fetch — the screen
/// shared by the daemon-side fetches that use it, navigation and file delivery alike.
///
/// The SSRF screen is deliberately narrow: it blocks only the unambiguous
/// metadata/link-local targets (see [`DENIED_IP_NETS`]), leaving loopback and
/// RFC1918 reachable by policy. It screens the address as written — a redirect
/// hop is a second address and must be screened again, which [`get_guarded`] does.
pub(crate) fn validate_url(url: &str) -> anyhow::Result<()> {
    use url::Host;

    let url = url.trim();
    if url.is_empty() {
        anyhow::bail!("URL cannot be empty");
    }
    // Block file:// case-insensitively — it bypasses SSRF controls.
    if url.to_ascii_lowercase().starts_with("file://") {
        anyhow::bail!("file:// URLs are not allowed");
    }
    // url::Url lowercases scheme+host, so "HTTP://EXAMPLE.COM" passes and
    // scheme-less strings ("example.com") stay rejected. Parse error or a
    // non-http(s) scheme is the same structural rejection.
    let parsed = url::Url::parse(url)
        .map_err(|_| anyhow::anyhow!("Only http:// and https:// URLs are allowed"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("Only http:// and https:// URLs are allowed");
    }

    let Some(host) = parsed.host() else {
        return Ok(());
    };
    let denied = |host: url::Host<&str>| {
        anyhow::anyhow!("host {host} is a denied link-local/metadata address (SSRF guard)")
    };
    let ip = match host {
        Host::Domain(domain) => {
            if domain.eq_ignore_ascii_case("metadata.google.internal") {
                return Err(denied(host));
            }
            None
        }
        Host::Ipv4(addr) => Some(std::net::IpAddr::V4(addr)),
        Host::Ipv6(addr) => Some(std::net::IpAddr::V6(addr)),
    };
    if ip.is_some_and(|ip| DENIED_IP_NETS.iter().any(|net| net.contains(&ip))) {
        return Err(denied(host));
    }
    Ok(())
}

/// An address as a log, a chat notice or a durable record may carry it: scheme,
/// host and path with the query and fragment dropped, so a signed URL's
/// credentials are neither quoted back to the user nor persisted. Anything that
/// is not an http(s) address is returned as written.
pub(crate) fn redact_address(value: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(value) else {
        return value.to_string();
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return value.to_string();
    }
    parsed.set_query(None);
    parsed.set_fragment(None);
    parsed.to_string()
}

/// The most hops a [`get_guarded`] fetch follows: a CDN chain is short, and a
/// bound keeps a loop from holding the caller.
const MAX_GUARDED_REDIRECTS: usize = 5;

/// GET `url`, screening every hop of the redirect chain with [`validate_url`].
///
/// The caller supplies the first address and screens it here; a hop the server
/// picks is a second address the caller never saw, so it is screened on arrival
/// instead of being trusted. `client` must not follow redirects of its own
/// ([`build_no_redirect_http_client`]) — otherwise this never sees them.
///
/// No timeout here: the caller bounds the transfer as a whole, because the bound
/// has to cover the body this returns too, not just the hops.
pub(crate) async fn get_guarded(
    client: &reqwest::Client,
    url: &str,
) -> anyhow::Result<reqwest::Response> {
    use anyhow::Context as _;

    let mut next = url.trim().to_string();
    for _ in 0..=MAX_GUARDED_REDIRECTS {
        validate_url(&next)
            .with_context(|| format!("refusing to fetch {}", redact_address(&next)))?;
        let response = client
            .get(&next)
            .send()
            .await
            .map_err(reqwest::Error::without_url)?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .context("a redirect answered with no location")?;
        next = url::Url::parse(&next)
            .ok()
            .and_then(|base| base.join(location).ok())
            .map(|target| target.to_string())
            .context("a redirect named an address that cannot be resolved")?;
    }
    anyhow::bail!(
        "{} redirected more than {MAX_GUARDED_REDIRECTS} times",
        redact_address(url)
    )
}

/// Build an HTTP client for large model downloads: `total_timeout` per
/// request plus a deliberate 30-second connect timeout.
///
/// The 30s connect timeout intentionally diverges from [`build_http_client`]'s
/// 10s — model CDNs can be slow to accept connections, so a 10s fail-fast
/// would abort large downloads. Do not "simplify" this back to 10s.
///
/// Returns `Result` so each caller keeps its own error handling (propagate via
/// `?` or degrade gracefully). `total_timeout` is a per-site argument — the
/// call sites' values are load-bearing (their retry loops wrap downloads in
/// matching outer `tokio::time::timeout`) and must not be unified.
pub(crate) fn build_download_client(total_timeout: Duration) -> anyhow::Result<reqwest::Client> {
    install_ring_provider();
    reqwest::Client::builder()
        .timeout(total_timeout)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(anyhow::Error::from)
}

/// Size-check policy applied after a streaming download completes.
pub(crate) enum DownloadSizeCheck {
    /// Require downloaded bytes to equal the Content-Length header (when present).
    Exact,
    /// No size validation.
    None,
}

/// A download whose bytes did not match the checksum the release published, so it
/// was removed rather than installed.
///
/// Typed so a caller that records the failure durably can say the file was rejected
/// instead of blaming the network — the downloader's own message names the local
/// temp path, and that is never recorded.
#[derive(Debug)]
pub(crate) struct ChecksumMismatch;

impl std::fmt::Display for ChecksumMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the downloaded file does not match the checksum published with the release")
    }
}

impl std::error::Error for ChecksumMismatch {}

/// A temp path beside `dest` that no other writer can be using.
///
/// Two fetchers of the same file — the running product's own daemon and a
/// `mahbot audio` process, say — must never share a temp path: each verifies
/// only the bytes it streamed itself, so interleaved writes into one file would
/// install a file whose bytes nobody verified. The process id and a per-process
/// sequence make the name unique, and the rename that follows publishes a whole
/// file, so the last fetcher to finish wins with a complete one. A process
/// killed mid-download leaves its part file behind; the name is never read back
/// by anything, so a leftover is inert until the next fetch of the same file
/// clears it ([`sweep_abandoned_parts`]).
fn temp_sibling(dest: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let mut tmp = dest.as_os_str().to_owned();
    tmp.push(format!(
        ".part-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    PathBuf::from(tmp)
}

/// Remove the part files a killed fetcher left beside `dest`.
///
/// A process killed before its guard could run leaves its temp behind, and the
/// model directories are the one place nothing else sweeps — so the next fetch of
/// the same file clears what the fetchers that are gone left there. Only a part
/// file whose own process is no longer there is taken: a fetch running in another
/// process (the daemon's, say) holds its own and is never touched. The names are
/// [`temp_sibling`]'s, so the process id is read straight out of one.
///
/// The walk is bounded to `dest`'s own name: a directory shared with another consumer
/// (the machine-wide model root is one) holds their files too, and none of them can
/// match this prefix.
///
/// [`download_verified`] runs this on a blocking thread: it walks a directory and
/// probes pids, and every download in the process pays for it.
fn sweep_abandoned_parts(dest: &Path) {
    let (Some(dir), Some(prefix)) = (
        dest.parent(),
        dest.file_name().and_then(|name| name.to_str()),
    ) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let pid = name
            .to_str()
            .and_then(|name| name.strip_prefix(prefix))
            .and_then(|rest| rest.strip_prefix(".part-"))
            .and_then(|rest| rest.split_once('-'))
            .and_then(|(pid, _seq)| pid.parse::<u32>().ok());
        if let Some(pid) = pid
            && pid != std::process::id()
            && !crate::util::process_alive(pid)
        {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// A temp file a download owns until it is renamed into place: dropped without
/// [`disarm`](Self::disarm), it removes whatever is at its path.
///
/// Every failure a download can meet therefore leaves nothing behind, which
/// matters most for the downloaders that retry — an orphan per attempt would
/// otherwise accumulate beside the models. (A process killed outright runs no
/// destructor; [`sweep_abandoned_parts`] is what takes that leftover, at the
/// next fetch of the same file.)
struct TempDownload {
    path: PathBuf,
    /// The rename published the file: there is nothing left to remove.
    disarmed: bool,
}

impl TempDownload {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            disarmed: false,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl Drop for TempDownload {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Create this fetch's part file, or take over one a dead process left at its name.
///
/// The name is [`temp_sibling`]'s, and its sequence never repeats within a process, so a
/// file already sitting there was left by a process whose pid was later reused: it is
/// not being written, and truncating it is what keeps one download from failing on
/// another's corpse. Anything but a regular file is refused — a model directory is
/// shared with other consumers, so a planted link must never be followed.
async fn open_part(path: &Path) -> std::io::Result<tokio::fs::File> {
    match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await
    {
        Ok(file) => Ok(file),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let meta = tokio::fs::symlink_metadata(path).await?;
            if !meta.file_type().is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("{} is not a regular file", path.display()),
                ));
            }
            tokio::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(path)
                .await
        }
        Err(e) => Err(e),
    }
}

/// On every failure path the temp file is removed (a failed rename included); on success it is
/// atomically renamed into place. A temp file a killed process left behind is taken by the next
/// fetch of the same file, and the temp name is unique per fetcher, so a concurrent download of the
/// same file can never interleave writes into this one.
pub(crate) async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    expected_sha256: &str,
    timeout: Option<Duration>,
    size_check: DownloadSizeCheck,
    mut progress: impl FnMut(u64, u64),
) -> anyhow::Result<()> {
    use anyhow::Context as _;
    use futures_util::StreamExt;
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncWriteExt;

    let mut request = client.get(url);
    if let Some(t) = timeout {
        request = request.timeout(t);
    }
    let response = request
        .send()
        .await
        .context("Failed to send download request")?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {} from {url}", response.status());
    }

    let content_length = response.content_length();
    let total_size = content_length.unwrap_or(0);

    // "Started" signal before the first byte is streamed.
    progress(0, total_size);

    // What killed fetchers left beside this file is dead weight: take it before this fetch adds
    // its own. Off the runtime thread — it is a directory walk and a process probe, and every
    // download in the process pays it.
    let sweep_dest = dest.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || sweep_abandoned_parts(&sweep_dest)).await;

    let mut temp = TempDownload::new(temp_sibling(dest));
    let mut file = open_part(temp.path())
        .await
        .context("Failed to create temp file")?;

    let mut hasher = (!expected_sha256.is_empty()).then_some(Sha256::new());
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        let len = chunk.len() as u64;
        downloaded += len;
        if let Some(h) = &mut hasher {
            h.update(&chunk);
        }
        file.write_all(&chunk)
            .await
            .context("Failed to write download chunk")?;
        progress(downloaded, total_size);
    }

    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    match size_check {
        DownloadSizeCheck::Exact => {
            if let Some(expected) = content_length
                && downloaded != expected
            {
                anyhow::bail!(
                    "Download size mismatch: expected {expected} bytes, got {downloaded} bytes"
                );
            }
        }
        DownloadSizeCheck::None => {}
    }

    if let Some(h) = hasher {
        let actual_hash = format!("{:x}", h.finalize());
        if actual_hash != expected_sha256 {
            return Err(anyhow::Error::new(ChecksumMismatch).context(format!(
                "SHA256 mismatch for {}: expected {expected_sha256}, got {actual_hash}",
                dest.display()
            )));
        }
    }

    tokio::fs::rename(temp.path(), dest)
        .await
        .with_context(|| format!("Failed to rename temp file to {}", dest.display()))?;
    temp.disarm();
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_validation_accepts_real_and_local_targets() {
        for url in [
            "https://example.com",
            "HTTP://EXAMPLE.COM",
            "https://EXAMPLE.com/path",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://192.168.1.1",
            "http://10.0.0.5",
            // Canonicalizes to 127.0.0.1 — loopback is not blocked.
            "http://0x7f000001/",
        ] {
            assert!(validate_url(url).is_ok(), "expected {url:?} to be accepted");
        }
    }

    #[test]
    fn url_validation_rejects_unsafe_targets() {
        let err = validate_url("").unwrap_err();
        assert!(err.to_string().contains("URL cannot be empty"));

        for url in ["ftp://example.com", "example.com"] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string()
                    .contains("Only http:// and https:// URLs are allowed"),
                "unexpected error for {url:?}: {err}"
            );
        }

        for url in ["FILE:///etc/passwd", "file:///etc/passwd"] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string().contains("not allowed"),
                "unexpected error for {url:?}: {err}"
            );
        }

        for url in [
            "http://169.254.169.254/",
            "http://0.0.0.0/",
            "http://0.1.2.3/",
            "http://[::1]/",
            "http://[::ffff:169.254.169.254]/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ] {
            let err = validate_url(url).unwrap_err();
            assert!(
                err.to_string().contains("SSRF guard"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    #[test]
    fn an_address_loses_its_credentials() {
        assert_eq!(
            redact_address("https://cdn.example.com/clip.mp4?sig=secret&exp=1#t=3"),
            "https://cdn.example.com/clip.mp4"
        );
        assert_eq!(
            redact_address("/tmp/workspace/clip.mp4"),
            "/tmp/workspace/clip.mp4"
        );
    }

    /// A non-2xx answer becomes the typed [`crate::util::error::HttpError`] carrying the
    /// status, while a 2xx answer is handed back as the response the caller reads.
    #[tokio::test]
    async fn check_response_maps_the_status_to_an_error_or_success() {
        let http_resp = http::Response::builder()
            .status(402)
            .body("Insufficient credits: please top up your account".to_string())
            .unwrap();
        let result = check_response(
            reqwest::Response::from(http_resp),
            "Video generation submission",
        )
        .await;
        assert!(result.is_err(), "expected error for 402 status");
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<crate::util::error::HttpError>()
                .map(|e| e.status),
            Some(402),
        );

        let http_resp = http::Response::builder()
            .status(200)
            .body(r#"{"ok": true}"#.to_string())
            .unwrap();
        assert!(
            check_response(reqwest::Response::from(http_resp), "test")
                .await
                .is_ok(),
            "expected success for 200 status"
        );
    }

    #[test]
    fn client_build_installs_ring_provider() {
        // reqwest 0.13's rustls-no-provider panics at Client build time if no rustls crypto
        // provider is installed, so every client factory installs the ring provider first —
        // this exercises the factory path and the idempotent double-install (the process
        // global is already set by earlier factories in the same test binary).
        install_ring_provider();
        let _client = build_http_client(Duration::from_secs(5));
        let _ = build_download_client(Duration::from_mins(1)).expect("download client builds");
    }

    /// A leftover part file of this very pid — the pid-reuse case, where the name is
    /// already taken by a process that is gone — is truncated rather than failing the fetch.
    #[tokio::test]
    async fn a_part_file_left_by_a_reused_pid_is_taken_over() {
        use tokio::io::AsyncWriteExt as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf.part-1-0");
        std::fs::write(&path, b"stale bytes").unwrap();
        let mut file = open_part(&path)
            .await
            .expect("a leftover regular file is taken");
        file.write_all(b"new").await.unwrap();
        // A tokio file buffers its writes: without this the bytes may never reach the disk
        // before the handle is dropped.
        file.flush().await.unwrap();
        drop(file);
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_part_file_that_is_a_link_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("somebody-elses");
        std::fs::write(&target, b"keep").unwrap();
        let path = dir.path().join("model.gguf.part-1-0");
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(open_part(&path).await.is_err(), "a link is not a part file");
        assert_eq!(std::fs::read(&target).unwrap(), b"keep");
    }
}

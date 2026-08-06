//! Ephemeral anonymous media upload bridge (video_edit local-clip and
//! image-input support).
//!
//! Uploads a local media file to the first available host in a fallback chain
//! and returns a GET-verified public HTTPS URL suitable for OpenRouter
//! `input_references`. All uploads are ephemeral by design; the caller owns
//! cleanup of the local temp file. Video uploads must be served back as
//! `video/*`; image uploads as `image/*` — verification stays per-kind.
//!
//! Chain: tmpfile.link (7-day) → uguu.se (3-h) → 0807.st (PoW + one 302) →
//! catbox.moe (hash-deduped, GET-validate) → x0.at (HEAD-404 trap — GET only).

use anyhow::Context;
use futures_util::StreamExt;
use reqwest::multipart::{Form, Part};
use sha2::{Digest, Sha256};

/// Maximum attempts for the 0807.st hashcash solve (expected 2^bits).
const POW_MAX_ATTEMPTS: u64 = 1 << 26;

/// Hard cap on the server-controlled PoW difficulty: solving past 2^24
/// hashes (~seconds) is never worth it — an over-cap challenge fails fast
/// and the fallback chain engages instead of pinning a blocking thread.
/// 0807.st's live challenge is 13 bits.
const POW_MAX_BITS: u32 = 24;

/// Parameters for one ephemeral upload: the bytes, the file name presented to
/// the host, the MIME declared in the multipart form, and the content-type
/// prefix the verified URL must be served with ("video/" or "image/").
struct UploadRequest<'a> {
    bytes: &'a [u8],
    file_name: &'a str,
    mime: &'a str,
    expected_prefix: &'a str,
}

/// An ephemeral-host uploader: takes the upload parameters and returns a
/// verified public HTTPS URL.
type Uploader<'a> =
    fn(&'a UploadRequest<'a>) -> futures_util::future::BoxFuture<'a, anyhow::Result<String>>;

/// Upload a video clip to the first working ephemeral host.
pub(crate) async fn upload_video_ephemeral(path: &std::path::Path) -> anyhow::Result<String> {
    upload_ephemeral(path, "video/mp4", "video/").await
}

/// Upload an image to the first working ephemeral host.
pub(crate) async fn upload_image_ephemeral(path: &std::path::Path) -> anyhow::Result<String> {
    let mime = crate::util::mime_for_extension(path);
    upload_ephemeral(path, mime, "image/").await
}

/// Upload `path` to the first working ephemeral host and return the verified
/// public HTTPS URL. Each host's URL is GET-validated (status, expected
/// content-type prefix, non-empty body) before being accepted.
async fn upload_ephemeral(
    path: &std::path::Path,
    mime: &str,
    expected_prefix: &str,
) -> anyhow::Result<String> {
    // Unreachable in practice (canonicalized files always have a file name),
    // but keep the fallback extension-bearing so hosts see a sane name.
    let fallback_name = if mime.starts_with("image/") {
        format!("image.{}", mime.trim_start_matches("image/"))
    } else {
        "clip.mp4".to_string()
    };
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .unwrap_or(&fallback_name)
        .to_string();
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("Failed to read media file {}", path.display()))?;

    let req = UploadRequest {
        bytes: &bytes,
        file_name: &file_name,
        mime,
        expected_prefix,
    };

    let uploaders: [Uploader; 5] = [
        |r| Box::pin(upload_tmpfile_link(r)),
        |r| Box::pin(upload_uguu(r)),
        |r| Box::pin(upload_0807_st(r)),
        |r| Box::pin(upload_catbox(r)),
        |r| Box::pin(upload_x0_at(r)),
    ];

    let mut last_error: Option<anyhow::Error> = None;
    for upload in uploaders {
        match upload(&req).await {
            Ok(url) => {
                tracing::info!(%url, "Media uploaded to ephemeral host");
                return Ok(url);
            }
            Err(e) => {
                tracing::warn!(error = %e, "Ephemeral media upload failed; trying next host");
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No ephemeral upload host available")))
}

/// GET-validate a media URL: 2xx status, expected content-type prefix,
/// non-empty body. Used to verify ephemeral-host uploads and user-supplied
/// public image URLs before they reach a billed request. The body is streamed
/// and discarded — only the first chunk is inspected. This is a served-type
/// check only, not magic-byte validation — a mislabeled container still
/// reaches OpenRouter, which rejects it at job time.
pub(crate) async fn verify_media_url(url: &str, expected_prefix: &str) -> anyhow::Result<()> {
    let resp = crate::util::http::media_http_client()
        .get(url)
        .send()
        .await
        .with_context(|| format!("Verification GET failed for {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("Verification GET returned HTTP {status} for {url}");
    }
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !content_type.starts_with(expected_prefix) {
        anyhow::bail!("Expected {expected_prefix}* content type from {url}, got {content_type}");
    }
    match resp.bytes_stream().next().await {
        Some(Ok(first)) if !first.is_empty() => Ok(()),
        Some(Ok(_)) | None => anyhow::bail!("Host served an empty body for {url}"),
        Some(Err(e)) => anyhow::bail!("Verification GET body read failed for {url}: {e}"),
    }
}

/// Multipart helper: one file part with the given field name and the
/// request's declared MIME.
fn file_part(field: &str, req: &UploadRequest<'_>) -> anyhow::Result<Form> {
    let part = Part::bytes(req.bytes.to_vec())
        .file_name(req.file_name.to_string())
        .mime_str(req.mime)?;
    Ok(Form::new().part(field.to_string(), part))
}

/// Read a JSON response body, bailing with the raw body on non-2xx or parse
/// failure (hosts return both JSON and plain-text error bodies).
async fn expect_json(resp: reqwest::Response, host: &str) -> anyhow::Result<serde_json::Value> {
    let status = resp.status();
    let body = resp
        .text()
        .await
        .with_context(|| format!("{host} response body read failed"))?;
    if !status.is_success() {
        anyhow::bail!("{host} upload failed: HTTP {status}: {body}");
    }
    serde_json::from_str(&body)
        .with_context(|| format!("{host} returned non-JSON response: {body}"))
}

/// Read a plain-text URL response body, bailing with the raw body on non-2xx
/// or a non-HTTPS response.
async fn expect_url(resp: reqwest::Response, host: &str) -> anyhow::Result<String> {
    let status = resp.status();
    let body = resp
        .text()
        .await
        .with_context(|| format!("{host} response body read failed"))?;
    if !status.is_success() {
        anyhow::bail!("{host} upload failed: HTTP {status}: {body}");
    }
    let url = body.trim();
    if !url.starts_with("https://") {
        anyhow::bail!("{host} returned an unexpected response: {body}");
    }
    Ok(url.to_string())
}

/// tmpfile.link — global Cloudflare/R2 CDN, 7-day lifetime, zero redirects.
/// The MIME declaration is required; without a correct `mime` the served type
/// degrades to `application/octet-stream`.
async fn upload_tmpfile_link(req: &UploadRequest<'_>) -> anyhow::Result<String> {
    let form = file_part("file", req)?;
    let resp = crate::util::http::media_http_client()
        .post("https://tmpfile.link/api/upload")
        .multipart(form)
        .send()
        .await
        .context("tmpfile.link upload request failed")?;
    let json = expect_json(resp, "tmpfile.link").await?;
    let url = json
        .get("downloadLink")
        .or_else(|| json.get("downloadLinkEncoded"))
        .and_then(serde_json::Value::as_str)
        .context("tmpfile.link response missing downloadLink")?;
    verify_media_url(url, req.expected_prefix).await?;
    Ok(url.to_string())
}

/// uguu.se — EU direct, 3-hour lifetime, zero redirects. Field is exactly
/// `files[]`; the server sniffs the MIME itself.
async fn upload_uguu(req: &UploadRequest<'_>) -> anyhow::Result<String> {
    let form = file_part("files[]", req)?;
    let resp = crate::util::http::media_http_client()
        .post("https://uguu.se/upload")
        .multipart(form)
        .send()
        .await
        .context("uguu.se upload request failed")?;
    let json = expect_json(resp, "uguu.se").await?;
    if json.get("success").and_then(serde_json::Value::as_bool) != Some(true) {
        anyhow::bail!("uguu.se upload reported failure: {json}");
    }
    let url = json
        .get("files")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|f| f.get("url"))
        .and_then(serde_json::Value::as_str)
        .context("uguu.se response missing files[0].url")?;
    verify_media_url(url, req.expected_prefix).await?;
    Ok(url.to_string())
}

/// 0807.st — global CDN, requires a hashcash proof-of-work and follows exactly
/// one 302 to a signed download URL. The signed URL must be verified at upload
/// time; OpenRouter follows the same redirect at job time.
async fn upload_0807_st(req: &UploadRequest<'_>) -> anyhow::Result<String> {
    let client = crate::util::http::media_http_client();

    // 1. Fetch the PoW challenge.
    let pow_resp = client
        .get("https://0807.st/pow")
        .send()
        .await
        .context("0807.st PoW request failed")?;
    let pow = expect_json(pow_resp, "0807.st PoW").await?;
    let id = pow
        .get("id")
        .and_then(serde_json::Value::as_str)
        .context("0807.st PoW missing id")?
        .to_string();
    let declared_bits = u32::try_from(
        pow.get("bits")
            .and_then(serde_json::Value::as_u64)
            .context("0807.st PoW missing bits")?,
    )
    .context("0807.st PoW bits out of range")?;
    let bits = declared_bits.min(POW_MAX_BITS);
    // ts arrives as a number today but may be string-typed; serializing a
    // JSON Value directly would quote a string field and break the submit.
    let ts = pow
        .get("ts")
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })
        .context("0807.st PoW missing ts")?;
    let sig = pow
        .get("sig")
        .and_then(serde_json::Value::as_str)
        .context("0807.st PoW missing sig")?
        .to_string();

    // 2. Solve hashcash: SHA-256(id + "." + nonce) with >= bits leading zeros.
    let id_for_solve = id.clone();
    let nonce = tokio::task::spawn_blocking(move || solve_pow(&id_for_solve, bits))
        .await
        .context("0807.st PoW solver panicked")?
        .with_context(|| format!("0807.st PoW unsolved at declared {declared_bits} bits"))?;

    // 3. Upload with the solved proof. expiry=24 keeps the clip ephemeral.
    let part = Part::bytes(req.bytes.to_vec())
        .file_name(req.file_name.to_string())
        .mime_str(req.mime)?;
    let form = Form::new()
        .part("file", part)
        .text("expiry", "24")
        .text("pow_id", id)
        .text("pow_ts", ts)
        .text("pow_bits", bits.to_string())
        .text("pow_sig", sig)
        .text("pow_nonce", nonce.to_string());
    let resp = client
        .post("https://0807.st/upload")
        .multipart(form)
        .send()
        .await
        .context("0807.st upload request failed")?;
    let json = expect_json(resp, "0807.st").await?;
    let url = json
        .get("url")
        .and_then(serde_json::Value::as_str)
        .context("0807.st response missing url")?;
    verify_media_url(url, req.expected_prefix).await?;
    Ok(url.to_string())
}

/// catbox.moe — US origin, hash-deduped: always GET-validate the returned URL
/// (a dedup hit may point at an existing upload).
async fn upload_catbox(req: &UploadRequest<'_>) -> anyhow::Result<String> {
    let part = Part::bytes(req.bytes.to_vec()).file_name(req.file_name.to_string());
    let form = Form::new()
        .text("reqtype", "fileupload")
        .part("fileToUpload", part);
    let resp = crate::util::http::media_http_client()
        .post("https://catbox.moe/user/api.php")
        .multipart(form)
        .send()
        .await
        .context("catbox.moe upload request failed")?;
    let url = expect_url(resp, "catbox.moe").await?;
    verify_media_url(&url, req.expected_prefix).await?;
    Ok(url)
}

/// x0.at — EU last resort. HEAD requests 404 even though GET works, so the
/// post-upload verification must use GET (which it does).
async fn upload_x0_at(req: &UploadRequest<'_>) -> anyhow::Result<String> {
    let form = file_part("file", req)?;
    let resp = crate::util::http::media_http_client()
        .post("https://x0.at/")
        .multipart(form)
        .send()
        .await
        .context("x0.at upload request failed")?;
    let url = expect_url(resp, "x0.at").await?;
    verify_media_url(&url, req.expected_prefix).await?;
    Ok(url)
}

/// Find a nonce such that `SHA-256(id + "." + nonce)` has at least `bits`
/// leading zero bits (hashcash-style, bytewise leading-zero counting).
fn solve_pow(id: &str, bits: u32) -> anyhow::Result<u64> {
    for nonce in 0..POW_MAX_ATTEMPTS {
        let mut hasher = Sha256::new();
        hasher.update(id.as_bytes());
        hasher.update(b".");
        hasher.update(nonce.to_string().as_bytes());
        let digest = hasher.finalize();
        if leading_zero_bits(&digest) >= bits {
            return Ok(nonce);
        }
    }
    anyhow::bail!("0807.st proof-of-work unsolved after {POW_MAX_ATTEMPTS} attempts (bits={bits})")
}

/// Count leading zero bits of a digest: each 0x00 byte contributes 8, then the
/// leading zeros of the first non-zero byte.
fn leading_zero_bits(digest: &[u8]) -> u32 {
    let mut count = 0;
    for &byte in digest {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_zero_bits_counts_bytes_and_bits() {
        // 0x00 contributes 8, then 0x05 (0000_0101) contributes 5 → 13.
        assert_eq!(leading_zero_bits(&[0x00, 0x05, 0xFF]), 13);
        assert_eq!(leading_zero_bits(&[0x01]), 7);
        assert_eq!(leading_zero_bits(&[0x00, 0x00, 0x80]), 16);
    }

    #[test]
    fn solve_pow_matches_example_nonce() {
        // Live-observed example: id "cf25698ba143dc6051399132c33359fc", bits 13,
        // nonce 12582 yields 000538c9... (13 leading zero bits). The solver may
        // find an earlier nonce, so only assert the bound.
        let nonce = solve_pow("cf25698ba143dc6051399132c33359fc", 13).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(b"cf25698ba143dc6051399132c33359fc.");
        hasher.update(nonce.to_string().as_bytes());
        assert!(leading_zero_bits(&hasher.finalize()) >= 13);
    }
}

use crate::retry::RETRY_AFTER_MAX_MS;
use crate::tools::image_catalog::{ImageModelInfo, check_image_capability};
use crate::util::error::retry_after_header;
use crate::util::http::{bearer_auth_header, read_error_body};
use crate::{Tool, Workspace};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Tool for generating images via OpenRouter's dedicated Image API.
///
/// Supports text-to-image and image-to-image generation. Accepts multiple
/// reference images on input. Returns the path to the generated file so the
/// agent can embed it as `[IMAGE:path]` in its reply.
pub struct ImageGenTool;

#[async_trait]
#[allow(clippy::too_many_lines)]
impl Tool for ImageGenTool {
    fn name(&self) -> &'static str {
        "image_gen"
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[IMAGE:")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate"
                },
                "images": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Paths to reference images for image-to-image generation"
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio (e.g. 16:9, 1:1, 4:3)"
                },
                "size": {
                    "type": "string",
                    "description": "Image size (e.g. 1K, 2K)"
                }
            }),
            &["prompt"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let prompt = super::get_str(&args, "prompt")?;
        let model = crate::config::CONFIG.image_gen_model();
        let aspect_ratio_arg = super::get_opt_str(&args, "aspect_ratio");
        let size = super::get_opt_str(&args, "size");
        let images: Vec<String> = super::get_str_array(&args, "images");

        // Capability and parameter decisions come from the catalog; a catalog
        // outage degrades to minimal user-provided parameters (fail-open).
        let catalog = crate::tools::image_catalog::get_catalog().await;
        let info = match &catalog {
            Some(catalog) => Some(check_image_capability(&model, catalog)?),
            None => None,
        };

        // Pre-flight reference-count validation BEFORE loading any files: a
        // too-many-refs error must not first read every reference into memory.
        // The catalog declares the per-model cap; the fail-open path (catalog
        // unavailable) falls back to the universal OpenRouter limit.
        if let Some(info) = info
            && !images.is_empty()
        {
            validate_reference_count(&model, info, images.len())?;
        } else if images.len() > super::MAX_REFERENCE_IMAGES_PER_REQUEST {
            anyhow::bail!(
                "Image generation supports at most {} reference image(s), got {}. \
                 Retry with fewer images.",
                super::MAX_REFERENCE_IMAGES_PER_REQUEST,
                images.len(),
            );
        }

        // Combined-size pre-flight before loading any file: the per-image
        // ceilings don't bound the total, and a pathological multi-reference
        // request must be refused without reading everything into memory first.
        if !images.is_empty() {
            crate::util::check_reference_total_input(
                &images.iter().map(PathBuf::from).collect::<Vec<_>>(),
            )
            .await?;
        }

        // Load reference images so file errors surface deterministically.
        let mut references = Vec::with_capacity(images.len());
        for img_path in &images {
            references.push(
                crate::util::load_reference_image(
                    Path::new(img_path),
                    super::MAX_REFERENCE_IMAGE_BYTES,
                )
                .await?,
            );
        }

        // Fail-open sends only an explicitly user-provided aspect ratio.
        let resolved_aspect_ratio = match info {
            Some(_) => Some(resolve_aspect_ratio(aspect_ratio_arg, &images)),
            None => aspect_ratio_arg.map(String::from),
        };

        let mut body = build_request_body(
            &model,
            prompt,
            resolved_aspect_ratio.as_deref(),
            size,
            &references,
            info,
        );

        // Aggregate body budget: per-image caps don't bound multi-reference
        // totals, so compress references further until the serialized request
        // fits the provider's ~2 MB body limit.
        super::fit_request_body_budget(&mut body, &mut references, super::MAX_REQUEST_BODY_BYTES)?;

        let api_base =
            crate::providers::ensure_base_url(&crate::config::CONFIG.provider_endpoint());
        let images_url = format!("{api_base}/images");
        let auth = bearer_auth_header().ok_or_else(|| {
            anyhow::anyhow!("Image generation: provider API key is not configured")
        })?;

        // Bounded retry loop over the dedicated 10-minute client; the loop is
        // also raced against global shutdown so a long generation cannot
        // block process teardown. Every call is recorded to telemetry at
        // call time (survives agent cancel/crash).
        let started = Instant::now();
        let shutdown = crate::shutdown::shutdown_token();
        // Publish the in-flight attempt so a shutdown-abort record reports
        // how many POSTs were actually made (the loop is mid-flight when the
        // shutdown branch fires).
        let attempt_counter = AtomicU32::new(0);
        let result = tokio::select! {
            () = shutdown.cancelled() => {
                // Record the actual elapsed time — the POST may have been in
                // flight for a long time before shutdown arrived.
                let elapsed_ms = i64::try_from(started.elapsed().as_millis()).unwrap_or(0);
                record_image_gen_call(
                    &model, &ws.path, elapsed_ms, false,
                    attempt_counter.load(Ordering::Relaxed), Some("shutdown"),
                    Some("shutdown during image generation"),
                ).await;
                anyhow::bail!("Shutting down — image generation aborted");
            }
            result = generate_image_with_retries(&images_url, &body, &auth, &attempt_counter) => result,
        };
        let elapsed_ms = i64::try_from(started.elapsed().as_millis()).unwrap_or(0);

        let (response_body, attempts, first_failure_class) = match result {
            Ok(outcome) => (outcome.body, outcome.attempts, outcome.first_failure_class),
            Err(failure) => {
                record_image_gen_call(
                    &model,
                    &ws.path,
                    elapsed_ms,
                    false,
                    failure.attempts,
                    Some(&failure.class),
                    Some(&failure.message),
                )
                .await;
                anyhow::bail!("{}", failure.message);
            }
        };

        let (b64_json, media_type) = match extract_response_image(&response_body) {
            Ok(x) => x,
            Err(err) => {
                // A 2xx body carrying a provider error (or no image data)
                // means the attempt completed and was billed — never retried.
                // The variant drives the telemetry class.
                let class = match &err {
                    ExtractImageError::Provider { .. } => "billed_error",
                    ExtractImageError::NoImageData => "no_image_data",
                };
                record_image_gen_call(
                    &model,
                    &ws.path,
                    elapsed_ms,
                    false,
                    attempts,
                    Some(class),
                    Some(&err.to_string()),
                )
                .await;
                anyhow::bail!("Image generation response error: {err}");
            }
        };

        let bytes = match STANDARD.decode(b64_json.as_bytes()) {
            Ok(bytes) => bytes,
            Err(e) => {
                // Provider returned malformed base64 — a response-processing
                // failure, recorded as such rather than as a success.
                record_image_gen_call(
                    &model,
                    &ws.path,
                    elapsed_ms,
                    false,
                    attempts,
                    Some("decode"),
                    Some(&format!("Failed to decode base64 image data: {e}")),
                )
                .await;
                anyhow::bail!("Failed to decode base64 image data from response: {e}");
            }
        };

        // Success is recorded only after the file is written, so a disk
        // failure is never recorded as a successful generation. For
        // retried-then-successful calls the first failure's class is carried
        // so analytics see the trigger cause, not just the retry count.
        let output_path = match super::save_generated_file(
            ws,
            &bytes,
            "image",
            extension_for_media_type(media_type.as_deref()),
        )
        .await
        {
            Ok(path) => path,
            Err(e) => {
                record_image_gen_call(
                    &model,
                    &ws.path,
                    elapsed_ms,
                    false,
                    attempts,
                    Some("file_write"),
                    Some(&e.to_string()),
                )
                .await;
                return Err(e);
            }
        };
        record_image_gen_call(
            &model,
            &ws.path,
            elapsed_ms,
            true,
            attempts,
            first_failure_class.as_deref(),
            None,
        )
        .await;

        Ok(self.format_media_result(&output_path))
    }
}

/// Default aspect ratio: supported by every model in the image-models catalog.
const DEFAULT_ASPECT_RATIO: &str = "9:16";

// ── Generation request: dedicated client, bounded retries, true causes ───

/// Total POST attempts (first attempt + 1 retry).
const IMAGE_GEN_MAX_ATTEMPTS: u32 = 2;

/// Retry only failures that surface within this window. Observed generations
/// complete in 92–99 s, so a failure inside this window is a prompt
/// transport/availability failure, not a cut-off generation — per OpenRouter's
/// all-or-nothing billing a generation that never completes is not billed,
/// making the retry safe. Never retry a near-full-timeout attempt: it may
/// have completed server-side (billed) with the response lost in transit.
const IMAGE_GEN_QUICK_FAILURE_MS: u64 = 60_000;

/// Backoff before the retry when the response carries no Retry-After header.
const IMAGE_GEN_BACKOFF_MS: u64 = 5_000;

/// Classification of one image-generation attempt failure, captured at the
/// HTTP boundary so the true cause survives into the surfaced error and
/// telemetry (a reqwest error's top-level `Display` hides the source chain,
/// e.g. "operation timed out").
enum AttemptFailure {
    /// Client-side timeout. `connect` distinguishes a connection timeout (the
    /// request never reached the provider — safe to retry) from the full
    /// 10-minute request timeout (the generation may have completed server-side).
    Timeout { elapsed_ms: u64, connect: bool },
    /// Transport-level failure (connection refused/reset, no HTTP response).
    Transport { elapsed_ms: u64 },
    /// Non-2xx HTTP response (provider error).
    Http {
        status: u16,
        body: String,
        retry_after_ms: Option<u64>,
        elapsed_ms: u64,
    },
    /// 2xx response whose body could not be read — the generation may have
    /// completed and been billed server-side; never retried.
    BodyRead,
    /// 2xx body that did not parse as JSON.
    Parse { message: String },
}

impl AttemptFailure {
    /// Whether this failure may be retried: prompt, transient failures only.
    /// Body-read and parse failures are never retried (billable-ambiguous or
    /// non-transient).
    fn retryable(&self) -> bool {
        match self {
            Self::Timeout {
                elapsed_ms,
                connect,
            } => {
                // Only connect timeouts are safe to retry — the request never
                // reached the provider. The elapsed guard is defensive: the
                // 10s connect cap keeps genuine connect timeouts well inside
                // the quick-failure window, while a full request timeout is a
                // possible hang and never retried.
                *connect && *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS
            }
            Self::Transport { elapsed_ms } => *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS,
            Self::Http {
                status, elapsed_ms, ..
            } => {
                matches!(status, 429 | 502 | 503 | 524 | 529)
                    && *elapsed_ms < IMAGE_GEN_QUICK_FAILURE_MS
            }
            Self::BodyRead | Self::Parse { .. } => false,
        }
    }

    fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Self::Http { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }

    /// Stable short label for telemetry.
    fn class_label(&self) -> String {
        match self {
            Self::Timeout { .. } => "timeout".to_string(),
            Self::Transport { .. } => "transport".to_string(),
            Self::Http { status, .. } => format!("http_{status}"),
            Self::BodyRead => "body_read".to_string(),
            Self::Parse { .. } => "parse".to_string(),
        }
    }
}

/// Successful image-generation request: the parsed 2xx body, the number of
/// POST attempts made (1 = no retry needed), and — when a retry was needed —
/// the class of the first failed attempt, so telemetry records the trigger
/// cause of retried-then-successful calls.
struct ImageGenOutcome {
    body: serde_json::Value,
    attempts: u32,
    first_failure_class: Option<String>,
}

/// Terminal request failure carrying the true cause for the surfaced error
/// and telemetry.
struct ImageGenFailure {
    class: String,
    attempts: u32,
    message: String,
}

/// POST the image-generation request with bounded retries.
///
/// Retry policy: auto-retry ONLY prompt failures — transport errors and HTTP
/// 429/502/503/524/529 — with backoff (Retry-After when present, else 5 s)
/// and at most one retry. Never retry: full-timeout hangs, any 4xx (402 =
/// insufficient credits), body-read failures after a 2xx (may have been
/// billed), or 200-with-error bodies (completed and billed).
async fn generate_image_with_retries(
    url: &str,
    body: &serde_json::Value,
    auth: &str,
    attempt_counter: &AtomicU32,
) -> Result<ImageGenOutcome, ImageGenFailure> {
    let mut failures: Vec<AttemptFailure> = Vec::new();
    for attempt in 1..=IMAGE_GEN_MAX_ATTEMPTS {
        attempt_counter.store(attempt, Ordering::Relaxed);
        match attempt_image_generation(url, body, auth).await {
            Ok(body) => {
                return Ok(ImageGenOutcome {
                    body,
                    attempts: attempt,
                    first_failure_class: failures.first().map(AttemptFailure::class_label),
                });
            }
            Err(failure) => {
                let retryable = failure.retryable();
                failures.push(failure);
                if !retryable || attempt == IMAGE_GEN_MAX_ATTEMPTS {
                    let last = failures.last().expect("just pushed");
                    return Err(ImageGenFailure {
                        class: last.class_label(),
                        attempts: attempt,
                        message: build_terminal_message(&failures),
                    });
                }
                let sleep_ms = failures
                    .last()
                    .and_then(AttemptFailure::retry_after_ms)
                    .map_or(IMAGE_GEN_BACKOFF_MS, |ms| {
                        ms.clamp(IMAGE_GEN_BACKOFF_MS, RETRY_AFTER_MAX_MS)
                    });
                if !crate::shutdown::sleep_or_shutdown(Duration::from_millis(sleep_ms)).await {
                    return Err(ImageGenFailure {
                        class: "shutdown".to_string(),
                        attempts: attempt,
                        message: "Shutting down — image generation aborted".to_string(),
                    });
                }
            }
        }
    }
    unreachable!("loop is bounded by IMAGE_GEN_MAX_ATTEMPTS")
}

/// One image-generation POST attempt with the dedicated 10-minute client.
/// Returns the parsed 2xx body, or a classified failure.
async fn attempt_image_generation(
    url: &str,
    body: &serde_json::Value,
    auth: &str,
) -> Result<serde_json::Value, AttemptFailure> {
    let started = Instant::now();
    let client = crate::util::http::image_gen_http_client();
    let response = match client
        .post(url)
        .header("Authorization", auth)
        .json(body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(0);
            return Err(if e.is_timeout() {
                AttemptFailure::Timeout {
                    elapsed_ms,
                    // A connect timeout never reached the provider (safe to
                    // retry); the full request timeout may have completed
                    // server-side. `is_connect()` covers DNS and TCP-level
                    // connect failures.
                    connect: e.is_connect(),
                }
            } else {
                tracing::debug!(error = %e, "Image generation transport error");
                AttemptFailure::Transport { elapsed_ms }
            });
        }
    };

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(0);
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let retry_after_ms = retry_after_header(response.headers());
        let body = read_error_body(response, "Image generation").await;
        return Err(AttemptFailure::Http {
            status,
            body,
            retry_after_ms,
            elapsed_ms,
        });
    }

    let body_text = match response.text().await {
        Ok(text) => text,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read image generation response body");
            return Err(AttemptFailure::BodyRead);
        }
    };
    let parsed =
        crate::util::http::parse_json_response(&body_text, "Image generation").map_err(|e| {
            AttemptFailure::Parse {
                message: e.to_string(),
            }
        })?;
    Ok(parsed)
}

/// Build the user-visible terminal error from the failure trail, stating the
/// actual cause (timeout vs HTTP status vs transport vs billable-ambiguous).
fn build_terminal_message(failures: &[AttemptFailure]) -> String {
    let last = failures.last().expect("at least one failure");
    let cause = match last {
        // Parse failures already carry a fully-formed message (error context
        // plus a raw-body preview) — used verbatim.
        AttemptFailure::Parse { message } => message.clone(),
        AttemptFailure::Timeout {
            elapsed_ms,
            connect,
        } => {
            let secs = elapsed_ms / 1000;
            // Only two combinations are reachable given the client's fixed
            // timeouts (10s connect cap, 10-min total): a connect timeout
            // fires inside the connect cap, a request timeout at the full cap.
            if *connect {
                format!(
                    "connection timed out after {secs} s — the request never reached \
                     the provider"
                )
            } else {
                format!(
                    "request timed out after {secs} s (client-side 10-minute limit) — \
                     the provider may still complete the generation server-side; \
                     not auto-retried"
                )
            }
        }
        AttemptFailure::Transport { elapsed_ms } => {
            if *elapsed_ms >= IMAGE_GEN_QUICK_FAILURE_MS {
                format!(
                    "transport error after {} s — the connection was lost mid-flight; the \
                     generation may have completed and been billed server-side; verify \
                     before retrying",
                    elapsed_ms / 1000,
                )
            } else {
                "transport error — the request did not reach the provider or no response \
                 was received"
                    .to_string()
            }
        }
        AttemptFailure::Http { status, body, .. } => describe_http_failure(*status, body),
        AttemptFailure::BodyRead => {
            "received a 2xx response but the body could not be read — the generation may \
             have completed and been billed server-side; verify before retrying"
                .to_string()
        }
    };
    let attempts = failures.len();
    let prefix = if attempts > 1 {
        format!("Image generation failed after {attempts} attempt(s): ")
    } else if matches!(last, AttemptFailure::Parse { .. }) {
        // A single-attempt parse failure keeps its fully-formed message
        // without an extra "Image generation failed: " prefix.
        String::new()
    } else {
        "Image generation failed: ".to_string()
    };
    format!("{prefix}{cause}")
}

/// Map an HTTP status to the typed provider error description, including the
/// provider's embedded error message when the body is parseable JSON.
fn describe_http_failure(status: u16, body: &str) -> String {
    let typed = match status {
        400 => "content policy violation or refusal (HTTP 400)".to_string(),
        402 => "insufficient credits — add credits to the provider account and retry (HTTP 402)"
            .to_string(),
        429 => "rate limit exceeded (HTTP 429)".to_string(),
        502 => "provider unavailable (HTTP 502)".to_string(),
        503 => "provider overloaded (HTTP 503)".to_string(),
        504 => "provider timeout (HTTP 504)".to_string(),
        524 | 529 => format!("edge timeout or overload (HTTP {status})"),
        _ => format!("HTTP {status}"),
    };
    let provider_msg = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").cloned())
        .and_then(|e| e.get("message").and_then(|m| m.as_str()).map(String::from));
    match provider_msg {
        Some(msg) => format!("{typed}: {msg}"),
        None => typed,
    }
}

/// Record one image-generation call in the logs store. Best-effort and at
/// call time so records survive agent cancel/crash (the session-finalize
/// flush drops stats when an agent is dropped without finalization).
async fn record_image_gen_call(
    model: &str,
    workspace: &str,
    duration_ms: i64,
    success: bool,
    attempts: u32,
    failure_class: Option<&str>,
    error_message: Option<&str>,
) {
    let Some(store) = crate::logs::LOG_STORE.get() else {
        return;
    };
    let rec = crate::stats::ImageGenCallRecord {
        model: model.to_string(),
        workspace: workspace.to_string(),
        duration_ms,
        success,
        attempts,
        failure_class: failure_class.map(String::from),
        error_message: error_message.map(String::from),
        recorded_at: crate::turso::now(),
    };
    if let Err(e) = store.record_image_gen_call(&rec).await {
        tracing::debug!(error = %e, "Failed to record image-gen call stat");
    }
}

/// Build the dedicated Image API request body from tool args and (when
/// available) the model's declared capabilities. `info` is `Some` exactly
/// when the catalog is available and the model passed the capability check;
/// `None` means the catalog is unavailable (fail-open) — only explicitly
/// user-provided parameters are sent.
///
/// Reference-count validation happens in `execute`'s pre-flight guard before
/// any file is loaded; this builder trusts the caller's count.
fn build_request_body(
    model: &str,
    prompt: &str,
    aspect_ratio: Option<&str>,
    size: Option<&str>,
    references: &[crate::util::ReferenceImage],
    info: Option<&ImageModelInfo>,
) -> serde_json::Value {
    let mut body = json!({
        "model": model,
        "prompt": prompt,
    });

    if let Some(info) = info {
        if info.declares("aspect_ratio")
            && let Some(ratio) = aspect_ratio
        {
            // "auto" is only valid where the model declares it; otherwise
            // fall back to the safe default.
            let ratio = if ratio == "auto" && !info.enum_contains("aspect_ratio", "auto") {
                DEFAULT_ASPECT_RATIO
            } else {
                ratio
            };
            body["aspect_ratio"] = json!(ratio);
        }

        if let Some(s) = size {
            if s.contains(['x', 'X', '×']) {
                if info.declares("size") {
                    body["size"] = json!(s);
                } else {
                    tracing::debug!("size `{s}` dropped — model `{model}` does not declare `size`");
                }
            } else if info.declares("resolution") {
                // Catalog resolution enums are uppercase ("1K", "2K", "4K", "512").
                body["resolution"] = json!(s.to_uppercase());
            } else {
                tracing::debug!(
                    "size `{s}` dropped — model `{model}` does not declare `resolution`"
                );
            }
        }

        if !references.is_empty() {
            body[super::INPUT_REFERENCES_KEY] = super::reference_json(references);
        }
    } else {
        // Fail-open: send only what the user explicitly provided.
        if let Some(ratio) = aspect_ratio {
            body["aspect_ratio"] = json!(ratio);
        }
        if let Some(s) = size {
            tracing::debug!(
                "size `{s}` dropped — catalog unavailable, only user-provided parameters are sent"
            );
        }
        if !references.is_empty() {
            body[super::INPUT_REFERENCES_KEY] = super::reference_json(references);
        }
    }

    body
}

/// Validate the reference count against the catalog's `input_references`
/// range. Runs in execute's pre-flight guard, before any file is loaded.
fn validate_reference_count(
    model: &str,
    info: &ImageModelInfo,
    count: usize,
) -> anyhow::Result<()> {
    match info.range_max("input_references") {
        #[allow(clippy::cast_possible_wrap)]
        Some(max) if count as i64 > max => anyhow::bail!(
            "Model `{model}` supports at most {max} reference image(s), \
             got {count}. Retry with fewer images.",
        ),
        Some(_) => {}
        None => anyhow::bail!(
            "Model `{model}` does not support reference images. \
             Retry without the `images` parameter."
        ),
    }
    Ok(())
}

/// All canonical aspect ratios supported by OpenRouter, mapped to their float
/// value (width / height). Used to find the closest match when auto-detecting
/// from a reference image.
static CANONICAL_ASPECT_RATIOS: &[(&str, f64)] = &[
    ("1:1", 1.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("4:3", 4.0 / 3.0),
    ("3:4", 3.0 / 4.0),
    ("3:2", 3.0 / 2.0),
    ("2:3", 2.0 / 3.0),
    ("4:5", 4.0 / 5.0),
    ("5:4", 5.0 / 4.0),
    ("1:2", 1.0 / 2.0),
    ("2:1", 2.0 / 1.0),
    ("1:4", 1.0 / 4.0),
    ("4:1", 4.0 / 1.0),
    ("21:9", 21.0 / 9.0),
    ("9:21", 9.0 / 21.0),
    ("1:8", 1.0 / 8.0),
    ("8:1", 8.0 / 1.0),
    ("9:19.5", 9.0 / 19.5),
    ("19.5:9", 19.5 / 9.0),
    ("9:20", 9.0 / 20.0),
    ("20:9", 20.0 / 9.0),
];

/// Resolve the effective aspect ratio: the user-provided value, the closest
/// canonical ratio detected from the first reference image, or the
/// [`DEFAULT_ASPECT_RATIO`] default.
fn resolve_aspect_ratio(aspect_ratio: Option<&str>, images: &[String]) -> String {
    match aspect_ratio {
        Some(ar) => ar.to_string(),
        None if !images.is_empty() => {
            if let Some(ratio) = detect_aspect_ratio_from_image(Path::new(&images[0])) {
                tracing::debug!(
                    "Auto-detected aspect ratio {ratio} from reference image `{}`",
                    images[0],
                );
                ratio.to_string()
            } else {
                tracing::debug!(
                    "Could not detect aspect ratio from reference image `{}`, falling back to {DEFAULT_ASPECT_RATIO}",
                    images[0],
                );
                DEFAULT_ASPECT_RATIO.to_string()
            }
        }
        None => DEFAULT_ASPECT_RATIO.to_string(),
    }
}

/// Detect the closest canonical aspect ratio from an image file.
///
/// Reads only the file header (no full decode) via the `imagesize` crate.
/// Returns `None` if the file cannot be read, is an unsupported format, or
/// has zero dimensions.
fn detect_aspect_ratio_from_image(path: &Path) -> Option<&'static str> {
    let size = imagesize::size(path).ok()?;
    find_closest_aspect_ratio(size.width, size.height)
}

/// Find the closest canonical aspect ratio string for the given dimensions.
///
/// Returns `None` when either dimension is zero.
#[allow(clippy::cast_precision_loss)]
fn find_closest_aspect_ratio(width: usize, height: usize) -> Option<&'static str> {
    // Guard against zero dimensions (would produce ∞ or panic at division)
    if width == 0 || height == 0 {
        return None;
    }

    let ratio = width as f64 / height as f64;

    // Find the closest canonical ratio via `min_by`. When two ratios are
    // equally close, the first in declaration order wins (a practical
    // impossibility with the given spacing, but `unwrap_or(Equal)` gives
    // the correct tie-break).
    CANONICAL_ASPECT_RATIOS
        .iter()
        .min_by(|a, b| {
            let da = (ratio - a.1).abs();
            let db = (ratio - b.1).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, _)| *name)
}

/// Why a 2xx image response could not be turned into an image. Both cases
/// mean the attempt completed and was billed server-side — never retried.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExtractImageError {
    /// The body carries a provider `error` object (a billed 200-with-error).
    Provider { err_type: String, message: String },
    /// The body has no usable `data[].b64_json` payload.
    NoImageData,
}

impl std::fmt::Display for ExtractImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider { err_type, message } => write!(f, "{err_type}: {message}"),
            Self::NoImageData => {
                write!(f, "response did not contain image data (data[].b64_json)")
            }
        }
    }
}

/// Extract the first non-empty `data[].b64_json` payload and its optional
/// `media_type` from a dedicated Image API response. `b64_json` is raw
/// base64 (not a data URI); `media_type` may be absent (PNG default).
///
/// Returns `Err` when the body carries a provider `error` object (surfacing
/// the embedded type/message — a billed 200-with-error response) or lacks
/// image data entirely; the variant drives the telemetry class.
fn extract_response_image(
    body: &serde_json::Value,
) -> Result<(String, Option<String>), ExtractImageError> {
    if let Some(err) = body.get("error") {
        let err_type = err
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("provider error");
        let message = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("no details");
        return Err(ExtractImageError::Provider {
            err_type: err_type.to_string(),
            message: message.to_string(),
        });
    }
    let Some(data) = body.get("data").and_then(|v| v.as_array()) else {
        return Err(ExtractImageError::NoImageData);
    };
    for entry in data {
        if let Some(b64) = entry.get("b64_json").and_then(|v| v.as_str())
            && !b64.is_empty()
        {
            let media_type = entry
                .get("media_type")
                .and_then(|v| v.as_str())
                .map(String::from);
            return Ok((b64.to_string(), media_type));
        }
    }
    Err(ExtractImageError::NoImageData)
}

/// Map a response media type to a file extension; PNG when absent or unknown.
#[must_use]
fn extension_for_media_type(media_type: Option<&str>) -> &'static str {
    match media_type {
        Some("image/jpeg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/svg+xml") => "svg",
        _ => "png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::image_catalog::{ImageCatalog, parse_catalog};

    /// Fixture catalog covering a hybrid image model (resolution + reference
    /// caps), a recraft-like model (aspect ratio only, declares "auto"), and a
    /// text-only model.
    fn fixture_catalog() -> ImageCatalog {
        parse_catalog(&json!({
            "data": [
                {
                    "id": "qwen/qwen-image-3-pro",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "resolution": { "type": "enum", "values": ["1K", "2K"] },
                        "aspect_ratio": { "type": "enum", "values": ["1:1", "9:16"] },
                        "input_references": { "type": "range", "min": 0, "max": 4 }
                    }
                },
                {
                    "id": "recraft/recraft-v4.1",
                    "architecture": { "output_modalities": ["image"] },
                    "supported_parameters": {
                        "aspect_ratio": { "type": "enum", "values": ["1:1", "9:16", "auto"] },
                        "input_references": { "type": "range", "min": 0, "max": 1 }
                    }
                },
                {
                    "id": "text-only/model",
                    "architecture": { "output_modalities": ["text"] },
                    "supported_parameters": {}
                }
            ]
        }))
        .expect("valid fixture")
    }

    /// Build `n` real validated reference images from one tiny PNG file.
    async fn refs(n: usize) -> Vec<crate::util::ReferenceImage> {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("ref.png");
        std::fs::write(&path, crate::util::test::noisy_png(1, 1)).unwrap();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(
                crate::util::load_reference_image(&path, super::super::MAX_REFERENCE_IMAGE_BYTES)
                    .await
                    .unwrap(),
            );
        }
        out
    }

    #[tokio::test]
    async fn test_build_request_body_catalog_driven() {
        let catalog = fixture_catalog();
        let qwen = catalog.find("qwen/qwen-image-3-pro");

        // size "2k" → resolution "2K" (catalog case); 9:16; two references.
        let refs = refs(2).await;
        let body = build_request_body(
            "qwen/qwen-image-3-pro",
            "a cat",
            Some("9:16"),
            Some("2k"),
            &refs,
            qwen,
        );
        assert_eq!(body["model"], "qwen/qwen-image-3-pro");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["resolution"], "2K");
        assert_eq!(body["aspect_ratio"], "9:16");
        assert_eq!(body["input_references"].as_array().unwrap().len(), 2);
        assert!(
            body["input_references"][0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "reference should be a data URI"
        );
        assert!(body.get("size").is_none());
        assert!(body.get("n").is_none());
    }

    #[tokio::test]
    async fn test_build_request_body_omits_undeclared_and_validates_caps() {
        let catalog = fixture_catalog();
        let recraft = catalog.find("recraft/recraft-v4.1");
        let qwen = catalog.find("qwen/qwen-image-3-pro");

        // No declared resolution → size dropped entirely.
        let body = build_request_body(
            "recraft/recraft-v4.1",
            "p",
            Some("9:16"),
            Some("2K"),
            &[],
            recraft,
        );
        assert!(body.get("resolution").is_none());
        assert_eq!(body["aspect_ratio"], "9:16");

        // '1024X1024' (uppercase separator) with no declared `size` → dropped,
        // never sent as a bogus resolution.
        let body = build_request_body(
            "qwen/qwen-image-3-pro",
            "p",
            None,
            Some("1024X1024"),
            &[],
            qwen,
        );
        assert!(body.get("resolution").is_none());
        assert!(body.get("size").is_none());

        // "auto" not declared by qwen → falls back to the 9:16 default.
        let body = build_request_body("qwen/qwen-image-3-pro", "p", Some("auto"), None, &[], qwen);
        assert_eq!(body["aspect_ratio"], "9:16");

        // recraft declares "auto" → passed through.
        let body = build_request_body(
            "recraft/recraft-v4.1",
            "p",
            Some("auto"),
            None,
            &[],
            recraft,
        );
        assert_eq!(body["aspect_ratio"], "auto");

        // Reference overflow → error (count validated pre-flight, not
        // truncated by the body builder).
        let err = validate_reference_count("qwen/qwen-image-3-pro", qwen.unwrap(), 5).unwrap_err();
        assert!(err.to_string().contains("at most 4 reference image(s)"));

        // Model without input_references → error when images provided.
        let text_only = catalog.find("text-only/model").unwrap();
        let err = validate_reference_count("text-only/model", text_only, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not support reference images")
        );
    }

    #[tokio::test]
    async fn test_build_request_body_fail_open_minimal() {
        // info = None (catalog unavailable): only user-provided params are sent.
        let refs = refs(1).await;
        let body = build_request_body("any/model", "p", Some("16:9"), Some("2k"), &refs, None);
        assert_eq!(body["model"], "any/model");
        assert_eq!(body["prompt"], "p");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["input_references"].as_array().unwrap().len(), 1);
        assert!(body.get("resolution").is_none());

        // No aspect ratio provided → not sent at all (no default in fail-open).
        let body = build_request_body("any/model", "p", None, None, &[], None);
        assert!(body.get("aspect_ratio").is_none());
        assert!(body.get("input_references").is_none());
    }

    #[test]
    fn test_check_image_capability_rejects_unsupported_models() {
        let catalog = fixture_catalog();
        assert!(check_image_capability("qwen/qwen-image-3-pro", &catalog).is_ok());
        let err = check_image_capability("unknown/model", &catalog).unwrap_err();
        assert!(err.to_string().contains("cannot generate images"));
        assert!(
            err.to_string()
                .contains("not in the OpenRouter image-models catalog")
        );
        let err = check_image_capability("text-only/model", &catalog).unwrap_err();
        assert!(err.to_string().contains("cannot generate images"));
        assert!(err.to_string().contains("does not list image output"));
    }

    #[test]
    fn test_check_image_capability_fail_open_on_shape_drift() {
        // Missing output_modalities (shape drift) is tolerated; explicitly
        // text-only models are still rejected.
        let catalog = parse_catalog(&json!({
            "data": [
                { "id": "drift/model", "supported_parameters": {} },
                {
                    "id": "text-only/model",
                    "architecture": { "output_modalities": ["text"] },
                    "supported_parameters": {}
                }
            ]
        }))
        .expect("valid");
        assert!(check_image_capability("drift/model", &catalog).is_ok());
        assert!(check_image_capability("text-only/model", &catalog).is_err());
    }

    #[test]
    fn test_extract_response_image_variants() {
        // b64_json is raw base64; media_type present.
        let body = json!({
            "data": [{ "b64_json": "aGVsbG8=", "media_type": "image/jpeg" }],
            "usage": { "cost": 0.001 }
        });
        let (b64, media_type) = extract_response_image(&body).expect("image");
        assert_eq!(b64, "aGVsbG8=");
        assert_eq!(media_type.as_deref(), Some("image/jpeg"));
        assert_eq!(STANDARD.decode(b64.as_bytes()).unwrap(), b"hello");

        // First non-empty entry wins; media_type absent.
        let body = json!({
            "data": [
                { "b64_json": "" },
                { "b64_json": "d29ybGQ=" }
            ]
        });
        let (b64, media_type) = extract_response_image(&body).expect("image");
        assert_eq!(b64, "d29ybGQ=");
        assert_eq!(media_type, None);

        // Empty/missing data → NoImageData.
        assert_eq!(
            extract_response_image(&json!({"data": []})).unwrap_err(),
            ExtractImageError::NoImageData
        );
        assert_eq!(
            extract_response_image(&json!({})).unwrap_err(),
            ExtractImageError::NoImageData
        );

        // A 200-with-error body surfaces the embedded type/message instead of
        // the generic "no image data" bail, as a distinct variant.
        let err_body = json!({
            "error": { "type": "provider_error", "message": "upstream refused" }
        });
        let err = extract_response_image(&err_body).unwrap_err();
        assert_eq!(err.to_string(), "provider_error: upstream refused");
        assert!(matches!(err, ExtractImageError::Provider { .. }));
    }

    #[test]
    fn test_extension_for_media_type() {
        assert_eq!(extension_for_media_type(Some("image/png")), "png");
        assert_eq!(extension_for_media_type(Some("image/jpeg")), "jpg");
        assert_eq!(extension_for_media_type(Some("image/webp")), "webp");
        assert_eq!(extension_for_media_type(Some("image/svg+xml")), "svg");
        assert_eq!(extension_for_media_type(None), "png");
        assert_eq!(
            extension_for_media_type(Some("application/octet-stream")),
            "png"
        );
    }

    // ── find_closest_aspect_ratio tests ──────────────────────────────

    #[test]
    fn test_closest_ratio_exact_match() {
        // Every canonical ratio should round-trip exactly.
        for &(ratio_str, ratio_val) in CANONICAL_ASPECT_RATIOS {
            let (w, h) = ratio_tuple_from_f64(ratio_val);
            let result = find_closest_aspect_ratio(w, h);
            assert_eq!(
                result,
                Some(ratio_str),
                "mismatch for {ratio_str} (w={w}, h={h})"
            );
        }
    }

    #[test]
    fn test_closest_ratio_between_candidates() {
        // 1400×900 ≈ 1.556 — closer to 3:2 (1.5) than to 16:9 (1.778)
        assert_eq!(find_closest_aspect_ratio(1400, 900), Some("3:2"));
        // 1700×900 ≈ 1.889 — closer to 16:9 (1.778) than to 3:2 (1.5)
        assert_eq!(find_closest_aspect_ratio(1700, 900), Some("16:9"));
        // 5×4 = 1.25 → exactly 5:4
        assert_eq!(find_closest_aspect_ratio(5, 4), Some("5:4"));
        // 17×20 = 0.85 — closer to 4:5 (0.8) than to 1:1 (1.0)
        assert_eq!(find_closest_aspect_ratio(17, 20), Some("4:5"));
    }

    #[test]
    fn test_closest_ratio_zero_dimensions() {
        assert_eq!(find_closest_aspect_ratio(0, 100), None);
        assert_eq!(find_closest_aspect_ratio(100, 0), None);
        assert_eq!(find_closest_aspect_ratio(0, 0), None);
    }

    /// Helper: convert a f64 ratio into integer width/height that produce
    /// the same ratio (within rounding). Used to construct test inputs.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn ratio_tuple_from_f64(ratio: f64) -> (usize, usize) {
        // Scale to avoid integer division rounding errors:
        // multiply by a large power of 10 then reduce.
        let scale = 10_000_000.0;
        let w = (ratio * scale).round() as usize;
        let h = scale as usize;
        (w, h)
    }

    // ── detect_aspect_ratio_from_image integration tests ──────────────

    /// A minimal valid 2×1 PNG (2:1 aspect ratio), base64-encoded.
    /// Generated with: python3 -c "..."
    const MINI_2X1_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAC0lEQVR4nGNgAAMAAAcAAbKGrPQAAAAASUVORK5CYII=";

    /// A minimal valid 16×9 PNG (16:9 aspect ratio), base64-encoded.
    const MINI_16X9_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAJCAIAAAC0SDtlAAAADklEQVR4nGNgGAVDEgAAAbkAAftY4pIAAAAASUVORK5CYII=";

    #[test]
    fn test_detect_aspect_ratio_from_real_png() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let png_bytes = STANDARD.decode(MINI_2X1_PNG_B64).expect("valid base64");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.png");
        std::fs::write(&path, &png_bytes).expect("write");

        assert_eq!(detect_aspect_ratio_from_image(&path), Some("2:1"));
    }

    #[test]
    fn test_detect_aspect_ratio_16x9_png() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};

        let png_bytes = STANDARD.decode(MINI_16X9_PNG_B64).expect("valid base64");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("wide.png");
        std::fs::write(&path, &png_bytes).expect("write");

        assert_eq!(detect_aspect_ratio_from_image(&path), Some("16:9"));
    }

    #[test]
    fn test_detect_aspect_ratio_missing_file() {
        let result = detect_aspect_ratio_from_image(Path::new("/nonexistent/image.png"));
        assert_eq!(result, None);
    }

    // ── Retry policy and true-cause reporting tests ────────────────────

    #[test]
    fn test_attempt_failure_retryable_decisions() {
        // Quick transport/timeout failures are retried (not billed per
        // OpenRouter all-or-nothing billing).
        assert!(AttemptFailure::Transport { elapsed_ms: 5_000 }.retryable());
        assert!(
            AttemptFailure::Timeout {
                elapsed_ms: 5_000,
                connect: true
            }
            .retryable()
        );
        // A near-full-timeout attempt is a hang — never retried.
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 600_000,
                connect: false
            }
            .retryable()
        );
        // A non-connect timeout is treated as a possible hang even when quick
        // (defensive; request timeouts only fire at the full cap in practice).
        assert!(
            !AttemptFailure::Timeout {
                elapsed_ms: 5_000,
                connect: false
            }
            .retryable()
        );

        // Transient provider statuses are retried; other statuses are not.
        for status in [429, 502, 503, 524, 529] {
            let f = AttemptFailure::Http {
                status,
                body: String::new(),
                retry_after_ms: None,
                elapsed_ms: 10_000,
            };
            assert!(f.retryable(), "status {status} should be retryable");
        }
        for status in [400, 401, 402, 404, 413, 500, 504] {
            let f = AttemptFailure::Http {
                status,
                body: String::new(),
                retry_after_ms: None,
                elapsed_ms: 10_000,
            };
            assert!(!f.retryable(), "status {status} should not be retryable");
        }

        // Billed-ambiguous and non-transient failures are never retried.
        assert!(!AttemptFailure::BodyRead.retryable());
        assert!(
            !AttemptFailure::Parse {
                message: "x".into()
            }
            .retryable()
        );
    }

    #[test]
    fn test_describe_http_failure_typed_codes() {
        let msg = describe_http_failure(402, "{}");
        assert!(msg.contains("insufficient credits"));

        // Provider body message is surfaced when the body is JSON.
        let msg = describe_http_failure(503, r#"{"error":{"message":"upstream busy"}}"#);
        assert!(msg.contains("provider overloaded (HTTP 503)"));
        assert!(msg.contains("upstream busy"));

        let msg = describe_http_failure(504, "plain text");
        assert!(msg.contains("provider timeout (HTTP 504)"));
        assert_eq!(describe_http_failure(418, "{}"), "HTTP 418");
    }

    #[test]
    fn test_build_terminal_message_states_true_cause() {
        // Full-cap request timeout: explicitly not auto-retried.
        let msg = build_terminal_message(&[AttemptFailure::Timeout {
            elapsed_ms: 600_000,
            connect: false,
        }]);
        assert!(msg.contains("timed out after 600 s"));
        assert!(msg.contains("not auto-retried"));

        // Retried quick failures: the attempt-count prefix carries the retry
        // context — the cause itself does not claim a retry.
        let msg = build_terminal_message(&[
            AttemptFailure::Timeout {
                elapsed_ms: 30_000,
                connect: true,
            },
            AttemptFailure::Timeout {
                elapsed_ms: 45_000,
                connect: true,
            },
        ]);
        assert!(msg.contains("failed after 2 attempt(s)"));
        assert!(msg.contains("connection timed out"));

        // A connect timeout is honest about never reaching the provider, and
        // never claims a retry it did not make.
        let msg = build_terminal_message(&[AttemptFailure::Timeout {
            elapsed_ms: 10_000,
            connect: true,
        }]);
        assert!(msg.contains("never reached the provider"));
        assert!(!msg.contains("retried"));

        // HTTP cause with provider message.
        let msg = build_terminal_message(&[AttemptFailure::Http {
            status: 503,
            body: r#"{"error":{"message":"down"}}"#.into(),
            retry_after_ms: None,
            elapsed_ms: 1_000,
        }]);
        assert!(msg.contains("provider overloaded (HTTP 503)"));

        // A long mid-flight transport failure warns about possible billing.
        let msg = build_terminal_message(&[AttemptFailure::Transport {
            elapsed_ms: 300_000,
        }]);
        assert!(msg.contains("may have completed and been billed"));

        // A retry followed by a parse failure keeps the attempt context.
        let msg = build_terminal_message(&[
            AttemptFailure::Transport { elapsed_ms: 5_000 },
            AttemptFailure::Parse {
                message: "Image generation response parse error: bad".into(),
            },
        ]);
        assert!(msg.contains("failed after 2 attempt(s)"));
        assert!(msg.contains("parse error"));
    }
}

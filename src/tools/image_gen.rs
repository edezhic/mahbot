use crate::tools::image_catalog::{ImageCatalog, ImageModelInfo, ImageOutput};
use crate::{Tool, Workspace};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use std::path::Path;

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

        // Load reference images before any request so count validation and
        // file errors surface deterministically.
        let mut reference_uris = Vec::with_capacity(images.len());
        for img_path in &images {
            reference_uris.push(
                crate::util::load_reference_image(
                    Path::new(img_path),
                    super::MAX_REFERENCE_IMAGE_BYTES,
                )
                .await?,
            );
        }

        // Capability and parameter decisions come from the catalog; a catalog
        // outage degrades to minimal user-provided parameters (fail-open).
        let catalog = crate::tools::image_catalog::get_catalog().await;
        let info = match &catalog {
            Some(catalog) => Some(check_image_capability(&model, catalog)?),
            None => None,
        };

        // Fail-open sends only an explicitly user-provided aspect ratio.
        let resolved_aspect_ratio = match info {
            Some(_) => Some(resolve_aspect_ratio(aspect_ratio_arg, &images)),
            None => aspect_ratio_arg.map(String::from),
        };

        let body = build_request_body(
            &model,
            prompt,
            resolved_aspect_ratio.as_deref(),
            size,
            &reference_uris,
            info,
        )?;

        let api_base =
            crate::providers::ensure_base_url(&crate::config::CONFIG.provider_endpoint());
        let images_url = format!("{api_base}/images");
        let response_body =
            crate::util::http::post_json_to_provider(&images_url, &body, "Image generation")
                .await?;

        let Some((b64_json, media_type)) = extract_response_image(&response_body) else {
            anyhow::bail!("Image generation response did not contain image data (data[].b64_json)");
        };

        let bytes = STANDARD.decode(b64_json.as_bytes()).map_err(|e| {
            anyhow::anyhow!("Failed to decode base64 image data from response: {e}")
        })?;

        let output_path = super::save_generated_file(
            ws,
            &bytes,
            "image",
            extension_for_media_type(media_type.as_deref()),
        )
        .await?;

        let path_str = output_path.to_string_lossy();
        let marker_prefix = self
            .media_marker()
            .expect("ImageGenTool always has a media marker");
        Ok(format!("{marker_prefix}{path_str}]"))
    }
}

/// Default aspect ratio: supported by every model in the image-models catalog.
const DEFAULT_ASPECT_RATIO: &str = "9:16";

/// Reject models that cannot generate images on the dedicated surface:
/// unknown model ids or models that explicitly declare text-only output.
/// Returns the model's catalog info for request building; models whose
/// catalog entry lacks output modalities (shape drift) are allowed through.
/// Catalog-driven — no per-model branches.
///
/// # Errors
///
/// - The model is absent from the catalog or explicitly declares no image output.
fn check_image_capability<'a>(
    model: &str,
    catalog: &'a ImageCatalog,
) -> anyhow::Result<&'a ImageModelInfo> {
    let Some(info) = catalog.find(model) else {
        anyhow::bail!(
            "Model `{model}` cannot generate images: it is not in the OpenRouter \
             image-models catalog. Set an image-capable model in Settings → \
             Image Generation and retry."
        );
    };
    if info.image_output == ImageOutput::TextOnly {
        anyhow::bail!(
            "Model `{model}` cannot generate images: the OpenRouter image-models catalog \
             does not list image output for it. Set an image-capable model in \
             Settings → Image Generation and retry."
        );
    }
    Ok(info)
}

/// Build the dedicated Image API request body from tool args and (when
/// available) the model's declared capabilities. `info` is `Some` exactly
/// when the catalog is available and the model passed the capability check;
/// `None` means the catalog is unavailable (fail-open) — only explicitly
/// user-provided parameters are sent.
///
/// # Errors
///
/// - Reference-image count exceeds the model's declared `input_references` cap.
/// - Reference images provided for a model that does not declare `input_references`.
fn build_request_body(
    model: &str,
    prompt: &str,
    aspect_ratio: Option<&str>,
    size: Option<&str>,
    reference_uris: &[String],
    info: Option<&ImageModelInfo>,
) -> anyhow::Result<serde_json::Value> {
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

        if !reference_uris.is_empty() {
            match info.range_max("input_references") {
                #[allow(clippy::cast_possible_wrap)]
                Some(max) if reference_uris.len() as i64 > max => anyhow::bail!(
                    "Model `{model}` supports at most {max} reference image(s), \
                     got {}. Retry with fewer images.",
                    reference_uris.len(),
                ),
                Some(_) => {}
                None => anyhow::bail!(
                    "Model `{model}` does not support reference images. \
                     Retry without the `images` parameter."
                ),
            }
            body["input_references"] = reference_json(reference_uris);
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
        if !reference_uris.is_empty() {
            body["input_references"] = reference_json(reference_uris);
        }
    }

    Ok(body)
}

/// Build the `input_references` array (image_url entries) for the dedicated surface.
fn reference_json(uris: &[String]) -> serde_json::Value {
    json!(
        uris.iter()
            .map(|uri| json!({
                "type": "image_url",
                "image_url": { "url": uri }
            }))
            .collect::<Vec<_>>()
    )
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

/// Extract the first non-empty `data[].b64_json` payload and its optional
/// `media_type` from a dedicated Image API response. `b64_json` is raw
/// base64 (not a data URI); `media_type` may be absent (PNG default).
fn extract_response_image(body: &serde_json::Value) -> Option<(String, Option<String>)> {
    let data = body.get("data")?.as_array()?;
    for entry in data {
        if let Some(b64) = entry.get("b64_json").and_then(|v| v.as_str())
            && !b64.is_empty()
        {
            let media_type = entry
                .get("media_type")
                .and_then(|v| v.as_str())
                .map(String::from);
            return Some((b64.to_string(), media_type));
        }
    }
    None
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

    fn refs(n: usize) -> Vec<String> {
        vec!["data:image/png;base64,a".to_string(); n]
    }

    #[test]
    fn test_build_request_body_catalog_driven() {
        let catalog = fixture_catalog();
        let qwen = catalog.find("qwen/qwen-image-3-pro");

        // size "2k" → resolution "2K" (catalog case); 9:16; two references.
        let body = build_request_body(
            "qwen/qwen-image-3-pro",
            "a cat",
            Some("9:16"),
            Some("2k"),
            &refs(2),
            qwen,
        )
        .unwrap();
        assert_eq!(body["model"], "qwen/qwen-image-3-pro");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["resolution"], "2K");
        assert_eq!(body["aspect_ratio"], "9:16");
        assert_eq!(body["input_references"].as_array().unwrap().len(), 2);
        assert_eq!(
            body["input_references"][0]["image_url"]["url"],
            "data:image/png;base64,a"
        );
        assert!(body.get("size").is_none());
        assert!(body.get("n").is_none());
    }

    #[test]
    fn test_build_request_body_omits_undeclared_and_validates_caps() {
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert!(body.get("resolution").is_none());
        assert!(body.get("size").is_none());

        // "auto" not declared by qwen → falls back to the 9:16 default.
        let body = build_request_body("qwen/qwen-image-3-pro", "p", Some("auto"), None, &[], qwen)
            .unwrap();
        assert_eq!(body["aspect_ratio"], "9:16");

        // recraft declares "auto" → passed through.
        let body = build_request_body(
            "recraft/recraft-v4.1",
            "p",
            Some("auto"),
            None,
            &[],
            recraft,
        )
        .unwrap();
        assert_eq!(body["aspect_ratio"], "auto");

        // Reference overflow → error, not truncation.
        let err = build_request_body("qwen/qwen-image-3-pro", "p", None, None, &refs(5), qwen)
            .unwrap_err();
        assert!(err.to_string().contains("at most 4 reference image(s)"));

        // Model without input_references → error when images provided.
        let text_only = catalog.find("text-only/model");
        let err = build_request_body("text-only/model", "p", None, None, &refs(1), text_only)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not support reference images")
        );
    }

    #[test]
    fn test_build_request_body_fail_open_minimal() {
        // info = None (catalog unavailable): only user-provided params are sent.
        let body =
            build_request_body("any/model", "p", Some("16:9"), Some("2k"), &refs(1), None).unwrap();
        assert_eq!(body["model"], "any/model");
        assert_eq!(body["prompt"], "p");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["input_references"].as_array().unwrap().len(), 1);
        assert!(body.get("resolution").is_none());

        // No aspect ratio provided → not sent at all (no default in fail-open).
        let body = build_request_body("any/model", "p", None, None, &[], None).unwrap();
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

        // Empty/missing data → None.
        assert!(extract_response_image(&json!({"data": []})).is_none());
        assert!(extract_response_image(&json!({})).is_none());
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
}

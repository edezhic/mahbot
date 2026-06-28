use crate::{Tool, Workspace};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use std::path::Path;

/// Tool for generating images via OpenRouter's chat completions API.
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

        let aspect_ratio = super::get_opt_str(&args, "aspect_ratio");
        let size = super::get_opt_str(&args, "size");

        let images: Vec<String> = super::get_str_array(&args, "images");

        let messages = if images.is_empty() {
            json!([{
                "role": "user",
                "content": prompt
            }])
        } else {
            let mut content_parts = Vec::new();
            content_parts.push(json!({
                "type": "text",
                "text": prompt
            }));

            for img_path in &images {
                let data_uri = crate::util::load_reference_image(
                    Path::new(img_path),
                    super::MAX_REFERENCE_IMAGE_BYTES,
                )
                .await?;
                content_parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": data_uri }
                }));
            }

            json!([{
                "role": "user",
                "content": content_parts
            }])
        };

        let mut body = json!({
            "model": model,
            "messages": messages,
            "modalities": ["image", "text"],
            "max_tokens": 4096,
        });

        // Auto-detect aspect ratio from the first reference image when one is
        // provided and the agent has not explicitly set `aspect_ratio`.
        let resolved_aspect_ratio: String = match aspect_ratio {
            Some(ar) => ar.into(),
            None if !images.is_empty() => {
                if let Some(ratio) = detect_aspect_ratio_from_image(Path::new(&images[0])) {
                    tracing::debug!(
                        "Auto-detected aspect ratio {ratio} from reference image `{}`",
                        images[0],
                    );
                    ratio.into()
                } else {
                    tracing::debug!(
                        "Could not detect aspect ratio from reference image `{}`, falling back to 9:16",
                        images[0],
                    );
                    "9:16".into()
                }
            }
            None => "9:16".into(),
        };

        body["image_config"] = json!({
            "aspect_ratio": resolved_aspect_ratio,
            "image_size": size.unwrap_or("2k"),
        });

        let endpoint = crate::config::CONFIG.provider_endpoint();
        let chat_url = crate::providers::ensure_chat_completions_url(&endpoint);

        let response_body =
            crate::util::http::post_json_to_provider(&chat_url, &body, "Image generation").await?;

        // Extract from choices[0].message.images[].image_url.url (OpenRouter format)
        let parts = extract_response_parts(&response_body);
        let Some(image_data) = parts.image_data else {
            anyhow::bail!("Image generation response did not contain image data in message.images");
        };

        let model_text = parts.text_content;

        let Some(bytes) = decode_data_uri(&image_data) else {
            anyhow::bail!("Failed to decode image data from response (expected base64 data URI)");
        };

        let output_path = super::save_generated_file(ws, &bytes, "image", "png").await?;

        let path_str = output_path.to_string_lossy();
        let marker_prefix = self
            .media_marker()
            .expect("ImageGenTool always has a media marker");
        let output = match &model_text {
            Some(text) => format!("{text}\n\n{marker_prefix}{path_str}]"),
            None => format!("{marker_prefix}{path_str}]"),
        };

        Ok(output)
    }
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

    // Find the closest canonical ratio by minimising absolute difference.
    // When two ratios are equally close, the first in declaration order wins
    // (a practical impossibility with the given spacing, but handled for
    // correctness).
    let mut best = CANONICAL_ASPECT_RATIOS[0];
    let mut best_diff = (ratio - best.1).abs();

    for entry in CANONICAL_ASPECT_RATIOS.iter().skip(1) {
        let diff = (ratio - entry.1).abs();
        if diff < best_diff {
            best = *entry;
            best_diff = diff;
        }
    }

    Some(best.0)
}

/// The parts extracted from an image generation response.
///
/// - `image_data`: the first non-empty `choices[0].message.images[].image_url.url`
/// - `text_content`: the optional text from `choices[0].message.content`
#[derive(Debug, PartialEq)]
pub(crate) struct ImageGenResponse {
    pub(crate) image_data: Option<String>,
    pub(crate) text_content: Option<String>,
}

/// Extract both image data and text content from an OpenRouter image generation
/// response via a single traversal of `body["choices"][0]["message"]`.
///
/// Images are in `choices[0].message.images[].image_url.url` (OpenRouter format).
/// Text content is in `choices[0].message.content` (Gemini-class models return both).
fn extract_response_parts(body: &serde_json::Value) -> ImageGenResponse {
    let message = body["choices"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"));

    let image_data = message
        .and_then(|msg| msg.get("images"))
        .and_then(|imgs| imgs.as_array())
        .and_then(|arr| {
            for img in arr {
                if let Some(url) = img["image_url"]["url"].as_str()
                    && !url.is_empty()
                {
                    return Some(url.to_string());
                }
            }
            None
        });

    let text_content = message
        .and_then(|msg| msg.get("content"))
        .and_then(|v| v.as_str())
        .filter(|t| !t.is_empty())
        .map(ToString::to_string);

    ImageGenResponse {
        image_data,
        text_content,
    }
}

/// Decode a base64 data URI like `data:image/png;base64,...`
fn decode_data_uri(data_uri: &str) -> Option<Vec<u8>> {
    let base64_part = data_uri.split(";base64,").nth(1)?;
    STANDARD.decode(base64_part).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper to build a minimal response body matching the OpenRouter image gen format.
    fn make_response(image_url: Option<&str>, text_content: Option<&str>) -> serde_json::Value {
        let msg = match image_url {
            Some(url) => {
                let images = json!([{"image_url": {"url": url}}]);
                match text_content {
                    Some(t) => json!({"images": images, "content": t}),
                    None => json!({"images": images}),
                }
            }
            None => match text_content {
                Some(t) => json!({"content": t}),
                None => json!({}),
            },
        };
        json!({
            "choices": [{"message": msg}]
        })
    }

    #[test]
    fn test_extract_response_parts_variants() {
        // Table-driven tests covering normal variants of extract_response_parts.
        //
        // Each case is:
        //   (image_url, text_content, expected_image_data, expected_text_content, label)
        //
        // - Empty text content → None (empty-string guard in extract_response_parts).
        let cases: Vec<(Option<&str>, Option<&str>, Option<&str>, Option<&str>, &str)> = vec![
            (
                Some("data:image/png;base64,abc123"),
                Some("Here is your generated image"),
                Some("data:image/png;base64,abc123"),
                Some("Here is your generated image"),
                "image_and_text",
            ),
            (
                Some("data:image/png;base64,def456"),
                None,
                Some("data:image/png;base64,def456"),
                None,
                "image_only",
            ),
            (
                None,
                Some("Just text, no image"),
                None,
                Some("Just text, no image"),
                "text_only",
            ),
            (
                Some("data:image/png;base64,ghi789"),
                Some(""),
                Some("data:image/png;base64,ghi789"),
                None,
                "empty_content — empty text yields None",
            ),
        ];
        for (img_url, text, exp_img, exp_text, label) in &cases {
            let body = make_response(*img_url, *text);
            let result = extract_response_parts(&body);
            assert_eq!(
                result.image_data,
                (*exp_img).map(|s| s.to_string()),
                "{label}: image_data mismatch",
            );
            assert_eq!(
                result.text_content,
                (*exp_text).map(|s| s.to_string()),
                "{label}: text_content mismatch",
            );
        }
    }

    #[test]
    fn test_extract_response_parts_graceful_empty() {
        // All these empty/missing response structures should yield (None, None)
        let cases: Vec<(&str, serde_json::Value)> = vec![
            ("empty image URL", make_response(Some(""), None)),
            (
                "null fields",
                json!({
                    "choices": [{
                        "message": {
                            "images": null,
                            "content": null,
                        }
                    }]
                }),
            ),
            ("empty choices array", json!({"choices": []})),
            ("missing choices key", json!({})),
        ];
        for (label, body) in &cases {
            let result = extract_response_parts(body);
            assert_eq!(
                result.image_data, None,
                "{label}: expected image_data to be None",
            );
            assert_eq!(
                result.text_content, None,
                "{label}: expected text_content to be None",
            );
        }
    }

    #[test]
    fn test_extract_response_parts_first_valid_image() {
        // Both cases should return the first non-empty image_url.url
        let cases: Vec<(&str, serde_json::Value, &str)> = vec![
            (
                "first of two valid URLs",
                json!({
                    "choices": [{
                        "message": {
                            "images": [
                                {"image_url": {"url": "data:image/png;base64,first"}},
                                {"image_url": {"url": "data:image/png;base64,second"}}
                            ]
                        }
                    }]
                }),
                "data:image/png;base64,first",
            ),
            (
                "skip empty, pick next valid",
                json!({
                    "choices": [{
                        "message": {
                            "images": [
                                {"image_url": {"url": ""}},
                                {"image_url": {"url": "data:image/png;base64,valid"}}
                            ]
                        }
                    }]
                }),
                "data:image/png;base64,valid",
            ),
        ];
        for (label, body, expected_url) in &cases {
            let result = extract_response_parts(body);
            assert_eq!(result.image_data, Some(expected_url.to_string()), "{label}");
            assert_eq!(result.text_content, None, "{label}");
        }
    }

    #[test]
    fn test_decode_data_uri_valid() {
        let result = decode_data_uri("data:image/png;base64,aGVsbG8=");
        assert_eq!(result, Some(b"hello".to_vec()));
    }

    #[test]
    fn test_decode_data_uri_invalid() {
        let result = decode_data_uri("not-a-data-uri");
        assert_eq!(result, None);
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

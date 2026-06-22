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

        body["image_config"] = json!({
            "aspect_ratio": aspect_ratio.unwrap_or("9:16"),
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
    fn test_extract_response_parts_image_and_text() {
        let body = make_response(
            Some("data:image/png;base64,abc123"),
            Some("Here is your generated image"),
        );
        let result = extract_response_parts(&body);
        assert_eq!(
            result.image_data,
            Some("data:image/png;base64,abc123".to_string())
        );
        assert_eq!(
            result.text_content,
            Some("Here is your generated image".to_string())
        );
    }

    #[test]
    fn test_extract_response_parts_image_only() {
        let body = make_response(Some("data:image/png;base64,def456"), None);
        let result = extract_response_parts(&body);
        assert_eq!(
            result.image_data,
            Some("data:image/png;base64,def456".to_string())
        );
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_text_only() {
        let body = make_response(None, Some("Just text, no image"));
        let result = extract_response_parts(&body);
        assert_eq!(result.image_data, None);
        assert_eq!(result.text_content, Some("Just text, no image".to_string()));
    }

    #[test]
    fn test_extract_response_parts_empty_content() {
        // Empty text content should yield None (preserving the empty-string guard)
        let body = make_response(Some("data:image/png;base64,ghi789"), Some(""));
        let result = extract_response_parts(&body);
        assert_eq!(
            result.image_data,
            Some("data:image/png;base64,ghi789".to_string())
        );
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_empty_image_url() {
        // An empty image_url.url should be skipped; falls to None if only empty URLs
        let body = json!({
            "choices": [{
                "message": {
                    "images": [{"image_url": {"url": ""}}]
                }
            }]
        });
        let result = extract_response_parts(&body);
        assert_eq!(result.image_data, None);
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_null_fields() {
        // Null/missing fields should be handled gracefully
        let body = json!({
            "choices": [{
                "message": {
                    "images": null,
                    "content": null
                }
            }]
        });
        let result = extract_response_parts(&body);
        assert_eq!(result.image_data, None);
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_empty_choices() {
        let body = json!({"choices": []});
        let result = extract_response_parts(&body);
        assert_eq!(result.image_data, None);
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_missing_choices() {
        let body = json!({});
        let result = extract_response_parts(&body);
        assert_eq!(result.image_data, None);
        assert_eq!(result.text_content, None);
    }

    #[test]
    fn test_extract_response_parts_multiple_images_picks_first() {
        // When multiple images are present, the first non-empty URL is returned
        let body = json!({
            "choices": [{
                "message": {
                    "images": [
                        {"image_url": {"url": "data:image/png;base64,first"}},
                        {"image_url": {"url": "data:image/png;base64,second"}}
                    ]
                }
            }]
        });
        let result = extract_response_parts(&body);
        assert_eq!(
            result.image_data,
            Some("data:image/png;base64,first".to_string())
        );
    }

    #[test]
    fn test_extract_response_parts_first_empty_second_valid() {
        // Skip empty URLs, pick the first non-empty one
        let body = json!({
            "choices": [{
                "message": {
                    "images": [
                        {"image_url": {"url": ""}},
                        {"image_url": {"url": "data:image/png;base64,valid"}}
                    ]
                }
            }]
        });
        let result = extract_response_parts(&body);
        assert_eq!(
            result.image_data,
            Some("data:image/png;base64,valid".to_string())
        );
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
}

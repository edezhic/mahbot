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

        let auth = crate::util::http::bearer_auth_header();
        let endpoint = crate::config::CONFIG.provider_endpoint();
        let chat_url = crate::providers::ensure_chat_completions_url(&endpoint);

        let client = crate::util::http::media_http_client();
        let response = match client
            .post(&chat_url)
            .header("Authorization", &auth)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => anyhow::bail!("API request failed: {e}"),
        };

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Image generation API error ({status}): {error_text}");
        }

        let response_body: serde_json::Value = match response.json().await {
            Ok(v) => v,
            Err(e) => anyhow::bail!("Failed to parse response: {e}"),
        };

        // Extract from choices[0].message.images[].image_url.url (OpenRouter format)
        let image_data = extract_image_data(&response_body);
        let Some(image_data) = image_data else {
            anyhow::bail!("Image generation response did not contain image data in message.images");
        };

        // Extract text content from the model response (Gemini-class models return both)
        let model_text = extract_text_content(&response_body);

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

/// Extract image data from an OpenRouter image generation response.
/// Images are in `choices[0].message.images[].image_url.url` (OpenRouter format).
fn extract_image_data(body: &serde_json::Value) -> Option<String> {
    let images = body["choices"]
        .as_array()?
        .first()?
        .get("message")?
        .get("images")?
        .as_array()?;
    for img in images {
        if let Some(url) = img["image_url"]["url"].as_str()
            && !url.is_empty()
        {
            return Some(url.to_string());
        }
    }
    None
}

/// Extract text content from the model response (Gemini-class models return both
/// text and images). Returns `None` if no text content is present.
fn extract_text_content(body: &serde_json::Value) -> Option<String> {
    let content = body["choices"]
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?;
    let text = content.as_str()?;
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Decode a base64 data URI like `data:image/png;base64,...`
fn decode_data_uri(data_uri: &str) -> Option<Vec<u8>> {
    let base64_part = data_uri.split(";base64,").nth(1)?;
    STANDARD.decode(base64_part).ok()
}

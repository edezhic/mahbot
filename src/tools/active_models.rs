//! Rendering of the `<active-models-opts>` block from the live image/video
//! model catalogs, plus the persisted model snapshot used for mid-session
//! change detection.
//!
//! The block is injected into Assistant sessions at session start and
//! re-emitted after compaction; only typed catalog fields are rendered — never
//! raw model name/description strings. Static per-model video-edit nuances
//! (see [`crate::tools::video_edit::video_edit_nuances`]) render whenever the
//! active video model matches, independently of catalog availability; the
//! remaining sections are omitted (fail-open) when their catalog is
//! unavailable and the static tool schema applies.

use std::fmt::Write;

use crate::tools::media_catalog::image::{self, ImageModelInfo, ParameterConstraint};
use crate::tools::media_catalog::video::{self, VideoModelInfo};

/// Cap per-list rendering — future catalogs must not bloat every Assistant
/// turn.
const MAX_LIST_VALUES: usize = 16;

/// Snapshot of the model ids rendered in the current `<active-models-opts>`
/// block. Persisted to `session_metadata.active_models` (JSON) and compared
/// against live config reads on each user message to detect mid-session
/// model changes. Each field is `Some` only when that model's section was
/// actually rendered (fail-open: a section absent from the block means no
/// baseline for that model, so a later switch of it never fires a
/// change-info). Records model ids only — a provider-endpoint switch with the
/// same model id is invisible until the next compaction re-renders the block
/// (accepted per ticket scope).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ModelSnapshot {
    pub(crate) image: Option<String>,
    pub(crate) video: Option<String>,
}

impl ModelSnapshot {
    /// Read the currently active image/video models for `user_name` (their
    /// per-user picker choice, falling back to the hardcoded default).
    pub(crate) async fn from_user(user_name: &str) -> Self {
        Self {
            image: Some(crate::users::resolve_image_gen_model(user_name).await),
            video: Some(crate::users::resolve_video_model(user_name).await),
        }
    }

    /// Serialize for the `session_metadata.active_models` column.
    #[must_use]
    pub(crate) fn to_json(&self) -> String {
        serde_json::json!({ "image": self.image, "video": self.video }).to_string()
    }

    /// Parse a persisted snapshot; `None` for absent/malformed values (no
    /// baseline to compare against).
    #[must_use]
    pub(crate) fn from_json(s: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(Self {
            image: v
                .get("image")
                .and_then(serde_json::Value::as_str)
                .map(String::from),
            video: v
                .get("video")
                .and_then(serde_json::Value::as_str)
                .map(String::from),
        })
    }
}

/// Which media surface changed mid-session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelKind {
    Image,
    Video,
}

impl ModelKind {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            ModelKind::Image => "image",
            ModelKind::Video => "video",
        }
    }
}

/// Render the full `<active-models-opts>` block for `user_name`'s active
/// models (their per-user picker choice, falling back to the hardcoded
/// default), or `None` when no section renders. Image and video sections render
/// independently — an unavailable catalog suppresses only its own section, and
/// static video-edit nuances still render without a catalog entry. The
/// returned snapshot records exactly which model ids were rendered (the
/// persisted baseline must never be re-derived from the live per-user
/// resolution).
pub(crate) async fn render_block(user_name: &str) -> Option<(String, ModelSnapshot)> {
    let current = ModelSnapshot::from_user(user_name).await;
    let (image, video) = tokio::join!(
        render_side(ModelKind::Image, current.image.as_deref()),
        render_side(ModelKind::Video, current.video.as_deref()),
    );

    let mut out = String::from("<active-models-opts>\n");
    let mut first = true;
    for (_, section) in [&image, &video].into_iter().flatten() {
        if !first {
            out.push('\n');
        }
        out.push_str(section);
        first = false;
    }
    if first {
        return None;
    }
    out.push_str("</active-models-opts>");
    Some((
        out,
        ModelSnapshot {
            image: image.as_ref().map(|(id, _)| id.clone()),
            video: video.as_ref().map(|(id, _)| id.clone()),
        },
    ))
}

/// Render one model's section for the block, or `None` when the model is
/// absent from the catalog / the catalog is unavailable / the model declares
/// no renderable capability envelope and has no static edit nuances (a
/// header-only section would be noise).
async fn render_side(kind: ModelKind, model: Option<&str>) -> Option<(String, String)> {
    let model = model?;
    let section = render_section(kind, model).await?;
    Some((model.to_string(), section))
}

/// Render the capability section for one model (used by the session-start
/// block and the mid-session change-info). Returns `None` when the model is
/// absent from the catalog, the catalog is unavailable, and the model has no
/// static edit nuances — callers fall back to a change-only line. The catalog
/// fetch is timeout-bounded internally (see [`crate::util::catalog_cache`]).
pub(crate) async fn render_section(kind: ModelKind, model: &str) -> Option<String> {
    match kind {
        ModelKind::Image => {
            let catalog = image::get_catalog().await?;
            catalog
                .find(model)
                .and_then(|info| render_image_section(model, info))
        }
        ModelKind::Video => {
            // Catalog fields are optional: static per-model edit nuances render
            // even when the catalog is unavailable (fail-open must not drop
            // nuance text on model switch).
            let catalog = video::get_catalog().await;
            let info = catalog.as_ref().and_then(|c| c.find(model));
            render_video_section(model, info)
        }
    }
}

/// Catalog `supported_parameters` keys → `image_gen` tool schema names. Only
/// mapped parameters render: the tool silently discards unknown keys, so
/// listing a raw catalog key would mislead the LLM into passing an
/// unactionable argument.
fn tool_param_name(catalog_key: &str) -> Option<&'static str> {
    match catalog_key {
        // Identity mapping — the runtime image_gen tool honors declares("size").
        "resolution" | "size" => Some("size"),
        "aspect_ratio" => Some("aspect_ratio"),
        "input_references" => Some("images"),
        _ => None,
    }
}

/// Render the declared image parameters (typed catalog fields only, mapped to
/// the `image_gen` tool parameter names). `None` when the model declares no
/// renderable envelope (only unmapped/unknown parameters) — a header-only
/// section would carry no actionable capability info. Rendering is derived
/// from the buffer: the header is the entire prefix written before any field,
/// so a longer buffer means the section rendered.
fn render_image_section(model: &str, info: &ImageModelInfo) -> Option<String> {
    let mut out = format!("Image model: {model}\n");
    let header = out.len();
    let mut params: Vec<(&str, &ParameterConstraint)> = info
        .supported_parameters
        .iter()
        .filter_map(|(name, constraint)| tool_param_name(name).map(|tool| (tool, constraint)))
        .collect();
    // A model declaring both "resolution" and "size" maps to the same tool
    // param — dedupe by tool name so the envelope lists it once.
    params.sort_by_key(|(name, _)| *name);
    params.dedup_by_key(|(name, _)| *name);
    for (name, constraint) in params {
        match constraint {
            ParameterConstraint::Enum(values) => {
                let _ = writeln!(out, "- {name}: {}", join_capped(values, MAX_LIST_VALUES));
            }
            ParameterConstraint::Range { max } => {
                // A model may declare more reference images than the provider
                // accepts — the advertised maximum is clamped to the ceiling.
                let max = if name == "images" {
                    super::effective_reference_max(*max)
                } else {
                    *max
                };
                let _ = writeln!(out, "- {name}: max {max}");
            }
            ParameterConstraint::Boolean => {
                let _ = writeln!(out, "- {name}: supported");
            }
            ParameterConstraint::Unknown => {}
        }
    }
    (out.len() > header).then_some(out)
}

/// Render the declared video capabilities (typed catalog fields only; the
/// field names match the `video_gen` tool schema) plus the static per-model
/// video-edit nuances. `None` when the model declares no capability fields
/// and no nuance data (a header-only section would be noise). Rendering is
/// derived from the buffer: the header is the entire prefix written before any
/// field, so a longer buffer means the section rendered — a helper collapsing
/// the repeated `- label: …` lines must therefore write to the buffer and
/// return nothing, never a flag.
fn render_video_section(model: &str, info: Option<&VideoModelInfo>) -> Option<String> {
    let mut out = format!("Video model: {model}\n");
    let header = out.len();
    if let Some(info) = info {
        if let Some(resolutions) = &info.resolutions
            && !resolutions.is_empty()
        {
            let _ = writeln!(
                out,
                "- resolution: {}",
                join_capped(resolutions, MAX_LIST_VALUES)
            );
        }
        if let Some(ratios) = &info.aspect_ratios
            && !ratios.is_empty()
        {
            let _ = writeln!(
                out,
                "- aspect_ratio: {}",
                join_capped(ratios, MAX_LIST_VALUES)
            );
        }
        if let Some(durations) = &info.durations
            && !durations.is_empty()
        {
            let _ = writeln!(out, "- duration: {} seconds", format_durations(durations));
        }
        if let Some(sizes) = &info.sizes
            && !sizes.is_empty()
        {
            let _ = writeln!(out, "- size: {}", join_capped(sizes, MAX_LIST_VALUES));
        }
        if let Some(frames) = &info.frame_images
            && !frames.is_empty()
        {
            let _ = writeln!(
                out,
                "- frame images: {}",
                join_capped(frames, MAX_LIST_VALUES)
            );
        }
        if let Some(audio) = info.generate_audio {
            let _ = writeln!(
                out,
                "- generate_audio: {}",
                if audio { "yes" } else { "no" }
            );
        }
        if let Some(seed) = info.seed {
            let _ = writeln!(out, "- seed: {}", if seed { "yes" } else { "no" });
        }
    }
    if let Some(nuance) = crate::tools::video_edit::video_edit_nuances(model) {
        let _ = writeln!(out, "- editing: {nuance}");
    }
    (out.len() > header).then_some(out)
}

/// Join values, capping the rendered list so future catalogs cannot bloat
/// every Assistant turn.
fn join_capped(values: &[String], cap: usize) -> String {
    if values.len() <= cap {
        values.join(", ")
    } else {
        format!("{}, +{} more", values[..cap].join(", "), values.len() - cap)
    }
}

/// Compact durations: a contiguous set of values renders as "min-max",
/// otherwise as a capped comma list. Duplicates are collapsed before the
/// contiguity check so `[5,6,6,8]` never renders as "5-8" (7 would be
/// advertised as supported).
#[expect(clippy::cast_possible_wrap)]
fn format_durations(durations: &[i64]) -> String {
    if let [single] = durations {
        return single.to_string();
    }
    let mut distinct: Vec<i64> = durations.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if let (Some(&min), Some(&max)) = (distinct.first(), distinct.last())
        && distinct.len() as i64 == max.saturating_sub(min).saturating_add(1)
    {
        return format!("{min}-{max}");
    }
    let values: Vec<String> = distinct.iter().map(i64::to_string).collect();
    join_capped(&values, MAX_LIST_VALUES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_from_json_tolerates_garbage() {
        assert!(ModelSnapshot::from_json("not json").is_none());
        assert!(ModelSnapshot::from_json("").is_none());
        assert!(ModelSnapshot::from_json("{\"image\":\"only\"}").is_some());
    }

    #[test]
    fn join_capped_truncates() {
        let values: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        assert_eq!(
            join_capped(&values, 16),
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, +4 more"
        );
        assert_eq!(join_capped(&values[..3], 16), "0, 1, 2");
    }

    /// Build a catalog entry declaring `params` — the shipped helper for the
    /// envelope cases below, whose rows differ only in what is declared.
    fn image_info(params: &[(&str, ParameterConstraint)]) -> ImageModelInfo {
        let mut info = ImageModelInfo::default();
        for (name, constraint) in params {
            info.supported_parameters
                .insert((*name).to_string(), constraint.clone());
        }
        info
    }

    /// One envelope row: the declared capabilities, and the lines the section
    /// must carry together with how many mapped lines it renders — `None` when
    /// no section may render at all.
    struct ImageEnvelopeCase {
        name: &'static str,
        info: ImageModelInfo,
        expected: Option<(&'static [&'static str], usize)>,
    }

    /// Guards the three decisions the image envelope makes: the declared
    /// reference maximum is clamped to the provider ceiling (the live catalog
    /// declares 20), `resolution`/`size` are one tool parameter listed once,
    /// and a model declaring nothing the tool accepts renders no section.
    #[test]
    fn the_image_envelope_clamps_the_reference_max_lists_size_once_and_omits_an_unmapped_model() {
        let cases = [
            ImageEnvelopeCase {
                name: "a max within the ceiling renders unchanged",
                info: image_info(&[("input_references", ParameterConstraint::Range { max: 4 })]),
                expected: Some((&["- images: max 4\n"], 1)),
            },
            ImageEnvelopeCase {
                name: "a declared max above the ceiling is clamped",
                info: image_info(&[("input_references", ParameterConstraint::Range { max: 20 })]),
                expected: Some((&["- images: max 16\n"], 1)),
            },
            ImageEnvelopeCase {
                name: "size is identity-mapped",
                info: image_info(&[("size", ParameterConstraint::Enum(vec!["512x512".into()]))]),
                expected: Some((&["- size: 512x512\n"], 1)),
            },
            ImageEnvelopeCase {
                name: "size and resolution collapse to one line",
                info: image_info(&[
                    (
                        "resolution",
                        ParameterConstraint::Enum(vec!["1K".into(), "2K".into()]),
                    ),
                    ("size", ParameterConstraint::Enum(vec!["512x512".into()])),
                ]),
                expected: Some((&[], 1)),
            },
            ImageEnvelopeCase {
                name: "a model with only unmapped params renders nothing",
                info: image_info(&[
                    ("n", ParameterConstraint::Range { max: 4 }),
                    ("future", ParameterConstraint::Unknown),
                ]),
                expected: None,
            },
        ];

        for case in &cases {
            let name = case.name;
            let rendered = render_image_section("test/model", &case.info);
            let Some((contains, mapped_lines)) = case.expected else {
                assert!(
                    rendered.is_none(),
                    "case: {name} — expected no section, got {rendered:?}"
                );
                continue;
            };
            let out = rendered.unwrap_or_else(|| panic!("case: {name} — expected a section"));
            assert!(
                out.starts_with("Image model: test/model\n"),
                "case: {name}: {out}"
            );
            for line in contains {
                assert!(
                    out.contains(line),
                    "case: {name} — expected {line:?}: {out}"
                );
            }
            assert_eq!(
                out.lines().filter(|line| line.starts_with("- ")).count(),
                mapped_lines,
                "case: {name} — mapped-line count: {out}"
            );
        }
    }
}

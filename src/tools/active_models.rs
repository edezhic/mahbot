//! Rendering of the `<active-models-opts>` block from the live image/video
//! model catalogs, plus the persisted model snapshot used for mid-session
//! change detection.
//!
//! The block is injected into Artist sessions at session start and re-emitted
//! after compaction; only typed catalog fields are rendered — never raw model
//! name/description strings. Static per-model video-edit nuances (see
//! [`crate::tools::video_edit::video_edit_nuances`]) render whenever the
//! active video model matches, independently of catalog availability; the
//! remaining sections are omitted (fail-open) when their catalog is
//! unavailable and the static tool schema applies.

use std::fmt::Write;

use crate::config::CONFIG;
use crate::tools::image_catalog::{self, ImageModelInfo, ParameterConstraint};
use crate::tools::video_catalog::{self, VideoModelInfo};

/// Cap per-list rendering — future catalogs must not bloat every Artist turn.
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
    /// Read the currently configured image/video models.
    #[must_use]
    pub(crate) fn from_config() -> Self {
        Self {
            image: Some(CONFIG.image_gen_model()),
            video: Some(CONFIG.video_model()),
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

/// Render the full `<active-models-opts>` block for the currently configured
/// models, or `None` when no section renders. Image and video sections render
/// independently — an unavailable catalog suppresses only its own section, and
/// static video-edit nuances still render without a catalog entry. The
/// returned snapshot records exactly which model ids were rendered (the
/// persisted baseline must never be re-derived from config).
pub(crate) async fn render_block() -> Option<(String, ModelSnapshot)> {
    let current = ModelSnapshot::from_config();
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
/// fetch is timeout-bounded internally (see [`crate::tools::catalog_cache`]).
pub(crate) async fn render_section(kind: ModelKind, model: &str) -> Option<String> {
    match kind {
        ModelKind::Image => {
            let catalog = image_catalog::get_catalog().await?;
            catalog
                .find(model)
                .and_then(|info| render_image_section(model, info))
        }
        ModelKind::Video => {
            // Catalog fields are optional: static per-model edit nuances render
            // even when the catalog is unavailable (fail-open must not drop
            // nuance text on model switch).
            let catalog = video_catalog::get_catalog().await;
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
/// section would carry no actionable capability info.
fn render_image_section(model: &str, info: &ImageModelInfo) -> Option<String> {
    let mut out = format!("Image model: {model}\n");
    let mut params: Vec<(&str, &ParameterConstraint)> = info
        .supported_parameters
        .iter()
        .filter_map(|(name, constraint)| tool_param_name(name).map(|tool| (tool, constraint)))
        .collect();
    // A model declaring both "resolution" and "size" maps to the same tool
    // param — dedupe by tool name so the envelope lists it once.
    params.sort_by_key(|(name, _)| *name);
    params.dedup_by_key(|(name, _)| *name);
    let mut rendered = false;
    for (name, constraint) in params {
        match constraint {
            ParameterConstraint::Enum(values) => {
                rendered = true;
                let _ = writeln!(out, "- {name}: {}", join_capped(values, MAX_LIST_VALUES));
            }
            ParameterConstraint::Range { max } => {
                rendered = true;
                let _ = writeln!(out, "- {name}: max {max}");
            }
            ParameterConstraint::Boolean => {
                rendered = true;
                let _ = writeln!(out, "- {name}: supported");
            }
            ParameterConstraint::Unknown => {}
        }
    }
    rendered.then_some(out)
}

/// Render the declared video capabilities (typed catalog fields only; the
/// field names match the `video_gen` tool schema) plus the static per-model
/// video-edit nuances. `None` when the model declares no capability fields
/// and no nuance data (a header-only section would be noise).
fn render_video_section(model: &str, info: Option<&VideoModelInfo>) -> Option<String> {
    let mut out = format!("Video model: {model}\n");
    let mut rendered = false;
    if let Some(info) = info {
        if let Some(resolutions) = &info.resolutions
            && !resolutions.is_empty()
        {
            rendered = true;
            let _ = writeln!(
                out,
                "- resolution: {}",
                join_capped(resolutions, MAX_LIST_VALUES)
            );
        }
        if let Some(ratios) = &info.aspect_ratios
            && !ratios.is_empty()
        {
            rendered = true;
            let _ = writeln!(
                out,
                "- aspect_ratio: {}",
                join_capped(ratios, MAX_LIST_VALUES)
            );
        }
        if let Some(durations) = &info.durations
            && !durations.is_empty()
        {
            rendered = true;
            let _ = writeln!(out, "- duration: {} seconds", format_durations(durations));
        }
        if let Some(sizes) = &info.sizes
            && !sizes.is_empty()
        {
            rendered = true;
            let _ = writeln!(out, "- size: {}", join_capped(sizes, MAX_LIST_VALUES));
        }
        if let Some(frames) = &info.frame_images
            && !frames.is_empty()
        {
            rendered = true;
            let _ = writeln!(
                out,
                "- frame images: {}",
                join_capped(frames, MAX_LIST_VALUES)
            );
        }
        if let Some(audio) = info.generate_audio {
            rendered = true;
            let _ = writeln!(
                out,
                "- generate_audio: {}",
                if audio { "yes" } else { "no" }
            );
        }
        if let Some(seed) = info.seed {
            rendered = true;
            let _ = writeln!(out, "- seed: {}", if seed { "yes" } else { "no" });
        }
    }
    if let Some(nuance) = crate::tools::video_edit::video_edit_nuances(model) {
        rendered = true;
        let _ = writeln!(out, "- editing: {nuance}");
    }
    rendered.then_some(out)
}

/// Join values, capping the rendered list so future catalogs cannot bloat
/// every Artist turn.
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
#[allow(clippy::cast_possible_wrap)]
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
    fn snapshot_json_roundtrip() {
        let snap = ModelSnapshot {
            image: Some("google/gemini-3.1-flash-image".into()),
            video: Some("minimax/hailuo-3".into()),
        };
        let parsed = ModelSnapshot::from_json(&snap.to_json()).expect("roundtrip");
        assert_eq!(parsed, snap);
    }

    #[test]
    fn snapshot_from_json_tolerates_garbage() {
        assert!(ModelSnapshot::from_json("not json").is_none());
        assert!(ModelSnapshot::from_json("").is_none());
        assert!(ModelSnapshot::from_json("{\"image\":\"only\"}").is_some());
    }

    #[test]
    fn format_durations_contiguous_range() {
        assert_eq!(format_durations(&[5, 6, 7, 8, 9]), "5-9");
        assert_eq!(format_durations(&[5, 7, 8]), "5, 7, 8");
        assert_eq!(format_durations(&[1]), "1");
        // Duplicates must not defeat the contiguity check: 7 is missing.
        assert_eq!(format_durations(&[5, 6, 6, 8, 9, 10]), "5, 6, 8, 9, 10");
        // A contiguous set with duplicates still compacts (all values covered).
        assert_eq!(format_durations(&[5, 6, 6, 7]), "5-7");
        // Extreme values must not overflow the contiguity check (debug panic).
        assert_eq!(format_durations(&[0, i64::MAX]), "0, 9223372036854775807");
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

    #[test]
    fn image_section_renders_tool_mapped_params_only() {
        let mut info = ImageModelInfo::default();
        info.supported_parameters.insert(
            "aspect_ratio".into(),
            ParameterConstraint::Enum(vec!["1:1".into(), "16:9".into()]),
        );
        info.supported_parameters.insert(
            "input_references".into(),
            ParameterConstraint::Range { max: 4 },
        );
        info.supported_parameters.insert(
            "resolution".into(),
            ParameterConstraint::Enum(vec!["1K".into(), "2K".into()]),
        );
        // Not exposed by the image_gen tool schema — must not render.
        info.supported_parameters
            .insert("n".into(), ParameterConstraint::Range { max: 4 });
        info.supported_parameters
            .insert("seed".into(), ParameterConstraint::Boolean);
        info.supported_parameters
            .insert("future".into(), ParameterConstraint::Unknown);

        let out = render_image_section("test/model", &info).expect("envelope rendered");
        assert!(out.starts_with("Image model: test/model\n"));
        assert!(out.contains("- aspect_ratio: 1:1, 16:9\n"));
        assert!(out.contains("- images: max 4\n"));
        assert!(out.contains("- size: 1K, 2K\n"));
        assert!(!out.contains("resolution"));
        assert!(!out.contains("input_references"));
        assert!(!out.contains("n:"));
        assert!(!out.contains("seed"));
        assert!(!out.contains("future"));
    }

    #[test]
    fn image_section_maps_size_identity_and_dedupes() {
        // "size" is exposed as-is (identity mapping).
        let mut size_only = ImageModelInfo::default();
        size_only.supported_parameters.insert(
            "size".into(),
            ParameterConstraint::Enum(vec!["512x512".into()]),
        );
        let out = render_image_section("m", &size_only).expect("envelope rendered");
        assert!(out.contains("- size: 512x512\n"));

        // A model declaring both "resolution" and "size" maps both to the same
        // tool param — the envelope must list it once.
        let mut both = ImageModelInfo::default();
        both.supported_parameters.insert(
            "resolution".into(),
            ParameterConstraint::Enum(vec!["1K".into(), "2K".into()]),
        );
        both.supported_parameters.insert(
            "size".into(),
            ParameterConstraint::Enum(vec!["512x512".into()]),
        );
        let out = render_image_section("m", &both).expect("envelope rendered");
        assert_eq!(out.matches("- size: ").count(), 1);
    }

    #[test]
    fn image_section_unmapped_only_is_omitted() {
        // A model declaring only unmapped/unknown params renders no envelope —
        // the section is omitted entirely (header-only would be noise).
        let mut info = ImageModelInfo::default();
        info.supported_parameters
            .insert("n".into(), ParameterConstraint::Range { max: 4 });
        info.supported_parameters
            .insert("future".into(), ParameterConstraint::Unknown);
        assert!(render_image_section("m", &info).is_none());
        assert!(render_image_section("m", &ImageModelInfo::default()).is_none());
    }

    #[test]
    fn video_section_skips_empty_and_uncapped_lists() {
        // Declared-but-empty lists must not render dangling lines — a model
        // with no declared fields and no nuance data renders no envelope at all.
        let empty = VideoModelInfo {
            resolutions: Some(vec![]),
            durations: Some(vec![]),
            frame_images: Some(vec![]),
            ..VideoModelInfo::default()
        };
        assert!(render_video_section("no-nuance-model", Some(&empty)).is_none());

        // Long frame-image lists go through the cap like every other list.
        let many_frames = VideoModelInfo {
            frame_images: Some((0..20).map(|i| format!("frame{i}")).collect()),
            ..VideoModelInfo::default()
        };
        let out = render_video_section("m", Some(&many_frames)).expect("envelope rendered");
        assert!(out.contains("- frame images: frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10, frame11, frame12, frame13, frame14, frame15, +4 more\n"));
    }

    #[test]
    fn video_section_renders_declared_fields_only() {
        let info = VideoModelInfo {
            resolutions: Some(vec!["2K".into()]),
            aspect_ratios: None,
            durations: Some(vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            sizes: None,
            frame_images: Some(vec!["first_frame".into(), "last_frame".into()]),
            generate_audio: Some(true),
            seed: Some(false),
        };
        let out = render_video_section("minimax/hailuo-3", Some(&info)).expect("envelope rendered");
        assert!(out.starts_with("Video model: minimax/hailuo-3\n"));
        assert!(out.contains("- resolution: 2K\n"));
        assert!(!out.contains("aspect_ratio"));
        assert!(out.contains("- duration: 5-15 seconds\n"));
        assert!(!out.contains("- size:"));
        assert!(out.contains("- frame images: first_frame, last_frame\n"));
        assert!(out.contains("- generate_audio: yes\n"));
        assert!(out.contains("- seed: no\n"));
        assert!(out.contains("- editing: localized instruction edits"));
    }

    #[test]
    fn video_section_renders_edit_nuances_without_catalog() {
        // Static per-model edit nuances render even with no catalog entry —
        // fail-open must not drop nuance text when the catalog is unavailable.
        let hailuo = render_video_section("minimax/hailuo-3", None).expect("nuance rendered");
        assert!(hailuo.starts_with("Video model: minimax/hailuo-3\n"));
        assert!(hailuo.contains("- editing: localized instruction edits"));
        assert!(!hailuo.contains("- duration:"));
        let seedance =
            render_video_section("bytedance/seedance-2.5", None).expect("nuance rendered");
        assert!(seedance.contains("- editing: whole-frame restyle"));
        // Edit vs generation disambiguation: the catalog duration range is
        // generation-only, so the Artist is not misled into passing it for edits.
        assert!(seedance.contains("applies to generation, not edits"));
        assert!(render_video_section("unknown/model", None).is_none());
    }
}

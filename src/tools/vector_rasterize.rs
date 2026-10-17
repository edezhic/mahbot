//! The `vector_rasterize` tool: a vector file in, an ordinary picture out.
//!
//! A vector file is markup, and the product could only read one as text. This
//! renders it with the engine's own rasterizer ([`resvg`] — pure Rust, in
//! process, nothing to configure) into a PNG, the product's usual picture format:
//! the assistant sees it through the ordinary tool-result payload, sends it to the
//! user through the ordinary `[IMAGE:path]` marker, and the file itself is a
//! picture like any other. Nothing about the way pictures are shown or delivered
//! changes, and neither does the document pipeline this action deliberately sits
//! outside of.
//!
//! SVG is the vector format accepted today. The tool is named and described by the
//! action rather than by that format, so a further one is a further input rather
//! than a new tool.
//!
//! Three rules the implementation exists to hold:
//!
//! * **The caller's boundary.** The drawing, and every resource it refers to, are
//!   resolved through the same gate an ordinary read goes through — on both the
//!   spelling and the canonical target, at the caller's own size limit — so a
//!   conversion never reaches a file the caller could not have read itself, and a
//!   reference is never an unbounded read. A reference outside the boundary is not
//!   a refusal: it is simply not drawn.
//! * **The machine's fonts, the product's font as the answer.** The engine
//!   resolves the families a file names against the machine's own collection, and
//!   would answer a family it cannot find with a hardcoded serif (or drop the span
//!   entirely); the database it is handed answers an unresolved family — and text
//!   that names none — with the font the product ships, so text is drawn rather
//!   than lost (see [`fonts`]).
//! * **A bounded, contained render.** The picture's long side is capped, the work
//!   runs on a blocking thread with a panic that becomes an ordinary refusal, and
//!   a compressed input is unwrapped here, bounded by the limit any input obeys.

use std::borrow::Cow;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use resvg::tiny_skia;
use resvg::usvg;
use resvg::usvg::fontdb;
use serde_json::json;
use tokio::sync::Semaphore;

use super::path::PathAccess;
use crate::{Tool, Workspace};

/// The longest side of a produced picture, in pixels: a larger drawing is scaled
/// down to it with its aspect ratio kept, and one already smaller is left at its
/// own size — never enlarged. 2048 is the same order as the page rasters the
/// document readers produce, and bounds one render at 16 MB of pixels.
const MAX_LONG_SIDE_PX: u32 = 2048;

/// The first two bytes of a gzip stream — how a compressed drawing is told from
/// markup without parsing it.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// How many renders may run at once, process-wide. The engine is synchronous and
/// one render holds a whole picture's worth of pixels for as long as it runs, so
/// this bounds the work the action can put on the machine at once.
static RENDERS: Semaphore = Semaphore::const_new(2);

/// The `vector_rasterize` tool. `access` is the caller's [`PathAccess`], fixed
/// when the role's toolset is built: the drawing and the resources it refers to
/// are read under it exactly as a `read` of them would be.
pub(crate) struct VectorRasterizeTool {
    access: PathAccess,
}

impl VectorRasterizeTool {
    /// The tool's own name: what [`Tool::name`] reports, the prompt asset that
    /// describes it, and the tool a command-line family fronts.
    const NAME: &'static str = "vector_rasterize";

    #[must_use]
    pub(crate) const fn new(access: PathAccess) -> Self {
        Self { access }
    }
}

#[async_trait]
impl Tool for VectorRasterizeTool {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn media_marker(&self) -> Option<&'static str> {
        Some("[IMAGE:")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "path": {
                    "type": "string",
                    "description": "Path to the vector file to render (an SVG today) — anywhere you may read files"
                }
            }),
            &["path"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        // The picture travels as the tool's `[IMAGE:...]` marker, the shape every
        // produced picture uses: the marker delivers the file to the user, and the
        // agent loop derives the assistant's own bounded copy of it in this same
        // step (`Agent::tool_result_content`).
        Ok(self.format_media_result(&self.rasterize(ws, &args).await?))
    }
}

impl VectorRasterizeTool {
    /// Render the file one call names, and answer the picture the product wrote.
    async fn rasterize(&self, ws: &Workspace, args: &serde_json::Value) -> anyhow::Result<PathBuf> {
        let requested = super::get_str(args, "path")?;
        // The drawing is resolved through the read gate before anything else, so a
        // path the caller may not read is never even stat'ed — and the boundary
        // every resource it refers to is held to is this same one.
        let drawing =
            super::path::resolve_read_target(ws.as_path(), requested, self.access).await?;
        let unreadable = |error: std::io::Error| {
            anyhow::anyhow!("cannot render {requested}: the file could not be read ({error})")
        };
        let meta = tokio::fs::metadata(&drawing).await.map_err(unreadable)?;
        if !meta.is_file() {
            anyhow::bail!(
                "cannot render {requested}: it is not a regular file — hint: name the vector file \
                 itself, not a directory"
            );
        }
        super::check_size_within(
            &meta,
            super::MAX_FILE_SIZE_BYTES,
            "Vector file is too large to render",
        )?;
        let data = tokio::fs::read(&drawing).await.map_err(unreadable)?;
        if data.is_empty() {
            anyhow::bail!(
                "cannot render {requested}: the file is empty — hint: write or fetch the drawing \
                 before rendering it"
            );
        }

        let dir = drawing.parent().unwrap_or(&drawing).to_path_buf();
        let root = ws.as_path().to_path_buf();
        let access = self.access;
        let name = requested.to_string();
        let _permit = RENDERS
            .acquire()
            .await
            .expect("the render semaphore is never closed");
        let rendered =
            tokio::task::spawn_blocking(move || render_png(&data, &dir, &root, access, &name))
                .await;
        let picture = match rendered {
            Ok(rendered) => rendered?,
            // A contained render: whatever killed the render on this file — a
            // panic in the engine is the case its own reports name — is this
            // tool's own refusal rather than the process's end.
            Err(_died) => anyhow::bail!(
                "cannot render {requested}: the renderer failed on this file — hint: try a \
                 different file"
            ),
        };
        super::save_generated_file(ws, &picture, "rasterized", "png").await
    }
}

/// Render one drawing into a PNG picture: parse it, size it, rasterize it, encode
/// it. Synchronous, and run on a blocking thread (see
/// [`VectorRasterizeTool::rasterize`]). `dir` is the drawing's own directory, so a
/// relative reference resolves beside it, and `name` is the caller's own spelling
/// of the input, which every refusal quotes back.
fn render_png(
    data: &[u8],
    dir: &Path,
    workspace_root: &Path,
    access: PathAccess,
    name: &str,
) -> anyhow::Result<Vec<u8>> {
    let markup = markup(data, name)?;
    let fonts = fonts();
    // A referenced file is loaded by the engine's own resolver — its format
    // handling (raster magics, nested SVG) is not restated here — once
    // `resolve_reference_target` has put the caller's own gate in front of it.
    let load_reference = usvg::ImageHrefResolver::default_string_resolver();

    let render_options = usvg::Options {
        resources_dir: Some(dir.to_path_buf()),
        // The engine's own font selection, on the database `fonts` built: the
        // machine's faces, with the shipped font answering what they cannot (and
        // answering the text that names no family, through the family below).
        font_family: fonts.family.clone(),
        fontdb: Arc::clone(&fonts.db),
        image_href_resolver: usvg::ImageHrefResolver {
            resolve_data: usvg::ImageHrefResolver::default_data_resolver(),
            resolve_string: Box::new(move |href, options| {
                let candidate = super::path::normalize_path(&options.get_abs_path(Path::new(href)));
                let Some(approved) =
                    super::path::resolve_reference_target(&candidate, workspace_root, access)
                else {
                    tracing::debug!(
                        href = %href,
                        "vector rasterize: a referenced resource is not readable, skipped"
                    );
                    return None;
                };
                load_reference(&approved.to_string_lossy(), options)
            }),
        },
        ..usvg::Options::default()
    };
    let tree =
        usvg::Tree::from_data(&markup, &render_options).map_err(|error| refusal(name, &error))?;
    let size = tree.size();
    let scale = render_scale(size);
    let (width, height) = (pixels(size.width() * scale), pixels(size.height() * scale));
    let Some(mut pixmap) = tiny_skia::Pixmap::new(width, height) else {
        anyhow::bail!(
            "cannot render {name}: a picture of {width}x{height} pixels cannot be produced"
        );
    };
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png().map_err(|error| {
        anyhow::anyhow!("cannot render {name}: the picture could not be encoded ({error})")
    })
}

/// The scale factor one drawing is rendered at: its own size, or the cap when it
/// is larger — and never more than one, so a small drawing stays crisp.
#[expect(clippy::cast_precision_loss)] // the cap is 2048, exactly an f32
fn render_scale(size: usvg::Size) -> f32 {
    (MAX_LONG_SIDE_PX as f32 / size.width().max(size.height())).min(1.0)
}

/// One axis of the produced picture, in pixels: rounded, at least one, and never
/// past the cap.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // clamped into [1, MAX_LONG_SIDE_PX] first
fn pixels(value: f32) -> u32 {
    value.round().clamp(1.0, MAX_LONG_SIDE_PX as f32) as u32
}

/// The drawing's own markup: the bytes as they are, or the gzip stream they are
/// wrapped in, unwrapped.
///
/// The unwrapping is this tool's own rather than the engine's, whose `svgz`
/// feature is deliberately not enabled: the engine's path reads a whole stream
/// into memory with no bound, while the file it was handed is already bounded —
/// and stays bounded here, by that same limit.
fn markup<'a>(data: &'a [u8], name: &str) -> anyhow::Result<Cow<'a, [u8]>> {
    if !data.starts_with(&GZIP_MAGIC) {
        return Ok(Cow::Borrowed(data));
    }
    let limit = super::MAX_FILE_SIZE_BYTES;
    let mut decoded = Vec::new();
    flate2::read::GzDecoder::new(data)
        .take(limit + 1)
        .read_to_end(&mut decoded)
        .map_err(|error| {
            anyhow::anyhow!(
                "cannot render {name}: the file is not a readable gzip-compressed SVG ({error})"
            )
        })?;
    if decoded.len() as u64 > limit {
        anyhow::bail!(
            "cannot render {name}: the file expands past the {limit} byte limit — hint: hand over \
             the uncompressed drawing, or a smaller one"
        );
    }
    Ok(Cow::Owned(decoded))
}

/// Why the engine would not read a file, said the way the caller can act on it.
fn refusal(name: &str, error: &usvg::Error) -> anyhow::Error {
    let reason = match error {
        usvg::Error::NotAnUtf8Str => "it is not SVG text (its bytes are not UTF-8)",
        usvg::Error::ParsingFailed(_) => "it is not valid SVG markup",
        usvg::Error::InvalidSize => {
            "it declares a width or height that is zero or negative — give the drawing a positive \
             width and height"
        }
        usvg::Error::ElementsLimitReached => "it has more elements than the renderer accepts",
        usvg::Error::MalformedGZip => "its gzip stream is malformed",
        // What is left of a compressed input: the engine's own `svgz` path is off,
        // so this is a stream that is compressed inside a stream already unwrapped.
        usvg::Error::SvgzFeatureNotEnabled => "its compressed form could not be unwrapped",
    };
    anyhow::anyhow!(
        "cannot render {name}: {reason} — hint: SVG is the vector format this renders today"
    )
}

/// The fonts one render draws text with: this machine's own collection with the
/// font the product ships registered in it, and the family that font declares.
struct Fonts {
    /// Every face the machine has, the shipped font among them.
    db: Arc<fontdb::Database>,
    /// The shipped font's family name — what a file naming no family resolves
    /// under, and what the generic `serif` answers with.
    family: String,
}

/// The process's font set, built once: finding the machine's fonts walks its font
/// directories and maps every face, which a conversion must not repeat.
fn fonts() -> &'static Fonts {
    static FONTS: OnceLock<Fonts> = OnceLock::new();
    FONTS.get_or_init(|| {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        let bundled = db
            .load_font_source(fontdb::Source::Binary(Arc::new(
                crate::docgen::TEXT_FONT_DATA.to_vec(),
            )))
            .first()
            .copied()
            .expect("the shipped font carries a face");
        let family = db
            .face(bundled)
            .and_then(|face| face.families.first().map(|(name, _)| name.clone()))
            .unwrap_or_default();
        // What answers a family the machine cannot supply: the engine's selector
        // ends every family list with its generic `serif`, and answers with what
        // that names — the machine's own serif, where the machine has one — so
        // this environment's generic is pointed at the font the product ships.
        db.set_serif_family(family.clone());
        Fonts {
            db: Arc::new(db),
            family,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;
    use std::io::Write as _;
    use tempfile::TempDir;

    /// A 40x20 drawing: a 20x20 red square at the left, nothing at the right.
    const RED_HALF: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
            <rect width="20" height="20" fill="#ff0000"/></svg>"##;

    /// A temporary workspace directory holding `files` (name, bytes).
    fn workspace(files: &[(&str, &[u8])]) -> (TempDir, Workspace) {
        let dir = TempDir::new().unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        let ws = Workspace::from_path(dir.path());
        (dir, ws)
    }

    /// Render `name` out of `ws` and answer the picture the tool produced.
    async fn rendered(access: PathAccess, ws: &Workspace, name: &str) -> RgbaImage {
        let answer = VectorRasterizeTool::new(access)
            .execute(ws, json!({ "path": name }))
            .await
            .expect("the drawing renders");
        let produced = answer
            .strip_prefix("[IMAGE:")
            .and_then(|marker| marker.strip_suffix(']'))
            .unwrap_or_else(|| panic!("the answer carries the delivery marker: {answer}"));
        let bytes = std::fs::read(produced).expect("the produced picture is on disk");
        image::load_from_memory(&bytes).unwrap().to_rgba8()
    }

    /// A solid 20x20 red PNG, for a drawing that refers to a picture.
    fn red_png(path: &Path) {
        RgbaImage::from_pixel(20, 20, image::Rgba([255, 0, 0, 255]))
            .save(path)
            .unwrap();
    }

    #[tokio::test]
    async fn a_drawing_becomes_a_picture_of_its_own_size_with_its_transparency() {
        let (_dir, ws) = workspace(&[("drawing.svg", RED_HALF.as_bytes())]);
        let picture = rendered(PathAccess::Unrestricted, &ws, "drawing.svg").await;

        assert_eq!(picture.dimensions(), (40, 20));
        assert_eq!(
            picture.get_pixel(5, 5).0,
            [255, 0, 0, 255],
            "the drawing's fill"
        );
        assert_eq!(
            picture.get_pixel(35, 5).0[3],
            0,
            "what the drawing leaves out stays transparent"
        );
    }

    #[tokio::test]
    async fn a_large_drawing_is_capped_and_a_small_one_is_never_enlarged() {
        let drawing = |side: &str| {
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="1000">
                   <rect width="{side}" height="1000" fill="#ff0000"/></svg>"##
            )
        };
        let (_dir, ws) = workspace(&[
            ("large.svg", drawing("4000").as_bytes()),
            ("small.svg", drawing("1").as_bytes()),
        ]);

        assert_eq!(
            rendered(PathAccess::Unrestricted, &ws, "large.svg")
                .await
                .dimensions(),
            (2048, 512),
            "a drawing past the cap is scaled down, its aspect ratio kept"
        );
        assert_eq!(
            rendered(PathAccess::Unrestricted, &ws, "small.svg")
                .await
                .dimensions(),
            (1, 1000),
            "a drawing inside the cap keeps its own size"
        );
    }

    #[tokio::test]
    async fn text_without_a_usable_machine_family_is_drawn_in_the_products_own_font() {
        // The same text three times: naming a family no machine has, naming none at
        // all, and naming the family the product's own font declares. All three
        // coming out identical is what says the first two were answered by the
        // shipped font — the machine's own serif (the engine's own answer) or no
        // face at all both look different, the last one by leaving the text out,
        // which the reference is checked for ink against.
        let drawing = |attributes: &str| {
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80">
                    <rect width="200" height="80" fill="#ffffff"/>
                    <text x="10" y="60" font-size="48" {attributes}>Wollstonecraft</text></svg>"##
            )
        };
        let shipped_family = fonts().family.clone();
        let (_dir, ws) = workspace(&[
            (
                "missing.svg",
                drawing(r#"font-family="No Such Family Ff5b1""#).as_bytes(),
            ),
            ("unnamed.svg", drawing("").as_bytes()),
            (
                "shipped.svg",
                drawing(&format!(r#"font-family="{shipped_family}""#)).as_bytes(),
            ),
        ]);
        let shipped = rendered(PathAccess::Unrestricted, &ws, "shipped.svg").await;

        assert!(
            shipped
                .pixels()
                .any(|pixel| pixel.0 != [255, 255, 255, 255]),
            "the shipped font draws this text at all"
        );
        for drawing in ["missing.svg", "unnamed.svg"] {
            assert_eq!(
                rendered(PathAccess::Unrestricted, &ws, drawing)
                    .await
                    .as_raw(),
                shipped.as_raw(),
                "{drawing} is drawn in the product's own font"
            );
        }
    }

    #[tokio::test]
    async fn a_resource_the_caller_could_not_read_is_not_drawn() {
        // Two pictures the drawing refers to the same way: one beside it, one
        // outside the workspace entirely (the reference rules themselves are
        // `path::resolve_reference_target`'s own test).
        let outside = TempDir::new().unwrap();
        let outside_path = outside.path().join("outside.png");
        red_png(&outside_path);
        let drawing = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
               <image href="sibling.png" width="20" height="20"/>
               <image href="{}" x="20" width="20" height="20"/></svg>"#,
            outside_path.display()
        );
        let (dir, ws) = workspace(&[("drawing.svg", drawing.as_bytes())]);
        red_png(&dir.path().join("sibling.png"));

        let picture = rendered(PathAccess::Workspace, &ws, "drawing.svg").await;

        assert_eq!(
            picture.get_pixel(5, 5).0,
            [255, 0, 0, 255],
            "a sibling of the drawing resolves beside it"
        );
        assert_eq!(
            picture.get_pixel(25, 5).0[3],
            0,
            "a reference outside the caller's boundary is not read"
        );
    }

    #[tokio::test]
    async fn a_compressed_drawing_renders_like_its_plain_form() {
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(RED_HALF.as_bytes()).unwrap();
        let compressed = gz.finish().unwrap();
        let (_dir, ws) = workspace(&[
            ("drawing.svg", RED_HALF.as_bytes()),
            ("drawing.svgz", compressed.as_slice()),
        ]);

        let plain = rendered(PathAccess::Unrestricted, &ws, "drawing.svg").await;
        let zipped = rendered(PathAccess::Unrestricted, &ws, "drawing.svgz").await;
        assert!(
            plain == zipped,
            "a compressed drawing renders as its plain form does"
        );
    }

    #[tokio::test]
    async fn an_unusable_file_is_refused_with_a_reason_the_caller_can_act_on() {
        let oversized = vec![b'a'; 10 * 1024 * 1024 + 1];
        let (dir, ws) = workspace(&[
            ("empty.svg", b""),
            ("notes.txt", b"this is not markup at all"),
            ("huge.svg", oversized.as_slice()),
        ]);
        red_png(&dir.path().join("picture.png"));

        let refused = |error: anyhow::Error| format!("{error:#}");
        for (name, reason) in [
            ("empty.svg", "is empty"),
            ("notes.txt", "not valid SVG markup"),
            ("picture.png", "not SVG text"),
            ("huge.svg", "too large"),
            ("absent.svg", "File not found"),
        ] {
            let error = VectorRasterizeTool::new(PathAccess::Unrestricted)
                .execute(&ws, json!({ "path": name }))
                .await
                .expect_err("the file is unusable");
            let text = refused(error);
            assert!(text.contains(reason), "{name}: {text}");
        }
    }
}

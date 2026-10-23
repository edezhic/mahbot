//! The `rasterize` tool: a file or a page in, an ordinary picture out.
//!
//! An SVG is markup, a PDF is a container, and an HTML file or a web address is
//! a page: the product could only read them as text or as extracted pages. This
//! renders them into a PNG, the product's usual picture format: the assistant
//! sees it through the ordinary tool-result payload, sends it to the user
//! through the ordinary `[IMAGE:path]` marker, and the file itself is a picture
//! like any other. Nothing about the way pictures are shown or delivered
//! changes, and neither does the document pipeline this action deliberately sits
//! outside of.
//!
//! SVG, one page of a PDF, a local HTML file and an http(s) address are the
//! inputs accepted today. The tool is named and described by the action rather
//! than by those formats, so a further one is a further input rather than a new
//! tool.
//!
//! Four rules the implementation exists to hold:
//!
//! * **The caller's boundary.** The drawing, and every resource it refers to, are
//!   resolved through the same gate an ordinary read goes through — on both the
//!   spelling and the canonical target, at the caller's own size limit — so a
//!   conversion never reaches a file the caller could not have read itself, and a
//!   reference is never an unbounded read. A reference outside the boundary is not
//!   a refusal: it is simply not drawn. An HTML page is held to the same gate,
//!   inside the caller's personal workspace (see [`super::rasterize_html`]).
//! * **The machine's fonts, the product's font as the answer.** The engine
//!   resolves the families a file names against the machine's own collection, and
//!   would answer a family it cannot find with a hardcoded serif (or drop the span
//!   entirely); the database it is handed answers an unresolved family — and text
//!   that names none — with the font the product ships, so text is drawn rather
//!   than lost (see [`fonts`]).
//! * **A bounded, contained render.** The picture's long side is capped; an
//!   in-process render runs on a blocking thread with a panic that becomes an
//!   ordinary refusal, a compressed input is unwrapped here (bounded by the limit
//!   any input obeys), and a browser capture is bounded by one budget.
//! * **One page of a PDF.** A PDF of several pages is rendered one named page at a
//!   time: the page's visible area (a crop honoured), a refusal that says how many
//!   pages there are when `page` is missing or past the end, and nothing
//!   interactive — form values, comments, links — in the picture.
//!
//! A browser capture never shares the in-process renders' permit pool, so it
//! cannot delay one; the capture's own rules live with it.

use std::borrow::Cow;
use std::io::Cursor;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use hayro::hayro_syntax::{DecryptionError, LoadPdfError, Pdf};
use image::ImageFormat;
use resvg::tiny_skia;
use resvg::usvg;
use resvg::usvg::fontdb;
use serde_json::json;
use tokio::sync::Semaphore;

use super::path::PathAccess;
use crate::{Tool, Workspace};

/// The longest side of a produced picture, in pixels: a drawing or page larger
/// than it is scaled down to it with its aspect ratio kept, and one already
/// smaller is left at its own size; an HTML page is captured at the fixed
/// viewport and then fitted down to it. 2048 is the same order as the page
/// rasters the document readers produce, and bounds one render at 16 MB of
/// pixels.
pub(super) const MAX_LONG_SIDE_PX: u32 = 2048;

/// The first two bytes of a gzip stream — how a compressed drawing is told from
/// markup without parsing it.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// The first five bytes of a PDF — how a PDF is told from SVG markup without
/// parsing it.
const PDF_MAGIC: &[u8; 5] = b"%PDF-";

/// How many in-process renders may run at once, process-wide. The engine is
/// synchronous and one render holds a whole picture's worth of pixels for as
/// long as it runs, so this bounds the work the action can put on the machine
/// at once. HTML renders take [`HTML_RENDERS`] instead: a browser capture must
/// never delay an in-process one.
static RENDERS: Semaphore = Semaphore::const_new(2);

/// How many browser captures may run at once, process-wide — one browser session
/// each, so this bounds the tabs the action can open on the owner's Chrome. A
/// separate permit pool from [`RENDERS`], so an HTML render never delays an
/// in-process one.
static HTML_RENDERS: Semaphore = Semaphore::const_new(2);

/// Which renderer an input is handed to, from its bytes and its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderKind {
    /// One page of a PDF, rendered in process.
    Pdf,
    /// SVG markup, plain or gzip-compressed, rendered in process.
    Svg,
    /// A page captured through the product's connected browser.
    Html,
}

/// The `rasterize` tool. `access` is the caller's [`PathAccess`], fixed when the
/// role's toolset is built: the input, the resources a drawing refers to and
/// every file the page server hands out are read under it exactly as a `read` of
/// them would be.
pub(crate) struct RasterizeTool {
    access: PathAccess,
}

impl RasterizeTool {
    /// The tool's own name: what [`Tool::name`] reports, the prompt asset that
    /// describes it, and the tool a command-line family fronts.
    const NAME: &'static str = "rasterize";

    #[must_use]
    pub(crate) const fn new(access: PathAccess) -> Self {
        Self { access }
    }
}

#[async_trait]
impl Tool for RasterizeTool {
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
                    "description": "What to render: an SVG, one page of a PDF, an HTML file inside your personal workspace, or an http(s) address"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The 1-based page to render, for a PDF of several pages; naming a page for any other input is refused."
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

impl RasterizeTool {
    /// Render the file or address one call names, and answer the picture the
    /// product wrote.
    async fn rasterize(&self, ws: &Workspace, args: &serde_json::Value) -> anyhow::Result<PathBuf> {
        let requested = super::get_str(args, "path")?;
        let page = super::get_opt_u64(args, "page")?;
        // An address is never resolved as a file: the page it names is fetched by
        // the product's own browser, under the URL policy a navigation obeys. Any
        // other address scheme is refused as one — `file://` included — rather
        // than looked up as a file path it cannot be.
        if let Some(url) = as_render_url(requested) {
            // A page names one page of a PDF, and a page has no pages to choose from.
            if page.is_some() {
                return Err(page_is_for_pdf(requested));
            }
            crate::util::http::validate_url(url)
                .map_err(|error| anyhow::anyhow!("cannot render {requested}: {error:#}"))?;
            let workspace = ws.as_path().to_string_lossy();
            let _permit = capture_permit().await;
            let picture = super::rasterize_html::render_url(url, requested, &workspace).await?;
            return super::save_generated_file(ws, &picture, "rasterized", "png").await;
        }
        if let Some(scheme) = non_http_address(requested) {
            anyhow::bail!(
                "cannot render {requested}: only an http(s) address is rendered as a page, and \
                 this one uses {scheme}"
            );
        }
        // The input is resolved through the read gate before anything else, so a
        // path the caller may not read is never even stat'ed — and the boundary
        // every resource a drawing refers to is held to is this same one.
        let drawing =
            super::path::resolve_read_target(ws.as_path(), requested, self.access).await?;
        let unreadable = |error: std::io::Error| {
            anyhow::anyhow!("cannot render {requested}: the file could not be read ({error})")
        };
        let meta = tokio::fs::metadata(&drawing).await.map_err(unreadable)?;
        if !meta.is_file() {
            anyhow::bail!(
                "cannot render {requested}: it is not a regular file — hint: name the file itself, \
                 not a directory"
            );
        }
        super::check_size_within(
            &meta,
            super::MAX_FILE_SIZE_BYTES,
            "File is too large to render",
        )?;
        let data = tokio::fs::read(&drawing).await.map_err(unreadable)?;
        if data.is_empty() {
            anyhow::bail!(
                "cannot render {requested}: the file is empty — hint: write or fetch the file \
                 before rendering it"
            );
        }

        let kind = classify(&data, &drawing);
        // A page names one page of a PDF; any other file is a single picture with
        // no pages to choose from, so naming one there is refused rather than
        // silently ignored.
        if page.is_some() && kind != RenderKind::Pdf {
            return Err(page_is_for_pdf(requested));
        }
        let name = requested.to_string();
        let picture = match kind {
            RenderKind::Pdf => {
                let render_name = name.clone();
                in_process_render(&name, move || {
                    render_pdf_page_png(&data, &render_name, page)
                })
                .await?
            }
            RenderKind::Svg => {
                let dir = drawing.parent().unwrap_or(&drawing).to_path_buf();
                let root = ws.as_path().to_path_buf();
                let access = self.access;
                let render_name = name.clone();
                in_process_render(&name, move || {
                    render_png(&data, &dir, &root, access, &render_name)
                })
                .await?
            }
            RenderKind::Html => self.html_picture(ws, &drawing, requested).await?,
        };
        super::save_generated_file(ws, &picture, "rasterized", "png").await
    }

    /// Render a local HTML page out of the caller's personal workspace: the page
    /// server only ever hands out files from under that root, so a file outside
    /// it is refused before anything is started.
    async fn html_picture(
        &self,
        ws: &Workspace,
        drawing: &Path,
        requested: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let Some(root) = personal_root() else {
            anyhow::bail!(
                "cannot render {requested}: a page is rendered from your personal workspace, and \
                 this call has none — hint: run the render from a user session"
            );
        };
        let root = tokio::fs::canonicalize(&root)
            .await
            .map(|root| crate::util::strip_verbatim_prefix(&root))
            .map_err(|error| {
                anyhow::anyhow!(
                    "cannot render {requested}: your personal workspace could not be read ({error})"
                )
            })?;
        if !crate::util::is_within(drawing, &root) {
            anyhow::bail!(
                "cannot render {requested}: a page is rendered from your personal workspace, and \
                 this file is outside it — hint: put the file in your personal workspace and name \
                 it there"
            );
        }
        let workspace = ws.as_path().to_string_lossy();
        let _permit = capture_permit().await;
        super::rasterize_html::render_file(&root, drawing, self.access, requested, &workspace).await
    }
}

/// One browser-capture permit, process-wide: a capture never takes one of the
/// in-process renders' permits and so never delays one.
async fn capture_permit() -> tokio::sync::SemaphorePermit<'static> {
    HTML_RENDERS
        .acquire()
        .await
        .expect("the browser-capture semaphore is never closed")
}

/// The one refusal for naming a page where there are none to choose from: only a
/// PDF has pages, and neither an address nor a single picture has any.
fn page_is_for_pdf(name: &str) -> anyhow::Error {
    anyhow::anyhow!("cannot render {name}: naming a page is for a PDF — only a PDF has pages")
}

/// The address a call names, when what it names is one: a trimmed `http`/`https`
/// URL with a host. Anything else is a local path, resolved as a file.
fn as_render_url(requested: &str) -> Option<&str> {
    let trimmed = requested.trim();
    let url = url::Url::parse(trimmed).ok()?;
    (matches!(url.scheme(), "http" | "https") && url.host().is_some()).then_some(trimmed)
}

/// The scheme of an address that is not http(s) — a `file://` a caller may try,
/// or any other scheme that carries a host (a single-letter drive prefix does
/// not, so a Windows path stays a path).
fn non_http_address(requested: &str) -> Option<String> {
    let trimmed = requested.trim();
    if trimmed.to_ascii_lowercase().starts_with("file://") {
        return Some("file".to_string());
    }
    let url = url::Url::parse(trimmed).ok()?;
    (url.host().is_some() && !matches!(url.scheme(), "http" | "https"))
        .then(|| url.scheme().to_string())
}

/// The root an HTML input must live under: the caller's personal workspace,
/// built from the caller's name only when it is a name the product itself builds
/// a workspace path from. No valid user identity — none bound, or a stored name
/// that is not a user name — means no personal workspace, and the hard boundary
/// refuses rather than widening to whatever workspace the call happens to run in.
fn personal_root() -> Option<PathBuf> {
    let user = crate::agent::tool_user_name();
    crate::users::is_valid_personal_user_name(&user)
        .then(|| crate::users::personal_workspace_path(&user))
}

/// Which renderer an input's bytes and name select: a PDF by its magic, a
/// compressed drawing by its gzip magic, then HTML by its markup or its name —
/// and SVG for everything else.
fn classify(data: &[u8], path: &Path) -> RenderKind {
    if data.starts_with(PDF_MAGIC) {
        RenderKind::Pdf
    } else if data.starts_with(&GZIP_MAGIC) {
        RenderKind::Svg
    } else if is_html_document(data, path) {
        RenderKind::Html
    } else {
        RenderKind::Svg
    }
}

/// Whether an input is an HTML page: what it opens with decides, and a file whose
/// name says `.html`/`.htm` is a page whatever its markup is.
fn is_html_document(data: &[u8], path: &Path) -> bool {
    let markup = data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data);
    if opens_as_html(markup) {
        return true;
    }
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("html" | "htm")
    )
}

/// Whether the markup opens as an HTML document, past the comments, processing
/// instructions and declarations a page may put before its doctype or root
/// element: an HTML doctype, or the first element being `<html`.
fn opens_as_html(markup: &[u8]) -> bool {
    let mut rest = markup;
    loop {
        rest = rest.trim_ascii_start();
        if let Some(after) = rest.strip_prefix(b"<!--") {
            let Some(end) = after.windows(3).position(|window| window == b"-->") else {
                return false;
            };
            rest = &after[end + 3..];
        } else if let Some(after) = rest.strip_prefix(b"<?") {
            let Some(end) = after.windows(2).position(|window| window == b"?>") else {
                return false;
            };
            rest = &after[end + 2..];
        } else if let Some(declaration) = rest.strip_prefix(b"<!") {
            let Some(end) = declaration.iter().position(|byte| *byte == b'>') else {
                return false;
            };
            let mut words = declaration[..end]
                .split(u8::is_ascii_whitespace)
                .filter(|word| !word.is_empty());
            if words
                .next()
                .is_some_and(|word| word.eq_ignore_ascii_case(b"doctype"))
                && words
                    .next()
                    .is_some_and(|word| word.eq_ignore_ascii_case(b"html"))
            {
                return true;
            }
            rest = &declaration[end + 1..];
        } else {
            break;
        }
    }
    rest.get(..5)
        .is_some_and(|tag| tag.eq_ignore_ascii_case(b"<html"))
        && rest
            .get(5)
            .is_some_and(|next| next.is_ascii_whitespace() || *next == b'>' || *next == b'/')
}

/// Run one in-process render under the shared permit, on a blocking thread,
/// turning a panic in the engine into this tool's own refusal rather than the
/// process's end.
async fn in_process_render(
    name: &str,
    render: impl FnOnce() -> anyhow::Result<Vec<u8>> + Send + 'static,
) -> anyhow::Result<Vec<u8>> {
    let _permit = RENDERS
        .acquire()
        .await
        .expect("the render semaphore is never closed");
    match tokio::task::spawn_blocking(render).await {
        Ok(rendered) => rendered,
        Err(_died) => anyhow::bail!(
            "cannot render {name}: the renderer failed on this file — hint: try a different file"
        ),
    }
}

/// Render one page of a PDF into a PNG picture: parse it, choose the page, size
/// the render, rasterize it, encode it. Synchronous, and run on a blocking
/// thread (see [`RasterizeTool::rasterize`]). `name` is the caller's own spelling
/// of the input, which every refusal quotes back, and `page_arg` is the page the
/// call named, if any.
#[expect(clippy::cast_precision_loss)] // the cap is 2048, exactly an f32
fn render_pdf_page_png(data: &[u8], name: &str, page_arg: Option<u64>) -> anyhow::Result<Vec<u8>> {
    // The structure parse also settles encryption: hayro decrypts with the empty
    // user password, so an owner-password-only document opens, while one that
    // needs a password is refused. A panic in the parse is this tool's own
    // unreadable-input refusal rather than the process's end.
    let pdf = match std::panic::catch_unwind(|| Pdf::new(data.to_vec())) {
        Ok(Ok(pdf)) => pdf,
        Ok(Err(LoadPdfError::Decryption(DecryptionError::PasswordProtected))) => anyhow::bail!(
            "cannot render {name}: it is password-protected — the tool cannot open an encrypted PDF"
        ),
        Ok(Err(_)) | Err(_) => {
            anyhow::bail!("cannot render {name}: its bytes do not form a PDF this tool can read")
        }
    };
    let count = pdf.pages().len();
    // No page is named or chosen in a document that has none.
    if count == 0 {
        anyhow::bail!("cannot render {name}: this PDF has no pages");
    }
    let page_number = match page_arg {
        Some(page) => {
            // A page argument the platform cannot index is past any document's
            // end, and is refused the same way.
            let page = usize::try_from(page).unwrap_or(usize::MAX);
            if page == 0 || page > count {
                anyhow::bail!(
                    "cannot render {name}: no page {page}: this document has {count} page(s)"
                );
            }
            page
        }
        None => match count {
            1 => 1,
            count => anyhow::bail!(
                "cannot render {name}: this PDF has {count} pages — name the page to render with \
                 \"page\""
            ),
        },
    };
    // The number was checked against the length of this same page tree, so the
    // lookup cannot miss.
    let page = pdf
        .pages()
        .get(page_number - 1)
        .expect("the page number was checked against this page tree");
    // A page is never enlarged past its own size — the cap only brings a large
    // one down — and its pixels come from the one page-to-pixels render the
    // document readings share; the cap and the PNG encoding are this tool's own.
    let Some(image) = crate::document::render_page_image(page, MAX_LONG_SIDE_PX as f32, 1.0) else {
        anyhow::bail!(
            "cannot render {name}: the page produced no pixels — hint: try a different page or \
             file"
        );
    };
    let mut png = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|error| {
            anyhow::anyhow!("cannot render {name}: the picture could not be encoded ({error})")
        })?;
    Ok(png)
}

/// Render one drawing into a PNG picture: parse it, size it, rasterize it, encode
/// it. Synchronous, and run on a blocking thread (see
/// [`RasterizeTool::rasterize`]). `dir` is the drawing's own directory, so a
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
                        "rasterize: a referenced resource is not readable, skipped"
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
        "cannot render {name}: {reason} — hint: this tool reads SVG markup, a PDF, and an HTML \
         page"
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

    /// The picture one successful call produced: the path its delivery marker
    /// names, and its bytes.
    async fn produced(
        access: PathAccess,
        ws: &Workspace,
        args: serde_json::Value,
    ) -> (String, Vec<u8>) {
        let answer = RasterizeTool::new(access)
            .execute(ws, args)
            .await
            .expect("the input renders");
        let produced = answer
            .strip_prefix("[IMAGE:")
            .and_then(|marker| marker.strip_suffix(']'))
            .unwrap_or_else(|| panic!("the answer carries the delivery marker: {answer}"));
        (
            produced.to_string(),
            std::fs::read(produced).expect("the produced picture is on disk"),
        )
    }

    /// The bytes of the picture one call produces.
    async fn rendered_png(access: PathAccess, ws: &Workspace, args: serde_json::Value) -> Vec<u8> {
        produced(access, ws, args).await.1
    }

    /// Render `name` out of `ws` and answer the picture the tool produced.
    async fn rendered(access: PathAccess, ws: &Workspace, name: &str) -> RgbaImage {
        let bytes = rendered_png(access, ws, json!({ "path": name })).await;
        image::load_from_memory(&bytes).unwrap().to_rgba8()
    }

    /// A PDF of `pages` pages, each a red rectangle of its `(width, height)`.
    fn pdf_with_pages(pages: &[(u32, u32)]) -> Vec<u8> {
        let entries: Vec<String> = pages
            .iter()
            .map(|(width, height)| format!("/MediaBox [0 0 {width} {height}]"))
            .collect();
        let contents: Vec<String> = pages
            .iter()
            .map(|(width, height)| format!("1 0 0 rg 0 0 {width} {height} re f"))
            .collect();
        let fixtures: Vec<(&str, &[u8])> = entries
            .iter()
            .zip(&contents)
            .map(|(entries, content)| (entries.as_str(), content.as_bytes()))
            .collect();
        crate::document::test_fixtures::multi_page_pdf(&fixtures)
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
            let error = RasterizeTool::new(PathAccess::Unrestricted)
                .execute(&ws, json!({ "path": name }))
                .await
                .expect_err("the file is unusable");
            let text = refused(error);
            assert!(text.contains(reason), "{name}: {text}");
        }
    }

    #[tokio::test]
    async fn a_pdf_page_becomes_a_picture_of_the_page() {
        // One small landscape page and one tall page past the cap: the first
        // keeps its own size and aspect ratio, the second is scaled down with its
        // long side capped and its portrait orientation kept.
        let small = pdf_with_pages(&[(200, 100)]);
        let large = pdf_with_pages(&[(100, 4000)]);
        let (_dir, ws) = workspace(&[("small.pdf", &small), ("large.pdf", &large)]);

        let bytes = rendered_png(
            PathAccess::Unrestricted,
            &ws,
            json!({ "path": "small.pdf" }),
        )
        .await;
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
            "a PDF page becomes an ordinary PNG"
        );
        let picture = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(
            picture.dimensions(),
            (200, 100),
            "a page inside the cap keeps its own size and landscape orientation"
        );

        let tall = rendered(PathAccess::Unrestricted, &ws, "large.pdf").await;
        assert_eq!(
            tall.height(),
            MAX_LONG_SIDE_PX,
            "a page past the cap is scaled down to its long side"
        );
        assert!(
            tall.width() < tall.height(),
            "the page's portrait orientation is kept"
        );
    }

    /// The page picture shows the page's visible area: a CropBox narrower than
    /// the page is what the picture measures, not the whole MediaBox.
    #[tokio::test]
    async fn a_pdf_pages_picture_honours_its_crop_box() {
        let cropped = crate::document::test_fixtures::multi_page_pdf(&[(
            "/MediaBox [0 0 400 200] /CropBox [100 50 300 150]",
            b"1 0 0 rg 0 0 400 200 re f",
        )]);
        let (_dir, ws) = workspace(&[("cropped.pdf", cropped.as_slice())]);
        let picture = rendered(PathAccess::Unrestricted, &ws, "cropped.pdf").await;
        assert_eq!(
            picture.dimensions(),
            (200, 100),
            "the picture measures the visible area, not the media box"
        );
    }

    #[tokio::test]
    async fn a_pdf_page_that_cannot_be_chosen_is_refused_with_the_count() {
        let two_pages = pdf_with_pages(&[(200, 100), (300, 150)]);
        let no_pages = pdf_with_pages(&[]);
        let (_dir, ws) = workspace(&[
            ("two.pdf", two_pages.as_slice()),
            ("none.pdf", no_pages.as_slice()),
            ("drawing.svg", RED_HALF.as_bytes()),
        ]);

        let refused = |error: anyhow::Error| format!("{error:#}");
        for (args, reason) in [
            (
                json!({ "path": "two.pdf" }),
                "this PDF has 2 pages — name the page to render",
            ),
            (
                json!({ "path": "two.pdf", "page": 5 }),
                "no page 5: this document has 2 page(s)",
            ),
            (
                json!({ "path": "none.pdf", "page": 1 }),
                "this PDF has no pages",
            ),
            (
                json!({ "path": "drawing.svg", "page": 1 }),
                "naming a page is for a PDF",
            ),
        ] {
            let error = RasterizeTool::new(PathAccess::Unrestricted)
                .execute(&ws, args)
                .await
                .expect_err("the page cannot be chosen");
            let text = refused(error);
            assert!(text.contains(reason), "{text}");
        }
    }

    /// A tiny valid PNG, which the capture stand-in answers with.
    fn tiny_png() -> Vec<u8> {
        let mut bytes = Vec::new();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255]))
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        bytes
    }

    /// A user's personal workspace — the live `<storage>/userspaces/<user>` root,
    /// the shared test root here — with `files` written into it.
    fn personal_workspace(user: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let personal = crate::users::personal_workspace_path(user);
        std::fs::create_dir_all(&personal).unwrap();
        for (name, bytes) in files {
            std::fs::write(personal.join(name), bytes).unwrap();
        }
        personal
    }

    #[test]
    fn an_input_is_classified_by_its_bytes_then_its_name() {
        let path = Path::new;
        assert_eq!(
            classify(b"<!DOCTYPE html><html></html>", path("notes.txt")),
            RenderKind::Html,
            "a doctype is HTML whatever the name says"
        );
        assert_eq!(
            classify(b"\xEF\xBB\xBF \n<!doctype HTML>", path("notes.txt")),
            RenderKind::Html,
            "a BOM and leading whitespace do not hide the doctype"
        );
        assert_eq!(
            classify(b"<!DOCTYPE  html><p>x</p>", path("notes.txt")),
            RenderKind::Html,
            "the doctype is compared by word, not by exact spacing"
        );
        assert_eq!(
            classify(
                b"<!-- saved from url -->\n<html lang=\"en\">",
                path("notes.txt")
            ),
            RenderKind::Html,
            "a leading comment does not hide the root element"
        );
        assert_eq!(
            classify(b"hello <html lang=\"en\">", path("notes.txt")),
            RenderKind::Svg,
            "HTML is what the markup opens with, not a tag mentioned later"
        );
        assert_eq!(
            classify(RED_HALF.as_bytes(), path("page.htm")),
            RenderKind::Html,
            "an HTML name classifies unusual markup"
        );
        assert_eq!(
            classify(b"<p>no doctype</p>", path("page.html")),
            RenderKind::Html
        );

        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(RED_HALF.as_bytes()).unwrap();
        assert_eq!(
            classify(&gz.finish().unwrap(), path("page.html")),
            RenderKind::Svg,
            "gzip keeps an SVG an SVG"
        );
        assert_eq!(
            classify(b"%PDF-1.7", path("page.html")),
            RenderKind::Pdf,
            "a PDF marker wins over the name"
        );
        assert_eq!(
            classify(RED_HALF.as_bytes(), path("drawing.svg")),
            RenderKind::Svg
        );
    }

    #[tokio::test]
    #[serial_test::serial(html_capture)]
    async fn a_page_outside_the_personal_workspace_is_refused() {
        let user = "rasterize-outside-probe";
        let personal = personal_workspace(user, &[]);
        assert!(personal.is_dir(), "the live personal root is in place");
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("page.html"), b"<!DOCTYPE html>").unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let _stand_in = crate::tools::rasterize_html::stand_in_for_capture(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(tiny_png())
        });
        let (_dir, ws) = workspace(&[]);

        let error = crate::agent::CURRENT_TOOL_USER_NAME
            .scope(user.to_string(), async {
                RasterizeTool::new(PathAccess::Unrestricted)
                    .execute(
                        &ws,
                        json!({ "path": outside.path().join("page.html").to_string_lossy() }),
                    )
                    .await
            })
            .await
            .expect_err("a page outside the personal workspace is refused");
        let text = format!("{error:#}");
        assert!(
            text.contains("a page is rendered from your personal workspace")
                && text.contains("put the file in your personal workspace"),
            "{text}"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "nothing was captured"
        );
    }

    #[tokio::test]
    async fn a_render_with_no_user_has_no_personal_workspace() {
        let (_dir, ws) = workspace(&[("page.html", b"<!DOCTYPE html>")]);

        let error = RasterizeTool::new(PathAccess::Unrestricted)
            .execute(&ws, json!({ "path": "page.html" }))
            .await
            .expect_err("no user means no personal workspace to render from");
        assert!(
            format!("{error:#}").contains("a page is rendered from your personal workspace"),
            "{error:#}"
        );
    }

    #[tokio::test]
    #[serial_test::serial(html_capture)]
    async fn a_page_inside_the_personal_workspace_renders_through_the_browser() {
        let user = "rasterize-inside-probe";
        let personal = personal_workspace(user, &[("page.html", b"<!DOCTYPE html><h1>hi</h1>")]);
        let png = tiny_png();
        let expected = png.clone();
        let _stand_in = crate::tools::rasterize_html::stand_in_for_capture(move |url| {
            let rest = url
                .strip_prefix("http://127.0.0.1:")
                .unwrap_or_else(|| panic!("the page URL names the loopback server: {url}"));
            let (port, rest) = rest
                .split_once('/')
                .unwrap_or_else(|| panic!("port then token: {url}"));
            assert!(port.parse::<u16>().is_ok(), "{url}");
            let (token, name) = rest
                .split_once('/')
                .unwrap_or_else(|| panic!("token then name: {url}"));
            assert_eq!(token.len(), 64, "the token is random hex: {url}");
            assert!(token.chars().all(|c| c.is_ascii_hexdigit()), "{url}");
            assert_eq!(name, "page.html", "{url}");
            Ok(expected.clone())
        });
        let (_dir, ws) = workspace(&[]);
        let path = personal.join("page.html");

        let (produced, bytes) = crate::agent::CURRENT_TOOL_USER_NAME
            .scope(user.to_string(), async {
                produced(
                    PathAccess::Unrestricted,
                    &ws,
                    json!({ "path": path.to_string_lossy() }),
                )
                .await
            })
            .await;
        assert!(
            produced.replace('\\', "/").contains("/generated/"),
            "{produced}"
        );
        assert_eq!(bytes, png, "the capture's bytes are the picture written");
    }

    #[tokio::test]
    #[serial_test::serial(html_capture)]
    async fn naming_a_page_for_a_page_or_an_address_is_refused() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let _stand_in = crate::tools::rasterize_html::stand_in_for_capture(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(tiny_png())
        });
        let (_dir, ws) = workspace(&[("page.html", b"<!DOCTYPE html><p>hi</p>")]);

        for args in [
            json!({ "path": "page.html", "page": 1 }),
            json!({ "path": "https://example.com/report", "page": 1 }),
        ] {
            let error = RasterizeTool::new(PathAccess::Unrestricted)
                .execute(&ws, args)
                .await
                .expect_err("a page is for a PDF");
            assert!(
                format!("{error:#}").contains("naming a page is for a PDF"),
                "{error:#}"
            );
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "nothing was captured"
        );
    }

    #[tokio::test]
    #[serial_test::serial(html_capture)]
    async fn an_address_renders_the_url_it_names() {
        let png = tiny_png();
        let expected = png.clone();
        let _stand_in = crate::tools::rasterize_html::stand_in_for_capture(move |url| {
            assert_eq!(url, "https://example.com/report");
            Ok(expected.clone())
        });
        let (_dir, ws) = workspace(&[]);

        let (_, bytes) = produced(
            PathAccess::Unrestricted,
            &ws,
            json!({ "path": " https://example.com/report " }),
        )
        .await;
        assert_eq!(bytes, png);
    }

    #[tokio::test]
    #[serial_test::serial(html_capture)]
    async fn a_non_http_address_is_never_captured() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let _stand_in = crate::tools::rasterize_html::stand_in_for_capture(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(tiny_png())
        });
        let (_dir, ws) = workspace(&[]);

        for path in ["file:///etc/passwd", "ftp://example.com/file"] {
            let error = RasterizeTool::new(PathAccess::Unrestricted)
                .execute(&ws, json!({ "path": path }))
                .await
                .expect_err("only http(s) addresses are pages");
            assert!(
                format!("{error:#}").contains("only an http(s) address"),
                "{error:#}"
            );
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "nothing was captured"
        );
    }
}

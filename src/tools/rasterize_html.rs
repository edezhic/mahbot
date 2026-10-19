//! Capturing an HTML page as a picture, through the product's connected browser.
//!
//! A `file://` URL is outside the product's URL policy, and Chrome needs a manual
//! per-extension toggle to open a local file, so a local page is served over a
//! temporary read-only loopback server that lives only while the capture runs.
//! The server hands out one file per request through the same read gate an
//! ordinary read goes through, and its addresses are unguessable to other local
//! processes: a long random path token, and the cookie it sets on the page so a
//! root-relative reference carries the token too. A wrong `Host` is refused.
//!
//! The capture runs in a chrome-use session of its own, through the interactive
//! tool's own dispatch ([`ChromeTool::dispatch_call`]), so it never touches an
//! agent's tab. The page is captured at a fixed 1280 px viewport at ratio 1,
//! clipped to the page's own box, and the browser fits that to the product's
//! long-side cap: one image of the whole page, at a size this process can decode
//! whole. Every browser step is bounded by one budget; the session is stopped on
//! every path that can run its cleanup, and the server's port is released even
//! when the capture is cut off.

use std::io::{ErrorKind, Read as _, Write as _};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;

use crate::chrome::contract::ChromeResponse;
use crate::chrome::spawn::{CliRun, CliSpawn, CliTimeout, spawn_cli};
use crate::chrome::{CliRecovery, SESSION_STOP_TIMEOUT};
use crate::tools::chrome::ChromeTool;
use crate::tools::path::PathAccess;
#[cfg(test)]
use crate::util::UnwrapPoison;

/// The viewport a page is captured at, with device pixel ratio 1: a fixed page
/// size is what makes the same page render the same on every machine.
const VIEWPORT_WIDTH: u32 = 1280;
const VIEWPORT_HEIGHT: u32 = 900;
const DEVICE_SCALE: u32 = 1;

/// One overall bound over every browser step of the capture. The readiness gate
/// and the teardown (the session stop) sit outside it: each has the product's own
/// bound, and a stop must still run when the steps were cut off. A page that has
/// not settled by then is refused as such, rather than reported as opened.
const CAPTURE_BUDGET: Duration = Duration::from_secs(150);

/// How long one best-effort settle wait (network idle, webfonts) may take before
/// the picture is taken anyway.
const SETTLE_BOUND: Duration = Duration::from_secs(6);

/// How long the server's authorization cookie may live in the owner's profile.
const COOKIE_MAX_AGE_SECS: u64 = 300;

/// Capture an HTML document out of the caller's personal workspace. `root` and
/// `document` are canonical, the document inside the root; the file is served
/// over a temporary loopback server for the duration of the capture only. `name`
/// is the caller's own spelling, quoted in every refusal, and `workspace` is what
/// a leftover browser helper is attributed to.
pub(super) async fn render_file(
    root: &Path,
    document: &Path,
    access: PathAccess,
    name: &str,
    workspace: &str,
) -> anyhow::Result<Vec<u8>> {
    let mut server = PageServer::start(root, document, access, name)?;
    let page_url = server.page_url().to_string();
    let outcome = capture(&page_url, name, workspace, Some(&mut server)).await;
    // The server lives only for the capture: however it ended — a refusal before
    // the browser was reached, or steps cut off by the budget — the port is
    // released before the caller returns.
    server.stop();
    outcome
}

/// Capture an http(s) address as a picture, with no page server.
pub(super) async fn render_url(url: &str, name: &str, workspace: &str) -> anyhow::Result<Vec<u8>> {
    capture(url, name, workspace, None).await
}

/// The steps' outcome, with the capture budget's expiry turned into the tool's
/// own refusal — the page's own settling has [`SETTLE_BOUND`] and is not this.
fn budgeted(
    outcome: Result<anyhow::Result<Vec<u8>>, tokio::time::error::Elapsed>,
    name: &str,
) -> anyhow::Result<Vec<u8>> {
    match outcome {
        Ok(result) => result,
        Err(_) => anyhow::bail!(
            "cannot render {name}: the page did not settle within {} seconds",
            CAPTURE_BUDGET.as_secs()
        ),
    }
}

/// The capture: readiness, the session of its own, the steps, and teardown.
/// `server`, when present, is the temporary page server that must stop before the
/// session stop, which may take its own minute.
async fn capture(
    page_url: &str,
    name: &str,
    workspace: &str,
    mut server: Option<&mut PageServer>,
) -> anyhow::Result<Vec<u8>> {
    #[cfg(test)]
    if let Some(hook) = capture_stand_in() {
        return hook(page_url);
    }
    ChromeTool::ensure_available()
        .await
        .map_err(|error| anyhow::anyhow!("cannot render {name}: {error:#}"))?;
    let cli = super::chrome_daemon::cli_path().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot render {name}: the browser helper is not installed. {}",
            super::chrome_daemon::CHROME_USE_INSTALL_HINT
        )
    })?;
    let mut session = RenderSession::new(cli, render_session_name());
    // Only the steps are inside the budget: a refusal at any of them still
    // reaches the teardown below, and a stop that runs past the budget is not
    // cut off.
    let picture = budgeted(
        tokio::time::timeout(CAPTURE_BUDGET, async {
            open_page(&session, page_url, name, workspace).await?;
            let page = page_box(&session, name, workspace).await?;
            let raw = screenshot(&session, page, name, workspace).await?;
            verified_picture(raw, name).await
        })
        .await,
        name,
    );
    // The server is only needed while the browser is looking at the page.
    if let Some(server) = server.as_mut() {
        server.stop();
    }
    session.stop().await;
    picture
}

/// One render's chrome-use session. The `mahbot-chrome-ephemeral-` prefix is
/// load-bearing: it is what the product's orphan sweep reclaims, so a render
/// killed outright leaves a tab the product closes for free rather than one
/// nothing ever will.
fn render_session_name() -> String {
    format!(
        "{}rasterize-{:016x}",
        crate::chrome::CLI_EPHEMERAL_PREFIX,
        rand::random::<u64>()
    )
}

/// Load `page_url` in a tab of this render's own, at the fixed viewport, and
/// refuse when the navigation did not commit to the page.
async fn open_page(
    session: &RenderSession,
    page_url: &str,
    name: &str,
    workspace: &str,
) -> anyhow::Result<()> {
    let cli = session.cli.as_path();
    let session_name = session.name.as_str();
    step(cli, session_name, &["tab", "new"], name, workspace).await?;
    let (width, height, scale) = (
        VIEWPORT_WIDTH.to_string(),
        VIEWPORT_HEIGHT.to_string(),
        DEVICE_SCALE.to_string(),
    );
    let viewport = [
        "set",
        "viewport",
        width.as_str(),
        height.as_str(),
        scale.as_str(),
    ];
    step(cli, session_name, &viewport, name, workspace).await?;
    let opened = step(cli, session_name, &["open", page_url], name, workspace).await?;
    // The page may have been opened into a fresh tab, so size it again.
    step(cli, session_name, &viewport, name, workspace).await?;
    // What the navigation committed to: a blank tab never loaded the page, and
    // an error page is Chrome's own report of why it could not (the same reading
    // the interactive `chrome` tool makes of its `open`).
    match crate::chrome::navigation_failure(&opened) {
        Some(crate::chrome::NavigationFailure::Blank) => anyhow::bail!(
            "cannot render {name}: the browser never loaded the page — the tab stayed on a blank \
             page"
        ),
        Some(crate::chrome::NavigationFailure::ErrorPage) => {
            // Best-effort: the probe names the net error when one renders.
            let code = step(
                cli,
                session_name,
                &["eval", crate::chrome::contract::ERROR_PAGE_PROBE_JS],
                name,
                workspace,
            )
            .await
            .ok()
            .and_then(|probe| crate::chrome::error_page_code(&probe));
            let reason = code
                .as_deref()
                .map_or("no error code", crate::chrome::contract::net_error_phrase);
            anyhow::bail!("cannot render {name}: the browser could not load the page ({reason})");
        }
        None => {}
    }
    // Best-effort settle: the page's own resources finish loading before the
    // picture is taken. Bounded product-side, so a page that never goes idle
    // costs a few seconds rather than the capture's whole budget.
    best_effort_step(cli, session_name, &["wait", "--load", "networkidle"], name).await;
    // Best-effort too: a page whose webfonts never settle is still worth
    // capturing.
    best_effort_step(
        cli,
        session_name,
        &["wait", "--fn", "document.fonts.status === 'loaded'"],
        name,
    )
    .await;
    Ok(())
}

/// The page's own size in CSS pixels: the box the capture is clipped to, so the
/// picture is the page and nothing the browser happened to draw around it.
async fn page_box(
    session: &RenderSession,
    name: &str,
    workspace: &str,
) -> anyhow::Result<(u32, u32)> {
    // The scrolling document's own size: the viewport for a page shorter than it,
    // the whole content for a taller one.
    const PAGE_BOX_JS: &str = "JSON.stringify({w: Math.max(document.documentElement.scrollWidth, \
        document.body ? document.body.scrollWidth : 0), h: \
        Math.max(document.documentElement.scrollHeight, document.body ? \
        document.body.scrollHeight : 0)})";
    let answer = step(
        session.cli.as_path(),
        session.name.as_str(),
        &["eval", PAGE_BOX_JS],
        name,
        workspace,
    )
    .await?;
    let size = answer
        .data
        .as_ref()
        .and_then(|data| data.get("result"))
        .and_then(serde_json::Value::as_str)
        .and_then(|result| serde_json::from_str::<serde_json::Value>(result).ok());
    let dimension = |key: &str| {
        size.as_ref()
            .and_then(|size| size.get(key))
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
    };
    match (dimension("w"), dimension("h")) {
        (Some(width), Some(height)) => Ok((width, height)),
        _ => anyhow::bail!("cannot render {name}: the page's size could not be read"),
    }
}

/// A settle wait: bounded by a short product clock of its own (chrome-use's
/// `--fn` form ignores a forwarded `--timeout`, so the declaration is what would
/// end it) and its failure is never the render's.
async fn best_effort_step(cli: &Path, session: &str, args: &[&str], name: &str) {
    let clocks = crate::chrome::probe_clocks(SETTLE_BOUND);
    match spawn_cli(CliSpawn {
        path: cli,
        args,
        session: Some(session),
        json: true,
        capture_stderr: false,
        timeout: CliTimeout::Bounded(clocks.kill),
        cancel_kills: true,
        input: None,
        chrome_side: clocks.chrome_side,
        recovery: CliRecovery::Suppressed,
    })
    .await
    {
        CliRun::Output(_) => {}
        CliRun::TimedOut => tracing::debug!(page = name, "rasterize: the settle wait timed out"),
        CliRun::SpawnFailure => {
            tracing::debug!(
                page = name,
                "rasterize: the settle wait could not be started"
            );
        }
    }
}

/// The whole page in one image, written by the browser to a temp file this call
/// owns. The clip is the page's own box, so what is captured is the page and
/// nothing the browser laid out around it; the cap makes the browser fit that to
/// the product's long-side limit — one image of the whole page, at a size this
/// process can decode whole.
async fn screenshot(
    session: &RenderSession,
    page: (u32, u32),
    name: &str,
    workspace: &str,
) -> anyhow::Result<Vec<u8>> {
    let path = std::env::temp_dir().join(format!("rasterize-{:016x}.png", rand::random::<u64>()));
    let _cleanup = TempCapture(path.clone());
    let path_text = path.to_string_lossy();
    let clip = format!("0,0,{},{}", page.0, page.1);
    let cap = super::rasterize::MAX_LONG_SIDE_PX.to_string();
    let shot = step(
        session.cli.as_path(),
        session.name.as_str(),
        &[
            "screenshot",
            "--full",
            "--clip",
            &clip,
            "--max-width",
            &cap,
            "--max-height",
            &cap,
            path_text.as_ref(),
        ],
        name,
        workspace,
    )
    .await;
    // The file is the capture: a truncated envelope must not lose a good
    // picture, and a reported capture with no picture is no capture at all.
    match (shot, std::fs::read(&path).ok()) {
        (_, Some(bytes)) => Ok(bytes),
        (Err(error), None) => Err(error),
        (Ok(_), None) => anyhow::bail!(
            "cannot render {name}: the browser reported a capture but wrote no picture"
        ),
    }
}

/// One chrome-use step in this render's session, through the interactive tool's
/// own dispatch ([`ChromeTool::dispatch_call`]): it runs to the action clocks and
/// may use chrome-use's relay self-heal, and a self-launched browser, a cut-off
/// answer and a leftover helper are read the same way the `chrome` tool reads
/// them.
async fn step(
    cli: &Path,
    session: &str,
    args: &[&str],
    name: &str,
    workspace: &str,
) -> anyhow::Result<ChromeResponse> {
    let started = Instant::now();
    let output = ChromeTool::dispatch_call(
        cli,
        session,
        args,
        crate::chrome::clocks(None, CliRecovery::Allowed),
        workspace,
    )
    .await
    .map_err(|error| anyhow::anyhow!("cannot render {name}: {error:#}"))?;
    if output.leftover_pipes {
        // A helper chrome-use left holding this call's output is the same fact
        // the interactive tool records; the call itself succeeded.
        ChromeTool::record_leftover(session, workspace, started.elapsed());
    }
    Ok(output.output)
}

/// One render's browser session, stopped exactly once: the explicit stop is the
/// product's own verified stop ([`session_stop_outcome`]), and the drop path is
/// the cancellation safety net (a detached best-effort stop) so a cut future
/// cannot leave it behind.
struct RenderSession {
    cli: PathBuf,
    name: String,
    stopped: bool,
}

impl RenderSession {
    fn new(cli: PathBuf, name: String) -> Self {
        Self {
            cli,
            name,
            stopped: false,
        }
    }

    async fn stop(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        stop_session(&self.cli, &self.name).await;
    }
}

impl Drop for RenderSession {
    fn drop(&mut self) {
        if self.stopped {
            return;
        }
        let (cli, name) = (self.cli.clone(), self.name.clone());
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move { stop_session(&cli, &name).await });
        }
    }
}

/// Stop one render's session through the product's own recovery stop: the answer
/// says whether the stop was confirmed, so a session left in an unknown state is
/// reported rather than assumed gone.
async fn stop_session(cli: &Path, name: &str) {
    let recovery =
        super::chrome_daemon::session_stop_outcome(cli, name, SESSION_STOP_TIMEOUT).await;
    if recovery != super::chrome_daemon::SessionRecovery::Stopped {
        tracing::debug!(
            session = name,
            stop = ?recovery,
            "rasterize: the render session stop was not confirmed"
        );
    }
}

/// The temporary screenshot file, removed on every exit path.
struct TempCapture(PathBuf);

impl Drop for TempCapture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The picture the browser wrote, checked to decode whole: a file cut off
/// mid-image is a broken picture, not a capture. The browser already fitted it to
/// the long-side cap, so the decode is bounded. Decoding runs on a blocking
/// thread.
async fn verified_picture(bytes: Vec<u8>, name: &str) -> anyhow::Result<Vec<u8>> {
    match tokio::task::spawn_blocking(move || (image::load_from_memory(&bytes).is_ok(), bytes))
        .await
    {
        Ok((true, picture)) => Ok(picture),
        _ => {
            anyhow::bail!("cannot render {name}: the browser's capture was not a readable picture")
        }
    }
}

// ── The temporary page server ───────────────────────────────────────────

/// The workspace the server hands out: `root` is the boundary, in the one
/// spelling the read gate answers in, and `document` the file the capture
/// navigates to.
struct ServerContext {
    root: PathBuf,
    document: PathBuf,
    access: PathAccess,
    token: String,
    addr: SocketAddr,
}

struct PageServer {
    #[cfg(test)]
    addr: SocketAddr,
    #[cfg(test)]
    token: String,
    page_url: String,
    stopped: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PageServer {
    /// `root` and `document` are canonical and already reduced to the spelling
    /// the read gate answers in, so the boundary, the document's relative path
    /// and every resolved request are read in one spelling (`\\?\…` on Windows
    /// would not agree with the stripped target the gate resolves).
    fn start(root: &Path, document: &Path, access: PathAccess, name: &str) -> anyhow::Result<Self> {
        let token = format!(
            "{:032x}{:032x}",
            rand::random::<u128>(),
            rand::random::<u128>()
        );
        let start_error =
            || anyhow::anyhow!("cannot render {name}: the temporary page server could not start");
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).with_context(start_error)?;
        listener.set_nonblocking(true).with_context(start_error)?;
        let addr = listener.local_addr().with_context(start_error)?;
        let page_url = build_page_url(document, root, addr, &token);
        let stopped = Arc::new(AtomicBool::new(false));
        let context = Arc::new(ServerContext {
            root: root.to_path_buf(),
            document: document.to_path_buf(),
            access,
            token: token.clone(),
            addr,
        });
        let stop = Arc::clone(&stopped);
        let thread = std::thread::Builder::new()
            .name("rasterize-http".to_string())
            .spawn(move || accept_loop(&listener, &context, &stop))
            .with_context(start_error)?;
        Ok(Self {
            #[cfg(test)]
            addr,
            #[cfg(test)]
            token,
            page_url,
            stopped,
            thread: Some(thread),
        })
    }

    fn page_url(&self) -> &str {
        &self.page_url
    }

    /// Release the listening socket. Idempotent: the accept loop polls, so this
    /// returns within one poll; a connection already accepted runs on its own
    /// thread to its own read or write bound and holds no listener.
    fn stop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for PageServer {
    fn drop(&mut self) {
        // Cancellation must not block on the accept thread; the explicit `stop`
        // is what guarantees the port is released before the caller returns.
        self.stopped.store(true, Ordering::Relaxed);
    }
}

/// Every accepted connection on a thread of its own: the browser opens several at
/// once and may park an idle preconnect, so a page's stylesheet must not wait
/// behind another socket. Only the accept loop is joined on stop; a handler
/// finishes on its own read and write bounds.
fn accept_loop(listener: &TcpListener, context: &Arc<ServerContext>, stopped: &AtomicBool) {
    while !stopped.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let context = Arc::clone(context);
                // A connection that cannot get a thread is just not served.
                let _ = std::thread::Builder::new().spawn(move || {
                    handle_connection(&mut stream, &context);
                });
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            // A transient accept failure (a descriptor limit, say) is not the
            // capture's end: the page may still be served once it clears.
            Err(_) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

fn handle_connection(stream: &mut TcpStream, context: &ServerContext) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let Some(request) = read_request(stream) else {
        return;
    };
    let _ = stream.write_all(&handle_request(&request, context));
}

/// One parsed request: the method, the target, and the two headers the
/// authorization rules read. Bodies are not read — the one accepted method
/// carries none.
struct Request {
    method: String,
    target: String,
    host: Option<String>,
    cookie: Option<String>,
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") || buffer.len() > 16 * 1024 {
            break;
        }
    }
    parse_request(std::str::from_utf8(&buffer).ok()?)
}

fn parse_request(raw: &str) -> Option<Request> {
    let mut lines = raw.split("\r\n");
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_string();
    let target = request_line.next()?.to_string();
    let mut request = Request {
        method,
        target,
        host: None,
        cookie: None,
    };
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((header, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match header.trim().to_ascii_lowercase().as_str() {
            "host" => request.host = Some(value.to_string()),
            "cookie" => request.cookie = Some(value.to_string()),
            _ => {}
        }
    }
    Some(request)
}

fn handle_request(request: &Request, context: &ServerContext) -> Vec<u8> {
    if request.method != "GET" {
        return error_response(405, "Method Not Allowed");
    }
    let host = format!("127.0.0.1:{}", context.addr.port());
    if request.host.as_deref() != Some(host.as_str()) {
        return error_response(400, "Bad Request");
    }
    if !request.target.starts_with('/') {
        return error_response(400, "Bad Request");
    }
    if !authorized(request, &context.token) {
        return error_response(403, "Forbidden");
    }
    let Some(mut components) = request_components(&request.target) else {
        return error_response(404, "Not Found");
    };
    // The document's address carries the token as its first segment; a
    // root-relative reference, authorized by the cookie, does not.
    if components
        .first()
        .is_some_and(|first| *first == context.token)
    {
        components.remove(0);
    }
    if components.is_empty() {
        return error_response(404, "Not Found");
    }
    let mut candidate = context.root.clone();
    for component in &components {
        candidate.push(component);
    }
    let Ok(canonical) = std::fs::canonicalize(&candidate)
        .map(|canonical| crate::util::strip_verbatim_prefix(&canonical))
    else {
        return error_response(404, "Not Found");
    };
    // The hard boundary: a canonical target outside the workspace — a symlink
    // that leaves it, a `..` that survived the component check — is not served.
    if !crate::util::is_within(&canonical, &context.root) {
        return error_response(404, "Not Found");
    }
    let Some(resolved) =
        crate::tools::path::resolve_reference_target(&canonical, &context.root, context.access)
    else {
        return error_response(404, "Not Found");
    };
    let Ok(bytes) = std::fs::read(&resolved) else {
        return error_response(404, "Not Found");
    };
    // The document is a page whatever its extension says: a file classified as
    // HTML by its bytes may be named `.txt`. Only the document sets the cookie;
    // subresources carry it from there.
    let document = resolved == context.document;
    let content_type = if document {
        "text/html; charset=utf-8"
    } else {
        content_type_for(&resolved)
    };
    success_response(
        content_type,
        bytes.len(),
        document.then_some(context.token.as_str()),
        &bytes,
    )
}

/// The file components the request's target names under the root, decoded once:
/// `None` when it is not a plain relative file path — no path at all, an empty,
/// `.` or `..` component, or a segment that decodes to a separator.
fn request_components(target: &str) -> Option<Vec<String>> {
    let url = url::Url::parse(&format!("http://127.0.0.1{target}")).ok()?;
    url.path_segments()?
        .map(|segment| {
            let decoded = percent_encoding::percent_decode_str(segment)
                .decode_utf8()
                .ok()?;
            (!decoded.is_empty()
                && decoded != "."
                && decoded != ".."
                && !decoded.contains(['/', '\\']))
            .then(|| decoded.into_owned())
        })
        .collect()
}

/// Whether the request carries the render's token: in its path, or in the cookie
/// the document response set. The cookie is what authorizes a page's
/// root-relative references, which resolve to the origin root and cannot carry
/// the path token.
fn authorized(request: &Request, token: &str) -> bool {
    if request.target.starts_with(&format!("/{token}/")) {
        return true;
    }
    let cookie_name = format!("mb-rasterize-{token}");
    request.cookie.as_ref().is_some_and(|cookie| {
        cookie.split(';').any(|part| {
            part.trim()
                .split_once('=')
                .is_some_and(|(name, _)| name.trim() == cookie_name)
        })
    })
}

fn success_response(
    content_type: &str,
    length: usize,
    token: Option<&str>,
    body: &[u8],
) -> Vec<u8> {
    let cookie = token.map_or_else(String::new, |token| {
        format!(
            "Set-Cookie: mb-rasterize-{token}=1; Path=/; HttpOnly; SameSite=Lax; \
             Max-Age={COOKIE_MAX_AGE_SECS}\r\n"
        )
    });
    let mut response = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {length}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\
         X-Content-Type-Options: nosniff\r\n\
         {cookie}\
         \r\n"
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

fn error_response(code: u16, reason: &str) -> Vec<u8> {
    let body = reason.as_bytes();
    let mut response = format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\
         X-Content-Type-Options: nosniff\r\n\
         \r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

/// The page address: the token, then the document's path relative to the root,
/// every segment percent-encoded.
fn build_page_url(document: &Path, root: &Path, addr: SocketAddr, token: &str) -> String {
    let relative = document.strip_prefix(root).unwrap_or(document);
    let mut url = url::Url::parse(&format!("http://127.0.0.1:{}", addr.port()))
        .expect("a literal base URL parses");
    {
        let mut segments = url
            .path_segments_mut()
            .expect("a base URL can carry path segments");
        segments.push(token);
        for component in relative.components() {
            if let std::path::Component::Normal(part) = component {
                segments.push(&part.to_string_lossy());
            }
        }
    }
    url.to_string()
}

/// The content type the server serves a file as, by its extension: the kinds a
/// page needs to render as it looks — styles, scripts, images, fonts and text.
/// Anything else is opaque bytes.
fn content_type_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("xml") => "application/xml",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("txt" | "md" | "csv") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

// ── Test seam ───────────────────────────────────────────────────────────

/// A test stand-in for the browser: set by tests so no capture touches a real
/// browser. Clone the hook out before invoking it, so the lock is never held
/// across the render.
#[cfg(test)]
type CaptureHook = Arc<dyn Fn(&str) -> anyhow::Result<Vec<u8>> + Send + Sync>;

#[cfg(test)]
static CAPTURE_STAND_IN: Mutex<Option<CaptureHook>> = Mutex::new(None);

/// Installs a capture stand-in for the duration of one test, clearing it on
/// drop. Every test that installs one holds the `html_capture` serial key.
#[cfg(test)]
pub(super) struct CaptureStandIn;

#[cfg(test)]
impl Drop for CaptureStandIn {
    fn drop(&mut self) {
        *CAPTURE_STAND_IN.lock().unwrap_poison() = None;
    }
}

#[cfg(test)]
pub(super) fn stand_in_for_capture(
    hook: impl Fn(&str) -> anyhow::Result<Vec<u8>> + Send + Sync + 'static,
) -> CaptureStandIn {
    *CAPTURE_STAND_IN.lock().unwrap_poison() = Some(Arc::new(hook));
    CaptureStandIn
}

#[cfg(test)]
fn capture_stand_in() -> Option<CaptureHook> {
    CAPTURE_STAND_IN.lock().unwrap_poison().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// A server over `document` (a name inside `root`), both canonical.
    fn served(root: &TempDir, document: &str) -> PageServer {
        let root = std::fs::canonicalize(root.path()).unwrap();
        let document = std::fs::canonicalize(root.join(document)).unwrap();
        PageServer::start(&root, &document, PathAccess::Workspace, "page").unwrap()
    }

    /// One request over a fresh connection, answered to EOF: status, headers
    /// (lower-cased names), and body.
    fn request(addr: SocketAddr, raw: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(raw.as_bytes()).unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        let split = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("a complete response");
        let head = std::str::from_utf8(&response[..split]).unwrap();
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .filter_map(|line| line.split_once(": "))
            .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
            .collect();
        (status, headers, response[split + 4..].to_vec())
    }

    fn get(server: &PageServer, target: &str, extra: &str) -> String {
        format!(
            "GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n{extra}Connection: close\r\n\r\n",
            server.addr.port()
        )
    }

    fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn a_documents_bytes_and_its_own_type_are_served() {
        let root = TempDir::new().unwrap();
        let document = b"<!DOCTYPE html><p>hello</p>";
        std::fs::write(root.path().join("page.txt"), document).unwrap();
        std::fs::write(root.path().join("style.css"), b"body{}").unwrap();
        std::fs::write(root.path().join("picture.png"), b"\x89PNG").unwrap();
        std::fs::write(root.path().join("face.woff2"), b"wOF2").unwrap();
        std::fs::create_dir_all(root.path().join("assets")).unwrap();
        std::fs::write(root.path().join("assets/app.css"), b"a{}").unwrap();
        let server = served(&root, "page.txt");

        let (status, headers, body) = request(
            server.addr,
            &get(&server, &format!("/{}/page.txt", server.token), ""),
        );
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "content-type"),
            Some("text/html; charset=utf-8"),
            "the document is a page whatever its extension says"
        );
        assert_eq!(body, document, "the file's bytes are served unchanged");
        assert!(
            header(&headers, "set-cookie").is_some(),
            "the document response sets the cookie"
        );

        for (name, content_type) in [
            ("style.css", "text/css; charset=utf-8"),
            ("picture.png", "image/png"),
            ("face.woff2", "font/woff2"),
            ("assets/app.css", "text/css; charset=utf-8"),
        ] {
            let (status, headers, _) = request(
                server.addr,
                &get(&server, &format!("/{}/{name}", server.token), ""),
            );
            assert_eq!(status, 200, "{name}");
            assert_eq!(
                header(&headers, "content-type"),
                Some(content_type),
                "{name}"
            );
            assert!(
                header(&headers, "set-cookie").is_none(),
                "{name}: only the document sets the cookie"
            );
        }
    }

    #[test]
    fn root_relative_references_need_the_cookie() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("page.html"), b"<html></html>").unwrap();
        std::fs::write(root.path().join("style.css"), b"body{}").unwrap();
        let server = served(&root, "page.html");

        let (status, headers, _) = request(server.addr, &get(&server, "/style.css", ""));
        assert_eq!(status, 403, "the path carries no token");
        assert!(header(&headers, "set-cookie").is_none());

        let cookie = format!("Cookie: mb-rasterize-{}=1\r\n", server.token);
        let (status, _, body) = request(server.addr, &get(&server, "/style.css", &cookie));
        assert_eq!(
            status, 200,
            "the cookie authorizes a root-relative reference"
        );
        assert_eq!(body, b"body{}");
    }

    #[test]
    fn nothing_outside_the_root_is_served() {
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.css"), b"secret").unwrap();
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("page.html"), b"<html></html>").unwrap();
        std::fs::create_dir(root.path().join("folder")).unwrap();
        let server = served(&root, "page.html");

        let (status, _, _) = request(
            server.addr,
            &get(&server, &format!("/{}/../secret.css", server.token), ""),
        );
        assert_eq!(status, 404, "a parent component never leaves the root");
        let (status, _, _) = request(
            server.addr,
            &get(&server, &format!("/{}/folder", server.token), ""),
        );
        assert_eq!(status, 404, "a directory is not a regular file");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                outside.path().join("secret.css"),
                root.path().join("link.css"),
            )
            .unwrap();
            let (status, _, _) = request(
                server.addr,
                &get(&server, &format!("/{}/link.css", server.token), ""),
            );
            assert_eq!(status, 404, "a symlink out of the root is not served");
        }
    }

    #[test]
    fn malformed_or_unauthorized_requests_are_refused() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("page.html"), b"<html></html>").unwrap();
        let server = served(&root, "page.html");
        let port = server.addr.port();

        let posted = format!(
            "POST /{}/page.html HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n",
            server.token
        );
        assert_eq!(request(server.addr, &posted).0, 405);

        let wrong_host = format!(
            "GET /{}/page.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
            server.token
        );
        assert_eq!(request(server.addr, &wrong_host).0, 400);

        let absent_host = format!("GET /{}/page.html HTTP/1.1\r\n\r\n", server.token);
        assert_eq!(request(server.addr, &absent_host).0, 400);

        let no_token = format!("GET /page.html HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n");
        assert_eq!(request(server.addr, &no_token).0, 403);
    }

    #[test]
    fn stopping_releases_the_port() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("page.html"), b"<html></html>").unwrap();
        let mut server = served(&root, "page.html");
        let addr = server.addr;

        server.stop();
        let listener = TcpListener::bind(addr).expect("the port is free after stop");
        drop(listener);
        assert!(
            TcpStream::connect(addr).is_err(),
            "nothing answers after stop"
        );
    }
}

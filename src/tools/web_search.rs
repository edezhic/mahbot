//! The `web_search` tool: one query against a chosen backend (Firecrawl or
//! Exa), and the per-session expand-id cache that tool instance owns.
//!
//! Not workspace file search (`tools::search`) and not link enrichment — this
//! tool reaches the public web. The cache is deliberately session-scoped: a new
//! tool instance starts empty, and an expand id is valid only in the session
//! that earned it.
//!
//! A flagged search (`pictures`) also lists pictures of the results' subjects —
//! discovery only, never an attached image: the agent fetches what it wants to
//! look at with its own tools.

use crate::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::net::ToSocketAddrs;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;

// ── Backend selection ───────────────────────────────────────────────────────────

/// Backend provider for web search.
pub enum WebSearchBackend {
    /// Firecrawl API (`api.firecrawl.dev/v2/search`).
    Firecrawl { key: String },
    /// Exa API (`api.exa.ai/search`).
    Exa { key: String },
}

impl WebSearchBackend {
    fn key(&self) -> &str {
        match self {
            Self::Firecrawl { key } | Self::Exa { key } => key,
        }
    }
}

// ── Firecrawl API types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ScrapeOptions {
    formats: Vec<String>,
    #[serde(rename = "onlyMainContent")]
    only_main_content: bool,
}

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    limit: usize,
    #[serde(rename = "scrapeOptions")]
    scrape_options: ScrapeOptions,
    /// Flag-gated: without it Firecrawl searches the web source alone, exactly
    /// as before. The image source is a second search (`limit` applies per
    /// source) and is what a flagged call buys.
    #[serde(skip_serializing_if = "Option::is_none")]
    sources: Option<Vec<SearchSource>>,
}

#[derive(Serialize)]
struct SearchSource {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize, Debug)]
struct WebResult {
    title: Option<String>,
    url: String,
    markdown: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ImageResult {
    /// `Option` rather than a defaulted string: a `null` entry costs one
    /// candidate, not the whole response.
    #[serde(default, rename = "imageUrl")]
    image_url: Option<String>,
    /// The page the picture was found on.
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default, rename = "imageWidth")]
    width: Option<u32>,
    #[serde(default, rename = "imageHeight")]
    height: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct SearchData {
    #[serde(default, deserialize_with = "null_as_default")]
    web: Vec<WebResult>,
    /// Present only for a call that asked for the image source.
    #[serde(default, deserialize_with = "null_as_default")]
    images: Vec<ImageResult>,
}

/// Deserialize an absent-or-null field as the field's default.
///
/// Both providers omit a list they did not produce in one response and send
/// `null` in another; a plain `#[serde(default)]` covers only the first, and a
/// parse failure here would cost a flagged call its text results.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Deserialize, Debug)]
struct SearchResponse {
    success: bool,
    data: SearchData,
}

// ── Exa API types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ExaContentsOptions {
    text: bool,
    /// The picture route: `imageLinks` makes Exa attach the per-picture list,
    /// and `richImageLinks` is the form that carries alt text. Requesting the
    /// rich form alone returns nothing, so both counts go out together.
    #[serde(skip_serializing_if = "Option::is_none")]
    extras: Option<ExaExtrasOptions>,
}

#[derive(Serialize)]
struct ExaExtrasOptions {
    #[serde(rename = "imageLinks")]
    image_links: usize,
    #[serde(rename = "richImageLinks")]
    rich_image_links: usize,
}

/// Exa's everyday search mode: fast and cheap. `lite` rejects any request field
/// outside its allowlist with a 400 — `contents.extras` included — so a picture
/// search has to leave it for [`EXA_PICTURE_SEARCH_MODE`], and a new
/// `ExaSearchBody` field must be checked against that allowlist. `lite` serves
/// cached page text only — freshness cannot be requested.
const EXA_SEARCH_MODE: &str = "lite";

/// The mode a picture search uses: the documented default, and the only
/// non-legacy mode that accepts `contents.extras` (`fast`/`instant` still do, but
/// both are marked "use `lite` instead", and `lite` cannot serve pictures).
const EXA_PICTURE_SEARCH_MODE: &str = "auto";

/// Pictures requested per result. They are free within the search price, and a
/// result keeps the first one that passes verification.
const PICTURES_PER_RESULT: usize = 3;

#[derive(Serialize)]
struct ExaSearchBody {
    query: String,
    #[serde(rename = "type")]
    search_type: &'static str,
    #[serde(rename = "numResults")]
    num_results: usize,
    contents: ExaContentsOptions,
}

#[derive(Deserialize, Debug)]
struct ExaImageLink {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    alt: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct ExaExtras {
    #[serde(
        default,
        rename = "richImageLinks",
        deserialize_with = "null_as_default"
    )]
    rich_image_links: Vec<ExaImageLink>,
}

#[derive(Deserialize, Debug)]
struct ExaResult {
    title: Option<String>,
    url: String,
    text: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    extras: ExaExtras,
}

#[derive(Deserialize, Debug)]
struct ExaSearchResponse {
    results: Vec<ExaResult>,
}

// ── Shared result formatting ────────────────────────────────────────────

/// A normalized search result shared by both backends.
struct SearchRow {
    title: Option<String>,
    url: String,
    text: Option<String>,
    description: Option<String>,
}

/// One backend call's answer: the text rows, and what happened to the pictures
/// the call asked for.
struct SearchAnswer {
    rows: Vec<SearchRow>,
    pictures: Pictures,
}

// ── Picture discovery ───────────────────────────────────────────────────────────

/// A picture of a result's subject: offered by the backend, then verified —
/// either by the size it reported or by a probe.
struct Picture {
    /// The address, exactly as the backend returned it: the agent fetches it.
    url: String,
    /// The alt text (Exa) or title (Firecrawl) describing what it shows.
    description: String,
    /// The page the picture was found on.
    page: String,
    /// The size the backend reported; `None` when a probe has to measure it.
    /// Never rendered.
    dims: Option<(u32, u32)>,
}

/// What a search that may ask for pictures learned about them.
enum Pictures {
    /// The call did not ask for pictures; output is exactly what it always was.
    NotRequested,
    /// The call asked and the picture part failed — the backend refused it, or
    /// the request never completed. The text results stand, and the output says
    /// the pictures were unavailable.
    Unavailable,
    /// Verified pictures, no more than the call's text results.
    Usable(Vec<Picture>),
}

/// Smallest size a subject picture may have. Below it, the image is a logo, an
/// avatar or an interface glyph.
const MIN_PICTURE_WIDTH: u32 = 200;
const MIN_PICTURE_HEIGHT: u32 = 150;

/// Address words for page furniture rather than the subject: logos, icons,
/// sprites, avatars, tracking pixels, placeholders and social cards.
const FURNITURE_URL_TOKENS: &[&str] = &[
    "logo",
    "logos",
    "icon",
    "icons",
    "favicon",
    "sprite",
    "sprites",
    "avatar",
    "avatars",
    "gravatar",
    "placeholder",
    "blank",
    "spacer",
    "transparent",
    "pixel",
    "beacon",
    "analytics",
    "adsct",
    "tracking",
    "badge",
    "social",
    "og",
    "opengraph",
    "twitter",
];

/// Alt/title words that admit the picture is not the subject.
const FURNITURE_TEXT_TOKENS: &[&str] = &[
    "logo",
    "logos",
    "icon",
    "icons",
    "avatar",
    "avatars",
    "placeholder",
    "signature",
    "sprite",
    "watermark",
    "badge",
    "twitter",
    "opengraph",
];

/// Word-wise view of a URL or an alt text, lowercased.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
}

/// Whether an alt/title admits to page furniture instead of the subject.
fn describes_furniture(text: &str) -> bool {
    tokens(text).any(|token| FURNITURE_TEXT_TOKENS.contains(&token.as_str()))
}

/// Whether a candidate can be the subject of its page, judged by the only facts
/// a search response carries for free: its address and its alt/title text.
///
/// Every rule is a rejection. A list of logos is worse than no pictures at all,
/// so anything ambiguous — a blank or non-web address, no descriptive text, a
/// furniture word, a page that merely ends in an image name — is dropped here
/// and, if it survives, verified against the image itself.
fn candidate_is_plausible(url: &str, description: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url.trim()) else {
        return false;
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return false;
    }
    // Words in the address are its identity; the query string is campaign tags.
    let without_query = url.split(['#', '?']).next().unwrap_or(url);
    if tokens(without_query).any(|token| FURNITURE_URL_TOKENS.contains(&token.as_str())) {
        return false;
    }
    let path = parsed.path().to_ascii_lowercase();
    // A wiki file (or map) page ends in an image name but is not the image.
    if path.contains("wiki/file:") || path.contains("special:") {
        return false;
    }
    let file = path.rsplit('/').next().unwrap_or_default();
    if file
        .rsplit_once('.')
        .is_some_and(|(_, extension)| !is_picture_extension(extension))
    {
        return false;
    }
    let description = description.trim();
    if description.is_empty() {
        return false;
    }
    // A filename is not a description of what the picture shows.
    if description.contains('/')
        || description
            .rsplit_once('.')
            .is_some_and(|(_, extension)| is_picture_extension(extension))
    {
        return false;
    }
    !describes_furniture(description)
}

/// Whether an extension names an image this crate reads natively, through the
/// same gate the `read` tool's content sniff uses, so the picture formats are
/// not restated here.
fn is_picture_extension(extension: &str) -> bool {
    image::ImageFormat::from_extension(extension.trim())
        .is_some_and(|format| crate::util::image_format_native_label(format).is_some())
}

/// Byte cap of a verification fetch: enough header bytes for every format the
/// `image` crate measures, including JPEGs with a large embedded thumbnail.
const PICTURE_PROBE_BYTES: usize = 64 * 1024;

/// Budget for a verification fetch: image CDNs can take several seconds to
/// answer even a range request (5.2 s measured on one host), and dropping a real
/// picture for its host's latency is worse than waiting — only a flagged call
/// pays this.
const PICTURE_PROBE_TIMEOUT_SECS: u64 = 10;

/// The probe client: it never follows a redirect — every hop would escape the
/// address guard below — and its short timeout keeps a slow host from holding
/// the search open.
fn probe_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        crate::util::http::build_no_redirect_http_client(Duration::from_secs(
            PICTURE_PROBE_TIMEOUT_SECS,
        ))
    })
}

/// Whether a candidate address may be fetched at all: a plain web URL whose host
/// resolves only to public addresses. Picture addresses come from indexed pages,
/// and a page must not be able to point the daemon's fetch at an internal
/// service — neither by naming one nor through a public name that resolves
/// inward.
async fn probe_allowed(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.trim_matches(['[', ']']).to_ascii_lowercase();
    if let Ok(address) = host.parse::<std::net::IpAddr>() {
        return is_public_ip(address);
    }
    if host == "localhost"
        || [".localhost", ".local", ".internal"]
            .iter()
            .any(|suffix| host.ends_with(suffix))
    {
        return false;
    }
    let port = parsed.port_or_known_default().unwrap_or(443);
    // Name resolution is blocking, so it runs off the async workers.
    let Ok(resolved) = tokio::task::spawn_blocking(move || {
        (host.as_str(), port)
            .to_socket_addrs()
            .map(|addresses| addresses.map(|address| address.ip()).collect::<Vec<_>>())
    })
    .await
    else {
        return false;
    };
    resolved.is_ok_and(|addresses| {
        !addresses.is_empty() && addresses.iter().all(|&address| is_public_ip(address))
    })
}

/// Whether an address belongs to the public internet rather than a local
/// network, a device or a reserved range.
fn is_public_ip(address: std::net::IpAddr) -> bool {
    match address {
        std::net::IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation())
        }
        // An IPv4-mapped address reaches the same IPv4 host, so it is judged as
        // one — `::ffff:127.0.0.1` is loopback.
        std::net::IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_public_ip(std::net::IpAddr::V4(v4)),
            None => {
                !(v6.is_loopback()
                    || v6.is_unique_local()
                    || v6.is_unicast_link_local()
                    || v6.is_unspecified()
                    || v6.is_multicast())
            }
        },
    }
}

/// Redirect hops a probe follows, each checked by [`probe_allowed`] first — the
/// client never follows one itself, so a hop cannot escape the guard.
const PICTURE_PROBE_HOPS: usize = 3;

/// The response for the first bytes of the picture at `url`, following up to
/// [`PICTURE_PROBE_HOPS`] validated redirects: image CDNs commonly redirect to
/// the host actually serving the bytes.
async fn probe_response(url: &str) -> Option<reqwest::Response> {
    let client = probe_http_client();
    let mut target = reqwest::Url::parse(url).ok()?;
    for _ in 0..=PICTURE_PROBE_HOPS {
        if !probe_allowed(target.as_str()).await {
            return None;
        }
        let response = client
            .get(target.clone())
            .header(
                reqwest::header::RANGE,
                format!("bytes=0-{}", PICTURE_PROBE_BYTES - 1),
            )
            .send()
            .await
            .ok()?;
        if !response.status().is_redirection() {
            return Some(response);
        }
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)?
            .to_str()
            .ok()?;
        target = target.join(location).ok()?;
    }
    None
}

/// Dimensions of the image at `url`, or `None` when it is not an image, does not
/// answer in time, or is not a format this crate reads. Only the leading bytes
/// are fetched: dimensions live in the header, and the picture itself is never
/// downloaded (viewing is the agent's own tool call, not the search's).
async fn probe_picture_dimensions(url: &str) -> Option<(u32, u32)> {
    let mut response = probe_response(url).await?;
    let is_image = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| {
            content_type
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("image/")
        });
    if !response.status().is_success() || !is_image {
        return None;
    }
    let mut bytes = Vec::with_capacity(PICTURE_PROBE_BYTES);
    while let Some(chunk) = response.chunk().await.ok().flatten() {
        if chunk.is_empty() {
            break;
        }
        bytes.extend_from_slice(&chunk);
        if bytes.len() >= PICTURE_PROBE_BYTES {
            break;
        }
    }
    let format = image::guess_format(&bytes).ok()?;
    image::ImageReader::with_format(std::io::Cursor::new(&bytes), format)
        .into_dimensions()
        .ok()
}

/// Confirm a picture really is an image at the size bar, keeping its address
/// exactly as the backend returned it.
async fn verify_picture(picture: Picture) -> Option<Picture> {
    let (width, height) = match picture.dims {
        Some(dims) => dims,
        None => probe_picture_dimensions(&picture.url).await?,
    };
    (width >= MIN_PICTURE_WIDTH && height >= MIN_PICTURE_HEIGHT).then_some(picture)
}

/// Verify one group of candidates — one result's pictures — and keep the first
/// that passes, so a rejected candidate (a social card that slipped past the
/// token filter) does not cost the result its other, already-paid-for pictures.
async fn first_verified(group: Vec<Picture>) -> Option<Picture> {
    for picture in group {
        if let Some(picture) = verify_picture(picture).await {
            return Some(picture);
        }
    }
    None
}

/// Verify groups concurrently — at most one picture each — and keep the first
/// `cap` in result order, deduped by address.
async fn verified_pictures(groups: Vec<Vec<Picture>>, cap: usize) -> Vec<Picture> {
    let mut seen = HashSet::new();
    futures_util::future::join_all(groups.into_iter().map(first_verified))
        .await
        .into_iter()
        .flatten()
        .filter(|picture| seen.insert(picture.url.clone()))
        .take(cap)
        .collect()
}

// ── Cache ────────────────────────────────────────────────────────────────────────

/// A cached search result with full text content.
#[derive(Debug, Clone)]
pub(crate) struct CachedResult {
    pub(crate) title: String,
    pub(crate) url: String,
    pub(crate) text: String,
}

/// In-memory cache shared between web search backends.
///
/// Holds search results that can later be expanded by ID.  Each agent
/// session gets its own cache, so entries are freed automatically when
/// the agent session ends.
pub(crate) struct WebSearchCache {
    cache: RwLock<HashMap<u64, CachedResult>>,
    next_id: AtomicU64,
}

impl WebSearchCache {
    pub(crate) fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Store a result and return its numeric expand ID.
    pub(crate) async fn cache_result(&self, title: String, url: String, text: String) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cache
            .write()
            .await
            .insert(id, CachedResult { title, url, text });
        id
    }

    /// Retrieve the full text for a cached result by expand ID.
    pub(crate) async fn expand_result(&self, id: u64) -> anyhow::Result<String> {
        let cache_guard = self.cache.read().await;

        let cached = cache_guard.get(&id).ok_or_else(|| {
            anyhow::anyhow!(
                "No cached result with expand id: {id}. This id does not exist \
                 in the current session's cache (it may be from a previous search \
                 in a different context). Use `web_search` with `query` parameter \
                 to run a new search first."
            )
        })?;

        let text = if cached.text.is_empty() {
            "No full text available for this result.".to_string()
        } else {
            cached.text.clone()
        };

        Ok(format!(
            "# {}\n\n**URL:** {}\n\n{}",
            cached.title, cached.url, text
        ))
    }

    /// Return the current counter value for expand-ID range checking.
    pub(crate) fn next_counter(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }
}

// ── Shared HTTP error handling ─────────────────────────────────────────

/// Check a web search API response for HTTP errors and extract structured
/// JSON error messages when available.
///
/// Both Firecrawl and Exa return structured JSON errors — this reads the
/// response body, extracts the error detail (string `error` field or the
/// nested-envelope cascade — see
/// [`crate::util::extract_provider_error_detail`]), and bails with a
/// descriptive message.  Returns the response on success (2xx) so the
/// caller can continue processing it.
pub(crate) async fn check_search_error(
    response: reqwest::Response,
) -> anyhow::Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = crate::util::http::read_error_body(response, "search").await;

    // APIs return structured JSON errors — try to extract for a better message.
    if let Some(error_msg) = search_error_detail(&body) {
        anyhow::bail!("search failed: {error_msg}");
    }

    anyhow::bail!("search failed with status {status}: {body}");
}

/// Extract a descriptive error message from a web-search error response body
/// (string `error` field or the nested-envelope cascade).
#[must_use]
fn search_error_detail(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| crate::util::extract_provider_error_detail(&v))
}

impl WebSearchCache {
    /// Parse and validate `web_search` tool arguments.
    ///
    /// Returns `(query, Option<expand_id>)` where `query` is the trimmed query
    /// string (empty if expanding).  Exactly one of `expand_id` is `Some` or
    /// `query` is non-empty.
    pub(crate) fn validate_execute_args<'a>(
        &self,
        args: &'a Value,
    ) -> anyhow::Result<(&'a str, Option<u64>)> {
        let query = super::get_opt_str_strict(args, "query")?.map(str::trim);
        let expand_id = super::get_opt_u64(args, "expand")?;

        match (query, expand_id) {
            (Some(_), Some(_)) => {
                anyhow::bail!(
                    "The web_search tool accepts either `query` (to run a new search) \
                     or `expand` (to expand a cached result), but not both at the same \
                     time. Please use one or the other."
                );
            }
            (None, None) => {
                anyhow::bail!(
                    "Missing required argument. Provide either `query` (to search \
                     the web) or `expand` (to expand a previous result by id)."
                );
            }
            (Some(query), None) => {
                if query.chars().count() <= 2 {
                    anyhow::bail!(
                        "Search query must be at least 3 characters long. \
                         Got: \"{query}\" ({} chars)",
                        query.chars().count()
                    );
                }

                Ok((query, None))
            }
            (None, Some(id)) => {
                if id >= self.next_counter() {
                    anyhow::bail!(
                        "Invalid expand id: {id}. No search has been run yet in this \
                         session, or the id exceeds any generated value. Use `web_search` \
                         with `query` parameter to run a new search first.",
                    );
                }

                // query is empty, expand_id is Some
                Ok(("", Some(id)))
            }
        }
    }
}

// ── Tool ────────────────────────────────────────────────────────────────────────

/// Web search tool backed by a configurable backend (Firecrawl or Exa).
///
/// Dispatches to the appropriate API based on [`WebSearchBackend`]. Results are
/// cached per-instance in-memory and can be expanded via the `expand` parameter
/// to retrieve full text content.  Each agent session gets its own tool instance,
/// so caches are automatically freed when the agent session ends.
pub struct WebSearchTool {
    backend: WebSearchBackend,
    cache: WebSearchCache,
}

impl WebSearchTool {
    #[must_use]
    pub fn new(backend: WebSearchBackend) -> Self {
        Self {
            backend,
            cache: WebSearchCache::new(),
        }
    }

    /// One backend request, with no fallback of its own.
    async fn backend_answer(&self, query: &str, pictures: bool) -> anyhow::Result<SearchAnswer> {
        match &self.backend {
            WebSearchBackend::Firecrawl { .. } => self.firecrawl_answer(query, pictures).await,
            WebSearchBackend::Exa { .. } => self.exa_answer(query, pictures).await,
        }
    }

    /// A search, with the one fallback rule: a flagged call whose picture part
    /// failed still returns its text results and reports the pictures
    /// unavailable rather than none.
    async fn search(&self, query: &str, pictures: bool) -> anyhow::Result<SearchAnswer> {
        match self.backend_answer(query, pictures).await {
            Ok(answer) => Ok(answer),
            Err(error) if pictures => {
                tracing::warn!(%error, "picture search failed; retrying as text only");
                let mut answer = self.backend_answer(query, false).await?;
                answer.pictures = Pictures::Unavailable;
                Ok(answer)
            }
            Err(error) => Err(error),
        }
    }

    /// One Firecrawl search. `success` is how a refused search is told apart from
    /// an empty one.
    async fn firecrawl_answer(&self, query: &str, pictures: bool) -> anyhow::Result<SearchAnswer> {
        let key = self.backend.key();

        let res = crate::util::http::media_http_client()
            .post("https://api.firecrawl.dev/v2/search")
            .header("Authorization", format!("Bearer {key}"))
            .json(&search_request_body(query, pictures))
            .send()
            .await?;

        let res = check_search_error(res).await?;
        let response = res.json::<SearchResponse>().await?;

        // `success` can be false with results present — render no results, and
        // report the pictures unavailable: the search that would have produced
        // them never ran.
        if !response.success {
            return Ok(SearchAnswer {
                rows: Vec::new(),
                pictures: if pictures {
                    Pictures::Unavailable
                } else {
                    Pictures::NotRequested
                },
            });
        }

        let rows = firecrawl_rows(response.data.web);
        let pictures = if pictures {
            Pictures::Usable(
                verified_pictures(firecrawl_candidates(response.data.images), rows.len()).await,
            )
        } else {
            Pictures::NotRequested
        };
        Ok(SearchAnswer { rows, pictures })
    }

    /// One Exa search: rows, plus one group of picture candidates per result
    /// when the flag asked for pictures.
    async fn exa_answer(&self, query: &str, pictures: bool) -> anyhow::Result<SearchAnswer> {
        let key = self.backend.key();

        let res = crate::util::http::media_http_client()
            .post("https://api.exa.ai/search")
            .header("x-api-key", key)
            .json(&exa_request_body(query, pictures))
            .send()
            .await?;

        let res = check_search_error(res).await?;
        let response = res.json::<ExaSearchResponse>().await?;

        let mut groups = Vec::new();
        let mut rows = Vec::with_capacity(response.results.len());
        for result in response.results {
            if pictures && let Some(group) = exa_picture_group(&result) {
                groups.push(group);
            }
            rows.push(SearchRow {
                title: result.title,
                url: result.url,
                text: result.text,
                description: None,
            });
        }

        let pictures = if pictures {
            Pictures::Usable(verified_pictures(groups, rows.len()).await)
        } else {
            Pictures::NotRequested
        };
        Ok(SearchAnswer { rows, pictures })
    }

    /// Render rows as the agent-facing search output: no-results message,
    /// header, then one entry per row (caching full text for `expand`), then the
    /// picture section a flagged call asked for.
    async fn format_search_output(
        &self,
        query: &str,
        rows: &[SearchRow],
        pictures: Pictures,
    ) -> String {
        let mut lines = Vec::new();
        if rows.is_empty() {
            lines.push(format!("No results found for: {query}"));
        } else {
            lines.push(format!("Search results for: {query}"));
            for (i, result) in rows.iter().enumerate() {
                let title = result.title.as_deref().unwrap_or("(no title)").to_string();

                let id = self
                    .cache
                    .cache_result(
                        title.clone(),
                        result.url.clone(),
                        result.text.clone().unwrap_or_default(),
                    )
                    .await;

                lines.push(format!("{}. [{title}]({url})", i + 1, url = result.url));
                if let Some(description) = result.description.as_deref().filter(|d| !d.is_empty()) {
                    lines.push(format!("   > {description}"));
                }
                lines.push(format!("   `expand` id: `{id}`"));
            }
        }

        // A picture call scrubs its own text (see `should_scrub_output`) and then
        // appends addresses verbatim.
        let mut output = if matches!(pictures, Pictures::NotRequested) {
            lines.join("\n")
        } else {
            crate::util::scrub_credentials(&lines.join("\n"))
        };
        if let Some(section) = picture_section(&pictures) {
            output.push_str("\n\n");
            output.push_str(&section);
        }
        output
    }
}

/// The picture section of a search answer, or `None` when the call did not ask
/// for pictures. "No usable pictures" and "the picture part of the search
/// failed" are different facts and are reported differently.
fn picture_section(pictures: &Pictures) -> Option<String> {
    match pictures {
        Pictures::NotRequested => None,
        Pictures::Unavailable => Some("Pictures unavailable for this query.".to_string()),
        Pictures::Usable(found) if found.is_empty() => {
            Some("No usable pictures for this query.".to_string())
        }
        Pictures::Usable(found) => {
            let mut lines = vec!["Pictures:".to_string()];
            lines.extend(found.iter().enumerate().map(|(i, picture)| {
                format!(
                    "{}. {} — \"{}\" (from {})",
                    i + 1,
                    picture.url,
                    picture.description,
                    picture.page
                )
            }));
            Some(lines.join("\n"))
        }
    }
}

/// The Firecrawl search body. Only `sources` is flag-gated: the scrape options
/// stay in place, so a text search keeps its markdown and its credit price, and
/// a picture search pays for the image source on top.
fn search_request_body(query: &str, pictures: bool) -> SearchRequest {
    SearchRequest {
        query: query.to_string(),
        limit: 10,
        scrape_options: ScrapeOptions {
            formats: vec!["markdown".to_string()],
            only_main_content: true,
        },
        sources: pictures.then(|| {
            vec![
                SearchSource { kind: "web" },
                SearchSource { kind: "images" },
            ]
        }),
    }
}

/// The Exa search body. A picture search leaves `lite` for the one documented
/// mode that accepts `contents.extras`; a text search's body is untouched.
///
/// Neither body carries a moderation switch: both backends keep their own
/// default when the field is absent, and a flagged call must not change the text
/// results by switching filtering on for itself.
fn exa_request_body(query: &str, pictures: bool) -> ExaSearchBody {
    ExaSearchBody {
        query: query.to_string(),
        search_type: if pictures {
            EXA_PICTURE_SEARCH_MODE
        } else {
            EXA_SEARCH_MODE
        },
        num_results: 10,
        contents: ExaContentsOptions {
            text: true,
            extras: pictures.then_some(ExaExtrasOptions {
                image_links: PICTURES_PER_RESULT,
                rich_image_links: PICTURES_PER_RESULT,
            }),
        },
    }
}

/// Firecrawl's web results as the shared row shape.
fn firecrawl_rows(web: Vec<WebResult>) -> Vec<SearchRow> {
    web.into_iter()
        .map(|r| SearchRow {
            title: r.title,
            url: r.url,
            text: r.markdown,
            description: r.description,
        })
        .collect()
}

/// Firecrawl's `images` source as candidate groups of one picture each.
///
/// Firecrawl usually reports a picture's size with the result, so most
/// candidates are judged without a fetch; one that arrives without a size is
/// probed like an Exa candidate. A picture with no hosting page is dropped — an
/// entry the agent cannot attribute is not a discovery.
fn firecrawl_candidates(images: Vec<ImageResult>) -> Vec<Vec<Picture>> {
    images
        .into_iter()
        .filter_map(|image| {
            let url = image.image_url.as_deref()?.trim().to_string();
            let page = image.url.as_deref()?.trim().to_string();
            if page.is_empty() {
                return None;
            }
            let description = image
                .title
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_string();
            let dims = image.width.zip(image.height);
            let big_enough = dims.is_none_or(|(width, height)| {
                width >= MIN_PICTURE_WIDTH && height >= MIN_PICTURE_HEIGHT
            });
            (big_enough && candidate_is_plausible(&url, &description)).then(|| {
                vec![Picture {
                    url,
                    description,
                    page,
                    dims,
                }]
            })
        })
        .collect()
}

/// Every picture of one Exa result its alt text presents as the subject, best
/// first: the first candidate verification accepts becomes that result's
/// picture.
///
/// The same address can appear twice with different alt text, so an address the
/// page labels as furniture anywhere is skipped even where another alt text
/// reads like a subject.
fn exa_picture_group(result: &ExaResult) -> Option<Vec<Picture>> {
    let links = &result.extras.rich_image_links;
    let page = result.url.trim();
    if page.is_empty() {
        return None;
    }
    let furniture: HashSet<&str> = links
        .iter()
        .filter(|link| link.alt.as_deref().is_some_and(describes_furniture))
        .filter_map(|link| link.url.as_deref())
        .collect();
    let group: Vec<Picture> = links
        .iter()
        .filter_map(|link| {
            let url = link.url.as_deref()?.trim();
            let description = link.alt.as_deref().unwrap_or_default().trim();
            (!furniture.contains(url) && candidate_is_plausible(url, description)).then(|| {
                Picture {
                    url: url.to_string(),
                    description: description.to_string(),
                    page: page.to_string(),
                    dims: None,
                }
            })
        })
        .collect();
    (!group.is_empty()).then_some(group)
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query. Mutually exclusive with `expand` — pass exactly one per call."
                },
                "expand": {
                    "type": "integer",
                    "description": "Expand ID of a result from a previous search in this same session; retrieves its full cached text instead of running a new search."
                },
                "pictures": {
                    "type": "boolean",
                    "description": "Also list pictures of the results' subjects, each with its address and hosting page."
                }
            },
            "oneOf": [
                {"required": ["query"]},
                {"required": ["expand"]}
            ]
        })
    }

    fn preserve_full_output(&self) -> bool {
        // Search results can exceed the 5 KB budget — the LLM needs the
        // relevant passages in full.
        true
    }

    fn side_effects(&self) -> bool {
        false // web search has no workspace side effects
    }

    fn should_scrub_output(&self, args: &serde_json::Value) -> bool {
        // A picture call scrubs its own text and then appends picture addresses,
        // which this pass would redact (`?token=…` in a signed image URL) — an
        // address the agent cannot fetch is not a discovered picture. The
        // formatting side reads the same fact from `Pictures::NotRequested`.
        !pictures_requested(args)
    }

    async fn execute(
        &self,
        _ws: &crate::Workspace,
        args: serde_json::Value,
    ) -> anyhow::Result<String> {
        let (query, expand_id) = self.cache.validate_execute_args(&args)?;
        // Read before the expand branch: a wrong-typed flag is a usage error on
        // either kind of call, and `pictures` with `expand` is accepted and
        // ignored — an expand call has no query to find pictures for.
        let pictures = super::get_opt_bool(&args, "pictures")?.unwrap_or(false);

        if let Some(id) = expand_id {
            tracing::debug!("Expanding result id: {}", id);
            return self.cache.expand_result(id).await;
        }

        let answer = self.search(query, pictures).await?;
        Ok(self
            .format_search_output(query, &answer.rows, answer.pictures)
            .await)
    }
}

/// Whether the call asks for pictures and is a search rather than an expand.
///
/// This is the reading `should_scrub_output` needs before `execute` has parsed
/// the arguments; `execute` reads the same flag through `get_opt_bool` so a
/// wrong type is a usage error.
fn pictures_requested(args: &serde_json::Value) -> bool {
    let expanding = !matches!(args.get("expand"), None | Some(Value::Null));
    !expanding && super::get_opt_bool(args, "pictures").ok().flatten() == Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_execute_args() {
        let cache = WebSearchCache::new();
        let cases: &[(&str, Value, Option<&str>, Option<&str>)] = &[
            ("no_args", json!({}), Some("either `query`"), None),
            (
                "both_args",
                json!({"query": "test", "expand": 1}),
                Some("not both"),
                None,
            ),
            (
                "query_too_short",
                json!({"query": "ab"}),
                Some("at least 3"),
                None,
            ),
            // Cache counter starts at 1, so any id >= 1 without a prior search fails
            (
                "expand_invalid_future_id",
                json!({"expand": 999_999_999}),
                Some("Invalid expand id"),
                None,
            ),
            // Wrong-typed expand is a usage error from the shared helper.
            (
                "expand_wrong_type",
                json!({"expand": "1"}),
                Some("usage: argument \"expand\" must be a non-negative integer"),
                None,
            ),
            // So is a wrong-typed query.
            (
                "query_wrong_type",
                json!({"query": 42}),
                Some("usage: argument \"query\" must be a string"),
                None,
            ),
            (
                "good_query",
                json!({"query": "rust programming"}),
                None,
                Some("rust programming"),
            ),
        ];
        for (name, args, err_contains, ok_query) in cases {
            let name = *name;
            match (err_contains, ok_query) {
                (Some(expected), _) => {
                    let err = cache.validate_execute_args(args).unwrap_err().to_string();
                    assert!(
                        err.contains(*expected),
                        "case: {name} — error should contain `{expected}`, got: {err}"
                    );
                }
                (None, Some(expected_query)) => {
                    let (query, expand_id) = cache.validate_execute_args(args).unwrap();
                    assert_eq!(query, *expected_query, "case: {name}");
                    assert!(expand_id.is_none(), "case: {name} — expected no expand_id");
                }
                (None, None) => unreachable!("case: {name}"),
            }
        }
    }

    /// Guards the error-message cascade: a string `error`, a nested envelope's
    /// `message`, the nested `raw`, and a body that is not JSON at all — the
    /// caller then falls back to status plus body instead of claiming a detail.
    #[test]
    fn the_error_detail_cascade_reads_a_string_message_raw_and_stops_at_non_json() {
        let cases: [(&str, &str, Option<&str>); 5] = [
            (
                "a string error field",
                r#"{"success":false,"error":"upstream down"}"#,
                Some("upstream down"),
            ),
            (
                "a nested envelope message",
                r#"{"error":{"message":"invalid API key"}}"#,
                Some("invalid API key"),
            ),
            (
                "a nested raw with no message",
                r#"{"error":{"code":"rate_limited","metadata":{"raw":"too many requests"}}}"#,
                Some("too many requests"),
            ),
            ("a non-JSON body", "plain text", None),
            ("an empty body", "", None),
        ];
        for (name, body, expected) in cases {
            assert_eq!(
                search_error_detail(body).as_deref(),
                expected,
                "case: {name}"
            );
        }
    }

    #[tokio::test]
    async fn cache_and_expand_cases() {
        let cache = WebSearchCache::new();
        let id = cache
            .cache_result(
                "Test Title".to_string(),
                "https://example.com".to_string(),
                "Full text content".to_string(),
            )
            .await;
        assert_eq!(id, 1, "first cached result should get id 1");

        let id2 = cache
            .cache_result(
                "Second".to_string(),
                "https://example.org".to_string(),
                "More content".to_string(),
            )
            .await;
        assert_eq!(id2, 2, "second cached result should get id 2");

        for (name, id, contains) in [
            (
                "the first result",
                1,
                &["Test Title", "https://example.com", "Full text content"][..],
            ),
            ("the second result", 2, &["Second", "More content"][..]),
        ] {
            let expanded = cache.expand_result(id).await.expect("a cached id expands");
            for needle in contains {
                assert!(
                    expanded.contains(needle),
                    "[{name}] expected {needle:?}: {expanded}"
                );
            }
        }

        let err = cache
            .expand_result(42)
            .await
            .expect_err("an uncached id is refused");
        assert!(
            err.to_string().contains("No cached result"),
            "error should mention no cached result: {err}"
        );
    }

    #[test]
    fn test_candidate_is_plausible() {
        // Rejections: pages that only look like pictures, tracking pixels, icons,
        // avatars, logos, social cards, and anything without a description.
        let rejected = [
            (
                "https://en.wikipedia.org/wiki/File:Barack_Obama.jpg",
                "A portrait of Barack Obama",
            ),
            ("https://t.co/1/i/adsct?event=x", "tracking pixel"),
            ("https://cdn.example.com/gallery/icon2.svg", "headphones"),
            (
                "https://secure.gravatar.com/avatar/8d63b097f5",
                "Christian Thomas",
            ),
            (
                "https://rkd.nl/",
                "Logo of RKD — Netherlands Institute for Art History",
            ),
            (
                "https://example.com/articles/2026/story.html",
                "A red-eyed tree frog",
            ),
            ("https://example.com/media/hero.jpg", ""),
            (
                "https://example.com/media/_images/cords.jpg",
                "_images/cords.jpg",
            ),
            // A social card by name, and one announced as such by its text.
            ("https://example.com/og-image-1200x630.png", "Some article"),
            (
                "https://cdn.example.com/media/9f3a.png",
                "Twitter card thumbnail",
            ),
            // A blank, relative or non-web address is not a picture.
            ("", "A red-eyed tree frog"),
            ("  ", "A red-eyed tree frog"),
            ("/media/frog.jpg", "A red-eyed tree frog"),
            ("data:image/png;base64,AAAA", "A red-eyed tree frog"),
        ];
        for (url, description) in rejected {
            assert!(
                !candidate_is_plausible(url, description),
                "should reject {url} ({description:?})"
            );
        }

        // Acceptances: real subject pictures, including an extensionless CDN URL
        // (hero photos are commonly served without an extension).
        let accepted = [
            (
                "https://i.natgeofe.com/n/f6ca1b91/red-eyed-tree-frog.JPG?w=374&h=249",
                "a red-eyed tree frog",
            ),
            (
                "https://digitalassets.tesla.com/tesla-contents/image/upload/f_auto,q_auto/Model-3-Hero-Desktop",
                "Model 3 Performance",
            ),
            (
                "https://cdn.mos.cms.futurecdn.net/oK4kKiMEXRjrM9MyS72Ygf.jpg",
                "Sony WH-1000XM5 headphones next to the Bose QC45",
            ),
            // A dotted directory is not an extension.
            ("https://cdn.example.com/v2.1/hero", "Model 3 front"),
            // Campaign tags in the query string are not the picture's identity.
            (
                "https://cdn.example.com/hero.jpg?utm_source=social",
                "Model 3 front",
            ),
        ];
        for (url, description) in accepted {
            assert!(
                candidate_is_plausible(url, description),
                "should accept {url} ({description:?})"
            );
        }
    }

    #[tokio::test]
    async fn test_probe_allowed_rejects_non_web_and_private_hosts() {
        // Literal addresses only: a hostname case would put DNS in the test.
        for url in [
            "https://93.184.216.34/hero.jpg",
            "http://8.8.8.8/hero.jpg",
            "https://[2606:4700:4700::1111]/hero.jpg",
        ] {
            assert!(probe_allowed(url).await, "should allow {url}");
        }
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/hero.jpg",
            "http://localhost:8080/hero.jpg",
            "http://127.0.0.1/hero.jpg",
            "http://192.168.1.10/hero.jpg",
            "http://169.254.169.254/latest/meta-data/",
            "http://172.16.0.1/hero.jpg",
            "http://[::1]/hero.jpg",
            "http://[::ffff:127.0.0.1]/hero.jpg",
            "http://[::ffff:169.254.169.254]/hero.jpg",
            "http://[fd00::1]/hero.jpg",
            "http://[fe80::1]/hero.jpg",
            "http://printer.local/hero.jpg",
            "http://metadata.internal/hero.jpg",
            "no-scheme.example.com/hero.jpg",
        ] {
            assert!(!probe_allowed(url).await, "should reject {url}");
        }
    }

    #[test]
    fn test_request_bodies_gate_pictures() {
        let plain = serde_json::to_value(search_request_body("frogs", false)).unwrap();
        assert!(
            plain.get("sources").is_none(),
            "unflagged Firecrawl body: {plain}"
        );
        assert!(
            plain.get("scrapeOptions").is_some(),
            "scraping must stay: {plain}"
        );
        let flagged = serde_json::to_value(search_request_body("frogs", true)).unwrap();
        assert_eq!(
            flagged["sources"],
            json!([{"type": "web"}, {"type": "images"}])
        );

        let plain = serde_json::to_value(exa_request_body("frogs", false)).unwrap();
        assert_eq!(plain["type"], EXA_SEARCH_MODE);
        assert!(
            plain["contents"].get("extras").is_none(),
            "unflagged Exa body: {plain}"
        );
        let flagged = serde_json::to_value(exa_request_body("frogs", true)).unwrap();
        assert_eq!(flagged["type"], EXA_PICTURE_SEARCH_MODE);
        assert_eq!(
            flagged["contents"]["extras"],
            json!({"imageLinks": PICTURES_PER_RESULT, "richImageLinks": PICTURES_PER_RESULT})
        );
    }

    #[test]
    fn test_firecrawl_candidates_filter() {
        let image =
            |url: &str, title: &str, width: Option<u32>, height: Option<u32>, page: &str| {
                ImageResult {
                    image_url: Some(url.to_string()),
                    url: Some(page.to_string()),
                    title: Some(title.to_string()),
                    width,
                    height,
                }
            };
        let candidates = firecrawl_candidates(vec![
            // A logo is small and a tracking pixel has no size at all.
            image(
                "https://site.test/logo.png",
                "Site logo",
                Some(80),
                Some(80),
                "https://a.test",
            ),
            // Neither address nor title says furniture: only the size does.
            image(
                "https://img.test/9f3a.jpg",
                "Jane Doe",
                Some(96),
                Some(96),
                "https://a.test",
            ),
            image(
                "https://site.test/pixel.gif",
                "Pixel",
                None,
                None,
                "https://a.test",
            ),
            // Without a hosting page the agent cannot attribute it.
            ImageResult {
                image_url: Some("https://img.test/frog.jpg".to_string()),
                url: None,
                title: Some("Red-eyed tree frog".to_string()),
                width: Some(1600),
                height: Some(900),
            },
            image(
                "https://img.test/frog-hero.jpg",
                "Red-eyed tree frog",
                Some(1600),
                Some(900),
                "https://a.test",
            ),
        ]);
        assert_eq!(candidates.len(), 1);
        let picture = &candidates[0][0];
        assert_eq!(picture.url, "https://img.test/frog-hero.jpg");
        assert_eq!(picture.page, "https://a.test");
        assert_eq!(picture.dims, Some((1600, 900)));
    }

    #[tokio::test]
    async fn test_verified_pictures_takes_first_per_group_and_caps() {
        let picture = |url: &str, (width, height): (u32, u32)| Picture {
            url: url.to_string(),
            description: "a red-eyed tree frog".to_string(),
            page: "https://site.test/page".to_string(),
            dims: Some((width, height)),
        };
        let groups = vec![
            // The first candidate is too small; the group's second must still
            // count.
            vec![
                picture("https://img.test/small.jpg", (120, 90)),
                picture("https://img.test/frog.jpg", (1200, 800)),
            ],
            vec![picture("https://img.test/other.jpg", (800, 600))],
            vec![picture("https://img.test/third.jpg", (800, 600))],
        ];
        let pictures = verified_pictures(groups, 2).await;
        assert_eq!(pictures.len(), 2, "capped at the text-result count");
        assert_eq!(pictures[0].url, "https://img.test/frog.jpg");
        assert_eq!(pictures[1].url, "https://img.test/other.jpg");
    }

    #[test]
    fn test_exa_picture_group_skips_an_address_labelled_as_furniture() {
        let result = ExaResult {
            title: Some("Headphones".to_string()),
            url: "https://site.test/headphones".to_string(),
            text: None,
            extras: ExaExtras {
                rich_image_links: vec![
                    ExaImageLink {
                        url: Some("https://site.test/default-guide.png".to_string()),
                        alt: Some("Sony WH-1000XM5".to_string()),
                    },
                    // The same address, described honestly elsewhere on the page.
                    ExaImageLink {
                        url: Some("https://site.test/default-guide.png".to_string()),
                        alt: Some("Site default logo image".to_string()),
                    },
                    ExaImageLink {
                        url: Some("https://img.test/xm5.jpg".to_string()),
                        alt: Some("Sony WH-1000XM5 headphones".to_string()),
                    },
                    ExaImageLink {
                        url: Some(String::new()),
                        alt: None,
                    },
                    ExaImageLink {
                        url: None,
                        alt: None,
                    },
                ],
            },
        };
        let group = exa_picture_group(&result).expect("a usable candidate");
        assert_eq!(group.len(), 1);
        assert_eq!(group[0].url, "https://img.test/xm5.jpg");
        assert_eq!(group[0].page, "https://site.test/headphones");
        assert_eq!(group[0].description, "Sony WH-1000XM5 headphones");
        assert!(group[0].dims.is_none(), "Exa reports no dimensions");
    }

    #[test]
    fn test_scrub_policy_covers_picture_calls_only() {
        let tool = WebSearchTool::new(WebSearchBackend::Exa { key: String::new() });
        // A picture search scrubs its own text, so the agent-level pass is off;
        // an expand or a plain search keeps the default, and an explicit null
        // is the same as an absent argument on either flag.
        assert!(!tool.should_scrub_output(&json!({"query": "frogs", "pictures": true})));
        assert!(tool.should_scrub_output(&json!({"query": "frogs"})));
        assert!(tool.should_scrub_output(&json!({"query": "frogs", "pictures": null})));
        assert!(tool.should_scrub_output(&json!({"expand": 3, "pictures": true})));
        assert!(
            !tool.should_scrub_output(&json!({"query": "frogs", "pictures": true, "expand": null}))
        );
    }

    #[tokio::test]
    async fn test_picture_section_keeps_addresses_intact() {
        let tool = WebSearchTool::new(WebSearchBackend::Exa {
            key: "test-key".to_string(),
        });
        let rows = vec![SearchRow {
            title: Some("Signed images".to_string()),
            url: "https://site.test/page".to_string(),
            text: None,
            description: Some("key api_key=abcdefgh12345678".to_string()),
        }];
        let signed = "https://cdn.test/frog.jpg?token=abcdefgh12345678";
        let output = tool
            .format_search_output(
                "frogs",
                &rows,
                Pictures::Usable(vec![Picture {
                    url: signed.to_string(),
                    description: "a red-eyed tree frog".to_string(),
                    page: "https://site.test/page".to_string(),
                    dims: None,
                }]),
            )
            .await;
        // The page text is scrubbed ...
        assert!(output.contains("*[REDACTED]"), "{output}");
        // ... while the picture address survives whole, with its page.
        assert!(output.contains(signed), "{output}");
        assert!(output.contains("(from https://site.test/page)"), "{output}");
        // Pictures never take expand ids: only the row was cached.
        assert_eq!(tool.cache.next_counter(), 2, "{output}");
        assert_eq!(
            output.matches("`expand` id: `").count(),
            1,
            "one expand id for one text result: {output}"
        );
    }

    #[tokio::test]
    async fn test_picture_section_reports_empty_and_refused_separately() {
        let tool = WebSearchTool::new(WebSearchBackend::Exa {
            key: "test-key".to_string(),
        });
        let rows = vec![SearchRow {
            title: Some("Result".to_string()),
            url: "https://site.test/page".to_string(),
            text: None,
            description: None,
        }];
        let empty = tool
            .format_search_output("frogs", &rows, Pictures::Usable(Vec::new()))
            .await;
        assert!(
            empty.ends_with("No usable pictures for this query."),
            "{empty}"
        );
        let refused = tool
            .format_search_output("frogs", &rows, Pictures::Unavailable)
            .await;
        assert!(
            refused.ends_with("Pictures unavailable for this query."),
            "{refused}"
        );
        // A flagged call that found no text results still says what happened to
        // its pictures instead of looking like a plain empty search.
        let no_rows = tool
            .format_search_output("frogs", &[], Pictures::Usable(Vec::new()))
            .await;
        assert_eq!(
            no_rows,
            "No results found for: frogs\n\nNo usable pictures for this query."
        );
        let no_rows_refused = tool
            .format_search_output("frogs", &[], Pictures::Unavailable)
            .await;
        assert_eq!(
            no_rows_refused,
            "No results found for: frogs\n\nPictures unavailable for this query."
        );
        let plain = tool
            .format_search_output("frogs", &rows, Pictures::NotRequested)
            .await;
        assert!(
            !plain.contains("ictures"),
            "unflagged output is unchanged: {plain}"
        );
    }
}

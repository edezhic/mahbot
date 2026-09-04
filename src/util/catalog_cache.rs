//! Generic single-flight, endpoint-keyed, TTL-cached fetch for the OpenRouter
//! model catalogs. Shared by the image and video catalog clients so their
//! cache/backoff behavior cannot drift. Also hosts the shared envelope parser
//! ([`parse_envelope`]) so the `{ "data": [...] }` contract is single-source.
//!
//! Caching: fetched once (single-flight), keyed by the configured provider
//! endpoint, and reused for [`CATALOG_TTL`]. A failed fetch — including a
//! timeout — is stored as a short-lived negative cache
//! ([`CATALOG_RETRY_BACKOFF`]) and degrades to fail-open: [`Catalog::get`]
//! returns `None` and callers proceed without capability data. Capability
//! checks can force a refresh on a membership miss via [`Catalog::refresh_for_miss`]
//! (cooldown-bounded); a plain [`Catalog::get`] never refreshes.

use crate::util::UnwrapPoison;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// How long a fetched catalog is considered fresh.
const CATALOG_TTL: Duration = Duration::from_hours(24);

/// Backoff before retrying a failed catalog fetch (negative cache).
const CATALOG_RETRY_BACKOFF: Duration = Duration::from_mins(1);

/// Bounds a single catalog fetch so a hung provider never stalls a caller
/// (session start, change detection, per-execute validation) beyond this
/// window. A timeout is recorded as a failure, so the negative-cache backoff
/// applies instead of re-fetching on every call.
///
/// Note: this is also a fail-open cutoff for slow-but-working providers — a
/// catalog that takes longer than 10s (previously bounded only by the media
/// client's ~2-min request timeout) now consistently fails open and is
/// retried only after the backoff window. Accepted: a bounded first-message
/// latency is worth more than a slow catalog.
const CATALOG_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// A single catalog fetch: URL → response body. Production resolves to
/// [`crate::util::http::get_json_from_provider`]; tests install a scripted
/// fetch so refresh/cooldown behavior is hermetic. A fn pointer keeps
/// `Catalog` const-constructible for the static catalog clients.
pub(crate) type FetchFuture = Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send>>;
pub(crate) type FetchFn = fn(url: &str, label: &str) -> FetchFuture;

fn fetch_from_provider(url: &str, label: &str) -> FetchFuture {
    let url = url.to_string();
    let label = label.to_string();
    Box::pin(async move { crate::util::http::get_json_from_provider(&url, &label).await })
}

/// Parse a `{ "data": [...] }` catalog envelope into a model map. The
/// per-entry parser is supplied by each catalog client (field shapes differ).
///
/// `label` feeds the error strings and must match the client's [`Catalog::new`]
/// label so parse errors stay consistent with log labels.
///
/// # Errors
///
/// - Missing/invalid `data` array or an empty model set.
pub(crate) fn parse_envelope<T>(
    body: &Value,
    label: &str,
    parse_model: fn(&Value) -> Option<(String, T)>,
) -> anyhow::Result<HashMap<String, T>> {
    let data = body
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("{label} response missing `data` array"))?;
    let mut models = HashMap::new();
    for entry in data {
        if let Some((id, info)) = parse_model(entry) {
            models.insert(id, info);
        }
    }
    if models.is_empty() {
        anyhow::bail!("{label} contained no models");
    }
    Ok(models)
}

struct CacheEntry<T> {
    endpoint: String,
    fetched_at: Instant,
    /// `None` = a failed fetch (negative cache for the backoff window).
    catalog: Option<Arc<T>>,
}

enum Lookup<T> {
    Fresh(Arc<T>),
    Backoff,
    Miss,
}

/// An endpoint-keyed catalog with a single-flight fetch and negative backoff.
pub(crate) struct Catalog<T> {
    /// URL path suffix appended to the provider base URL, e.g. `/images/models`.
    path: &'static str,
    /// Human-readable label used in fetch error logs.
    label: &'static str,
    /// Parses a fetched response body into the catalog type.
    parse: fn(&Value) -> anyhow::Result<T>,
    /// Injected fetch transport — resolves to [`fetch_from_provider`] in
    /// production; tests install a scripted fetch.
    fetch: FetchFn,
    cache: OnceLock<RwLock<Option<CacheEntry<T>>>>,
    /// Serializes concurrent fetches — one network call regardless of how many
    /// callers hit the catalog at once.
    fetch_lock: OnceLock<tokio::sync::Mutex<()>>,
    /// Per-endpoint timestamp of the last miss-triggered refresh attempt, for
    /// the cooldown.
    last_miss_refresh: OnceLock<RwLock<Option<(String, Instant)>>>,
}

impl<T> Catalog<T> {
    pub(crate) const fn new(
        path: &'static str,
        label: &'static str,
        parse: fn(&Value) -> anyhow::Result<T>,
    ) -> Self {
        Self {
            path,
            label,
            parse,
            fetch: fetch_from_provider,
            cache: OnceLock::new(),
            fetch_lock: OnceLock::new(),
            last_miss_refresh: OnceLock::new(),
        }
    }

    /// Test-only: replace the fetch transport with a scripted one.
    #[cfg(test)]
    pub(crate) const fn with_fetch(mut self, fetch: FetchFn) -> Self {
        self.fetch = fetch;
        self
    }

    /// Return the cached catalog for `endpoint` if fresh, otherwise fetch it
    /// (single-flight). Returns `None` (fail-open) when the catalog is
    /// unavailable — retried after a short negative-cache backoff.
    pub(crate) async fn get(&self, endpoint: &str) -> Option<Arc<T>> {
        let base = crate::providers::ensure_base_url(endpoint);
        match self.lookup(&base) {
            Lookup::Fresh(catalog) => return Some(catalog),
            Lookup::Backoff => return None,
            Lookup::Miss => {}
        }
        let _guard = self.fetch_lock().await;
        match self.lookup(&base) {
            Lookup::Fresh(catalog) => return Some(catalog),
            Lookup::Backoff => return None,
            Lookup::Miss => {}
        }
        self.fetch_and_store(base).await
    }

    /// Fetch the catalog for the already-normalized `base` (bypassing any cache
    /// check) and store the outcome: fresh catalog on success, negative cache on
    /// failure. Returns the fetched catalog, or `None` on any failure (fail-open).
    async fn fetch_and_store(&self, base: String) -> Option<Arc<T>> {
        let url = format!("{base}{}", self.path);
        let fetched =
            tokio::time::timeout(CATALOG_FETCH_TIMEOUT, (self.fetch)(&url, self.label)).await;
        match fetched {
            Err(_) => {
                tracing::warn!(
                    catalog = self.label,
                    "Timed out fetching catalog — proceeding without capability data"
                );
                self.store_failure(base);
                None
            }
            Ok(Ok(body)) => match (self.parse)(&body) {
                Ok(catalog) => {
                    let catalog = Arc::new(catalog);
                    *self.cache_lock().write().unwrap_poison() = Some(CacheEntry {
                        endpoint: base,
                        fetched_at: Instant::now(),
                        catalog: Some(catalog.clone()),
                    });
                    Some(catalog)
                }
                Err(e) => {
                    tracing::warn!(catalog = self.label, error = %e, "Failed to parse catalog — proceeding without capability data");
                    self.store_failure(base);
                    None
                }
            },
            Ok(Err(e)) => {
                tracing::warn!(catalog = self.label, error = %e, "Failed to fetch catalog — proceeding without capability data");
                self.store_failure(base);
                None
            }
        }
    }

    /// Forced refresh for a capability-check membership miss: the model id is
    /// absent from the cached snapshot, so the snapshot may predate the model's
    /// OpenRouter listing. Atomic — acquires the fetch lock (single-flight),
    /// bypasses the TTL check, fetches, stores. Bounded by a per-endpoint
    /// cooldown ([`CATALOG_RETRY_BACKOFF`]): inside the window no refetch
    /// happens and the last fetched snapshot is returned as-is, so absence
    /// confirmed by a successful fresh fetch stands (the caller rejects).
    /// Returns the refreshed/last snapshot, or `None` when the fetch failed
    /// (fail-open — the caller must not reject on a stale/missing catalog).
    pub(crate) async fn refresh_for_miss(&self, endpoint: &str) -> Option<Arc<T>> {
        let base = crate::providers::ensure_base_url(endpoint);
        let _guard = self.fetch_lock().await;
        if self.miss_refresh_in_cooldown(&base) {
            // Cooldown window: re-check against the last fetched snapshot.
            return match self.lookup(&base) {
                Lookup::Fresh(catalog) => Some(catalog),
                _ => None,
            };
        }
        self.record_miss_refresh(&base);
        self.fetch_and_store(base).await
    }

    fn miss_refresh_in_cooldown(&self, endpoint: &str) -> bool {
        self.last_miss_refresh
            .get_or_init(|| RwLock::new(None))
            .read()
            .unwrap_poison()
            .as_ref()
            .is_some_and(|(ep, at)| ep == endpoint && at.elapsed() < CATALOG_RETRY_BACKOFF)
    }

    fn record_miss_refresh(&self, endpoint: &str) {
        *self
            .last_miss_refresh
            .get_or_init(|| RwLock::new(None))
            .write()
            .unwrap_poison() = Some((endpoint.to_string(), Instant::now()));
    }

    fn lookup(&self, endpoint: &str) -> Lookup<T> {
        let guard = self.cache_lock().read().unwrap_poison();
        let Some(cache) = guard.as_ref() else {
            return Lookup::Miss;
        };
        if cache.endpoint != endpoint {
            return Lookup::Miss;
        }
        match &cache.catalog {
            Some(catalog) if cache.fetched_at.elapsed() < CATALOG_TTL => {
                Lookup::Fresh(catalog.clone())
            }
            None if cache.fetched_at.elapsed() < CATALOG_RETRY_BACKOFF => Lookup::Backoff,
            _ => Lookup::Miss,
        }
    }

    fn store_failure(&self, endpoint: String) {
        *self.cache_lock().write().unwrap_poison() = Some(CacheEntry {
            endpoint,
            fetched_at: Instant::now(),
            catalog: None,
        });
    }

    fn cache_lock(&self) -> &RwLock<Option<CacheEntry<T>>> {
        self.cache.get_or_init(|| RwLock::new(None))
    }

    async fn fetch_lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.fetch_lock
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await
    }

    /// Test-only: seed the cache directly (no network).
    #[cfg(test)]
    pub(crate) fn seed(&self, endpoint: &str, catalog: Option<Arc<T>>) {
        self.seed_at(endpoint, catalog, Instant::now());
    }

    /// Test-only: seed the cache with an explicit fetch time (e.g. to simulate
    /// an expired backoff window).
    #[cfg(test)]
    pub(crate) fn seed_at(&self, endpoint: &str, catalog: Option<Arc<T>>, fetched_at: Instant) {
        *self.cache_lock().write().unwrap_poison() = Some(CacheEntry {
            endpoint: endpoint.to_string(),
            fetched_at,
            catalog,
        });
    }

    /// Test-only: classify the current lookup state for `endpoint`.
    #[cfg(test)]
    pub(crate) fn lookup_state(&self, endpoint: &str) -> LookupState {
        match self.lookup(endpoint) {
            Lookup::Fresh(_) => LookupState::Fresh,
            Lookup::Backoff => LookupState::Backoff,
            Lookup::Miss => LookupState::Miss,
        }
    }
}

/// Test-only: [`Lookup`] stripped of the payload.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LookupState {
    Fresh,
    Backoff,
    Miss,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Default)]
    struct FakeCatalog;

    fn parse_fake(body: &Value) -> anyhow::Result<FakeCatalog> {
        if body.get("data").and_then(Value::as_array).is_none() {
            anyhow::bail!("missing data array");
        }
        Ok(FakeCatalog)
    }

    #[test]
    fn lookup_is_endpoint_keyed_with_negative_backoff() {
        let catalog = Catalog::new("/fake/models", "Fake catalog", parse_fake);
        let endpoint = "https://openrouter.ai/api/v1";
        let other = "https://other.example/api/v1";

        // Fresh catalog → Fresh; a different provider endpoint → Miss.
        catalog.seed(endpoint, Some(Arc::new(FakeCatalog)));
        assert_eq!(catalog.lookup_state(endpoint), LookupState::Fresh);
        assert_eq!(catalog.lookup_state(other), LookupState::Miss);

        // Failed fetch → Backoff inside the window, Miss once it expires.
        catalog.seed(endpoint, None);
        assert_eq!(catalog.lookup_state(endpoint), LookupState::Backoff);
        catalog.seed_at(
            endpoint,
            None,
            Instant::now()
                .checked_sub(CATALOG_RETRY_BACKOFF + Duration::from_secs(1))
                .expect("clock is past the backoff window"),
        );
        assert_eq!(catalog.lookup_state(endpoint), LookupState::Miss);
    }

    #[test]
    fn parse_envelope_rejects_missing_or_empty_data() {
        fn parse_id(entry: &Value) -> Option<(String, ())> {
            entry
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_string(), ()))
        }

        // Missing `data` → exact error string (label literal verbatim).
        let err = parse_envelope(&json!({}), "Image models catalog", parse_id).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Image models catalog response missing `data` array"
        );
        // Non-array `data` → same error.
        assert!(parse_envelope(&json!({"data": {}}), "Image models catalog", parse_id).is_err());

        // Empty model set → exact error string.
        let err =
            parse_envelope(&json!({"data": []}), "Video models catalog", parse_id).unwrap_err();
        assert_eq!(err.to_string(), "Video models catalog contained no models");
        // All entries skipped by the per-entry parser → same error.
        assert!(
            parse_envelope(
                &json!({"data": [{"name": "x"}]}),
                "Video models catalog",
                parse_id
            )
            .is_err()
        );

        // Success path maps parsed entries.
        let models = parse_envelope(
            &json!({"data": [{"id": "m1"}, {"id": "m2"}]}),
            "Fake models catalog",
            parse_id,
        )
        .expect("parsed");
        assert_eq!(models.len(), 2);
    }
}

//! Generic single-flight, endpoint-keyed, TTL-cached fetch for the OpenRouter
//! model catalogs. Shared by the image and video catalog clients so their
//! cache/backoff behavior cannot drift.
//!
//! Caching: fetched once (single-flight), keyed by the configured provider
//! endpoint, and reused for [`CATALOG_TTL`]. A failed fetch — including a
//! timeout — is stored as a short-lived negative cache
//! ([`CATALOG_RETRY_BACKOFF`]) and degrades to fail-open: [`Catalog::get`]
//! returns `None` and callers proceed without capability data.

use crate::util::UnwrapPoison;
use serde_json::Value;
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
    cache: OnceLock<RwLock<Option<CacheEntry<T>>>>,
    /// Serializes concurrent fetches — one network call regardless of how many
    /// callers hit the catalog at once.
    fetch_lock: OnceLock<tokio::sync::Mutex<()>>,
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
            cache: OnceLock::new(),
            fetch_lock: OnceLock::new(),
        }
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

        let url = format!("{base}{}", self.path);
        let fetched = tokio::time::timeout(
            CATALOG_FETCH_TIMEOUT,
            crate::util::http::get_json_from_provider(&url, self.label),
        )
        .await;
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
}

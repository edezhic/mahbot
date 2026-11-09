//! The machine-wide model root, and the fixed table of files the speech engines read.
//!
//! The root is the inference crate's own ([`mahbot_inference::paths::root`]) — one
//! location per operating system, no environment variable, no home-relative
//! fallback, no search anywhere else. The installer creates it; the product only
//! ever reports its absence, naming the path.
//!
//! Inside it the product knows exactly the file names of the sets it fetches, taken from the
//! engine's published table rather than restated here. A purpose fetches its own set and
//! nothing else — the wake word's listening tier takes the pair alone, and the tagger arrives
//! with the first transcription — so spoken replies never make recognition require, verify or
//! download their weights, and the other way round. The folder is never enumerated for model
//! files, so anything else in it — the other consumer's corpus folders above all — is never read,
//! written, overwritten or removed; the one other name touched is a part file of the
//! product's own (`<name>.part-<pid>-<seq>`) that a killed fetch left behind.
//!
//! Identity is the digest, never the name: the recognition body sits under this exact name
//! in other repositories with other bytes, so each entry carries its own source and pin.
//! What a file *is* — body, projector or tagger — comes from its own metadata: the engine
//! refuses a pair whose halves do not belong together, so a swap is caught there.
//!
//! The digest is checked once per process start, in the background ([`check_once`]), and the
//! result is remembered: a load only stats the files, so the first use of an engine never pays
//! a multi-gigabyte hash. A file the check finds wrong is recorded, and the next load fetches
//! it again; a missing file is left to the on-demand fetch, so a start never downloads.

use crate::util::UnwrapPoison;
use anyhow::{Context as _, Result, bail};
use mahbot_inference::ced::CED_MODEL;
use mahbot_inference::loader::{self, Role};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, info, warn};

/// The root, or the error naming it. Missing is the installer's fault to fix, not
/// something the product repairs by creating a substitute location.
#[cfg(not(test))]
pub(crate) fn root() -> Result<PathBuf> {
    loader::models_dir()
}

/// No test resolves the live root: the set there is the owner's, and a test that reached it
/// would hash and map gigabytes of their files. Everything else here takes the directory as
/// an argument and is exercised against a temporary one.
#[cfg(test)]
pub(crate) fn root() -> Result<PathBuf> {
    bail!("speech does not run under test")
}

/// Which engine a fetch is for. A purpose is a set: its own files, its own presence check and
/// its own fetch, so one engine never waits on the other's gigabytes.
#[derive(Clone, Copy)]
pub(crate) enum Purpose {
    /// The recognition pair alone: all the wake word's listening tier loads.
    Listening,
    /// The recognition pair plus the audio tagger every transcription depends on.
    Recognition,
    /// The synthesiser's talker pair.
    Synthesis,
}

impl Purpose {
    /// The name a log line leads with.
    pub(crate) const fn title(self) -> &'static str {
        match self {
            Self::Listening => "Speech recognition (listening)",
            Self::Recognition => "Speech recognition",
            Self::Synthesis => "Speech synthesis",
        }
    }

    /// What the set is, as a failure speaks of it.
    pub(crate) const fn what(self) -> &'static str {
        match self {
            Self::Listening | Self::Recognition => "speech",
            Self::Synthesis => "speech synthesis",
        }
    }
}

/// One file of the set: the engine's name for it, the bytes it must be, and where
/// those bytes come from. The source is per entry because it cannot be derived: the
/// recognition body's upstream file is named differently, so a URL built from our
/// name would 404.
struct ModelFile {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
    url: &'static str,
    timeout: Duration,
}

/// The files one purpose fetches.
fn model_files(purpose: Purpose) -> Result<Vec<ModelFile>> {
    match purpose {
        Purpose::Listening => recognition_pair(),
        Purpose::Recognition => {
            let mut files = recognition_pair()?;
            files.push(tagger());
            Ok(files)
        }
        Purpose::Synthesis => synthesis_pair(),
    }
}

/// The recognition body and its projector: the pair the listening tier and a transcription both
/// open.
fn recognition_pair() -> Result<Vec<ModelFile>> {
    let Some((body, projector)) = loader::pair_files(Role::Recognition) else {
        bail!("the inference crate publishes no recognition pair");
    };
    Ok(vec![
        ModelFile {
            name: body,
            bytes: 1_282_435_552,
            sha256: "3893b8926065bbff3da7586d21d8711a9b4fa4fa8f12cd0cefad58e31b2660b6",
            url: "https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/664b960e358348943aefcdc26b3cf09eff1fb0ac/Qwen3-ASR-1.7B.Q4_K_M.gguf",
            timeout: LARGE_FILE_TIMEOUT,
        },
        ModelFile {
            name: projector,
            bytes: 355_709_344,
            sha256: "46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d",
            url: "https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/resolve/36a678687ba7d07a74ca70ccb0e36902e005fb80/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
            timeout: LARGE_FILE_TIMEOUT,
        },
    ])
}

/// The audio tagger a transcription depends on; the wake word never reads it.
fn tagger() -> ModelFile {
    ModelFile {
        name: CED_MODEL,
        bytes: 43_707_680,
        sha256: "c391ed8697a1b08d7c1a463e4940a5c3a2f670e0544ab0d8ee23b544583602a8",
        url: "https://huggingface.co/mudler/ced-gguf/resolve/b5e9a4aad6438763c8da16079d77563fbed35c65/ced-small-f16.gguf",
        timeout: SMALL_FILE_TIMEOUT,
    }
}

/// The synthesiser's talker pair: the body and its projector, both from the one repository
/// that publishes them under exactly these names. The repository carries no model card, so the
/// digest is the whole provenance — and the projector's role is in its metadata, so the
/// recognition projector can never stand in for it.
fn synthesis_pair() -> Result<Vec<ModelFile>> {
    let Some((body, projector)) = loader::pair_files(Role::Synthesis) else {
        bail!("the inference crate publishes no synthesis pair");
    };
    Ok(vec![
        ModelFile {
            name: body,
            bytes: 1_035_965_280,
            sha256: "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129",
            url: "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/ca27d74bc954b73dadab5b71ca265d87fc861a7c/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
            timeout: LARGE_FILE_TIMEOUT,
        },
        ModelFile {
            name: projector,
            bytes: 446_422_912,
            sha256: "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2",
            url: "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/ca27d74bc954b73dadab5b71ca265d87fc861a7c/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            timeout: LARGE_FILE_TIMEOUT,
        },
    ])
}

/// Names this product wrote into the root in an earlier release and no longer uses.
/// A name dropped from [`model_files`] is moved here, so the file it left behind is
/// removed once; the folder is never scanned to find candidates. The list is the root's,
/// not a purpose's: a retired name is dead whichever engine's fetch removes it.
const RETIRED: &[&str] = &[];

/// The large files need a long request; the 44 MB tagger finishes far inside a
/// shorter one.
const LARGE_FILE_TIMEOUT: Duration = Duration::from_mins(30);
const SMALL_FILE_TIMEOUT: Duration = Duration::from_mins(5);

/// Files the once-per-start check found not to match their pins. The fetch path downloads them
/// again; nothing hashes a load for the rest of the process's life.
static BAD: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

fn is_bad(name: &'static str) -> bool {
    BAD.lock().unwrap_poison().contains(&name)
}

fn mark_bad(name: &'static str) {
    let mut bad = BAD.lock().unwrap_poison();
    if !bad.contains(&name) {
        bad.push(name);
    }
}

fn clear_bad(name: &'static str) {
    BAD.lock().unwrap_poison().retain(|bad| *bad != name);
}

/// Whether a purpose's set is worth loading: every file at its pinned size and none known to be
/// the wrong bytes. The size is a stat; the bytes are [`check_once`]'s business, so a load never
/// hashes. A set needed before the check finishes is judged on its sizes alone.
pub(crate) fn usable(dir: &Path, purpose: Purpose) -> bool {
    model_files(purpose).is_ok_and(|files| {
        files
            .iter()
            .all(|file| size_of(&file_at(dir, file)) == Some(file.bytes) && !is_bad(file.name))
    })
}

/// Hash every model file that is present, once per process start, off the request path. Nothing
/// is fetched: a missing file is the on-demand fetch's business, and a file that does not match
/// its pin is recorded so the next load fetches it again. A set needed before this finishes
/// still loads on its size alone, so the check never delays a request.
pub async fn check_once() {
    let Ok(dir) = root() else { return };
    let started = std::time::Instant::now();
    // `Recognition` is the superset of `Listening`: the pair is hashed once, with the tagger.
    for purpose in [Purpose::Recognition, Purpose::Synthesis] {
        let Ok(files) = model_files(purpose) else {
            continue;
        };
        for file in files {
            let path = file_at(&dir, &file);
            match verify(&path, &file).await {
                // Absent: the first use fetches it, and the fetch verifies as it streams.
                Ok(()) | Err(Why::Missing) => {}
                Err(Why::Mismatch(why)) => {
                    warn!(
                        file = %path.display(),
                        "The model file does not match its pin ({why}); the next load fetches it again"
                    );
                    mark_bad(file.name);
                }
            }
        }
    }
    info!(
        "Speech model integrity checked in {:.1}s",
        started.elapsed().as_secs_f64()
    );
}

/// Make one purpose's set usable under `dir`: a file at its pinned size and not known to be the
/// wrong bytes is kept, and the ones missing, mis-sized or known wrong are fetched.
/// `fetch_starting` is called with `dir` before the first byte of a fetch that really begins, so
/// a caller binds its notice to a real download.
///
/// Nothing here hashes: the digest is [`check_once`]'s once-per-start business, and a download
/// verifies as it streams. A file that cannot be fetched is an error naming its path.
///
/// One fetch per set family runs at a time: the two recognition tiers share the pair, and a
/// second fetch of the same gigabytes is pure waste. A tier that arrives during a fetch waits for
/// it and then finds the files in place.
pub(crate) async fn ensure_in(
    dir: &Path,
    purpose: Purpose,
    client: &reqwest::Client,
    fetch_starting: &mut impl FnMut(&Path),
) -> Result<()> {
    static RECOGNITION_FETCH: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
    static SYNTHESIS_FETCH: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
    let _one_fetch = match purpose {
        Purpose::Listening | Purpose::Recognition => RECOGNITION_FETCH.lock().await,
        Purpose::Synthesis => SYNTHESIS_FETCH.lock().await,
    };
    if !dir.is_dir() {
        bail!(
            "the model directory {} does not exist — the installer creates it",
            dir.display()
        );
    }
    remove_retired(dir);

    let files = model_files(purpose)?;
    for file in &files {
        let path = file_at(dir, file);
        let size_matches = size_of(&path) == Some(file.bytes);
        if size_matches && !is_bad(file.name) {
            continue;
        }
        if size_matches {
            warn!(
                file = %path.display(),
                "The model file does not match its pin, fetching it again"
            );
        }
        fetch_starting(dir);
        download(client, file, &path).await?;
        clear_bad(file.name);
    }
    Ok(())
}

/// Why a file is not usable: absent, or present and not the pinned bytes.
enum Why {
    Missing,
    Mismatch(String),
}

/// Check one file against its pin: the size first (instant), then the digest. Only the
/// once-per-start check calls this; a load never hashes.
async fn verify(path: &Path, file: &ModelFile) -> Result<(), Why> {
    match size_of(path) {
        None => return Err(Why::Missing),
        Some(len) if len != file.bytes => {
            return Err(Why::Mismatch(format!(
                "{len} bytes, the pin says {}",
                file.bytes
            )));
        }
        Some(_) => {}
    }
    let owned = path.to_path_buf();
    let expected = file.sha256;
    match tokio::task::spawn_blocking(move || crate::util::verify_sha256(&owned, expected)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(Why::Mismatch(format!("{e:#}"))),
        Err(_) => Err(Why::Mismatch("it could not be read".to_string())),
    }
}

fn size_of(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .filter(std::fs::Metadata::is_file)
        .map(|meta| meta.len())
}

fn file_at(dir: &Path, file: &ModelFile) -> PathBuf {
    dir.join(file.name)
}

/// Fetch one file: streamed to a sibling temp file, hashed as it arrives, and renamed
/// into place only once it matches its pin, so a reader in the same directory never
/// sees a half-written file.
async fn download(client: &reqwest::Client, file: &ModelFile, path: &Path) -> Result<()> {
    info!(file = %path.display(), "Fetching a speech model file");
    crate::util::http::download_verified(
        client,
        file.url,
        path,
        file.sha256,
        Some(file.timeout),
        crate::util::http::DownloadSizeCheck::None,
        |downloaded, total| {
            if let Some(percent) = downloaded.saturating_mul(100).checked_div(total) {
                debug!(file = %path.display(), "{percent}% of {total} bytes");
            }
        },
    )
    .await
    .with_context(|| format!("fetching {}", path.display()))
}

/// Remove the retired names from the root, and nothing else: a name outside the list is
/// somebody else's, and a file that is not there is not an error.
fn remove_retired(dir: &Path) {
    for name in RETIRED {
        let path = dir.join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => info!(file = %path.display(), "Removed a retired model file"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                warn!(file = %path.display(), error = %e, "Could not remove a retired model file");
            }
        }
    }
}

/// The HTTP client a fetch uses: no client-wide timeout, because each request carries its own
/// bound (a gigabyte body needs a long one), and the ring TLS provider the rest of the product
/// installs.
pub(crate) fn download_client() -> Result<reqwest::Client> {
    crate::util::http::install_ring_provider();
    reqwest::Client::builder()
        .user_agent(concat!(
            "mahbot/",
            env!("CARGO_PKG_VERSION"),
            " (speech model downloader)"
        ))
        .build()
        .context("Failed to create HTTP client for model download")
}

/// Why a set is not usable, in the terms the caller reports: a failed load and a load
/// still in progress are different news for someone who just spoke. Both name the directory the
/// set lives in, so the failure carries the path rather than sending the owner to the log for it.
pub(crate) fn unloaded_message(dir: &Path, purpose: Purpose, failed: bool) -> String {
    let what = purpose.what();
    if failed {
        format!(
            "the local {what} model under {} failed to load (the loader logged why)",
            dir.display()
        )
    } else {
        format!(
            "the local {what} model under {} is still loading",
            dir.display()
        )
    }
}

/// A purpose's set on disk at its pinned sizes, for a test that needs [`usable`] without
/// gigabytes: the files are sparse.
#[cfg(test)]
pub(crate) fn write_sparse_set(dir: &Path, purpose: Purpose) {
    for file in model_files(purpose).expect("a published set") {
        std::fs::File::create(dir.join(file.name))
            .expect("a file")
            .set_len(file.bytes)
            .expect("a file of the pinned size");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temp dir")
    }

    #[test]
    fn the_table_names_what_the_engine_opens() {
        let files = model_files(Purpose::Listening).expect("the recognition pair is published");
        let (body, projector) = loader::pair_files(Role::Recognition).expect("a pair");
        assert_eq!(files[0].name, body);
        assert_eq!(files[1].name, projector);

        let files = model_files(Purpose::Recognition).expect("the recognition set is published");
        assert_eq!(files[0].name, body);
        assert_eq!(files[1].name, projector);
        assert_eq!(files[2].name, CED_MODEL);

        let files = model_files(Purpose::Synthesis).expect("the synthesis pair is published");
        let (body, projector) = loader::pair_files(Role::Synthesis).expect("a pair");
        assert_eq!(files[0].name, body);
        assert_eq!(files[1].name, projector);

        for purpose in [Purpose::Listening, Purpose::Recognition, Purpose::Synthesis] {
            for file in &model_files(purpose).expect("a published set") {
                assert_eq!(file.sha256.len(), 64, "{} has a digest", file.name);
                assert!(file.bytes > 0, "{} has a size", file.name);
                // Pinned by commit: a branch would move under the pin.
                assert!(file.url.starts_with("https://"), "{}", file.url);
                assert!(!file.url.contains("/main/"), "{} is not pinned", file.url);
            }
        }
    }

    #[tokio::test]
    async fn a_missing_root_is_reported_by_its_path() {
        let dir = scratch();
        let missing = dir.path().join("models");
        // The client is only built here; nothing is fetched.
        crate::util::http::install_ring_provider();
        let client = reqwest::Client::new();
        let err = ensure_in(&missing, Purpose::Recognition, &client, &mut |_| {})
            .await
            .expect_err("a missing root is an error");
        assert!(
            err.to_string().contains(&missing.display().to_string()),
            "the failure names the path: {err}"
        );
    }

    #[tokio::test]
    async fn a_file_that_is_not_the_pinned_bytes_is_a_mismatch() {
        let dir = scratch();
        let file = &model_files(Purpose::Recognition).expect("the pair")[0];
        let path = dir.path().join(file.name);
        assert!(matches!(verify(&path, file).await, Err(Why::Missing)));
        std::fs::write(&path, b"not the model").expect("a file");
        match verify(&path, file).await {
            Err(Why::Mismatch(why)) => assert!(why.contains("bytes"), "{why}"),
            _ => panic!("a file of the wrong size is a mismatch"),
        }
        assert!(
            !usable(dir.path(), Purpose::Recognition),
            "an unusable set is not present"
        );
    }
}

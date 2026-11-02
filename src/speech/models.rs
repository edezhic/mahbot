//! The machine-wide model root, and the fixed table of files recognition reads.
//!
//! The root is the inference crate's own ([`mahbot_inference::paths::root`]) — one
//! location per operating system, no environment variable, no home-relative
//! fallback, no search anywhere else. The installer creates it; the product only
//! ever reports its absence, naming the path.
//!
//! Inside it the product knows exactly three file names, taken from the engine's
//! published table rather than restated here. The folder is never enumerated for model
//! files, so anything else in it — the other consumer's corpus folders above all — is
//! never read, written, overwritten or removed; the one other name touched is a part file
//! of the product's own (`<name>.part-<pid>-<seq>`) that a killed fetch left behind.
//!
//! Identity is the digest, never the name: the recognition body sits under this exact name
//! in other repositories with other bytes, so each entry carries its own source and pin.
//! What a file *is* — body, projector or tagger — comes from its own metadata: the engine
//! refuses a pair whose halves do not belong together, so a swap is caught there.

use anyhow::{Context as _, Result, bail};
use mahbot_inference::ced::CED_MODEL;
use mahbot_inference::loader::{self, Role};
use std::path::{Path, PathBuf};
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
    bail!("speech recognition does not run under test")
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

/// The files this product fetches: the recognition body and its projector, plus the
/// audio tagger every recognition call depends on. The synthesis pair belongs to
/// synthesis and is not fetched here.
fn model_files() -> Result<Vec<ModelFile>> {
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
        ModelFile {
            name: CED_MODEL,
            bytes: 43_707_680,
            sha256: "c391ed8697a1b08d7c1a463e4940a5c3a2f670e0544ab0d8ee23b544583602a8",
            url: "https://huggingface.co/mudler/ced-gguf/resolve/b5e9a4aad6438763c8da16079d77563fbed35c65/ced-small-f16.gguf",
            timeout: SMALL_FILE_TIMEOUT,
        },
    ])
}

/// Names this product wrote into the root in an earlier release and no longer uses.
/// A name dropped from [`model_files`] is moved here, so the file it left behind is
/// removed once; the folder is never scanned to find candidates.
const RETIRED: &[&str] = &[];

/// The two large files need a long request; the 44 MB tagger finishes far inside a
/// shorter one.
const LARGE_FILE_TIMEOUT: Duration = Duration::from_mins(30);
const SMALL_FILE_TIMEOUT: Duration = Duration::from_mins(5);

/// Whether the whole set is on disk at its pinned size. Cheap (three stats) and no
/// digest, so it answers "is a load worth attempting" rather than "are the bytes
/// right" — [`ensure_in`] is what answers the second.
pub(crate) fn present(dir: &Path) -> bool {
    model_files().is_ok_and(|files| {
        files
            .iter()
            .all(|file| size_of(&file_at(dir, file)) == Some(file.bytes))
    })
}

/// Verify every file against its pin and fetch nothing: the check the awaited path makes,
/// where a wrong file is a failure naming it rather than a download in the caller's way.
pub(crate) async fn verify_set(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        bail!(
            "the model directory {} does not exist — the installer creates it",
            dir.display()
        );
    }
    for file in &model_files()? {
        let path = file_at(dir, file);
        match verify(&path, file).await {
            Ok(()) => {}
            Err(Why::Missing) => bail!(
                "{} is not there — the product fetches the model set into the directory the \
                 installer creates",
                path.display()
            ),
            Err(Why::Mismatch(why)) => {
                bail!("{} does not match its pin ({why})", path.display());
            }
        }
    }
    Ok(())
}

/// Make the set usable under `dir`: every file checked against its pin, and the ones
/// missing or not matching fetched. `fetch_starting` is called with `dir` before the first
/// byte of a fetch that really begins, so a caller binds its notice to a real download.
///
/// A file that does not match its pin and cannot be fetched again is an error naming
/// its path — the engine verifies nothing itself, so this is the only place a wrong
/// file is caught.
pub(crate) async fn ensure_in(
    dir: &Path,
    client: &reqwest::Client,
    fetch_starting: &mut impl FnMut(&Path),
) -> Result<()> {
    if !dir.is_dir() {
        bail!(
            "the model directory {} does not exist — the installer creates it",
            dir.display()
        );
    }
    remove_retired(dir);

    let files = model_files()?;
    for file in &files {
        let path = file_at(dir, file);
        match verify(&path, file).await {
            Ok(()) => continue,
            Err(Why::Missing) => {}
            Err(Why::Mismatch(why)) => warn!(
                file = %path.display(),
                "The model file does not match its pin ({why}), fetching it again"
            ),
        }
        fetch_starting(dir);
        download(client, file, &path).await?;
    }
    Ok(())
}

/// Why a file is not usable: absent, or present and not the pinned bytes.
enum Why {
    Missing,
    Mismatch(String),
}

/// Check one file against its pin: the size first (instant), then the digest.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temp dir")
    }

    #[test]
    fn the_table_names_what_the_engine_opens() {
        let files = model_files().expect("the recognition pair is published");
        let (body, projector) = loader::pair_files(Role::Recognition).expect("a pair");
        assert_eq!(files[0].name, body);
        assert_eq!(files[1].name, projector);
        assert_eq!(files[2].name, CED_MODEL);
        for file in &files {
            assert_eq!(file.sha256.len(), 64, "{} has a digest", file.name);
            assert!(file.bytes > 0, "{} has a size", file.name);
            // Pinned by commit: a branch would move under the pin.
            assert!(file.url.starts_with("https://"), "{}", file.url);
            assert!(!file.url.contains("/main/"), "{} is not pinned", file.url);
        }
    }

    #[tokio::test]
    async fn a_missing_root_is_reported_by_its_path() {
        let dir = scratch();
        let missing = dir.path().join("models");
        // The client is only built here; nothing is fetched.
        crate::util::http::install_ring_provider();
        let client = reqwest::Client::new();
        let err = ensure_in(&missing, &client, &mut |_| {})
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
        let file = &model_files().expect("the pair")[0];
        let path = dir.path().join(file.name);
        assert!(matches!(verify(&path, file).await, Err(Why::Missing)));
        std::fs::write(&path, b"not the model").expect("a file");
        match verify(&path, file).await {
            Err(Why::Mismatch(why)) => assert!(why.contains("bytes"), "{why}"),
            _ => panic!("a file of the wrong size is a mismatch"),
        }
        assert!(!present(dir.path()), "an unusable set is not present");
    }
}

//! The document kit runner — the one place the embedded JavaScript document kit
//! is materialized and invoked.
//!
//! The kit (`assets/docgen/document-kit.js`, built from `assets/docgen/kit.js`
//! and the shared `assets/docgen/rules.json`) is a self-contained bundle that
//! runs on the managed bun runtime. It is not a prompt asset: it is megabytes
//! of raw bytes, and the prompt folder pins LF endings and is not for binaries
//! — so the bundle and the PDF text font it embeds
//! (`assets/docgen/NotoSans-Regular.ttf`) are compiled in with `include_bytes!`
//! and written out at run time.
//!
//! # Protocol
//!
//! A request is a JSON object naming the operation (`op`) and the paths that
//! operation reads and writes. The runner always writes that object to a
//! `request-<nonce>.json` file, always points the kit at a `result-<nonce>.json`
//! file, and always reads the RESULT FILE — never stdout, which is only the
//! run's diagnostics. Both files live in the kit's `scratch/` directory, which
//! is emptied when the kit is prepared. A result is `{"ok": bool, "outputs":
//! [...], "missing": [...], "placeholders": N, "unsupported": [...], "class":
//! "...", "error": "..."}`, where `placeholders` is a `fill_template` result
//! only, `unsupported` lists the distinct characters the embedded PDF font could
//! not draw for a PDF-writing operation, and `class` is present only on a
//! failure the kit blamed on the request ("usage": the caller can fix it, see
//! [`read_result`]). A failure of the run itself (a timeout, a kill, a spawn
//! failure) never reaches the result file and is reported by the runner, with
//! its own class token.
//!
//! # Containment
//!
//! The materialized kit directory is a subdirectory of the product's storage
//! root. In the product's own layout (`$HOME` outside the temp roots) that is a
//! place no workspace, no tool-accessible root and no OS temp root an agent may
//! write covers — unlike the temp-root path it used to occupy. A sandbox launch
//! that points `HOME` at a temp root puts the storage root there too, where a
//! read-only agent may write, and the guarantee is only as good as the layout.
//!
//! The kit runs with that directory as its cwd, never the caller's workspace:
//! bun auto-loads `$cwd/bunfig.toml` and runs its `preload` scripts (verified —
//! `--config` does not suppress it), so a workspace holding a `bunfig.toml`
//! would run attacker-authored code as the OS owner. The invocation also passes
//! `--no-install` and `--no-env-file`, so neither an install nor a stray env
//! file can run either.

use anyhow::Result;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The committed, self-contained kit bundle. Deliberately raw bytes rather than
/// a prompt asset (see the module docs).
const KIT_SCRIPT: &[u8] = include_bytes!("../assets/docgen/document-kit.js");

/// The Cyrillic-covering PDF text font the kit embeds into the PDFs it writes
/// or annotates (see `assets/docgen/NotoSans-LICENSE.txt`).
const KIT_PDF_FONT: &[u8] = include_bytes!("../assets/docgen/NotoSans-Regular.ttf");

/// The materialized-kit directory name under the product's storage root. The
/// per-content subdirectory keeps a new binary from ever reusing an older
/// kit's files.
const KIT_DIR_NAME: &str = "mahbot-document-kit";
const KIT_SCRIPT_NAME: &str = "document-kit.js";
const KIT_FONT_NAME: &str = "NotoSans-Regular.ttf";

/// The kit's own scratch directory, inside the version directory: a run's
/// request and result files and the probe's tree. It is emptied when the kit is
/// prepared, so the leftovers of a killed run do not accumulate in a storage
/// root nothing else sweeps.
const KIT_SCRATCH_NAME: &str = "scratch";

/// Bytes of the content hash that name the per-version directory.
const KIT_HASH_BYTES: usize = 8;

/// Characters of a failed run's output carried into the error when the kit
/// itself said nothing.
const OUTPUT_TAIL_CHARS: usize = 2000;

/// What one kit operation produced.
pub(crate) struct KitOutcome {
    /// Absolute paths of the files the kit wrote, in the order it wrote them.
    pub outputs: Vec<PathBuf>,
    /// Placeholder names the request supplied no value for (a `fill_template`
    /// sample's `{name}` fields), left as they were in the produced file.
    pub missing: Vec<String>,
    /// The distinct `{name}` template tags the filler compiled or substituted,
    /// counted once per name — a `fill_template` result only, and only over the
    /// parts the filler rewrites (docx/pptx: the rendered parts plus the notes
    /// slides; xlsx: the worksheets plus the shared-string table). It is not a
    /// count of substitution sites, and a tag in a part outside that set — a
    /// chart label, a comment — is not in it at all. `Some(0)` means the parts
    /// it rewrites held none, which is the one fact a filled copy cannot state
    /// about itself. `None` for every other operation, none of which reports a
    /// count.
    pub placeholders: Option<usize>,
    /// The distinct characters the embedded PDF font could not draw, in the
    /// order the kit first drew them — the PDF-writing operations only, and
    /// empty for every operation that draws no text. Each entry is one
    /// character, missing from the produced file rather than present in it.
    pub unsupported: Vec<String>,
}

/// Run one kit operation and report what it produced.
///
/// `request` must be a JSON object naming `op`. The runner adds the result path
/// and the materialized font path itself, so a caller never names either.
pub(crate) async fn run(request: Value) -> Result<KitOutcome> {
    let value = invoke(request).await?;
    Ok(KitOutcome {
        outputs: string_array(&value, "outputs")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
        missing: string_array(&value, "missing"),
        placeholders: value
            .get("placeholders")
            .and_then(Value::as_u64)
            .and_then(|count| usize::try_from(count).ok()),
        unsupported: string_array(&value, "unsupported"),
    })
}

/// Run the kit's own `probe` operation once per process, into a scratch
/// directory removed afterwards, and cache SUCCESS only.
///
/// A failure is retried on the next call because a kit that is broken now may
/// become usable: the product replaces the managed runtime with the newest
/// release on every start, so a permanent refusal would outlive its cause.
pub(crate) async fn probe() -> Result<()> {
    static PROBED: OnceLock<()> = OnceLock::new();
    if PROBED.get().is_some() {
        return Ok(());
    }
    // Refused before the kit is materialized: writing megabytes out only to
    // report an unavailable runtime is work for nothing.
    managed_runtime()?;
    let dir = kit_directory().await?;
    let scratch = dir
        .join(KIT_SCRATCH_NAME)
        .join(format!("probe-{:016x}", rand::random::<u64>()));
    let result = invoke(json!({ "op": "probe", "scratch": scratch })).await;
    // The scratch tree is the kit's own artifact, never something the tool
    // reserved, so it is removed here whatever the probe reported.
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result?;
    let _ = PROBED.set(());
    Ok(())
}

/// One kit operation end to end: fill in the result path, write the request,
/// run the kit, read and parse the result, then remove both files (best effort,
/// the error path included). Shared by [`run`] and [`probe`], which interpret
/// the parsed result differently.
async fn invoke(mut request: Value) -> Result<Value> {
    // Resolved before anything is materialized: the kit is megabytes, and
    // writing it out only to refuse the call is work for nothing.
    let runtime = managed_runtime()?;
    let op = request
        .get("op")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let dir = kit_directory().await?;
    let nonce = format!("{:016x}", rand::random::<u64>());
    let request_path = dir
        .join(KIT_SCRATCH_NAME)
        .join(format!("request-{nonce}.json"));
    let result_path = dir
        .join(KIT_SCRATCH_NAME)
        .join(format!("result-{nonce}.json"));

    // The runner owns both paths, so a caller can never point the kit at a file
    // of its own choosing; the font is materialized beside the kit.
    request["result"] = Value::String(result_path.to_string_lossy().into_owned());
    request["font"] = Value::String(dir.join(KIT_FONT_NAME).to_string_lossy().into_owned());

    let outcome = async {
        let body = serde_json::to_vec(&request).map_err(|e| {
            crate::tools::internal_fault(&format!("failed to encode the document kit request: {e}"))
        })?;
        tokio::fs::write(&request_path, body).await.map_err(|e| {
            crate::tools::internal_fault(&format!(
                "failed to write the document kit request for `{op}`: {e}"
            ))
        })?;
        let output = run_kit(dir, &runtime, &request_path).await?;
        read_result(&result_path, &output, &op).await
    }
    .await;
    let _ = tokio::fs::remove_file(&request_path).await;
    let _ = tokio::fs::remove_file(&result_path).await;
    outcome
}

/// The managed script runtime, or the mahbot-side fault that says why the
/// document kit cannot run: nothing here is the request's fault, so it carries
/// the `internal` token the model needs to tell it from a call it can fix. The
/// path is absent both when the runtime is not installed and while the product
/// replaces it at startup, so the message says it is unavailable rather than
/// claiming an install state this side cannot know.
fn managed_runtime() -> Result<PathBuf> {
    crate::tools::bun::bun_binary_path().ok_or_else(|| {
        crate::tools::internal_fault(
            "the document kit cannot run: the managed bun runtime is unavailable",
        )
    })
}

/// Run the kit on the managed runtime with the kit directory as cwd: the
/// materialized script, then the request file as its argument.
///
/// A failure of the RUN itself — one that timed out, was killed or would not
/// start — is the product's own, and the runner reports it with its own class
/// token ("timeout:", "io:", "terminated:"), as it does for the custom tool:
/// the model reads one class and a subject, not two classes for one failure.
async fn run_kit(dir: &Path, runtime: &Path, request_path: &Path) -> Result<String> {
    let args = vec![
        "--no-install".to_string(),
        "--no-env-file".to_string(),
        dir.join(KIT_SCRIPT_NAME).to_string_lossy().into_owned(),
        request_path.to_string_lossy().into_owned(),
    ];
    // The kit is the product's own code and needs none of the owner's
    // environment, which is what this entry point is for.
    crate::tools::shell::run_internal_program_with_timeout(dir, runtime, &args, "the document kit")
        .await
}

/// Read and parse the kit's result file.
///
/// A missing or unreadable result file, or an `ok: false` the kit did NOT blame
/// on the request, is a mahbot-side fault ([`crate::tools::internal_fault`]): the
/// kit's own code failed on data it is supposed to handle. A failure the kit DID
/// blame on the request (`"class": "usage"` — a page the document does not have,
/// a field it does not declare) carries the `usage` token instead, because the
/// model can fix that by changing the call.
async fn read_result(path: &Path, output: &str, op: &str) -> Result<Value> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return Err(crate::tools::internal_fault(&format!(
                "the document kit gave no result for `{op}` ({}){}",
                e.kind(),
                output_tail(output)
            )));
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(e) => {
            return Err(crate::tools::internal_fault(&format!(
                "the document kit returned an unreadable result for `{op}`: {e}{}",
                output_tail(output)
            )));
        }
    };
    if value.get("ok").and_then(Value::as_bool) == Some(true) {
        return Ok(value);
    }
    let message = value
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let reason = if message.is_empty() {
        format!("no reason given{}", output_tail(output))
    } else {
        message.to_string()
    };
    if value.get("class").and_then(Value::as_str) == Some("usage") {
        anyhow::bail!("usage: {reason}");
    }
    Err(crate::tools::internal_fault(&format!(
        "the document kit could not complete `{op}`: {reason}"
    )))
}

/// The materialized-kit directory for THIS build's kit bytes, prepared on every
/// call so a kit file that is missing or truncated is restored before the call
/// runs. The path itself is cached; the length-checked writes are not, because
/// the files may have gone with the directory (and the first write moves
/// megabytes onto the disk, so it is done through async I/O rather than on a
/// worker thread that could be running other work).
async fn kit_directory() -> Result<&'static Path> {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    // Once per process, and before this call's own request is written: the sweep
    // is what a killed run cannot do for itself, and `OnceCell` is what makes
    // the ordering hold — every caller waits for it, so none of them can have a
    // file inside the directory it clears.
    static SWEPT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    // The kit must not be materialized where an agent can write: a writable
    // `bunfig.toml` beside it would run attacker-authored code (see the module
    // docs). Boot resolves the storage root before any agent runs, so its
    // absence here is a boot-order bug, not the caller's fault.
    let root = crate::config::CONFIG.try_storage_root().ok_or_else(|| {
        crate::tools::internal_fault(
            "the document kit cannot be prepared: the storage root is not resolved",
        )
    })?;
    let dir = DIR.get_or_init(|| root.join(KIT_DIR_NAME).join(content_version()));
    materialize(dir).await.map_err(|e| {
        crate::tools::internal_fault(&format!(
            "failed to prepare the document kit at {}: {e}",
            dir.display()
        ))
    })?;
    SWEPT
        .get_or_init(|| async { sweep_kit_directory(dir).await })
        .await;
    Ok(dir)
}

/// The directory name for the embedded kit bytes: a new binary that changes the
/// kit (or the font) gets a new directory, so a stale kit is never reused.
fn content_version() -> String {
    let mut hasher = Sha256::new();
    hasher.update(KIT_SCRIPT);
    hasher.update(KIT_PDF_FONT);
    let hex = crate::util::hex_string(&hasher.finalize());
    hex[..KIT_HASH_BYTES * 2].to_string()
}

/// Create the kit directory and put both files in place.
async fn materialize(dir: &Path) -> std::io::Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    write_if_missing(&dir.join(KIT_SCRIPT_NAME), KIT_SCRIPT).await?;
    write_if_missing(&dir.join(KIT_FONT_NAME), KIT_PDF_FONT).await
}

/// Drop what the kit directory should not keep: the scratch of a run that was
/// killed, and the directories other content versions left behind (each is
/// megabytes of bundle and font, and nothing else sweeps the storage root).
///
/// The instance flock makes this the only process under the storage root, so no
/// sibling can be a version another process is running. Best effort — whatever
/// will not go is left for the next process.
async fn sweep_kit_directory(dir: &Path) {
    let scratch = dir.join(KIT_SCRATCH_NAME);
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    let _ = tokio::fs::create_dir_all(&scratch).await;
    let Some(parent) = dir.parent() else {
        return;
    };
    let Ok(mut entries) = tokio::fs::read_dir(parent).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.path() != dir {
            let _ = tokio::fs::remove_dir_all(entry.path()).await;
        }
    }
}

/// Write `bytes` to `path` unless a file of exactly that length is already
/// there. A replacement goes through a unique sibling name + rename, so
/// concurrent callers cannot race on a shared temporary and no reader ever sees
/// a half-written file; a file of the wrong length is a leftover from a partial
/// write (or a truncated copy) and is replaced.
async fn write_if_missing(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if tokio::fs::metadata(path)
        .await
        .is_ok_and(|m| m.len() == bytes.len() as u64)
    {
        return Ok(());
    }
    let tmp = path.with_extension(format!("{:016x}.part", rand::random::<u64>()));
    tokio::fs::write(&tmp, bytes).await?;
    // A rename replaces an existing destination on every platform this builds
    // for (Windows included), so it is not removed first: that would open a
    // window in which a concurrently spawning sibling finds no kit at all.
    match tokio::fs::rename(&tmp, path).await {
        Ok(()) => Ok(()),
        // Another caller can have won the race with byte-identical content; the
        // length check is what says its copy is as good as ours.
        Err(_)
            if tokio::fs::metadata(path)
                .await
                .is_ok_and(|m| m.len() == bytes.len() as u64) =>
        {
            let _ = tokio::fs::remove_file(&tmp).await;
            Ok(())
        }
        Err(e) => {
            let _ = tokio::fs::remove_file(&tmp).await;
            Err(e)
        }
    }
}

/// The output tail carried into an error when the kit gave no message of its
/// own; empty when the run said nothing.
fn output_tail(output: &str) -> String {
    let tail = output.trim();
    if tail.is_empty() {
        String::new()
    } else {
        format!("\n{}", crate::util::truncate(tail, OUTPUT_TAIL_CHARS))
    }
}

/// The string members of a JSON array field, or an empty list when it is
/// absent or not an array.
fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sources the committed bundle is built from, in the order
    /// `assets/docgen/build.sh` hashes them.
    const KIT_SOURCES: [&[u8]; 4] = [
        include_bytes!("../assets/docgen/kit.js"),
        include_bytes!("../assets/docgen/rules.json"),
        include_bytes!("../assets/docgen/package.json"),
        include_bytes!("../assets/docgen/bun.lock"),
    ];

    /// How much of the bundle's head carries its build provenance.
    const HEADER_BYTES: usize = 512;

    /// The class a kit failure carries is the whole contract with the model: a
    /// failure the kit blamed on the request (`"class": "usage"`) is the caller's
    /// to fix, and everything else is the product's — never the other way round.
    #[tokio::test]
    async fn a_kit_failure_carries_the_class_the_kit_reported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = dir.path().join("result.json");

        let usage =
            br#"{"ok":false,"error":"no page 5: this document has 1 page(s)","class":"usage"}"#;
        tokio::fs::write(&result, usage).await.expect("write");
        let error = read_result(&result, "", "pdf_rotate")
            .await
            .expect_err("a failed result");
        assert_eq!(
            error.to_string(),
            "usage: no page 5: this document has 1 page(s)"
        );

        // The same failure with no class of its own is the product's: no change
        // to the call can recover it.
        tokio::fs::write(&result, br#"{"ok":false,"error":"the kit broke"}"#)
            .await
            .expect("write");
        let error = read_result(&result, "", "create")
            .await
            .expect_err("a failed result");
        let message = error.to_string();
        assert!(message.starts_with("internal: "), "{message}");
        assert!(message.contains("the kit broke"), "{message}");

        // So is a run that wrote no result at all.
        tokio::fs::remove_file(&result).await.expect("remove");
        let error = read_result(&result, "the run said nothing", "create")
            .await
            .expect_err("a failed result");
        assert!(error.to_string().starts_with("internal: "), "{error}");
    }

    /// The kit file is written once per content version and then left alone: only
    /// a file of the wrong LENGTH is a leftover to replace, and no temporary
    /// sibling survives either outcome.
    #[tokio::test]
    async fn write_if_missing_keeps_a_complete_copy_and_replaces_a_partial_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kit.js");

        write_if_missing(&path, b"first").await.expect("write");
        write_if_missing(&path, b"other").await.expect("write");
        assert_eq!(
            tokio::fs::read(&path).await.expect("read"),
            b"first",
            "a file of the expected length is this build's own copy"
        );

        write_if_missing(&path, b"a much longer body")
            .await
            .expect("write");
        assert_eq!(
            tokio::fs::read(&path).await.expect("read"),
            b"a much longer body"
        );

        let mut entries = tokio::fs::read_dir(dir.path()).await.expect("read dir");
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.expect("entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(names, ["kit.js"], "no temporary sibling is left behind");
    }

    /// Only the version in use survives, and its scratch starts empty: an older
    /// version's directory is megabytes nothing else sweeps away, and a killed
    /// run's request and result files would otherwise stay in the storage root
    /// for good.
    #[tokio::test]
    async fn the_kit_directory_is_swept_once() {
        let root = tempfile::tempdir().expect("tempdir");
        let parent = root.path().join(KIT_DIR_NAME);
        let current = parent.join("current");
        let stale = parent.join("stale");
        let scratch = current.join(KIT_SCRATCH_NAME);
        tokio::fs::create_dir_all(&stale).await.expect("create");
        tokio::fs::write(stale.join("document-kit.js"), b"old")
            .await
            .expect("write");
        tokio::fs::create_dir_all(scratch.join("probe-dead"))
            .await
            .expect("create");
        tokio::fs::write(scratch.join("request-dead.json"), b"{}")
            .await
            .expect("write");

        sweep_kit_directory(&current).await;

        assert!(current.exists(), "the version in use must survive");
        assert!(!stale.exists(), "an older version must be dropped");
        let mut names = Vec::new();
        let mut entries = tokio::fs::read_dir(&scratch).await.expect("read scratch");
        while let Some(entry) = entries.next_entry().await.expect("entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert!(
            names.is_empty(),
            "the scratch of a killed run must be dropped, found: {names:?}"
        );
    }

    /// An edited `kit.js` without a rebuild changes nothing and would otherwise
    /// fail nothing, leaving the embedded bytes and their source out of step.
    #[test]
    fn the_committed_bundle_is_the_build_of_the_committed_sources() {
        let mut hasher = Sha256::new();
        for source in KIT_SOURCES {
            hasher.update(source);
        }
        let expected = crate::util::hex_string(&hasher.finalize());
        let header = String::from_utf8_lossy(&KIT_SCRIPT[..HEADER_BYTES.min(KIT_SCRIPT.len())]);
        assert!(
            header.contains(&expected),
            "assets/docgen/document-kit.js was not built from the committed kit.js/rules.json/\
             package.json/bun.lock — run assets/docgen/build.sh and commit the bundle"
        );
    }
}

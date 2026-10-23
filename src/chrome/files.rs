//! Local files at the edge of the browser surfaces: the paths a caller hands a
//! page (`upload`) and the picture a capture writes back (`screenshot`).
//!
//! Path rules are the surface's own (the command line resolves against the
//! directory it was run in, the agent tool against its workspace), so the ONE
//! resolution here takes that root as an operand. There is deliberately no access
//! level and no allowlist: any file the caller can read may be attached. The one
//! refusal beyond ordinary file correctness is the product's own store rule — its
//! live database files are never opened, by anything.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use crate::chrome::contract::{ChromeResponse, eval_result};

/// Resolve one local file a caller named for an attach, against the root its
/// surface resolves relative paths from. `~` is expanded; the answer is the
/// absolute, symlink-resolved path the browser is handed — and the one the
/// result reports, so a same-named file elsewhere can never be a silent
/// substitute. A refusal names the reason: missing, not a regular file, or
/// unreadable. The product's own live stores are refused by location, spelled
/// directly or through a link.
pub(crate) fn resolve_local_file(raw: &str, base: &Path) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("empty file path".to_string());
    }
    let expanded = crate::util::expand_tilde(raw);
    let candidate = if expanded.is_absolute() {
        expanded
    } else {
        base.join(expanded)
    };
    let display = candidate.display();
    // Resolve first, so a symlinked spelling of a store file meets the same
    // refusal — `realpath` holds no descriptor, so it cannot touch the store's lock.
    let resolved = std::fs::canonicalize(&candidate)
        .map(|path| crate::util::strip_verbatim_prefix(&path))
        .map_err(|e| format!("{}: {display}", reason(&e)))?;
    let display = resolved.display();
    // The product's own live stores are refused by location: the service holds
    // whole-file locks on them, and opening one from inside it releases those locks.
    if let Some(denial) = crate::tools::path::live_store_denial(&resolved, "attach") {
        return Err(denial.to_string());
    }
    match std::fs::metadata(&resolved) {
        Ok(meta) if !meta.is_file() => return Err(format!("not a regular file: {display}")),
        Ok(_) => {}
        Err(e) => return Err(format!("{}: {display}", reason(&e))),
    }
    // Metadata can succeed where reading does not (a file with no read
    // permission), and that is the caller's own operand to fix rather than a
    // failure for chrome-use to discover mid-stream.
    std::fs::File::open(&resolved).map_err(|e| format!("{}: {display}", reason(&e)))?;
    Ok(resolved)
}

/// The plain words for an unreadable-path reason, or the OS's own message.
fn reason(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "no such file".to_string(),
        std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        _ => e.to_string(),
    }
}

/// Validate a capture is a real PNG: magic bytes + IHDR dimensions (bytes
/// 16..24, big-endian). chrome-use can report success yet write a
/// corrupt/truncated file, and the tool's image payload fails open on
/// undecodable rasters — so this gates the `[IMAGE:…]` marker (and the CLI's
/// reported picture) instead of passing a phantom image on.
pub(crate) fn validate_png(path: &Path) -> anyhow::Result<(u32, u32)> {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    // Only the header is needed — magic bytes + IHDR dimensions (bytes 16..24).
    let mut header = [0u8; 24];
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to read screenshot {}", path.display()))?;
    std::io::Read::read_exact(&mut file, &mut header).map_err(|_| {
        anyhow::anyhow!(
            "Screenshot at {} is not a valid PNG (truncated or wrong format)",
            path.display()
        )
    })?;
    if header[..8] != PNG_MAGIC {
        anyhow::bail!(
            "Screenshot at {} is not a valid PNG (wrong format)",
            path.display()
        );
    }
    let width = u32::from_be_bytes(header[16..20].try_into().expect("slice is 4 bytes"));
    let height = u32::from_be_bytes(header[20..24].try_into().expect("slice is 4 bytes"));
    if width == 0 || height == 0 {
        anyhow::bail!(
            "Screenshot at {} has an empty IHDR ({}x{})",
            path.display(),
            width,
            height
        );
    }
    Ok((width, height))
}

/// The counts chrome-use's `upload` reply carries: the files streamed into the
/// page (`uploaded`) and the ones still on its input afterwards (`attached`).
/// `None` means the browser side did not state it — a fact of its own, never a
/// zero.
pub(crate) fn upload_counts(resp: &ChromeResponse) -> (Option<u64>, Option<u64>) {
    let count = |key: &str| {
        resp.data
            .as_ref()
            .and_then(|data| data.get(key))
            .and_then(Value::as_u64)
    };
    (count("uploaded"), count("attached"))
}

/// JS answering `true` when the files a caller names would land in a file input
/// that takes one at a time (`input[type=file]` without `multiple`), `false` for
/// a multi-file input, and `null` when the target cannot be judged from the page.
///
/// The target shapes chrome-use resolves are resolved here too: the input itself,
/// the label that pairs with it, the container that holds it (the element and up
/// to eight ancestors, open shadow roots included). A ref (`@e1`) is chrome-use's
/// own handle and invisible to the page, so it is judged only when the page
/// offers exactly one file input; anything ambiguous answers `null` and never
/// refuses the attach.
#[must_use]
pub(crate) fn single_file_input_probe(selector: &str) -> String {
    format!(
        "(() => {{ \
           const found = new Set(); \
           const add = (el) => {{ if (el && el.tagName === 'INPUT' && el.type === 'file') found.add(el); }}; \
           const subtree = (el) => {{ \
             if (!el) return; \
             add(el); \
             if (el.control) add(el.control); \
             for (const root of [el, el.shadowRoot]) {{ \
               if (root && root.querySelectorAll) root.querySelectorAll('input[type=file]').forEach(add); \
             }} \
           }}; \
           let el = null; \
           const sel = '{}'; \
           try {{ el = document.querySelector(sel); }} catch (e) {{ el = null; }} \
           if (el) {{ \
             subtree(el); \
             for (let up = el.parentElement, i = 0; up && i < 8 && found.size === 0; up = up.parentElement, i++) subtree(up); \
           }} else if (sel.startsWith('@')) {{ \
             document.querySelectorAll('input[type=file]').forEach(add); \
           }} \
           const only = [...found]; \
           return only.length === 1 ? !only[0].multiple : null; \
         }})()",
        crate::chrome::escape_js_single_quoted(selector)
    )
}

/// `Some(true)` when [`single_file_input_probe`] answered "a one-file input";
/// `None` when the answer could not be read (a failed step, an ambiguous target,
/// a page that no longer answers) — an unreadable answer never refuses the attach.
pub(crate) fn single_file_input(resp: &ChromeResponse) -> Option<bool> {
    eval_result(resp)?.as_bool()
}

/// The one-file question a several-file attach must ask before sending: the probe
/// script, or `None` when there is nothing to ask — one file makes no list a real
/// selection could not have produced.
#[must_use]
pub(crate) fn one_file_probe(selector: &str, files: usize) -> Option<String> {
    (files > 1).then(|| single_file_input_probe(selector))
}

/// What the one-file question answered about a several-file attach.
pub(crate) enum OneFileTarget {
    /// Nothing stands in the way: the target takes several files, or there was
    /// nothing to ask (one file, or no file at all).
    Proceed,
    /// The target's shape could not be judged: send the files and say the check
    /// was skipped.
    Unjudged,
    /// The target is a one-file input: the refusal to print, nothing attached.
    Refused(String),
}

/// Judge a several-file attach from [`single_file_input`]'s answer — the one rule
/// both browser surfaces follow.
#[must_use]
pub(crate) fn judge_one_file_target(selector: &str, probe: Option<bool>) -> OneFileTarget {
    match probe {
        Some(true) => OneFileTarget::Refused(format!(
            "the file input at {selector} takes one file at a time — attach one file, or name a \
             target that accepts several"
        )),
        Some(false) => OneFileTarget::Proceed,
        None => OneFileTarget::Unjudged,
    }
}

/// What a several-file attach says when the target's shape could not be judged: the
/// files were sent, but the one-file check was skipped. The shapes the probe cannot
/// judge: a ref on a page with more than one file input, an input held by an
/// ancestor more than eight levels above the named element, or one behind a closed
/// shadow root.
#[must_use]
pub(crate) fn unchecked_several_files_note() -> &'static str {
    "the target's file-input shape could not be judged from the page, so several files were sent \
     without the one-file check"
}

/// The chrome-use `upload` argv for an attach whose paths are resolved: the verb,
/// the target, and the absolute files it is handed.
#[must_use]
pub(crate) fn upload_argv(selector: &str, files: &[PathBuf]) -> Vec<String> {
    let mut argv = vec!["upload".to_string(), selector.to_string()];
    argv.extend(files.iter().map(|p| p.to_string_lossy().into_owned()));
    argv
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(name: &str) -> PathBuf {
        let dir = crate::util::test::test_root().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resolve_local_file_names_the_reason() {
        let dir = fixture("chrome-files-resolve");
        std::fs::write(dir.join("a.txt"), b"x").unwrap();

        // Relative paths resolve against the surface's own root, and the answer is
        // the absolute file — the one the result reports.
        let resolved = resolve_local_file("a.txt", &dir).expect("a real file resolves");
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("a.txt"));

        let missing = resolve_local_file("absent.txt", &dir).expect_err("missing is refused");
        assert!(missing.contains("no such file"), "{missing}");
        assert!(missing.contains("absent.txt"), "{missing}");

        let directory =
            resolve_local_file(dir.to_str().unwrap(), &dir).expect_err("a dir is not a file");
        assert!(directory.contains("not a regular file"), "{directory}");
    }

    #[test]
    fn validate_png_accepts_real_png() {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let png_path = fixture("chrome-files-valid").join("valid.png");
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&png_path, &buf).unwrap();

        assert_eq!(validate_png(&png_path).unwrap(), (2, 2));
    }

    #[test]
    fn validate_png_rejects_corrupt_file() {
        let png_path = fixture("chrome-files-corrupt").join("corrupt.png");
        std::fs::write(&png_path, b"not a png").unwrap();

        let err = validate_png(&png_path).unwrap_err().to_string();
        assert!(err.contains("not a valid PNG"), "err: {err}");
    }

    #[test]
    fn single_file_input_reads_only_a_boolean_answer() {
        let answer = |result: Value| {
            ChromeResponse::from_value(&json!({
                "success": true,
                "data": { "origin": "https://x", "result": result }
            }))
        };
        assert_eq!(single_file_input(&answer(json!(true))), Some(true));
        assert_eq!(single_file_input(&answer(json!(false))), Some(false));
        assert_eq!(single_file_input(&answer(Value::Null)), None);
    }

    #[test]
    fn the_probe_quotes_the_selector_into_its_script() {
        assert!(single_file_input_probe("a'b").contains("a\\'b"));
    }
}

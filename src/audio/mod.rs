//! Audio: the microphone, the voice command pipeline, wake-word detection and the machine's audio
//! output.
//!
//! Capture goes through the platform's own host API (`cpal`: CoreAudio, WASAPI, ALSA), so the
//! microphone and the pipeline that turns a recording into a composer message are on every
//! platform. Recognition and spoken replies are not part of this subsystem: both live in
//! [`crate::speech`].
//!
//! The wake word ([`wake_word`]) is on every platform too: it feeds VAD-gated windows to the
//! inference crate's listener and acts on the decision, and its recording cues ([`cue`]) play
//! through the shared output device.

pub(crate) mod cue;
pub(crate) mod output;
pub(crate) mod util;
pub mod voice;
pub(crate) mod wake_word;

/// The audio library's soname on Linux — the one the shim (`vendor/alsa-sys-dlopen`) resolves
/// every `snd_*` call through. The literal exists once per side because the shim is a separate
/// crate the published package replaces with the registry one; [`the_probe_and_the_shim_name_one_library`]
/// keeps them from drifting.
#[cfg(target_os = "linux")]
const ALSA_SONAME: &[u8] = b"libasound.so.2\0";

/// Whether the platform's audio library is loadable on this host.
///
/// On Linux the library is looked up on first use, never at start — the product must run on a
/// server that has no audio stack at all — and a host without it reports voice unavailable. Every
/// other platform's audio stack is part of the system. This is the one answer the product gives:
/// the ALSA shim's wrappers resolve the same soname and report nothing of their own, so a host
/// that passes this probe cannot have every call fall back.
#[must_use]
pub fn library_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::sync::OnceLock;
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // Nothing calls `dlclose`, so the handle stays open for the life of the process — the
            // shim's wrappers need the library loaded, and one reference is the whole cost.
            let handle = unsafe {
                libc::dlopen(
                    ALSA_SONAME.as_ptr().cast(),
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };
            !handle.is_null()
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// The one sentence every voice surface says on a host without the audio library.
pub(crate) const LIBRARY_UNAVAILABLE: &str = concat!(
    "Voice is unavailable: the audio library (libasound2 on Linux) is not installed on this ",
    "machine."
);

/// Open the speech output device at start when the owner has speech output on. No model is
/// touched: both engines load on first use, and a phrase is not refused while a cold device
/// open runs.
pub fn open_speech_output_if_enabled() {
    if !library_available() {
        tracing::info!("{LIBRARY_UNAVAILABLE}");
        return;
    }
    if crate::speech::synth::is_config_enabled() {
        let _ = output::ensure_audio_output();
    }
}

/// The retired per-user model tree, deleted once: the old encoder, its weights, the cache file
/// beside them and the orphan embedding model. Nothing reads that folder now — recognition and
/// synthesis use the machine-wide root ([`crate::speech::models`]) — and its weights are
/// gigabytes. A failure is said plainly: the tree is dead weight, not a reason to refuse anything.
pub(crate) fn remove_retired_models() {
    let Some(root) = crate::config::CONFIG.try_storage_root() else {
        return;
    };
    let dir = root.join("models");
    if !dir.exists() {
        return;
    }
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => tracing::info!(dir = %dir.display(), "Removed the retired speech model tree"),
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "Could not remove the retired speech model tree");
        }
    }
}

#[cfg(test)]
mod tests {
    use sha2::Digest as _;
    use std::collections::BTreeSet;

    /// The shim's two sources: the frozen bindings and the header macros beside them. `None` for the
    /// published package, which replaces the shim with the registry crate.
    fn shim() -> Option<(String, String)> {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/alsa-sys-dlopen/src");
        Some((
            std::fs::read_to_string(root.join("generated.rs")).ok()?,
            std::fs::read_to_string(root.join("lib.rs")).ok()?,
        ))
    }

    /// The probe and the shim resolve one library: a drift here would report voice available while
    /// every `snd_*` wrapper answered with a failure value.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_probe_and_the_shim_name_one_library() {
        let Some((generated, _)) = shim() else {
            return;
        };
        let soname = std::str::from_utf8(&super::ALSA_SONAME[..super::ALSA_SONAME.len() - 1])
            .expect("the soname is text");
        assert!(
            generated.contains(soname),
            "the shim resolves a different soname than the probe ({soname})"
        );
    }

    /// The surface the audio crate compiles against carries every name that crate asks for. It is
    /// frozen, so this is what a regeneration has to keep true: a surface wrapped from a header
    /// older than the crate expects compiles the shim and fails in the crate that consumes it,
    /// which no build of the shim can see.
    ///
    /// The names come from the audio crate's own source in the cargo registry, so the check runs
    /// where that source is on disk and skips where the machine keeps it elsewhere.
    /// `scripts/linux-cross-check.sh` is the proof that does not depend on the layout: it compiles
    /// the crate against the surface for both Linux targets.
    #[test]
    fn the_frozen_surface_carries_every_name_the_audio_crate_uses() {
        let Some((generated, lib)) = shim() else {
            return;
        };
        let Some(used) = spelled() else {
            return;
        };
        assert!(
            !used.is_empty(),
            "the `alsa` sources on this machine name no surface symbol: the walk read them wrong"
        );
        // A walk that read almost nothing would compare almost nothing: these five names sit in
        // different modules (pcm, ctl, seq, io and the header macros), so finding them shows the
        // walk reaches real code.
        for anchor in [
            "snd_pcm_open",
            "snd_ctl_open",
            "snd_seq_event_data",
            "SND_PCM_NONBLOCK",
            "__va_list_tag",
        ] {
            assert!(used.contains(anchor), "the walk did not reach `{anchor}`");
        }
        let surface = format!("{generated}\n{lib}");
        let missing: Vec<&str> = used
            .iter()
            .map(String::as_str)
            .filter(|name| !defines(&surface, name))
            .collect();
        assert!(
            missing.is_empty(),
            "the frozen surface is missing {} name(s) `alsa` uses: {missing:?}",
            missing.len()
        );
    }

    /// The committed surface is an artefact: a formatter run inside the vendored crate, or a hand
    /// edit, would change it without the recipe and take its byte-for-byte property with it. The pin
    /// is this constant — `regenerate.sh` writes the bytes, not the pin, so a legitimate bump moves
    /// this value in the same change — and `.gitattributes` keeps those bytes the same on a checkout
    /// that would otherwise rewrite the file's line endings.
    #[test]
    fn the_frozen_surface_is_the_one_the_recipe_writes() {
        let Some((generated, _)) = shim() else {
            return;
        };
        assert_eq!(
            crate::util::hex_string(&sha2::Sha256::digest(generated.as_bytes())),
            "8457f90893d987230642c68409c7d152d085a35f5d39e4020e9734d28a285e8a",
            "the frozen surface changed: rebuild it with vendor/alsa-sys-dlopen/regenerate.sh, \
             then move this pin with it"
        );
    }

    /// The surface carries no function the layer cannot reach: a function is reachable only by being
    /// named, so one the layer's sources never spell is a binding nothing can invoke — the other half
    /// of the completeness check above, which only says the surface is a superset of what the layer
    /// needs. "Named" is that check's walk, so a name the layer only re-exports counts as reachable,
    /// exactly as it does for the transform. Functions are the only items checked here because they
    /// are the only ones reached by name; a type or a constant is reached through the signature of
    /// something kept, which the digest above and the recipe below cover instead.
    #[test]
    fn the_surface_carries_no_function_the_audio_crate_cannot_reach() {
        let Some((generated, _)) = shim() else {
            return;
        };
        let Some(spelled) = spelled() else {
            return;
        };
        let declared: Vec<&str> = generated
            .lines()
            .filter_map(|line| line.strip_prefix("pub unsafe fn "))
            .filter_map(|rest| rest.split('(').next())
            .collect();
        assert!(
            !declared.is_empty(),
            "the surface declares no function: the walk read it wrong"
        );
        let unreachable: Vec<&str> = declared
            .iter()
            .copied()
            .filter(|name| !spelled.contains(*name))
            .collect();
        assert!(
            unreachable.is_empty(),
            "the surface binds {} function(s) `alsa` never spells: {unreachable:?}",
            unreachable.len()
        );
    }

    /// The recipe is the authority for the committed bytes, so it has to reproduce them: a transform
    /// that drifted from the file it produced would have the next dependency bump regenerate
    /// something else. Skipped where `rustc` or the two pinned inputs are not on this machine: both
    /// inputs live outside the crate, and cargo never fetches the published `alsa-sys` at all — this
    /// repository patches it out — so a fresh clone skips here and the digest above is what still
    /// holds. `vendor/alsa-sys-dlopen/regenerate.sh` is the same proof by hand, anywhere the inputs
    /// are; the two lines below mirror its invocation, so a third input is a third edit there and
    /// here.
    #[test]
    fn the_recipe_reproduces_the_frozen_surface() {
        let Some((generated, _)) = shim() else {
            return;
        };
        let (Some(bindings), Some(layer)) = (published_bindings(), audio_crate_source_dir()) else {
            return;
        };
        let work = crate::util::test::test_root().join("alsa-recipe");
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).expect("create the recipe's scratch directory");
        let transform = work.join("transform");
        let recipe = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vendor/alsa-sys-dlopen/regenerate.rs");
        let compiled = std::process::Command::new("rustc")
            .args(["--edition", "2021", "-O", "-o"])
            .arg(&transform)
            .arg(&recipe)
            .output();
        let Ok(compiled) = compiled else {
            return;
        };
        assert!(
            compiled.status.success(),
            "the recipe does not compile: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let produced = std::process::Command::new(&transform)
            .arg(&bindings)
            .arg(&layer)
            .output()
            .expect("run the recipe");
        assert!(
            produced.status.success(),
            "the recipe failed: {}",
            String::from_utf8_lossy(&produced.stderr)
        );
        assert_eq!(
            crate::util::hex_string(&sha2::Sha256::digest(&produced.stdout)),
            crate::util::hex_string(&sha2::Sha256::digest(generated.as_bytes())),
            "the recipe no longer writes the committed surface: \
             run vendor/alsa-sys-dlopen/regenerate.sh"
        );
    }

    /// The audio crate's source, as found in the cargo registry: the version `Cargo.lock` resolves,
    /// whose source is on disk wherever that crate was compiled. Only the files directly under
    /// `src` — the `direct` directory holds the crate's own copy of the ioctl structures, so the
    /// names in it are the crate's own, not the surface's.
    fn audio_crate_sources() -> Vec<String> {
        let Some(directory) = audio_crate_source_dir() else {
            return Vec::new();
        };
        let Ok(files) = std::fs::read_dir(directory) else {
            return Vec::new();
        };
        files
            .flatten()
            .map(|file| file.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
            .filter_map(|path| std::fs::read_to_string(path).ok())
            .collect()
    }

    /// The `src` directory of the `alsa` crate this build compiles, as the cargo registry holds it —
    /// the keep-set the recipe reads.
    fn audio_crate_source_dir() -> Option<std::path::PathBuf> {
        let directory = registry_crate("alsa")?.join("src");
        directory.is_dir().then_some(directory)
    }

    /// The published `alsa-sys` bindings the recipe reads, where the registry still holds the crate
    /// this repository patches out — cargo never builds it, so it may never have been fetched.
    fn published_bindings() -> Option<std::path::PathBuf> {
        let bindings = registry_crate("alsa-sys")?.join("src/generated.rs");
        bindings.is_file().then_some(bindings)
    }

    /// One crate's directory in the cargo registry, matched on the version `Cargo.lock` resolves.
    fn registry_crate(package: &str) -> Option<std::path::PathBuf> {
        let name = format!("{package}-{}", locked_version(package)?);
        let cargo_home = match std::env::var_os("CARGO_HOME") {
            Some(cargo_home) => std::path::PathBuf::from(cargo_home),
            None => std::path::PathBuf::from(std::env::var_os("HOME")?).join(".cargo"),
        };
        let registries = std::fs::read_dir(cargo_home.join("registry/src")).ok()?;
        for registry in registries.flatten() {
            let Ok(crates) = std::fs::read_dir(registry.path()) else {
                continue;
            };
            for entry in crates.flatten() {
                if entry.file_name().to_string_lossy() == name {
                    return Some(entry.path());
                }
            }
        }
        None
    }

    /// Every name the audio crate's sources spell, or `None` where its source is not on this machine:
    /// a vendored checkout, a directory registry or an offline mirror keeps it somewhere this cannot
    /// guess, so a miss says nothing about the surface. `scripts/linux-cross-check.sh` is the proof
    /// that does not depend on the layout.
    fn spelled() -> Option<BTreeSet<String>> {
        let sources = audio_crate_sources();
        if sources.is_empty() {
            return None;
        }
        let mut names = BTreeSet::new();
        for source in &sources {
            names.extend(names_in(source));
        }
        Some(names)
    }

    /// The version `Cargo.lock` resolves for one package, so the check reads the crate the build
    /// actually compiles rather than whichever copy the registry cache happens to hold.
    fn locked_version(package: &str) -> Option<String> {
        let lock = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
        )
        .ok()?;
        let mut lines = lock.lines();
        while let Some(line) = lines.next() {
            if line == format!("name = \"{package}\"") {
                let version = lines.next()?.strip_prefix("version = \"")?;
                return Some(version.strip_suffix('"')?.to_owned());
            }
        }
        None
    }

    /// The surface's own names in the shape the bindings emit them, so `snd_pcm` is not read as
    /// present because `snd_pcm_t` is.
    fn defines(surface: &str, name: &str) -> bool {
        [
            format!("fn {name}("),
            format!("type {name} ="),
            format!("struct {name} {{"),
            format!("union {name} {{"),
            format!("const {name}:"),
        ]
        .iter()
        .any(|shape| surface.contains(shape.as_str()))
    }

    /// Every name one `alsa` source takes from the frozen surface: the `snd_*` and `SND_*`
    /// identifiers it spells, plus the ones it names only through its `alsa` alias for `alsa_sys` —
    /// `__va_list_tag` is the only name without that prefix. The surface declares no modules, so
    /// every `alsa::`-qualified name in code is a surface item; the crate's module paths
    /// (`alsa::pcm`) live in its doc comments, which are stripped below.
    ///
    /// Comments are stripped because they talk about names the surface deliberately does not carry —
    /// `alsa` says so itself, beside the flag it builds by hand.
    fn names_in(source: &str) -> BTreeSet<String> {
        let code = without_comments(source);
        let mut names = BTreeSet::new();
        let mut current = String::new();
        for ch in code.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                current.push(ch);
                continue;
            }
            if current.starts_with("snd_") || current.starts_with("SND_") {
                names.insert(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
        if current.starts_with("snd_") || current.starts_with("SND_") {
            names.insert(current);
        }
        for (index, _) in code.match_indices("alsa::") {
            let rest = &code[index + "alsa::".len()..];
            let end = rest
                .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .unwrap_or(rest.len());
            if end > 0 {
                names.insert(rest[..end].to_owned());
            }
        }
        names
    }

    /// One source file with its comments and string literals removed: `//` to the end of the line,
    /// `/* */`, and `"…"` whole. Both are dropped rather than read, because a `snd_*` name in either
    /// is prose: `alsa` labels its own errors `from_const("snd_hctl_elem_id_get_name", …)` beside the
    /// call it actually makes, and its comments name functions the surface deliberately lacks.
    ///
    /// This reads the shapes `alsa`'s sources are written in, not Rust: no raw string with a quote
    /// inside it, no char literal holding one, no nested block comment. A source that had one would
    /// be mis-read silently, which is why the assumption is written down here beside the code that
    /// makes it rather than left implicit.
    ///
    /// `vendor/alsa-sys-dlopen/regenerate.rs` carries the same scan — it is the recipe's keep-set —
    /// and the two cannot share code, so a change here is a change there.
    fn without_comments(source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let mut code = String::with_capacity(source.len());
        let mut index = 0;
        while index < chars.len() {
            match chars[index] {
                '/' if chars.get(index + 1) == Some(&'/') => {
                    while index < chars.len() && chars[index] != '\n' {
                        index += 1;
                    }
                }
                '/' if chars.get(index + 1) == Some(&'*') => {
                    index += 2;
                    while index < chars.len()
                        && !(chars[index] == '*' && chars.get(index + 1) == Some(&'/'))
                    {
                        index += 1;
                    }
                    index += 2;
                }
                '"' => {
                    index += 1;
                    while index < chars.len() {
                        if chars[index] == '\\' {
                            index += 2;
                            continue;
                        }
                        let done = chars[index] == '"';
                        index += 1;
                        if done {
                            break;
                        }
                    }
                }
                ch => {
                    code.push(ch);
                    index += 1;
                }
            }
        }
        code
    }
}

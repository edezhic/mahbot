//! Making the product's own tools visible in the owner's own search path.
//!
//! One guarded block in the file the owner's own shell reads names the standard
//! directories and nothing else: the old private tools folder is never put on the
//! search path. The decision is remade after each successful read of his environment,
//! from the path that environment carries; his files are read only to recognise the
//! blocks the product itself wrote, so a stale or duplicated block is replaced by
//! exactly one correct block and a block an earlier release left in another shell's
//! startup file is taken out wherever it sits. A shell whose startup file cannot be
//! resolved gets no block, and then no earlier block is taken out anywhere either —
//! one of them may be the only thing keeping a directory on his search path.
//!
//! A rewrite preserves the file's remaining bytes and its permissions, follows a
//! symlinked configuration file rather than replacing the link, and lands atomically;
//! Windows gets the `Path` value written as the system writes it, without expanding
//! the references existing entries contain. When nothing can be done safely, nothing
//! is written and the reason is recorded at WARN — the level that survives the 8-hour
//! INFO retention — once per distinct reason, carrying no value read from the owner's environment.
//!
//! [`sync`] is the only entry point, called by
//! [`crate::shell_env::run_reader_loop`] with the environment it just published.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::shell_env::OwnerEnv;
use crate::util::UnwrapPoison;

/// The line opening a block the product appended to the owner's startup file,
/// byte-compatible with what is already in owners' files.
#[cfg(unix)]
const BLOCK_START: &str = "# >>> mahbot managed binaries >>>";

#[cfg(unix)]
const BLOCK_END: &str = "# <<< mahbot managed binaries <<<";

/// Make the product's own directories visible in the owner's own search path, from the
/// environment the owner's commands get: never from the text of his files, and never
/// before a read has succeeded. The work is blocking, so it rides the blocking pool —
/// the reader's next read waits for it, nothing on a command's path does.
pub(crate) async fn sync(env: Arc<OwnerEnv>) {
    #[cfg(any(unix, windows))]
    if tokio::task::spawn_blocking(move || sync_blocking(&env))
        .await
        .is_err()
    {
        // A panic in the blocking work is recorded at WARN rather than disappearing with
        // the join handle. The join error's own display is the panic payload, which is not
        // guaranteed free of a value out of the owner's environment.
        begin_sync();
        record_failure("the work that makes the product's own tools visible did not finish");
        end_sync();
    }
    // A target with neither a shell startup file nor a user `Path` value has nothing to
    // make visible.
    #[cfg(not(any(unix, windows)))]
    drop(env);
}

/// [`sync`] on the platform's own module, with one whole sync's failure bookkeeping.
#[cfg(any(unix, windows))]
fn sync_blocking(env: &OwnerEnv) {
    let dirs = visible_dirs();
    begin_sync();
    if dirs.is_empty() {
        // A host where neither directory resolves has nothing to put on a search path, and
        // an empty body would render as a trailing entry naming the current directory. It
        // is still worth a record: the owner would otherwise never learn why.
        record_failure(
            "the product's own tools have no directory to make visible (the owner's home \
             directory could not be resolved)",
        );
    } else {
        #[cfg(unix)]
        unix::sync(env, &dirs);
        #[cfg(windows)]
        windows::sync(env, &dirs);
    }
    end_sync();
}

/// The directories the owner's own terminal must resolve by bare name: the folder the
/// product is installed in ([`crate::util::managed_bin::mahbot_install_dir`]), the browser
/// helper's own directory, and the runtime's own directory. The system-wide directory is
/// not named (it is on his search path by default), and neither is the private tools
/// folder an earlier release used.
#[cfg(any(unix, windows))]
#[must_use]
fn visible_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    // The product's own directory, named as itself rather than through the helper's.
    dirs.extend(crate::util::managed_bin::mahbot_install_dir());
    // On Windows the helper sits beside the product; on unix its fallback is the parent.
    #[cfg(not(unix))]
    dirs.extend(crate::util::managed_bin::chrome_use_user_bin_dir());
    dirs.extend(crate::util::managed_bin::bun_bin_dir());
    dirs
}

/// Whether `dir` is NOT visible in the `PATH` the owner's own environment carries, the
/// occurrences the product's own contribution could have supplied (`made_of_them` of them)
/// excluded. An occurrence beyond those is the owner's own and is the only thing that
/// counts as visible. Entries are compared the way the platform compares them ([`same_entry`]).
/// Windows passes `0`: its user `Path` entries are recognised separately, by `entry_names`.
#[cfg(any(unix, windows))]
#[must_use]
fn dir_missing(env: &OwnerEnv, dir: &Path, made_of_them: usize) -> bool {
    path_entries(env)
        .iter()
        .filter(|entry| same_entry(entry, dir))
        .count()
        <= made_of_them
}

/// The entries of the `PATH` the owner's environment carries, split the platform's own way
/// (`:` on unix, `;` on Windows).
#[cfg(any(unix, windows))]
#[must_use]
fn path_entries(env: &OwnerEnv) -> Vec<PathBuf> {
    env.vars()
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("PATH"))
        .map_or_else(Vec::new, |(_, value)| {
            std::env::split_paths(value).collect()
        })
}

/// The separators a path on this host may end with, which name the same directory as the
/// text without them: `/` everywhere, `\` as well on Windows. On unix `\` is an ordinary
/// name byte, so an entry spelled `~/.local/bin\` names a directory of its own.
#[cfg(any(unix, windows))]
const TRAILING_SEPARATORS: &[char] = if cfg!(unix) { &['/'] } else { &['/', '\\'] };

/// Whether a `PATH` entry of this host or one of the product's own directories names `dir`,
/// compared the way the platform's own lookup compares them: byte for byte on unix,
/// case-insensitively on Windows, a trailing separator ignored either way.
#[cfg(any(unix, windows))]
#[must_use]
fn same_entry(entry: &Path, dir: &Path) -> bool {
    names(entry, dir, TRAILING_SEPARATORS, cfg!(not(unix)))
}

/// Whether an entry of a Windows `Path` value names `dir`: the same comparison, with both
/// of a Windows path's separators ignored.
#[cfg(any(windows, test))]
#[must_use]
fn same_entry_windows(entry: &Path, dir: &Path) -> bool {
    names(entry, dir, &['/', '\\'], true)
}

/// Whether `entry` and `dir` are the same directory: their text equal once a trailing
/// separator in `separators` is ignored. A path of nothing but separators is left as it stands.
#[cfg(any(unix, windows))]
#[must_use]
fn names(entry: &Path, dir: &Path, separators: &[char], fold_case: bool) -> bool {
    /// `path` as the comparison reads it.
    fn text(path: &Path, separators: &[char]) -> String {
        let text = path.to_string_lossy();
        let trimmed = text.trim_end_matches(separators);
        if trimmed.is_empty() {
            text.into_owned()
        } else {
            trimmed.to_string()
        }
    }
    let (entry, dir) = (text(entry, separators), text(dir, separators));
    if fold_case {
        entry.eq_ignore_ascii_case(&dir)
    } else {
        entry == dir
    }
}

/// The lines a sync can record, kept as two sets rather than one: a sync touches several of
/// the owner's files, so a reason is compared against the previous whole sync. Without that
/// a reason that keeps failing would be one WARN row per read.
#[cfg(any(unix, windows))]
struct Failures {
    /// The reasons the sync in progress has recorded.
    current: Vec<String>,
    /// The reasons the sync before it recorded.
    previous: Vec<String>,
}

#[cfg(any(unix, windows))]
static FAILURES: Mutex<Failures> = Mutex::new(Failures {
    current: Vec::new(),
    previous: Vec::new(),
});

/// The line both platforms' failures carry; only its fields differ.
#[cfg(any(unix, windows))]
const PATH_FAILURE_MESSAGE: &str =
    "could not make the product's own tools visible in the owner's own search path";

/// The audit line both platforms write when one of the product's own directories is put into
/// what the owner's own shell reads.
#[cfg(any(unix, windows))]
const PATH_VISIBLE_MESSAGE: &str =
    "made the product's own tools visible in the owner's own search path";

/// One refusal for both platforms: a directory whose text cannot be written into the owner's
/// own configuration without changing what it means there.
#[cfg(any(unix, windows))]
const UNSAFE_DIRECTORY: &str = "a directory that cannot be written into the owner's own search \
                                path (the product's own tools directory looks unsafe)";

/// Open one sync: the reasons it records are collected for [`end_sync`].
#[cfg(any(unix, windows))]
fn begin_sync() {
    FAILURES.lock().unwrap_poison().current.clear();
}

/// Record a failure of the sync in progress, once per distinct reason — the sync in progress
/// and the one before it both count, so one reason from two files is one row. The line carries
/// the reason and nothing else, never a value read from the owner's environment.
#[cfg(any(unix, windows))]
fn record_failure(reason: &str) {
    {
        let mut failures = FAILURES.lock().unwrap_poison();
        let known = failures
            .current
            .iter()
            .chain(&failures.previous)
            .any(|seen| seen == reason);
        failures.current.push(reason.to_string());
        if known {
            return;
        }
    }
    tracing::warn!(reason = %reason, "{PATH_FAILURE_MESSAGE}");
}

/// Close the sync: its reasons become the ones the next sync compares against, so a failure
/// that persists stays one row while one that comes back is reported again.
#[cfg(any(unix, windows))]
fn end_sync() {
    let mut failures = FAILURES.lock().unwrap_poison();
    failures.previous = std::mem::take(&mut failures.current);
}

/// What one sync changed in what the owner's own shell reads.
#[cfg(any(unix, windows))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Written {
    /// Nothing: it already says exactly what it must.
    Nothing,
    /// One of the product's own directories is on his search path because of this write, and from
    /// nowhere else. A block counts as having reached his search path even where the file it sits
    /// in is not one his shell reads ([`read_earlier_blocks`] holds that approximation), so a
    /// directory only such a block named is never reported here.
    Added,
    /// Nothing was added by that same measure: the write changed the product's own earlier
    /// contribution — a block brought to the current spelling or taken out, duplicate `Path`
    /// entries — whichever copy had put it there. By the same approximation this is also what a
    /// write that does reach his real search path is reported as.
    Removed,
}

/// The owner's own `PATH` is set by his shell's startup files, so the block goes into the one
/// file the shell he actually runs reads — never one invented for him, and never a file whose
/// mere presence would change which files that shell reads.
#[cfg(unix)]
mod unix {
    use std::fs;
    use std::ops::Range;
    use std::path::{Path, PathBuf};

    use super::{
        BLOCK_END, BLOCK_START, OwnerEnv, PATH_VISIBLE_MESSAGE, UNSAFE_DIRECTORY, Written,
        dir_missing, record_failure, same_entry,
    };

    /// Bring the owner's startup file in line with `dirs`. His shell and home are resolved once
    /// here and handed to [`run`]: the shell only says which startup file is his.
    pub(super) fn sync(env: &OwnerEnv, dirs: &[PathBuf]) {
        let (shell, home) = crate::shell_env::owner_shell_and_home();
        let name = shell.as_deref().map(crate::shell_env::shell_basename);
        let inherited = inherited_path_entries();
        let outcome = match (name.as_deref(), home.as_deref()) {
            (Some(name), Some(home)) => run(name, home, env, &inherited, dirs).map(|written| match written {
                Written::Nothing => {}
                // A write into his own file happens once, when the block is first needed or has
                // to change, so this is the audit line for a change to his configuration.
                Written::Added => {
                    tracing::info!(shell = name, "{PATH_VISIBLE_MESSAGE}");
                }
                // The write changed only the product's own earlier block.
                Written::Removed => {
                    tracing::info!(
                        shell = name,
                        "brought the product's own block in the owner's own startup file to its \
                         current spelling, or took it out where his own files already name those \
                         directories; by the product's own measure (a block of its own counts as \
                         having reached his search path even where the file it sits in is not one \
                         his shell reads) this added no directory to that path"
                    );
                }
            }),
            (None, _) => Err("the owner's shell could not be determined".to_string()),
            (_, None) => Err("the owner's home directory could not be determined".to_string()),
        };
        if let Err(reason) = outcome {
            record_failure(&reason);
        }
    }

    /// The entries of the search path the daemon itself was started with: the read's shell
    /// inherits this process's environment and applies the owner's startup files on top, so an
    /// entry here may be the product's own block's, not the owner's own arrangement.
    fn inherited_path_entries() -> Vec<PathBuf> {
        std::env::var_os("PATH")
            .map_or_else(Vec::new, |value| std::env::split_paths(&value).collect())
    }

    /// Resolve the file today's block belongs in, read it and every other file the product may
    /// have left a block in, and rewrite today's file when what it must say differs from what it
    /// says. Nothing else is written until his own file holds its block, so a write that fails
    /// there cannot cost him a directory an earlier block was the only source of; each rewrite is
    /// atomic, and a crash between the files leaves an earlier block standing until the next read.
    /// `inherited` is the search path the daemon itself was started with, which the read's shell
    /// inherited: it is what tells an occurrence the product's own block left behind from one the
    /// owner's own arrangement gives him (see [`Self::inherited_path_entries`]).
    fn run(
        shell: &str,
        home: &Path,
        env: &OwnerEnv,
        inherited: &[PathBuf],
        dirs: &[PathBuf],
    ) -> Result<Written, String> {
        // Refuse before anything is read or written: with a directory the block cannot name on
        // this shell, today's file can get no block — and taking the earlier blocks out would
        // leave him without the directory they were keeping on his search path.
        render(shell, dirs, home)?;
        // A shell with no startup file the product can resolve is the same case, and the
        // reason is recorded instead.
        let target = target_file(shell, home, env)?;
        remove_leftover_temp(&target);
        // A missing file is an empty one, read before any of his files is written: one that
        // cannot be read is as much a reason not to touch the others as one that cannot be written.
        let existing = read_startup_file(&target, FILE_UNREADABLE)?.unwrap_or_default();
        let regions = block_regions(&existing);
        // The product's own earlier blocks are read wherever they could sit, and the rewrite
        // each of those files needs is held back until his own file is in place.
        let mut made = vec![0; dirs.len()];
        let strips = read_earlier_blocks(home, env, &target, dirs, &mut made);
        add_contributions(&existing, &regions, dirs, home, &mut made);
        // An entry the daemon itself was started with is not the owner's evidence: the shell
        // that started it had the product's own block applied too. Counting only the fresh
        // occurrences would read the block's own work as his arrangement and take the block out
        // of the file his terminal reads, so they count against the block instead.
        for (count, dir) in made.iter_mut().zip(dirs) {
            if *count > 0 {
                *count += inherited
                    .iter()
                    .filter(|entry| same_entry(entry, dir))
                    .count();
            }
        }
        // Only what his own environment does not already give him is named: naming it again
        // would put it on his search path twice.
        let missing: Vec<PathBuf> = dirs
            .iter()
            .zip(&made)
            .filter(|(dir, count)| dir_missing(env, dir, **count))
            .map(|(dir, _)| dir.clone())
            .collect();
        // The refusals were raised before any of this was read, so rendering cannot fail.
        let desired = render(shell, &missing, home)?;
        let Some(content) = planned_content(&existing, &regions, &desired) else {
            strip_earlier_blocks(strips);
            return Ok(Written::Nothing);
        };
        // Everything else waits on this write: a failure here leaves the earlier blocks where
        // they are.
        write_atomically(&target, &content)?;
        strip_earlier_blocks(strips);
        // A write is an addition only where it is what puts a directory of the product's own on
        // his search path: the directory his own environment does not have at all and no earlier
        // block named.
        let added = dirs
            .iter()
            .zip(&made)
            .any(|(dir, count)| *count == 0 && dir_missing(env, dir, 0));
        Ok(if added {
            Written::Added
        } else {
            Written::Removed
        })
    }

    /// One earlier block of the product's own, with the content that file must have once it is
    /// gone.
    struct Strip {
        /// The shell that file belongs to, which only says which file it is.
        shell: &'static str,
        path: PathBuf,
        content: String,
    }

    /// Read every startup file that could hold an earlier block of the product's own, count how
    /// many times each of `dirs` those blocks name (today's own file is counted by the caller),
    /// and report what each of those files must be rewritten to. Nothing is written here: the
    /// strips are applied by [`strip_earlier_blocks`], once his own file holds its block.
    ///
    /// An earlier release appended the same block to `~/.zshrc` and `~/.bashrc` — whichever
    /// belonged to the login shell, and both when that could not be determined — so a block can
    /// sit in a file today's shell never reads, and one naming the old private tools folder would
    /// leave that folder on the owner's search path. Only a block the product itself wrote is
    /// taken out (see [`block_regions`]); every other byte is left as it is, and no block is ever
    /// added to these files.
    ///
    /// Every block found here is counted here and taken out here, which is what keeps a directory
    /// of `dirs` from being dropped: what a counted block contributed is either still on his
    /// search path without it or named by today's block (see [`dir_missing`]). A block naming the
    /// private tools folder is counted for nothing and leaves his search path with the block.
    ///
    /// Counting a block approximates its having reached his search path. The ordinary error is
    /// harmless — a block in a file he does not source is still counted, so a directory can be
    /// named once more by today's block, and the read after the next one settles it. A body entry
    /// [`add_contributions`] cannot read at all is the one harmful direction: it is counted for
    /// nothing while its block is taken out, and reaching it takes a hand-edit of a block the
    /// product wrote.
    fn read_earlier_blocks(
        home: &Path,
        env: &OwnerEnv,
        target: &Path,
        dirs: &[PathBuf],
        made: &mut [usize],
    ) -> Vec<Strip> {
        let mut strips = Vec::new();
        for (file_shell, path) in candidates(env, home, target) {
            remove_leftover_temp(&path);
            let content = match read_startup_file(&path, FILE_WITH_AN_EARLIER_BLOCK) {
                Ok(Some(content)) => content,
                Ok(None) => continue,
                Err(reason) => {
                    record_failure(&reason);
                    continue;
                }
            };
            let regions = block_regions(&content);
            if regions.is_empty() {
                continue;
            }
            add_contributions(&content, &regions, dirs, home, made);
            strips.push(Strip {
                shell: file_shell,
                path,
                content: strip_regions(&content, &regions),
            });
        }
        strips
    }

    /// Take the earlier blocks out, [once the owner's own file holds its
    /// block](run), recording the change rather than per-start noise.
    fn strip_earlier_blocks(strips: Vec<Strip>) {
        for Strip {
            shell,
            path,
            content,
        } in strips
        {
            match write_atomically(&path, &content) {
                Ok(()) => tracing::info!(
                    shell = shell,
                    "removed an earlier mahbot block from another of the owner's own startup files"
                ),
                Err(reason) => record_failure(&reason),
            }
        }
    }

    /// The startup files the product may have left its own block in, besides the one the owner's
    /// own shell reads today: the plain `~/.zshrc` and `~/.bashrc` an earlier release appended to
    /// for the login shell, plus the file every other shell the product knows reads.
    fn candidates(env: &OwnerEnv, home: &Path, target: &Path) -> Vec<(&'static str, PathBuf)> {
        let mut files = vec![
            ("zsh", home.join(".zshrc")),
            ("zsh", zsh_startup_file(home, env)),
            ("bash", home.join(".bashrc")),
            ("bash", login_profile_file(home)),
            ("sh", home.join(".profile")),
            ("fish", fish_startup_file(home, env)),
        ];
        if let Ok(file) = env_startup_file(env) {
            files.push(("sh", file));
        }
        // Files are recognised by what they really are: `~/.zshrc` symlinked to
        // `$ZDOTDIR/.zshrc` is one startup file, and sweeping it as another would take today's
        // block out on every read and let [`run`] put it back.
        let canonical = |path: &Path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut seen: Vec<PathBuf> = vec![canonical(target)];
        let mut kept: Vec<(&'static str, PathBuf)> = Vec::with_capacity(files.len());
        for file in files {
            let key = canonical(&file.1);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            kept.push(file);
        }
        kept
    }

    /// The reason a startup file that could not be read is recorded with.
    const FILE_UNREADABLE: &str = "the owner's startup file could not be read";

    /// The same, for one of the other files the product may have left a block in.
    const FILE_WITH_AN_EARLIER_BLOCK: &str =
        "a startup file holding an earlier mahbot block could not be read";

    /// The content of one of the owner's own startup files, or `None` when it does not exist.
    fn read_startup_file(path: &Path, what: &str) -> Result<Option<String>, String> {
        match fs::read_to_string(path) {
            Ok(content) => Ok(Some(content)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_reason(what, &e)),
        }
    }

    /// The file the owner's own terminal really reads for `shell`, which is not the same file for
    /// every shell or on every unix: macOS starts all of these as login shells, elsewhere each
    /// shell reads the file its own convention names, and an unknown shell is refused rather than
    /// guessed at ([`crate::shell_env::knows_shell`]). This is the only file a block is ever
    /// written into — a block an earlier release left elsewhere is taken out, nothing added.
    fn target_file(shell: &str, home: &Path, env: &OwnerEnv) -> Result<PathBuf, String> {
        let macos = cfg!(target_os = "macos");
        match shell {
            "zsh" => Ok(zsh_startup_file(home, env)),
            "bash" if macos => Ok(login_profile_file(home)),
            "bash" => Ok(home.join(".bashrc")),
            "fish" => Ok(fish_startup_file(home, env)),
            other if crate::shell_env::knows_shell(other) => {
                if macos {
                    Ok(home.join(".profile"))
                } else {
                    env_startup_file(env)
                }
            }
            _ => {
                Err("the owner's shell is not one whose startup file the product knows".to_string())
            }
        }
    }

    /// `~/.zshrc`, or `$ZDOTDIR/.zshrc` when the owner moved zsh's own startup directory.
    fn zsh_startup_file(home: &Path, env: &OwnerEnv) -> PathBuf {
        env_dir(env, "ZDOTDIR").map_or_else(|| home.join(".zshrc"), |dir| dir.join(".zshrc"))
    }

    /// `~/.config/fish/config.fish`, or `$XDG_CONFIG_HOME/fish/config.fish` when the owner moved
    /// fish's configuration directory.
    fn fish_startup_file(home: &Path, env: &OwnerEnv) -> PathBuf {
        env_dir(env, "XDG_CONFIG_HOME").map_or_else(
            || home.join(".config").join("fish").join("config.fish"),
            |dir| dir.join("fish").join("config.fish"),
        )
    }

    /// The directory the owner's own environment names in `name`, or none when the variable is
    /// absent, empty, or not an absolute path.
    fn env_dir(env: &OwnerEnv, name: &str) -> Option<PathBuf> {
        env.vars()
            .iter()
            .find(|(var, _)| var.as_os_str() == std::ffi::OsStr::new(name))
            .map(|(_, value)| PathBuf::from(value))
            .filter(|dir| dir.is_absolute())
    }

    /// The first of the login profile files a login bash reads when it exists, else
    /// `~/.bash_profile`.
    fn login_profile_file(home: &Path) -> PathBuf {
        [".bash_profile", ".bash_login", ".profile"]
            .iter()
            .map(|name| home.join(name))
            .find(|path| path.exists())
            .unwrap_or_else(|| home.join(".bash_profile"))
    }

    /// The one startup file an interactive POSIX shell reads on a non-macOS unix: the file `$ENV`
    /// names. A relative value is refused like an empty one — the shell resolves it against its
    /// own directory, not the daemon's.
    fn env_startup_file(env: &OwnerEnv) -> Result<PathBuf, String> {
        env.vars()
            .iter()
            .find(|(name, _)| name.as_os_str() == std::ffi::OsStr::new("ENV"))
            .map(|(_, value)| PathBuf::from(value))
            .filter(|path| path.is_absolute() && path.exists())
            .ok_or_else(|| {
                "the owner's shell reads no startup file the product can write safely".to_string()
            })
    }

    /// The body line [`render`] writes and [`body_entries`] reads back: the product's directories
    /// after whatever the owner's own path already holds.
    const SH_BODY_PREFIX: &str = "export PATH=\"$PATH:";
    const SH_BODY_SUFFIX: &str = "\"";
    const FISH_BODY_PREFIX: &str = "set -gx PATH $PATH ";

    /// The block body without its markers: one line that puts `dirs` after whatever the owner's
    /// own path already holds, so his own installs keep precedence. Empty for an empty list — an
    /// empty body would name the current directory. fish's `PATH` is a list and `set -gx PATH`
    /// appends to it; `fish_add_path` is deliberately not used, because it is a function an older
    /// fish may not have.
    fn render(shell: &str, dirs: &[PathBuf], home: &Path) -> Result<String, String> {
        if dirs.is_empty() {
            return Ok(String::new());
        }
        let fish = shell == "fish";
        let mut rendered = Vec::with_capacity(dirs.len());
        for dir in dirs {
            rendered.push(render_dir(dir, home, fish)?);
        }
        Ok(if fish {
            format!("{FISH_BODY_PREFIX}{}", rendered.join(" "))
        } else {
            format!("{SH_BODY_PREFIX}{}{SH_BODY_SUFFIX}", rendered.join(":"))
        })
    }

    /// One directory as the shell line spells it: `$HOME/<relative>`, refused unless it really is
    /// inside the home whose startup file the block goes into — and with fish's unquoted form in
    /// mind, where a `set` would split a directory holding whitespace and a glob character would
    /// expand. The home check is also the precondition [`run`] applies before anything is written,
    /// so a state it refuses cannot leave the owner's other files stripped. The byte check guards
    /// the interpolation rather than today's two directories.
    fn render_dir(dir: &Path, home: &Path, fish: bool) -> Result<String, String> {
        let Ok(relative) = dir.strip_prefix(home) else {
            return Err(
                "the product's own tools are not under the owner's home (the block would name \
                 another home's directories)"
                    .to_string(),
            );
        };
        let relative = relative.to_string_lossy();
        // `"`, `` ` ``, `$` and `\` end the posix form's quoting early, a newline splits the
        // file, and `:` separates the entries of the posix form.
        if relative.contains(['"', '`', '$', '\\', '\n', ':']) {
            return Err(UNSAFE_DIRECTORY.to_string());
        }
        let token = format!("$HOME/{relative}");
        // The fish form is unquoted: whitespace would split the line and a glob character would
        // expand.
        if fish && token.contains([' ', '\t', '*', '?', '[']) {
            return Err(UNSAFE_DIRECTORY.to_string());
        }
        Ok(token)
    }

    /// The file's new full content, or `None` when the file already is what it must be. `regions`
    /// are the product's own previous appends ([`block_regions`]) and `desired` the body of the
    /// block the file must carry — empty when his own environment already gives him every
    /// directory, which is why it also says whether a block is written. Those appends are taken
    /// out first, whatever they name; everything else comes back byte for byte.
    fn planned_content(existing: &str, regions: &[Range<usize>], desired: &str) -> Option<String> {
        let appended = block_text(desired);
        // Already exactly right, including the owner's own file mode.
        if regions.len() == 1 && existing[regions[0].clone()] == appended {
            return None;
        }
        let cleaned = strip_regions(existing, regions);
        match (regions.is_empty(), desired.is_empty()) {
            // Nothing to add and nothing of the product's own to take out.
            (true, true) => None,
            // The directories are visible from his environment, so nothing is added — but a stale
            // or duplicated block is still taken out.
            (_, true) => Some(cleaned),
            // Otherwise exactly one correct block follows everything else the file held.
            _ => Some(append_block(&cleaned, desired)),
        }
    }

    /// `content` with one freshly rendered block appended. A last line without a newline is
    /// closed first, so the block never continues a line of the owner's own.
    fn append_block(content: &str, desired: &str) -> String {
        let mut out = String::from(content);
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&block_text(desired));
        out
    }

    /// The block as it is written today, byte for byte: one leading newline before the start
    /// marker and one trailing newline after the end marker — exactly what an append of an earlier
    /// release wrote, so a file it touched is recognised.
    fn block_text(desired: &str) -> String {
        format!("\n{BLOCK_START}\n{desired}\n{BLOCK_END}\n")
    }

    /// The byte ranges of `content` the product's own previous appends occupy: the run from a
    /// [`BLOCK_START`] line through the [`BLOCK_END`] line that follows it, with the newline its
    /// append wrote in front — the exact shape [`append_block`] leaves. A pair counts only when
    /// exactly one line sits between the markers, so a stray marker or any second line takes
    /// nothing of the owner's with it.
    fn block_regions(content: &str) -> Vec<Range<usize>> {
        let mut regions: Vec<Range<usize>> = Vec::new();
        // The line the marker opened on and whether a body line has been seen: 0 = no body line
        // yet, 1 = exactly the one body line the product writes, 2 = a run the product never wrote.
        let mut opened: Option<(usize, u8)> = None;
        let mut offset = 0;
        for line in content.split_inclusive('\n') {
            let line_start = offset;
            offset += line.len();
            let text = line.strip_suffix('\n').unwrap_or(line);
            if text == BLOCK_START {
                opened = Some((line_start, 0));
            } else if text == BLOCK_END {
                if let Some((from, 1)) = opened.take() {
                    // The newline the append wrote in front of the marker is taken back, so a strip
                    // removes exactly what an append added. It is the append's own when the file
                    // begins with it or a blank line precedes the marker; directly under one of the
                    // owner's lines it is his, and taking it would glue two of his lines together.
                    // `claimed` keeps a region from taking a newline the region before it owns.
                    let claimed = regions.last().map_or(0, |previous| previous.end);
                    let appended = from > claimed
                        && from > 0
                        && content.as_bytes()[from - 1] == b'\n'
                        && (from == 1 || content.as_bytes()[from - 2] == b'\n');
                    let from = if appended { from - 1 } else { from };
                    regions.push(from..offset);
                }
            } else if let Some(open) = opened.as_mut() {
                open.1 = if open.1 == 0 && is_product_body(text) {
                    1
                } else {
                    2
                };
            }
        }
        regions
    }

    /// Whether `text` is a body line of the product's own block for this platform.
    fn is_product_body(text: &str) -> bool {
        !body_entries(text).is_empty()
    }

    /// The directories one body line names, in the spelling the block writes them, for either
    /// form (posix or fish). Empty for a line that is not a body line at all.
    fn body_entries(line: &str) -> Vec<&str> {
        if let Some(rest) = line
            .strip_prefix(SH_BODY_PREFIX)
            .and_then(|rest| rest.strip_suffix(SH_BODY_SUFFIX))
        {
            return rest.split(':').collect();
        }
        if let Some(rest) = line.strip_prefix(FISH_BODY_PREFIX) {
            return rest.split(' ').collect();
        }
        Vec::new()
    }

    /// Add, for each of `counts`, how many times the product's own appends in `content` name that
    /// directory, so the occurrences in the owner's own `PATH` can be reduced by exactly those.
    ///
    /// An entry counts when it *names* the directory the way the platform's own lookup reads one
    /// ([`same_entry`]): absolute or as the `$HOME/<relative>` token today's block writes, with or
    /// without a trailing separator, through a leading home reference however a hand-edit spells
    /// it. Counting by exact text would leave a respelled entry counted for nothing while its
    /// block is taken out all the same, and a count coming up short is what costs him a directory
    /// (see [`read_earlier_blocks`]). The count is per directory, never per block: an earlier
    /// block names a folder the block does not name today, so counting blocks would credit a
    /// directory with occurrences it never had.
    fn add_contributions(
        content: &str,
        regions: &[Range<usize>],
        dirs: &[PathBuf],
        home: &Path,
        counts: &mut [usize],
    ) {
        for region in regions {
            for line in content[region.clone()].lines() {
                for entry in body_entries(line) {
                    let entry = PathBuf::from(expand_home(entry, home));
                    for (count, dir) in counts.iter_mut().zip(dirs) {
                        if same_entry(&entry, dir) {
                            *count += 1;
                        }
                    }
                }
            }
        }
    }

    /// `entry` with a leading home reference — `$HOME/`, `${HOME}/`, `~/` — replaced by the
    /// owner's own home. Anything else, a `$HOME` not followed by a separator included, stands.
    fn expand_home(entry: &str, home: &Path) -> String {
        for reference in ["$HOME", "${HOME}", "~"] {
            if let Some(rest) = entry.strip_prefix(reference)
                && rest.starts_with('/')
            {
                return format!("{}{rest}", home.to_string_lossy());
            }
        }
        entry.to_string()
    }

    /// `content` with the product's own regions taken out.
    fn strip_regions(content: &str, regions: &[Range<usize>]) -> String {
        if regions.is_empty() {
            return content.to_string();
        }
        let mut cleaned = String::with_capacity(content.len());
        let mut cursor = 0;
        for region in regions {
            cleaned.push_str(&content[cursor..region.start]);
            cursor = region.end;
        }
        cleaned.push_str(&content[cursor..]);
        cleaned
    }

    /// Replace `target` with `content` in one atomic step, keeping everything about the file that
    /// is not its content. A symlinked configuration file is written through, including a link
    /// whose destination does not exist yet ([`resolved_target`]), a missing directory is created,
    /// and the write lands on a sibling temp file renamed over the target, carrying the target's
    /// own permissions.
    fn write_atomically(target: &Path, content: &str) -> Result<(), String> {
        let resolved = resolved_target(target);
        if let Some(parent) = resolved.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| {
                io_reason(
                    "the directory of the owner's startup file could not be created",
                    &e,
                )
            })?;
        }
        let Some(tmp) = temp_path(&resolved) else {
            return Err("the owner's startup file has no name to write next to".to_string());
        };
        // Whatever sits at the staging name is a leftover from an interrupted write — or something
        // else in the owner's own home. Removing it (the link itself, never what it points at) is
        // what keeps a write from landing through it.
        let _ = fs::remove_file(&tmp);
        if let Err(e) = write_temp(&tmp, &resolved, content) {
            let _ = fs::remove_file(&tmp);
            return Err(io_reason(
                "the owner's startup file could not be written",
                &e,
            ));
        }
        if let Err(e) = fs::rename(&tmp, &resolved) {
            let _ = fs::remove_file(&tmp);
            return Err(io_reason(
                "the owner's startup file could not be replaced",
                &e,
            ));
        }
        Ok(())
    }

    /// The sibling a rewrite stages into before the rename.
    fn temp_path(resolved: &Path) -> Option<PathBuf> {
        resolved
            .file_name()
            .map(|name| resolved.with_file_name(format!("{}.mahbot_tmp", name.to_string_lossy())))
    }

    /// Remove the temp file an interrupted write may have left beside `target`.
    fn remove_leftover_temp(target: &Path) {
        if let Some(tmp) = temp_path(&resolved_target(target)) {
            let _ = fs::remove_file(tmp);
        }
    }

    /// The path a write to `target` must land on: whatever a symlink at `target` points at — even
    /// when that file does not exist yet — or `target` itself.
    fn resolved_target(target: &Path) -> PathBuf {
        let Ok(destination) = fs::read_link(target) else {
            return target.to_path_buf();
        };
        let dest = match target.parent() {
            Some(parent) if destination.is_relative() => parent.join(destination),
            _ => destination,
        };
        fs::canonicalize(&dest).unwrap_or(dest)
    }

    /// The temp file's own write: created fresh, never opened, with the target's own permissions,
    /// so the content is never briefly readable under the process umask — and, because the umask
    /// masks what `open` is given, those exact bits are set again before the rename.
    fn write_temp(tmp: &Path, target: &Path, content: &str) -> std::io::Result<()> {
        use std::io::Write as _;
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

        let mode = fs::metadata(target)
            .ok()
            .map(|metadata| metadata.permissions().mode() & 0o777);
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        if let Some(mode) = mode {
            options.mode(mode);
        }
        options.open(tmp)?.write_all(content.as_bytes())?;
        if let Some(mode) = mode {
            fs::set_permissions(tmp, fs::Permissions::from_mode(mode))?;
        }
        Ok(())
    }

    /// A failure's reason for the log: the module's own sentence with only the I/O kind appended.
    /// Never the path and never the `io::Error`'s display, which carry the owner's own files.
    #[must_use]
    fn io_reason(what: &str, error: &std::io::Error) -> String {
        format!("{what} ({})", error.kind())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::ffi::OsString;
        use tempfile::TempDir;

        /// A daemon started by a launcher inherits no search path of its own.
        const NO_INHERITED: &[PathBuf] = &[];

        /// Two directories inside a fake home, the shape the block always has.
        fn home_dirs(home: &Path) -> Vec<PathBuf> {
            vec![home.join(".local/bin"), home.join(".bun/bin")]
        }

        /// The block body those two directories render to.
        fn block_body(home: &Path) -> String {
            render("zsh", &home_dirs(home), home).expect("render")
        }

        /// How many times each of `dirs` is named by the product's own blocks in `content`.
        fn made(content: &str, dirs: &[PathBuf], home: &Path) -> Vec<usize> {
            let mut counts = vec![0; dirs.len()];
            add_contributions(content, &block_regions(content), dirs, home, &mut counts);
            counts
        }

        fn env_with_path(path: &str) -> OwnerEnv {
            OwnerEnv::new(vec![(OsString::from("PATH"), OsString::from(path))])
        }

        /// The two halves [`run`] runs around the write of the owner's own file, with what each
        /// file's earlier blocks contributed.
        fn sweep(home: &Path, env: &OwnerEnv, target: &Path, dirs: &[PathBuf]) -> Vec<usize> {
            let mut made = vec![0; dirs.len()];
            let strips = read_earlier_blocks(home, env, target, dirs, &mut made);
            strip_earlier_blocks(strips);
            made
        }

        /// `planned_content` over content the product has not touched before.
        fn plan(existing: &str, desired: &str) -> Option<String> {
            planned_content(existing, &block_regions(existing), desired)
        }

        #[test]
        fn planned_content_appends_only_when_a_directory_is_missing() {
            let home = Path::new("/home/o");
            let desired = block_body(home);
            let appended = plan("export FOO=1\n", &desired).expect("append");
            assert_eq!(
                appended,
                format!("export FOO=1\n\n{BLOCK_START}\n{desired}\n{BLOCK_END}\n")
            );
            assert_eq!(plan("export FOO=1\n", ""), None);
        }

        #[test]
        fn planned_content_replaces_a_stale_block_and_collapses_duplicates() {
            let home = Path::new("/home/o");
            let desired = block_body(home);
            let fresh = block_text(&desired);
            let stale = block_text("export PATH=\"$PATH:$HOME/.mahbot/bin\"");
            let expected = format!("export FOO=1\n{fresh}");

            let replaced = plan(&format!("export FOO=1\n{stale}"), &desired).expect("rewrite");
            assert_eq!(replaced, expected);
            assert!(
                !replaced.contains(".mahbot/bin"),
                "the private tools folder is gone"
            );

            let collapsed =
                plan(&format!("export FOO=1\n{fresh}{fresh}"), &desired).expect("collapse");
            assert_eq!(collapsed, expected);
        }

        #[test]
        fn planned_content_removes_a_stale_block_when_nothing_is_needed() {
            let home = Path::new("/home/o");
            let desired = block_body(home);
            let stale = block_text("export PATH=\"$PATH:$HOME/.mahbot/bin\"");
            assert_eq!(
                plan(&format!("export FOO=1\n{stale}"), ""),
                Some("export FOO=1\n".to_string())
            );
            let with_block = format!("export FOO=1\n{}", block_text(&desired));
            assert_eq!(plan(&with_block, ""), Some("export FOO=1\n".to_string()));
            assert_eq!(plan(&with_block, &desired), None);
        }

        #[test]
        fn planned_content_round_trips_the_owners_file_and_is_idempotent() {
            let home = Path::new("/home/o");
            let desired = block_body(home);
            for (original, restored) in [
                ("export FOO=1\n", "export FOO=1\n"),
                ("export FOO=1", "export FOO=1\n"),
            ] {
                let written = plan(original, &desired).expect("append");
                let cleaned = strip_regions(&written, &block_regions(&written));
                assert_eq!(cleaned, restored);
                assert_eq!(plan(&written, &desired), None);
            }
        }

        #[test]
        fn render_writes_the_posix_and_the_fish_form() {
            let home = Path::new("/home/o");
            let dirs = home_dirs(home);
            assert_eq!(
                render("zsh", &dirs, home).expect("posix"),
                "export PATH=\"$PATH:$HOME/.local/bin:$HOME/.bun/bin\""
            );
            assert_eq!(
                render("fish", &dirs, home).expect("fish"),
                "set -gx PATH $PATH $HOME/.local/bin $HOME/.bun/bin"
            );
        }

        #[test]
        fn render_refuses_a_directory_outside_the_owners_home_and_unwritable_bytes() {
            let home = Path::new("/home/o");
            assert!(render("zsh", &[PathBuf::from("/opt/tools/bin")], home).is_err());

            for unsafe_dir in [
                "/home/o/a\"b",
                "/home/o/a`b",
                "/home/o/a$b",
                "/home/o/a\nb",
                "/home/o/a\\b",
                "/home/o/a:b",
            ] {
                let dirs = vec![PathBuf::from(unsafe_dir)];
                assert!(render("zsh", &dirs, home).is_err(), "{unsafe_dir}");
            }
            let spaced = vec![PathBuf::from("/home/o/a b")];
            assert!(render("fish", &spaced, home).is_err());
            assert_eq!(
                render("zsh", &spaced, home).expect("posix"),
                "export PATH=\"$PATH:$HOME/a b\""
            );
        }

        #[test]
        fn a_directory_is_visible_only_beyond_what_the_block_contributed() {
            let home = Path::new("/home/o");
            let local = home.join(".local/bin");
            let bun = home.join(".bun/bin");

            assert!(dir_missing(
                &env_with_path("/home/o/.local/bin:/usr/bin"),
                &bun,
                0
            ));
            assert!(!dir_missing(
                &env_with_path("/home/o/.local/bin:/usr/bin"),
                &local,
                0
            ));

            let both = env_with_path("/usr/bin:/home/o/.local/bin:/home/o/.bun/bin");
            assert!(dir_missing(&both, &local, 1));
            assert!(dir_missing(&both, &bun, 1));

            let doubled =
                env_with_path("/usr/bin:/home/o/.local/bin:/home/o/.bun/bin:/home/o/.bun/bin");
            assert!(dir_missing(&doubled, &bun, 2));
            assert!(!dir_missing(&doubled, &bun, 1));

            let mixed =
                env_with_path("/home/o/.bun/bin:/usr/bin:/home/o/.local/bin:/home/o/.bun/bin");
            assert!(!dir_missing(&mixed, &bun, 1));
            assert!(dir_missing(&mixed, &local, 1));

            assert!(dir_missing(&OwnerEnv::new(Vec::new()), &local, 0));
        }

        #[test]
        fn the_block_names_only_the_directories_his_own_files_do_not_give_him() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let dirs = home_dirs(&home);
            let env = env_with_path(&format!("{}/.local/bin:/usr/bin", home.display()));
            fs::write(home.join(".zshrc"), "export A=1\n").expect("write");

            assert!(matches!(
                run("zsh", &home, &env, NO_INHERITED, &dirs).expect("run"),
                Written::Added
            ));
            assert_eq!(
                fs::read_to_string(home.join(".zshrc")).expect("read"),
                format!(
                    "export A=1\n\n{BLOCK_START}\nexport PATH=\"$PATH:$HOME/.bun/bin\"\n{BLOCK_END}\n"
                )
            );
            assert!(matches!(
                run("zsh", &home, &env, NO_INHERITED, &dirs).expect("run"),
                Written::Nothing
            ));
            fs::write(
                home.join(".zshrc"),
                format!(
                    "export A=1\n{}",
                    block_text("export PATH=\"$PATH:$HOME/.bun/bin/\"")
                ),
            )
            .expect("write");
            assert!(matches!(
                run("zsh", &home, &env, NO_INHERITED, &dirs).expect("run"),
                Written::Removed
            ));
            let respelled = fs::read_to_string(home.join(".zshrc")).expect("read");
            assert!(respelled.contains("$PATH:$HOME/.bun/bin\""), "{respelled}");
            assert!(matches!(
                run("zsh", &home, &env, NO_INHERITED, &dirs).expect("run"),
                Written::Nothing
            ));
            assert_eq!(render("zsh", &[], &home).expect("render"), "");
            assert_eq!(render("fish", &[], &home).expect("render"), "");
        }

        #[test]
        fn a_block_whose_own_entries_the_daemon_was_started_with_is_kept() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let dirs = home_dirs(&home);
            let with_block = format!("export MINE=1\n{}", block_text(&block_body(&home)));
            fs::write(home.join(".zshrc"), &with_block).expect("write");

            let path = format!(
                "{local}:{bun}:{local}:{bun}",
                local = home.join(".local/bin").display(),
                bun = home.join(".bun/bin").display()
            );
            let inherited = home_dirs(&home);
            assert!(matches!(
                run("zsh", &home, &env_with_path(&path), &inherited, &dirs).expect("run"),
                Written::Nothing
            ));
            assert_eq!(
                fs::read_to_string(home.join(".zshrc")).expect("read"),
                with_block
            );

            assert!(matches!(
                run("zsh", &home, &env_with_path(&path), NO_INHERITED, &dirs).expect("run"),
                Written::Removed
            ));
            assert_eq!(
                fs::read_to_string(home.join(".zshrc")).expect("read"),
                "export MINE=1\n"
            );
        }

        #[test]
        fn the_block_contributes_only_the_directories_it_names() {
            let home = Path::new("/home/o");
            let dirs = home_dirs(home);

            let earlier = format!(
                "export FOO=1\n{}",
                block_text("export PATH=\"$PATH:/home/o/.mahbot/bin:/home/o/.bun/bin\"")
            );
            assert_eq!(made(&earlier, &dirs, home), vec![0, 1]);

            let current = block_text(&block_body(home));
            assert_eq!(made(&current, &dirs, home), vec![1, 1]);
            let one = format!(
                "export FOO=1\n{}",
                block_text("export PATH=\"$PATH:$HOME/.bun/bin\"")
            );
            assert_eq!(made(&one, &dirs, home), vec![0, 1]);

            let doubled = format!("{current}{current}");
            assert_eq!(made(&doubled, &dirs, home), vec![2, 2]);
            let fish = block_text("set -gx PATH $PATH $HOME/.local/bin $HOME/.bun/bin");
            assert_eq!(made(&fish, &dirs, home), vec![1, 1]);

            let respelled = block_text("export PATH=\"$PATH:$HOME/.local/bin/:$HOME/.bun/bin\"");
            assert_eq!(made(&respelled, &dirs, home), vec![1, 1]);
            let absolute_separated =
                block_text("export PATH=\"$PATH:/home/o/.local/bin/:/home/o/.bun/bin\"");
            assert_eq!(made(&absolute_separated, &dirs, home), vec![1, 1]);
            let braces = block_text("export PATH=\"$PATH:${HOME}/.local/bin:~/.bun/bin\"");
            assert_eq!(made(&braces, &dirs, home), vec![1, 1]);
            let computed = block_text("export PATH=\"$PATH:$HOME/.$(echo local)/bin\"");
            assert_eq!(made(&computed, &dirs, home), vec![0, 0]);

            assert_eq!(made("export FOO=1\n", &dirs, home), vec![0, 0]);
        }

        #[test]
        fn an_earlier_block_is_taken_out_of_another_shells_startup_file() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let zshrc = home.join(".zshrc");
            let bashrc = home.join(".bashrc");
            let dirs = home_dirs(&home);
            let stale = block_text(&format!(
                "export PATH=\"$PATH:{}/.mahbot/bin:{}/.bun/bin\"",
                home.display(),
                home.display()
            ));
            fs::write(&zshrc, "export MINE=1\n").expect("write zshrc");
            fs::write(&bashrc, format!("export A=1\n{stale}export B=2\n")).expect("write bashrc");
            let env = OwnerEnv::new(Vec::new());

            let made = sweep(&home, &env, &zshrc, &dirs);
            assert_eq!(
                fs::read_to_string(&bashrc).expect("read bashrc"),
                "export A=1\nexport B=2\n"
            );
            assert_eq!(
                fs::read_to_string(&zshrc).expect("read zshrc"),
                "export MINE=1\n"
            );
            assert_eq!(made, vec![0, 1]);
            assert_eq!(sweep(&home, &env, &zshrc, &dirs), vec![0, 0]);

            fs::write(&bashrc, format!("export A=1\n{stale}export B=2\n")).expect("write bashrc");
            assert_eq!(
                sweep(&home, &env, &home.join(".bash_profile"), &dirs),
                vec![0, 1]
            );
            assert_eq!(
                fs::read_to_string(&bashrc).expect("read bashrc"),
                "export A=1\nexport B=2\n"
            );
        }

        /// The reachable shape of "no file the product can write": an interactive POSIX shell
        /// reads the file `$ENV` names, so one that names nothing leaves it none. An earlier block
        /// may be the only thing keeping a directory on his search path, so nothing is taken out
        /// anywhere. (On the other side of the platform the same property is pinned by the test
        /// below: every refusal precedes the reads, whichever one it is.)
        #[cfg(not(target_os = "macos"))]
        #[test]
        fn a_shell_with_no_startup_file_the_product_can_write_keeps_every_block() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let dirs = home_dirs(&home);
            let stale = block_text(&format!(
                "export PATH=\"$PATH:{}/.mahbot/bin:{}/.bun/bin\"",
                home.display(),
                home.display()
            ));
            fs::write(home.join(".zshrc"), format!("export MINE=1\n{stale}")).expect("write zshrc");
            fs::write(home.join(".bashrc"), format!("export B=1\n{stale}")).expect("write bashrc");
            let env = OwnerEnv::new(Vec::new());
            let Err(reason) = run("sh", &home, &env, NO_INHERITED, &dirs) else {
                panic!("an interactive `sh` reads no startup file where `$ENV` names none");
            };
            assert!(reason.contains("startup file"), "{reason}");
            for file in [".zshrc", ".bashrc"] {
                let after = fs::read_to_string(home.join(file)).expect("read");
                assert!(
                    after.contains(stale.trim()),
                    "{file} keeps its block: {after}"
                );
            }
        }

        #[test]
        fn a_block_that_cannot_be_rendered_leaves_the_other_files_alone() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let env = OwnerEnv::new(Vec::new());
            let elsewhere = home_dirs(&dir.path().join("other"));
            let stale = block_text("export PATH=\"$PATH:/other/.bun/bin\"");
            fs::write(home.join(".bashrc"), format!("export B=1\n{stale}")).expect("write bashrc");

            let Err(reason) = run("zsh", &home, &env, NO_INHERITED, &elsewhere) else {
                panic!("the directories are not under this home, so nothing can be written");
            };
            assert!(reason.contains("not under the owner's home"), "{reason}");
            assert!(
                fs::read_to_string(home.join(".bashrc"))
                    .expect("read bashrc")
                    .contains(".bun/bin"),
                "the earlier block is still there"
            );
        }

        #[test]
        fn todays_own_file_is_not_swept_as_another_shells_file() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            let zdot = home.join(".config").join("zsh");
            fs::create_dir_all(&zdot).expect("mkdir zdot");
            let env = OwnerEnv::new(vec![(
                OsString::from("ZDOTDIR"),
                zdot.clone().into_os_string(),
            )]);
            let dirs = home_dirs(&home);
            fs::write(
                zdot.join(".zshrc"),
                format!("export MINE=1\n{}", block_text(&block_body(&home))),
            )
            .expect("write zshrc");
            std::os::unix::fs::symlink(zdot.join(".zshrc"), home.join(".zshrc"))
                .expect("symlink zshrc");

            let candidates = candidates(&env, &home, &zdot.join(".zshrc"));
            assert!(
                !candidates.contains(&("zsh", home.join(".zshrc"))),
                "the symlinked spelling is today's own file, not another one"
            );
            assert_eq!(sweep(&home, &env, &zdot.join(".zshrc"), &dirs), vec![0, 0]);
            assert!(
                fs::read_to_string(zdot.join(".zshrc"))
                    .expect("read zshrc")
                    .contains("MINE"),
                "his file is untouched"
            );
        }

        #[test]
        fn the_zsh_file_of_the_previous_arrangement_is_swept_even_under_zdotdir() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            let zdot = dir.path().join("zdot");
            fs::create_dir_all(&home).expect("mkdir home");
            fs::create_dir_all(&zdot).expect("mkdir zdot");
            let env = OwnerEnv::new(vec![(
                OsString::from("ZDOTDIR"),
                zdot.clone().into_os_string(),
            )]);
            let dirs = home_dirs(&home);
            let stale = block_text("export PATH=\"$PATH:$HOME/.mahbot/bin\"");
            fs::write(home.join(".zshrc"), format!("export A=1\n{stale}")).expect("write zshrc");
            fs::write(zdot.join(".zshrc"), "export MINE=1\n").expect("write zdot zshrc");

            let candidates = candidates(&env, &home, &zdot.join(".zshrc"));
            assert!(
                candidates.contains(&("zsh", home.join(".zshrc"))),
                "the plain zsh file is a candidate"
            );
            assert_eq!(sweep(&home, &env, &zdot.join(".zshrc"), &dirs), vec![0, 0]);
            assert_eq!(
                fs::read_to_string(home.join(".zshrc")).expect("read zshrc"),
                "export A=1\n"
            );
            assert_eq!(
                fs::read_to_string(zdot.join(".zshrc")).expect("read zdot"),
                "export MINE=1\n"
            );
        }

        #[test]
        fn write_atomically_creates_a_missing_parent_directory() {
            let dir = TempDir::new().expect("tempdir");
            let target = dir.path().join(".config").join("fish").join("config.fish");
            write_atomically(&target, "new").expect("write");
            assert_eq!(fs::read_to_string(&target).expect("read"), "new");
            assert!(!target.with_file_name("config.fish.mahbot_tmp").exists());
        }

        #[cfg(unix)]
        #[test]
        fn write_atomically_preserves_the_targets_mode() {
            use std::os::unix::fs::PermissionsExt as _;

            let dir = TempDir::new().expect("tempdir");
            let target = dir.path().join(".zshrc");
            fs::write(&target, "old").expect("write");
            fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).expect("chmod");

            write_atomically(&target, "new").expect("write");
            assert_eq!(fs::read_to_string(&target).expect("read"), "new");
            assert_eq!(
                fs::metadata(&target).expect("stat").permissions().mode() & 0o777,
                0o600
            );
            assert!(!dir.path().join(".zshrc.mahbot_tmp").exists());
        }

        #[cfg(unix)]
        #[test]
        fn write_atomically_writes_through_a_symlink() {
            let dir = TempDir::new().expect("tempdir");
            let real = dir.path().join("dotfiles").join("zshrc");
            fs::create_dir_all(real.parent().expect("parent")).expect("mkdir");
            fs::write(&real, "old").expect("write real");
            let link = dir.path().join(".zshrc");
            std::os::unix::fs::symlink(&real, &link).expect("symlink");

            write_atomically(&link, "new").expect("write");
            assert!(
                fs::symlink_metadata(&link)
                    .expect("stat link")
                    .file_type()
                    .is_symlink(),
                "the link is still a link"
            );
            assert_eq!(fs::read_to_string(&real).expect("read real"), "new");
        }

        #[cfg(unix)]
        #[test]
        fn write_atomically_fills_in_a_dangling_symlink() {
            let dir = TempDir::new().expect("tempdir");
            let real = dir.path().join("dotfiles").join("zshrc");
            fs::create_dir_all(real.parent().expect("parent")).expect("mkdir");
            let link = dir.path().join(".zshrc");
            std::os::unix::fs::symlink(&real, &link).expect("symlink");

            write_atomically(&link, "new").expect("write");
            assert!(
                fs::symlink_metadata(&link)
                    .expect("stat link")
                    .file_type()
                    .is_symlink(),
                "the link is still a link"
            );
            assert_eq!(fs::read_to_string(&real).expect("read real"), "new");
        }

        #[cfg(unix)]
        #[test]
        fn a_leftover_temp_file_is_removed() {
            let dir = TempDir::new().expect("tempdir");
            let target = dir.path().join(".zshrc");
            fs::write(&target, "export MINE=1\n").expect("write zshrc");
            let tmp = temp_path(&target).expect("temp path");
            fs::write(&tmp, "half a block").expect("write temp");

            remove_leftover_temp(&target);
            assert!(!tmp.exists());
            assert_eq!(
                fs::read_to_string(&target).expect("read zshrc"),
                "export MINE=1\n"
            );

            let elsewhere = dir.path().join("elsewhere");
            fs::write(&elsewhere, "his own file").expect("write elsewhere");
            std::os::unix::fs::symlink(&elsewhere, &tmp).expect("symlink temp");
            write_atomically(&target, "export MINE=2\n").expect("write");
            assert!(!tmp.exists());
            assert_eq!(
                fs::read_to_string(&elsewhere).expect("read elsewhere"),
                "his own file"
            );
            assert_eq!(
                fs::read_to_string(&target).expect("read zshrc"),
                "export MINE=2\n"
            );
        }

        #[cfg(unix)]
        #[test]
        fn a_write_that_cannot_land_leaves_the_earlier_blocks_in_place() {
            use std::os::unix::fs::PermissionsExt as _;

            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let dirs = home_dirs(&home);
            let stale = block_text(&format!(
                "export PATH=\"$PATH:{}/.bun/bin\"",
                home.display()
            ));
            fs::write(home.join(".zshrc"), "export MINE=1\n").expect("write zshrc");
            fs::write(home.join(".bashrc"), format!("export B=1\n{stale}")).expect("write bashrc");
            fs::set_permissions(&home, fs::Permissions::from_mode(0o500)).expect("chmod home");

            let env = OwnerEnv::new(Vec::new());
            assert!(
                run("zsh", &home, &env, NO_INHERITED, &dirs).is_err(),
                "the write cannot land"
            );
            let after = fs::read_to_string(home.join(".bashrc")).expect("read bashrc");
            assert!(
                after.contains(".bun/bin"),
                "the earlier block is still there: {after}"
            );
            assert_eq!(
                fs::read_to_string(home.join(".zshrc")).expect("read zshrc"),
                "export MINE=1\n"
            );

            fs::set_permissions(&home, fs::Permissions::from_mode(0o700)).expect("chmod back");
        }

        #[test]
        fn target_file_maps_the_shells_whose_startup_file_is_known() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let env = OwnerEnv::new(Vec::new());

            assert_eq!(
                target_file("zsh", &home, &env).expect("zsh"),
                home.join(".zshrc")
            );
            assert_eq!(
                target_file("fish", &home, &env).expect("fish"),
                home.join(".config/fish/config.fish")
            );
            let moved = OwnerEnv::new(vec![(
                OsString::from("XDG_CONFIG_HOME"),
                home.join("cfg").into_os_string(),
            )]);
            assert_eq!(
                target_file("fish", &home, &moved).expect("fish"),
                home.join("cfg/fish/config.fish")
            );
            let empty = OwnerEnv::new(vec![(OsString::from("XDG_CONFIG_HOME"), OsString::new())]);
            assert_eq!(
                target_file("fish", &home, &empty).expect("fish"),
                home.join(".config/fish/config.fish")
            );
            assert!(target_file("csh", &home, &env).is_err());
            assert!(target_file("tcsh", &home, &env).is_err());

            if cfg!(target_os = "macos") {
                assert_eq!(
                    target_file("bash", &home, &env).expect("bash"),
                    home.join(".bash_profile")
                );
                fs::write(home.join(".profile"), "").expect("write profile");
                assert_eq!(
                    target_file("bash", &home, &env).expect("bash"),
                    home.join(".profile")
                );
            } else {
                assert_eq!(
                    target_file("bash", &home, &env).expect("bash"),
                    home.join(".bashrc")
                );
            }
        }

        #[test]
        fn target_file_of_zsh_follows_zdotdir() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");
            let zdot = dir.path().join("zdot");
            let env = OwnerEnv::new(vec![(
                OsString::from("ZDOTDIR"),
                zdot.clone().into_os_string(),
            )]);

            assert_eq!(
                target_file("zsh", &home, &env).expect("zsh"),
                zdot.join(".zshrc")
            );
            let empty = OwnerEnv::new(vec![(OsString::from("ZDOTDIR"), OsString::new())]);
            assert_eq!(
                target_file("zsh", &home, &empty).expect("zsh"),
                home.join(".zshrc")
            );
        }

        #[test]
        fn target_file_for_the_posix_shells_is_the_profile_or_the_env_file() {
            let dir = TempDir::new().expect("tempdir");
            let home = dir.path().join("home");
            fs::create_dir_all(&home).expect("mkdir home");

            if cfg!(target_os = "macos") {
                assert_eq!(
                    target_file("sh", &home, &OwnerEnv::new(Vec::new())).expect("sh"),
                    home.join(".profile")
                );
                assert_eq!(
                    target_file("ksh", &home, &OwnerEnv::new(Vec::new())).expect("ksh"),
                    home.join(".profile")
                );
            } else {
                assert!(target_file("sh", &home, &OwnerEnv::new(Vec::new())).is_err());
                let startup = dir.path().join("startup");
                fs::write(&startup, "").expect("write startup");
                let env = OwnerEnv::new(vec![(
                    OsString::from("ENV"),
                    startup.clone().into_os_string(),
                )]);
                assert_eq!(target_file("sh", &home, &env).expect("sh"), startup);
            }
        }

        #[test]
        fn block_regions_keep_the_owners_own_lines_whole() {
            let home = Path::new("/home/o");
            let body = block_body(home);
            let block = block_text(&body);
            let doubled = format!("{block}{block}");
            assert_eq!(block_regions(&doubled).len(), 2);
            assert_eq!(strip_regions(&doubled, &block_regions(&doubled)), "");

            let unclosed = format!("export KEEP=1\n{BLOCK_START}\nexport KEEP=2\n");
            assert!(block_regions(&unclosed).is_empty());
            assert_eq!(
                strip_regions(&unclosed, &block_regions(&unclosed)),
                unclosed
            );

            let stray = format!("{BLOCK_START}\nexport KEEP=1\n{block}");
            assert_eq!(
                strip_regions(&stray, &block_regions(&stray)),
                format!("{BLOCK_START}\nexport KEEP=1\n")
            );

            let between = format!(
                "export A=1\n{block}export B=2\n{BLOCK_START}\n{body}\n{BLOCK_END}\nexport C=3\n"
            );
            assert_eq!(
                strip_regions(&between, &block_regions(&between)),
                "export A=1\nexport B=2\nexport C=3\n"
            );
        }

        #[test]
        fn block_regions_never_take_a_pair_the_owner_wrote() {
            let home = Path::new("/home/o");
            let body = block_body(home);
            let owned = format!("{BLOCK_START}\nexport KEEP=1\n{BLOCK_END}\n");
            assert!(block_regions(&owned).is_empty());
            assert_eq!(strip_regions(&owned, &block_regions(&owned)), owned);

            let his_then_body = format!("{BLOCK_START}\nexport KEEP=1\n{body}\n{BLOCK_END}\n");
            assert!(block_regions(&his_then_body).is_empty());
            assert_eq!(
                strip_regions(&his_then_body, &block_regions(&his_then_body)),
                his_then_body
            );
            let body_then_his = format!("{BLOCK_START}\n{body}\nexport KEEP=1\n{BLOCK_END}\n");
            assert!(block_regions(&body_then_his).is_empty());
            assert_eq!(
                strip_regions(&body_then_his, &block_regions(&body_then_his)),
                body_then_his
            );

            let genuine = block_text(&block_body(home));
            assert_eq!(strip_regions(&genuine, &block_regions(&genuine)), "");

            let with_append = format!("{}{owned}", block_text(&block_body(home)));
            assert_eq!(
                strip_regions(&with_append, &block_regions(&with_append)),
                owned
            );
        }

        /// `install.sh` writes this product's own block and the product then keeps it in step,
        /// so the two must spell it the same way: markers byte for byte, and a body line this
        /// module reads back naming the folder the product's update moves itself into.
        #[test]
        fn the_install_script_writes_the_product_s_own_block() {
            let script =
                std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh"))
                    .expect("install.sh is part of the tree");

            assert!(
                script.contains(&format!("BLOCK_START='{BLOCK_START}'")),
                "install.sh's opening marker: {BLOCK_START}"
            );
            assert!(
                script.contains(&format!("BLOCK_END='{BLOCK_END}'")),
                "install.sh's closing marker: {BLOCK_END}"
            );
            assert!(
                script.contains(&format!("SH_BODY_PREFIX='{SH_BODY_PREFIX}'")),
                "install.sh's body-line opening: {SH_BODY_PREFIX}"
            );
            assert!(
                script.contains(&format!("SH_BODY_SUFFIX='{SH_BODY_SUFFIX}'")),
                "install.sh's body-line closing: {SH_BODY_SUFFIX}"
            );
            assert!(
                script.contains(&format!("FISH_BODY_PREFIX='{FISH_BODY_PREFIX}'")),
                "install.sh's fish body-line opening: {FISH_BODY_PREFIX}"
            );

            let home = directories::UserDirs::new().expect("a home directory");
            let home = home.home_dir();
            let install = crate::util::managed_bin::mahbot_install_dir().expect("a home directory");
            let body = |name: &str| -> String {
                script
                    .lines()
                    .find_map(|line| line.strip_prefix(name)?.strip_suffix('\''))
                    .unwrap_or_else(|| panic!("install.sh must set {name}…'"))
                    .to_string()
            };
            for name in ["SH_BODY='", "FISH_BODY='"] {
                let line = body(name);
                let named: Vec<String> = body_entries(&line)
                    .into_iter()
                    .map(|entry| expand_home(entry, home))
                    .collect();
                assert_eq!(
                    named,
                    vec![install.to_string_lossy().to_string()],
                    "install.sh's {name} line must name the product's own install directory"
                );
            }
        }
    }
}

/// The reasons one sync records are compared against the sync before it — the bookkeeping
/// both platforms share — so it is exercised here.
#[cfg(all(test, any(unix, windows)))]
mod failure_tests {
    use super::{FAILURES, begin_sync, end_sync, record_failure};
    use crate::util::UnwrapPoison;

    #[test]
    fn each_failure_of_one_sync_is_recorded_and_survives_the_sync() {
        /// The reasons the next sync would compare against.
        fn recorded() -> Vec<String> {
            FAILURES.lock().unwrap_poison().previous.clone()
        }

        begin_sync();
        record_failure("the first reason");
        record_failure("the second reason");
        end_sync();
        assert_eq!(recorded(), vec!["the first reason", "the second reason"]);

        begin_sync();
        record_failure("the first reason");
        end_sync();
        assert_eq!(recorded(), vec!["the first reason"]);

        begin_sync();
        end_sync();
        assert!(recorded().is_empty());
    }
}

/// The `Path` value the owner's own environment must end up holding, or `None` when the value
/// already is what it must be. Pure text work, so these rules hold on every host.
///
/// Each of the product's own directories keeps its first entry where it is and loses every later
/// one, so his order is never changed and no directory is named twice; a directory with no entry
/// is added as the literal path the platform resolved, but only when his own environment carries
/// it from nowhere else. `raw` is the value as the key holds it, or `None` when the key holds no
/// value at all — the two are not the same state, because a value that is there keeps the empty
/// entries it has (an empty entry names the current directory, his own arrangement). The
/// [`Written`] says whether the write puts a directory on his search path or only takes the
/// product's own duplicates out.
#[cfg(any(windows, test))]
fn planned_path_value(
    raw: Option<&str>,
    env: &OwnerEnv,
    dirs: &[PathBuf],
) -> Result<Option<(String, Written)>, String> {
    let entries: Vec<&str> = raw.map_or_else(Vec::new, |raw| raw.split(';').collect());

    let mut kept: Vec<&str> = Vec::with_capacity(entries.len());
    let mut present = vec![false; dirs.len()];
    for entry in entries {
        // The first entry naming one of the directories stays where it is; every later one is a
        // duplicate.
        if let Some(index) = dirs.iter().position(|dir| entry_names(entry, dir, env))
            && std::mem::replace(&mut present[index], true)
        {
            continue;
        }
        kept.push(entry);
    }

    let mut written = Written::Removed;
    let mut value = kept.join(";");
    // Whether the value already holds an entry: the owner's own entries are always still there —
    // an empty one included, which he is appended to rather than written over — while a key
    // holding no value has none.
    let mut holds_an_entry = !kept.is_empty();
    for (index, dir) in dirs.iter().enumerate() {
        if present[index] || !dir_missing(env, dir, 0) {
            continue;
        }
        // `%` expands and `;` splits the entry the owner's environment reads.
        let text = dir.to_string_lossy();
        if text.contains(['%', ';']) {
            return Err(UNSAFE_DIRECTORY.to_string());
        }
        if holds_an_entry {
            value.push(';');
        }
        value.push_str(&text);
        holds_an_entry = true;
        written = Written::Added;
    }
    // What the value was, absence included: a key holding none with nothing to add is left alone,
    // because writing would create exactly the empty entry the product must never leave behind.
    let changed = match raw {
        Some(raw) => value != raw,
        None => !value.is_empty(),
    };
    Ok(changed.then_some((value, written)))
}

/// Whether a raw `Path` entry names `dir`: compared the way that platform's own lookup compares
/// them, with the references the entry contains resolved from the owner's own environment — what
/// the platform and the helper's own installer do before using or comparing one. Only ever used
/// to recognise an entry: a reference is never written out expanded.
#[cfg(any(windows, test))]
#[must_use]
fn entry_names(entry: &str, dir: &Path, env: &OwnerEnv) -> bool {
    same_entry_windows(Path::new(entry), dir)
        || same_entry_windows(Path::new(&resolve_references(entry, env)), dir)
}

/// `text` with every `%NAME%` the owner's environment defines replaced by its value. A `%`
/// without a partner, and an undefined name, are left as they stand.
#[cfg(any(windows, test))]
#[must_use]
fn resolve_references(text: &str, env: &OwnerEnv) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('%') {
        let after = &rest[start + 1..];
        let name = after.find('%').map(|end| &after[..end]);
        let value = name.filter(|name| !name.is_empty()).and_then(|name| {
            env.vars()
                .iter()
                .find(|(var, _)| var.to_string_lossy().eq_ignore_ascii_case(name))
                .map(|(_, value)| value.to_string_lossy().into_owned())
        });
        if let (Some(name), Some(value)) = (name, value) {
            out.push_str(&rest[..start]);
            out.push_str(&value);
            rest = &after[name.len() + 1..];
        } else {
            out.push_str(&rest[..=start]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Windows has no startup file to append to: the owner's own `Path` value lives in his
/// environment key, which is the place `setx` writes and a terminal opened afterwards reads.
#[cfg(windows)]
mod windows {
    use std::path::PathBuf;

    use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_EXPAND_SZ, REG_SZ,
        REG_VALUE_TYPE, RegCloseKey, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        HWND_BROADCAST, SMTO_ABORTIFHUNG, SendMessageTimeoutW, WM_SETTINGCHANGE,
    };

    use super::{OwnerEnv, PATH_VISIBLE_MESSAGE, Written, planned_path_value, record_failure};
    use crate::util::registry::{os_reason, utf16_text, wide};

    /// The per-user environment key. The machine-wide environment is deliberately left alone:
    /// nothing here is the owner's own unless it is per-user.
    const ENVIRONMENT_KEY: &str = "Environment";

    /// The value name, spelled `Path`; value names are case-insensitive there anyway.
    const PATH_NAME: &str = "Path";

    /// How long the change broadcast waits for one window to answer: generous for a busy program,
    /// bounded so a hung one cannot hold the reader.
    const BROADCAST_TIMEOUT_MS: u32 = 5_000;

    /// The owner's own `Path` value as the key holds it: its type and its raw text.
    struct RawValue {
        kind: REG_VALUE_TYPE,
        text: String,
    }

    /// Bring the owner's own `Path` value in line with `dirs`.
    pub(super) fn sync(env: &OwnerEnv, dirs: &[PathBuf]) {
        match run(env, dirs) {
            // A write into his own environment happens once, when the value is first needed or has
            // to change, so this is the audit line for a change to it.
            Ok(Written::Added) => {
                tracing::info!("{PATH_VISIBLE_MESSAGE}");
            }
            Ok(Written::Removed) => {
                tracing::info!(
                    "removed duplicate entries naming the product's own tools from the owner's \
                     own search path"
                );
            }
            Ok(Written::Nothing) => {}
            Err(reason) => record_failure(&reason),
        }
    }

    /// Read the value, decide what it must become, and store it — the key closed on every path out.
    fn run(env: &OwnerEnv, dirs: &[PathBuf]) -> Result<Written, String> {
        let key = open_key()?;
        let outcome = match plan(key, env, dirs) {
            Ok(Some(planned)) => store(key, planned.kind, &planned.value).map(|()| planned.written),
            Ok(None) => Ok(Written::Nothing),
            Err(reason) => Err(reason),
        };
        // SAFETY: `key` came from `open_key` and nothing uses it after this.
        unsafe { RegCloseKey(key) };
        outcome
    }

    /// Open `HKCU\Environment` for reading and writing.
    fn open_key() -> Result<HKEY, String> {
        let name = wide(ENVIRONMENT_KEY);
        let mut key: HKEY = 0;
        // SAFETY: `name` is a NUL-terminated UTF-16 string that outlives the
        // call, `key` is ours to fill, and the call has no other precondition.
        let status = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                name.as_ptr(),
                0,
                KEY_QUERY_VALUE | KEY_SET_VALUE,
                &raw mut key,
            )
        };
        if status != ERROR_SUCCESS {
            return Err(os_reason(
                "the owner's own environment key could not be opened",
                status,
            ));
        }
        Ok(key)
    }

    /// The value the key must end up holding — its type and its text — or `None` when the value
    /// already is what it must be. The type is the one the owner's value already has, so a
    /// `REG_EXPAND_SZ` stays one and a reference he adds later still expands.
    fn plan(key: HKEY, env: &OwnerEnv, dirs: &[PathBuf]) -> Result<Option<Planned>, String> {
        let current = query_value(key)?;
        let kind = current.as_ref().map_or(REG_EXPAND_SZ, |value| value.kind);
        let raw = current.as_ref().map(|value| value.text.as_str());
        Ok(
            planned_path_value(raw, env, dirs)?.map(|(value, written)| Planned {
                kind,
                value,
                written,
            }),
        )
    }

    /// What the key must end up holding.
    struct Planned {
        kind: REG_VALUE_TYPE,
        /// The value's new text.
        value: String,
        /// Whether the write puts a directory on his search path or only takes duplicates out.
        written: Written,
    }

    /// The owner's own `Path` value, or `None` when the key holds none. The size is queried first
    /// and the buffer allocated here, as the API documents, so nothing is left to free.
    fn query_value(key: HKEY) -> Result<Option<RawValue>, String> {
        let name = wide(PATH_NAME);
        let mut kind: REG_VALUE_TYPE = 0;
        let mut size: u32 = 0;
        // SAFETY: a size query — no data pointer — as the API documents, with
        // `name` outliving the call and the two out-parameters ours.
        let status = unsafe {
            RegQueryValueExW(
                key,
                name.as_ptr(),
                std::ptr::null(),
                &raw mut kind,
                std::ptr::null_mut(),
                &raw mut size,
            )
        };
        if status == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        if status != ERROR_SUCCESS {
            return Err(os_reason(
                "the owner's own Path value could not be read",
                status,
            ));
        }
        // The product edits a string value; anything else is left alone.
        if !matches!(kind, REG_SZ | REG_EXPAND_SZ) {
            return Err("the owner's own Path value is not one the product can edit".to_string());
        }
        let capacity = size as usize;
        let mut data = vec![0u8; capacity];
        let mut read = size;
        // SAFETY: `data` holds `capacity` bytes and `read` tells the API how many
        // of them it may fill; `name` outlives the call.
        let status = unsafe {
            RegQueryValueExW(
                key,
                name.as_ptr(),
                std::ptr::null(),
                &raw mut kind,
                data.as_mut_ptr(),
                &raw mut read,
            )
        };
        if status != ERROR_SUCCESS {
            return Err(os_reason(
                "the owner's own Path value could not be read",
                status,
            ));
        }
        Ok(Some(RawValue {
            kind,
            text: utf16_text(&data),
        }))
    }

    /// Store the value with the type it already has, then tell running programs the environment
    /// changed.
    fn store(key: HKEY, kind: REG_VALUE_TYPE, text: &str) -> Result<(), String> {
        let name = wide(PATH_NAME);
        let mut data: Vec<u16> = text.encode_utf16().collect();
        data.push(0);
        let Ok(bytes) = u32::try_from(data.len() * std::mem::size_of::<u16>()) else {
            return Err("the owner's own Path value is too long to store".to_string());
        };
        // SAFETY: `name` and `data` are the NUL-terminated UTF-16 value name and
        // value data the API reads, and both outlive the call.
        let status = unsafe {
            RegSetValueExW(
                key,
                name.as_ptr(),
                0,
                kind,
                data.as_ptr().cast::<u8>(),
                bytes,
            )
        };
        if status != ERROR_SUCCESS {
            return Err(os_reason(
                "the owner's own Path value could not be stored",
                status,
            ));
        }
        broadcast_change();
        Ok(())
    }

    /// Tell running programs the environment changed — the same broadcast `setx` makes. Best
    /// effort: a window that does not answer in time is not a failure of the write.
    fn broadcast_change() {
        let payload = wide(ENVIRONMENT_KEY);
        // SAFETY: `payload` is a NUL-terminated UTF-16 string that outlives the
        // call; the send is bounded by the timeout and its result is not used.
        unsafe {
            SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                0,
                payload.as_ptr() as isize,
                SMTO_ABORTIFHUNG,
                BROADCAST_TIMEOUT_MS,
                std::ptr::null_mut(),
            );
        }
    }
}

/// The rules [`planned_path_value`] applies are pure text work: the value is always split the
/// Windows way, because that is what reads it.
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    /// The two directories the block names, written the way this host compares paths.
    fn dirs() -> Vec<PathBuf> {
        vec![
            PathBuf::from("/home/o/.local/bin"),
            PathBuf::from("/home/o/.bun/bin"),
        ]
    }

    fn environment(vars: &[(&str, &str)]) -> OwnerEnv {
        OwnerEnv::new(
            vars.iter()
                .map(|(name, value)| (OsString::from(name), OsString::from(value)))
                .collect(),
        )
    }

    /// A `PATH` value written the way the shell that produced it writes one on this host.
    fn path_var(entries: &[&str]) -> String {
        std::env::join_paths(entries)
            .expect("join paths")
            .to_string_lossy()
            .into_owned()
    }

    /// An environment whose `PATH` holds the system directory only.
    fn own_env() -> OwnerEnv {
        environment(&[("PATH", "/usr/bin"), ("HOME", "/home/o")])
    }

    /// The planned value alone; whether the write adds a directory has its own test.
    fn planned(raw: Option<&str>, env: &OwnerEnv, dirs: &[PathBuf]) -> Option<String> {
        planned_path_value(raw, env, dirs)
            .expect("plan")
            .map(|(value, _)| value)
    }

    #[test]
    fn the_owners_own_order_and_his_references_survive() {
        assert_eq!(
            planned(Some("C:\\tools;%HOME%/.bun/bin"), &own_env(), &dirs()),
            Some("C:\\tools;%HOME%/.bun/bin;/home/o/.local/bin".to_string())
        );
    }

    #[test]
    fn a_duplicate_entry_is_collapsed_and_never_re_appended() {
        let env = environment(&[
            ("PATH", &path_var(&["/home/o/.local/bin", "/usr/bin"])),
            ("HOME", "/home/o"),
        ]);
        assert_eq!(
            planned(Some("%HOME%/.bun/bin;/home/o/.bun/bin"), &env, &dirs()),
            Some("%HOME%/.bun/bin".to_string())
        );
    }

    #[test]
    fn nothing_is_written_when_the_value_already_says_it() {
        assert_eq!(
            planned(
                Some("/home/o/.local/bin;%HOME%/.bun/bin"),
                &own_env(),
                &dirs()
            ),
            None
        );
        let visible = environment(&[
            (
                "PATH",
                &path_var(&["/home/o/.local/bin", "/home/o/.bun/bin", "/usr/bin"]),
            ),
            ("HOME", "/home/o"),
        ]);
        assert_eq!(planned(Some("C:\\tools;;D:\\x"), &visible, &dirs()), None);
    }

    #[test]
    fn a_directory_that_cannot_be_written_is_refused_rather_than_written() {
        for unsafe_dir in ["/home/o/%TMP%/bin", "/home/o/a;b"] {
            let dirs = vec![PathBuf::from(unsafe_dir)];
            assert!(
                planned_path_value(Some("C:\\tools"), &own_env(), &dirs).is_err(),
                "{unsafe_dir}"
            );
        }
    }

    #[test]
    fn a_write_that_only_takes_duplicates_out_is_not_an_addition() {
        let written = |raw: &str, env: &OwnerEnv| {
            planned_path_value(Some(raw), env, &dirs())
                .expect("plan")
                .map(|(_, written)| written)
        };
        assert_eq!(
            written(
                "/home/o/.local/bin;/home/o/.local/bin;%HOME%/.bun/bin",
                &own_env()
            ),
            Some(Written::Removed)
        );
        assert_eq!(
            written("/home/o/.local/bin", &own_env()),
            Some(Written::Added)
        );
        let visible = environment(&[
            ("PATH", &path_var(&["/home/o/.bun/bin", "/usr/bin"])),
            ("HOME", "/home/o"),
        ]);
        assert_eq!(written("/home/o/.local/bin", &visible), None);
    }

    #[test]
    fn an_empty_entry_is_kept_and_a_missing_directory_goes_after_it() {
        assert_eq!(
            planned(Some(""), &own_env(), &dirs()),
            Some(";/home/o/.local/bin;/home/o/.bun/bin".to_string())
        );
        assert_eq!(
            planned(Some(";C:\\tools"), &own_env(), &dirs()),
            Some(";C:\\tools;/home/o/.local/bin;/home/o/.bun/bin".to_string())
        );
    }

    #[test]
    fn a_value_that_is_not_there_names_only_the_product_directories() {
        assert_eq!(
            planned(None, &own_env(), &dirs()),
            Some("/home/o/.local/bin;/home/o/.bun/bin".to_string())
        );
        let visible = environment(&[
            (
                "PATH",
                &path_var(&["/home/o/.local/bin", "/home/o/.bun/bin"]),
            ),
            ("HOME", "/home/o"),
        ]);
        assert_eq!(planned(None, &visible, &dirs()), None);
        assert_eq!(
            planned_path_value(None, &own_env(), &dirs())
                .expect("plan")
                .map(|(_, written)| written),
            Some(Written::Added)
        );
    }

    #[test]
    fn an_entry_is_recognised_the_way_the_platform_expands_it() {
        let env = environment(&[("USERPROFILE", "C:\\Users\\o")]);
        let dir = PathBuf::from("C:\\Users\\o\\Programs\\chrome-use");
        assert!(entry_names(
            "C:\\Users\\o\\Programs\\chrome-use\\",
            &dir,
            &env
        ));
        assert!(entry_names(
            "%USERPROFILE%\\Programs\\chrome-use",
            &dir,
            &env
        ));
        assert!(entry_names(
            "C:\\USERS\\O\\programs\\CHROME-USE",
            &dir,
            &env
        ));
        assert!(!entry_names("C:\\Users\\o\\Programs\\other", &dir, &env));
        assert!(!entry_names("", &dir, &env));

        assert_eq!(resolve_references("a%NOPE%b", &env), "a%NOPE%b");
        assert_eq!(resolve_references("a%b", &env), "a%b");
        assert_eq!(
            resolve_references("%USERPROFILE%\\x", &env),
            "C:\\Users\\o\\x"
        );
    }
}

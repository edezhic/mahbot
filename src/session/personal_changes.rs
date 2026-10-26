//! Between-turns change report for a user's personal workspace: what was added,
//! changed or removed there between turns, at the head of the message every
//! Assistant turn answers ([`prepend_changes`]) — an automation's, a notice's or a
//! delegation's turn included — so it never answers from stale knowledge.
//!
//! The observed state lives in a row of its own — `session_personal_changes`, a
//! child of the Assistant's `session_metadata` row
//! ([`crate::session::assistant_agent_id`]), written there and only when it
//! changes, and deleted with that row — never inside the watched folder, which
//! stays the user's alone. [`refresh_after_turn`] rewrites it at the end of
//! the Assistant's own turn, so what it absorbs is everything that moved during
//! that turn: the product's own writes, and equally an edit the user made while
//! the turn was running. Telling writers apart — a raw shell command included —
//! would cost more than that window is worth, and the block never claims the user
//! made a change; between turns the report is exact.
//!
//! Cost: one plain walk per observation, carrying size and mtime across turns so
//! an unchanged file is never read ([`scan`] says what is read, [`diff`] what that
//! decides). A walk of a personal folder holding tens of thousands of entries
//! measures ~0.12–0.15 s, and a turn pays for one on the reply path plus one at
//! its end; content is read only for the files whose stamp moved. The folder's
//! first observation, which does read every file once, does not ([`establish`]),
//! and while it runs the account gets no block at all.
//!
//! What the folder's own git rules exclude is not the user's material at all:
//! [`Rules`] answers that question while the change set is gathered, so an ignored
//! path never enters the delta in the first place — and never reads as removed,
//! however it came to be ignored.
//!
//! The session-start `<workspace-structure>` picture ([`personal_folder_picture`]) is
//! built from the same walk predicate ([`is_skipped`]) and the same rules
//! ([`Rules`]) as the report, so the two blocks can never describe different
//! material; it costs one metadata-only walk and never reads a file's contents.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ignore::{IncrementalIgnore, WalkBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::util::UnwrapPoison;
use crate::{Role, Workspace};

/// One entry's fingerprint. `digest` is `None` for an entry whose content was not
/// read (a newly seen path on the report path) or could not be (above
/// [`HASH_MAX_BYTES`], an I/O error, or an entry that is not a regular file);
/// [`diff`] says what that means.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileState {
    size: u64,
    mtime_ns: i64,
    digest: Option<u64>,
}

/// Relative path → fingerprint. `BTreeMap` keeps the serialized JSON stable so
/// "did anything move?" is a string comparison.
type Snapshot = BTreeMap<String, FileState>;

/// Files at or below this size are read and hashed whole; a larger one is judged
/// by its stamp alone, so a folder's multi-gigabyte blobs are never read at all.
/// The price is that a mere touch of such a file reports as a change — the
/// lesser evil against never noticing a real edit deep inside one we never read.
const HASH_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Read buffer for streaming a file through the hasher.
const READ_BUFFER_BYTES: usize = 64 * 1024;

/// A stamp the platform refused to report (a file older than the epoch, or one
/// whose stamp does not fit). It is not a fact about the file, so it can never
/// decide a comparison: such a file is read and its content decides.
const STAMP_UNKNOWN: i64 = i64::MIN;

/// Hash the whole content of `path` — never the mtime or the size. The first 8
/// bytes of SHA-256 are the digest: 64 bits tells one revision of a file from
/// another far beyond what a folder can hold, and keeps the stored baseline
/// small. `None` on any I/O error.
fn digest_of(path: &Path) -> Option<u64> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    Some(u64::from_be_bytes(head))
}

/// A file's mtime as nanoseconds since the epoch, or [`STAMP_UNKNOWN`] when the
/// platform reports none (a stamp before 1970, or one outside `i64`).
fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or(STAMP_UNKNOWN)
}

/// Whether the walker should skip `entry` (and, for a directory, its whole
/// subtree). Only what the product or the OS keeps in the personal folder is
/// pruned at any depth: the `.git` repository metadata, the `.compaction` store
/// and macOS's `.DS_Store`. A user's own dotfile — `.env`, `.gitignore`,
/// `.config/` — is the user's material and a hand edit to it must be reported
/// unless the folder's own git rules hide it ([`Rules`], the other half of the
/// walk predicate); the product's own things at the workspace root (`MEMORY.md`
/// and the `generated/`, `uploads/`, `notes/` and `archive/` directories) are
/// never watched either. `archive/` holds only what the assistant retired with
/// its `archive` tool, so a move into it is the assistant's own bookkeeping
/// rather than a change to the user's material — and the picture stays a sketch
/// of what the user has, not of what was put aside.
///
/// This predicate and [`Rules`] together are the definition of the user's file
/// material inside a personal folder — the report ([`scan`]) and the session-start
/// picture ([`personal_folder_picture`]) both walk through it, so the two can never
/// describe different material. The `shared/` catalogue
/// ([`is_owner_personal_folder`]) is product material only in the folder that
/// holds it, and the sibling enumerations ([`crate::tools::custom`],
/// `research_cleanup`) are named here so a new product-owned path lands in both
/// places.
fn is_skipped(entry: &ignore::DirEntry, owner_folder: bool) -> bool {
    if entry.depth() == 0 {
        return false; // the watched root itself
    }
    let Some(name) = entry.file_name().to_str() else {
        return false;
    };
    if name == ".DS_Store" {
        return true;
    }
    if matches!(name, ".git" | ".compaction") && entry.file_type().is_some_and(|t| t.is_dir()) {
        return true;
    }
    if entry.depth() == 1 {
        if name == "MEMORY.md" {
            return true;
        }
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            if matches!(name, "generated" | "uploads" | "notes" | "archive") {
                return true;
            }
            // The product's custom-tool catalogue lives in the owner's own
            // folder alone; in anyone else's personal folder the name `shared/`
            // is that user's own material, whose edits must be reported.
            if name == "shared" && owner_folder {
                return true;
            }
        }
    }
    false
}

/// Hard cap on how many child folders the session-start picture names; the rest
/// are summarized as a count, so the block never grows with the folder's size.
const MAX_PICTURED_FOLDERS: usize = 40;

/// A folder's own git ignore rules, for one observation — a cheap clonable handle,
/// since [`scan`] hands a clone to its walk's entry filter (which must own what it
/// captures) and still asks the same cached answers for the delta afterwards.
///
/// What is hidden is what git itself would ignore inside this folder, judged by
/// the rules that live in it: every repository the folder holds (itself included)
/// contributes its `.gitignore` files and its `.git/info/exclude`, at the
/// boundaries git applies them — a nested repository's rules replace the outer
/// one's for the paths inside it, which is why each repository gets a matcher of
/// its own rather than one rooted at the folder. Nothing above the folder is
/// applied (no parent `.gitignore`, no `core.excludesFile`, no global or system
/// excludes), `.ignore` files are not git rules, and hidden files are not an
/// ignored class. A folder holding no repository therefore hides nothing.
///
/// The one thing that outranks a rule is the index: a path git still tracks is
/// never hidden. Hiding also fails open — when the index cannot be read, the paths
/// its rules would hide stay visible rather than risk dropping a tracked one. That
/// index is the only git this module runs, and only for a path a rule would hide.
///
/// Everything is built on first use and cached for this one observation: a matcher
/// never notices an ignore file edited mid-observation, and the next starts clean.
#[derive(Clone)]
struct Rules {
    root: PathBuf,
    state: Arc<Mutex<RulesState>>,
}

#[derive(Default)]
struct RulesState {
    /// Repository (folder-relative) → the matcher over its own rules.
    matchers: HashMap<PathBuf, IncrementalIgnore>,
    /// Folder-relative directory → the repository governing it; `None` outside
    /// every repository. A walk asks this once per directory, not per entry.
    governing: HashMap<PathBuf, Option<PathBuf>>,
    /// Repository → the paths its index tracks, repository-relative and
    /// `/`-separated as git spells them. `None` when git could not answer.
    tracked: HashMap<PathBuf, Option<BTreeSet<String>>>,
}

impl Rules {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            state: Arc::new(Mutex::new(RulesState::default())),
        }
    }

    /// Whether the walk should hide `entry` — [`RulesState::hidden`] on the
    /// entry's folder-relative path. Every walk prunes through this, so the report
    /// and the picture can never disagree about material. An entry outside the
    /// folder (or the folder itself) is never hidden.
    fn hides(&self, entry: &ignore::DirEntry) -> bool {
        let Ok(rel) = entry.path().strip_prefix(&self.root) else {
            return false;
        };
        if rel.as_os_str().is_empty() {
            return false;
        }
        self.hidden(rel, entry.file_type().is_some_and(|t| t.is_dir()))
    }

    /// Whether `rel` (folder-relative) is hidden by the folder's own rules.
    fn hidden(&self, rel: &Path, is_dir: bool) -> bool {
        self.state
            .lock()
            .unwrap_poison()
            .hidden(&self.root, rel, is_dir)
    }
}

impl RulesState {
    /// Whether `rel` (folder-relative) is hidden: a path some repository's rules
    /// exclude and git does not track. A path git could not be asked about stays
    /// visible (see [`Rules`]).
    fn hidden(&mut self, root: &Path, rel: &Path, is_dir: bool) -> bool {
        if self.excluded(root, rel, is_dir) && self.tracked(root, rel, is_dir) == Some(false) {
            return true;
        }
        // A directory above `rel` can be hidden by a different repository than the
        // one governing `rel` — a nested repository inside a folder its enclosing
        // repository ignores — and the walk prunes there, so everything below is
        // gone from the change set for the same reason. Only a directory whose
        // repository differs is asked: within one repository the matcher has
        // already weighed every ancestor, and outside every repository there is
        // nothing to ask.
        let Some(governing) = self.repository(root, rel) else {
            return false;
        };
        let mut above = rel.parent();
        while let Some(dir) = above.filter(|dir| !dir.as_os_str().is_empty()) {
            if self.repository(root, dir).as_deref() != Some(governing.as_path())
                && self.excluded(root, dir, true)
                && self.tracked(root, dir, true) == Some(false)
            {
                return true;
            }
            above = dir.parent();
        }
        false
    }

    /// Whether the rules alone exclude `rel` — no index consulted, so a path the
    /// folder no longer holds is judged by the same rules that hid it.
    fn excluded(&mut self, root: &Path, rel: &Path, is_dir: bool) -> bool {
        let Some(repo) = self.repository(root, rel) else {
            return false;
        };
        let in_repo = rel.strip_prefix(&repo).unwrap_or(rel);
        self.matchers
            .entry(repo.clone())
            .or_insert_with(|| build_matcher(&root.join(&repo)))
            .matched(in_repo, is_dir)
            .is_ignore()
    }

    /// Whether git tracks `rel` (or, for a directory, any path beneath it), or
    /// `None` when the repository's index cannot be read.
    fn tracked(&mut self, root: &Path, rel: &Path, is_dir: bool) -> Option<bool> {
        let repo = self.repository(root, rel)?;
        let tracked = self
            .tracked
            .entry(repo.clone())
            .or_insert_with(|| load_tracked(&root.join(&repo)))
            .as_ref()?;
        let key = slash_path(rel.strip_prefix(&repo).unwrap_or(rel))?;
        if tracked.contains(&key) {
            return Some(true);
        }
        if !is_dir {
            return Some(false);
        }
        // A directory is kept when anything beneath it is tracked: its own entry
        // is not in the index, the files under it are.
        let prefix = format!("{key}/");
        Some(
            tracked
                .range(prefix.clone()..)
                .next()
                .is_some_and(|path| path.starts_with(&prefix)),
        )
    }

    /// The repository governing `rel` (folder-relative): the closest directory at
    /// or above its parent that holds a `.git` entry, never above `root` itself —
    /// `None` outside every repository. Each level's answer is cached, so the walk
    /// asks once per directory it enters.
    fn repository(&mut self, root: &Path, rel: &Path) -> Option<PathBuf> {
        let dir = rel.parent().unwrap_or(Path::new("")).to_path_buf();
        if let Some(known) = self.governing.get(&dir) {
            return known.clone();
        }
        let found = if crate::git::commands::is_git_repo(&root.join(&dir)) {
            Some(dir.clone())
        } else if dir.as_os_str().is_empty() {
            None
        } else {
            self.repository(root, &dir)
        };
        self.governing.insert(dir, found.clone());
        found
    }
}

/// The matcher for the rules in force inside one repository.
///
/// `standard_filters(false)` settles every default the walk needs — parents,
/// global excludes, `.ignore` files and the hidden class are all off — and the
/// two git sources of a repository's own rules are switched back on:
/// `require_git` stays at its default, so `.gitignore` files and
/// `.git/info/exclude` apply exactly where git applies them, a nested
/// repository's rules replacing these for the paths inside that nested one.
fn build_matcher(repo: &Path) -> IncrementalIgnore {
    let mut builder = WalkBuilder::new(repo);
    builder
        .standard_filters(false)
        .git_ignore(true)
        .git_exclude(true);
    builder
        .build_matchers()
        .pop()
        .expect("a walk root always yields its matcher")
}

/// The paths `repo`'s index tracks, repository-relative as git spells them, or
/// `None` when git cannot answer (its own doc says what that covers).
fn load_tracked(repo: &Path) -> Option<BTreeSet<String>> {
    let Some(tracked) = crate::git::commands::tracked_paths(repo) else {
        tracing::debug!(
            repo = %repo.display(),
            "personal-workspace repository index unreadable — its ignored paths stay visible"
        );
        return None;
    };
    Some(tracked.into_iter().collect())
}

/// `path`'s components joined by `/` — the spelling git prints and always uses,
/// so a tracked set and a walked path compare the same on every platform.
/// `None` for a path that is not UTF-8.
fn slash_path(path: &Path) -> Option<String> {
    let mut out = String::new();
    for component in path.components() {
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(component.as_os_str().to_str()?);
    }
    Some(out)
}

/// The session-start picture of `root`: the folders sitting directly inside it,
/// each with the number of files it holds at any depth, plus how many files sit
/// directly in the folder itself. No path is ever named, and the walk reads no
/// file's contents — metadata alone.
///
/// The material counted is exactly what the change report watches: the same
/// walk, the same predicate ([`is_skipped`]) and the same rules ([`Rules`]) as
/// [`scan`]. `None` when `root` does not exist yet (personal folders are created
/// lazily), when it cannot be opened, or when it holds nothing of the user's — so
/// the block is silent rather than empty. A walk error deeper inside warns and
/// continues, leaving the counts what could be counted.
pub(crate) fn personal_folder_picture(root: &Path) -> Option<String> {
    if let Err(e) = std::fs::read_dir(root) {
        if e.kind() == std::io::ErrorKind::NotFound {
            return None; // not created yet is not a fault
        }
        tracing::warn!(root = %root.display(), error = %e, "personal-workspace folder unreadable");
        return None;
    }
    let owner_folder = is_owner_personal_folder(root);
    let rules = Rules::new(root);
    let mut folders: BTreeMap<String, usize> = BTreeMap::new();
    let mut root_files = 0usize;
    for result in WalkBuilder::new(root)
        .standard_filters(false)
        .follow_links(false)
        .filter_entry(move |entry| !is_skipped(entry, owner_folder) && !rules.hides(entry))
        .build()
    {
        let entry = match result {
            Ok(entry) => entry,
            Err(e) => {
                // The folder itself was probed readable above, so an error here
                // is about one path: the other folders are still pictured.
                tracing::warn!(error = %e, "personal-workspace walk error");
                continue;
            }
        };
        if entry.depth() == 0 {
            continue; // the folder itself
        }
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
        };
        let Some(rel) = rel.to_str() else {
            continue; // a non-UTF-8 path is not material the report carries either
        };
        let Ok(meta) = std::fs::metadata(path).or_else(|_| std::fs::symlink_metadata(path)) else {
            continue; // an entry whose metadata cannot be read is not counted
        };
        if meta.is_dir() {
            // A real immediate child directory is the user's folder even when
            // nothing countable sits inside it; a link to a directory is neither
            // recorded nor descended into (the walk never follows one).
            if entry.depth() == 1 && entry.file_type().is_some_and(|t| t.is_dir()) {
                folders.entry(rel.to_string()).or_default();
            }
            continue;
        }
        match rel.split_once(std::path::MAIN_SEPARATOR) {
            Some((folder, _)) => *folders.entry(folder.to_string()).or_default() += 1,
            None => root_files += 1,
        }
    }
    if folders.is_empty() && root_files == 0 {
        return None; // only kept-aside areas, or nothing at all
    }
    Some(render_personal_folder_lines(&folders, root_files))
}

/// "1 file" / "3 files" — the plural a count line needs.
fn counted(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// Render the picture: one `- <name>/ — <n> files` line per folder, name-sorted
/// and at most [`MAX_PICTURED_FOLDERS`] of them, the folders left out summarized
/// as a count, and finally how many files sit directly in the folder.
fn render_personal_folder_lines(folders: &BTreeMap<String, usize>, root_files: usize) -> String {
    let mut out = String::new();
    for (name, count) in folders.iter().take(MAX_PICTURED_FOLDERS) {
        let _ = writeln!(out, "- {name}/ — {}", counted(*count, "file"));
    }
    let left = folders.len().saturating_sub(MAX_PICTURED_FOLDERS);
    if left > 0 {
        let _ = writeln!(out, "…and {} not named", counted(left, "folder"));
    }
    let _ = write!(
        out,
        "{} directly in the folder",
        counted(root_files, "file")
    );
    out
}

/// The path below the userspaces root when `path` lies inside it, `None` when it
/// does not. Both sides are compared canonically
/// ([`crate::stored_workspace_path`]): this relates a [`Workspace`]'s stored path
/// — canonical, and possibly behind a symlink — to the userspaces tree, and
/// `Path::starts_with` is lexical, so a storage root behind a symlink (macOS's
/// `/tmp` → `/private/tmp`, a symlinked `HOME`) would never match a raw prefix.
fn under_userspaces(path: &Path) -> Option<PathBuf> {
    let root = crate::stored_workspace_path(&crate::users::userspaces_root());
    crate::stored_workspace_path(path)
        .strip_prefix(root)
        .ok()
        .map(Path::to_path_buf)
}

/// The account whose personal folder `path` is or lies in — the folder itself or
/// anything under it (a delegated round, a targeted project round and a
/// custom-role launch all work below it, and a managed workspace placed inside
/// one counts too): the first component below the userspaces root, which is the
/// name [`crate::users::personal_workspace_path`] builds every personal folder
/// from.
fn personal_account_in(path: &Path) -> Option<String> {
    under_userspaces(path)?
        .components()
        .next()?
        .as_os_str()
        .to_str()
        .map(str::to_string)
}

/// Whether `root` is the owner's own personal folder — the one that holds the
/// product's custom-tool catalogue ([`crate::tools::custom`]), so that its
/// `shared/` is product material rather than the user's.
///
/// [`crate::users::personal_workspace_path`] builds every personal folder by
/// joining [`crate::users::userspaces_root`] with the account's name, and a walk's
/// root is always what that join returned, so undoing the join names the account
/// the walked folder belongs to. The comparison is component-wise
/// ([`Path::strip_prefix`]) — a trailing separator or a `.` component decides
/// nothing — and the account is tested with the product's single account test
/// ([`crate::users::is_admin_name`]). Only the owner's own folder counts: no
/// folder named after the owner anywhere else, and none below the real one, is
/// the product's, and a guest's own `shared/` stays watched.
///
/// The path is taken as it stands rather than canonically
/// ([`under_userspaces`]), and deliberately: canonicalizing moves an account
/// folder that is itself a link out of the userspaces tree, which would stop
/// pruning the catalogue that does live there (the walk follows such a root).
fn is_owner_personal_folder(root: &Path) -> bool {
    let Ok(rel) = root.strip_prefix(crate::users::userspaces_root()) else {
        return false;
    };
    let mut parts = rel.components();
    let account = parts.next().and_then(|c| c.as_os_str().to_str());
    parts.next().is_none() && account.is_some_and(crate::users::is_admin_name)
}

/// Dig the offending path out of an `ignore` walk error. The path lives in the
/// public `WithPath` variant, wrapped by `WithLineNumber`/`WithDepth`; `ignore`
/// 0.4.33 exposes no `path()` accessor of its own.
fn error_path(err: &ignore::Error) -> Option<&Path> {
    match err {
        ignore::Error::WithPath { path, .. } => Some(path),
        ignore::Error::WithLineNumber { err, .. } | ignore::Error::WithDepth { err, .. } => {
            error_path(err)
        }
        _ => None,
    }
}

/// Copy `rel`'s previous state — and, when it is a directory prefix, every
/// previous entry beneath it — into `out`, unchanged. Used wherever the walk
/// could not observe an entry at all, so a transient error can never read as the
/// user deleting files, and a later successful observation still reports the real
/// change. The snapshot is a sorted `BTreeMap`, so the `rel/` subtree is one
/// contiguous `range`.
fn freeze(previous: Option<&Snapshot>, rel: &str, out: &mut Snapshot) {
    let Some(previous) = previous else {
        return;
    };
    if let Some(state) = previous.get(rel) {
        out.insert(rel.to_string(), state.clone());
    }
    // Keys beginning with `rel/` sort contiguously between `rel/` and `rel0`
    // ('0' is the byte just past '/'), so the range cannot spill into a sibling
    // like `rel.txt`.
    for (path, state) in previous.range(format!("{rel}/")..format!("{rel}0")) {
        out.insert(path.clone(), state.clone());
    }
}

/// `workspace`'s personal account, when the turn ran in a personal folder —
/// the folder itself, or anything under it (a delegated round, a targeted
/// project round and a custom-role launch all work under a path-derived
/// workspace, and a managed workspace placed inside one counts too). `None`
/// for a workspace that is neither inside the userspaces root nor named after a
/// personal workspace, which is what leaves an ordinary managed workspace
/// untouched.
///
/// Where the turn actually worked decides first — that is where its writes
/// landed — and the personal name is the fallback, for a folder the product
/// reaches through a symlink of its own. [`personal_account_of`] is what refuses
/// a name that cannot address a folder.
fn personal_user_of(workspace: &Workspace) -> Option<String> {
    personal_account_in(Path::new(&workspace.path))
        .or_else(|| crate::users::personal_user_name(&workspace.name).map(str::to_string))
}

/// What one scan is for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scan {
    /// The turn's report: nothing is stored, so only what [`diff`] consults is read.
    Report,
    /// A refresh: every entry it keeps is fingerprinted, so a later touch stays
    /// decidable.
    Refresh,
    /// The establishment's walk ([`establish`]). The same as [`Scan::Refresh`],
    /// with the two liberties only a walk that is off the reply path may take: an
    /// unreadable baseline is re-fingerprinted from scratch rather than handed
    /// back, and the walk gives up on a forced stop instead of holding the exit
    /// for a folder-wide read.
    Whole,
}

impl Scan {
    /// Whether the walk's result is stored (a report's is not).
    fn stores(self) -> bool {
        !matches!(self, Self::Report)
    }
}

/// Walk `root` and fingerprint every entry the folder holds, reusing a stored
/// digest when the size and a reported stamp both match exactly. `None` when
/// `root` is not a readable directory or a walk error's path cannot be recovered
/// (fail-open); every other entry the walk could not observe keeps its previous
/// state ([`freeze`]).
///
/// Directories are not entries of their own — a directory is implied by the paths
/// below it — so an added empty directory is not reported. The walk never descends
/// into a symlink either, which leaves the files behind a directory link out of
/// the snapshot: replacing a real directory with a link to one therefore reports
/// the files that were there as removed.
///
/// Entries the walk must not see are pruned by its entry filter: the product's own
/// material ([`is_skipped`]) and whatever the folder's own git rules hide
/// ([`Rules`]) — the filter [`personal_folder_picture`] walks through too, so the
/// two blocks describe the same material. Filtering there is what keeps an ignored
/// path out of the delta instead of merely out of its rendering. `mode` decides what
/// is read ([`Scan`]) and nothing else.
fn scan(root: &Path, previous: Option<&Snapshot>, mode: Scan, rules: &Rules) -> Option<Snapshot> {
    let stores = mode.stores();
    if std::fs::read_dir(root).is_err() {
        // Fail open — no report, no baseline, and never a blocked turn.
        tracing::warn!(root = %root.display(), "personal-workspace folder unreadable");
        return None;
    }
    let mut out = Snapshot::new();
    let whole_walk = mode == Scan::Whole;
    let owner_folder = is_owner_personal_folder(root);
    let walk_rules = rules.clone();
    // Cloned once: the loop below runs per entry, and a whole-folder walk on a
    // large folder would clone the token tens of thousands of times.
    let stop = crate::shutdown::shutdown_token();
    for result in WalkBuilder::new(root)
        .standard_filters(false)
        .follow_links(false)
        .filter_entry(move |entry| !is_skipped(entry, owner_folder) && !walk_rules.hides(entry))
        .build()
    {
        // A forced stop gives up a whole-folder walk rather than hold the exit for
        // a folder-wide read. The token, not `aborting()`: a graceful drain exists
        // to let in-flight work finish, and the drain's own watch fires the token
        // as soon as the foreground work drains.
        if whole_walk && stop.is_cancelled() {
            return None;
        }
        let entry = match result {
            Ok(entry) => entry,
            Err(e) => {
                // A root-level walk error means the folder itself went
                // unreadable after the probe — fail open (no report), not a
                // half-walk that would look like mass deletion.
                if e.depth() == Some(0) {
                    return None;
                }
                // A deeper error misses that path (for a directory: its whole
                // subtree) — freeze it.
                let Some(rel) = error_path(&e)
                    .and_then(|path| path.strip_prefix(root).ok())
                    .and_then(Path::to_str)
                else {
                    // The path is unrecoverable, so what was missed is
                    // unknowable — fail open rather than report a partial walk.
                    tracing::warn!(error = %e, "personal-workspace walk error without a path");
                    return None;
                };
                tracing::warn!(
                    path = %rel,
                    error = %e,
                    "personal-workspace walk error — freezing its previous state"
                );
                freeze(previous, rel, &mut out);
                continue;
            }
        };
        if entry.depth() == 0 {
            continue;
        }
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
        };
        let Some(rel) = rel.to_str() else {
            continue; // a non-UTF-8 path cannot be listed or diffed
        };
        // `std::fs::metadata` follows symlinks (the walker does not descend
        // into them), matching the read tool: a symlinked file is fingerprinted
        // by what it points at. A symlink whose target is gone falls back to the
        // link itself — the entry the folder now holds — so the user's
        // replacement of a file is reported rather than silently missed.
        let meta = std::fs::metadata(path).or_else(|_| std::fs::symlink_metadata(path));
        let Ok(meta) = meta else {
            tracing::warn!(
                path = %path.display(),
                "personal-workspace entry metadata unreadable — freezing its previous state"
            );
            freeze(previous, rel, &mut out);
            continue;
        };
        if meta.is_dir() {
            continue; // directories are implied by the paths below them
        }
        let size = meta.len();
        let mtime = mtime_ns(&meta);
        let digest = if meta.is_file() {
            match previous.and_then(|prev| prev.get(rel)) {
                // An unchanged, reported stamp means the same bytes: reuse it.
                Some(prev)
                    if prev.size == size && prev.mtime_ns == mtime && mtime != STAMP_UNKNOWN =>
                {
                    prev.digest
                }
                // A report keeps nothing, and `diff` consults no digest for a path
                // it has never seen (`added`) or one whose size moved (`changed`).
                None if !stores => None,
                Some(prev) if !stores && prev.size != size => None,
                // Above the cap the stamp is all there is to compare.
                _ if size > HASH_MAX_BYTES => None,
                // Everything else — a stamp that moved under an unchanged size, and
                // a stamp the platform will not report at all — is decided by the
                // content, the only thing here that can decide it.
                _ => digest_of(path),
            }
        } else {
            // A FIFO, a device or a socket is never opened, hashed or waited on;
            // its own stamp is all the walk has to offer, and recording it still
            // reports the user replacing a file with one.
            None
        };
        out.insert(
            rel.to_string(),
            FileState {
                size,
                mtime_ns: mtime,
                digest,
            },
        );
    }
    Some(out)
}

/// The three sorted change classes between two snapshots.
#[derive(Debug, Default)]
struct Changes {
    added: Vec<String>,
    changed: Vec<String>,
    removed: Vec<String>,
}

/// Classify `current` against `previous`; a rename therefore shows as removed +
/// added. A path present in both is `changed` when its size differs — airtight,
/// no digest needed — or when it cannot be shown to hold the same bytes: its
/// stamp moved, or the platform reports no stamp for it at all, and its digest
/// says otherwise. Equal digests mean a touch (nothing reported); a digest
/// missing on either side (above [`HASH_MAX_BYTES`], an entry that is not a
/// regular file, or one that failed to read) cannot be compared, and such a file
/// is reported rather than silently never — a stamp that moved is the most a
/// stamp-only entry can offer. An entry above the cap under an unreportable stamp
/// is the one case with neither evidence to compare, and it follows the same
/// preference: reported, because missing a real edit is the worse failure.
/// Iterating the `BTreeMap`s keeps each list sorted.
///
/// A recorded path the walk no longer produces is `removed` unless the folder's
/// own rules hide it ([`Rules::hidden`]): an ignored path leaves the report
/// silently, whether a rule was added since the baseline or the baseline itself
/// predates this filtering. A tracked path is never hidden, so deleting one is
/// still reported, and a path that stopped being ignored is absent from the
/// baseline — it comes back as `added`.
fn diff(previous: &Snapshot, current: &Snapshot, rules: &Rules) -> Changes {
    let mut changes = Changes::default();
    for (path, cur) in current {
        match previous.get(path) {
            None => changes.added.push(path.clone()),
            Some(prev) => {
                let same_content = matches!(
                    (prev.digest, cur.digest),
                    (Some(before), Some(after)) if before == after
                );
                let moved = prev.mtime_ns != cur.mtime_ns
                    || prev.mtime_ns == STAMP_UNKNOWN
                    || cur.mtime_ns == STAMP_UNKNOWN;
                if prev.size != cur.size || (moved && !same_content) {
                    changes.changed.push(path.clone());
                }
            }
        }
    }
    for path in previous.keys() {
        if !current.contains_key(path) && !rules.hidden(Path::new(path), false) {
            changes.removed.push(path.clone());
        }
    }
    changes
}

/// Hard entry cap across all three classes; the tail summarizes the rest.
const MAX_LISTED_ENTRIES: usize = 40;

/// Secondary stop condition on the rendered block size, in bytes.
const MAX_BLOCK_BYTES: usize = 2 * 1024;

/// Render the change block, or `None` when nothing changed. Classes render in
/// the order added → changed → removed, one per line, stopping at the entry or
/// byte cap; everything left out is summarized by a final `…and N more` line.
fn render(changes: &Changes) -> Option<String> {
    let groups = [
        ("added: ", &changes.added),
        ("changed: ", &changes.changed),
        ("removed: ", &changes.removed),
    ];
    let total: usize = groups.iter().map(|(_, list)| list.len()).sum();
    if total == 0 {
        return None;
    }

    let mut lines = String::new();
    let mut shown = 0usize;
    let mut capped = false;
    for (label, list) in groups {
        let mut names: Vec<&str> = Vec::new();
        let mut cost = lines.len();
        for name in list {
            if shown >= MAX_LISTED_ENTRIES {
                capped = true;
                break;
            }
            let add = name.len() + if names.is_empty() { label.len() } else { 2 };
            if cost + add > MAX_BLOCK_BYTES {
                capped = true;
                break;
            }
            cost += add;
            names.push(name);
            shown += 1;
        }
        if !names.is_empty() {
            let _ = writeln!(lines, "{label}{}", names.join(", "));
        }
        if capped {
            break;
        }
    }
    if total > shown {
        let _ = write!(lines, "…and {} more", total - shown);
    }

    let asset = crate::prompt::load_prompt("context/personal_changes.md");
    Some(crate::prompt::substitute(
        &asset,
        &[("{{changes}}", lines.trim_end())],
    ))
}

/// Prepend the between-turns change report for `workspace`'s personal folder to
/// the Assistant's user message `msg`; `msg` comes back unchanged for every other
/// role, and whenever there is nothing to say: no personal folder, no baseline (or
/// no session row at all), nothing changed, or an unresolvable/unwalkable folder —
/// fail-open throughout. This is the only gate and the only prepend site the
/// message-append path uses, and it resolves the folder exactly like
/// [`refresh_after_turn`], so the block can only ever describe a folder whose
/// baseline is also the one being advanced.
pub(crate) async fn prepend_changes(msg: &str, workspace: &Workspace, role: &Role) -> String {
    if !matches!(role, Role::Assistant) {
        return msg.to_string();
    }
    let Some(user_name) = personal_account_of(workspace) else {
        return msg.to_string();
    };
    // The folder's first observation is still walking, or a turn that ended
    // during it has not been absorbed yet: report nothing rather than a delta
    // from a baseline that is not settled (see `establish`).
    if ESTABLISHING.lock().unwrap_poison().contains_key(&user_name) {
        return msg.to_string();
    }
    let agent_id = crate::session::assistant_agent_id(&user_name);
    // `None` (no session row, no baseline recorded yet, or a read error) means
    // there is nothing to compare against — never fabricate a report from one.
    let Some(Some(stored)) = crate::session::store()
        .personal_changes_baseline(&agent_id)
        .await
    else {
        return msg.to_string();
    };
    let root = crate::users::personal_workspace_path(&user_name);
    // Decoding the baseline, walking the folder and reading what moved all run on
    // the blocking pool: the baseline is megabytes of JSON and the walk is I/O.
    let scanned = tokio::task::spawn_blocking(move || {
        let previous: Snapshot = match serde_json::from_str(&stored) {
            Ok(previous) => previous,
            Err(e) => {
                tracing::warn!(%e, "Unparseable personal-changes baseline — skipping the report");
                return None;
            }
        };
        // One `Rules` for the whole observation: the walk and the removal filter
        // judge by the same rules, and the index is read at most once.
        let rules = Rules::new(&root);
        let current = scan(&root, Some(&previous), Scan::Report, &rules)?;
        Some(diff(&previous, &current, &rules))
    })
    .await;
    let Some(Some(changes)) = scanned.ok() else {
        return msg.to_string();
    };
    match render(&changes) {
        Some(block) => format!("{block}\n\n{msg}"),
        None => msg.to_string(),
    }
}

/// Absorb into the baseline whatever the product itself did inside a personal
/// workspace, so a later report cannot blame the user for it.
///
/// Gated exactly like the report it serves: only the Assistant's own turn counts
/// ([`Role::Assistant`]), only one whose user message was recorded (an
/// undelivered report must leave its changes pending rather than swallow them),
/// and only when the turn ran in that user's personal folder. A delegated round
/// or a custom-role launch that happens to run in the folder therefore advances
/// nothing — absorbing a hand edit the user made between messages would be
/// worse than letting such a round's own writes surface in the next block.
pub(crate) async fn refresh_after_turn(role: &Role, workspace: &Workspace, recorded: bool) {
    if !recorded || !matches!(role, Role::Assistant) {
        return;
    }
    let Some(user_name) = personal_account_of(workspace) else {
        return;
    };
    refresh(&user_name).await;
}

/// The account a turn's change report and baseline concern: the owner of the
/// personal folder that turn ran in ([`personal_user_of`]), refused when the name
/// cannot address a folder. `None` for a turn that ran in no personal folder,
/// which is what leaves an ordinary managed workspace untouched.
fn personal_account_of(workspace: &Workspace) -> Option<String> {
    let user_name = personal_user_of(workspace)?;
    crate::users::is_valid_personal_user_name(&user_name).then_some(user_name)
}

/// Refresh the stored baseline for `user_name`'s personal workspace; the name
/// comes from [`personal_account_of`], so it can address a folder. The write
/// lands on the baseline's own child row, whose foreign key rejects it while the
/// session's row is absent (`SessionStore::update_personal_changes`), so an
/// account with no Assistant metadata row yet is never given a phantom one
/// — and, since that write is what the walk exists for, such an account is not
/// walked at all. An unwalkable folder is a silent no-op and the DB write is
/// skipped when nothing moved. A baseline that is missing or unreadable is the
/// case whose walk is not paid here: it hands off to [`establish`].
///
/// Two turns in the same folder (an automation overlapping a user turn) can interleave
/// their read-write here. Either write is a fresh observation of the same folder,
/// so the loser costs a few milliseconds of staleness, not a wrong report.
async fn refresh(user_name: &str) {
    // A turn ending while the folder's first observation is still running is
    // absorbed by that observation's own next walk (see `establish`), so the walk
    // is not paid for here as well.
    if request_another_walk(user_name) {
        return;
    }
    let agent_id = crate::session::assistant_agent_id(user_name);
    // `None` is no session row at all: the baseline could never be recorded, so
    // there is nothing to walk for.
    let Some(stored) = crate::session::store()
        .personal_changes_baseline(&agent_id)
        .await
    else {
        return;
    };
    if stored.is_none() {
        establish(user_name, agent_id);
        return;
    }
    match observe(
        crate::users::personal_workspace_path(user_name),
        stored,
        Scan::Refresh,
    )
    .await
    {
        Observation::Fresh(json) => store_baseline(user_name, &agent_id, &json).await,
        Observation::Settled => {}
        Observation::Unusable => establish(user_name, agent_id),
    }
}

/// Fingerprint `user_name`'s folder for the first time — the one observation
/// that reads and hashes every file it keeps — off the reply path: the user
/// never waits for it, and no block is rendered while it runs (see
/// [`prepend_changes`]). A change made during that short window is absorbed
/// silently, the same accepted class as the one an in-flight turn absorbs.
///
/// [`ESTABLISHING`] carries that window: it marks the account as being observed,
/// keeps a burst of turns from stacking identical whole-folder walks, and records
/// that a turn ended since the last walk — such a turn's own writes are not in
/// that observation, so one more (incremental) observation runs before the window
/// closes, and the product's writes never reach the next block.
///
/// A baseline write that keeps failing (a broken store) leaves no baseline stored,
/// so every later turn retries this observation — off the reply path, logged, and
/// still silent: the only price is a background walk per turn until a write
/// lands, and recovery is automatic once one does.
fn establish(user_name: &str, agent_id: String) {
    let user_name = user_name.to_string();
    let root = crate::users::personal_workspace_path(&user_name);
    // The guard is built here, next to the insert that sets the mark, and moved
    // into the task below — see [`Establishing`] for what that guarantees.
    let establishing = {
        let mut establishing = ESTABLISHING.lock().unwrap_poison();
        // Already observing: this turn's writes are not in the last walk, so ask
        // for one more (see the loop below) instead of walking here as well. Both
        // happen under one lock, so two turns cannot both start an observation.
        if let Some(in_flight) = establishing.get_mut(&user_name) {
            in_flight.again = true;
            return;
        }
        let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
        establishing.insert(
            user_name.clone(),
            InFlight {
                again: false,
                owner,
            },
        );
        Establishing {
            user_name: user_name.clone(),
            owner,
        }
    };
    tokio::spawn(async move {
        // Holding the guard for the task's lifetime keeps the mark set until this
        // observation really ends, and its owner is the only mark this task may
        // close.
        let owner = establishing.owner;
        // What the caller read may have moved on since (a predecessor stored its
        // observation, or another turn's establishment did): comparing against what
        // is stored NOW keeps a folder already observed on the incremental walk and
        // leaves the whole-folder read — through an unreadable baseline included —
        // to the case it is for, a folder nobody has observed yet.
        let Some(mut previous) = crate::session::store()
            .personal_changes_baseline(&agent_id)
            .await
        else {
            return; // no metadata row: there would be nothing to record
        };
        loop {
            // A walk that stores nothing (an unwalkable folder, or one that moved in
            // no way the baseline does not already record) still owes the window its
            // last check: a turn that ended while it ran has writes this observation
            // does not hold.
            if let Observation::Fresh(json) =
                observe(root.clone(), previous.clone(), Scan::Whole).await
            {
                store_baseline(&user_name, &agent_id, &json).await;
                previous = Some(json);
            }
            if !take_again_or_finish(&user_name, owner) {
                return;
            }
        }
    });
}

/// An account's first observation being in flight, and whether a turn has ended
/// since its last walk. While an account is in here the block stays silent for it
/// (see [`prepend_changes`]).
struct InFlight {
    /// A turn ended since the last walk, so one more observation is owed.
    again: bool,
    /// Which observation set the mark (see [`Establishing`]).
    owner: u64,
}

/// Accounts being observed right now.
static ESTABLISHING: std::sync::Mutex<BTreeMap<String, InFlight>> =
    std::sync::Mutex::new(BTreeMap::new());

/// Hands each observation the token that owns its [`ESTABLISHING`] mark.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

/// Whether an observation of `user_name` is in flight; if it is, record that a
/// turn has ended since its last walk — those writes are not in it, so it runs one
/// more before the window closes.
fn request_another_walk(user_name: &str) -> bool {
    let mut establishing = ESTABLISHING.lock().unwrap_poison();
    match establishing.get_mut(user_name) {
        Some(in_flight) => {
            in_flight.again = true;
            true
        }
        None => false,
    }
}

/// Under one lock: `true` while `owner`'s observation must keep observing — a turn
/// ended since its last walk, and the request is consumed — and `false` when the
/// window can close, which also clears the mark so the block resumes for the
/// account. Taking the request and clearing the mark together is what keeps a turn
/// ending in between from being lost; a mark another observation owns is left
/// alone, exactly as in [`Establishing`].
fn take_again_or_finish(user_name: &str, owner: u64) -> bool {
    let mut establishing = ESTABLISHING.lock().unwrap_poison();
    match establishing.get_mut(user_name) {
        Some(in_flight) if in_flight.owner != owner => false,
        Some(in_flight) if in_flight.again => {
            in_flight.again = false;
            true
        }
        Some(_) => {
            establishing.remove(user_name);
            false
        }
        None => false,
    }
}

/// Clears the [`ESTABLISHING`] mark it set however the observation ends — the
/// window closing normally, an early return, a panic, or a runtime that drops the
/// spawned future without ever polling it — or the account would stay silent and
/// never get a baseline.
struct Establishing {
    user_name: String,
    owner: u64,
}

impl Drop for Establishing {
    fn drop(&mut self) {
        let mut establishing = ESTABLISHING.lock().unwrap_poison();
        // A later observation for the same account may already own a fresh mark
        // by now: that one is not this guard's to clear.
        if establishing
            .get(&self.user_name)
            .is_some_and(|in_flight| in_flight.owner == self.owner)
        {
            establishing.remove(&self.user_name);
        }
    }
}

/// One observation attempt's outcome.
enum Observation {
    /// An observation to store as the baseline.
    Fresh(String),
    /// Nothing to store: the folder could not be walked, or it moved in no way
    /// the stored baseline does not already record.
    Settled,
    /// The stored baseline cannot be read, so the folder has to be fingerprinted
    /// from scratch — the one observation that must not happen on the reply path.
    Unusable,
}

/// Walk `root` and return the observation to store.
///
/// `stored` is the baseline the walk compares against; `None` is no stored
/// baseline (a session predating the baseline), whose empty previous snapshot
/// makes this the folder's first observation.
async fn observe(root: PathBuf, stored: Option<String>, mode: Scan) -> Observation {
    // Decoding the baseline, walking the folder and re-encoding what the walk
    // found all run on the blocking pool: the baseline is megabytes of JSON.
    tokio::task::spawn_blocking(move || {
        let previous: Snapshot = match stored.as_deref() {
            None => Snapshot::new(),
            Some(stored) => match serde_json::from_str(stored) {
                Ok(previous) => previous,
                Err(e) => {
                    // An unreadable baseline means the folder has to be
                    // fingerprinted from scratch — which only the establishment's
                    // walk may pay ([`Scan::Whole`]); a refresh hands the case back
                    // to it instead of reading the whole folder on the reply path.
                    tracing::warn!(%e, "Unreadable personal-changes baseline");
                    if mode != Scan::Whole {
                        return Observation::Unusable;
                    }
                    Snapshot::new()
                }
            },
        };
        // `None` is an unwalkable folder (already warned about inside `scan`).
        let Some(current) = scan(&root, Some(&previous), mode, &Rules::new(&root)) else {
            return Observation::Settled;
        };
        let json = match serde_json::to_string(&current) {
            Ok(json) => json,
            Err(e) => {
                tracing::warn!(%e, "Failed to serialize personal-changes baseline");
                return Observation::Settled;
            }
        };
        if stored.as_deref() == Some(json.as_str()) {
            Observation::Settled
        } else {
            Observation::Fresh(json)
        }
    })
    .await
    .unwrap_or(Observation::Settled)
}

/// Store one observation as `user_name`'s baseline.
async fn store_baseline(user_name: &str, agent_id: &str, json: &str) {
    if let Err(e) = crate::session::store()
        .update_personal_changes(agent_id, json)
        .await
    {
        tracing::warn!(
            user = %user_name,
            error = %e,
            "Failed to persist personal-changes baseline"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::params;
    use crate::util::test::{init_git_repo, init_temp_repo};

    /// The rendered picture read back: its folder counts and its direct-file count.
    /// The parity test reads the block the assistant sees rather than the map behind
    /// it.
    fn pictured_folders(picture: &str) -> (BTreeMap<String, usize>, usize) {
        let count_of = |rest: &str| {
            rest.split_once(' ')
                .expect("a count and a noun")
                .0
                .parse::<usize>()
                .expect("a number")
        };
        let mut folders = BTreeMap::new();
        let mut direct = 0;
        for line in picture.lines() {
            if let Some(rest) = line.strip_prefix("- ") {
                let (name, count) = rest.split_once("/ — ").expect("a folder line");
                folders.insert(name.to_string(), count_of(count));
            } else {
                direct = count_of(
                    line.split_once(" directly in the folder")
                        .expect("the direct-file line")
                        .0,
                );
            }
        }
        (folders, direct)
    }

    /// The folder's first observation runs off the reply path, so its baseline
    /// lands a moment after the refresh that started it returns — and until that
    /// observation has closed, the block stays silent for the account. A walk a
    /// forced stop abandoned leaves no baseline behind, and re-driving the refresh
    /// once nothing is in flight is what the next turn does.
    async fn wait_for_baseline(user: &str) -> Option<String> {
        let agent_id = crate::session::assistant_agent_id(user);
        for _ in 0..200 {
            // While the observation is in flight, waiting is all this may do: a
            // refresh here would ask it for another walk and perturb the state
            // under test.
            if ESTABLISHING.lock().unwrap_poison().contains_key(user) {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                continue;
            }
            if let Some(Some(json)) = crate::session::store()
                .personal_changes_baseline(&agent_id)
                .await
            {
                return Some(json);
            }
            refresh(user).await;
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        None
    }

    #[test]
    fn scan_diff_classifies_added_changed_removed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        for (name, body) in [
            ("a.txt", "aaa"),
            ("b.txt", "bbb"),
            ("c.txt", "ccc"),
            ("d.txt", "ddd"),
        ] {
            std::fs::write(root.join(name), body).unwrap();
        }
        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();

        // Let the mtime clock advance so a rewrite is observable even on a
        // coarse filesystem.
        std::thread::sleep(std::time::Duration::from_millis(20));

        std::fs::write(root.join("a.txt"), "aaa-edited").unwrap(); // content changed
        std::fs::write(root.join("b.txt"), "bbb").unwrap(); // identical bytes (touch)
        std::fs::write(root.join("e.txt"), "eee").unwrap(); // new
        std::fs::rename(root.join("c.txt"), root.join("c2.txt")).unwrap(); // rename
        std::fs::remove_file(root.join("d.txt")).unwrap(); // delete

        let after = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        let changes = diff(&before, &after, &Rules::new(root));
        assert_eq!(
            changes.added,
            vec!["c2.txt".to_string(), "e.txt".to_string()]
        );
        assert_eq!(changes.changed, vec!["a.txt".to_string()]);
        assert_eq!(
            changes.removed,
            vec!["c.txt".to_string(), "d.txt".to_string()]
        );

        // The report's scan stores nothing, so it reads only what `diff`
        // consults: a size change decides on the size alone, while the one
        // identical-size rewrite (b.txt) is read to tell a touch from an edit.
        let report = scan(root, Some(&before), Scan::Report, &Rules::new(root)).unwrap();
        let report_changes = diff(&before, &report, &Rules::new(root));
        assert_eq!(report_changes.changed, vec!["a.txt".to_string()]);
        assert_eq!(
            report["a.txt"].digest, None,
            "a size change must not cost a read on the report path"
        );
        assert!(
            report["b.txt"].digest.is_some(),
            "an identical-size rewrite must still be read"
        );

        // A folder the walk cannot even open is fail-open: no report, no
        // baseline, never a blocked turn.
        assert!(
            scan(
                &root.join("gone"),
                None,
                Scan::Refresh,
                &Rules::new(&root.join("gone"))
            )
            .is_none()
        );
    }

    /// A stamp the platform will not report can never decide a comparison, so
    /// such a file is settled by its content: a same-size edit under one is
    /// reported, and an untouched file is not — the sentinel is not a stamp that
    /// "did not move".
    #[test]
    fn scan_reads_a_file_whose_stamp_the_platform_will_not_report() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let file = root.join("old.txt");
        std::fs::write(&file, "one").unwrap();

        // A stamp before the epoch is the one such mtime a filesystem will
        // actually hold; where it refuses, there is nothing to exercise.
        let before_epoch = std::time::UNIX_EPOCH
            .checked_sub(std::time::Duration::from_secs(60))
            .unwrap();
        let handle = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
        if handle.set_modified(before_epoch).is_err() {
            return;
        }
        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        if before["old.txt"].mtime_ns != STAMP_UNKNOWN {
            return; // the platform would not store a pre-epoch stamp
        }
        assert!(before["old.txt"].digest.is_some());
        // Same size, same (unreportable) stamp, different bytes.
        std::fs::write(&file, "two").unwrap();
        handle.set_modified(before_epoch).unwrap();
        let after = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(after["old.txt"].mtime_ns, STAMP_UNKNOWN);
        assert_eq!(
            diff(&before, &after, &Rules::new(root)).changed,
            vec!["old.txt".to_string()],
            "content must decide when the stamp cannot"
        );
    }

    #[test]
    fn scan_watches_user_dotfiles_but_skips_product_owned_entries() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        for dir in [
            ".git",
            ".compaction",
            "generated",
            "uploads",
            "shared",
            "notes",
            "archive/2026-01-01",
            "archive",
            ".config",
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        let skipped = [
            ".git/config",
            ".compaction/c.txt",
            "generated/g.txt",
            "uploads/u.txt",
            "notes/n.txt",
            "archive/2026-01-01/old.txt",
            ".DS_Store",
            "MEMORY.md",
        ];
        // `shared/` belongs to the product in the owner's own folder alone: in any
        // other root that name is the user's own material (see the dedicated
        // catalogue test below).
        let watched = [
            ".env",
            ".gitignore",
            ".config/settings.toml",
            "keep.txt",
            "shared/s.txt",
        ];
        for path in skipped.iter().chain(watched.iter()) {
            std::fs::write(root.join(path), "one").unwrap();
        }

        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(
            before.keys().cloned().collect::<Vec<_>>(),
            vec![
                ".config/settings.toml".to_string(),
                ".env".to_string(),
                ".gitignore".to_string(),
                "keep.txt".to_string(),
                "shared/s.txt".to_string(),
            ],
            "only product/OS-owned entries must be hidden"
        );

        std::thread::sleep(std::time::Duration::from_millis(20));
        for path in skipped.iter().chain(watched.iter()) {
            std::fs::write(root.join(path), "two-longer").unwrap();
        }

        let after = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(
            after.keys().cloned().collect::<Vec<_>>(),
            before.keys().cloned().collect::<Vec<_>>(),
            "skipped entries must never enter the snapshot"
        );
        let changes = diff(&before, &after, &Rules::new(root));
        assert!(changes.added.is_empty());
        assert!(changes.removed.is_empty());
        assert_eq!(changes.changed.len(), watched.len(), "{changes:?}");
        for path in watched {
            assert!(changes.changed.contains(&path.to_string()), "missed {path}");
        }
    }

    /// The rules come from repositories inside the folder and nowhere else: the
    /// folder's own `.gitignore` is inert while it holds no repository (even with
    /// a repository above it), a nested repository's rules govern the paths inside
    /// it, and no rule from above the folder reaches any of them.
    #[test]
    fn rules_come_from_repositories_inside_the_folder_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        // An outer repository with its own rules: above the folder, so never
        // applied — and its presence must not switch the folder's own
        // `.gitignore` on either.
        let outer = tmp.path();
        init_git_repo(outer);
        std::fs::write(outer.join(".gitignore"), "*.log\n").unwrap();
        let root = outer.join("inner");
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join(".gitignore"), "*.log\n").unwrap();
        std::fs::write(root.join("a.log"), "one").unwrap();
        std::fs::write(root.join("b.txt"), "one").unwrap();

        let no_repo = scan(&root, None, Scan::Refresh, &Rules::new(&root)).unwrap();
        assert_eq!(
            no_repo.keys().cloned().collect::<Vec<_>>(),
            vec![
                ".gitignore".to_string(),
                "a.log".to_string(),
                "b.txt".to_string()
            ],
            "a folder holding no repository hides nothing"
        );

        // A repository inside the folder: its rules count, and only inside it.
        init_git_repo(&nested);
        std::fs::write(nested.join(".gitignore"), "*.tmp\n").unwrap();
        // `.git/info/exclude` is a local rule and counts; a `.ignore` file is not
        // a git rule at all.
        std::fs::create_dir_all(nested.join(".git/info")).unwrap();
        std::fs::write(nested.join(".git/info/exclude"), "*.exe\n").unwrap();
        std::fs::write(nested.join(".ignore"), "*.lock\n").unwrap();
        std::fs::write(nested.join("c.tmp"), "one").unwrap();
        std::fs::write(nested.join("d.log"), "one").unwrap();
        std::fs::write(nested.join("e.exe"), "one").unwrap();
        std::fs::write(nested.join("f.lock"), "one").unwrap();

        let nested_repo = scan(&root, None, Scan::Refresh, &Rules::new(&root)).unwrap();
        assert!(!nested_repo.contains_key("nested/c.tmp"), "{nested_repo:?}");
        assert!(!nested_repo.contains_key("nested/e.exe"), "{nested_repo:?}");
        assert!(
            nested_repo.contains_key("nested/f.lock"),
            "a `.ignore` file is not a git rule: {nested_repo:?}"
        );
        assert!(
            nested_repo.contains_key("nested/d.log"),
            "rules above the nested repository must not reach it: {nested_repo:?}"
        );
        assert!(nested_repo.contains_key("a.log"), "{nested_repo:?}");
    }

    /// A nested repository inside a folder its enclosing repository ignores is gone
    /// as a whole: the walk prunes at the ignored folder, so nothing beneath it is
    /// reported as removed even though the nested repository's own rules never hid
    /// those paths.
    #[test]
    fn a_nested_repository_inside_an_ignored_folder_is_not_reported_removed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        init_git_repo(root);
        let inner = root.join("vendor/inner");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join("keep.txt"), "one").unwrap();
        init_git_repo(&inner);
        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert!(before.contains_key("vendor/inner/keep.txt"), "{before:?}");

        std::fs::write(root.join(".gitignore"), "vendor/\n").unwrap();
        let after = scan(root, Some(&before), Scan::Report, &Rules::new(root)).unwrap();
        assert!(!after.contains_key("vendor/inner/keep.txt"), "{after:?}");
        assert!(
            diff(&before, &after, &Rules::new(root)).removed.is_empty(),
            "an ignored folder takes the nested repository inside it with it"
        );
    }

    /// Adding an ignore rule is invisible: a recorded path the rules now exclude
    /// is not reported as removed, and a path that stops being ignored comes back
    /// as added — new to the world the report describes.
    #[test]
    fn a_new_ignore_rule_never_reports_a_removal() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        init_git_repo(root);
        std::fs::write(root.join(".gitignore"), "# no rules yet\n").unwrap();
        std::fs::write(root.join("kept.txt"), "kept").unwrap();
        std::fs::write(root.join("noise.log"), "noise").unwrap();
        std::fs::create_dir(root.join("build")).unwrap();
        std::fs::write(root.join("build/a.o"), "noise").unwrap();
        std::fs::write(root.join("build/b.o"), "noise").unwrap();
        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert!(before.contains_key("noise.log"), "{before:?}");

        // The rule is added: the paths it covers leave the change set silently.
        std::fs::write(root.join(".gitignore"), "*.log\nbuild/\n").unwrap();
        let after = scan(root, Some(&before), Scan::Report, &Rules::new(root)).unwrap();
        assert!(!after.contains_key("noise.log"), "{after:?}");
        assert!(!after.contains_key("build/a.o"), "{after:?}");
        let changes = diff(&before, &after, &Rules::new(root));
        assert!(
            changes.removed.is_empty(),
            "an ignored path may never read as removed: {changes:?}"
        );

        // A baseline recorded under the rule does not carry those paths, so
        // lifting the rule reports them as new — never as a resurrection.
        let under_rule = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        std::fs::write(root.join(".gitignore"), "# rule lifted\n").unwrap();
        let lifted = scan(root, Some(&under_rule), Scan::Report, &Rules::new(root)).unwrap();
        assert_eq!(
            diff(&under_rule, &lifted, &Rules::new(root)).added,
            vec![
                "build/a.o".to_string(),
                "build/b.o".to_string(),
                "noise.log".to_string()
            ]
        );
    }

    /// A tracked path is never hidden by a pattern: the index is git's own answer
    /// to the ignore question, and a tracked file keeps its directory in the walk.
    #[test]
    fn rules_never_hide_a_tracked_path() {
        let (_dir, root) = init_temp_repo();
        std::fs::write(root.join(".gitignore"), "*.log\nbuild/\n").unwrap();
        std::fs::write(root.join("keep.log"), "tracked").unwrap();
        std::fs::write(root.join("skip.log"), "untracked").unwrap();
        std::fs::create_dir(root.join("build")).unwrap();
        std::fs::write(root.join("build/keep.txt"), "tracked").unwrap();
        std::fs::write(root.join("build/skip.txt"), "untracked").unwrap();
        for path in ["keep.log", "build/keep.txt"] {
            let status = std::process::Command::new("git")
                .args(["add", "-f", path])
                .current_dir(&root)
                .status()
                .expect("git add");
            assert!(status.success(), "git add -f {path}");
        }

        let snapshot = scan(&root, None, Scan::Refresh, &Rules::new(&root)).unwrap();
        assert!(snapshot.contains_key("keep.log"), "{snapshot:?}");
        assert!(
            snapshot.contains_key("build/keep.txt"),
            "a tracked file keeps its ignored directory: {snapshot:?}"
        );
        assert!(!snapshot.contains_key("skip.log"), "{snapshot:?}");
        assert!(!snapshot.contains_key("build/skip.txt"), "{snapshot:?}");

        // A tracked file that is deleted is reported: it is not ignored, it is
        // gone, and the pattern it matches never overrides the index.
        std::fs::remove_file(root.join("keep.log")).unwrap();
        let after = scan(&root, Some(&snapshot), Scan::Report, &Rules::new(&root)).unwrap();
        assert_eq!(
            diff(&snapshot, &after, &Rules::new(&root)).removed,
            vec!["keep.log".to_string()]
        );
    }

    /// The tracked exemption holds inside a nested repository as well: its index
    /// is asked about its own paths, spelled the way git spells them, not at the
    /// folder's prefix.
    #[test]
    fn rules_never_hide_a_tracked_path_inside_a_nested_repository() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let inner = root.join("project");
        std::fs::create_dir(&inner).unwrap();
        init_git_repo(&inner);
        std::fs::write(inner.join(".gitignore"), "*.log\n").unwrap();
        std::fs::write(inner.join("keep.log"), "tracked").unwrap();
        std::fs::write(inner.join("skip.log"), "untracked").unwrap();
        let status = std::process::Command::new("git")
            .args(["add", "-f", "keep.log"])
            .current_dir(&inner)
            .status()
            .expect("git add");
        assert!(status.success(), "git add -f keep.log");

        let snapshot = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert!(snapshot.contains_key("project/keep.log"), "{snapshot:?}");
        assert!(!snapshot.contains_key("project/skip.log"), "{snapshot:?}");
    }

    /// The catalogue decision ([`is_owner_personal_folder`]): `shared/` is product
    /// material in the owner's own personal folder alone — the folder the product
    /// resolves from the owner's account — so the product's writes into the
    /// catalogue never reach the block, while a folder that merely carries the
    /// owner's name (anywhere but the userspaces root, or below the real folder)
    /// and a guest's own `shared/` stay the user's material.
    #[test]
    fn scan_prunes_shared_in_the_owners_own_folder_alone() {
        let owner = crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME);
        // The decision itself, on the shapes a walk's root can be: the owner's own
        // folder — spelled as the product builds it, or with a `.` component
        // appended, since the comparison is component-wise — and nothing else.
        assert!(is_owner_personal_folder(&owner));
        assert!(is_owner_personal_folder(&owner.join(".")));
        assert!(!is_owner_personal_folder(&crate::users::userspaces_root()));
        assert!(!is_owner_personal_folder(&owner.join("nested")));
        let outside = tempfile::TempDir::new().unwrap();
        assert!(!is_owner_personal_folder(outside.path()));

        // The owner's own folder: its `shared/` is the product's catalogue.
        let catalogue = owner.join("shared");
        std::fs::create_dir_all(&catalogue).unwrap();
        // Non-runnable extensions: the probes must be invisible to the custom-tool
        // discovery other tests run against this shared folder.
        let catalogue_probe = crate::util::test::ProbeFile(catalogue.join("installed_probe.txt"));
        let personal_probe = crate::util::test::ProbeFile(owner.join("personal_changes_probe.txt"));
        std::fs::write(&catalogue_probe.0, "one").unwrap();
        std::fs::write(&personal_probe.0, "one").unwrap();

        let owner_snapshot = scan(&owner, None, Scan::Refresh, &Rules::new(&owner)).unwrap();
        assert!(
            owner_snapshot.contains_key("personal_changes_probe.txt"),
            "the owner's own material stays watched: {owner_snapshot:?}"
        );
        assert!(
            !owner_snapshot
                .keys()
                .any(|path| path.starts_with("shared/")),
            "the catalogue is the product's own: {owner_snapshot:?}"
        );

        // A guest's own folder keeps its `shared/`.
        let guest = crate::users::userspaces_root().join("pc_shared_guest");
        std::fs::create_dir_all(guest.join("shared")).unwrap();
        std::fs::write(guest.join("shared/own.txt"), "one").unwrap();
        let guest_snapshot = scan(&guest, None, Scan::Refresh, &Rules::new(&guest)).unwrap();
        assert_eq!(
            guest_snapshot.keys().cloned().collect::<Vec<_>>(),
            vec!["shared/own.txt".to_string()],
            "another account's `shared/` is that account's own material"
        );
        std::fs::remove_dir_all(&guest).unwrap();

        // A folder that merely carries the owner's name is not the product's own.
        let lookalike = outside.path().join(crate::users::ADMIN_USER_NAME);
        std::fs::create_dir_all(lookalike.join("shared")).unwrap();
        std::fs::write(lookalike.join("shared/own.txt"), "one").unwrap();
        let lookalike_snapshot =
            scan(&lookalike, None, Scan::Refresh, &Rules::new(&lookalike)).unwrap();
        assert_eq!(
            lookalike_snapshot.keys().cloned().collect::<Vec<_>>(),
            vec!["shared/own.txt".to_string()],
            "only the product's own folder holds the catalogue"
        );
    }

    #[test]
    fn render_caps_listed_entries_and_summarizes_the_rest() {
        let changes = Changes {
            added: (0..100).map(|i| format!("f{i:03}.txt")).collect(),
            changed: Vec::new(),
            removed: Vec::new(),
        };
        let block = render(&changes).unwrap();
        assert!(block.contains("…and 60 more"), "tail missing: {block}");
        assert!(
            !block.contains("\n\n"),
            "a truncated block must not carry a stray blank line: {block}"
        );
        let listed = block
            .lines()
            .find_map(|line| line.strip_prefix("added: "))
            .unwrap()
            .split(", ")
            .count();
        assert_eq!(listed, MAX_LISTED_ENTRIES);

        // The untruncated form ends exactly like the truncated one.
        let short = render(&Changes {
            added: vec!["a.txt".to_string()],
            ..Changes::default()
        })
        .unwrap();
        assert!(
            short.ends_with("a.txt\n</personal-changes>"),
            "unexpected block tail: {short}"
        );
    }

    /// An entry the walk cannot stat at all keeps its previous state instead of
    /// reading as a deletion, and the `rel/` range copy takes its whole subtree
    /// with it without spilling into a sibling that merely shares the prefix.
    #[test]
    fn freeze_keeps_the_prefix_subtree_and_no_sibling() {
        let mut previous = Snapshot::new();
        for path in ["dir", "dir/a.txt", "dir/sub/b.txt", "dir.txt", "other.txt"] {
            previous.insert(
                path.to_string(),
                FileState {
                    size: 1,
                    mtime_ns: 2,
                    digest: Some(3),
                },
            );
        }
        let mut out = Snapshot::new();
        freeze(Some(&previous), "dir", &mut out);
        assert_eq!(
            out.keys().cloned().collect::<Vec<_>>(),
            vec![
                "dir".to_string(),
                "dir/a.txt".to_string(),
                "dir/sub/b.txt".to_string(),
            ],
            "the frozen prefix must take its subtree, and only it"
        );

        // With no previous observation there is nothing to keep.
        let mut fresh = Snapshot::new();
        freeze(None, "dir", &mut fresh);
        assert!(fresh.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn scan_reports_a_file_replaced_by_a_dangling_symlink() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("gone.txt"), "here").unwrap();
        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert!(before.contains_key("gone.txt"));

        // Replaced by a dangling symlink: `metadata` (which follows the link)
        // fails, so the link itself — the entry the folder now holds — is what the
        // walk records. Never followed, and the replacement is reported rather
        // than silently frozen away.
        std::fs::remove_file(root.join("gone.txt")).unwrap();
        std::os::unix::fs::symlink("x", root.join("gone.txt")).unwrap();

        let after = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(
            after["gone.txt"].digest, None,
            "a broken link is never followed"
        );
        let changes = diff(&before, &after, &Rules::new(root));
        assert_eq!(changes.changed, vec!["gone.txt".to_string()]);
    }

    #[cfg(unix)]
    #[test]
    fn scan_never_opens_a_fifo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("ok.txt"), "ok").unwrap();
        let fifo = root.join("pipe");
        let c_path = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path; `mkfifo` only mints
        // the node.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) }, 0);

        // A regression that opens the FIFO blocks in `File::open` forever; the
        // timeout turns that hang into a failure instead of a stuck suite.
        let (tx, rx) = std::sync::mpsc::channel();
        let root = root.to_path_buf();
        std::thread::spawn(move || {
            let _ = tx.send(scan(&root, None, Scan::Refresh, &Rules::new(&root)));
        });
        let snapshot = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a FIFO must not block the scan")
            .expect("the folder is readable");
        assert!(snapshot.contains_key("ok.txt"));
        let pipe = snapshot
            .get("pipe")
            .expect("a FIFO is an entry the folder holds and must be observed");
        assert_eq!(pipe.digest, None, "a FIFO must never be hashed");
    }

    #[tokio::test]
    #[serial_test::serial(personal_changes)]
    async fn refresh_and_prepend_round_trip() {
        crate::util::test::init_test_stores().await;
        let user = "pc_e2e_user";
        let agent_id = crate::session::assistant_agent_id(user);
        let conn = &crate::session::store().conn;
        conn.execute(
            "DELETE FROM session_metadata WHERE agent_id = ?1",
            params![agent_id.clone()],
        )
        .await
        .unwrap();
        let root = crate::users::personal_workspace_path(user);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), "one").unwrap();
        let personal = crate::users::personal_workspace_struct(user);

        // No metadata row: prepend is a no-op and refresh must not create one.
        assert_eq!(
            prepend_changes("hi", &personal, &Role::Assistant).await,
            "hi"
        );
        refresh(user).await;
        let rows: i64 = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = ?1",
                params![agent_id.clone()],
            )
            .await
            .unwrap()[0]
            .get::<i64>(0)
            .unwrap();
        assert_eq!(rows, 0, "refresh must not create a phantom metadata row");

        let now = crate::db::now();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, last_activity, user_name, workspace_name, role, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'assistant', ?2)",
            params![agent_id.clone(), now, user, format!("personal:{user}")],
        )
        .await
        .unwrap();

        // The first refresh establishes the baseline silently, off the reply
        // path.
        refresh(user).await;
        assert_eq!(
            prepend_changes("hi", &personal, &Role::Assistant).await,
            "hi"
        );
        let stored = wait_for_baseline(user)
            .await
            .expect("the folder's first observation must land");
        assert!(stored.contains("a.txt"), "{stored}");

        // A hand edit is reported at the head of the next message.
        std::fs::write(root.join("a.txt"), "an edited body").unwrap();
        std::fs::write(root.join("b.txt"), "brand new").unwrap();
        let msg = prepend_changes("hello", &personal, &Role::Assistant).await;
        assert!(
            msg.ends_with("hello"),
            "original message must stay at the tail: {msg}"
        );
        assert!(
            msg.starts_with("<personal-changes>"),
            "block shell missing: {msg}"
        );
        assert!(msg.contains("changed: a.txt"), "missed change: {msg}");
        assert!(msg.contains("added: b.txt"), "missed addition: {msg}");

        // The turn's own writes land after the report, so the refresh that
        // closes the turn absorbs them and the next message carries nothing.
        std::fs::write(root.join("written-by-the-turn.md"), "product output").unwrap();
        refresh(user).await;
        assert_eq!(
            prepend_changes("hello", &personal, &Role::Assistant).await,
            "hello"
        );

        // A workspace in no personal folder (a project one), a name that cannot
        // address a folder, and an account with no metadata row are all no-ops.
        let outside = std::env::temp_dir();
        let project = Workspace::ephemeral_run("project-ws", &outside);
        assert_eq!(prepend_changes("x", &project, &Role::Assistant).await, "x");
        let mut unnameable = Workspace::ephemeral_run("dir-x", &outside);
        unnameable.name = "personal:bad/name".to_string();
        assert_eq!(
            prepend_changes("x", &unnameable, &Role::Assistant).await,
            "x"
        );
        // The role gate is the module's, so a first turn and a later one cannot
        // drift: only the Assistant's own turns carry the block.
        assert_eq!(
            prepend_changes("x", &personal, &Role::Manager).await,
            "x",
            "only the Assistant's turns carry the report"
        );
        assert_eq!(
            prepend_changes(
                "x",
                &crate::users::personal_workspace_struct("pc_e2e_absent_user"),
                &Role::Assistant,
            )
            .await,
            "x"
        );

        conn.execute(
            "DELETE FROM session_metadata WHERE agent_id = ?1",
            params![agent_id],
        )
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A file above the hash cap is decided by its stamp alone, so a same-size
    /// edit deep inside it — both ends byte-identical — is still reported.
    #[test]
    fn diff_reports_an_oversized_file_by_its_stamp() {
        use std::io::{Seek, SeekFrom, Write as _};
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let big = root.join("big.bin");
        std::fs::File::create(&big)
            .unwrap()
            .set_len(HASH_MAX_BYTES + 1)
            .unwrap();

        let before = scan(root, None, Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(
            before["big.bin"].digest, None,
            "an oversized file is never read"
        );
        assert_eq!(before["big.bin"].size, HASH_MAX_BYTES + 1);

        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut file = std::fs::OpenOptions::new().write(true).open(&big).unwrap();
        file.seek(SeekFrom::Start(HASH_MAX_BYTES / 2)).unwrap();
        file.write_all(b"x").unwrap();

        let after = scan(root, Some(&before), Scan::Refresh, &Rules::new(root)).unwrap();
        assert_eq!(
            diff(&before, &after, &Rules::new(root)).changed,
            vec!["big.bin".to_string()]
        );
    }

    /// The refresh gate: a turn's workspace resolves to the personal account
    /// whose folder it ran in — by path for anything inside it, by the personal
    /// name otherwise — and to nothing for a workspace outside the userspaces
    /// root.
    #[tokio::test]
    #[serial_test::serial(personal_changes)]
    async fn refresh_after_turn_resolves_the_owned_personal_folder() {
        crate::util::test::init_test_stores().await;
        let root = crate::users::userspaces_root();
        let folder = root.join("pc_gate_user");
        let inner = folder.join("delegated");
        std::fs::create_dir_all(&inner).unwrap();
        let project_dir = root.join("..").join("pc_gate_project");
        std::fs::create_dir_all(&project_dir).unwrap();

        let own = crate::users::personal_workspace_struct("pc_gate_user");
        assert_eq!(personal_user_of(&own).as_deref(), Some("pc_gate_user"));

        // Both sides are compared canonically, so a storage root behind a
        // symlink (macOS temp dirs are) still resolves.
        let delegated = Workspace::ephemeral_run("dir-delegated-0", &inner);
        assert_eq!(
            personal_user_of(&delegated).as_deref(),
            Some("pc_gate_user")
        );
        let project = Workspace::from_path(&project_dir);
        assert_eq!(
            personal_user_of(&project),
            None,
            "a managed workspace is not watched"
        );

        // A personal workspace whose folder the product reaches through a
        // symlink of its own falls back to the personal name.
        let mut linked = Workspace::ephemeral_run("dir-linked-0", &project_dir);
        linked.name = crate::users::personal_workspace_name("pc_gate_user");
        assert_eq!(personal_user_of(&linked).as_deref(), Some("pc_gate_user"));

        // The gate's effect: only a turn that ran in the folder advances its
        // baseline, and the write lands on that folder's own account.
        let agent_id = crate::session::assistant_agent_id("pc_gate_user");
        crate::session::store()
            .conn
            .execute(
                "INSERT INTO session_metadata (agent_id, last_activity, user_name, role, created_at) \
                 VALUES (?1, ?2, ?3, 'assistant', ?2)",
                params![agent_id.clone(), crate::db::now(), "pc_gate_user"],
            )
            .await
            .unwrap();
        std::fs::write(inner.join("hand.txt"), "hand written").unwrap();
        refresh_after_turn(&crate::Role::Assistant, &delegated, true).await;
        let stored = wait_for_baseline("pc_gate_user")
            .await
            .expect("a turn inside the folder stores its baseline");
        assert!(stored.contains("delegated/hand.txt"), "{stored}");

        // A round that ran in the folder but was not the Assistant's own
        // recorded turn — a delegation or a custom-role launch — must advance
        // nothing, or a hand edit would be swallowed for good; such a round's
        // own writes surface in the next block instead.
        std::fs::write(inner.join("hand.txt"), "hand edited again").unwrap();
        refresh_after_turn(&crate::Role::Custom, &delegated, true).await;
        refresh_after_turn(&crate::Role::Assistant, &delegated, false).await;
        let personal = crate::users::personal_workspace_struct("pc_gate_user");
        let report = prepend_changes("x", &personal, &Role::Assistant).await;
        assert!(report.contains("changed: delegated/hand.txt"), "{report}");

        // A turn outside the folder leaves that baseline as it was.
        std::fs::remove_file(inner.join("hand.txt")).unwrap();
        refresh_after_turn(&crate::Role::Assistant, &project, true).await;
        assert_eq!(
            crate::session::store()
                .personal_changes_baseline(&agent_id)
                .await
                .flatten()
                .as_deref(),
            Some(stored.as_str())
        );

        crate::session::store()
            .conn
            .execute(
                "DELETE FROM session_metadata WHERE agent_id = ?1",
                params![agent_id],
            )
            .await
            .unwrap();
        let _ = std::fs::remove_dir_all(&folder);
        let _ = std::fs::remove_dir_all(&project_dir);
    }

    /// The session-start picture counts exactly the material the report watches:
    /// a parallel [`scan`] of the same root names the same immediate-child
    /// folders with the same counts, and the files it carries at the root are the
    /// direct count — the block and the report can never describe different
    /// material.
    #[test]
    fn personal_folder_picture_counts_exactly_what_the_report_watches() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        // Not the owner's own folder, so `shared/` here is the user's own
        // material: the catalogue name is pruned in the owner's folder alone.
        assert!(!is_owner_personal_folder(root));
        init_git_repo(root);
        for dir in [
            ".compaction",
            "generated",
            "uploads",
            "notes",
            "archive/2026-01-01",
            "archive",
            "album/deep",
            "album",
            ".config",
            "empty",
            "shared",
            "cache/deep",
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        // Kept-aside material: never named and never counted.
        for path in [
            ".git/config",
            ".compaction/dump.md",
            "generated/big.bin",
            "uploads/u.png",
            "notes/topic.md",
            "archive/2026-01-01/old.md",
            "MEMORY.md",
            ".DS_Store",
            "album/.DS_Store",
        ] {
            std::fs::write(root.join(path), "one").unwrap();
        }
        // The user's own material — dotfiles included unless the folder's own
        // rules exclude them: this is a repository, so `secret.txt` and the whole
        // `cache/` folder are gone from both blocks, while `.env` stays.
        std::fs::write(root.join(".gitignore"), "secret.txt\ncache/\n").unwrap();
        for path in [
            "album/one.jpg",
            "album/deep/two.jpg",
            ".config/settings.toml",
            "shared/s.txt",
            "top.txt",
            ".env",
            "secret.txt",
            "cache/one.bin",
        ] {
            std::fs::write(root.join(path), "one").unwrap();
        }
        // A link to a directory is neither named nor descended into — the report
        // records no material behind one either.
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("album"), root.join("linked")).unwrap();

        let picture = personal_folder_picture(root).expect("picture");
        let snapshot =
            scan(root, None, Scan::Refresh, &Rules::new(root)).expect("the report's own walk");
        let mut watched: BTreeMap<String, usize> = BTreeMap::new();
        let mut watched_direct = 0usize;
        for key in snapshot.keys() {
            match key.split_once(std::path::MAIN_SEPARATOR) {
                Some((folder, _)) => *watched.entry(folder.to_string()).or_default() += 1,
                None => watched_direct += 1,
            }
        }

        let (pictured, pictured_direct) = pictured_folders(&picture);

        assert_eq!(watched_direct, 3, "{snapshot:?}");
        assert!(!snapshot.contains_key("secret.txt"), "{snapshot:?}");
        assert!(snapshot.contains_key(".env"), "{snapshot:?}");
        assert_eq!(pictured_direct, watched_direct);
        // The empty folder is the user's folder although the report records no
        // material in it (directories are implied by the paths below them).
        assert_eq!(pictured.get("empty"), Some(&0));
        assert_eq!(pictured.len(), watched.len() + 1);
        for (folder, count) in &watched {
            assert_eq!(pictured.get(folder), Some(count), "{folder} in {picture}");
        }
        for hidden in [
            "generated",
            "uploads",
            "notes",
            "archive",
            ".git",
            ".compaction",
            "MEMORY.md",
            "linked",
            "cache", // an ignored folder is pruned whole by both blocks
        ] {
            assert!(!picture.contains(hidden), "{hidden} in {picture}");
        }
    }

    /// The picture names at most [`MAX_PICTURED_FOLDERS`] folders, summarizes the
    /// rest as a count, and stays byte-stable across walks; material added deep
    /// inside a folder moves that folder's count without changing the block's
    /// shape.
    #[test]
    fn personal_folder_picture_bounds_names_and_is_stable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let total = MAX_PICTURED_FOLDERS + 3;
        // Created out of order: the picture comes out name-sorted regardless.
        for i in 0..total {
            std::fs::create_dir(root.join(format!("f{:03}", (i * 7) % total))).unwrap();
        }
        std::fs::write(root.join("top.txt"), "one").unwrap();
        std::fs::create_dir(root.join("f000/deep")).unwrap();
        std::fs::write(root.join("f000/deep/a.txt"), "one").unwrap();

        let picture = personal_folder_picture(root).expect("picture");
        let lines: Vec<&str> = picture.lines().collect();
        assert_eq!(lines.len(), MAX_PICTURED_FOLDERS + 2, "{picture}");
        assert_eq!(lines[0], "- f000/ — 1 file");
        for (i, line) in lines.iter().enumerate().take(MAX_PICTURED_FOLDERS).skip(1) {
            assert_eq!(*line, format!("- f{i:03}/ — 0 files"));
        }
        assert_eq!(lines[MAX_PICTURED_FOLDERS], "…and 3 folders not named");
        assert_eq!(
            lines[MAX_PICTURED_FOLDERS + 1],
            "1 file directly in the folder"
        );
        assert_eq!(
            personal_folder_picture(root).as_deref(),
            Some(picture.as_str()),
            "two walks over an unchanged folder must agree byte for byte"
        );

        std::fs::create_dir(root.join("f001/nested")).unwrap();
        std::fs::write(root.join("f001/nested/b.txt"), "one").unwrap();
        std::fs::write(root.join("f001/nested/c.txt"), "one").unwrap();
        let grown = personal_folder_picture(root).expect("picture");
        assert_eq!(
            grown.lines().count(),
            picture.lines().count(),
            "a changed count must not change the block's shape: {grown}"
        );
        assert!(grown.contains("- f001/ — 2 files"), "{grown}");
    }

    /// No material of the user's means no block: a folder that does not exist yet
    /// (personal folders are created lazily), one holding only kept-aside areas,
    /// and — unlike those — a folder holding only an empty child folder, which is
    /// still the user's own and is pictured at zero.
    #[test]
    fn personal_folder_picture_is_absent_without_user_material() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        assert!(personal_folder_picture(&root.join("absent")).is_none());

        let kept_aside = root.join("kept-aside");
        std::fs::create_dir_all(kept_aside.join(".git")).unwrap();
        std::fs::write(kept_aside.join(".git/config"), "one").unwrap();
        std::fs::write(kept_aside.join("MEMORY.md"), "one").unwrap();
        assert!(personal_folder_picture(&kept_aside).is_none());

        let only_empty = root.join("only-empty");
        std::fs::create_dir_all(only_empty.join("album")).unwrap();
        assert_eq!(
            personal_folder_picture(&only_empty).as_deref(),
            Some("- album/ — 0 files\n0 files directly in the folder")
        );
    }
}

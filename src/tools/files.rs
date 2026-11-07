//! The `files` tool: move, rename and delete inside the caller's own personal workspace.
//!
//! A guest Assistant has no shell, so this is its only way to arrange what it produced
//! (`generated/`) and what arrived (`uploads/`) — the two folders the media sweep empties —
//! into folders of its own that the sweep never visits. The admin's Assistant has a shell and
//! needs no such tool, so this one is built for a guest alone (see `assistant_core_tools` in
//! `agent/role.rs`).
//!
//! Every operation is a real filesystem rename or removal, so the frame is enforced on both
//! sides: the source is resolved without following a symlink, the destination is refused
//! inside the product's own folders, and nothing that already exists is ever overwritten.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::path::{closed_area, normalize_path};
use crate::{Tool, Workspace};

/// The dated shelf an item can be moved onto instead of a folder of the caller's own.
const ARCHIVE_DIR: &str = "archive";

/// Top-level folders that are not operands themselves, though what they hold is. `generated/`
/// and `uploads/` are the media sweep's own two folders, so neither is a destination either.
const SERVICE_AREAS: &[&str] = &["generated", "uploads", ARCHIVE_DIR];

/// How many entries a partly failed delete names before it stops listing.
const REMAINS_NAMED: usize = 20;

pub(crate) struct FilesTool;

#[async_trait]
impl Tool for FilesTool {
    fn name(&self) -> &'static str {
        "files"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "operation": {
                    "type": "string",
                    "enum": ["move", "rename", "delete"],
                    "description": "What to do with `path`: move it into a folder, rename it in place, or delete it for good."
                },
                "path": {
                    "type": "string",
                    "description": "The one file or folder to act on, relative to your workspace root."
                },
                "to": {
                    "type": "string",
                    "description": "`move` only: the destination folder, created when it is missing; `.` or an empty value is the top of your workspace, and `archive` is today's dated folder under archive/."
                },
                "name": {
                    "type": "string",
                    "description": "The new name: required by `rename`, optional for `move` (which otherwise keeps the current name)."
                }
            }),
            &["operation", "path"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        if !crate::users::is_personal_workspace(&ws.name) {
            anyhow::bail!(
                "cannot use files: it arranges the caller's own personal workspace, and `{}` is \
                 not one",
                ws.name
            );
        }
        let path = arg(&args, "path");
        if path.is_empty() {
            anyhow::bail!("`path` must name the one file or folder to act on");
        }
        let root = crate::stored_workspace_path(ws.as_path());
        let source = resolve_item(&root, &path).await?;
        match arg(&args, "operation").as_str() {
            "move" => move_item(&root, &source, &args).await,
            "rename" => rename_item(&root, &source, &args).await,
            "delete" => delete_item(&root, &source).await,
            "" => anyhow::bail!("`operation` is required: one of `move`, `rename`, `delete`"),
            other => anyhow::bail!(
                "unknown operation `{other}`: `operation` is one of `move`, `rename`, `delete`"
            ),
        }
    }
}

/// One resolved operand: its workspace-relative spelling and the absolute path it names.
struct Item {
    relative: PathBuf,
    absolute: PathBuf,
}

/// A trimmed string argument, empty when absent.
fn arg(args: &serde_json::Value, key: &str) -> String {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Resolve one operand without following a symlink: the path is normalized into the workspace
/// frame, and every existing component is refused when it is a link.
async fn resolve_item(root: &Path, spelled: &str) -> Result<Item> {
    let relative = workspace_relative(root, spelled)?;
    refuse_symlinks(root, &relative).await?;
    Ok(Item {
        absolute: root.join(&relative),
        relative,
    })
}

/// The workspace-relative spelling of `spelled`: a relative path is joined to the root, an
/// absolute one must already be inside it, and `..` never escapes.
fn workspace_relative(root: &Path, spelled: &str) -> Result<PathBuf> {
    let expanded = crate::util::expand_tilde(spelled);
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        root.join(expanded)
    };
    normalize_path(&absolute)
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| anyhow::anyhow!("{spelled} is outside your personal workspace"))
}

/// Refuse a path that runs through a symlink. A move or a delete through a link acts on what
/// it points at, and a link in a destination chain would route the operation out of the frame;
/// components that do not exist yet cannot be links.
async fn refuse_symlinks(root: &Path, relative: &Path) -> Result<()> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        if tokio::fs::symlink_metadata(&current)
            .await
            .is_ok_and(|meta| meta.file_type().is_symlink())
        {
            anyhow::bail!(
                "refused: {} is a symlink — this tool moves, renames and deletes real items and \
                 never follows a link to do it",
                current.display()
            );
        }
    }
    Ok(())
}

/// The service area the path's first component names, if it is one. The match ignores case: a
/// case-insensitive volume spells one folder many ways.
fn first_area(relative: &Path) -> Option<&'static str> {
    let first = relative.components().next()?.as_os_str().to_str()?;
    SERVICE_AREAS
        .iter()
        .copied()
        .find(|area| area.eq_ignore_ascii_case(first))
}

/// Whether the path is the workspace's own root memory file, in any spelling the volume accepts.
fn is_root_memory(relative: &Path) -> bool {
    let mut components = relative.components();
    let first = components.next().and_then(|c| c.as_os_str().to_str());
    components.next().is_none()
        && first.is_some_and(|name| name.eq_ignore_ascii_case(crate::prompt::MEMORY_FILE))
}

/// The service area this exact path names — the folder itself, not what it holds.
fn exact_area(relative: &Path) -> Option<&'static str> {
    let area = first_area(relative)?;
    let mut components = relative.components();
    components.next();
    components.next().is_none().then_some(area)
}

/// Refusals shared by every operation on an existing item.
async fn refuse_source(source: &Item, verb: &str) -> Result<()> {
    if source.relative.as_os_str().is_empty() {
        anyhow::bail!("cannot {verb}: that is the workspace itself");
    }
    if is_root_memory(&source.relative) {
        anyhow::bail!(
            "cannot {verb} {}: the memory is corrected, not moved aside — edit the stale part out \
             of it instead",
            crate::prompt::MEMORY_FILE
        );
    }
    if let Some(area) = closed_area(&source.relative) {
        anyhow::bail!(
            "cannot {verb} {}: `{area}/` is kept by the product and closed to this tool",
            source.relative.display()
        );
    }
    if let Some(area) = exact_area(&source.relative) {
        anyhow::bail!(
            "cannot {verb} {area}/: that is one of the workspace's own folders — act on what is \
             inside it instead"
        );
    }
    if tokio::fs::symlink_metadata(&source.absolute).await.is_err() {
        anyhow::bail!(
            "cannot {verb} {}: no such file or folder",
            source.relative.display()
        );
    }
    Ok(())
}

/// Refuse a destination name the product keeps for itself.
fn refuse_reserved(target: &Path) -> Result<()> {
    if let Some(area) = closed_area(target) {
        anyhow::bail!(
            "cannot put it there: `{area}/` is kept by the product and closed to this tool"
        );
    }
    if is_root_memory(target) {
        anyhow::bail!(
            "cannot put it there: `{}` is the workspace's own memory file — correct it in place \
             instead",
            crate::prompt::MEMORY_FILE
        );
    }
    Ok(())
}

/// Refuse a destination the product's own folders claim: nothing new lands in `generated/` or
/// `uploads/` (the sweep empties both), and `archive/` is entered only through its own
/// destination or from inside itself.
fn refuse_destination(source: &Path, target: &Path) -> Result<()> {
    refuse_reserved(target)?;
    let Some(area) = first_area(target) else {
        return Ok(());
    };
    if area == ARCHIVE_DIR {
        if first_area(source) == Some(ARCHIVE_DIR) {
            return Ok(());
        }
        anyhow::bail!(
            "cannot put it there: `archive/` takes an item through the `archive` destination, \
             which shelves it under today's date"
        );
    }
    anyhow::bail!(
        "cannot put it there: `{area}/` is one of the two folders the media sweep empties — move \
         the item into a folder of your own instead"
    )
}

/// A new name: one bare component, with no folder and no `..` in it.
fn bare_name(name: &str) -> Result<String> {
    let mut components = Path::new(name).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(only)), None) => Ok(only.to_string_lossy().into_owned()),
        _ => anyhow::bail!("`name` must be a bare name with no folder in it, got `{name}`"),
    }
}

/// The optional new name a `move` gives the item on arrival.
fn optional_name(args: &serde_json::Value) -> Result<Option<String>> {
    let name = arg(args, "name");
    if name.is_empty() {
        return Ok(None);
    }
    bare_name(&name).map(Some)
}

/// Whether `target` would land inside `source`, compared on real paths: a case-varied spelling
/// names the same folder on a case-insensitive volume, and the destination may not exist yet, so
/// its frame is measured from the deepest ancestor that does.
async fn moves_inside(source: &Path, target: &Path) -> bool {
    let Ok(real_source) = tokio::fs::canonicalize(source).await else {
        return false;
    };
    let mut ancestor = target.parent();
    while let Some(path) = ancestor {
        if let Ok(real) = tokio::fs::canonicalize(path).await {
            return real.starts_with(&real_source);
        }
        ancestor = path.parent();
    }
    false
}

/// Whether two spellings name the same entry on disk — a case-only rename on a case-insensitive
/// volume is the one collision that is not one.
async fn same_entry(left: &Path, right: &Path) -> bool {
    match (
        tokio::fs::canonicalize(left).await,
        tokio::fs::canonicalize(right).await,
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Move one item into a folder of the caller's choosing, or onto the dated archive shelf.
async fn move_item(root: &Path, source: &Item, args: &serde_json::Value) -> Result<String> {
    refuse_source(source, "move").await?;
    let to = arg(args, "to");
    let name = optional_name(args)?;
    let archive = normalize_path(Path::new(&to))
        .to_str()
        .is_some_and(|spelled| spelled.eq_ignore_ascii_case(ARCHIVE_DIR));
    let mut target = if archive {
        // What came from a swept folder keeps its provenance under the day's own folder; an
        // item already on the shelf is shelved under its bare name rather than nested again.
        let mirrored = if first_area(&source.relative) == Some(ARCHIVE_DIR) {
            PathBuf::from(source.relative.file_name().unwrap_or_default())
        } else {
            source.relative.clone()
        };
        PathBuf::from(ARCHIVE_DIR)
            .join(chrono::Local::now().format("%Y-%m-%d").to_string())
            .join(mirrored)
    } else {
        let folder = workspace_relative(root, &to)?;
        let target_name: &OsStr = match &name {
            Some(name) => OsStr::new(name),
            None => source.relative.file_name().unwrap_or_default(),
        };
        folder.join(target_name)
    };
    if archive && let Some(name) = &name {
        target.set_file_name(name);
    }
    if archive {
        refuse_reserved(&target)?;
    } else {
        refuse_destination(&source.relative, &target)?;
    }
    place(root, source, &target, "move").await?;
    Ok(format!(
        "Moved {} → {}",
        source.relative.display(),
        target.display()
    ))
}

/// Rename one item in place.
async fn rename_item(root: &Path, source: &Item, args: &serde_json::Value) -> Result<String> {
    refuse_source(source, "rename").await?;
    if matches!(first_area(&source.relative), Some("generated" | "uploads")) {
        anyhow::bail!(
            "cannot rename {}: `generated/` and `uploads/` are temporary — move the item into a \
             folder of your own under the name you want instead",
            source.relative.display()
        );
    }
    let name = arg(args, "name");
    if name.is_empty() {
        anyhow::bail!("`name` must be the new name of the item");
    }
    let target = source
        .relative
        .parent()
        .unwrap_or(Path::new(""))
        .join(bare_name(&name)?);
    if target == source.relative {
        anyhow::bail!(
            "cannot rename {}: that is its current name",
            source.relative.display()
        );
    }
    refuse_destination(&source.relative, &target)?;
    place(root, source, &target, "rename").await?;
    Ok(format!(
        "Renamed {} → {}",
        source.relative.display(),
        target.display()
    ))
}

/// The move itself, shared by `move` and `rename`: nothing existing is overwritten, a folder
/// never lands inside itself, the destination folder is created when it is missing, and the
/// link refusal is applied to the whole destination chain.
///
/// The existence probe and the rename are two syscalls, so only a concurrent creator landing
/// exactly between them could be replaced; nothing in the product writes to that folder while an
/// agent's turn runs.
async fn place(root: &Path, source: &Item, target: &Path, verb: &str) -> Result<()> {
    let absolute = root.join(target);
    if absolute == source.absolute {
        anyhow::bail!(
            "cannot {verb} {}: it is already at that path",
            source.relative.display()
        );
    }
    let is_dir = tokio::fs::symlink_metadata(&source.absolute)
        .await
        .is_ok_and(|meta| meta.is_dir());
    if is_dir && moves_inside(&source.absolute, &absolute).await {
        anyhow::bail!(
            "cannot {verb} {}: a folder cannot be moved inside itself",
            source.relative.display()
        );
    }
    if tokio::fs::symlink_metadata(&absolute).await.is_ok()
        && !same_entry(&source.absolute, &absolute).await
    {
        anyhow::bail!(
            "cannot {verb} {}: {} already exists — choose another name",
            source.relative.display(),
            target.display()
        );
    }
    refuse_symlinks(root, target).await?;
    if let Some(parent) = absolute.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            anyhow::anyhow!(
                "cannot {verb} {}: the destination folder is unwritable: {e}",
                source.relative.display()
            )
        })?;
    }
    tokio::fs::rename(&source.absolute, &absolute)
        .await
        .map_err(|e| anyhow::anyhow!("cannot {verb} {}: {e}", source.relative.display()))
}

/// Delete one item for good: a file, or a folder whole with everything inside it. There is no
/// bin to recover it from, so a partial failure says plainly what is still there.
async fn delete_item(root: &Path, source: &Item) -> Result<String> {
    refuse_source(source, "delete").await?;
    let is_dir = tokio::fs::symlink_metadata(&source.absolute)
        .await
        .is_ok_and(|meta| meta.is_dir());
    let outcome = if is_dir {
        tokio::fs::remove_dir_all(&source.absolute).await
    } else {
        tokio::fs::remove_file(&source.absolute).await
    };
    let Err(error) = outcome else {
        return Ok(if is_dir {
            format!(
                "Deleted {} — the folder and everything inside it",
                source.relative.display()
            )
        } else {
            format!("Deleted {}", source.relative.display())
        });
    };
    let mut remains = Vec::new();
    collect_remaining(&source.absolute, &mut remains, REMAINS_NAMED + 1);
    if remains.is_empty() {
        return Ok(format!("Deleted {}", source.relative.display()));
    }
    let more = remains.len() > REMAINS_NAMED;
    remains.truncate(REMAINS_NAMED);
    let listed: Vec<String> = remains
        .iter()
        .filter_map(|path| path.strip_prefix(root).ok())
        .map(|path| path.display().to_string())
        .collect();
    anyhow::bail!(
        "deleting {} did not finish: {error}; still there: {}{}",
        source.relative.display(),
        listed.join(", "),
        if more { ", …" } else { "" }
    )
}

/// Collect up to `cap` entries still present under `path`, the path itself included. An entry
/// that is gone is not reported, and a link is named but never followed: what a partial delete
/// left behind is what is on disk, not what a link points at.
fn collect_remaining(path: &Path, out: &mut Vec<PathBuf>, cap: usize) {
    if out.len() >= cap {
        return;
    }
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    out.push(path.to_path_buf());
    if !meta.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        collect_remaining(&entry.path(), out, cap);
        if out.len() >= cap {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_at(root: &Path, name: &str) -> Workspace {
        crate::workspace::test_ws_named(&root.to_string_lossy(), name)
    }

    async fn run(ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        FilesTool.execute(ws, args).await
    }

    fn read(root: &Path, relative: &str) -> String {
        std::fs::read_to_string(root.join(relative)).unwrap()
    }

    /// One workspace with a note, a swept output, a swept attachment, a folder and the memory.
    fn seed(root: &Path) {
        for (name, body) in [
            ("MEMORY.md", "memory"),
            ("notes/old.md", "old"),
            ("notes/keep.md", "keep"),
            ("generated/pic.png", "picture"),
            ("uploads/inbox.pdf", "attachment"),
            ("drafts/inner/draft.md", "nested"),
        ] {
            let path = root.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }
    }

    fn setup() -> (tempfile::TempDir, std::path::PathBuf, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        seed(&root);
        let ws = workspace_at(&root, "personal:alice");
        (dir, root, ws)
    }

    fn today() -> String {
        chrono::Local::now().format("%Y-%m-%d").to_string()
    }

    /// A move creates the folder it names, renames on the way when asked, and the top of the
    /// workspace is a destination of its own.
    #[tokio::test]
    async fn move_creates_the_destination_folder_and_renames_on_the_way() {
        let (_dir, root, ws) = setup();
        assert_eq!(
            run(
                &ws,
                json!({"operation": "move", "path": "notes/old.md", "to": "album/2026", "name": "retired.md"})
            )
            .await
            .unwrap(),
            "Moved notes/old.md → album/2026/retired.md"
        );
        assert!(!root.join("notes/old.md").exists());
        assert_eq!(read(&root, "album/2026/retired.md"), "old");

        assert_eq!(
            run(
                &ws,
                json!({"operation": "move", "path": "album/2026/retired.md", "to": "."})
            )
            .await
            .unwrap(),
            "Moved album/2026/retired.md → retired.md"
        );
        assert_eq!(read(&root, "retired.md"), "old");
    }

    /// The two folders the sweep empties are fine sources and never destinations.
    #[tokio::test]
    async fn move_takes_files_out_of_the_swept_folders() {
        let (_dir, root, ws) = setup();
        run(
            &ws,
            json!({"operation": "move", "path": "generated/pic.png", "to": "album"}),
        )
        .await
        .unwrap();
        run(
            &ws,
            json!({"operation": "move", "path": "uploads/inbox.pdf", "to": "album", "name": "letter.pdf"}),
        )
        .await
        .unwrap();
        assert_eq!(read(&root, "album/pic.png"), "picture");
        assert_eq!(read(&root, "album/letter.pdf"), "attachment");
        assert!(!root.join("generated/pic.png").exists());
        assert!(!root.join("uploads/inbox.pdf").exists());
    }

    /// A rename stays in its folder, and the two temporary folders take none: the same result
    /// comes from moving the item out under the wanted name.
    #[tokio::test]
    async fn rename_is_in_place_and_refused_inside_the_swept_folders() {
        let (_dir, root, ws) = setup();
        assert_eq!(
            run(
                &ws,
                json!({"operation": "rename", "path": "notes/old.md", "name": "new.md"})
            )
            .await
            .unwrap(),
            "Renamed notes/old.md → notes/new.md"
        );
        assert_eq!(read(&root, "notes/new.md"), "old");

        for swept in ["generated/pic.png", "uploads/inbox.pdf"] {
            let err = run(
                &ws,
                json!({"operation": "rename", "path": swept, "name": "renamed.bin"}),
            )
            .await
            .expect_err("a swept folder takes no rename");
            assert!(err.to_string().contains("temporary"), "{err}");
            assert!(root.join(swept).exists(), "{swept} must be untouched");
        }
    }

    /// The dated shelf is one destination among the others and keeps the source's own path.
    #[tokio::test]
    async fn the_archive_destination_keeps_the_relative_path_under_todays_folder() {
        let (_dir, root, ws) = setup();
        let day = today();
        assert_eq!(
            run(
                &ws,
                json!({"operation": "move", "path": "notes/old.md", "to": "archive"})
            )
            .await
            .unwrap(),
            format!("Moved notes/old.md → archive/{day}/notes/old.md")
        );
        assert_eq!(read(&root, &format!("archive/{day}/notes/old.md")), "old");

        // An item already on the shelf is shelved under its bare name, not nested again.
        assert_eq!(
            run(
                &ws,
                json!({"operation": "move", "path": format!("archive/{day}/notes/old.md"), "to": "archive"})
            )
            .await
            .unwrap(),
            format!("Moved archive/{day}/notes/old.md → archive/{day}/old.md")
        );
    }

    /// A rename inside the archive shelf is ordinary work on its contents, and the shelf is
    /// recognised through any spelling of its own name.
    #[tokio::test]
    async fn the_archive_shelf_takes_renames_and_any_spelling_of_its_name() {
        let (_dir, root, ws) = setup();
        let day = today();
        run(
            &ws,
            json!({"operation": "move", "path": "notes/old.md", "to": "./archive"}),
        )
        .await
        .unwrap();
        assert_eq!(
            run(
                &ws,
                json!({"operation": "rename", "path": format!("archive/{day}/notes/old.md"), "name": "older.md"}),
            )
            .await
            .unwrap(),
            format!("Renamed archive/{day}/notes/old.md → archive/{day}/notes/older.md")
        );
        assert_eq!(read(&root, &format!("archive/{day}/notes/older.md")), "old");
    }

    /// What a partial delete reports is what is still on disk, and a link is never followed into
    /// another tree.
    #[tokio::test]
    async fn remaining_names_the_leftovers_without_following_links() {
        let (_dir, root, ws) = setup();
        let mut remains = Vec::new();
        collect_remaining(&root.join("drafts"), &mut remains, 20);
        assert_eq!(remains.len(), 3, "{remains:?}");

        let mut capped = Vec::new();
        collect_remaining(&root.join("drafts"), &mut capped, 2);
        assert_eq!(capped.len(), 2, "{capped:?}");

        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            std::fs::write(outside.path().join("precious.md"), "x").unwrap();
            std::os::unix::fs::symlink(outside.path(), root.join("drafts/escape")).unwrap();
            let mut remains = Vec::new();
            collect_remaining(&root.join("drafts"), &mut remains, 20);
            assert_eq!(remains.len(), 4, "{remains:?}");
            assert!(
                remains
                    .iter()
                    .all(|path| !path.to_string_lossy().contains("precious")),
                "a link is named, never followed: {remains:?}"
            );

            // Deleting the folder takes the link, never what it points at.
            run(&ws, json!({"operation": "delete", "path": "drafts"}))
                .await
                .unwrap();
            assert!(outside.path().join("precious.md").exists());
        }
    }

    /// A file goes, and a folder goes whole with everything inside it — the swept folders and
    /// the archive shelf included.
    #[tokio::test]
    async fn delete_takes_a_file_and_a_folder_whole() {
        let (_dir, root, ws) = setup();
        assert_eq!(
            run(&ws, json!({"operation": "delete", "path": "notes/old.md"}))
                .await
                .unwrap(),
            "Deleted notes/old.md"
        );
        assert!(!root.join("notes/old.md").exists());
        assert!(root.join("notes/keep.md").exists());

        let answer = run(&ws, json!({"operation": "delete", "path": "drafts"}))
            .await
            .unwrap();
        assert!(
            answer.contains("folder and everything inside it"),
            "{answer}"
        );
        assert!(!root.join("drafts").exists());

        let day = today();
        std::fs::create_dir_all(root.join(format!("archive/{day}"))).unwrap();
        std::fs::write(root.join(format!("archive/{day}/retired.md")), "old").unwrap();
        for gone in [
            "generated/pic.png",
            "uploads/inbox.pdf",
            &format!("archive/{day}"),
        ] {
            run(&ws, json!({"operation": "delete", "path": gone}))
                .await
                .unwrap();
            assert!(!root.join(gone).exists(), "{gone} must be gone");
        }
    }

    /// Nothing existing is overwritten, and a rename that only changes the case still works.
    #[tokio::test]
    async fn nothing_is_overwritten_and_a_case_only_rename_still_works() {
        let (_dir, root, ws) = setup();
        let err = run(
            &ws,
            json!({"operation": "move", "path": "notes/keep.md", "to": "notes", "name": "old.md"}),
        )
        .await
        .expect_err("a taken name is a refusal");
        assert!(err.to_string().contains("already exists"), "{err}");
        assert_eq!(read(&root, "notes/old.md"), "old");
        assert_eq!(read(&root, "notes/keep.md"), "keep");

        run(
            &ws,
            json!({"operation": "rename", "path": "notes/keep.md", "name": "Keep.md"}),
        )
        .await
        .unwrap();
        let names: Vec<String> = std::fs::read_dir(root.join("notes"))
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"Keep.md".to_string()), "{names:?}");
        assert!(!names.contains(&"keep.md".to_string()), "{names:?}");
    }

    /// The source frame: the workspace itself, the root memory, the product's own folders, the
    /// closed areas, symlinks and everything outside the personal workspace are refused.
    #[tokio::test]
    async fn refusals_cover_the_root_the_memory_the_closed_areas_and_symlinks() {
        let (_dir, root, ws) = setup();
        let refused = |err: anyhow::Error, expected: &str| {
            assert!(
                err.to_string().contains(expected),
                "expected `{expected}`, got: {err}"
            );
        };

        refused(
            run(&ws, json!({"operation": "delete", "path": "."}))
                .await
                .unwrap_err(),
            "the workspace itself",
        );
        refused(
            run(&ws, json!({"operation": "delete", "path": "MEMORY.md"}))
                .await
                .unwrap_err(),
            "the memory is corrected",
        );
        for folder in ["generated", "uploads", "archive", "GENERATED"] {
            refused(
                run(&ws, json!({"operation": "delete", "path": folder}))
                    .await
                    .unwrap_err(),
                "one of the workspace's own folders",
            );
        }
        for closed in [
            "shared/tool.ts",
            ".git/config",
            ".compaction/store.db",
            ".hyperframes/state",
            ".secrets/key",
        ] {
            refused(
                run(&ws, json!({"operation": "delete", "path": closed}))
                    .await
                    .unwrap_err(),
                "closed to this tool",
            );
        }
        refused(
            run(
                &ws,
                json!({"operation": "move", "path": "notes/old.md", "to": "../escape"}),
            )
            .await
            .unwrap_err(),
            "outside your personal workspace",
        );
        refused(
            run(
                &ws,
                json!({"operation": "rename", "path": "notes/old.md", "name": "a/b.md"}),
            )
            .await
            .unwrap_err(),
            "bare name",
        );

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("notes/old.md"), root.join("link.md")).unwrap();
            refused(
                run(&ws, json!({"operation": "delete", "path": "link.md"}))
                    .await
                    .unwrap_err(),
                "is a symlink",
            );
        }
    }

    /// The destination frame: neither swept folder takes anything, the archive is entered only
    /// through its own destination, the root memory is not a name to write, and no link in the
    /// chain is followed.
    #[tokio::test]
    async fn destinations_never_enter_the_swept_folders_the_memory_or_a_symlink() {
        let (_dir, root, ws) = setup();
        let refused = |err: anyhow::Error, expected: &str| {
            assert!(
                err.to_string().contains(expected),
                "expected `{expected}`, got: {err}"
            );
        };

        for destination in [
            "generated",
            "uploads",
            "generated/nested",
            "archive/2026-01-01",
        ] {
            refused(
                run(
                    &ws,
                    json!({"operation": "move", "path": "notes/old.md", "to": destination}),
                )
                .await
                .unwrap_err(),
                "cannot put it there",
            );
        }
        refused(
            run(&ws, json!({"operation": "move", "path": "notes/old.md", "to": ".", "name": "generated"}))
                .await
                .unwrap_err(),
            "cannot put it there",
        );
        refused(
            run(&ws, json!({"operation": "move", "path": "notes/old.md", "to": ".", "name": "memory.md"}))
                .await
                .unwrap_err(),
            "memory file",
        );

        #[cfg(unix)]
        {
            // A link in the destination chain is refused before anything is created through it.
            std::os::unix::fs::symlink(root.join("generated"), root.join("into")).unwrap();
            refused(
                run(
                    &ws,
                    json!({"operation": "move", "path": "notes/old.md", "to": "into"}),
                )
                .await
                .unwrap_err(),
                "is a symlink",
            );
            assert!(root.join("notes/old.md").exists());
            assert!(!root.join("generated/old.md").exists());
        }
    }

    /// A managed workspace has no personal folder to arrange, and the tool says so.
    #[tokio::test]
    async fn the_tool_only_serves_a_personal_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("x.md"), "x").unwrap();
        let ws = workspace_at(&root, "proj");
        let err = run(&ws, json!({"operation": "delete", "path": "x.md"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("is not one"), "{err}");
    }
}

//! The `archive` tool: move one file or folder out of the way without losing
//! it.
//!
//! A guest Assistant has no shell, so without this its only way to retire a file
//! was to empty or overwrite it — leaving a stale file that reads later exactly
//! like a current one. The item, a folder whole with its internal structure,
//! lands at `archive/<year>-<month>-<day>/<its own relative path>` inside the
//! same personal workspace, so where it came from stays recognisable.
//!
//! The source goes through the ordinary read gate (existence, the workspace
//! frame, a canonical target inside it), the workspace's own [`SERVICE_AREAS`]
//! are refused, and so is the root `MEMORY.md`, which is corrected rather than
//! archived. The admin's Assistant has a shell and needs no such tool, so this
//! one is built for a guest alone (see `assistant_core_tools` in `agent/role.rs`).

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::path::{PathAccess, resolve_read_target};
use crate::{Tool, Workspace};

/// The folder archived items land in, inside the personal workspace.
const ARCHIVE_DIR: &str = "archive";

/// The workspace's own service areas — top-level names that are never
/// archivable: `.compaction/` is the compaction store and `.git/` the repository
/// `ensure_personal_workspace` initialises for the personal-folder change
/// baseline.
const SERVICE_AREAS: &[&str] = &[
    "shared",
    "uploads",
    "generated",
    ".compaction",
    ".git",
    ARCHIVE_DIR,
];

/// The memory file at the workspace root: corrected, never archived.
const MEMORY_FILE: &str = "MEMORY.md";

pub(crate) struct ArchiveTool;

#[async_trait]
impl Tool for ArchiveTool {
    fn name(&self) -> &'static str {
        "archive"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &json!({
                "path": {
                    "type": "string",
                    "description": "The one file or folder to archive, relative to your workspace root."
                }
            }),
            &["path"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: serde_json::Value) -> Result<String> {
        if !crate::users::is_personal_workspace(&ws.name) {
            anyhow::bail!(
                "cannot archive: this tool moves things inside the caller's own personal \
                 workspace, and `{}` is not one",
                ws.name
            );
        }
        let path = args
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim();
        if path.is_empty() {
            anyhow::bail!("cannot archive: `path` must name the one file or folder to archive");
        }

        // The product's own spelling of a canonical path: a verbatim prefix on
        // Windows is stripped, exactly as it is on the resolved source, so the
        // two are measured in the same frame.
        let root = crate::stored_workspace_path(ws.as_path());
        let source = resolve_read_target(&root, path, PathAccess::Workspace).await?;
        let relative = source.strip_prefix(&root).map_err(|_| {
            anyhow::anyhow!("cannot archive {path}: it is outside your personal workspace")
        })?;
        if relative.as_os_str().is_empty() {
            anyhow::bail!("cannot archive {path}: that is the workspace itself");
        }
        let top = relative
            .components()
            .next()
            .and_then(|component| component.as_os_str().to_str())
            .unwrap_or_default();
        if SERVICE_AREAS.contains(&top) {
            let areas: Vec<String> = SERVICE_AREAS
                .iter()
                .map(|area| format!("`{area}/`"))
                .collect();
            anyhow::bail!(
                "cannot archive {path}: `{top}/` is one of the workspace's own service areas ({})",
                areas.join(", ")
            );
        }
        if relative == Path::new(MEMORY_FILE) {
            anyhow::bail!(
                "cannot archive {path}: the memory is corrected, not archived — edit the stale \
                 part out of {MEMORY_FILE} instead"
            );
        }

        let day = chrono::Local::now().format("%Y-%m-%d").to_string();
        let folder = root.join(ARCHIVE_DIR).join(day);
        let destination = free_destination(&folder, relative).await;
        let parent = destination.parent().unwrap_or(&folder);
        // A rename needs its destination folder to exist, so the day's folder is
        // created first: a rename that then fails leaves it empty rather than
        // never created.
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            anyhow::anyhow!("cannot archive {path}: the archive folder is unwritable: {e}")
        })?;
        tokio::fs::rename(&source, &destination)
            .await
            .map_err(|e| {
                anyhow::anyhow!("cannot archive {path}: moving it to the archive failed: {e}")
            })?;

        let moved_to = destination
            .strip_prefix(&root)
            .unwrap_or(&destination)
            .display();
        Ok(format!("Archived {} → {moved_to}", relative.display()))
    }
}

/// The free destination for `relative` under `folder`: its own name, or that
/// name with `-2`, `-3`, … appended when it is taken.
///
/// An entry is anything at that path — a file, a folder, a dangling symlink — so
/// a move never lands on something that already exists.
async fn free_destination(folder: &Path, relative: &Path) -> PathBuf {
    let parent = relative
        .parent()
        .map_or_else(|| folder.to_path_buf(), |parent| folder.join(parent));
    let name = relative.file_name().unwrap_or_default();
    let mut candidate = parent.join(name);
    for n in 2.. {
        if tokio::fs::symlink_metadata(&candidate).await.is_err() {
            break;
        }
        candidate = parent.join(suffixed(name, n));
    }
    candidate
}

/// `name` with `-{n}` appended (`notes.md` → `notes.md-2`): one suffix rule for
/// files and folders alike, and the name's own bytes are kept, so a non-UTF-8
/// name is never rewritten lossily.
#[must_use]
fn suffixed(name: &OsStr, n: u32) -> OsString {
    let mut suffixed = name.to_os_string();
    suffixed.push(format!("-{n}"));
    suffixed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_at(root: &Path, name: &str) -> Workspace {
        crate::workspace::test_ws_named(&root.to_string_lossy(), name)
    }

    async fn archive(ws: &Workspace, path: &str) -> anyhow::Result<String> {
        ArchiveTool.execute(ws, json!({ "path": path })).await
    }

    /// The day the move happens, in the tool's own spelling.
    fn today() -> String {
        chrono::Local::now().format("%Y-%m-%d").to_string()
    }

    /// One file and one folder keep their relative paths under the day's own
    /// folder, and a second item landing on a taken name gets a numeric suffix.
    #[tokio::test]
    async fn items_keep_their_relative_paths_and_a_taken_name_gets_a_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let ws = workspace_at(&root, "personal:alice");
        std::fs::create_dir_all(root.join("notes")).unwrap();
        std::fs::write(root.join("notes/old.md"), "first").unwrap();
        std::fs::create_dir_all(root.join("drafts/inner")).unwrap();
        std::fs::write(root.join("drafts/inner/draft.md"), "nested").unwrap();
        assert!(
            !root.join(ARCHIVE_DIR).exists(),
            "the archive folder waits for the first move"
        );

        assert_eq!(
            archive(&ws, "notes/old.md").await.unwrap(),
            format!("Archived notes/old.md → archive/{}/notes/old.md", today())
        );
        assert!(!root.join("notes/old.md").exists());
        assert_eq!(
            std::fs::read_to_string(root.join(format!("archive/{}/notes/old.md", today())))
                .unwrap(),
            "first"
        );

        // The same relative path archived again: the taken name is kept whole.
        std::fs::write(root.join("notes/old.md"), "second").unwrap();
        assert_eq!(
            archive(&ws, "notes/old.md").await.unwrap(),
            format!("Archived notes/old.md → archive/{}/notes/old.md-2", today())
        );
        assert_eq!(
            std::fs::read_to_string(root.join(format!("archive/{}/notes/old.md", today())))
                .unwrap(),
            "first"
        );
        assert_eq!(
            std::fs::read_to_string(root.join(format!("archive/{}/notes/old.md-2", today())))
                .unwrap(),
            "second"
        );

        // A folder moves whole, its internal structure kept.
        assert_eq!(
            archive(&ws, "drafts").await.unwrap(),
            format!("Archived drafts → archive/{}/drafts", today())
        );
        assert!(!root.join("drafts").exists());
        assert_eq!(
            std::fs::read_to_string(
                root.join(format!("archive/{}/drafts/inner/draft.md", today()))
            )
            .unwrap(),
            "nested"
        );
        // A folder landing on a taken name is suffixed the same way.
        std::fs::create_dir_all(root.join("drafts")).unwrap();
        assert_eq!(
            archive(&ws, "drafts").await.unwrap(),
            format!("Archived drafts → archive/{}/drafts-2", today())
        );
    }

    /// Everything the tool must not touch is refused, and a refusal creates no
    /// archive folder.
    #[tokio::test]
    async fn refusals_cover_the_memory_the_service_areas_and_anything_outside() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let ws = workspace_at(&root, "personal:alice");
        for area in [
            "shared",
            "uploads",
            "generated",
            ".compaction",
            ".git",
            "archive",
        ] {
            std::fs::create_dir_all(root.join(area)).unwrap();
            std::fs::write(root.join(area).join("stale.md"), "x").unwrap();
        }
        std::fs::write(root.join("MEMORY.md"), "facts").unwrap();
        std::fs::write(root.join("notes.md"), "x").unwrap();

        for path in [
            "",
            "MEMORY.md",
            "shared/stale.md",
            "uploads",
            "generated/stale.md",
            ".compaction/stale.md",
            ".git/stale.md",
            "archive/stale.md",
            "missing.md",
            "../escape.md",
        ] {
            assert!(
                archive(&ws, path).await.is_err(),
                "`{path}` must be refused"
            );
        }
        assert!(
            !root.join(ARCHIVE_DIR).join(today()).exists(),
            "a refusal archives nothing — not even the day's folder"
        );

        // A managed workspace is not a personal one: no pathway between accounts.
        let project = workspace_at(&root, "my-project");
        let err = archive(&project, "notes.md").await.unwrap_err().to_string();
        assert!(err.contains("personal workspace"), "{err}");
    }
}

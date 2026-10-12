//! The optional `place` argument of the two dispatch tools (`analyze` and
//! `send`): a holder that declares it — the admin Assistant, and a custom role
//! whose grants rebuild the tool — can point a round at a chosen directory
//! instead of its own workspace.
//!
//! Resolution is exact and every refusal is explicit — a place that resolves
//! to nothing usable never falls back to the caller's own workspace. A value is
//! either a registered workspace named exactly (case-sensitively) or an
//! absolute existing directory; an absolute path that IS a registered
//! workspace is that workspace, with its identity and its stored context, never
//! a bare directory.
//!
//! The resolved directory becomes the round's EXECUTION workspace (the
//! sub-agents' root, context and search key) while the round's durable delivery
//! key stays the caller's own workspace name (`jobs.workspace_name`), so the
//! result always lands in the conversation that asked.
//!
//! A coder round additionally refuses a registered workspace, anything inside
//! one, the filesystem root and the product's own data directory: a ticket-less
//! coder must never write into a tree whose workspace pipeline owns it. These
//! are guardrails, not a security boundary — the coder's shell is unchanged.

use crate::Workspace;
use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};

/// Which delegation tool is resolving a place — the two differ in what they
/// refuse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DelegationKind {
    /// The read-only analyst round: a registered workspace and a plain
    /// directory are both legitimate places.
    Analyze,
    /// The coder round: refuses registered workspaces (and anything inside
    /// them) and the product's own data directory.
    Coder,
}

/// The registered workspace occupying exactly `dir` — "this path IS that
/// workspace, with its identity and its stored context". `Path` equality is
/// component-based, so it tolerates the trailing separator a registered
/// `workspaces.path` carries.
fn registered_at(registered: &[Workspace], dir: &Path) -> Option<Workspace> {
    registered.iter().find(|ws| dir == ws.as_path()).cloned()
}

/// Resolve a `place` that must address a registered project — by its exact
/// registered name or by that project's own folder. Any other folder (a personal
/// workspace included) is refused: the Manager exists only for registered
/// projects.
pub(crate) async fn resolve_project(raw: &str) -> Result<Workspace> {
    let raw = raw.trim();
    if !raw.is_empty() {
        let expanded = crate::util::expand_tilde(raw);
        if expanded.is_absolute() {
            let dir = canonical_dir(raw, &expanded)?;
            if let Some(ws) = registered_at(&registered_workspaces().await?, &dir) {
                return Ok(ws);
            }
        } else if let Some(ws) = crate::workspace::get_by_name(raw).await? {
            return Ok(ws);
        }
    }
    Err(refuse_project(raw))
}

fn refuse_project(raw: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place \"{raw}\" is not a registered project — hint: pass a registered project's \
         name or its own folder; personal workspaces have no Manager"
    )
}

/// Resolve the optional `place` argument into the workspace the round runs in.
///
/// `Ok(None)` means no place — the round runs in the caller's own workspace
/// exactly as before; an empty or whitespace-only value counts as no place.
/// Everything else is decided here and now, before anything is dispatched.
pub(crate) async fn resolve(kind: DelegationKind, args: &Value) -> Result<Option<Workspace>> {
    let raw = match args.get("place") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => s.as_str(),
        Some(other) => return Err(super::wrong_type("place", "a string", other)),
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }

    // A value that is not an absolute path can only be a registered workspace
    // name: registered names are letters and underscores, never a path.
    let expanded = crate::util::expand_tilde(raw);
    if !expanded.is_absolute() {
        let Some(ws) = crate::workspace::get_by_name(raw).await? else {
            return Err(refuse_unresolvable(kind, raw));
        };
        if kind == DelegationKind::Coder {
            return Err(refuse_registered(raw, &ws.name));
        }
        return Ok(Some(ws));
    }

    let dir = canonical_dir(raw, &expanded)?;
    let registered = registered_workspaces().await?;
    let exact = registered_at(&registered, &dir);
    if kind == DelegationKind::Coder {
        if let Some(ws) = exact {
            return Err(refuse_registered(raw, &ws.name));
        }
        if let Some(ws) = registered
            .iter()
            .find(|ws| crate::util::is_within(&dir, ws.as_path()))
        {
            return Err(refuse_inside(raw, &ws.name));
        }
        if let Some(root) = data_dir(&dir) {
            return Err(refuse_data_dir(raw, &root));
        }
        return Ok(Some(dir_workspace(&dir)));
    }
    // The analyst may be pointed at a registered workspace: it then carries
    // that workspace's identity and stored context, never a bare directory.
    match exact {
        Some(ws) => Ok(Some(ws)),
        None => Ok(Some(dir_workspace(&dir))),
    }
}

/// The execution workspace for a directory recorded on a round that already
/// started (`jobs.exec_dir`): the registered workspace at that path when one
/// exists now, else a directory-scoped synthetic one. `Err` when the directory
/// is gone or unusable — that failure ends the resumed round instead of leaving
/// it hanging. A recorded directory is never re-refused: the refusals are the
/// dispatch-time guardrail.
pub(crate) async fn workspace_at_dir(dir: &str) -> Result<Workspace> {
    let path = Path::new(dir);
    let (_, is_dir) = path_state(path);
    if !is_dir {
        bail!("the place directory {dir} is no longer available");
    }
    let canonical = crate::stored_workspace_path(path);
    match registered_at(&registered_workspaces().await?, &canonical) {
        Some(ws) => Ok(ws),
        None => Ok(dir_workspace(&canonical)),
    }
}

/// The registered workspaces — nothing else in the product looks a workspace up
/// by its path, so the path-identity rules read the (small) table here.
async fn registered_workspaces() -> Result<Vec<Workspace>> {
    crate::workspace::store()
        .list()
        .await
        .context("list registered workspaces")
}

/// Canonicalize an existing directory place, refusing the filesystem root (it
/// is not a project directory, and every path would count as inside it).
fn canonical_dir(raw: &str, expanded: &Path) -> Result<PathBuf> {
    let (exists, is_dir) = path_state(expanded);
    if !exists {
        return Err(refuse_missing(raw));
    }
    if !is_dir {
        return Err(refuse_not_a_directory(raw));
    }
    let canonical = crate::stored_workspace_path(expanded);
    if canonical.parent().is_none() {
        return Err(refuse_root(raw));
    }
    Ok(canonical)
}

/// A place path's metadata, off the async worker pool (`with_block_in_place`
/// per the util's convention for fast blocking syscalls in async callers):
/// whether it exists and whether it is a directory.
fn path_state(path: &Path) -> (bool, bool) {
    crate::util::with_block_in_place(|| (path.exists(), path.is_dir()))
}

fn refuse_missing(raw: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place path \"{raw}\" does not exist — hint: pass an absolute path to an \
         existing directory"
    )
}

fn refuse_not_a_directory(raw: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place path \"{raw}\" is a file, not a directory — hint: pass an absolute \
         path to a directory"
    )
}

fn refuse_root(raw: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place \"{raw}\" is the filesystem root — hint: pass a project directory instead"
    )
}

/// The product's own data directory when `dir` lies inside it OUTSIDE the
/// personal workspaces — a personal workspace directory is a legitimate
/// place, the `userspaces` container itself is not.
///
/// The storage root is only the containment base: it is derived as the personal
/// root's parent, so the fallback rule lives in exactly one place
/// ([`crate::users::userspaces_root`]), and the refusal message still names it.
/// It is spelled canonically like `dir` (macOS puts the temp root behind the
/// `/var` → `/private/var` symlink), independently of whether the personal root
/// itself exists yet.
fn data_dir(dir: &Path) -> Option<PathBuf> {
    let personal = crate::stored_workspace_path(&crate::users::userspaces_root());
    let root = crate::stored_workspace_path(personal.parent()?);
    if dir != personal && crate::util::is_within(dir, &personal) {
        return None;
    }
    crate::util::is_within(dir, &root).then_some(root)
}

/// Drop the search engine a delegated round created for its synthetic `dir-…`
/// workspace: its lifetime is bounded by the run that served it, exactly like a
/// research run root — a one-off directory must never keep a picker and a file
/// watcher alive forever. A round in a registered folder uses that workspace's
/// own engine and is left alone.
///
/// Accepted trade-off: two rounds placed in the same directory share one
/// `dir-…` name, so releasing while another such round is still running only
/// costs that round a picker re-scan on its next search — never a wrong result.
/// A round cut by drain keeps its engine for the resume that reuses it; a cancel
/// of that round releases it, even after the directory itself is gone — see
/// [`release_placed_engine`].
pub(crate) fn release_exec_engine(exec: &Workspace) {
    if !exec.name.starts_with(DIR_WORKSPACE_PREFIX) || !crate::search_engine::registry_initialized()
    {
        return;
    }
    crate::search_engine::remove_engine(&exec.name);
}

/// Release the search engine a round placed in `dir` created, named by the
/// directory alone: the synthetic name is a pure function of the canonical path,
/// so this reaches the engine even when the directory is no longer there — which
/// is exactly the state a cancel can leave (its rows are gone too, so the round's
/// own exit never runs and nothing else will release it). A registered folder's
/// engine is not touched: the derivation only ever names a `dir-…` workspace, and
/// a name nothing handed an engine releases nothing.
pub(crate) fn release_placed_engine(dir: &str) {
    let canonical = crate::stored_workspace_path(Path::new(dir));
    release_exec_engine(&dir_workspace(&canonical));
}

/// The name prefix of a directory-scoped synthetic workspace — impossible in a
/// registered name (`validate_name` allows only letters and underscores), so it
/// identifies one unambiguously.
const DIR_WORKSPACE_PREFIX: &str = "dir-";

/// A directory-scoped execution workspace for a delegation round placed in a
/// folder. The name is `dir-<basename>-<hash of the canonical path>`: the
/// embedded `-` is impossible in a registered workspace name (`validate_name`
/// allows only letters and underscores), so a name-keyed surface — the
/// per-workspace search index above all — can never hand this directory a
/// registered workspace's state, and the path hash keeps two directories apart.
/// The derivation is deterministic, so a resumed round re-derives the same
/// agent/session/search identity.
///
/// The short key is only ever an in-memory execution identity, bounded by the
/// round that created it — never a durable launch address.
fn dir_workspace(canonical: &Path) -> Workspace {
    let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
    let hash = crate::util::hex_string(&digest[..6]);
    let base: String = canonical
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(32)
        .collect();
    let base = if base.is_empty() { "dir" } else { &base };
    Workspace::ephemeral_run(&format!("{DIR_WORKSPACE_PREFIX}{base}-{hash}"), canonical)
}

fn refuse_unresolvable(kind: DelegationKind, raw: &str) -> anyhow::Error {
    match kind {
        DelegationKind::Analyze => anyhow::anyhow!(
            "usage: place \"{raw}\" is not a registered workspace and is not an absolute path — \
             hint: pass the exact name of a registered workspace, or an absolute path to an \
             existing directory"
        ),
        DelegationKind::Coder => anyhow::anyhow!(
            "usage: place \"{raw}\" is not an absolute path — hint: this tool only runs in a \
             directory outside the registered workspaces (a registered workspace name never works \
             here); pass an absolute path to an existing directory"
        ),
    }
}

fn refuse_registered(raw: &str, workspace: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place \"{raw}\" is the registered workspace \"{workspace}\" — hint: work in a \
         registered workspace goes through that workspace's board and its Manager, never through \
         this tool; pass an absolute path to a directory outside the registered workspaces"
    )
}

fn refuse_inside(raw: &str, workspace: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place path \"{raw}\" is inside the registered workspace \"{workspace}\" — hint: \
         work in a registered workspace goes through that workspace's board and its Manager, never \
         through this tool; pass an absolute path to a directory outside the registered workspaces"
    )
}

fn refuse_data_dir(raw: &str, root: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place path \"{raw}\" is inside MahBot's own data directory ({}) — hint: pass a \
         project directory instead; a personal workspace directory is allowed",
        root.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{create_test_workspace, init_management_test_stores, test_root};
    use serde_json::json;

    /// Canonicalize a real path the way [`resolve`] and `stored_workspace_path`
    /// do — the macOS temp dir sits behind the `/var` → `/private/var` symlink,
    /// so comparing raw `tempdir()` paths would never match.
    fn canon(path: &Path) -> PathBuf {
        crate::util::strip_verbatim_prefix(&std::fs::canonicalize(path).unwrap())
    }

    /// Register a workspace the way production rows are stored: the canonical
    /// path with the platform's trailing separator appended.
    async fn register_workspace(dir: &Path, name: &str) -> String {
        let stored = format!("{}{}", dir.display(), std::path::MAIN_SEPARATOR);
        create_test_workspace(&stored, name).await;
        stored
    }

    #[tokio::test]
    async fn absent_null_and_blank_places_mean_no_place() {
        init_management_test_stores().await;
        for kind in [DelegationKind::Analyze, DelegationKind::Coder] {
            for args in [
                json!({}),
                json!({"place": null}),
                json!({"place": ""}),
                json!({"place": "   "}),
            ] {
                let ws = resolve(kind, &args).await.expect("no place resolves");
                assert!(ws.is_none(), "{kind:?} / {args} must be no place");
            }
        }
    }

    #[tokio::test]
    async fn plain_directory_resolves_to_a_synthetic_workspace() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        let target = canonical.to_str().unwrap();

        for kind in [DelegationKind::Analyze, DelegationKind::Coder] {
            let ws = resolve(kind, &json!({ "place": target }))
                .await
                .expect("plain directory resolves")
                .expect("a target is present");
            assert_eq!(ws.path, target, "{kind:?}");
            assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
            assert!(ws.name.contains('-'), "got {}", ws.name);
        }

        let first = resolve(DelegationKind::Analyze, &json!({ "place": target }))
            .await
            .unwrap()
            .unwrap();
        let second = resolve(DelegationKind::Analyze, &json!({ "place": target }))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.name, second.name, "the derivation is deterministic");
    }

    #[tokio::test]
    async fn registered_workspace_resolves_by_name_and_by_path() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        let stored = register_workspace(&canonical, "ws_delegation_alpha").await;
        let plain = canonical.to_str().unwrap();

        // By exact name — the analyst round accepts it.
        let by_name = resolve(
            DelegationKind::Analyze,
            &json!({ "place": "ws_delegation_alpha" }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(by_name.name, "ws_delegation_alpha");
        assert_eq!(by_name.path, stored, "the registered spelling is kept");

        // The same directory by its separator-less absolute path is that
        // workspace — the path match tolerates the registered row's trailing
        // separator — identity and stored context included.
        let by_path = resolve(DelegationKind::Analyze, &json!({ "place": plain }))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_path.name, "ws_delegation_alpha");
        assert_eq!(by_path.path, stored);

        // A coder round refuses a registered workspace, by name and by path.
        for args in [
            json!({ "place": "ws_delegation_alpha" }),
            json!({ "place": plain }),
        ] {
            let err = resolve(DelegationKind::Coder, &args)
                .await
                .expect_err("the coder refuses a registered workspace")
                .to_string();
            assert!(err.starts_with("usage:"), "{err}");
            assert!(err.contains("ws_delegation_alpha"), "{err}");
            assert!(err.contains("board") && err.contains("Manager"), "{err}");
        }
    }

    /// The Manager addressee's `place` is one registered project — by its
    /// registered name or by that project's own folder; a plain directory (a
    /// personal workspace included) is refused.
    #[tokio::test]
    async fn project_place_resolves_by_name_and_folder_only() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        let plain = canonical.to_str().unwrap();
        register_workspace(&canonical, "ws_project_alpha").await;

        for place in ["ws_project_alpha", plain] {
            let ws = resolve_project(place).await.expect("a registered project");
            assert_eq!(ws.name, "ws_project_alpha", "{place}");
        }

        let other = tempfile::tempdir().unwrap();
        let err = resolve_project(other.path().to_str().unwrap())
            .await
            .expect_err("a plain directory has no Manager")
            .to_string();
        assert!(err.contains("not a registered project"), "{err}");
    }

    #[tokio::test]
    async fn directory_inside_a_registered_workspace_is_synthetic_for_analysts_only() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        register_workspace(&canonical, "ws_delegation_nested").await;
        let nested = canonical.join("inner");
        std::fs::create_dir(&nested).unwrap();
        let target = nested.to_str().unwrap();

        let ws = resolve(DelegationKind::Analyze, &json!({ "place": target }))
            .await
            .unwrap()
            .unwrap();
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
        assert_ne!(ws.name, "ws_delegation_nested");

        let err = resolve(DelegationKind::Coder, &json!({ "place": target }))
            .await
            .expect_err("the coder refuses anything inside a registered workspace")
            .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("ws_delegation_nested"), "{err}");
        assert!(err.contains("inside"), "{err}");
    }

    #[tokio::test]
    async fn unresolvable_places_are_each_refused_clearly() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());

        let unregistered = resolve(DelegationKind::Analyze, &json!({ "place": "no_such_ws" }))
            .await
            .expect_err("unregistered name")
            .to_string();
        assert!(unregistered.starts_with("usage:"), "{unregistered}");
        assert!(
            unregistered.contains("not a registered workspace"),
            "{unregistered}"
        );

        // The coder's variant must not send the caller down the registered-name
        // form it always refuses.
        let coder_unregistered = resolve(DelegationKind::Coder, &json!({ "place": "no_such_ws" }))
            .await
            .expect_err("unregistered name for the coder")
            .to_string();
        assert!(
            coder_unregistered.starts_with("usage:"),
            "{coder_unregistered}"
        );
        assert!(
            coder_unregistered.contains("never works here"),
            "{coder_unregistered}"
        );
        assert!(
            !coder_unregistered.contains("not a registered workspace"),
            "the coder's hint must not offer the registered-name form: {coder_unregistered}"
        );

        let relative = resolve(DelegationKind::Analyze, &json!({ "place": "sub/dir" }))
            .await
            .expect_err("relative value")
            .to_string();
        assert!(relative.starts_with("usage:"), "{relative}");
        assert!(relative.contains("not an absolute path"), "{relative}");

        let missing_path = canonical.join("does-not-exist");
        let missing = resolve(
            DelegationKind::Coder,
            &json!({ "place": missing_path.to_str().unwrap() }),
        )
        .await
        .expect_err("missing absolute path")
        .to_string();
        assert!(missing.starts_with("usage:"), "{missing}");
        assert!(missing.contains("does not exist"), "{missing}");

        let file = canonical.join("a-file.txt");
        std::fs::write(&file, "x").unwrap();
        let file_err = resolve(
            DelegationKind::Coder,
            &json!({ "place": file.to_str().unwrap() }),
        )
        .await
        .expect_err("a file place")
        .to_string();
        assert!(file_err.starts_with("usage:"), "{file_err}");
        assert!(file_err.contains("not a directory"), "{file_err}");

        let non_string = resolve(DelegationKind::Analyze, &json!({ "place": 5 }))
            .await
            .expect_err("a non-string place")
            .to_string();
        assert!(non_string.starts_with("usage:"), "{non_string}");
        assert!(non_string.contains("must be a string"), "{non_string}");
    }

    #[tokio::test]
    async fn filesystem_root_is_refused_for_both_kinds() {
        init_management_test_stores().await;
        for kind in [DelegationKind::Analyze, DelegationKind::Coder] {
            let err = resolve(kind, &json!({ "place": "/" }))
                .await
                .expect_err("the filesystem root is refused")
                .to_string();
            assert!(err.starts_with("usage:"), "{err}");
            assert!(err.contains("filesystem root"), "{err}");
        }
    }

    #[tokio::test]
    async fn coder_refuses_the_data_directory_but_allows_personal_workspaces() {
        init_management_test_stores().await;
        let probe = test_root().join("data-probe-dir");
        std::fs::create_dir_all(&probe).unwrap();
        let err = resolve(
            DelegationKind::Coder,
            &json!({ "place": probe.to_str().unwrap() }),
        )
        .await
        .expect_err("the data directory is refused for the coder")
        .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("data directory"), "{err}");

        // The `userspaces` container itself is NOT a personal workspace — a
        // ticket-less coder there could write sibling folders next to every
        // account's personal workspace.
        let container = test_root().join("userspaces");
        std::fs::create_dir_all(&container).unwrap();
        let err = resolve(
            DelegationKind::Coder,
            &json!({ "place": container.to_str().unwrap() }),
        )
        .await
        .expect_err("the userspaces container itself is refused for the coder")
        .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("data directory"), "{err}");

        let personal = test_root().join("userspaces").join("delegation-probe");
        std::fs::create_dir_all(&personal).unwrap();
        let ws = resolve(
            DelegationKind::Coder,
            &json!({ "place": personal.to_str().unwrap() }),
        )
        .await
        .expect("a personal workspace directory is allowed")
        .expect("a target is present");
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
    }

    #[tokio::test]
    async fn workspace_at_dir_matches_registered_or_derives() {
        init_management_test_stores().await;

        let registered_dir = tempfile::tempdir().unwrap();
        let registered_canonical = canon(registered_dir.path());
        let registered_path = registered_canonical.to_str().unwrap();
        create_test_workspace(registered_path, "ws_delegation_resume").await;
        let ws = workspace_at_dir(registered_path).await.unwrap();
        assert_eq!(ws.name, "ws_delegation_resume");
        assert_eq!(ws.path, registered_path);

        let plain_dir = tempfile::tempdir().unwrap();
        let plain_canonical = canon(plain_dir.path());
        let ws = workspace_at_dir(plain_canonical.to_str().unwrap())
            .await
            .unwrap();
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);

        let missing = workspace_at_dir("/definitely/not/here/delegation")
            .await
            .expect_err("a vanished target directory cannot resume");
        assert!(
            missing.to_string().contains("no longer available"),
            "{missing}"
        );
    }
}

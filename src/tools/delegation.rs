//! The optional `place` argument of the two dispatch tools (`analyze` and `send`): a holder that
//! declares it — the admin Assistant, and a custom role whose grants rebuild the tool — can point
//! a round at a chosen directory instead of its own workspace.
//!
//! Resolution is exact and every refusal is explicit — a place that resolves to nothing usable
//! never falls back to the caller's own workspace. A value is either a managed workspace named
//! exactly (case-sensitively) or an absolute existing directory; an absolute path that IS a
//! managed workspace is that workspace, with its identity and stored context.
//!
//! The resolved directory becomes the round's EXECUTION workspace (the sub-agents' root, context
//! and search key) while its durable delivery key stays the caller's own workspace name
//! (`jobs.workspace_name`), so the result lands in the conversation that asked.
//!
//! A worker round additionally refuses a managed workspace, anything inside one, the filesystem
//! root and the product's own data directory: a ticket-less worker must never write into a tree
//! whose workspace pipeline owns it. Guardrails, not a security boundary — the shell is unchanged.
//!
//! A chain step that names no folder of its own inherits the one the nearest earlier step named,
//! and [`resolve_inherited`] is the same rule with the refusals addressed to that step — the folder
//! was never written there, so it is never refused over how the earlier step spelled it.

use crate::Workspace;
use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};

/// Which delegation tool is resolving a place — the two differ in what they refuse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DelegationKind {
    /// The read-only analyst round: a managed workspace and a plain directory.
    Analyze,
    /// The worker round: refuses managed workspaces (and anything inside one).
    Worker,
}

/// The managed workspace occupying exactly `dir` — "this path IS that workspace".
fn managed_at(managed: &[Workspace], dir: &Path) -> Option<Workspace> {
    managed.iter().find(|ws| dir == ws.as_path()).cloned()
}

/// Who addressed a folder to the resolver: the caller in its own `place`, or a chain step that
/// names none and inherits the earlier step's folder. The checks are the same; only the refusal
/// differs, because an inherited folder was never written on that step.
#[derive(Clone, Copy)]
pub(crate) enum Provenance<'a> {
    /// The caller named it in its own `place`.
    Named,
    /// A chain step inherits it: where the step stands in its chain (`at`) and what it runs as
    /// (`runs`), so the refusal can say which step and how to make it start.
    Inherited { at: (usize, usize), runs: &'a str },
}

impl Provenance<'_> {
    /// The refusal an inherited folder gets, when this is one: the step, the folder and why that
    /// step cannot work there, ending in the way out.
    fn inherited(self, folder: &str, why: &str) -> Option<anyhow::Error> {
        let Self::Inherited {
            at: (index, total),
            runs,
        } = self
        else {
            return None;
        };
        Some(anyhow::anyhow!(
            "usage: step {index} of {total} runs as {runs}, and the folder it would inherit \
             \"{folder}\" {why} — hint: name a folder on that step that {runs} can work in, or do \
             not chain it"
        ))
    }
}

/// One refused place: the folder, the caller's own subject for it, why, and the caller's way out.
/// Written once so the caller's voice and an inherited folder's cannot drift apart.
struct Refusal<'a> {
    /// The folder as the caller wrote it, which the inherited voice names.
    folder: &'a str,
    /// What the caller's own refusal names as its subject — `place "<raw>"` or `place path "<raw>"`.
    subject: String,
    /// The reason, as it reads after the subject or after the folder: `is the filesystem root`.
    why: String,
    /// The way out for a caller that named the place itself; an inherited folder's way out is the
    /// chain's own ([`Provenance::inherited`]).
    hint: &'static str,
}

impl Refusal<'_> {
    /// The refusal in the voice of whoever addressed the folder.
    fn say(self, provenance: Provenance<'_>) -> anyhow::Error {
        if let Some(refusal) = provenance.inherited(self.folder, &self.why) {
            return refusal;
        }
        anyhow::anyhow!("usage: {} {} — hint: {}", self.subject, self.why, self.hint)
    }
}

/// Resolve a `place` that must address a managed workspace — by its exact name, or the directory
/// that is it.
pub(crate) async fn resolve_managed_workspace(raw: &str) -> Result<Workspace> {
    let raw = raw.trim();
    if !raw.is_empty() {
        let expanded = crate::util::expand_tilde(raw);
        if expanded.is_absolute() {
            let dir = canonical_dir(raw, &expanded, Provenance::Named)?;
            if let Some(ws) = managed_at(&stored_workspaces().await?, &dir) {
                return Ok(ws);
            }
        } else if let Some(ws) = crate::workspace::get_by_name(raw).await? {
            return Ok(ws);
        }
    }
    Err(refuse_not_managed(raw))
}

fn refuse_not_managed(raw: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "usage: place \"{raw}\" is not a managed workspace — hint: pass a managed workspace's \
         name or its own folder; personal workspaces have no Manager"
    )
}

/// Resolve the optional `place` argument into the workspace the round runs in.
pub(crate) async fn resolve(kind: DelegationKind, args: &Value) -> Result<Option<Workspace>> {
    let raw = super::get_opt_str_strict(args, "place")?.map(str::trim);
    let Some(raw) = raw.filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    resolve_place(kind, raw, Provenance::Named).await.map(Some)
}

/// Resolve the folder a chain step inherits, by the rule a worker's own `place` gets, with every
/// refusal naming the step instead of a `place` it never wrote. A folder handed on by name becomes
/// the directory it names first, so one folder spelled either way gets one answer.
pub(crate) async fn resolve_inherited(
    address: &str,
    provenance: Provenance<'_>,
) -> Result<Workspace> {
    let Some(workspace) = crate::users::resolve_workspace(address).await? else {
        return Err(refuse_missing(address).say(provenance));
    };
    resolve_place(DelegationKind::Worker, &workspace.path, provenance).await
}

/// Resolve one `place` value — a caller's own, or a folder a chain step inherited — into the
/// workspace the round runs in.
async fn resolve_place(
    kind: DelegationKind,
    raw: &str,
    provenance: Provenance<'_>,
) -> Result<Workspace> {
    // A value that is not an absolute path can only be a managed workspace name.
    let expanded = crate::util::expand_tilde(raw);
    if !expanded.is_absolute() {
        let Some(ws) = crate::workspace::get_by_name(raw).await? else {
            return Err(refuse_unresolvable(kind, raw));
        };
        if kind == DelegationKind::Worker {
            return Err(refuse_managed_workspace(raw, &ws.name).say(provenance));
        }
        return Ok(ws);
    }

    let dir = canonical_dir(raw, &expanded, provenance)?;
    let managed = stored_workspaces().await?;
    let exact = managed_at(&managed, &dir);
    if kind == DelegationKind::Worker {
        if let Some(ws) = exact {
            return Err(refuse_managed_workspace(raw, &ws.name).say(provenance));
        }
        if let Some(ws) = managed
            .iter()
            .find(|ws| crate::util::is_within(&dir, ws.as_path()))
        {
            return Err(refuse_inside(raw, &ws.name).say(provenance));
        }
        if let Some(root) = data_dir(&dir) {
            return Err(refuse_data_dir(raw, &root).say(provenance));
        }
        return Ok(dir_workspace(&dir));
    }
    // The analyst may be pointed at a managed workspace: it then carries that workspace's context.
    match exact {
        Some(ws) => Ok(ws),
        None => Ok(dir_workspace(&dir)),
    }
}

/// The execution workspace for a directory recorded on a round that already resolved one.
pub(crate) async fn workspace_at_dir(dir: &str) -> Result<Workspace> {
    let path = Path::new(dir);
    let (_, is_dir) = path_state(path);
    if !is_dir {
        bail!("the place directory {dir} is no longer available");
    }
    let canonical = crate::stored_workspace_path(path);
    match managed_at(&stored_workspaces().await?, &canonical) {
        Some(ws) => Ok(ws),
        None => Ok(dir_workspace(&canonical)),
    }
}

/// Every stored `workspaces` row — nothing else in the product looks at these.
async fn stored_workspaces() -> Result<Vec<Workspace>> {
    crate::workspace::store()
        .list()
        .await
        .context("list managed workspaces")
}

/// Canonicalize an existing directory place, refusing the filesystem root.
fn canonical_dir(raw: &str, expanded: &Path, provenance: Provenance<'_>) -> Result<PathBuf> {
    let (exists, is_dir) = path_state(expanded);
    if !exists {
        return Err(refuse_missing(raw).say(provenance));
    }
    if !is_dir {
        return Err(refuse_not_a_directory(raw).say(provenance));
    }
    let canonical = crate::stored_workspace_path(expanded);
    if canonical.parent().is_none() {
        return Err(refuse_root(raw).say(provenance));
    }
    Ok(canonical)
}

/// A place path's metadata, off the async worker pool (`with_block_in_place`).
fn path_state(path: &Path) -> (bool, bool) {
    crate::util::with_block_in_place(|| (path.exists(), path.is_dir()))
}

fn refuse_missing(raw: &str) -> Refusal<'_> {
    Refusal {
        folder: raw,
        subject: format!("place path \"{raw}\""),
        why: "does not exist".to_string(),
        hint: "pass an absolute path to an existing directory",
    }
}

fn refuse_not_a_directory(raw: &str) -> Refusal<'_> {
    Refusal {
        folder: raw,
        subject: format!("place path \"{raw}\""),
        why: "is a file, not a directory".to_string(),
        hint: "pass an absolute path to a directory",
    }
}

fn refuse_root(raw: &str) -> Refusal<'_> {
    Refusal {
        folder: raw,
        subject: format!("place \"{raw}\""),
        why: "is the filesystem root".to_string(),
        hint: "pass a project directory instead",
    }
}

/// The product's own data directory when `dir` lies inside it OUTSIDE the personal workspaces —
/// a personal workspace directory is a legitimate place, the `userspaces` container itself is not.
///
/// The storage root is only the containment base: it is derived as the personal root's parent, so
/// the rule lives in one place ([`crate::users::userspaces_root`]). It is spelled canonically
/// like `dir`, independently of whether the personal root itself exists yet.
fn data_dir(dir: &Path) -> Option<PathBuf> {
    let personal = crate::stored_workspace_path(&crate::users::userspaces_root());
    let root = crate::stored_workspace_path(personal.parent()?);
    if dir != personal && crate::util::is_within(dir, &personal) {
        return None;
    }
    crate::util::is_within(dir, &root).then_some(root)
}

/// Drop the search engine a delegated round created for its synthetic `dir-…` workspace: its
/// lifetime is bounded by the run that served it, exactly like a research run root — a one-off
/// directory must never keep a picker and a file watcher alive forever. A round in a managed
/// workspace's folder uses that workspace's engine and is left alone.
///
/// Accepted trade-off: two rounds placed in the same directory share one `dir-…` name, so
/// releasing while another is still running only costs it a picker re-scan — never a wrong
/// result. A round cut by drain keeps its engine for the resume that reuses it; a cancel of that
/// round releases it ([`release_placed_engine`]).
pub(crate) fn release_exec_engine(exec: &Workspace) {
    if !exec.name.starts_with(DIR_WORKSPACE_PREFIX)
        || !crate::workspace_index::registry_initialized()
    {
        return;
    }
    crate::workspace_index::remove_engine(&exec.name);
}

/// Release the search engine a round placed in `dir` created, named by the directory alone: the
/// synthetic name is a pure function of the canonical path, so this reaches the engine even when
/// the directory is gone — the state a cancel leaves. A managed workspace's engine is not
/// touched: the derivation only ever names a `dir-…` workspace.
pub(crate) fn release_placed_engine(dir: &str) {
    let canonical = crate::stored_workspace_path(Path::new(dir));
    release_exec_engine(&dir_workspace(&canonical));
}

/// The name prefix of a directory-scoped synthetic workspace — impossible in a managed
/// workspace name.
const DIR_WORKSPACE_PREFIX: &str = "dir-";

/// A directory-scoped execution workspace for a delegation round placed in a folder:
/// `dir-<basename>-<hash of the canonical path>`. The embedded `-` is impossible in a managed
/// workspace name (`validate_name` allows only letters and underscores), so a name-keyed surface
/// — the per-workspace search index above all — can never hand this directory a managed
/// workspace's state, and the path hash keeps two directories apart. Deterministic, so a resumed
/// round re-derives the same agent/session/search identity.
///
/// The short key is only ever an in-memory execution identity, bounded by the round that created
/// it — never a durable launch address.
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
            "usage: place \"{raw}\" is not a managed workspace and is not an absolute path — \
             hint: pass the exact name of a managed workspace, or an absolute path to an \
             existing directory"
        ),
        DelegationKind::Worker => anyhow::anyhow!(
            "usage: place \"{raw}\" is not an absolute path — hint: this tool only runs in a \
             directory outside the managed workspaces (a managed workspace name never works \
             here); pass an absolute path to an existing directory"
        ),
    }
}

/// The way out shared by the two managed-workspace refusals.
const THROUGH_THE_BOARD: &str = "work in a managed workspace goes through that workspace's board \
     and its Manager, never through this tool; pass an absolute path to a directory outside the \
     managed workspaces";

fn refuse_managed_workspace<'a>(raw: &'a str, workspace: &str) -> Refusal<'a> {
    Refusal {
        folder: raw,
        subject: format!("place \"{raw}\""),
        why: format!("is the managed workspace \"{workspace}\""),
        hint: THROUGH_THE_BOARD,
    }
}

fn refuse_inside<'a>(raw: &'a str, workspace: &str) -> Refusal<'a> {
    Refusal {
        folder: raw,
        subject: format!("place path \"{raw}\""),
        why: format!("is inside the managed workspace \"{workspace}\""),
        hint: THROUGH_THE_BOARD,
    }
}

fn refuse_data_dir<'a>(raw: &'a str, root: &Path) -> Refusal<'a> {
    Refusal {
        folder: raw,
        subject: format!("place path \"{raw}\""),
        why: format!("is inside MahBot's own data directory ({})", root.display()),
        hint: "pass a project directory instead; a personal workspace directory is allowed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::{create_test_workspace, init_management_test_stores, test_root};
    use serde_json::json;

    /// Canonicalize a real path the way [`resolve`] and `stored_workspace_path` do.
    fn canon(path: &Path) -> PathBuf {
        crate::util::strip_verbatim_prefix(&std::fs::canonicalize(path).unwrap())
    }

    /// Add a workspace the way production rows are stored: the canonical directory, named.
    async fn add_managed_workspace(dir: &Path, name: &str) -> String {
        let stored = format!("{}{}", dir.display(), std::path::MAIN_SEPARATOR);
        create_test_workspace(&stored, name).await;
        stored
    }

    /// What a `place` value means: no place at all, or a refusal.
    enum PlaceExpect {
        NoPlace,
        /// Substrings the refusal must carry.
        Refused(&'static [&'static str]),
    }

    /// One `place` value, for both delegation kinds.
    struct PlaceValueCase {
        name: &'static str,
        args: serde_json::Value,
        expected: PlaceExpect,
    }

    #[tokio::test]
    async fn place_value_cases() {
        init_management_test_stores().await;
        for case in [
            // Nothing names a place: the round runs in the caller's own one.
            PlaceValueCase {
                name: "an absent place",
                args: json!({}),
                expected: PlaceExpect::NoPlace,
            },
            PlaceValueCase {
                name: "a null place",
                args: json!({"place": null}),
                expected: PlaceExpect::NoPlace,
            },
            PlaceValueCase {
                name: "an empty place",
                args: json!({"place": ""}),
                expected: PlaceExpect::NoPlace,
            },
            PlaceValueCase {
                name: "a blank place",
                args: json!({"place": "   "}),
                expected: PlaceExpect::NoPlace,
            },
            // The root is refused by name, never resolved to a place no round can use.
            PlaceValueCase {
                name: "the filesystem root",
                args: json!({"place": "/"}),
                expected: PlaceExpect::Refused(&["usage: ", "filesystem root"]),
            },
        ] {
            for kind in [DelegationKind::Analyze, DelegationKind::Worker] {
                let refusal = resolve(kind, &case.args)
                    .await
                    .err()
                    .map(|err| err.to_string());
                match (&case.expected, refusal) {
                    (PlaceExpect::NoPlace, None) => {}
                    (PlaceExpect::NoPlace, Some(err)) => {
                        panic!("[{}] {:?} must be no place: {err}", case.name, kind);
                    }
                    (PlaceExpect::Refused(needles), Some(err)) => {
                        for needle in *needles {
                            assert!(
                                err.contains(needle),
                                "[{}] {:?} must say {needle:?}: {err}",
                                case.name,
                                kind
                            );
                        }
                    }
                    (PlaceExpect::Refused(_), None) => {
                        panic!("[{}] {:?} must be refused", case.name, kind);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn managed_workspace_resolves_by_name_and_by_path() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        let stored = add_managed_workspace(&canonical, "ws_delegation_alpha").await;
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
        assert_eq!(by_name.path, stored, "the managed spelling is kept");

        // The same directory by its separator-less absolute path is that
        // workspace — the path match tolerates the managed row's trailing
        // separator — identity and stored context included.
        let by_path = resolve(DelegationKind::Analyze, &json!({ "place": plain }))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_path.name, "ws_delegation_alpha");
        assert_eq!(by_path.path, stored);

        // A worker round refuses a managed workspace, by name and by path.
        for args in [
            json!({ "place": "ws_delegation_alpha" }),
            json!({ "place": plain }),
        ] {
            let err = resolve(DelegationKind::Worker, &args)
                .await
                .expect_err("the worker refuses a managed workspace")
                .to_string();
            assert!(err.starts_with("usage:"), "{err}");
            assert!(err.contains("ws_delegation_alpha"), "{err}");
            assert!(err.contains("board") && err.contains("Manager"), "{err}");
        }
    }

    #[tokio::test]
    async fn directory_inside_a_managed_workspace_is_synthetic_for_analysts_only() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        add_managed_workspace(&canonical, "ws_delegation_nested").await;
        let nested = canonical.join("inner");
        std::fs::create_dir(&nested).unwrap();
        let target = nested.to_str().unwrap();

        let ws = resolve(DelegationKind::Analyze, &json!({ "place": target }))
            .await
            .unwrap()
            .unwrap();
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
        assert_ne!(ws.name, "ws_delegation_nested");

        let err = resolve(DelegationKind::Worker, &json!({ "place": target }))
            .await
            .expect_err("the worker refuses anything inside a managed workspace")
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

        let not_managed = resolve(DelegationKind::Analyze, &json!({ "place": "no_such_ws" }))
            .await
            .expect_err("a name that is not a managed workspace")
            .to_string();
        assert!(not_managed.starts_with("usage:"), "{not_managed}");
        assert!(
            not_managed.contains("not a managed workspace"),
            "{not_managed}"
        );

        // The worker's variant must not send the caller down the managed-name
        // form it always refuses.
        let worker_not_managed = resolve(DelegationKind::Worker, &json!({ "place": "no_such_ws" }))
            .await
            .expect_err("a workspace name that is not managed, for the worker")
            .to_string();
        assert!(
            worker_not_managed.starts_with("usage:"),
            "{worker_not_managed}"
        );
        assert!(
            worker_not_managed.contains("never works here"),
            "{worker_not_managed}"
        );
        assert!(
            !worker_not_managed.contains("not a managed workspace"),
            "the worker's hint must not offer the managed-name form: {worker_not_managed}"
        );

        let relative = resolve(DelegationKind::Analyze, &json!({ "place": "sub/dir" }))
            .await
            .expect_err("relative value")
            .to_string();
        assert!(relative.starts_with("usage:"), "{relative}");
        assert!(relative.contains("not an absolute path"), "{relative}");

        let missing_path = canonical.join("does-not-exist");
        let missing = resolve(
            DelegationKind::Worker,
            &json!({ "place": missing_path.to_str().unwrap() }),
        )
        .await
        .expect_err("missing absolute path")
        .to_string();
        assert_eq!(
            missing,
            format!(
                "usage: place path \"{}\" does not exist — hint: pass an absolute path to an \
                 existing directory",
                missing_path.display()
            ),
            "the caller's own refusal names the folder once"
        );

        let file = canonical.join("a-file.txt");
        std::fs::write(&file, "x").unwrap();
        let file_err = resolve(
            DelegationKind::Worker,
            &json!({ "place": file.to_str().unwrap() }),
        )
        .await
        .expect_err("a file place")
        .to_string();
        assert_eq!(
            file_err,
            format!(
                "usage: place path \"{}\" is a file, not a directory — hint: pass an absolute path \
                 to a directory",
                file.display()
            ),
            "and neither does this one"
        );

        let non_string = resolve(DelegationKind::Analyze, &json!({ "place": 5 }))
            .await
            .expect_err("a non-string place")
            .to_string();
        assert!(non_string.starts_with("usage:"), "{non_string}");
        assert!(non_string.contains("must be a string"), "{non_string}");
    }

    #[tokio::test]
    async fn worker_refuses_the_data_directory_but_allows_personal_workspaces() {
        init_management_test_stores().await;
        let probe = test_root().join("data-probe-dir");
        std::fs::create_dir_all(&probe).unwrap();
        let err = resolve(
            DelegationKind::Worker,
            &json!({ "place": probe.to_str().unwrap() }),
        )
        .await
        .expect_err("the data directory is refused for the worker")
        .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("data directory"), "{err}");

        // The `userspaces` container itself is NOT a personal workspace — a
        // ticket-less worker there could write sibling folders next to every
        // account's personal workspace.
        let container = test_root().join("userspaces");
        std::fs::create_dir_all(&container).unwrap();
        let err = resolve(
            DelegationKind::Worker,
            &json!({ "place": container.to_str().unwrap() }),
        )
        .await
        .expect_err("the userspaces container itself is refused for the worker")
        .to_string();
        assert!(err.starts_with("usage:"), "{err}");
        assert!(err.contains("data directory"), "{err}");

        let personal = test_root().join("userspaces").join("delegation-probe");
        std::fs::create_dir_all(&personal).unwrap();
        let ws = resolve(
            DelegationKind::Worker,
            &json!({ "place": personal.to_str().unwrap() }),
        )
        .await
        .expect("a personal workspace directory is allowed")
        .expect("a target is present");
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
    }

    /// The provenance an inherited folder's refusal carries in these tests.
    fn inherited() -> Provenance<'static> {
        Provenance::Inherited {
            at: (2, 3),
            runs: "a worker",
        }
    }

    /// A folder a chain step inherits is judged by what it is, never by how the step that named it
    /// spelled it: the same folder written as a name and as a directory gets one answer. Every
    /// refusal names the step and the folder, and ends in the way out.
    #[tokio::test]
    async fn an_inherited_folder_is_judged_by_what_it_is_not_how_it_was_spelled() {
        init_management_test_stores().await;
        let dir = tempfile::tempdir().unwrap();
        let canonical = canon(dir.path());
        add_managed_workspace(&canonical, "ws_inherited_alpha").await;
        let plain = canonical.to_str().unwrap();

        // A worker cannot be handed a managed workspace, whichever way the earlier step wrote it.
        for address in ["ws_inherited_alpha", plain] {
            let err = resolve_inherited(address, inherited())
                .await
                .expect_err("a worker refuses a managed workspace")
                .to_string();
            assert!(err.starts_with("usage:"), "{err}");
            assert!(err.contains("step 2 of 3"), "the step is named: {err}");
            assert!(err.contains("a worker"), "what it runs as is named: {err}");
            assert!(
                err.contains("ws_inherited_alpha"),
                "the folder it would inherit is named: {err}"
            );
            assert!(err.contains("hint:"), "{err}");
            assert!(err.contains("do not chain it"), "{err}");
        }
        assert_eq!(
            resolve_inherited("ws_inherited_alpha", inherited())
                .await
                .expect_err("refused")
                .to_string(),
            resolve_inherited(plain, inherited())
                .await
                .expect_err("refused")
                .to_string(),
            "one folder, one answer, however it was spelled"
        );

        // A plain directory is the folder a worker takes.
        let scratch = tempfile::tempdir().unwrap();
        let scratch_canonical = canon(scratch.path());
        let ws = resolve_inherited(scratch_canonical.to_str().unwrap(), inherited())
            .await
            .expect("a plain directory is a worker's folder");
        assert!(ws.name.starts_with("dir-"), "got {}", ws.name);
        assert_eq!(ws.path, scratch_canonical.to_string_lossy());

        // A `personal:` key becomes the directory it names: a worker may work there.
        let personal = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let ws = resolve_inherited(&personal, inherited())
            .await
            .expect("a personal workspace is a folder a worker may work in");
        assert_eq!(
            ws.path,
            crate::stored_workspace_path(&crate::users::personal_workspace_path(
                crate::users::ADMIN_USER_NAME
            ))
            .to_string_lossy()
        );
    }

    /// A folder that is gone, the root, the product's own data directory or a tree inside a managed
    /// workspace is refused for a step that would inherit it, exactly as for a caller that named it.
    #[tokio::test]
    async fn an_inherited_folder_a_worker_cannot_work_in_is_refused() {
        init_management_test_stores().await;
        let probe = test_root().join("inherited-probe");
        std::fs::create_dir_all(&probe).unwrap();
        let project = tempfile::tempdir().unwrap();
        let canonical = canon(project.path());
        add_managed_workspace(&canonical, "ws_inherited_nested").await;
        let nested = canonical.join("inner");
        std::fs::create_dir(&nested).unwrap();
        for (address, needle) in [
            (
                tempfile::tempdir()
                    .unwrap()
                    .path()
                    .join("gone")
                    .to_string_lossy()
                    .into_owned(),
                "does not exist",
            ),
            ("/".to_string(), "filesystem root"),
            (probe.to_string_lossy().into_owned(), "data directory"),
            (
                nested.to_string_lossy().into_owned(),
                "inside the managed workspace \"ws_inherited_nested\"",
            ),
        ] {
            let err = resolve_inherited(&address, inherited())
                .await
                .expect_err("a worker cannot work there")
                .to_string();
            assert!(err.starts_with("usage:"), "{err}");
            assert!(err.contains(needle), "{address}: {err}");
            assert!(err.contains("step 2 of 3"), "{err}");
            assert!(err.contains("hint:"), "{err}");
        }
    }
}

//! Tool grants with kinds: the grammar of a custom role's `custom_roles.tools`
//! column, and the kind vocabulary of the dispatch tools.
//!
//! One entry is `"tool"` (the tool with its narrowest kind) or
//! `"tool:kind1,kind2"` (the named kinds). [`kinds_of`] is the single source of
//! each dispatch tool's kind vocabulary — the same const arrays the tool enums
//! expose ([`AnalyzeDepth::STRS`], [`Addressee::STRS`]) — so a kind can never be
//! advertised by a tool it does not exist on. The catalog a role may be given is
//! decided elsewhere ([`crate::Role::custom_role_tool_catalog`]); this module
//! owns the grammar and the vocabulary only.

use anyhow::Result;

use crate::tools::analyze::AnalyzeDepth;
use crate::tools::send::Addressee;

/// One granted tool with the kinds it holds. A tool with no kind vocabulary (a
/// kindless tool) has an empty `kinds`; so does a bare entry for a tool that
/// has one — such an entry implies the vocabulary's narrowest kind
/// ([`held_kinds`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolGrant {
    pub tool: String,
    pub kinds: Vec<String>,
}

/// The kind vocabulary of a dispatch tool, in interface order. `None` for a
/// tool with no kinds.
#[must_use]
pub(crate) fn kinds_of(tool: &str) -> Option<&'static [&'static str]> {
    match tool {
        "analyze" => Some(AnalyzeDepth::STRS),
        "send" => Some(Addressee::STRS),
        _ => None,
    }
}

/// The tool name of one raw entry — everything before the first `:`. The
/// grammar's only spelling of that rule, so every reader of a stored entry
/// agrees on what names the tool.
#[must_use]
pub(crate) fn tool_name(entry: &str) -> &str {
    entry.split(':').next().unwrap_or(entry)
}

/// Put a tool's kinds into interface order (the vocabulary's own order).
fn canonical_kinds(tool: &str, kinds: &mut [String]) {
    if let Some(vocab) = kinds_of(tool) {
        kinds.sort_by_key(|kind| vocab.iter().position(|v| v == kind).unwrap_or(usize::MAX));
    }
}

/// Parse the stored entries into grants: one entry per tool, in the stored
/// order, with each tool's kinds in the interface order. A tool named twice is
/// refused, like an unknown kind.
pub(crate) fn parse(entries: &[String]) -> Result<Vec<ToolGrant>> {
    let mut out: Vec<ToolGrant> = Vec::with_capacity(entries.len());
    for entry in entries {
        let grant = parse_entry(entry)?;
        anyhow::ensure!(
            !out.iter().any(|g| g.tool == grant.tool),
            "tool '{}' is named twice",
            grant.tool
        );
        out.push(grant);
    }
    Ok(out)
}

/// Parse the stored entries, dropping only the entries the grammar refuses.
/// The read side (building a role's tool set) uses this so a single bad row
/// cannot cost the role its whole tool set — the validated write side never
/// stores one, so this only ever narrows to what the record can serve.
pub(crate) fn parse_lossy(entries: &[String]) -> Vec<ToolGrant> {
    let mut out: Vec<ToolGrant> = Vec::with_capacity(entries.len());
    for entry in entries {
        let Ok(grant) = parse_entry(entry) else {
            continue;
        };
        // One entry per tool, as `parse` holds: a repeat is dropped like any
        // other entry the grammar refuses.
        if !out.iter().any(|g| g.tool == grant.tool) {
            out.push(grant);
        }
    }
    out
}

/// Parse one entry into a grant: split on its first `:`, split the kinds on
/// `,`, and refuse a kind on a tool that has no kind vocabulary (or an unknown
/// kind, or a kind named twice on one tool).
fn parse_entry(entry: &str) -> Result<ToolGrant> {
    let tool = tool_name(entry);
    let mut kinds: Vec<String> = match entry.split_once(':') {
        None => Vec::new(),
        Some((_, raw)) => {
            let vocab = kinds_of(tool).ok_or_else(|| {
                anyhow::anyhow!("tool '{tool}' takes no kinds, but '{entry}' names some")
            })?;
            let mut kinds: Vec<String> = Vec::new();
            for kind in raw.split(',') {
                let kind = kind.trim();
                anyhow::ensure!(
                    vocab.contains(&kind),
                    "unknown kind '{kind}' for tool '{tool}'"
                );
                anyhow::ensure!(
                    !kinds.iter().any(|k| k == kind),
                    "kind '{kind}' is named twice for tool '{tool}'"
                );
                kinds.push(kind.to_string());
            }
            kinds
        }
    };
    canonical_kinds(tool, &mut kinds);
    Ok(ToolGrant {
        tool: tool.to_string(),
        kinds,
    })
}

/// The kinds a grant holds, with the narrowest kind filled in when the entry
/// named none (empty for a tool with no kind vocabulary).
#[must_use]
pub(crate) fn held_kinds(grant: &ToolGrant) -> Vec<String> {
    if !grant.kinds.is_empty() {
        return grant.kinds.clone();
    }
    match kinds_of(&grant.tool).and_then(|vocab| vocab.first()) {
        Some(narrowest) => vec![(*narrowest).to_string()],
        None => Vec::new(),
    }
}

/// The stored entries with every bare dispatch-tool entry spelled out with the
/// kind it implies — what the admin surfaces list.
#[must_use]
pub(crate) fn describe_all(entries: &[String]) -> Vec<String> {
    parse_lossy(entries)
        .iter()
        .map(|grant| {
            encode(&ToolGrant {
                tool: grant.tool.clone(),
                kinds: held_kinds(grant),
            })
        })
        .collect()
}

/// Flip one kind of `tool` in the stored entries: add it when absent, remove it
/// when present, drop the entry when its last kind goes, and keep the kinds in
/// interface order and the entry's own position in the list. A tool with no
/// kind vocabulary is left untouched.
#[must_use]
pub(crate) fn toggle(entries: &[String], tool: &str, kind: &str) -> Vec<String> {
    // Nothing to flip on a tool that has no kinds, nor on a kind the tool's
    // vocabulary does not name.
    let Some(vocab) = kinds_of(tool) else {
        return entries.to_vec();
    };
    if !vocab.contains(&kind) {
        return entries.to_vec();
    }
    // An entry the grammar refuses is carried through untouched — rewriting it
    // is the write side's job, and it refuses such a record outright — so a
    // record the picker renders checkmarks for is one this can still edit. The
    // read is the same lenient one the picker and the launch path use.
    let mut out: Vec<String> = Vec::with_capacity(entries.len() + 1);
    let mut found = false;
    for entry in entries {
        let Ok(mut grant) = parse_entry(entry) else {
            out.push(entry.clone());
            continue;
        };
        if grant.tool != tool {
            out.push(entry.clone());
            continue;
        }
        found = true;
        // A bare entry holds the narrowest kind, so the flip starts from what
        // the tool actually offers.
        let mut kinds = held_kinds(&grant);
        match kinds.iter().position(|k| k == kind) {
            Some(at) => {
                kinds.remove(at);
            }
            None => kinds.push(kind.to_string()),
        }
        canonical_kinds(tool, &mut kinds);
        // A grant is a held tool: an entry whose last kind just went is dropped.
        if !kinds.is_empty() {
            grant.kinds = kinds;
            out.push(encode(&grant));
        }
    }
    if !found {
        out.push(encode(&ToolGrant {
            tool: tool.to_string(),
            kinds: vec![kind.to_string()],
        }));
    }
    out
}

/// Encode one grant back into its stored entry.
#[must_use]
fn encode(grant: &ToolGrant) -> String {
    if grant.kinds.is_empty() {
        grant.tool.clone()
    } else {
        format!("{}:{}", grant.tool, grant.kinds.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    /// A slice of grants as the stored entries they encode to.
    fn encoded(grants: &[ToolGrant]) -> Vec<String> {
        grants.iter().map(encode).collect()
    }

    #[test]
    fn parse_reads_kinds_and_canonicalizes_order() {
        let parsed = parse(&entries(&["send:manager,worker", "read"])).unwrap();
        assert_eq!(
            parsed,
            vec![
                ToolGrant {
                    tool: "send".to_string(),
                    kinds: vec!["worker".to_string(), "manager".to_string()],
                },
                ToolGrant {
                    tool: "read".to_string(),
                    kinds: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn parse_names_each_tool_once() {
        assert!(
            parse(&entries(&["analyze:quick", "analyze:deep"])).is_err(),
            "a second entry for the same tool is refused"
        );
        assert!(
            parse(&entries(&["read", "read"])).is_err(),
            "a repeated kindless tool is refused too"
        );
    }

    /// The lenient read: only the entries the grammar refuses are dropped, so
    /// one bad row cannot cost the rest of the record.
    #[test]
    fn parse_lossy_drops_only_the_bad_entries() {
        let lost = parse_lossy(&entries(&[
            "read",
            "analyze:bogus",
            "send:worker",
            "read:quick",
        ]));
        assert_eq!(
            encoded(&lost),
            entries(&["read", "send:worker"]),
            "the parseable entries survive in their stored order"
        );
    }

    /// A bare dispatch-tool entry holds the vocabulary's narrowest kind, and
    /// the admin-facing spelling fills it in.
    #[test]
    fn held_kinds_and_describe_all_fill_in_the_narrowest_kind() {
        let parsed = parse(&entries(&["analyze", "send:worker,role", "read"])).unwrap();
        assert_eq!(held_kinds(&parsed[0]), vec!["quick".to_string()]);
        assert_eq!(
            held_kinds(&parsed[1]),
            vec!["worker".to_string(), "role".to_string()]
        );
        assert!(
            held_kinds(&parsed[2]).is_empty(),
            "a tool with no kind vocabulary holds none"
        );
        assert_eq!(
            describe_all(&entries(&["analyze", "send", "read", "send:manager"])),
            entries(&["analyze:quick", "send:worker", "read"])
        );
    }

    /// Flipping a kind adds it, removes it, and takes the entry with the last
    /// one — in interface order, and in place.
    #[test]
    fn toggle_adds_removes_and_drops_an_entry() {
        assert_eq!(
            toggle(&entries(&["read"]), "send", "manager"),
            entries(&["read", "send:manager"])
        );
        assert_eq!(
            toggle(&entries(&["send:manager", "read"]), "send", "worker"),
            entries(&["send:worker,manager", "read"]),
            "the kind lands in interface order and the entry keeps its position"
        );
        assert_eq!(
            toggle(&entries(&["send:manager"]), "send", "manager"),
            Vec::<String>::new(),
            "the entry goes with its last kind"
        );
        assert_eq!(
            toggle(&entries(&["analyze"]), "analyze", "quick"),
            Vec::<String>::new(),
            "the implied narrowest kind is what a bare entry holds"
        );
        assert_eq!(
            toggle(&entries(&["analyze"]), "analyze", "deep"),
            entries(&["analyze:quick,deep"]),
            "the implied kind stays alongside the added one"
        );
        assert_eq!(
            toggle(&entries(&["read"]), "read", "quick"),
            entries(&["read"]),
            "a tool with no kind vocabulary is left untouched"
        );
    }

    #[test]
    fn parse_refuses_unknown_kind_and_kinds_on_a_kindless_tool() {
        assert!(parse(&entries(&["analyze:bogus"])).is_err());
        assert!(parse(&entries(&["read:quick"])).is_err());
    }

    /// The GRANT vocabulary is [`Addressee::STRS`] — the tool's own set — and never
    /// the wider one [`Addressee::parse`] reads: an entry reaches the tool's own
    /// parser only after this module has validated it, so a role can never be
    /// granted the owner's own assistant (an address of the command line alone).
    #[test]
    fn a_grant_can_never_name_the_command_lines_own_assistant() {
        assert!(parse(&entries(&["send:assistant"])).is_err());
        assert!(
            parse(&entries(&["send:worker,assistant"])).is_err(),
            "a known kind beside an unknown one does not make the entry valid"
        );
    }

    #[test]
    fn encode_roundtrips() {
        let parsed = parse(&entries(&["send:worker,role", "read", "analyze"])).unwrap();
        assert_eq!(
            encoded(&parsed),
            entries(&["send:worker,role", "read", "analyze"])
        );
    }
}

//! Tool grants: the grammar of a custom role's `custom_roles.tools` column, and the kind vocabulary
//! of the dispatch tools.
//!
//! One entry is `"tool"` (the tool with its narrowest kind), `"tool:kind1,kind2"` (the named kinds),
//! or `"custom:<script>"` — one script from the admin's `shared` folder, which is also the only way
//! a role holds the `custom` dispatcher. [`kinds_of`] is the single source of each dispatch tool's
//! kind vocabulary — the same const arrays the tool enums expose ([`AnalyzeDepth::STRS`],
//! [`Addressee::STRS`]) — so a kind can never be advertised by a tool it does not exist on. The
//! catalog a role may be given is decided elsewhere ([`crate::Role::custom_role_tool_catalog`]);
//! this module owns the grammar and the vocabulary only.

use anyhow::Result;

use crate::tools::analyze::AnalyzeDepth;
use crate::tools::custom::is_tool_name;
use crate::tools::send::Addressee;

/// The prefix that makes an entry name one custom script rather than a tool.
const SCRIPT_PREFIX: &str = "custom:";

/// One stored entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Grant {
    /// `"tool"` or `"tool:kind1,kind2"`.
    Tool { tool: String, kinds: Vec<String> },
    /// `"custom:<script>"`.
    Script { name: String },
}

impl Grant {
    /// The tool this entry names; `None` for a script entry.
    #[must_use]
    pub(crate) fn tool(&self) -> Option<&str> {
        match self {
            Self::Tool { tool, .. } => Some(tool),
            Self::Script { .. } => None,
        }
    }

    /// The script this entry names; `None` for a tool entry.
    #[must_use]
    pub(crate) fn script(&self) -> Option<&str> {
        match self {
            Self::Script { name } => Some(name),
            Self::Tool { .. } => None,
        }
    }

    /// Whether the two entries name the same thing: a tool by its name (its kinds are its interface,
    /// not its identity) or a script by its name. A tool and a script never collide, whatever they
    /// are called.
    #[must_use]
    pub(crate) fn names_the_same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tool { tool: a, .. }, Self::Tool { tool: b, .. })
            | (Self::Script { name: a }, Self::Script { name: b }) => a == b,
            _ => false,
        }
    }
}

/// Whether a raw entry names a script rather than a tool.
#[must_use]
pub(crate) fn is_script_entry(entry: &str) -> bool {
    entry.starts_with(SCRIPT_PREFIX)
}

/// The script a raw entry names, or `None` when it names a tool.
#[must_use]
pub(crate) fn script_of(entry: &str) -> Option<&str> {
    entry.strip_prefix(SCRIPT_PREFIX)
}

/// The entry that names one script.
#[must_use]
pub(crate) fn script_entry(name: &str) -> String {
    format!("{SCRIPT_PREFIX}{name}")
}

/// The scripts a stored list names, in stored order, dropping entries the grammar refuses — the
/// same lenient read the tool set is built from.
#[must_use]
pub(crate) fn scripts_of(entries: &[String]) -> Vec<String> {
    parse_lossy(entries)
        .iter()
        .filter_map(|grant| grant.script().map(str::to_string))
        .collect()
}

#[must_use]
pub(crate) fn kinds_of(tool: &str) -> Option<&'static [&'static str]> {
    match tool {
        "analyze" => Some(AnalyzeDepth::STRS),
        "send" => Some(Addressee::STRS),
        _ => None,
    }
}

/// The tool name of a raw tool entry — everything before the first `:`. A script entry
/// ([`is_script_entry`]) names a script instead, whose name is [`script_of`]; every reader of a
/// stored entry checks that first.
#[must_use]
pub(crate) fn tool_name(entry: &str) -> &str {
    entry.split(':').next().unwrap_or(entry)
}

fn canonical_kinds(tool: &str, kinds: &mut [String]) {
    if let Some(vocab) = kinds_of(tool) {
        kinds.sort_by_key(|kind| vocab.iter().position(|v| v == kind).unwrap_or(usize::MAX));
    }
}

/// Parse the stored entries into grants: one entry per tool, in the stored order, with each tool's
/// kinds in the interface order. A tool named twice, or a script named twice, is refused — like an
/// unknown kind.
pub(crate) fn parse(entries: &[String]) -> Result<Vec<Grant>> {
    let mut out: Vec<Grant> = Vec::with_capacity(entries.len());
    for entry in entries {
        let grant = parse_entry(entry)?;
        anyhow::ensure!(
            !out.iter().any(|held| held.names_the_same(&grant)),
            "'{entry}' is named twice"
        );
        out.push(grant);
    }
    Ok(out)
}

/// Parse the stored entries, dropping only the entries the grammar refuses. The read side (building
/// a role's tool set) uses this so a single bad row cannot cost the role its whole tool set — the
/// validated write side never stores one, so this only ever narrows to what the record can serve.
pub(crate) fn parse_lossy(entries: &[String]) -> Vec<Grant> {
    let mut out: Vec<Grant> = Vec::with_capacity(entries.len());
    for entry in entries {
        let Ok(grant) = parse_entry(entry) else {
            continue;
        };
        // One entry per name, as `parse` holds: a repeat is dropped like any other entry the
        // grammar refuses.
        if !out.iter().any(|held| held.names_the_same(&grant)) {
            out.push(grant);
        }
    }
    out
}

/// Parse one entry: a script entry (`custom:<script>`, whose name is a plain file name) or a tool
/// entry (split on its first `:`, with the kinds split on `,` and refused on a tool that has no kind
/// vocabulary, or an unknown kind, or a kind named twice on one tool).
fn parse_entry(entry: &str) -> Result<Grant> {
    if let Some(name) = script_of(entry) {
        anyhow::ensure!(
            is_tool_name(name),
            "'{entry}' does not name a script — hint: a script's name is its file name without the \
             extension"
        );
        return Ok(Grant::Script {
            name: name.to_string(),
        });
    }
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
    Ok(Grant::Tool {
        tool: tool.to_string(),
        kinds,
    })
}

#[must_use]
pub(crate) fn held_kinds(grant: &Grant) -> Vec<String> {
    let Grant::Tool { tool, kinds } = grant else {
        return Vec::new();
    };
    if !kinds.is_empty() {
        return kinds.clone();
    }
    match kinds_of(tool).and_then(|vocab| vocab.first()) {
        Some(narrowest) => vec![(*narrowest).to_string()],
        None => Vec::new(),
    }
}

/// The entries as a display shows them: a bare tool entry with the narrowest kind it implies filled
/// in (`analyze` → `analyze:quick`), a script entry as its own name.
#[must_use]
pub(crate) fn describe_all(entries: &[String]) -> Vec<String> {
    parse_lossy(entries).iter().map(describe).collect()
}

fn describe(grant: &Grant) -> String {
    match grant {
        Grant::Script { name } => script_entry(name),
        Grant::Tool { tool, .. } => encode(&Grant::Tool {
            tool: tool.clone(),
            kinds: held_kinds(grant),
        }),
    }
}

#[must_use]
pub(crate) fn toggle(entries: &[String], tool: &str, kind: &str) -> Vec<String> {
    // Nothing to flip on a tool that has no kinds, nor on a kind the tool's vocabulary does not
    // name.
    let Some(vocab) = kinds_of(tool) else {
        return entries.to_vec();
    };
    if !vocab.contains(&kind) {
        return entries.to_vec();
    }
    // An entry the grammar refuses is carried through untouched — rewriting it is the write side's
    // job, and it refuses such a record outright — so a record the picker renders checkmarks for is
    // one this can still edit. The read is the same lenient one the picker and the launch path use.
    let mut out: Vec<String> = Vec::with_capacity(entries.len() + 1);
    let mut found = false;
    for entry in entries {
        let Ok(grant) = parse_entry(entry) else {
            out.push(entry.clone());
            continue;
        };
        // A script entry (or another tool's) is carried through untouched.
        if grant.tool() != Some(tool) {
            out.push(entry.clone());
            continue;
        }
        found = true;
        let mut kinds = held_kinds(&grant);
        match kinds.iter().position(|k| k == kind) {
            Some(at) => {
                kinds.remove(at);
            }
            None => kinds.push(kind.to_string()),
        }
        canonical_kinds(tool, &mut kinds);
        if !kinds.is_empty() {
            out.push(encode(&Grant::Tool {
                tool: tool.to_string(),
                kinds,
            }));
        }
    }
    if !found {
        out.push(encode(&Grant::Tool {
            tool: tool.to_string(),
            kinds: vec![kind.to_string()],
        }));
    }
    out
}

#[must_use]
fn encode(grant: &Grant) -> String {
    match grant {
        Grant::Script { name } => script_entry(name),
        Grant::Tool { tool, kinds } if kinds.is_empty() => tool.clone(),
        Grant::Tool { tool, kinds } => format!("{tool}:{}", kinds.join(",")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

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

    /// The grant vocabulary is [`Addressee::STRS`], never the wider vocabulary [`Addressee::parse`]
    /// reads: an entry is validated here before it ever reaches the tool's parser, so neither
    /// owner-side address is grantable.
    #[test]
    fn a_grant_can_never_name_the_command_lines_own_assistant_or_the_guest_handover() {
        assert!(parse(&entries(&["send:assistant"])).is_err());
        assert!(parse(&entries(&["send:guest"])).is_err());
        assert!(
            parse(&entries(&["send:worker,assistant"])).is_err(),
            "a known kind beside an unknown one does not make the entry valid"
        );
    }

    /// A script entry is one script, held beside any number of others and beside tools of the same
    /// name: `custom:read` and `read` are two different things.
    #[test]
    fn script_entries_are_individual_and_never_collapse() {
        let stored = entries(&["custom:read", "custom:weather", "read"]);
        let parsed = parse(&stored).unwrap();
        assert_eq!(
            parsed,
            vec![
                Grant::Script {
                    name: "read".to_string()
                },
                Grant::Script {
                    name: "weather".to_string()
                },
                Grant::Tool {
                    tool: "read".to_string(),
                    kinds: Vec::new()
                },
            ]
        );
        assert_eq!(scripts_of(&stored), entries(&["read", "weather"]));
        assert_eq!(
            describe_all(&stored),
            stored,
            "a script entry is shown as itself"
        );
        assert_eq!(
            toggle(&stored, "read", "quick"),
            stored,
            "a tool with no kinds leaves every entry alone, scripts included"
        );
    }

    /// The grammar refuses a script entry that names no plain file name, and one script named
    /// twice; the lenient read drops exactly those entries.
    #[test]
    fn a_script_entry_is_a_plain_file_name_named_once() {
        for bad in ["custom:", "custom:.hidden", "custom:a/b", "custom:a\\b"] {
            assert!(parse(&entries(&[bad])).is_err(), "{bad:?} must be refused");
            assert!(
                parse_lossy(&entries(&[bad])).is_empty(),
                "{bad:?} must be dropped on the read side"
            );
        }
        assert!(
            parse(&entries(&["custom:weather", "custom:weather"])).is_err(),
            "one script named twice is refused"
        );
        assert_eq!(
            parse_lossy(&entries(&["custom:weather", "custom:weather"])).len(),
            1,
            "the repeat is dropped like any other refused entry"
        );
    }
}

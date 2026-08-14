//! Read-only shell command validation (tree-sitter AST walker).
//!
//! [`check_command`] validates shell commands against a set of rules that
//! distinguish safe inspection commands from workspace-mutating ones.
//! Used by [`crate::tools::shell::ShellTool`] when operating in [`ShellMode::ReadOnly`].
//!
//! # Design
//!
//! The command is parsed with the tree-sitter bash grammar and validated by
//! one recursive walk over the syntax tree. The walker is **fail-closed**:
//! any parse error, missing node, or unrecognized node kind rejects the whole
//! command, and the temp-write contract (see the banner) is enforced by
//! validating every output redirect and every temp-gated mutator path against
//! the tracked current directory and temp-variable bindings.
//!
//! The command verb always comes from the tree's authoritative `command_name`
//! field, so substitution-formed verbs (`$(echo rm)`, backtick verbs) are
//! structurally rejected. Known parser gaps (herestrings, extglob, unquoted
//! `%(` in git --format, `<>` redirects, multiple heredocs per line, escaped
//! `$()` in unquoted heredoc bodies, unterminated quotes/heredocs) over-reject
//! fail-closed — the denial messages hint the plainer spellings. Retained
//! text-scanning helpers used by out-of-scope consumers live in
//! [`crate::tools::shell::scan`]; this module only uses word-level helpers on
//! words reconstructed from the syntax tree.

use std::path::Path;

use tree_sitter::{Node, Parser};

use super::scan::{self, CdScan};

/// Shell execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellMode {
    /// Full shell access — all commands allowed.
    Full,
    /// Read-only shell — only inspection commands allowed.
    ReadOnly,
}

// ── Const tables ─────────────────────────────────────────────────────────

/// Commands rejected unconditionally (any invocation).
///
/// NOTE: Script interpreters (bash, python, node, etc.) and container tools
/// (docker, kubectl, etc.) are intentionally NOT in this list. They are
/// general-purpose tools commonly used for read-only inspection (e.g.,
/// `python --version`, `docker ps`, `kubectl get pods`). Shell prefix
/// stripping covers dangerous wrapper patterns (sudo, eval, exec). The
/// trade-off accepts false negatives through indirection (e.g.,
/// `sh -c "rm -rf /"`, `python3 -c "__import__('os').system('rm -rf /')"`)
/// in favor of not breaking legitimate read-only usage.
const MUTATING_COMMANDS: &[&str] = &[
    // ── File mutation ──
    "shred",
    "mkfifo",
    "mknod",
    "ln",
    "install",
    "truncate",
    "fallocate",
    "split",
    "csplit",
    "patch",
    "scp",
    "sftp",
    "chmod",
    "chown",
    "chattr",
    "chflags",
    "setfacl",
    "rsync",
    "unzip",
    "vim",
    "vi",
    "nvim",
    "nano",
    "pico",
    "emacs",
    "ed",
    "code",
    "gedit",
    "sponge",
    "kill",
    "pkill",
    "killall",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "make",
    "cmake",
    // ── Package managers ──
    "npm",
    "yarn",
    "pnpm",
    "pip",
    "pip3",
    "pipenv",
    "poetry",
    "brew",
    "port",
];

/// Safe git subcommands (read-only inspection).
const GIT_SAFE_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "blame",
    "annotate",
    "shortlog",
    "describe",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "rev-list",
    "for-each-ref",
    "grep",
    "help",
    "version",
    "name-rev",
    "count-objects",
    "verify-pack",
    "verify-commit",
    "verify-tag",
    "check-attr",
    "check-ignore",
    "check-mailmap",
    "check-ref-format",
    "cat-file",
    "cherry",
    "diff-files",
    "diff-index",
    "diff-tree",
    "fmt-merge-msg",
    "fsck",
    "merge-base",
    "whatchanged",
    "reflog",
    "range-diff",
    "request-pull",
    "worktree list",
    "hash-object",
    "stripspace",
    "remote",
    "branch",
    "tag",
    "show-ref",
    "ls-remote",
];

/// Safe cargo subcommands that only affect build artifacts in `target/` or are purely
/// read-only queries. Commands that modify source files or `Cargo.lock` must NOT be added
/// here — they should get targeted rejection messages with tailored suggestions instead.
const CARGO_SAFE_SUBCOMMANDS: &[&str] = &[
    "build",
    "check",
    "test",
    "clippy",
    "rustc",
    "metadata",
    "tree",
    "locate-project",
    "pkgid",
    "report",
    "search",
    "info",
    "clean",
    "doc",
    "fmt",
    "version",
    "verify-project",
    "read-manifest",
    "help",
    "bench",
];

/// Git subcommands that always write — rejected regardless of flags. The
/// tailored suggestions double as denial-message education; the generic
/// rejection message covers any subcommand not listed here.
const GIT_ALWAYS_MUTATE: &[(&str, &str, &str)] = &[
    (
        "mktag",
        "`git mktag` is not allowed — it always writes a tag object to the object database.",
        "use `git verify-tag` or `git cat-file` to inspect existing tag objects.",
    ),
    (
        "mktree",
        "`git mktree` is not allowed — it always writes a tree object to the object database.",
        "use `git ls-tree` to inspect existing tree objects.",
    ),
    (
        "merge-file",
        "`git merge-file` is not allowed — it mutates files or writes to the object database.",
        "use `git diff` to compare files, or `diff`/`diff3` for three-way comparisons.",
    ),
    (
        "merge-tree",
        "`git merge-tree` is not allowed — it writes tree objects in its default mode.",
        "use `git merge-base` to find the merge base, or `git diff-tree` to inspect trees.",
    ),
    // push/clean mutate even with dry-run/force-looking tokens as values
    // (`git push -o --dry-run` is a real push) — unconditional rejects.
    (
        "push",
        "`git push` is not allowed — it writes to a remote repository.",
        "inspect remote state with `git ls-remote`, `git remote show`, or `git status`.",
    ),
    (
        "clean",
        "`git clean` is not allowed — it deletes untracked files.",
        "inspect untracked files with `git status` or `git ls-files --others`.",
    ),
];

/// `git branch`/`git tag` list-mode triggers and verify flags. A bare
/// positional after any list-mode trigger is a pattern (read-only listing);
/// without one it is a ref name being created or deleted — reject. Verify
/// mode (`git tag -v <tag>`) treats positionals as read-only verify targets.
/// Long options match by exact name or unambiguous abbreviation, mirroring
/// git's option parsing; ambiguous or unknown long options fail closed
/// (create-active).
const GIT_REF_SHORT_LIST: &[char] = &['l'];
/// Tag list shorts: `-l` and `-n` (list with optional attached message-count
/// value — `git tag -n 1` lists, `1` is a pattern).
const GIT_TAG_SHORT_LIST: &[char] = &['l', 'n'];

/// `git remote` mutation verbs — positionals name the remote; always reject.
const GIT_REMOTE_MUTATIONS: &[&str] = &[
    "add",
    "remove",
    "rm",
    "rename",
    "set-url",
    "set-head",
    "set-branches",
    "update",
    "prune",
];

// ── Context and validation state ─────────────────────────────────────────

/// Immutable context for one validation: the session workspace root, the
/// allowed OS temp roots, and the session shell's temp-variable baseline.
pub(super) struct CheckContext {
    pub(crate) workspace_root: std::path::PathBuf,
    pub(crate) temp_roots: Vec<std::path::PathBuf>,
    pub(crate) temp_vars: Vec<(String, String)>,
}

impl CheckContext {
    /// Context for a session rooted at `workspace_root` with the standard
    /// OS temp roots and the session shell's temp-variable bindings.
    pub(super) fn for_workspace(workspace_root: &Path) -> Self {
        Self {
            workspace_root: workspace_root.to_path_buf(),
            temp_roots: crate::tools::path::allowed_temp_roots(),
            temp_vars: vec![("TMPDIR".to_string(), super::TMPDIR_BASELINE.to_string())],
        }
    }
}

/// Binding state for one tracked variable.
#[derive(Debug, Clone)]
struct VarBinding {
    /// Bound value; `None` = unknown concrete value provably under a temp
    /// root (`NAME=$(mktemp -d)`).
    value: Option<String>,
    /// When true, subsequent expansions of this variable reject (fail-closed).
    /// Set by binding to a non-temp path; cleared by re-binding to a valid
    /// temp root.
    poisoned: bool,
}

/// Mutable state threaded through the validation of a single command string:
/// the tracked current directory and temp-variable bindings.
struct ValidationState<'a> {
    ctx: &'a CheckContext,
    /// Tracked current directory. `None` = fail-closed: relative paths reject
    /// (any non-literal-absolute `cd` form resets tracking).
    cwd: Option<std::path::PathBuf>,
    /// Number of `cd`/`pushd`/`popd` verbs (incl. eval-body cds) executed so
    /// far in the current-shell context. Construct parsers compare a part's
    /// count against the construct start to detect a cd that leaks into the
    /// parent shell (if/for/while/case/select bodies run in the current
    /// shell): the leaked CWD is untrackable, so the outer tracking resets.
    cd_count: u64,
    /// Tracked variable bindings by name: the session temp variables
    /// (TMPDIR/TMP/TEMP) plus any variable assigned a temp-root value
    /// (`SNAP=$(mktemp -d)`, `SNAP=/tmp/x`, ...) in the command chain.
    vars: std::collections::HashMap<String, VarBinding>,
    /// Directories provably created by an approved `mkdir` earlier in this
    /// command chain (normalized temp paths). A later `cd` into one of them
    /// tracks even when the directory did not exist at validation time.
    created_dirs: std::collections::HashSet<std::path::PathBuf>,
}

impl<'a> ValidationState<'a> {
    /// Initial state: CWD = the session's workspace root; temp vars bound from
    /// the context's session-environment snapshot.
    fn new(ctx: &'a CheckContext) -> ValidationState<'a> {
        let mut vars = std::collections::HashMap::new();
        for (name, value) in &ctx.temp_vars {
            vars.insert(
                name.clone(),
                VarBinding {
                    value: Some(value.clone()),
                    poisoned: false,
                },
            );
        }
        ValidationState {
            ctx,
            cwd: Some(ctx.workspace_root.clone()),
            cd_count: 0,
            vars,
            created_dirs: std::collections::HashSet::new(),
        }
    }

    /// Snapshot of the current tracking state. Used for command substitutions
    /// and construct parts, which execute in a subshell or conditional
    /// boundary: their `cd`/export must not leak to the outer tracking.
    fn snapshot(&self) -> ValidationState<'a> {
        ValidationState {
            ctx: self.ctx,
            cwd: self.cwd.clone(),
            cd_count: self.cd_count,
            vars: self.vars.clone(),
            created_dirs: self.created_dirs.clone(),
        }
    }
}

// ── Main validation entry ────────────────────────────────────────────────

/// Validate a shell command for read-only execution.
///
/// Parses the command with the tree-sitter bash grammar and walks the syntax
/// tree once, validating every executed command, redirect, and substitution
/// against the tracked state. Returns `Ok(())` if the command is safe, or
/// `Err(String)` with a descriptive rejection message.
///
/// Validation is scoped to the session being validated (workspace root +
/// shell environment from `ctx`) — never the daemon process's environment.
pub(super) fn check_command(command_str: &str, ctx: &CheckContext) -> Result<(), String> {
    let trimmed = command_str.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let mut state = ValidationState::new(ctx);
    parse_and_walk(trimmed, &mut state)
}

/// Fail-closed rejection for a command the bash grammar cannot parse. Parser
/// gaps over-reject (herestrings, `<>`, multiple heredocs per line, unquoted
/// `%(` in git --format); the banner lists the accepted over-rejection
/// classes.
fn parse_error(cmd: &str) -> String {
    format!(
        "⚠️ Read-only mode: the command could not be parsed as valid bash — rejected fail-closed.\n\
         Command: `{cmd}`\n\
         If this command is a known bash construct (herestrings, `<>` redirects, unquoted `%(` in git --format), \
         rewrite it in a plainer form (e.g. use `$()` not backticks, quote format strings, use a quoted heredoc delimiter)."
    )
}

// ── AST walker ───────────────────────────────────────────────────────────

/// Walk flags for one command-ish unit.
#[derive(Clone, Copy, Default)]
struct WalkFlags {
    /// `!`-negation context: `time` is demoted to the external command.
    negated: bool,
    /// Condition/case-arm head: `time` demoted for the first unit only.
    time_external: bool,
}

/// Shared walker context: the raw source (for word text reconstruction) and
/// the state at the start of the most recently walked command-ish unit at the
/// current construct level — the value a hoisted redirect of an enclosing
/// `redirected_statement` binds to (bash opens a redirect when its owning
/// command starts, so `cd /tmp > out` rejects `out` against the pre-cd cwd
/// while `cd /tmp && cat > out` allows it against /tmp). Walkers that descend
/// into inner commands (construct bodies, pipelines, substitution bodies)
/// wrap their traversal in [`LastStartGuard`] so the inner commands cannot
/// shift this value.
struct W<'a> {
    /// Owned source text — `eval` bodies are re-parsed, so the source is not
    /// tied to the top-level command's lifetime.
    src: String,
    last_start: ValidationState<'a>,
}

/// RAII restore of `w.last_start` to its value at guard creation. A hoisted
/// redirect binds to the state at its owning command's start; inner commands
/// clobber `w.last_start` to their own starts, so every walker that walks a
/// body of inner commands (constructs, pipelines, substitution bodies) must
/// wrap the traversal in this guard — forgetting it silently reintroduces the
/// redirect-state class of workspace writes.
struct LastStartGuard<'a, 'w> {
    w: &'w mut W<'a>,
    saved: ValidationState<'a>,
}

impl<'a, 'w> LastStartGuard<'a, 'w> {
    fn new(w: &'w mut W<'a>) -> Self {
        let saved = w.last_start.snapshot();
        LastStartGuard { w, saved }
    }

    /// Reborrow the walker context for the guarded traversal.
    fn w(&mut self) -> &mut W<'a> {
        self.w
    }
}

impl Drop for LastStartGuard<'_, '_> {
    fn drop(&mut self) {
        self.w.last_start = self.saved.snapshot();
    }
}

/// True when `kind` is a command-ish node: an executed unit the walker
/// dispatches on (as opposed to an operator, keyword, or word node).
fn is_commandish(kind: &str) -> bool {
    matches!(
        kind,
        "command"
            | "list"
            | "do_group"
            | "redirected_statement"
            | "pipeline"
            | "negated_command"
            | "if_statement"
            | "while_statement"
            | "until_statement"
            | "for_statement"
            | "c_style_for_statement"
            | "case_statement"
            | "compound_statement"
            | "subshell"
            | "function_definition"
            | "test_command"
            | "declaration_command"
            | "unset_command"
            | "variable_assignment"
            | "variable_assignments"
            | "command_substitution"
            | "process_substitution"
    )
}

/// Node kinds whose children are inert text/expansion syntax — they never
/// execute commands themselves, so the walker only descends into them to find
/// nested `$(...)`/process substitutions (via [`walk_word_substitutions`]).
fn is_wordish(kind: &str) -> bool {
    matches!(
        kind,
        "word"
            | "string"
            | "raw_string"
            | "ansi_c_string"
            | "translated_string"
            | "concatenation"
            | "expansion"
            | "simple_expansion"
            | "special_variable_name"
            | "arithmetic_expansion"
            | "binary_expression"
            | "unary_expression"
            | "postfix_expression"
            | "parenthesized_expression"
            | "ternary_expression"
            | "subscript"
            | "array"
            | "number"
            | "extglob_pattern"
            | "regex"
            | "brace_expression"
            | "variable_name"
            | "string_content"
            | "test_operator"
            | "heredoc_content"
            | "heredoc_body"
            | "comment"
    )
}

/// Raw source text of a node (the walker never normalizes — downstream
/// predicates work on the exact source, quotes and escapes preserved).
fn node_text(node: Node, w: &W) -> String {
    node.utf8_text(w.src.as_bytes()).unwrap_or("").to_string()
}

/// Recursively validate every executed substitution inside a word-ish node:
/// `$(...)`/backtick and process-substitution bodies are validated as nested
/// commands (subshell semantics — state changes discarded); arithmetic and
/// parameter-expansion nodes are descended into so only their nested
/// `$(...)` bodies execute. String content is never treated as a command
/// (`bash -c "rm f"` stays allowed — documented script-content residual —
/// while `echo "$(rm f)"` rejects).
fn walk_word_substitutions<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
) -> Result<(), String> {
    // The node itself may be a substitution (`echo $(touch f)` — the walker
    // descends into every word-ish child, and the substitution is one of them).
    match node.kind() {
        "command_substitution" => {
            // Backtick form: fail-closed reject (hint the `$()` spelling).
            if node_text(node, w).starts_with('`') {
                return Err(format!(
                    "⚠️ Read-only mode: backtick command substitution is not allowed — its content cannot be safely tracked.\n\
                     Command: `{}`\n\
                     Suggestion: use `$()` instead of backticks, e.g. `echo \"$(ls)\"`.",
                    w.src
                ));
            }
            let mut snap = state.snapshot();
            walk_substitution_body(node, w, &mut snap, flags)
        }
        "process_substitution" => {
            let mut snap = state.snapshot();
            walk_substitution_body(node, w, &mut snap, flags)
        }
        // command_name contains the verb word — its substitutions execute
        // (`if $(touch f); then ...` runs touch). herestring_redirect content
        // also executes (`cat <<< "$(touch f)"` runs touch). test_command
        // (`[[ ]]`) executes substitutions in its expressions.
        kind if is_wordish(kind)
            || kind == "variable_assignment"
            || kind == "command_name"
            || kind == "herestring_redirect"
            || kind == "test_command" =>
        {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                walk_word_substitutions(child, w, state, flags)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Validate the inner commands of a `$(...)`/backtick/process-substitution
/// node (the node's children after the introducer, before the closer). The
/// body runs in a subshell against its own snapshot — it must not disturb
/// `w.last_start`, which the enclosing redirected_statement reads as the
/// state at its owning command's start.
fn walk_substitution_body<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
) -> Result<(), String> {
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let inner: Vec<Node> = children
        .iter()
        .copied()
        .filter(|c| is_commandish(c.kind()))
        .collect();
    if inner.is_empty() {
        // Empty or pure-expansion body (`$( )`) — nothing executes.
        return Ok(());
    }
    walk_sequence_of(&inner, w, state, flags, &[])
}

/// Dispatch one node to its handler. The conservative default arm rejects —
/// any node kind the walker does not explicitly recognize fails closed (the
/// grammar may grow new shapes; over-rejection is accepted).
fn walk_node<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    extras: Vec<String>,
) -> Result<(), String> {
    if node.is_error() || node.is_missing() {
        return Err(parse_error(&w.src));
    }
    match node.kind() {
        "program" | "list" | "do_group" => {
            let mut cursor = node.walk();
            let children: Vec<Node> = node.children(&mut cursor).collect();
            walk_sequence_of(&children, w, state, flags, &extras)
        }
        "command" => walk_command(node, w, state, flags, extras),
        "redirected_statement" => walk_redirected(node, w, state, flags, extras),
        "pipeline" => walk_pipeline(node, w, state, flags, &extras),
        "negated_command" => {
            let inner = node
                .children(&mut node.walk())
                .find(|c| is_commandish(c.kind()))
                .expect("negated command has a command child");
            let f = WalkFlags {
                negated: true,
                ..flags
            };
            walk_node(inner, w, state, f, extras)
        }
        "if_statement" => walk_if(node, w, state, &extras),
        "while_statement" | "until_statement" => walk_while_until(node, w, state, &extras),
        "for_statement" => walk_for(node, w, state, &extras),
        "c_style_for_statement" => walk_c_style_for(node, w, state, &extras),
        "case_statement" => walk_case(node, w, state, &extras),
        "compound_statement" => walk_compound(node, w, state, &extras),
        "subshell" => walk_subshell(node, w, state, &extras),
        "function_definition" => walk_function(node, w, state, &extras),
        "test_command" => walk_word_substitutions(node, w, state, WalkFlags::default()),
        "variable_assignment" | "variable_assignments" => {
            walk_word_substitutions(node, w, state, WalkFlags::default())?;
            bind_assignments(node, w, state)
        }
        "declaration_command" => walk_declaration(node, w, state),
        "unset_command" => walk_unset(node, w, state),
        "command_substitution" | "process_substitution" => {
            let mut snap = state.snapshot();
            walk_substitution_body(node, w, &mut snap, flags)
        }
        kind if is_wordish(kind) => walk_word_substitutions(node, w, state, flags),
        "file_redirect" | "heredoc_redirect" | "herestring_redirect" => {
            validate_redirect(node, w, state)
        }
        "comment" => Ok(()),
        _ => Err(unrecognized_node(node, w)),
    }
}

fn unrecognized_node(node: Node, w: &W) -> String {
    format!(
        "⚠️ Read-only mode: the command contains an unrecognized shell construct (`{}`) — rejected fail-closed.\n\
         Command: `{}`",
        node.kind(),
        w.src
    )
}

/// Walk a sequence of sibling nodes (a program, list, construct part, or
/// substitution body): commands execute in order with threaded state; `&`
/// restores the chain-start state (the whole chain ran in a background
/// child); stray `;;`-family terminators reject. `time_external` applies to
/// the first command-ish child only (condition/case-arm heads demote `time`).
/// `extras` (words swallowed into hoisted redirect nodes) append to the LAST
/// command's arguments.
fn walk_sequence_of<'a>(
    nodes: &[Node],
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    extras: &[String],
) -> Result<(), String> {
    let mut chain_start = state.snapshot();
    let last_cmd = nodes.iter().rposition(|n| is_commandish(n.kind()));
    let mut first_cmd = true;
    for (i, child) in nodes.iter().enumerate() {
        let kind = child.kind();
        if !is_commandish(kind) {
            match kind {
                "&" => {
                    // The whole chain up to `&` ran in a background child —
                    // its state changes are discarded.
                    *state = chain_start.snapshot();
                    chain_start = state.snapshot();
                }
                ";;" | ";&" | ";;&" => {
                    return Err(format!(
                        "⚠️ Read-only mode: a stray `{kind}` terminator appears outside a case construct — rejected fail-closed (bash rejects it too).\n\
                         Command: `{}`",
                        w.src
                    ));
                }
                "ERROR" => return Err(parse_error(&w.src)),
                _ => {} // `;` `&&` `||` `|` newlines and keywords — state threads
            }
            continue;
        }
        let child_flags = if first_cmd {
            first_cmd = false;
            flags
        } else {
            WalkFlags {
                time_external: false,
                ..flags
            }
        };
        w.last_start = state.snapshot();
        let child_extras = if Some(i) == last_cmd {
            extras.to_vec()
        } else {
            Vec::new()
        };
        walk_node(*child, w, state, child_flags, child_extras)?;
    }
    Ok(())
}

/// One command's words, reconstructed from the syntax tree with the
/// parser-split re-join: an unbraced expansion mid-word (`$SNAP/$RANDOM/f`)
/// parses as two nodes (the first ending in a bare `$`) — the adapter merges
/// them back into one word so path/flag predicates see the real token.
fn collect_command_words(cmd: Node, w: &W) -> (Vec<String>, Vec<String>) {
    let mut words: Vec<String> = Vec::new();
    let mut assignments: Vec<String> = Vec::new();
    let mut cursor = cmd.walk();
    for child in cmd.children(&mut cursor) {
        match child.kind() {
            "command_name" => {
                let mut c2 = child.walk();
                for inner in child.children(&mut c2) {
                    if is_wordish(inner.kind()) {
                        push_word(&mut words, node_text(inner, w));
                    }
                }
            }
            "variable_assignment" => {
                // Kept in the word list too — resolve_verb/apply_env_bindings
                // expect leading assignments at the front (`TMPDIR=/tmp time
                // cd /tmp` must demote time via the assignment at index 0).
                let text = node_text(child, w);
                assignments.push(text.clone());
                push_word(&mut words, text);
            }
            "$" => push_word(&mut words, "$".to_string()), // translated-string split `$"..."`
            kind if is_wordish(kind) => push_word(&mut words, node_text(child, w)),
            _ => {}
        }
    }
    (words, assignments)
}

/// Push a word, merging it into the previous one when that word ends with a
/// bare `$` (the parser's unbraced-expansion split).
fn push_word(words: &mut Vec<String>, word: String) {
    if let Some(prev) = words.last_mut()
        && prev.ends_with('$')
    {
        prev.push_str(&word);
    } else {
        words.push(word);
    }
}

/// Validate one simple command: substitution recursion, env bindings, verb
/// resolution (cd/eval routing), then the keep-set dispatch.
fn walk_command<'a>(
    cmd: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    mut extras: Vec<String>,
) -> Result<(), String> {
    w.last_start = state.snapshot();
    let (mut words, assignments) = collect_command_words(cmd, w);
    words.append(&mut extras);

    // Substitutions execute BEFORE this command's own env bindings (bash
    // expands them with the shell's current variables). A `time`-prefixed
    // command's grammar also allows a subshell operand as a child (`time
    // ( rm -rf ./x )`) — it runs in a child shell, so validate it against a
    // snapshot (`time ( cd /tmp && cat > out )` allows the temp write inside
    // the subshell).
    let mut cursor = cmd.walk();
    for child in cmd.children(&mut cursor) {
        if child.kind() == "subshell" {
            let mut snap = state.snapshot();
            walk_node(child, w, &mut snap, WalkFlags::default(), Vec::new())?;
        } else {
            walk_word_substitutions(child, w, state, flags)?;
        }
    }
    for a in &assignments {
        check_git_env_binding(a)?;
    }
    let mut words_refs: Vec<&str> = words.iter().map(String::as_str).collect();
    check_words(&mut words_refs, state, flags, &words)
}

/// The ported segment validator: verb resolution (cd/eval/prefix routing),
/// temp bindings, and the mutator/git/cargo/flag dispatch. Operates on the
/// AST-reconstructed words; redirects and substitutions are validated by the
/// walker, not here.
#[allow(clippy::too_many_lines)] // security-critical command validator
fn check_words(
    words: &mut [&str],
    state: &mut ValidationState,
    flags: WalkFlags,
    originals: &[String],
) -> Result<(), String> {
    if words.is_empty() {
        return Ok(());
    }
    let negated = flags.negated || flags.time_external;
    let (verb_idx, verb) = match resolve_verb(words, negated) {
        VerbResolution::Informational | VerbResolution::None => {
            apply_env_bindings(words, state, None)?;
            return Ok(());
        }
        VerbResolution::Verb {
            class: VerbClass::Unprovable,
            ..
        } => {
            if !is_bare_substitution_segment(words) {
                let cmd = originals.join(" ");
                return reject(
                    &cmd,
                    "the command verb cannot be proven safe (concatenated quotes, escapes, or substitution-formed).",
                    "write the command name literally (e.g. `cd`, `rm`) so it can be validated.",
                );
            }
            apply_env_bindings(words, state, None)?;
            return Ok(());
        }
        VerbResolution::Verb {
            idx,
            class: VerbClass::Literal(v),
        } => (idx, v),
    };
    if matches!(verb, "cd" | "pushd" | "popd") {
        process_cd_words(words, verb_idx, verb, state);
        return Ok(());
    }
    if verb == "eval" {
        handle_eval_body(&words[verb_idx + 1..], state)?;
        return Ok(());
    }
    // Reserved words at command position: genuine stray terminators — bash
    // rejects them as syntax errors; fail closed rather than treat them as
    // commands. `{`/`}` are brace-group openers/closers whose group the
    // grammar flattened into a plain command (`time { rm f; }` parses the
    // group's words as the command's, and a stray `}` command follows) —
    // bash accepts the group and runs its body, so it must not fall through
    // as an unmodeled command either.
    if matches!(verb, "{" | "}") {
        let cmd = originals.join(" ");
        return reject(
            &cmd,
            "a brace group (`{ ...; }`) could not be parsed as a construct here — rejected fail-closed rather than validate its flattened words.",
            "write the command without the brace group, or quote it if it is meant as an argument.",
        );
    }
    if matches!(
        verb,
        ")" | "fi" | "done" | "esac" | "then" | "do" | "elif" | "else" | ";;" | ";&" | ";;&"
    ) {
        let cmd = originals.join(" ");
        return reject(
            &cmd,
            &format!(
                "`{verb}` is a shell control keyword appearing outside its construct — rejected fail-closed (bash rejects it too)."
            ),
            "remove the stray keyword, or complete the construct it belongs to.",
        );
    }

    apply_env_bindings(words, state, Some((verb_idx, verb)))?;

    let segment = originals.join(" ");

    // Effective command: strip shell prefixes and env assignments for the
    // blocklist dispatch.
    let first_word = super::first_command_word(&segment);
    if first_word.is_empty() {
        return Ok(());
    }
    let first_word = match classify_verb_word(first_word) {
        VerbClass::Literal(v) => v,
        VerbClass::Unprovable => {
            return reject(
                &segment,
                "the command verb cannot be proven safe (concatenated quotes, escapes, or substitution-formed).",
                "write the command name literally (e.g. `rm`, `touch`) so it can be validated.",
            );
        }
    };

    // 'mktemp' creates a temp directory and outputs its path — always allowed.
    if first_word == "mktemp" {
        return Ok(());
    }

    // Temp-gated mutator dispatch (scratch, temp, unconditional).
    for check in MUTATOR_CHECKS {
        if !check.verbs.contains(&first_word) {
            continue;
        }
        if check
            .rejects
            .is_none_or(|reject| reject(&segment, first_word, state))
        {
            let (education, fallback) = check.suggestions;
            return reject(
                &segment,
                &check.rejection.replace("{verb}", first_word),
                if has_unresolved_var_path(&segment, state) {
                    education
                } else {
                    fallback
                },
            );
        }
        if first_word == "mkdir" {
            record_mkdir_targets(&segment, state);
        }
        return Ok(());
    }

    // Git-specific checks
    if first_word == "git" {
        return check_git_segment(&segment);
    }

    // Cargo-specific checks
    if first_word == "cargo" {
        return check_cargo_segment(&segment);
    }

    // Flag-dependent checks: reject commands that use mutation flags.
    for check in FLAG_CHECKS {
        if first_word == check.verb && (check.predicate)(&segment, state) {
            return reject(&segment, check.rejection, check.suggestion);
        }
    }

    Ok(())
}

// ── Redirects and heredocs ───────────────────────────────────────────────

/// A `redirected_statement` hoists its command's redirects above the maximal
/// list/pipeline suffix ending at that command. The body is walked first
/// (source order), then each redirect validates against `w.last_start` — the
/// state at the redirect's owning command's start: the body itself for a
/// simple/assignment-only command (`!` unwraps to its negated command), the
/// LAST member for a list/pipeline body (bash feeds the redirect to the last
/// pipeline member), and the construct itself for a construct body. Construct
/// walkers, pipelines, and substitution bodies wrap their inner traversal in
/// [`LastStartGuard`] so an inner command's cd cannot shift the redirect state
/// (`true && { cd /tmp; cat; } > f` opens f in the pre-internal-cd cwd).
/// Words the parser swallowed into the redirect nodes (the grammar groups
/// `rm a > f c` as command(rm a) + file_redirect(> f c); `c` is a real rm
/// argument) append to the last command's argument list.
fn walk_redirected<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    extras: Vec<String>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let Some((body_idx, &body)) = children
        .iter()
        .enumerate()
        .find(|(_, c)| is_commandish(c.kind()))
    else {
        return Err(unrecognized_node(node, w));
    };
    let redirects = &children[body_idx + 1..];

    let mut extra = extras;
    for r in redirects {
        collect_redirect_extras(*r, w, &mut extra);
    }
    if !extra.is_empty() && !matches!(body.kind(), "command" | "list" | "pipeline") {
        // Compound-command bodies take no arguments — bash errors on the
        // extra word; fail closed rather than model the error.
        return Err(format!(
            "⚠️ Read-only mode: a word follows a redirect on a compound command — rejected fail-closed (bash rejects it as a syntax error).\n\
             Command: `{}`",
            w.src
        ));
    }

    w.last_start = state.snapshot();
    walk_body(body, w, state, flags, extra)?;

    let mut redirect_state = w.last_start.snapshot();
    // A `!`-negated body demotes a leading `time` to the external command
    // (its args never bind before the body expands); condition/case-arm heads
    // do the same — pass the walk flags so the owner's prefix scan matches
    // the verb resolution.
    let owner_negated = flags.negated || flags.time_external || body.kind() == "negated_command";
    let (mut assignments, assignment_only) = owning_command(body).map_or_else(
        || (Vec::new(), false),
        |n| owning_command_assignments(n, w, owner_negated),
    );
    if assignment_only {
        // An assignment-only command applies its bindings to the current shell
        // BEFORE the heredoc body expands and BEFORE its redirect opens —
        // unlike a command owner, whose marker-line redirect is pre-assignment
        // — so the applied state IS the redirect state here.
        for a in &assignments {
            bind_assignment_word(a, &mut redirect_state);
        }
        assignments = Vec::new();
    }
    for r in redirects {
        if r.kind() == "heredoc_redirect" {
            // Body substitutions expand after the owning command's assignments
            // apply but before it runs; `&&`/`||` tails run after it (post-body
            // `state`).
            validate_heredoc(*r, w, &mut redirect_state, &assignments, state)?;
        } else {
            validate_file_redirect(*r, w, &mut redirect_state)?;
        }
    }
    Ok(())
}

/// The command that owns a hoisted redirect: the body itself for a simple or
/// assignment-only command, or the last command of a list/pipeline body (bash
/// feeds the redirect to the last pipeline member). A `!`-negated body unwraps
/// to its inner command (`! VAR=val cmd` binds VAR=val before the heredoc
/// body expands). Construct bodies return None — they own their redirects as
/// a whole (see walk_redirected).
fn owning_command(body: Node) -> Option<Node> {
    match body.kind() {
        "command" | "variable_assignment" | "variable_assignments" => Some(body),
        "negated_command" => body
            .children(&mut body.walk())
            .find(|c| is_commandish(c.kind()))
            .and_then(owning_command),
        "list" | "pipeline" => body
            .children(&mut body.walk())
            .filter(|c| is_commandish(c.kind()))
            .last()
            .and_then(owning_command),
        _ => None,
    }
}

/// The env assignments of the redirect's owning command, and whether it is an
/// assignment-only command. `NAME=value` text of an assignment-only command,
/// or the `VAR=val` prefix words of a simple command. Under a FORWARDING
/// keyword-`time` — whose operand is a full command list — assignments parse
/// as plain words (`time TMPDIR=/etc cat`) yet bind before the heredoc body
/// expands, so they are collected from the word list too; quoted `"time"`, a
/// `!`-negated list (`! time ...` — `negated`), and a nested `time time` are
/// the external command whose args never bind. A `time`-wrapped command whose
/// timed operand is only assignments is assignment-only (`time TMPDIR=/etc
/// <<EOF` opens its redirect after the binding). After `command`/`builtin` a
/// `FOO=bar` word is the command NAME and never binds. Construct owners
/// cannot take an assignment prefix (`VAR=val { ...; }` is a bash syntax
/// error) — none.
fn owning_command_assignments(owner: Node, w: &W, negated: bool) -> (Vec<String>, bool) {
    match owner.kind() {
        "variable_assignment" => (vec![node_text(owner, w)], true),
        "variable_assignments" => (
            owner
                .children(&mut owner.walk())
                .filter(|c| c.kind() == "variable_assignment")
                .map(|c| node_text(c, w))
                .collect(),
            true,
        ),
        "command" => {
            let (words, mut assignments) = collect_command_words(owner, w);
            let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();
            let mut assignment_only = true;
            let mut i = 0;
            // Leading env assignments parse as variable_assignment nodes
            // (already collected); `time`'s operand is a full command list, so
            // its env assignments parse as plain words yet bind before the
            // heredoc body expands — collect them. Only a FORWARDING
            // keyword-`time` collects (external `time` passes its args to a
            // child — nothing binds). After `command`/`builtin` a `FOO=bar`
            // word is the command NAME — nothing after binds.
            while i < words.len() {
                let unquoted = scan::strip_outer_quotes(&words[i]).map_or("", |(c, _)| c);
                match unquoted {
                    "command" | "builtin" => {
                        assignment_only = false;
                        break;
                    }
                    "time" => {
                        let Some(past) = consume_time_prefix(&word_refs, i, negated) else {
                            assignment_only = false;
                            break;
                        };
                        i = past;
                        // The timed operand is a full command list: its head
                        // may carry a single `!` negation (`time ! FOO=bar cat`
                        // binds FOO=bar before the body expands), then env
                        // assignments.
                        if words.get(i).is_some_and(|o| o == "!") {
                            i += 1;
                        }
                        while words.get(i).is_some_and(|o| super::is_env_assignment(o)) {
                            assignments.push(words[i].clone());
                            i += 1;
                        }
                    }
                    _ => {
                        // A real verb word — the command is not assignment-only.
                        assignment_only = false;
                        break;
                    }
                }
            }
            (assignments, assignment_only)
        }
        _ => (Vec::new(), false),
    }
}

/// Walk the body of a `redirected_statement`, updating `w.last_start` to the
/// state at the body's last command start.
fn walk_body<'a>(
    body: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    extras: Vec<String>,
) -> Result<(), String> {
    match body.kind() {
        "command" => walk_command(body, w, state, flags, extras),
        "list" => {
            let mut cursor = body.walk();
            let children: Vec<Node> = body.children(&mut cursor).collect();
            walk_sequence_of(&children, w, state, flags, &extras)
        }
        "pipeline" => walk_pipeline(body, w, state, flags, &extras),
        kind if is_commandish(kind) => walk_node(body, w, state, flags, extras),
        _ => Err(unrecognized_node(body, w)),
    }
}

/// Collect argument words the parser swallowed into a hoisted redirect node:
/// a `file_redirect`'s words after the first target, and a `heredoc_redirect`'s
/// same-line marker words (`cat <<EOF x` makes `x` a cat argument). Body-line
/// words (parser glitches like escaped `\$HOME` in an unquoted body) are NOT
/// collected — the heredoc guard minimal-scans them as body content.
fn collect_redirect_extras(node: Node, w: &W<'_>, out: &mut Vec<String>) {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    match node.kind() {
        "file_redirect" => {
            // First word-ish child is the target; the rest are real args.
            let mut seen_target = false;
            for c in &children {
                if c.kind() == "file_descriptor" {
                    continue;
                }
                if is_wordish(c.kind()) {
                    if !seen_target {
                        seen_target = true;
                        continue;
                    }
                    out.push(node_text(*c, w));
                }
            }
        }
        "heredoc_redirect" => {
            let mut marker_end = 0usize;
            let mut after_start = false;
            for c in &children {
                if c.kind() == "heredoc_start" {
                    after_start = true;
                    marker_end = c.end_byte();
                    continue;
                }
                if !after_start || c.kind() == "heredoc_body" || c.kind() == "heredoc_end" {
                    continue;
                }
                if is_wordish(c.kind())
                    && !node_text(*c, w).starts_with('\n')
                    && !text_between_has_newline(&w.src, marker_end, c.start_byte())
                {
                    out.push(node_text(*c, w));
                }
            }
        }
        _ => {}
    }
}

/// True when the source text between `from` and `to` contains a newline.
fn text_between_has_newline(src: &str, from: usize, to: usize) -> bool {
    src.get(from..to).is_some_and(|s| s.contains('\n'))
}

/// Validate one redirect node (file_redirect or heredoc_redirect) against
/// `state` — the state at its owning command's start.
fn validate_redirect<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    match node.kind() {
        "file_redirect" => validate_file_redirect(node, w, state),
        "heredoc_redirect" => {
            // Standalone heredoc (no owning command): body and tail both run
            // against the current state; no assignment prefix applies.
            let mut snap = state.snapshot();
            validate_heredoc(node, w, state, &[], &mut snap)
        }
        "herestring_redirect" => Err(parse_error(&w.src)),
        _ => Err(unrecognized_node(node, w)),
    }
}

/// Validate a `> file`-family redirect: output ops (`>`, `>>`, `>|`, `>&`
/// with a path target, `&>`, `&>>`) must target /dev/null or a path under a
/// temp root; fd-dups (`>&2`, `2>&1`) and input ops (`<`, `<&`) are always
/// allowed. A missing target rejects. Substitutions anywhere in the node
/// (targets and words swallowed after the target) execute and are validated
/// first — `cat < $(rm -rf /)` and `echo hi > /tmp/out $(rm -rf /)` reject.
fn validate_file_redirect<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    for c in &children {
        walk_word_substitutions(*c, w, state, WalkFlags::default())?;
    }
    let mut op: Option<String> = None;
    let mut target: Option<Node> = None;
    let mut fd_dup_target = false;
    for c in &children {
        match c.kind() {
            "file_descriptor" => {}
            ">" | ">>" | ">|" | ">&" | "<" | "<&" | "&>" | "&>>" | "<>" => {
                op = Some(c.kind().to_string());
            }
            kind if is_wordish(kind) || kind == "number" => {
                if target.is_none() {
                    target = Some(*c);
                    fd_dup_target = c.kind() == "number";
                }
            }
            "ERROR" => return Err(parse_error(&w.src)),
            _ => {}
        }
    }
    let Some(op) = op else {
        return Err(unrecognized_node(node, w));
    };
    let is_output = matches!(op.as_str(), ">" | ">>" | ">|" | ">&" | "&>" | "&>>");
    let is_dup = op == ">&" || op == "<&";
    if !is_output {
        return Ok(()); // input redirects (`<`, `<&`) are read-only
    }
    if is_dup && fd_dup_target {
        return Ok(()); // fd duplication (`2>&1`, `cat >&2`)
    }
    let Some(target) = target else {
        return Err(format!(
            "⚠️ Read-only mode: command contains a disallowed output redirect (bare redirect with no target).\n\
             Command: `{}`\n\
             Suggestion: write the redirect target explicitly, or drop the redirect.",
            w.src
        ));
    };
    let target_text = node_text(target, w);
    if target_text == "/dev/null" {
        return Ok(());
    }
    if writes_outside_temp(&target_text, state) {
        return Err(disallowed_redirect_err(&w.src, &target_text));
    }
    Ok(())
}

/// Rejection message for a redirect targeting a non-temp path.
fn disallowed_redirect_err(cmd: &str, target: &str) -> String {
    format!(
        "⚠️ Read-only mode: command contains a disallowed output redirect (`{target}`).\n\
         Command: `{cmd}`\n\
         Redirects are only allowed to /dev/null, 2>&1, 1>&2, or paths under /tmp, /var/tmp, or the OS temp directory.\n\
         Suggestion: pipe to a pager (e.g., `| less`) or use `| head` to limit output."
    )
}

/// Validate a heredoc redirect: the marker text must be a plain delimiter;
/// marker-line children execute and are validated (`cat <<EOF | rm f` must
/// reject); an unquoted body's substitutions execute and are validated,
/// while its raw text is minimal-scanned for backticks/`$(` (invisible to the
/// parser). Quoted-delimiter bodies are literal — nothing executes.
/// Bash expansion order: the body's substitutions expand AFTER the owning
/// command's VAR=value assignments apply but BEFORE the command runs, so the
/// body validates against `cmd_state` (the owning command's start state —
/// pre-cd for `cd /tmp <<EOF`) PLUS `assignments` applied; marker-line words
/// (arguments) and the marker-line redirect expand pre-assignment against
/// plain `cmd_state`. A `|`-glued pipeline member forks WITH the command
/// (pre-command `cmd_state`); `&&`/`||` tails run after it (`tail_state`).
fn validate_heredoc<'a>(
    node: Node,
    w: &mut W<'a>,
    cmd_state: &mut ValidationState<'a>,
    assignments: &[String],
    tail_state: &mut ValidationState<'a>,
) -> Result<(), String> {
    let mut body_state = cmd_state.snapshot();
    for a in assignments {
        bind_assignment_word(a, &mut body_state);
    }
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let mut unquoted = true;
    let mut body: Option<Node> = None;
    let mut marker_line_end = 0usize;
    let mut seen_start = false;
    let mut seen_body = false;

    for c in &children {
        match c.kind() {
            "<<" | "<<-" | "&&" | "||" => {}
            "heredoc_start" => {
                seen_start = true;
                marker_line_end = c.end_byte();
                let marker = node_text(*c, w);
                // The delimiter must be a single bare or quoted word; any
                // metacharacter glued into it (`cat <<EOF| rm f`,
                // `cat <<EOF>file` — the parser also errors on these, but the
                // guard splits the marker text itself) is rejected.
                let bare = marker.strip_prefix(['\'', '"']).unwrap_or(&marker);
                let bare = bare.strip_suffix(['\'', '"']).unwrap_or(bare);
                if bare.is_empty()
                    || bare.chars().any(|ch| {
                        ch.is_whitespace()
                            || matches!(ch, '|' | '>' | '<' | '&' | ';' | '(' | ')' | '`')
                    })
                {
                    return Err(format!(
                        "⚠️ Read-only mode: malformed heredoc delimiter `{marker}` — rejected fail-closed.\n\
                         Command: `{}`\n\
                         Suggestion: give the heredoc a plain word delimiter (e.g. `<<'EOF'`) and put commands after the terminator line.",
                        w.src
                    ));
                }
                unquoted = !marker.starts_with(['\'', '"']);
            }
            "heredoc_body" => {
                seen_body = true;
                body = Some(*c);
            }
            "heredoc_end" => break,
            "file_redirect" => {
                // A redirect on the marker line opens when the command starts
                // — against the owning command's start state, before its
                // VAR=value assignments apply (`TMPDIR=/etc cat <<EOF >
                // "$TMPDIR/f"` opens $TMPDIR pre-assignment).
                validate_file_redirect(*c, w, cmd_state)?;
            }
            "pipeline" => {
                // A `|`-glued member forks WITH the owning command (both run
                // at pipeline start) — validate against the pre-command state
                // (`cd /tmp <<EOF | rm f` runs rm in the pre-cd cwd).
                let mut snap = cmd_state.snapshot();
                walk_node(*c, w, &mut snap, WalkFlags::default(), Vec::new())?;
            }
            kind if is_commandish(kind) => {
                walk_node(*c, w, tail_state, WalkFlags::default(), Vec::new())?;
            }
            kind if is_wordish(kind) => {
                // A word after the marker: same-line words are the owning
                // command's arguments — their substitutions expand
                // pre-assignment (`cat <<EOF "$(rm -rf /)"` runs the
                // substitution); body-line words are parser glitches of body
                // content — in unquoted bodies walk substitutions against the
                // body state (post-assignment) and minimal-scan the raw text
                // (backticks/`$(` execute there).
                let marker_line = !seen_body
                    && seen_start
                    && !node_text(*c, w).starts_with('\n')
                    && !text_between_has_newline(&w.src, marker_line_end, c.start_byte());
                if marker_line {
                    walk_word_substitutions(*c, w, cmd_state, WalkFlags::default())?;
                } else if unquoted {
                    walk_word_substitutions(*c, w, &mut body_state, WalkFlags::default())?;
                    minimal_body_scan(&node_text(*c, w))?;
                }
            }
            _ => walk_word_substitutions(*c, w, cmd_state, WalkFlags::default())?,
        }
    }

    // Body handling: quoted delimiter → literal (skip); unquoted → recurse
    // into parsed substitution children and minimal-scan raw text.
    if let Some(body_node) = body
        && unquoted
    {
        let mut bcur = body_node.walk();
        let bchildren: Vec<Node> = body_node.children(&mut bcur).collect();
        if bchildren.is_empty() {
            minimal_body_scan(&node_text(body_node, w))?;
        } else {
            for bc in &bchildren {
                match bc.kind() {
                    "command_substitution"
                    | "process_substitution"
                    | "expansion"
                    | "arithmetic_expansion" => {
                        let mut snap = body_state.snapshot();
                        walk_word_substitutions(*bc, w, &mut snap, WalkFlags::default())?;
                    }
                    "heredoc_content" => minimal_body_scan(&node_text(*bc, w))?,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// Minimal scan of raw unquoted-heredoc-body text: backticks and unescaped
/// `$(` execute in bash but are invisible to the parser — reject (fail-closed;
/// escaped `$()` over-rejects, an accepted parser-gap class).
fn minimal_body_scan(text: &str) -> Result<(), String> {
    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().expect("i < len");
        if c == '\\' {
            i += c.len_utf8();
            if i < text.len() {
                i += text[i..].chars().next().expect("i < len").len_utf8();
            }
            continue;
        }
        if c == '`' || (c == '$' && text[i + c.len_utf8()..].starts_with('(')) {
            return Err("⚠️ Read-only mode: command substitution inside an unquoted heredoc body is not allowed.\n\
                 Suggestion: quote the heredoc delimiter (e.g. `<<'EOF'`) to make the body literal, or remove the `$()`/backticks.".to_string());
        }
        i += c.len_utf8();
    }
    Ok(())
}

// ── Construct walkers ────────────────────────────────────────────────────

/// Merge a construct part's state into the outer state: a cd executed in the
/// part (which runs in the current shell) leaves the real CWD untrackable —
/// reset fail-closed. Variable bindings made in the part are uncertain (the
/// branch may not execute) — poison them.
fn note_part_cd(part: &ValidationState, base: &mut ValidationState) {
    if part.cd_count != base.cd_count {
        base.cwd = None;
        base.cd_count = part.cd_count;
    }
    for (name, pb) in &part.vars {
        match base.vars.get(name) {
            Some(bb) if !bb.poisoned && !pb.poisoned && bb.value == pb.value => {}
            None | Some(_) => {
                base.vars.insert(
                    name.clone(),
                    VarBinding {
                        poisoned: true,
                        value: pb.value.clone(),
                    },
                );
            }
        }
    }
}

/// True when a list separator (`;` or newline) appears in the source gap
/// between the construct keyword (`after`) and the first command (`start`) —
/// the separator consumes the condition-head `time` demotion.
fn head_has_separator(src: &str, after: usize, start: usize) -> bool {
    src.get(after..start)
        .is_some_and(|s| s.contains(['\n', ';']))
}

/// Walk one construct part (a sequence of command-ish children) against a
/// snapshot of the pre-construct state, then merge via [`note_part_cd`].
/// `time_external` marks a condition/case-arm head.
fn walk_part<'a>(
    nodes: &[Node],
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    time_external: bool,
    head_end: usize,
) -> Result<(), String> {
    let mut part = state.snapshot();
    // The head demotion covers the FIRST unit only: a separator before the
    // first command consumes it, returning `time` to reserved
    // (`if\n time cd /tmp` tracks the cd; `if time cd /tmp` does not). The
    // parser drops newlines from the AST, so the check also reads the source
    // gap between the construct keyword and the first command.
    let demote = time_external
        && !nodes
            .iter()
            .take_while(|n| !is_commandish(n.kind()))
            .any(|n| n.kind() == ";")
        && !nodes
            .iter()
            .find(|n| is_commandish(n.kind()))
            .is_some_and(|c| head_has_separator(&w.src, head_end, c.start_byte()));
    walk_sequence_of(
        nodes,
        w,
        &mut part,
        WalkFlags {
            time_external: demote,
            ..WalkFlags::default()
        },
        &[],
    )?;
    note_part_cd(&part, state);
    Ok(())
}

/// Children of a construct node split at the given keyword kinds; returns the
/// child ranges for each section: `(before, between, after)` — the node
/// kinds `start`/`mid`/`end` bound the sections.
fn split_children<'t>(
    node: Node<'t>,
    _w: &W<'_>,
    start: &str,
    mid: &str,
    end: &str,
) -> (Vec<Node<'t>>, Vec<Node<'t>>, Vec<Node<'t>>, bool) {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let mut first: Vec<Node> = Vec::new();
    let mut second: Vec<Node> = Vec::new();
    let mut third: Vec<Node> = Vec::new();
    let mut phase = 0u8;
    let mut closed = false;
    for c in &children {
        match c.kind() {
            k if k == start => phase = 1,
            k if k == mid => {
                phase = 2;
                closed = true;
            }
            k if k == end => {
                phase = 3;
                closed = true;
            }
            _ => match phase {
                1 => first.push(*c),
                2 => second.push(*c),
                _ => third.push(*c),
            },
        }
    }
    (first, second, third, closed)
}

fn walk_if<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let (cond, body, _rest, _) = split_children(node, w, "if", "then", "fi");
    walk_part(&cond, w, state, true, keyword_end(node, "if"))?;
    walk_part(&body, w, state, false, 0)?;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "elif_clause" => {
                let (econd, ebody, _, _) = split_children(child, w, "elif", "then", "fi");
                walk_part(&econd, w, state, true, keyword_end(child, "elif"))?;
                walk_part(&ebody, w, state, false, 0)?;
            }
            "else_clause" => {
                // else_clause children: `else`, body commands — the body is
                // the FIRST field (unlike elif_clause, which has a condition
                // before `then`).
                let (ebody, _, _, _) = split_children(child, w, "else", "then", "fi");
                walk_part(&ebody, w, state, false, 0)?;
            }
            _ => {}
        }
    }
    reject_redirect_extras(extras, w)
}

/// End byte of the first child of `node` with kind `kind` (a construct
/// keyword), for the condition-head separator check.
fn keyword_end(node: Node<'_>, kind: &str) -> usize {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|c| c.kind() == kind)
        .map_or(0, |c| c.end_byte())
}

fn redirect_extras_on_construct(w: &W) -> String {
    format!(
        "⚠️ Read-only mode: a word follows a redirect on a compound command — rejected fail-closed (bash rejects it as a syntax error).\n\
         Command: `{}`",
        w.src
    )
}

/// Reject when a hoisted redirect swallowed argument words on a compound
/// command (bash rejects the shape as a syntax error too).
fn reject_redirect_extras(extras: &[String], w: &W) -> Result<(), String> {
    if extras.is_empty() {
        Ok(())
    } else {
        Err(redirect_extras_on_construct(w))
    }
}

fn walk_while_until<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    // The grammar collapses `until` into `while_statement` (only the keyword
    // child differs), so the keyword is probed, not dispatched on node.kind().
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let (cond, _, _, _) = split_children(node, w, "while", "do", "done");
    if cond.is_empty() {
        let (cond, _, _, _) = split_children(node, w, "until", "do", "done");
        walk_part(&cond, w, state, true, keyword_end(node, "until"))?;
    } else {
        walk_part(&cond, w, state, true, keyword_end(node, "while"))?;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "do_group" {
            let mut part = state.snapshot();
            walk_node(child, w, &mut part, WalkFlags::default(), Vec::new())?;
            note_part_cd(&part, state);
        }
    }
    reject_redirect_extras(extras, w)
}

fn walk_for<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    // Header: `for`/`select`, variable_name, optional `in` + words, `;`.
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut header_words: Vec<String> = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "in" | "for" | "select" | ";" | "\n" | "variable_name" => {}
            kind if is_wordish(kind) => {
                header_words.push(node_text(child, w));
            }
            "do_group" => {
                let mut part = state.snapshot();
                if let Some(binding) = loop_var_binding(&header_words, state) {
                    let name = loop_var_name(node, w);
                    part.vars.insert(name, binding);
                }
                walk_node(child, w, &mut part, WalkFlags::default(), Vec::new())?;
                note_part_cd(&part, state);
            }
            _ => {
                // Substitutions in the header word list execute
                // (`for i in $(ls)`) — validate them.
                walk_word_substitutions(child, w, state, WalkFlags::default())?;
            }
        }
    }
    reject_redirect_extras(extras, w)
}

fn loop_var_name(node: Node, w: &W) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "variable_name" {
            return node_text(child, w);
        }
    }
    String::new()
}

fn walk_c_style_for<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    // Arithmetic header substitutions execute (`for ((i=$(rm f); ...))`).
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "do_group" => {
                let mut part = state.snapshot();
                walk_node(child, w, &mut part, WalkFlags::default(), Vec::new())?;
                note_part_cd(&part, state);
            }
            _ => walk_word_substitutions(child, w, state, WalkFlags::default())?,
        }
    }
    reject_redirect_extras(extras, w)
}

fn walk_case<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    // Subject and patterns execute substitutions (bash expands them); each
    // arm body is a conditional boundary.
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "case_item" => {
                let mut part = state.snapshot();
                walk_case_item(child, w, &mut part)?;
                note_part_cd(&part, state);
            }
            "case" | "in" | "esac" | ";" | "\n" => {}
            _ => walk_word_substitutions(child, w, state, WalkFlags::default())?,
        }
    }
    reject_redirect_extras(extras, w)
}

fn walk_case_item<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let mut body: Vec<Node> = Vec::new();
    for c in &children {
        match c.kind() {
            // Pattern section — substitutions execute.
            ")" | ";;" | ";&" | ";;&" | "|" => {}
            kind if is_wordish(kind) => {
                walk_word_substitutions(*c, w, state, WalkFlags::default())?;
            }
            kind if is_commandish(kind) => body.push(*c),
            _ => {}
        }
    }
    if body.is_empty() {
        return Ok(());
    }
    // The arm-head `time` demotion is consumed by a separator after the `)`
    // (like condition heads): `case x in x) time cd /tmp;; esac` demotes,
    // `case x in x)\n time cd /tmp;; esac` tracks.
    let paren_end = children
        .iter()
        .find(|c| c.kind() == ")")
        .map_or(0, tree_sitter::Node::end_byte);
    let demote = !body
        .first()
        .is_some_and(|c| head_has_separator(&w.src, paren_end, c.start_byte()));
    let mut part = state.snapshot();
    walk_sequence_of(
        &body,
        w,
        &mut part,
        WalkFlags {
            time_external: demote,
            ..WalkFlags::default()
        },
        &[],
    )?;
    note_part_cd(&part, state);
    Ok(())
}

fn walk_compound<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    // `(( ... ))` arithmetic command — substitutions inside execute; nothing
    // else does.
    if children.iter().any(|c| c.kind() == "((") {
        reject_redirect_extras(extras, w)?;
        for c in &children {
            walk_word_substitutions(*c, w, state, WalkFlags::default())?;
        }
        return Ok(());
    }
    // `{ ...; }` brace group — runs in the current shell: a cd inside leaks.
    let entry_count = state.cd_count;
    let body: Vec<Node> = children
        .iter()
        .copied()
        .filter(|c| is_commandish(c.kind()))
        .collect();
    walk_sequence_of(&body, w, state, WalkFlags::default(), extras)?;
    if state.cd_count != entry_count {
        state.cwd = None;
    }
    Ok(())
}

fn walk_subshell<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut snap = state.snapshot();
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let body: Vec<Node> = children
        .iter()
        .copied()
        .filter(|c| is_commandish(c.kind()))
        .collect();
    walk_sequence_of(&body, w, &mut snap, WalkFlags::default(), extras)
}

fn walk_function<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    extras: &[String],
) -> Result<(), String> {
    reject_redirect_extras(extras, w)?;
    // A function body does not execute at definition time — validate it
    // against a snapshot and discard.
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut snap = state.snapshot();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if is_commandish(child.kind()) {
            walk_node(child, w, &mut snap, WalkFlags::default(), Vec::new())?;
        }
    }
    Ok(())
}

fn walk_pipeline<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
    flags: WalkFlags,
    extras: &[String],
) -> Result<(), String> {
    // Pipeline members run in subshells: each is validated against a snapshot
    // of the pipeline-start state; `state` itself is never mutated.
    let mut g = LastStartGuard::new(w);
    let w = g.w();
    let mut cursor = node.walk();
    let members: Vec<Node> = node
        .children(&mut cursor)
        .filter(|c| is_commandish(c.kind()))
        .collect();
    let base = state.snapshot();
    for (i, m) in members.iter().enumerate() {
        let mflags = if i == 0 {
            flags
        } else {
            WalkFlags {
                time_external: false,
                ..flags
            }
        };
        let mextras = if i + 1 == members.len() {
            extras.to_vec()
        } else {
            Vec::new()
        };
        let mut mstate = base.snapshot();
        walk_node(*m, w, &mut mstate, mflags, mextras)?;
    }
    Ok(())
}

// ── Declaration and unset handlers ───────────────────────────────────────

/// `declare`/`local`/`typeset`/`readonly`/`export` bind their assignments
/// (`export TMPDIR=/etc` poisons TMPDIR; `export "TMPDIR=/tmp"` binds /tmp).
/// Option words and bare names (no `=`) do not change the tracked binding.
/// Non-identifier words (`export A=1 B=2 rm f`) are skipped by bash (only the
/// names are exported; `rm` never executes) — the walker leaves them unbound
/// (allow-ward, safe).
fn walk_declaration<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    // Two passes mirroring bash: ALL substitutions expand before ANY
    // assignment applies (`export MYTMP=/tmp x=$(echo $MYTMP)` expands with
    // MYTMP unset), so every child's substitutions walk against the
    // pre-binding state first, then the bindings apply in order.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_word_substitutions(child, w, state, WalkFlags::default())?;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "variable_assignment" => {
                let text = node_text(child, w);
                check_git_env_binding(&text)?;
                bind_assignment_word(&text, state);
            }
            "string" => {
                // `export "TMPDIR=/tmp"` — the string_content holds NAME=value.
                let text = node_text(child, w);
                if let Some((inner, _)) = scan::strip_outer_quotes(&text) {
                    check_git_env_binding(inner)?;
                    bind_assignment_word(inner, state);
                }
            }
            // `export "TMPDIR"=/etc` parses as a concatenation (string + word);
            // bind by its raw text (bind_assignment_word strips quotes).
            kind if is_wordish(kind) => {
                let text = node_text(child, w);
                check_git_env_binding(&text)?;
                bind_assignment_word(&text, state);
            }
            _ => {}
        }
    }
    Ok(())
}

/// `unset NAME` models an EMPTY binding (bash expands `$NAME` to ''), not a
/// return to the session baseline. Options (`-f`, `-v`, `-n`) are skipped.
fn walk_unset<'a>(
    node: Node,
    w: &mut W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Substitutions execute (`unset $(rm -rf /)`) — walk them first.
        walk_word_substitutions(child, w, state, WalkFlags::default())?;
        match child.kind() {
            "variable_name" | "word" => {
                let text = node_text(child, w);
                let name = scan::strip_quoted_word(&text);
                if !name.starts_with('-') && !name.is_empty() {
                    state.vars.insert(
                        name.to_string(),
                        VarBinding {
                            value: Some(String::new()),
                            poisoned: false,
                        },
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Bind the variable assignments of a standalone `variable_assignment` /
/// `variable_assignments` list member (`VAR=1; echo $VAR`).
fn bind_assignments<'a>(
    node: Node,
    w: &W<'a>,
    state: &mut ValidationState<'a>,
) -> Result<(), String> {
    if node.kind() == "variable_assignment" {
        let text = node_text(node, w);
        check_git_env_binding(&text)?;
        bind_assignment_word(&text, state);
        return Ok(());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "variable_assignment" {
            let text = node_text(child, w);
            check_git_env_binding(&text)?;
            bind_assignment_word(&text, state);
        }
    }
    Ok(())
}

// ── Temp-write contract machinery ────────────────────────────────────────

/// Synthetic anchor path for a temp-tagged variable (`NAME=$(mktemp -d)`):
/// the first allowed temp root plus one level. The concrete value is unknown,
/// but any `mktemp` result lives exactly one level below a temp root, so
/// `..` chains over this anchor produce the same under-temp verdict as the
/// real value.
fn temp_anchor_path(ctx: &CheckContext) -> String {
    let root = ctx
        .temp_roots
        .first()
        .map_or_else(|| "/tmp".to_string(), |p| p.to_string_lossy().into_owned());
    format!("{root}/{TEMP_ANCHOR_SEGMENT}")
}

/// Placeholder segment for a temp-tagged variable's unknown concrete value.
const TEMP_ANCHOR_SEGMENT: &str = "__mahbot_temp__";

/// Placeholder segment for an unbound variable used as an opaque suffix after
/// a temp anchor. Must not contain `/` or `..`.
const OPAQUE_SEGMENT: &str = "__mahbot_opaque__";

/// True when the partial expansion text is provably under a temp root (a
/// literal temp prefix or a temp-anchored variable has been emitted).
fn is_under_temp_prefix(out: &str, state: &ValidationState) -> bool {
    let p = Path::new(out);
    p.is_absolute() && crate::tools::path::is_path_under_roots(p, &state.ctx.temp_roots)
}

/// Parse a `$NAME` or `${NAME}` reference at the start of `rest`.
/// Returns `(name, total_consumed_len)`.
fn parse_var_ref(rest: &str) -> Option<(&str, usize)> {
    let after_dollar = rest.strip_prefix('$')?;
    if let Some(braced) = after_dollar.strip_prefix('{') {
        let end = braced.find('}')?;
        let name = &braced[..end];
        if name.is_empty()
            || name.contains([':', '-', '=', '?', '+', '/', '#', '%', '!', '@', '*', '['])
        {
            return None;
        }
        return Some((name, end + 3)); // $ { name }
    }
    let name = after_dollar
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>();
    if name.is_empty() {
        return None;
    }
    Some((&after_dollar[..name.len()], name.len() + 1))
}

/// Resolved value of a tracked variable.
enum VarValue {
    /// Concrete value (temp-tracked path text).
    Concrete(String),
    /// Unknown concrete value, provably under a temp root (`NAME=$(mktemp -d)`).
    TempRoot,
    /// Unbound variable — only `$RANDOM` is a safe opaque suffix after a temp
    /// anchor; anything else fails closed.
    Opaque,
    /// Poisoned binding (bound to a non-temp path) — expansions reject.
    Blocked,
}

fn resolve_var(name: &str, state: &ValidationState) -> VarValue {
    match name {
        "PWD" => match state.vars.get(name) {
            // Explicit assignment wins: `PWD=/etc` must not be masked by the
            // tracked CWD (`cd /tmp && PWD=/etc touch $PWD/f` → /etc/f).
            Some(b) => {
                if b.poisoned {
                    VarValue::Blocked
                } else {
                    match &b.value {
                        Some(v) => VarValue::Concrete(v.clone()),
                        None => VarValue::TempRoot,
                    }
                }
            }
            None => match &state.cwd {
                Some(cwd) => VarValue::Concrete(cwd.to_string_lossy().into_owned()),
                None => VarValue::Blocked,
            },
        },
        "HOME" => VarValue::Blocked,
        _ => match state.vars.get(name) {
            None => VarValue::Opaque,
            Some(VarBinding { poisoned: true, .. }) => VarValue::Blocked,
            Some(VarBinding { value: None, .. }) => VarValue::TempRoot,
            Some(VarBinding {
                value: Some(v),
                poisoned: false,
            }) if !v.is_empty() => VarValue::Concrete(v.clone()),
            // Empty binding (`unset NAME` — bash expands to '').
            Some(VarBinding {
                value: Some(_),
                poisoned: false,
            }) => VarValue::Concrete(String::new()),
        },
    }
}

/// Expand `$VAR` / `${VAR}` references in `word`. Inside single quotes nothing
/// expands. Returns `None` (reject) when the expansion is unprovable: an
/// unbound variable without a temp anchor, a poisoned variable, `$HOME`,
/// `$PWD`-when-untracked, or a `..` escape segment after an opaque suffix.
///
/// Temp-tagged variables (`NAME=$(mktemp -d)`) expand to a synthetic anchor
/// path (first allowed temp root + one level — the same depth as a real
/// `mktemp` result, so `..` chains escape at the same depth). The `$RANDOM`
/// builtin after a temp anchor expands to an opaque placeholder segment
/// (`$SNAP/$RANDOM/f`); any other unbound variable fails closed (its ambient
/// value is outside the guard's tracking model). The tail after the first
/// opaque segment is unverifiable, so any `..` path segment there fails
/// closed.
fn expand_vars(word: &str, single_quoted: bool, state: &ValidationState) -> Option<String> {
    if single_quoted {
        // No expansion inside single quotes — a literal `$` is not a valid
        // temp path.
        if word.contains('$') {
            return None;
        }
        return Some(word.to_string());
    }
    let mut out = String::with_capacity(word.len());
    let mut opaque_from: Option<usize> = None;
    let mut i = 0;
    while i < word.len() {
        let rest = &word[i..];
        // Escaped dollar/backslash/backtick: literal character, no expansion.
        if let Some(next) = rest.strip_prefix('\\') {
            // Word-final backslash cannot resolve to a concrete path —
            // fail closed instead of panicking.
            let c = next.chars().next()?;
            if matches!(c, '$' | '\\' | '`') {
                out.push(c);
            } else {
                out.push('\\');
                out.push(c);
            }
            i += 1 + c.len_utf8();
            continue;
        }
        if let Some((name, len)) = parse_var_ref(rest) {
            match resolve_var(name, state) {
                VarValue::Concrete(text) => out.push_str(&text),
                VarValue::TempRoot => out.push_str(&temp_anchor_path(state.ctx)),
                VarValue::Opaque => {
                    if name != "RANDOM" {
                        return None;
                    }
                    if opaque_from.is_none() && !is_under_temp_prefix(&out, state) {
                        return None;
                    }
                    if opaque_from.is_none() {
                        opaque_from = Some(out.len());
                    }
                    out.push_str(OPAQUE_SEGMENT);
                }
                VarValue::Blocked => return None,
            }
            i += len;
        } else {
            let c = rest.chars().next().expect("i < len");
            // An unescaped `$(`/backtick (command substitution) or an
            // unparseable `${...}` (parameter expansion with modifiers, e.g.
            // `${FOO:-../etc}`) has an unprovable output, so the word cannot
            // resolve to a concrete path — fail-closed (the escaped forms
            // were handled above; single-quoted words never reach this
            // branch; plain `${NAME}` parses as a variable reference).
            if c == '`'
                || (c == '$'
                    && rest
                        .as_bytes()
                        .get(1)
                        .is_some_and(|b| matches!(*b, b'(' | b'{')))
            {
                return None;
            }
            out.push(c);
            i += c.len_utf8();
        }
    }
    // The tail after the first opaque suffix is unverifiable: a `..` path
    // segment there could escape through the unknown value (fail-closed).
    if let Some(start) = opaque_from
        && out[start..].split('/').any(|seg| seg == "..")
    {
        return None;
    }
    Some(out)
}

/// Resolve a path word (mutator argument, redirect target, or flag value)
/// into an absolute path against the tracked state: shell-variable expansion
/// (`$VAR`/`${VAR}`), balanced surrounding quotes, tilde handling, and
/// resolution against the tracked current directory. Returns `None` when the
/// word cannot be safely resolved — unknown/poisoned variables, tilde paths
/// (never under temp), unbalanced/mixed quotes, relative paths without a
/// tracked cwd.
fn resolve_path_word(word: &str, state: &ValidationState) -> Option<std::path::PathBuf> {
    let word = word.trim();
    if word.is_empty() || word.starts_with('~') {
        return None;
    }
    let (clean, single_quoted) =
        scan::strip_outer_quotes(word).map_or((word, false), |(c, q)| (c, q));
    let expanded = expand_vars(clean, single_quoted, state)?;
    let p = Path::new(&expanded);
    if p.is_absolute() {
        return Some(p.to_path_buf());
    }
    // Relative globs stay rejected — they could match workspace files even
    // under a temp CWD (the glob expands at runtime against the real CWD).
    if crate::tools::path::contains_glob(&expanded, false) {
        return None;
    }
    // Relative path: resolve against the tracked cwd, if any.
    let cwd = state.cwd.as_ref()?;
    Some(cwd.join(p))
}

/// True when a path word resolves outside every allowed temp root (or cannot
/// be proven under one) — the temp-write contract gate.
fn writes_outside_temp(word: &str, state: &ValidationState) -> bool {
    let Some(p) = resolve_path_word(word, state) else {
        return true;
    };
    !is_path_under_temp(&p, state.ctx)
}

fn is_path_under_temp(path: &std::path::Path, ctx: &CheckContext) -> bool {
    if !crate::tools::path::is_path_under_roots(path, &ctx.temp_roots) {
        return false;
    }
    // Canonicalize (walking up parents for not-yet-existing paths): a symlink
    // inside a temp root pointing at a non-temp dir (or vice versa) must not
    // be treated as a temp path — writes through it land outside.
    let mut probe = path;
    loop {
        if let Ok(canon) = std::fs::canonicalize(probe) {
            return crate::tools::path::is_path_under_roots(&canon, &ctx.temp_roots);
        }
        let Some(parent) = probe.parent() else {
            return true;
        };
        if parent == probe {
            return true;
        }
        probe = parent;
    }
}

// ── Variable binding machinery ───────────────────────────────────────────

/// Reject `GIT_*` env bindings (quote-stripped) except the documented
/// `GIT_PAGER` carve-out — git only paginates on a TTY and the shell tool
/// captures via pipes, so a pager binding can never spawn. Covers
/// GIT_EXTERNAL_DIFF, GIT_SSH_COMMAND, GIT_CONFIG_*, GIT_DIR, GIT_EXEC_PATH,
/// GIT_TRACE*, GIT_ASKPASS — all invisible to the subcommand allowlist.
/// Also fires on non-git commands (`GIT_DIR=/tmp ls`): fail-closed trade-off
/// closing transitive git invocation (make/cargo inheriting GIT_*).
fn check_git_env_binding(word: &str) -> Result<(), String> {
    let w = scan::strip_quoted_word(word);
    if let Some((name, _)) = w.split_once('=')
        && name.starts_with("GIT_")
        && name != "GIT_PAGER"
    {
        return reject(
            word,
            &format!(
                "`{name}` environment bindings are not allowed — git env vars can execute programs (GIT_EXTERNAL_DIFF, GIT_SSH_COMMAND, ...)."
            ),
            "run git without GIT_* environment assignments.",
        );
    }
    Ok(())
}

fn apply_env_bindings(
    words: &[&str],
    state: &mut ValidationState,
    export_verb: Option<(usize, &str)>,
) -> Result<(), String> {
    let first_non_assign = words
        .iter()
        .position(|w| !super::is_env_assignment(w))
        .unwrap_or(words.len());
    for w in &words[..first_non_assign] {
        check_git_env_binding(w)?;
        bind_assignment_word(w, state);
    }
    if let Some((idx, v)) = export_verb
        && v == "export"
    {
        for w in &words[idx + 1..] {
            check_git_env_binding(w)?;
            bind_assignment_word(w, state);
        }
    }
    Ok(())
}

/// Bind a single `NAME=value` assignment word, stripping balanced surrounding
/// quotes first so `export "TMPDIR=/etc"` and `"TMPDIR"=/etc` bind by their
/// unquoted name and value.
fn bind_assignment_word(w: &str, state: &mut ValidationState) {
    let Some((name, value)) = w.split_once('=') else {
        return;
    };
    apply_single_binding(name, value, state);
}

/// Bind a single `NAME=value` assignment to a tracked variable.
///
/// Any variable assigned a value that provably resolves under a temp root is
/// tracked for path resolution (TMPDIR/TMP/TEMP as before, plus e.g.
/// `SNAP=/tmp/x`). `NAME=$(mktemp -d)` / `` NAME=`mktemp -d` `` bind the
/// variable to a fresh temp root with an unknown concrete value. A non-temp
/// binding poisons the variable (fail-closed); a temp binding clears the
/// poison.
///
/// The stored value is the *resolved* path (shell expansion happens at
/// assignment time, so `export TMPDIR="$TMPDIR/x"` binds the expanded path).
fn apply_single_binding(name: &str, value: &str, state: &mut ValidationState) {
    // Quoted assignment names (`"TMPDIR"=/etc`) bind by their unquoted content.
    let name = scan::strip_quoted_word(name);
    // Strip balanced surrounding quotes from the value (`export TMPDIR="/tmp"`).
    // A single-quoted value is a literal — no expansion, no substitution —
    // so it can never be an mktemp temp binding.
    let (clean, single_quoted) =
        scan::strip_outer_quotes(value).map_or((value, false), |(c, q)| (c, q));
    // `NAME=$(mktemp -d)` binds a fresh temp root with an unknown value.
    if !single_quoted {
        if let Some(binding) = mktemp_binding(clean, state) {
            state.vars.insert(name.to_string(), binding);
            return;
        }
        // Any other command substitution has an unprovable output. Resolving
        // the raw `$(...)` string as a literal path would fabricate a temp
        // value against a tracked temp cwd (`cd /tmp; SNAP=$(mktemp -d -p
        // /tmp/nonexistent_x)`): at runtime mktemp errors, SNAP binds empty,
        // and the chained write lands outside every temp root. Bind such
        // variables poisoned (fail-closed).
        if substitution_body(clean).is_some() {
            state.vars.insert(
                name.to_string(),
                VarBinding {
                    value: Some(clean.to_string()),
                    poisoned: true,
                },
            );
            return;
        }
    }
    let resolved = if !clean.is_empty() && !clean.starts_with('~') {
        if single_quoted && clean.contains('$') {
            None // literal `$` (e.g. `'$(mktemp -d)'`) never resolves under temp
        } else {
            resolve_path_word(clean, state)
        }
    } else {
        None
    };
    let under_temp = resolved
        .as_ref()
        .is_some_and(|p| is_path_under_temp(p, state.ctx));
    state.vars.insert(
        name.to_string(),
        VarBinding {
            value: Some(
                resolved.map_or_else(|| clean.to_string(), |p| p.to_string_lossy().into_owned()),
            ),
            poisoned: !under_temp,
        },
    );
}

/// Extract the command body from a `$(...)` or `` `...` `` substitution
/// (outer quotes already stripped).
fn substitution_body(value: &str) -> Option<&str> {
    if let Some(inner) = value.strip_prefix("$(") {
        return inner.strip_suffix(')');
    }
    if let Some(inner) = value.strip_prefix('`') {
        return inner.strip_suffix('`');
    }
    None
}

/// Detect a `NAME=$(mktemp ...)` / `` NAME=`mktemp ...` `` assignment and
/// return a temp-tagged binding (unknown concrete value, provably under a
/// temp root). Returns `None` when the value is not an mktemp substitution or
/// when the invocation's landing directory is not provably under a temp root
/// (the variable then falls through to the normal binding path, which poisons
/// it — fail-closed).
///
/// The landing directory comes from `-p DIR` / `--tmpdir=DIR` or from a
/// positional template. `-p DIR` is honored only when DIR provably exists (at
/// validation time or created by `mkdir` earlier in the chain): mktemp never
/// creates that directory, so a missing one makes it error and bind the
/// variable empty — a false temp anchor. The template is honored only when it
/// provably resolves under a temp root (an absolute temp path, or a variable
/// expanding to one). Absolute non-temp templates (e.g. under /etc, the
/// workspace, $HOME), relative templates (resolved against the shell cwd) and
/// the `-- <template>` forms carrying them all fail closed. The
/// space-separated `--tmpdir DIR` form is not provable — GNU treats the value
/// as a template, macOS as a directory — so it fails closed.
fn mktemp_binding(value: &str, state: &ValidationState) -> Option<VarBinding> {
    let inner = substitution_body(value)?;
    let mut words = inner.split_whitespace();
    if words.next()? != "mktemp" {
        return None;
    }
    let args: Vec<&str> = words.collect();
    let mut target_dir: Option<&str> = None;
    let mut template: Option<&str> = None;
    let mut after_ddash = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i];
        if a == "--" {
            after_ddash = true;
            i += 1;
            continue;
        }
        if !after_ddash {
            if a == "-p" {
                if i + 1 < args.len() {
                    target_dir = Some(args[i + 1]);
                    i += 2;
                    continue;
                }
                return None; // missing dir value → fail-closed
            }
            if a == "--tmpdir" {
                return None;
            }
            if let Some(dir) = a.strip_prefix("--tmpdir=") {
                target_dir = Some(dir);
                i += 1;
                continue;
            }
            if a == "-t" {
                if i + 1 < args.len() {
                    i += 2;
                    continue;
                }
                return None;
            }
            if matches!(a, "-d" | "-q" | "-u" | "--dry-run" | "--quiet") {
                i += 1;
                continue;
            }
            if a.starts_with('-') {
                return None;
            }
        }
        if template.is_some() {
            return None;
        }
        template = Some(a);
        i += 1;
    }
    if let Some(t) = template {
        let clean = scan::strip_quoted_word(t);
        if !clean.ends_with("XXXXXX") {
            return None;
        }
        let p = resolve_path_word(clean, state)?;
        let parent = p.parent()?;
        let parent_norm = crate::tools::path::normalize_path(parent);
        if !is_path_under_temp(&p, state.ctx)
            || !path_exists_or_created(parent, &parent_norm, state)
        {
            return None;
        }
    }
    match target_dir {
        None => Some(VarBinding {
            value: None,
            poisoned: false,
        }),
        Some(dir) => {
            let clean = scan::strip_quoted_word(dir);
            let raw = resolve_path_word(clean, state)?;
            let normalized = crate::tools::path::normalize_path(&raw);
            if is_path_under_temp(&normalized, state.ctx)
                && path_exists_or_created(&raw, &normalized, state)
            {
                Some(VarBinding {
                    value: None,
                    poisoned: false,
                })
            } else {
                None
            }
        }
    }
}

// ── cd / eval / mkdir ────────────────────────────────────────────────────

/// A `cd`/`mktemp` target is provably reachable at runtime when the raw path
/// resolves through the filesystem (`is_dir` follows every component,
/// matching `chdir`/`stat` semantics) or — for `..`-free paths — was recorded
/// by an earlier `mkdir` in the chain. A raw path containing `..` is only
/// provable when it already exists: a missing prefix component fails the
/// command at runtime even when the normalized tail exists.
fn path_exists_or_created(
    raw: &std::path::Path,
    normalized: &std::path::Path,
    state: &ValidationState,
) -> bool {
    raw.is_dir()
        || (!raw
            .components()
            .any(|c| c == std::path::Component::ParentDir)
            && state.created_dirs.contains(normalized))
}

/// Resolve the target of a `cd`/`pushd`/`popd` verb at `cd_idx` (eval bodies
/// are re-parsed and reach this through the normal command walk). pushd/popd
/// always reset fail-closed; only `cd` tracks. Option skipping and target
/// extraction go through [`scan::cd_target_after_options`]; a bare
/// `cd`/`cd -P` or an invalid option (`cd -e`) resets fail-closed — the cd
/// errors at runtime, so tracking would approve a chained write that lands
/// in the real CWD.
fn process_cd_words(words: &[&str], cd_idx: usize, verb: &str, state: &mut ValidationState) {
    // Every executed cd/pushd/popd moves the real CWD in the current shell —
    // construct walkers compare this counter to detect the leak.
    state.cd_count += 1;
    if verb != "cd" {
        state.cwd = None;
        return;
    }
    let (target, next) = match scan::cd_target_after_options(words, cd_idx + 1) {
        CdScan::Target(target, next) => (target, next),
        CdScan::Bare | CdScan::BadOption => {
            state.cwd = None;
            return;
        }
    };
    // A word after the target is an extra operand (`cd /tmp extra`). The
    // runtime shell ignores extras and executes the cd to the first target;
    // the guard cannot prove the target across shells — reset fail-closed.
    if words.get(next).is_some() {
        state.cwd = None;
        return;
    }
    if target.starts_with('/') {
        let p = std::path::PathBuf::from(target);
        if is_path_under_temp(&p, state.ctx) {
            // Track only when the directory exists or was provably created by
            // `mkdir` earlier in this chain. A nonexistent target — or one
            // whose raw path has a missing prefix component (`/tmp/a/../b`
            // with `a` absent) — fails the `cd` at runtime and a chained
            // write would land in the real CWD (fail-closed).
            let normalized = crate::tools::path::normalize_path(&p);
            state.cwd = if path_exists_or_created(&p, &normalized, state) {
                Some(p)
            } else {
                None
            };
        } else {
            // Filesystem existence checks are expected and acceptable;
            // a nonexistent target fails the command, so
            // tracking resets fail-closed.
            state.cwd = if p.is_dir() { Some(p) } else { None };
        }
    } else {
        // Relative cd under a tracked temp cwd tracks iff the target — after
        // variable expansion — resolves to a path under a temp root that
        // exists at validation time or was created by `mkdir` earlier in the
        // chain. A nonexistent target fails the `cd` at runtime, leaving the
        // real CWD one level shallower: tracking the deeper path would let a
        // `..` chained write climb past the temp root (fail-closed). `~` and
        // `-` never resolve under temp, and unresolvable targets (`$HOME`,
        // `$OLDPWD`, unbound variables) reject.
        let Some(cwd) = &state.cwd else {
            state.cwd = None;
            return;
        };
        if !is_path_under_temp(cwd, state.ctx) {
            state.cwd = None;
            return;
        }
        if target.starts_with('~') || target == "-" {
            state.cwd = None;
            return;
        }
        let Some(resolved) = resolve_path_word(target, state) else {
            state.cwd = None;
            return;
        };
        let normalized = crate::tools::path::normalize_path(&resolved);
        state.cwd = if is_path_under_temp(&normalized, state.ctx)
            && path_exists_or_created(&resolved, &normalized, state)
        {
            Some(normalized)
        } else {
            None
        };
    }
}

/// An `eval` segment evaluates its body in the current shell: the body is
/// decoded (one layer of surrounding quotes) and re-parsed + walked as the
/// command text itself — a `cd` inside tracks (the real CWD moves), and a
/// mutator inside rejects. Unparseable bodies fail closed.
fn handle_eval_body(body_words: &[&str], state: &mut ValidationState) -> Result<(), String> {
    let joined = body_words.join(" ");
    let decoded = if let Some((content, _)) = scan::strip_outer_quotes(&joined) {
        content.to_string()
    } else {
        // Unquoted body: strip quotes so the re-parse sees the words
        // eval would split on.
        let mut s = String::with_capacity(joined.len());
        s.extend(joined.chars().filter(|c| !matches!(c, '\'' | '"')));
        s
    };
    if decoded.trim().is_empty() {
        return Ok(());
    }
    parse_and_walk(&decoded, state)
}

/// Parse `text` with the bash grammar and walk it against `state` (used for
/// the top-level command and for re-parsed `eval` bodies).
fn parse_and_walk(text: &str, state: &mut ValidationState) -> Result<(), String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .map_err(|_| "failed to initialize the bash parser".to_string())?;
    let tree = parser
        .parse(text, None)
        .ok_or_else(|| "failed to parse the command".to_string())?;
    let root = tree.root_node();
    if root.has_error() {
        return Err(parse_error(text));
    }
    let mut w = W {
        src: text.to_string(),
        last_start: state.snapshot(),
    };
    walk_node(root, &mut w, state, WalkFlags::default(), Vec::new())
}

/// Record temp directories created by an approved `mkdir` segment (normalized
/// paths) so a later `cd` in the same command chain can track them even though
/// they did not exist at validation time. With `-p`/`--parents` the ancestors
/// are recorded too — `mkdir -p` creates them along the way.
fn record_mkdir_targets(segment: &str, state: &mut ValidationState) {
    let parents = segment
        .split_whitespace()
        .any(|w| w == "-p" || w == "--parents");
    for arg in non_flag_path_args(segment) {
        let Some(resolved) = resolve_path_word(&arg, state) else {
            continue;
        };
        let dir = crate::tools::path::normalize_path(&resolved);
        if !is_path_under_temp(&dir, state.ctx) {
            continue;
        }
        if parents {
            // `-p` creates the whole chain, so every ancestor is provable.
            state.created_dirs.insert(dir.clone());
            let mut d = dir;
            while let Some(parent) = d.parent() {
                d = parent.to_path_buf();
                if !is_path_under_temp(&d, state.ctx) {
                    break;
                }
                state.created_dirs.insert(d.clone());
            }
        } else if dir.parent().is_some_and(|p| {
            // Without `-p`, mkdir fails when the parent is absent at runtime:
            // a target is only provably created when its immediate parent
            // exists at validation time or was created by `mkdir` earlier in
            // this chain.
            p.is_dir()
                || state
                    .created_dirs
                    .contains(&crate::tools::path::normalize_path(p))
        }) {
            state.created_dirs.insert(dir);
        }
    }
}

/// Temp-binding for a for-loop variable: only when every static prefix of the
/// header word list provably resolves under a temp root (`for f in /tmp/*.db`)
/// is the variable temp-bound; a non-temp prefix (`for db in ~/.mahbot/db/*.db`)
/// leaves it unbound (existing fail-closed semantics apply).
fn loop_var_binding(words: &[String], state: &ValidationState) -> Option<VarBinding> {
    if words.is_empty() {
        return None;
    }
    words
        .iter()
        .all(|w| {
            // Static prefix: up to the first glob metacharacter.
            let prefix = w.find(['*', '?', '[']).map_or(w.as_str(), |g| &w[..g]);
            if prefix.is_empty() {
                return false;
            }
            resolve_path_word(prefix, state).is_some_and(|p| {
                is_path_under_temp(&crate::tools::path::normalize_path(&p), state.ctx)
            })
        })
        .then_some(VarBinding {
            value: None,
            poisoned: false,
        })
}

// ── Verb resolution ──────────────────────────────────────────────────────

/// Classification of a command verb word: provably literal (possibly with
/// balanced surrounding quotes) or unprovable (concatenated quotes, ANSI-C
/// quoting, escapes, substitutions — the guard cannot prove what the word
/// resolves to without full shell word-expansion modeling).
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerbClass<'a> {
    Literal(&'a str),
    Unprovable,
}

/// Result of scanning a command's leading env assignments and shell prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerbResolution<'a> {
    /// Verb word found at `idx` in the word list.
    Verb { idx: usize, class: VerbClass<'a> },
    /// A forwarding prefix with informational/invalid options (`command -v`,
    /// `builtin -p`, `time -p -p`): the command is provably never executed,
    /// so the segment leaves the real CWD and environment untouched.
    Informational,
    /// No verb word (empty command, only env assignments).
    None,
}

/// First effective command word: after env assignments and the forwarding
/// prefixes (`command`/`builtin`, plus `time` at the head), each prefix's
/// option grammar decides whether the command executes:
///
/// - `command`: `-p` (repeatable, including combined `-pp`) forwards; `-v`/`-V`
///   (alone or combined) and invalid options are informational; `--` ends
///   options and forwards.
/// - `builtin`: `--` forwards; any other option is invalid → informational.
/// - `time`: forwards ONLY as the unquoted first word of the full command,
///   outside a `!`-negated list and outside a condition/case-arm head (the
///   first word of an `if`/`elif`/`while`/`until` condition or case-arm
///   body — `if time cd /tmp` runs external time with the timed cd in a
///   child). Quoted `"time"`, an env assignment or `command`/`builtin`
///   before it, a nested `time time`, or `! time` all resolve to the
///   external `time` command — its timed command runs in a child, so the
///   guard must not track a cwd the shell does not reach. The timed operand
///   is a full command list whose head may carry a single `!` negation
///   (`time ! cmd` runs cmd negated in the CURRENT shell — bash-verified;
///   the cd still propagates) before its env assignments. At most one
///   UNQUOTED `-p` follows (a quoted `"-p"` is the timed command, not an
///   option); any further word (including `--`, a second `-p`, or `-v`) IS
///   the timed command, and a `-`-prefixed one does not exist → informational.
///
/// Prefixes may be composed (`command builtin cd`, `time command cd`,
/// `builtin -- command cd`) — each level forwards to the current shell, so
/// the loop continues until a non-forwarding verb is found. Non-forwarding
/// prefixes (`env`/`sudo`/`nice`/…) run in a child shell and are returned as
/// the verb (their command's CWD change never reaches the parent).
///
/// Leading env assignments (`TMPDIR=/tmp cd /tmp`) are valid before the first
/// prefix, and after `time` (whose operand is a full command list — `time
/// FOO=bar cd /tmp` runs the cd). After `command`/`builtin` a `FOO=bar` word
/// is the COMMAND NAME (`command FOO=bar cd /tmp` errors 127 and the cd never
/// runs), so assignments are never skipped there.
///
/// The verb word is classified by [`classify_verb_word`]: balanced-quoted
/// words (`"cd"`) stay literal and modeled; concatenated (`"c"d`, `c'd'`),
/// ANSI-C (`$'cd'`), escaped (`c\144`), or substitution-formed verbs are
/// unprovable. `eval` is not skipped: its body is handled as a current-shell
/// construct.
fn resolve_verb<'a>(words: &[&'a str], negated: bool) -> VerbResolution<'a> {
    let mut i = 0;
    // Leading env assignments are valid before the first prefix
    // (`TMPDIR=/tmp cd /tmp` runs the cd). After `command`/`builtin` a
    // `FOO=bar` word is the COMMAND NAME (`command FOO=bar cd /tmp` errors
    // with 127 — the cd never runs), so assignments are skipped only here
    // and after `time`, whose operand is a full command list (`time FOO=bar
    // cd /tmp` runs the cd).
    while i < words.len() && super::is_env_assignment(words[i]) {
        i += 1;
    }
    loop {
        if i >= words.len() {
            return VerbResolution::None;
        }
        let w = words[i];
        let unquoted = scan::strip_outer_quotes(w).map_or("", |(c, _)| c);
        if !matches!(unquoted, "command" | "builtin" | "time") {
            return VerbResolution::Verb {
                idx: i,
                class: classify_verb_word(w),
            };
        }
        let opts = &words[i + 1..];
        let mut j = 0;
        let executes = match unquoted {
            "command" => loop {
                let Some(o) = opts.get(j) else {
                    return VerbResolution::Informational;
                };
                let oq = scan::strip_outer_quotes(o).map_or("", |(c, _)| c);
                if oq == "--" {
                    j += 1;
                    break true;
                }
                if oq.starts_with('-') && oq.len() > 1 && oq[1..].bytes().all(|b| b == b'p') {
                    j += 1;
                    continue;
                }
                if oq.starts_with('-') && oq.len() > 1 {
                    return VerbResolution::Informational;
                }
                break true;
            },
            "builtin" => {
                let Some(o) = opts.first() else {
                    return VerbResolution::Informational;
                };
                let oq = scan::strip_outer_quotes(o).map_or("", |(c, _)| c);
                if oq == "--" {
                    j = 1;
                    true
                } else if oq.starts_with('-') && oq.len() > 1 {
                    return VerbResolution::Informational;
                } else {
                    true
                }
            }
            _ => {
                // `time` is the reserved word only as the unquoted first word
                // of the FULL segment (`i == 0`, raw word `time`), outside a
                // `!`-negated list and outside a condition/case-arm head.
                // Quoted `"time"`, an env assignment or `command`/`builtin`
                // before it, a nested `time time`, and `! time` all resolve
                // to the external `time` command — its timed command runs in
                // a child, so the real CWD never changes and tracking would
                // approve a chained write that lands in the workspace
                // (fail-closed: return it as a verb).
                let Some(past) = consume_time_prefix(words, i, negated) else {
                    return VerbResolution::Verb {
                        idx: i,
                        class: classify_verb_word(w),
                    };
                };
                // `time` consumes at most one `-p`, matched on the RAW word:
                // keyword parsing treats a quoted `"-p"` as the timed command
                // (`-p: command not found`, nothing executes), not as the
                // option. Anything after the option is the timed command; a
                // `-`-prefixed one does not exist (`time -p -p` → `-p:
                // command not found`), so nothing executes.
                j = past - (i + 1);
                if opts.get(j).is_some_and(|o| {
                    let oq = scan::strip_outer_quotes(o).map_or("", |(c, _)| c);
                    oq.starts_with('-') && oq.len() > 1
                }) {
                    return VerbResolution::Informational;
                }
                true
            }
        };
        if !executes {
            return VerbResolution::Informational;
        }
        let idx = i + 1 + j;
        if idx >= words.len() {
            return VerbResolution::None;
        }
        // The forwarded verb may itself be a forwarding prefix — continue the
        // loop so composed prefixes resolve to the real verb. Only `time`
        // takes a full command list, so a leading `!` negation (its operand is
        // a pipeline, bash-verified: `time ! cmd` runs cmd negated in the
        // current shell) and env assignments are valid again there (`time
        // FOO=bar cd /tmp` runs the cd); after `command`/`builtin` the next
        // word is the command name and must NOT be skipped.
        i = idx;
        if unquoted == "time" {
            if i < words.len() && words[i] == "!" {
                i += 1;
            }
            while i < words.len() && super::is_env_assignment(words[i]) {
                i += 1;
            }
        }
    }
}

/// Consume a forwarding keyword-`time` prefix at `words[i]`, returning the
/// index just past it (and its single optional unquoted `-p` option), or
/// `None` when the word is the external `time` command: quoted `"time"`, an
/// env assignment or `!` negation before it, a `command`/`builtin` before it,
/// or a nested `time time` — only the raw unquoted word `time` as the first
/// token of the full segment is the keyword.
fn consume_time_prefix(words: &[&str], i: usize, negated: bool) -> Option<usize> {
    if i != 0 || words.get(i) != Some(&"time") || negated {
        return None;
    }
    let mut j = i + 1;
    if words.get(j) == Some(&"-p") {
        j += 1;
    }
    Some(j)
}

/// Classify a verb word as provably literal or unprovable.
///
/// A balanced-quoted word (`"cd"`, `'cd'`) is literal (the shell concatenates
/// nothing and the quoted content is the command name). Anything with `$`
/// (variables, `$(...)`, ANSI-C `$'...'`), backslashes, brace-expansion
/// metacharacters (`{touch,}` expands to `touch` — an unmodeled word list),
/// or quote characters outside a balanced pair (`"c"d`, `c'd'`) cannot be
/// normalized without full word-expansion modeling — unprovable, fail-closed.
fn classify_verb_word(w: &str) -> VerbClass<'_> {
    if w.contains(['$', '\\', '{', '}', ',']) {
        return VerbClass::Unprovable;
    }
    if let Some((content, _)) = scan::strip_outer_quotes(w) {
        if content.is_empty() || content.contains(['\'', '"']) {
            return VerbClass::Unprovable;
        }
        return VerbClass::Literal(content);
    }
    if w.contains(['\'', '"']) {
        return VerbClass::Unprovable;
    }
    VerbClass::Literal(w)
}

/// True when the command's words are exactly one bare substitution span at
/// command position (a standalone `$(...)`/backtick — its content executes in
/// a subshell and was already validated by the walker).
fn is_bare_substitution_segment(words: &[&str]) -> bool {
    let Some(first) = words.first() else {
        return false;
    };
    if words.len() != 1 {
        return false;
    }
    let s = first.trim();
    scan::substitution_span(s, 0).is_some_and(|(_, next)| next == s.len() && !s.starts_with("${"))
}

// ── Keep-set dispatch tables ─────────────────────────────────────────────

/// Rejection template helper.
fn reject<T>(cmd: &str, why: &str, suggestion: &str) -> Result<T, String> {
    Err(format!(
        "⚠️ Read-only mode: {why}\n\
         Command: `{cmd}`\n\
         Suggestion: {suggestion}"
    ))
}

/// Scratch-file mutators allowed when all explicit path args are under temp.
const SCRATCH_MUTATORS: &[&str] = &["tee", "touch", "mkdir"];

/// Temp-directory mutators allowed when all explicit path args are under temp.
///
/// These are commands that modify files on disk but are allowed in read-only mode
/// when all path arguments target temp directories (/tmp, /var/tmp, or the OS temp
/// directory). The prompt tells agents: "Writing to the OS temp directory is allowed."
const TEMP_MUTATORS: &[&str] = &[
    "cp", "mv", "rm", "rmdir", "unlink", "gzip", "gunzip", "bzip2", "xz", "zstd", "zip",
];

/// One temp-gated mutator dispatch row.
struct MutatorCheck {
    verbs: &'static [&'static str],
    /// Reject predicate: true when the invocation writes outside temp and must
    /// be rejected; `None` = always rejected (unconditional mutator).
    rejects: Option<fn(&str, &str, &ValidationState) -> bool>,
    /// Rejection reason template with a `{verb}` placeholder.
    rejection: &'static str,
    /// Suggestion strings: the first educates about recognized temp-variable
    /// spellings when a `$` path failed to resolve; the second is the generic
    /// read-only-alternatives fallback.
    suggestions: (&'static str, &'static str),
}

/// True when a scratch mutator writes outside temp.
fn scratch_rejects(segment: &str, _verb: &str, state: &ValidationState) -> bool {
    !scratch_paths_under_temp(segment, state)
}

/// True when a temp mutator writes outside temp; `cp` only needs its
/// destination under temp (sources are read-only — copying from anywhere
/// into temp is allowed; copying into the workspace stays blocked).
fn temp_rejects(segment: &str, verb: &str, state: &ValidationState) -> bool {
    !temp_mutator_paths_under_temp(segment, verb, state)
}

const READONLY_ALTERNATIVES: &str = "use read-only alternatives to inspect files, e.g. `cat`, `head`, `tail`, `ls`, `file`, `stat`.";

/// Temp-gated mutator dispatch, iterated in order by [`check_words`]:
/// scratch mutators (tee/touch/mkdir), temp mutators (cp/mv/rm/…), then the
/// unconditional mutators (always rejected).
const MUTATOR_CHECKS: &[MutatorCheck] = &[
    MutatorCheck {
        verbs: SCRATCH_MUTATORS,
        rejects: Some(scratch_rejects),
        rejection: "`{verb}` is not allowed outside temp directories — it modifies the workspace.",
        suggestions: (
            "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`.",
            READONLY_ALTERNATIVES,
        ),
    },
    MutatorCheck {
        verbs: TEMP_MUTATORS,
        rejects: Some(temp_rejects),
        rejection: "`{verb}` is not allowed outside temp directories — it modifies files outside /tmp.",
        suggestions: (
            "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`.",
            "use paths under /tmp, /var/tmp, or the OS temp directory, or use read-only alternatives like `cat`, `head`, `tail`, `ls`, `file`, `stat`.",
        ),
    },
    MutatorCheck {
        verbs: MUTATING_COMMANDS,
        rejects: None,
        rejection: "`{verb}` is not allowed — it modifies the workspace.",
        suggestions: (READONLY_ALTERNATIVES, READONLY_ALTERNATIVES),
    },
];

/// A flag-dependent command check: if the command's first word matches `verb`
/// and the `predicate` returns true, the command is rejected with the given message.
struct FlagCheck {
    verb: &'static str,
    predicate: fn(&str, &ValidationState) -> bool,
    rejection: &'static str,
    suggestion: &'static str,
}

/// Flag-dependent checks: reject commands that use mutation flags.
/// Each entry tests a specific verb + predicate combination.
const FLAG_CHECKS: &[FlagCheck] = &[
    FlagCheck {
        verb: "sed",
        predicate: has_sed_mutation,
        rejection: "`sed` in-place editing (`-i`/`-I`/`--in-place`) is not allowed outside temp directories — it modifies files in-place.",
        suggestion: "use `sed` without in-place flags to output to stdout, e.g. `sed 's/a/b/' file`, or use `-i` with a path under /tmp.",
    },
    FlagCheck {
        verb: "awk",
        predicate: has_inplace,
        rejection: "`awk -i inplace` is not allowed — it edits files in-place.",
        suggestion: "use `awk` without `-i inplace` to output to stdout, e.g. `awk '{print $1}' file`, or use `-i inplace` with a path under /tmp.",
    },
    FlagCheck {
        verb: "dd",
        predicate: has_dd_mutation,
        rejection: "`dd of=...` is not allowed outside temp directories — it writes a file.",
        suggestion: "use `dd of=/tmp/...` to write under the OS temp directory, or use read-only alternatives like `cat`, `head`, `tail`, `ls`, `file`, `stat`.",
    },
    FlagCheck {
        verb: "curl",
        predicate: has_curl_mutation,
        rejection: "`curl` with output flags is not allowed outside temp directories.",
        suggestion: "use `curl` without output flags to display content in stdout, or use `curl -o /tmp/...` to save to temp.",
    },
    FlagCheck {
        verb: "tar",
        predicate: is_tar_mutating,
        rejection: "`tar` is only allowed in list mode (`-t`/`--list`) — extraction/creation modifies files.",
        suggestion: "use `tar -tf archive.tar.gz` to list contents, or `tar -xzf archive.tar.gz -C /tmp` to extract to temp.",
    },
    FlagCheck {
        verb: "base64",
        predicate: has_base64_mutation,
        rejection: "`base64` with output flags is not allowed outside temp directories.",
        suggestion: "use `base64` without output flags to print to stdout, or use `base64 -o /tmp/...` to save to temp.",
    },
    FlagCheck {
        verb: "wget",
        predicate: has_wget_mutation,
        rejection: "`wget` with output flags is not allowed outside temp directories.",
        suggestion: "use `curl` without output flags to display content in stdout, or use `wget -O /tmp/...` to save to temp.",
    },
];

/// Collect all non-flag, non-redirect, non-heredoc path-like arguments from a
/// command segment (the AST-reconstructed word list joined with spaces), so
/// temp-gated mutators see every real path argument. Skips shell flags,
/// standalone redirect operators and their targets, and self-contained
/// redirect tokens (see [`scan::classify_shell_token`]).
fn non_flag_path_args(segment: &str) -> Vec<String> {
    let words = scan::split_words_keeping_substitutions(segment);
    let Some(cmd_idx) = super::find_first_command_word_index(&words) else {
        return vec![];
    };
    let mut args = Vec::new();
    let mut skip_redirect_target = false;
    for w in &words[cmd_idx + 1..] {
        if skip_redirect_target {
            skip_redirect_target = false;
            continue;
        }
        if w.starts_with('-') {
            continue;
        }
        if let scan::TokenKind::Redirect { needs_target } = scan::classify_shell_token(w) {
            skip_redirect_target = needs_target;
            continue;
        }
        args.push(w.to_string());
    }
    args
}

/// True when a path argument contains a `$` variable that failed to resolve —
/// used to tailor rejection messages with the recognized temp-variable
/// spellings (a literal temp path, or `NAME=$(mktemp -d)`).
fn has_unresolved_var_path(segment: &str, state: &ValidationState) -> bool {
    non_flag_path_args(segment)
        .iter()
        .any(|p| p.contains('$') && resolve_path_word(p, state).is_none())
}

/// True when every explicit path argument resolves under an allowed temp root.
fn scratch_paths_under_temp(segment: &str, state: &ValidationState) -> bool {
    let paths = non_flag_path_args(segment);
    !paths.is_empty() && paths.iter().all(|p| !writes_outside_temp(p, state))
}

/// True when a temp mutator's path arguments all resolve under an allowed
/// temp root. `cp` is special-cased: only the destination must be under temp
/// (sources are read-only — copying from anywhere into temp is allowed;
/// copying into the workspace stays blocked).
fn temp_mutator_paths_under_temp(segment: &str, first_word: &str, state: &ValidationState) -> bool {
    if first_word == "cp" {
        return cp_destination_under_temp(segment, state);
    }
    scratch_paths_under_temp(segment, state)
}

/// Identify the `cp` destination: the value of
/// `-t`/`--target-directory`, or the last non-flag path argument. With no
/// identifiable destination the command is rejected.
fn cp_destination_under_temp(segment: &str, state: &ValidationState) -> bool {
    let words = scan::split_words_keeping_substitutions(segment);
    let Some(cmd_idx) = super::find_first_command_word_index(&words) else {
        return false;
    };
    let rest = &words[cmd_idx + 1..];

    if let Some(val) = output_flag_value(
        rest,
        &["-t", "--target-directory"],
        Some("--target-directory="),
    ) {
        return !writes_outside_temp(val, state);
    }
    // Last non-flag path argument is the destination.
    let Some(dest) = rest.iter().rfind(|w| {
        !w.starts_with('-')
            && !matches!(
                scan::classify_shell_token(w),
                scan::TokenKind::Redirect { .. }
            )
    }) else {
        return false;
    };
    !writes_outside_temp(dest, state)
}

fn word_has_substitution(w: &str) -> bool {
    let mut found = false;
    scan::for_each_substitution(w, |_, _, _, _| {
        found = true;
        false // stop scanning at the first hit
    });
    found
}

fn shell_word(word: &str) -> String {
    let word = word
        .strip_prefix('$')
        .filter(|rest| rest.starts_with(['\'', '"']))
        .unwrap_or(word);
    word.replace(['\'', '"', '\\'], "")
}

/// `$'...'`/`$"..."` tokens with backslash escapes are unprovable: escape
/// semantics are shell/version-dependent (`$'\x2di'` → `-i`), so treat them
/// as unprovable flag candidates — mirroring [`classify_verb_word`]'s
/// Unprovable verdict for `$`/`\`-words. Used by the sed/awk in-place gates,
/// the clippy `--fix` and git `--output` gates, and the git exec-vector scan
/// (`git log --format=$'%h\t%s'` is over-rejected, an accepted fail-closed
/// contract). `${...}`/`$var` are deliberately not treated as unprovable:
/// they also appear in legit read-only scripts (`sed "s/a/$var/" file`), and
/// their values are not decodable anyway.
fn is_unprovable_flag_token(part: &str) -> bool {
    (part.starts_with("$'") || part.starts_with("$\"")) && part.contains('\\')
}

/// True when `w` contains a substitution span at a flag-delivering position:
/// the shell-normalized word starts with `-` (a substitution may hide behind
/// it — `-$(echo o)`) or starts with `$`/backtick and holds an active span
/// (`$(...)`, backtick, `${...}`, `<(cmd)`) — ANY such span, provable or
/// not: whole-word `$(echo foo)` and whole-word arithmetic (`curl $((1+2))
/// URL`) over-reject, as do quoted flag-VALUE positions (`curl -d
/// "$(echo foo)" URL`); mid-word arithmetic is exempt from the field-split
/// arm (see [`unquoted_span_could_field_split_flag`]). Literal operands
/// (`file$(echo x)`, `"s/a/$(echo b)/"`) never match — their first expanded
/// char is fixed. An UNQUOTED mid-word span can additionally field-split a
/// standalone flag out of a literal-prefixed word (`file$(echo 'a -o')` → fields `filea`,
/// `-o`) — see [`unquoted_span_could_field_split_flag`]. Used by the flag
/// scans so substitution-formed mutation flags (`sed $(echo -i)`,
/// `curl -$(echo o)`, `curl file$(echo 'a -o') URL`) fail closed. Bare
/// `$var` stays exempt here — temp-tracked variables (`$TMPDIR/...`) must
/// stay usable as flag values — and is handled at the fixed-dictionary
/// positions that need it ([`contains_bare_var`] in the sed/hash-object/awk
/// flag checks). The equivalent `${var}` spelling is NOT exempt: an unquoted
/// `${...}` span fails closed even mid-word (`curl http://x/${port}/`
/// rejects, `curl http://x/$port/` allows) — an accepted asymmetry: bare
/// `$var` operands are pervasive in read-only commands (URLs, temp paths,
/// sed scripts) and stay usable, while the braced spelling at a
/// mutation-relevant word is over-rejected like any other unprovable span.
/// The curl/wget output gates have NO bare-`$var` arm (only this span-based
/// predicate), so `curl $port URL` with `port='-o /etc/passwd'` can
/// field-split the flag out — a pre-existing accepted residual (the old
/// plain-split guard allowed it too). `p` is the caller's shell-normalized
/// word ([`shell_word`]); it must equal `shell_word(w)` (debug-asserted) —
/// the cluster/sed gates already hold it for their own scans and pass it,
/// the output gates compute it per word in their caller.
fn substitution_could_form_flag(w: &str, p: &str) -> bool {
    debug_assert_eq!(p, shell_word(w));
    (word_has_substitution(w) && (p.starts_with('-') || p.starts_with(['$', '`'])))
        || unquoted_span_could_field_split_flag(w)
}

/// True when an UNQUOTED `$(...)`/backtick/`${...}` span in `w` could
/// field-split a standalone `-`-prefixed flag at runtime (`file$(echo 'a -o')`
/// → fields `filea`, `-o`), which the span-aware tokenizer keeps as ONE word
/// and the plain-split old guard saw as separate tokens. Only unquoted
/// expansions word-split — double-quoted ones concatenate, and process
/// substitution yields a single `/dev/fd/N` path — and a span that provably
/// echoes a single whitespace-free field cannot create a boundary
/// (`file$(echo x)` → `filex`). Any unprovable span (or one provably echoing
/// whitespace) fails closed: its output could be `a -o`.
fn unquoted_span_could_field_split_flag(w: &str) -> bool {
    let mut splits = false;
    scan::for_each_substitution(w, |span, content, _, in_double| {
        if !in_double
            && !span.starts_with("$((") // arithmetic: numeric output, no split
            && !span.starts_with("<(") // process substitution: single /dev/fd/N path
            && !span.starts_with(">(")
            && !matches!(
                simple_echo_output(content),
                Some(out) if !out.contains(char::is_whitespace)
            )
        {
            splits = true;
            return false;
        }
        true
    });
    splits
}

/// A shell word decomposed into fixed literal text and substitution spans.
enum WordPart {
    /// Fixed literal text (or a substitution body that provably echoes plain
    /// tokens — [`simple_echo_output`]).
    Lit(String),
    /// An unprovable expansion: its output can be ANY string, incl. empty.
    Any,
}

/// Decompose `w` into literal parts and substitution spans via the shared
/// substitution scan (no new quote tracking). A body that provably echoes
/// plain tokens is folded to its fixed output; everything else (commands,
/// `${...}`, nested/indirect bodies) is [`WordPart::Any`].
fn word_parts(w: &str) -> Vec<WordPart> {
    let mut parts: Vec<WordPart> = Vec::new();
    let mut pos = 0;
    scan::for_each_substitution(w, |span, content, end, _| {
        let start = end - span.len();
        if start > pos {
            parts.push(WordPart::Lit(w[pos..start].to_string()));
        }
        parts.push(match simple_echo_output(content) {
            Some(out) => WordPart::Lit(out),
            None => WordPart::Any,
        });
        pos = end;
        true
    });
    if pos < w.len() {
        parts.push(WordPart::Lit(w[pos..].to_string()));
    }
    parts
}

/// Fixed output of a substitution body that provably echoes plain tokens
/// (`$(echo foo bar)`); `None` when the output is unprovable (any other
/// command, `-`-prefixed or quoted args, `$var`, nested spans, ...).
fn simple_echo_output(body: &str) -> Option<String> {
    let body = body.trim();
    let (cmd, args) = body.split_once(char::is_whitespace)?;
    if cmd != "echo" {
        return None;
    }
    let toks: Vec<&str> = args.split_whitespace().collect();
    if toks.is_empty()
        || toks.iter().any(|t| {
            t.starts_with('-')
                || !t.chars().all(|c| {
                    c.is_ascii_alphanumeric()
                        || matches!(c, '_' | '.' | '/' | '+' | ':' | '@' | '%' | '=' | ',' | '-')
                })
        })
    {
        return None;
    }
    Some(toks.join(" "))
}

/// True when a substitution word `w` could expand to a mutation token at a
/// fixed-token gate. ANY unprovable span (`$(...)`, backtick, `${...}`,
/// `<(cmd)` — quoted or not) makes the output arbitrary, and an unquoted
/// expansion additionally field-splits, so the token can appear as its own
/// argv word (`HEAD${x:- --output=...}` → fields `HEAD`, `--output=...`);
/// every such word fails closed. When every span provably echoes plain tokens
/// ([`simple_echo_output`]), the folded expansion's whitespace-fields are
/// still checked (`inpl$(echo ace)` → `inplace`; `xx$(echo a b)--fix` → fields
/// `xxa`, `b--fix`) — excluding fields that start with `benign_prefix` (a
/// longer flag the token is a prefix of, e.g. `--output-indicator-*` for the
/// `--output` gate). ANSI-C `$'...'` escapes fail closed via the early return
/// below; bare `$var` is the callers' rule ([`contains_bare_var`]);
/// substitution-free words return false and callers keep their own literal
/// matching. The unprovable-span check runs on the RAW word (quote context
/// intact — a single-quoted span is literal, not an expansion).
fn word_could_form_token(w: &str, token: &str, benign_prefix: &str) -> bool {
    if !word_has_substitution(w) {
        return is_unprovable_flag_token(w);
    }
    let parts = word_parts(&shell_word(w));
    if parts.iter().any(|p| matches!(p, WordPart::Any)) {
        return true;
    }
    let mut full = String::new();
    for p in parts {
        if let WordPart::Lit(s) = p {
            full.push_str(&s);
        }
    }
    full.split_whitespace()
        .filter(|f| benign_prefix.is_empty() || !f.starts_with(benign_prefix))
        .any(|f| f.starts_with(token))
}

/// True when `w` contains a bare `$name` expansion (unbraced, not `$(`, `${`,
/// `$'`/`$"`: `$x`, `--$x`, `"$x"`, positional/special `$1`/`$@`/`$?`/`$$`).
/// The expanded value is unprovable. `$` inside single quotes or after a
/// backslash is literal (`'$x'`, `"\$x"`); `'ex'$x` concatenates and matches.
/// The general `$var` exemption elsewhere stays (see
/// [`word_has_substitution`]); gates that need a bare-var arm reach it
/// directly or via the [`unprovable_flag_word`] /
/// [`word_could_form_token_or_bare_var`] combos (git reflog subcommand, fsck
/// flags, sed/awk in-place, hash-object `-w`, dd `of=`, clippy `--fix`, git
/// `--output`).
fn contains_bare_var(w: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut it = w.chars().peekable();
    while let Some(c) = it.next() {
        let was_escaped = escaped;
        super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped);
        if c == '$'
            && !in_single
            && !was_escaped
            && it.peek().is_some_and(|n| {
                n.is_ascii_alphanumeric()
                    || matches!(n, '_' | '@' | '#' | '?' | '$' | '!' | '-' | '*')
            })
        {
            return true;
        }
    }
    false
}

/// True when `w` could deliver a mutation flag at a fixed-dictionary flag
/// position: a substitution word that could form a flag
/// ([`substitution_could_form_flag`]) or a bare-`$var` word whose expansion is
/// arbitrary and starts the field (`$w` could be `-w`, `$i` could be `-i`).
/// Fail-closed: any such word counts as the flag. The bare-`$var` arm fires on
/// ANY word starting with `$`/`` ` `` — including read-only temp operands
/// (`git hash-object $TMPDIR/f file`, `sed s/a/b/ $TMPDIR/x`) and positional
/// ref names (`git branch --list $name feature`) — the accepted over-rejection
/// contract of all three call sites (the cluster gate serving git config and
/// hash-object, sed in-place, branch/tag ref subcommand). The arm is
/// word-start-gated, so a mid-word bare var that
/// field-splits a flag (`git hash-object --$w file`, `w='x -w'` → fields
/// `--x`, `-w`) escapes while the `$(...)` spelling is caught — accepted
/// residual, unlike the fixed-token gates which fire on any-position bare
/// vars ([`word_could_form_token_or_bare_var`]). `p` contract: see
/// [`substitution_could_form_flag`].
fn unprovable_flag_word(w: &str, p: &str) -> bool {
    debug_assert_eq!(p, shell_word(w));
    substitution_could_form_flag(w, p) || (contains_bare_var(w) && p.starts_with(['$', '`']))
}

/// True when a substitution word could expand to `token` at a fixed-token
/// gate ([`word_could_form_token`]) or contains a bare `$var` — its unquoted,
/// arbitrary value can field-split the token out as its own argv field
/// (`--message-format=$fmt` → `--fix`, `if=$src` → `of=`). The shared
/// fail-closed tail of the awk/dd/clippy gates; gates needing a provable
/// fixed prefix (awk's `inplace` operand — its bare-`$var` prefix check is
/// inlined in [`has_inplace`]) or an exemption (git `--output`'s
/// `--output-indicator` prefix and `git remote` verb) keep their own
/// combination.
fn word_could_form_token_or_bare_var(w: &str, token: &str) -> bool {
    word_could_form_token(w, token, "") || contains_bare_var(w)
}

/// True when `w` contains any expansion the fixed-dictionary gates fail
/// closed on: a substitution span ([`word_has_substitution`] — incl. bodies
/// that provably echo single fields; these gates don't fold) or a bare
/// `$var` ([`contains_bare_var`]). Used at git reflog-subcommand and fsck
/// flag positions; the finer substitution-only / bare-var-armed gates use
/// the predicates directly.
fn word_has_unprovable_expansion(w: &str) -> bool {
    word_has_substitution(w) || contains_bare_var(w)
}

/// True when any single-dash token in `command` contains a char from
/// `mutation`, combined-cluster aware (`-df`, `-wt`, quoted `'-do'`).
/// `value_taking` chars consume the rest of their token (attached value), so
/// later chars are never misread as flags (`-tw` is `-t w`, `-dio` is
/// `-d -i 'o'`). Long flags and tokens without a leading dash never match.
/// Tokens are normalized as the shell delivers them ([`shell_word`]). A word
/// that could deliver a mutation char via a substitution or word-start bare
/// `$var` fails closed ([`unprovable_flag_word`]).
fn has_cluster_char(command: &str, mutation: &[char], value_taking: &[char]) -> bool {
    scan::split_words_keeping_substitutions(command)
        .into_iter()
        .any(|w| {
            let p = shell_word(w);
            if unprovable_flag_word(w, &p) {
                return true; // unprovable: the word could deliver a mutation char
            }
            if !p.starts_with('-') || p.starts_with("--") {
                return false;
            }
            let b = p.as_bytes();
            let mut k = 1;
            while k < b.len() {
                let c = b[k] as char;
                if mutation.contains(&c) {
                    return true;
                }
                if value_taking.contains(&c) {
                    return false; // value-taking flag: rest of token is its value
                }
                k += 1;
            }
            false
        })
}

/// Check if `sed` has an in-place flag in a way that mutates files outside temp.
/// When all file operands after the flag are under temp, returns `false` (allow).
fn has_sed_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);
    // In-place flag position: any single-dash flag containing `i`/`I`
    // (-i, -iSUFFIX, -I, -nix/-Ei clusters; no other sed short flag has
    // i/I), or the GNU long form `--in-place[=SUFFIX]` at any unique
    // abbreviation (`--i` is the shortest; no other GNU sed long option
    // starts with `i`). Over-rejection is intentional (fail-closed):
    // attached `-e`/`-f`/`-l` args containing `i` (e.g. `-e's/x/i/'`) and
    // operands literally named like a flag (`sed -- -info.txt`) are
    // rejected too. Tokens are normalized as the shell delivers them
    // ([`shell_word`]), and `$'...'`/`$"..."` tokens with escapes are
    // unprovable flag candidates ([`is_unprovable_flag_token`]). The flag
    // POSITION fails closed on ANY substitution word — even one provably not
    // `-i` (`sed $(echo foo) file`) — unlike awk's `-i` OPERAND gate, which
    // folds provable echoes (`awk -i $(echo otherlib) file` stays allowed):
    // a substitution here could hide the `-i` itself, and sed's operands are
    // temp-gated after the flag (no proof of read-only otherwise).
    let i_pos = parts.iter().position(|part| {
        let p = shell_word(part);
        (p.starts_with('-') && !p.starts_with("--") && p.contains(['i', 'I']))
            || p.starts_with("--i")
            || is_unprovable_flag_token(part)
            || unprovable_flag_word(part, &p)
    });
    let Some(i_pos) = i_pos else {
        return false; // no in-place flag → not a sed mutation
    };

    // Collect file operands after -i.
    // Skip flags (-n, -e, etc.), sed expressions, and backup-extension tokens.
    // Sed expressions and backup extensions are non-absolute tokens before the
    // first absolute-path file operand. Once we've seen a file operand, all
    // subsequent non-flag tokens are treated as file operands (multi-file).
    // The absolute check normalizes via [`shell_word`], so quoted/escaped
    // paths (e.g. '/etc/passwd') count as operands — which also classifies
    // `/`-addressed scripts (e.g. '/pat/d') as operands, an accepted fail-closed
    // over-rejection matching the unquoted behavior. Unprovable `$'...'`/`$"..."`
    // tokens with escapes ([`is_unprovable_flag_token`]) also count as
    // operands — they may resolve to a workspace path.
    let mut file_operands: Vec<&str> = Vec::new();
    let mut seen_file_operand = false;

    for part in &parts[i_pos + 1..] {
        if part.starts_with('-') {
            continue; // skip flags like -n, -e, -e 'expr'
        }
        if seen_file_operand {
            // Already past the sed expression — this is a file operand
            file_operands.push(part);
            continue;
        }
        // Before the first file operand: skip non-absolute tokens (sed expression
        // like 's/a/b/' or backup extension like .bak for `-i .bak`).
        if Path::new(&shell_word(part)).is_absolute() || is_unprovable_flag_token(part) {
            seen_file_operand = true;
            file_operands.push(part);
        }
        // Non-absolute tokens are sed expressions or backup extensions — skip
    }

    if file_operands.is_empty() {
        return true; // in-place flag without file operand → reject (conservative)
    }

    file_operands.iter().any(|p| writes_outside_temp(p, state))
}

/// Check if `awk -i inplace` is present (GNU awk in-place edit). The flag
/// position must be literal `-i` or a substitution/bare-`$var` word that could
/// deliver it; the operand must be provably `inplace` — a substitution word
/// that could expand to it fails closed, while a body provably echoing
/// something else (`awk -i $(echo otherlib) file`) and a bare `$var` whose
/// fixed prefix cannot start the word (`awk -i lib$suffix file`, the first
/// argv field is `lib`) stay allowed.
fn has_inplace(command: &str, _state: &ValidationState) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);
    parts.windows(2).any(|w| {
        let p0 = shell_word(w[0]);
        let p1 = shell_word(w[1]);
        // Operand: `inplace` literal or provably-folded; a bare `$var` whose
        // literal prefix cannot start the word could still expand to `inplace`
        // (`inpl$ace` → fail closed) while a pinned prefix (`lib$suffix` → the
        // first field is `lib`, never `inplace`) stays allowed.
        let bare_operand = contains_bare_var(w[1]) && {
            let prefix = &p1[..p1.find('$').unwrap_or(p1.len())];
            prefix.is_empty() || "inplace".starts_with(prefix)
        };
        (p0 == "-i" || word_could_form_token_or_bare_var(w[0], "-i"))
            && (p1 == "inplace" || word_could_form_token(w[1], "inplace", "") || bare_operand)
    })
}

/// Resolve the `of=` operand value of `w` when the whole word is provable:
/// shell-normalized (quoted `"of=..."` matches) with provably-echoed spans
/// folded ([`word_parts`] + [`WordPart::Lit`] concatenation) and bare `$var`
/// left for the temp resolver. `None` when `w` cannot be an `of=` operand or
/// its value is unprovable (the caller then fails closed when the word could
/// still form one).
fn dd_of_value(w: &str) -> Option<String> {
    let p = shell_word(w);
    let mut out = String::new();
    for part in word_parts(&p) {
        match part {
            WordPart::Lit(s) => out.push_str(&s),
            WordPart::Any => return None,
        }
    }
    out.strip_prefix("of=").map(str::to_string)
}

/// Check if `dd of=...` writes outside temp. Every `of=` operand is evaluated
/// — no early return on a temp-gated one (GNU dd honors the last):
/// `dd of=/tmp/x $(echo of=/etc/passwd) ...` rejects. A literal or provably-
/// echoed value is temp-gated (`of=$(echo /tmp/x).txt` → `/tmp/x.txt`,
/// `of=$SNAP/x` resolves via the bound variable, `$(echo of=/tmp/y)` folds to
/// the same form); an unprovable value — or any bare `$var` in the command,
/// whose value could field-split an `of=` operand out (`if=$src`) — fails
/// closed. `if=`/`bs=`/`obs=`/`oflag=` operands are not mutation-determining.
fn has_dd_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);
    for w in &parts {
        if let Some(val) = dd_of_value(w) {
            if writes_outside_temp(&val, state) {
                return true;
            }
        } else if word_could_form_token_or_bare_var(w, "of=") {
            return true;
        }
    }
    false
}

/// Curl short flags that take an attached or separate value. Their attached
/// values consume the rest of the token, so `-do` (data "o") and
/// `-HContent-Type:o` are never misread as `-o`, and `-dO` is never misread
/// as `-O`. `-h` (help) takes an optional attached subject (`-ho` is help for
/// "o", not `-o`). `-o` itself is handled by [`has_output_mutation`], not
/// listed here.
const CURL_VALUE_TAKING: &[char] = &[
    'A', 'b', 'c', 'C', 'd', 'D', 'e', 'E', 'F', 'H', 'h', 'K', 'm', 'P', 'Q', 'r', 't', 'T', 'u',
    'U', 'w', 'x', 'X', 'y', 'Y', 'z',
];

/// Output-flag family for one tool: single-dash chars that unconditionally
/// write to CWD (`-O`), the temp-gated `-o` output char, value-taking chars,
/// and long flags that unconditionally write to CWD.
struct OutputFlagSpec {
    always_blocked_short: &'static [char],
    output_short: char,
    value_taking: &'static [char],
    always_blocked_long: &'static [&'static str],
}

const CURL_OUTPUT_FLAGS: OutputFlagSpec = OutputFlagSpec {
    always_blocked_short: &['O'],
    output_short: 'o',
    value_taking: CURL_VALUE_TAKING,
    always_blocked_long: &["--remote-name", "--remote-name-all"],
};

const BASE64_OUTPUT_FLAGS: OutputFlagSpec = OutputFlagSpec {
    always_blocked_short: &[],
    output_short: 'o',
    value_taking: &['i', 'b'],
    always_blocked_long: &[],
};

/// Check if a curl/base64-style command has output flags that write outside
/// temp. Every output flag is evaluated — no early return on a temp-gated one:
/// curl takes the first `-o`/`--output` and adds a CWD write per URL with `-O`,
/// while base64 takes the last `-o`/`--output`, so an outside-temp write
/// anywhere is fail-closed regardless of which flag ultimately wins.
/// `-O`/`--remote-name`/`--remote-name-all` always write to CWD; `-o`/`--output`
/// (space and `=` forms) are temp-gated with attached or next-token values;
/// value-taking chars consume the rest of their token (`-do` is data "o",
/// `-dio` is `-d -i 'o'`). Long flags and `=`-forms match after shell
/// normalization ([`shell_word`]), so quoted spellings (`'-so'`, `--'output'`)
/// are seen; `$'...'`/`$"..."` tokens with escapes are unprovable flag
/// candidates — same fail-closed contract as the sed gate. A substitution
/// word that could form an output flag is rejected UNCONDITIONALLY (the
/// substituted flag's value is unprovable — no sed-style operand
/// temp-gating, contrast [`has_sed_mutation`]).
fn has_output_mutation(command: &str, state: &ValidationState, spec: &OutputFlagSpec) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);

    // A substitution word could deliver an output flag (`-o`, `-O`,
    // `--output`) — its output is unprovable, so fail closed (see
    // [`substitution_could_form_flag`]).
    if parts
        .iter()
        .any(|w| substitution_could_form_flag(w, &shell_word(w)))
    {
        return true;
    }

    for (i, part) in parts.iter().enumerate() {
        let w = shell_word(part);
        if spec.always_blocked_long.contains(&w.as_str()) {
            return true;
        }
        if let Some(path) = w.strip_prefix("--output=") {
            if writes_outside_temp(path, state) {
                return true;
            }
        } else if w == "--output"
            && let Some(next) = parts.get(i + 1)
            && writes_outside_temp(next, state)
        {
            return true;
        }
    }
    // ANSI-C `$'...'`/`$"..."` tokens with escapes are unprovable flag
    // candidates (`$'\x2do'` delivers `-o`) — same fail-closed contract as
    // the sed gate.
    if parts.iter().any(|p| is_unprovable_flag_token(p)) {
        return true;
    }

    // Single-dash forms, standalone or in a combined cluster (`-so`, `-sO`,
    // `-OJ`, `-LO`, `-so/tmp/x`, quoted `'-so'`).
    for (i, part) in parts.iter().enumerate() {
        let p = shell_word(part);
        if !p.starts_with('-') || p.starts_with("--") || p.len() < 2 {
            continue;
        }
        let b = p.as_bytes();
        let mut k = 1;
        while k < b.len() {
            let c = b[k] as char;
            if spec.always_blocked_short.contains(&c) {
                return true;
            }
            if c == spec.output_short {
                let path = if k + 1 < b.len() {
                    &p[k + 1..]
                } else {
                    parts.get(i + 1).copied().unwrap_or("")
                };
                if !path.is_empty() && writes_outside_temp(path, state) {
                    return true;
                }
                break; // -o value consumed (attached or next token)
            }
            if spec.value_taking.contains(&c) {
                break; // value-taking flag: rest of token is its value
            }
            k += 1;
        }
    }

    false
}

/// Check if curl has output flags that write outside temp (see
/// [`has_output_mutation`]). `-o <path>`/`--output <path>` is allowed when the
/// path is under temp; `-O`/`--remote-name`/`--remote-name-all` always write
/// to CWD and are blocked, standalone or in a cluster (`-sO`, `-OJ`, `-LO`).
fn has_curl_mutation(command: &str, state: &ValidationState) -> bool {
    has_output_mutation(command, state, &CURL_OUTPUT_FLAGS)
}

/// Check if `wget` output flags write outside temp.
/// `-O <path>` / `--output-document <path>` / `-P <path>` / `--directory-prefix <path>`
/// are allowed when the path is under temp.
/// Without output flags, wget writes to CWD → always blocked. A substitution
/// word that could form an output flag is rejected unconditionally (unprovable
/// value — same contract as [`has_output_mutation`]).
fn has_wget_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);

    if parts
        .iter()
        .any(|w| substitution_could_form_flag(w, &shell_word(w)))
    {
        return true;
    }

    // -O/--output-document and -P/--directory-prefix (space and = forms):
    // allowed when the path is under temp.
    if let Some(path) = output_flag_value(
        &parts,
        &["-O", "--output-document"],
        Some("--output-document="),
    ) {
        return writes_outside_temp(path, state);
    }
    if let Some(path) = output_flag_value(
        &parts,
        &["-P", "--directory-prefix"],
        Some("--directory-prefix="),
    ) {
        return writes_outside_temp(path, state);
    }

    // No known output flag → wget would write to CWD with URL's filename
    true
}

/// Characters that are non-operation tar flags (format/output modifiers).
/// These can appear alongside the operation flag in combined forms (e.g.
/// `-tvf` combines `t` (list) with `v` (verbose) and `f` (file)).
const TAR_SAFE_CHARS: &[char] = &['v', 'f', 'z', 'j', 'J'];

/// Check if tar is using only `-t`/`--list` (list) mode. Handles combined flags.
///
/// This is the **whitelist** for [`is_tar_mutating`]'s negative detection
/// strategy. Add new safe/list-only operations (e.g. `--diff`/`--compare`)
/// here rather than adding blacklist checks to [`is_tar_mutating`].
fn is_tar_list_only(command: &str) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);
    // Find the operation flag/option
    for part in &parts {
        // --list is always safe
        if *part == "--list" {
            return true;
        }
        if part.starts_with('-') && !part.starts_with("--") {
            // Skip non-operation flags
            if part.len() == 2 && TAR_SAFE_CHARS.contains(&part.chars().nth(1).unwrap()) {
                continue;
            }
            // Check if this contains only 't' (and maybe v/f/z/j/J) as operation flags
            let ops: String = part
                .chars()
                .skip(1) // skip leading '-'
                .filter(|c| !TAR_SAFE_CHARS.contains(c))
                .collect();
            if !ops.is_empty() {
                return ops == "t";
            }
        }
    }
    // No operation flag found — reject (conservative)
    false
}

/// Check if `tar` is in mutating mode (i.e., will extract/create).
///
/// # Negative detection strategy
///
/// Unlike the other [`FLAG_CHECKS`] entries which use **positive detection**
/// (a predicate that returns `true` only when a specific mutating flag is
/// found), this function uses **negative detection**: it returns `true` when
/// the command is *not* explicitly list-only.
///
/// The reason is that `tar` combines short flags into strings (e.g. `-xvf`
/// or `-czf`), making it impractical to enumerate every mutating flag
/// combination. Instead, it's simpler to maintain a whitelist of safe
/// operations in [`is_tar_list_only`] and deny everything else.
///
/// The `!is_tar_list_only()` fallback automatically catches **all** mutating
/// operations — extraction (`-x`), creation (`-c`), appending (`-r`), update
/// (`-u`), deletion (`--delete`), etc. No individual blacklist checks are
/// needed.
///
/// ## For maintainers
///
/// *   **New mutating operations** require no changes — they are already
///     caught by the negative-detection fallback.
/// *   **New safe/list operations** (e.g. `--diff`/`--compare` for comparing
///     archive contents without modifying anything) should be added to
///     [`is_tar_list_only`] to whitelist them.
/// *   Do **not** add positive blacklist checks to this function — they
///     would be dead code, masked by the fallback.
fn is_tar_mutating(command: &str, _state: &ValidationState) -> bool {
    !is_tar_list_only(command)
}

/// Check if `base64` writes output outside temp (see [`has_output_mutation`]).
/// The decode gate is deliberately absent: `-o` writes a file in encode mode
/// too. `-i`/`-b` are value-taking on BSD/macOS (`-dio` is `-d -i 'o'`, not an
/// output flag); GNU base64 has no `-o`/`-i` at all, so no write happens there.
fn has_base64_mutation(command: &str, state: &ValidationState) -> bool {
    has_output_mutation(command, state, &BASE64_OUTPUT_FLAGS)
}

/// Check if `cargo clippy` has `--fix` before `--` (auto-fix; after `--` a
/// `--fix` token is a lint name). shell_word closes quoted spellings; a
/// substitution word that could expand to `--fix` fails closed — ANY
/// unprovable span (an unquoted one field-splits, `-p foo${x:- --fix}` →
/// fields `foo`, `--fix`) or bare `$var` (its value is unquoted and arbitrary,
/// `--message-format=$fmt` can field-split `--fix` out) — while provably-
/// echoed bodies stay allowed (`$(echo foo)`, `--message-format=$(echo json)`).
fn has_clippy_fix(command: &str) -> bool {
    let parts: Vec<&str> = scan::split_words_keeping_substitutions(command);
    let dashdash_pos = parts.iter().position(|p| shell_word(p) == "--");
    parts.iter().enumerate().any(|(i, part)| {
        dashdash_pos.is_none_or(|dd| i < dd)
            && (shell_word(part) == "--fix" || word_could_form_token_or_bare_var(part, "--fix"))
    })
}

/// Check if `cargo fmt` provably runs with `--check` (the read-only proof;
/// `cargo fmt` without it reformats files). The proof is a literal `--check`
/// word — shell-normalized, so quoted forms count — and other words cannot
/// disable it (`cargo fmt --check --emit=$(echo json)` stays check mode).
/// `cargo fmt $(echo --check)` (no literal proof) is not provably read-only
/// and fails closed.
fn has_cargo_fmt_check(command: &str) -> bool {
    scan::split_words_keeping_substitutions(command)
        .iter()
        .any(|p| shell_word(p) == "--check")
}

// ── Git checks ───────────────────────────────────────────────────────────

/// Long-option kinds for the collapsed branch/tag table.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefLongKind {
    /// List-mode trigger — bare positionals become patterns.
    List,
    /// Value-taking option (`--sort`): consumes the next token; NOT a list
    /// trigger (git creates a ref when a bare word follows `--sort <key>`).
    Value,
}

/// Collapsed `git branch`/`git tag` long-option table: the list-mode triggers
/// plus the value-taking `--format`/`--sort` and the tag verify flag. Every other long
/// option is treated as unknown (fail-closed — a bare word after it is a ref
/// name). Unambiguous abbreviations of these rows fire (mirroring git's
/// option prefixing); an abbreviation matching multiple rows is ambiguous →
/// no list/verify/value mode, so a following bare word rejects. When git
/// itself would see the abbreviation as ambiguous (e.g. `--m` → `--merged`
/// vs `--move`) it errors at runtime — nothing executes, so the over-firing
/// list mode is safe.
const GIT_REF_LONG_OPTS: &[(&str, RefLongKind)] = &[
    ("--merged", RefLongKind::List),
    ("--no-merged", RefLongKind::List),
    ("--contains", RefLongKind::List),
    ("--no-contains", RefLongKind::List),
    ("--points-at", RefLongKind::List),
    ("--list", RefLongKind::List),
    ("--format", RefLongKind::Value),
    ("--verify", RefLongKind::List),
    ("--sort", RefLongKind::Value),
];

fn resolve_ref_long(kind: &str, sub: &str) -> Option<RefLongKind> {
    let base = kind.split('=').next().unwrap_or(kind);
    if base.len() < 3 {
        return None; // bare `--`/`-`-length tokens are never options
    }
    // `--verify` is tag-only; on branch it is an unknown option (create-active).
    let mut matches = GIT_REF_LONG_OPTS
        .iter()
        .filter(|(name, _)| name.starts_with(base) && !(sub == "branch" && *name == "--verify"));
    let first = matches.next()?;
    if matches.next().is_some() {
        return None; // ambiguous within the table — fail closed
    }
    Some(first.1)
}

/// Rejection for a git subcommand that mutates (mutation flags, unprovable
/// words, or ref-name positionals).
fn git_mutation_rejection(subcommand: &str) -> String {
    format!(
        "⚠️ Read-only mode: `git {subcommand}` is not allowed — it mutates.\n\
         Suggestion: inspect state with read-only git commands instead \
         (e.g. `git status`, `git log`, `git diff`, `git show`)."
    )
}

/// Check a `git branch`/`git tag` subcommand with the collapsed rule set:
/// list/verify mode from the retained triggers, `--sort` value consumption,
/// and the bare-name rule — a bare positional outside list/verify mode is a
/// ref name being created or modified → reject. Mutation flags need no table:
/// a bare name after them rejects; without a name git errors at runtime.
fn check_git_ref_subcommand(subcommand: &str, sub: &str) -> Result<(), String> {
    let words = scan::split_words_keeping_substitutions(subcommand);
    let mut list = false;
    let mut verify = false;
    let mut saw_name = false;
    let mut consume_next = false;
    for raw in words.iter().skip(1) {
        let w = shell_word(raw);
        // A substitution/bare-`$var` word is unprovable (could deliver a
        // mutation flag or a ref name) — fail closed.
        if unprovable_flag_word(raw, &w) {
            return Err(git_mutation_rejection(subcommand));
        }
        // Name-less mutations: git writes config without a bare ref-name
        // positional (`-u<upstream>`, `--set-upstream-to=<x>`,
        // `--unset-upstream`, `--edit-description`). The bare-name rule below
        // cannot see them, so they reject here. Other mutation flags
        // (`-d`, `--delete`, `-m`, ...) need a ref name — covered by the
        // bare-name rule.
        if w.starts_with('-') && w.len() > 1 {
            let is_name_less_mutation = if w.starts_with("--") {
                [
                    "--set-upstream-to",
                    "--unset-upstream",
                    "--edit-description",
                ]
                .iter()
                .any(|m| w == *m || w.starts_with(&format!("{m}=")))
            } else {
                w[1..].contains('u') // branch `-u<upstream>` (attached value)
            };
            if is_name_less_mutation {
                return Err(git_mutation_rejection(subcommand));
            }
        }
        if consume_next {
            consume_next = false;
            continue; // `--sort` value — never a ref name
        }
        if w == "--" || w == "--end-of-options" {
            saw_name = true; // post-separator tokens are positionals
            continue;
        }
        if let Some(kind) = resolve_ref_long(&w, sub) {
            match kind {
                RefLongKind::List => list = true,
                RefLongKind::Value => {
                    if !w.contains('=') {
                        consume_next = true;
                    }
                }
            }
        } else if w.starts_with('-') && !w.starts_with("--") && w.len() > 1 {
            // Short cluster: `l` → list; `v` (tag only) → verify. An unknown
            // short stops the scan — the rest of the token is its attached
            // value, which could contain `l`/`v` (`-F/file` must not set
            // list mode via the `l` in "/file").
            let b = w.as_bytes();
            let mut k = 1;
            while k < b.len() {
                let c = b[k] as char;
                let list_shorts = if sub == "tag" {
                    GIT_TAG_SHORT_LIST
                } else {
                    GIT_REF_SHORT_LIST
                };
                if list_shorts.contains(&c) {
                    list = true;
                } else if sub == "tag" && c == 'v' {
                    verify = true;
                } else {
                    break;
                }
                k += 1;
            }
        } else if !w.starts_with('-') {
            saw_name = true;
        }
        // Unknown/ambiguous long options (`--abbrev=10`, `--bogus`) are
        // skipped — git errors at runtime; only a FOLLOWING bare word is a
        // ref name (fail-closed, create-active).
    }
    if verify || list || !saw_name {
        return Ok(());
    }
    Err(format!(
        "⚠️ Read-only mode: `git {subcommand}` is not allowed — it names a {sub} ref, which creates or modifies it.\n\
         Suggestion: use `git {sub} --list` or `git {sub} --merged` to list existing {sub}es read-only."
    ))
}

/// For a matched git subcommand, check for mutation verbs across all
/// argument positions (used for `git remote`).
///
/// Bare-word tokens (remote verbs like `add`, `remove`, `prune`) are checked
/// only at the first non-flag argument position; later bare words can be
/// legit names (`git remote show add` — `add` is a remote name).
fn check_git_subcommand_mutation(subcommand: &str, mutation_tokens: &[&str]) -> Result<(), String> {
    let words = scan::split_words_keeping_substitutions(subcommand);
    let (flag_tokens, bare_tokens): (Vec<&str>, Vec<&str>) = mutation_tokens
        .iter()
        .copied()
        .partition(|t| t.starts_with('-'));
    let short_chars: Vec<char> = flag_tokens
        .iter()
        .filter_map(|t| {
            let b = t.as_bytes();
            (b.len() == 2 && b[0] == b'-').then(|| b[1] as char)
        })
        .collect();
    // Flag-based mutation token check (all positions).
    for arg in words.iter().skip(1) {
        let a = shell_word(arg);
        if !a.starts_with('-') {
            continue;
        }
        let is_mutating = if a.starts_with("--") {
            matches_mutation_token(&a, &flag_tokens)
        } else {
            short_chars.iter().any(|c| a[1..].contains(*c))
        };
        if is_mutating {
            return Err(git_mutation_rejection(subcommand));
        }
    }
    // Bare-word mutation verbs checked only at the first non-flag position.
    if !bare_tokens.is_empty()
        && let Some(first_non_flag_arg) = words
            .iter()
            .skip(1)
            .find(|w| !shell_word(w).starts_with('-'))
    {
        let bare = shell_word(first_non_flag_arg);
        let is_mutating = word_has_substitution(first_non_flag_arg)
            || matches_mutation_token(&bare, &bare_tokens);
        if is_mutating {
            return Err(git_mutation_rejection(subcommand));
        }
    }
    Ok(())
}

/// True when `word` is a mutation token or the `token=value` form of one.
fn matches_mutation_token(word: &str, tokens: &[&str]) -> bool {
    tokens.contains(&word)
        || tokens
            .iter()
            .any(|t| word.strip_prefix(t).is_some_and(|r| r.starts_with('=')))
}

/// Extract the full subcommand from a git command's words.
///
/// Skips leading environment variable assignments, the `git` command word,
/// global flags and their values, and collects all remaining words as the
/// subcommand.
fn extract_git_subcommand(segment: &str) -> String {
    let words = scan::split_words_keeping_substitutions(segment);
    let Some(git_idx) = super::find_first_command_word_index(&words) else {
        return String::new();
    };
    let git_word = words[git_idx]
        .rsplit('/')
        .next()
        .expect("rsplit always yields at least one element");
    let git_word = scan::strip_quoted_word(git_word);
    if git_word != "git" {
        return String::new();
    }
    let remaining = &words[git_idx + 1..];
    if let Some(sub_start) = super::find_first_non_flag_index(remaining, true) {
        remaining[sub_start..].join(" ")
    } else {
        String::new()
    }
}

/// Exec-flag blocks applied across the whole command. Git abbreviates
/// unambiguous long-option prefixes, so any token of length > 2 that is a
/// prefix of a blocked flag is rejected (bare `--` is the path separator).
/// Exact `--filter`/`--text` exemptions: `--filter` shadows `--filters`;
/// `--text` is the benign `-a` flag on the diff family + grep
/// ([`GIT_TEXT_BENIGN_SUBCOMMANDS`]) and an exec vector elsewhere.
const GIT_EXEC_LONG_FLAGS: &[&str] = &[
    "--ext-diff",
    "--textconv",
    "--show-signature",
    "--filters",
    "--upload-pack",
    "--exec",
    "--receive-pack",
];

/// Subcommands where git resolves exact `--text` to the benign `-a/--text`
/// flag (diff family, grep, shortlog/rev-list which take diff options).
const GIT_TEXT_BENIGN_SUBCOMMANDS: &[&str] = &[
    "grep",
    "diff",
    "log",
    "show",
    "blame",
    "annotate",
    "diff-files",
    "diff-index",
    "diff-tree",
    "whatchanged",
    "shortlog",
    "rev-list",
    "range-diff",
    "stash",
];

fn check_git_exec_flags(trimmed: &str, base: &str, words: &[&str]) -> Result<(), String> {
    for w in words {
        if w.contains("$'") && w.contains('\\') {
            return reject(
                trimmed,
                "ANSI-C quoted git arguments with backslash escapes cannot be proven safe.",
                "write git flags and arguments literally.",
            );
        }
        let wq = shell_word(w);
        let opt = wq.split('=').next().unwrap_or(&wq);
        if opt.len() <= 2 {
            continue;
        }
        let text_benign = opt == "--text" && GIT_TEXT_BENIGN_SUBCOMMANDS.contains(&base);
        if opt != "--filter"
            && !text_benign
            && GIT_EXEC_LONG_FLAGS.iter().any(|f| f.starts_with(opt))
        {
            return reject(
                trimmed,
                &format!(
                    "`{opt}` is not allowed in read-only mode — it selects a git feature that executes external programs."
                ),
                "drop the flag; the corresponding git feature stays disabled without it.",
            );
        }
    }
    Ok(())
}

/// Subcommand-scoped exec flags. `-O`/`--open-files-in-pager` (grep only)
/// and the help viewers `-w/--web`, `-m/--man`, `-i/--info` exec external
/// programs. Long-option prefixes are blocked; short-flag matches are
/// case-sensitive.
fn check_git_scoped_flags(trimmed: &str, base: &str, words: &[&str]) -> Result<(), String> {
    for w in words {
        let wq = shell_word(w);
        let opt = wq.split('=').next().unwrap_or(&wq);
        let is_scoped = match base {
            "grep" => {
                (opt.len() > 2 && "--open-files-in-pager".starts_with(opt))
                    || (wq.starts_with('-') && !wq.starts_with("--") && wq[1..].contains('O'))
            }
            "help" => {
                (opt.len() > 2
                    && ["--web", "--man", "--info"]
                        .iter()
                        .any(|f| f.starts_with(opt)))
                    || (wq.starts_with('-')
                        && !wq.starts_with("--")
                        && wq[1..].contains(['w', 'm', 'i']))
            }
            _ => false,
        };
        if is_scoped {
            return reject(
                trimmed,
                &format!(
                    "`{wq}` on `git {base}` is not allowed in read-only mode — it runs an external program."
                ),
                "drop the flag.",
            );
        }
    }
    Ok(())
}

/// Reject `--output` on any git subcommand — it writes the command's output
/// to a file. The unprovable-span rejection doubles as the blanket
/// substitution net for the arguments this gate scans. `--output-indicator-*`
/// do not write and are exempt — but only the literal prefix; the fail-closed
/// checks still run on what follows it. Runs before
/// [`check_git_read_only_extensions`], whose read forms would bypass it.
fn check_git_output_flag(trimmed: &str, subcommand: &str) -> Result<(), String> {
    if subcommand.starts_with("config") {
        return Ok(());
    }
    let words = scan::split_words_keeping_substitutions(subcommand);
    let is_remote = words.first().is_some_and(|w| shell_word(w) == "remote");
    if words.iter().any(|w| {
        let p = shell_word(w);
        p == "--output"
            || p.starts_with("--output=")
            || word_could_form_token(w, "--output", "--output-indicator")
            || (!is_remote && contains_bare_var(w))
    }) {
        return reject(
            trimmed,
            "`--output` is not allowed in read-only mode — it writes the git output to a file.",
            "drop the flag; use a shell redirect like `git diff > /tmp/out` to save output to the OS temp directory.",
        );
    }
    Ok(())
}

/// Reject git invocations that can execute programs via flag/config/env
/// channels invisible to the subcommand allowlist. Runs FIRST in
/// [`check_git_segment`] — before the extension allowlist, whose allowlisted
/// forms would otherwise bypass the scan. Fail-closed: repo-redirect globals
/// (`-C`/`--git-dir`/`--work-tree`) are rejected too, so an agent-written
/// hostile repo config cannot be reached via redirect. Accepted residual: a
/// repo with pre-existing hostile config (`.git/config` diff.external/
/// filter.*/core.fsmonitor, .gitattributes textconv, global LFS filters) can
/// still exec drivers from plain `git diff`/`status` — trusted-workspace
/// model, out of scope here.
fn check_git_exec_vectors(trimmed: &str) -> Result<(), String> {
    let words = scan::split_words_keeping_substitutions(trimmed);
    let Some(git_idx) = super::find_first_command_word_index(&words) else {
        return Ok(());
    };
    // Env bindings before `git` — catches `env`/`sudo`-wrapped forms that the
    // binding layer (which only sees current-shell segments) cannot.
    for w in &words[..git_idx] {
        if super::is_env_assignment(w) {
            check_git_env_binding(w)?;
        }
    }
    // Global-region flags between `git` and the subcommand. Position-sensitive:
    // `git log -c` / `git diff -c` (combined diff) live in the subcommand
    // region and stay allowed. `-c` injects config that executes programs;
    // `-C` redirects the repo (attached/clustered forms `-cfoo`, `-pc`, `-pC`
    // included). Long globals redirect the repo or inject config.
    let sub_idx = super::find_first_non_flag_index(&words[git_idx + 1..], true)
        .map_or(words.len(), |i| git_idx + 1 + i);
    let base = words.get(sub_idx).copied().unwrap_or("");
    for w in &words[git_idx + 1..sub_idx] {
        let wq = shell_word(w);
        if wq.starts_with('-') && !wq.starts_with("--") && wq[1..].contains(['c', 'C']) {
            return reject(
                trimmed,
                "`-c`/`-C` git global options are not allowed in read-only mode — they inject config or redirect the repository, both of which can execute programs.",
                "run git without global `-c`/`-C` options (use `cd` to change directory).",
            );
        }
        for opt in [
            "--git-dir",
            "--work-tree",
            "--config-env",
            "--config-file",
            "--exec-path",
        ] {
            if wq == opt || wq.starts_with(&format!("{opt}=")) {
                return reject(
                    trimmed,
                    &format!(
                        "`{opt}` is not allowed in read-only mode — it redirects the repository or injects config, enabling program execution."
                    ),
                    "run git against the workspace repository without repo-redirect or config-injection options.",
                );
            }
        }
    }
    check_git_exec_flags(trimmed, base, &words)?;
    check_git_scoped_flags(trimmed, base, &words)
}

/// Phase-3 read-only git allowlist rules: `stash show`, `config` read forms,
/// `rebase --show-current`, and `submodule status`.
///
/// Returns `Some(result)` when `subcommand` matches one of these prefixes
/// (the caller returns it directly), `None` when the subcommand is outside
/// these rules and should fall through to the general safe-list match.
/// Matching is exact: mutating siblings stay blocked.
fn check_git_read_only_extensions(trimmed: &str, subcommand: &str) -> Option<Result<(), String>> {
    if subcommand.starts_with("stash") {
        let words = scan::split_words_keeping_substitutions(subcommand);
        let stash_cmd = shell_word(words.get(1).copied().unwrap_or(""));
        if matches!(stash_cmd.as_str(), "show" | "list") {
            return Some(Ok(()));
        }
        return Some(reject(
            trimmed,
            "`git stash` is not allowed — it modifies the working tree.",
            "use `git stash list` to view stashes, or `git diff` to preview changes.",
        ));
    }
    if subcommand.starts_with("config") {
        let words = scan::split_words_keeping_substitutions(subcommand);
        let rest = &words[1..];
        if rest.iter().any(|w| {
            let w = shell_word(w);
            matches!(
                w.as_str(),
                "--list" | "-l" | "--get" | "--get-all" | "--get-regexp" | "--name-only"
            )
        }) {
            return Some(Ok(()));
        }
        if rest.iter().any(|w| word_has_substitution(w)) {
            return Some(reject(
                trimmed,
                "`git config` with a substitution cannot be proven read-only.",
                "write git config arguments literally.",
            ));
        }
        if rest.iter().any(|w| {
            let w = shell_word(w);
            matches!(
                w.as_str(),
                "--add"
                    | "--unset"
                    | "--unset-all"
                    | "--edit"
                    | "--remove-section"
                    | "--rename-section"
                    | "--replace-all"
            )
        }) || has_cluster_char(&rest.join(" "), &['e'], &['f'])
        {
            return Some(reject(
                trimmed,
                "`git config` write/edit forms are not allowed — they modify repository or global config.",
                "use `git config user.name` (key read), `git config --list`, or `git config --get <key>` to inspect configuration.",
            ));
        }
        match rest.iter().filter(|w| !w.starts_with('-')).count() {
            0 | 1 => Some(Ok(())),
            _ => Some(reject(
                trimmed,
                "`git config` with a value is not allowed — it writes configuration.",
                "use `git config user.name` (key read) or `git config --list` to inspect configuration.",
            )),
        }
    } else if subcommand.starts_with("rebase") {
        let words = scan::split_words_keeping_substitutions(subcommand);
        if words.len() >= 2 && shell_word(words[1]) == "--show-current" {
            return Some(Ok(()));
        }
        Some(reject(
            trimmed,
            "`git rebase` is not allowed — it rewrites branch history.",
            "use `git rebase --show-current` to see the in-progress rebase, or `git log` to inspect history.",
        ))
    } else if subcommand.starts_with("submodule") {
        let words = scan::split_words_keeping_substitutions(subcommand);
        match words.get(1) {
            None => Some(Ok(())),
            Some(sub) if shell_word(sub) == "status" => Some(Ok(())),
            Some(sub) => Some(reject(
                trimmed,
                &format!("`git submodule {sub}` is not allowed — it modifies submodules."),
                "use `git submodule status` to inspect submodule state.",
            )),
        }
    } else {
        None
    }
}

fn check_git_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();
    let subcommand = extract_git_subcommand(trimmed);

    // Exec-vector scan FIRST — before the extension allowlist and the
    // empty-subcommand early return (`git --git-dir=/tmp/evil` bare still
    // rejects the repo redirect).
    check_git_exec_vectors(trimmed)?;

    if subcommand.is_empty() || subcommand == "git" {
        return Ok(());
    }

    // `--output` file-write vector — before the extension allowlist (stash read forms bypass it).
    check_git_output_flag(trimmed, &subcommand)?;

    // Phase-3 read-only allowlist rules (stash/config/rebase/submodule).
    if let Some(result) = check_git_read_only_extensions(trimmed, &subcommand) {
        return result;
    }

    // git subcommands that always write — rejected regardless of flags
    // (prefix-matched). push/clean are unconditional rejects here.
    for &(prefix, why, suggestion) in GIT_ALWAYS_MUTATE {
        if subcommand.starts_with(prefix) {
            return reject(trimmed, why, suggestion);
        }
    }

    let matched_safe = GIT_SAFE_SUBCOMMANDS
        .iter()
        .copied()
        .find(|safe| subcommand == *safe || subcommand.starts_with(&format!("{safe} ")));

    if matched_safe.is_none() {
        return Err(format!(
            "⚠️ Read-only mode: the `git {subcommand}` subcommand is not allowed — it may mutate the repository.\n\
             Command: `{trimmed}`\n\
             Allowed git subcommands for read-only mode: status, log, diff, show, blame, branch, tag, remote,\n\
             stash list/show, show-ref, ls-remote, submodule status, config reads, rebase --show-current,\n\
             and other inspection-only commands. Suggestion: use these for repository exploration."
        ));
    }

    // Additional mutation checks for branch/tag/remote/hash-object/reflog/fsck.
    match matched_safe {
        Some("branch") => check_git_ref_subcommand(&subcommand, "branch")?,
        Some("tag") => check_git_ref_subcommand(&subcommand, "tag")?,
        Some("remote") => {
            check_git_subcommand_mutation(&subcommand, GIT_REMOTE_MUTATIONS)?;
        }
        Some("hash-object") => {
            // git hash-object -w writes the object to the database; without -w
            // it only computes and outputs the hash (read-only). The -w flag
            // can appear anywhere, alone or in a combined cluster (`-wt blob
            // <file>`). `-t` is value-taking and consumes the rest of its
            // token (`-tw` is `-t w`, not `-w`).
            if has_cluster_char(trimmed, &['w'], &['t']) {
                return reject(
                    trimmed,
                    "`git hash-object` with `-w` is not allowed — it writes objects to the object database.",
                    "use `git hash-object` without `-w` to compute the hash without storing the object.",
                );
            }
        }
        Some("reflog") => {
            // expire/delete mutate the reflog; show/list/exists are read-only.
            let words = scan::split_words_keeping_substitutions(&subcommand);
            if let Some(raw) = words.get(1) {
                let reflog_sub = shell_word(raw);
                if word_has_unprovable_expansion(raw)
                    || reflog_sub == "expire"
                    || reflog_sub == "delete"
                {
                    return reject(
                        trimmed,
                        &format!("`git reflog {raw}` is not allowed — it modifies the reflog."),
                        "use `git reflog show` or bare `git reflog` to view reflog entries.",
                    );
                }
            }
        }
        // `git fsck --lost-found` writes dangling objects into `.git/lost-found/`.
        Some("fsck")
            if scan::split_words_keeping_substitutions(&subcommand)
                .iter()
                .any(|raw| {
                    shell_word(raw).starts_with("--l") || word_has_unprovable_expansion(raw)
                }) =>
        {
            return reject(
                trimmed,
                "`git fsck --lost-found` is not allowed — it writes dangling objects to `.git/lost-found/`.",
                "drop the flag; dangling objects are still listed on stdout without `--lost-found`.",
            );
        }
        _ => {}
    }

    Ok(())
}

// ── Cargo checks ─────────────────────────────────────────────────────────

fn check_cargo_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();
    let canonical = super::canonical_command(trimmed);
    let subcommand = canonical.strip_prefix("cargo ").unwrap_or(&canonical);

    if subcommand.is_empty() || subcommand == "cargo" {
        return Ok(());
    }

    let base = scan::split_words_keeping_substitutions(subcommand)
        .first()
        .copied()
        .unwrap_or("");

    // ── Help/version exemption ─────────────────────────────────────
    // `-h`/`--help`/`-V`/`--version` appearing as a standalone token BEFORE a
    // `--` separator is a pure read and is allowed for ANY cargo subcommand,
    // including `run`. Tokens after `--` stay blocked (`cargo run -- --help`).
    let words = scan::split_words_keeping_substitutions(trimmed);
    let help_version_before_ddash = words
        .iter()
        .take_while(|w| **w != "--")
        .any(|w| matches!(*w, "-h" | "--help" | "-V" | "--version"));
    if help_version_before_ddash {
        return Ok(());
    }

    // ── Specific rejection messages for subcommands that modify source files ──
    match base {
        "update" => {
            if words.contains(&"--dry-run") {
                return Ok(());
            }
            return reject(
                trimmed,
                "`cargo update` is not allowed — it modifies Cargo.lock.",
                "use `cargo update --dry-run` to preview what would change without modifying anything; \
                 if the real update is required, state that in your final response — it is outside read-only scope.",
            );
        }
        "generate-lockfile" => {
            return reject(
                trimmed,
                "`cargo generate-lockfile` is not allowed — it creates or overwrites Cargo.lock.",
                "generating a lockfile is outside read-only scope; if it is required, state that in your final response.",
            );
        }
        "run" => {
            return reject(
                trimmed,
                "`cargo run` is not allowed — it may write files.",
                "use `cargo run --help` or `cargo run -h` to inspect the program's CLI without running it; \
                 if running the program is actually required, state that in your final response — it is outside read-only scope.",
            );
        }
        _ => {}
    }

    let is_safe = CARGO_SAFE_SUBCOMMANDS.contains(&base);

    if !is_safe {
        return Err(format!(
            "⚠️ Read-only mode: `cargo {base}` is not in the allowed cargo subcommands list.\n\
             Command: `{trimmed}`\n\
             Allowed cargo subcommands: {}\n\
             Suggestion: use `cargo check`, `cargo test`, `cargo clippy`, `cargo doc`, etc.",
            CARGO_SAFE_SUBCOMMANDS.join(", ")
        ));
    }

    // cargo clippy --fix rejection (only when --fix appears BEFORE --)
    if base == "clippy" && has_clippy_fix(trimmed) {
        return Err(format!(
            "⚠️ Read-only mode: `cargo clippy --fix` is not allowed — it auto-applies fixes.\n\
             Command: `{trimmed}`\n\
             Suggestion: use `cargo clippy` without `--fix` to see warnings only,\n\
             or use `cargo clippy -- --fix` to pass `--fix` as a lint name (not auto-fix)."
        ));
    }

    // cargo fmt without --check rejection
    if base == "fmt" && !has_cargo_fmt_check(trimmed) {
        return reject(
            trimmed,
            "`cargo fmt` without `--check` is not allowed — it reformats files.",
            "use `cargo fmt --check` to verify formatting without modifying files.",
        );
    }

    Ok(())
}

// ── Flag detection helpers ───────────────────────────────────────────────

/// Return the value of a flag that takes a separate word as its argument.
fn flag_value<'a>(parts: &'a [&'a str], flag: &str) -> Option<&'a str> {
    parts.windows(2).find_map(|w| {
        if w[0] == flag {
            let val = w[1];
            if val.starts_with('-') && val != "-" {
                None
            } else {
                Some(val)
            }
        } else {
            None
        }
    })
}

/// Return the value of a flag using `=` syntax (e.g., `--output=/tmp/file`, `of=/tmp/out`).
fn flag_value_equals<'a>(parts: &'a [&'a str], prefix: &str) -> Option<&'a str> {
    parts.iter().find_map(|p| p.strip_prefix(prefix))
}

/// Value of the first of `flags` found in space-separated form, falling back
/// to the `=` form of `equals_prefix` when set.
fn output_flag_value<'a>(
    parts: &'a [&'a str],
    flags: &[&str],
    equals_prefix: Option<&str>,
) -> Option<&'a str> {
    flags
        .iter()
        .find_map(|f| flag_value(parts, f))
        .or_else(|| equals_prefix.and_then(|p| flag_value_equals(parts, p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::shell::{NON_DELEGATING_PREFIXES, SHELL_PREFIXES};

    /// Hermetic test context.
    ///
    /// The workspace root is a fixed absolute path that is NOT under any OS
    /// temp root (the "hermetic-test trap": a fixture workspace created under
    /// `tempfile::tempdir()` lies inside the allowed temp roots, making
    /// 'workspace write blocked' assertions impossible). Relative paths
    /// resolve against this root and are therefore rejected as workspace
    /// writes, matching production semantics where the workspace lives outside
    /// the OS temp dirs.
    fn test_ctx() -> CheckContext {
        CheckContext {
            workspace_root: std::path::PathBuf::from("/__mahbot_readonly_test_ws__"),
            temp_roots: crate::tools::path::allowed_temp_roots(),
            // Match the session shell env (TMPDIR = super::TMPDIR_BASELINE); TMP/TEMP unset.
            temp_vars: vec![(
                "TMPDIR".to_string(),
                super::super::TMPDIR_BASELINE.to_string(),
            )],
        }
    }

    fn ok(cmd: &str) {
        let ctx = test_ctx();
        assert!(
            check_command(cmd, &ctx).is_ok(),
            "expected ALLOW but got REJECT for: `{cmd}`"
        );
    }

    fn assert_rejected(cmd: &str) {
        let ctx = test_ctx();
        assert!(
            check_command(cmd, &ctx).is_err(),
            "expected REJECT but got ALLOW for: `{cmd}`"
        );
    }

    /// Assert each case in a table-driven test.
    fn run_cases(cases: &[(&str, bool)]) {
        for &(command, allowed) in cases {
            if allowed {
                ok(command);
            } else {
                assert_rejected(command);
            }
        }
    }

    /// Assert all items in `items` are rejected when formatted with `template`.
    fn assert_all_rejected(items: &[&str], template: impl Fn(&str) -> String) {
        for &item in items {
            assert_rejected(&template(item));
        }
    }

    /// Assert all items in `items` are allowed when formatted with `template`.
    fn assert_all_allowed(items: &[&str], template: impl Fn(&str) -> String) {
        for &item in items {
            ok(&template(item));
        }
    }

    // ── Empty / whitespace ──────────────────────────────────────────

    #[test]
    fn empty_whitespace_and_unknown() {
        let cases = [
            ("", true),
            ("   ", true),
            ("some_obscure_tool --flag", true),
        ];

        run_cases(&cases);
    }

    // ── Git allowlist ──────────────────────────────────────────────

    /// Tests that ALL entries in the production [`GIT_SAFE_SUBCOMMANDS`] constant
    /// are accepted. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_git_safe_subcommands_allowed() {
        assert_all_allowed(GIT_SAFE_SUBCOMMANDS, |subcmd| format!("git {subcmd}"));
    }

    #[test]
    fn git_individual_commands() {
        let cases = [
            ("git commit -m test", false),
            ("git push", false),
            ("git stash", false),
            ("git stash list", true),
            ("git merge feature", false),
            ("git rebase main", false),
            // branch/tag name-creation bypass (regression tests)
            ("git branch my-feature", false),
            ("git branch --force my-feature", false),
            ("git branch -f my-feature", false),
            ("git tag v1.0", false),
            ("git tag -f v1.0", false),
            ("git tag --force v1.0", false),
            ("git branch -u origin/main", false),
            ("git branch --set-upstream-to=origin/main", false),
            // branch mutation flags as first arg (bypasses name-creation check)
            ("git branch --track feature", false),
            ("git branch --no-track feature", false),
            ("git branch -t", true), // git lists with no name — read-only
            ("git branch --unset-upstream", false),
            // tag message flag bypasses
            ("git tag -m msg v1.0", false),
            ("git tag --message msg v1.0", false),
            ("git tag -F file v1.0", false),
            ("git tag --file file v1.0", false),
            ("git tag -e v1.0", false),
            ("git tag --edit v1.0", false),
            // remote inspection commands
            ("git remote", true),
            ("git remote -v", true),
            ("git remote show origin", true),
            ("git remote get-url origin", true),
            // mktag always mutates
            ("git mktag <tag_object", false),
            ("git mktag", false),
            // mktree always mutates
            ("git mktree <tree_contents", false),
            ("git mktree", false),
            // merge-file always mutates (removed from safe list)
            ("git merge-file a.txt b.txt c.txt", false),
            ("git merge-file -p a.txt b.txt c.txt", false),
            // merge-tree always mutates (removed from safe list)
            ("git merge-tree base branch1 branch2", false),
            ("git merge-tree --write-tree base branch1 branch2", false),
            // hash-object: read-only without -w, mutating with -w
            ("git hash-object file.txt", true),
            ("git hash-object --stdin", true),
            ("git hash-object -w file.txt", false),
            ("git hash-object -w --stdin", false),
            ("git hash-object file.txt -w", false),
            // reflog: read-only commands allowed, expire/delete rejected
            ("git reflog", true),
            ("git reflog show", true),
            ("git reflog list", true),
            ("git reflog show HEAD", true),
            ("git reflog expire --all", false),
            ("git reflog delete HEAD@{0}", false),
            // fsck: --lost-found writes dangling objects into .git/lost-found/
            // (git abbreviates long options, so --lost/--l trigger it too;
            // quotes are stripped by shell_word);
            // --no-lost-found and the read-only flags stay allowed
            ("git fsck", true),
            ("git fsck --strict --full", true),
            ("git fsck --no-lost-found", true),
            ("git fsck --lost-found", false),
            ("git fsck \"--lost-found\"", false),
            ("git fsck --lost", false),
            ("git fsck --l", false),
            // --output file-write vector (diff/log family writes the file;
            // blame-class truncates the target to 0 bytes; dash values are
            // valid filenames)
            ("git diff --output=/tmp/x", false),
            ("git show --output /tmp/x", false),
            ("git log -p --output=/tmp/x", false),
            ("git blame --output=src/lib.rs", false),
            ("git diff-tree --output=/tmp/x", false),
            ("git diff --output -1", false),
        ];

        run_cases(&cases);
    }

    // ── Git --bare flag (regression: was skipped as a git global flag) ─

    #[test]
    fn git_bare_flag() {
        let cases = [
            ("git --bare status", true),
            ("git --bare log --oneline", true),
            ("git --bare diff", true),
            ("git --bare push", false),
            ("git --bare commit -m test", false),
            ("git --bare reset --hard", false),
        ];

        run_cases(&cases);
    }

    // ── Cargo allowlist ────────────────────────────────────────────

    /// Tests that ALL entries in the production [`CARGO_SAFE_SUBCOMMANDS`] constant
    /// (except `"fmt"`, which requires `--check`) are accepted.
    /// Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_cargo_safe_subcommands_allowed() {
        for subcmd in CARGO_SAFE_SUBCOMMANDS {
            if *subcmd == "fmt" {
                continue; // requires --check flag — tested via cargo_individual_commands
            }
            ok(&format!("cargo {subcmd}"));
        }
    }

    #[test]
    fn cargo_individual_commands() {
        let cases = [
            ("cargo clippy --fix", false),
            ("cargo clippy -- --fix", true),
            ("cargo fmt", false),
            ("cargo fmt --check", true),
            ("cargo fmt -- --check", true),
            ("cargo fix", false),
            // cargo update and generate-lockfile are rejected with tailored messages
            ("cargo update", false),
            ("cargo update --dry-run", true), // dry-run preview — read-only
            ("cargo generate-lockfile", false),
        ];

        run_cases(&cases);
    }

    /// Combined single-dash flag clusters: mutation chars inside a cluster
    /// (`git branch -df`, `-mv`, `git tag -am`, `git hash-object -wt blob`,
    /// `git config -e`) are detected, while read-only clusters (`-vv`, `-a`,
    /// `-n1`) and quoted spellings stay correctly classified.
    #[test]
    fn git_combined_short_flag_clusters() {
        let cases = [
            // branch: delete/force clusters
            ("git branch -df feature", false),
            ("git branch -fd feature", false),
            ("git branch -Dd feature", false),
            ("git branch '-df' feature", false),
            ("git branch -v '-df' feature", false), // quoted cluster after a safe flag
            // branch: move/copy/force clusters
            ("git branch -mv old new", false),
            ("git branch -cv old new", false),
            ("git branch -fv main origin/main", false),
            // branch: attached -u upstream value
            ("git branch -uorigin/main", false),
            // tag: annotate/force/message clusters
            ("git tag -am 'msg' v1.0", false),
            ("git tag -afm 'msg' v1.0", false),
            ("git tag -fam 'msg' v1.0", false),
            ("git tag -ma v1.0", false),
            ("git tag -mprobe-msg v1.0", false),
            ("git tag -F/file v1.0", false),
            ("git tag '-am' 'msg' v1.0", false),
            ("git tag -n '-am' 'msg' v1.0", true), // git usage-errors — safe; mutation shorts aren't tracked in the collapsed table
            // hash-object: -w inside a cluster with the -t type flag
            ("git hash-object -wt blob a.txt", false),
            ("git hash-object -wtblob a.txt", false),
            ("git hash-object '-wt' blob a.txt", false),
            ("git hash-object -w -t blob a.txt", false),
            // config: -e edit short form (single and quoted)
            ("git config -e", false),
            ("git config '-e'", false),
            ("git config -f /tmp/cfg -e", false),
            // config: quoted long write forms (shell-stripped quotes deliver
            // --edit/--unset); quoted read forms stay allowed
            ("git config --'edit'", false),
            ("git config --'unset' key", false),
            ("git config --'list'", true),
            // read-only clusters stay allowed
            ("git branch -v", true),
            ("git branch -vv", true),
            ("git branch -a", true),
            ("git branch -r", true),
            ("git tag -n", true),
            ("git tag -n1", true),
            ("git hash-object -t blob a.txt", true),
            ("git hash-object --stdin", true),
            ("git config -l", true),
        ];

        run_cases(&cases);
    }

    // ── Unconditional rejections ──────────────────────────────────

    /// Tests that ALL entries in the production [`MUTATING_COMMANDS`] constant
    /// are rejected. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_mutating_commands_rejected() {
        // Use an absolute non-temp path to prevent accidental test drift
        // if items are later moved to TEMP_MUTATORS.
        assert_all_rejected(MUTATING_COMMANDS, |cmd| format!("{cmd} /etc/blocked_test"));
    }

    /// Tests that all git remote mutation verbs are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_remote_mutation_verbs_rejected() {
        assert_all_rejected(GIT_REMOTE_MUTATIONS, |verb| {
            format!("git remote {verb} origin")
        });
    }

    /// Tests that safe branch/tag listing commands are accepted (non-regression
    /// for the name-creation bypass fix — these must NOT be blocked).
    #[test]
    fn git_branch_tag_safe_listing() {
        let cases = [
            // branch: no args (list local branches)
            ("git branch", true),
            // branch: standard listing flags
            ("git branch --list", true),
            ("git branch --list feature", true),
            ("git branch -l", true),
            ("git branch -l feature", true),
            ("git branch --merged", true),
            ("git branch --merged main", true),
            ("git branch --no-merged", true),
            ("git branch --no-merged main", true),
            ("git branch -a", true),
            ("git branch -r", true),
            ("git branch -v", true),
            ("git branch -vv", true),
            ("git branch --sort=-committerdate", true),
            ("git branch --contains abc123", true),
            ("git branch --points-at abc123", true),
            ("git branch --format='%(refname)'", true),
            // branch: value-consuming space forms (value consumed, no bare name)
            ("git branch --format %(refname)", false), // parser gap (unquoted `%(`) — fail-closed
            ("git branch --sort committerdate", true),
            // branch/tag: quoted multi-word values stay one consumed token
            ("git branch --format \"%(refname) %(objectname)\"", true),
            ("git branch --format=\"%(refname) %(objectname)\"", true),
            ("git tag --format \"%(refname) %(objectname)\"", true),
            ("git tag --format=\"%(refname) %(objectname)\"", true),
            // branch: optional-value `=`-forms (read-only without a name)
            ("git branch --abbrev=10", true),
            ("git branch --color=always", true),
            ("git branch --column=dense", true),
            // branch: unambiguous long-option abbreviations of list filters
            ("git branch --mer main", true),
            ("git branch --con HEAD", true),
            ("git branch --no-contains HEAD", true),
            ("git branch --no-merged main", true),
            // branch: --list after a bare word turns it into a pattern (loosening)
            ("git branch feature --list", true),
            // branch: listing-only actions without a name
            ("git branch --show-current", true),
            // tag: no args (list tags)
            ("git tag", true),
            // tag: standard listing flags
            ("git tag --list", true),
            ("git tag -l", true),
            ("git tag -l v1*", true),
            ("git tag --contains abc123", true),
            ("git tag --merged main", true),
            ("git tag --points-at abc123", true),
            ("git tag -n", true),
            ("git tag -v v1.0", true), // verify mode: positionals are verify targets
            ("git tag --verify v1.0", true),
            ("git tag -n 1", true), // -n optional value NOT consumed → pattern
            ("git tag feature --list", true), // list after bare word → pattern
        ];

        run_cases(&cases);
    }

    /// Tests that every branch/tag name-creation and mutation vector from the
    /// git 2.50.1 tables is rejected: optional-value space forms, separators,
    /// unambiguous long abbreviations, consumed-token ordering, and negated
    /// flags that leave a bare ref name.
    #[test]
    fn git_branch_tag_creation_bypass_rejected() {
        let cases = [
            // optional-value space forms leave a bare ref name → create
            ("git branch --abbrev 10", false),
            ("git branch --color always", false),
            ("git branch --column dense", false),
            // separators end option parsing — a bare name still creates
            ("git branch -- foo", false),
            ("git branch --end-of-options foo", false),
            ("git tag --end-of-options foo", false),
            // negated flags are create-active (no value, no list mode)
            ("git branch --no-list foo", false),
            ("git branch --no-verbose foo", false),
            ("git branch --no-delete foo", false),
            ("git branch --no-move foo", false),
            ("git branch --no-force foo", false),
            ("git branch --no-points-at foo", false),
            ("git tag --no-annotate foo", false),
            ("git tag --no-force foo", false),
            // unambiguous long-option abbreviations of mutation flags
            ("git branch --forc foo", false),
            ("git branch --tr foo", false),
            ("git branch --uns foo", false),
            ("git branch --mov foo", false),
            ("git branch --del foo", false),
            ("git branch --edit-d foo", false),
            ("git tag --mes foo", false),
            // unknown long/short flags fail closed → bare word is a name
            ("git branch --mes foo", false),
            ("git branch --bogus foo", false),
            // consumed tokens never trigger list mode
            ("git branch --sort --merged foo", false),
            ("git branch --sort --list foo", false),
            ("git branch --format %(refname) foo", false),
            ("git tag --sort --list foo", false),
            // quoted value consumed; a trailing bare word is still a name
            (
                "git branch --format \"%(refname) %(objectname)\" foo",
                false,
            ),
            ("git tag --format \"%(refname) %(objectname)\" foo", false),
            // mutation tokens still reject when git consumes them as values
            ("git branch --merged -d", true), // git errors at runtime — safe
            ("git branch --sort -d foo", false),
            // `-a`/`-r` are list-only: a bare name after them is rejected
            ("git branch -a foo", false),
            ("git branch -r foo", false),
            // tag: create-active flags without list mode
            ("git tag -i foo", false),
            ("git tag --column foo", false),
            ("git tag --sort=key foo", false),
        ];

        run_cases(&cases);
    }

    /// Tests that mutation flags in later argument positions are correctly caught,
    /// and that bare-word remote verbs in position 2+ after a safe sub-subcommand
    /// are correctly allowed (no false positives).
    #[test]
    fn git_mutation_in_later_position() {
        let cases = [
            // branch: mutation flag -d in position 2 (after a safe flag)
            ("git branch --sort=-committerdate -d", true), // git errors at runtime — safe
            // branch: mutation flag --delete in position 2 (after a safe flag)
            ("git branch --sort=-committerdate --delete feature", false),
            // tag: mutation flag --delete in position 2 (after -l and pattern)
            ("git tag -l v1.* --delete", true), // git errors (list + delete conflict) — safe
            // tag: mutation flag -d in position 2
            ("git tag -l v1.* -d", true), // git errors (list + delete conflict) — safe
            // remote: bare-word "add" in position 2 after safe sub-subcommand
            // should be ALLOWED (it's a remote name, not a mutation verb)
            ("git remote show add", true),
            // remote: bare-word verb "update" after a flag should be REJECTED
            // (first non-flag argument)
            ("git remote -v update", false),
        ];

        run_cases(&cases);
    }

    // ── Flag-dependent tests ──────────────────────────────────────

    #[test]
    fn flag_dependent_tests() {
        let cases = [
            // sed
            ("sed 's/a/b/' file", true),
            ("sed -i 's/a/b/' file", false),
            ("sed -i.bak 's/a/b/' file", false),
            // awk
            ("awk '{print $1}' file", true),
            ("awk -i inplace '{print $1}' file", false),
            // dd
            ("dd if=/dev/zero bs=1 count=10", true),
            ("dd if=/dev/zero of=file bs=1 count=10", false),
            // curl
            ("curl https://example.com", true),
            ("curl -o file https://example.com", false),
            ("curl -O https://example.com/file", false),
            // curl combined short-flag clusters
            ("curl -so out URL", false),
            ("curl -sO URL", false),
            ("curl -OJ URL", false),
            ("curl -LO URL", false),
            ("curl '-so' out URL", false),
            ("curl -so/tmp/x URL", true),
            ("curl -do out URL", true), // -d consumes 'o' (data), not output
            ("curl -HContent-Type:o URL", true), // -H value contains 'o'
            ("curl -ho URL", true),     // -h help subject 'o', not output
            ("curl $'\\x2do' /etc/passwd URL", false), // unprovable → -o delivered
            // curl: every output flag evaluated — -O/--remote-name* write to
            // CWD even after a temp-gated -o/--output (no early return), and a
            // later outside-temp flag rejects even when curl would first-wins
            // (fail-closed over-rejection)
            ("curl -O --output /tmp/x URL", false),
            ("curl -o out --output /tmp/x URL", false),
            ("curl -o /tmp/x --output out URL", false),
            ("curl -o /tmp/x --output /tmp/y URL", true),
            ("curl --output=/tmp/x URL", true),
            ("curl --output=out URL", false),
            ("curl --'output' out URL", false), // quoted long form
            ("curl --remote-name-all URL", false), // per-URL CWD writes
            // tar
            ("tar -tf archive.tar.gz", true),
            ("tar -xzf archive.tar.gz", false),
            ("tar -czf archive.tar.gz dir/", false),
            ("tar --list -f archive.tar.gz", true),
            // base64
            ("base64 -d file.txt", true),
            ("base64 -d -o out.bin file.txt", false),
            ("base64 --decode --output out.bin file.txt", false),
            // base64 combined short-flag clusters
            ("base64 -do out", false),
            ("base64 '-do' out", false),
            ("base64 -do/tmp/x", true),
            ("base64 -o out file", false), // encode-mode output write
            ("base64 -dio out", true),     // -i consumes 'o' (input file) on BSD
            ("base64 $'\\x2do' /etc/passwd", false), // unprovable → -o delivered
            // base64: last output flag wins — any outside-temp write rejects
            // (also covers the --output= form)
            ("base64 --output /tmp/x -o out file", false),
            ("base64 -o out --output /tmp/x file", false), // fail-closed over-rejection
            ("base64 -o /tmp/x --output /tmp/y file", true),
            ("base64 --output=/tmp/x file", true),
            ("base64 --output=out file", false),
        ];

        run_cases(&cases);
    }

    // ── Chained commands ───────────────────────────────────────────

    #[test]
    fn chained_commands() {
        let cases = [
            ("cargo check && cargo test", true),
            ("cargo check && rm file", false),
            ("git status && cargo fmt", false),
            ("git log --oneline | head -20", true),
            ("cargo check; rm file", false),
        ];

        run_cases(&cases);
    }

    // ── Redirect tests ─────────────────────────────────────────────

    #[test]
    fn redirect_tests() {
        let cases = [
            // Original redirect tests
            ("echo hello > file.txt", false),
            ("echo hello > /dev/null", true),
            ("echo hello > /tmp/output.txt", true),
            ("cmd 2>&1", true),
            ("echo \"hello > world\"", true),
            ("echo hello >> /tmp/log", true),
            ("echo hello >| /tmp/force", true),
            ("cargo build > /dev/null 2>&1", true),
            // /var/tmp redirect tests
            ("echo hello > /var/tmp/output.txt", true),
            ("echo hello >> /var/tmp/log", true),
            // Redirect operators refactor tests
            ("cmd > output.txt", false),
            ("cmd 1>&2", true),
            ("cmd 2> /tmp/errors.log", true),
            ("cmd 2> errors.log", false),
            ("cmd >&2", true), // fd-dup — approved allow
            ("echo \\> /tmp/file", true),
            ("echo \\>", true),
            ("echo \\\\\\> file", true),
            ("echo \"> /tmp/foo", false), // unterminated quote — parser error, fail-closed
            ("echo '> /tmp/foo", false),  // unterminated quote — parser error, fail-closed
        ];

        run_cases(&cases);
    }

    // ── New-walker minimal suites ──────────────────────────────────

    /// Parse errors and grammar-gap shapes fail closed (accepted over-
    /// rejections from the parser-gap list: herestrings with redirects,
    /// `<>`, multiple heredocs per line, unquoted `%(` in git --format,
    /// unterminated quotes/heredocs, malformed substitutions).
    #[test]
    fn parse_error_fail_closed() {
        let cases = [
            ("cat > /tmp/x <<< hi", false),
            ("cat <> f", false),
            ("cat <<A <<B\nx\nA\ny\nB", false),
            ("git branch --format %(refname)", false),
            ("echo \"unterminated", false),
            ("cat <<EOF > /tmp/out", false), // unterminated heredoc
            ("echo $(cmd 2>/dev/null} done", false), // malformed substitution
            ("cat <<", false),               // dangling heredoc marker
            ("echo hi ;;", false),           // stray terminator
        ];
        run_cases(&cases);
    }

    /// Heredoc marker-line tails execute and must be validated (the marker-
    /// line swallow family): `cat <<EOF | rm f` runs rm; `<<EOF > f` is a
    /// real redirect; `<<EOF && cmd` runs cmd. Glued metachars
    /// (`<<EOF| rm f`, `<<EOF>file`) reject via the marker-text guard.
    #[test]
    fn heredoc_marker_line_guard() {
        let cases = [
            ("cat <<EOF | rm f\nbody\nEOF", false),
            ("cat <<EOF && touch f\nbody\nEOF", false),
            ("cat <<EOF && echo hi\nbody\nEOF", true),
            ("cat <<EOF | grep x\nbody\nEOF", true),
            ("cat <<EOF > /tmp/out\nbody\nEOF", true),
            ("cat <<EOF > workspace_file\nbody\nEOF", false),
            ("cat <<EOF| rm f\nbody\nEOF", false),
            ("cat <<EOF>file\nbody\nEOF", false),
            ("cat <<EOF && cd /tmp && touch f\nbody\nEOF", true),
        ];
        run_cases(&cases);
    }

    /// Substitutions in declaration/unset children execute and must be
    /// validated — `unset $(rm -rf /)`, `export x=$(rm -rf /)`, quoted and
    /// array forms, and process substitutions all reject. Bash expands ALL
    /// substitutions before ANY assignment applies, so an earlier binding
    /// cannot mask a later substitution's unbound variable (`export MYTMP=/tmp
    /// x=$(touch $MYTMP/f)` expands `$MYTMP` unset → `touch /f`). Benign
    /// bindings stay allowed.
    #[test]
    fn declaration_unset_substitutions() {
        let cases = [
            ("unset $(rm -rf /)", false),
            ("unset `rm -rf /`", false),
            ("declare $(rm -rf /)", false),
            ("export x=$(rm -rf /)", false),
            ("declare \"x=$(rm -rf /)\"", false),
            ("declare -a arr=( $(rm -rf /) )", false),
            ("export x=<(rm -rf /)", false),
            ("X=$(rm -rf /)", false),
            // Binding-ordering: earlier bindings do not leak into later
            // substitution walks (bash expands with the pre-binding state).
            ("export MYTMP=/tmp x=$(touch $MYTMP/f)", false),
            ("export TMPDIR=/tmp x=$(touch $TMPDIR/f)", true),
            ("unset TMPDIR", true),
            ("export TMPDIR=/tmp", true),
            ("declare x=1", true),
            ("export A=1 B=2", true),
        ];
        run_cases(&cases);
    }

    /// Heredoc body/marker-line substitutions and the marker-line redirect
    /// expand BEFORE the owning command runs (bash: `cd /tmp <<EOF` +
    /// `$(pwd)` prints the pre-cd cwd), so they validate against the command's
    /// start state — a cd-owning heredoc cannot smuggle workspace writes via
    /// the post-cd cwd. The body also expands AFTER the command's VAR=value
    /// assignments apply (marker-line words/redirects see the pre-assignment
    /// value; an assignment-only owner applies them BEFORE both — bash feeds
    /// no command, so the bindings land in the current shell first). A
    /// `|`-glued member forks at pipeline start (pre-command cwd); the
    /// `&&`/`||` tail runs after the command (post-cd temp writes stay
    /// allowed). Construct-owning redirects — direct or as the last member of
    /// a list/pipeline body — open at the construct's start, before any inner
    /// command's cd.
    #[test]
    fn cd_owning_heredoc_state() {
        let cases = [
            // Body/marker-line substitutions and the marker-line redirect
            // expand before the owning command runs (pre-cd).
            ("cd /tmp <<EOF\n$(rm -rf .)\nEOF", false),
            ("cd /tmp <<EOF\n$(touch rel)\nEOF", false),
            ("cd /tmp <<EOF\n$(mkdir x)\nEOF", false),
            ("cd /tmp <<EOF\n$(rm -rf x)\nEOF", false),
            ("cd /tmp <<EOF\n$(echo hi > wsfile)\nEOF", false),
            ("cd /tmp <<EOF > rel\nbody\nEOF", false),
            // A `|`-glued member forks WITH the command at pipeline start —
            // `rm f` runs in the pre-cd (workspace) cwd.
            ("cd /tmp <<EOF | rm f\nbody\nEOF", false),
            ("cd /tmp <<EOF | touch /tmp/f\nbody\nEOF", true),
            // Body substitutions see the owning command's VAR=value
            // assignments (bash expands them after the assignment applies);
            // marker-line words/redirects see the pre-assignment value.
            ("TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            ("TMPDIR=/tmp/xyzzy cat <<EOF\n$(touch $TMPDIR/x)\nEOF", true),
            (
                "cd /tmp && TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            ("a | TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            (
                "TMPDIR=/tmp/xyzzy cat <<EOF > \"$TMPDIR/f\"\nbody\nEOF",
                true,
            ),
            // Construct-owning redirects open at the CONSTRUCT's start (pre-
            // internal-cd): `{ cd /tmp; cat; }` body expansions and `> f` run
            // in the workspace — also when the construct is the LAST member of
            // a list body (`true && { cd /tmp; cat; }` starts pre-internal-cd).
            ("{ cd /tmp; cat; } <<EOF\n$(rm -rf .)\nEOF", false),
            ("{ cd /tmp; } <<EOF\n$(rm -rf .)\nEOF", false),
            ("( cd /tmp; cat; ) <<EOF\n$(rm -rf .)\nEOF", false),
            ("{ cd /tmp; cat; } <<EOF > rel\nbody\nEOF", false),
            ("{ cd /tmp; cat; } > f", false),
            (
                "if true; then cd /tmp; touch f; fi <<EOF\n$(rm -rf .)\nEOF",
                false,
            ),
            (
                "for x in a; do cd /tmp; touch f; done <<EOF\n$(rm -rf .)\nEOF",
                false,
            ),
            ("true && { cd /tmp; cat; } <<EOF\n$(rm -rf .)\nEOF", false),
            ("true && ( cd /tmp; cat; ) <<EOF\n$(rm -rf .)\nEOF", false),
            (
                "true && if cd /tmp; then cat; fi <<EOF\n$(rm -rf .)\nEOF",
                false,
            ),
            ("true && { cd /tmp; cat; } > f", false),
            ("true && { cd /tmp; cat; } <<EOF > rel\nbody\nEOF", false),
            // Assignment-only owners apply their bindings BEFORE the body
            // expands and BEFORE the redirect opens (unlike command owners).
            ("TMPDIR=/etc <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            (
                "cd /tmp && TMPDIR=/etc <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            ("TMPDIR=/etc <<EOF > \"$TMPDIR/f\"\nbody\nEOF", false),
            ("TMPDIR=/tmp/xyzzy <<EOF > \"$TMPDIR/f\"\nbody\nEOF", true),
            ("TMPDIR=/tmp/xyzzy <<EOF\n$(touch $TMPDIR/x)\nEOF", true),
            ("A=1 B=2 <<EOF > /tmp/f\nbody\nEOF", true),
            // Substitution bodies run in subshells — their internal cds must
            // not shift the redirect state (`cat $(cd /tmp && cat) <<EOF`
            // still expands its body in the workspace).
            ("cat $(cd /tmp && cat) <<EOF\n$(rm -rf .)\nEOF", false),
            ("export x=$(cd /tmp && cat) <<EOF\n$(rm -rf .)\nEOF", false),
            ("cat $(cd /tmp && cat) <<EOF > rel\nbody\nEOF", false),
            ("echo $(cd /tmp && cat) > f", false),
            ("cat $(cd /tmp && cat) <<EOF\n$(echo hi)\nEOF", true),
            // Benign tails and bodies stay allowed.
            ("cd /tmp <<EOF && touch f\nbody\nEOF", true),
            ("{ cat; } <<EOF\n$(echo hi > /tmp/out)\nEOF", true),
            ("( cat; ) <<EOF\n$(echo hi)\nEOF", true),
            ("true && { cat; } <<EOF\n$(echo hi > /tmp/out)\nEOF", true),
            ("cat <<EOF\n$(echo hi > /tmp/out)\nEOF", true),
        ];
        run_cases(&cases);
    }

    /// `!`- and `time`-wrapped heredoc owners apply their VAR=value
    /// assignments before the body expands (bash-verified): `!` unwraps to
    /// the negated command, and `time`'s operand is a full command list whose
    /// assignments parse as plain words (`time TMPDIR=/etc cat`).
    #[test]
    fn negated_time_heredoc_owners() {
        let cases = [
            // Negated command owners: body sees the assignment.
            ("! TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            ("! TMPDIR=/etc <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            (
                "cd /tmp && ! TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            ("! TMPDIR=/etc <<EOF > \"$TMPDIR/f\"\nbody\nEOF", false),
            (
                "! TMPDIR=/etc time cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            // Time owners: the timed command's assignments bind first.
            ("time TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            ("time TMPDIR=/etc <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            (
                "time -p TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            (
                "time -p ! TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            (
                "time ! TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                false,
            ),
            ("time ! TMPDIR=/etc <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            // Assignment-only time owners open the marker-line redirect
            // post-binding.
            ("time TMPDIR=/etc <<EOF > \"$TMPDIR/f\"\nbody\nEOF", false),
            // `! { ...; }` parses broken (negated_command swallows `! { cd
            // /tmp`; `cat` and `} <<EOF` become separate top-level commands)
            // — the rejection comes from the stray `}`/`{` reserved-word
            // checks, not construct-state binding; bash runs the body at the
            // construct's start (workspace).
            ("! { cd /tmp; cat; } <<EOF\n$(rm -rf .)\nEOF", false),
            // Benign temp-bound forms stay allowed.
            (
                "! TMPDIR=/tmp/xyzzy cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
            ("! TMPDIR=/tmp/xyzzy <<EOF\n$(touch $TMPDIR/x)\nEOF", true),
            (
                "time TMPDIR=/tmp/xyzzy cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
            (
                "time ! TMPDIR=/tmp/xyzzy <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
            (
                "TMPDIR=/tmp/xyzzy time cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
        ];
        run_cases(&cases);
    }

    /// Substitutions in redirect nodes execute and must be validated:
    /// input-redirect targets (`cat < $(rm -rf /)`), words swallowed into a
    /// file_redirect after the target, and quoted/unquoted heredoc marker-line
    /// words. Substitution-formed targets stay fail-closed (`cat > $(mktemp
    /// -d)/f` — the guard cannot prove the substitution resolves under temp).
    #[test]
    fn redirect_substitution_gaps() {
        let cases = [
            ("cat < $(rm -rf /)", false),
            ("cat < \"$(rm -rf /)\"", false),
            ("cat < <(rm -rf /)", false),
            ("cat < `rm -rf /`", false),
            ("echo hi > /tmp/out $(rm -rf /)", false),
            ("cat <<EOF \"$(rm -rf /)\"\nbody\nEOF", false),
            ("cat <<EOF $(rm -rf /)\nbody\nEOF", false),
            ("cat > $(mktemp -d)/f", false),
            ("cat < /tmp/in", true),
        ];
        run_cases(&cases);
    }

    /// Arithmetic expansions recurse into embedded substitutions only:
    /// `$(( 1 + 2 ))` is inert, `$((1+$(rm f)))` rejects.
    #[test]
    fn arithmetic_expansion_guard() {
        let cases = [
            ("echo $((1+2))", true),
            ("echo $[1+2]", true),
            ("echo \"$((1+2))\"", true),
            ("echo $((1+$(touch f)))", false),
            ("(( 1 + 2 ))", true),
            ("for ((i=0; i>3; i++)); do echo hi; done", true),
        ];
        run_cases(&cases);
    }

    /// The source-span re-join adapter: unbraced expansions split by the
    /// parser re-join into one word for path resolution (safety requirement —
    /// without it unbound variables could make temp paths look literal).
    #[test]
    fn expansion_split_rejoin() {
        let cases = [
            ("cd /tmp && touch $SNAP/$RANDOM/f", false), // $SNAP unbound — fail closed
            ("SNAP=/tmp && touch $SNAP/$RANDOM/f", true),
            ("SNAP=/tmp && touch pre$SNAP/f", false),
            ("SNAP=/tmp && touch /tmp/$SNAP/f", true),
            ("SNAP=$(mktemp -d) && touch $SNAP/$RANDOM/f", true),
            ("touch ${TMPDIR}/x", true), // braced split re-join
            ("echo $SNAP/$RANDOM/f", true),
        ];
        run_cases(&cases);
    }

    /// Declaration/unset temp bindings: `export TMPDIR=/etc` poisons;
    /// `unset TMPDIR` models an EMPTY binding (bash expands to ''), so a
    /// chained write resolves against the CWD and rejects.
    #[test]
    fn declaration_unset_bindings() {
        let cases = [
            ("declare TMPDIR=/etc && touch $TMPDIR/f", false),
            ("local TMPDIR=/etc && touch $TMPDIR/f", false),
            ("typeset TMPDIR=/etc && touch $TMPDIR/f", false),
            ("readonly TMPDIR=/etc && touch $TMPDIR/f", false),
            ("export TMPDIR=/tmp && touch $TMPDIR/f", true),
            ("unset TMPDIR; touch $TMPDIR/f", false),
            ("unset TMPDIR; touch /tmp/f", true),
            ("unset -f TMPDIR; touch /tmp/f", true),
            (
                "export TMPDIR=/tmp && unset TMPDIR && touch $TMPDIR/f",
                false,
            ),
        ];
        run_cases(&cases);
    }

    // ── mktemp (temp dir, allowed) ────────────────────────────────

    #[test]
    fn mktemp_allowed() {
        let cases = [("mktemp", true), ("mktemp -t mahbot.XXXXXX", true)];

        run_cases(&cases);
    }

    // ── Prefix stripping (P0) ──────────────────────────────────────

    /// Tests that ALL delegating shell prefixes (those that forward their
    /// arguments as a command) correctly dispatch commands for read-only
    /// validation. Excludes non-delegating builtins (`cd`, `pushd`, `popd`,
    /// `export`, `source`, `.`).
    ///
    /// Three command scenarios are tested for every prefix:
    /// - `rm file` — a mutating command that must be rejected.
    /// - `git push` — a mutating git subcommand that must be rejected
    ///   (ensuring no prefix masks the git command word).
    /// - `git status` — a safe git command that must be allowed.
    #[test]
    fn shell_prefixes_delegating() {
        let cases = [
            ("rm file", false),
            ("git push", false),
            ("git status", true),
        ];

        for prefix in SHELL_PREFIXES {
            if NON_DELEGATING_PREFIXES.contains(prefix) {
                continue;
            }
            for &(command, allowed) in &cases {
                let cmd = format!("{prefix} {command}");
                if allowed {
                    ok(&cmd);
                } else {
                    assert_rejected(&cmd);
                }
            }
        }
    }

    // ── Prefix / env stripping regression tests (P0) ──────────────

    #[test]
    fn prefix_bypass_and_env() {
        let cases = [
            // Prefix stripping with flags
            ("sudo -E rm file", false),
            ("sudo git status", true),
            ("sudo cargo check", true),
            // Git prefix bypass
            ("sudo git push", false),
            ("env git push", false),
            ("GIT_DIR=/tmp sudo git push", false),
            ("sudo git stash list", true),
            ("cd", true),
            ("cd ..", true),
            // VAR=val stripping
            ("FOO=bar rm file", false),
            ("VAR=val sudo rm -rf /", false),
            ("GIT_DIR=/tmp git status", false), // GIT_* env rejected (exec vector)
        ];

        run_cases(&cases);
    }
    #[allow(clippy::too_many_lines)] // data-driven bypass battery

    /// Git exec vectors: flags/config/env channels that run programs are
    /// rejected, even on allowlisted subcommands. Includes the analyst-flagged
    /// gaps (repo redirect, --config-env/--config-file, export channel,
    /// env/sudo-wrapped forms) and documented fail-closed over-rejections.
    #[test]
    fn git_exec_vectors_rejected() {
        let cases = [
            // ── GIT_* env bindings (current-shell, export, and wrapped) ──
            ("GIT_DIR=/tmp git status", false),
            ("GIT_EXTERNAL_DIFF=/bin/echo git diff", false),
            ("GIT_SSH_COMMAND=/bin/echo git ls-remote origin", false),
            ("GIT_CONFIG_COUNT=1 git log", false),
            ("GIT_EXEC_PATH=/tmp/evil git request-pull", false),
            ("GIT_ASKPASS=/bin/echo git ls-remote origin", false),
            ("GIT_TRACE=/tmp/trace git status", false),
            ("export GIT_EXTERNAL_DIFF=/bin/echo && git diff", false),
            (
                "export GIT_SSH_COMMAND=/bin/echo; git ls-remote origin",
                false,
            ),
            ("env GIT_EXTERNAL_DIFF=/bin/echo git diff", false),
            ("sudo GIT_DIR=/tmp git status", false),
            ("GIT_PAGER=/bin/echo git log", true), // carve-out: no TTY → pager never spawns
            // ── -c/-C global options (position-sensitive: log/diff -c stay allowed) ──
            ("git -c diff.external=/bin/echo diff", false),
            ("git -c core.fsmonitor=/bin/echo status", false),
            ("git -c user.name=me log", false),
            ("git -cfoo=bar status", false), // attached form
            ("git -pc status", false),       // cluster
            ("git -C /tmp/repo status", false),
            ("git -C/tmp/repo diff", false),   // attached form
            ("git -pC /tmp/repo diff", false), // cluster
            // ── repo-redirect / config-injection long globals ──
            ("git --git-dir=/tmp/evil/.git diff", false),
            ("git --git-dir /tmp/evil status", false),
            ("git --work-tree=/tmp/evil status", false),
            ("git --config-env=core.fsmonitor=F status", false),
            ("git --config-file=/tmp/evil status", false),
            ("git --exec-path=/tmp/evil request-pull", false),
            // ── exec flags (anywhere in the command; --no-* forms stay allowed) ──
            ("git diff --ext-diff", false),
            ("git diff --no-index --ext-diff /tmp/a /tmp/b", false),
            ("git log -p --textconv", false),
            ("git log --show-signature", false),
            ("git cat-file --filters HEAD:src/lib.rs", false),
            ("git cat-file --text HEAD:a.txt", false), // abbrev of --textconv: execs the driver
            ("git ls-remote --upload-pack=/bin/echo origin", false),
            ("git ls-remote --upload-pack /bin/echo origin", false),
            ("git push --dry-run --exec=/bin/echo origin", false), // extension bypass
            ("git stash show --textconv", false),                  // extension bypass
            (
                "git push --dry-run --receive-pack=/bin/echo origin main",
                false,
            ),
            (
                "git push --dry-run --receive-pack /bin/echo origin main",
                false,
            ),
            // ── git abbreviated long-option prefixes (--upload-p= == --upload-pack=) ──
            ("git ls-remote --upload-p=/bin/echo origin", false),
            ("git ls-remote --upload-pa /bin/echo origin", false),
            ("git ls-remote --exe=/bin/echo origin", false),
            ("git push --dry-run --exe=/bin/echo origin", false),
            ("git push --dry-run --exe /bin/echo origin", false),
            (
                "git push --dry-run --receive-p=/bin/echo origin main",
                false,
            ),
            ("git grep --open-files-in-p=/bin/echo pattern", false),
            ("git grep --open-files-in-p /bin/echo pattern", false),
            // defense-in-depth (this git build rejects these spellings itself)
            ("git diff --ext-d", false),
            ("git log --textc", false),
            ("git log --show-signat", false),
            ("git cat-file --filt HEAD:src/lib.rs", false), // ambiguous in git, blocked anyway
            // ── grep -O / --open-files-in-pager ──
            ("git grep -O /bin/echo pattern", false),
            ("git grep -O/bin/echo pattern", false),
            ("git grep -nO /bin/echo pattern", false),
            ("git grep -IO /bin/echo pattern", false),
            ("git grep --open-files-in-pager=/bin/echo pattern", false),
            ("git grep --open-files-in-pager pattern", false),
            // ── help viewers (web/info/man all exec external programs) ──
            ("git help --web", false),
            ("git help -w", false),
            ("git help -aw", false),
            ("git help -m git", false),
            ("git help -i git", false),
            ("git help --man git", false),
            ("git help --info git", false),
            ("git help --m git", false),
            // ── full-path git (basename allowlist bypass) ──
            ("/usr/bin/git push", false),
            ("/usr/bin/git -C /tmp status", false),
            // ── documented fail-closed over-rejections ──
            ("git grep -e '--ext-diff' pattern", false), // literal pattern
            ("git grep -e '-Ofoo' pattern", false),      // literal pattern
            ("git log --format=$'%h\\t%s'", false),      // ANSI-C unprovable token
            ("git log -- --ext-diff", false),            // pathname named like a flag
        ];

        run_cases(&cases);
    }

    /// Positive controls for the exec-vector scan: ordinary read-only git and
    /// the legitimate spellings that must NOT be caught (position/subcommand
    /// scoping, the bare `--` path separator, the benign `--text` flag,
    /// --no-* disabling forms, GIT_PAGER carve-out).
    #[test]
    fn git_exec_vector_positives() {
        let cases = [
            // Ordinary read-only git still passes
            ("git status", true),
            ("git log", true),
            ("git diff", true),
            ("git show HEAD", true),
            ("git blame src/lib.rs", true),
            // bare `--` path separator is never an option
            ("git log -- src/lib.rs", true),
            ("git log --oneline -- path", true),
            ("git diff -- file", true),
            ("git show HEAD -- file", true),
            ("git grep -- pattern", true),
            // exact `--text` is the benign -a flag where git has it
            ("git grep --text pattern", true),
            ("git diff --text", true),
            ("git log --text", true),
            ("git blame --text src/lib.rs", true),
            // Position-sensitive: -c after the subcommand is combined diff
            ("git log -c", true),
            ("git diff -c", true),
            ("git log --oneline -c", true),
            // -w on diff/log is ignore-whitespace (help-scoped -w only)
            ("git diff -w", true),
            ("git log -w", true),
            // -O on log is a benign order-file read (grep-scoped -O only)
            ("git log -Oorderfile", true),
            // grep short flags without uppercase -O
            ("git grep -o pattern", true),
            ("git grep -n pattern", true),
            ("git grep -I pattern", true),
            ("git grep -c pattern", true),
            ("git grep -C3 pattern", true),
            ("git grep --regexp=-Ofoo pattern", true), // long form spared
            // --no-* disabling forms stay allowed
            ("git diff --no-ext-diff", true),
            ("git diff --no-textconv", true),
            ("git log --no-show-signature", true),
            // --filter exact stays allowed (legit flag shadowing --filters)
            ("git log --filter=blob:none", true),
            ("git rev-list --filter=blob:none HEAD", true),
            ("git cat-file --filter blob:none HEAD", true),
            // help non-viewer flags and topics
            ("git help", true),
            ("git help -a", true),
            ("git help --all", true),
            ("git help -g", true),
            ("git help -c core.pager", true),
            ("git help --config", true),
            // regular ls-remote flags stay allowed
            ("git ls-remote --heads origin", true),
            // paginate globals (no c/C)
            ("git -p log", true),
            ("git -P log", true),
            // GIT_PAGER carve-out (pager never spawns on piped stdout)
            ("GIT_PAGER=less git log", true),
            // quoted args that only look like flags (no backslash escapes)
            ("git log --format=$'%h'", true),
            ("git log --grep=--ext-diff", true),
        ];

        run_cases(&cases);
    }

    /// Path-spelled git invocations route to the git layer exactly like plain
    /// `git`: basename dispatch + basename-matched subcommand extraction, so
    /// mutating forms reject and read-only forms stay allowed. Per-spelling
    /// extraction output is pinned by `test_extract_git_subcommand`; this
    /// matrix pins the end-to-end routing per spelling shape.
    #[test]
    fn git_path_spelled_invocations() {
        let cases = [
            // absolute / relative paths
            ("/usr/bin/git init", false),
            ("/usr/bin/git status", true),
            ("./git init", false),
            ("./git log", true),
            // quoted git and quotes inside the final path component are
            // rejected fail-closed: bare forms at the verb gate, prefix-
            // forwarded forms via git-layer validation
            ("'git' init", false),
            ("/usr/bin/'git' init", false),
            ("sudo /usr/bin/'git' reset --hard", false),
            ("env /usr/bin/'git' commit -m x", false),
            ("sudo /usr/bin/'git' status", true),
            // prefixes in front of the path
            ("sudo /usr/bin/git init", false),
            ("sudo /usr/bin/git status", true),
            ("env /usr/bin/git push", false),
            // combined short-flag cluster through the same gate
            ("/usr/bin/git branch -df feature", false),
        ];

        run_cases(&cases);
    }

    // ── extract_git_subcommand unit tests ──────────────────────────

    #[test]
    fn test_extract_git_subcommand() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: &'static str,
        }

        let cases = [
            Case {
                name: "basic",
                input: "git status",
                expected: "status",
            },
            Case {
                name: "with global flag",
                input: "git -C /repo diff",
                expected: "diff",
            },
            Case {
                name: "with config",
                input: "git -c user.name=me log",
                expected: "log",
            },
            Case {
                name: "with git dir",
                input: "git --git-dir /repo status",
                expected: "status",
            },
            Case {
                name: "env assignment",
                input: "GIT_DIR=/tmp git status",
                expected: "status",
            },
            Case {
                name: "no git",
                input: "cargo build",
                expected: "",
            },
            Case {
                name: "git only",
                input: "git",
                expected: "",
            },
            Case {
                name: "full subcommand",
                input: "git branch -d feature",
                expected: "branch -d feature",
            },
            Case {
                name: "with double dash",
                input: "git -- diff",
                expected: "diff",
            },
            Case {
                name: "stash list",
                input: "git stash list",
                expected: "stash list",
            },
            Case {
                name: "stderr capture suffix skipped",
                input: "git --version 2>&1",
                expected: "",
            },
            Case {
                name: "stderr capture after subcommand",
                input: "git status 2>&1",
                expected: "status 2>&1",
            },
            Case {
                name: "multiple env",
                input: "CC=gcc CXX=g++ git status",
                expected: "status",
            },
            Case {
                name: "multiple flags",
                input: "git -C /repo --git-dir /other status",
                expected: "status",
            },
            Case {
                name: "with sudo skipped",
                input: "sudo git status",
                expected: "status",
            },
            Case {
                name: "absolute path",
                input: "/usr/bin/git status",
                expected: "status",
            },
            Case {
                name: "absolute path with global flag",
                input: "/opt/homebrew/bin/git -C /repo diff",
                expected: "diff",
            },
            Case {
                name: "prefixed absolute path",
                input: "sudo /usr/bin/git push",
                expected: "push",
            },
            Case {
                name: "relative path",
                input: "./git log",
                expected: "log",
            },
            Case {
                name: "quoted git",
                input: "'git' branch -d feature",
                expected: "branch -d feature",
            },
            Case {
                name: "quotes in final path component",
                input: "/usr/bin/'git' status",
                expected: "status",
            },
            Case {
                name: "prefixed quotes in final path component",
                input: "sudo /usr/bin/'git' push",
                expected: "push",
            },
            Case {
                name: "with env skipped",
                input: "env git status",
                expected: "status",
            },
            Case {
                name: "env and sudo",
                input: "GIT_DIR=/tmp sudo git status",
                expected: "status",
            },
            Case {
                name: "sudo push",
                input: "sudo git push",
                expected: "push",
            },
            Case {
                name: "flag with multiple args",
                input: "git branch --merged master",
                expected: "branch --merged master",
            },
        ];

        for case in &cases {
            assert_eq!(
                extract_git_subcommand(case.input),
                case.expected,
                "case: {}",
                case.name
            );
        }
    }

    // ── Temp / scratch directory tests ─────────────────────────────

    #[test]
    fn temp_scratch_tests() {
        let cases = [
            (
                "cat > /tmp/test_match.rs << 'EOF'\nfn test() { match x { \"a\" => 1, _ => 0 } }\nEOF",
                true,
            ),
            ("echo hello > /private/tmp/mahbot_test_out.txt", true),
            ("tee /tmp/scratch.log", true),
            ("touch /tmp/scratch.txt", true),
            ("mkdir -p /tmp/scratch_dir", true),
            ("tee output.log", false),
            ("rm /tmp/scratch.txt", true),
            // ── TEMP_MUTATORS (cp, mv, rm, gzip, etc.) ──
            ("cp /tmp/a /tmp/b", true),
            ("mv /tmp/a /tmp/b", true),
            ("cp /tmp/a /etc/passwd", false),
            ("mv /tmp/a /etc/passwd", false),
            ("rm /etc/passwd", false),
            ("rmdir /tmp/scratch_dir", true),
            ("gzip /tmp/file.txt", true),
            ("gunzip /tmp/file.txt.gz", true),
            ("bzip2 /tmp/file.txt", true),
            ("xz /tmp/file.txt", true),
            ("zstd /tmp/file.txt", true),
            ("zip /tmp/out.zip /tmp/file1 /tmp/file2", true),
            // cp: only the DESTINATION must be under temp (sources are
            // read-only — copying from anywhere into temp is allowed;
            // copying into the workspace stays blocked).
            ("cp /etc/passwd /tmp/out", true), // source outside temp, dest temp
            // ── Flag-based temp-aware checks ──
            // curl -o to temp → allowed
            ("curl -o /tmp/file URL", true),
            ("curl --output /tmp/file URL", true),
            ("curl -o /etc/passwd URL", false),
            ("curl --output /etc/passwd URL", false),
            // curl -O/--remote-name always blocked (writes to CWD)
            ("curl -O URL", false),
            ("curl --remote-name URL", false),
            // curl -o with a mix: -o OK but -O still blocked
            ("curl -o /tmp/file -O URL", false), // -O always blocks
            // wget -O to temp → allowed
            ("wget -O /tmp/file URL", true),
            ("wget --output-document /tmp/file URL", true),
            ("wget -O /etc/passwd URL", false),
            ("wget --output-document /etc/passwd URL", false),
            ("wget -P /tmp/dir URL", true),
            ("wget --directory-prefix /tmp/dir URL", true),
            ("wget -P /etc/dir URL", false),
            // wget without output flags → always blocked
            ("wget URL", false),
            // sed -i to temp → allowed
            ("sed -i 's/a/b/' /tmp/file", true),
            ("sed -i.bak 's/a/b/' /tmp/file", true),
            ("sed -i 's/a/b/' /tmp/file1 /tmp/file2", true),
            ("sed -i 's/a/b/' /etc/passwd", false),
            ("sed -i 's/a/b/' /tmp/file /etc/passwd", false), // mixed
            ("sed -i '' /tmp/file", true),                    // empty backup ext
            // sed -i with -e flag
            ("sed -i -e 's/a/b/' /tmp/file", true),
            ("sed -i -e 's/a/b/' /etc/passwd", false),
            // sed -i with separate backup extension
            ("sed -i .bak 's/a/b/' /tmp/file", true),
            ("sed -i .bak 's/a/b/' /etc/passwd", false),
            // sed in-place family: attached suffix, clusters, uppercase -I
            ("sed -ix 's/a/b/' /tmp/file", true),
            ("sed -ix 's/a/b/' /etc/passwd", false),
            ("sed -nix 's/a/b/' /etc/passwd", false),
            ("sed -I 's/a/b/' /tmp/file", true),
            ("sed -I 's/a/b/' /etc/passwd", false),
            ("sed -nIx 's/a/b/' /etc/passwd", false),
            // GNU long form --in-place (token-level; behavioral check needs GNU sed)
            ("sed --in-place 's/a/b/' /tmp/file", true),
            ("sed --in-place=bak 's/a/b/' /tmp/file", true),
            ("sed --in-place 's/a/b/' /etc/passwd", false),
            ("sed --in-place=bak 's/a/b/' /etc/passwd", false),
            // GNU getopt_long abbreviations (--i is the shortest unique prefix)
            ("sed --i 's/a/b/' /etc/passwd", false),
            // Quote-wrapped / concatenated flags (shell strips quotes)
            ("sed '-i' 's/a/b/' /tmp/file", true),
            ("sed '-i' 's/a/b/' /etc/passwd", false),
            ("sed '-i'x 's/a/b/' /tmp/file", true), // concatenates to -ix
            ("sed '-i'x 's/a/b/' /etc/passwd", false),
            ("sed '-nix' 's/a/b/' /etc/passwd", false),
            ("sed --'in-place' 's/a/b/' /tmp/file", true), // → --in-place
            ("sed --'in-place' 's/a/b/' /etc/passwd", false),
            // Backslash-escaped and ANSI-C-quoted flags (shell delivers -i/-ix)
            ("sed \\-ix 's/a/b/' /tmp/file", true),
            ("sed \\-ix 's/a/b/' /etc/passwd", false),
            ("sed \\--in-place 's/a/b/' /etc/passwd", false),
            ("sed $'-ix' 's/a/b/' /etc/passwd", false),
            // $'...'/$"..." tokens with escapes are unprovable flag candidates
            ("sed $'\\x2di' 's/a/b/' /tmp/file", true), // → -i; temp still allowed
            ("sed $'\\x2di' 's/a/b/' /etc/passwd", false),
            ("sed $'\\055i' 's/a/b/' /etc/passwd", false), // octal escape
            ("sed $'\\x2dix' 's/a/b/' /etc/passwd", false), // → -ix
            ("sed $'\\u002di' 's/a/b/' /etc/passwd", false), // any escape → unprovable
            ("sed $'\\\\x2di' 's/a/b/' /etc/passwd", false), // literal backslash → over-rejected
            ("sed $\"\\x2di\" 's/a/b/' /etc/passwd", false), // locale quoting → over-rejected
            // $'...' operands with escapes count as operands (may be a path)
            ("sed -i 's/a/b/' $'\\x2fetc\\x2fpasswd' /tmp/file", false),
            // Documented over-rejections (fail-closed contract)
            ("sed -e's/x/i/' file", false), // attached -e script arg
            ("sed -- -info.txt", false),    // operand literally named like a -i flag
            // Quote-wrapped operands (normalized for the absolute check)
            ("sed -i 's/a/b/' '/tmp/file'", true),
            ("sed -i 's/a/b/' '/etc/passwd' /tmp/file", false), // mixed
            // dd of= to temp → allowed
            ("dd if=/dev/zero of=/tmp/out bs=1024 count=1", true),
            ("dd of=/tmp/out", true),
            ("dd if=/dev/zero of=/etc/passwd bs=1024 count=1", false),
            ("dd of=/etc/passwd", false),
            // base64 -d -o to temp → allowed
            ("base64 -d -o /tmp/out input.txt", true),
            ("base64 -d --output /tmp/out input.txt", true),
            ("base64 -d -o /etc/passwd input.txt", false),
            ("base64 -d --output /etc/passwd input.txt", false),
            // base64 -d without -o → stdout, always allowed
            ("base64 -d input.txt", true),
            // base64 -o to temp → allowed (encode or decode mode)
            ("base64 -o /tmp/out input.txt", true),
            // ── Multiple path arguments (security bypass) ──
            // Multiple path args under temp → should be allowed
            ("tee /tmp/scratch.log /tmp/out.txt", true),
            ("touch /tmp/a.txt /tmp/b.txt", true),
            // Mixed: one temp, one non-temp → should be rejected
            ("tee /tmp/scratch.log /etc/passwd", false),
            ("touch /tmp/scratch.txt /etc/cron.d/evil", false),
            ("mkdir -p /tmp/dir /etc/cron.d", false),
            // Mixed with redirects → only path args checked
            ("tee /tmp/scratch.log /etc/passwd > /dev/null", false),
            ("tee /tmp/scratch.log /tmp/out.txt > /dev/null", true),
            // Combined redirect tokens (2>/dev/null style)
            ("tee /tmp/scratch.log /etc/passwd 2>/dev/null", false),
            ("tee /tmp/scratch.log /tmp/out.txt 2>&1", true),
            // Heredoc with scratch mutator → heredoc body not treated as path
            ("tee /tmp/scratch.log << 'EOF'\nbody\nEOF", true),
            (
                "tee /tmp/scratch.log /tmp/out.txt << 'EOF'\nbody\nEOF",
                true,
            ),
            // 1> standalone redirect (separate target) → not a path arg
            ("tee /tmp/scratch.log 1>/dev/null", true),
            // Bash &> combined redirect → not collected as path arg
            ("tee /tmp/scratch.log &>/dev/null", true),
            ("tee /tmp/scratch.log &>>/dev/null", true),
            // 1> with space-separated target → redirect target not collected as path
            ("tee /tmp/scratch.log 1> /dev/null", true),
            // Generic digit-prefixed redirects ({digit}> and {digit}<)
            ("tee /tmp/scratch.log 3> /dev/null", true),
            ("tee /tmp/scratch.log 3< /dev/null", true),
            // Digit-prefixed heredoc (e.g. 3<<EOF) → body not treated as path
            ("tee /tmp/scratch.log 3<< 'EOF'\nbody\nEOF", true),
            // Multi-digit fd redirect with space-separated target (10> /dev/null)
            ("tee /tmp/scratch.log 10> /dev/null", true),
            // &> standalone redirect with space before target
            ("tee /tmp/scratch.log &> /dev/null", true),
            ("tee /tmp/scratch.log &>> /dev/null", true),
        ];

        run_cases(&cases);
    }

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are allowed with temp paths
    /// and rejected with non-temp paths, preventing coverage drift.
    #[test]
    fn all_temp_mutators_allowed_with_temp_paths() {
        for &cmd in TEMP_MUTATORS {
            let args = match cmd {
                "cp" | "mv" => "/tmp/a /tmp/b",
                "zip" => "/tmp/out.zip /tmp/file1",
                _ => "/tmp/scratch.txt",
            };
            ok(&format!("{cmd} {args}"));
        }
    }

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are rejected when used
    /// with a non-temp absolute path, preventing accidental test drift.
    #[test]
    fn all_temp_mutators_rejected_with_non_temp() {
        for &cmd in TEMP_MUTATORS {
            assert_rejected(&format!("{cmd} /etc/blocked_test"));
        }
    }

    /// Renders every [`MUTATOR_CHECKS`] rejection template so malformed
    /// `{verb}` placeholders fail in CI instead of at runtime, and sanity-
    /// checks the static git always-mutate rows.
    #[test]
    fn mutator_table_rows_render() {
        for check in MUTATOR_CHECKS {
            for &verb in check.verbs {
                let rendered = check.rejection.replace("{verb}", verb);
                assert!(
                    !rendered.contains('{'),
                    "unsubstituted placeholder in rejection: {rendered}"
                );
                assert!(
                    rendered.contains(verb),
                    "verb missing from rendered rejection: {rendered}"
                );
            }
        }
        for &(_, why, suggestion) in GIT_ALWAYS_MUTATE {
            assert!(!why.is_empty() && !suggestion.is_empty());
        }
    }

    // ── Phase 1 acceptance: heredocs ──────────────────────────────────────

    /// Every fixed heredoc pattern AND its mutating sibling (guard contract,
    /// phase 1.1).
    #[test]
    fn heredoc_acceptance() {
        let cases = [
            // Write command immediately after the terminator must NOT escape
            // scanning (live bypass: leaked_guard_test.txt).
            ("cat <<EOF\nbody\nEOF\ntouch workspace_file", false),
            ("cat <<EOF\nbody\nEOF\ntouch /tmp/scratch_file", true),
            // Write on the delimiter line must be scanned (live bypass).
            ("cat <<EOF > workspace_file", false),
            ("cat <<EOF > /tmp/out", false), // unterminated heredoc — parser error, fail-closed
            // Heredoc body containing mutator-shaped text is literal — allowed.
            ("cat <<EOF\nrm -rf /tmp\nEOF", true),
            ("cat <<EOF\ncat > workspace_file\nEOF", true),
            // Two-heredoc chain: bodies excluded, delimiter-line redirect scanned.
            (
                "cat <<EOF1 <<EOF2 > /tmp/out\nbody1 > x\nEOF1\nbody2 > y\nEOF2",
                false, // multiple heredocs per line — parser error, fail-closed
            ),
            // <<- with tab-indented terminator.
            ("cat <<-EOF\n\tbody\n\tEOF", true),
            // <<- strips ALL leading tabs (bash semantics): a 2+-tab-indented
            // terminator still terminates, so a write on the line after it
            // must be scanned (QA round: the matcher stripped only one tab,
            // swallowing the write as body text).
            ("cat <<-EOF\n\tbody\n\t\tEOF", true),
            ("cat <<-EOF\n\tbody\n\t\tEOF\ntouch workspace_file", false),
            ("cat <<-EOF\n\tbody\n\t\t\tEOF\ntouch workspace_file", false),
            ("cat <<-EOF\n\tbody\n\t\tEOF\ntouch /tmp/scratch_file", true),
            // CRLF line endings: write after terminator still blocked.
            ("cat <<EOF\r\nbody\r\nEOF\r\ntouch workspace_file", false),
            ("cat <<EOF\r\nbody\r\nEOF", true),
            // Terminator at end of input (no trailing newline).
            ("cat <<EOF\nbody\nEOF", true),
            // Quoted delimiter.
            ("cat <<'EOF'\nbody\nEOF", true),
            // Heredoc with workspace write after, chained with && on the
            // line after the terminator.
            ("cat <<EOF\nbody\nEOF\n&& touch workspace_file", false),
            // Double-quoted `$()` in an unquoted heredoc body still expands
            // in bash (quote chars are literal in heredoc bodies) — blocked.
            ("cat <<EOF\n\"$(touch workspace_file)\"\nEOF", false),
            // Multi-tab <<- terminator with a substitution body and a write
            // after the terminator — all caught.
            (
                "cat <<-EOF\n\t$(touch workspace_file)\n\t\tEOF\ntouch /tmp/ok",
                false,
            ),
            // A command on the terminator line itself is NOT a terminator in
            // bash (unterminated heredoc — nothing executes), so it must not
            // be treated as a command.
            ("cat <<EOF\nbody\nEOF && touch workspace_file", false), // parser terminates at EOF; the tail is over-rejected (bash treats it as body)
            // fd-prefixed heredoc body not scanned as command text.
            ("tee /tmp/x 3<< 'EOF'\nbody\nEOF", true),
            // ── Command substitutions in heredoc bodies ─────────────
            // Real bash executes `$(...)`/backticks inside UNQUOTED heredoc
            // bodies, so those spans must remain scanned (reviewer round).
            ("cat <<EOF\n$(touch workspace_file)\nEOF", false),
            ("cat <<EOF\n$(echo hi > workspace_file)\nEOF", false),
            ("cat <<EOF\n`touch workspace_file`\nEOF", false),
            // Temp-targeted body substitutions stay allowed.
            ("cat <<EOF\n$(touch /tmp/x)\nEOF", true),
            ("cat <<EOF\n$(echo hi > /tmp/out)\nEOF", true),
            // Nested substitution in an unquoted body.
            ("cat <<EOF\n$(echo $(touch workspace_file))\nEOF", false),
            // Escaped `\$(` in an unquoted body is literal in bash — the
            // minimal body scan over-rejects it (accepted parser-gap class).
            ("cat <<EOF\n\\$(touch workspace_file)\nEOF", false),
            // Quoted delimiters make the body literal — no expansion.
            ("cat <<'EOF'\n$(touch workspace_file)\nEOF", true),
            ("cat <<\"EOF\"\n$(touch workspace_file)\nEOF", true),
            // Plain mutator-shaped body text (no substitution) stays literal.
            ("cat <<EOF\ntouch workspace_file\nEOF", true),
            // Multiple heredocs: a body substitution in the first body is
            // still scanned.
            ("cat <<A <<B\n$(touch workspace_file)\nA\nB", false),
            // Body substitution followed by a workspace write after the
            // terminator — both caught.
            ("cat <<EOF\n$(echo hi)\nEOF\ntouch workspace_file", false),
            // Body substitution under a temp cd — allowed (state at the
            // heredoc's position in the chain applies).
            ("cd /tmp && cat <<EOF\n$(touch rel)\nEOF", true), // Dangling `<<` markers (no delimiter): bash errors on these and
            // executes nothing — fail-closed parse errors in the new walker.
            ("cat <<", false),
            ("cat << ", false),
            ("cat <<-", false),
            ("cat <<- ", false),
            ("3<< ", false),
            ("3<<-", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: substitutions ────────────────────────────────

    /// `$(...)` and backtick contents are validated as nested commands.
    #[test]
    fn substitution_acceptance() {
        let cases = [
            // Suppressed-stderr read inside a substitution — allowed.
            ("echo $(cat /etc/passwd 2>/dev/null)", true),
            ("echo `cat /etc/passwd 2>/dev/null`", false), // backticks — fail-closed (use $())
            // Write inside a substitution — blocked.
            ("echo $(touch workspace_file)", false),
            ("echo `touch workspace_file`", false),
            // Redirect to temp inside a substitution — allowed.
            ("echo $(echo hi > /tmp/out)", true),
            // Redirect to workspace inside a substitution — blocked.
            ("echo $(echo hi > out.txt)", false),
            ("echo $(echo hi > /etc/passwd)", false),
            // Nested substitution.
            ("echo $(echo $(rm -rf /tmp/x))", true),
            ("echo $(echo $(touch ws_file))", false),
            // Substitution with a mutating command under temp — allowed.
            ("echo $(rm -rf /tmp/scratch_dir)", true),
            // Substitution that writes to workspace via a temp-qualified var is
            // rejected because the relative target resolves to the workspace.
            ("echo $(tee /tmp/ok > ws_out)", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 1.3 acceptance: double-quoted substitutions ────────────────

    /// bash executes `$(...)`/backticks inside double quotes
    /// (`echo "$(touch f)"` runs `touch f`), so writes hidden in
    /// double-quoted substitutions must be rejected (QA round: the guard
    /// previously skipped all double-quoted content, leaving this spelling
    /// of the phase-1.3 bypass open). Literal quoted text (no substitution)
    /// and escaped/single-quoted `$(` stay allowed.
    #[test]
    fn double_quoted_substitution_acceptance() {
        let cases = [
            // Flat double-quoted substitution with a workspace write — blocked.
            ("echo \"$(touch workspace_file)\"", false),
            ("echo \"$(echo hi > workspace_file)\"", false),
            ("echo \"`touch workspace_file`\"", false),
            ("echo \"$(rm -f workspace_file)\"", false),
            ("echo \"$(rm -f /tmp/scratch_file)\"", true),
            // Nested double-quoted substitution — blocked at the inner level.
            ("echo \"$(echo $(touch nested))\"", false),
            ("echo \"$(echo hi > $(echo workspace_file))\"", false),
            // Temp-targeted double-quoted substitution — allowed.
            ("echo \"$(echo hi > /tmp/out)\"", true),
            ("echo \"$(touch /tmp/scratch)\"", true),
            // Plain double-quoted substitution (pure read) — allowed.
            ("echo \"$(echo hi)\"", true),
            ("echo \"$(ls 2>&1)\"", true),
            // Literal double-quoted text with a `>` — not a redirect.
            ("echo \"a > b\"", true),
            // Outside-quote redirect after a double-quoted substitution.
            ("echo \"$(echo hi)\" > /tmp/out", true),
            ("echo \"$(echo hi)\" > workspace_file", false),
            // Escaped `\$(` inside double quotes is literal — allowed.
            ("echo \"abc\\$(touch workspace_file)\"", true),
            // Escaped backtick inside double quotes is literal — allowed.
            ("echo \"abc\\`touch workspace_file\\`\"", true),
            // Single-quoted `$(` is literal — allowed.
            ("echo '$(touch workspace_file)'", true),
            // Backtick substitution spanning a quote-aware body.
            ("echo \"`echo hi; echo hi2`\"", false), // backticks — fail-closed (use $())
            // Double-quoted substitution with an internal cd — state applies.
            ("echo \"$(cd /tmp; touch rel)\"", true),
            ("echo \"$(cd /etc; touch rel)\"", false),
            // Poisoned export in a prior segment applies to a double-quoted
            // substitution.
            ("export TMPDIR=/etc\necho \"$(touch $TMPDIR/x)\"", false),
            (
                "export TMPDIR=/__mahbot_readonly_test_ws__\necho \"$(echo hi > $TMPDIR/out)\"",
                false,
            ),
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\necho \"$(touch $TMPDIR/x)\"",
                true,
            ),
            // Herestring content expands substitutions — a write in a
            // double-quoted herestring word must be caught.
            ("cat <<< \"$(touch workspace_file)\"", false),
            ("cat <<< \"$(echo hi)\"", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 1.3×2 integration: substitution × cd/export state ──────────

    /// Substitutions must be validated with the state at their segment's
    /// position — prior segments' `cd`/export bindings apply (reviewer round:
    /// the upfront whole-string scan validated every substitution against the
    /// initial state, leaving the poisoned-export bypass open and rejecting
    /// legit temp writes after a `cd`).
    #[test]
    fn substitution_state_acceptance() {
        let cases = [
            // Export-poisoning bypass: `export TMPDIR=/etc` in a prior segment
            // must poison expansions inside a later substitution (phase 1.3
            // "verifiably closed").
            ("export TMPDIR=/etc\necho $(touch $TMPDIR/x)", false),
            ("export TMPDIR=/etc\ncat $(echo hi > $TMPDIR/out)", false),
            // Same via the workspace-root spelling.
            (
                "export TMPDIR=/__mahbot_readonly_test_ws__\necho $(echo hi > $TMPDIR/out)",
                false,
            ),
            // Env-prefix binding in a prior segment poisons a later
            // substitution (env-prefix form).
            ("TMPDIR=/etc true\necho $(touch $TMPDIR/x)", false),
            // Poison cleared by a temp-root rebind before the substitution.
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\necho $(touch $TMPDIR/x)",
                true,
            ),
            // cd to temp + relative write inside a substitution — allowed
            // (false-positive fix: the upfront scan rejected this against the
            // workspace-root CWD).
            ("cd /tmp && echo $(touch rel)", true),
            ("cd /tmp && echo $(echo hi > out.txt)", true),
            // cd to a non-temp dir + relative write inside a substitution —
            // blocked.
            ("cd /etc && echo $(touch rel)", false),
            // Substitution-internal `;` must not fragment the substitution:
            // a `cd` inside the subshell applies to the rest of the
            // substitution content.
            ("echo $(cd /tmp; touch rel)", true),
            ("echo $(cd /etc; touch rel)", false),
            // Export inside a substitution poisons only the substitution's
            // own environment (snapshot semantics), not the outer command.
            ("echo $(export TMPDIR=/etc; touch $TMPDIR/x)", false),
            (
                "echo $(export TMPDIR=/etc; echo hi)\ntouch $TMPDIR/outer",
                true,
            ),
            // Mutator inside a substitution after a temp cd stays blocked
            // when it targets the workspace (relative to the subshell CWD).
            ("cd /tmp && echo $(touch ws_file)", true),
            // A workspace write AFTER a substitution is still caught.
            ("cd /tmp && echo $(touch rel) && touch ws_file", true),
            ("echo $(echo hi) && touch ws_file", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: redirect-target delimiters ───────────────────

    #[test]
    fn redirect_delimiter_acceptance() {
        let cases = [
            // `)` terminates a redirect target (suppressed-stderr read).
            ("echo $(cmd 2>/dev/null)", true),
            ("echo $(cmd 2>/dev/null) done", true),
            // Redirect to /etc inside a substitution — blocked.
            ("echo $(echo x > /etc/passwd)", false),
            // Temp-file redirect ending in `)` — allowed.
            ("echo $(echo x > /tmp/out)", true),
            // Relative-file redirect ending in `)` — blocked.
            ("echo $(echo x > out.txt)", false),
            // `}` terminates a redirect target too (brace-closed groups).
            ("echo $(cmd 2>/dev/null} done", false), // malformed substitution — parser error, fail-closed
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: newline as command separator ─────────────────

    #[test]
    fn newline_separator_acceptance() {
        let cases = [
            // Multi-line temp setup script: each line parses as its own command.
            ("touch /tmp/a\necho hi > /tmp/b\ncat /tmp/a", true),
            // Workspace write across a newline — blocked.
            ("echo hi\ntouch workspace_file", false),
            ("touch workspace_file\necho hi", false),
            // Heredoc body with mutator-shaped text is not a command.
            ("cat <<EOF\nrm -rf /tmp\nEOF\necho done", true),
            // Backslash-newline continuation joins the logical line.
            ("echo hello \\\nworld", true),
            ("touch /tmp/a \\\n/tmp/b", true),
            // Quote-aware newline: quoted newline is not a separator.
            ("echo 'a\nb'", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: variable/tilde/quote expansion ───────────────

    #[test]
    fn expansion_acceptance() {
        let cases = [
            // Temp writes via env vars and quoted env vars — allowed.
            ("touch $TMPDIR/out.txt", true),
            ("touch \"$TMPDIR/out.txt\"", true),
            ("touch ${TMPDIR}/out.txt", true),
            ("echo hi > $TMPDIR/out", true),
            ("echo hi > \"$TMPDIR/out\"", true),
            ("tee $TMPDIR/scratch.log", true),
            ("mkdir -p $TMPDIR/scratch_dir", true),
            ("rm $TMPDIR/scratch.txt", true),
            ("sed -i 's/a/b/' /tmp/file", true),
            ("dd of=$TMPDIR/out bs=1 count=1", true),
            ("curl -o $TMPDIR/file URL", true),
            // $PWD resolves to the workspace (initial CWD) — blocked.
            ("touch $PWD/out.txt", false),
            ("echo hi > $PWD/out", false),
            // Unknown/unset variables — blocked.
            ("touch $FOO/out.txt", false),
            ("touch $TMP/out.txt", false), // TMP unset in the session env
            // Tilde / $HOME — never under temp — blocked.
            ("touch ~/out.txt", false),
            ("touch $HOME/out.txt", false),
            // Single-quoted (unexpanded) var — blocked.
            ("touch '$TMPDIR/out.txt'", false),
            // Unbalanced quotes — blocked.
            ("touch \"$TMPDIR/out.txt", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: export poisoning ─────────────────────────────

    #[test]
    fn export_poisoning_acceptance() {
        let cases = [
            // Binding TMPDIR to a non-temp path poisons it (export form).
            ("export TMPDIR=/etc\necho hi > $TMPDIR/out", false),
            // Env-prefix form poisons too.
            ("TMPDIR=/etc touch $TMPDIR/out", false),
            // Plain assignment segment poisons.
            ("TMPDIR=/etc\ntouch $TMPDIR/out", false),
            // Re-binding to a valid temp root clears the poison.
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\ntouch $TMPDIR/out",
                true,
            ),
            // Env-prefix rebind to temp clears.
            ("TMPDIR=/etc\nTMPDIR=/tmp touch $TMPDIR/out", true),
            // Binding TMP (initially unset) to temp enables it.
            ("export TMP=/tmp\ntouch $TMP/out", true),
            // Bare export / export NAME are no-ops.
            ("export\ntouch $TMPDIR/out", true),
            ("export TMP\ntouch $TMPDIR/out", true),
            // Quoted value in export.
            ("export TMPDIR=\"/tmp\"\ntouch $TMPDIR/out", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: copy vs move/delete semantics ────────────────

    #[test]
    fn copy_move_acceptance() {
        let cases = [
            // cp: only the destination must be under temp.
            ("cp /etc/passwd /tmp/out", true),
            ("cp /tmp/a /tmp/b", true),
            ("cp -t /tmp/d s1 s2", true),
            ("cp --target-directory /tmp/d s1 s2", true),
            ("cp --target-directory=/tmp/d s1", true),
            // cp into the workspace stays blocked.
            ("cp /etc/passwd ws_file", false),
            ("cp /tmp/a /etc/passwd", false),
            ("cp -t /workspace/d s1", false),
            ("cp", false),
            // mv/rm: all args must be under temp (move from workspace to temp
            // deletes a source → blocked).
            ("mv /tmp/a /tmp/b", true),
            ("mv ws_file /tmp/out", false),
            ("rm /tmp/scratch.txt", true),
            ("rm ws_file", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: current-directory tracking ───────────────────

    #[test]
    fn cwd_tracking_acceptance() {
        let cases = [
            // cd to temp + relative write — allowed.
            ("cd /tmp && touch f", true),
            ("cd /tmp && echo hi > out.txt", true),
            ("cd /tmp && tee scratch.log", true),
            // cd to a non-temp dir then relative write — blocked.
            ("cd /tmp && cd /etc && touch f", false),
            ("cd / && touch f", false),
            // Relative cd under a tracked temp cwd tracks iff the normalized
            // result stays under a temp root AND the directory exists or was
            // created by `mkdir` earlier in the chain (a nonexistent target
            // fails the `cd` at runtime, so tracking its deeper path would let
            // a `..` chained write escape the temp root).
            ("mkdir -p /tmp/src && cd /tmp && cd src && touch f", true),
            (
                "mkdir -p /tmp/a/b/c && cd /tmp && cd a/b/c && echo hi > out.txt",
                true,
            ),
            ("cd /tmp && cd .. && touch f", false),
            ("cd /tmp && cd ../etc && touch f", false),
            ("cd /tmp && cd ../../../etc && touch f", false),
            ("cd /tmp && cd /etc && cd src && touch f", false),
            // Nonexistent relative targets fail the `cd` at runtime — the
            // chained write lands one level shallower, where a `..` chain
            // escapes the temp root. Fail closed in every chaining spelling.
            ("cd /tmp && cd a/b/c && touch ../x", false),
            ("cd /tmp ; cd a/b/c ; touch ../x", false),
            ("cd /tmp\ncd a/b/c\ntouch ../x", false),
            (
                "mkdir -p /tmp/a/b/c && cd /tmp && cd a/b/c && touch ../x",
                true,
            ),
            ("cd .. && touch f", false),
            ("cd - && touch f", false),
            ("cd && touch f", false),
            ("cd $TMPDIR && touch f", false),
            ("cd ~ && touch f", false),
            // pushd/popd always reset.
            ("cd /tmp && pushd /tmp && touch f", false),
            ("cd /tmp && popd && touch f", false),
            // cd to nonexistent dir + write — blocked (non-temp target).
            ("cd /nonexistent_dir_xyz_1059 && touch f", false),
            // cd to a nonexistent temp dir fails closed in every chaining
            // spelling: at runtime the cd fails and the write lands in the
            // real CWD (the workspace). The dir must exist at validation time
            // or have been created by `mkdir` earlier in the same chain.
            ("cd /tmp/nonexistent_x && touch f", false),
            ("cd /tmp/nonexistent_x ; touch f", false),
            ("cd /tmp/nonexistent_x\ntouch f", false),
            ("cd /tmp/nonexistent_x || touch f", false),
            ("mkdir -p /tmp/snap && cd /tmp/snap && touch f", true),
            ("mkdir -p /tmp/snap; cd /tmp/snap; touch f", true),
            ("mkdir -p /tmp/snap\ncd /tmp/snap\ntouch f", true),
            ("mkdir /tmp/snap; cd /tmp/snap; touch f", true),
            // mkdir -p records ancestors too.
            ("mkdir -p /tmp/a/b; cd /tmp/a; touch f", true),
            // Non-`-p` mkdir into a missing parent errors at runtime — the
            // target is not provably created, so the chained cd/write fails
            // closed in every spelling.
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; touch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y\ncd /tmp/absent_parent_x/dir_y\ntouch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y || cd /tmp/absent_parent_x/dir_y || touch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; rm -rf x",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; tee out",
                false,
            ),
            // A non-`-p` target whose parent was created earlier in the chain
            // (transitively) is provable.
            (
                "mkdir -p /tmp/chain_a; mkdir /tmp/chain_a/chain_b; cd /tmp/chain_a/chain_b; touch f",
                true,
            ),
            // Relative cd targets are expanded before tracking: a target that
            // expands to a non-temp location fails closed.
            ("cd /tmp && cd $TMPDIR && touch f", true),
            ("cd /tmp && cd $HOME && touch f", false),
            ("cd /tmp && cd ~ && touch f", false),
            ("cd /tmp && cd - && touch f", false),
            ("cd /tmp && cd $OLDPWD && rm -rf x", false),
            ("cd /tmp && cd $OLDPWD && touch f", false),
            ("cd /tmp && cd $FOO && touch f", false),
            // $PWD after tracked cd resolves to the tracked CWD.
            ("cd /tmp && touch $PWD/out", true),
            ("cd /tmp && touch $PWD/../../etc/passwd", false),
            // A nonexistent relative target must not be tracked: `$PWD` would
            // then resolve one level deeper than the real CWD.
            ("cd /tmp ; cd src ; touch $PWD/../etc/passwd", false),
            (
                "mkdir -p /tmp/src && cd /tmp && cd src && touch $PWD/out",
                true,
            ),
            // cd to workspace + rm — blocked.
            ("cd / && rm f", false),
            // Relative path args without cd resolve to the workspace — blocked.
            ("touch f", false),
            // Relative globs stay rejected even under a temp CWD.
            ("cd /tmp && touch *.log", false),
            // `..`-normalized escape from a temp CWD — blocked.
            ("cd /tmp && echo hi > ../etc/passwd", false),
            // cp destination "." = current dir (workspace → blocked, temp → allowed).
            ("cp /tmp/a .", false),
            ("cd /tmp && cp /tmp/a .", true),
            // cd option flags are skipped before resolving the target.
            ("cd /tmp && cd -P /tmp && touch f", true),
            ("cd /tmp && cd -L /tmp && echo hi > out.txt", true),
            ("cd /tmp && cd -L -- /tmp && touch f", true),
            ("mkdir -p /tmp/snap && cd -P /tmp/snap && touch f", true),
            ("cd /tmp && cd -P /tmp/nonexistent_x && touch f", false),
            // Prefixed cd forms (`command`/`builtin`/`time`/`eval`) run the cd
            // in the current shell — routed and tracked like a bare cd.
            ("cd /tmp && command cd /tmp && touch f", true),
            ("cd /tmp && builtin cd /tmp && touch f", true),
            (
                "mkdir -p /tmp/snap && cd /tmp && time cd /tmp/snap && touch f",
                true,
            ),
            ("cd /tmp && eval cd /tmp && echo hi > out.txt", true),
            ("cd /tmp && command cd -L -- /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    #[test]
    fn cd_flag_forms_fail_closed() {
        let cases = [
            // Option flags were parsed as the cd target (tracking `/tmp/-P`
            // etc.) while the runtime cd lands in /etc, $HOME, or $OLDPWD —
            // the chained write escapes every temp root.
            ("cd /tmp && cd -P /etc && touch f", false),
            ("cd /tmp && cd -L /etc && rm -rf x", false),
            ("cd /tmp && cd -PL /etc && touch f", false),
            ("cd /tmp && cd -P && touch f", false),
            ("cd /tmp && cd -L && touch f", false),
            ("cd /tmp && cd -PL && touch f", false),
            ("cd /tmp && cd -- $HOME && touch f", false),
            ("cd /tmp && cd -- - && touch f", false),
            ("cd /tmp && cd -- - && rm -rf x", false),
            ("cd /tmp && cd -P - && touch f", false),
            // Invalid options error at runtime (the cd never happens), so
            // tracking them would approve a write that lands in the real CWD.
            ("cd /tmp && cd -e /tmp && touch f", false),
            ("cd /tmp && cd -eP /tmp && touch f", false),
            ("cd /tmp && cd -Pe /tmp && touch f", false),
            ("cd /tmp && cd -@ /tmp && touch f", false),
            ("cd /tmp && cd -x /tmp && touch f", false),
            ("cd /tmp && cd -P-L /tmp && touch f", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: git read-only operations ─────────────────────

    #[test]
    fn git_read_only_acceptance() {
        let cases = [
            // Allowed read forms (phase 3).
            ("git stash show", true),
            ("git stash show stash@{0}", true),
            ("git show-ref", true),
            ("git ls-remote", true),
            ("git submodule status", true),
            ("git submodule status --recursive", true),
            ("git submodule", true),
            ("git config user.name", true),
            ("git config --global user.name", true),
            ("git config --list", true),
            ("git config -l", true),
            ("git config --get user.name", true),
            ("git config --get-all core.pager", true),
            ("git config --name-only --get-regexp '^core\\.'", true),
            ("git rebase --show-current", true),
            ("git push --dry-run", false), // push is an unconditional reject
            ("git push -n", false),        // push is an unconditional reject
            ("git push -n origin main", false), // push is an unconditional reject
            ("git clean -n", false),       // clean is an unconditional reject
            ("git clean --dry-run", false), // clean is an unconditional reject
            ("git clean -ndx", false),     // clean is an unconditional reject
            ("git --version 2>&1", true),
            ("git status 2>&1", true),
            // --output-indicator-* share the --output prefix but do not write
            ("git diff --output-indicator-new=+", true),
            ("git diff --output-indicator-old=-", true),
            (
                "git diff --output-indicator-new=+ --output-indicator-old=-",
                true,
            ),
            // Mutating siblings stay blocked.
            ("git config user.name Egor", false),
            ("git config --global user.name Egor", false),
            ("git config --add core.pager less", false),
            ("git config --edit", false),
            ("git config --unset user.name", false),
            ("git rebase --continue", false),
            ("git rebase main", false),
            ("git rebase", false),
            ("git push origin main", false),
            ("git push", false),
            ("git push --dry-run -f", false),
            ("git push -fn", false),
            ("git clean -f", false),
            ("git clean -n -f", false),
            ("git clean -fd", false),
            ("git clean", false),
            ("git stash pop", false),
            ("git stash apply", false),
            ("git stash drop", false),
            ("git stash push", false),
            ("git stash", false),
            // Exact stash subcommand matching: a mutation whose message
            // contains a read-form word must stay blocked (QA round: the
            // substring check allowed `git stash push -m "stash show"`).
            ("git stash push -m \"stash show\"", false),
            ("git stash push -m \"stash list\"", false),
            ("git stash store 0123abcd \"stash show\"", false),
            ("git stash show --stat stash@{0}", true),
            ("git stash list --oneline", true),
            // --output writes a file even on read forms (stash show/list)
            ("git stash show --output=/tmp/x", false),
            ("git stash list --output=/tmp/x", false),
            ("git submodule foreach", false),
            ("git submodule update", false),
            ("git submodule add https://example.com/repo.git", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: cargo read-only invocations ──────────────────

    #[test]
    fn cargo_read_only_acceptance() {
        let cases = [
            // Toolchain specifier no longer becomes the parsed subcommand.
            ("cargo +nightly build", true),
            ("cargo +stable check", true),
            ("cargo +nightly test", true),
            ("cargo +nightly clippy", true),
            ("cargo +nightly doc", true),
            ("cargo +nightly run", false),
            ("cargo +nightly fix", false),
            ("cargo +nightly update", false),
            // stderr-capture suffix no longer becomes the parsed subcommand.
            ("cargo --version 2>&1", true),
            ("cargo build 2>&1", true),
            ("git --version 2>&1", true),
            // Help/version exemption for any subcommand, before `--`.
            ("cargo fix --help", true),
            ("cargo run --help", true),
            ("cargo run -h", true),
            ("cargo nextest --version", true),
            ("cargo --version", true),
            ("cargo build -V", true),
            // update --dry-run allowed; plain update blocked.
            ("cargo update --dry-run", true),
            ("cargo update", false),
            // Still blocked.
            ("cargo fix", false),
            ("cargo run", false),
            ("cargo run -- --help", false),
            ("cargo clippy --fix", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: keep-blocked battery ─────────────────────────

    #[test]
    fn keep_blocked_battery() {
        let cases = [
            // Process control — all forms.
            ("kill 1234", false),
            ("kill -9 1234", false),
            ("pkill -f mahbot", false),
            ("killall chrome", false),
            // ln in all forms.
            ("ln -s /tmp/a /tmp/b", false),
            ("ln /tmp/a /tmp/b", false),
            // unzip extract.
            ("unzip archive.zip", false),
            ("unzip -o archive.zip -d /tmp/out", false),
            // Live-DB sidecar deletion (tilde / $HOME spellings).
            ("rm ~/.mahbot/db/board.db-wal", false),
            ("rm $HOME/.mahbot/db/board.db-wal", false),
            // rm -rf target.
            ("rm -rf target", false),
            // cargo mutators.
            ("cargo clippy --fix", false),
            ("cargo run", false),
            ("cargo fix", false),
            // git init even in temp.
            ("git init", false),
            ("git init /tmp/x", false),
            // Non-read-only git mutations.
            ("git commit -m test", false),
            ("git reset --hard", false),
            // sed/awk/dd/curl/wget on workspace targets.
            ("sed -i 's/a/b/' file", false),
            ("awk -i inplace '{print $1}' file", false),
            ("dd if=/dev/zero of=file bs=1 count=1", false),
            ("curl -o out.txt URL", false),
            ("wget -O out.txt URL", false),
        ];

        run_cases(&cases);
    }

    // ── Temp-value variable binding ──────────────────────────────────────

    /// `NAME=$(mktemp -d)` / backtick spellings bind a fresh temp root with an
    /// unknown concrete value; writes via `$NAME` resolve under that root.
    /// Non-temp and unknown bindings poison the variable (fail-closed);
    /// temp re-binds clear the poison.
    #[test]
    fn temp_var_binding_acceptance() {
        let cases = [
            // mktemp substitution binds a temp root; writes via $SNAP allowed.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d)\necho hi > \"$SNAP/out\"", true),
            ("SNAP=$(mktemp -d)\ncp /etc/passwd \"$SNAP/out\"", true),
            ("SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/a/b\"", true),
            ("SNAP=$(mktemp -d)\nrm -rf \"$SNAP\"", true),
            ("SNAP=$(mktemp -d)\nrm -rf \"$SNAP/\"", true),
            // Backtick and quoted-substitution spellings.
            ("SNAP=`mktemp -d`\ntouch \"$SNAP/f\"", false), // backticks — fail-closed (use $())
            ("SNAP=\"$(mktemp -d)\"\ntouch \"$SNAP/f\"", true),
            // export form.
            ("export SNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            // Binding inside a substitution applies within the subshell.
            ("echo $(SNAP=$(mktemp -d); touch \"$SNAP/f\")", true),
            // Braced / unbraced references.
            ("SNAP=$(mktemp -d)\ntouch $SNAP/f", true),
            ("SNAP=$(mktemp -d)\ntouch ${SNAP}/f", true),
            // Concrete temp bindings track too.
            ("SNAP=/tmp/snap\ntouch \"$SNAP/f\"", true),
            ("SNAP=$TMPDIR/snap\ntouch \"$SNAP/f\"", true),
            (
                "SNAP=/tmp/snap\ndir=$SNAP/.mahbot/db\nmkdir -p \"$dir\"",
                true,
            ),
            // Non-temp / unknown bindings poison → blocked.
            ("SNAP=/etc\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=/__mahbot_readonly_test_ws__/snap\nmkdir -p \"$SNAP/db\"",
                false,
            ),
            ("SNAP=$FOO\ntouch \"$SNAP/f\"", false),
            // Temp rebind clears; non-temp rebind stays poisoned.
            ("SNAP=/etc\nSNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d)\nSNAP=/etc\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d)\nSNAP=/tmp/other\ntouch \"$SNAP/f\"",
                true,
            ),
            // Env-prefix form.
            ("SNAP=/tmp/snap touch $SNAP/f", true),
            ("SNAP=/etc touch $SNAP/f", false),
            // Unbound variable without a temp anchor stays blocked.
            ("touch $FOO/f", false),
            // mktemp with a template still binds.
            (
                "SNAP=$(mktemp -d /tmp/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                true,
            ),
            // Positional templates that provably resolve under a temp root
            // bind — including the `-- <template>` form and variables
            // expanding to a temp path.
            (
                "SNAP=$(mktemp -d -- /tmp/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                true,
            ),
            (
                "SNAP=$(mktemp -d \"$TMPDIR/mahbot.XXXXXX\")\ntouch \"$SNAP/f\"",
                true,
            ),
            // Absolute non-temp, $HOME, relative (shell-cwd) and workspace
            // templates fail closed — no temp-root binding, writes via the
            // variable reject.
            (
                "SNAP=$(mktemp -d /etc/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d /__mahbot_readonly_test_ws__/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d $HOME/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            ("SNAP=$(mktemp -d snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d ./snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            // The `-- <template>` form carries the same rules.
            (
                "SNAP=$(mktemp -d -- /etc/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            ("SNAP=$(mktemp -d -- snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d -- $HOME/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A second positional template makes mktemp error at runtime —
            // the variable would bind empty, so no temp anchor.
            (
                "SNAP=$(mktemp -d /tmp/a.XXXXXX /tmp/b.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A template without trailing X's makes mktemp error at runtime —
            // the variable would bind empty, so no temp anchor.
            ("SNAP=$(mktemp -d /tmp/foo)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d /tmp/foo.XX)\ntouch \"$SNAP/f\"", false),
            // mktemp -p/--tmpdir to a provably existing temp dir binds; to a
            // non-temp dir does not. mktemp never creates the -p directory,
            // so a nonexistent one makes it error at runtime and bind the
            // variable empty — the chained write would land outside every
            // temp root (false temp anchor), so it fails closed.
            ("SNAP=$(mktemp -d -p /tmp)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d --tmpdir=/tmp)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d -p /etc)\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d --tmpdir=/__mahbot_readonly_test_ws__)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d -p /tmp/nonexistent_x)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d -p /tmp/nonexistent_x) ; touch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d --tmpdir=/tmp/nonexistent_x)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A -p dir created by `mkdir` earlier in the chain is provable.
            (
                "mkdir -p /tmp/snapdir; SNAP=$(mktemp -d -p /tmp/snapdir)\ntouch \"$SNAP/f\"",
                true,
            ),
            // The space-separated `--tmpdir DIR` form is not provable (GNU
            // optional-arg semantics error there; macOS joins DIR as a
            // template under /tmp) → fail-closed.
            ("SNAP=$(mktemp -d --tmpdir /tmp)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d --tmpdir /etc)\ntouch \"$SNAP/f\"", false),
            // `--suffix=` is GNU-only; macOS mktemp rejects it → fail-closed.
            (
                "SNAP=$(mktemp -d --suffix=.foo /tmp/x.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // The same fail-closed shapes must hold when the initial cwd is a
            // tracked temp dir: the raw `$(mktemp ...)` substitution string
            // must never resolve as a literal path against a tracked temp cwd
            // and bind as a temp value — at runtime mktemp errors, SNAP binds
            // empty, and the chained write lands outside every temp root.
            (
                "cd /tmp; SNAP=$(mktemp -d -p /tmp/nonexistent_x); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp\nSNAP=$(mktemp -d -p /tmp/nonexistent_x)\ntouch $SNAP/f",
                false,
            ),
            (
                "cd /tmp && SNAP=$(mktemp -d -p /tmp/nonexistent_x) && touch $SNAP/f",
                false,
            ),
            ("cd /tmp; SNAP=$(mktemp -d -p /etc); touch $SNAP/f", false),
            (
                "cd /tmp; SNAP=$(mktemp -d /etc/foo.XXXXXX); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir /tmp); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir=/tmp/nonexistent_x); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --suffix=.foo /tmp/x.XXXXXX); touch $SNAP/f",
                false,
            ),
            // Workspace-targeting chains through the unprovable substitution
            // stay blocked.
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); rm -rf $SNAP/Users/egordezic/Desktop/mahbot",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); echo pwn > $SNAP/Users/egordezic/Desktop/mahbot/x",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); tee $SNAP/Users/egordezic/Desktop/mahbot/x",
                false,
            ),
            // A substitution used directly in a path argument is equally
            // unprovable: at runtime mktemp errors, the empty expansion makes
            // the write land at /f or the workspace path.
            ("cd /tmp; touch $(mktemp -d -p /etc)/f", false),
            ("cd /tmp; echo hi > $(mktemp -d -p /etc)/f", false),
            (
                "cd /tmp; rm -rf $(mktemp -d -p /etc)/Users/egordezic/Desktop/mahbot",
                false,
            ),
            ("cd /tmp; touch `mktemp -d -p /etc`/f", false),
            // Provable bindings keep working under a tracked temp cwd.
            ("cd /tmp; SNAP=$(mktemp -d); touch $SNAP/f", true),
            ("cd /tmp; SNAP=$(mktemp -d -p /tmp); touch $SNAP/f", true),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir=/tmp); touch $SNAP/f",
                true,
            ),
            // A single-quoted `$(...)` is a literal, not a substitution: it
            // never binds a temp root. A substitution embedded in a larger
            // value is unprovable too — fail-closed.
            ("SNAP='$(mktemp -d)'\ntouch \"$SNAP/f\"", false),
            (
                "cd /tmp; SNAP=pre$(mktemp -d -p /etc); touch $SNAP/f",
                false,
            ),
            // Parameter expansions with modifiers (`${FOO:-../etc}`) are
            // unprovable: the default can carry `..` and escape the temp
            // root at runtime, so words containing them fail to resolve.
            ("cd /tmp; echo hi > /tmp/${FOO:-../etc}/f", false),
            ("echo hi > /tmp/${FOO:-../etc}/f", false),
            ("cd /tmp; touch /tmp/${FOO//x/../etc}/f", false),
            ("cd /tmp; echo hi > \"/tmp/${FOO:-../etc}/f\"", false),
        ];

        run_cases(&cases);
    }

    /// Opaque suffixes (`$$`, `$RANDOM`) under a temp prefix resolve; without
    /// a temp anchor, or after an escape segment, they stay fail-closed. An
    /// arbitrary unbound variable is never treated as an opaque suffix —
    /// only the `$RANDOM` builtin is.
    #[test]
    fn opaque_suffix_acceptance() {
        let cases = [
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/f\"", true),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$$/f\"", true),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM\"", true),
            ("SNAP=$(mktemp -d)\necho hi > \"$SNAP/$RANDOM/out\"", true),
            ("touch \"/tmp/$RANDOM/f\"", true),
            // Without a temp anchor the opaque variable is unprovable.
            ("touch \"$RANDOM/f\"", false),
            // Arbitrary unbound variables fail closed even after a temp anchor.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$FOO/f\"", false),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$PATH/f\"", false),
            // `..` after an opaque suffix is unverifiable → blocked.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/../f\"", false),
            (
                "SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/../../etc/passwd\"",
                false,
            ),
            // `..` before the opaque suffix normalizes concretely → allowed.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/../$RANDOM/f\"", true),
        ];

        run_cases(&cases);
    }

    /// The documented snapshot-query procedure (idiomatic and literal
    /// spellings) passes; workspace-targeting variants of the same shapes stay
    /// blocked.
    #[test]
    fn snapshot_query_procedure_acceptance() {
        let cases = [
            // Idiomatic spelling.
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/.mahbot/db\"\ncp ~/.mahbot/db/board.db \"$SNAP/.mahbot/db/\"\nHOME=\"$SNAP\" mahbot debug --db board \"SELECT 1\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Full for-loop spelling from the deleted ops doc.
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/.mahbot/db\"\nfor db in ~/.mahbot/db/*.db; do\ncp \"$db\" \"$SNAP/.mahbot/db/\"\ncp \"$db-wal\" \"$SNAP/.mahbot/db/\" 2>/dev/null || true\ndone\nHOME=\"$SNAP\" mahbot debug --db sessions \"SELECT COUNT(*) FROM messages\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Literal spelling: concrete temp bindings.
            (
                "SNAP=/tmp/mahbot-snap\ndir=$SNAP/.mahbot/db\nmkdir -p \"$dir\"\ncp ~/.mahbot/db/board.db \"$dir/\"\nHOME=\"$SNAP\" mahbot debug --db board \"SELECT 1\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Workspace-targeting variants of the same shapes stay blocked.
            (
                "SNAP=/__mahbot_readonly_test_ws__/snap\nmkdir -p \"$SNAP/.mahbot/db\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d)\ncp ~/.mahbot/db/board.db /__mahbot_readonly_test_ws__/out",
                false,
            ),
            (
                "SNAP=$(mktemp -d)\nrm -rf /__mahbot_readonly_test_ws__",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// Denial messages teach the recognized spelling: a literal temp path or
    /// `NAME=$(mktemp -d)` for unresolvable variables, and the help form for
    /// `cargo run` without bypass guidance.
    #[test]
    fn denial_message_education() {
        let ctx = test_ctx();
        let err = check_command("touch $FOO/out", &ctx).unwrap_err();
        assert!(
            err.contains("$(mktemp -d)"),
            "scratch-mutator denial should teach the variable spelling: {err}"
        );
        let err = check_command("rm $FOO/out", &ctx).unwrap_err();
        assert!(
            err.contains("$(mktemp -d)"),
            "temp-mutator denial should teach the variable spelling: {err}"
        );
        let err = check_command("cargo run", &ctx).unwrap_err();
        assert!(
            err.contains("--help"),
            "cargo run denial should suggest the help form: {err}"
        );
        for banned in ["target/debug", "built binary", "full shell", "escalate"] {
            assert!(
                !err.contains(banned),
                "cargo run denial must not teach a bypass: {err}"
            );
        }
    }

    #[test]
    fn symlink_escape_fails_closed() {
        // A symlink inside a temp root pointing at an existing non-temp dir:
        // writes through it land outside every temp root. The target must
        // exist for canonicalize to resolve through it; `/etc` is a stable
        // non-temp anchor that does not depend on the test process cwd (the
        // checkout can itself live under a temp root in CI/dev setups).
        let target = std::path::PathBuf::from("/etc");
        let link = std::env::temp_dir().join(format!("mahbot_ro_probe_{}", std::process::id()));
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let link_str = link.to_string_lossy().into_owned();
        let cases = [
            (format!("cd {link_str} && touch f"), false),
            (format!("cd {link_str} ; touch f"), false),
            (format!("touch {link_str}/f"), false),
            (format!("rm -rf {link_str}/x"), false),
            (format!("echo hi > {link_str}/out.txt"), false),
            (format!("tee {link_str}/x"), false),
            (format!("cp /tmp/a {link_str}/dest"), false),
        ];
        for (command, allowed) in &cases {
            if *allowed {
                ok(command);
            } else {
                assert_rejected(command);
            }
        }
        let _ = std::fs::remove_file(&link);
    }

    #[test]
    fn prefixed_cd_and_dotdot_escapes_fail_closed() {
        let cases = [
            // Prefixed cd forms (`command`/`builtin`/`time`/`eval`) run the cd
            // in the current shell; a non-temp target must reset fail-closed.
            ("cd /tmp && command cd /etc && touch f", false),
            ("cd /tmp && command cd -- /etc && touch f", false),
            ("cd /tmp && builtin cd $HOME && touch f", false),
            ("cd /tmp && builtin cd -P /etc && touch f", false),
            ("cd /tmp && time cd ~ && touch f", false),
            ("cd /tmp && eval cd \"$HOME\" && touch f", false),
            (
                "cd /tmp && command cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            (
                "cd /tmp && builtin cd /__mahbot_readonly_test_ws__ && touch f",
                false,
            ),
            // `..`-collapsed missing-prefix targets (mkdir-created tail).
            (
                "mkdir -p /tmp/qa_b && cd /tmp/qa_a/../qa_b && touch ../x",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && cd qa_a/../qa_b && touch ../x",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && SNAP=$(mktemp -d -p qa_a/../qa_b) && touch $SNAP/f",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && SNAP=$(mktemp -d qa_a/../qa_b.XXXXXX) && touch $SNAP/f",
                false,
            ),
            // The `..`-collapsed shape via the -P flag path.
            (
                "mkdir -p /tmp/qa_b && cd /tmp && cd -P qa_a/../qa_b && touch ../x",
                false,
            ),
            // Prefixed cd to a temp target stays accepted (routed, tracks).
            ("cd /tmp && command cd /tmp && touch f", true),
            ("cd /tmp && builtin cd -P /tmp && touch f", true),
            // Prefix execution options and quoted verbs route and track.
            ("cd /tmp && command -p cd /tmp && touch f", true),
            ("cd /tmp && time -p cd /tmp && touch f", true),
            ("cd /tmp && command \"cd\" /tmp && touch f", true),
            ("cd /tmp && eval 'cd /tmp' && touch f", true),
            ("cd /tmp && eval \"cd /tmp\" && touch f", true),
            // Informational/invalid prefix options never execute the cd, so
            // they are not routed — tracking stays where the real CWD is.
            ("cd /tmp && command -v cd /tmp && touch f", true),
            ("cd /tmp && builtin -p cd /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    #[test]
    fn quoted_eval_and_brace_escapes_fail_closed() {
        let cases = [
            // Quoted eval bodies run in the current shell: a cd inside moves
            // the real CWD behind the guard — fail closed.
            ("cd /tmp && eval 'cd /etc' && touch f", false),
            ("cd /tmp && eval \"cd /etc\" && touch f", false),
            ("cd /tmp && eval \"cd $HOME\" && touch f", false),
            ("cd /tmp && command eval 'cd /etc' && touch f", false),
            ("cd /tmp && time eval 'cd /etc' && touch f", false),
            ("cd /tmp && builtin eval 'cd /etc' && touch f", false),
            ("cd /tmp && eval 'cd /etc' && rm -rf x", false),
            // eval bodies with cd + extra commands reset fail-closed.
            ("cd /tmp && eval 'cd /etc; echo hi' && touch f", false),
            ("cd /tmp && eval 'cd /tmp && cd /etc' && touch f", false),
            // A `;` inside the body is a real separator — the body validates
            // as a list: the cd tracks and the temp-scoped touch stays
            // allowed (matches bash — the outer `touch g` lands in /tmp).
            ("cd /tmp && eval 'cd /tmp; touch f' && touch g", true),
            // Quoted eval bodies WITHOUT a cd verb are validated as command
            // text (the body executes in the current shell — a mutator
            // inside rejects).
            ("eval 'echo hi'", true),
            ("eval 'ls'", true),
            ("eval \"echo hi\"", true),
            ("eval 'touch /tmp/x'", true),
            ("eval 'rm -rf /tmp/x'", true),
            ("eval 'touch f'", false),
            ("eval 'echo hi && touch f'", false),
            ("eval 'cd /tmp; rm -rf /tmp/x'", true),
            // UNQUOTED eval bodies are visible and validated normally.
            ("eval echo hi", true),
            ("eval touch /tmp/x", true),
            ("eval touch f", false),
            ("eval rm -rf /__mahbot_readonly_test_ws__", false),
            // Brace groups run in the current shell: a cd inside changes the
            // real CWD without tracking, and a mutator inside writes
            // unvalidated — both fail closed.
            ("cd /tmp && { cd /etc; } && touch f", false),
            ("cd /tmp && { cd /etc } && touch f", false),
            ("cd /tmp && { cd /etc;\n} && touch f", false),
            ("cd /tmp && { cd /etc; } && rm -rf x", false),
            ("{ touch f; }", false),
            ("cd /tmp && { cd /tmp; } && touch f", false),
        ];

        run_cases(&cases);
    }

    #[test]
    fn prefix_option_and_quoted_verb_fail_closed() {
        let cases = [
            // `command -p` / `time -p` execute the cd in the current shell —
            // a non-temp target must reset fail-closed.
            ("cd /tmp && command -p cd /etc && touch f", false),
            ("cd /tmp && command -p cd -P /etc && touch f", false),
            ("cd /tmp && time -p cd /etc && touch f", false),
            ("cd /tmp && time -p cd ~ && touch f", false),
            (
                "cd /tmp && command -p cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            // Quoted cd/pushd verbs still execute in the current shell.
            ("cd /tmp && command \"cd\" /etc && touch f", false),
            ("cd /tmp && builtin 'cd' $HOME && touch f", false),
            ("cd /tmp && time \"cd\" /etc && touch f", false),
            ("cd /tmp && command 'pushd' /etc && touch f", false),
            ("\"cd\" /etc && touch f", false),
            (
                "cd /tmp && \"cd\" /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// Composed forwarding prefixes (`command builtin cd`, `time command cd`,
    /// `builtin -- command cd`, ...) execute the forwarded verb in the current
    /// shell — a cd hidden behind composed prefixes routes through cd tracking
    /// (fail-closed for non-temp targets) instead of falling through as an
    /// unmodeled non-mutator. Informational composed forms never execute.
    #[test]
    fn composed_forwarding_prefixes_fail_closed() {
        let cases = [
            // cd behind two forwarding prefixes — tracking updates/resets.
            ("cd /tmp && command builtin cd /etc && touch f", false),
            ("cd /tmp && command command cd /etc && touch f", false),
            ("cd /tmp && builtin command cd /etc && touch f", false),
            ("cd /tmp && command -p builtin cd /etc && touch f", false),
            ("cd /tmp && time command cd /etc && touch f", false),
            ("cd /tmp && time builtin cd /etc && touch f", false),
            ("cd /tmp && builtin -- command cd /etc && touch f", false),
            // Three levels.
            (
                "cd /tmp && command builtin command cd /etc && touch f",
                false,
            ),
            // Workspace-target variant.
            (
                "cd /tmp && command builtin cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            // Temp target behind composed prefixes tracks — write allowed.
            ("cd /tmp && command builtin cd /tmp && touch f", true),
            ("time command cd /tmp && touch f", true),
            // Combined -p flags still forward through the composition.
            ("cd /tmp && command -pp builtin cd /etc && touch f", false),
            // Informational composed forms never execute — tracking unchanged.
            ("cd /tmp && command -v builtin && touch f", true),
            ("cd /tmp && command builtin -v cd && touch f", true),
            ("cd /tmp && builtin command -v cd && touch f", true),
            ("cd /tmp && time command -v cd && touch f", true),
            // An env-assignment word after `command`/`builtin` is the COMMAND
            // name (`command FOO=bar cd /tmp` errors 127 — the cd never runs),
            // so tracking stays unchanged and a chained write fails closed.
            ("command FOO=bar cd /tmp; touch f", false),
            ("builtin FOO=bar cd /tmp; touch f", false),
            ("command -- FOO=bar cd /tmp; touch f", false),
            ("command -p FOO=bar cd /tmp; touch f", false),
            ("command -pp FOO=bar cd /tmp; touch f", false),
            ("command builtin FOO=bar cd /tmp; touch f", false),
            ("command -pp builtin FOO=bar cd /tmp; rm -rf x", false),
            ("command -p FOO=bar cd /tmp; rm -rf x", false),
            // `time` takes a full command list: the env assignment is valid
            // and the timed cd runs (tracked to the temp dir).
            ("time FOO=bar cd /tmp && touch f", true),
            ("time FOO=bar cd /etc && touch f", false),
            ("time -p FOO=bar cd /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `time` is a forwarding prefix only as the UNQUOTED first word of a
    /// full segment, outside a `!`-negated list (bash 3.2 probe-verified):
    /// quoted `"time"`/`'time'`, an env assignment before it, a
    /// `command`/`builtin` before it, a nested `time time`, and `! time` all
    /// resolve to the external `time` command whose timed command runs in a
    /// child — the guard must not track a cwd the shell never reaches
    /// (fail-closed: a chained relative write against the workspace rejects).
    #[test]
    fn time_prefix_requires_head_position_and_unquoted() {
        let cases = [
            // Quoted `time` — P0 shape: the timed cd runs in a child, the
            // real CWD stays in the workspace, so a chained relative write
            // must reject against it.
            ("\"time\" cd /tmp && touch f", false),
            ("'time' cd /tmp && touch f", false),
            ("\"time\" cd /tmp; rm -rf x", false),
            // Bare quoted segment is allowed-but-untracked (nothing runs in
            // the workspace); P0 protection comes from the chained-write
            // rejection above.
            ("\"time\" cd /tmp", true),
            ("'time' cd /tmp", true),
            ("\"time\" cd /tmp && cd /tmp && touch f", true),
            // The shared-path quote-normalization still surfaces mutators
            // behind quoted `time` to the blocklist dispatch (external time
            // executes its command).
            ("\"time\" rm -rf /__mahbot_readonly_test_ws__", false),
            ("\"time\" git commit -m test", false),
            // The `cd /tmp && "time" cd /etc && touch f` shape flips from
            // REJECT to ALLOW: the first cd tracks /tmp, the quoted-time cd
            // runs in a child (no tracking change), and the write resolves
            // against the tracked /tmp — matching the real shell (the
            // pre-fix REJECT came from buggy /etc tracking).
            ("cd /tmp && \"time\" cd /etc && touch f", true),
            ("cd /tmp && 'time' cd /etc && touch f", true),
            // Composed prefixes demote `time` to the external command.
            ("command time cd /tmp && touch f", false),
            ("builtin time cd /tmp && touch f", false),
            ("command -- time cd /tmp && touch f", false),
            ("command time cd /etc && touch f", false),
            ("time time cd /tmp && touch f", false),
            ("command time export SNAP=/tmp; cd $SNAP; touch f", false),
            // Env assignment before `time` demotes it to external.
            ("TMPDIR=/tmp time cd /tmp && touch f", false),
            ("TMPDIR=/tmp time cd /etc && touch f", false),
            ("TMPDIR=/tmp time export SNAP=/tmp; touch $SNAP/f", false),
            (
                "TMPDIR=/tmp time export SNAP=/tmp; cd $SNAP; touch f",
                false,
            ),
            // `!` negation demotes `time` (external) but not `cd` (current
            // shell — the pinned `! cd /tmp && touch f` ALLOW stays).
            ("! time cd /tmp; touch f", false),
            ("! time cd /tmp; rm -rf x", false),
            ("! time cd /tmp && touch f", false),
            ("! time cd /tmp | cat; touch f", false),
            ("! cd /tmp; touch f", true),
            ("! cd /tmp && touch f", true),
            // Unquoted head-of-list `time` keeps forwarding (current-shell
            // timed command — tracking unchanged).
            ("time cd /tmp && touch f", true),
            ("time cd /etc && touch f", false),
            ("time -p cd /tmp && touch f", true),
            ("time command cd /tmp && touch f", true),
            ("time builtin cd /tmp && touch f", true),
            ("time export SNAP=/tmp; touch $SNAP/f", true),
            (
                "time export SNAP=/tmp && cd /tmp && cd $SNAP && touch f",
                true,
            ),
            ("time FOO=bar cd /tmp && touch f", true),
            ("time -p FOO=bar cd /tmp && touch f", true),
            // Quoted `-p` after unquoted head-of-list `time` is the TIMED
            // COMMAND (`-p: command not found` — nothing executes, no
            // tracking), not the option: a chained write lands in the
            // workspace and must reject. Bare segment is allowed-but-
            // untracked (the command errors, the real CWD is untouched).
            ("time \"-p\" cd /tmp; touch f", false),
            ("time \"-p\" cd /tmp; rm -rf x", false),
            ("time '-p' cd /tmp; touch f", false),
            ("time \"-p\" cd /tmp && touch f", false),
            ("time \"-p\" cd /tmp || touch f", false),
            ("time \"-p\" cd /tmp", true),
            // `'time' '-p'`: quoted time resolves external (child process),
            // and the quoted option would be its timed command either way —
            // no tracking, chained write rejects.
            ("'time' '-p' cd /tmp; touch f", false),
            // Unquoted `-p` option with a quoted timed command that errors.
            ("time -p \"-p\" cd /tmp; touch f", false),
            // Unquoted `-p` forwards a quoted timed command (`"cd"` is the
            // builtin — current shell, tracking unchanged).
            ("time -p \"cd\" /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `time` is the EXTERNAL command at the first word of an
    /// if/elif/while/until condition or case-arm body (probe-verified on
    /// bash 3.2: the timed cd runs in a child, the parent CWD stays put) —
    /// the guard must not track a cwd the shell does not reach, so a write
    /// chained inside the condition rejects against the real CWD. A
    /// separator/newline before the first word returns `time` to the
    /// reserved word (`if \n time cd /tmp` tracks), and bodies keep it
    /// reserved.
    #[test]
    fn time_in_construct_conditions_is_external() {
        let cases = [
            // Condition-head / arm-head time: external — no tracking,
            // chained write rejects against the workspace.
            ("if time cd /tmp; touch f; then :; fi", false),
            ("if time cd /tmp && touch f; then :; fi", false),
            ("if time cd /tmp; rm -rf x; then :; fi", false),
            ("while time cd /tmp; touch f; do :; done", false),
            ("until time cd /tmp; touch f; do :; done", false),
            (
                "if false; then :; elif time cd /tmp; touch f; then :; fi",
                false,
            ),
            ("case x in x) time cd /tmp; touch f;; esac", false),
            // `!` after the keyword demotes too (already negated).
            ("if ! time cd /tmp; touch f; then :; fi", false),
            // A newline before the first word returns `time` to reserved:
            // tracked, chained temp write allowed.
            ("if\n time cd /tmp; touch f; then :; fi", true),
            // After a separator inside the condition / arm, `time` is
            // reserved again — the demotion covers the first unit only.
            ("if true && time cd /tmp && touch f; then :; fi", true),
            ("if cd /tmp && time cd /etc && touch f; then :; fi", false),
            ("case x in x) true && time cd /tmp && touch f;; esac", true),
            // Bodies keep `time` reserved (current-shell timed command).
            ("if true; then time cd /tmp; touch f; fi", true),
            ("if false; then :; else time cd /tmp; touch f; fi", true),
            ("while true; do time cd /tmp; touch f; break; done", true),
        ];

        run_cases(&cases);
    }

    // ── Fail-closed-by-default residuals (unmodeled constructs) ─────────

    /// Concatenated-quote / ANSI-C / escape / substitution-formed cd verbs
    /// reset (chained relative writes reject); the same verb forms for
    /// mutators reject outright. Each residual spelling gets its own case.
    #[test]
    fn unmodeled_verb_forms_fail_closed() {
        let cases = [
            // Concatenated-quote cd verbs — chained writes fail closed.
            ("cd /tmp && \"c\"d /etc && touch f", false),
            ("cd /tmp && \"c\"d /tmp && touch f", false),
            ("cd /tmp && c'd' /etc && touch f", false),
            ("cd /tmp && \"c\"'d' /etc && touch f", false),
            // ANSI-C quoting.
            ("cd /tmp && $'cd' /etc && touch f", false),
            ("cd /tmp && $'cd' /tmp && touch f", false),
            // Escape-formed verb.
            ("cd /tmp && c\\144 /etc && touch f", false),
            // Substitution-formed verb.
            ("cd /tmp && $(printf c)d /etc && touch f", false),
            ("cd /tmp && $(printf c)d /tmp && touch f", false),
            // Concatenated-quote pushd.
            ("cd /tmp && \"p\"ushd /etc && touch f", false),
            // Concatenated-quote mutators — rejected outright.
            ("\"r\"m -rf /__mahbot_readonly_test_ws__", false),
            ("\"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"c\"p /etc/passwd /__mahbot_readonly_test_ws__/out", false),
            ("\"g\"it commit -m x", false),
            ("\"m\"kdir /__mahbot_readonly_test_ws__/x", false),
            ("$'rm' -rf /__mahbot_readonly_test_ws__", false),
            ("r\\155 -rf /__mahbot_readonly_test_ws__", false),
            ("$(printf r)m -rf /__mahbot_readonly_test_ws__", false),
            // Quoted mutators inside eval bodies — rejected outright.
            ("eval '\"t\"ouch /__mahbot_readonly_test_ws__/x'", false),
            ("eval '\"c\"d /etc' && touch f", false),
            // Fully-quoted mutators execute the unquoted command — the
            // blocklist dispatch normalizes the verb (item 2).
            ("\"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("\"touch\" /__mahbot_readonly_test_ws__/x", false),
            ("\"cp\" /etc/passwd /__mahbot_readonly_test_ws__/out", false),
            ("\"mkdir\" /__mahbot_readonly_test_ws__/x", false),
            ("\"sed\" -i s/a/b/ /__mahbot_readonly_test_ws__/f", false),
            ("\"git\" commit -m x", false),
            ("\"git\" status", true),
            // Prefix-wrapped quoted mutators route through the same
            // normalization (sudo/env/nice/npx forward the command).
            ("sudo \"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("env \"touch\" /__mahbot_readonly_test_ws__/x", false),
            (
                "nice \"cp\" /etc/passwd /__mahbot_readonly_test_ws__/out",
                false,
            ),
            ("npx \"git\" push", false),
            // Fully-quoted PREFIXES are stripped before the blocklist dispatch
            // too — `"env"` forwards like `env`, so the quoted mutator verb
            // behind it must still be found (quote-level-deep variant).
            ("\"env\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"exec\" \"r\"m -rf /__mahbot_readonly_test_ws__", false),
            ("\"npx\" \"g\"it push", false),
            ("\"env\" \"git\" push", false),
            ("\"nice\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"nohup\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"sudo\" \"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("\"env\" \"ls\" /tmp", true),
            ("\"env\" \"git\" status", true),
            // Brace-expansion verbs (`{touch,}` expands to `touch`) cannot be
            // normalized to a single word — unprovable, rejected (item 2).
            ("{touch,} f", false),
            ("{cp,} a b", false),
            ("{rm,} -rf /__mahbot_readonly_test_ws__", false),
            ("{touch,} /tmp/ok", false),
            ("cd /tmp && {touch,} f", false),
            ("{touch,} /tmp/x && echo hi", false),
            // Brace expansion in ARGUMENT position is not a verb — benign
            // commands with brace-expanded args stay allowed.
            ("echo {a,b}", true),
            ("ls {a,b}", true),
            // Fully-quoted benign verbs stay modeled and allowed.
            ("\"ls\" -la /tmp", true),
            ("\"echo\" hi", true),
            ("sudo \"ls\" /tmp", true),
            // Concatenated-quote benign verbs are unprovable — reject.
            ("\"c\"at /etc/passwd", false),
            // Fully-quoted cd stays modeled (balanced quotes are provable).
            ("cd /tmp && command \"cd\" /tmp && touch f", true),
            ("\"cd\" /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `--` after a forwarding prefix executes the command in the current
    /// shell; a non-temp target must reset fail-closed. Informational/invalid
    /// prefix options never execute the command — tracking stays unchanged.
    #[test]
    fn prefix_dashdash_and_informational_options() {
        let cases = [
            ("cd /tmp && command -- cd /etc && touch f", false),
            ("cd /tmp && builtin -- cd /etc && touch f", false),
            ("cd /tmp && command -p -- cd /etc && touch f", false),
            ("cd /tmp && command -- 'cd' /etc && touch f", false),
            ("cd /tmp && command -- pushd /etc && touch f", false),
            // Forwarding forms with a temp target keep tracking.
            ("cd /tmp && command -- cd /tmp && touch f", true),
            ("cd /tmp && builtin -- cd -P /tmp && touch f", true),
            ("cd /tmp && command -p -- cd /tmp && touch f", true),
            // command -p is repeatable and forwards.
            ("cd /tmp && command -p -p cd /etc && touch f", false),
            ("cd /tmp && command -p -p cd /tmp && touch f", true),
            // Combined short flags forward like repeated -p (`-pp` executes);
            // a -v/-V anywhere in the combination is informational.
            ("cd /tmp && command -pp cd /etc && touch f", false),
            ("cd /tmp && command -pp cd /tmp && touch f", true),
            ("cd /tmp && command -ppp cd /tmp && touch f", true),
            ("command -pp rm -rf /__mahbot_readonly_test_ws__", false),
            ("cd /tmp && command -pp -v cd /etc && touch f", true),
            // Informational options never execute the command: tracking stays
            // where the real CWD is (a chained temp write stays allowed).
            ("cd /tmp && command -v cd /tmp && touch f", true),
            ("cd /tmp && command -V cd && touch f", true),
            ("cd /tmp && command -p -v cd /etc && touch f", true),
            ("cd /tmp && command -pv cd /etc && touch f", true),
            ("cd /tmp && command -x cd /etc && touch f", true),
            ("cd /tmp && builtin -p cd /tmp && touch f", true),
            // time consumes at most one -p: the second becomes the timed
            // command (`-p: command not found`), so cd never runs — tracking
            // unchanged. A chained relative write is validated against the
            // real (unchanged) CWD and must reject when that is the workspace.
            ("time -p -p cd /tmp && touch f", false),
            ("cd /tmp && time -p -p cd /etc && touch f", true),
            ("cd /tmp && time -p -p cd /tmp && touch f", true),
            // `--` after time is the timed command (not found) — informational.
            ("cd /tmp && time -- cd /etc && touch f", true),
            ("cd /tmp && time -v cd /etc && touch f", true),
            // time -p forwards.
            ("cd /tmp && time -p cd /etc && touch f", false),
        ];

        run_cases(&cases);
    }

    /// `cd` in a pipeline member or after a single `&` runs in a subshell or
    /// child — the parent CWD does not change, and the state before the
    /// boundary continues after it.
    #[test]
    fn pipeline_and_background_cd_fail_closed() {
        let cases = [
            // Pipeline member cd — parent CWD unchanged (workspace).
            ("cd /tmp | true && touch f", false),
            ("cd /tmp | cat && touch f", false),
            ("cd /tmp | true\n touch f", false),
            // State before the pipeline continues after it.
            ("cd /tmp && ls | grep x && touch f", true),
            ("cd /tmp && cd /etc | true && touch f", true),
            // Constructs as pipeline members run in subshells — their state
            // never leaks into the parent chain.
            ("ls | { SNAP=/tmp/x; } && touch $SNAP/f", false),
            ("ls | { cd /etc; } && touch f", false),
            ("ls | ( SNAP=/tmp/x ) && touch $SNAP/f", false),
            ("ls | if true; then SNAP=/tmp/x; fi && touch $SNAP/f", false),
            // A brace cd in a pipeline is contained (subshell) — the parent
            // cwd stays /tmp for the chained write.
            ("cd /tmp && ls | { cd /tmp; } && touch f", true),
            // Single-& background cd — parent CWD unchanged (workspace).
            ("cd /tmp & touch f", false),
            ("cd /tmp & rm -rf x", false),
            ("cd /tmp & touch /tmp/x", true),
            ("cd /tmp && touch a & touch f", false),
            ("cd /tmp & echo done", true),
            // `&&` is not a background separator.
            ("cd /tmp && touch f", true),
            // Redirect operators containing & are not separators.
            ("echo hi 2>&1", true),
            ("echo hi &> /tmp/out", true),
            ("echo hi >& /tmp/out", true),
            ("echo hi <&1", true),
            ("cat /tmp/x >& /tmp/out", true),
            // An ESCAPED `\>`/`\<` at segment end is a literal argument, not a
            // redirect operator — the following `&`/`|` IS a separator, so the
            // trailing command is validated independently (fail-closed).
            ("echo a\\> & touch f", false),
            ("echo a\\< & touch f", false),
            ("echo a\\> & touch /tmp/ok", true),
            ("echo a\\> | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo a\\> | grep x && touch /tmp/ok", true),
            ("echo a\\> 2>&1 & touch f", false),
            // A QUOTED `>`/`<` is a literal argument — the trailing quote keeps
            // the `&`/`|` a separator, so the trailing command is validated
            // independently (fail-closed).
            ("echo \">\" & touch f", false),
            ("echo '>' & touch f", false),
            ("echo \">\" | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo '<' | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo \"a>b\" & touch f", false),
            ("echo hi \"2>&1\" & touch f", false),
            ("echo x > \"/tmp/out\" & touch f", false),
            ("echo \"a\" > /tmp/out & touch f", false),
            // `>|` (clobber redirect) stays whole; `\>` before `|` does not.
            ("echo hi >| /tmp/out", true),
            // `||` is an OR-list, not a pipeline: the left side runs in the
            // current shell and the right side only on failure. State threads
            // like `&&` — a cd on the left tracks (baseline behavior).
            ("cd /tmp || touch f", true),
            ("cd /tmp || rm -rf x", true),
            ("cd /tmp || true && touch f", true),
            ("cd /etc || true && touch f", false),
            ("false || rm -rf /__mahbot_readonly_test_ws__", false),
            ("false || touch /__mahbot_readonly_test_ws__/x", false),
            // A subshell `||` is non-leaking — the parent cwd stays /tmp.
            ("cd /tmp && (cd /etc || true) && touch f", true),
            ("cd /tmp/nonexistent_x || touch f", false),
        ];

        run_cases(&cases);
    }

    /// Explicit PWD assignment must not be masked by tracked-cwd resolution.
    #[test]
    fn explicit_pwd_assignment_fails_closed() {
        let cases = [
            ("export PWD=/etc && touch $PWD/f", false),
            ("PWD=/etc touch $PWD/f", false),
            ("cd /tmp && export PWD=/etc && touch $PWD/f", false),
            ("cd /tmp && PWD=/etc touch $PWD/f", false),
            ("cd /tmp && export PWD=/etc && touch /tmp/ok", true),
            // A temp PWD binding resolves normally.
            ("cd /tmp && PWD=/tmp && touch $PWD/f", true),
            // Untracked PWD still resolves to the workspace root.
            ("touch $PWD/f", false),
            ("cd /tmp && touch $PWD/out", true),
        ];

        run_cases(&cases);
    }

    /// The brace-group eval shape (`{ eval 'cd /etc'; }`) runs the eval body
    /// in the current shell — chained relative writes fail closed.
    #[test]
    fn brace_eval_body_fails_closed() {
        let cases = [
            ("cd /tmp && { eval 'cd /etc'; } && touch f", false),
            ("cd /tmp && { eval \"cd /etc\"; } && touch f", false),
            ("cd /tmp && { eval 'cd /tmp'; } && touch f", false),
            ("cd /tmp && { eval 'cd /etc' ; } && rm -rf x", false),
            // A brace group with only reads stays allowed.
            ("cd /tmp && { ls; } && touch f", true),
            ("{ echo hi; } && touch /tmp/x", true),
            // A cd-SHAPED argument word is not a cd — no reset (position-aware).
            ("cd /tmp && { echo cd; } && touch f", true),
            ("cd /tmp && { echo \"hello cd\"; } && touch f", true),
            ("cd /tmp && { grep cd file; } && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `cd` with extra operands: the runtime shell ignores extras and executes
    /// the cd to the first target — the guard cannot prove the target across
    /// shells, so tracking resets fail-closed.
    #[test]
    fn cd_extra_operands_fail_closed() {
        let cases = [
            ("cd /tmp extra && touch f", false),
            ("cd /tmp && cd /etc extra && touch f", false),
            ("cd /tmp && cd -P /tmp extra && touch f", false),
            ("cd /tmp && command cd /tmp extra && touch f", false),
            ("eval 'cd /tmp extra' && touch f", false),
        ];

        run_cases(&cases);
    }

    // ── Control constructs (fail-closed bodies/conditions, no leakage) ────

    /// Control-construct bodies and conditions get the same checks as
    /// top-level commands: workspace-targeting mutations and disallowed
    /// redirects reject; pure reads and temp-scoped writes stay allowed —
    /// across every construct class, both case spellings, nested constructs,
    /// constructs in substitutions, `!`-negated commands, and `&`/`|&`
    /// backgrounds.
    #[test]
    fn control_construct_bodies_and_conditions() {
        let cases = [
            // if / elif / else bodies and conditions.
            ("if true; then touch f; fi", false),
            ("if true; then touch /tmp/f; fi", true),
            ("if false; then echo hi; fi", true),
            ("if true; then echo hi > f; fi", false),
            ("if true; then echo hi > /tmp/out; fi", true),
            ("if cd /etc; then echo hi; fi", true),
            ("if true; then rm -rf f; else echo hi; fi", false),
            ("if false; then echo a; elif true; then touch f; fi", false),
            ("if false; then echo a; else touch /tmp/f; fi", true),
            ("if true\nthen touch f; fi", false),
            // while / until conditions and bodies.
            ("while true; do touch f; done", false),
            ("while true; do touch /tmp/f; done", true),
            ("while grep -q x file; do echo hi; done", true),
            ("until false; do rm -rf x; done", false),
            ("until false; do touch /tmp/f; done", true),
            // for: word lists, arithmetic headers, loop-var temp binding.
            ("for x in a b; do touch f; done", false),
            ("for x in a b; do touch /tmp/f; done", true),
            ("for x in /tmp/*.db; do rm \"$x\"; done", true),
            (
                "for db in ~/.mahbot/db/*.db; do cp \"$db\" /tmp/x/; done",
                true,
            ),
            ("for x in a; do rm \"$x\"; done", false),
            ("for ((i=0; i>3; i++)); do echo hi; done", true),
            ("for ((i=0; i>3; i++)); do touch f; done", false),
            ("for x; do echo hi; done", true),
            // `do` is a LIST WORD on the runtime shell (bash 3.2 iterates over
            // it in `for x in do; do ...`); omitting the `;`/newline before
            // `do` is a syntax error there, and the guard's rejection matches.
            ("for x in do; do echo hi; done", true),
            ("for x in a b do echo hi; done", false),
            // select.
            ("select x in a b; do touch f; done", false),
            ("select x in a b; do echo hi; done", true),
            // case: `|`-pattern and plain `;;` spellings.
            ("case x in a|b) touch f;; esac", false),
            ("case x in a|b) echo hi;; esac", true),
            ("case x in a) touch /tmp/f;; b) touch /tmp/g;; esac", true),
            ("case x in a) touch f;; esac", false),
            ("case x in\na) echo hi;;\nesac", true),
            ("case x in\nx) touch f;;\nesac", false),
            // subshells.
            ("( touch f )", false),
            ("( touch /tmp/f )", true),
            ("( cd /tmp && touch f )", true),
            ("( cd /etc && touch f )", false),
            // brace groups.
            ("{ touch f; }", false),
            ("{ touch /tmp/f; }", true),
            ("cd /tmp && { rm -rf x; }", true),
            // function definitions.
            ("f() { touch f; }", false),
            ("f() { touch /tmp/f; }", true),
            ("function f { rm -rf x; }", false),
            ("function f() { touch /tmp/f; }", true),
            ("function f () { touch /tmp/f; }", true),
            ("function f() { touch f; }", false),
            ("f () ( touch /tmp/f )", true),
            // nested constructs.
            ("if true; then for x in a; do touch f; done; fi", false),
            ("if true; then for x in a; do touch /tmp/f; done; fi", true),
            ("while true; do if false; then touch f; fi; done", false),
            (
                "if true; then while false; do cd /tmp; done; fi; touch f",
                false,
            ),
            // constructs in substitutions (subshell semantics).
            ("echo $(if true; then touch f; fi)", false),
            ("echo $(if true; then touch /tmp/f; fi)", true),
            ("echo $(for x in /tmp/*; do echo $x; done)", true),
            ("echo $(while true; do touch f; done)", false),
            // !-negated commands run in the current shell.
            ("! cd /tmp && touch f", true),
            ("! cd /etc && touch f", false),
            ("! ls && touch /tmp/x", true),
            // & and |& backgrounds.
            ("cd /tmp & touch f", false),
            ("cd /tmp & touch /tmp/x", true),
            ("cd /tmp |& true && touch f", false),
            ("ls |& grep x && touch /tmp/x", true),
            // Heredocs inside construct bodies: the body is stripped before
            // the walker; a workspace redirect on the delimiter line rejects
            // (the terminator must be on its own line — `EOF; fi` is not a
            // valid terminator in bash either, so nothing runs there).
            ("if true; then cat <<EOF\nbody\nEOF\nfi", true),
            ("if true; then cat <<EOF > f\nbody\nEOF\nfi", false),
            ("while true; do cat <<EOF > /tmp/out\nbody\nEOF\ndone", true),
            // Unterminated constructs reject.
            ("while true; do echo hi", false),
            ("for x in a b; do echo hi", false),
            ("{ echo hi; ", false),
            ("( echo hi ", false),
            // Header/pattern substitutions are still validated.
            ("for x in $(touch f); do echo hi; done", false),
            ("for x in $(touch /tmp/x); do echo hi; done", true),
            ("for ((i=$(touch f); i<3; i++)); do echo hi; done", false),
            ("case $(touch f) in a) echo hi;; esac", false),
            ("case $(touch /tmp/x) in a) echo hi;; esac", true),
            ("if $(touch f); then echo hi; fi", false),
            ("if $(touch /tmp/x); then echo hi; fi", true),
        ];

        run_cases(&cases);
    }

    /// Nothing set inside if/while/for/case/select bodies or subshells leaks
    /// after the construct: post-construct commands are validated from the
    /// pre-construct cwd — the `if false; then cd /tmp; fi; touch f` shape in
    /// both the `;` and newline spellings rejects.
    #[test]
    fn construct_state_does_not_leak() {
        let cases = [
            ("if false; then cd /tmp; fi; touch f", false),
            ("if false; then cd /tmp; fi\ntouch f", false),
            ("if true; then cd /tmp; fi; touch f", false),
            ("while false; do cd /tmp; done; touch f", false),
            ("for x in a; do cd /tmp; done; touch f", false),
            ("case x in a) cd /tmp;; esac; touch f", false),
            ("( cd /tmp ); touch f", false),
            ("f() { cd /tmp; }; touch f", false),
            // Pre-construct state threads INTO constructs.
            ("cd /tmp && if true; then touch f; fi", true),
            ("cd /tmp && while true; do touch f; done", true),
            ("cd /tmp && { touch f; }", true),
            (
                "SNAP=$(mktemp -d)\nif true; then touch \"$SNAP/f\"; fi",
                true,
            ),
            // Construct changes never affect a chained write after the
            // construct, even with an intervening separator.
            ("if false; then cd /tmp; fi ; touch f", false),
            // A cd inside a body or condition runs in the CURRENT shell (bash
            // if/for/while/case bodies are not subshells) — the leaked CWD is
            // untrackable, so chained relative writes fail closed.
            ("cd /tmp && if true; then cd /etc; fi && touch f", false),
            ("cd /tmp && if true; then cd /etc; fi\ntouch f", false),
            ("cd /tmp && if cd /etc; then :; fi && touch f", false),
            (
                "cd /tmp && if false; then :; elif true; then cd /etc; fi && touch f",
                false,
            ),
            (
                "cd /tmp && if false; then :; else cd /etc; fi && touch f",
                false,
            ),
            ("cd /tmp && for x in 1; do cd /etc; done && touch f", false),
            (
                "cd /tmp && while true; do cd /etc; break; done && touch f",
                false,
            ),
            (
                "cd /tmp && until false; do cd /etc; break; done && touch f",
                false,
            ),
            ("cd /tmp && case a in a) cd /etc;; esac && touch f", false),
            (
                "cd /tmp && if true; then if true; then cd /etc; fi; fi && touch f",
                false,
            ),
            // No cd in the construct: pre-construct tracking threads through.
            ("cd /tmp && if true; then echo hi; fi && touch f", true),
            ("cd /tmp && for x in 1; do echo hi; done && touch f", true),
            // Subshell / pipeline-member cds do NOT leak — state continues.
            ("cd /tmp && (cd /etc) && touch f", true),
            ("cd /tmp && (cd /tmp) && touch f", true),
            // Non-temp var rebindings inside current-shell construct bodies
            // leak at runtime (`if true; then export TMPDIR=/etc; fi; touch
            // $TMPDIR/f` writes to /etc) — the binding poisons the outer
            // tracking so the chained write fails closed.
            (
                "if true; then export TMPDIR=/etc; fi; touch $TMPDIR/f",
                false,
            ),
            (
                "if true; then export TMPDIR=/etc; fi\ntouch $TMPDIR/f",
                false,
            ),
            (
                "cd /tmp && if true; then export TMPDIR=/etc; fi && touch $TMPDIR/f",
                false,
            ),
            (
                "for x in 1; do export TMPDIR=/etc; done; touch $TMPDIR/f",
                false,
            ),
            (
                "while true; do export TMPDIR=/etc; break; done; touch $TMPDIR/f",
                false,
            ),
            (
                "case a in a) export TMPDIR=/etc;; esac; touch $TMPDIR/f",
                false,
            ),
            (
                "if true; then export TMPDIR=/etc; else export TMPDIR=/tmp; fi; touch $TMPDIR/f",
                false,
            ),
            // A temp-scoped rebind inside a body leaks but stays safe.
            (
                "if true; then export TMPDIR=/tmp; fi; touch $TMPDIR/f",
                true,
            ),
            // Subshell bodies do NOT leak their bindings.
            ("( export TMPDIR=/etc ); touch $TMPDIR/f", true),
            // A variable FIRST bound inside a conditional part is uncertain
            // (the branch may not execute, leaving it unset at runtime —
            // `if false; then D=/tmp; fi; touch $D/f` writes to /f), so it
            // poisons even when the binding is temp-scoped.
            ("if false; then D=/tmp; fi; touch $D/f", false),
            ("if true; then D=/tmp; fi; touch $D/f", false),
            ("if false; then D=/tmp; fi\ntouch $D/f", false),
            (
                "cd /tmp && if false; then D=/tmp; fi; cd $D && touch f",
                false,
            ),
            ("while false; do D=/tmp; done; touch $D/f", false),
            ("for x in 1; do D=/tmp; done; touch $D/f", false),
            ("case a in a) D=/tmp;; esac; touch $D/f", false),
            (
                "if true; then if false; then D=/tmp; fi; fi; touch $D/f",
                false,
            ),
            // Subshell bindings are discarded entirely — $D stays unbound.
            ("( D=/tmp ); touch $D/f", false),
        ];

        run_cases(&cases);
    }

    /// Quoted and prefix-composed `export` verbs bind exactly like a plain
    /// export (the shell concatenates quotes and the forwarding prefixes run
    /// in the current shell), so a non-temp binding poisons the tracked
    /// variable and a chained `$VAR` expansion fails closed. Non-forwarding
    /// prefixes (`env`/`sudo`) run the export in a child shell — nothing
    /// leaks, so the outer binding stays.
    #[test]
    fn quoted_and_prefixed_export_bindings() {
        let cases = [
            // Quoted export verbs.
            (
                "\"export\" TMPDIR=/__mahbot_readonly_test_ws__; touch $TMPDIR/f",
                false,
            ),
            ("'export' TMPDIR=/etc; touch $TMPDIR/f", false),
            // Quoted assignment words bind by their unquoted name.
            (
                "export \"TMPDIR=/__mahbot_readonly_test_ws__\"; touch $TMPDIR/f",
                false,
            ),
            ("export TMPDIR=\"/etc\"; touch $TMPDIR/f", false),
            ("export \"TMPDIR\"=/etc; touch $TMPDIR/f", false),
            // Forwarding prefixes that run export in the current shell.
            ("command export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("builtin export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("time export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("command -p export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("builtin -- export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("command builtin export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("time -p export TMPDIR=/etc; touch $TMPDIR/f", false),
            // Non-forwarding prefixes run export in a child — no leak.
            ("env export TMPDIR=/etc; touch $TMPDIR/f", true),
            ("sudo export TMPDIR=/etc; touch $TMPDIR/f", true),
            // Temp-target bindings through the same spellings stay allowed.
            ("\"export\" TMPDIR=/tmp; touch $TMPDIR/f", true),
            ("export \"TMPDIR=/tmp\"; touch $TMPDIR/f", true),
            ("command export TMPDIR=/tmp; touch $TMPDIR/f", true),
            ("\"export\" SNAP=/tmp/x; touch \"$SNAP/f\"", true),
            // Informational export forms never execute — no binding.
            ("command -v export; touch $TMPDIR/f", true),
            // Plain spellings (regression anchors).
            ("export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("export TMPDIR=/tmp; touch $TMPDIR/f", true),
        ];

        run_cases(&cases);
    }

    /// The two pinned temp-idiot shapes stay allowed: a temp-cwd cd followed
    /// by a temp-scoped removal in a body, and a temp-dir binding followed by
    /// a temp-scoped copy in a body.
    #[test]
    fn pinned_temp_idiot_shapes_allowed() {
        let cases = [
            // temp-cwd cd followed by temp-scoped removal in a brace body.
            ("cd /tmp && { rm -rf x; }", true),
            ("cd /tmp && { rm -rf x; } && touch /tmp/ok", true),
            // temp-dir binding followed by temp-scoped copy in a brace body.
            ("SNAP=$(mktemp -d) && { cp /etc/passwd \"$SNAP/\"; }", true),
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/d\"\n{ cp /etc/passwd \"$SNAP/d/\"; }",
                true,
            ),
            // The same shapes with workspace targets stay blocked.
            ("cd /tmp && { rm -rf /__mahbot_readonly_test_ws__; }", false),
            (
                "SNAP=$(mktemp -d) && { cp /etc/passwd /__mahbot_readonly_test_ws__/; }",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// `time`-prefixed operands that are not simple commands still execute and
    /// must be validated (all bash-verified): a `!`-negated operand runs in
    /// the CURRENT shell (`time ! rm f` deletes; `time ! cd /tmp` propagates
    /// the cd), and a subshell/arithmetic operand runs in a child (`time
    /// ( rm f )`) — the grammar's `command` production allows the subshell as
    /// a child node, which the walker dispatches via walk_node. External-time
    /// forms never bind their args: quoted `"time"` and `! time` leave the
    /// heredoc body on the baseline TMPDIR (safe-direction fixes), while the
    /// leading-assignment form (`TMPDIR=/etc time cat`) still binds (bash
    /// applies the command's assignment prefix to the body expansion) and
    /// `time FOO=bar ! cmd` binds FOO for the body even though `!` as a
    /// command errors. `VAR=val time ! cmd` is over-rejected: external time
    /// fails on the `!` operand (nothing executes).
    #[test]
    fn time_prefixed_non_simple_operands() {
        let cases = [
            // `!`-negated timed operands run in the current shell.
            ("time ! rm -rf ./x", false),
            ("time ! touch f", false),
            ("time -p ! rm -rf ./x", false),
            // A second `!` is a bash syntax error — over-rejection is safe.
            ("time ! ! rm -rf ./x", false),
            // Bash runs external time on the `!` operand — nothing executes.
            ("VAR=val time ! rm -rf ./x", false),
            ("time ! echo hi", true),
            ("time ! cd /tmp && cat > f", true),
            // Subshell operands run in a child shell.
            ("time ( rm -rf ./x )", false),
            ("time ( git commit -m test )", false),
            ("time ( ( rm -rf ./x ) )", false),
            ("time ( rm -rf ./x ) | cat", false),
            ("time ( rm -rf ./x ) <<EOF\nbody\nEOF", false),
            ("time (( i = $(rm -rf ./x) ))", false),
            ("time ( echo hi )", true),
            ("time ( cd /tmp && touch f )", true),
            // External time never binds its args before the body expands.
            (
                "\"time\" TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
            (
                "! time TMPDIR=/etc cat <<EOF\n$(touch $TMPDIR/x)\nEOF",
                true,
            ),
            // The leading assignment before external time binds for the body.
            ("TMPDIR=/etc time cat <<EOF\n$(touch $TMPDIR/x)\nEOF", false),
            // FOO binds for the body even though `!` as a command errors.
            ("time FOO=/etc ! touch f <<EOF\n$(touch $FOO/x)\nEOF", false),
            // Flattened brace-group operands reject via the stray `{`/`}`
            // reserved-word checks (bash-valid temp writes over-reject).
            ("time { rm -rf ./x; }", false),
            ("time { cat > /tmp/out; }", false),
        ];
        run_cases(&cases);
    }
}

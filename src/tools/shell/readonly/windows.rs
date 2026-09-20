//! Windows (`cmd.exe`) platform layer of the read-only shell guard.
//!
//! Active only for [`super::ShellPlatform::Windows`]: there the command string is
//! executed by `cmd.exe /C` while the guard's rules are unix-shaped — a word
//! containing `\` is classified unprovable and a leading `/` token is a path
//! operand rather than a switch. The layer closes both directions: the
//! destructive cmd.exe verbs are refused and the legitimate Windows spellings
//! (`%TEMP%\scratch`, `> NUL`, `del/q`) are admitted; the null device and the
//! temp roots are dispatched by platform in the shared file (`is_null_target`,
//! `writes_outside_temp`). The unix verdicts and their tests are untouched: the
//! shared dispatch calls this layer once per command string ([`check_line`],
//! which scans the whole tree and checks the platform itself) and, behind its
//! own platform branch, for every command segment ([`check_segment`]).
//!
//! `grep_engine/windows.rs` is the sibling cmd.exe reader — the grep
//! interception's own, with its own fail-closed policy — so a cmd.exe rule
//! change belongs in both: its verb reading (the tables below on this side,
//! `grep_verb` on the other's) and the `%` expansion, which the two sides read
//! on purpose by different rules ([`expand_percent_vars`] here against the
//! engine's coarse `grep_engine::windows::has_percent_expansion`).
//!
//! # Refused
//!
//! [`DENIED_VERBS`], [`TEMP_GATED_VERBS`], [`CWD_DESTINATION_VERBS`],
//! [`TEMP_GATED_DESTINATION_VERBS`] (with [`COPY_DENIED_SWITCHES`]),
//! [`LIST_ONLY_VERBS`], [`QUERY_VERBS`], [`CLOCK_VERBS`] with an operand, a drive
//! switch with an operand, a switch glued to its operand by cmd's parameter
//! delimiter ([`glued_delimiter_switch`]), and the `set` forms [`check_set`]
//! names. Each table is authoritative on its own category and states its own
//! rule; so does every predicate below. A command string the bash grammar cannot
//! parse at all is refused by the shared parse before this layer sees a segment —
//! cmd-only shapes (`if exist x ( … )`, cmd's `for %i in (…) do …` loops) are
//! refused that way, since no cmd.exe parser is added.
//!
//! # Matching
//!
//! Verb matching is case-insensitive, extension-insensitive, aware of
//! path-qualified spellings and of cmd's glued switch ([`verb_key`],
//! [`split_glued_switch`]): `C:\Windows\System32\FORMAT.COM`, `"C:\Program
//! Files\x\del.exe"`, `format` and `del/q` name the same verb as their plain
//! spellings — a word carrying a path separator names a program only through a
//! known executable extension.
//!
//! A path word is read only where cmd.exe and the bash parse the guard reads
//! agree on the operand; a disagreement the layer models is refused
//! fail-closed:
//!
//! - whitespace — literal or arriving from a `%VAR%`, since cmd expands before
//!   it tokenises — is one operand only when double-quoted. cmd quotes with `"`
//!   alone, so a single-quoted OPERAND is ordinary text and is refused; a VERB
//!   word is judged under the literal name the shared balanced-quote strip reads
//!   on both platforms ([`verb_key`]) — `'del' C:\ws\x.txt` is refused while
//!   `'del' "%TEMP%\x.txt"` stays a temp write, so the quotes hide nothing;
//! - an unquoted `,`/`=` is cmd's parameter delimiter, and `;`/`&`/`|`/`<`/`>`
//!   are its separators and redirects — a spelling the bash parse reads as one
//!   word only through a `\` escape, where cmd.exe still splits or redirects
//!   the operand. Behind a switch the same delimiter is refused outright
//!   ([`glued_delimiter_switch`]): cmd.exe splits the token there and hands the
//!   text behind it to the verb as another argument, so the token is either a
//!   switch the gate drops or one fused word it cannot parse — never the operand
//!   cmd would hand over;
//! - `*`/`?`/`~` (a glob the program reading the path expands), `^` (cmd's
//!   escape, which hides a character the layer must see) and `!` (cmd's delayed
//!   expansion, which the model does not follow) anywhere in the expanded text;
//! - every relative spelling: no relative path OPERAND is provable (see
//!   below), so the temp gate accepts absolute paths only.
//!
//! A `;`-split line is refused whole, which is not a path-word rule: cmd.exe
//! keeps one line, and several of its internal commands split their ARGUMENTS at
//! the `;`, so the fragment the bash parse reads as a second command would become
//! an extra operand of the first (`del %TEMP%\a;C:\ws\x.exe`). cmd.exe has no
//! comment syntax either, so a `#` is an ordinary character to it: `echo hi # &
//! del C:\ws\x.txt` is a comment for the bash parse and two commands for cmd.
//! A heredoc is refused outright: cmd.exe has no `<<`, so the body's later lines
//! are command lines of its own that this reader never inspects — a body line
//! written `del C:\ws\x.txt` would run without ever being judged.
//! All three are checked once, over the whole tree ([`check_line`]) — each is a
//! property of the LINE rather than of a nesting position, so all three also hold
//! inside a substituted command, a parenthesised group, a function body, a case
//! branch and a pipeline member.
//!
//! # Accepted limits (settled, not chased)
//!
//! Each is recorded as a limit rather than chased. Where the cmd.exe side of one
//! could not be measured on this host it is argued (see `# Unverifiable from this
//! host`):
//!
//! - wrappers and other interpreters are NOT chased, and on Windows they are
//!   idiomatic: `cmd /c …` (the shell itself — the Windows counterpart of the
//!   unix limit for a command handed to `sh -c`), `powershell`/`pwsh`, `start`,
//!   `call`, `mshta`, `rundll32`, `forfiles`, `wmic`, and the scripting and
//!   execution hosts `cscript`/`wscript`. (`forfiles` is what the sanitation
//!   temp-cleanup task uses to report the newest file in a tree, so refusing it
//!   would break a documented read-only workflow.)
//! - a VERB word spelled through cmd's escape (`d^el`, `^del`) or its variable
//!   syntax (`%DEL%`, `!DEL!`), and a decorated or trailing-dot name (`del.`,
//!   `format.com.`), match no table and pass. An OPERAND carrying the same
//!   characters is refused instead, the unexpanded `%…%` form below excepted.
//! - a word that escapes a separator or a redirect (`dir \& del C:\ws\x.exe`,
//!   `whoami \> C:\ws\out.txt`) is admitted unless it is a path OPERAND of a
//!   verb the layer reads: cmd.exe reads the backslash as an ordinary character
//!   and then splits or redirects at the character behind it, while the bash
//!   parse keeps one word.
//! - a command on no table at all passes — the same "not a complete account"
//!   contract the unix tables have. The writers among them are the sharp edge:
//!   `certutil` and `sort` are named by no rule here, and neither is a command
//!   that picks its own destination when only a source is given (`makecab
//!   C:\ws\x.txt` defaults its output into cmd's own directory, the unmodelled
//!   one the paragraph after this list describes). The layer models the mutator
//!   families it knows and discloses the rest.
//! - a path-qualified spelling of a known verb written with forward slashes and
//!   no executable extension reaches no Windows table: [`verb_key`] reads a
//!   separator-carrying word as a program only through a known executable
//!   extension, and the bash parse keeps the whole word literal, so the shared
//!   dispatch judges the basename cmd.exe would resolve instead
//!   ([`dispatch_word`]) — deliberately, since the same fallback is what keeps a
//!   `/usr/bin/rm`-shaped word a verb on unix. A name the shared tables carry is
//!   therefore still refused (`C:/Windows/System32/rm C:\ws\x.txt`), while a
//!   Windows-only one passes (`C:/Windows/System32/del C:\ws\x.txt`, and likewise
//!   `erase`, `rd`, `md`, `move`, `copy`, `xcopy`, `ren`, `replace`, `mklink` and
//!   even the denied `format`). The same spelling with backslashes is the shared
//!   classifier's unprovable word and is refused.
//! - whole-line divergences beyond the ones the layer models: the bash parse
//!   folds a `\` before a line break and a line break inside a quoted word into
//!   one command, where cmd.exe reads the text after the break as a command of
//!   its own (`dir C:\ws \` then `del C:\ws\x.txt`, or the same second command
//!   inside an unterminated quoted word). The divergences the layer does model —
//!   a `#` comment, an unquoted `;`, a heredoc — are refused whole-line (see
//!   `# Matching`).
//! - an OPERAND carrying a `%…%` whose name is not `[A-Za-z0-9_]` (`del
//!   "%TEMP%\%中%\x.txt"`) is read as literal text: the model follows no such
//!   variable, yet cmd resolves one of that shape, so a defined name would put
//!   the write outside temp at runtime.
//!
//! cmd resolves a relative operand against its own directory, which is not
//! modelled — and that directory is not reliably outside the temp roots, because
//! the temp cleaner runs its shell in a workspace built at the private temp
//! root. Writes under temp therefore name it absolutely: a literal path, or
//! `%TMP%`/`%TEMP%`, the variables the shells are handed (from the same single
//! source as the accepted roots). A temp path that carries whitespace must be
//! quoted, exactly as cmd requires. A daemon temp path that itself carries one
//! of the refused characters (`~`, `!`, `^` — an 8.3 short-name component, say)
//! makes even the literal spelling unprovable, which is the accepted
//! over-rejection direction, but it is the one rule here that can deny a
//! legitimate temp write. The cd family itself is allowed and not
//! modelled further: it writes nothing, a target fused with a separator
//! (`cd..\..`, `cd\Users`) is an unprovable word and is refused, and the spaced
//! spelling (`cd ..`) works.
//!
//! Only the temp variables are modelled, and they resolve against the very
//! bindings the shells are handed. `set` rebinding one of them is refused: the
//! runtime `%TEMP%` would then name a location the model still reads as the
//! daemon temp root. Any other `NAME=value` (through `set` or the shell's own
//! syntax, which cmd.exe cannot run) is not tracked, so a later `%NAME%` operand
//! is unresolvable and refused. The bindings the shared unix assignment path
//! records are not consulted: cmd.exe cannot execute `NAME=value` at all, so
//! resolving through one would approve a path the runtime never builds.
//!
//! `robocopy`/`xcopy` are gated on the destination only, so a tree copy INTO
//! temp from anywhere is allowed and `/MIR`'s and `/PURGE`'s deletions inside
//! that destination are accepted as part of the same grant;
//! [`COPY_DENIED_SWITCHES`] is the known set of copy switches that reach beyond
//! that grant, and a switch outside it keeps the destination-only grant. A copy
//! with a single path argument is refused: cmd's destination is optional and
//! then defaults to cmd's own directory, which read-only mode never proved is
//! the accepted location (see [`destination_under_temp`]). `move` and `replace`
//! have the same optional destination and are refused the same way when they name
//! fewer than two paths ([`CWD_DESTINATION_VERBS`]). A switch-SHAPED
//! argument carrying a path signal (`/ws/b.txt`) is read as an operand — cmd.exe
//! accepts `/` as a path separator — and must satisfy the gate like any other
//! path; one carrying no path signal (`/s`, `/ws`) stays a switch and is
//! dropped from the gate. A switch-shaped token carrying cmd's parameter
//! delimiter is refused before any of that ([`glued_delimiter_switch`]).
//! `ren`'s second operand is a bare name that cmd anchors on the FIRST operand's
//! directory, so the temp grant here is effectively unreachable: the
//! fully-qualified spelling the gate accepts (`ren %TEMP%\a.txt %TEMP%\b.txt`) is
//! one cmd refuses to run, while the spelling cmd runs (`ren %TEMP%\a.txt b.txt`)
//! is refused as a workspace write.
//! The shared tables are matched case-insensitively on Windows (`TAR`,
//! `shutdown.exe` and `Git push` all dispatch), while the unix verdicts stay
//! case-sensitive: a destructive verb written in another case is not recognised
//! there, which is a settled limit of the shared tables rather than a gap this
//! layer's fold closes.
//!
//! # Unverifiable from this host
//!
//! No Windows host or CI is available to this crate, so cmd.exe's own behaviour
//! is argued rather than measured: how it splits and quotes arguments (a `\`
//! before a space is ordinary text for cmd but an escape for the bash parse the
//! guard reads), its parameter delimiters and separators, its glued switches,
//! its device names, its delayed `!VAR!`/caret escapes and its case-insensitive
//! verb dispatch. Every verdict the layer reaches rests on that reading rather
//! than on anything observed here — the refusals under `# Matching` and the
//! limits under `# Accepted limits` alike. Chiefly argued: that Win32 strips a
//! trailing dot or space off a name component, that cmd.exe resolves a bare
//! `;`-bearing word the way it resolves the fragments the guard splits it into,
//! that a `%…%` outside the name grammar stays literal, that a heredoc's body
//! lines are commands cmd.exe would run, that an omitted `move`/`replace`
//! destination defaults to cmd's own directory ([`CWD_DESTINATION_VERBS`]), and
//! that `ren`/`mklink` refuse an omitted second operand rather than defaulting
//! it.

use std::borrow::Cow;
use std::path::PathBuf;

use tree_sitter::Node;

use super::scan;
use super::{CheckContext, ShellPlatform, rejection_message};

// ── Verdict ──────────────────────────────────────────────────────────────

/// Verdict of the Windows layer for one command segment.
pub(super) enum WinVerdict {
    /// The layer models this verb and permits the invocation: the shared
    /// unix-shaped dispatch must not run for it.
    Allow,
    /// Refused, with the full rejection text (`rejection_message`).
    Refuse(String),
    /// Not a Windows-layer verb: fall through to the shared dispatch.
    Pass,
}

// ── Verb tables ──────────────────────────────────────────────────────────

/// Executable extensions stripped from a verb word before matching.
const VERB_EXTENSIONS: &[&str] = &["exe", "com", "bat", "cmd"];

/// System-level commands refused outright: unlike the file-scoped mutators
/// below there is no temp-scoped grant that would make any of them safe.
#[rustfmt::skip] // one line per family, so each group stays unambiguous
const DENIED_VERBS: &[&str] = &[
    // volumes, partitions, disks, filesystem and boot configuration
    "format", "diskpart", "diskcomp", "diskcopy", "chkdsk", "chkntfs", "fsutil",
    "defrag", "convert", "bcdedit", "bootcfg", "mountvol", "subst", "wbadmin",
    "vssadmin", "label", "diskperf",
    // archive extraction to a destination the layer cannot model
    "expand",
    // system repair, event-log, power and security policy configuration
    "bcdboot", "wevtutil", "secedit", "powercfg", "sfc", "auditpol", "dism",
    "gpupdate", "verifier",
    // permissions, ownership, attributes
    "takeown", "setx",
    // processes and services
    "taskkill", "tskill", "net", "net1",
    // registry and file associations
    "regedit", "regedt32", "regini", "regsvr32", "assoc", "ftype",
    // scheduled tasks, service/package installation, session end
    "at", "instsrv", "logoff", "msiexec", "wusa",
    // network reconfiguration
    "netsh", "route",
];

/// File- and tree-scoped mutators: refused unless every path argument resolves
/// under the accepted temp location — the same single grant the unix
/// `rm`/`rmdir`/`mkdir` gate keeps.
const TEMP_GATED_VERBS: &[&str] = &[
    "del", "erase", "rd", "rmdir", "md", "mkdir", "move", "ren", "rename", "mklink", "replace",
];

/// Temp-gated verbs whose destination cmd.exe may omit, defaulting it to its own
/// process directory — the unmodelled cwd the module doc describes. They need the
/// rule the copy-shaped verbs get from [`destination_under_temp`]: a write whose
/// destination is not proven stays refused. `ren`/`mklink` error out without
/// their second operand, so only `move` and `replace` qualify.
const CWD_DESTINATION_VERBS: &[&str] = &["move", "replace"];

/// Copy-shaped mutators: only the destination (the last path argument) must be
/// under temp — sources are read-only, mirroring the unix `cp` grant.
const TEMP_GATED_DESTINATION_VERBS: &[&str] = &["copy", "xcopy", "robocopy"];

/// The known switches of the copy-shaped verbs that touch something other than
/// the destination: the source tree (`/MOVE`, `/MOV` delete what was copied),
/// the registry (`/REG`), an arbitrary log file (`/LOG:`, `/UNILOG:`), file
/// times (`/TIMFIX`), the never-terminating monitor modes (`/MOT:`, `/MON:`,
/// which re-run the copy until interrupted) and the source files' archive
/// attribute (`xcopy /M`). Refused outright — the monitor modes included
/// although they delete nothing by themselves, because they are not a read-only
/// shape. Not a family-closed list: a switch outside it keeps the
/// destination-only grant.
const COPY_DENIED_SWITCHES: &[&str] = &[
    "move", "mov", "reg", "log", "unilog", "m", "mot", "mon", "timfix",
];

/// Attribute/ACL/encryption tools: the list form and its path operands are
/// read-only, while a switch (`/grant`, `/c`, `/w`) or an attribute operator
/// (`+r`, `-h`) is a mutation.
///
/// The read-only spelling cannot be separated from the mutating one
/// unambiguously — `attrib /s` and `icacls /T` only list, but the same shape
/// carries `attrib +r` and `icacls /grant` — so any switch or operator refuses
/// the tool rather than risk a mutating spelling.
const LIST_ONLY_VERBS: &[&str] = &["attrib", "icacls", "cacls", "cipher", "compact"];

/// Verbs whose read-only spelling is a named subcommand/switch: the FIRST
/// argument must be one of these (quote-stripped, case-folded).
const QUERY_VERBS: &[(&str, &[&str])] = &[
    ("reg", &["query"]),
    ("sc", &["query", "queryex"]),
    ("schtasks", &["/query"]),
];

/// cmd's clock builtins: bare (or `/t`) they only print, an operand sets the
/// machine clock. Matched on the segment's FIRST word rather than on
/// `verb_idx`, because the shared resolver reads a leading `time` as the bash
/// reserved word and lands the index on its operand (`time 12:00`), while
/// cmd.exe runs its builtin on the whole line.
const CLOCK_VERBS: &[&str] = &["time", "date"];

/// cmd.exe's internal commands that no other table covers: `cd`/`chdir` and
/// `pushd`/`popd` (allowed outright by [`check_segment`], whatever is glued to
/// them) and `set`. The rest of the `cmd /?` list either already sits in a
/// [`VERB_TABLES`] entry or has no verdict the glue split could change — `del`,
/// `date` and `assoc` are the table verbs' own spellings.
///
/// The four cwd spellings are one family HERE because the guard only needs to
/// admit them; the engine's own reading of the same family differs by what it
/// does with each — `grep_engine::is_cd_segment` recognises all four so none is
/// left with a stale tracked cwd, while `grep_engine::resolve_cd` tracks
/// `cd`/`chdir`/`pushd` as navigation and refuses `popd` fail-closed, since the
/// directory it returns to lives only in the pushed stack that model does not
/// keep. A change to either side — the family, or what a member of it does to
/// the cwd — is checked against the other.
const INTERNAL_VERBS: &[&str] = &["cd", "chdir", "popd", "pushd", "set"];

// ── Verb key ─────────────────────────────────────────────────────────────

/// Split cmd.exe's glued switch off a raw command word: an internal command
/// takes its switches with no separating space (`del/q C:\ws\x.txt` runs
/// `del /q …`, `rd/s/q X` runs `rd /s /q X`). The split is what makes such a
/// verb readable at all: left whole, the word is a separator-carrying spelling
/// with no known executable extension — not a plain command name — so the
/// mutator behind it would never reach its table. The word is split only when
/// the text before the first `/` names a verb the layer acts on
/// ([`is_glue_splittable`]), so a path-qualified spelling keeps its basename
/// (`subdir/rm.exe` stays the program `rm.exe`, `C:/Windows/format.com` the
/// program `format.com`). An `@` prefix stays on the head — [`verb_key`] strips
/// it.
fn split_glued_switch(word: &str) -> (&str, Vec<&str>) {
    let bare = word.strip_prefix('@').unwrap_or(word);
    let Some((head, _)) = bare.split_once('/') else {
        return (word, Vec::new());
    };
    if !is_glue_splittable(head) {
        return (word, Vec::new());
    }
    let at = word.len() - bare.len() + head.len();
    (&word[..at], switch_cluster(&word[at..]))
}

/// The switch tokens of a glued cluster: cmd starts a new switch at every `/`
/// (`rd/s/q X` is `rd /s /q X`). The split is load-bearing, not cosmetic: a
/// cluster left whole carries a `/`, which [`switch_is_operand`] reads as the
/// path signal of a `/ws/b.txt`-shaped operand — the switches would then have to
/// satisfy the temp gate and every glued invocation would be refused.
fn switch_cluster(glue: &str) -> Vec<&str> {
    let mut switches = Vec::new();
    let mut start = 0;
    for (at, _) in glue.match_indices('/').skip(1) {
        switches.push(&glue[start..at]);
        start = at;
    }
    switches.push(&glue[start..]);
    switches
}

/// The flat verb tables the glue split consults. A table whose verbs may take a
/// glued switch belongs here, or its glued spellings silently miss the split
/// ([`is_glue_splittable`]); [`CWD_DESTINATION_VERBS`] is not listed because its
/// verbs all sit on [`TEMP_GATED_VERBS`] as well, so they are covered through
/// that entry. [`QUERY_VERBS`] pairs a verb with its permitted spellings, so
/// [`is_glue_splittable`] consults it separately.
const VERB_TABLES: &[&[&str]] = &[
    DENIED_VERBS,
    TEMP_GATED_VERBS,
    TEMP_GATED_DESTINATION_VERBS,
    LIST_ONLY_VERBS,
    CLOCK_VERBS,
    INTERNAL_VERBS,
];

/// True when a command word's leading text names a verb this layer acts on, and
/// so may carry a glued switch. Splitting the token is cmd.exe's own behaviour
/// for its internal commands (argued from documented behaviour like every
/// cmd.exe reading here — see `# Unverifiable from this host`); the table verbs
/// that are external programs (`format`, `taskkill`, `xcopy`, …) are split the
/// same way because their glued spelling cannot be argued from this host and
/// over-rejection is the accepted failure direction. Balanced quotes are
/// stripped first, so a quoted head (`"del"/q`) is read as the name cmd.exe
/// resolves.
fn is_glue_splittable(head: &str) -> bool {
    let head = scan::strip_quoted_word(head);
    VERB_TABLES
        .iter()
        .any(|table| table.iter().any(|verb| verb.eq_ignore_ascii_case(head)))
        || QUERY_VERBS
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(head))
}

/// The guard's verb key for a raw command word: the glued switch is split off
/// ([`split_glued_switch`]), balanced outer quotes stripped, cmd's `@` no-echo
/// prefix dropped, the directory prefix dropped on `\` or `/`, a known
/// executable extension removed, ASCII-lowercased
/// (`C:\Windows\System32\FORMAT.COM` → `format`, `"C:\Program Files\x\del.exe"`
/// → `del`, `@del` → `del`, `del/q` → `del`). A word carrying a separator names
/// a program only through a known executable extension; without one it is not a
/// plain command name. `None` when the word is not a plain command name — a
/// `$`/`%`/`!` word (indirection), a glob, a separator-carrying word with no
/// executable extension, or an empty/still-quoted/exotic spelling — so the
/// caller falls through to the shared dispatch, which either refuses the word as
/// unprovable or matches its own tables against the folded basename. It is never
/// this layer's deny list that decides such a word (an accepted limit).
/// Deliberate drift seam: the shared `command_word_basename`/
/// `canonical_command` pair keeps selecting the shell's output profile from the
/// unplatformed spelling, so a later platform-aware change there would make this
/// deny list and the profile dispatch disagree.
pub(super) fn verb_key(word: &str) -> Option<String> {
    verb_key_of(split_glued_switch(word).0)
}

/// The command word the shared dispatch matches its tables against on
/// `platform`.
///
/// On Windows cmd.exe dispatches case-insensitively and ignores the executable
/// extension, so `raw` is read through [`verb_key`] — and, for a word the layer
/// cannot read as a plain command name (a variable, a quoted spelling), through
/// the lowercased `fallback`. On unix `fallback` is returned unchanged: the
/// shared tables stay case-sensitive there.
pub(super) fn dispatch_word(platform: ShellPlatform, raw: &str, fallback: String) -> String {
    if platform == ShellPlatform::Windows {
        verb_key(raw).unwrap_or_else(|| fallback.to_ascii_lowercase())
    } else {
        fallback
    }
}

/// [`verb_key`] for a word already stripped of any glued switch — the head of
/// [`split_glued_switch`].
fn verb_key_of(word: &str) -> Option<String> {
    let w = scan::strip_quoted_word(word);
    let w = w.strip_prefix('@').unwrap_or(w);
    if w.is_empty()
        || w.contains([
            '@', '$', '`', '%', '!', '*', '?', '[', ']', '{', '}', ',', '~', '\'', '"',
        ])
    {
        return None;
    }
    let basename = w
        .rsplit(['\\', '/'])
        .next()
        .expect("rsplit always yields at least one segment");
    if basename.is_empty() || basename.chars().any(char::is_whitespace) {
        return None;
    }
    let (stem, has_extension) = match basename.rsplit_once('.') {
        Some((stem, ext))
            if !stem.is_empty() && VERB_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext)) =>
        {
            (stem, true)
        }
        _ => (basename, false),
    };
    // A separator-carrying word only names a program through a known executable
    // extension (`C:\Windows\System32\FORMAT.COM`): without one cmd.exe would
    // apply PATHEXT or fail, and a path word the bash parse split on whitespace
    // can hide the real command in a later fragment — neither is a proof.
    if !has_extension && stem.len() != w.len() {
        return None;
    }
    Some(stem.to_ascii_lowercase())
}

// ── Lexical Windows path model ───────────────────────────────────────────

/// Split a folded Windows path into its prefix (`c:` / `\\server\share` / `""`)
/// and the remaining segment text. A `\\`-leading path with an empty server or
/// share is degenerate and fails (fail-closed).
fn split_prefix(path: &str) -> Option<(String, &str)> {
    if let Some(len) = drive_letter(path) {
        // A drive prefix is two ASCII bytes, so the slice is a char boundary.
        return Some((path[..len].to_ascii_lowercase(), &path[len..]));
    }
    if let Some(stripped) = path.strip_prefix(r"\\") {
        let mut parts = stripped.splitn(3, '\\');
        let server = parts.next().unwrap_or("");
        let share = parts.next().unwrap_or("");
        if server.is_empty() || share.is_empty() {
            return None;
        }
        let end = 2 + server.len() + 1 + share.len();
        return Some((path[..end].to_ascii_lowercase(), &path[end..]));
    }
    Some((String::new(), path))
}

/// The length of a leading drive-letter prefix (`c:`), if any.
fn drive_letter(path: &str) -> Option<usize> {
    let bytes = path.as_bytes();
    (bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic()).then_some(2)
}

/// True when a folded path is drive- or UNC-absolute: a drive letter followed by
/// a separator (`c:\x`), or a `\\`-prefixed UNC/device path. A bare `c:x` is
/// drive-*relative* and is not absolute.
fn is_absolute(folded: &str) -> bool {
    if folded.starts_with(r"\\") {
        return true;
    }
    drive_letter(folded).is_some_and(|len| folded.as_bytes().get(len) == Some(&b'\\'))
}

/// Normalize a Windows path to its comparison form (see the module doc):
/// `/` folded to `\`, parts joined by single `\`, `.` dropped, `..` resolved
/// lexically (climbing above the root fails), ASCII-lowercased, no trailing
/// separator, prefix preserved.
fn normalize(path: &str) -> Option<String> {
    let folded = path.replace('/', "\\");
    let (prefix, rest) = split_prefix(&folded)?;
    let mut segments: Vec<String> = Vec::new();
    for segment in rest.split('\\') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            segment => {
                // Win32 strips trailing dots and spaces off a component, so
                // `C:\Temp\.. \x` reaches outside temp while reading as a
                // literal `.. ` segment. A component the platform would rewrite
                // cannot be compared lexically — fail closed.
                if segment.ends_with(['.', ' ']) {
                    return None;
                }
                segments.push(segment.to_ascii_lowercase());
            }
        }
    }
    let mut out = prefix;
    for segment in &segments {
        out.push('\\');
        out.push_str(segment);
    }
    Some(out)
}

/// The inner text of a balanced pair of surrounding DOUBLE quotes. cmd.exe
/// quotes only with `"`, so a single-quoted word keeps its quotes (and the
/// lexical model then fails closed on it).
fn strip_double_quotes(word: &str) -> Option<&str> {
    word.strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .filter(|inner| !inner.contains('"'))
}

/// The text cmd.exe delivers for one word: balanced double quotes stripped, the
/// word as written otherwise. `None` when the word carries a quote character cmd
/// does not interpret — a single-quoted word is ordinary text whose characters
/// are not the ones the layer must judge, and an unbalanced `"` is not a quoted
/// argument — so every classifier that would read it as a switch or a name fails
/// closed instead.
fn cmd_token(word: &str) -> Option<&str> {
    match strip_double_quotes(word) {
        Some(inner) => Some(inner),
        None if word.contains(['"', '\'']) => None,
        None => Some(word),
    }
}

/// The absolute, unfolded spelling of a path operand: balanced double quotes
/// stripped, `%VAR%` expanded, `/` folded to `\`. `None` for every spelling
/// whose operand the layer cannot prove — the classes and their reasons are in
/// the module doc's `# Matching`.
fn resolve_raw(word: &str, ctx: &CheckContext) -> Option<String> {
    let raw = cmd_token(word)?;
    // `cmd_token` strips only a balanced double-quoted pair, so a delivered text
    // differing from the word is exactly "cmd read this as one quoted token".
    let quoted = raw != word;
    if raw.is_empty() {
        return None;
    }
    let expanded = expand_percent_vars(raw, ctx)?.replace('/', "\\");
    if expanded.contains(['*', '?', '~', '^', '!']) {
        return None;
    }
    // cmd expands before it tokenises, so a delimiter or separator arriving from
    // a `%VAR%` value splits the operand exactly as a literal one does; only the
    // double-quoted spelling is one operand for both readers.
    if !quoted
        && (expanded.chars().any(char::is_whitespace)
            || expanded.contains([',', '=', ';', '&', '|', '<', '>']))
    {
        return None;
    }
    is_absolute(&expanded).then_some(expanded)
}

/// [`resolve_raw`] folded to the comparison form of [`normalize`].
fn resolve(word: &str, ctx: &CheckContext) -> Option<String> {
    normalize(&resolve_raw(word, ctx)?)
}

/// True when `path` (already normalized) is at or under one of the context's
/// accepted temp roots.
fn under_roots(path: &str, ctx: &CheckContext) -> bool {
    ctx.temp_roots.iter().any(|root| {
        let Some(root) = normalize(&root.to_string_lossy()) else {
            return false;
        };
        !root.is_empty()
            && (path == root
                || path
                    .strip_prefix(&root)
                    .is_some_and(|rest| rest.starts_with('\\')))
    })
}

/// Compare a `canonicalize` result against the accepted temp roots: the verbatim
/// prefix is dropped and the string goes through [`normalize`] — `canonicalize`
/// returns the on-disk case, so a raw comparison never matches the lowercased
/// roots.
fn canonical_under_roots(canonical: &str, ctx: &CheckContext) -> bool {
    normalize(&strip_verbatim(canonical)).is_some_and(|path| under_roots(&path, ctx))
}

/// Reparse-point (junction/symlink) escape check: the canonical location of
/// `path` — of its nearest existing ancestor when the leaf is still new — must
/// still be under an accepted temp root, the analogue of the unix
/// `is_path_under_temp` parent walk. Nothing this host can resolve (a
/// Windows-shaped path off Windows) leaves the lexical verdict standing.
fn resolves_inside_temp(path: &str, ctx: &CheckContext) -> bool {
    let mut probe = PathBuf::from(path);
    loop {
        if let Ok(canonical) = probe.canonicalize() {
            return canonical_under_roots(&canonical.to_string_lossy(), ctx);
        }
        match probe.parent() {
            // The empty-parent guard is load-bearing: off Windows
            // `Path::new(r"C:\Temp\x").parent()` is `""`, and an empty path
            // canonicalizes to the host cwd — which would reject every
            // Windows temp path in the unit lane.
            Some(parent) if !parent.as_os_str().is_empty() && parent != probe => {
                probe = parent.to_path_buf();
            }
            _ => return true,
        }
    }
}

/// True when the word provably names a location under an accepted temp root.
pub(super) fn under_temp(word: &str, ctx: &CheckContext) -> bool {
    let Some(path) = resolve(word, ctx) else {
        return false;
    };
    under_roots(&path, ctx) && resolves_inside_temp(&path, ctx)
}

/// Drop the verbatim prefix `canonicalize` adds on Windows (`\\?\`, and
/// `\\?\UNC\` for a share), so a canonical path compares against the ordinary
/// spelling of the temp roots.
fn strip_verbatim(canonical: &str) -> String {
    if let Some(rest) = canonical.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    canonical
        .strip_prefix(r"\\?\")
        .unwrap_or(canonical)
        .to_string()
}

/// Expand cmd.exe's `%NAME%` references against the temp variables the shells
/// are handed — the same single source as the accepted temp roots, so the name
/// the layer resolves and the environment cmd gives the child cannot drift. The
/// name matches case-insensitively; a name the layer does not bind fails closed
/// — an unbound one expands to nothing and the empty expansion is not a provable
/// path, while cmd's own `%USERPROFILE%`/`%CD%` resolve to values this model
/// cannot see. A `%` with no partner, or with a non-name between, stays literal
/// — an argued reading of cmd's expansion, not a measured one (an accepted
/// limit; see the module doc).
///
/// The pairing above is this reader's, and the interception reads the same
/// character with a coarser rule of its own: `grep_engine::windows::has_percent_expansion`
/// refuses any word carrying two `%` at all, whatever sits between them. That is
/// a deliberate superset, not an older draft of this function — a served
/// member's text rides its spec file and never reaches cmd.exe's command line,
/// so the engine only needs to know whether an expansion could change what the
/// program receives, while this reader must decide what the expansion IS (an
/// operand path, or an unprovable one). The two readings answer different
/// questions and are not to be reconciled into one.
fn expand_percent_vars(word: &str, ctx: &CheckContext) -> Option<String> {
    let mut out = String::with_capacity(word.len());
    let mut rest = word;
    while let Some(start) = rest.find('%') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('%') else {
            break; // no partner: the dangling `%` and the tail stay literal
        };
        out.push_str(&rest[..start]);
        let name = &after[..end];
        if is_var_name(name) {
            let value = &ctx
                .temp_vars
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))?
                .1;
            out.push_str(value);
        } else {
            let literal_end = start + 1 + end + 1;
            out.push_str(&rest[start..literal_end]);
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

/// True when `name` is a `%NAME%` reference cmd.exe would resolve: non-empty
/// ASCII letters, digits and underscores.
fn is_var_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The platform's null device in the spellings cmd.exe accepts, matched
/// case-insensitively: `nul`, `nul:` and the extension-qualified `nul.txt`
/// (drop any `:…` stream suffix and any `.…` extension), double-quoted or bare.
/// A separator or a quote character cmd does not interpret makes the spelling an
/// ordinary path in the cwd, not the device.
pub(super) fn is_null_device(target: &str) -> bool {
    let Some(word) = cmd_token(target) else {
        return false;
    };
    if word.is_empty() || word.contains(['\\', '/']) {
        return false;
    }
    let name = word.split(':').next().unwrap_or_default();
    let name = name.split('.').next().unwrap_or_default();
    name.eq_ignore_ascii_case("nul")
}

/// The whole-line Windows divergences — a comment, an unquoted `;` and a
/// heredoc — checked once over the whole syntax tree, before the walk. Each is a
/// property of the LINE for cmd.exe rather than of a nesting position (see the
/// module doc), so this scan is the layer's only decision point for any of them:
/// a rule hooked into the walker instead would police the positions that walker
/// happens to visit and skip the rest.
pub(super) fn check_line(root: Node, src: &str, ctx: &CheckContext) -> Result<(), String> {
    if ctx.platform != ShellPlatform::Windows {
        return Ok(());
    }
    // An explicit stack, not recursion: this walks the raw tree, whose depth is
    // the command string's, and a guard that aborts on a deeply nested command
    // is worse than one that over-rejects it.
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "comment" => {
                return Err(rejection_message(
                    src,
                    "cmd.exe has no comment syntax — `#` is an ordinary character, so what the \
                     shell parser drops as a comment is an argument or a separator to cmd.exe.",
                    "drop the comment text — write only the commands you want to run (a `#` inside \
                     quotes stays accepted).",
                ));
            }
            "heredoc_redirect" => {
                return Err(rejection_message(
                    src,
                    "cmd.exe has no `<<` — the body's later lines are command lines of its own, \
                     which this guard never reads as commands.",
                    "write the command in the plain spelling cmd.exe accepts.",
                ));
            }
            ";" => {
                return Err(rejection_message(
                    src,
                    "cmd.exe does not split a command on `;` — several of its internal commands \
                     take it as an argument delimiter, so what the shell parser reads as a second \
                     command would be an extra operand of the first.",
                    "sequence commands with `&&` (or `&`), which both readers split on.",
                ));
            }
            _ => {}
        }
        let mut cursor = node.walk();
        stack.extend(node.children(&mut cursor));
    }
    Ok(())
}

/// One command segment's Windows verdict. `words` are the bash-parsed words;
/// `verb_idx` is the index the shared resolver found for the effective verb
/// (cmd.exe has no pre-verb prefixes, but the shared `time`-prefix handling
/// lands the index on the operand, which is what must be judged). `cmd` is the
/// raw command text used in rejection messages.
#[expect(clippy::too_many_lines)] // one branch per verb table, in refusal order
pub(super) fn check_segment(
    words: &[&str],
    verb_idx: usize,
    cmd: &str,
    ctx: &CheckContext,
) -> WinVerdict {
    // cmd also accepts switches glued to the command name (`del/q`, `time/t`,
    // `set/a`), so the verb word's glue is split off and re-joined to the
    // arguments, where every switch-consuming branch below sees it. The split
    // and the key of the segment's first word are computed once: the clock check
    // below and the main path both read that same word whenever the resolver
    // landed the verb index on it.
    let first = words.first().copied().unwrap_or_default();
    let (first_word, first_glue) = split_glued_switch(first);
    let first_key = verb_key(first_word);
    if let Some(first) = first_key.as_deref()
        && CLOCK_VERBS.contains(&first)
    {
        let args = glued_args(&first_glue, words.get(1..).unwrap_or_default());
        let bare = args
            .iter()
            .all(|arg| cmd_token(arg).is_some_and(|token| token.eq_ignore_ascii_case("/t")));
        return if bare {
            WinVerdict::Allow
        } else {
            WinVerdict::Refuse(rejection_message(
                cmd,
                "`time`/`date` with an operand sets the machine clock.",
                "drop the operand — the bare form (or `/t`) only reports.",
            ))
        };
    }
    let (verb_word, glue, verb) = if verb_idx == 0 {
        (first_word, first_glue, first_key)
    } else {
        let Some(word) = words.get(verb_idx) else {
            return WinVerdict::Pass;
        };
        let (word, glue) = split_glued_switch(word);
        (word, glue, verb_key(word))
    };
    let Some(verb) = verb else {
        return WinVerdict::Pass;
    };
    // The cd family writes nothing, so it is allowed outright — and the layer
    // must own the verdict: no relative operand is provable here
    // ([`resolve_raw`]), so a tracked directory would buy nothing, while the
    // shared dispatch would read `cd C:\ws` as an unknown command word.
    if matches!(verb.as_str(), "cd" | "chdir" | "popd" | "pushd") {
        return WinVerdict::Allow;
    }
    let args = glued_args(&glue, words.get(verb_idx + 1..).unwrap_or_default());
    let args = args.as_ref();

    // A bare drive switch (`D:`) moves the process to that drive: it writes no
    // file and needs no verdict of its own. An operand after it is a different
    // thing — cmd.exe takes none, and reading the rest as words of one command
    // would hide a mutator (`D: del C:\ws\x.txt`) — so it is refused.
    if is_drive_switch(verb_word) {
        return if args.is_empty() {
            WinVerdict::Allow
        } else {
            WinVerdict::Refuse(rejection_message(
                cmd,
                "a drive switch (`D:`) takes no operand in read-only mode.",
                "put the drive switch on its own (`C:` then the command), or name a fully-qualified path.",
            ))
        };
    }

    // `set` is cmd's variable builtin: the layer's view of the temp variables
    // must not drift from the one the children run with, so the shared unix
    // assignment machinery must not see it either.
    if verb == "set" {
        return check_set(args, cmd, ctx);
    }

    if DENIED_VERBS.contains(&verb.as_str()) {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            &format!(
                "`{verb}` is not allowed in read-only mode — it changes system, service, \
                 registry, volume or account state outside any workspace scope."
            ),
            "use a read-only inspection command; this operation has no temp-scoped grant on any platform.",
        ));
    }

    // cmd's parameter delimiter glued behind a switch (`del /f,C:\ws\x.txt`,
    // `xcopy /move=C:\ws\x.txt`), for the verbs this layer acts on
    // ([`is_glue_splittable`]): cmd.exe splits the token there and hands the text
    // behind the delimiter to the verb as another argument. That text is never
    // read — the token stays switch-shaped, and a drive-qualified operand behind
    // the delimiter leaves [`switch_is_operand`] false too — and on the
    // copy-shaped verbs the glued deny-list spelling is invisible twice over,
    // since a copy switch's name stops at `:`/`+` ([`denied_copy_switch`]) and so
    // `/move,` and `/log=` never reached [`COPY_DENIED_SWITCHES`] either, leaving
    // the line inside the destination-only grant. Refused outright.
    if is_glue_splittable(&verb) && args.iter().copied().any(glued_delimiter_switch) {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            &format!(
                "`{verb}` with a switch glued to its operand by cmd.exe's parameter delimiter \
                 (`,` or `=`) is not allowed in read-only mode — cmd.exe hands the text behind the \
                 delimiter to the command as another argument, which this guard cannot read as a \
                 switch or as a path."
            ),
            "separate the switch from its operand with a space (or drop the operand).",
        ));
    }

    // A verb on either mutator table holds the same single temp grant, so the
    // gate is asked with both in hand — a verb added to the narrower
    // [`CWD_DESTINATION_VERBS`] alone must not reach the shared dispatch ungated.
    // The omitted-destination rule stays the narrower table's own.
    let needs_destination = CWD_DESTINATION_VERBS.contains(&verb.as_str());
    let temp_gated = TEMP_GATED_VERBS.contains(&verb.as_str()) || needs_destination;
    if temp_gated {
        let paths = path_args(args);
        // Its own rule and its own why: a segment with no path has nothing for
        // the grant to cover — not the same statement as a path outside temp.
        if paths.is_empty() {
            return WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb}` names no path, and read-only mode proves writes only under the \
                     daemon's temp location."
                ),
                super::temp_path_or_alternatives_hint(ctx.platform),
            ));
        }
        if !all_paths_under_temp(&paths, ctx) {
            return WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb}` is not allowed outside the temp directory — it deletes, moves \
                     or creates files outside the daemon's temp location."
                ),
                super::temp_path_or_alternatives_hint(ctx.platform),
            ));
        }
        if needs_destination && paths.len() < 2 {
            return WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb}` without a destination writes into cmd.exe's own directory — a \
                     location read-only mode never proved is the accepted one."
                ),
                "name both the source and a destination under the daemon temp root the \
                 session's `%TMP%`/`%TEMP%` point at.",
            ));
        }
        return WinVerdict::Allow;
    }

    if TEMP_GATED_DESTINATION_VERBS.contains(&verb.as_str()) {
        if let Some(offending) = args.iter().copied().find(|tok| denied_copy_switch(tok)) {
            return WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb} {offending}` is not allowed in read-only mode — that switch \
                     deletes the sources, writes the registry, or logs to an arbitrary file."
                ),
                "drop the switch; only a plain copy into a temp destination is permitted.",
            ));
        }
        let paths = path_args(args);
        return if destination_under_temp(&paths, ctx) {
            WinVerdict::Allow
        } else {
            // This branch spells its own temp route instead of taking the shared
            // `super::temp_path_or_alternatives_hint` the temp-gate branches
            // above use, and the duplication is deliberate: "sources may be read
            // from anywhere" is this family's own half of the grant, which that
            // helper cannot carry, and it is a `&'static str` selector, so
            // composing the two would mean building a string per refusal. An
            // edit to this advice belongs here, not in the helper.
            WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb}` is not allowed — only a destination under the daemon temp \
                     location is permitted."
                ),
                "name a destination under the daemon temp root the session's `%TMP%`/`%TEMP%` \
                 point at; sources may be read from anywhere.",
            ))
        };
    }

    if LIST_ONLY_VERBS.contains(&verb.as_str()) {
        return if args
            .iter()
            .any(|tok| is_switch(tok) || tok.starts_with(['+', '-']))
        {
            WinVerdict::Refuse(rejection_message(
                cmd,
                &format!(
                    "`{verb}` is not allowed in read-only mode — its read-only spelling cannot be \
                     told apart from a mutating one, so no switch or attribute operator is \
                     permitted."
                ),
                "drop the switch or attribute operator and run the bare form.",
            ))
        } else {
            WinVerdict::Allow
        };
    }

    if let Some((_, spellings)) = QUERY_VERBS.iter().find(|(name, _)| *name == verb) {
        // The slash is not part of the spelling: `schtasks /query` and the
        // glued `schtasks/query` are the same command.
        if args
            .first()
            .copied()
            .and_then(cmd_token)
            .is_some_and(|tok| {
                spellings.iter().any(|spelling| {
                    spelling
                        .trim_start_matches('/')
                        .eq_ignore_ascii_case(tok.trim_start_matches('/'))
                })
            })
        {
            return WinVerdict::Allow;
        }
        let listed = spellings
            .iter()
            .map(|spelling| format!("`{verb} {spelling}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return WinVerdict::Refuse(rejection_message(
            cmd,
            &format!("`{verb}` is only permitted in its read-only inspection spelling — {listed}."),
            "use the inspection spelling; the mutating forms change system state outside any workspace scope.",
        ));
    }

    WinVerdict::Pass
}

/// `set` is cmd's variable builtin. The value is everything after the first
/// space, so the words are re-joined before splitting, and cmd strips the outer
/// double quotes of the quoted spelling (`set "NAME=value"`) — both spellings
/// bind the same variable. `set` alone lists the environment and `set NAME`
/// queries one variable: both only read, and only the name's shape has to hold.
///
/// A binding is refused when the name is not a plain environment name — a
/// `/a`/`/p` switch form evaluates an expression or reads input, an empty name
/// binds nothing addressable, and cmd rewrites the line before it binds, so
/// `set TEMP^=C:\ws` would bind `TEMP` under the layer's feet — and when it
/// binds something the guard resolves: a temp variable or a `GIT_*` name. The
/// temp-variable policy behind that is in the module doc.
fn check_set(args: &[&str], cmd: &str, ctx: &CheckContext) -> WinVerdict {
    let joined = args.join(" ");
    let Some(joined) = cmd_token(&joined) else {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            "`set` with a quote character cmd.exe does not interpret — the binding cannot be read.",
            "write a plain `set NAME=value` binding, quoting it with double quotes if it carries spaces.",
        ));
    };
    let (name, binding) = match joined.split_once('=') {
        Some((name, _)) => (name, true),
        None if joined.is_empty() => return WinVerdict::Allow,
        None => (joined, false),
    };
    if !is_var_name(name) {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            &format!(
                "`set {joined}` — only a bare name or a `NAME=value` binding is modelled: a switch \
                 form evaluates or reads instead of binding, and cmd rewrites the line before it \
                 binds, so a name outside the plain charset proves nothing."
            ),
            "use a plain `set NAME=value` (letters, digits and underscores in the name), or a bare `set`.",
        ));
    }
    if !binding {
        return WinVerdict::Allow;
    }
    if ctx
        .temp_vars
        .iter()
        .any(|(temp_name, _)| temp_name.eq_ignore_ascii_case(name))
    {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            &format!(
                "`set {name}=…` rebinds a temp variable the guard resolves — the value the \
                 shells are handed — so a rebinding cannot be followed."
            ),
            "leave the temp variables alone; name the location directly (a literal path under the \
             daemon temp root, or `%TMP%`/`%TEMP%`).",
        ));
    }
    // cmd's environment is case-insensitive, so the shared `GIT_*` binding rule
    // applies to the folded name — otherwise `set git_dir=…` slips a transitive
    // git invocation past the exec-vector scan.
    if super::git_env_name_denied(&name.to_ascii_uppercase()) {
        return WinVerdict::Refuse(rejection_message(
            cmd,
            "`set` with a `GIT_*` variable is not allowed in read-only mode — git reads it as an exec vector.",
            "drop the binding; inspect the repository with `git status`/`git log`/`git ls-remote` instead.",
        ));
    }
    WinVerdict::Allow
}

// ── Argument helpers ─────────────────────────────────────────────────────

/// A verb word's glued switches ([`split_glued_switch`]) followed by the segment's
/// own arguments, so a switch-consuming branch reads the glued spelling exactly
/// as the spaced one. Borrowed when there is no glue — the common case, and the
/// reason this does not copy the arguments for every modelled segment.
fn glued_args<'x, 'a>(glue: &'x [&'a str], rest: &'x [&'a str]) -> Cow<'x, [&'a str]> {
    if glue.is_empty() {
        return Cow::Borrowed(rest);
    }
    Cow::Owned(glue.iter().copied().chain(rest.iter().copied()).collect())
}

/// True when a token is a cmd.exe switch or a unix-style flag — never a path
/// argument. A token cmd would not deliver as written (a single-quoted word) is
/// an operand, so it must satisfy the temp gate like any other.
fn is_switch(tok: &str) -> bool {
    cmd_token(tok).is_some_and(|tok| tok.starts_with('/') || tok.starts_with('-'))
}

/// A bare cmd.exe drive switch (`D:`), which moves the process cwd to that
/// drive's own current directory — a location the layer does not model.
fn is_drive_switch(word: &str) -> bool {
    cmd_token(word).is_some_and(|word| {
        word.len() == 2
            && word.ends_with(':')
            && word.starts_with(|c: char| c.is_ascii_alphabetic())
    })
}

/// True when a copy-shaped verb's switch is one of the refused
/// [`COPY_DENIED_SWITCHES`]: delivered as cmd delivers it, the token must start
/// with `/` or `-` and its name is the text up to the first `:` or `+` — so
/// `/LOG:C:\ws\run.log`, `/LOG+:x` and `-move` are all caught. A token cmd would
/// not deliver as written (a single-quoted word) is not a switch and is read as
/// an operand instead; these verbs gate their destination only, so such a token
/// goes unjudged unless it is the last path argument.
fn denied_copy_switch(token: &str) -> bool {
    let Some(tok) = cmd_token(token) else {
        return false;
    };
    let Some(name) = tok.strip_prefix(['/', '-']) else {
        return false;
    };
    let name = name.split([':', '+']).next().unwrap_or_default();
    COPY_DENIED_SWITCHES
        .iter()
        .any(|switch| name.eq_ignore_ascii_case(switch))
}

/// True when a switch-shaped token is really an operand: cmd.exe accepts `/` as a
/// path separator, so `/ws/b.txt` (or `/b.txt`) names a file, not a switch. A
/// token carrying a `:` is a switch with its argument (`/R:1`, `/XF:*.log`), and
/// one carrying no path signal at all is an ordinary switch (`/s`, `/-Y`).
fn switch_is_operand(tok: &str) -> bool {
    let Some(tok) = cmd_token(tok) else {
        return false;
    };
    let Some(rest) = tok.strip_prefix(['/', '-']) else {
        return false;
    };
    !rest.contains(':') && rest.contains(['.', '\\', '/'])
}

/// True when a switch-shaped token carries cmd's parameter delimiter after the
/// switch text (`/f,C:\ws\x.txt`, `/move=C:\ws`): cmd.exe splits the token there
/// and hands the text behind the delimiter to the verb as another argument — so
/// the token is neither a switch the layer may drop nor a word it can read as a
/// path. The segment is refused (see `# Matching`).
fn glued_delimiter_switch(tok: &str) -> bool {
    cmd_token(tok)
        .and_then(|tok| tok.strip_prefix(['/', '-']))
        .is_some_and(|rest| rest.contains([',', '=']))
}

/// The path arguments of a Windows command segment: every word that is not a
/// switch — plus a switch-shaped spelling that carries a path signal, which cmd
/// reads as an operand, so it must satisfy the temp gate like any other path.
fn path_args<'a>(args: &[&'a str]) -> Vec<&'a str> {
    args.iter()
        .copied()
        .filter(|tok| !is_switch(tok) || switch_is_operand(tok))
        .collect()
}

/// True when every path argument of a segment resolves under an accepted temp
/// root. Vacuously true with no path argument — that case has its own refusal at
/// the call site.
fn all_paths_under_temp(paths: &[&str], ctx: &CheckContext) -> bool {
    paths.iter().copied().all(|path| under_temp(path, ctx))
}

/// True when the LAST path argument (the destination of a copy-shaped verb)
/// resolves under an accepted temp root. A copy naming fewer than two paths is
/// refused — the why is in the module doc. Takes a segment's path arguments, as
/// [`all_paths_under_temp`] does.
fn destination_under_temp(paths: &[&str], ctx: &CheckContext) -> bool {
    paths.len() >= 2
        && paths
            .last()
            .copied()
            .is_some_and(|dest| under_temp(dest, ctx))
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::shell::readonly::{
        CheckContext, ShellPlatform, ValidationState, check_command, is_null_target,
    };
    use std::path::Path;

    /// The fixture's temp root, in Windows spelling: the host's own temp
    /// directory on a Windows host — the reparse-point walk canonicalizes the
    /// nearest existing ancestor, so the root must exist there — and the
    /// fabricated `C:\Temp` elsewhere, where nothing resolves and the lexical
    /// verdict carries (the walk's fallback). It is what the context binds
    /// `%TMP%`/`%TEMP%` to, and it is expressed in its canonical comparison form:
    /// `canonicalize` answers in the on-disk case behind a `\\?\` verbatim
    /// prefix, which `canonical_under_roots` strips off its own probes.
    fn temp_root() -> String {
        if cfg!(windows) {
            let temp = std::env::temp_dir();
            let canonical = temp.canonicalize().unwrap_or(temp);
            strip_verbatim(&canonical.to_string_lossy())
        } else {
            r"C:\Temp".to_string()
        }
    }

    /// A Windows-platform context over the layer's own lexical model: the
    /// platform is pinned (the host must never decide what these assert), while
    /// the temp root and its `%TMP%`/`%TEMP%` bindings come from the host — the
    /// layer expands the variables, so the production `%TEMP%` spelling is
    /// exercised on any host.
    ///
    /// Every temp OPERAND below is written double-quoted. The lane has to hold on
    /// a host whose temp path carries whitespace (`C:\Users\John Doe\…`, a common
    /// install), where cmd.exe hands an unquoted operand to the program in
    /// pieces — the layer refuses that spelling, which is its cmd-parity rule and
    /// not a fixture failure. The unquoted spelling's verdicts are pinned by
    /// `whitespace_after_expansion_fails_closed`.
    fn win_ctx() -> CheckContext {
        let mut ctx = CheckContext::for_platform(Path::new(r"C:\ws"), ShellPlatform::Windows);
        ctx.temp_roots = vec![PathBuf::from(temp_root())];
        // Both names the Windows shells are handed (see `crate::temp`).
        ctx.temp_vars = vec![
            ("TMP".to_string(), temp_root()),
            ("TEMP".to_string(), temp_root()),
        ];
        ctx
    }

    fn ok(cmd: &str) {
        let ctx = win_ctx();
        assert!(
            check_command(cmd, &ctx).is_ok(),
            "expected ALLOW but got REJECT for: `{cmd}`"
        );
    }

    fn assert_rejected(cmd: &str) {
        let ctx = win_ctx();
        assert!(
            check_command(cmd, &ctx).is_err(),
            "expected REJECT but got ALLOW for: `{cmd}`"
        );
    }

    #[test]
    fn verb_key_strips_quotes_prefixes_and_executable_extensions() {
        let cases = [
            (r"C:\Windows\System32\FORMAT.COM", Some("format")),
            (r#""C:\Program Files\x\del.exe""#, Some("del")),
            ("DEL.BAT", Some("del")),
            // A separator-carrying word names a program only through a known
            // executable extension.
            ("/usr/bin/rm", None),
            ("/usr/bin/rm.exe", Some("rm")),
            ("dir", Some("dir")),
            // cmd's `@` no-echo prefix is not part of the command name.
            ("@del", Some("del")),
            ("@FORMAT.COM", Some("format")),
            // A switch glued to a modelled verb is not part of its name; a
            // word whose head is not a verb keeps its basename.
            ("del/q", Some("del")),
            ("RD/S/Q", Some("rd")),
            ("@del/q", Some("del")),
            ("format/q", Some("format")),
            ("attrib/q", Some("attrib")),
            ("\"del\"/q", Some("del")),
            ("C:/Windows/format.com", Some("format")),
            ("bin/rm", None),
            // Only known executable extensions are stripped.
            ("nul.txt", Some("nul.txt")),
            // Indirection, globs and non-names are never plain command names.
            ("%BIN%", None),
            ("!BIN!", None),
            ("$del", None),
            ("de*", None),
            ("de@l", None),
            ("de l", None),
            ("''", None),
        ];
        for (word, expected) in cases {
            assert_eq!(verb_key(word).as_deref(), expected, "word `{word}`");
        }
    }

    /// cmd's internal commands accept their switch glued to the name
    /// (`del/q X` runs `del /q X`), so the glued spelling must get the verdict
    /// of the spaced one — refused where it writes outside temp, admitted
    /// inside it, and never read as a command whose arguments were never gated.
    #[test]
    fn glued_switches_keep_the_spaced_verdict() {
        assert_rejected(r"del/q C:\ws\x.txt");
        assert_rejected(r"rd/s/q C:\ws\tree");
        assert_rejected(r#"move/y C:\ws\a.txt "%TEMP%\b.txt""#);
        assert_rejected(r#"copy/y "%TEMP%\a.txt" C:\ws\b.txt"#);
        // A table verb that is an external program is split the same way (its
        // glued spelling cannot be argued from this host, so it over-rejects).
        assert_rejected(r"taskkill/f /im notepad.exe");
        assert_rejected(r"format/q C:");
        ok(r#"del/q "%TEMP%\x.txt""#);
        ok(r#"rd/s/q "%TEMP%\junk""#);
        ok(r#"md/q "%TEMP%\new""#);
        ok(r#"copy/y C:\ws\a.txt "%TEMP%\b.txt""#);
        // `set /a` evaluates an expression instead of binding a literal.
        assert_rejected("set/a x=1");
        assert_rejected("set/p x=");
        // `cd/d` is a builtin's glued switch: the cd is allowed and the layer
        // never reads it as a program whose arguments would go ungated.
        ok(r#"md "%TEMP%\new" && cd/d "%TEMP%\new""#);
    }

    #[test]
    fn path_model_resolves_drive_and_unc_prefixes() {
        assert_eq!(normalize(r"C:\A\..\B\.\c"), Some(r"c:\b\c".to_string()));
        assert_eq!(normalize(r"c:\a\..\..\b"), None); // climbs above the root
        assert_eq!(
            normalize(r"\\Server\Share\A\..\B"),
            Some(r"\\server\share\b".to_string())
        );
        assert_eq!(split_prefix(r"c:\x"), Some(("c:".to_string(), r"\x")));
        assert_eq!(split_prefix("relative"), Some((String::new(), "relative")));
        // Degenerate UNC and empty paths cannot be proven (fail-closed).
        assert_eq!(split_prefix(r"\\"), None);
        assert_eq!(normalize(""), Some(String::new()));
    }

    #[test]
    fn percent_variables_expand_case_insensitively_and_fail_closed() {
        let ctx = win_ctx();
        let expected = format!("{}\\x", temp_root());
        assert_eq!(
            expand_percent_vars(r"%temp%\x", &ctx).as_deref(),
            Some(expected.as_str())
        );
        assert_eq!(
            expand_percent_vars(r"%TMP%\x", &ctx).as_deref(),
            Some(expected.as_str())
        );
        // A `%` with no partner or a non-name between it stays literal.
        assert_eq!(expand_percent_vars("100%", &ctx).as_deref(), Some("100%"));
        assert_eq!(expand_percent_vars("%%", &ctx).as_deref(), Some("%%"));
        assert_eq!(expand_percent_vars("%a b%", &ctx).as_deref(), Some("%a b%"));
        // An unbound name has no provable expansion — including a name the
        // shared unix assignment path would have recorded: cmd.exe cannot run
        // `NAME=value`, so that binding never exists, and the layer reads the
        // context rather than the shared state.
        assert_eq!(expand_percent_vars("%UNSET%", &ctx), None);
        assert_eq!(expand_percent_vars("%FOO%", &ctx), None);
    }

    /// Every denied verb is refused by its bare name and by a path-qualified,
    /// capitalized, extension-qualified spelling.
    #[test]
    fn denied_verbs_are_refused_in_every_spelling() {
        for verb in DENIED_VERBS {
            assert_rejected(verb);
            assert_rejected(&format!("{verb} /?"));
            assert_rejected(&format!(
                r"C:\Windows\System32\{}.EXE",
                verb.to_ascii_uppercase()
            ));
        }
        // The `@` no-echo prefix does not hide the verb.
        assert_rejected(r"@del C:\ws\x.txt");
        assert_rejected(r"@format C:");
        // A path word the bash parse split at its whitespace must not hide the
        // real verb in a later fragment.
        assert_rejected(r"C:\Program Files\format.com C:");
        // A verb with a variable embedded in its name matches no table — but it
        // is not admitted either: the exact-pinned bash grammar errors on the
        // spelling, so the shared parse failure refuses the line before this
        // layer's tables are consulted.
        assert_rejected(r"del%X% C:\ws\x.txt");
    }

    #[test]
    fn file_mutators_need_every_path_under_temp() {
        let temp = temp_root();
        // The literal absolute spelling, and the `%TEMP%` spelling production
        // hands the layer (the layer expands it to the same text).
        ok(&format!(r#"del "{temp}\scratch.txt""#));
        ok(r#"del "%TEMP%\scratch.txt""#);
        ok(r#"DEL "%TEMP%\scratch.txt""#);
        ok(r#"move "%TEMP%\a.txt" "%TEMP%\b.txt""#);
        ok(r#"md "%TEMP%\new""#);
        ok(r#"replace "%TEMP%\new.txt" "%TEMP%""#);
        // `/` is a path separator for cmd.exe, so a switch-shaped token carrying
        // a path signal is an operand and must satisfy the gate like any other.
        assert_rejected(r#"move "%TEMP%\a.txt" /ws/b.txt"#);
        assert_rejected(r#"del "%TEMP%\a.txt" /b.txt"#);
        ok(r#"move /-Y "%TEMP%\a.txt" "%TEMP%\b.txt""#);
        assert_rejected(r"del C:\ws\src\main.rs");
        assert_rejected(r"del %TEMP%\..\..\ws\main.rs");
        assert_rejected("del src\\main.rs"); // relative: cmd resolves it against a directory no model has
        assert_rejected(r#"del C:\ws\src\main.rs "%TEMP%\a.txt""#); // one outside is enough
        // Nothing to prove — refused for naming no path, not for the temp
        // location the other spellings are refused for.
        let err = check_command("del", &win_ctx()).unwrap_err();
        assert!(err.contains("names no path"), "{err}");
        assert_rejected(r#"erase "%TEMP%\a.txt" "%TEMP%d\b.txt""#);
        // `move`/`replace` with their destination omitted: cmd writes into its own
        // directory, a location the layer does not model.
        assert_rejected(r#"move "%TEMP%\junk.rs""#);
        assert_rejected(r#"move /y "%TEMP%\junk.rs""#);
        assert_rejected(r#"replace "%TEMP%\junk.txt""#);
        assert_rejected(r#"replace /A "%TEMP%\junk.txt""#);
        // `ren`'s usable spelling — a bare new name — is refused with the relative
        // operand it carries.
        assert_rejected(r#"ren "%TEMP%\a.txt" b.txt"#);
        // A single operand outside temp is refused for that write, not for the
        // missing destination — the fix the agent is pointed at is the path.
        let err = check_command(r"move C:\ws\x.txt", &win_ctx()).unwrap_err();
        assert!(
            err.contains("not allowed outside the temp directory"),
            "a single outside-temp operand should be refused as a workspace write: {err}"
        );
    }

    /// Spellings Windows rewrites before touching the filesystem: the lexical
    /// comparison must not read them as the path they look like.
    #[test]
    fn platform_rewritten_paths_fail_closed() {
        // Drive-relative: `<cwd on C>\Temp\scratch.txt` at runtime, not
        // `C:\Temp\scratch.txt`.
        assert_rejected(r"del c:Temp\scratch.txt");
        // Win32 strips a component's trailing space and dot, so this is
        // `C:\Temp\..\x.txt` — outside temp.
        assert_rejected(r"del %TEMP%\.. \x.txt");
        assert_rejected(r"del %TEMP%\..\x.txt.");
    }

    /// cmd.exe splits an unquoted word carrying whitespace where the bash parse
    /// reads one word, so the spelling must fail closed.
    #[test]
    fn whitespace_in_an_operand_fails_closed() {
        // One bash word, two cmd operands — cmd would delete `b.txt` relative
        // to the real cwd.
        assert_rejected(r"del %TEMP%\a\ b.txt");
        // The spelling cmd reads as one operand.
        ok(r#"del "%TEMP%\a b.txt""#);
    }

    /// A single-quoted token is ordinary text for cmd, not a quoted argument: a
    /// classifier that read it as a switch would drop a real operand from the
    /// temp gate (`del "…\a.txt" '/C:\ws\x'` leaves `'/C:\ws\x'` unchecked).
    #[test]
    fn single_quoted_tokens_are_ordinary_text() {
        assert_rejected(r#"del "%TEMP%\a.txt" '/C:\ws\x'"#);
        assert_rejected(r#"del "%TEMP%\a.txt" '-y'"#);
        assert_rejected(r#"move "%TEMP%\a.txt" 'C:\ws\x'"#);
        assert_rejected(r"dir > '%TEMP%\listing.txt'");
        assert_rejected(r#"copy '%TEMP%\a.txt' "C:\ws\b.txt""#);
        // A single-quoted VERB word is judged under its literal name, so a
        // destructive verb cannot hide in quotes — and a word whose rule passes
        // stays a pass, quotes or not.
        assert_rejected(r"'del' C:\ws\x.txt");
        ok(r#"'del' "%TEMP%\x.txt""#);
    }

    /// Whitespace that only appears once a `%VAR%` is expanded: cmd expands
    /// before it tokenises, so a temp path with a space in it (a `%TEMP%` under
    /// `C:\Users\John Doe\...`) splits the operand where the layer reads one
    /// word — the first half is not under temp at all. The quoted spelling is the
    /// one cmd keeps whole, and it is what the rest of this lane writes.
    #[test]
    fn whitespace_after_expansion_fails_closed() {
        let mut ctx = win_ctx();
        let spaced = format!(r"{}\John Doe", temp_root());
        ctx.temp_roots = vec![PathBuf::from(&spaced)];
        ctx.temp_vars = vec![("TEMP".to_string(), spaced)];
        assert!(
            check_command(r"del %TEMP%\x.txt", &ctx).is_err(),
            "expanded whitespace must split the operand, not be read as one word"
        );
        // The other half of the lane's contract: a root that carries a space
        // must not cost the quoted spelling its temp grant. Only the lexical
        // gate is asserted — the fabricated root does not exist, so a verdict
        // through the reparse-point walk would assert the host instead.
        let quoted = resolve(r#""%TEMP%\x.txt""#, &ctx).expect("the quoted spelling resolves");
        assert!(
            under_roots(&quoted, &ctx),
            "a quoted temp path must stay under a temp root that carries a space"
        );
    }

    /// cmd's caret escape and its delayed (`!NAME!`) expansion are not modelled,
    /// and both hide the characters the layer must judge: cmd sees a workspace
    /// file where the lexical model sees a temp path, or a path that only exists
    /// once `setlocal enabledelayedexpansion` expands a name the model does not
    /// track.
    #[test]
    fn caret_escapes_and_delayed_expansion_fail_closed() {
        assert_rejected(r"del C:\ws\^..\..\ws\main.rs");
        assert_rejected(r"del %TEMP%\^..\..\ws\main.rs");
        assert_rejected(r"dir > C:\ws\^..\out.txt");
        assert_rejected(r"del ^C:\ws\f.txt");
        assert_rejected(r"del %TEMP%\sub\!A!\f.txt");
        assert_rejected(
            r"set DEST=C:\ws\src\main.rs && setlocal enabledelayedexpansion && del !DEST!",
        );
    }

    /// cmd's parameter delimiters: unlike the bash parse, cmd splits an
    /// unquoted `,`/`=` into a parameter boundary, so `del a.txt,C:\ws\b.txt`
    /// deletes a workspace file the layer reads as one temp-scoped operand. Glued
    /// behind a switch (`del /f,C:\ws\x.txt`) the delimiter is refused outright,
    /// which is also what stops the copy-shaped verbs' glued deny-list spellings
    /// from slipping past their destination-only grant.
    #[test]
    fn parameter_delimiters_and_separators_fail_closed() {
        assert_rejected(r"copy %TEMP%\a.txt,C:\ws\b.txt");
        assert_rejected(r"del %TEMP%\a.txt,C:\ws\b.txt");
        assert_rejected(r"move %TEMP%\a.txt=C:\ws\b.txt");
        assert_rejected(r#"del /f,C:\ws\x.txt "%TEMP%\junk.txt""#);
        assert_rejected(r#"del /f=C:\ws\x.txt "%TEMP%\junk.txt""#);
        assert_rejected(r#"rd /s,C:\ws\dir "%TEMP%\junk""#);
        // The copy family's twin of the shape, and the reason the rule matters
        // there: a switch name is read up to the first `:` or `+`
        // ([`denied_copy_switch`]), so `/move,` and `/log=` never reached
        // [`COPY_DENIED_SWITCHES`] while the glued token was dropped as a switch —
        // the destination-only grant then admitted the line.
        assert_rejected(r#"xcopy /move,C:\ws\y C:\ws\x.txt "%TEMP%\junk.txt""#);
        assert_rejected(r#"robocopy /log=C:\ws\l C:\ws\x.txt "%TEMP%\junk.txt""#);
        // cmd's separators and redirects: the bash parse keeps them inside one
        // word through a `\` escape, where the line-level `;` rule cannot see
        // them, while cmd.exe still splits or redirects the operand.
        assert_rejected(r"del %TEMP%\x\;C:\ws\y.txt");
        assert_rejected(r"del %TEMP%\a\&b.txt");
        assert_rejected(r"del %TEMP%\x\>C:\ws\out.txt");
        assert_rejected(r"del %TEMP%\x\|C:\ws\y.txt");
        assert_rejected(r"del %TEMP%\x\<C:\ws\y.txt");
        // The quoted spelling is one parameter for cmd too.
        ok(r#"del "%TEMP%\a,b.txt""#);
        ok(r#"copy C:\ws\a.txt "%TEMP%\b,c.txt""#);
        ok(r#"del "%TEMP%\a&b.txt""#);
        ok(r#"del "%TEMP%\a;b.txt""#);
    }

    /// A `;` is bash's command separator but cmd.exe's ARGUMENT delimiter for its
    /// internal commands, so the fragment the parse reads as a second command is
    /// an extra operand of the first — `del %TEMP%\a;C:\ws\x.exe` deletes a
    /// workspace file. The line is refused on the Windows platform, while `&&`
    /// (which both readers split on) keeps working.
    ///
    /// The divergence is a property of the LINE for cmd.exe, not of a nesting
    /// position: the guard reads the raw tree once, so a `;` inside a construct
    /// is refused here too (see `line_divergences_are_refused_inside_every_construct`).
    #[test]
    fn semicolon_split_lines_are_refused() {
        assert_rejected(r"del %TEMP%\a;C:\ws\x.exe");
        assert_rejected(r"del %TEMP%\a;..\..\ws\x.exe");
        assert_rejected(r"dir C:\ws;");
        assert_rejected(r#"del "%TEMP%\a";cmd /c del C:\ws\x.exe"#);
        ok(r#"del "%TEMP%\a" && del "%TEMP%\b""#);
        // Over-rejection, accepted: an unquoted `;` anywhere in the line is
        // refused, even where cmd would treat it as ordinary text (`echo a;b`).
        assert_rejected(r"echo a;b");
        ok(r#"echo "a;b""#);
        ok(r#"dir "C:\ws;a""#);
    }

    /// cmd.exe has no comment syntax: to it the `#` is an ordinary character, so
    /// the text the bash parse drops as a comment still reaches it — in
    /// `echo hi # & del C:\ws\x.txt` it deletes the workspace file. Refused on
    /// the Windows platform, where a comment node is the divergence; unix keeps
    /// its verdict (a comment executes nothing for either reader there).
    ///
    /// Like the `;` and the heredoc, the divergence is a property of the LINE
    /// rather than of a nesting position
    /// (see `line_divergences_are_refused_inside_every_construct`).
    #[test]
    fn comment_lines_are_refused() {
        assert_rejected(r"echo hi # & del C:\ws\x.txt");
        assert_rejected(r"# & format C:");
        assert_rejected(r"dir C:\ws # list");
        // Quoted or word-internal, the `#` is ordinary text for both readers.
        ok(r#"echo "a # b""#);
        ok(r"echo a#b");
    }

    /// Every whole-line divergence is decided once, over the raw syntax tree,
    /// before the walk — so each holds in every position a command can sit in,
    /// including the constructs whose children the walker filters (a subshell, a
    /// command substitution, a brace group, a function body, a case branch and a
    /// pipeline member). A rule hooked into the walker instead would police only
    /// the positions that walker happens to visit. The comment and `;` cases name
    /// a harmless command (`dir`), so the rule's own message is what must come
    /// back — not a parse error or a mutator verdict from the construct around
    /// it.
    #[test]
    fn line_divergences_are_refused_inside_every_construct() {
        for cmd in [
            "(dir C:\\ws # c\n)",
            "echo $(dir C:\\ws # c\n)",
            "{ dir C:\\ws # c\n}",
            "f() { dir C:\\ws # c\n }",
            "case x in a) dir C:\\ws ;; # c\n esac",
            "dir C:\\ws | # c\n more",
        ] {
            let err = check_command(cmd, &win_ctx()).unwrap_err();
            assert!(err.contains("no comment syntax"), "{cmd}: {err}");
        }
        // The same constructs carrying a `;`-split line instead: the fragment
        // after the `;` is an extra operand of `del` for cmd.exe.
        for cmd in [
            "(del %TEMP%\\a;C:\\ws\\x.exe)",
            "echo $(del %TEMP%\\a;C:\\ws\\x.exe)",
            "{ del %TEMP%\\a;C:\\ws\\x.exe\n}",
            "f() { del %TEMP%\\a;C:\\ws\\x.exe\n }",
            "case x in a) del %TEMP%\\a;C:\\ws\\x.exe ;; esac",
            "del %TEMP%\\a;C:\\ws\\x.exe | more",
        ] {
            let err = check_command(cmd, &win_ctx()).unwrap_err();
            assert!(err.contains("argument delimiter"), "{cmd}: {err}");
        }
        // A heredoc is refused the same way, in the same positions.
        for cmd in [
            "cat <<EOF\nx\nEOF",
            "(cat <<EOF\nx\nEOF\n)",
            "{ cat <<EOF\nx\nEOF\n}",
            "echo $(cat <<EOF\nx\nEOF\n)",
            "f() { cat <<EOF\nx\nEOF\n }",
            "case x in a) cat <<EOF\nx\nEOF\n ;; esac",
            "cat <<EOF\nx\nEOF | more",
        ] {
            let err = check_command(cmd, &win_ctx()).unwrap_err();
            assert!(err.contains("no `<<`"), "{cmd}: {err}");
        }
    }

    /// A tripwire for the module doc's `# Accepted limits` section: one admitted
    /// spelling per limit, so a later change that starts refusing one comes past
    /// this test first. Not the whole list — the wrapper limit is pinned by
    /// `unmodelled_verbs_fall_through_to_the_shared_dispatch`, and the unix
    /// case-variant one by `unix_platform_verdicts_never_use_the_windows_rules`.
    #[test]
    fn documented_limits_are_not_closed() {
        // A word escaping a separator or a redirect: cmd.exe reads the `\` as an
        // ordinary character and splits or redirects at the character behind it,
        // so the second command runs although the bash parse reads one word.
        ok(r"dir \& del C:\ws\x.exe");
        ok(r"whoami \> C:\ws\out.txt");
        // A VERB word spelled through cmd's escape or its variable syntax, and a
        // decorated or trailing-dot name: none matches a table.
        ok(r"%DEL% C:\ws\x.txt");
        ok(r"!DEL! C:\ws\x.txt");
        ok(r"d^el C:\ws\x.txt");
        ok(r"DEL. C:\ws\x.txt");
        ok(r"format.com. C:");
        // A command on no table at all — the writers among them, and one that
        // picks its own destination when it is given only a source.
        ok(r"certutil -decode C:\ws\x.b64 C:\ws\out.bin");
        ok(r"sort /o C:\ws\out.txt C:\ws\a.txt");
        ok(r"makecab C:\ws\x.txt");
        // A path-qualified verb with forward slashes and no executable
        // extension: a literal word for the bash parse, and no table for
        // `verb_key` — which reads a separator-carrying word as a program only
        // through a known extension. The shared dispatch then judges the
        // basename cmd.exe would resolve, so a name the shared tables carry is
        // still refused in this spelling while a Windows-only one passes. The
        // backslash spelling is the shared classifier's unprovable word instead.
        ok(r"C:/Windows/System32/del C:\ws\x.txt");
        ok(r"C:/Windows/System32/copy C:\ws\a.txt C:\ws\b.txt");
        // The sharpest one: `format` IS on a Windows deny table, and this
        // spelling still reaches the shared dispatch by its basename only.
        ok(r"C:/Windows/System32/format C:");
        assert_rejected(r"C:/Windows/System32/rm C:\ws\x.txt");
        assert_rejected(r"C:\Windows\System32\del C:\ws\x.txt");
        // Whole-line divergences beyond the modelled ones: the bash parse folds
        // a `\` before a line break and a line break inside a quoted word into
        // one command, while cmd.exe reads a second command after the break.
        ok("dir C:\\ws \\\n del C:\\ws\\x.exe");
        ok("echo \"a\n del C:\\ws\\x.exe\"");
        // An operand `%…%` whose name is not `[A-Za-z0-9_]`: read as literal
        // text rather than expanded.
        ok(r#"del "%TEMP%\%中%\x.txt""#);
    }

    #[test]
    fn clock_builtins_refuse_an_operand() {
        ok("date /t");
        ok("time /t");
        ok("date/t");
        ok("time/t");
        ok("date");
        ok("time");
        assert_rejected("date 12/31/2026");
        assert_rejected("time 12:34:56.78");
        assert_rejected("time/t 12:00");
        assert_rejected("@time 12:00");
    }

    #[test]
    fn copy_shaped_verbs_gate_the_destination_only() {
        ok(r#"copy C:\ws\a.txt "%TEMP%\b.txt""#);
        ok(r#"xcopy C:\ws\src "%TEMP%\backup" /E"#);
        // A tree copy into temp from anywhere, `/MIR` deletions included.
        ok(r#"robocopy C:\ws "%TEMP%\mirror" /MIR"#);
        assert_rejected(r#"copy "%TEMP%\a.txt" C:\ws\b.txt"#);
        assert_rejected(r#"xcopy "%TEMP%\mirror" C:\ws\mirror /E"#);
        assert_rejected(r"robocopy C:\ws C:\ws\backup /MIR");
        // cmd's destination is optional and then defaults to cmd's own
        // directory, which read-only mode never proved is the accepted
        // location, so a single path argument is refused.
        assert_rejected(r#"copy "%TEMP%\report.txt""#);
        assert_rejected(r#"xcopy "%TEMP%\report.txt""#);
        assert_rejected(r"copy C:\ws\a.txt");
        assert_rejected("copy");
        // Switches that stop being a destination-only copy: source deletion,
        // registry/log writes, the archive attribute, and the monitor modes
        // (`/MOT:`/`/MON:` re-run the copy until interrupted).
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /MOVE"#);
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /MOV"#);
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /LOG:C:\ws\run.log"#);
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /REG"#);
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /MOT:5"#);
        assert_rejected(r#"robocopy C:\ws "%TEMP%\x" /MON:3"#);
        assert_rejected(r#"xcopy C:\ws "%TEMP%\x" /M"#);
        ok(r#"xcopy C:\ws "%TEMP%\x" /A"#);
        // A colon-bearing switch stays a switch and the destination gate holds.
        ok(r#"robocopy C:\ws "%TEMP%\x" /E /R:1 /W:1"#);
    }

    #[test]
    fn list_only_verbs_allow_the_bare_form() {
        ok("attrib");
        ok(r"attrib C:\ws\a.txt");
        ok(r"icacls C:\ws\a.txt");
        assert_rejected(r"attrib +r C:\ws\a.txt");
        assert_rejected(r"attrib -h C:\ws\a.txt");
        assert_rejected(r"attrib /s C:\ws");
        assert_rejected(r"icacls C:\ws\a.txt /grant everyone:f");
        assert_rejected(r"icacls C:\ws\a.txt /deny everyone:d");
        assert_rejected(r"cipher /w:C:\ws");
        assert_rejected(r"compact /c C:\ws\a.txt");
    }

    #[test]
    fn query_verbs_allow_only_the_inspection_spelling() {
        ok(r"reg query HKLM\Software");
        ok("sc query state= all");
        ok("sc queryex wuauserv");
        ok("schtasks /query /fo LIST");
        // The glued spelling is the spaced one (`schtasks/query`).
        ok("reg/query HKLM\\Software");
        ok("schtasks/query /fo LIST");
        assert_rejected(r"reg add HKLM\Software\X /v Y");
        assert_rejected(r"reg delete HKLM\Software\X");
        assert_rejected(r"reg import %TEMP%\x.reg");
        assert_rejected(r"reg/add HKLM\Software\X /v Y");
        assert_rejected("sc start wuauserv");
        assert_rejected("sc stop wuauserv");
        assert_rejected("schtasks /create /tn x /tr y");
        assert_rejected("schtasks /delete /tn x");
    }

    /// Only the daemon temp variables are part of the model: rebinding one is
    /// refused, any other binding is allowed but not tracked — so a later
    /// `%NAME%` reference is unbound and refused.
    #[test]
    fn set_binds_only_what_the_model_can_follow() {
        ok("set");
        ok("set PATH");
        ok("set FOO=bar");
        ok(r#"set "FOO=bar baz""#);
        ok(r"set FOO=%TEMP%\x");
        // Rebinding a temp variable would make every later expansion name a path
        // cmd no longer uses — any spelling, any casing.
        assert_rejected(r"set TEMP=C:\ws");
        assert_rejected(r"set temp=C:\ws");
        assert_rejected(r#"set "TEMP=%TEMP%\x""#);
        assert_rejected(r"set TMP=C:\ws && del %TMP%\x.txt");
        // The binding is not tracked, so using it cannot be resolved.
        assert_rejected(r"set FOO=%TEMP%\x && del %FOO%\x.txt");
        assert_rejected("set /a x=1");
        assert_rejected("set /p x=");
        assert_rejected("set =value");
        // A quote character cmd.exe does not interpret leaves the binding
        // unreadable; the double-quoted spelling is the one cmd assigns, and a
        // temp name is refused through it too.
        assert_rejected(r"set TEMP='C:\ws'");
        assert_rejected(r#"set "TEMP='C:\ws'""#);
        // cmd's transform pass rewrites the line before the builtin binds, so a
        // name carrying an escape would set a variable spelled differently from
        // the text the layer reads.
        assert_rejected(r"set TEMP^=C:\ws");
        assert_rejected(r"set TEMP =C:\ws");
        assert_rejected(r"set TEM%P%=C:\ws");
        // cmd's environment is case-insensitive, so the unix `GIT_*` binding
        // rule applies to any casing; `GIT_PAGER` is the documented carve-out.
        assert_rejected(r"set GIT_SSH_COMMAND=C:\ws\evil.exe && git ls-remote x");
        assert_rejected(r"set git_dir=C:\ws");
        ok("set GIT_PAGER=cat");
    }

    /// cmd.exe dispatches `Git`/`GIT`/`GIT.EXE` to the same program, so a
    /// mixed-case or extension-qualified verb must get exactly the `git`
    /// verdict.
    #[test]
    fn mixed_case_git_verbs_keep_their_verdict() {
        for verb in ["Git", "GIT", "git.exe", "GIT.EXE", "Git.EXE"] {
            for args in [
                "push origin main",
                "clean -fdx",
                "checkout -f .",
                "status",
                "log --oneline",
            ] {
                let ctx = win_ctx();
                let mixed = check_command(&format!("{verb} {args}"), &ctx);
                assert_eq!(
                    mixed.is_ok(),
                    check_command(&format!("git {args}"), &ctx).is_ok(),
                    "`{verb} {args}` must match the `git {args}` verdict"
                );
            }
        }
        assert_rejected("Git push origin main");
        assert_rejected("GIT clean -fdx");
        assert_rejected("GIT.EXE clean -fdx");
        ok("Git status");
        ok("git.exe log --oneline");
    }

    /// No relative path operand is provable: cmd resolves one against a
    /// directory the guard does not model, and that directory is not reliably
    /// outside the temp roots — the temp cleaner runs its shell in a workspace
    /// built at the private temp root, where a workspace-anchored relative path
    /// would satisfy the gate while cmd writes wherever a `cd` left it. The temp
    /// gate therefore accepts absolute spellings only.
    #[test]
    fn relative_operands_are_refused() {
        let ctx = win_ctx();
        // No relative word has a proving form.
        assert_eq!(resolve_raw("x.txt", &ctx), None);
        assert!(!under_temp("x.txt", &ctx));
        // cmd's other relative spellings: root-relative (the drive's own current
        // directory) and drive-relative.
        assert_eq!(resolve_raw(r"\x.txt", &ctx), None);
        assert_eq!(resolve_raw("c:x.txt", &ctx), None);
        assert!(!under_temp(r"\x.txt", &ctx));
        // The temp cleaner's session: the workspace IS the private temp root, so
        // an anchored relative operand would have satisfied the gate.
        let mut cleaner =
            CheckContext::for_platform(Path::new(&temp_root()), ShellPlatform::Windows);
        cleaner.temp_roots = vec![PathBuf::from(temp_root())];
        cleaner.temp_vars = vec![
            ("TMP".to_string(), temp_root()),
            ("TEMP".to_string(), temp_root()),
        ];
        assert!(
            check_command(r"del x.txt", &cleaner).is_err(),
            "a relative operand must be refused even when the workspace is a temp root"
        );
        assert!(
            check_command(r"cd ..\.. && del x.txt", &cleaner).is_err(),
            "the directory change must not make a relative operand provable"
        );
        assert!(
            check_command(&format!(r#"del "{}\x.txt""#, temp_root()), &cleaner).is_ok(),
            "the absolute spelling of the same write stays admitted"
        );
        // A relative operand stays refused in every spelling, whatever the cd
        // family around it does.
        assert_rejected(r"cd %TEMP% && del x.txt");
        assert_rejected(r"cd /d %TEMP% && del x.txt");
        assert_rejected(r"md %TEMP%\new && cd %TEMP%\new && del x.txt");
        assert_rejected(r"CD %TEMP% && del x.txt");
        assert_rejected(r"pushd %TEMP% && del x.txt");
        assert_rejected(r"cd %TEMP% && cd.. && del x.txt");
        assert_rejected(r"cd %TEMP% && dir > out.txt");
        assert_rejected(r"cd %TEMP% && copy x.txt C:\ws\y.txt");
        assert_rejected(r"cd %dir% && del x.txt");
        // The absolute spelling of the same write stays admitted.
        ok(r#"cd %TEMP% && del "%TEMP%\x.txt""#);
    }

    /// The layer owns the cd family outright: it writes nothing, and the shared
    /// dispatch would read a cmd path as an unknown command word. The target is
    /// not modelled — a spelling fused with a separator (`cd..\..`, `cd\Users`)
    /// is unprovable and refused — and no cd can make a relative operand
    /// writable (see `relative_operands_are_refused`).
    #[test]
    fn cd_family_is_owned_by_the_layer() {
        ok("cd");
        ok(r"cd C:\ws");
        ok("cd..");
        ok(r"cd /d %TEMP%");
        ok("cdrecord /dev/x"); // a longer name is not the builtin
        // cmd's own spellings of the family: case-folded, glued switch, and the
        // `chdir` alias are the same builtin.
        ok(r"CD C:\ws");
        ok("cd/d C:\\ws");
        ok(r"chdir C:\ws");
        ok("pushd C:\\ws");
        ok("popd");
        assert_rejected(r"cd..\..");
        assert_rejected(r"cd\Users");
        // A bare drive switch writes nothing and is allowed; one carrying an
        // operand is not a shape cmd.exe has, and reading the rest as words of
        // one command would hide a mutator.
        ok("C:");
        ok(r"C: && dir C:\ws");
        assert_rejected(r"D: del C:\ws\x.txt");
        assert_rejected(r#"D: del "%TEMP%\x.txt""#);
    }

    /// A non-ASCII command word must be judged, never panic: a panic here aborts
    /// a phase dispatch. The word is either an unmodelled program (allowed) or
    /// unprovable (refused) — what matters is that it has a verdict.
    #[test]
    fn non_ascii_command_words_are_judged_not_panicked() {
        for cmd in [
            "céd",
            "céd/d",
            "céd..",
            "céd\\x",
            "c€x /x",
            "\u{4e2d}\u{6587} C:\\ws",
            "del céd.txt",
            "«rd» C:\\ws",
        ] {
            let _ = check_command(cmd, &win_ctx());
            let _ = verb_key(cmd);
            let _ = split_glued_switch(cmd);
        }
        // Refusals still hold around a non-ASCII operand.
        assert_rejected("del céd.txt"); // relative: no absolute spelling, so refused
        assert_rejected(r#"del "céd.txt""#); // quoted, but still not under temp
        assert_rejected(r"céd\del C:\ws\x.txt"); // unprovable verb
        ok(r#"dir C:\ws\céd && del "%TEMP%\céd.txt""#);
    }

    /// The exact command block the sanitation temp-cleanup task is given on
    /// Windows must keep running: the removal verbs are inside its scan roots
    /// (which are the guard's accepted temp roots, passed as literal absolute
    /// paths), and the inspection rows — `forfiles` included — are not chased as
    /// wrappers.
    #[test]
    fn sanitation_windows_tool_block_still_works() {
        let temp = temp_root();
        ok("whoami /user");
        ok(&format!(r#"dir /a /q /tw "{temp}\junk""#));
        ok(&format!(r#"dir /a:l /s /b "{temp}""#));
        ok(&format!(r#"dir /a "{temp}""#));
        ok(&format!(
            r#"forfiles /P "{temp}" /S /M * /C "cmd /c echo @fdate @ftime @path""#
        ));
        ok(&format!(r#"del /f /q "{temp}\junk.txt""#));
        ok(&format!(r#"rd /s /q "{temp}\junk""#));
        ok(&format!(r#"rd "{temp}\junk""#));
    }

    #[test]
    fn redirects_accept_the_platform_null_device_and_temp_paths() {
        let temp = temp_root();
        ok("dir > NUL");
        ok("dir > nul:");
        ok("dir > NUL.txt");
        ok(&format!(r#"dir > "{temp}\listing.txt""#));
        ok(r#"dir > "%TEMP%\listing.txt""#);
        ok("dir 2>&1");
        assert_rejected(r"dir > C:\ws\listing.txt");
        assert_rejected("dir > NULX");
        // cmd does not interpret single quotes: the target is an ordinary file
        // in the cwd, not the device.
        assert_rejected("dir > 'NUL'");
        // A separator makes the spelling an ordinary path, not the device.
        assert_rejected(r"dir > NUL\..\..\ws\x.txt");
        assert_rejected("dir > NUL.x\\foo");
        // `/dev/null` is a unix spelling: under cmd.exe it is a real path.
        assert_rejected("dir > /dev/null");
    }

    /// `canonicalize` returns the on-disk case behind a `\\?\` verbatim prefix,
    /// so the comparison must fold before matching the lowercased temp roots.
    #[test]
    fn canonical_paths_are_folded_before_comparing() {
        let ctx = win_ctx();
        assert!(canonical_under_roots(
            &format!(r"\\?\{}\MahBot\X.txt", temp_root().to_ascii_uppercase()),
            &ctx
        ));
        assert!(!canonical_under_roots(
            r"\\?\C:\definitely-not-temp\x",
            &ctx
        ));
    }

    #[test]
    fn unmodelled_verbs_fall_through_to_the_shared_dispatch() {
        // `forfiles` is what the sanitation temp-cleanup task reports a tree's
        // newest mtime with, so refusing it would break a documented read-only
        // workflow.
        ok(r"forfiles /p %TEMP% /m * /d -7");
        ok(r"dir /b C:\ws");
        ok(r"where cargo.exe");
        ok(r"type C:\ws\Cargo.toml");
        ok(r"findstr /s /i TODO C:\ws\src\*.rs");
        // Interpreters stay unchased, as on unix — the shell itself included,
        // with its own `/c`.
        ok(r"powershell -Command Get-ChildItem");
        ok(r"cmd /c del C:\ws\x.txt");
        // cmd.exe dispatches case-insensitively and ignores the executable
        // extension, so the shared tables are matched on the layer's verb key:
        // `tar`/`curl`/`sed`/`dd` are flag-gated and `shutdown` is an
        // unconditional mutator, in any casing and with or without `.exe`.
        assert_rejected(r"TAR -xf C:\ws\a.tar");
        assert_rejected(r"CURL -o C:\ws\f https://example.com");
        assert_rejected(r"SHUTDOWN /s");
        assert_rejected(r"tar.exe -xf C:\ws\a.tar");
        assert_rejected(r"Tar.EXE -xf C:\ws\a.tar");
        assert_rejected(r"sed.exe -i C:\ws\f");
        assert_rejected(r"dd.exe of=C:\ws\f");
        assert_rejected(r"shutdown.exe /s");
        ok(r"tar.exe -tf C:\ws\a.tar");
        // The unix mutator tables get the same treatment, extension included.
        assert_rejected(r"rm.exe C:\ws\x.txt");
        assert_rejected(r"C:\tools\rm.exe C:\ws\x.txt");
        ok(r#"rm.exe "%TEMP%\x.txt""#);
        // A quoted absolute executable path is a command word cmd.exe runs, so it
        // is admitted too.
        ok(r#""C:\Program Files\mahbot\mahbot.exe" --version"#);
    }

    /// The platform is a runtime value, so this host can pin both lanes: the
    /// Windows-only rules (the null device, the case fold) must not reach a
    /// unix verdict. The uppercase unix verdicts asserted here are the
    /// documented case-sensitivity limit of the shared tables, not a Windows
    /// behaviour — the fold is what makes the difference.
    #[test]
    fn unix_platform_verdicts_never_use_the_windows_rules() {
        let unix = CheckContext::for_platform(Path::new("/__ws__"), ShellPlatform::Unix);
        let state = ValidationState::new(&unix);
        assert!(!is_null_target("NUL", &state));
        assert!(!is_null_target("nul:", &state));
        // `NUL` is not the unix null device, so the unix redirect verdict stands.
        assert!(check_command("echo hi > NUL", &unix).is_err());
        assert!(check_command("tar -xf a.tar", &unix).is_err());
        assert!(check_command("TAR -xf a.tar", &unix).is_ok());
        assert!(check_command("shutdown /s", &unix).is_err());
        assert!(check_command("SHUTDOWN /s", &unix).is_ok());
        assert!(check_command("shutdown.exe /s", &unix).is_ok());
        // A `;` separates commands for both readers on unix, so the Windows
        // `;`-refusal must not reach a unix verdict.
        assert!(check_command("echo a; echo b", &unix).is_ok());
        // A comment executes nothing on unix, so only Windows refuses it.
        assert!(check_command("echo a # b", &unix).is_ok());
        // A heredoc is ordinary bash: only Windows refuses the `<<`.
        assert!(check_command("cat <<EOF\nbody\nEOF", &unix).is_ok());
    }
}

//! One reading of a command text's line breaks, for the platform whose own
//! reader ends a command at them.
//!
//! On unix this module is a no-op: `sh -c` runs a `\n`-separated script as written, so
//! [`executable`] hands the text back byte for byte and no rule below is reached. On
//! Windows it is the reading the *execution* uses — the value
//! [`build_shell_command`](crate::tools::shell::build_shell_command) hands over as its
//! `/C` argument — because `cmd.exe` reads one line and ends the command at a bare
//! break: the lines after it never run, no error is reported and the call returns the
//! first line's status. A plain break is therefore respelled as the platform's
//! unconditional separator `&`: the lines run in order, a failing line does not stop
//! the rest, and the status reported is the last line's. A break is one break however
//! it is spelled — `\n`, `\r\n` and a longer run of carriage returns all end the line
//! before them ([`line_end`]) — so a rule that reads the line before a break reads the
//! same shape under any of them. A trailing break, and blank lines, add no command of
//! their own: a text whose breaks all stand at its end is the one line it was written
//! as, handed over as that line — byte for byte the text the same command without them
//! hands over ([`spans_commands`] decides which texts those are). A text ending on an
//! odd caret run is the one shape a trailing break changes: that break is the character
//! the caret escapes, so the text is refused whole rather than handed over.
//! A line this reader *swallows* rather than separates (a comment, a label:
//! [`SWALLOWING_LINE`]) is refused whole rather than run as less than it is. The text
//! is kept exactly as written only where the reader is left waiting for the rest of a
//! command: behind an odd caret run (whose escape reaches the first character of the
//! line it continues) and inside a group this reading has proved the reader opened
//! ([`opens_group`]). A break behind `&`/`&&`/`|`/`||`/`<`/`>` is the third shape, and
//! the one that goes rather than stays: the operator was written for an operand this
//! platform takes from the line after it, so the two lines are joined into the one
//! command both readers already read there.
//!
//! # What is argued, and what was measured
//!
//! No Windows host or CI exists in this project, so this platform's reading is
//! argued the way the two sibling modules argue theirs: the model takes
//! [`ShellPlatform`] and [`ShellMode`] as data (never a `cfg` branch of its own), so
//! both lanes are unit-testable from any host — but nothing here was measured
//! against `cmd.exe`. Every rule fails toward a refusal when its premise does not
//! hold — a wrong premise there costs a command that would have run — except the
//! rules that run a text *as this platform reads it* rather than refuse it: a wrong
//! premise about one of those runs a different command. Those rules are the list
//! below, and they are the caret's reach (1), the spelling of an escaped operator
//! (2), the operator's operand (3), the group proof ([`opens_group`] — where the
//! reader is looking for a command decides whether a `(` opens the block a break may
//! stand inside; 4–5), the spelling the comment command is recognised by (6), a line
//! that begins with a redirection (7) and a line break spelled as one lone carriage
//! return (8). The trailing-break rule rests on the one premise the owner measured
//! himself — the interpreter reads no line past the first break — so it is not part
//! of this list. The owner's hand check settles each of them in a minute, from a
//! plain prompt, one experiment per line:
//!
//! 1. `copy a.txt ^`, then `b.txt` on the next line — does that copy, i.e. does the
//!    caret's escape reach past the break into the line it continues? (This reading
//!    hands the two lines over as written, so a caret that stopped at the break would
//!    run `b.txt` as a command of its own.)
//! 2. `dir ^& x`, then `dir y` on the next line — is the escaped `&` ordinary text, so
//!    that the break after it is the separator this reading respells? (The same
//!    question for `^|`, `^<`, `^>`, `^(` and `^)`; the read-only mode's own checks
//!    read each of them as an operator and refuse the text, [`ESCAPED_SYNTAX`].)
//! 3. `dir &&`, then `next` on the next line — does `next` run, i.e. does an operator
//!    at the end of a line take its operand from the next one? (This reading joins the
//!    two lines into one command, so a wrong answer here runs what the platform does
//!    not.)
//! 4. `for %f in (a` then `b) do echo %f` — does the `(` open the block its `in` set
//!    is, so that the break inside it belongs to the block? (This reading hands the two
//!    lines over as written, so a `(` that opened nothing would lose the second.)
//! 5. `echo (a) done`, then `dir` on the next line — is `(a) done` echoed as text, or
//!    is `a` run as a command inside a block? (A `(` in an argument opens no block to
//!    this reading, [`opens_group`], so it respells the break and runs
//!    `echo (a) done&dir`.)
//! 6. `rem. note`, then `dir` on the next line — does `dir` run? (`rem[ note` asks the
//!    same; this reading respells the break and runs both, because a character
//!    appended to `rem` is not the comment command and the separator after it stands.
//!    A spelling that *is* the comment command takes the rest of the line, and the
//!    line after the break is swallowed — which is why a plain `rem` or a label is
//!    refused in every mode, [`SWALLOWING_LINE`].)
//! 7. `dir`, then `> out.txt` on the next line — is the file created as that line's own,
//!    the way the same two lines create it on unix?
//!    (This reading respells the break as `& `, so the next line keeps its redirect.)
//! 8. `dir`, a single carriage return, then `del x` — does this platform end a line at
//!    a carriage return with no line feed behind it? (This reading hands such a text
//!    over as written, so if it does, everything after that carriage return is lost
//!    silently.)
//!
//! Five residuals are recorded rather than read:
//!
//! - a break whose next line opens with a redirection is respelled as `& ` rather than
//!   `&`, so no `&>` is spelled and the redirection stays the next line's own. Where
//!   the interpreter takes a bare redirection as a command, that is what unix does
//!   with the two lines; where it does not, that line fails with the interpreter's own
//!   error instead of creating the file (hand check item 7);
//! - a caret-continued break is left as written although the guard's reader ends a
//!   command there and judges the words after it as a command of its own, while this
//!   platform runs them as ARGUMENTS of the command the caret spread over the two
//!   lines. Acceptable because the verb that executes is the verb the guard judged
//!   (never one it never looked at: a character the caret escapes into that reader's
//!   syntax is refused above) and because this shape must keep running as this
//!   platform reads it. It is the one place the words that reach a program differ
//!   from the guard's judgment, and the verbs whose rule gates their *destination*
//!   (`copy`, `move`, `ren`, `xcopy`) are where those extra operands could land in it;
//! - a redirection written immediately before a caret suppresses the escape of the
//!   line the caret continues, so a `&` there is a connector where this reading takes
//!   it for the escaped literal: read-only mode refuses that shape
//!   ([`ESCAPED_SYNTAX`]), the full mode keeps the text exactly as written;
//! - a break this reading *keeps* (inside a proven group, or behind a caret) reaches
//!   the grep engine with the break still in it, where that engine's Windows segmenter
//!   fails closed: a search inside such a command is refused as an unserved search, so
//!   a text does the same work whether or not it holds a search only where its breaks
//!   are the plain ones. That policy is the engine's own and unchanged here — the same
//!   text was handed to the same segmenter before this reading existed — and the
//!   refusal that carries it ends with this reading's own remedy, offered by the text's
//!   own shape (through the engine's
//!   [`unserved_failure`](crate::tools::shell::grep_engine::unserved_failure)), because
//!   a text only a file can run is exactly what a file fixes;
//! - a break standing at the very end of the text — the only break a single-line text
//!   has, and the last one a longer text has — is not part of the line this platform
//!   runs, while the guard's reader reads the ORIGINAL text and folds such a break into
//!   the line it stands on (a `\` immediately before it is that reader's line
//!   continuation, so its last word ends one character earlier than the one that runs).
//!   The command and every word before that character are the same, and this is the
//!   reading a single-line text always had: the platform's own reader reaches past no
//!   break, which is the one premise of this module the owner measured on his own
//!   machine.
//!
//! # Where a `(` opens a group
//!
//! Only where this platform's reader is looking for a command: at the start of the
//! text or of a command, after a separator this interpreter runs itself (`&`, `&&`,
//! `||`), first inside a group, after the `)` of a block it runs itself, where its `@`
//! prefixes the block, and where an `if`/`for` statement writes its body (the command
//! its condition governs, the `do`/`else` after the statement's own `)`, the `in (…)`
//! set). A `(` in an argument is ordinary text whose command ends at a break inside it,
//! so the two shapes cannot be told apart from the text alone: [`opens_group`] proves
//! the group where it can and the whole command is refused where it cannot
//! ([`UNPROVEN_GROUP`]) — as for a `(` anywhere in the command a single `|` hands to a
//! child interpreter, and for a `)` reaching past a `(` the reading cannot name
//! ([`UNPAIRED_PAREN`]). A `(` or `)` on a line a comment or a label has taken is that
//! line's text, so the reader's counter never sees it and [`Words::swallowing`] leaves
//! the group it stands in open — which is the text ending inside a group
//! ([`unclosed_cause`]) rather than a shape of its own.
//!
//! # Refusals
//!
//! Every other shape is refused whole, with the mode's own remedy ([`refusal`]); each
//! cause constant below states its own rule. The one split among them is the reading
//! they belong to: a divergence of a *break* belongs to the read-only guard's reader
//! (`readonly/windows.rs`), which reads the ORIGINAL text and is handed the text it
//! judged in [`ShellMode::ReadOnly`] alone — so those shapes are refused there and run
//! as this platform reads them in the other mode ([`divergent`], and the `bash_quote`
//! leg of [`unclosed_at_end`]). What this reading takes from that reader in both modes
//! is only [`TEXT_BLOCK`]: whether a `<<` sits inside a quoted word tells a text block
//! from text that merely spells one.
//!

use std::borrow::Cow;

use super::grep_engine::REFUSAL_FRAME;
use super::{ShellMode, ShellPlatform};

/// The platform's own limit on its single-line form: `cmd.exe`'s command line,
/// the `/C "…"` hand-off this shell spawns with, counted in the platform's own
/// unit, the UTF-16 code unit. The text the platform *reads* is what must fit it —
/// the text this module's reading produces, not the text the agent wrote —
/// because the interpreter refuses a longer line with its own line-length error
/// rather than running it; a refusal here says the same thing, with the number,
/// before anything runs.
///
/// Public to the shell because the tool's own guidance states the same limit.
pub(super) const COMMAND_LINE_CAP: usize = 8191;

/// What the `/C "…"` hand-off leaves of [`COMMAND_LINE_CAP`] for the command text
/// itself: the interpreter's own path (28 characters for the `cmd.exe` of a stock
/// install), the ` /C "` switch and the closing quote take the reserved 64, which
/// leaves room for a system directory spelling longer than a stock install's. Both
/// the refusal below and the tool's own guidance state this number, so neither
/// advertises a limit the text is refused under while it is shorter.
pub(super) const TEXT_UNIT_LIMIT: usize = COMMAND_LINE_CAP - 64;

/// A break inside the platform's own `"` span: its reader ends the command there.
const BREAK_IN_DOUBLE_QUOTES: &str = "a line break falls inside a double-quoted argument, and this platform's reader \
     ends the command at that break, so the argument would reach its program \
     truncated";

/// A break behind an odd run of `\`: ordinary text for `cmd.exe`, a line
/// continuation for the guard's bash-based reader.
const ODD_BACKSLASHES: &str = "an odd number of backslashes immediately before a line break, and this platform \
     treats `\\` as an ordinary character: the break there is a command separator, \
     while the read-only guard's reader folds the two lines into one";

/// A `<<` outside both readings' quotes: a text block fed to the interpreter,
/// which this platform has no text blocks for.
const TEXT_BLOCK: &str = "the command feeds a text block to the interpreter, and this platform has no \
     text blocks";

/// A caret over a character the read-only guard's reader reads as an operator or a
/// group (`&`, `|`, `<`, `>`, or a `(`/`)` inside the line): this platform reads the
/// escaped character literally, so the two readings do not agree about which text is
/// a command and neither can be shown to be what the other judged.
const ESCAPED_SYNTAX: &str = "a caret escapes a character this platform reads literally and the read-only \
     guard's reader reads as an operator or a group (`&`, `|`, `<`, `>`, `(`, `)`): \
     the two do not agree about which text is a command";

/// A caret at the end of a line whose escape reaches the `(` or `)` the next line
/// opens with: the escape makes that character ordinary text to this platform's
/// reader, and this reading takes a character an escape carried across a break as
/// nothing but a character, so the pairing it would read the following breaks from is
/// not the platform's own (`read_paren` counts parens this reading names itself).
const ESCAPED_PAREN: &str = "a caret at the end of a line escapes the `(` or `)` the next line opens with, and \
     this platform's reader takes that character as an ordinary one: its own group \
     counter is left out of step with the text, so the lines after the break cannot \
     be read as written";

/// A caret at the end of a line whose escape is spent on the break of a blank line:
/// this platform's reader takes the break itself as the escaped character, which a
/// respelling of the lines cannot carry — the break a blank line ends on is one the
/// rebuilt text drops.
const CARET_INTO_BLANK_LINE: &str = "a caret at the end of a line escapes the first character of the line it \
     continues, and that line is blank: this platform's reader is left taking the \
     line break itself as the escaped character, which the respelt lines cannot carry";

/// A break whose first significant character after it is `&` or `|`: the
/// connector has no left-hand command to join, so neither respelling can leave
/// the operator the agent wrote intact (the guard's reader refuses such a line
/// outright, and this platform reads it as a command opening with the connector).
const LEADING_CONNECTOR: &str = "the first significant character after a line break is a connector (`&` or `|`), \
     which joins the text after the break to nothing: this platform's reader has no \
     left-hand command for it, and no respelling of the break can leave the \
     operator the command was written with intact";

/// A `(` whose group this reading cannot prove: this platform's reader opens one
/// only where it is looking for a command, so a `(` written in an argument
/// (`echo a (`) is ordinary text whose line the reader ends at a break inside it,
/// while the same text may be the block an `if` statement or a `for` body was
/// written as. Running either as the other is what is refused here.
const UNPROVEN_GROUP: &str = "a `(` in the command may or may not open a bracketed group — this platform's \
     reader opens one only where it is looking for a command, and a `(` written in an \
     argument is ordinary text whose command ends where the arguments end — so \
     nothing that follows it can be read for certain";

/// A `)` this platform's counter cannot pair, written where its reader is looking
/// for a command: the reader discards the rest of that line, so nothing written
/// after it runs.
const UNPAIRED_PAREN: &str = "a `)` closes no group and stands where this platform's reader is looking for a \
     command: the reader discards the rest of that line, so the text after it cannot \
     be run as written";

/// A command whose last line leaves an `if`/`for` statement waiting for its own
/// text — its `do`, the command its condition governs, or the statement its set
/// belongs to: this platform's reader is left waiting for what the text never
/// supplies.
const UNFINISHED_STATEMENT: &str = "the command's last line leaves an `if`/`for` statement waiting for its own \
     text (its `do`, or the command its condition governs), and this platform's \
     reader is left waiting for what the text never supplies";

/// A break behind an `if`/`for` statement whose own text is written on the same
/// line: this platform's reader runs that text to the end of the line (or of the
/// block the statement is written in), so a separator here would join it — the
/// line after the break would run under the condition, or once per iteration.
const STATEMENT_SPANS_LINE: &str = "the line before the break ends an `if`/`for` statement whose own command is \
     written on it, and this platform's reader runs that command to the end of the \
     line: a separator here would become part of the statement, so the line after \
     the break would run only under the condition — or once per iteration — rather \
     than as a command of its own";

/// A comment (`rem`) or a label (`:`) line: this platform's reader takes the rest
/// of such a line as the comment's own text or the label's own name — the
/// separator a break would be joined with included — so the lines after the break
/// would be swallowed with no error and no status of their own.
const SWALLOWING_LINE: &str = "the line before the break is a comment (`rem`) or a label (`:`), and this \
     platform's reader takes the rest of such a line as the comment's text or the \
     label's name: the separator a break would be joined with would be part of it, \
     so the lines after the break would never run";

/// A break whose join would put two operator characters together: `>`+`>` spells
/// an append, `<`+`<` a text block, and `&`+`>` a separator followed by a
/// redirection — none of which the command wrote.
const DOUBLED_OPERATOR: &str = "joining the lines at this break would put two operator characters together, \
     spelling an operator the command never wrote (`>>`, `<<`, `&>` where it wrote the \
     characters separately)";

/// A caret whose escape has nothing left to take, at the end of the text that
/// would run: the reader is left waiting for what the caret escapes. An escaped
/// caret (`^^`) is not this shape.
const DANGLING_CARET: &str = "the command ends with a caret whose escape has nothing left to take, and this \
     platform's reader is left waiting for what it escapes";

/// What one line break means to the platform's own reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Break<'a> {
    /// The reader continues across it: the break stays exactly as written,
    /// whichever spelling it had.
    Continued(&'a str),
    /// The break goes: the operator before it and the operand after it become
    /// the one line the shell parser already read there.
    Joined,
    /// The reader ends the command there: respelled as `&`. The first significant
    /// byte after the break travels with it — `None` where nothing but the
    /// platform's own separators follows, i.e. where the break separates nothing —
    /// because that byte is what the separator is owed to and spaced by.
    Separator(Option<u8>),
}

/// The bracketed group a `(` opened, as far as the walk must know it: this
/// platform's reader pairs a `)` with the group its own counter holds open, and a
/// statement reads its `)` differently from a plain group's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Group {
    /// A `(` at a command position: no statement owns its `)`, which leaves a new
    /// command and nothing else.
    Plain,
    /// A `for`'s `in (…)` set: its `)` leaves the statement waiting for its `do`.
    Set,
    /// A statement's own body: its `)` satisfies the statement, and only a `do`/
    /// `else` of the same statement may still follow.
    Body,
    /// A `(` this reading cannot prove opened one. A `)` is counted against whatever
    /// this platform's counter holds even where the `)` stands in an argument, so an
    /// entry is recorded for a `(` this reading cannot name (the `)`s then pair the
    /// way this platform pairs them) and refused where a break or a `)` reaches it.
    Unproven,
}

/// Where in a statement's own grammar the command being read is. This platform's
/// `if` and `for` are the two commands whose word positions can open a bracketed
/// group at an argument position, so their words are the ones read here and nothing
/// else is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Stmt {
    /// Nothing of the command has been read: a `@` may still prefix its first word.
    #[default]
    Start,
    /// `@` alone was read: the word it prefixes follows.
    Prefixed,
    /// The first word opened no statement.
    Plain,
    /// `if` was read: its `not`, its `/i` switch or its condition follows.
    If,
    /// `if not` was read: its condition follows.
    Not,
    /// A condition keyword, a comparison operator or a `for`'s `in` was read: the
    /// word that completes it follows.
    Keyword,
    /// `for` was read: its variable and its `in` set follow.
    For,
    /// A `for`'s `in` was read: the set's `(` follows.
    Set,
    /// The statement's own text is still to come: the `do`/`else` of the statement
    /// whose group just closed, or the command a completed condition governs.
    Await,
    /// The command has left any statement: no `(` here opens a group.
    Done,
}

/// The word being read: where it starts, and what its first character was — whether it
/// stands where this platform's reader is looking for a command ([`Walk::at_command`],
/// which is what tells a command's first word from a later one), and whether a caret
/// escape carried it.
struct Word {
    at: usize,
    command: bool,
    escaped: bool,
}

/// The command being read, word by word: the positions a statement puts at an
/// argument position — an `if` condition, a `for` set, a `do`/`else` body — are read
/// off the words, because only they say where this platform's reader is looking for
/// a command. A word is the bytes since its first significant character; this
/// platform's separators end it, and so do `(` and `)`.
#[derive(Default)]
struct Words {
    /// The word being read, when one is open.
    open: Option<Word>,
    /// Where in a statement's grammar the command is.
    stmt: Stmt,
    /// Whether the `)` last read closed a group of a statement: the `do`/`else` that
    /// may follow it is what the chain is for.
    chain: bool,
    /// Whether this line holds a word where this platform's reader looks for a
    /// command that makes it take the rest of the LINE — a separator joined at the
    /// break behind it, and a paren and a label's `:` included — as text: its parser
    /// switches to a comment at such a `rem`, and a label is its own line's name, so
    /// the lines after such a break would be swallowed ([`SWALLOWING_LINE`]).
    swallowing: bool,
    /// Whether an `if`/`for` statement's own text is written on this line: this
    /// platform's reader runs such a text to the end of the line, so a `&` appended
    /// at a break would be read as part of the statement rather than as a separator
    /// ([`STATEMENT_SPANS_LINE`]).
    absorbing: bool,
    /// The group depth that bounds it: the block the statement is written in, whose
    /// end (like the line's) this platform's reader bounds such a text at.
    absorbing_bound: usize,
}

impl Words {
    /// Start a word at `at`, unless one is open: the word's first character decides
    /// where it begins, whether the reader is looking for a command there, and
    /// whether a caret escape carried it.
    fn start(&mut self, at: usize, command: bool, escaped: bool) {
        if self.open.is_none() {
            self.open = Some(Word {
                at,
                command,
                escaped,
            });
        }
    }

    /// End the word being read at `end`, reading it the way a statement's own words
    /// are read off a command. `depth` is the group the word is read in, which is what
    /// bounds a statement whose text starts here.
    fn finish(&mut self, bytes: &[u8], end: usize, depth: usize) {
        let Some(open) = self.open.take() else { return };
        let word = &bytes[open.at..end];
        let command = open.command;
        let escaped_start = open.escaped;
        let previous = self.stmt;
        let chain = self.chain;
        self.stmt = stmt_step(previous, chain, word, command);
        // A word read where this platform's reader looks for a command — at the start
        // of one (its `@` prefix included), after a completed condition, or after a
        // statement's own `)` — makes it take the rest of the LINE when its spelling is
        // the comment command (`rem`) or a label (`:name`). A label's name is its own
        // first character, so the escape a caret carried over that character makes it
        // inert and names no label — while the word a `rem` is written as still spells
        // itself (`^rem` is this platform's `rem`: the escape was over a letter, which
        // had no special meaning to lose).
        if (command || matches!(previous, Stmt::Prefixed | Stmt::Await) || chain)
            && (is_comment_command(word) || (!escaped_start && is_label_word(word)))
        {
            self.swallowing = true;
        }
        // The word a statement's own text starts with opens a text this platform's
        // reader runs to the end of the line, so a break behind it cannot carry a
        // separator. A comparison's other string is that text's condition rather than
        // its start, and a `do`/`else` a statement's own `)` leaves still only awaits
        // the text — the word after either is where the body begins.
        if (previous == Stmt::Await || chain)
            && matches!(self.stmt, Stmt::Done | Stmt::If | Stmt::For)
        {
            self.absorbing = true;
            self.absorbing_bound = depth;
        }
        // The chain is about the one `do`/`else` that may follow the `)` just read.
        self.chain = false;
    }

    /// End that absorption where the block the statement is written in has closed:
    /// this platform's reader bounds the statement's text at the line's end or at the
    /// end of that block, and only the second of the two is reached mid-text.
    fn bounded(&mut self, depth: usize) {
        if self.absorbing && depth < self.absorbing_bound {
            self.absorbing = false;
        }
    }

    /// Start a fresh statement machine and discard the words after it, keeping the one
    /// thing that outlives them: a statement's own text already running
    /// ([`Words::absorbing`]), whose words belong to the statement that opened the
    /// group — only the line's end and the block the statement is written in bound
    /// such a text ([`Words::bounded`]) — so losing it at a parenthesis would join the
    /// line after a break into the statement. A `(` or `)` a comment's or a label's
    /// line has taken never reaches here ([`read_paren`] returns before this, the
    /// reader's counter never seeing that paren).
    fn restart(&mut self) {
        let (absorbing, bound) = (self.absorbing, self.absorbing_bound);
        *self = Self::default();
        self.absorbing = absorbing;
        self.absorbing_bound = bound;
    }

    /// Whether the command read so far is a statement still waiting for its own
    /// text: a `for` without its `do`, an `if` without the command its condition
    /// governs, a set without the statement it belongs to.
    fn unfinished(&self) -> bool {
        matches!(
            self.stmt,
            Stmt::Prefixed
                | Stmt::If
                | Stmt::Not
                | Stmt::Keyword
                | Stmt::For
                | Stmt::Set
                | Stmt::Await
        )
    }
}

/// One word further into the statement a command spells out: `command` marks a word
/// standing where this platform's reader is looking for a command, and `chain`
/// whether the `)` just read closed a group of a statement.
fn stmt_step(stmt: Stmt, chain: bool, word: &[u8], command: bool) -> Stmt {
    // `@` in front of a word is this platform's own suppression prefix, not part of
    // it. A word that was nothing but `@` leaves the state as it was — the word it
    // prefixes, or the group it precedes, still follows.
    let word = word.strip_prefix(b"@").unwrap_or(word);
    if word.is_empty() {
        return if command { Stmt::Prefixed } else { stmt };
    }
    // The `do`/`else` a statement's own `)` leaves room for, and nothing else: the
    // statement's own text is written already, so any other word is a command the
    // `)` left — and the `(` a statement's text would open is not one after it.
    if chain {
        return if word.eq_ignore_ascii_case(b"else") || word.eq_ignore_ascii_case(b"do") {
            Stmt::Await
        } else {
            statement_word(word, Stmt::Done)
        };
    }
    if command || stmt == Stmt::Prefixed {
        return statement_word(word, Stmt::Plain);
    }
    match stmt {
        // A statement starts only where this platform's reader looks for a command,
        // and one that is over never starts again.
        Stmt::Start | Stmt::Prefixed | Stmt::Plain | Stmt::Done => Stmt::Plain,
        // `if`: its `not`, its `/i` switch, a condition keyword, a comparison
        // operator, or the string a comparison reads.
        Stmt::If | Stmt::Not => {
            if stmt == Stmt::If && word.eq_ignore_ascii_case(b"not") {
                Stmt::Not
            } else if word.starts_with(b"/") {
                // The `if` switch (`/i`): the condition still follows.
                stmt
            } else if is_condition_keyword(word) || is_comparison_operator(word) {
                Stmt::Keyword
            } else {
                Stmt::Await
            }
        }
        // The condition keyword's own operand, or a comparison's other string.
        Stmt::Keyword => Stmt::Await,
        // `for %f in (…) do …`: the variable, then the `in`, then the set's `(`.
        Stmt::For => {
            if word.eq_ignore_ascii_case(b"in") {
                Stmt::Set
            } else {
                Stmt::For
            }
        }
        // The set is written: the `do` the statement owes follows.
        Stmt::Set => {
            if word.eq_ignore_ascii_case(b"do") {
                Stmt::Await
            } else {
                Stmt::Done
            }
        }
        // The statement's own text: the `do`/`else` its group's `)` leaves, and a
        // body, which may be a statement of its own. A comparison's own spelling is
        // the condition's text rather than the statement's command: its spaced
        // operator still awaits its second string, and the operator with its second
        // string joined to it (`x ==1`) leaves only the command the condition
        // governs.
        Stmt::Await => {
            if is_comparison_operator(word) {
                Stmt::Keyword
            } else if word.starts_with(b"==") {
                Stmt::Await
            } else {
                statement_word(word, Stmt::Done)
            }
        }
    }
}

/// What a word that may be a statement's own text opens: a nested `if`/`for`, or
/// nothing at all — with `none` the state a word that is one leaves.
fn statement_word(word: &[u8], none: Stmt) -> Stmt {
    if word.eq_ignore_ascii_case(b"if") {
        Stmt::If
    } else if word.eq_ignore_ascii_case(b"for") {
        Stmt::For
    } else {
        none
    }
}

/// The text the platform's interpreter is handed: the reading of the line the agent
/// wrote ([`executable`]) — what runs.
///
/// A type of its own, beside [`Written`], because both texts are `&str` and a call
/// that passed them the other way round would compile while quoting the agent a
/// command it never wrote.
#[derive(Clone, Copy)]
pub(super) struct Executed<'a>(pub(super) &'a str);

/// The line the agent wrote, exactly as it wrote it: what a refusal and a
/// started-session message quote, and what never runs.
#[derive(Clone, Copy)]
pub(super) struct Written<'a>(pub(super) &'a str);

/// The text the platform actually reads for `command`.
///
/// On unix this is `command` itself, unchanged byte for byte: that interpreter reads
/// the whole text. On Windows it is the reading further down for a text whose breaks
/// separate commands, and otherwise the text itself — or, where every break stands at
/// its end ([`spans_commands`]), the one line that text is, which has no break for this
/// platform's reader to reach. `Err` is the agent-facing refusal (nothing runs): a shape
/// the two readings disagree about.
///
/// The reading is only the reading: the platform's line limit is measured on the texts
/// actually handed to the interpreter, at each hand-off, and [`check_command_line`] is
/// where that measurement and its rule live.
pub(super) fn executable(
    command: &str,
    platform: ShellPlatform,
    mode: ShellMode,
) -> Result<Cow<'_, str>, String> {
    // The unix interpreter reads the whole text itself.
    if platform == ShellPlatform::Unix {
        return Ok(Cow::Borrowed(command));
    }
    // A text whose breaks all stand at its end is the one line it was written as: this
    // platform's reader ends the command at the first of them and never reaches the
    // rest, so the reading below — every rule of it about what the line *after* a break
    // is read as — has nothing to decide, and a single-line command is never respelled,
    // the shapes only this platform accepts included.
    let text = if spans_commands(command) {
        Cow::Owned(respell(command, mode)?)
    } else {
        Cow::Borrowed(before_first_break(command))
    };
    Ok(text)
}

/// Where the line a break at `break_at` ends stops: every `\r` standing immediately
/// before the break belongs to the break rather than to the line, so one break reads as
/// one break however it is spelled and a rule that reads the line before it — the caret
/// the reader is left waiting for, the comment command, the guard's own `\` — reads the
/// same shape under any run of them. Every spelling of the platform's break this module
/// takes is measured here: the walk's own ([`break_ends`]), the one line a text with no
/// second command is ([`before_first_break`]), and the caret run the reader leaves
/// pending ([`spans_commands`]).
fn line_end(text: &str, break_at: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = break_at;
    while end > 0 && bytes[end - 1] == b'\r' {
        end -= 1;
    }
    end
}

/// Whether this platform's reader is given anything to read after a break: a character
/// after the break that is not one of the platform's own separators — a blank line and
/// the break after it give it nothing — or a caret escaping the break, whose line is the
/// continued one's own text. Neither is true of the break a text whose breaks all stand
/// at its end has, which is the one line [`executable`] hands over as written. The
/// premise is the one the owner measured on Windows: `cmd.exe` ends the command at the
/// first break and the lines after it never run, so a reading of them has nothing to
/// decide.
fn spans_commands(command: &str) -> bool {
    let bytes = command.as_bytes();
    let Some(break_at) = bytes.iter().position(|b| *b == b'\n') else {
        return false;
    };
    bytes[break_at + 1..].iter().any(|b| !is_separator(*b))
        // Every `\r` of a break is part of it, not a character of the line the caret
        // stands on.
        || trailing_run(&command[..line_end(command, break_at)], b'^') % 2 == 1
}

/// The line a text with no second command is: everything before its first break — or
/// the whole text, when it carries no break at all.
fn before_first_break(command: &str) -> &str {
    match command.find('\n') {
        Some(i) => &command[..line_end(command, i)],
        None => command,
    }
}

/// The platform's own command-line limit, measured on one text the interpreter is
/// handed: nothing on the platform whose interpreter takes the text as it comes
/// (`sh -c`), the refusal below on the one that reads a single line.
///
/// The refusal: the text the platform reads has to fit that line, and the interpreter
/// refuses a longer one with its own line-length error instead of reporting a status
/// for the command the agent wrote. The measurement is in the platform's own unit —
/// UTF-16 code units, where a character outside the basic plane counts twice — so a
/// line of astral-plane characters is not under-measured the way a `chars()` count
/// under-measures it. `mode` selects the remedy, exactly as every shape refusal does.
///
/// The shell measures it at each hand-off to an interpreter, and nowhere else. What is
/// measured is therefore always the text being handed over: a fold of a served rewrite,
/// or the reading the shell is spawned with when no plan serves it. A served rewrite's
/// whole reading is never measured: it is handed over one fold at a time and is longer
/// than the line it replaced, so measuring it whole would refuse a command whose own
/// folds all fit. A text no interpreter is handed — a lone call of this service's own
/// image, which the runner spawns from its own argv — is measured nowhere.
pub(super) fn check_command_line(
    text: &str,
    platform: ShellPlatform,
    mode: ShellMode,
) -> Result<(), String> {
    if platform != ShellPlatform::Windows {
        return Ok(());
    }
    let units = text.encode_utf16().count();
    if units <= TEXT_UNIT_LIMIT {
        return Ok(());
    }
    let cause = format!(
        "a command line on this platform holds {COMMAND_LINE_CAP} units in all, and the \
         interpreter's own path and its `/C` switch take about {} of them, so the command \
         text must fit in {TEXT_UNIT_LIMIT}: this text needs {units}",
        COMMAND_LINE_CAP - TEXT_UNIT_LIMIT
    );
    Err(refusal(&cause, mode))
}

/// The line the break at `i` ends, where that line stops, and that break's own bytes —
/// the run of `\r`s standing immediately before it and the break itself. A break the
/// reader continues across is re-emitted from those bytes, byte for byte, and the
/// position is the word end the walk measures ([`line_end`]), returned here rather
/// than re-derived from the bytes' length.
fn break_ends(command: &str, i: usize, start: usize) -> (&str, usize, &str) {
    let end = line_end(command, i);
    (&command[start..end], end, &command[end..=i])
}

/// The first significant byte at or after every position of `bytes`, computed once
/// from the end: the break rules read it and so does the separator the rebuild
/// spells from it, so one fold answers both (a blank run would otherwise be
/// rescanned for every break inside it). Byte-level on purpose — `&` and `|` are
/// ASCII, so a multi-byte character can never be mistaken for either (a
/// continuation byte of one is not one of them). `None` where nothing but the
/// platform's own separators follows.
fn next_significant(bytes: &[u8]) -> Vec<Option<u8>> {
    let mut following = vec![None; bytes.len() + 1];
    let mut seen = None;
    for j in (0..bytes.len()).rev() {
        if !is_separator(bytes[j]) {
            seen = Some(bytes[j]);
        }
        following[j] = seen;
    }
    following
}

/// What the walk's group stack holds when a break or a `)` reaches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    /// No group is open.
    None,
    /// Every open group was proven.
    Proven,
    /// A `(` this reading cannot prove is open.
    Unknown,
}

/// A line that does not end where its break is: this platform's reader runs such a
/// line to its own end, so a separator joined at the break behind it would be read as
/// part of the line rather than as a separator — the lines after it would run under a
/// statement's condition, once per iteration, or never at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Takes {
    /// Nothing: the line ends at the break, as a line is meant to.
    #[default]
    Nothing,
    /// The line ends an `if`/`for` statement's own command, which the reader runs to
    /// the end of the line ([`STATEMENT_SPANS_LINE`]).
    Statement,
    /// The line is one the reader takes whole — a comment or a label
    /// ([`SWALLOWING_LINE`]).
    Line,
}

/// What the walk has read when it meets a break: the line that ends on it and the
/// state the rules read.
struct BreakSite<'a> {
    /// The line the break ends.
    segment: &'a str,
    /// The first significant byte after the break, `None` where nothing but the
    /// platform's own separators follows: a break with nothing after it separates
    /// nothing, so the rules that would otherwise refuse the break leave it alone.
    following: Option<u8>,
    /// The character a caret at the end of the line escapes — the first character
    /// of the line after the break — when the line ends on an odd caret run: this
    /// platform's own way of spreading one command over several lines, whose escape
    /// reaches into the line it continues. `None` with [`BreakSite::caret_unescaped`]
    /// set when there is no such line at all.
    caret_escapes: Option<u8>,
    /// Whether the line ends on a caret whose escape has nothing to take: the text
    /// ends here, so this platform's reader is left waiting for what it escapes
    /// ([`DANGLING_CARET`]).
    caret_unescaped: bool,
    /// What the groups this reading has opened and paired left open.
    groups: Open,
    /// Whether the command read so far is a statement still waiting for its own
    /// text — a `for` without its `do`, an `if` without its command.
    statement_unfinished: bool,
    /// The guard's own reading of the text so far, when one of its quoted words is
    /// open: this platform's reader quotes with `"` alone.
    bash_quote: Option<u8>,
    /// The line's last significant character.
    last: Option<u8>,
    /// What the line before the break takes with it ([`Takes`]).
    takes: Takes,
}

/// What this platform's reader makes of one break, in the reading's own order: the
/// caret's continuation first, then the two readings' disagreement, then the groups
/// and statements that are open, then the operator the line ends on (and the
/// redirection the line after it opens with). `Err` is the refusal for a break
/// nothing can be respelled into.
fn break_kind<'a>(
    site: &BreakSite<'_>,
    spelling: &'a str,
    mode: ShellMode,
) -> Result<Break<'a>, String> {
    if site.caret_unescaped {
        // The text ends on a caret: the character its escape would take is not there,
        // and this platform's reader is left waiting for it. Refused here rather than
        // at the end of the walk because the break behind such a caret is one the
        // ordinary rules would drop — the rebuilt text would end in an even caret run
        // and lose the shape.
        return Err(refusal(DANGLING_CARET, mode));
    }
    if let Some(target) = site.caret_escapes {
        // The caret escapes the break and the first character of the line it
        // continues, which this reading consumes as literal text (the walk's own
        // escape), so the two lines are read as the one command here.
        if target == b'\n' {
            // A break is not a character the respelling can carry: the rebuilt text
            // drops the break a blank line ends on.
            return Err(refusal(CARET_INTO_BLANK_LINE, mode));
        }
        // A paren is the other target that cannot be read that way: pairing it is the
        // group stack's own work, and a group this platform's counter does not open
        // would leave a break after it with no reading the two agree on.
        if matches!(target, b'(' | b')') {
            return Err(refusal(ESCAPED_PAREN, mode));
        }
        // An operator is literal text here and structure to the read-only guard's
        // reader: that reader judged a different text, so that mode refuses the shape
        // while the full mode runs it as written.
        if mode == ShellMode::ReadOnly && matches!(target, b'&' | b'|' | b'<' | b'>') {
            return Err(refusal(ESCAPED_SYNTAX, mode));
        }
        return Ok(Break::Continued(spelling));
    }
    // Nothing follows the break: this platform's reader ends the command here and the
    // text after it is the platform's own separators alone — a blank line, another
    // break, a run of both — so every rule below, each of them about what the line
    // *after* the break is read as, has nothing to decide. The break is dropped:
    // [`rebuild`] joins nothing at it, and the text is the one line it was written as.
    if site.following.is_none() {
        return Ok(Break::Separator(None));
    }
    // The shapes the two readings disagree about are the same inside a group as
    // outside it, so they are decided before anything continues.
    if let Some(cause) = divergent(site, mode) {
        return Err(refusal(&cause, mode));
    }
    match site.groups {
        // A break continues the command if the `(` opened a group and ends it there if
        // it did not, and the text cannot be shown to be the block it may have been.
        Open::Unknown => return Err(refusal(UNPROVEN_GROUP, mode)),
        // A break inside a bracketed group: this platform's own block spans lines.
        Open::Proven => return Ok(Break::Continued(spelling)),
        Open::None => {}
    }
    if site.statement_unfinished {
        // A statement's own text is missing, and the group it belongs to has closed:
        // this platform's reader is left waiting for what the text never supplies.
        return Err(refusal(UNFINISHED_STATEMENT, mode));
    }
    // The line before the break takes the text after it with it, so a separator joined
    // here would be read as part of that line.
    match site.takes {
        // The reader runs a statement's own command to the end of the line: the line
        // after the break would run under the condition — or once per iteration —
        // instead of as a command of its own.
        Takes::Statement => return Err(refusal(STATEMENT_SPANS_LINE, mode)),
        // The reader takes the line whole — a comment or a label — so the lines after
        // the break would never run at all.
        Takes::Line => return Err(refusal(SWALLOWING_LINE, mode)),
        Takes::Nothing => {}
    }
    // A line opening with a redirection behind an operator: joining would put the
    // two characters together into an operator the command never wrote.
    let operator = site
        .last
        .is_some_and(|c| matches!(c, b'&' | b'|' | b'<' | b'>'));
    if operator && matches!(site.following, Some(b'<' | b'>')) {
        return Err(refusal(DOUBLED_OPERATOR, mode));
    }
    if operator {
        // The operand after the break joins the operator before it on one line, the
        // single command the shell parser already reads there.
        return Ok(Break::Joined);
    }
    Ok(Break::Separator(site.following))
}

/// The divergence of one break, when this platform's reader and the guard's own
/// would read it differently: an odd run of `\` (a line continuation for the guard's
/// reader, an ordinary character here) and a break the guard's reader holds inside a
/// quoted word while this platform's reader ends the command there, where the two do
/// not agree about which text is a command. A quoted word holding the break is named
/// first, and the odd `\` run only where no quoted word does: a `\` inside a `'…'`
/// word is ordinary text to that reader, and one before a break inside a `"…"`
/// word escapes the break itself, so in either spelling what diverges is the word
/// the reader carries across the break.
///
/// Both are the *guard's* reading, so both are decided in the mode whose guard reads
/// the original text ([`ShellMode::ReadOnly`]): the full mode consults no such reader
/// (`readonly::check_command` is called on the read-only path alone), and there the
/// platform's own reading of the two lines is the only one there is.
fn divergent(site: &BreakSite<'_>, mode: ShellMode) -> Option<String> {
    // A connector with no left-hand command to join is a shape neither reader can
    // run, in either mode.
    if matches!(site.following, Some(b'&' | b'|')) {
        return Some(LEADING_CONNECTOR.to_owned());
    }
    if mode != ShellMode::ReadOnly {
        return None;
    }
    if let Some(open) = site.bash_quote {
        return Some(divergent_quote_cause(open));
    }
    if trailing_run(site.segment, b'\\') % 2 == 1 {
        return Some(ODD_BACKSLASHES.to_owned());
    }
    None
}

/// Read `command` line by line and rebuild the text the platform will run: the
/// reading the module doc describes, spelled out for a reader that has one line.
fn respell(command: &str, mode: ShellMode) -> Result<String, String> {
    let bytes = command.as_bytes();
    let mut lines: Vec<(&str, Break<'_>)> = Vec::new();
    let following = next_significant(bytes);
    let mut walk = Walk::default();
    // Where the line being read starts.
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];

        // A break is never the character an escape takes: `cmd.exe`'s caret before
        // one is its own continuation (read below), the shell parser's `\` before
        // one is the line continuation refused below, and a break inside the
        // platform's `"` span is refused outright.
        if b == b'\n' {
            let (segment, kind) = walk.at_break(command, bytes, i, start, &following, mode)?;
            lines.push((segment, kind));
            start = i + 1;
            i += 1;
            continue;
        }

        // The shell parser's own reading of this character, tracked beside the
        // platform's own.
        let (quote, escaped) = bash_step(walk.bash_quote, walk.bash_escaped, b);
        walk.bash_quote = quote;
        walk.bash_escaped = escaped;

        // A carriage return is not a character this platform's reader sees in the
        // phase these rules read: it ends no word, is not the one the line's last is,
        // and is not the one an escape takes (a `^` before it escapes the break after
        // it). It stays in the text as written.
        if b == b'\r' {
            i += 1;
            continue;
        }

        // Two adjacent `<` are a text block to the shell parser whether or not a
        // caret escape is pending over the first one: the escape makes it literal
        // here while the parser, which has no caret escape, still reads `<<`. This
        // platform has no text block, so a spelling it cannot run as written must
        // not have its lines respelled into commands. Checked outside both
        // readings' quotes, and before the escape below consumes the character.
        if b == b'<' && bytes.get(i + 1) == Some(&b'<') && !walk.quote && walk.bash_quote.is_none()
        {
            return Err(refusal(TEXT_BLOCK, mode));
        }

        // The character a caret escapes opens no span and changes no group for this
        // platform's reader — it is ordinary text — so it is consumed here, and what
        // the guard's own reader makes of it is what [`read_escaped`] decides. It does
        // open the word it is written in, though: a token whose first letter an escape
        // carried is the token it spells (`^rem` is this platform's `rem`), while the
        // character itself is inert ([`Word::escaped`]).
        if walk.escaped {
            read_escaped(b, mode)?;
            walk.words.start(i, walk.at_command(), true);
            walk.escaped = false;
        } else if walk.quote {
            walk.quote = b != b'"';
        } else {
            walk.step(bytes, i, b, mode)?;
        }
        i += 1;
    }

    // The line the run ends on, kept verbatim.
    let tail = &command[start..];
    // The word the text ends on is finished for the same reason a break's own word is
    // (`Words::finish`'s separators): a statement completed by the last word is not one
    // still waiting for its own text.
    walk.words.finish(bytes, bytes.len(), walk.groups.len());
    walk.close(&lines, tail, mode)
}

/// The state one line of the walk leaves for the next: the spans the reader holds
/// open, the groups it counts, the words a statement's own positions are read off,
/// and the escape a caret leaves pending. Each `bool` here is a separate notion a
/// reader of this text keeps — an open span of one grammar, an escape of the other,
/// a caret's escape, a pipeline, and the block position a `(` after a `)` is read
/// from — so they are kept apart on purpose.
#[expect(clippy::struct_excessive_bools)] // each bool a notion of its own
#[derive(Default)]
struct Walk {
    /// The platform's own `"` span, when one is open: `cmd.exe` quotes with `"`
    /// only, and a caret's escape swallows a quote it precedes.
    quote: bool,
    /// The shell parser's own reading of the same text (the read-only guard's bash
    /// grammar), advanced beside it because the two do not agree: `'` quotes there
    /// and not here, and `\` escapes there and not here. The open span, when one is
    /// open, and a `\` whose escape is pending.
    bash_quote: Option<u8>,
    bash_escaped: bool,
    /// A caret's escape is pending over the next character — or was left pending by
    /// the break a caret at the end of a line escapes.
    escaped: bool,
    /// The bracketed groups this reading has opened, each with the shape that opened
    /// it. A `)` is counted against whatever this platform's counter holds — wherever
    /// the `)` itself stands — so an entry is recorded for every `(`, including one
    /// whose group the reading cannot prove; such an entry is refused where a break or
    /// a `)` reaches it ([`read_paren`]).
    groups: Vec<Group>,
    /// How many entries of [`Walk::groups`] are [`Group::Unproven`], kept beside the
    /// stack so the break rules and a `)` read it without rescanning a stack a text
    /// can make as long as itself ([`Walk::open`], [`Walk::pop_group`]).
    unproven: usize,
    /// The last unescaped character of the line that is not a space or a tab — what a
    /// break's separator reading looks at, and where a `(` sits.
    last: Option<u8>,
    /// Whether that last character is the `|` of a pipeline: a single `|` is, while
    /// the `||` this platform concatenates command lines with is not, and the group a
    /// `(` after either of them opens is counted by a different interpreter
    /// ([`opens_group`]).
    piped: bool,
    /// Whether the command being read is the right-hand side of that single `|` — a
    /// command a child interpreter runs. Set where that connector is read and cleared
    /// where the connector after it is read, or where a break leaves the walk the new
    /// command it ends — but, unlike [`Walk::piped`], kept across the ordinary
    /// characters of the command, because `| @(` is the pipeline's own spelling too
    /// ([`opens_group`]).
    child_pipe: bool,
    /// Whether the `)` last read closed a block of this platform's own rather than a
    /// statement's: its reader is looking for a command again there, so a `(` after it
    /// is a block of its own ([`opens_group`]).
    after_block: bool,
    /// The words of the command being read, for the statement positions that sit at
    /// an argument position ([`opens_group`]).
    words: Words,
}

impl Walk {
    /// Read one ordinary character of a line into the state above: this platform's
    /// quote, the caret's pending escape, a group's `(`/`)`, and the words a
    /// statement's own positions are read off. A `<<` is checked by the caller, which
    /// is where the guard's own reading of it is known.
    fn step(&mut self, bytes: &[u8], i: usize, b: u8, mode: ShellMode) -> Result<(), String> {
        match b {
            // A quote is a character of the line like any other — an argument
            // boundary — so it becomes the break's last significant character:
            // `dir >"out.txt"` ends its command after the quoted target, not
            // after the operator the target was written behind.
            b'"' => {
                self.words.start(i, self.at_command(), false);
                self.quote = true;
                self.last = Some(b);
                self.piped = false;
                self.after_block = false;
            }
            b'^' => self.escaped = true,
            b'(' | b')' => {
                read_paren(b, bytes, i, self, mode)?;
                self.last = Some(b);
                if b == b'(' {
                    // A group starts a command first inside it; what the statement
                    // that opened it is still reading is [`read_paren`]'s business.
                    self.after_block = false;
                }
                self.piped = false;
            }
            b' ' | b'\t' => {
                // The word being read ends here, but the command's own words
                // outlive the separator: an `if` condition and a `do`/`else` are
                // words, and the `(` they open follows a space.
                self.words.finish(bytes, i, self.groups.len());
            }
            b'&' | b'|' => {
                // A connector ends the word before it and starts a command after it
                // (`&&if exist x (` is the statement `&& if exist x (`), so the
                // `do`/`else` a statement's own `)` leaves can no longer follow.
                self.words.finish(bytes, i, self.groups.len());
                self.words.chain = false;
                // One `|` hands the command after it to a child interpreter; `||`, like
                // `&`/`&&`, is a command this interpreter runs itself.
                self.piped = b == b'|' && self.last != Some(b'|');
                self.child_pipe = self.piped;
                self.after_block = false;
                self.last = Some(b);
            }
            _ => {
                self.words.start(i, self.at_command(), false);
                self.last = Some(b);
                self.piped = false;
                self.after_block = false;
            }
        }
        Ok(())
    }

    /// Whether this platform's reader is looking for a command at the character about
    /// to be read: the start of a command (after a separator this interpreter runs
    /// itself — the single `|` included, whose right-hand side a child interpreter
    /// runs as a command of its own), the first word inside a group this reading
    /// proved (a `for`'s set holds files, not commands), the word after the `)` of a
    /// block it proved, and the word this platform's own `@` prefixes. [`opens_group`]
    /// reads the `(` such a position opens off the same state — except at the pipeline,
    /// whose group the child interpreter counts and this reading cannot.
    fn at_command(&self) -> bool {
        match self.last {
            None | Some(b'&' | b'|') => true,
            Some(b'(') => matches!(self.groups.last(), Some(Group::Plain | Group::Body)),
            Some(b')') => self.after_block,
            Some(b'@') => self.words.stmt == Stmt::Prefixed,
            _ => false,
        }
    }

    /// Which groups are open, as the break rules read them.
    fn open(&self) -> Open {
        if self.unproven > 0 {
            Open::Unknown
        } else if self.groups.is_empty() {
            Open::None
        } else {
            Open::Proven
        }
    }

    /// Record a `(`, keeping the unprovable count beside the stack.
    fn push_group(&mut self, group: Group) {
        self.unproven += usize::from(group == Group::Unproven);
        self.groups.push(group);
    }

    /// Take the last `(` back off, keeping that count.
    fn pop_group(&mut self) -> Option<Group> {
        let group = self.groups.pop();
        if group == Some(Group::Unproven) {
            self.unproven -= 1;
        }
        group
    }

    /// Read the break at `i`: the line it ends, what the break means, and the state
    /// the line after it is read with.
    fn at_break<'a>(
        &mut self,
        command: &'a str,
        bytes: &[u8],
        i: usize,
        start: usize,
        following: &[Option<u8>],
        mode: ShellMode,
    ) -> Result<(&'a str, Break<'a>), String> {
        let (segment, end, spelling) = break_ends(command, i, start);
        // The word the line ends on is finished here, before the break rules read the
        // statement state: this platform's separators end a word at a break too, so a
        // statement completed on this line is not read as one still waiting. The line
        // stops where the break's own bytes start, every `\r` of them included
        // ([`line_end`]).
        self.words.finish(bytes, end, self.groups.len());
        self.bash_escaped = false;
        if self.quote {
            return Err(refusal(BREAK_IN_DOUBLE_QUOTES, mode));
        }
        // A caret at the end of the line — the walk's own pending escape — escapes this
        // break and reaches the first byte after it that this platform's reader sees: a
        // carriage return is one it never sees, the same skip the walk makes inside a
        // line.
        let caret = self.escaped;
        let target = caret
            .then(|| bytes[i + 1..].iter().copied().find(|b| *b != b'\r'))
            .flatten();
        let site = BreakSite {
            segment,
            following: following[i + 1],
            caret_escapes: target,
            caret_unescaped: caret && target.is_none(),
            groups: self.open(),
            statement_unfinished: self.words.unfinished(),
            bash_quote: self.bash_quote,
            last: self.last,
            takes: if self.words.swallowing {
                Takes::Line
            } else if self.words.absorbing {
                Takes::Statement
            } else {
                Takes::Nothing
            },
        };
        let kind = break_kind(&site, spelling, mode)?;
        // A pending escape survives the break by itself — the walk's own escape branch
        // consumes it on the next line — and the comment's or the label's reach is the
        // line's own, so what follows the break is otherwise read afresh.
        self.words.swallowing = false;
        // A break that ends a command leaves the next line a new one, and the
        // statement machine starts again; the breaks `carried` names leave it
        // running instead, and under a joined one the operand continues the line
        // the operator was written on — so what `(` opens is still read off it.
        match kind {
            Break::Joined => {}
            _ if carried(&site) => {}
            _ => {
                self.last = None;
                self.piped = false;
                self.child_pipe = false;
                self.after_block = false;
                self.words = Words::default();
            }
        }
        Ok((segment, kind))
    }

    /// The text the lines rebuild into, or the refusal for a text that ends inside a
    /// span or bracket it never closes, mid-way through a statement's own grammar, or
    /// with a caret that has nothing left to escape.
    fn close(
        &self,
        lines: &[(&str, Break<'_>)],
        tail: &str,
        mode: ShellMode,
    ) -> Result<String, String> {
        // A caret the walk left pending where the text ends has nothing to escape either
        // — the text ends without a break for it to reach, which is the same shape the
        // break site refuses a step earlier ([`DANGLING_CARET`]). Read before the open
        // spans and groups, as the break site reads it, so a text ending in one is
        // refused for the same cause with and without a break behind it.
        if self.escaped {
            return Err(refusal(DANGLING_CARET, mode));
        }
        if let Some(cause) = unclosed_at_end(
            self.quote,
            self.bash_quote,
            self.open(),
            self.words.unfinished(),
            mode,
        ) {
            return Err(refusal(&cause, mode));
        }
        Ok(rebuild(lines, tail))
    }
}

/// Read the character a caret escape reaches: literal text here, while the guard's
/// reader reads a `&`/`|`/`<`/`>`/`(`/`)` as an operator or a group — so the shape
/// is refused ([`ESCAPED_SYNTAX`]) in the mode whose guard read the original text,
/// and run as written in the other. Either way the character is consumed as this
/// platform reads it: it opens no span and changes no group here, and stays part of
/// the word it is written in.
fn read_escaped(b: u8, mode: ShellMode) -> Result<(), String> {
    if mode == ShellMode::ReadOnly && matches!(b, b'&' | b'|' | b'<' | b'>' | b'(' | b')') {
        return Err(refusal(ESCAPED_SYNTAX, mode));
    }
    Ok(())
}

/// Whether a break leaves the word being read and the statement machine running: the
/// caret's continuation does, and so does a break inside a group while a statement
/// still waits for its own text — this platform's reader appending the next line to
/// that statement — and so does a break with nothing after it, which separates nothing
/// and so ends nothing: the text is read as the same line whether it stands there or
/// not ([`spans_commands`]).
fn carried(site: &BreakSite<'_>) -> bool {
    site.caret_escapes.is_some()
        || site.following.is_none()
        || (site.groups == Open::Proven && site.statement_unfinished)
}

/// Read one `(` or `)`: which group it opens, or what closing one leaves the
/// statement being read. An unprovable `(` is still an entry — this platform's
/// reader counts one wherever it stands, so the stack mirrors its counter — and
/// what it hides is refused where a break or a `)` reaches it.
///
/// `Err` is the refusal for the one shape that would put this reading's groups and
/// this platform's own counter out of step ([`UNPROVEN_GROUP`]), and for a `)` the
/// counter cannot pair where this platform's reader is looking for a command
/// ([`UNPAIRED_PAREN`]).
fn read_paren(
    paren: u8,
    bytes: &[u8],
    i: usize,
    walk: &mut Walk,
    mode: ShellMode,
) -> Result<(), String> {
    // The word the paren ends is finished before the opener is read: the statement
    // positions that sit at an argument position are read off it — and that word is
    // what decides that this line's rest is this platform's comment text or a label's
    // name, the paren that ends it included.
    walk.words.finish(bytes, i, walk.groups.len());
    // A paren the line's own text has taken is text to this platform's reader: its
    // counter never sees it ([`SWALLOWING_LINE`]), so nothing here closes and nothing
    // opens one.
    if walk.words.swallowing {
        return Ok(());
    }
    if paren == b'(' {
        let group = opens_group(
            walk.last,
            walk.piped,
            walk.child_pipe,
            walk.after_block,
            &walk.words,
        )
        .unwrap_or(Group::Unproven);
        walk.push_group(group);
        // A group at a command position starts a command of its own, so the words read
        // so far belong to the statement that opened it. A `(` this reading could not
        // prove opened none: its words are an argument's and stay, or the statement a
        // break behind it is refused for would look finished here.
        if group != Group::Unproven {
            walk.words.restart();
        }
        return Ok(());
    }
    match walk.pop_group() {
        Some(Group::Plain) => {
            // The block this platform's reader runs itself is closed, so it is looking
            // for a command again: a `(` after the `)` opens a group of its own.
            walk.words.restart();
            walk.after_block = true;
        }
        Some(Group::Body) => {
            // The statement's own group is closed: only its `do`/`else` may still
            // follow, and it is read off this `)` and no other. Its own text is
            // written, and this platform's reader runs that text to the end of the
            // line or of the block the statement sits in.
            walk.words.restart();
            walk.words.chain = true;
            walk.words.absorbing = true;
            walk.words.absorbing_bound = walk.groups.len();
        }
        Some(Group::Set) => {
            // The `for` awaits its `do`, which may still follow.
            walk.words.restart();
            walk.words.stmt = Stmt::Set;
        }
        // A `)` this platform's reader counts against a `(` this reading cannot
        // name, with a proven group below it: it may be closing that group, so what
        // the text after it holds is not known either.
        Some(Group::Unproven) if walk.groups.len() != walk.unproven => {
            return Err(refusal(UNPROVEN_GROUP, mode));
        }
        // A `)` this platform's counter cannot pair, where its reader is looking for
        // a command — behind a separator, first inside a group, after a statement's own
        // group (`if exist x (echo a) )`) or at the start: the reader discards the rest
        // of the line there, so nothing written after it runs.
        None if matches!(walk.last, None | Some(b'&' | b'|' | b'(' | b')')) => {
            return Err(refusal(UNPAIRED_PAREN, mode));
        }
        // A group whose `)` leaves a new command, a `)` counting a `(` this reading
        // only recorded, and a `)` this platform's counter ignores where its reader
        // is not looking for a command: nothing follows from any of them.
        Some(Group::Unproven) | None => {}
    }
    // A block that closed here may hold the statement a text of its own is bounded
    // by: the end of that block bounds it as the line's end does.
    walk.words.bounded(walk.groups.len());
    Ok(())
}

/// The cause for a text that ends inside a span or bracket it never closes or mid-way
/// through a statement's own grammar: this platform's reader would be left waiting for
/// the rest of the command. `None` when the text ends where it should. A caret left
/// with nothing to escape is the same kind of waiting, refused where the walk reads it
/// ([`DANGLING_CARET`]).
///
/// `bash_quote` is the guard's reader alone, so a span it holds open is decided in
/// the mode whose guard read the original text ([`divergent`] says why): no
/// *divergence* of a break is decided anywhere but there, and a `'`-word or a
/// caret-escaped quote is no span to this platform's.
fn unclosed_at_end(
    platform_quote: bool,
    bash_quote: Option<u8>,
    groups: Open,
    statement_unfinished: bool,
    mode: ShellMode,
) -> Option<String> {
    if platform_quote {
        return Some(unclosed_cause(b'"'));
    }
    if mode == ShellMode::ReadOnly
        && let Some(open) = bash_quote
    {
        return Some(unclosed_cause(open));
    }
    match groups {
        Open::Unknown => return Some(UNPROVEN_GROUP.to_owned()),
        Open::Proven => return Some(unclosed_cause(b'(')),
        Open::None => {}
    }
    if statement_unfinished {
        return Some(UNFINISHED_STATEMENT.to_owned());
    }
    None
}

/// Rebuild the platform's one line from the lines the walk read: each segment
/// verbatim, minus the break it ended on — or with that break exactly as written,
/// carriage returns and all, where the reader continues across it.
fn rebuild(lines: &[(&str, Break<'_>)], tail: &str) -> String {
    let mut out = String::new();
    // Whether the text emitted so far ends a command: the separator a break owes is
    // owed only to a command it separates from, so a blank run — of lines, or of the
    // spaces a blank line holds — takes one separator, not one per break. What
    // follows a break travels on the `Break` itself, folded once with the break rules.
    let mut ends_command = false;
    for (segment, kind) in lines {
        out.push_str(segment);
        ends_command |= !is_blank(segment);
        match kind {
            // Always as written, a blank line included: the blank line stays
            // inside the continued shape.
            Break::Continued(spelling) => out.push_str(spelling),
            Break::Joined => {}
            Break::Separator(following) => {
                if ends_command && following.is_some() {
                    out.push('&');
                    // Spaced where a redirection opens the line after it: no `&>` is
                    // spelled, and the redirection stays the next line's own.
                    if matches!(following, Some(b'<' | b'>')) {
                        out.push(' ');
                    }
                    ends_command = false;
                }
            }
        }
    }
    out.push_str(tail);
    out
}

/// The reading of a text that ends inside a span or bracket `open` never closes:
/// the platform's reader would be left waiting for the rest of the command.
fn unclosed_cause(open: u8) -> String {
    let shape = match open {
        b'"' => "double-quoted argument",
        b'\'' => "single-quoted word",
        _ => "bracketed group",
    };
    format!(
        "the text ends inside an unclosed {shape} — this platform's reader would be \
         left waiting for the rest of the command"
    )
}

/// The reading of a break this platform's reader ends the command at, while the
/// shell parser that read the command reads one quoted word straight across it:
/// `open` is the quote that opened that word's span. `cmd.exe` quotes with `"`
/// only, and each reader's own escape (`\` there, `^` here) hides a quote from
/// one of them, so the two spans do not line up and the readers do not agree about
/// which text is a command — neither deleting the break nor respelling it keeps
/// what either reader took as quoted.
fn divergent_quote_cause(open: u8) -> String {
    let shape = match open {
        b'\'' => "single-quoted word",
        _ => "double-quoted word",
    };
    format!(
        "a line break falls inside a {shape} — this platform's reader quotes with \
         `\"` only, so the command ends at that break while the shell parser that \
         read this command reads one quoted word across it, and the two do not agree \
         about which text is a command"
    )
}

/// Which group the `(` reached now opens, read off where it sits and the statement
/// before it: this platform's reader opens one only where it is looking for a
/// command, and only an `if`/`for` statement puts one at an argument position.
/// `None` is a `(` written where this reading cannot prove one — in an argument, or
/// anywhere in the command a single `|` handed to a child interpreter, whose own
/// group counting is not modelled here (`echo x | (` … `)`) — so a break reaching it
/// is refused rather than read.
fn opens_group(
    last: Option<u8>,
    piped: bool,
    child_pipe: bool,
    after_block: bool,
    words: &Words,
) -> Option<Group> {
    // The right-hand side of a single `|` is a command a child interpreter runs, and
    // its own command line is what counts the groups in it — a reading this module
    // does not model, so nothing there can be proven. The pipeline's prefix is part of
    // its spelling (`| @(` is the same position), which is what makes this a state of
    // the command rather than of the last character ([`Walk::child_pipe`]).
    if child_pipe {
        return None;
    }
    match last {
        // The start of the text or of a command: after a separator — `||` included,
        // which this platform runs in this interpreter like `&`/`&&` — or first inside
        // a group.
        None | Some(b'&' | b'(') => Some(Group::Plain),
        Some(b'|') if !piped => Some(Group::Plain),
        // The `)` that closed a block of this platform's own leaves the reader looking
        // for a command again, and so does the `@` its no-echo prefix is written with:
        // `@( … )` is one of the blocks it suppresses the echo of.
        Some(b')') if after_block => Some(Group::Plain),
        Some(b'@') if words.stmt == Stmt::Prefixed => Some(Group::Plain),
        // A `for`'s `in` set: the statement awaits its `do` after it.
        _ if words.stmt == Stmt::Set => Some(Group::Set),
        // A statement's own body: the command that follows a completed condition, the
        // `do`/`else` body, and the block a statement's own `)` leaves room for.
        _ if words.stmt == Stmt::Await || words.chain => Some(Group::Body),
        _ => None,
    }
}

/// Whether `word` is one of the condition keywords this platform's `if` takes
/// before its operand.
fn is_condition_keyword(word: &[u8]) -> bool {
    [&b"exist"[..], b"defined", b"errorlevel", b"cmdextversion"]
        .iter()
        .any(|keyword| word.eq_ignore_ascii_case(keyword))
}

/// Whether `word` is one of the comparison operators this platform's `if` takes
/// between the two strings it compares — the spaced spelling of a comparison, and
/// the `==` of the joined one.
fn is_comparison_operator(word: &[u8]) -> bool {
    [&b"=="[..], b"equ", b"neq", b"lss", b"leq", b"gtr", b"geq"]
        .iter()
        .any(|operator| word.eq_ignore_ascii_case(operator))
}

/// How many `c` characters end `text` — the run a break's caret and backslash
/// rules count immediately before it.
fn trailing_run(text: &str, c: u8) -> usize {
    text.bytes().rev().take_while(|b| *b == c).count()
}

/// The shell parser's own quote state after `b`: the open span, when one is open,
/// and whether a `\`'s escape is pending over the next character. `\` outside a
/// single-quoted word escapes the next character, so a quote it escapes opens or
/// closes no word; `cmd.exe`'s caret escapes nothing there, which is why the two
/// readings of the same text are tracked separately.
fn bash_step(quote: Option<u8>, escaped: bool, b: u8) -> (Option<u8>, bool) {
    if escaped {
        (quote, false)
    } else if let Some(open) = quote {
        if b == open {
            (None, false)
        } else if open == b'"' && b == b'\\' {
            (quote, true)
        } else {
            (quote, false)
        }
    } else if b == b'\\' {
        (None, true)
    } else if matches!(b, b'\'' | b'"') {
        (Some(b), false)
    } else {
        (None, false)
    }
}

/// Whether `c` is one of the separators the platform's own reader skips over: a
/// space, a tab, a carriage return, and — for the forward-looking scan, which has
/// to find the first character of the next line that is not blank — a break.
/// Nothing else is blank to `cmd.exe`: a character Unicode calls whitespace
/// (U+00A0, U+3000) is an ordinary character here, so a line holding one has text
/// on it and its break is not dropped.
fn is_separator(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Whether `text` holds nothing but the platform's own separators: a blank line.
fn is_blank(text: &str) -> bool {
    !text.bytes().any(|b| !is_separator(b))
}

/// Whether `word` is this platform's comment command: `rem` (its own `@` prefix
/// allowed), read with one of the delimiters its parser ends a word at — or with
/// nothing after it. The `,`/`;`/`=` delimiters count like the spaces, so `rem,x`
/// takes its line here as well. A character appended to `rem` (`.`, `/`, `:`, `[`,
/// `+`) is not the command: the parser drops that character later and the control
/// operators after it are read as commands of their own.
fn is_comment_command(word: &[u8]) -> bool {
    let word = word.strip_prefix(b"@").unwrap_or(word);
    word.split(|b| matches!(b, b',' | b';' | b'='))
        .next()
        .is_some_and(|head| head.eq_ignore_ascii_case(b"rem"))
}

/// Whether `word` is a label's own name: this platform's reader takes the rest of
/// such a line — the separator a break would join there included — as that name
/// ([`SWALLOWING_LINE`]). Read where its reader is looking for a command, like the
/// comment command above, and under the same `@` prefix.
fn is_label_word(word: &[u8]) -> bool {
    word.strip_prefix(b"@").unwrap_or(word).starts_with(b":")
}

/// The agent-facing refusal for one cause, in the reading's mode-selected form: the
/// shared frame ([`REFUSAL_FRAME`], so the three refusal renderers cannot word it
/// differently), the cause, that nothing ran, and the mode's remedy ([`remedy`], which
/// owns the choice between the two).
fn refusal(cause: &str, mode: ShellMode) -> String {
    format!("{REFUSAL_FRAME}{cause}. Nothing ran. {}", remedy(mode))
}

/// The remedy a refusal offers in `mode` ([`refusal`]), and the sentence the guidance
/// hands the agent for the same shapes ([`super::render_command_line_notes`]) — one
/// prompt asset for both, so the tool cannot promise a way out its own refusals do not
/// honour, nor word the same sentence twice. The mode picks the asset, never the cause:
/// a file runs any of these texts whole, line breaks and all, so the full mode offers it
/// for every cause, while the read-only mode has none to offer and says so itself.
pub(super) fn remedy(mode: ShellMode) -> String {
    crate::prompt::load_prompt(match mode {
        ShellMode::Full => "tool/shell_command_lines_remedy_full.md",
        ShellMode::ReadOnly => "tool/shell_command_lines_remedy_read_only.md",
    })
    .trim()
    .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Windows lane's reading, driven from this host (the platform is data).
    fn read(command: &str) -> Result<String, String> {
        executable(command, ShellPlatform::Windows, ShellMode::Full).map(Cow::into_owned)
    }

    /// The refusal for `command`: a shape the two readings disagree about, or one
    /// this platform's reader cannot be left with.
    fn refused(command: &str) -> String {
        read(command).expect_err("the shape is refused, not read")
    }

    /// The refusal for `command` in the mode whose guard reads the original text:
    /// the divergences of that reader alone are decided there.
    fn refused_read_only(command: &str) -> String {
        match executable(command, ShellPlatform::Windows, ShellMode::ReadOnly) {
            Err(cause) => cause,
            Ok(text) => panic!("the shape is refused in read-only mode, not read: {text:?}"),
        }
    }

    /// Assert the reading of `command` is `expected`, text and all.
    fn reads(command: &str, expected: &str) {
        assert_eq!(read(command).as_deref(), Ok(expected), "{command:?}");
    }

    #[test]
    fn unix_and_single_line_texts_are_handed_back_untouched() {
        for text in [
            "dir /b",
            "for %f in (*.txt) do (del %f)",
            "if exist x (echo y)",
        ] {
            assert!(
                matches!(
                    executable(text, ShellPlatform::Unix, ShellMode::Full),
                    Ok(Cow::Borrowed(borrowed)) if borrowed == text
                ),
                "{text:?}"
            );
            assert!(
                matches!(
                    executable(text, ShellPlatform::Windows, ShellMode::Full),
                    Ok(Cow::Borrowed(borrowed)) if borrowed == text
                ),
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_bare_break_becomes_the_unconditional_separator() {
        // That a failing first line does not stop the rest, and that the status
        // reported is the last line's, are properties of `&` itself.
        reads(
            "dir C:\\ws\ndel C:\\ws\\x.txt",
            "dir C:\\ws&del C:\\ws\\x.txt",
        );
        // An even `\` run is a literal backslash before a separator.
        reads("dir C:\\ws\\\\\ndel x", "dir C:\\ws\\\\&del x");
    }

    #[test]
    fn blank_lines_and_a_trailing_break_change_nothing() {
        reads("a\n\nb", "a&b");
        reads("a\n", "a");
        reads("\na", "a");
        reads("a\n\n", "a");
        // A line holding only a space or a tab is blank to this platform too: the
        // separator belongs to the break before it, not after it, and the spaces
        // themselves stay as written.
        reads("a\n \t\nb", "a& \tb");
        // A character Unicode calls whitespace is an ordinary one to `cmd.exe`:
        // the line has text on it, so its break is a separator like any other and
        // the line is not glued into the one after it.
        reads("a\n\u{A0}\nb", "a&\u{A0}&b");
        reads("a\n\u{3000}\nb", "a&\u{3000}&b");
        // A text whose breaks all stand at its end is the single line it was written
        // as: this platform's reader never reaches past the first break, so none of
        // the reading's rules — every one about what the line *after* a break is read
        // as — is reached, and a trailing break can never turn such a text into a
        // refusal. A shape only this platform accepts, a shape whose `"` span is left
        // open, a statement's own text, and the guard's own `\`-before-a-break
        // divergence all run in the mode they run in without the break.
        for (text, one_line) in [
            ("echo a (", "echo a (\n"),
            ("echo \"a", "echo \"a\n"),
            ("if exist x", "if exist x\n"),
            ("dir C:\\ws\\", "dir C:\\ws\\\n"),
            ("dir C:\\ws\\", "dir C:\\ws\\\n\n"),
            ("rem note", "rem note\n"),
        ] {
            reads(one_line, text);
            assert_eq!(
                executable(one_line, ShellPlatform::Windows, ShellMode::ReadOnly).as_deref(),
                Ok(before_first_break(one_line)),
                "{one_line:?}"
            );
        }
        // Nothing after the break is true of a break wherever it stands, so the same
        // holds for the trailing break of a longer text: it is dropped like the blank
        // lines' breaks, and the shape it ends on is not read as a break either. The
        // text is read as the same line with and without it, so a statement the text
        // ends on is still one the reader is left waiting for.
        for (text, written) in [
            ("dir&rem note", "dir\nrem note\n"),
            ("echo a&dir C:\\ws\\", "echo a\ndir C:\\ws\\\n"),
            ("dir&dir C:\\ws\\", "dir\ndir C:\\ws\\\n"),
        ] {
            reads(written, text);
            assert_eq!(
                executable(written, ShellPlatform::Windows, ShellMode::ReadOnly).as_deref(),
                Ok(text),
                "{written:?}"
            );
        }
        assert_eq!(
            read("dir\nif exist x\n"),
            read("dir\nif exist x"),
            "a trailing break changes nothing about the statement the text ends on"
        );
    }

    #[test]
    fn a_crlf_break_is_one_break() {
        reads("a\r\nb", "a&b");
        // The spelling of the break is the only difference: nothing else about the
        // reading depends on it, down to the word a line ends on and the character a
        // caret's escape reaches.
        for (lf, crlf) in [
            ("if exist x echo\nmore", "if exist x echo\r\nmore"),
            (
                "if exist x (\ndir\n) echo b",
                "if exist x (\r\ndir\r\n) echo b",
            ),
            ("if exist x (dir) cls\ndir", "if exist x (dir) cls\r\ndir"),
            (
                "if exist x echo (a) done\nmore",
                "if exist x echo (a) done\r\nmore",
            ),
            ("if x ==1\nmore", "if x ==1\r\nmore"),
            ("echo a ^\nb", "echo a ^\r\nb"),
            ("echo a ^\n\n", "echo a ^\r\n\r\n"),
            ("dir || (a\nb)", "dir || (a\r\nb)"),
        ] {
            match (read(lf), read(crlf)) {
                (Ok(left), Ok(right)) => assert_eq!(right.replace("\r\n", "\n"), left, "{crlf:?}"),
                (Err(left), Err(right)) => assert_eq!(right, left, "{crlf:?}"),
                pair => panic!("{crlf:?}: the spelling of the break changed the reading: {pair:?}"),
            }
        }
    }

    /// One break reads as one break however it is spelled: every `\r` standing
    /// immediately before the break belongs to the break, so a rule that reads the line
    /// before it — the caret the reader is left waiting for, the comment command, the
    /// guard's own `\` — reads the same shape under a longer run of them as under one,
    /// and a shape refused under one spelling is refused under the other.
    #[test]
    fn a_run_of_carriage_returns_is_the_one_break() {
        for (one, run) in [
            // A caret whose escape has nothing left to take, so the reader is left
            // waiting for a character the text never supplies.
            ("echo a ^\r\n", "echo a ^\r\r\n"),
            // A comment line, which takes the rest of its line — the separator a break
            // would be joined with included — as the comment's own text.
            ("dir\nrem\r\ndel x", "dir\nrem\r\r\ndel x"),
            // A label line, whose name is the rest of its line.
            ("dir\n:: note\r\ndel x", "dir\n:: note\r\r\ndel x"),
            // A line whose own text runs to its end: the separator would become part of
            // the statement rather than a command of its own.
            ("if exist x echo\r\nmore", "if exist x echo\r\r\nmore"),
        ] {
            assert_eq!(refused(run), refused(one), "{run:?}");
        }
        assert!(refused("echo a ^\r\r\n").contains("left waiting"));
        assert!(refused("dir\nrem\r\r\ndel x").contains("comment"));
        // The shapes the reading goes on to re-spell read the same under either spelling
        // of the break: an operator joining its operand, and the separator of a plain
        // line.
        for (one, run) in [
            ("dir &&\r\nnext", "dir &&\r\r\nnext"),
            ("dir\r\n\r\r\ndel x", "dir\r\r\n\r\r\ndel x"),
        ] {
            assert_eq!(
                read(run).expect("read"),
                read(one).expect("read"),
                "{run:?}"
            );
        }
        // The guard's own reading of an odd `\` run is decided in the mode whose guard
        // read the original text, and reads the run of carriage returns the same way.
        for (one, run) in [
            ("dir \\\\\\\r\ndel x", "dir \\\\\\\r\r\ndel x"),
            ("(echo a\\\r\ndel x)", "(echo a\\\r\r\ndel x)"),
        ] {
            assert_eq!(refused_read_only(run), refused_read_only(one), "{run:?}");
        }
        // A break the platform's reader continues across is kept byte for byte, the run
        // of carriage returns included: nothing is re-spelled into a command the agent
        // did not write.
        reads("echo a ^\r\r\nb", "echo a ^\r\r\nb");
        reads("(\r\r\ndir\r\r\n)", "(\r\r\ndir\r\r\n)");
        // A carriage return with no line feed behind it is no break to this reading:
        // such a text is the one line it was written as, handed over byte for byte,
        // never respelled into a second command. The platform's reader is the
        // unmeasured half of that premise — a carriage return it ends a line at would
        // lose everything after it (hand check item 8).
        reads("dir\rdel x", "dir\rdel x");
        reads("echo a\rb", "echo a\rb");
    }

    #[test]
    fn the_platforms_own_continuations_stay_as_written() {
        // An odd caret run escapes the break, whose continuation is the command the
        // caret spread: the two lines are one command and are left as written.
        reads("echo a ^\nb", "echo a ^\nb");
        reads("copy a.txt ^\nb.txt", "copy a.txt ^\nb.txt");
        // An even one is a literal caret before a break that ends the command.
        reads("echo a ^^\nb", "echo a ^^&b");
        // An escaped caret at the end of the text is not a pending escape either.
        reads("dir\necho b^^", "dir&echo b^^");
        // A break inside a bracketed group, blank line included, is the
        // platform's own multi-line block.
        reads(
            "for %f in (*.txt) do (\ndel x\n)",
            "for %f in (*.txt) do (\ndel x\n)",
        );
        reads(
            "for %f in (*.txt) do (\n\ndel x\n)",
            "for %f in (*.txt) do (\n\ndel x\n)",
        );
        // The group openers this platform puts at an argument position: the `)`
        // of a statement's own group, and the `if` condition's end.
        reads(
            "if exist C:\\ws\\x.txt (\ndel C:\\ws\\x.txt\n)",
            "if exist C:\\ws\\x.txt (\ndel C:\\ws\\x.txt\n)",
        );
        reads(
            "if exist x (echo a\n) else (\necho b\n)",
            "if exist x (echo a\n) else (\necho b\n)",
        );
        reads(
            "@if not defined DEBUG (\necho release\n)",
            "@if not defined DEBUG (\necho release\n)",
        );
        reads("if %a%==1 (\necho one\n)", "if %a%==1 (\necho one\n)");
        // The rest of this platform's `if` conditions: a comparison's second
        // string, a comparison operator, and the `/i` switch.
        reads(
            "if \"%a%\" == \"1\" (\necho one\n)",
            "if \"%a%\" == \"1\" (\necho one\n)",
        );
        reads("if %a% equ 1 (\necho one\n)", "if %a% equ 1 (\necho one\n)");
        reads(
            "if /i \"A\"==\"a\" (\necho same\n)",
            "if /i \"A\"==\"a\" (\necho same\n)",
        );
        // A statement's body written as another statement, and the `else if` chain.
        reads(
            "for %f in (*.txt) do if exist %f (\ndel %f\n)",
            "for %f in (*.txt) do if exist %f (\ndel %f\n)",
        );
        reads(
            "if exist x (\necho a\n) else if exist y (\necho b\n)",
            "if exist x (\necho a\n) else if exist y (\necho b\n)",
        );
        // A `for`'s set spans lines: this platform appends the next line to it and
        // the statement's own `do` still follows.
        reads("for %f in (a\nb) do echo %f", "for %f in (a\nb) do echo %f");
        reads(
            "for %f in (*.txt\n*.md) do (\ndel %f\n)",
            "for %f in (*.txt\n*.md) do (\ndel %f\n)",
        );
        // A `(` that opens no group is ordinary text, so a break after both
        // parens are closed is the separator it is. The statement is over by then,
        // which the separators after it show.
        reads("echo (a) done\nmore", "echo (a) done&more");
        reads(
            "echo (see the log) below\ndir",
            "echo (see the log) below&dir",
        );
        // The word a line ends on is a word like any other: a statement completed by
        // it is not one still waiting for its own text, so a statement body on the
        // block's last line — with a command after the `)` and with none — leaves a
        // text the reader is not waiting for.
        reads("if exist x (\ndir\n) echo b", "if exist x (\ndir\n) echo b");
        reads(
            "for %f in (*.txt) do (\necho %f\n) echo t",
            "for %f in (*.txt) do (\necho %f\n) echo t",
        );
        reads(
            "for %f in (*.txt) do (\necho %f\n)& dir",
            "for %f in (*.txt) do (\necho %f\n)& dir",
        );
        // A statement's own `)` ends its word positions: what follows at the same
        // position is a command of its own — and a statement whose own text that word
        // belongs to has still absorbed the line, which is why the break behind it is
        // refused rather than respelled (see the statement test below).
        assert!(refused("if exist x (dir) cls\ndir").contains("runs that command"));
        assert!(refused("if exist x (dir)\ndir").contains("runs that command"));
    }

    /// A statement's own text written on a line runs to that line's end, so a
    /// separator behind it would be read as part of the statement: the following line
    /// would run under the condition, or once per iteration, instead of as a command
    /// of its own. The whole text is refused rather than run as something else.
    #[test]
    fn a_break_behind_a_statements_own_text_is_refused() {
        let task = "runs that command to the end";
        for text in [
            // The command its condition governs, written on the line.
            "if exist x echo a\ndel C:\\ws\\x.txt",
            "if not exist out mkdir out\ncopy a b",
            // The same through the block a statement's body is written as: the `)`
            // does not end the statement for this platform's reader.
            "if exist x (dir)\ncopy a b",
            "if exist x (\ndir\n) echo b\ncopy a b",
            // A `do` body, and a nested statement as one.
            "for %f in (*.log) do type %f\ndel *.log",
            "if exist x if exist y dir\ncopy a b",
            // A statement whose first letter a caret escape carried is the statement it
            // spells: the escape was over a letter and removed nothing but itself.
            "dir & ^if exist x echo b\ncopy a b",
        ] {
            assert!(refused(text).contains(task), "{text:?}");
        }
        // A statement whose text the text itself ends is not followed by anything, and
        // a statement behind the break is a command of its own.
        reads("echo a\nif exist x echo b", "echo a&if exist x echo b");
        // A break with nothing after it separates nothing, so a trailing break cannot
        // turn such a text into a refusal: the text the platform runs is the same.
        reads("echo a\nif exist x echo b\n", "echo a&if exist x echo b");
        reads(
            "for %f in (*.txt) do (\ndel %f\n)\n",
            "for %f in (*.txt) do (\ndel %f\n)",
        );
        reads(
            "if exist x (\ndir\n) echo b\n",
            "if exist x (\ndir\n) echo b",
        );
        // A `(` on the line does not end a statement's own text: whether the group it
        // opens is one this reading can name (the words restart inside it) or not (they
        // stay the argument's), the line after the break would still be read inside the
        // statement.
        for text in [
            "if exist x echo (a) done\nmore",
            "for %f in (*.txt) do echo %f (x)\ndir",
            "if not exist out mkdir out (x)\ncopy a b",
            "if exist x (echo a) & (b c)\ndir",
        ] {
            assert!(refused(text).contains("runs that command"), "{text:?}");
            // The reading of the breaks is the same in both modes: the guard's
            // divergences are all this module decides in one of them alone.
            assert!(
                refused_read_only(text).contains("runs that command"),
                "{text:?}"
            );
        }
        // A comparison's operator with its second string joined to it is the
        // condition's text, not the statement's command: the reader is left waiting for
        // what the condition governs, which is the cause it reports.
        assert!(refused("if x ==1\nmore").contains("waiting for its own"));
        // A block bounds the statement written in it, so the break behind the block is
        // the separator it is.
        reads(
            "(if exist x mkdir out)\ncopy a b",
            "(if exist x mkdir out)&copy a b",
        );
        reads(
            "(for %f in (*.log) do type %f)\ndel *.log",
            "(for %f in (*.log) do type %f)&del *.log",
        );
        // A statement whose own text is bounded by a `(` it does not own — an argument
        // paren, whose group this reading cannot name — ends where that group closes.
        reads(
            "echo (if exist y mkdir z)\ndir2",
            "echo (if exist y mkdir z)&dir2",
        );
        // A statement whose *own* command is a block is not bounded by it: the block is
        // part of the text this platform's reader runs to the end of the line.
        assert!(
            refused("if exist x (if exist y (\ndir\n))\ncopy a b").contains("runs that command"),
        );
        // An agent's own `&` behind a statement's text is the agent's own reading, and
        // the break behind *that* is refused like any other (see the test above).
        assert!(refused("if exist x (\ndir\n) & echo b\ndir2").contains("runs that command"),);
        // A single-line command is never respelled, whatever it holds (see
        // [`unix_and_single_line_texts_are_handed_back_untouched`]).
        reads("if not exist out mkdir out", "if not exist out mkdir out");
    }

    #[test]
    fn a_statement_cut_by_a_break_is_refused_rather_than_split() {
        // The statement's own text is missing, and the break ends the command where
        // this platform's reader is still waiting for it: running the next line as a
        // command of its own would run what the statement never named.
        for text in [
            "for %f in\ndo echo %f",
            "for %f in (a b)\ndo echo %f",
            "if exist x\ndel C:\\ws\\x.txt",
            "if not exist x\ndel C:\\ws\\x.txt",
        ] {
            assert!(refused(text).contains("waiting"), "{text:?}");
        }
        // The split between a `for`'s set and its `do`.
        assert!(refused("for %f in (a\nb)\ndo echo %f").contains("waiting"));
        // A statement whose own command the last line carries is not one of these:
        // its text is written, and this reading refuses the break for a different
        // reason ([`STATEMENT_SPANS_LINE`], see the statement test below).
        assert!(refused("if exist x echo\nmore").contains("runs that command to the end"));
    }

    #[test]
    fn a_paren_whose_group_cannot_be_told_is_refused() {
        // A `)` this platform's reader counts against a group this reading cannot
        // name: it may close the group below it, so the lines after it are unknown.
        assert!(
            refused("if exist x (\ndir C:\\ws\\(a)\ndel C:\\ws\\x.txt\n)")
                .contains("bracketed group")
        );
        // A `)` this platform's counter cannot pair, where its reader is looking for
        // a command: its own reading discards the rest of the line there.
        assert!(refused("dir\n) echo b").contains("discards"));
        assert!(refused("(\ndel C:\\ws\\x.txt)\necho tail\n)").contains("discards"));
        // The same `)` right after a statement's own group: the reader is looking for
        // a command there too, so the rest of the line is discarded as well.
        assert!(refused("if exist x (echo a\n) )\ndir").contains("discards"));
        // A `(` a pipeline's right-hand side opens: the interpreter the pipeline
        // hands that side to is the one that counts it, so a break reaching it is
        // refused rather than read as the block it may have been. `||` — which this
        // platform runs in this interpreter, like `&`/`&&` — is not that case, and
        // neither is the pipeline's own `@(` prefix, which is the same position.
        for text in [
            "echo x | (\ndir\n)",
            "if exist x | (\ndir\n)",
            "for %f in (a) do | (\ndir\n)",
            "if exist x (dir) else | (\ndir\n)",
            "dir | @(\nb\n)",
        ] {
            assert!(refused(text).contains("bracketed group"), "{text:?}");
        }
        reads("dir || (a\nb)", "dir || (a\nb)");
        reads("dir && (a\nb)", "dir && (a\nb)");
        reads("dir & (a\nb)", "dir & (a\nb)");
        reads("dir || @(\nb\n)", "dir || @(\nb\n)");
        // A `)` that closed a block of this platform's own leaves the reader looking
        // for a command, so the `(` after it opens a block like the one before it —
        // and so does the `@` this platform suppresses the echo of a block with.
        reads("(\ndir\n)\n(\ndel x\n)", "(\ndir\n)&(\ndel x\n)");
        reads("(a)\n(b\nc)", "(a)&(b\nc)");
        reads("@(\necho a\necho b\n)", "@(\necho a\necho b\n)");
        assert!(refused("echo (a) (b\nc)").contains("bracketed group"));
        assert!(refused("echo () (b\nc)").contains("bracketed group"));
        // A parenthesised word inside a proven block, where this platform's `)` is the
        // counter's and not the argument's: the block it closes is not the one the text
        // was written as, so the command is refused rather than run as the two
        // different texts the two readings make of it. (The hand-check list in the
        // module doc is where this one is settled.)
        assert!(refused("if exist x (\necho (a)\n)").contains("bracketed group"));
        assert!(refused("if exist x (\ndir C:\\ws (x86)\\bin\n)").contains("bracketed group"));
        // A break behind an operator joins the lines into one command, so the line
        // the operand continues is the line the operator was written on: a `(` there
        // is still in the operand's position and opens no proven group.
        assert!(refused("dir >\n(echo a\necho b)").contains("bracketed group"));
        // A `(` after a statement's own `)` is the same: the statement is over, so
        // the `(` stands in the argument of the command after it and opens no group
        // of a statement — while the `if`/`for` that may start there keeps its own.
        assert!(refused("if exist x (dir) echo (b\nc)").contains("bracketed group"));
        assert!(refused("if exist x (dir) cls (b\nc)").contains("bracketed group"));
        reads(
            "if exist x (dir) if exist y (b\nc)",
            "if exist x (dir) if exist y (b\nc)",
        );
    }

    #[test]
    fn a_break_after_an_operator_is_joined_to_its_operand() {
        reads("dir &&\nnext", "dir &&next");
        reads("dir >\nout.txt", "dir >out.txt");
        reads("dir <\nin.txt", "dir <in.txt");
        reads("dir |\nmore", "dir |more");
        // A quoted operand is the break's last significant character, so the break
        // is not behind the operator the target was written after: it ends the
        // command instead of joining the two lines.
        reads("dir >\"out.txt\"\nnext", "dir >\"out.txt\"&next");
        // The joined line is one command, and a `(` after the operator a command
        // starts with opens a group the block's own breaks are read against.
        reads("dir &&\n(echo a\necho b)", "dir &&(echo a\necho b)");
    }

    #[test]
    fn a_join_that_would_spell_an_operator_of_its_own_is_refused() {
        // The two characters the join would put together spell an operator the command
        // never wrote, so the lines are refused rather than joined.
        for text in [
            "dir >\n> next",
            "dir >\n< next",
            "dir <\n< next",
            "dir <\n> next",
            "dir &\n> out.txt",
            "dir |\n> out.txt",
            "dir &&\n> out.txt",
        ] {
            assert!(refused(text).contains("operator"), "{text:?}");
        }
        // A plain break before a line that opens with a redirection is a separator
        // like any other: the two operator characters are kept apart by a space, so
        // nothing spells `&>`, and the second line runs as the line it is — a
        // redirection with no command, which is an error this platform reports.
        reads("dir\n> out.txt", "dir& > out.txt");
        reads("dir\n>> out.txt", "dir& >> out.txt");
        reads("dir\n< in.txt", "dir& < in.txt");
        // The redirect written on the line after a blank one is the same shape.
        reads("dir\n\n> out.txt", "dir& > out.txt");
        // The redirection the *joined* line takes is the operator's operand, so the
        // line after a break opening with anything else is its own text.
        reads("dir >\nout.txt", "dir >out.txt");
    }

    #[test]
    fn shapes_the_two_readers_disagree_about_are_refused() {
        // A break inside this platform's own `"` span: the argument would reach
        // its program truncated.
        assert!(refused("echo \"a\nb\"").contains("double-quoted"));
        // A caret at the end of a line whose escape reaches a paren: the character is
        // ordinary text here, so this reading's own group counter — the pairs its
        // breaks are read against — would be left out of step with the text.
        assert!(refused("if exist x ^\n(\ndel y\n)").contains("escapes"));
        assert!(refused("dir &&^\n(del x\ndel y)").contains("escapes"));
        // A text block: this platform has none, and a caret over its first `<`
        // does not hide the `<<` from the parser that reads the text.
        assert!(refused("cat <<EOF\nx\nEOF").contains("text block"));
        assert!(refused("cat ^<<EOF\nx\nEOF").contains("text block"));
        // A span or bracket the text never closes, and a caret with nothing left
        // to escape.
        assert!(refused("echo \"unterminated\nb").contains("double-quoted"));
        assert!(refused("echo a\necho \"b").contains("unclosed"));
        assert!(refused("(\ndir\n").contains("unclosed"));
        assert!(refused("dir\necho b^").contains("left waiting"));
        // A connector as the first significant character after a break: behind a
        // separator break, and behind an operator whose join would otherwise eat
        // the agent's operator into one it never wrote.
        assert!(refused("dir\n&& next").contains("connector"));
        assert!(refused("dir &\n& x").contains("connector"));
        assert!(refused("dir |\n| more").contains("connector"));
        // A `(` whose group cannot be proved: this platform's reader opens one only
        // where it is looking for a command, and the text after the break may be
        // the block's body or a command of its own. A `(` in an argument of any
        // command, a `#`-comment-looking argument, and a `(` after an `if`'s
        // command word instead of after its condition.
        for text in [
            "echo a (\necho b)",
            "echo (\nb)",
            "echo 'total ('\necho more",
            "echo a # (note\necho b",
            "if exist x echo (\nmore",
            "dir C:\\ws (\nmore",
            "echo (a\necho b)",
        ] {
            assert!(refused(text).contains("bracketed group"), "{text:?}");
        }
    }

    /// The caret's escape has nothing to take in the text that runs, whichever
    /// break the caret was written behind: a break the respelling drops must not
    /// turn a dangling caret into a live one.
    #[test]
    fn a_caret_left_with_nothing_to_escape_is_refused() {
        // A caret at the end of the last line has nothing to reach, whether the text
        // ends there or a break the respelling drops follows it.
        for text in [
            "dir\necho b^",
            "dir\necho b^\n",
            "dir\necho b^\r\n",
            "echo a ^\n",
            "echo a ^\r\n",
            // Caret-only lines: their carets pair off across the breaks the respelling
            // deletes, so the walk's own pending escape is what refuses the text.
            "||\n^\n^\n^",
            "||\n^\n^\n^\n",
        ] {
            assert!(refused(text).contains("left waiting"), "{text:?}");
        }
        // A caret whose escape is left taking the break of a blank line — the
        // spelling this platform reads as a literal line break — is refused too: the
        // rebuilt text drops that break, so the escape would reach the line's first
        // character instead.
        for text in ["echo a ^\n\n", "echo a ^\r\n\r\n", "dir ^\n\n(x)"] {
            assert!(refused(text).contains("left taking"), "{text:?}");
        }
        // The same caret with a line to reach runs as written, and a break after the
        // text does not change what it escapes.
        reads("echo a ^\nb", "echo a ^\nb");
        reads("echo a ^\nb\n", "echo a ^\nb");
        // An even run is a literal caret before a break that ends the command.
        reads("echo a ^^\nb", "echo a ^^&b");
        // A blank line after the line a caret continues is a command of its own: the
        // continuation takes only the line after it, so the blank line's break needs
        // the separator its own blankness would otherwise drop.
        reads("echo a ^\n  \nb", "echo a ^\n  &b");
        reads("echo a ^\n \t\nb", "echo a ^\n \t&b");
    }

    #[test]
    fn the_guards_own_divergences_are_refused_in_the_mode_that_reads_them() {
        // An odd `\` run before a break is a line continuation for the guard's reader
        // and an ordinary character here, and a `'…'` word is one quoted word there
        // and two ordinary pieces of text here: in read-only mode the guard judged a
        // text the execution would not run, so the shape is refused whole.
        for text in [
            "echo 'a\nb'",
            "echo It's here\ndir C:\\ws",
            "echo ^'a\ndel C:\\ws\\x.txt\necho ^'b",
            "echo ^\"a\ndel C:\\ws\\x.txt\necho ^\"b",
            "echo ^\"a b\ndir",
            "(echo a\\\ndel C:\\ws\\x.txt)",
            "(echo 'a\ndel C:\\ws\\x.txt')",
            "dir C:\\ws\\\ndel x",
            // A `'…'` word whose line ends on an odd `\` run.
            "echo 'a\\\nb'",
            // A `'…'` word left open at the end of the text: a quoted word to that
            // reader, ordinary text to this platform, and no span of its own.
            "echo a\necho 'b",
            // An operator the caret escapes: literal text here, structure there —
            // inside a line, and as the character a trailing caret's escape reaches.
            "dir ^& x\ndir y",
            "echo a ^>\nbuild.bat",
            "dir /b *.txt ^\n| findstr x",
        ] {
            assert!(read(text).is_ok(), "{text:?}");
            let cause = refused_read_only(text);
            assert!(cause.contains("Nothing ran."), "{text:?}: {cause}");
        }
        // Each divergence is refused under the rule that actually holds: a `\` before a
        // break inside the `'…'` word is ordinary text to the guard's reader too, so the
        // quoted word the break falls in is what it names, while a `\` run holding the
        // break outside any quoted word is the line continuation.
        let quoted = refused_read_only("echo 'a\\\nb'");
        assert!(quoted.contains("single-quoted word"), "{quoted}");
        let continued = refused_read_only("(echo a\\\ndel C:\\ws\\x.txt)");
        assert!(
            continued.contains("odd number of backslashes"),
            "{continued}"
        );
        // The full mode consults no such reader: there the platform's own reading of
        // the two lines is the only one there is, so the same texts are run as the
        // platform reads them.
        reads("echo 'a\nb'", "echo 'a&b'");
        reads("dir C:\\ws\\\ndel x", "dir C:\\ws\\&del x");
        reads("dir ^& x\ndir y", "dir ^& x&dir y");
        reads("echo a ^>\nbuild.bat", "echo a ^>&build.bat");
        // The caret escapes the operator, so the two lines stay one command.
        reads("dir /b *.txt ^\n| findstr x", "dir /b *.txt ^\n| findstr x");
    }

    #[test]
    fn the_cap_is_the_platforms_own_limit_and_the_remedy_is_the_modes() {
        let at_cap = "a".repeat(TEXT_UNIT_LIMIT);
        let over_cap = "a".repeat(TEXT_UNIT_LIMIT + 1);
        assert!(check_command_line(&at_cap, ShellPlatform::Windows, ShellMode::Full).is_ok());
        // The limit is on the platform's own unit, the UTF-16 code unit: a line of
        // characters outside the basic plane counts twice, so a line that looks
        // short to a `chars()` count still does not fit.
        let astral = "\u{1F600}".repeat(TEXT_UNIT_LIMIT / 2 + 1);
        assert!(astral.chars().count() < TEXT_UNIT_LIMIT);
        let over_cap_in_units =
            check_command_line(&astral, ShellPlatform::Windows, ShellMode::Full)
                .expect_err("over the cap in the platform's units");
        assert!(over_cap_in_units.contains("8191"), "{over_cap_in_units}");
        // Nothing but this platform has the limit: the unix lane takes the text
        // itself, whatever its length.
        assert!(check_command_line(&over_cap, ShellPlatform::Unix, ShellMode::Full).is_ok());
        assert!(check_command_line(&astral, ShellPlatform::Unix, ShellMode::Full).is_ok());
        // The limit is measured by `check_command_line` alone: `executable` produces
        // the reading of an over-cap text without complaint, and measuring that
        // reading is what refuses it.
        let over_cap_reading =
            read(&over_cap).expect("the reading of an over-cap text is produced");
        assert!(
            check_command_line(&over_cap_reading, ShellPlatform::Windows, ShellMode::Full).is_err()
        );
        // The limit is on the text that will be executed, and a trailing break the
        // platform never reads is no part of it: the line that fits still fits with
        // one, and the line that does not is still refused.
        assert!(
            check_command_line(
                &read(&format!("{at_cap}\n")).expect("read"),
                ShellPlatform::Windows,
                ShellMode::Full
            )
            .is_ok()
        );
        assert!(
            check_command_line(
                &read(&format!("{over_cap}\n")).expect("read"),
                ShellPlatform::Windows,
                ShellMode::Full
            )
            .is_err()
        );

        let full = check_command_line(&over_cap, ShellPlatform::Windows, ShellMode::Full)
            .expect_err("over the cap");
        let read_only = check_command_line(&over_cap, ShellPlatform::Windows, ShellMode::ReadOnly)
            .expect_err("over the cap");
        // The message states the line's own limit, what the interpreter's hand-off
        // takes of it, and the number left for the text — a text refused under it
        // is never left looking shorter than the limit it was refused by.
        assert!(full.contains("8191"), "{full}");
        assert!(full.contains(&TEXT_UNIT_LIMIT.to_string()), "{full}");
        assert!(full.contains("Nothing ran."), "{full}");
        assert!(full.contains(&remedy(ShellMode::Full)), "{full}");
        assert!(
            read_only.contains(&remedy(ShellMode::ReadOnly)),
            "{read_only}"
        );
        assert_ne!(full, read_only);
    }

    /// A comment (`rem`) or a label (`:`) word, read where this platform's reader
    /// looks for a command: it takes the rest of that line — including the separator a
    /// break would be joined with, and a paren — as the comment's text or the label's
    /// name, so the lines after that break would be swallowed with no error and no
    /// status of their own. The whole text is refused rather than run as less than it
    /// is.
    #[test]
    fn a_swallowed_line_at_a_break_is_refused_rather_than_separated() {
        let task = "comment (`rem`) or a label";
        for text in [
            // The comment command at a command position, in this platform's own
            // spellings.
            "rem note\ndir",
            "rem\ndir",
            "@rem note\ndir",
            "REM NOTE\ndir",
            "  rem note\ndir",
            "rem,note\ndir",
            // A word whose first letter a caret escape carried is the word it spells:
            // the escape was over a letter, which had no special meaning to lose.
            "dir & ^rem x\ndir2",
            "del x & ^\nrem note\nmore",
            // A label word, which takes the rest of the line whatever it holds — read
            // at every command position, a separator's included, like the comment.
            ":: note\ndir",
            ":label\ndir",
            "echo a\n:: b\nmore",
            "dir & :label\nmore",
            "dir | :label\nmore",
            "(dir) :label\nmore",
            // The `@` prefix, with and without a space between it and the label.
            "@:label\ndir",
            "@ :label\ndir",
            "@\t:label\ndir",
        ] {
            assert!(refused(text).contains(task), "{text:?}");
            assert!(refused_read_only(text).contains(task), "{text:?}");
        }
        // A word that merely spells one, a comment in an argument, and a comment
        // inside a quoted word: this platform's reader looks for a command at a
        // command position alone.
        reads("remainder\ndir", "remainder&dir");
        reads("rem.txt\ndir", "rem.txt&dir");
        reads("echo rem\ndir", "echo rem&dir");
        reads("echo \"rem note\"\ndir", "echo \"rem note\"&dir");
        reads("echo :label\ndir", "echo :label&dir");
        // `rem` with a character appended is not this platform's comment command: its
        // parser drops the character later and reads the control operators after it.
        reads("rem. note\ndir", "rem. note&dir");
        // A label's first character is its own, so an escape over it makes that
        // character inert and no label of it — the word is the ordinary one it spells,
        // and the break behind it is the separator it is.
        reads(
            "echo a ^\n:not-a-label\nmore",
            "echo a ^\n:not-a-label&more",
        );
        // Nothing follows the swallowed line, so the break loses nothing: a trailing
        // break adds no command, and a line after the swallowed one keeps the break it
        // already has.
        reads("rem note\n", "rem note");
        reads("dir\nrem note", "dir&rem note");
        reads("echo a\n:: b", "echo a&:: b");
        // A comment word read where the reader looks for a command inside a proven
        // group takes that line's rest, its `)` included: this reading's counter never
        // sees that paren either, so the group the `(` opened is never closed and the
        // text is refused for the unclosed group it is. A `)` on a line after the
        // comment's own line closes that group as written.
        for text in ["echo a\n(rem note)", "(rem note)\ndir"] {
            assert!(
                refused(text).contains("unclosed bracketed group"),
                "{text:?}"
            );
        }
        // A `rem` written in an argument — after a `(` this reading cannot prove, and
        // after the `)` closing one — is an argument like any other, and the parens
        // are the reader's own counter's.
        reads("echo (a) rem note\ndir", "echo (a) rem note&dir");
        reads("echo (rem note)\ndir", "echo (rem note)&dir");
        reads("dir (x) rem note\ndir2", "dir (x) rem note&dir2");
        // A paren on a comment's line is that line's text, so the `)` inside one closes
        // no group — while a `rem` read after the `)` of a block this reading proved is
        // a command like any other.
        for text in ["rem note (x)\ndir", "(dir) rem note\ndir2"] {
            assert!(refused(text).contains(task), "{text:?}");
        }
        // A break this reading keeps — inside the group the comment line sits in — is
        // left exactly as written: the platform's own reader ends that line at the
        // break, so the `)` on a later line is that counter's own.
        reads("(rem note\ndir\n)", "(rem note\ndir\n)");
    }
}

//! The runner's own decomposition of a rewritten command line.
//!
//! This service is a **windowed** program, so at an interactive prompt a shell
//! neither waits for it nor hands back its status. Where such a call's own result
//! matters the runner runs it itself: the image is spawned as argv with the streams
//! wired here, waited for, and its own status and output reported — never a fold's
//! shell text, which would hand the call to an interpreter. A member the runner
//! cannot reproduce as one plain call of the image is refused loudly, never run with
//! output nobody can vouch for; a shape this reading cannot reach stays with the
//! shell. The product's file search is that same image, so its refusal or
//! stale-helper answer must never arrive looking like "no matches".
//!
//! The analyzer's per-member data is rendered twice from one traversal: as the line
//! the shell would have run (`grep_engine::join_rewritten`) and as this plan. The
//! engine's Windows drift guard asserts the rendered text is the plan's members
//! rejoined with their connectors — a plan and a line that disagreed about a member
//! would be a search whose output came from somewhere the agent did not look. One
//! divergence is deliberate: a bare-name own-image member (an agent's `mahbot -V`,
//! carried verbatim) keeps the typed spelling in the rendered line while its
//! [`Run::Own`] step spawns the absolute path ([`Plan::exe`]). On Windows the
//! rendered line is never executed — the plan is — so its quoting serves only the
//! read-only guard and that drift guard.
//!
//! A plan is a linear chain of steps in *pipe groups* — maximal runs joined by `|` — each one
//! and waited for on its own:
//!
//! - consecutive verbatim members become ONE [`Run::Shell`] step (`cmd.exe /C "<text>"`, their
//!   connectors inside), which keeps cmd.exe's reading — quoting, redirects, builtins, `%VAR%`
//!   expansion — inside a step and keeps `cd a && grep x .` searching the directory the `cd`
//!   chose. Both ends of a `|` are steps of their own: cmd.exe pipes exactly the stdout of the
//!   command immediately before a `|`, gates the pipeline on that command's connector, and runs
//!   what a following `&&` reaches with the shell's stdin rather than the pipe's — folding
//!   either end would report a match cmd.exe never produced, run a pipeline whose gated producer
//!   failed, or leave `cat` in `grep x . | head -3 && cat` reading the search's leftovers;
//! - a member that is this service's own image becomes a [`Run::Own`] step, spawned as argv with
//!   no interpreter between: nothing on the line can reinterpret an argument and the status is
//!   that member's own. Decided per member by [`build`], whichever caller it came from — a
//!   served search, or an agent's `mahbot -V` — so such a line never reaches the shell;
//! - a group's members are piped by the runner (one member per spawn, a task copying each stdout
//!   into the next member's stdin), so `grep … | head -3` is that pipeline, not the shell's; a
//!   consumer that exits drops the producer's next write as a shell pipeline does;
//! - a group runs only when cmd.exe's connector says so ([`runs_after`]), and the run reports
//!   the LAST executed group's LAST member's status — cmd's own rule, which the
//!   `[exit status: …]` annotation, the engine's sentinel classification and its refusal marker
//!   read: for `… | engine` that is the engine's status, not the interpreter's. The gating step
//!   is the group's first — the member before its first `|`, a step of its own above.
//!
//! Every [`Step`] carries its directory, derived by [`build`] from the members: the line's own
//! `cd` members ([`tracked_cd`], [`grep_engine::resolve_cd`] underneath) move the directory the
//! next step starts in; a served [`Member::Own`] carries the cwd its search tracked; a fold runs
//! where the line was where it opened, a `cd` inside it being that fold's own reading.
//!
//! Deliberately refused — the tool error [`refusal_message`] renders, never a partial result or
//! a status from somewhere else:
//!
//! - a redirection the runner cannot apply to a member it spawns itself
//!   ([`redirects_supported`]): a dup that is not the stdout merge (`>&2`), `&>`, `<>`,
//!   multi-digit descriptors, a missing target, a `%…%`-carrying or drive-relative one. `2>&1`
//!   *is* applied, at the destination the member's stdout has (see the residuals);
//! - an own-image member in a shape [`build`] cannot run as one plain call
//!   ([`own_image_reference`]): a launcher spelling (`start mahbot …`, `cmd /c mahbot …`), a
//!   cmd.exe would have expanded (`mahbot -V %TEMP%`), a spelling of the file name that is not
//!   the program the runner runs (`.\mahbot.exe -V`), plus a line the model cannot read while it
//!   names the image by that path ([`spelling_in_text`]) or where a command can start
//!   ([`own_image_in_text`], which sees through a refused line's `@`/`(` punctuation);
//! - a fold whose shell state a later fold would have observed (environment it sets, directory
//!   stack it pushes): each fold is a fresh cmd.exe and that state cannot be approximated;
//! - a `cd` member the tracking cannot follow ([`tracked_cd`], [`keyword_cd`]): a spelling the
//!   model reads but cannot track (`cd..`, `cd.txt`, `cd/d`, `@cd`), an unresolvable target, any
//!   of them beside a pipe or inside cmd's keyword forms (`if exist x cd sub`) — a following
//!   step would run in the wrong directory, a wrong answer rather than a missing one;
//! - a rewritten line whose members cannot be read as a chain.
//!
//! A line the serve decision did not plan — `mahbot debug …`, `mahbot chrome …`, `mahbot
//! bench-openrouter …`, `mahbot -V` — reaches the same runner: [`own_image_plan`] hands its
//! members to [`build`], so `cd sub && mahbot -V && echo done` and `mahbot -V | head -3` are the
//! runner's too and no caller has a second reading of what a member is.
//!
//! Accepted residuals — what is left to the shell, or not covered at all:
//!
//! - an own-image call spelled through what this model cannot read keeps the shell's behaviour,
//!   closed by decision rather than grown per spelling: a batch wrapper, a `%VAR%` expanding to
//!   the image's path, a launcher outside [`LAUNCHER_VERBS`], and among the known launchers a
//!   word the grammar does not resolve (a quoted call, `cmd /c call mahbot -V`,
//!   `start /d C:\ws mahbot -V`) — while what they do resolve is refused, as is a call in a
//!   refused line, read for command-start words ([`own_image_in_text`]) and for the image's path
//!   spelling anywhere in the text ([`spelling_in_text`]);
//! - a script this product runs on its own behalf (a custom tool's payload, an owner's
//!   automation, a diagnostics step) is not a line this runner sees: cmd.exe runs it, so an
//!   own-image call inside it gets a shell's answer for a windowed image;
//! - a bare-name `mahbot …` this model reads but never in command position — cmd's `for`, `if`,
//!   `if`, `else` forms — and one glued to a connector inside a text read as one word (another
//!   command's quoted argument, a refused line) stays the shell's: refusing them would refuse
//!   the bare name wherever it merely *stands*, and with it every line that mentions it (`echo
//!   mahbot`, `grep -rn mahbot .`). Glued connectors *between* its members are split, so
//!   `echo hi&&mahbot -V` is taken;
//! - the image's own *file name* (`mahbot`, `mahbot.exe`) counts as this service — the identity
//!   cmd.exe would hand the call, and the only reading that holds for the spelling an agent
//!   types; no other spelling is recognised, so a renamed image is the shell's to run;
//! - an own step gets the environment this runner hands every command
//!   ([`super::build_program_command`]: the owner's own, or the reduced fallback), while cmd.exe
//!   would have passed a fold's `set` on: `set FOO=1 && mahbot …` runs without `FOO` — not
//!   detectable from the line, so stated rather than refused (the state a *later fold* would
//!   observe is refused);
//! - redirection is the runner's, so a target word is its file and a `%…%` pair in one is
//!   refused rather than expanded; the engine's argv refuses such words too;
//! - a member's `2>&1` copies at the member's stdout destination rather than cmd.exe's shared
//!   handle: the streams arrive whole per reader and interleave per chunk into the next member's
//!   input, neither being cmd.exe's write order. A search's merged stderr (refusal text, size
//!   marker when stdin-fed) lands in the captured stdout, scanned whole
//!   ([`super::engine_failure`]): the text stays visible, a last-member search still refuses
//!   through its exit status, and another member's failure becomes the engine's refusal with its
//!   message as the cause;
//! - the run's capture cap is one budget per stream ([`CaptureBudgets`]);
//! - leftover descendants can hold a capture pipe open: the drain bound is the single-child
//!   run's and resolves the same way — the whole tree is ended ([`super::stop_leftover`]) and the
//!   last member's status is the answer; only a stop that did not take reports the leftover
//!   ([`super::ShellRunResult::ExitedWithLeftovers`]). No program path builds a plan, so a plan
//!   always takes [`super::LeftoverPolicy::Command`];
//! - a stop ends the whole run's tree, not one member's — every member's process group on unix
//!   (already-reaped ones included, reaching the descendants they left) and the run's own job on
//!   Windows ([`super::Tree`]) — hence a plan stop is a whole-run stop, and a reported pid names
//!   the waiting group's first member;
//! - a fold's stderr may interleave differently with another step's than inside one cmd process,
//!   and stdout ordering is preserved by member, not by line.
//!
//! The plan's steps are plain data, so the executor's mechanics are driven from any host with
//! shell members, and [`runs_after`] is pinned by the executor's connector test.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::FutureExt;
use futures_util::stream::FuturesUnordered;
use futures_util::stream::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};

use super::grep_engine;
use super::grep_engine::windows;
use super::scan;
use super::windows_line::{Executed, Written};
use super::{
    CaptureBudget, Captured, KillOnDrop, Readers, RunOwner, SHELL_PIPE_READ_CAP, ShellPlatform,
    ShellRunResult, Stream, Tree, Watchdog, build_program_command, build_shell_command,
};

/// A plan: the rewritten line decomposed into the steps the runner spawns, in order.
#[derive(Debug)]
pub(super) struct Plan {
    pub(super) steps: Vec<Step>,
    /// This service's own image: the program every [`Run::Own`] step runs and the path
    /// [`own_image_plan`] compares words against. One field, so the argv's image and the one
    /// the engine rendered into the text cannot be told apart.
    pub(super) exe: PathBuf,
}

/// One step of a plan: one thing the runner spawns and waits for.
#[derive(Debug)]
pub(super) struct Step {
    pub(super) run: Run,
    /// The directory the step runs in (see the module docs).
    pub(super) cwd: PathBuf,
    pub(super) join: Join,
}

/// What one step runs.
#[derive(Debug)]
pub(super) enum Run {
    /// One `cmd.exe /C "<text>"` fold of adjacent verbatim members; a member that
    /// feeds a pipe is never folded (see the module docs).
    Shell { text: String },
    /// A call of this service's own image, spawned directly by the runner.
    Own {
        /// The member's argv after the image path, as cmd.exe would have delivered it.
        args: Vec<String>,
        /// The member's redirect tokens, verbatim (`[">", "out.txt"]`).
        redirects: Vec<String>,
    },
}

/// How a step joins the one before it: cmd.exe's connectors plus the pipeline link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Join {
    First,
    And,
    Or,
    /// `&`: runs regardless, and its failure does not stop the line. A bare line break ENDS the
    /// command rather than sequencing it, so the shell respells a plain break as this `&`.
    Always,
    /// `|`: one member of the pipeline the step before it feeds.
    Pipe,
}

impl Join {
    /// The connector's spelling: what the fold text joins its members with, and what the drift
    /// guard compares against the rewrite.
    #[must_use]
    pub(super) fn spelling(self) -> &'static str {
        match self {
            Join::First => "",
            Join::And => "&&",
            Join::Or => "||",
            Join::Always => "&",
            Join::Pipe => "|",
        }
    }
}

/// One member of a line as the decomposition needs it: the text cmd.exe must run, or the argv the
/// runner spawns the image with. It is the analyzer's per-member decision, not a re-reading of the
/// rewrite (the rendering walks the same list), and [`own_image_plan`] walks an unplanned line into
/// the same shape, so [`build`] is the one folding rule for both. Neither reading decides *what* a
/// member is: an own-image call arriving as [`Member::Shell`] is classified by [`build`].
#[derive(Debug)]
pub(super) enum Member {
    /// A verbatim member: cmd.exe runs it inside its fold unless [`build`] finds a plain call of
    /// this service's own image.
    Shell { text: String },
    /// A member that runs this service's own image for a reason its caller already decided: a
    /// served search, whose engine argv ([`grep_engine`]) the runner spawns itself.
    Own {
        argv: Vec<String>,
        redirects: Vec<String>,
        /// The cwd the analyzer tracked — authoritative for a search, whose spec names it too.
        cwd: PathBuf,
    },
}

/// Why a plan could not be built: a member cannot run as one plain call of the image, or the line
/// cannot be decomposed. Never a fallback to the shell — the reason becomes the agent-facing tool
/// error and no command runs.
#[derive(Debug)]
pub(super) enum Refusal {
    /// A member names this service's own image in a shape the runner cannot run
    /// ([`own_image_reference`]).
    OwnImageInShell,
    /// The command-position word names the image's file name through a path the runner cannot
    /// match ([`names_image_file`]).
    OwnImageSpelling(String),
    /// The member cannot run as it stands and must not go to the shell: redirect tokens the runner
    /// cannot apply ([`redirects_supported`]), or a line shape it cannot decompose.
    Shape(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Causes name the program as the agent spelled it: `mahbot`.
            Refusal::OwnImageInShell => write!(
                f,
                "the line also runs `mahbot` itself in a shape the runner cannot run as \
                 one plain call of the image — a launcher, or a word the shell would \
                 expand first — so it is refused rather than run with output nobody can \
                 vouch for"
            ),
            Refusal::OwnImageSpelling(spelling) => write!(
                f,
                "a shell member runs `{spelling}`, which names `mahbot`'s file name but \
                 is not the program the runner spawns — a call the runner cannot \
                 reproduce, so it is refused rather than run with output nobody can \
                 vouch for"
            ),
            Refusal::Shape(reason) => write!(f, "{reason}"),
        }
    }
}

/// Build the plan for `members` — in order, each with its following connector — tracking the
/// line's own cwd from `root_cwd`.
///
/// The one reading of a line for both callers: the analyzer's rewrite (served searches arrive as
/// [`Member::Own`]) and an unplanned line (all [`Member::Shell`]). Consecutive verbatim members
/// become ONE [`Run::Shell`] step — unless one feeds a pipe — a plain call of this image is
/// always an own step, and a fold can never swallow such a call. Windows only.
pub(super) fn build(
    members: &[(Member, String)],
    exe: &Path,
    root_cwd: &Path,
) -> Result<Plan, Refusal> {
    let mut steps: Vec<Step> = Vec::new();
    // The fold being accumulated: its text, its first member's join and its cwd. Flushed when an
    // own step or the line's end ends it.
    let mut fold: Option<Fold> = None;
    // Where the line's own `cd` members have left it: the directory every fold starts in and every
    // own step runs in, read with the analyzer's tracking ([`tracked_cd`]) so it cannot disagree
    // with the engine's spec files.
    let mut tracked = root_cwd.to_path_buf();
    // The directory the tracking lost, when a member moved it in a spelling this model cannot
    // follow ([`tracked_cd`], [`keyword_cd`]): kept until a step of its own needs it, because a
    // move made inside a fold is not observable outside it.
    let mut untracked_cd: Option<String> = None;
    for (index, (member, _)) in members.iter().enumerate() {
        let join = join_before(members, index)?;
        // This member's OWN following connector: a `|` there means cmd.exe takes its stdout as the
        // pipe's input, which no fold may share.
        let feeds_pipe = members[index].1 == "|";
        // A step of its own is spawned in the tracked directory, so an untracked `cd`
        // is refused where one begins (a member inside the `cd`'s own fold has
        // cmd.exe's own state and needs nothing of `tracked`).
        if (matches!(member, Member::Own { .. }) || fold.is_none())
            && let Some(cause) = &untracked_cd
        {
            return Err(Refusal::Shape(cause.clone()));
        }
        let (args, redirects, cwd) = match member {
            Member::Own {
                argv,
                redirects,
                cwd,
            } => (argv.clone(), redirects.clone(), cwd.clone()),
            Member::Shell { text } => {
                let words = member_words(text);
                // A plain call of this service's own image is not the shell's,
                // whatever its caller took it for: the runner waits for it and reports
                // the child's own status. It becomes an own step here, which is why
                // the fold breaks.
                if let Some((args, redirects)) = own_call(&words, exe) {
                    // An own step needs the tracked directory even with a fold open.
                    if let Some(cause) = &untracked_cd {
                        return Err(Refusal::Shape(cause.clone()));
                    }
                    let cwd = tracked.clone();
                    (args, redirects, cwd)
                } else {
                    // Not a plain call, yet still unreproducible as one — a launcher spelling, an
                    // unmatched path, an expanded word. Refused, never handed over.
                    if let Some(refusal) = own_image_reference(&words, text, exe) {
                        return Err(refusal);
                    }
                    // A `cd` inside one of cmd's keyword forms (`if exist x cd sub`): whether it
                    // runs at all depends on the keyword's condition, which this model does not
                    // evaluate, so the directory the line is left in is unknown — noted, and
                    // refused wherever a later step of its own would need it.
                    if let Some(cause) = keyword_cd(&words, text) {
                        untracked_cd = Some(cause);
                    }
                    // The directory the line is in *before* this member's own `cd` runs: the fold
                    // this member may open starts there and runs the `cd` itself.
                    let fold_cwd = tracked.clone();
                    match tracked_cd(&words, text, &tracked, feeds_pipe || join == Join::Pipe) {
                        Ok(Some(moved)) => tracked = moved,
                        Ok(None) => {}
                        Err(cause) => untracked_cd = Some(cause),
                    }
                    // cmd.exe pipes exactly the stdout of the command before a `|` and gates the
                    // pipeline on its connector, so both ends of a pipe are steps of their own: a
                    // fold around the head would send earlier members' output through the pipe
                    // too, one around the consumer would let what follows read its pipe alone.
                    if feeds_pipe {
                        flush_fold(&mut steps, &mut fold);
                    }
                    fold_push(&mut fold, text, join, fold_cwd);
                    // A member that *takes* a pipe ends its fold too: what follows it gets the
                    // shell's stdin, not that pipe's, so `… | head -3 && cat` must not leave the
                    // consumer reading the search's leftovers.
                    if feeds_pipe || join == Join::Pipe {
                        flush_fold(&mut steps, &mut fold);
                    }
                    continue;
                }
            }
        };
        flush_fold(&mut steps, &mut fold);
        redirects_supported(&redirects).map_err(Refusal::Shape)?;
        // An own step ends the fold before it and leaves the line where it ran.
        tracked.clone_from(&cwd);
        steps.push(Step {
            run: Run::Own { args, redirects },
            cwd,
            join,
        });
    }
    flush_fold(&mut steps, &mut fold);
    // A fold is a fresh cmd.exe: state it sets dies with it, and a later fold that would have
    // observed that state (cmd's errorlevel excepted — the run carries it) cannot be reproduced.
    if let Some(reason) = state_across_folds(&steps) {
        return Err(Refusal::Shape(reason));
    }
    Ok(Plan {
        steps,
        exe: exe.to_path_buf(),
    })
}

/// Add one verbatim member to the open fold, opening a fold at `start_cwd` (the directory the line
/// was in before this member's own `cd`, which the fold runs itself) when there is none.
fn fold_push(fold: &mut Option<Fold>, text: &str, join: Join, start_cwd: PathBuf) {
    match fold {
        Some(open) => {
            open.text.push(' ');
            open.text.push_str(join.spelling());
            open.text.push(' ');
            open.text.push_str(text);
        }
        None => {
            *fold = Some(Fold {
                text: text.to_string(),
                join,
                cwd: start_cwd,
            });
        }
    }
}

/// The fold being accumulated by [`build`]: its text, its join and its cwd.
struct Fold {
    text: String,
    join: Join,
    cwd: PathBuf,
}

/// Push the open fold (if any) as one [`Run::Shell`] step.
fn flush_fold(steps: &mut Vec<Step>, fold: &mut Option<Fold>) {
    if let Some(open) = fold.take() {
        steps.push(Step {
            run: Run::Shell { text: open.text },
            cwd: open.cwd,
            join: open.join,
        });
    }
}

/// The join of the member at `index` to the one before it: the previous member's own following
/// connector. Only connectors a fold can spell on ONE line reach this table — the shell respells
/// a plain break as `&` and refuses what it cannot read — so `"\n"` falls to the refusal.
fn join_before(members: &[(Member, String)], index: usize) -> Result<Join, Refusal> {
    if index == 0 {
        return Ok(Join::First);
    }
    match members[index - 1].1.as_str() {
        "&&" => Ok(Join::And),
        "||" => Ok(Join::Or),
        "&" => Ok(Join::Always),
        "|" => Ok(Join::Pipe),
        other => Err(Refusal::Shape(format!(
            "the rewritten line carries the connector `{other}`, which this \
             platform's runner does not read"
        ))),
    }
}

/// The first fold that changes shell state a LATER fold would have observed — the one class this
/// decomposition cannot reproduce. Each fold is its own `cmd.exe`, so state it sets dies with it;
/// an own step gets the environment the caller built
/// ([`crate::tools::shell::apply_agent_env`]), never one a fold set.
fn state_across_folds(steps: &[Step]) -> Option<String> {
    for (index, step) in steps.iter().enumerate() {
        let Run::Shell { text } = &step.run else {
            continue;
        };
        if !steps[index + 1..]
            .iter()
            .any(|later| matches!(later.run, Run::Shell { .. }))
        {
            continue;
        }
        if let Some(verb) = state_verb(text) {
            return Some(format!(
                "the shell member `{verb}` changes state the line's later members would \
                 have seen, and the runner gives each member its own interpreter"
            ));
        }
    }
    None
}

/// The first state-changing cmd.exe builtin one of a fold's members names, or `None`. A fold's text
/// is the segmenter's members rejoined, so it is read back as those.
///
/// Absent on purpose: `cd`/`chdir`, whose directory the analyzer tracks and whose move is cmd's
/// is cmd's business; and the console's verbs (`prompt`, `title`, `doskey`, `chcp`, `color`,
/// `verify`, `break`), whose whole effect is a console this service's program owns.
///
/// `pushd`/`popd` stay despite being cwd-family: the stack they push on is state a later fold could
/// read, and no fold here carries a stack.
fn state_verb(text: &str) -> Option<String> {
    /// cmd.exe's own verbs whose effect outlives the command that set it *and* is
    /// observable by a later fold: the environment and its scope, plus the directory
    /// stack's pair.
    const STATE_VERBS: &[&str] = &["set", "setlocal", "endlocal", "path", "pushd", "popd"];
    for (member, _) in
        windows::segment_command(text).expect("a fold is the members the segmenter read, rejoined")
    {
        let words = member_words(&member);
        // The command word where cmd.exe starts the command ([`command_word_index`]):
        // a `set` behind a redirection is still the member's own command.
        let Some(verb) = command_word_index(&words).map(|index| &words[index]) else {
            continue;
        };
        if let Some(key) = windows::verb_key(command_word(&verb.value))
            && STATE_VERBS.contains(&key.as_str())
        {
            return Some(key);
        }
    }
    None
}

/// The closed set of redirect spellings the runner can apply: `>`/`1>` and `>>`/`1>>` (stdout),
/// `2>`/`2>>` (stderr), `<` (stdin), the stdout merge `2>&1`, and `nul`/`NUL` as the null device.
///
/// A target-taking spelling is read as the shared token classifier reads it; everything else is a
/// named refusal, never an approximation: a dup that is not the merge (`>&2`), the combined `&>`,
/// a multi-digit descriptor, a missing target, a target with two or more `%`
/// ([`windows::has_percent_expansion`] — the shell spawns `cmd /C` with `/V:OFF`, so `!` is
/// ordinary), and a drive-relative target ([`windows::is_drive_relative`]). The reason names the
/// spelling, because the agent can change it and the runner cannot. The merge is applied because
/// its destination is the runner's already and `… 2>&1 | head` is common; it copies rather than
/// sharing cmd.exe's handle, which is the ordering the residuals state.
fn redirects_supported(redirects: &[String]) -> Result<(), String> {
    parse_redirects(redirects).map(|_| ())
}

/// One redirect the runner applies to a member it spawns itself.
#[derive(Debug)]
enum Redirect {
    Stdin(String),
    Stdout { target: String, append: bool },
    Stderr { target: String, append: bool },
    MergeIntoStdout,
}

/// Parse a member's verbatim redirect spellings into the runner's own reading;
/// `Err` is the agent-facing refusal for a spelling outside the closed set.
fn parse_redirects(redirects: &[String]) -> Result<Vec<Redirect>, String> {
    let mut parsed = Vec::new();
    let mut words = redirects.iter();
    while let Some(token) = words.next() {
        let spelling = split_redirect(token).map_err(|why| unsupported(token, why))?;
        let (stream, append, glued) = match spelling {
            // The merge names both descriptors, so no word is its target.
            RedirectSpelling::MergeIntoStdout => {
                parsed.push(Redirect::MergeIntoStdout);
                continue;
            }
            RedirectSpelling::Operator {
                stream,
                append,
                glued,
            } => (stream, append, glued),
        };
        // A spelling with no target of its own takes the next word; an operator at the
        // end of the list is a target the shell never got either.
        let target = if let Some(glued) = glued {
            glued
        } else {
            let Some(word) = words.next() else {
                return Err(format!(
                    "the runner cannot apply the redirection `{token}` to a member it \
                     spawns itself — its target word is missing"
                ));
            };
            word
        };
        // A target cmd.exe would have expanded: the runner opens the file itself, so the pair
        // would reach the file system as text the shell never delivered.
        if windows::has_percent_expansion(target) {
            return Err(format!(
                "the runner cannot apply the redirection `{token}` to a member it spawns \
                 itself — its target `{target}` carries a `%…%` pair, which cmd.exe \
                 would have expanded before the member ran"
            ));
        }
        // A drive-relative target (`> C:out.txt`) names a file in ANOTHER drive's current
        // directory, which nothing here tracks: joined onto the runner's cwd it would open a
        // different file silently. Refused, as the cwd tracker refuses it for `cd`.
        let delivered = windows::unquote_word(target);
        if windows::is_drive_relative(&delivered) {
            return Err(format!(
                "the runner cannot apply the redirection `{token}` to a member it spawns \
                 itself — its target `{delivered}` is drive-relative, naming another \
                 drive's current directory"
            ));
        }
        let target = target.to_string();
        parsed.push(match stream {
            Stream::Stdin => Redirect::Stdin(target),
            Stream::Stdout => Redirect::Stdout { target, append },
            Stream::Stderr => Redirect::Stderr { target, append },
        });
    }
    Ok(parsed)
}

/// One redirect token's reading: the stream it takes over with its target, or the merge.
#[derive(Debug, Clone, Copy)]
enum RedirectSpelling<'a> {
    /// An operator with its stream and append flag (`append` is `false` for `<`).
    Operator {
        stream: Stream,
        append: bool,
        /// The target glued to the operator (`>out.txt`), or `None` when the next word is it.
        glued: Option<&'a str>,
    },
    /// `2>&1`.
    MergeIntoStdout,
}

/// The redirect one token spells, or why it is outside the runner's closed set.
///
/// Read exactly where the shared classifier sees an operator, and nowhere else: a token opening
/// with `>`/`<` is a redirect, one carrying it later (`x>out.txt`, which the engine refuses before
/// any plan is built) is not. Only the runner's descriptors are read: `>`/`>>` take `1` at most,
/// `<` takes no digit, and a second digit, a dup (`>&2`, `3>&1`) or `&>` is a refusal.
fn split_redirect(token: &str) -> Result<RedirectSpelling<'_>, Unsupported> {
    if token == "2>&1" {
        return Ok(RedirectSpelling::MergeIntoStdout);
    }
    // A `&`-carrying spelling names another descriptor's destination (`>&2`, `1>&2`, `3>&1`, `&>`)
    // rather than a file the runner could open.
    if token.contains('&') {
        return Err(Unsupported::Descriptor);
    }
    let (stream, append, tail) = match token.as_bytes() {
        [b'1', b'>', b'>', ..] => (Stream::Stdout, true, &token[3..]),
        [b'1', b'>', ..] => (Stream::Stdout, false, &token[2..]),
        [b'2', b'>', b'>', ..] => (Stream::Stderr, true, &token[3..]),
        [b'2', b'>', ..] => (Stream::Stderr, false, &token[2..]),
        [b'>', b'>', ..] => (Stream::Stdout, true, &token[2..]),
        [b'>', ..] => (Stream::Stdout, false, &token[1..]),
        [b'<', ..] => (Stream::Stdin, false, &token[1..]),
        _ => return Err(Unsupported::Spelling),
    };
    // `<>file` is the read-write redirect: two ways into one file, which the closed set lacks.
    if tail.starts_with('>') {
        return Err(Unsupported::Spelling);
    }
    Ok(RedirectSpelling::Operator {
        stream,
        append,
        glued: (!tail.is_empty()).then_some(tail),
    })
}

/// Why one redirect token is outside the runner's closed set, as [`split_redirect`] read it: the
/// wording is decided where the spelling was rejected, not re-derived later.
#[derive(Debug, Clone, Copy)]
enum Unsupported {
    /// A dup that is not the merge (`>&2`, `1>&2`, `3>&1`) or the combined `&>`: another
    /// descriptor's destination, which the runner has none of.
    Descriptor,
    /// A spelling the closed set does not have at all (a multi-digit descriptor, `<>`).
    Spelling,
}

/// The refusal for one redirect token the runner cannot apply, naming the spelling. A
/// dup that is not the merge is explained on its own terms (it takes over a descriptor
/// the runner has no destination for); every other spelling names the set it applies.
#[must_use]
fn unsupported(token: &str, why: Unsupported) -> String {
    match why {
        Unsupported::Descriptor => format!(
            "the runner cannot apply the redirection `{token}` to a member it spawns \
             itself — the one descriptor merge it applies is `2>&1`, whose destination \
             the member's stdout already has, and its combined `&>` is not applied either"
        ),
        Unsupported::Spelling => format!(
            "the runner cannot apply the redirection `{token}` to a member it spawns \
             itself — it applies `2>&1`, `>`, `1>`, `>>`, `1>>`, `2>`, `2>>` and `<`, \
             the target-taking ones with their target glued to the operator or as the \
             next word"
        ),
    }
}

impl Plan {
    /// The single own-image step of a plan [`own_image_plan`] built: its argv, redirect tokens and
    /// cwd. `None` for a compound line (the members around the call are steps of their own).
    #[must_use]
    pub(super) fn direct_own(&self) -> Option<(&[String], &[String], &Path)> {
        let [step] = self.steps.as_slice() else {
            return None;
        };
        match &step.run {
            Run::Own { args, redirects } => Some((args, redirects, &step.cwd)),
            Run::Shell { .. } => None,
        }
    }
}

/// Apply one own-image member's redirects to the command that runs it, and report which streams
/// they took over and which merge the caller must perform.
///
/// Targets resolve against `cwd` — cmd.exe resolves a relative redirect against the process's
/// working directory, not the daemon's — and `nul`/`NUL` is the null device. The list applies
/// left to right: a `2>&1` between two redirects takes the stdout the ones before left.
pub(super) fn apply_redirects(
    cmd: &mut tokio::process::Command,
    redirects: &[String],
    cwd: &Path,
) -> io::Result<AppliedRedirects> {
    let parsed = parse_redirects(redirects).map_err(io::Error::other)?;
    let mut applied = AppliedRedirects::default();
    // What the member's stdout is at each point: `None` while it is still the caller's (the
    // capture, the pipe into the next member, a session's output file).
    let mut stdout: Option<RedirectDest> = None;
    for redirect in &parsed {
        match redirect {
            Redirect::Stdin(target) => {
                cmd.stdin(stdin_stdio(target, cwd)?);
                applied.streams.push(Stream::Stdin);
            }
            Redirect::Stdout { target, append } => {
                let dest = RedirectDest::open(target, *append, cwd)?;
                cmd.stdout(dest.stdio()?);
                applied.streams.push(Stream::Stdout);
                stdout = Some(dest);
            }
            Redirect::Stderr { target, append } => {
                cmd.stderr(RedirectDest::open(target, *append, cwd)?.stdio()?);
                applied.streams.push(Stream::Stderr);
            }
            // Stderr joins the file stdout was sent to (`> out.txt 2>&1`); with stdout the
            // caller's the merge happens at that destination, which the caller wires. A `2>`
            // written *before* the merge is overridden: its file was opened and truncated, then
            // lost stderr to the merge.
            Redirect::MergeIntoStdout => {
                if let Some(dest) = &stdout {
                    cmd.stderr(dest.stdio()?);
                    applied.streams.push(Stream::Stderr);
                } else {
                    applied.streams.retain(|stream| *stream != Stream::Stderr);
                    applied.merge_into_stdout = true;
                }
            }
        }
    }
    Ok(applied)
}

/// What a member's redirect tokens left for the caller's own wiring.
#[derive(Debug, Default)]
pub(super) struct AppliedRedirects {
    /// The streams the redirects took over; the caller wires none of them. A merged stderr whose
    /// stdout the caller still owns is not among them — that merge is the caller's part.
    pub(super) streams: Vec<Stream>,
    /// `2>&1` with stdout left to the caller: the stderr belongs wherever the caller sends its
    /// stdout. A caller whose two streams are one destination already reads only
    /// [`Self::streams`].
    pub(super) merge_into_stdout: bool,
}

/// Where a redirect sends one of the member's streams: the null device, or a file the runner
/// opened. The file is kept open rather than handed over as the stream's `Stdio`, so a `2>&1` after
/// it sends stderr to the SAME file: two handles on one path would each write from an offset.
enum RedirectDest {
    /// `nul`/`NUL`.
    Null,
    /// The file the redirect named, open for writing.
    File(std::fs::File),
}

impl RedirectDest {
    /// Open one output redirect's target: the null device for `nul`/`NUL` (any case), the file
    /// otherwise — created and truncated for `>`, created and appended for `>>`, resolved against
    /// `cwd`.
    fn open(target: &str, append: bool, cwd: &Path) -> io::Result<Self> {
        // The target cmd.exe would deliver — quote removal applied first — judged as the device
        // in the spellings cmd.exe accepts (`nul`, `nul:`, `nul.txt`, quoted or bare) by
        // [`scan::is_null_device`], and opened as a file when it is not one.
        let target = windows::unquote_word(target);
        if scan::is_null_device(&target) {
            return Ok(RedirectDest::Null);
        }
        let path = cwd.join(&target);
        Ok(RedirectDest::File(if append {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)?
        } else {
            std::fs::File::create(path)?
        }))
    }

    /// The stdio for one stream sent here. A file is *cloned*: the clone shares the open file's
    /// offset, which is what cmd.exe's descriptor merge gives the two streams.
    fn stdio(&self) -> io::Result<Stdio> {
        match self {
            RedirectDest::Null => Ok(Stdio::null()),
            RedirectDest::File(file) => Ok(Stdio::from(file.try_clone()?)),
        }
    }
}

/// The stdio an input redirect's target names: the null device, or the file cmd.exe would
/// have opened, resolved against `cwd`.
fn stdin_stdio(target: &str, cwd: &Path) -> io::Result<Stdio> {
    // The same reading [`RedirectDest::open`] makes of an output target.
    let target = windows::unquote_word(target);
    if scan::is_null_device(&target) {
        return Ok(Stdio::null());
    }
    Ok(Stdio::from(std::fs::File::open(cwd.join(&target))?))
}

/// Whether a whole line the segmenter refused names this service's own image where cmd.exe would
/// run a command from. The text is the line the interpreter is handed, so an unreadable spelling is
/// still read at its first word and for the image's path anywhere in it ([`spelling_in_text`]).
#[must_use]
fn own_image_in_text(text: &str, exe: &Path) -> bool {
    let Some(words) = windows::tokenize(text) else {
        // A text this model cannot read whole is still read at its first word — the one place a
        // command can start — beside the image's path anywhere in it.
        return names_image(text.split_whitespace().next().unwrap_or(""), exe)
            || spelling_in_text(text, exe);
    };
    names_image_in_words(&words, exe)
}

/// Whether one text's words name this service's own image where cmd.exe would run a command from:
/// its opening word, a word after a connector (`&&`, `||`, `&`, `|`, read on the word's own
/// spelling so a quoted `"&"` is an argument), or a launcher's command word
/// ([`launcher_hands_image`]). Any other position is ordinary text; cmd.exe's punctuation
/// ([`command_word`], [`is_decoration`]) does not consume it, so `@mahbot -V` and `( mahbot -V )`
/// are calls.
#[must_use]
fn names_image_in_words(words: &[grep_engine::GrepWord], exe: &Path) -> bool {
    let mut starts_command = true;
    let mut index = 0;
    while let Some(word) = words.get(index) {
        // A redirection does not consume the command position either: `> out.txt mahbot
        // -V` is the call, and a target word after a target-taking operator is part of
        // the redirection rather than a word a command could start at.
        if word.redirect {
            index += if word.needs_target { 2 } else { 1 };
            continue;
        }
        if starts_command {
            if names_image(&word.value, exe) {
                return true;
            }
            let launcher = windows::verb_key(command_word(&word.value))
                .is_some_and(|key| LAUNCHER_VERBS.contains(&key.as_str()));
            if launcher && launcher_hands_image(words, index, exe).is_some() {
                return true;
            }
        }
        // cmd.exe's punctuation does not consume the command position: the word after a
        // group's `(` or a lone `@` is still a place a command can start.
        starts_command = matches!(word.raw.as_str(), "&&" | "||" | "&" | "|")
            || (starts_command && is_decoration(&word.value));
        index += 1;
    }
    false
}

/// The refusal for a member that names this service's own image in a shape the runner
/// cannot run as one plain call, or `None` when it names no image at all (the member then
/// stays with its fold). Three causes reach here: a command-position word naming the image
/// by a path the runner cannot match ([`names_image_file`]); a word that IS the image
/// ([`is_own_image_word`]) while the member is not one plain call (a `%…%` word); and a
/// launcher ([`launcher_hands_image`]) whose grammar hands the image on rather than
/// running it.
fn own_image_reference(words: &[grep_engine::GrepWord], text: &str, exe: &Path) -> Option<Refusal> {
    // The command word: not cmd.exe's punctuation, and not a redirection's target.
    let first = command_word_index(words)?;
    let word = &words[first];
    if is_own_image_word(&word.value, exe) {
        return Some(Refusal::OwnImageInShell);
    }
    if let Some(spelling) = names_image_file(&word.value, exe) {
        return Some(Refusal::OwnImageSpelling(spelling.to_string()));
    }
    let launcher = windows::verb_key(command_word(&word.value))
        .is_some_and(|key| LAUNCHER_VERBS.contains(&key.as_str()));
    if launcher {
        return match launcher_hands_image(words, first, exe) {
            Some(HandsOn::ToCommand) => Some(Refusal::OwnImageInShell),
            // The launcher's command word is not one this model can read: the cause claims only
            // what was read, since a word after such a launcher is not proof of what it runs.
            Some(HandsOn::Unreadable) => Some(Refusal::Shape(format!(
                "the line's member `{text}` runs a launcher whose own command word this \
                 model cannot read while `mahbot` appears after it, so the call is not \
                 one the runner can make"
            ))),
            None => None,
        };
    }
    None
}

/// The verbs that hand the image on to something else rather than running it: cmd.exe's
/// `start` and `call`, and an explicit interpreter. A member whose command word is one of
/// these is refused when the word its own grammar hands the command to is the image.
const LAUNCHER_VERBS: &[&str] = &["start", "call", "cmd", "powershell", "pwsh"];

/// Whether a launcher at `words[launcher]` hands the line to this service's own image, and
/// how narrowly that can be read: the word its own grammar gives its command to
/// ([`launched_command`]) — or, when that word is one this model cannot see (a `cmd` whose
/// switch it cannot read), any word after the launcher.
///
/// A word merely *carried* by the launcher's command is not one: `cmd /c dir mahbot` lists
/// the directory, `powershell Get-Process mahbot` asks for a process, and both are ordinary
/// commands the shell may run.
fn launcher_hands_image(
    words: &[grep_engine::GrepWord],
    launcher: usize,
    exe: &Path,
) -> Option<HandsOn> {
    match launched_command(words, launcher) {
        Some(at) => names_image(&words[at].value, exe).then_some(HandsOn::ToCommand),
        None => words[launcher + 1..]
            .iter()
            .any(|word| names_image(&word.value, exe))
            .then_some(HandsOn::Unreadable),
    }
}

/// How a launcher hands the line on this service's own image, as far as this model can read
/// it: the two readings refuse the line, and only [`HandsOn::ToCommand`] says the image is
/// the word the launcher runs — which is why the causes the two render differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandsOn {
    /// The word the launcher's own grammar hands its command to is the image.
    ToCommand,
    /// That word is not one this model can see, while a word after the launcher names the image.
    Unreadable,
}

/// The word a launcher's own grammar hands its command to, or `None` when this model cannot
/// see one: `cmd`'s `/c` (or `/k`) text; `start`'s first word that is neither a switch nor a
/// title; `call`'s word; an interpreter's first word that is not a switch
/// ([`is_switch_word`]). `None` keeps the caller on its wider reading, and a shape this
/// grammar does not resolve is left to the shell, as the residuals state.
fn launched_command(words: &[grep_engine::GrepWord], launcher: usize) -> Option<usize> {
    let key = windows::verb_key(command_word(&words[launcher].value))?;
    let rest = &words[launcher + 1..];
    let offset = match key.as_str() {
        "cmd" => {
            // cmd.exe's own `/c`/`/k` (`-c`/`-k` too), the spelling that carries the text
            // cmd.exe runs. A word without the switch character, or a bare `c`, is an
            // ordinary argument.
            let switch = rest.iter().position(|word| {
                is_switch_word(&word.value)
                    && matches!(word.value[1..].to_ascii_lowercase().as_str(), "c" | "k")
            })?;
            switch + 1
        }
        "start" => rest
            .iter()
            .position(|word| !is_switch_word(&word.value) && !word.raw.starts_with('"'))?,
        "call" => 0,
        "powershell" | "pwsh" => rest.iter().position(|word| !is_switch_word(&word.value))?,
        _ => return None,
    };
    let at = launcher + 1 + offset;
    (at < words.len()).then_some(at)
}

/// Whether one word is one of cmd.exe's own switch spellings (`/wait`, `/d`,
/// `-Command`): a switch character followed by a single component. A word carrying another
/// separator after it is a path (`/Users/owner/mahbot`), not a switch.
#[must_use]
fn is_switch_word(word: &str) -> bool {
    word.starts_with(['/', '-']) && !word[1..].contains(['/', '\\'])
}

/// Whether one word of a command line names this service's own image at all — by identity
/// ([`is_own_image_word`]) or its file name through a path ([`names_image_file`]). The
/// *detection* reading: it never says the member can be run as a plain call.
#[must_use]
fn names_image(word: &str, exe: &Path) -> bool {
    is_own_image_word(word, exe) || names_image_file(word, exe).is_some()
}

/// The spelling of one word that names this service's own image's file name through a path
/// — `.\mahbot.exe`, `target\debug\mahbot.exe` — where a bare `mahbot`/`mahbot.exe` is
/// [`is_own_image_word`]'s identity case already. `None` for a word naming some other file:
/// the runner cannot tell which file cmd.exe would resolve it to, so such a member is
/// refused rather than run.
#[must_use]
fn names_image_file<'a>(word: &'a str, exe: &Path) -> Option<&'a str> {
    let name = image_file_name(exe)?;
    // A word with no separator is a bare name, already [`is_own_image_word`]'s answer.
    let (_, file) = command_word(word).rsplit_once(['\\', '/'])?;
    windows::verb_key(file)
        .is_some_and(|key| Some(key) == windows::verb_key(name))
        .then_some(word)
}

/// Whether one word names this service's own image *by identity*: its full path — case,
/// separator and verbatim prefix aside, as the engine's cwd gate compares — or its bare file
/// name, `mahbot.exe` and `mahbot` being one verb to cmd.exe's dispatch. Only the `.exe`
/// suffix folds, so `mahbot.com`/`mahbot.bat`/`mahbot.cmd` name a file of their own and a key
/// spelled through one stays the shell's. What this accepts is the program the runner runs.
#[must_use]
fn is_own_image_word(word: &str, exe: &Path) -> bool {
    if windows::same_spelling(Path::new(command_word(word)), exe) {
        return true;
    }
    let Some(name) = image_file_name(exe) else {
        return false;
    };
    windows::verb_key(command_word(word)).is_some_and(|key| Some(key) == windows::verb_key(name))
}

/// One word read as a command word rather than as text: cmd.exe's `@` prefix and the group
/// parentheses at its ends dropped. cmd.exe reads `(`/`)` as delimiters outside quotes and
/// `@` as the prefix of the command it opens, so the command word of `(mahbot`, `@mahbot`
/// and `mahbot)` is `mahbot`.
#[must_use]
fn command_word(word: &str) -> &str {
    word.trim_start_matches(['@', '(']).trim_end_matches(')')
}

/// The index of the word a member's command starts at: cmd.exe's punctuation (`@`, group
/// parentheses) and a redirection take no command position, so `> out.txt mahbot -V` names
/// its command in `mahbot`.
/// `None` for a member with no command word at all.
#[must_use]
fn command_word_index(words: &[grep_engine::GrepWord]) -> Option<usize> {
    let mut index = 0;
    while let Some(word) = words.get(index) {
        if word.redirect {
            index += if word.needs_target { 2 } else { 1 };
            continue;
        }
        if is_decoration(&word.value) {
            index += 1;
            continue;
        }
        return Some(index);
    }
    None
}

/// Whether one word is nothing but cmd.exe's punctuation — a group's `(`/`)`, the
/// no-echo `@` — and so names no command of its own. Such a word keeps the
/// command position it stands in (`( mahbot -V )` is a call where a command can
/// start); it is never a word the runner can run.
#[must_use]
fn is_decoration(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| matches!(c, '(' | ')' | '@'))
}

/// The words of one member text the segmenter read — total for the segmenter's own
/// text, so a text that does not tokenize panics rather than falling open to the shell.
#[must_use]
fn member_words(text: &str) -> Vec<grep_engine::GrepWord> {
    windows::tokenize(text).expect("the segmenter's own text always tokenizes")
}

/// This image's file name — the last component of its path, split on every separator Windows
/// accepts. Not `Path::file_name`: on a `/`-separator host that would leave `C:\…\mahbot.exe`
/// whole, so the reading is textual and answers the same everywhere.
#[must_use]
fn image_file_name(exe: &Path) -> Option<&str> {
    exe.to_str()?
        .rsplit(['\\', '/'])
        .next()
        .filter(|name| !name.is_empty())
}

/// The coarse half of [`own_image_in_text`]: whether the raw text carries the image's path
/// spelling, case, separators and verbatim prefix aside. The one signal left for a line the
/// cmd.exe model cannot read.
#[must_use]
fn spelling_in_text(text: &str, exe: &Path) -> bool {
    let spelled = fold_path(&crate::util::strip_verbatim_prefix(exe).to_string_lossy());
    !spelled.is_empty() && fold_path(text).contains(&spelled)
}

/// The comparison spelling of a path or a line: `/` folded to `\`, lowercased. Enough for the
/// coarse check, which only asks whether one spelling appears in a line.
#[must_use]
fn fold_path(text: &str) -> String {
    text.replace('/', "\\").to_lowercase()
}

/// What a line that the serve decision did not plan turned out to be.
pub(super) enum OwnImage {
    /// The line names no image: the shell runs it exactly as it always did.
    None,
    /// The line names the image and the runner's decomposition covers it — one step for a
    /// lone call, the whole chain for a line naming the image among other members.
    Direct(Plan),
    /// The line names the image in a shape the runner cannot run; the cause is for
    /// [`refusal_message`].
    Refused(String),
}

/// Read a command line the serve decision did not plan for a call of this service's own
/// image — `mahbot debug …`, `mahbot chrome …`, `mahbot bench-openrouter …`, `mahbot -V`.
///
/// `None` for a line that names no own image. A line that names it is read with the same
/// cmd.exe model the rewrite is built from: every member as [`Member::Shell`], classified by
/// [`build`], the cwd tracked across the members. A shape that reading cannot represent — a
/// launcher spelling, an unmatched path, an inapplicable redirect, an untrackable `cd` — is
/// refused ([`Refusal`], rendered by [`refusal_message`]) rather than run. A line the model
/// cannot read is still read for a call where a command can start ([`own_image_in_text`]) and
/// for the image's path spelling ([`spelling_in_text`]); one that only mentions the bare name
/// in unreadable text is the shell's.
///
/// The refusal quotes `written`, the line the agent wrote: `executed` is the platform's
/// reading of it ([`crate::tools::shell::windows_line`] may have respelled it).
#[must_use]
fn own_image_plan(
    executed: Executed<'_>,
    written: Written<'_>,
    exe: &Path,
    cwd: &Path,
) -> OwnImage {
    let Some(members) = windows::segment_command(executed.0) else {
        // A line the segmenter refused is read twice: for a call where a command can start
        // ([`own_image_in_text`] — its first word, one after a connector, one a launcher
        // hands the image to, so `cd sub && mahbot -V (x)` is still the call) and for the
        // image's path anywhere in the text ([`spelling_in_text`], which sees the member no
        // position reading can). A line that merely *mentions* the name is ordinary text.
        return if own_image_in_text(executed.0, exe) || spelling_in_text(executed.0, exe) {
            OwnImage::Refused(format!(
                "the line names `mahbot` but cannot be read as cmd.exe's own chain of \
                 commands (`{}`), and a call in a line the runner cannot read is not \
                 one it can make itself — run it as a line of its own",
                written.0
            ))
        } else {
            OwnImage::None
        };
    };
    // The line is the runner's only when a member names the image — in command position or
    // through a launcher; one that only *carries* the name (`grep -rn mahbot .`) is the
    // shell's. This pre-pass keeps [`build`]'s refusals off such a line.
    if !members
        .iter()
        .any(|(text, _)| names_image_in_words(&member_words(text), exe))
    {
        return OwnImage::None;
    }
    let line: Vec<(Member, String)> = members
        .into_iter()
        .map(|(text, conn)| (Member::Shell { text }, conn))
        .collect();
    // Tracked from the workspace root the analyzer starts from, so both readings of a
    // directory spell it the same ([`tracked_cd`]).
    let root = grep_engine::canonical_or_lexical(cwd, ShellPlatform::Windows);
    match build(&line, exe, &root) {
        Ok(plan) => OwnImage::Direct(plan),
        Err(refusal) => OwnImage::Refused(refusal.to_string()),
    }
}

/// The plan a line that runs this service's own image is to be run by, or [`OwnImage::None`]
/// for a line that is the shell's: the one dispatch of the own-image reading, shared by the
/// shell tool and the background sessions.
///
/// Windows only by construction: elsewhere the shell waits for its children, so the image's
/// path (`current_exe()`) would have no reader. A failed lookup is fail-open, exactly as for
/// the analyzer's rewrite: nothing is recognised as the image, so the line runs as today.
#[must_use]
pub(super) fn plan_for_command(
    executed: Executed<'_>,
    written: Written<'_>,
    cwd: &Path,
) -> OwnImage {
    if super::SHELL_PLATFORM != super::ShellPlatform::Windows {
        return OwnImage::None;
    }
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            tracing::debug!(
                err = %e,
                "no own executable path: a line naming this service's image runs as the shell runs it"
            );
            return OwnImage::None;
        }
    };
    own_image_plan(executed, written, &exe, cwd)
}

/// The argv and redirect tokens of one member's plain call of this service's own image, or
/// `None` when the member is not one: no command word at all (a redirect-only member), a
/// command word that does not name the image by identity ([`is_own_image_word`]), or a word
/// carrying a `%…%` pair cmd.exe would have expanded (argv would reach the image as text the
/// shell never delivered) — the last refused as [`Refusal::OwnImageInShell`] rather than run.
///
/// The command word is where cmd.exe starts the command — not its punctuation, not a
/// redirection (`> out.txt mahbot -V` is still the call) — and the argv is the words after it,
/// unquoted by cmd.exe's model. Redirect tokens ride along verbatim, in cmd.exe's order, for
/// [`build`] and [`apply_redirects`].
fn own_call(words: &[grep_engine::GrepWord], exe: &Path) -> Option<(Vec<String>, Vec<String>)> {
    let first = command_word_index(words)?;
    if !is_own_image_word(&words[first].value, exe) {
        return None;
    }
    let mut argv = Vec::new();
    let mut redirects = Vec::new();
    let mut index = 0;
    while let Some(word) = words.get(index) {
        if word.redirect {
            redirects.push(word.raw.clone());
            if word.needs_target
                && let Some(target) = words.get(index + 1)
            {
                redirects.push(target.raw.clone());
                index += 1;
            }
        } else if index > first {
            if windows::has_percent_expansion(&word.raw) {
                return None;
            }
            argv.push(word.value.clone());
        }
        index += 1;
    }
    Some((argv, redirects))
}

/// The cwd a `cd`-family member hands the members after it, or `None` when the member is not
/// one: the analyzer's own tracking ([`grep_engine::resolve_cd`], the one reading of cmd.exe's
/// `cd` grammar), asked of an unplanned line so the directory a member runs in cannot disagree
/// with the engine's cwd gate.
///
/// `Err` is the refusal for a directory change this tracking cannot follow: a spelling the verb
/// key does not know (`cd..`, `cd\Users`, `cd/d`, `@cd`), a redirection before the command word, a
/// target the resolver refuses (a drive-relative `cd C:foo`), or any of them beside a pipe
/// (`piped`), where the shell gives the pipeline its own interpreter. A step run in the wrong
/// directory is a wrong answer, not a missing one — and the runner spawns every step itself, so
/// nothing downstream could notice.
///
/// Windows-only by construction; the `home` [`grep_engine::resolve_cd`] takes is unix-only, never
/// consulted here.
fn tracked_cd(
    words: &[grep_engine::GrepWord],
    text: &str,
    cwd: &Path,
    piped: bool,
) -> Result<Option<PathBuf>, String> {
    let platform = ShellPlatform::Windows;
    // The member's command word, where cmd.exe starts the command (punctuation and a
    // leading redirection take no position), with whether it is the member's first. A
    // member with no command word at all (a redirect-only text) has no verb, so its first
    // spelling stands in.
    let command = command_word_index(words).map(|at| (at, words[at].value.as_str()));
    let (leads, verb) = match command {
        Some((at, verb)) => (at == 0, verb),
        None => (false, text.split_whitespace().next().unwrap_or("")),
    };
    // One side of a pipe is its own interpreter in cmd.exe, and which directory the members
    // after it see is unanswerable here: the analyzer refuses the same shape for a served
    // line ([`grep_engine`]'s `CdUntrackable`).
    if piped && names_cd_family(verb) {
        return Err(untrackable_cd(
            text,
            "runs on one side of a pipe, which the shell gives an interpreter of its own",
        ));
    }
    if grep_engine::is_cd_segment(verb, platform) {
        if !leads {
            return Err(untrackable_cd(
                text,
                "cannot be read where the tracking starts",
            ));
        }
        return grep_engine::resolve_cd(text, cwd, Path::new(""), platform)
            .map(Some)
            .map_err(|_| untrackable_cd(text, "is a form the runner cannot track"));
    }
    if grep_engine::is_cd_spelling(verb, platform) {
        return Err(untrackable_cd(
            text,
            "is a spelling of cmd.exe's own directory change that the runner cannot track",
        ));
    }
    Ok(None)
}

/// Whether one word names cmd.exe's directory-change family in any spelling the tracking reads:
/// the family's own names ([`grep_engine::is_cd_segment`]) or cmd's fused ones
/// ([`grep_engine::is_cd_spelling`]).
#[must_use]
fn names_cd_family(word: &str) -> bool {
    let platform = ShellPlatform::Windows;
    grep_engine::is_cd_segment(word, platform) || grep_engine::is_cd_spelling(word, platform)
}

/// The refusal for a member that changes the shell's directory inside one of cmd.exe's keyword
/// forms (`if exist x cd sub`, `for %f in (*) do cd sub`, `else cd sub`), or `None` when the
/// command word is no keyword form or names no directory change of the family.
///
/// The keyword's own condition decides whether that `cd` runs at all, and this model does not
/// evaluate conditions, so the members after it would run in a directory the shell might not
/// have used. The reading is wider than the grammar's (any word after the keyword, not only the
/// command position its condition hands its own command), so a `cd` merely standing in the
/// member (`if exist x echo cd sub`) is refused too — the direction a wrong directory must not
/// be read in.
fn keyword_cd(words: &[grep_engine::GrepWord], text: &str) -> Option<String> {
    /// cmd.exe's own keyword forms a command can run from.
    const KEYWORD_FORMS: &[&str] = &["if", "else", "for"];
    let at = command_word_index(words)?;
    let key = windows::verb_key(&words[at].value);
    if !key
        .as_deref()
        .is_some_and(|key| KEYWORD_FORMS.contains(&key))
    {
        return None;
    }
    words[at + 1..]
        .iter()
        .any(|word| names_cd_family(command_word(&word.value)))
        .then(|| {
            format!(
                "the line's member `{text}` runs a directory change inside one of the \
                 shell's own keyword forms, whose condition decides whether it runs at \
                 all, so the directory the line is left in is not something this model \
                 can read"
            )
        })
}

/// The refusal for a `cd`-family member this tracking cannot follow, saying what
/// about it the runner could not read.
fn untrackable_cd(text: &str, why: &str) -> String {
    format!(
        "the line's `cd` member `{text}` {why}, so a step of its own after it cannot be \
         run in the directory the shell would have used"
    )
}

/// The agent-facing failure for a line [`own_image_plan`] refused. Its frame is
/// the shared [`super::REFUSAL_FRAME`], so the same cause cannot reach the
/// agent worded two ways through the two entry points; what follows the frame is
/// the consequence of *this* one, that the answer is not the command's output,
/// where the search's own entry point states that it is not an empty match set.
#[must_use]
pub(super) fn refusal_message(cause: &str) -> String {
    format!(
        "{}{cause}. This is not the command's output.",
        super::REFUSAL_FRAME
    )
}

/// Whether a group whose first step joins with `join` runs after the group before
/// it reported `previous` — cmd.exe's own rule, on cmd's own errorlevel: `&&`
/// needs a zero, `||` any non-zero (a member killed by a signal has no code at
/// all, which is not a success either), and both `&` and the line's first step
/// always run. A skipped group leaves the errorlevel in place, which is why the
/// caller keeps the previous status rather than clearing it.
///
/// `Pipe` never starts a group ([`pipe_groups`]) and answers "runs": a pipeline
/// member is not sequenced by a connector.
#[must_use]
fn runs_after(join: Join, previous: Option<i32>) -> bool {
    match join {
        Join::And => previous == Some(0),
        Join::Or => previous != Some(0),
        Join::First | Join::Always | Join::Pipe => true,
    }
}

/// The plan's pipe groups: each is the range of steps whose members are connected
/// by real pipes, in order. A step whose join is [`Join::Pipe`] continues the
/// group before it; every other join starts a new one, and the first step always
/// starts one.
#[must_use]
fn pipe_groups(steps: &[Step]) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    for (index, step) in steps.iter().enumerate() {
        if index > 0 && step.join != Join::Pipe {
            groups.push(start..index);
            start = index;
        }
    }
    if !steps.is_empty() {
        groups.push(start..steps.len());
    }
    groups
}

// Execution

/// One spawned member of a group.
struct MemberRun {
    child: Child,
    /// The member's pid, taken from the child at spawn — what the memory watchdog
    /// samples, while the member is in its set.
    pid: u32,
    /// The member's exit status once it was reaped.
    status: Option<std::process::ExitStatus>,
}

/// How many bytes a plan run's capture readers may keep: one budget per stream
/// ([`CaptureBudget`], which states the split, what it buys and why), shared by
/// every member that writes that stream.
struct CaptureBudgets {
    stdout: CaptureBudget,
    stderr: CaptureBudget,
}

impl CaptureBudgets {
    fn new() -> Self {
        Self {
            stdout: CaptureBudget::new(SHELL_PIPE_READ_CAP),
            stderr: CaptureBudget::new(SHELL_PIPE_READ_CAP),
        }
    }
}

/// Run a plan: every group in order, each member spawned with the containment the
/// single-child runner gives its one child: its output captured into the run's streams, and the
/// run's bounds — one deadline, one memory cadence, the capture budget of the stream each member
/// writes, one output drain — applied to the plan as a whole.
///
/// The result maps onto the same [`ShellRunResult`] variants a shell run produces, so the caller
/// cannot tell a plan run from one: `Completed`, `TimedOut`, `ExitedWithLeftovers`,
/// `MemoryExceeded`, `SpawnFailed`. The status of a completed run is the last executed group's
/// last member's.
///
/// A leftover descendant holding a capture pipe is stopped with the run's tree
/// ([`super::stop_leftover`]): the last member's status is the answer, and only a stop that did not
/// take reports it. A plan serves only an agent's rewritten line, so it takes the agent's shaping
/// ([`super::LeftoverPolicy`]) and never reports a leftover the status already settled.
pub(super) async fn run(
    plan: &Plan,
    timeout: Duration,
    drain_limit: Duration,
    memory_limit: Option<u64>,
    owner: RunOwner,
) -> ShellRunResult {
    let start = std::time::Instant::now();
    let mut tree = Tree::new(owner);
    // Kill-on-drop: an aborted task (cancellation, a sibling's panic, runtime teardown) must
    // not orphan a member. Disarmed on every path that reaped them or ended the tree itself.
    let mut kill_guard = KillOnDrop::new(tree.clone());
    let cancel = tokio_util::sync::CancellationToken::new();
    let budgets = CaptureBudgets::new();
    let mut readers = Readers::default();
    // The run's bounds, watched across every group: one deadline, one cadence.
    let mut watchdog = Watchdog::new(Some(timeout));
    // cmd's errorlevel: the last executed group's status gates the next, and a skipped group
    // leaves it in place.
    let mut previous: Option<i32> = None;
    // The status the run reports: the last executed group's last member's.
    let mut reported: Option<std::process::ExitStatus> = None;

    for group in pipe_groups(&plan.steps) {
        if !runs_after(plan.steps[group.start].join, previous) {
            continue;
        }
        let mut members =
            match spawn_group(plan, group, &mut tree, &mut readers, &budgets, &cancel).await {
                Ok(members) => members,
                // The group's members were killed and reaped by the spawn path, and every
                // earlier group was too, so the guard is disarmed rather than left to end a
                // reaped group's tree ([`KillOnDrop`]'s contract). The capture readers are
                // cancelled too: nothing collects them here, and one left waiting on a
                // member's pipe would outlive the run.
                Err(e) => {
                    cancel.cancel();
                    kill_guard.disarm();
                    return ShellRunResult::SpawnFailed(e);
                }
            };
        // The pid a stop names: the group the run was waiting for.
        let named_pid = members.first().map(|member| member.pid);
        match wait_group(&mut members, &mut watchdog, memory_limit).await {
            GroupEnd::Reaped => {
                let last = members.last().and_then(|member| member.status);
                // cmd's errorlevel for the next group's connector, and the status reported.
                previous = last.and_then(|status| status.code());
                reported = last;
            }
            GroupEnd::TimedOut => {
                end_group(&tree, &mut members, &mut kill_guard).await;
                let (stdout, stderr) = collect_stopped(&cancel, &mut readers).await;
                return ShellRunResult::TimedOut {
                    stdout,
                    stderr,
                    pid: named_pid,
                    elapsed: start.elapsed(),
                };
            }
            GroupEnd::MemoryExceeded { used, limit } => {
                end_group(&tree, &mut members, &mut kill_guard).await;
                let (stdout, stderr) = collect_stopped(&cancel, &mut readers).await;
                return ShellRunResult::MemoryExceeded {
                    stdout,
                    stderr,
                    pid: named_pid,
                    elapsed: start.elapsed(),
                    used,
                    limit,
                };
            }
            GroupEnd::Failed(e) => {
                // A member that could not be reaped may still be running: the guard stays
                // armed and ends the run's tree. The capture readers end when the tree's
                // pipes close, as the single-child runner leaves its own on a failed spawn.
                return ShellRunResult::SpawnFailed(e);
            }
        }
    }

    // Every group ran or was skipped: the run ended on its own, so nothing is killed here and
    // the job is kept rather than closed.
    kill_guard.disarm();
    let drained = readers.drain(drain_limit).await;
    // The first group of a non-empty plan always runs, and a plan is built from a non-empty
    // line: no status here means a plan nobody built — a failed run, not a panic.
    let Some(status) = reported else {
        return ShellRunResult::SpawnFailed(io::Error::other("the plan has no step to run"));
    };
    if drained {
        tree.retain_after_completion();
        let (stdout, stderr) = readers.collect().await;
        return ShellRunResult::Completed {
            stdout,
            stderr,
            status,
            elapsed: start.elapsed(),
        };
    }
    // The drain bound was exceeded: a leftover process still holds a capture pipe. A plan run is
    // an agent's rewritten line, so it takes the agent's shaping
    // ([`super::leftover_result`], [`super::LeftoverPolicy::Command`]): the whole tree is stopped
    // and a stop that took is a plain completed run. No containment is named in the residual
    // case — the members are one process group each on unix and one job on Windows.
    super::leftover_result(
        super::LeftoverPolicy::Command,
        tree,
        &mut readers,
        &cancel,
        status,
        None,
        start,
    )
    .await
}

/// Spawn one group's members, left to right, connecting consecutive members with real pipes the
/// runner owns, and starting a reader for every stream the group leaves to the run's capture.
///
/// A member that cannot be spawned ends the group: the run's tree is ended and the members this
/// group already started are reaped here — the caller disarms the kill guard on that path, which
/// is [`KillOnDrop`]'s contract. A redirect the runner cannot open ends the group the same way.
async fn spawn_group(
    plan: &Plan,
    group: std::ops::Range<usize>,
    tree: &mut Tree,
    readers: &mut Readers,
    budgets: &CaptureBudgets,
    cancel: &tokio_util::sync::CancellationToken,
) -> io::Result<Vec<MemberRun>> {
    let mut members: Vec<MemberRun> = Vec::with_capacity(group.len());
    // What the member before this one left for this one to read, or `None` before
    // the group's first member, whose stdin is inherited (as the single shell
    // child's is).
    let mut feed: Option<Pipe> = None;
    for index in group.clone() {
        let step = &plan.steps[index];
        let mut cmd = match &step.run {
            Run::Shell { text } => build_shell_command(text, &step.cwd),
            Run::Own { args, .. } => build_program_command(&plan.exe, args, &step.cwd),
        };
        // The member's own redirects, applied by the runner (the spellings were
        // validated when the plan was built): the streams they take over are not wired here,
        // and a `2>&1` whose stdout the runner still owns tells the wiring below that the
        // member's stderr shares that destination.
        let applied = match &step.run {
            Run::Own { redirects, .. } => match apply_redirects(&mut cmd, redirects, &step.cwd) {
                Ok(applied) => applied,
                // A redirect the runner cannot open — a missing directory, a directory target
                // — ends the group exactly as a failed spawn does: this group's earlier members
                // are reaped here, which lets the caller disarm the kill guard.
                Err(e) => {
                    end_group_members(tree, &mut members).await;
                    return Err(e);
                }
            },
            Run::Shell { .. } => AppliedRedirects::default(),
        };
        let fed_by_pipe = feed.is_some();
        if fed_by_pipe && !applied.streams.contains(&Stream::Stdin) {
            cmd.stdin(Stdio::piped());
        }
        // stdout is piped unless the member redirected it: a non-last member's pipe is the next
        // member's input, the last member's is the run's capture.
        if !applied.streams.contains(&Stream::Stdout) {
            cmd.stdout(Stdio::piped());
        }
        if !applied.streams.contains(&Stream::Stderr) {
            cmd.stderr(Stdio::piped());
        }
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                end_group_members(tree, &mut members).await;
                return Err(e);
            }
        };
        // The platform answers `None` only once the child is reaped, which the wait path below
        // is what does: a member this runner spawned has its pid, and it is what the memory
        // watchdog samples.
        let pid = child
            .id()
            .expect("a freshly spawned child still has its pid");
        tree.attach(pid);
        // Wire this member's stdin from the member before it.
        match feed.take() {
            // The first member of the group: its stdin is whatever this process has, exactly as
            // the single shell child's is.
            None => {}
            Some(pipe) => match child.stdin.take() {
                Some(write_end) => connect_pipe(pipe, write_end),
                // The member's own redirect took its stdin (`… | grep x . < in.txt`): the
                // redirect wins, as cmd.exe's does, so the producer's pipe is closed by
                // dropping the read ends.
                None => drop(pipe),
            },
        }
        // What this member leaves for the next one — its stdout, and its stderr too when it
        // merged the two — or for the run's capture.
        let piped = index + 1 < group.end;
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        if piped {
            feed = Some(Pipe {
                stdout: stdout.take(),
                // A merged stderr is one of the streams the consumer's input carries (see
                // [`Pipe`]); without a merge it stays the run's stderr channel, below.
                stderr: if applied.merge_into_stdout {
                    stderr.take()
                } else {
                    None
                },
            });
        }
        if let Some(pipe) = stdout {
            readers.push(
                Captured::Stdout,
                pipe,
                budgets.stdout.clone(),
                cancel.clone(),
            );
        }
        if let Some(pipe) = stderr {
            // A merged stderr is the member's stdout destination, so its bytes join the run's
            // captured stdout — and the stdout budget, the stream they were charged to.
            let (stream, budget) = if applied.merge_into_stdout {
                (Captured::Stdout, budgets.stdout.clone())
            } else {
                (Captured::Stderr, budgets.stderr.clone())
            };
            readers.push(stream, pipe, budget, cancel.clone());
        }
        members.push(MemberRun {
            child,
            pid,
            status: None,
        });
    }
    Ok(members)
}

/// The pipe between two members of a group: what the member before left for the member after.
struct Pipe {
    /// The producer's stdout, or `None` when a redirect of its own sent stdout elsewhere. A
    /// `2>&1` written before such a `>` still left the merged stderr feeding this pipe (see
    /// [`connect_pipe`]); a producer with neither stream leaves a pipe nothing writes into, and
    /// its input is closed — the end-of-input cmd.exe's own pipeline gives it.
    stdout: Option<ChildStdout>,
    /// The producer's stderr, when it merged into its stdout: the pipe's input too when a later
    /// `>` took stdout over (see [`connect_pipe`]), otherwise the same destination as that
    /// stdout, fed by a second pump.
    stderr: Option<ChildStderr>,
}

/// Connect the pipe the member before left to the member just spawned: its stdout into the
/// consumer's stdin, plus its stderr when the producer merged the two.
///
/// A producer whose own redirect sent stdout elsewhere leaves its merged stderr — written before
/// that `>` — as the pipe's input, and otherwise leaves nothing to read, so the consumer's input
/// is dropped, the end-of-input cmd.exe's pipeline gives it either way.
fn connect_pipe(pipe: Pipe, write_end: ChildStdin) {
    match (pipe.stdout, pipe.stderr) {
        // Two streams, one destination: both pumps write through the shared handle.
        (Some(stdout), Some(stderr)) => {
            let shared = SharedStdin::new(write_end);
            spawn_merged_copy(stdout, shared.clone());
            spawn_merged_copy(stderr, shared);
        }
        (Some(stdout), None) => spawn_pipe_copy(stdout, write_end),
        // The member's own `>` sent stdout to a file, but its `2>&1` came first: cmd.exe had
        // taken this pipe for the merge, so the merged stream is the pipe's input even though
        // nothing writes stdout into it.
        (None, Some(stderr)) => spawn_pipe_copy(stderr, write_end),
        (None, None) => drop(write_end),
    }
}

/// The write end of one pipe between two members, shared by the pumps that feed it: the
/// producer's stdout, plus its stderr when it merged the two. Each pump takes the handle for one
/// chunk, so both streams reach the consumer whole; the consumer's input closes when the last
/// handle is dropped.
#[derive(Clone)]
struct SharedStdin(Arc<tokio::sync::Mutex<ChildStdin>>);

impl SharedStdin {
    fn new(write_end: ChildStdin) -> Self {
        SharedStdin(Arc::new(tokio::sync::Mutex::new(write_end)))
    }
}

/// Copy one member's stream into the next member's stdin — the pipe the runner
/// owns between two members of a group. Both a plain stdout pipe and a merged
/// stderr that `2>&1` handed to the pipe before a later `>` took stdout over are
/// one stream here.
///
/// The task ends when either end does: the consumer exiting drops the producer's
/// next write (the broken pipe a shell pipeline gives it, which the engine's own
/// exit code already handles), and the producer exiting closes the consumer's
/// input. It is deliberately not bounded by the run's capture budget: a pipeline
/// member's stream must reach its consumer whole, exactly as it did through the
/// shell's own pipe.
fn spawn_pipe_copy(
    mut from: impl tokio::io::AsyncRead + Unpin + Send + 'static,
    mut to: ChildStdin,
) {
    tokio::spawn(async move {
        let _ = tokio::io::copy(&mut from, &mut to).await;
    });
}

/// Copy one of a member's two merged streams into the shared write end of the pipe it feeds
/// (`2>&1`), one chunk at a time so the streams interleave rather than one pump waiting out the
/// other's whole stream.
///
/// The task ends when either end does, as [`spawn_pipe_copy`]'s: a failed write is the consumer
/// gone, a closed read end the producer gone.
fn spawn_merged_copy(
    mut from: impl tokio::io::AsyncRead + Unpin + Send + 'static,
    to: SharedStdin,
) {
    tokio::spawn(async move {
        let mut buf = [0_u8; MERGED_COPY_CHUNK];
        loop {
            match from.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    let mut write_end = to.0.lock().await;
                    if write_end.write_all(&buf[..read]).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
}

/// One read of a merged pipe pump ([`spawn_merged_copy`]): the chunk one write into the shared
/// handle carries.
const MERGED_COPY_CHUNK: usize = 8 * 1024;

/// How waiting for one group's members ended. The bounds behind these outcomes are the shared
/// [`Watchdog`]'s, so only the payload differs from the single-child runner's [`WaitOutcome`].
enum GroupEnd {
    /// Every member was reaped; their statuses are on them.
    Reaped,
    /// The run's deadline expired while a member was still running.
    TimedOut,
    /// A memory sample crossed the ceiling, carried with the ceiling for the report.
    MemoryExceeded { used: u64, limit: u64 },
    /// A member could not be reaped at all.
    Failed(io::Error),
}

/// Wait for every member of one group under the run's shared bounds, recording each member's
/// status as it is reaped.
async fn wait_group(
    members: &mut [MemberRun],
    watchdog: &mut Watchdog,
    memory_limit: Option<u64>,
) -> GroupEnd {
    // The statuses land here rather than on the members, because each member's own wait future
    // borrows it for as long as the group is waited for. Every member arrives unwaited and
    // un-reaped ([`spawn_group`] is the only constructor), so one without a status is exactly
    // one still running — and the pids the watchdog samples are the same reading of the group.
    debug_assert!(
        members.iter().all(|member| member.status.is_none()),
        "a member is waited for as soon as it is spawned, never twice"
    );
    let mut statuses: Vec<Option<std::process::ExitStatus>> = vec![None; members.len()];
    // The pids the watchdog samples, with the member each belongs to: every member (its pid is
    // taken at spawn), a reaped one leaving both sets at once.
    let mut live: Vec<(usize, u32)> = members
        .iter()
        .enumerate()
        .map(|(index, member)| (index, member.pid))
        .collect();
    let end = {
        let mut pending: FuturesUnordered<_> = members
            .iter_mut()
            .enumerate()
            .map(|(index, member)| member.child.wait().map(move |result| (index, result)))
            .collect();
        loop {
            tokio::select! {
                biased;
                Some((index, result)) = pending.next() => {
                    match result {
                        Ok(status) => {
                            statuses[index] = Some(status);
                            live.retain(|(at, _)| *at != index);
                        }
                        Err(e) => break GroupEnd::Failed(e),
                    }
                    if statuses.iter().all(Option::is_some) {
                        break GroupEnd::Reaped;
                    }
                }
                () = &mut watchdog.deadline => break GroupEnd::TimedOut,
                _ = watchdog.sample.tick() => {
                    let pids: Vec<u32> = live.iter().map(|(_, pid)| *pid).collect();
                    if let Some(limit) = memory_limit
                        && let Some(sample) = watchdog.measure(&pids, limit)
                    {
                        break GroupEnd::MemoryExceeded {
                            used: sample.used,
                            limit: sample.limit,
                        };
                    }
                }
            }
        }
    };
    match end {
        GroupEnd::Reaped => {
            for (member, status) in members.iter_mut().zip(statuses) {
                member.status = status;
            }
            GroupEnd::Reaped
        }
        other => other,
    }
}

/// End the group the run was waiting for, as the single-child stop does: each member is
/// signalled (the fallback for a run whose containment could not be established, which is why
/// the direct kill comes first), the tree ends them together, then they are reaped.
async fn end_group(tree: &Tree, members: &mut [MemberRun], kill_guard: &mut KillOnDrop) {
    end_group_members(tree, members).await;
    // Reaped — disarm the guard (the explicit kill already ran).
    kill_guard.disarm();
}

/// [`end_group`]'s kill-and-reap, for a caller that owns the disarm decision: every member is
/// signalled, the run's tree ends them together, and they are reaped.
async fn end_group_members(tree: &Tree, members: &mut [MemberRun]) {
    for member in members.iter_mut() {
        let _ = member.child.start_kill();
    }
    tree.terminate();
    for member in members.iter_mut() {
        let _ = member.child.wait().await;
    }
}

/// Collect a run that is being stopped: cancel the capture readers and take what they hold, so
/// every stop path — the deadline, the memory ceiling — reports its partial output instead of a
/// hang. The kill itself is the caller's, because what has to be ended differs.
async fn collect_stopped(
    cancel: &tokio_util::sync::CancellationToken,
    readers: &mut Readers,
) -> (Vec<u8>, Vec<u8>) {
    cancel.cancel();
    readers.collect().await
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::super::{SHELL_PIPE_TAIL_CAP, TRUNCATED_MIDDLE};
    use super::*;
    use std::ops::Range;

    /// The image and workspace every case runs against: the decomposition never touches either, so
    /// the paths only have to be the shapes a real install has.
    const EXE: &str = r"C:\Program Files\MahBot\mahbot.exe";
    const ROOT: &str = r"C:\ws";

    /// One verbatim member and the connector that follows it. The directory its fold starts in is
    /// [`build`]'s own tracking, not a field of the member.
    fn shell(text: &str, conn: &str) -> (Member, String) {
        (
            Member::Shell {
                text: text.to_string(),
            },
            conn.to_string(),
        )
    }

    /// One served member: its argv after the image path, its redirect tokens, the cwd its search
    /// tracked, and the connector that follows it.
    fn own(argv: &[&str], redirects: &[&str], cwd: &str, conn: &str) -> (Member, String) {
        (
            Member::Own {
                argv: argv.iter().map(|word| (*word).to_string()).collect(),
                redirects: redirects.iter().map(|word| (*word).to_string()).collect(),
                cwd: PathBuf::from(cwd),
            },
            conn.to_string(),
        )
    }

    /// The engine's own hand-off words for a served member: the decomposition does not care what
    /// they are, only that they travel as argv.
    fn served(cwd: &str, conn: &str) -> (Member, String) {
        own(
            &["__grep-engine", "--spec-file", r"C:\tmp\spec.json"],
            &[],
            cwd,
            conn,
        )
    }

    /// The plan for a line whose members are handed over as the analyzer hands them, tracked from
    /// [`ROOT`] — the root every case reads its `cd` members against.
    fn plan(members: &[(Member, String)]) -> Plan {
        build_at(ROOT, members).expect("a decomposable line")
    }

    /// [`build`] for one case, with `root` (the analyzer's own canonicalization of it) as the cwd
    /// its tracking starts from.
    fn build_at(root: &str, members: &[(Member, String)]) -> Result<Plan, Refusal> {
        build(members, Path::new(EXE), &tracked_root_of(root))
    }

    /// The workspace root the decomposition tracks from: `root` under the analyzer's own
    /// canonicalization, which a `/`-separator host resolves like any other relative path.
    fn tracked_root_of(root: &str) -> PathBuf {
        grep_engine::canonical_or_lexical(Path::new(root), ShellPlatform::Windows)
    }

    /// The plan [`own_image_plan`] built for a line, or a panic naming what it read instead.
    fn direct_plan(command: &str) -> Plan {
        match own_image_plan(
            Executed(command),
            Written(command),
            Path::new(EXE),
            Path::new(ROOT),
        ) {
            OwnImage::Direct(plan) => plan,
            OwnImage::Refused(cause) => panic!("{command}: refused: {cause}"),
            OwnImage::None => panic!("{command}: not read as an own-image line"),
        }
    }

    /// The agent-facing cause of the refusal for a line.
    fn refusal_cause(command: &str) -> String {
        match own_image_plan(
            Executed(command),
            Written(command),
            Path::new(EXE),
            Path::new(ROOT),
        ) {
            OwnImage::Refused(cause) => cause,
            OwnImage::Direct(_) | OwnImage::None => {
                panic!("{command}: expected a refusal")
            }
        }
    }

    /// The text one step runs, for the shell steps of a plan.
    fn shell_text(step: &Step) -> &str {
        match &step.run {
            Run::Shell { text } => text,
            Run::Own { .. } => panic!("expected a shell step"),
        }
    }

    /// The cwds of a plan's steps.
    fn step_cwds(plan: &Plan) -> Vec<&Path> {
        plan.steps.iter().map(|step| step.cwd.as_path()).collect()
    }

    /// One member of a plan-shape case: a verbatim shell member or a served
    /// search, with the connector that follows it — the analyzer's own data.
    #[derive(Clone, Copy)]
    enum ShapeMember {
        Shell(&'static str, &'static str),
        Served(&'static str, &'static str),
    }

    /// One assertion a plan-shape case makes about the plan's steps.
    #[derive(Clone, Copy)]
    enum ShapeCheck {
        Steps(usize),
        Join(usize, Join),
        ShellText(usize, &'static str),
        Cwd(usize, &'static str),
        /// The cwd the decomposition tracks from [`ROOT`] itself.
        CwdTrackedRoot(usize),
        /// The tracked root with `child` joined to it.
        CwdUnderRoot(usize, &'static str),
        CwdEquals(usize, usize),
        PipeGroups(&'static [(usize, usize)]),
        OwnStep(usize),
        OwnArgs(usize, &'static str),
        RunsAfter(usize, Option<i32>, bool),
    }

    struct ShapeCase {
        name: &'static str,
        members: &'static [ShapeMember],
        checks: &'static [ShapeCheck],
    }

    /// The plan's member shape, as one table: how many steps a line's members
    /// become, how they join, which of them the runner spawns, and the cwd every
    /// step carries. Each case is one line the analyzer can hand over.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the plan's shapes
    fn plan_shape_cases() {
        let cases: &[ShapeCase] = &[
            // `grep -rn x .` — nothing but a served search: one own step in the
            // workspace root, sequenced as the line's first.
            ShapeCase {
                name: "a lone search is one own step",
                members: &[ShapeMember::Served(ROOT, "")],
                checks: &[
                    ShapeCheck::Steps(1),
                    ShapeCheck::Join(0, Join::First),
                    ShapeCheck::Cwd(0, ROOT),
                    ShapeCheck::OwnStep(0),
                    ShapeCheck::OwnArgs(0, "__grep-engine"),
                ],
            },
            // `grep -rn x . | head -3`: one pipe group of two members — the
            // engine, then the tail the runner feeds it to.
            ShapeCase {
                name: "a pipelined search is one pipe group",
                members: &[
                    ShapeMember::Served(ROOT, "|"),
                    ShapeMember::Shell("head -3", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(2),
                    ShapeCheck::Join(1, Join::Pipe),
                    ShapeCheck::PipeGroups(&[(0, 2)]),
                ],
            },
            // `cd src && grep -rn x .` — the fold is cmd.exe's own `cd`, and the
            // served step runs in the directory that `cd` chose.
            ShapeCase {
                name: "a cd before a search is a fold and the tracked cwd",
                members: &[
                    ShapeMember::Shell("cd src", "&&"),
                    ShapeMember::Served(r"C:\ws\src", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(2),
                    ShapeCheck::ShellText(0, "cd src"),
                    ShapeCheck::CwdTrackedRoot(0),
                    ShapeCheck::Join(1, Join::And),
                    ShapeCheck::Cwd(1, r"C:\ws\src"),
                ],
            },
            // `cd src && grep -rn x . | head -3`: one fold, one pipe group —
            // three members, two steps.
            ShapeCase {
                name: "a cd search and tail is a fold then a pipe group",
                members: &[
                    ShapeMember::Shell("cd src", "&&"),
                    ShapeMember::Served(r"C:\ws\src", "|"),
                    ShapeMember::Shell("head -3", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(3),
                    ShapeCheck::Join(2, Join::Pipe),
                    ShapeCheck::PipeGroups(&[(0, 1), (1, 3)]),
                    ShapeCheck::ShellText(0, "cd src"),
                    ShapeCheck::CwdEquals(2, 1),
                ],
            },
            // `cd src && type f.txt | grep -n needle`: the producer is a step of
            // its own, so the plan is three steps.
            ShapeCase {
                name: "a member that feeds a pipe is a step of its own",
                members: &[
                    ShapeMember::Shell("cd src", "&&"),
                    ShapeMember::Shell("type f.txt", "|"),
                    ShapeMember::Served(r"C:\ws\src", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(3),
                    ShapeCheck::PipeGroups(&[(0, 1), (1, 3)]),
                    ShapeCheck::ShellText(0, "cd src"),
                    ShapeCheck::Join(0, Join::First),
                    ShapeCheck::ShellText(1, "type f.txt"),
                    ShapeCheck::Join(1, Join::And),
                    ShapeCheck::OwnStep(2),
                    ShapeCheck::Join(2, Join::Pipe),
                    ShapeCheck::CwdTrackedRoot(0),
                    ShapeCheck::CwdUnderRoot(1, "src"),
                    ShapeCheck::Cwd(2, r"C:\ws\src"),
                ],
            },
            // A member that takes a pipe is a step of its own: the connector
            // after the consumer runs in cmd's own process, not in the pipe.
            ShapeCase {
                name: "the pipeline consumer is a step of its own",
                members: &[
                    ShapeMember::Served(ROOT, "|"),
                    ShapeMember::Shell("head -3", "&&"),
                    ShapeMember::Shell("cat", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(3),
                    ShapeCheck::PipeGroups(&[(0, 2), (2, 3)]),
                    ShapeCheck::OwnStep(0),
                    ShapeCheck::ShellText(1, "head -3"),
                    ShapeCheck::Join(1, Join::Pipe),
                    ShapeCheck::ShellText(2, "cat"),
                    ShapeCheck::Join(2, Join::And),
                ],
            },
            // Two consumers and a tail after them: every pipe member is its own
            // step, and the tail is gated by the last consumer's own status.
            ShapeCase {
                name: "two consumers and a tail are three steps",
                members: &[
                    ShapeMember::Served(ROOT, "|"),
                    ShapeMember::Shell("head -3", "|"),
                    ShapeMember::Shell("sort", "&&"),
                    ShapeMember::Shell("echo done", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(4),
                    ShapeCheck::PipeGroups(&[(0, 3), (3, 4)]),
                    ShapeCheck::Join(3, Join::And),
                ],
            },
            // Two members around a `|` are two steps even with nothing after the consumer: the
            // pipe is the runner's, so both ends are spawned.
            ShapeCase {
                name: "a bare pipe has two steps",
                members: &[
                    ShapeMember::Shell("type a.txt", "|"),
                    ShapeMember::Shell("sort", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(2),
                    ShapeCheck::ShellText(0, "type a.txt"),
                    ShapeCheck::ShellText(1, "sort"),
                    ShapeCheck::PipeGroups(&[(0, 2)]),
                ],
            },
            // `grep a f1 && grep b f2`: two served members, each in its own group, the second
            // sequenced by cmd's `&&` and running in the first's cwd.
            ShapeCase {
                name: "two searches are two sequenced own steps",
                members: &[
                    ShapeMember::Served(ROOT, "&&"),
                    ShapeMember::Served(ROOT, ""),
                ],
                checks: &[
                    ShapeCheck::Steps(2),
                    ShapeCheck::Join(1, Join::And),
                    ShapeCheck::RunsAfter(1, Some(0), true),
                    ShapeCheck::RunsAfter(1, Some(1), false),
                ],
            },
            // An own-image member breaks the fold, and the fold after it runs in that cwd.
            ShapeCase {
                name: "an own step breaks the fold and carries the cwd",
                members: &[
                    ShapeMember::Shell("cd src", "&&"),
                    ShapeMember::Served(r"C:\ws\src", "&&"),
                    ShapeMember::Shell("type note.txt", ""),
                ],
                checks: &[
                    ShapeCheck::Steps(3),
                    ShapeCheck::ShellText(0, "cd src"),
                    ShapeCheck::ShellText(2, "type note.txt"),
                    ShapeCheck::Cwd(2, r"C:\ws\src"),
                ],
            },
            // The pipe-group split: `|` continues a group, every other join starts one.
            ShapeCase {
                name: "pipe groups split on every join but a pipe",
                members: &[
                    ShapeMember::Shell("echo one", "&&"),
                    ShapeMember::Served(ROOT, "|"),
                    ShapeMember::Shell("head -1", "&&"),
                    ShapeMember::Served(ROOT, "||"),
                    ShapeMember::Shell("tail -1", ""),
                ],
                checks: &[ShapeCheck::PipeGroups(&[(0, 1), (1, 3), (3, 4), (4, 5)])],
            },
        ];
        for case in cases {
            let members: Vec<(Member, String)> = case
                .members
                .iter()
                .map(|member| match member {
                    ShapeMember::Shell(text, conn) => shell(text, conn),
                    ShapeMember::Served(cwd, conn) => served(cwd, conn),
                })
                .collect();
            let plan = plan(&members);
            for check in case.checks {
                match check {
                    ShapeCheck::Steps(steps) => {
                        assert_eq!(plan.steps.len(), *steps, "{}: step count", case.name);
                    }
                    ShapeCheck::Join(at, join) => {
                        assert_eq!(plan.steps[*at].join, *join, "{}: step {at}", case.name);
                    }
                    ShapeCheck::ShellText(at, text) => {
                        assert_eq!(
                            shell_text(&plan.steps[*at]),
                            *text,
                            "{}: step {at}",
                            case.name
                        );
                    }
                    ShapeCheck::Cwd(at, cwd) => {
                        assert_eq!(
                            plan.steps[*at].cwd,
                            PathBuf::from(cwd),
                            "{}: step {at} cwd",
                            case.name
                        );
                    }
                    ShapeCheck::CwdTrackedRoot(at) => {
                        assert_eq!(
                            plan.steps[*at].cwd,
                            tracked_root_of(ROOT),
                            "{}: step {at} cwd",
                            case.name
                        );
                    }
                    ShapeCheck::CwdUnderRoot(at, child) => {
                        assert_eq!(
                            plan.steps[*at].cwd,
                            tracked_root_of(ROOT).join(child),
                            "{}: step {at} cwd",
                            case.name
                        );
                    }
                    ShapeCheck::CwdEquals(a, b) => {
                        assert_eq!(
                            plan.steps[*a].cwd, plan.steps[*b].cwd,
                            "{}: steps {a} and {b} cwd",
                            case.name
                        );
                    }
                    ShapeCheck::PipeGroups(groups) => {
                        let expected: Vec<Range<usize>> =
                            groups.iter().map(|(start, end)| *start..*end).collect();
                        assert_eq!(pipe_groups(&plan.steps), expected, "{}: groups", case.name);
                    }
                    ShapeCheck::OwnStep(at) => {
                        assert!(
                            matches!(plan.steps[*at].run, Run::Own { .. }),
                            "{}: step {at} is an own step",
                            case.name
                        );
                    }
                    ShapeCheck::OwnArgs(at, first) => {
                        let Run::Own { args, .. } = &plan.steps[*at].run else {
                            panic!("{}: step {at} is not an own step", case.name);
                        };
                        assert_eq!(args[0], *first, "{}: step {at} argv", case.name);
                    }
                    ShapeCheck::RunsAfter(at, after, expected) => {
                        assert_eq!(
                            runs_after(plan.steps[*at].join, *after),
                            *expected,
                            "{}: step {at}",
                            case.name
                        );
                    }
                }
            }
        }
        assert_eq!(pipe_groups(&[]), Vec::<Range<usize>>::new());
    }

    /// A member whose redirect the runner applies, carried with the step.
    #[test]
    fn a_redirect_rides_its_own_step() {
        let plan = plan(&[own(
            &["__grep-engine", "--spec-file", r"C:\tmp\spec.json"],
            &[">", "out.txt"],
            ROOT,
            "",
        )]);
        let Run::Own { redirects, .. } = &plan.steps[0].run else {
            panic!("expected an own step");
        };
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);
    }

    /// The fold rejoins its members with the connectors the rewrite spells: here the `&` a plain
    /// line break was respelled into, and cmd's own `&&`.
    #[test]
    fn a_fold_joins_its_members_with_their_own_connectors() {
        let plan = plan(&[
            shell("echo one", "&"),
            shell("echo two", "&&"),
            shell("echo three", ""),
        ]);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(
            shell_text(&plan.steps[0]),
            "echo one & echo two && echo three"
        );
    }

    /// A verbatim member that is a plain call of this service's own program is not the shell's to
    /// run, whatever its caller took it for: [`build`] converts it into an own step — the shell
    /// never sees the call — and the fold around it breaks there. This is what makes a rewritten
    /// `mahbot … && grep …` the runner's rather than a refusal.
    #[test]
    fn a_fold_member_that_calls_the_image_becomes_an_own_step() {
        for (members, args) in [
            (vec![shell("mahbot -V", "")], ["-V"].as_slice()),
            (
                vec![shell("echo hi", "&&"), shell("mahbot -V", "")],
                ["-V"].as_slice(),
            ),
            (
                vec![shell(r#""C:\Program Files\MahBot\mahbot.exe" debug"#, "")],
                ["debug"].as_slice(),
            ),
            // cmd.exe's `@` no-echo prefix, glued and standing alone: not a word of the command
            // it opens, so the call is one plain call and its argv is what follows the prefix.
            (vec![shell("@mahbot -V", "")], ["-V"].as_slice()),
            (vec![shell("@ mahbot -V", "")], ["-V"].as_slice()),
            // A redirection written before the command takes no command position: the call is the
            // one it names, and the redirection is its own.
            (vec![shell("> out.txt mahbot -V", "")], ["-V"].as_slice()),
            (
                vec![
                    shell("grep -n x f.txt", "&&"),
                    shell("2> err.txt mahbot debug", ""),
                ],
                ["debug"].as_slice(),
            ),
        ] {
            let plan = build_at(ROOT, &members).expect("a call of the image is the runner's");
            let own: Vec<&Step> = plan
                .steps
                .iter()
                .filter(|step| matches!(step.run, Run::Own { .. }))
                .collect();
            assert_eq!(own.len(), 1, "{members:?}");
            let Run::Own { args: argv, .. } = &own[0].run else {
                unreachable!("filtered to the own step");
            };
            assert_eq!(argv, args, "{members:?}");
        }
        // The redirection written before the command rides the own step, in cmd.exe's order.
        let redirected = plan(&[shell("> out.txt mahbot -V", "")]);
        let Run::Own { redirects, .. } = &redirected.steps[0].run else {
            panic!("a call of the image is an own step");
        };
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);
        for text in [
            "echo mahbot",
            "type mahbot.exe",
            "echo (mahbot -V)",
            "echo @mahbot -V",
            "echo hi > mahbot.exe",
            "echo hi < mahbot -V.txt",
            "cmd /c dir mahbot",
            "cmd -c dir mahbot",
            r#"cmd /c "dir mahbot""#,
            r#"powershell -c "Get-Process mahbot""#,
            r"start /d C:\ws notepad mahbot",
            "mahbot.bat -V",
            "mahbot.com -V",
        ] {
            let plan = plan(&[shell(text, "&&"), served(ROOT, "")]);
            assert_eq!(plan.steps.len(), 2, "{text}");
        }
    }

    /// A member naming this service's own program in a shape the runner cannot run as one plain
    /// call refuses the line, and the cause names the shape: a launcher, which hands the image on
    /// instead of running it, and a path spelling that names the program's file name but not the
    /// program the runner runs. The launcher shapes this grammar does not resolve are the shell's.
    #[test]
    fn a_fold_naming_the_image_in_a_shape_the_runner_cannot_run_refuses_the_line() {
        for (members, spelling) in [
            (vec![shell("start mahbot chrome", "")], false),
            (vec![shell("call mahbot.exe -V", "")], false),
            (vec![shell("cmd /c mahbot bench-openrouter", "")], false),
            (vec![shell("cmd -c mahbot -V", "")], false),
            (vec![shell(r".\mahbot.exe -V", "")], true),
            (vec![shell(r"target\debug\mahbot.exe -V", "")], true),
            (
                vec![shell("echo hi", "&&"), shell(r".\mahbot.exe -V", "")],
                true,
            ),
        ] {
            let refusal = build_at(ROOT, &members).expect_err("a shape the runner cannot run");
            match refusal {
                Refusal::OwnImageSpelling(spelled) => {
                    assert!(
                        spelling,
                        "{members:?}: unexpected spelling cause: {spelled}"
                    );
                }
                Refusal::OwnImageInShell => {
                    assert!(!spelling, "{members:?}: expected the spelling cause");
                }
                Refusal::Shape(reason) => {
                    panic!("{members:?}: unexpected cause: {reason}");
                }
            }
        }
        // A launcher whose own command word this model cannot read: refused too, with a cause that
        // claims only what was read — a word after it is not proof of what it runs. A bare
        // `c`/`k` letter is an ordinary argument, not cmd's switch.
        for text in ["cmd /x dir mahbot", "cmd dir c mahbot", "cmd dir k mahbot"] {
            let refusal = build_at(ROOT, &[shell(text, "")])
                .expect_err("a launcher whose command word cannot be read");
            let Refusal::Shape(reason) = refusal else {
                panic!("{text}: expected the shape refusal: {refusal}");
            };
            assert!(
                reason.contains("command word this model cannot read"),
                "{text}: {reason}"
            );
            assert!(
                !reason.contains("runs `mahbot` itself"),
                "{text}: the cause must not claim what the reading did not see: {reason}"
            );
        }
    }

    /// A redirect the runner cannot apply refuses the line, naming the spelling — the stdout
    /// merge is not one of them any more ([`redirects_supported`]).
    #[test]
    fn an_unsupported_redirect_refuses_the_line() {
        let refusal = build_at(
            ROOT,
            &[own(
                &["__grep-engine", "--spec-file", r"C:\tmp\spec.json"],
                &[">&2"],
                ROOT,
                "",
            )],
        )
        .expect_err("a dup the runner has no destination for is refused");
        let Refusal::Shape(reason) = refusal else {
            panic!("expected the redirect refusal");
        };
        assert!(reason.contains("`>&2`"), "{reason}");
        assert!(
            reason.contains("descriptor merge it applies is `2>&1`"),
            "{reason}"
        );
    }

    /// A fold whose shell state a later fold would have observed is refused — each fold is a
    /// fresh cmd.exe, and state that dies with it cannot be approximated. Allowed: no later fold
    /// can read their state — a `cd` (the analyzer tracks the directory), `pushd <target>`
    /// (tracked the same way), a console verb (a console child owns a console of its own).
    #[test]
    fn a_state_changing_fold_before_another_fold_refuses_the_line() {
        for member in ["set MAHBOT_TEST=1", "path C:\\other", "pushd src"] {
            let refusal = build_at(
                ROOT,
                &[
                    shell(member, "&&"),
                    served(ROOT, "&&"),
                    shell("echo done", ""),
                ],
            )
            .expect_err("the state dies with the fold");
            assert!(matches!(refusal, Refusal::Shape(_)), "{member}: {refusal}");
        }

        // Nothing after the state: nothing could have observed it.
        plan(&[shell("set MAHBOT_TEST=1", "&&"), served(ROOT, "")]);
        plan(&[shell("cd src", "&&"), served(r"C:\ws\src", "")]);
        plan(&[shell("pushd src", "&&"), served(r"C:\ws\src", "")]);
        plan(&[
            shell("color 0A", "&&"),
            served(ROOT, "&&"),
            shell("echo done", ""),
        ]);
    }

    /// A member that is a directory change cmd.exe reads and the tracking cannot
    /// follow refuses the line: with the runner spawning every step itself, nothing
    /// downstream can still notice that the members after it would run in a
    /// directory the shell would not have used.
    #[test]
    fn a_cd_the_tracking_cannot_follow_refuses_the_line() {
        for member in [
            // cmd's own spellings of the builtin with its target or switch fused to
            // the name, and its `@` no-echo prefix: all of them are cmd's `cd`, none
            // of them is the verb key the tracking reads.
            "cd..",
            r"cd\Users",
            "cd/d C:\\ws",
            "@cd C:\\ws",
            "chdir..",
            "> out.txt cd src",
            "cd C:ws",
        ] {
            let refusal = build_at(ROOT, &[shell(member, "&&"), served(ROOT, "")])
                .expect_err("a directory change the tracking cannot follow");
            let Refusal::Shape(reason) = refusal else {
                panic!("{member}: expected the shape refusal: {refusal}");
            };
            assert!(reason.contains("`cd` member"), "{member}: {reason}");
            assert!(reason.contains(member), "{member}: {reason}");
        }
        for members in [
            vec![shell("cd src", "|"), served(ROOT, "")],
            vec![shell("dir", "|"), shell("cd src", "&&"), served(ROOT, "")],
        ] {
            let refusal = build_at(ROOT, &members).expect_err("a `cd` a pipeline owns");
            let Refusal::Shape(reason) = refusal else {
                panic!("expected the shape refusal: {refusal}");
            };
            assert!(reason.contains("side of a pipe"), "{reason}");
        }
        for members in [
            vec![served(ROOT, "&&"), shell("cd..", "")],
            vec![served(ROOT, "|"), shell("cd..", "")],
            vec![shell("cd..", "&&"), shell("echo done", "")],
            vec![
                shell("cd..", "&&"),
                shell("echo hi", "&&"),
                shell("echo done", ""),
            ],
            vec![shell("if exist src cd src", "&&"), shell("echo done", "")],
        ] {
            assert!(!plan(&members).steps.is_empty(), "{members:?}");
        }
        for members in [
            vec![shell("cd..", "&&"), shell("mahbot -V", "")],
            vec![
                shell("echo hi", "&&"),
                shell("cd..", "&&"),
                shell("mahbot -V", ""),
            ],
        ] {
            assert!(build_at(ROOT, &members).is_err(), "{members:?}");
        }
        // A directory change inside one of cmd's keyword forms: whether it runs at all is the
        // keyword's condition to decide, so the directory the line is left in is unknown wherever
        // a later step would need it.
        for member in [
            "if exist src cd src",
            "if not exist x cd sub",
            "else cd sub",
            "for %f in (*) do cd sub",
            "if exist x (cd sub)",
        ] {
            let refusal = build_at(ROOT, &[shell(member, "&&"), served(ROOT, "")])
                .expect_err("a `cd` a keyword form owns");
            let Refusal::Shape(reason) = refusal else {
                panic!("{member}: expected the shape refusal: {refusal}");
            };
            assert!(reason.contains("keyword form"), "{member}: {reason}");
        }
        for (member, tracked) in [
            ("cd src", r"C:\ws\src"),
            ("chdir src", r"C:\ws\src"),
            ("CD \\ws", r"C:\ws"),
        ] {
            let plan = plan(&[shell(member, "&&"), served(tracked, "")]);
            assert_eq!(plan.steps[1].cwd, PathBuf::from(tracked), "{member}");
        }
    }

    /// The redirect set the runner applies by itself, and the spellings it refuses by name.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the closed set
    fn redirects_supported_is_the_closed_set_it_documents() {
        struct RedirectCase {
            name: &'static str,
            command: &'static [&'static str],
            refused: bool,
        }
        let cases = [
            RedirectCase {
                name: "stdout to a file",
                command: &[">", "out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "fd 1 to a file",
                command: &["1>", "out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stdout append",
                command: &[">>", "out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "fd 1 append",
                command: &["1>>", "out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stderr to a file",
                command: &["2>", "err.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stderr append",
                command: &["2>>", "err.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stdin from a file",
                command: &["<", "in.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stdout to the lowercase nul device",
                command: &[">", "nul"],
                refused: false,
            },
            RedirectCase {
                name: "stderr to the uppercase NUL device",
                command: &["2>", "NUL"],
                refused: false,
            },
            RedirectCase {
                name: "all three streams at once",
                command: &[">", "out.txt", "2>", "err.txt", "<", "in.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stdout glued to its target",
                command: &[">out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "fd 1 append glued to its target",
                command: &["1>>log.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stderr glued to its target",
                command: &["2>err.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stderr append glued to its target",
                command: &["2>>err.txt"],
                refused: false,
            },
            RedirectCase {
                name: "stdin glued to its target",
                command: &["<in.txt"],
                refused: false,
            },
            RedirectCase {
                name: "fd 1 to fd 0",
                command: &["1>0"],
                refused: false,
            },
            RedirectCase {
                name: "a drive-rooted target",
                command: &[">", r"C:\out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "a quoted drive-rooted target",
                command: &[">", r#""C:\out.txt""#],
                refused: false,
            },
            RedirectCase {
                name: "the stdout merge alone",
                command: &["2>&1"],
                refused: false,
            },
            RedirectCase {
                name: "the stdout merge beside a redirect",
                command: &[">", "out.txt", "2>&1"],
                refused: false,
            },
            RedirectCase {
                name: "the stdout merge before a redirect",
                command: &["2>&1", ">", "out.txt"],
                refused: false,
            },
            RedirectCase {
                name: "fd 1 duplicated to stderr",
                command: &["1>&2"],
                refused: true,
            },
            RedirectCase {
                name: "a bare stderr dup",
                command: &[">&2"],
                refused: true,
            },
            RedirectCase {
                name: "an append with no target",
                command: &[">>"],
                refused: true,
            },
            RedirectCase {
                name: "an input redirect with no target",
                command: &["<"],
                refused: true,
            },
            RedirectCase {
                name: "fd 10 is outside the set",
                command: &["10>", "out.txt"],
                refused: true,
            },
            RedirectCase {
                name: "a redirect with no target",
                command: &[">"],
                refused: true,
            },
            RedirectCase {
                name: "stdin duplicated from stderr",
                command: &["<&2"],
                refused: true,
            },
            RedirectCase {
                name: "a read-write redirect",
                command: &["<>file"],
                refused: true,
            },
            RedirectCase {
                name: "a percent-expanded target",
                command: &[">", r"%TEMP%\out.txt"],
                refused: true,
            },
            RedirectCase {
                name: "a glued percent-expanded target",
                command: &[r">%TEMP%\out.txt"],
                refused: true,
            },
            RedirectCase {
                name: "a drive-relative target",
                command: &[">", "C:out.txt"],
                refused: true,
            },
            RedirectCase {
                name: "a glued drive-relative target",
                command: &[">C:out.txt"],
                refused: true,
            },
            RedirectCase {
                name: "a quoted drive-relative target",
                command: &[">", r#""C:out.txt""#],
                refused: true,
            },
            RedirectCase {
                name: "a glued quoted drive-relative target",
                command: &[r#">"C:out.txt""#],
                refused: true,
            },
            // Quote removal can sit between the colon and the path (`C:"out.txt`).
            RedirectCase {
                name: "a drive-relative target split by quote removal",
                command: &[">", r#""C:""out.txt""#],
                refused: true,
            },
            RedirectCase {
                name: "a drive-relative stdin target",
                command: &["<", "C:in.txt"],
                refused: true,
            },
            RedirectCase {
                name: "fd 3 duplicated to stdout",
                command: &["3>&1"],
                refused: true,
            },
        ];
        for case in cases {
            let redirects: Vec<String> = case.command.iter().map(|w| (*w).to_string()).collect();
            let outcome = redirects_supported(&redirects);
            if case.refused {
                let reason = outcome.expect_err("outside the closed set");
                assert!(
                    reason.contains(&format!("`{}`", case.command[0])),
                    "{}: {:?}: {reason}",
                    case.name,
                    case.command
                );
            } else {
                assert_eq!(outcome, Ok(()), "{}: {:?}", case.name, case.command);
            }
        }
    }

    /// The own-image reading: command position and a launcher count, argument position does not.
    /// A path with a space is one word only where the line quotes it — cmd.exe splits the bare
    /// spelling, so the bare one names no program (and is deliberately not read as this image).
    #[test]
    fn own_image_in_text_reads_the_command_position() {
        let exe = Path::new(EXE);
        for names in [
            "mahbot -V",
            "MAHBOT debug --db board",
            r#""C:\Program Files\MahBot\mahbot.exe" chrome x"#,
            r#""\\?\C:\Program Files\MahBot\mahbot.exe" -V"#,
            r#""C:/Program Files/MahBot/mahbot.exe" -V"#,
            "start mahbot -V",
            "call mahbot.exe -V",
            "cmd /c mahbot debug",
            "powershell -c mahbot",
            "pwsh mahbot.exe -V",
            r".\mahbot.exe -V",
            r"target\debug\mahbot.exe -V",
            r"C:\somewhere\else\mahbot -V",
            r"start .\mahbot.exe",
            "cd sub && mahbot -V",
            "type f.txt | mahbot -V",
            "echo hi & mahbot -V",
            "@mahbot -V",
            "(mahbot -V)",
            "( mahbot -V )",
            "(cd x & mahbot -V)",
        ] {
            assert!(own_image_in_text(names, exe), "{names}");
        }
        for text in [
            "grep -rn mahbot .",
            "grep -n mahbot.exe f.txt",
            "echo mahbot",
            "type mahbot.exe",
            "copy mahbot.exe back\\",
            "where mahbot",
            r"C:\Program Files\MahBot\mahbot.exe chrome x",
            // Another file's name, however it is spelled.
            r".\grep.exe -V",
            r"target\debug\other.exe -V",
            r#"echo "&&" mahbot"#,
            "echo ( mahbot -V )",
            "echo @ mahbot -V",
            // A word a launcher only *carries* is an ordinary argument: `cmd /c dir mahbot` lists
            // the directory, `powershell Get-Process mahbot` asks for a process, `start` hands its
            // command to `notepad`.
            "cmd /c dir mahbot",
            "cmd /c rd /s /q mahbot",
            "cmd /d /c echo mahbot",
            "powershell Get-Process mahbot",
            "powershell -c Get-Process mahbot",
            "start /wait notepad",
            r#"start "" notepad mahbot"#,
        ] {
            assert!(!own_image_in_text(text, exe), "{text}");
        }
    }

    /// A line that is one own-image call becomes a one-step plan, with the member's argv and
    /// redirect tokens.
    #[test]
    fn own_image_plan_reads_a_lone_call() {
        let exe = Path::new(EXE);
        let cwd = Path::new(ROOT);
        let OwnImage::Direct(direct) = own_image_plan(
            Executed(r#""C:\Program Files\MahBot\mahbot.exe" debug --db board "select 1""#),
            Written(r#""C:\Program Files\MahBot\mahbot.exe" debug --db board "select 1""#),
            exe,
            cwd,
        ) else {
            panic!("a lone own-image call is run by the runner");
        };
        let Some((args, redirects, step_cwd)) = direct.direct_own() else {
            panic!("a direct plan is one own step");
        };
        assert_eq!(args, &["debug", "--db", "board", "select 1"]);
        assert!(redirects.is_empty());
        assert_eq!(step_cwd, tracked_root_of(ROOT));

        let OwnImage::Direct(redirected) = own_image_plan(
            Executed(r#""C:\Program Files\MahBot\mahbot.exe" -V > out.txt"#),
            Written(r#""C:\Program Files\MahBot\mahbot.exe" -V > out.txt"#),
            exe,
            cwd,
        ) else {
            panic!("a redirected lone call is run by the runner");
        };
        let Some((args, redirects, _)) = redirected.direct_own() else {
            panic!("a direct plan is one own step");
        };
        assert_eq!(args, &["-V"]);
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);

        // The glued spelling of the same redirect is the same call, and the one an agent types.
        let OwnImage::Direct(glued) = own_image_plan(
            Executed(r#""C:\Program Files\MahBot\mahbot.exe" -V >out.txt"#),
            Written(r#""C:\Program Files\MahBot\mahbot.exe" -V >out.txt"#),
            exe,
            cwd,
        ) else {
            panic!("a redirected lone call is run by the runner");
        };
        let Some((args, redirects, _)) = glued.direct_own() else {
            panic!("a direct plan is one own step");
        };
        assert_eq!(args, &["-V"]);
        assert_eq!(redirects, &[">out.txt".to_string()]);
    }

    /// A line that names the image among other members is decomposed with the analyzer's own
    /// cmd.exe model: every plain call of the image is an own step, every other member one the
    /// folds around it run, and the cwd is tracked across the members.
    #[test]
    fn own_image_plan_decomposes_a_compound_line() {
        let root = tracked_root_of(ROOT);
        let sub = root.join("sub");
        let plan = direct_plan(&format!(r#"cd sub && "{EXE}" --version && echo done"#));
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(shell_text(&plan.steps[0]), "cd sub");
        assert_eq!(shell_text(&plan.steps[2]), "echo done");
        assert_eq!(plan.steps[2].join, Join::And);
        assert_eq!(
            step_cwds(&plan),
            vec![root.as_path(), sub.as_path(), sub.as_path()]
        );
        let Run::Own { args, .. } = &plan.steps[1].run else {
            panic!("the call of the image is an own step");
        };
        assert_eq!(args, &["--version"]);

        let plan = direct_plan(&format!(r#""{EXE}" -V | head -3"#));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::Pipe);
        assert_eq!(pipe_groups(&plan.steps), vec![0..2]);
        assert_eq!(shell_text(&plan.steps[1]), "head -3");
        assert_eq!(step_cwds(&plan), vec![root.as_path(), root.as_path()]);

        let plan = direct_plan(&format!(r#""{EXE}" debug --db board "SELECT 1" > out.txt"#));
        let Some((args, redirects, _)) = plan.direct_own() else {
            panic!("a lone call is one own step");
        };
        assert_eq!(args, &["debug", "--db", "board", "SELECT 1"]);
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);

        let plan = direct_plan(&format!(r#""{EXE}" -V 2>&1 | head -2"#));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::Pipe);
        let Run::Own { args, redirects } = &plan.steps[0].run else {
            panic!("the call of the image is an own step");
        };
        assert_eq!(args, &["-V"]);
        assert_eq!(redirects, &["2>&1".to_string()]);
    }

    /// A shape the generic decomposition cannot represent is refused through the plan's own
    /// refusals, never handed to the shell.
    #[test]
    fn an_unrepresentable_own_image_line_is_refused() {
        // A launcher spelling: the member names the image, but not in a command position the
        // runner can spawn.
        for command in ["start mahbot -V", "cmd /c mahbot bench-openrouter"] {
            let cause = refusal_cause(command);
            assert!(cause.contains("runs `mahbot` itself"), "{command}: {cause}");
        }
        // A dup the runner has no destination for, on a member it spawns itself.
        let cause = refusal_cause("mahbot -V >&2");
        assert!(cause.contains("`>&2`"), "{cause}");
        // A `cd` the tracking cannot follow: the call after it would run in a directory the shell
        // would not have used.
        let cause = refusal_cause("cd /x && mahbot -V");
        assert!(cause.contains("`cd` member"), "{cause}");
        // A `%…%` argument cmd.exe would have expanded before the image saw it: not one plain
        // call, so refused.
        let cause = refusal_cause("mahbot -V %TEMP%");
        assert!(cause.contains("runs `mahbot` itself"), "{cause}");
        // A call the line *opens* with in a line the segmenter refused, whose text carries no
        // spelling of the image's path: the bare name is read in command position, so the call is
        // refused rather than left to the shell.
        let cause = refusal_cause("mahbot -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        // The same line with the call further along it: the reading walks the connectors the
        // tokenizer still sees next to the words a command can start at — and the path spelling
        // catches what it cannot see.
        let cause = refusal_cause("cd sub && mahbot -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        let cause = refusal_cause(r"cd sub && C:\Program Files\MahBot\mahbot.exe -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        // cmd.exe's own punctuation does not hide a call: the `@` no-echo prefix
        // (whose launcher form hands the image on) and a group's parentheses, in
        // both spellings the model can be given.
        for command in ["@start mahbot -V", "@cmd /c mahbot -V"] {
            let cause = refusal_cause(command);
            assert!(cause.contains("runs `mahbot` itself"), "{command}: {cause}");
        }
        for command in [
            "(mahbot -V)",
            "( mahbot -V )",
            "mahbot -V )",
            "(cd x & mahbot -V)",
        ] {
            let cause = refusal_cause(command);
            assert!(cause.contains("cannot be read"), "{command}: {cause}");
        }
    }

    /// Every line that names no own image is left exactly as the shell left it.
    #[test]
    fn a_line_that_names_no_own_image_is_the_shells() {
        let exe = Path::new(EXE);
        let cwd = Path::new(ROOT);
        for command in [
            "grep -rn mahbot .",
            "echo mahbot > out.txt",
            "type mahbot.exe",
            "echo hi && echo done",
            // An unreadable line that only *mentions* the name: the command
            // position is `echo`'s, so the line stays the shell's.
            "echo (x) mahbot",
            "echo @ mahbot -V",
            // The residual, pinned: a `mahbot` call this model reads but never in
            // command position is the shell's, because recognising the bare name
            // wherever it merely *stands* would refuse ordinary text.
            "for /f %i in ('mahbot -V') do echo x",
        ] {
            assert!(
                matches!(
                    own_image_plan(Executed(command), Written(command), exe, cwd),
                    OwnImage::None
                ),
                "{command}"
            );
        }
        // On the platform whose shell waits for its children there is nothing to
        // plan and nothing to refuse: the gate is [`plan_for_command`]'s, the one
        // dispatch the callers use.
        if crate::tools::shell::SHELL_PLATFORM == ShellPlatform::Unix {
            assert!(matches!(
                plan_for_command(
                    Executed("mahbot -V && echo done"),
                    Written("mahbot -V && echo done"),
                    cwd
                ),
                OwnImage::None
            ));
        }
    }

    /// The cwd every member carries, for the line the analyzer plans and for the
    /// same line the runner tracks itself: a fold starts in the cwd of the member
    /// `cd src && grep -rn x . | head -3`, as the analyzer hands it over: the
    /// tail's fold starts in the directory the search tracked, not in the
    /// line's own cwd.
    #[test]
    fn the_cwd_a_fold_starts_in_is_the_members_own() {
        let root = tracked_root_of(ROOT);
        let plan = plan(&[
            shell("cd src", "&&"),
            served(r"C:\ws\src", "|"),
            shell("head -3", ""),
        ]);
        assert_eq!(
            step_cwds(&plan),
            vec![
                root.as_path(),
                Path::new(r"C:\ws\src"),
                Path::new(r"C:\ws\src")
            ]
        );
        // ... which is the same directory the runner tracks for the same line
        // spelled with this service's own call: the two readings agree member for member.
        let generic = direct_plan(&format!(r#"cd src && "{EXE}" -V | head -3"#));
        assert_eq!(generic.steps.len(), plan.steps.len());
        assert_eq!(generic.steps[1].cwd, generic.steps[2].cwd);
        assert!(generic.steps[2].cwd.ends_with("src"));
    }

    /// `steps` as a plan of shell members, each running `sh -c "<text>"`.
    #[cfg(unix)]
    fn shell_plan(steps: &[(&str, Join)]) -> Plan {
        Plan {
            steps: steps
                .iter()
                .map(|(text, join)| Step {
                    run: Run::Shell {
                        text: (*text).to_string(),
                    },
                    cwd: std::env::temp_dir(),
                    join: *join,
                })
                .collect(),
            exe: PathBuf::from(EXE),
        }
    }

    #[cfg(unix)]
    async fn run_plan(plan: &Plan) -> ShellRunResult {
        run(
            plan,
            Duration::from_secs(20),
            Duration::from_secs(5),
            None,
            RunOwner::Agent,
        )
        .await
    }

    /// A group runs only when cmd's connector says it should; a skipped group leaves the
    /// previous status in place.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_group_runs_only_when_its_connector_says_so() {
        let skipped = run_plan(&shell_plan(&[
            ("exit 3", Join::First),
            ("echo never", Join::And),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = skipped else {
            panic!("a completed run");
        };
        assert!(stdout.is_empty(), "the `&&` group must not run");
        assert_eq!(status.code(), Some(3), "the failed group's status stays");

        let ran = run_plan(&shell_plan(&[
            ("exit 3", Join::First),
            ("echo after-failure", Join::Or),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = ran else {
            panic!("a completed run");
        };
        assert_eq!(String::from_utf8_lossy(&stdout), "after-failure\n");
        assert_eq!(status.code(), Some(0), "the last member's status");
    }

    /// Two members of one pipe group are connected by a pipe the runner owns: the producer's
    /// whole stream reaches the consumer, and the run reports the consumer's status.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_pipe_group_connects_its_members() {
        let result = run_plan(&shell_plan(&[
            ("printf 'a\\nb\\nc\\n'", Join::First),
            ("head -2", Join::Pipe),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = result else {
            panic!("a completed run");
        };
        assert_eq!(String::from_utf8_lossy(&stdout), "a\nb\n");
        assert_eq!(status.code(), Some(0));
    }

    /// The pipe's input is the head's own stdout alone: the head runs as a step of its own, so
    /// what the group before it printed stays in the run's capture. A fold around the head would
    /// have fed the consumer `hi` first, and `head -1` would have printed that.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_pipelines_input_is_its_heads_own_stdout() {
        let result = run_plan(&shell_plan(&[
            ("echo hi", Join::First),
            (r"printf 'x\ny\n'", Join::And),
            ("head -1", Join::Pipe),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, .. } = result else {
            panic!("a completed run");
        };
        // The head's own group prints its line; the consumer's line is `x`.
        assert_eq!(String::from_utf8_lossy(&stdout), "hi\nx\n");
    }

    /// A pipeline is gated by its head's connector: an unsatisfied `&&` before the head skips the
    /// whole group and leaves the run on the group before it. A fold around the head would have run
    /// the pipeline anyway and reported its last member's status.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsatisfied_head_takes_its_pipeline_with_it() {
        let result = run_plan(&shell_plan(&[
            ("exit 3", Join::First),
            ("echo x", Join::And),
            ("cat", Join::Pipe),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = result else {
            panic!("a completed run");
        };
        assert!(stdout.is_empty(), "the `&&` group must not run");
        assert_eq!(status.code(), Some(3), "the failed group's status stays");
    }

    /// The run's streams are its members' output in execution order, and the status is the last
    /// executed group's last member's.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_run_reports_the_last_members_status_and_concatenated_output() {
        let result = run_plan(&shell_plan(&[
            ("echo one", Join::First),
            ("echo two; exit 4", Join::Always),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = result else {
            panic!("a completed run");
        };
        assert_eq!(String::from_utf8_lossy(&stdout), "one\ntwo\n");
        assert_eq!(status.code(), Some(4));
    }

    /// `steps` as a plan of own-image steps running `sh` — one host program spawned as argv, the
    /// way an own step spawns this service's image — so the redirect and pipe wiring of an own
    /// step is driven here too.
    #[cfg(unix)]
    fn program_plan(steps: &[(&[&str], &[&str], Join)]) -> Plan {
        Plan {
            steps: steps
                .iter()
                .map(|(args, redirects, join)| Step {
                    run: Run::Own {
                        args: args.iter().map(|word| (*word).to_string()).collect(),
                        redirects: redirects.iter().map(|word| (*word).to_string()).collect(),
                    },
                    cwd: std::env::temp_dir(),
                    join: *join,
                })
                .collect(),
            exe: PathBuf::from("/bin/sh"),
        }
    }

    /// `2>&1` on a member whose stdout the run captures: the member's stderr is its stdout
    /// destination, so its bytes join the captured stdout and nothing is left on stderr.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_merged_stderr_joins_the_captured_stdout() {
        let result = run_plan(&program_plan(&[(
            &["-c", "echo out; echo err >&2"],
            &["2>&1"],
            Join::First,
        )]))
        .await;
        let ShellRunResult::Completed { stdout, stderr, .. } = result else {
            panic!("a completed run");
        };
        // Per reader: the member's stdout, then the stderr the merge added beside
        // it (the module docs' residuals).
        assert_eq!(String::from_utf8_lossy(&stdout), "out\nerr\n");
        assert!(
            stderr.is_empty(),
            "a merged stderr has no stderr channel: {}",
            String::from_utf8_lossy(&stderr)
        );
    }

    /// `2>&1` on a member of a pipe group: both streams are pumped into the next member's input
    /// through one shared handle, and the consumer sees the end of that input when the last pump
    /// drops it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_merged_stderr_reaches_the_pipelines_consumer() {
        let result = run_plan(&program_plan(&[
            (&["-c", "echo out; echo err >&2"], &["2>&1"], Join::First),
            (&["-c", "cat"], &[], Join::Pipe),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, .. } = result else {
            panic!("a completed run");
        };
        // The two streams arrive whole, in whatever order the pumps ran.
        let captured = String::from_utf8_lossy(&stdout);
        let mut lines: Vec<&str> = captured.lines().collect();
        lines.sort_unstable();
        assert_eq!(lines, ["err", "out"]);
    }

    /// `> out.txt 2>&1`: the merge follows the stdout the member's own `>` left, so both streams
    /// are the one file, not the run's capture the merge would have taken had it come first.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_merge_after_a_redirect_takes_that_redirects_destination() {
        let dir = std::env::temp_dir();
        let target = dir.join(format!("plan-merge-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&target);
        let result = run_plan(&program_plan(&[(
            &["-c", "echo out; echo err >&2"],
            &[">", &target.to_string_lossy(), "2>&1"],
            Join::First,
        )]))
        .await;
        let ShellRunResult::Completed { stdout, .. } = result else {
            panic!("a completed run");
        };
        assert!(
            stdout.is_empty(),
            "the member's own redirect took its streams: {}",
            String::from_utf8_lossy(&stdout)
        );
        let written = std::fs::read_to_string(&target).expect("the redirect's file");
        let mut lines: Vec<&str> = written.lines().collect();
        lines.sort_unstable();
        assert_eq!(lines, ["err", "out"]);
        let _ = std::fs::remove_file(&target);
    }

    /// `2> err.txt 2>&1`: the merge comes later, so it re-points stderr at stdout's destination —
    /// the caller's capture — exactly as cmd.exe's left-to-right application does. The file the
    /// earlier redirect opened was created and truncated before the merge took it away.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_merge_after_a_redirect_overrides_it() {
        let dir = std::env::temp_dir();
        let target = dir.join(format!("plan-override-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&target);
        let result = run_plan(&program_plan(&[(
            &["-c", "echo out; echo err >&2"],
            &["2>", &target.to_string_lossy(), "2>&1"],
            Join::First,
        )]))
        .await;
        let ShellRunResult::Completed { stdout, stderr, .. } = result else {
            panic!("a completed run");
        };
        // Both streams arrive in the run's stdout, in the order the readers were
        // created (the module docs' residuals).
        assert_eq!(String::from_utf8_lossy(&stdout), "out\nerr\n");
        assert!(
            stderr.is_empty(),
            "the later merge left no stderr channel: {}",
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(
            std::fs::read_to_string(&target).expect("the redirect's file"),
            ""
        );
        let _ = std::fs::remove_file(&target);
    }

    /// A member that floods stdout cannot truncate another member's stderr: the two
    /// streams have a budget each, the split the single-child runner's two pipes
    /// have. The flood is written before the marker, so the stdout budget is spent
    /// by the time the marker arrives — a shared budget would lose it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_flooded_stdout_does_not_truncate_stderr() {
        let result = run_plan(&program_plan(&[(
            &["-c", "head -c 1048576 /dev/zero; echo marker >&2"],
            &[],
            Join::First,
        )]))
        .await;
        let ShellRunResult::Completed { stdout, stderr, .. } = result else {
            panic!("a completed run");
        };
        assert_eq!(
            stdout.len(),
            SHELL_PIPE_READ_CAP + SHELL_PIPE_TAIL_CAP + TRUNCATED_MIDDLE.len(),
            "the head is the cap and the rest is kept as the stream's end"
        );
        assert_eq!(String::from_utf8_lossy(&stderr), "marker\n");
    }

    /// `2>&1 > out.txt` on a member of a pipe group: the merge came first, so the
    /// pipe was already the stderr's destination when the `>` took stdout over —
    /// the consumer reads the merged stream while the member's stdout goes to the file.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_merge_before_a_redirect_still_feeds_the_pipe() {
        let dir = std::env::temp_dir();
        let target = dir.join(format!("plan-merge-pipe-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&target);
        let result = run_plan(&program_plan(&[
            (
                &["-c", "echo out; echo err >&2"],
                &["2>&1", ">", &target.to_string_lossy()],
                Join::First,
            ),
            (&["-c", "cat"], &[], Join::Pipe),
        ]))
        .await;
        let ShellRunResult::Completed { stdout, .. } = result else {
            panic!("a completed run");
        };
        // The consumer prints what the pipe carried — the producer's merged stderr,
        // not the stdout its own `>` took.
        assert_eq!(String::from_utf8_lossy(&stdout), "err\n");
        assert_eq!(
            std::fs::read_to_string(&target).expect("the redirect's file"),
            "out\n"
        );
        let _ = std::fs::remove_file(&target);
    }

    /// A run past its deadline ends the whole run and reports a `TimedOut` naming
    /// the group it waited for — with what the members had already written still collected.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_plan_past_its_deadline_is_stopped_and_named() {
        let result = run(
            &shell_plan(&[("echo started; sleep 30", Join::First)]),
            Duration::from_secs(1),
            Duration::from_secs(5),
            None,
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::TimedOut {
            stdout,
            pid,
            elapsed,
            ..
        } = result
        else {
            panic!("expected a timeout, got {result:?}");
        };
        assert!(pid.is_some(), "the group the run waited for is named");
        assert!(
            elapsed < Duration::from_secs(20),
            "the watchdog must not wait out the member: {elapsed:?}"
        );
        assert!(
            String::from_utf8_lossy(&stdout).contains("started"),
            "the member's output up to the stop is collected: {}",
            String::from_utf8_lossy(&stdout)
        );
    }

    /// A run over its memory ceiling is ended like a timeout and reported as a
    /// memory failure of its own — never as one that ran out of time. The memory is
    /// held by the member itself (a shell variable), and the ceiling is injected.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_plan_over_the_memory_ceiling_is_stopped() {
        /// Small enough that the member below crosses it long before the test
        /// machine's own limit could be relevant.
        const CEILING: u64 = 16 * 1024 * 1024;
        /// What the member holds — 64 MiB, four times the ceiling.
        const PAYLOAD: usize = 64 * 1024 * 1024;

        let result = run(
            &shell_plan(&[(
                &format!("x=$(yes x | head -c {PAYLOAD}); sleep 30"),
                Join::First,
            )]),
            Duration::from_secs(30),
            Duration::from_secs(5),
            Some(CEILING),
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::MemoryExceeded {
            used,
            limit,
            elapsed,
            ..
        } = result
        else {
            panic!("expected a memory failure, got {result:?}");
        };
        assert_eq!(limit, CEILING);
        assert!(
            used > limit,
            "the tripping sample exceeds the ceiling: {used}"
        );
        assert!(
            elapsed < Duration::from_secs(20),
            "the watchdog must not wait out the member: {elapsed:?}"
        );
    }

    /// A member that leaves a descendant holding the capture pipe open past the drain bound is
    /// stopped with the run's tree: the last member's status and the output collected up to the
    /// detection are the result, the split the single-child runner gives — and the descendant
    /// does not survive it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_leftover_holder_past_the_drain_bound_is_stopped() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let pid_path = tmp.path().join("leftover.pid");
        let command = format!("echo out; sleep 999 & echo $! > {}", pid_path.display());
        let result = run(
            &shell_plan(&[(command.as_str(), Join::First)]),
            Duration::from_secs(20),
            Duration::from_millis(300),
            None,
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::Completed { stdout, status, .. } = result else {
            panic!("expected Completed, got {result:?}");
        };
        assert_eq!(status.code(), Some(0), "the member's own status");
        assert!(
            String::from_utf8_lossy(&stdout).contains("out"),
            "what the member wrote before the bound is collected: {}",
            String::from_utf8_lossy(&stdout)
        );
        // The descendant that held the pipe is gone with the rest of the tree.
        let pid: i32 = std::fs::read_to_string(&pid_path)
            .expect("the member wrote the descendant's pid")
            .trim()
            .parse()
            .expect("valid pid file");
        crate::util::test::wait_for_death(pid).await;
    }

    /// A member that cannot be spawned fails the run as `SpawnFailed` — the single-child runner's
    /// variant — and ends the tree rather than leaving what was running behind.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_member_that_cannot_be_spawned_fails_the_run() {
        let mut plan = shell_plan(&[("echo never", Join::First)]);
        plan.steps[0].cwd = PathBuf::from("/nonexistent-plan-cwd");
        let result = run_plan(&plan).await;
        let ShellRunResult::SpawnFailed(e) = result else {
            panic!("expected a spawn failure, got {result:?}");
        };
        assert!(!e.to_string().is_empty(), "the failure names its cause");
    }

    /// A redirect the runner cannot open for a later member of a group ends the group exactly as a
    /// failed spawn does: the members already spawned are killed and reaped.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_redirect_that_cannot_be_opened_ends_the_group() {
        let dir = std::env::temp_dir();
        let marker = dir.join(format!("plan-cleanup-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let target = dir.join(format!("plan-missing-{}/out.txt", std::process::id()));
        let result = run_plan(&program_plan(&[
            // A member that would write its marker a second from now: the marker's absence is
            // the kill having reached it.
            (
                &["-c", &format!("sleep 1; echo done > {}", marker.display())],
                &[],
                Join::First,
            ),
            // Its own redirect names a directory that does not exist, so the runner cannot open
            // it — while the member before it is already running, the pipe being their connect.
            (
                &["-c", "cat"],
                &[">", &target.to_string_lossy()],
                Join::Pipe,
            ),
        ]))
        .await;
        let ShellRunResult::SpawnFailed(_) = result else {
            panic!("expected a spawn failure, got {result:?}");
        };
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(
            !marker.exists(),
            "the group's earlier member must be killed, not left running"
        );
    }
}

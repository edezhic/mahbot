//! The runner's own decomposition of a rewritten command line.
//!
//! # The guarantee
//!
//! Every step that runs this service's own image is spawned — and waited for — by
//! the runner itself. This service is built as a **windowed** program
//! (`windows_subsystem = "windows"`), and the platform documents such an image as
//! neither allocated a console nor blocking the shell that starts it: a
//! `cmd.exe /C "… mahbot.exe …"` line returns as soon as cmd.exe has *started* the
//! image, so the shell's own status and whatever the capture held at that instant
//! would reach the agent as the command's answer — a truncated or empty result,
//! or a status that came from the interpreter. The product's file search is the
//! same image: its call is the one whose refusal, truncation or stale-helper
//! answer must never arrive looking like "no matches".
//!
//! Nothing on cmd.exe's side can change that: it is what a GUI-subsystem image
//! *is*. So the runner takes those members over — it spawns the image as argv,
//! waits for it, and reports its status — and the shell is left with only the
//! members it can genuinely be asked to run.
//!
//! # One line, two renderings
//!
//! The analyzer hands over the rewritten line's members — every served search as
//! the engine's argv, every other member as the text cmd.exe must run — and the
//! same per-member data is rendered twice: as the one command line the shell
//! would have run (`grep_engine::join_rewritten`) and as this plan. Both come
//! from one traversal, and the drift guard in the engine's Windows lane asserts
//! that the rendered text is the plan's members rejoined with their connectors: a
//! plan and a line that disagreed about a member would be a search whose output
//! came from somewhere the agent did not look.
//!
//! One divergence is deliberate and worth naming. A *bare-name* own-image member
//! — an agent's own `mahbot -V`, which the rewrite carries verbatim, since the
//! analyzer spells by the image's absolute path only the members it renders itself
//! (its served searches) — keeps the spelling the agent typed in the rendered line,
//! while its step is a [`Run::Own`] spawned from the image's absolute path
//! ([`Plan::exe`]).
//!
//! On Windows the rendered line is never executed — the plan is — so its cmd.exe
//! quoting exists for the two readers that do need it: the read-only guard, which
//! validates the line the shell tool hands it, and that drift guard.
//!
//! # Just enough cmd.exe
//!
//! A plan is a linear chain of steps, grouped into *pipe groups* — each a maximal
//! run of members joined by `|` — and each group is spawned and waited for on its
//! own:
//!
//! - consecutive verbatim members become ONE [`Run::Shell`] step: a single
//!   `cmd.exe /C "<text>"` whose text is those members rejoined by their own
//!   connectors. The fold is what keeps cmd.exe's own reading — its quoting,
//!   redirects, builtins and `%VAR%` expansion — *inside* a step, and it is what
//!   keeps `cd a && grep x .` searching the directory the line's `cd` chose (a
//!   cmd.exe per member would lose that cwd). The exception is the two ends of a
//!   `|`: both the member whose *own* following connector is `|` and the member it
//!   feeds are steps of their own, because cmd.exe pipes exactly the stdout of the
//!   command immediately before a `|`, gates the pipeline on that command's
//!   connector, and runs what a following `&&` reaches with the shell's stdin
//!   rather than the pipe's — folding either end would report a match cmd.exe never
//!   produced, run a pipeline whose gated producer failed, or leave `cat` in
//!   `grep x . | head -3 && cat` reading the search's leftovers;
//! - a member that is this service's own image becomes a [`Run::Own`] step,
//!   spawned as argv with no interpreter in between: nothing on the line can
//!   reinterpret an argument, and the status the run reports is that member's own.
//!   This is decided per member, by [`build`], whichever caller the member came
//!   from — a served search is one, and so is an agent's own `mahbot -V` that the
//!   analyzer left verbatim beside a search — which is what keeps a line that runs
//!   this service's program out of the shell's hands entirely;
//! - a group's members are connected by pipes the runner owns — one member
//!   spawned per member, with a task copying each member's stdout into the next
//!   member's stdin — so `grep … | head -3` is that pipeline and not the shell's.
//!   A consumer that exits drops the producer's next write exactly as a shell
//!   pipeline does (the engine's own broken-pipe handling is unchanged);
//! - a group runs only when cmd.exe's connector says it should ([`runs_after`]),
//!   and the run reports the LAST executed group's LAST member's status — cmd's
//!   own rule, and the one the tool's `[exit status: …]` annotation, the engine
//!   sentinel classification and the engine refusal marker read: for a
//!   `… | engine` pipeline that is the engine's own status, not the interpreter's.
//!   The step a group is gated by is its first — the pipeline's head, the member
//!   immediately before the group's first `|` — because the exception above makes
//!   that member a step of its own, so the connector read here is the head's own
//!   and not an earlier member's.
//!
//! # Where each step runs
//!
//! Every [`Step`] carries the directory it runs in, and [`build`] derives it from
//! the members themselves: the line's own `cd` members, read with the analyzer's
//! tracking ([`tracked_cd`], [`grep_engine::resolve_cd`] underneath), move the
//! directory the next step starts in. A served [`Member::Own`] carries the cwd its
//! own search tracked — the same value its spec file names — and a fold runs in the
//! directory the line was in where the fold opened, which is what lets a `cd`
//! before a search survive the fold boundary between the two. A `cd` inside a fold
//! is that fold's own reading: `cmd.exe /C` runs it there, so the members after it
//! inside the fold stay with that fold's cmd.exe, while the move is what the step
//! after the fold starts in.
//!
//! # Deliberately refused
//!
//! Where the runner cannot reproduce what the shell would have done, the shape is
//! refused *visibly* — the tool error [`refusal_message`] renders, never a partial
//! result or a status from somewhere else:
//!
//! - a redirection the runner cannot apply to a member it spawns itself
//!   ([`redirects_supported`]): a dup that is not the stdout merge (`>&2`), a
//!   combined-output spelling (`&>`), a read-write `<>`, multi-digit descriptors,
//!   a missing target, a `%…%`-carrying or drive-relative one. `2>&1` *is* applied — at
//!   the destination the member's stdout has, which is what cmd.exe's own merge
//!   leaves (see the residuals);
//! - a member that names this service's own image in a shape [`build`] cannot run
//!   as one plain call of it ([`own_image_reference`]): a launcher spelling
//!   (`start mahbot …`, `cmd /c mahbot …`), a word cmd.exe would have expanded
//!   before any program saw it (`mahbot -V %TEMP%`), and a spelling of the image's
//!   file name that is not the program the runner runs (`.\mahbot.exe -V`), plus
//!   any line the cmd.exe model cannot read while it names the image by that path
//!   ([`spelling_in_text`]) or where a command can start ([`own_image_in_text`] —
//!   the reading that also sees through the `@`/`(` punctuation a refused line
//!   carries). Every one of them is a call the shell does not wait for, so leaving
//!   it to the shell would hand the agent a truncated or empty answer;
//! - a fold whose shell state a later fold would have observed (the environment it
//!   sets, the directory stack it pushes): each fold is a fresh cmd.exe, and state
//!   that dies with its fold cannot be approximated;
//! - a `cd`-family member the cwd tracking cannot follow ([`tracked_cd`],
//!   [`keyword_cd`]): a spelling of cmd.exe's own family the model reads but cannot
//!   track (`cd..`, `cd.txt`, `cd/d`, `@cd`), a target it cannot resolve, any of them
//!   on one side of a pipe, and any of them inside one of the shell's own keyword
//!   forms (`if exist x cd sub`) — refused where a step of its own would afterwards
//!   run in the tracked directory, because cmd.exe keeps the move inside the fold that
//!   made it, and a step in the wrong directory is a wrong answer, not a missing one;
//! - a rewritten line whose members cannot be read as a chain.
//!
//! A line the serve decision did not plan — an agent running `mahbot debug …`,
//! `mahbot chrome …`, `mahbot bench-openrouter …`, `mahbot -V` — reaches the same
//! runner: [`own_image_plan`] hands its members to [`build`], which classifies them
//! with that same cmd.exe model, tracks the cwd across them and refuses the shapes
//! above. So `cd sub && mahbot -V && echo done` and `mahbot -V | head -3` are the
//! runner's too, and no caller has a second reading of what a member is.
//!
//! # Residuals
//!
//! What the guarantee covers is the command lines the shell tool and the
//! background sessions run. What is left to the shell, or not covered at all:
//!
//! - An own-image call spelled through something this model cannot read keeps the
//!   shell's behaviour, and the reading here is closed by decision rather than grown
//!   per spelling: a batch wrapper (`mahbot.bat`), a `%VAR%` expanding to the image's
//!   path, a launcher outside [`LAUNCHER_VERBS`] — and among the known launchers a
//!   command word this grammar does not resolve (a quoted call, `cmd /c "mahbot -V"`;
//!   one launcher handing to another, `cmd /c call mahbot -V`; a `start` switch that
//!   takes the word after it, `start /d C:\ws mahbot -V`). What the known launchers do
//!   resolve is refused instead, as is a call in a line the segmenter refused, read
//!   for the words a command can start at ([`own_image_in_text`]) and for the image's
//!   path spelling anywhere in the text ([`spelling_in_text`]).
//! - A script this product runs on its own behalf — a custom tool's payload, an
//!   owner's alarm program, a diagnostics step — is not a command line this runner
//!   sees: cmd.exe runs it, and a call of this service's own image inside it is left
//!   to the shell, with whatever a shell gives a windowed image.
//! - One class of bare-name call stays the shell's without being refused, and it is
//!   a windowed image the shell does not wait for: a `mahbot …` command this model
//!   reads but never in command position — the keyword forms cmd.exe runs a command
//!   from (`for … do`, `if`, `else`): `for /f %i in ('mahbot -V') do …`,
//!   `if exist x mahbot -V`, `else mahbot -V`, `for %f in (*) do mahbot -V` — and a
//!   call glued to a connector inside a text this model reads as one word, which is
//!   another command's quoted argument (`cmd /c "echo hi&&mahbot -V"`) or a line the
//!   segmenter refused (`ls (x)&&mahbot -V`). No position reading sees those, and
//!   refusing them would mean refusing the bare name wherever it merely *stands* —
//!   and with it every ordinary line that mentions it (`echo mahbot`,
//!   `grep -rn mahbot .`). Glued connectors *between* its members the segmenter does
//!   split, so `echo hi&&mahbot -V` is a line this runner takes.
//! - A command word that is the image's own *file name* (`mahbot`, `mahbot.exe`)
//!   counts as this service's own image — the identity cmd.exe would have handed the
//!   call, and the only reading that makes the guarantee hold for the spelling an
//!   agent types. No *other* spelling of the name is recognised, so a renamed image
//!   (a version-suffixed copy, a workspace build) is the shell's to run.
//! - An own step is spawned with the environment this runner hands every command
//!   ([`super::build_program_command`]'s — the owner's own, or the reduced
//!   fallback), while cmd.exe would have passed
//!   a fold's own `set` on to it: `set FOO=1 && mahbot …` runs the call without `FOO`
//!   — not detectable from the line, so stated rather than refused (the state-carrying
//!   shape a *later fold* would have observed is refused).
//! - The runner applies a member's redirection itself, so a target word is the
//!   runner's file and a `%…%` pair in one is refused rather than expanded (see the
//!   refusal list above); the engine's own argv refuses such words likewise.
//! - A member's `2>&1` is a copy at the member's stdout destination rather than
//!   cmd.exe's shared handle, so the two streams arrive whole per reader (in the
//!   order the readers were created) and interleave per chunk into the next member's
//!   input — neither is cmd.exe's write order. A served search's merged stderr (its
//!   refusal text, the stream-size marker when it is stdin-fed) lands in the captured
//!   stdout rather than the stderr channel the parent strips and scans: the text stays
//!   visible, and a search that is the last member still refuses through its exit
//!   status. That channel is scanned whole ([`super::engine_failure`]), so another
//!   member's failure turns the run into the engine's refusal with that member's
//!   message as the cause.
//! - The run's capture cap is one budget per stream ([`CaptureBudgets`]), the cap
//!   the single-child runner's two pipes had.
//! - A member's leftover descendants can hold a capture pipe open after the
//!   member exited: the post-run drain bound is the same one the single-child run
//!   gives, and its overrun is the same
//!   [`super::ShellRunResult::DrainTimedOut`].
//! - A stop ends the whole run's tree, not one member's: every member's process
//!   group on unix — the groups of members already reaped included, which is what
//!   reaches a descendant they left behind — and the run's own job on Windows
//!   ([`super::Tree`]). That is why a plan stop is a whole-run stop, and why the
//!   variants reporting a pid name the waiting group's first member rather than the
//!   member that finished last.
//! - A fold's stderr may interleave differently with another step's than it did
//!   inside one cmd process, and a member's stdout ordering is preserved by
//!   member rather than by line: each stream is assembled in execution order.
//!
//! # Reasoned, not measured
//!
//! No Windows host runs in this project's lane, so cmd.exe's reading of the
//! connectors and redirects modelled here is argued from its documentation, the
//! way the engine's own `windows` line model is. The plan's steps are plain data,
//! so the executor's mechanics are driven from any host with shell members, and
//! [`runs_after`] (the one decision with a host oracle: cmd's errorlevel) is pinned
//! there by the executor's connector test. What no host here can show is cmd.exe's
//! agreement with the fold text the runner hands it.

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
use super::{
    CaptureBudget, Captured, KillOnDrop, Readers, RunOwner, SHELL_PIPE_READ_CAP, ShellPlatform,
    ShellRunResult, Stream, Tree, Watchdog, build_program_command, build_shell_command,
};

/// A plan: the rewritten command line decomposed into the steps the runner
/// spawns, in execution order.
#[derive(Debug)]
pub(super) struct Plan {
    pub(super) steps: Vec<Step>,
    /// This service's own image — the program every [`Run::Own`] step runs, and
    /// the path [`own_image_plan`] compares a command's words against. One field
    /// rather than a lookup per step, so the image the plan's argv names and the
    /// image [`super::grep_engine`] rendered into the text cannot be told apart.
    pub(super) exe: PathBuf,
}

/// One step of a plan: one thing the runner spawns and waits for.
#[derive(Debug)]
pub(super) struct Step {
    pub(super) run: Run,
    /// The directory the step runs in (see the module docs' "Where each step
    /// runs"): a cmd.exe fold at the cwd its first member was tracked at, an
    /// own-image member at its own member's cwd.
    pub(super) cwd: PathBuf,
    /// How the step joins the one before it.
    pub(super) join: Join,
}

/// What one step runs.
#[derive(Debug)]
pub(super) enum Run {
    /// One `cmd.exe /C "<text>"` run — a maximal fold of adjacent verbatim
    /// members, their own connectors inside the text. A member that feeds a pipe
    /// is never folded, so its `cmd.exe` is its own (see the module docs' folding
    /// rule).
    Shell { text: String },
    /// A call of this service's own image, spawned directly by the runner.
    Own {
        /// The member's argv after the image path, as cmd.exe would have
        /// delivered it (the engine verb and its spec, or an agent's own words).
        args: Vec<String>,
        /// The member's redirect tokens, verbatim, as the shell's parser read
        /// them (`[">", "out.txt"]`).
        redirects: Vec<String>,
    },
}

/// How a step joins the one before it — cmd.exe's own connectors, plus the
/// pipeline link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Join {
    /// Nothing before it: the line's first step.
    First,
    /// `&&`: the step runs only when the group before it succeeded.
    And,
    /// `||`: the step runs only when the group before it failed.
    Or,
    /// `&` — and the newline connector cmd.exe reads the same way: always.
    Always,
    /// `|`: the step is one member of the pipeline the step before it feeds, so
    /// the two are members of one group rather than steps that follow each other.
    Pipe,
}

impl Join {
    /// The connector's spelling in the shell's own line: what the fold text
    /// joins its members with, and what the drift guard compares against the
    /// rewrite.
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

/// One member of a line, as the decomposition needs it: the text cmd.exe must
/// run for a verbatim member, or the argv the runner spawns the image with for a
/// member that runs it. This is the analyzer's per-member decision, not a
/// re-reading of the rewrite — the rendering walks the same list (see the module
/// docs' "One line, two renderings") — and [`own_image_plan`] walks a line the
/// serve decision did not plan into the same shape, so [`build`] is the one
/// folding rule for both.
///
/// Neither reading decides *what* a member is on its way in: an own-image call
/// arriving as [`Member::Shell`] is classified by [`build`] itself, which is what
/// makes the two callers agree about a line that runs this service's program
/// beside a search.
#[derive(Debug)]
pub(super) enum Member {
    /// A verbatim member: cmd.exe runs it, inside its fold — unless it turns out
    /// to be a plain call of this service's own image, which [`build`] converts
    /// into the runner's own step wherever it came from.
    Shell {
        /// The member's text, as cmd.exe must receive it.
        text: String,
    },
    /// A member that runs this service's own image for a reason its caller
    /// already decided: a served search, whose engine argv
    /// ([`grep_engine`]) the runner spawns itself.
    Own {
        /// The member's argv after the image path.
        argv: Vec<String>,
        /// The member's redirect tokens, verbatim.
        redirects: Vec<String>,
        /// The cwd the analyzer tracked for the member — authoritative for a
        /// search, whose spec file names the same directory.
        cwd: PathBuf,
    },
}

/// Why a plan could not be built. Every one of these is a *refusal*: the command
/// must not be handed to the shell either, because the shell cannot be waited on
/// for the member the refusal is about — the reason becomes the agent-facing tool
/// error, and no command runs.
#[derive(Debug)]
pub(super) enum Refusal {
    /// A member names this service's own image — in command position, through a
    /// launcher, or in a line the model cannot read — and the shell does not wait
    /// for it ([`own_image_reference`]).
    OwnImageInShell,
    /// A member's command-position word names the image's file name through a
    /// path the runner cannot match to the program it runs
    /// ([`names_image_file`]); the string is that spelling.
    OwnImageSpelling(String),
    /// The member cannot be run as it stands and the shell must not run it: its
    /// redirect tokens are ones the runner cannot apply to a member it spawns
    /// itself ([`redirects_supported`]), or the line's own shape cannot be
    /// decomposed into steps the runner can run (a fold whose state a later fold
    /// would have observed, a `cd` the tracking cannot follow). The string is the
    /// cause's own text.
    Shape(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The agent-facing causes name the program as the agent spelled it:
            // `mahbot` is the word the call was written with.
            Refusal::OwnImageInShell => write!(
                f,
                "the line also runs `mahbot` itself, which a shell does not wait for — \
                 such a call would come back truncated or empty"
            ),
            Refusal::OwnImageSpelling(spelling) => write!(
                f,
                "a shell member runs `{spelling}`, which names `mahbot`'s file name but \
                 is not the program the runner spawns — and a shell does not wait for \
                 such an image either"
            ),
            Refusal::Shape(reason) => write!(f, "{reason}"),
        }
    }
}

/// Build the plan for `members` — the line's members in order, each with the
/// connector that follows it (as the renderer walks them) — tracking the line's
/// own cwd from `root_cwd`.
///
/// The one reading of a line, for both callers: the analyzer's rewrite (whose
/// served searches arrive as [`Member::Own`]) and a line the serve decision did
/// not plan (whose members all arrive as [`Member::Shell`] and are classified
/// here). Folding, cwd tracking and the refusals are the module docs' story; the
/// three properties this function must keep are that consecutive verbatim members
/// become ONE [`Run::Shell`] step whose text is those members rejoined by their
/// connectors spelled the way the rewrite spells them — unless one of them feeds
/// a pipe, which is a step of its own (the module docs' folding rule) — that a
/// plain call of this service's own image is always an own step, whether it
/// arrived as one or had to be converted, and that a fold can never swallow such
/// a call.
///
/// Windows only, and by construction: this is the platform whose shell does not
/// wait for this service's own image, so it is the only one whose caller builds a
/// plan at all.
pub(super) fn build(
    members: &[(Member, String)],
    exe: &Path,
    root_cwd: &Path,
) -> Result<Plan, Refusal> {
    let mut steps: Vec<Step> = Vec::new();
    // The fold being accumulated: its text, the join its first member had and the
    // cwd its cmd.exe runs in. A fold is flushed when an own step or the end of
    // the line ends it.
    let mut fold: Option<Fold> = None;
    // Where the line's own `cd` members have left it: the directory every
    // own-image step runs in and every fold starts in. Read with the analyzer's
    // own tracking ([`tracked_cd`]), so the plan's cwd cannot disagree with the
    // directory the engine's spec files name.
    let mut tracked = root_cwd.to_path_buf();
    // The directory the tracking lost, when a member moved it in a spelling this
    // model cannot follow ([`tracked_cd`], [`keyword_cd`]): the refusal is kept
    // until a step of its own would need that directory, because a move cmd.exe
    // made inside a fold is not observable outside it.
    let mut untracked_cd: Option<String> = None;
    for (index, (member, _)) in members.iter().enumerate() {
        let join = join_before(members, index)?;
        // This member's OWN following connector: a `|` there means cmd.exe takes
        // this member's stdout as the pipe's input, which no fold may share (see
        // the module docs' folding rule).
        let feeds_pipe = members[index].1 == "|";
        // A step of its own is spawned in the tracked directory — an own member,
        // or a Shell member that opens a fold — so an untracked `cd` is refused
        // where one of them begins (a member inside the `cd`'s own fold has
        // cmd.exe's own state, and needs nothing of `tracked`).
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
                // The member's words, read once for every reading below.
                let words = member_words(text);
                // A member that is a plain call of this service's own image is not
                // the shell's to run, whatever its caller took it for: the shell
                // does not wait for the image, and the fold around it would hand
                // the call to exactly that shell. It becomes an own step in this
                // member's place, which is why the fold breaks here.
                if let Some((args, redirects)) = own_call(&words, exe) {
                    // An own step needs the tracked directory even when a fold is
                    // open around it (that fold is flushed below).
                    if let Some(cause) = &untracked_cd {
                        return Err(Refusal::Shape(cause.clone()));
                    }
                    let cwd = tracked.clone();
                    (args, redirects, cwd)
                } else {
                    // Not a plain call, but still a call the shell cannot be waited
                    // on for — a launcher spelling, a path that is not the program
                    // the runner runs, a word cmd.exe would have expanded before any
                    // program saw it. Refused, never handed over.
                    if let Some(refusal) = own_image_reference(&words, text, exe) {
                        return Err(refusal);
                    }
                    // A `cd` inside one of the shell's own keyword forms (`if exist
                    // x cd sub`, `for %f in (*) do cd sub`): whether it runs at all
                    // depends on the keyword's own condition, which this model does
                    // not evaluate, so which directory the line is left in is
                    // unknown from here on — noted, and refused wherever a later
                    // step of its own would need it.
                    if let Some(cause) = keyword_cd(&words, text) {
                        untracked_cd = Some(cause);
                    }
                    // The directory the line is in *before* this member's own `cd`
                    // runs: the fold this member may open starts there and runs the
                    // `cd` itself, so the members after the fold feel the move.
                    let fold_cwd = tracked.clone();
                    match tracked_cd(&words, text, &tracked, feeds_pipe || join == Join::Pipe) {
                        Ok(Some(moved)) => tracked = moved,
                        Ok(None) => {}
                        Err(cause) => untracked_cd = Some(cause),
                    }
                    // cmd.exe pipes exactly the stdout of the command immediately
                    // before a `|`, and gates the pipeline on that command's own
                    // connector — so both ends of a pipe are steps of their own,
                    // never folded with their neighbours: a fold around the head
                    // would send every earlier member's output through the pipe too
                    // and gate the group by the fold's first connector instead of
                    // by the head's, and a fold around the consumer would let what
                    // follows it in the line read the pipe cmd.exe gave the
                    // consumer alone (the fold's rule below).
                    if feeds_pipe {
                        flush_fold(&mut steps, &mut fold);
                    }
                    fold_push(&mut fold, text, join, fold_cwd);
                    // A member that *takes* a pipe ends its fold too: cmd.exe hands
                    // it the pipe's input and gives what follows it in cmd's own
                    // process — with the shell's stdin, not that pipe's — so
                    // `grep x . | head -3 && cat` must not leave `cat` reading the
                    // search's leftovers.
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
    // A fold is a fresh cmd.exe: state it sets dies with it, and a later fold that
    // would have observed that state (cmd's errorlevel excepted — that one the
    // run carries itself) is a shape the runner cannot reproduce.
    if let Some(reason) = state_across_folds(&steps) {
        return Err(Refusal::Shape(reason));
    }
    Ok(Plan {
        steps,
        exe: exe.to_path_buf(),
    })
}

/// Add one verbatim member to the open fold, opening a fold at `start_cwd` when
/// there is none: the directory the line was in before this member's own `cd`,
/// which the fold runs itself — `cd a && grep x .` searches `a` because the `cd`
/// and the search are members of ONE cmd.exe, and the runner starts that cmd.exe
/// where the line's own `cd` members had left the line.
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

/// The fold being accumulated by [`build`]: its text so far, the connector it
/// started with, and the cwd its cmd.exe runs in.
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

/// The join the member at `index` connects to the member before it with: the
/// previous member's own following connector, in cmd.exe's vocabulary. `First`
/// for the line's first member.
fn join_before(members: &[(Member, String)], index: usize) -> Result<Join, Refusal> {
    if index == 0 {
        return Ok(Join::First);
    }
    match members[index - 1].1.as_str() {
        "&&" => Ok(Join::And),
        "||" => Ok(Join::Or),
        // A newline is the unconditional separator `&` spells — the rewrite
        // re-emits it as `&` for exactly that reason.
        "&" | "\n" => Ok(Join::Always),
        "|" => Ok(Join::Pipe),
        other => Err(Refusal::Shape(format!(
            "the rewritten line carries the connector `{other}`, which this \
             platform's runner does not read"
        ))),
    }
}

/// The first fold, if any, that changes shell state a LATER fold would have
/// observed — the one class of fold this decomposition cannot reproduce (see the
/// module docs). Each fold is its own `cmd.exe`, so state a fold sets dies with
/// it, and cmd.exe's own reading of the line would have carried that state into
/// every member after it. Only a later *fold* can observe it here: an own-image
/// member is spawned by the runner, and the runner hands a command the one
/// environment the caller built for it — the owner's own, or the reduced
/// fallback ([`crate::tools::shell::apply_agent_env`]) — never one a previous
/// fold set (see the module docs' residuals).
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

/// The first state-changing cmd.exe builtin one of a fold's members names, or
/// `None` for a fold that names none. A fold's text is the members the segmenter
/// read, rejoined, so it is read back as those members here. Read through cmd.exe's
/// own verb key, like every other verb list in this module tree.
///
/// Absent on purpose: the `cd`/`chdir` half of the cwd family, because the analyzer
/// tracks its directory ([`Member::Own`]'s cwd is that tracking) and a fold's own
/// `cd` is cmd's business; and the console's own verbs (`prompt`, `title`,
/// `doskey`, `chcp`, `color`, `verify`, `break`), whose whole effect is the
/// console, and a console program this service starts owns a console of its own
/// (`super::tree`) — no later member can observe them either way.
///
/// `pushd`/`popd` stay despite being cwd-family: their directory is tracked like
/// `cd`'s, but the stack they push on is state a later fold could read, and no fold
/// here carries a stack.
fn state_verb(text: &str) -> Option<String> {
    /// cmd.exe's own verbs whose effect outlives the command that set it *and* is
    /// observable by a later fold: the environment and its scope, plus the
    /// directory stack's own pair.
    const STATE_VERBS: &[&str] = &["set", "setlocal", "endlocal", "path", "pushd", "popd"];
    for (member, _) in
        windows::segment_command(text).expect("a fold is the members the segmenter read, rejoined")
    {
        let words = member_words(&member);
        // The command word, where cmd.exe starts the command (see
        // [`command_word_index`]): a `set` behind a redirection is still the
        // member's own command.
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

/// The closed set of redirect spellings the runner can apply to a member it
/// spawns itself: `>`/`1>` and `>>`/`1>>` for stdout, `2>`/`2>>` for stderr, `<`
/// for stdin, and the stdout merge `2>&1` — with `nul`/`NUL` as the null-device
/// target for any of the target-taking ones.
///
/// A target-taking spelling is read the way the shared token classifier reads it
/// ([`super::scan::classify_shell_token`]): glued to its operator (`>out.txt`) or
/// as the following word (`>` `out.txt`), which cmd.exe treats the same.
///
/// Everything else is a named refusal rather than an approximation: a dup that is
/// not the stdout merge (`>&2`), cmd.exe's combined `&>`, a multi-digit
/// descriptor and a missing target are the spellings a member's own redirect list
/// can carry, a target word carrying two or more `%` is one cmd.exe would have
/// expanded before the member ran ([`windows::has_percent_expansion`] — the shell
/// tool spawns `cmd /C` without `/V:ON`, so `!` is ordinary text), and a
/// drive-relative target ([`windows::is_drive_relative`]) is one it resolves
/// against another drive's current directory; the reason names the spelling,
/// because the agent can change it and the runner cannot.
///
/// The merge is applied rather than refused because it is the one dup whose
/// destination the runner already owns (the member's stdout), and it is common
/// enough in an agent's own line — `… 2>&1 | head` — that refusing it would
/// refuse the pipeline. What it does *not* reproduce is cmd.exe's handle sharing:
/// the runner copies the member's stderr to that destination instead, which is
/// why the module docs' residuals state the ordering that costs.
fn redirects_supported(redirects: &[String]) -> Result<(), String> {
    parse_redirects(redirects).map(|_| ())
}

/// One redirect the runner applies to a member it spawns itself, read from the
/// member's verbatim spellings.
#[derive(Debug)]
enum Redirect {
    /// `<`: the member's standard input, from `target`.
    Stdin(String),
    /// `>`/`1>` (truncate) or `>>`/`1>>` (append): the member's stdout.
    Stdout { target: String, append: bool },
    /// `2>`/`2>>`: the member's stderr.
    Stderr { target: String, append: bool },
    /// `2>&1`: the member's stderr into whatever its stdout is.
    MergeIntoStdout,
}

/// Parse a member's verbatim redirect spellings into the runner's own reading.
/// `Err` carries the agent-facing refusal for a spelling outside the set
/// [`redirects_supported`] documents.
fn parse_redirects(redirects: &[String]) -> Result<Vec<Redirect>, String> {
    let mut parsed = Vec::new();
    let mut words = redirects.iter();
    while let Some(token) = words.next() {
        let spelling = split_redirect(token).map_err(|why| unsupported(token, why))?;
        let (stream, append, glued) = match spelling {
            // The merge names both of its descriptors, so no word is its target.
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
        // A spelling with no target of its own takes the word after it: the
        // tokenizer pushes the operator's own following word, so an operator at
        // the end of the list is a target the shell never got either.
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
        // A target cmd.exe would have expanded before the member ran: the runner
        // opens the file itself, so the pair would reach the file system as text
        // the shell never delivered.
        if windows::has_percent_expansion(target) {
            return Err(format!(
                "the runner cannot apply the redirection `{token}` to a member it spawns \
                 itself — its target `{target}` carries a `%…%` pair, which cmd.exe \
                 would have expanded before the member ran"
            ));
        }
        // A drive-relative target (`> C:out.txt`, quoted or not) names a file in
        // the current directory of ANOTHER drive, which nothing here tracks: joined
        // onto the runner's cwd it would open a different file than cmd.exe would,
        // with no sign that it did. Judged, and named in the cause, as cmd.exe
        // delivers the word — the quote removal [`RedirectDest::open`] applies —
        // and refused, as the cwd tracker refuses the same spelling for `cd`.
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

/// One redirect token's own reading: the stream it takes over with the reading of
/// its target, or the self-contained stdout merge.
#[derive(Debug, Clone, Copy)]
enum RedirectSpelling<'a> {
    /// An operator with its stream and append flag (`append` is `false` for `<`,
    /// which is not an appending operator).
    Operator {
        stream: Stream,
        append: bool,
        /// The target glued to the operator (`>out.txt`), or `None` when the next
        /// word is the target (`>` `out.txt`).
        glued: Option<&'a str>,
    },
    /// `2>&1`.
    MergeIntoStdout,
}

/// The redirect one token spells, or why it is outside the runner's closed set
/// (see [`redirects_supported`]).
///
/// Read exactly where the shared classifier sees an operator
/// ([`super::scan::classify_shell_token`]) and nowhere else: a token that opens
/// with `>`/`<` is a redirect, one that carries the operator later (`x>out.txt` —
/// cmd.exe's own split, which the engine refuses before any plan is built) is not.
/// On top of that, only the runner's own descriptors are read: `>`/`>>` take `1`
/// at most, `<` takes no digit, and a second digit, a descriptor dup (`>&2`,
/// `3>&1`) or the combined `&>` is a refusal whatever its target.
fn split_redirect(token: &str) -> Result<RedirectSpelling<'_>, Unsupported> {
    if token == "2>&1" {
        return Ok(RedirectSpelling::MergeIntoStdout);
    }
    // A `&`-carrying spelling names another descriptor's destination — a dup
    // (`>&2`, `1>&2`, `3>&1`) or cmd.exe's combined `&>` — rather than a file the
    // runner could open.
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
    // `<>file` is cmd.exe's read-write redirect: two ways into one file, which the
    // runner's closed set does not have.
    if tail.starts_with('>') {
        return Err(Unsupported::Spelling);
    }
    Ok(RedirectSpelling::Operator {
        stream,
        append,
        glued: (!tail.is_empty()).then_some(tail),
    })
}

/// Why one redirect token is outside the runner's closed set, as [`split_redirect`]
/// read it — so the refusal's wording is decided where the spelling was rejected
/// rather than re-derived from the token's text afterwards.
#[derive(Debug, Clone, Copy)]
enum Unsupported {
    /// A descriptor dup the merge is not (`>&2`, `1>&2`, `3>&1`) or the combined
    /// `&>`: another descriptor's destination, which the runner has none of.
    Descriptor,
    /// A spelling the closed set does not have at all (a multi-digit descriptor, a
    /// read-write `<>`).
    Spelling,
}

/// The refusal for one redirect token the runner cannot apply, naming the
/// spelling. A descriptor dup the merge is not is explained on its own terms (it
/// takes over a descriptor the runner has no destination for), while every other
/// spelling names the set the runner does apply.
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
    /// The single own-image step of a plan [`own_image_plan`] built — its argv,
    /// its redirect tokens and its cwd. `None` for any other plan: a plan of a
    /// compound line (the members around the call are steps of their own) has no
    /// single step to read, which is how a background session's launch fails a
    /// command it could only run as a chain.
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

/// Apply one own-image member's redirects to the command that runs it, and report
/// which of the member's streams they took over — and which merge the caller has
/// to perform — so it can leave the rest to its own wiring.
///
/// Targets resolve against `cwd`: cmd.exe resolves a relative redirect against the
/// process's working directory, and this member's is `cwd` — not the daemon's own.
/// `nul`/`NUL` is the platform's null device rather than a file of that name. The
/// list is applied left to right, exactly as cmd.exe reads it: a `2>&1` between
/// two redirects takes the stdout the ones before it left, so a `>` written after
/// it does not reach back to it.
pub(super) fn apply_redirects(
    cmd: &mut tokio::process::Command,
    redirects: &[String],
    cwd: &Path,
) -> io::Result<AppliedRedirects> {
    let parsed = parse_redirects(redirects).map_err(io::Error::other)?;
    let mut applied = AppliedRedirects::default();
    // What the member's stdout is at each point of the list: `None` while it is
    // still the caller's (the run's capture, the pipe into the next member, a
    // background session's output file).
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
            // Stderr joins the file stdout was sent to (`> out.txt 2>&1`); with
            // stdout still the caller's the merge is performed at that
            // destination, which the caller wires (see the module docs'
            // residuals). A `2>` written *before* this merge is overridden by it,
            // exactly as cmd.exe's own left-to-right application does — the
            // re-pointed stderr is the caller's again, and the file the earlier
            // redirect opened has been created and truncated, which is what cmd.exe
            // did before the merge took it away.
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
    /// The member's streams the redirects took over. The caller wires none of
    /// them; a merged stderr whose stdout the caller still owns is not among
    /// them — the caller owns that destination, so performing the merge is its
    /// part.
    pub(super) streams: Vec<Stream>,
    /// `2>&1` with stdout left to the caller (no `>` of the member's own has
    /// taken it over): the member's stderr belongs wherever the caller sends its
    /// stdout — the run's capture, the pipe into the next member of the group, or
    /// a background session's output file. A caller whose own two streams are one
    /// destination already (the background session's output file) has nothing left
    /// to do, so it reads only [`Self::streams`].
    pub(super) merge_into_stdout: bool,
}

/// Where a redirect sends one of the member's streams: cmd.exe's own two places —
/// the platform's null device, or a file the runner opened.
///
/// The open file is kept rather than handed over as the stream's `Stdio`, so a
/// `2>&1` after it sends stderr to the SAME open file: cmd.exe merges the
/// descriptors, and two handles on one path would each write from an offset of
/// their own.
enum RedirectDest {
    /// `nul`/`NUL`.
    Null,
    /// The file the redirect named, open for writing.
    File(std::fs::File),
}

impl RedirectDest {
    /// Open one output redirect's target: the platform's null device for
    /// `nul`/`NUL` (cmd.exe reads the word in any case), the file otherwise —
    /// created and truncated for `>`, created and appended for `>>`, resolved
    /// against `cwd`.
    fn open(target: &str, append: bool, cwd: &Path) -> io::Result<Self> {
        // The target cmd.exe would deliver — its own quote removal applied first —
        // which is then judged as the device in the spellings cmd.exe accepts
        // (`nul`, `nul:`, `nul.txt`, quoted or bare) by the one function that owns
        // them ([`scan::is_null_device`]) and opened as a file when it is not one.
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

    /// The stdio for one stream sent here. A file is *cloned*: the clone shares
    /// the open file's offset, which is what cmd.exe's descriptor merge gives the
    /// two streams, and it is dropped with the child when the child exits.
    fn stdio(&self) -> io::Result<Stdio> {
        match self {
            RedirectDest::Null => Ok(Stdio::null()),
            RedirectDest::File(file) => Ok(Stdio::from(file.try_clone()?)),
        }
    }
}

/// The stdio an input redirect's target names: the platform's null device, or the
/// file cmd.exe would have opened, resolved against `cwd`.
fn stdin_stdio(target: &str, cwd: &Path) -> io::Result<Stdio> {
    // The same reading [`RedirectDest::open`] makes of an output target: cmd.exe's
    // quote removal, then the device spellings, then the file it would open.
    let target = windows::unquote_word(target);
    if scan::is_null_device(&target) {
        return Ok(Stdio::null());
    }
    Ok(Stdio::from(std::fs::File::open(cwd.join(&target))?))
}

/// Whether a whole line the segmenter refused names this service's own image where
/// cmd.exe would have run a command from. This is [`own_image_plan`]'s fallback for
/// the line [`windows::segment_command`] refused: the text is the raw line, so a
/// spelling the model cannot read whole is still read at its first word and for the
/// image's path spelling anywhere in it ([`spelling_in_text`]) — the little an
/// unreadable line can still say.
#[must_use]
fn own_image_in_text(text: &str, exe: &Path) -> bool {
    let Some(words) = windows::tokenize(text) else {
        // A text this model cannot read whole is still read at its first word —
        // the one place a command can start in a line nothing precedes — beside
        // the image's own path spelling anywhere in it: `mahbot -V "unclosed` is
        // the call its first word says it is, and a text merely *carrying* the
        // path is caught by the spelling.
        return names_image(text.split_whitespace().next().unwrap_or(""), exe)
            || spelling_in_text(text, exe);
    };
    names_image_in_words(&words, exe)
}

/// Whether the words of one text name this service's own image where cmd.exe would
/// have run a command from: the word the text opens with, a word that follows one
/// of the connectors cmd.exe reads (`&&`, `||`, `&`, `|` — the spellings
/// [`join_before`] maps onto [`Join`], read on the word's own spelling so a quoted
/// `"&"` counts as the argument it is), or the word a launcher ([`LAUNCHER_VERBS`])
/// hands its command to ([`launcher_hands_image`]). A word naming the image in any
/// other position is ordinary text (a grep pattern, an operand, a program's
/// argument), which is why the position is part of the test — and so is cmd.exe's
/// own punctuation ([`command_word`], [`is_decoration`]): `@mahbot -V` and
/// `( mahbot -V )` are calls.
#[must_use]
fn names_image_in_words(words: &[grep_engine::GrepWord], exe: &Path) -> bool {
    let mut starts_command = true;
    let mut index = 0;
    while let Some(word) = words.get(index) {
        // A redirection does not consume the command position either: `> out.txt
        // mahbot -V` is the call it is, and the target word written after a
        // target-taking operator is part of the redirection rather than a word a
        // command could start at.
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
        // cmd.exe's own punctuation does not consume the command position: the
        // word after a group's `(` — or after a `@` standing alone — is still a
        // place a command can start, so `( mahbot -V )` is the call it is.
        starts_command = matches!(word.raw.as_str(), "&&" | "||" | "&" | "|")
            || (starts_command && is_decoration(&word.value));
        index += 1;
    }
    false
}

/// The refusal for a member that names this service's own image in a shape the
/// runner cannot run as one plain call of it, or `None` for a member that names no
/// image at all — such a member stays with the fold it belongs to.
///
/// Three causes reach here, and the cause says which: a command-position word that
/// names the image by a path the runner cannot match to the program it runs
/// ([`names_image_file`] — `.\mahbot.exe`, `target\debug\mahbot.exe`); a word
/// that IS the image ([`is_own_image_word`]) while the member is not one plain call
/// of it (a `%…%` word cmd.exe would have expanded); and a launcher
/// ([`launcher_hands_image`]) whose own grammar runs the image while handing the
/// call to a shell that does not wait for it — or, when that launcher's own command
/// word is one this model cannot read, the shape refusal, which claims only that
/// `mahbot` sits after such a launcher.
fn own_image_reference(words: &[grep_engine::GrepWord], text: &str, exe: &Path) -> Option<Refusal> {
    // The command word: not cmd.exe's own punctuation, and not a redirection
    // (`> out.txt mahbot -V` names its command in `mahbot`).
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
            // The launcher's own command word is not one this model can read: the
            // cause claims only what was read, since a word merely sitting after
            // such a launcher is not proof that the image is what it runs.
            Some(HandsOn::Unreadable) => Some(Refusal::Shape(format!(
                "the line's member `{text}` runs a launcher whose own command word this \
                 model cannot read while `mahbot` appears after it, and a call a shell \
                 might be starting is not one a shell waits for"
            ))),
            None => None,
        };
    }
    None
}

/// The verbs that hand the image to a shell rather than running it themselves:
/// cmd.exe's own `start` builtin, `call`, and an explicit interpreter launch.
/// Every one of them is a call the runner cannot wait for either, so a member
/// whose command word is one of these is refused when the word its own grammar
/// hands the command to is the image ([`launcher_hands_image`]).
const LAUNCHER_VERBS: &[&str] = &["start", "call", "cmd", "powershell", "pwsh"];

/// Whether a launcher at `words[launcher]` hands the line to this service's own
/// image, and how narrowly that could be read: the word the launcher's own
/// grammar gives its command to ([`launched_command`]) — or, when that word is
/// not one this model can see (a `cmd` whose switch it cannot read), any word
/// after the launcher, the wider reading that cannot narrow further without
/// missing a call.
///
/// A word merely *carried* by the launcher's command is not one: `cmd /c dir
/// mahbot` lists the directory, `powershell Get-Process mahbot` asks for a
/// process, and both are ordinary commands the shell may run.
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

/// How a launcher hands the line on this service's own image, as far as this model
/// can read it: the two readings refuse the line, and only [`HandsOn::ToCommand`]
/// says the image is the word the launcher runs — which is why the causes the two
/// render differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandsOn {
    /// The word the launcher's own grammar hands its command to is the image.
    ToCommand,
    /// That word is not one this model can see, while a word after the launcher
    /// names the image.
    Unreadable,
}

/// The word a launcher's own grammar hands its command to, or `None` when this
/// model cannot see one. Per launcher: `cmd` runs the text of its `/c` (or `/k`)
/// switch; `start` runs the first word that is neither one of its own switches nor
/// a window title (a quoted first word); `call` runs the word it is handed; an
/// interpreter (`powershell`, `pwsh`) runs the first word that is not one of its
/// own switches ([`is_switch_word`]). `None` is what keeps a caller on its wider
/// reading: a `cmd` whose switch this model cannot read (a command glued to it)
/// has no position to judge — and a shape this grammar does not resolve (a quoted
/// command word, a launcher inside a launcher, a `start` switch that takes the word
/// after it) is left to the shell, as the residuals state.
fn launched_command(words: &[grep_engine::GrepWord], launcher: usize) -> Option<usize> {
    let key = windows::verb_key(command_word(&words[launcher].value))?;
    let rest = &words[launcher + 1..];
    let offset = match key.as_str() {
        "cmd" => {
            // cmd.exe's own `/c`/`/k` (`-c`/`-k` too) — the spelling that carries
            // the text cmd.exe runs. A word without the switch character is an
            // ordinary argument, and so is a bare `c`.
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
/// `-Command`): a switch character followed by a single component. A word carrying
/// another separator after it is a path (`/Users/owner/mahbot`), not a switch —
/// the same distinction cmd.exe's own readers make.
#[must_use]
fn is_switch_word(word: &str) -> bool {
    word.starts_with(['/', '-']) && !word[1..].contains(['/', '\\'])
}

/// Whether one word of a command line names this service's own image at all — by
/// identity ([`is_own_image_word`]) or by the image's file name through a path
/// ([`names_image_file`]). This is the *detection* reading: it decides whether a
/// member is the runner's to look at, never that the member can be run as a plain
/// call.
#[must_use]
fn names_image(word: &str, exe: &Path) -> bool {
    is_own_image_word(word, exe) || names_image_file(word, exe).is_some()
}

/// The spelling of one word that names this service's own image's file name
/// through a path — `.\mahbot.exe`, `target\debug\mahbot.exe` — where a bare
/// `mahbot`/`mahbot.exe` is [`is_own_image_word`]'s identity case already.
///
/// `None` for a word that names some other file entirely. Such a word is a call
/// the runner cannot make: it cannot tell which file cmd.exe would resolve the
/// path to (it is not the path the runner runs, or it would be identity), the
/// shell would run some other file or none, and the shell does not wait for an
/// image like this one either way — so the member is refused rather than run or
/// handed over.
#[must_use]
fn names_image_file<'a>(word: &'a str, exe: &Path) -> Option<&'a str> {
    let name = image_file_name(exe)?;
    // A word with no separator is a bare name, which [`is_own_image_word`] has
    // already answered for: only a path spelling reaches this.
    let (_, file) = command_word(word).rsplit_once(['\\', '/'])?;
    windows::verb_key(file)
        .is_some_and(|key| Some(key) == windows::verb_key(name))
        .then_some(word)
}

/// Whether one word of a command line names this service's own image *by
/// identity*: its full path — case, separator and verbatim prefix aside, the same
/// comparison the engine's cwd gate uses — or its bare file name, `mahbot.exe` and
/// `mahbot` being one verb to cmd.exe's dispatch. Only the `.exe` suffix folds, so
/// `mahbot.com`/`mahbot.bat`/`mahbot.cmd` name a file of their own and a line
/// spelled through one stays the shell's (see the residuals). The reading the
/// runner can act on: a word this accepts is the program the runner itself runs.
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

/// One word read as a command word rather than as text: cmd.exe's `@` no-echo
/// prefix and the group parentheses it opens or closes with dropped from the
/// ends. cmd.exe reads `(`/`)` as command delimiters wherever they stand outside
/// quotes and `@` as the prefix of the command it opens, so the command word of
/// `(mahbot`, `@mahbot` and `mahbot)` is `mahbot` — the `@` being the prefix the
/// read-only guard's own verb key drops too.
#[must_use]
fn command_word(word: &str) -> &str {
    word.trim_start_matches(['@', '(']).trim_end_matches(')')
}

/// The index of the word a member's command starts at: cmd.exe's own punctuation
/// (`@`, group parentheses) and a redirection take no command position, so
/// `> out.txt mahbot -V` names its command in `mahbot` — and the redirection's
/// target is part of the redirection, never a command of its own.
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

/// This image's file name — the last component of its path, split on the
/// separators Windows itself accepts. Not `Path::file_name`: on a host whose own
/// separator is `/` that would leave a `C:\…\mahbot.exe` path whole, so this
/// reading is textual and answers the same on every host the lane runs on.
#[must_use]
fn image_file_name(exe: &Path) -> Option<&str> {
    exe.to_str()?
        .rsplit(['\\', '/'])
        .next()
        .filter(|name| !name.is_empty())
}

/// The coarse half of [`own_image_in_text`]: whether the raw text carries the
/// image's path spelling, case, separators and verbatim prefix aside. The one
/// signal left for a line cmd.exe's model cannot read — a spelling that stops the
/// model can still be the line that hands the image to the shell.
#[must_use]
fn spelling_in_text(text: &str, exe: &Path) -> bool {
    let spelled = fold_path(&crate::util::strip_verbatim_prefix(exe).to_string_lossy());
    !spelled.is_empty() && fold_path(text).contains(&spelled)
}

/// The comparison spelling of a path or a line: `/` folded to `\`, lowercased.
/// Good enough for the coarse check above, which is only ever asked whether one
/// path's spelling appears in a line.
#[must_use]
fn fold_path(text: &str) -> String {
    text.replace('/', "\\").to_lowercase()
}

/// What a line that the serve decision did not plan turned out to be.
pub(super) enum OwnImage {
    /// The line does not name this service's own image: the shell runs it
    /// exactly as it always did.
    None,
    /// The line names the image and the runner's own decomposition covers it —
    /// one step for a lone call, the whole chain for a line that names the image
    /// among other members. The runner runs it.
    Direct(Plan),
    /// The line names the image in a shape the runner cannot run; the cause is
    /// for [`refusal_message`].
    Refused(String),
}

/// Read a command line the serve decision did not plan for a call of this
/// service's own image — an agent running `mahbot debug …`, `mahbot chrome …`,
/// `mahbot bench-openrouter …`, `mahbot -V`.
///
/// `None` for every line that names no own image. A line that names the image is
/// read with the same cmd.exe model the analyzer's rewrite is built from — every
/// member's text as [`Member::Shell`], [`build`] classifying each of them: a
/// member that is a plain call of the image becomes an own step the runner spawns
/// itself, every other member becomes one the folds around it run, and the cwd is
/// tracked across the members so the image runs in the directory cmd.exe would
/// have run it in. A shape that reading cannot represent — a launcher spelling, a
/// path that is not the program the runner runs, a redirect the runner cannot
/// apply, a `cd` the tracking cannot follow — is refused ([`Refusal`], rendered by
/// [`refusal_message`]) rather than handed to a shell that will not wait for the
/// image. A line the cmd.exe model cannot read is still read for a call where a
/// command can start ([`own_image_in_text`], the same reading a member gets) and for
/// the image's path spelling ([`spelling_in_text`]); a line that only mentions the
/// bare name in text neither reader can read is the shell's (see the residuals).
///
/// Windows-only by construction, like every reading in this module: the platform
/// gate — and the `current_exe()` lookup with it — is [`plan_for_command`]'s.
#[must_use]
fn own_image_plan(command: &str, exe: &Path, cwd: &Path) -> OwnImage {
    let Some(members) = windows::segment_command(command) else {
        // A line the segmenter refused is read twice: for a call where a command can
        // start ([`own_image_in_text`] — the line's first word, one after a
        // connector, one a launcher hands the image to, so both `mahbot -V (x)` and
        // `cd sub && mahbot -V (x)` are the no-wait calls they are) and for the
        // image's own path spelling anywhere in the text ([`spelling_in_text`] — the
        // member no command-position reading can see). Only a line that merely
        // *mentions* the name is ordinary text (see the residuals).
        return if own_image_in_text(command, exe) || spelling_in_text(command, exe) {
            OwnImage::Refused(format!(
                "the line names `mahbot` but cannot be read as cmd.exe's own chain of \
                 commands (`{command}`), and a line the shell has to run cannot be \
                 waited for — run it as a line of its own"
            ))
        } else {
            OwnImage::None
        };
    };
    // The line is the runner's only when a member names the image — in command
    // position or through a launcher; a member that only *carries* the name
    // (`grep -rn mahbot .`) is ordinary text and the shell's. This pre-pass is what
    // keeps [`build`]'s refusals off a line that never mentions the image.
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
    // Tracked from the workspace root the analyzer's own tracking starts from, so
    // both readings of a directory spell it the same ([`tracked_cd`]).
    let root = grep_engine::canonical_or_lexical(cwd, ShellPlatform::Windows);
    match build(&line, exe, &root) {
        Ok(plan) => OwnImage::Direct(plan),
        Err(refusal) => OwnImage::Refused(refusal.to_string()),
    }
}

/// The plan a line that runs this service's own image is to be run by, or
/// [`OwnImage::None`] for a line that is the shell's: the one dispatch of the
/// own-image reading, shared by the shell tool and by the background sessions, so
/// both agree about which lines the runner takes over and which it refuses.
///
/// Windows only, and by construction: on any other platform this platform's shell
/// waits for its children, so the image's own path (a `current_exe()` syscall)
/// would have no reader. A failed lookup is fail-open, exactly as it is for the
/// analyzer's rewrite: nothing can be recognised as the image, so the line runs as
/// the shell runs it today.
#[must_use]
pub(super) fn plan_for_command(command: &str, cwd: &Path) -> OwnImage {
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
    own_image_plan(command, &exe, cwd)
}

/// The argv and redirect tokens of one member's plain call of this service's own
/// image, or `None` when the member is not one: it has no command word at all (a
/// redirect-only member), its command word does not name the image by identity
/// ([`is_own_image_word`]), or one of its words carries a `%…%` pair cmd.exe would
/// have expanded (the runner spawns argv directly, so the word would reach the
/// image as text the shell never delivered). Such a call is one the shell does not
/// wait for either, so it is refused as [`Refusal::OwnImageInShell`] rather than run
/// or handed over.
///
/// The command word is the one where cmd.exe starts the command: not its own
/// punctuation (`@ mahbot -V` is the call `@mahbot -V` is), and not a redirection
/// — a redirect written *before* the command (`> out.txt mahbot -V`) is the
/// member's own and rides along with the rest. The argv is the words after the
/// command word, unquoted by cmd.exe's model — the words argv would have carried
/// with no interpreter in between — and every redirect token rides along verbatim
/// in the order cmd.exe applies them, for [`build`] to validate and
/// [`apply_redirects`] to apply.
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

/// The cwd a `cd`-family member hands the members after it, or `None` when the
/// member is not one — the analyzer's own tracking
/// ([`grep_engine::resolve_cd`], the one reading of cmd.exe's `cd` grammar), asked
/// of a line the serve decision did not plan so the directory a member runs in
/// cannot disagree with the directory the engine's cwd gate names.
///
/// `Err` is the refusal for a member cmd.exe reads as a directory change this
/// tracking cannot follow: a spelling of the family the verb key does not know
/// (`cd..`, `cd\Users`, `cd/d`, `@cd`), a redirection written *before* the member's
/// command word, a target the resolver itself refuses (a drive-relative `cd C:foo`),
/// and any of them on one side of a pipe (`piped`), where the shell gives the
/// pipeline its own interpreter. Running a step of its own after it in a directory
/// the shell would not have used would be a wrong answer, not a missing one — and
/// the runner spawns every step itself, so nothing downstream can still notice the
/// divergence.
///
/// Windows-only by construction, like every other reading in this module; the
/// `home` [`grep_engine::resolve_cd`] takes is unix-only (a bare `cd` targets
/// `$HOME` there), so the placeholder is never consulted — the same one
/// `grep_engine` passes when it has no home.
fn tracked_cd(
    words: &[grep_engine::GrepWord],
    text: &str,
    cwd: &Path,
    piped: bool,
) -> Result<Option<PathBuf>, String> {
    let platform = ShellPlatform::Windows;
    // The member's command word, where cmd.exe starts the command (its own
    // punctuation and a redirection before the command take no command position),
    // with whether that word is the member's first — which is where the shared
    // resolver reads a member from. A member with no command word at all (a
    // redirect-only text) has no verb, so its first spelling stands in for one.
    let command = command_word_index(words).map(|at| (at, words[at].value.as_str()));
    let (leads, verb) = match command {
        Some((at, verb)) => (at == 0, verb),
        None => (false, text.split_whitespace().next().unwrap_or("")),
    };
    // One side of a pipe is its own interpreter in cmd.exe, and which directory the
    // members after it see is not something this reading can answer: the analyzer
    // refuses the same shape for a served line ([`grep_engine`]'s `CdUntrackable`),
    // and a wrong tracked cwd is a wrong answer rather than a missing one.
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

/// Whether one word names cmd.exe's directory-change family in any of the
/// spellings the tracking reads: the family's own names
/// ([`grep_engine::is_cd_segment`]) or cmd's fused ones
/// ([`grep_engine::is_cd_spelling`]).
#[must_use]
fn names_cd_family(word: &str) -> bool {
    let platform = ShellPlatform::Windows;
    grep_engine::is_cd_segment(word, platform) || grep_engine::is_cd_spelling(word, platform)
}

/// The refusal for a member that changes the shell's directory inside one of
/// cmd.exe's own keyword forms (`if exist x cd sub`, `for %f in (*) do cd sub`,
/// `else cd sub`), or `None` for a member whose command word is no keyword form or
/// that names no directory change of the family.
///
/// The keyword's own condition decides whether that `cd` runs at all — a model
/// that does not evaluate conditions cannot tell — so the members after it would
/// run in a directory the shell might not have used. The reading is deliberately
/// wider than the grammar's (any word after the keyword, not only the command
/// position the condition hands its own command): a `cd` this model reads as
/// merely standing in the member (`if exist x echo cd sub`) is refused too, which
/// is the direction a wrong directory must not be read in.
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
/// the shared [`grep_engine::REFUSAL_FRAME`], so the same cause cannot reach the
/// agent worded two ways through the two entry points; what follows the frame is
/// the consequence of *this* one, that the answer is not the command's output,
/// where the search's own entry point states that it is not an empty match set.
#[must_use]
pub(super) fn refusal_message(cause: &str) -> String {
    format!(
        "{}{cause}. This is not the command's output.",
        grep_engine::REFUSAL_FRAME
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

// ── Execution ─────────────────────────────────────────────────────────────

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
/// single-child runner gives its one child, its output captured into the run's
/// streams, and the run's bounds — one deadline, one memory cadence, the capture
/// budget of the stream each member writes, one output drain — applied to the plan
/// as a whole.
///
/// The result maps onto the same [`ShellRunResult`] variants a shell run produces,
/// so the caller cannot tell a plan run from one: `Completed`, `TimedOut`,
/// `DrainTimedOut`, `MemoryExceeded`, `SpawnFailed`. The status of a `Completed`
/// run is the last executed group's last member's (see the module docs).
pub(super) async fn run(
    plan: &Plan,
    timeout: Duration,
    drain_limit: Duration,
    memory_limit: Option<u64>,
    owner: RunOwner,
) -> ShellRunResult {
    let start = std::time::Instant::now();
    let mut tree = Tree::new(owner);
    // Kill-on-drop: an aborted task (a cancellation drop, a panic in a sibling
    // tool, runtime teardown) must not orphan any of the run's members. Disarmed
    // on every path that reaped them or ended the tree itself.
    let mut kill_guard = KillOnDrop::new(tree.clone());
    let cancel = tokio_util::sync::CancellationToken::new();
    // The run's capture budgets, one per stream (see [`CaptureBudgets`]).
    let budgets = CaptureBudgets::new();
    let mut readers = Readers::default();
    // The run's bounds, watched across every group: one deadline, one cadence.
    let mut watchdog = Watchdog::new(timeout);
    // cmd's errorlevel: the last executed group's status decides whether the next
    // one runs, and a skipped group leaves it in place.
    let mut previous: Option<i32> = None;
    // The status the run reports: the last executed group's last member's.
    let mut reported: Option<std::process::ExitStatus> = None;
    // The pid a stop names: the group the run was waiting for (see
    // [`ShellRunResult::TimedOut`]).
    let mut named_pid: Option<u32> = None;

    for group in pipe_groups(&plan.steps) {
        if !runs_after(plan.steps[group.start].join, previous) {
            continue;
        }
        let mut members =
            match spawn_group(plan, group, &mut tree, &mut readers, &budgets, &cancel).await {
                Ok(members) => members,
                // The group's own members were killed and reaped by the spawn path
                // — a failed spawn or a redirect it could not open — and every
                // earlier group was reaped too, so nothing is left for the guard to
                // end: it is disarmed rather than left to end a reaped group's tree
                // ([`KillOnDrop`]'s contract). The capture readers are cancelled
                // too: nothing collects them on this path, and a reader left waiting
                // on a member's pipe would outlive the run it belongs to.
                Err(e) => {
                    cancel.cancel();
                    kill_guard.disarm();
                    return ShellRunResult::SpawnFailed(e);
                }
            };
        named_pid = members.first().map(|member| member.pid);
        match wait_group(&mut members, &mut watchdog, memory_limit).await {
            GroupEnd::Reaped => {
                let last = members.last().and_then(|member| member.status);
                // cmd's errorlevel for the next group's connector, and the status
                // the run reports when the plan ends here.
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
                // A member that could not be reaped may still be running: the guard
                // stays armed and ends the run's tree on the way out. The capture
                // readers are left to end when the tree's pipes close, as the
                // single-child runner leaves its own on a failed spawn.
                return ShellRunResult::SpawnFailed(e);
            }
        }
    }

    // Every group ran or was skipped: the run ended on its own, so nothing is
    // killed here and the job is kept rather than closed.
    kill_guard.disarm();
    if readers.drain(drain_limit).await {
        tree.retain_after_completion();
        let (stdout, stderr) = readers.collect().await;
        // The first group of a non-empty plan always runs, and a plan is built
        // from a non-empty line: no status here means a plan nobody built
        // ([`Plan::steps`] empty), which is a failed run rather than a panic.
        let Some(status) = reported else {
            return ShellRunResult::SpawnFailed(io::Error::other("the plan has no step to run"));
        };
        return ShellRunResult::Completed {
            stdout,
            stderr,
            status,
            elapsed: start.elapsed(),
        };
    }
    // The drain bound was exceeded: a leftover process still holds a capture
    // pipe. End the run's tree — the members' process groups on unix, the job on
    // Windows — then collect what the readers hold, mirroring the single-child
    // path.
    let ended = tree.terminate();
    let (stdout, stderr) = collect_stopped(&cancel, &mut readers).await;
    ShellRunResult::DrainTimedOut {
        stdout,
        stderr,
        // Named as killed only when the tree really was ended (see the variant),
        // and this platform's containment is the whole run rather than one pid.
        pid: named_pid.filter(|_| ended),
        elapsed: start.elapsed(),
    }
}

/// Spawn one group's members, left to right, connecting consecutive members with
/// real pipes the runner owns, and starting a reader for every stream the group
/// leaves for the run's capture.
///
/// A member that cannot be spawned ends the group: the run's tree is ended and the
/// members this group already started are reaped here — the caller disarms the kill
/// guard on that path, because those members were reaped and the earlier groups
/// were too, which is [`KillOnDrop`]'s contract. A member whose own redirect the
/// runner cannot open ends the group the same way, for the same reason.
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
        // validated when the plan was built): the streams they take over are not
        // wired here, and a `2>&1` whose stdout the runner still owns tells the
        // wiring below that the member's stderr shares that destination.
        let applied = match &step.run {
            Run::Own { redirects, .. } => match apply_redirects(&mut cmd, redirects, &step.cwd) {
                Ok(applied) => applied,
                // A redirect the runner cannot open — a missing directory, a
                // directory target — ends the group exactly as a failed spawn
                // does: this group's earlier members are reaped here, which is
                // what lets the caller disarm the kill guard on that path.
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
        // stdout is piped unless the member redirected it: a non-last member's
        // pipe is the next member's input, the last member's is the run's
        // capture.
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
        // The platform answers `None` only once the child is reaped, which the wait
        // path below is what does: a member this runner spawned has its pid, and it
        // is what the memory watchdog samples.
        let pid = child
            .id()
            .expect("a freshly spawned child still has its pid");
        tree.attach(pid);
        // Wire this member's stdin from the member before it.
        match feed.take() {
            // The first member of the group: its stdin is whatever this process
            // has, exactly as the single shell child's is.
            None => {}
            Some(pipe) => match child.stdin.take() {
                Some(write_end) => connect_pipe(pipe, write_end),
                // The member's own redirect took its stdin (`… | grep x . < in.txt`):
                // the redirect wins, exactly as cmd.exe's does, so the producer's
                // pipe is closed by dropping the read ends.
                None => drop(pipe),
            },
        }
        // What this member leaves for the next one — its stdout, and its stderr
        // too when the member merged the two — or for the run's capture.
        let piped = index + 1 < group.end;
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        if piped {
            feed = Some(Pipe {
                stdout: stdout.take(),
                // A merged stderr is one of the streams the consumer's input
                // carries (see [`Pipe`]); without a merge it stays the run's own
                // stderr channel, below.
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
            // A merged stderr is the member's stdout destination, so its bytes
            // join the run's captured stdout (the module docs' residuals) — and
            // the stdout budget, the stream they were charged to.
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

/// The pipe between two members of a group: what the member before left for the
/// member after it to read.
struct Pipe {
    /// The producer's stdout, or `None` when a redirect of its own sent stdout
    /// elsewhere. A `2>&1` written before such a `>` still left the merged stderr
    /// feeding this pipe (see [`connect_pipe`]); only a producer with neither
    /// stream leaves a pipe nothing writes into, and its input is closed, which is
    /// the end-of-input cmd.exe's own pipeline gives it.
    stdout: Option<ChildStdout>,
    /// The producer's stderr, when the producer merged it into its stdout
    /// (`2>&1`): the pipe's input too, when a later `>` of its own took stdout over
    /// (see [`connect_pipe`]), otherwise the same destination as that stdout, fed
    /// by a second pump.
    stderr: Option<ChildStderr>,
}

/// Connect the pipe the member before left to the member just spawned: its stdout
/// into the consumer's stdin, plus its stderr when the producer merged the two.
///
/// A producer whose own redirect sent stdout elsewhere leaves its merged stderr —
/// when it wrote `2>&1` before that `>` — as the pipe's input, and otherwise
/// leaves nothing to read, so the consumer's input is dropped — the end-of-input
/// cmd.exe's own pipeline gives it either way.
fn connect_pipe(pipe: Pipe, write_end: ChildStdin) {
    match (pipe.stdout, pipe.stderr) {
        // Two streams, one destination: both pumps write through the shared
        // handle (see [`SharedStdin`]).
        (Some(stdout), Some(stderr)) => {
            let shared = SharedStdin::new(write_end);
            spawn_merged_copy(stdout, shared.clone());
            spawn_merged_copy(stderr, shared);
        }
        (Some(stdout), None) => spawn_pipe_copy(stdout, write_end),
        // The member's own `>` sent stdout to a file, but its `2>&1` was written
        // before that redirect: cmd.exe had taken this pipe for the merge, so the
        // merged stream is the pipe's input even though nothing writes stdout
        // into it.
        (None, Some(stderr)) => spawn_pipe_copy(stderr, write_end),
        (None, None) => drop(write_end),
    }
}

/// The write end of one pipe between two members, shared by the pumps that feed
/// it: the producer's stdout, plus its stderr when the producer merged the two.
/// Each pump takes the handle for one chunk, so both streams reach the consumer
/// whole; the consumer's input closes when the last handle is dropped, which is
/// the end-of-input its own reads need.
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

/// Copy one of a member's two merged streams into the shared write end of the pipe
/// it feeds (`2>&1`), one chunk at a time so the two streams interleave rather
/// than one pump waiting out the other's whole stream.
///
/// The task ends when either end does, exactly as [`spawn_pipe_copy`]'s: a failed
/// write is the consumer gone, and a closed read end is the producer gone.
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

/// One read of a merged pipe pump ([`spawn_merged_copy`]): the chunk a single
/// write into the shared handle carries.
const MERGED_COPY_CHUNK: usize = 8 * 1024;

/// How waiting for one group's members ended. The bounds behind these outcomes
/// are the shared [`Watchdog`]'s, so only the payload differs from the
/// single-child runner's [`WaitOutcome`].
enum GroupEnd {
    /// Every member was reaped; their statuses are on them.
    Reaped,
    /// The run's deadline expired while a member was still running.
    TimedOut,
    /// A memory sample crossed the ceiling: the sample and the ceiling are
    /// carried so the failure can report both.
    MemoryExceeded { used: u64, limit: u64 },
    /// A member could not be reaped at all.
    Failed(io::Error),
}

/// Wait for every member of one group under the run's shared bounds, recording
/// each member's status as it is reaped.
async fn wait_group(
    members: &mut [MemberRun],
    watchdog: &mut Watchdog,
    memory_limit: Option<u64>,
) -> GroupEnd {
    // The statuses land here rather than on the members, because the member's own
    // wait future borrows it (below) for as long as the group is being waited for.
    // Every member arrives unwaited and un-reaped ([`spawn_group`] is the only
    // constructor, and it returns right before this), so a member without a status
    // recorded here is exactly one still running, and the pids the watchdog samples
    // — every member's — are the same reading of the group: a member leaves both
    // sets at the moment its status lands.
    debug_assert!(
        members.iter().all(|member| member.status.is_none()),
        "a member is waited for as soon as it is spawned, never twice"
    );
    let mut statuses: Vec<Option<std::process::ExitStatus>> = vec![None; members.len()];
    // The pids the watchdog samples, with the member each belongs to: every member
    // (its pid is taken at spawn), a reaped one leaving both sets at once.
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

/// End the group the run was waiting for, the way the single-child stop does:
/// each member is signalled (the fallback for a run whose containment could not
/// be established, which is why the direct kill comes first) and the tree ends
/// them together, then they are reaped so nothing outlives the tool call.
async fn end_group(tree: &Tree, members: &mut [MemberRun], kill_guard: &mut KillOnDrop) {
    end_group_members(tree, members).await;
    // Reaped — disarm the guard (the explicit kill already ran).
    kill_guard.disarm();
}

/// [`end_group`]'s kill-and-reap, for a caller that owns the disarm decision:
/// every member is signalled, the run's tree ends them together, and they are
/// reaped.
async fn end_group_members(tree: &Tree, members: &mut [MemberRun]) {
    for member in members.iter_mut() {
        let _ = member.child.start_kill();
    }
    tree.terminate();
    for member in members.iter_mut() {
        let _ = member.child.wait().await;
    }
}

/// Collect a run that is being stopped: cancel the capture readers and take what
/// they hold, so every stop path — the deadline, the memory ceiling, the output
/// drain — reports its partial output instead of a hang. The kill itself is the
/// caller's, because what has to be ended differs: a group's members are reaped
/// through [`end_group`], while the drain path's were all reaped already and only
/// the run's tree can still hold a leftover process.
async fn collect_stopped(
    cancel: &tokio_util::sync::CancellationToken,
    readers: &mut Readers,
) -> (Vec<u8>, Vec<u8>) {
    cancel.cancel();
    readers.collect().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    /// The image and workspace every case here runs against; the plan's own
    /// decomposition never touches either, so the paths only have to be the
    /// shapes a real install has.
    const EXE: &str = r"C:\Program Files\MahBot\mahbot.exe";
    const ROOT: &str = r"C:\ws";

    /// One verbatim member and the connector that follows it — the analyzer's
    /// per-member data. The directory its fold starts in is [`build`]'s own
    /// tracking (from the root the plan is built at), not a field of the member.
    fn shell(text: &str, conn: &str) -> (Member, String) {
        (
            Member::Shell {
                text: text.to_string(),
            },
            conn.to_string(),
        )
    }

    /// One served member: its argv after the image path, its redirect tokens, the
    /// cwd its search tracked, and the connector that follows it.
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

    /// The engine's own hand-off words for a served member (the plan does not
    /// care what they are, and neither does the decomposition — only that they
    /// travel as argv).
    fn served(cwd: &str, conn: &str) -> (Member, String) {
        own(
            &["__grep-engine", "--spec-file", r"C:\tmp\spec.json"],
            &[],
            cwd,
            conn,
        )
    }

    /// The plan for a line whose members are handed over as the analyzer hands
    /// them, tracked from [`ROOT`] — the root every case here reads its `cd`
    /// members against.
    fn plan(members: &[(Member, String)]) -> Plan {
        build_at(ROOT, members).expect("a decomposable line")
    }

    /// [`build`] for one case, with `root` (the analyzer's own canonicalization of
    /// it) as the cwd its tracking starts from.
    fn build_at(root: &str, members: &[(Member, String)]) -> Result<Plan, Refusal> {
        build(members, Path::new(EXE), &tracked_root_of(root))
    }

    /// The workspace root the decomposition tracks from: `root` under the
    /// analyzer's own canonicalization, which a host whose separator is `/`
    /// resolves like any other relative path.
    fn tracked_root_of(root: &str) -> PathBuf {
        grep_engine::canonical_or_lexical(Path::new(root), ShellPlatform::Windows)
    }

    /// The plan [`own_image_plan`] built for a line, or a panic naming what it
    /// read instead.
    fn direct_plan(command: &str) -> Plan {
        match own_image_plan(command, Path::new(EXE), Path::new(ROOT)) {
            OwnImage::Direct(plan) => plan,
            OwnImage::Refused(cause) => panic!("{command}: refused: {cause}"),
            OwnImage::None => panic!("{command}: not read as an own-image line"),
        }
    }

    /// The agent-facing cause of the refusal for a line.
    fn refusal_cause(command: &str) -> String {
        match own_image_plan(command, Path::new(EXE), Path::new(ROOT)) {
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

    /// `grep -rn x .` — the one shape that is nothing but a served search: one
    /// own step in the workspace root, sequenced as the line's first.
    #[test]
    fn a_lone_search_is_one_own_step() {
        let plan = plan(&[served(ROOT, "")]);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].join, Join::First);
        assert_eq!(plan.steps[0].cwd, PathBuf::from(ROOT));
        let Run::Own { args, .. } = &plan.steps[0].run else {
            panic!("the served member must be an own step");
        };
        assert_eq!(args[0], "__grep-engine");
    }

    /// `grep -rn x . | head -3`: a pipe group of two members — the engine, then
    /// the tail the runner feeds it to — while the run's line stays two members
    /// wide.
    #[test]
    fn a_pipelined_search_is_one_pipe_group() {
        let plan = plan(&[served(ROOT, "|"), shell("head -3", "")]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::Pipe);
        // One group, two members: the pipe is the runner's, not a step boundary.
        assert_eq!(pipe_groups(&plan.steps), vec![0..2]);
    }

    /// `cd src && grep -rn x .` — the fold is cmd.exe's own `cd`, and the served
    /// step runs in the directory that `cd` chose, which the analyzer tracked
    /// (not in the line's own cwd, and not in whatever the fold's cmd.exe would
    /// have left behind).
    #[test]
    fn a_cd_before_a_search_is_a_fold_and_the_tracked_cwd() {
        let plan = plan(&[shell("cd src", "&&"), served(r"C:\ws\src", "")]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(shell_text(&plan.steps[0]), "cd src");
        // The fold starts where the line does: it runs the `cd` itself.
        assert_eq!(plan.steps[0].cwd, tracked_root_of(ROOT));
        assert_eq!(plan.steps[1].join, Join::And);
        assert_eq!(plan.steps[1].cwd, PathBuf::from(r"C:\ws\src"));
    }

    /// `cd src && grep -rn x . | head -3`: one fold, one pipe group — three
    /// members, two steps. The tail's fold starts in the cwd the served member
    /// ran in, which is where the step after the engine's `cd` left the line.
    #[test]
    fn a_cd_search_and_tail_is_a_fold_then_a_pipe_group() {
        let plan = plan(&[
            shell("cd src", "&&"),
            served(r"C:\ws\src", "|"),
            shell("head -3", ""),
        ]);
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[2].join, Join::Pipe);
        // The fold, then the pipeline the engine feeds.
        assert_eq!(pipe_groups(&plan.steps), vec![0..1, 1..3]);
        assert_eq!(shell_text(&plan.steps[0]), "cd src");
        assert_eq!(plan.steps[2].cwd, plan.steps[1].cwd);
    }

    /// `cd src && type f.txt | grep -n needle`: the producer is a step of its own,
    /// so the plan is three steps — the `cd`'s fold (`&&`), the pipe's head
    /// (`&&`), and the served search (`|`) — not two. Folding the producer into
    /// the `cd` would gate the pipeline on the `cd`'s connector instead of the
    /// producer's own, and would send the fold's whole stdout through the pipe.
    #[test]
    fn a_member_that_feeds_a_pipe_is_a_step_of_its_own() {
        let root = tracked_root_of(ROOT);
        let plan = plan(&[
            shell("cd src", "&&"),
            shell("type f.txt", "|"),
            served(r"C:\ws\src", ""),
        ]);
        assert_eq!(plan.steps.len(), 3);
        // One-member groups: the head runs on its own, the search is the one
        // member the producer's stdout feeds.
        assert_eq!(pipe_groups(&plan.steps), vec![0..1, 1..3]);
        assert_eq!(shell_text(&plan.steps[0]), "cd src");
        assert_eq!(plan.steps[0].join, Join::First);
        assert_eq!(shell_text(&plan.steps[1]), "type f.txt");
        assert_eq!(plan.steps[1].join, Join::And);
        assert!(matches!(plan.steps[2].run, Run::Own { .. }));
        assert_eq!(plan.steps[2].join, Join::Pipe);
        // Each step runs where the line's own `cd` tracking leaves it: the `cd`'s
        // fold at the root (it runs the `cd` itself), the head in the directory
        // that `cd` chose, and the search at the cwd its own tracking named.
        assert_eq!(
            step_cwds(&plan),
            vec![
                root.as_path(),
                root.join("src").as_path(),
                Path::new(r"C:\ws\src")
            ]
        );
    }

    /// A member that *takes* a pipe is a step of its own too: what a following
    /// connector reaches in `grep x . | head -3 && cat` runs in cmd's own process,
    /// with the shell's stdin rather than the pipe the consumer alone was given.
    #[test]
    fn the_pipeline_consumer_is_a_step_of_its_own() {
        let piped = plan(&[served(ROOT, "|"), shell("head -3", "&&"), shell("cat", "")]);
        assert_eq!(piped.steps.len(), 3);
        assert_eq!(pipe_groups(&piped.steps), vec![0..2, 2..3]);
        assert!(matches!(piped.steps[0].run, Run::Own { .. }));
        assert_eq!(shell_text(&piped.steps[1]), "head -3");
        assert_eq!(piped.steps[1].join, Join::Pipe);
        assert_eq!(shell_text(&piped.steps[2]), "cat");
        assert_eq!(piped.steps[2].join, Join::And);

        // Two consumers and a tail after them: every pipe member is its own step,
        // and the tail is gated by the last consumer's own status.
        let tailed = plan(&[
            served(ROOT, "|"),
            shell("head -3", "|"),
            shell("sort", "&&"),
            shell("echo done", ""),
        ]);
        assert_eq!(tailed.steps.len(), 4);
        assert_eq!(pipe_groups(&tailed.steps), vec![0..3, 3..4]);
        assert_eq!(tailed.steps[3].join, Join::And);

        // Two members around a `|` are two steps even with nothing after the
        // consumer: the pipe is the runner's, so both ends are spawned by it.
        let bare = plan(&[shell("type a.txt", "|"), shell("sort", "")]);
        assert_eq!(bare.steps.len(), 2);
        assert_eq!(shell_text(&bare.steps[0]), "type a.txt");
        assert_eq!(shell_text(&bare.steps[1]), "sort");
        assert_eq!(pipe_groups(&bare.steps), vec![0..2]);
    }

    /// `grep a f1 && grep b f2`: two served members, each in its own group, the
    /// second sequenced by cmd's `&&` and running in the first's cwd.
    #[test]
    fn two_searches_are_two_sequenced_own_steps() {
        let plan = plan(&[served(ROOT, "&&"), served(ROOT, "")]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::And);
        assert!(runs_after(plan.steps[1].join, Some(0)));
        assert!(!runs_after(plan.steps[1].join, Some(1)));
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

    /// The fold rejoins its members with the connectors the rewrite spells,
    /// including the newline cmd.exe reads as `&`.
    #[test]
    fn a_fold_joins_its_members_with_their_own_connectors() {
        let plan = plan(&[
            shell("echo one", "\n"),
            shell("echo two", "&&"),
            shell("echo three", ""),
        ]);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(
            shell_text(&plan.steps[0]),
            "echo one & echo two && echo three"
        );
    }

    /// The fold is broken by an own-image member, and the fold after it runs in
    /// that member's cwd — the cwd the analyzer tracked and the line's own `cd`s
    /// left behind.
    #[test]
    fn an_own_step_breaks_the_fold_and_carries_the_cwd() {
        let plan = plan(&[
            shell("cd src", "&&"),
            served(r"C:\ws\src", "&&"),
            shell("type note.txt", ""),
        ]);
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(shell_text(&plan.steps[0]), "cd src");
        assert_eq!(shell_text(&plan.steps[2]), "type note.txt");
        assert_eq!(plan.steps[2].cwd, PathBuf::from(r"C:\ws\src"));
    }

    /// A verbatim member that is a plain call of this service's own program is not
    /// the shell's to run, whatever its caller took it for: [`build`] converts it
    /// into an own step — the shell never sees the call — and the fold around it
    /// breaks there. This is what makes a line that runs this service beside a
    /// search (a rewritten `mahbot … && grep …`) the runner's rather than a
    /// refusal.
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
            // cmd.exe's `@` no-echo prefix, glued and standing alone: not a word of
            // the command it opens, so the call is one plain call of the image and
            // its argv is what follows the prefix.
            (vec![shell("@mahbot -V", "")], ["-V"].as_slice()),
            (vec![shell("@ mahbot -V", "")], ["-V"].as_slice()),
            // A redirection written before the command takes no command position:
            // the call is the one it names, and the redirection is its own.
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
        // The redirection written before the command rides the own step, in the
        // order cmd.exe applies it.
        let redirected = plan(&[shell("> out.txt mahbot -V", "")]);
        let Run::Own { redirects, .. } = &redirected.steps[0].run else {
            panic!("a call of the image is an own step");
        };
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);
        // The same words in argument position are ordinary text: a search's
        // pattern, an operand, another program's argument.
        for text in [
            "echo mahbot",
            "type mahbot.exe",
            "echo (mahbot -V)",
            "echo @mahbot -V",
            // A redirection naming a file the image's name spells is a file of that
            // name, not a call: the target of a redirect is never a command word.
            "echo hi > mahbot.exe",
            "echo hi < mahbot -V.txt",
            // A launcher that merely carries the name runs something else, and so
            // does a file the image's name spells through another extension — only
            // `.exe` is this image to cmd.exe's dispatch.
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

    /// A member that names this service's own program in a shape the runner cannot
    /// run as one plain call of it refuses the line, and the cause names the shape: a
    /// launcher, which hands the image to a shell that does not wait for it, and a
    /// path spelling that names the program's file name but not the program the
    /// runner runs. Both would otherwise reach the agent as a truncated or empty
    /// result (the launcher shapes this grammar does not resolve are the shell's —
    /// see the module's residuals).
    #[test]
    fn a_fold_naming_the_image_in_a_shape_the_runner_cannot_run_refuses_the_line() {
        for (members, spelling) in [
            (vec![shell("start mahbot chrome", "")], false),
            (vec![shell("call mahbot.exe -V", "")], false),
            (vec![shell("cmd /c mahbot bench-openrouter", "")], false),
            // cmd.exe's other switch character is the same launch.
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
        // A launcher whose own command word this model cannot read: refused too, with
        // a cause that claims only what was read — a word merely sitting after such a
        // launcher is not proof of what it runs. A bare `c`/`k` letter is an ordinary
        // argument, not cmd.exe's switch, so it does not make the word after it the
        // launcher's command.
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

    /// A redirect the runner cannot apply to a member it spawns itself refuses
    /// the line, naming the spelling — the stdout merge is not one of them any
    /// more ([`redirects_supported`]).
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

    /// A fold whose shell state a later fold would have observed is refused — each
    /// fold is a fresh cmd.exe, and state that dies with its fold cannot be
    /// approximated. Allowed, because no later fold can read their state: a `cd`
    /// (the analyzer tracks the directory) and a `pushd <target>` (tracked the same
    /// way), and a verb whose whole effect is the console — a console child owns a
    /// console of its own.
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
        // The readers of a fold's state in a later fold are the environment and the
        // directory stack, never the console.
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
            // A redirection before the command word: `resolve_cd` reads the verb
            // from the member's first word, which here is the redirect.
            "> out.txt cd src",
            // A drive-relative target names the cwd of ANOTHER drive.
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
        // A directory change on either side of a pipe: the shell gives each side an
        // interpreter of its own, so which directory the members after it see is not
        // something this tracking can answer.
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
        // The same spelling with only members of its own fold after it: cmd.exe runs
        // the move itself inside that one interpreter, so nothing the runner spawns
        // reads the directory it left — the line is the runner's.
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
        // …but a step of its own after the move is spawned in the tracked directory:
        // an own image call, and its conversion out of an open fold.
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
        // A directory change inside one of the shell's own keyword forms: whether it
        // runs at all is the keyword's condition to decide, so which directory the
        // line is left in is unknown wherever a later step would need it.
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
        // The spellings the verb key does read are tracked, not refused — a `cd`
        // with a spaced target, case-folded, and its `chdir` alias.
        for (member, tracked) in [
            ("cd src", r"C:\ws\src"),
            ("chdir src", r"C:\ws\src"),
            ("CD \\ws", r"C:\ws"),
        ] {
            let plan = plan(&[shell(member, "&&"), served(tracked, "")]);
            assert_eq!(plan.steps[1].cwd, PathBuf::from(tracked), "{member}");
        }
    }

    /// The redirect set the runner applies by itself, and the spellings it
    /// refuses with the spelling named.
    #[test]
    fn redirects_supported_is_the_closed_set_it_documents() {
        for supported in [
            vec![">", "out.txt"],
            vec!["1>", "out.txt"],
            vec![">>", "out.txt"],
            vec!["1>>", "out.txt"],
            vec!["2>", "err.txt"],
            vec!["2>>", "err.txt"],
            vec!["<", "in.txt"],
            vec![">", "nul"],
            vec!["2>", "NUL"],
            vec![">", "out.txt", "2>", "err.txt", "<", "in.txt"],
            // A target glued to its operator is the same redirect — cmd.exe splits
            // neither apart, and this is the spelling an agent actually types.
            vec![">out.txt"],
            vec!["1>>log.txt"],
            vec!["2>err.txt"],
            vec!["2>>err.txt"],
            vec!["<in.txt"],
            vec!["1>0"],
            // A drive-rooted target is the runner's file like any other path: only
            // the drive-RELATIVE spelling below names a directory it cannot know.
            vec![">", r"C:\out.txt"],
            vec![">", r#""C:\out.txt""#],
            // The stdout merge: self-contained (no target word of its own), and
            // beside a `>` its destination is that file, not the caller's.
            vec!["2>&1"],
            vec![">", "out.txt", "2>&1"],
            vec!["2>&1", ">", "out.txt"],
        ] {
            let redirects: Vec<String> = supported.iter().map(|w| (*w).to_string()).collect();
            assert_eq!(redirects_supported(&redirects), Ok(()), "{supported:?}");
        }
        for refused in [
            vec!["1>&2"],
            vec![">&2"],
            vec![">>"],
            vec!["<"],
            vec!["10>", "out.txt"],
            vec![">"],
            vec!["<&2"],
            vec!["<>file"],
            vec![">", r"%TEMP%\out.txt"],
            vec![r">%TEMP%\out.txt"],
            // A drive-relative target names another drive's current directory —
            // quoted or not, since cmd.exe's quote removal is applied before the
            // file is opened.
            vec![">", "C:out.txt"],
            vec![">C:out.txt"],
            vec![">", r#""C:out.txt""#],
            vec![r#">"C:out.txt""#],
            // Quote removal can also sit between the colon and the path
            // (`"C:""out.txt"` → `C:"out.txt`), which is drive-relative the same way.
            vec![">", r#""C:""out.txt""#],
            vec!["<", "C:in.txt"],
            // A dup of another descriptor: only the stdout merge has a
            // destination the runner already owns.
            vec!["3>&1"],
        ] {
            let spellings: Vec<String> = refused.iter().map(|w| (*w).to_string()).collect();
            let reason = redirects_supported(&spellings).expect_err("outside the set");
            assert!(
                reason.contains(&format!("`{}`", refused[0])),
                "{refused:?}: {reason}"
            );
        }
    }

    /// The pipe-group split: `|` continues a group, every other join starts one.
    #[test]
    fn pipe_groups_split_on_every_join_but_a_pipe() {
        let plan = plan(&[
            shell("echo one", "&&"),
            served(ROOT, "|"),
            shell("head -1", "&&"),
            served(ROOT, "||"),
            shell("tail -1", ""),
        ]);
        let groups: Vec<Range<usize>> = pipe_groups(&plan.steps);
        assert_eq!(groups, vec![0..1, 1..3, 3..4, 4..5]);
        assert_eq!(pipe_groups(&[]), Vec::<Range<usize>>::new());
    }

    /// The own-image reading: command position and a launcher count, argument
    /// position does not. A path with a space in it is one word only where the
    /// line quotes it — cmd.exe splits the bare spelling, so the bare one names
    /// no program at all (and is deliberately not read as this image).
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
            // A path spelling of the same file name: the word names the image's
            // name through a path, which the shell would have to resolve (and would
            // not wait for either way), so the member is the runner's to refuse or
            // run — never the shell's to start.
            r".\mahbot.exe -V",
            r"target\debug\mahbot.exe -V",
            r"C:\somewhere\else\mahbot -V",
            r"start .\mahbot.exe",
            // The connectors cmd.exe reads: a command starts after each of them, so
            // the call is the no-wait call wherever in the line it sits. The quoted
            // `"&"` below is an argument, not a connector.
            "cd sub && mahbot -V",
            "type f.txt | mahbot -V",
            "echo hi & mahbot -V",
            // cmd.exe's own punctuation: the `@` no-echo prefix and a group's
            // parentheses are not command words, and the word after them is still
            // where a command can start.
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
            // A quoted connector is an argument like any other.
            r#"echo "&&" mahbot"#,
            // cmd.exe's punctuation in argument position is ordinary text, and so
            // is a quoted `@`/`(`: nothing there starts a command.
            "echo ( mahbot -V )",
            "echo @ mahbot -V",
            // A word a launcher only *carries* in its command is an ordinary
            // argument: `cmd /c dir mahbot` lists the directory, `powershell
            // Get-Process mahbot` asks for a process, and `start` hands its command
            // to `notepad`.
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

    /// A line that is one own-image call becomes a one-step plan, with the
    /// member's own argv and redirect tokens.
    #[test]
    fn own_image_plan_reads_a_lone_call() {
        let exe = Path::new(EXE);
        let cwd = Path::new(ROOT);
        let OwnImage::Direct(direct) = own_image_plan(
            r#""C:\Program Files\MahBot\mahbot.exe" debug --db board "select 1""#,
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
            r#""C:\Program Files\MahBot\mahbot.exe" -V > out.txt"#,
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

        // The glued spelling of the same redirect is the same call, and it is the
        // one an agent types.
        let OwnImage::Direct(glued) = own_image_plan(
            r#""C:\Program Files\MahBot\mahbot.exe" -V >out.txt"#,
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

    /// A line that names the image among other members is decomposed with the
    /// analyzer's own cmd.exe model: every plain call of the image is an own step
    /// the runner spawns itself, every other member is one the folds around it
    /// run, and the cwd is tracked across the members.
    #[test]
    fn own_image_plan_decomposes_a_compound_line() {
        // `cd sub && <image> --version && echo done`: the `cd` is the first
        // fold's, the call is the runner's, and the fold after it runs where the
        // call ran — the directory the `cd` tracked.
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

        // `<image> -V | head -3`: one pipe group — the call and the tail the
        // runner feeds it to.
        let plan = direct_plan(&format!(r#""{EXE}" -V | head -3"#));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::Pipe);
        assert_eq!(pipe_groups(&plan.steps), vec![0..2]);
        assert_eq!(shell_text(&plan.steps[1]), "head -3");
        assert_eq!(step_cwds(&plan), vec![root.as_path(), root.as_path()]);

        // `<image> debug --db board "SELECT 1" > out.txt`: the call travels as
        // argv and its target is the runner's own file.
        let plan = direct_plan(&format!(r#""{EXE}" debug --db board "SELECT 1" > out.txt"#));
        let Some((args, redirects, _)) = plan.direct_own() else {
            panic!("a lone call is one own step");
        };
        assert_eq!(args, &["debug", "--db", "board", "SELECT 1"]);
        assert_eq!(redirects, &[">".to_string(), "out.txt".to_string()]);

        // `<image> -V 2>&1 | head -2`: the merge rides the own step, and the tail
        // is the shell's member of the same pipe group.
        let plan = direct_plan(&format!(r#""{EXE}" -V 2>&1 | head -2"#));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].join, Join::Pipe);
        let Run::Own { args, redirects } = &plan.steps[0].run else {
            panic!("the call of the image is an own step");
        };
        assert_eq!(args, &["-V"]);
        assert_eq!(redirects, &["2>&1".to_string()]);
    }

    /// A shape the generic decomposition cannot represent is refused through the
    /// plan's own refusals, never handed to a shell that does not wait for the
    /// image.
    #[test]
    fn an_unrepresentable_own_image_line_is_refused() {
        // A launcher spelling: the member names the image, but not in the command
        // position the runner can spawn.
        for command in ["start mahbot -V", "cmd /c mahbot bench-openrouter"] {
            let cause = refusal_cause(command);
            assert!(cause.contains("runs `mahbot` itself"), "{command}: {cause}");
        }
        // A dup the runner has no destination for, on a member it spawns itself.
        let cause = refusal_cause("mahbot -V >&2");
        assert!(cause.contains("`>&2`"), "{cause}");
        // A `cd` the tracking cannot follow: the call after it would run in a
        // directory the shell would not have used.
        let cause = refusal_cause("cd /x && mahbot -V");
        assert!(cause.contains("`cd` member"), "{cause}");
        // A `%…%` argument cmd.exe would have expanded before the image saw it:
        // the member is left to the shell, which is the no-wait call again.
        let cause = refusal_cause("mahbot -V %TEMP%");
        assert!(cause.contains("runs `mahbot` itself"), "{cause}");
        // A call the line *opens* with in a line the segmenter refused, whose text
        // carries no spelling of the image's path: the bare name is read in
        // command position, so the call is refused rather than handed to a shell
        // that does not wait for it.
        let cause = refusal_cause("mahbot -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        // The same line, with the call further along it: the reading walks the
        // connectors the tokenizer still sees next to the words a command can start
        // at — and the path spelling catches what it cannot see.
        let cause = refusal_cause("cd sub && mahbot -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        let cause = refusal_cause(r"cd sub && C:\Program Files\MahBot\mahbot.exe -V (x)");
        assert!(cause.contains("cannot be read"), "{cause}");
        // cmd.exe's own punctuation does not hide a call: the `@` no-echo prefix
        // (whose launcher form hands the image to a shell that does not wait) and a
        // group's parentheses, in both spellings the model can be given.
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
                matches!(own_image_plan(command, exe, cwd), OwnImage::None),
                "{command}"
            );
        }
        // On the platform whose shell waits for its children there is nothing to
        // plan and nothing to refuse: the gate is [`plan_for_command`]'s, the one
        // dispatch the callers use.
        if crate::tools::shell::SHELL_PLATFORM == ShellPlatform::Unix {
            assert!(matches!(
                plan_for_command("mahbot -V && echo done", cwd),
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
        // spelled with this service's own call: the two readings agree member for
        // member.
        let generic = direct_plan(&format!(r#"cd src && "{EXE}" -V | head -3"#));
        assert_eq!(generic.steps.len(), plan.steps.len());
        assert_eq!(generic.steps[1].cwd, generic.steps[2].cwd);
        assert!(generic.steps[2].cwd.ends_with("src"));
    }

    // ── The executor (unix lane: the plan's steps are plain data, so the
    //    mechanics are driven here with `sh` members) ──────────────────────

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

    /// A group runs only when cmd's connector says it should, and a skipped group
    /// leaves the previous status in place.
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

    /// Two members of one pipe group are connected by a pipe the runner owns: the
    /// producer's whole stream reaches the consumer, and the run reports the
    /// consumer's status (cmd's rule for a pipeline's last member).
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

    /// The pipe's input is the head's own stdout alone: the head runs as a step of
    /// its own, so what the group before it printed stays in the run's capture and
    /// does not reach the consumer. A fold around the head would have fed the
    /// consumer `hi` first, and `head -1` would have printed that.
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

    /// A pipeline is gated by its head's connector: an unsatisfied `&&` before the
    /// head skips the whole group and leaves the run on the group before it. A fold
    /// around the head would have run the pipeline anyway and reported its last
    /// member's status.
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

    /// The run's streams are its members' output in execution order, and the
    /// status is the last executed group's last member's.
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

    /// `steps` as a plan of own-image steps running `sh` — one host program
    /// spawned as argv, the way an own step spawns this service's image — so the
    /// redirect and pipe wiring of the member the runner spawns itself is driven
    /// here too.
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

    /// `2>&1` on a member whose stdout the run captures: the member's stderr is
    /// its stdout destination, so its bytes join the run's captured stdout and
    /// nothing is left on the stderr channel.
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

    /// `2>&1` on a member of a pipe group: both of the member's streams are
    /// pumped into the next member's input through one shared handle, and the
    /// consumer still sees the end of that input when the last pump drops it.
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

    /// `> out.txt 2>&1`: the merge follows the stdout the member's own `>` left,
    /// so both streams are the one file — not the run's capture, which the merge
    /// would have taken had it come first.
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

    /// `2> err.txt 2>&1`: the merge comes later, so it re-points stderr at stdout's
    /// destination — the caller's capture — exactly as cmd.exe's left-to-right
    /// application does. The file the earlier redirect opened was created and
    /// truncated before the merge took it away, which is what cmd.exe did too.
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
        assert_eq!(stdout.len(), SHELL_PIPE_READ_CAP);
        assert_eq!(String::from_utf8_lossy(&stderr), "marker\n");
    }

    /// `2>&1 > out.txt` on a member of a pipe group: the merge came first, so the
    /// pipe was already the stderr's destination when the `>` took stdout over —
    /// the consumer reads the merged stream while the member's stdout goes to the
    /// file.
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

    // ── The executor's stops and failures: the single-child runner's own
    //    bounds and failure paths, pinned for the plan that carries them ──────

    /// A run past its deadline ends the whole run and reports a `TimedOut` naming
    /// the group it waited for — with what the members had already written still
    /// collected.
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

    /// A member that leaves a descendant holding the capture pipe open past the
    /// drain bound ends the run as `DrainTimedOut` rather than hanging on a stream
    /// nothing will close — the same bound, and the same variant, the single-child
    /// runner gives.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_leftover_holder_past_the_drain_bound_ends_the_run() {
        let result = run(
            &shell_plan(&[("echo out; sleep 30 &", Join::First)]),
            Duration::from_secs(20),
            Duration::from_millis(300),
            None,
            RunOwner::Agent,
        )
        .await;
        let ShellRunResult::DrainTimedOut { stdout, .. } = result else {
            panic!("expected a drain timeout, got {result:?}");
        };
        assert!(
            String::from_utf8_lossy(&stdout).contains("out"),
            "what the member wrote before the bound is collected: {}",
            String::from_utf8_lossy(&stdout)
        );
    }

    /// A member that cannot be spawned fails the run as `SpawnFailed` — the same
    /// variant the single-child runner produces — and ends the tree rather than
    /// leaving what was already running behind.
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

    /// A redirect the runner cannot open for a later member of a group ends the
    /// group exactly as a failed spawn does: the members already spawned are killed
    /// and reaped, so nothing keeps running behind a run that has already failed.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_redirect_that_cannot_be_opened_ends_the_group() {
        let dir = std::env::temp_dir();
        let marker = dir.join(format!("plan-cleanup-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let target = dir.join(format!("plan-missing-{}/out.txt", std::process::id()));
        let result = run_plan(&program_plan(&[
            // A member that would write its marker a second from now: the marker's
            // absence is the kill having reached it.
            (
                &["-c", &format!("sleep 1; echo done > {}", marker.display())],
                &[],
                Join::First,
            ),
            // Its own redirect names a directory that does not exist, so the runner
            // cannot open it — while the member before it is already running, the
            // pipe being the two members' connect.
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

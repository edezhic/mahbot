//! Whole-tree containment for a command run.
//!
//! Ending a run must end everything the run started. A stop — its timeout, the
//! point where its output collection is given up on
//! ([`super::ShellRunResult::DrainTimedOut`]) or the run being torn down (its tool
//! call dropped, its agent destroyed) — takes the whole tree on both platforms.
//! An abrupt death of the service takes it too, but only where something holds the
//! tree on the run's behalf: a background session's watcher or job handle (`bg`)
//! and, on Windows, every run. Two cases are not a stop:
//!
//! - a *foreground* run that finished on its own is left alone; its stragglers
//!   outlive both the command and, on unix, the service, exactly as they always
//!   did. Windows keeps the containment in force instead, so nothing it left
//!   behind can outlive the service ([`Tree::retain_after_completion`]).
//! - a *background session* belongs to the agent rather than to a tool call, so
//!   its leftovers go when its command does, on both platforms (`bg`: the unix
//!   watcher's group kill when the lifeline closes, this module's `terminate` on
//!   Windows).
//!
//! The two platforms reach all of that with their own primitive, and the split
//! between them is deliberate — a shared seam over the two would put the unix
//! behaviour at risk.
//!
//! - **unix**: the child leads its own process group from the moment it is
//!   spawned (`process_group(0)` in [`super::build_shell_command`]), so
//!   [`Tree::terminate`] is one `kill(-pgid, SIGKILL)` per member the run
//!   attached — best-effort, a failure is
//!   logged — reaching every descendant. The attached members are one shared
//!   list, not a per-clone snapshot, so a kill guard built before the spawns
//!   still ends every member attached afterwards ([`Tree::attach`]).
//! - **windows**: one fresh job object per run, created with
//!   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, and each just-spawned member assigned
//!   to it ([`Tree::attach`]). Descent is inheritance — the child of a job member
//!   joins the job — and breakaway is deliberately not permitted, so
//!   `CREATE_BREAKAWAY_FROM_JOB` cannot hand a descendant to another job.
//!   [`Tree::terminate`] ends the job with `TerminateJobObject`; nothing here asks
//!   politely. The handle is the guarantee: closing the last one ends every process
//!   in the job, which is what an abrupt death of the service does.
//!
//! A run is one child for the shell tool's ordinary path and one member per plan
//! step for a rewritten line ([`super::plan`]); both attach every process they
//! spawn to one tree, so a stop takes whichever shape the run had.
//!
//! # Which runs are contained (windows)
//!
//! [`RunOwner`] draws the boundary and it is the *only* thing the two owners
//! differ in. [`RunOwner::Agent`] — the shell tool in both modes, which is also
//! the path a ticket's diagnostics commands take, and the `custom` tool's script
//! run — gets the job. [`RunOwner::Service`] — a user's alarm program — does
//! not: it is not an agent's work and keeps exactly the treatment it had before
//! job containment existed. Everything the service launches for its own
//! operation — the browser, the replacement instance a self-update starts, the
//! browser-automation command-line tool and the helper process it forks, the
//! bun runtime's own install/update/probe runs, git — never comes through this
//! module.
//!
//! # No console window (windows)
//!
//! A process this service starts must never put a console window on the screen.
//! Spawning with `CREATE_NO_WINDOW` hands the child a console that has no window
//! — and any process the child starts in turn inherits that console unless it
//! asks for its own — so for a console-program child the guarantee covers
//! descendants too. (That inheritance half is the platform's documented console
//! rule, reasoned here rather than observed: no Windows host runs in this
//! project's lane. macOS/Linux need nothing — a process started there cannot
//! create a window. The images the flag cannot act on are named below.)
//!
//! A parent with no console at all hands every console child it starts without
//! creation flags a brand-new **visible** console. The replacement instance is
//! therefore never created detached: it gets `CREATE_NO_WINDOW` like every other
//! spawn, and its own children get the same flag from it — the flag is inert for
//! the instance itself (see below), and it is what keeps the console programs it
//! starts in turn from putting a window on the screen. The rationale for the
//! instance's own spawn is in `self_update::spawn_new_instance_from`.
//!
//! Three of the images those sites launch are not console programs at all — the
//! browser, the file manager, and this service's own image — and the flag is inert
//! for them: the platform gives such a program no console in the first place, so
//! `CREATE_NO_WINDOW` has nothing to suppress. The service is that kind of image
//! (it is built for the windowed subsystem, `windows_subsystem = "windows"`). Its
//! own sites — the grep engine's probe, the runner's own-image steps
//! ([`super::plan`], through the same builders) and the replacement instance —
//! still set the flag, and what such an image starts in turn is outside the
//! service's reach, exactly as it is for the browser and the file manager. The one
//! spawn of this image that is not the service's to flag — the `self-replace` crate
//! starts its file-swap copy itself — is the residual below. What changes for such
//! an image is that it asks for no console of its own: a launch with no console has
//! none to hand on, while one started from a console hands that console over, and
//! the flag is inert for a windowed image either way.
//!
//! Covered, one entry per spawn site: git (`git::commands::git_command`); the
//! cargo probe and both install modes (`self_update::verify_cargo_on_path`,
//! `self_update::run_cargo_with_timeout`); the replacement instance
//! (`self_update::spawn_new_instance_from`); the bun runtime's probe
//! (`tools::bun::bun_cli_version`); the browser-automation CLI in all its runs —
//! probe, version, `tasklist`, install and restart
//! (`tools::chrome_daemon::cli_probe`, `cli_version`, `tasklist_has`,
//! `install_chrome_use`, `run_cli`) plus the shared `chrome::spawn::spawn_cli` —
//! and the browser itself (`tools::chrome_daemon::spawn_chrome_detached`); every
//! agent command through the shell builders (`tools::shell::build_shell_command`,
//! `tools::shell::build_program_command` — the shell tool, the diagnostics
//! runner, the `custom` tool's script, a user's alarm program, and every member
//! of a rewritten line's plan ([`super::plan`], which spawns its own-image steps
//! through that same builder rather than a site of its own) plus the grep
//! engine's own probe (`tools::shell::grep_engine::probe_engine`); and the file
//! manager the owner asked for (`gui::editor::perform_reveal_in_finder`).
//!
//! Not covered, on purpose: the terminal the owner opens inside the app
//! (`gui::shell`, spawned through a pseudoconsole) and the owner's own browser
//! (`gui::open_url` hands the address to `ShellExecuteExW` in process, so no
//! console program exists to flash) — their windows are the owner's business.
//! Residual, stated rather than implied: a command that asks the platform for a
//! window of its own (`CREATE_NEW_CONSOLE`, or `cmd`'s `start` builtin) gets one,
//! and so does anything it starts; a console program started by one of the three
//! non-console images (the browser, the file manager, this service's own image)
//! is beyond this flag's reach; and an un-flagged console child of a console-less
//! parent gets a new visible console. The `self-replace` crate's file-swap copy of
//! this service's own binary is an un-flagged child of a console-less parent, and it
//! is the windowed subsystem, not a creation flag, that keeps it off the screen: the
//! copy is this service's image, so the platform gives it no console to show. An
//! instance launched by an earlier build runs the older, console-program image for
//! its lifetime, so its own file-swap copy can still show one window during an
//! update started by it.
//!
//! A console program this service starts therefore owns a console of its own — one
//! extra hidden `conhost.exe` per spawn — and it is not attached to the console its
//! parent would otherwise have handed it, so console control events on that console
//! (a console-window close) never reach it. Whether such a child can still *use* a
//! console handle it inherited from this process is not something this host can
//! observe: the handle values it receives are unchanged, but they belong to another
//! console.
//!
//! No usable standard input either: an agent's command inherits no console for its
//! input — a console-less process has none to hand on — so a command that would
//! have read the console (`set /p`, `pause`, `more`, anything interactive) has
//! none instead of waiting for the owner's keystrokes. The shell tool never hands
//! a command the owner's prompt, and its background form gives the command no
//! input at all (null stdin, unless the command's own redirect took it over). What
//! a command must read it takes from a file (`<`), from a pipe (`… | cmd`), or from
//! a redirection the runner applies itself ([`super::plan`]).
//!
//! The guarantee is a per-spawn flag, so a new spawn site could be added without
//! one and nothing would say so: `every_production_spawn_is_windowless` below is
//! the tripwire, and the count of sites it insists on, and the names it reads back
//! out of the inventory above, are the covered list itself.
//!
//! # When containment cannot be established (windows)
//!
//! The assignment can be refused — a job the platform cannot nest ours under, a
//! security limit, a job that is already terminating. The run then proceeds
//! exactly as it did before, without added delay, and the reason is written to
//! the service log: visible, never silent, and never an error the caller has to
//! handle.
//!
//! # Accepted limitations (windows)
//!
//! - Between the child's spawn and its assignment to the job there is a window
//!   in which it can start a process that never joins the job; such a runaway
//!   behaves exactly as it did before this containment existed. Closing the
//!   window needs the process created suspended and resumed through its primary
//!   thread handle, which the standard library does not expose. A command short
//!   enough to finish inside that window is logged the same way a refused
//!   assignment is: either way the run has no containment, and the pid of a
//!   command that is already gone is not worth a different treatment in the log.
//! - The guarantee is termination, not a graceful request: there is no `SIGTERM`
//!   equivalent to send first and no grace period to give.
//! - A program that expects to detach itself from whatever launched it, or that
//!   manages this kind of containment itself, may behave differently as an agent
//!   command: breakaway is not permitted, so a child it creates with
//!   `CREATE_BREAKAWAY_FROM_JOB` fails to start. Giving itself — or a child — a
//!   job of its own still succeeds through job nesting, which costs it nothing:
//!   the process is in both jobs, so ending ours ends it too (terminating a job
//!   terminates the jobs nested inside it). A browser-automation helper process an
//!   agent starts by running that tool from a command line is such a descendant,
//!   and the service's own launch exemption cannot reach it.

// The unix half's shared pid list needs the poison-tolerant lock helper.
#[cfg(unix)]
use crate::util::UnwrapPoison;

// The job-object half is the only thing below that needs any of these.
#[cfg(not(unix))]
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
#[cfg(not(unix))]
use std::sync::Arc;
#[cfg(not(unix))]
use std::sync::atomic::{AtomicBool, Ordering};

/// Who owns a run — on Windows the only thing that decides whether the run is
/// contained; the boundary is the module docs' `Which runs are contained`.
///
/// `Copy` keeps `clippy::needless_pass_by_value` quiet on [`Tree::new`]'s
/// by-value owner, whose Windows match only looks at the variant.
#[derive(Clone, Copy)]
pub(super) enum RunOwner {
    /// An agent's work: gets the job.
    Agent,
    /// A user's alarm program — `alarms::run_program_outcome` is its only caller.
    /// Never given a job; unix contains it like any other run, because there the
    /// process group belongs to the spawn rather than to the owner.
    Service,
}

// ── Unix ─────────────────────────────────────────────────────────────

/// Unix containment is the process group the child leads from the moment it is
/// spawned — see the module docs.
#[cfg(unix)]
#[derive(Clone)]
pub(super) struct Tree {
    /// The pids of the run's members, each of which `process_group(0)` also made
    /// its own process group's id. Empty until [`Tree::attach`], and after a spawn
    /// that produced no pid at all. A single-child run attaches one (the shell
    /// tool's child); a plan run attaches one per member it spawns
    /// ([`super::plan`]), every one of them a group leader of its own.
    ///
    /// Shared by every clone a run holds, like the Windows job: the plan runner
    /// builds its kill guard before it spawns, so a member attached afterwards
    /// must still be seen by that guard.
    pids: std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
}

#[cfg(unix)]
impl Tree {
    /// Nothing to set up per run before the spawn: the group is a property of
    /// the spawn itself. `_owner` decides the Windows job only, so both owners
    /// are contained the same way here — deliberately.
    pub(super) fn new(_owner: RunOwner) -> Self {
        Self {
            pids: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Record one spawned member's process group.
    pub(super) fn attach(&mut self, pid: u32) {
        self.pids.lock().unwrap_poison().push(pid);
    }

    /// End every process in the run's groups — the direct children included,
    /// which is why a caller that got `true` only has to reap them. `true` means
    /// the groups were signalled, not that every member died: a signal to a group
    /// with no live member left is not an error here
    /// ([`super::kill_process_group`] logs its own failure).
    ///
    /// A member that has already been reaped stays in the list: its process group is
    /// where anything it left running lives, which is what a stop after the reap is
    /// for. A group that is fully gone leaves its pgid free for reuse, so a signal
    /// meant for it can land elsewhere — an accepted window, not the purpose of the
    /// retention, and the alternative (forgetting the pid) is what loses the leftover
    /// kill the drain path relies on.
    pub(super) fn terminate(&self) -> bool {
        for pid in self.pids.lock().unwrap_poison().iter() {
            super::kill_process_group(*pid, libc::SIGKILL);
        }
        true
    }

    /// Nothing to keep: the group is the kernel's, not a handle we own, so it
    /// cannot be lost by ending the run.
    #[expect(
        clippy::unused_self,
        reason = "the shared call shape consumes the tree; unix has nothing to keep"
    )]
    pub(super) fn retain_after_completion(self) {}
}

// ── Windows ──────────────────────────────────────────────────────────

/// Windows containment is a job object — see the module docs.
#[cfg(not(unix))]
#[derive(Clone)]
pub(super) struct Tree {
    /// `None` when the run has no containment: a service launch
    /// ([`RunOwner::Service`]), or a job that could not be created or assigned
    /// (fail-open — see the module docs).
    job: Option<Arc<Job>>,
}

#[cfg(not(unix))]
impl Tree {
    /// Create the run's job. A job that cannot be created is not a failure to
    /// run: the run proceeds uncontained, with a warning naming the reason.
    pub(super) fn new(owner: RunOwner) -> Self {
        let job = match owner {
            RunOwner::Service => None,
            RunOwner::Agent => match Job::create() {
                Ok(job) => Some(Arc::new(job)),
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "cannot create a job object for this command — it runs without \
                         process-tree containment, as commands did before containment"
                    );
                    None
                }
            },
        };
        Self { job }
    }

    /// Put the just-spawned child in the job, so that every descendant it starts
    /// is in it too. A refusal is fail-open: the run keeps the behaviour it had
    /// before job containment existed, and the reason is logged.
    pub(super) fn attach(&mut self, pid: u32) {
        let Some(job) = &self.job else {
            return;
        };
        if let Err(e) = job.assign(pid) {
            tracing::warn!(
                pid,
                err = %e,
                "cannot put this command's process tree under containment — it may \
                 have finished before the assignment, and anything it left running \
                 may outlive it, as it did before containment"
            );
            self.job = None;
        }
    }

    /// End every process in the run's job — the direct child included, which is
    /// why a caller that got `true` only has to reap it. `false` means the tree
    /// was not ended: the run has no job at all (fail-open, see the module docs)
    /// or the termination call failed, so a caller that still holds the child has
    /// to end it itself. [`super::KillOnDrop`] does not — it holds only the tree —
    /// so a run dropped in that case leaves its direct child running, exactly as
    /// it did before this module existed.
    pub(super) fn terminate(&self) -> bool {
        let Some(job) = &self.job else {
            return false;
        };
        job.terminate()
    }

    /// Keep the job's handle for the rest of the process lifetime: a run that
    /// finished is not a stop, but its handle must not be closed while the job
    /// still holds a process (see the module docs). A job that holds nothing is
    /// closed instead — there is nothing for the handle to keep in.
    ///
    /// The handle is therefore leaked on purpose, one per completed command that
    /// left a live descendant. Dropping it would give up the only thing standing
    /// between those processes and outliving the service, so the leak is the
    /// feature — do not "fix" it.
    pub(super) fn retain_after_completion(self) {
        let Some(job) = self.job else {
            return;
        };
        if !job.holds_processes() {
            return;
        }
        std::mem::forget(job);
    }
}

#[cfg(not(unix))]
struct Job {
    handle: OwnedHandle,
    /// Set once `TerminateJobObject` has succeeded, and never cleared: a later
    /// call — the waiter after an explicit stop, the kill-on-drop guard — is then
    /// a cheap no-op instead of a second call that could log a second failure,
    /// and a concurrent failed call cannot undo a success.
    ended: AtomicBool,
}

#[cfg(not(unix))]
impl Job {
    /// A fresh, empty, unnamed job, carrying the limit the module docs describe.
    fn create() -> std::io::Result<Self> {
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        // SAFETY: an unnamed job (a null name) with default security (no
        // SECURITY_ATTRIBUTES) is the documented anonymous shape.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let job = Self {
            // SAFETY: the call just returned this owned handle; nothing else
            // holds it.
            handle: unsafe { OwnedHandle::from_raw_handle(handle as _) },
            ended: AtomicBool::new(false),
        };
        // SAFETY: all-zero is a valid value for every field of the struct, and
        // the limit flag below is the only one set.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: the class names the struct passed and the length is exactly
        // that struct's size.
        let configured = unsafe {
            SetInformationJobObject(
                job.raw(),
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                u32::try_from(std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                    .expect("job limits fit in u32"),
            )
        };
        if configured == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    fn raw(&self) -> windows_sys::Win32::Foundation::HANDLE {
        self.handle.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE
    }

    /// Assign a process to the job. The process is named by the pid of a child
    /// we just spawned — its `Child` keeps its own process handle private, and a
    /// pid that names a live process cannot name anything else.
    fn assign(&self, pid: u32) -> std::io::Result<()> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
        };

        // SAFETY: a plain open by pid — no pseudo-handle — asking for the two
        // rights `AssignProcessToJobObject` requires.
        let process = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid) };
        if process == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: both handles are live for the duration of the call.
        let assigned = unsafe { AssignProcessToJobObject(self.raw(), process) };
        // The last error must be read before anything else can overwrite it.
        let failure = std::io::Error::last_os_error();
        // SAFETY: `process` came from the `OpenProcess` above and is not used
        // again.
        unsafe { CloseHandle(process) };
        if assigned == 0 {
            return Err(failure);
        }
        Ok(())
    }

    /// `TerminateJobObject`: end every process in the job, the direct child
    /// included, and report whether the call succeeded. A job already ended this
    /// way is reported as ended without a second call; a failed call is not
    /// remembered, so a caller that falls back is free to retry.
    fn terminate(&self) -> bool {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        if self.ended.load(Ordering::SeqCst) {
            return true;
        }
        // SAFETY: the handle is live and owned by `self`. The exit code is ours
        // to choose; 1 is a non-zero code, which is all a process ended this way
        // can report — Windows has no signal for it to show in its exit status.
        let terminated = unsafe { TerminateJobObject(self.raw(), 1) };
        if terminated == 0 {
            tracing::warn!(
                err = %std::io::Error::last_os_error(),
                "TerminateJobObject failed — this command's processes may survive"
            );
        } else {
            // Only ever set, never cleared: a failure here cannot undo a success
            // another thread already latched.
            self.ended.store(true, Ordering::SeqCst);
        }
        terminated != 0
    }

    /// Whether the job still holds a live process — the question
    /// [`Tree::retain_after_completion`] asks before deciding the handle is
    /// worth keeping. A query that fails reports `true`: keeping a handle we
    /// cannot inspect is the safe direction, dropping the only handle to
    /// processes that may still be alive is not.
    fn holds_processes(&self) -> bool {
        use windows_sys::Win32::System::JobObjects::{
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JobObjectBasicAccountingInformation,
            QueryInformationJobObject,
        };

        // SAFETY: all-zero is a valid value for every field of the struct; the
        // call fills it.
        let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: the class names the struct passed, the length is exactly that
        // struct's size, and no return length is asked for.
        let queried = unsafe {
            QueryInformationJobObject(
                self.raw(),
                JobObjectBasicAccountingInformation,
                std::ptr::from_mut(&mut accounting).cast(),
                u32::try_from(std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>())
                    .expect("job accounting fits in u32"),
                std::ptr::null_mut(),
            )
        };
        queried == 0 || accounting.ActiveProcesses > 0
    }
}

#[cfg(test)]
mod tests {
    /// Files that are test-only end to end — `util/test.rs` is `#![cfg(test)]`
    /// and `db/store_lock_check.rs` is included as `#[cfg(all(unix, test))]`.
    /// Their spawns are test scaffolding, not commands the service starts.
    const TEST_ONLY_FILES: [&str; 2] = ["src/util/test.rs", "src/db/store_lock_check.rs"];

    /// The verdict on a `#[cfg(...)]` predicate, on the question "can this code
    /// be the windows production path?" — `No` is the only verdict that lets the
    /// sweep ignore the code inside. Anything it cannot decide (a feature, a
    /// debug assertion) is `Unknown`, and swept: a gate this scanner does not
    /// understand must never be able to hide a spawn from it.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Gate {
        Yes,
        No,
        Unknown,
    }

    /// One leaf of a predicate. `unix` is never windows, `test` is never the
    /// production build, and a `target_os`/`target_family` names windows or not.
    fn gate_atom(predicate: &str) -> Gate {
        let p = predicate.trim();
        if p == "windows" {
            return Gate::Yes;
        }
        if p == "unix" || p == "test" {
            return Gate::No;
        }
        if p.starts_with("target_os") || p.starts_with("target_family") {
            return if p.contains("\"windows\"") {
                Gate::Yes
            } else {
                Gate::No
            };
        }
        Gate::Unknown
    }

    /// The contents of `name(...)`, when the predicate is exactly that call.
    fn gate_bound<'a>(predicate: &'a str, name: &str) -> Option<&'a str> {
        predicate
            .strip_prefix(name)?
            .strip_prefix('(')?
            .strip_suffix(')')
    }

    /// Split `all`/`any` arguments at nesting depth zero.
    fn gate_args(predicate: &str) -> Vec<&str> {
        let mut args = Vec::new();
        let mut depth = 0_usize;
        let mut quoted = false;
        let mut start = 0_usize;
        for (i, c) in predicate.char_indices() {
            match c {
                '"' => quoted = !quoted,
                '(' if !quoted => depth += 1,
                ')' if !quoted => depth = depth.saturating_sub(1),
                ',' if !quoted && depth == 0 => {
                    args.push(&predicate[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        args.push(&predicate[start..]);
        args.into_iter().map(str::trim).collect()
    }

    /// The verdict on a `#[cfg(...)]` predicate by [Kleene] three-valued logic,
    /// its leaves being the ones [`gate_atom`] knows.
    ///
    /// [Kleene]: https://en.wikipedia.org/wiki/Three-valued_logic
    fn gate(predicate: &str) -> Gate {
        let p = predicate.trim();
        if let Some(inner) = gate_bound(p, "not") {
            return match gate(inner) {
                Gate::Yes => Gate::No,
                Gate::No => Gate::Yes,
                Gate::Unknown => Gate::Unknown,
            };
        }
        if let Some(inner) = gate_bound(p, "all") {
            let mut result = Gate::Yes;
            for arg in gate_args(inner) {
                match gate(arg) {
                    Gate::No => return Gate::No,
                    Gate::Unknown => result = Gate::Unknown,
                    Gate::Yes => {}
                }
            }
            return result;
        }
        if let Some(inner) = gate_bound(p, "any") {
            let mut result = Gate::No;
            for arg in gate_args(inner) {
                match gate(arg) {
                    Gate::Yes => return Gate::Yes,
                    Gate::Unknown => result = Gate::Unknown,
                    Gate::No => {}
                }
            }
            return result;
        }
        gate_atom(p)
    }

    /// The one creation flag every covered spawn sets.
    const WINDOWLESS_FLAG: &str = "CREATE_NO_WINDOW";

    /// `CREATE_NEW_CONSOLE` is a window by construction; `DETACHED_PROCESS` is
    /// forbidden for the reason the module docs' guarantee section gives.
    const FORBIDDEN_FLAGS: [&str; 2] = ["CREATE_NEW_CONSOLE", "DETACHED_PROCESS"];

    /// The number of spawn sites the sweep must see — the module docs' inventory,
    /// counted. A site that constant and that inventory do not account for fails
    /// the sweep, and a scan that stopped reading the tree finds far fewer.
    const SPAWN_SITES: usize = 16;

    fn indent(line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    /// The predicate of a `#[cfg(...)]` attribute, when the line carries one.
    fn cfg_predicate(line: &str) -> Option<&str> {
        let rest = line.trim_start().strip_prefix("#[cfg(")?;
        Some(&rest[..rest.rfind(')')?])
    }

    /// The lines a `#[cfg(...)]`-gated item spans, starting at its attribute.
    ///
    /// Indentation alone would end the item at the first blank line or at the
    /// first line at column 0 — and a gated test module can hold a multi-line
    /// string literal whose lines are exactly that, which would leave the rest of
    /// the module scanned as production code. So the item's head is walked to the
    /// brace that opens its body (a wrapped signature opens it further down), and
    /// a braced item is then closed where rustfmt puts its closing brace: the first
    /// line at the item's own indentation that starts with `}`. Both are bounded by
    /// the item — skipping too little is a loud false positive, skipping too much
    /// would hide a spawn.
    ///
    /// What that close is not: it cannot tell a brace from a string literal that
    /// holds a line of exactly that shape (the item's indentation, then `}`), which
    /// still ends the item early and leaves the rest of its body scanned as
    /// production. No such literal line exists in the tree today, and the direction
    /// is the loud one — a test-only spawn reported as a violation, never a spawn
    /// hidden.
    fn gated_item_end(lines: &[&str], at: usize) -> usize {
        let gate_indent = indent(lines[at]);
        // The item's own attributes and doc comments, then the line that opens it.
        let mut opener = at + 1;
        while opener < lines.len()
            && (lines[opener].trim().is_empty()
                || lines[opener].trim_start().starts_with("//")
                || lines[opener].trim_start().starts_with("#["))
        {
            opener += 1;
        }
        let mut braced = false;
        for start in opener..lines.len() {
            if indent(lines[start]) < gate_indent {
                break;
            }
            let text = lines[start].trim_end();
            if text.ends_with('{') {
                braced = true;
                opener = start;
                break;
            }
            // The item is a statement, or it is complete on this line (`fn f() {}`).
            if text.ends_with('}') || text.ends_with(';') {
                return start + 1;
            }
            // Only a wrapped head reads on, and only onto a more indented line.
            let Some(next) = lines.get(start + 1) else {
                break;
            };
            if next.trim().is_empty()
                || indent(next) < gate_indent
                || next.trim_start().starts_with("//")
            {
                break;
            }
        }
        if braced {
            // `}` alone closes an item, `};` closes a `let … = { … }` block: both
            // are the item's end. Ending early is loud; running past it would hide
            // a spawn, so the search never leaves the item's indentation.
            for (end, line) in lines.iter().enumerate().skip(opener + 1) {
                if indent(line) == gate_indent && line.trim_start().starts_with('}') {
                    return end + 1;
                }
            }
        }
        let mut end = opener + 1;
        while end < lines.len()
            && (lines[end].trim().is_empty() || indent(lines[end]) > gate_indent)
        {
            end += 1;
        }
        end
    }

    /// The lines the sweep must not read: comments (the module docs and ordinary
    /// comments name the flags), and every `#[cfg(...)]`-gated item that cannot be
    /// the windows production path.
    fn skipped_lines(lines: &[&str]) -> Vec<bool> {
        let mut skipped: Vec<bool> = lines
            .iter()
            .map(|l| l.trim_start().starts_with("//"))
            .collect();
        for (i, text) in lines.iter().enumerate() {
            if cfg_predicate(text).is_some_and(|p| gate(p) == Gate::No) {
                skipped[i..gated_item_end(lines, i)].fill(true);
            }
        }
        skipped
    }

    /// The name of the function a line declares, if it declares one.
    fn fn_name(line: &str) -> Option<String> {
        let name: String = line
            .split_once("fn ")?
            .1
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        (!name.is_empty()).then_some(name)
    }

    /// The function a line sits in: name, first line, first line after the body.
    fn enclosing_fn(lines: &[&str], at: usize) -> Option<(String, usize, usize)> {
        let start = (0..at).rev().find(|&i| {
            !lines[i].trim_start().starts_with("//")
                && indent(lines[i]) < indent(lines[at])
                && fn_name(lines[i]).is_some()
        })?;
        let mut end = at + 1;
        while end < lines.len()
            && (lines[end].trim().is_empty() || indent(lines[end]) > indent(lines[start]))
        {
            end += 1;
        }
        Some((fn_name(lines[start])?, start, end))
    }

    /// Whether a function body applies the windowless flag: the flag must sit
    /// inside a `creation_flags(...)` call — naming it without applying it (a bare
    /// `use …::CREATE_NO_WINDOW`) does not count. rustfmt wraps that call once a
    /// site is deep enough, so the argument list is read up to its closing
    /// parenthesis rather than per line.
    fn applies_windowless_flag(body: &str) -> bool {
        let mut rest = body;
        while let Some(at) = rest.find("creation_flags(") {
            let call = &rest[at..];
            let args = &call[..call.find(')').unwrap_or(call.len())];
            if args.contains(WINDOWLESS_FLAG) {
                return true;
            }
            rest = &call["creation_flags(".len()..];
        }
        false
    }

    /// The names the module docs' "No console window (windows)" section writes in
    /// backticks: the covered sites as they are recorded for a reader. That section
    /// *is* the covered list, so the sweep reads it back rather than trusting it —
    /// a site the sweep finds and the section does not name fails.
    fn inventoried_sites() -> Vec<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file!());
        let source = std::fs::read_to_string(&path).expect("read this module");
        let section = source
            .lines()
            .skip_while(|l| !l.contains(" # No console window (windows)"))
            .skip(1)
            .take_while(|l| !l.trim_start().starts_with("//! # "))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !section.is_empty(),
            "the module docs' window-guarantee section is gone"
        );
        section
            .split('`')
            .skip(1)
            .step_by(2)
            .map(|token| token.rsplit("::").next().unwrap_or(token))
            .map(|token| token.trim_end_matches("()"))
            .filter(|token| {
                !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
            .map(str::to_string)
            .collect()
    }

    /// Tripwire: every process the service starts must be created without a
    /// console window. See the module docs' "No console window (windows)"
    /// section, whose inventory this test reads back.
    ///
    /// It is a source scan, not a proof: it reads rustfmt-shaped code and
    /// best-effort `#[cfg]` gating, it judges per enclosing function (a second
    /// spawn added to a function that already applies the flag is caught by the
    /// site count, not site-specifically), and a `#[cfg]` on a `mod` declaration
    /// elsewhere in the tree is invisible to it — such a site fails loudly, and
    /// gating the spawn itself is the fix. A `(file, symbol)` allow-list plus the
    /// count would also catch an added site, but the count alone would not say
    /// which site it was, and the list would need hand-maintenance on every site
    /// change; the scan names the offending site itself.
    #[test]
    fn every_production_spawn_is_windowless() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let inventory = inventoried_sites();

        let mut sites = 0_usize;
        let mut violations: Vec<String> = Vec::new();
        for file in crate::util::test::rs_files_under(&root.join("src")) {
            let rel = crate::util::test::rel_source_path(root, &file);
            if TEST_ONLY_FILES.contains(&rel.as_str()) {
                continue;
            }
            let content = std::fs::read_to_string(&file).expect("read source file");
            let lines: Vec<&str> = content.lines().collect();
            let skipped = skipped_lines(&lines);
            for (i, text) in lines.iter().enumerate() {
                if skipped[i] {
                    continue;
                }
                let starts_process = text.contains("Command::new");
                let reserved = FORBIDDEN_FLAGS.iter().find(|f| text.contains(**f)).copied();
                if !starts_process && reserved.is_none() {
                    continue;
                }
                let Some((name, _, end)) = enclosing_fn(&lines, i) else {
                    violations.push(format!(
                        "{rel}:{}: a process is created outside any function",
                        i + 1
                    ));
                    continue;
                };
                if starts_process {
                    sites += 1;
                    if !applies_windowless_flag(&lines[i..end].join("\n")) {
                        violations.push(format!(
                            "{rel}:{} `{name}` starts a process without {WINDOWLESS_FLAG}",
                            i + 1
                        ));
                    } else if !inventory.iter().any(|listed| listed == &name) {
                        violations.push(format!(
                            "{rel}:{} `{name}` applies {WINDOWLESS_FLAG} but the module docs' \
                             covered-site inventory does not name it",
                            i + 1
                        ));
                    }
                }
                if let Some(flag) = reserved {
                    violations.push(format!(
                        "{rel}:{} `{name}` uses {flag} — see the module docs' window guarantee",
                        i + 1
                    ));
                }
            }
        }

        assert_eq!(
            sites, SPAWN_SITES,
            "the spawn sweep saw {sites} sites, not the {SPAWN_SITES} the module docs \
             inventory lists — a new site needs the flag and a name in that list, and a \
             scan that stopped reading the tree would find fewer"
        );
        assert!(
            violations.is_empty(),
            "the service must never put a console window on the screen:\n{violations:#?}"
        );
    }

    /// The one line the module docs' premise about this service's own image rests
    /// on: it is a windowed image because the crate root says so, and with the
    /// `not(test)` guard the binary's own test harness stays a console program. No
    /// other lane would notice the attribute's removal (the cross-checks only
    /// compile), so the source is read back — the same cheap tripwire the spawn
    /// sweep above is.
    #[test]
    fn the_binary_is_built_as_a_windowed_image() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
        let source = std::fs::read_to_string(&path).expect("read the binary's crate root");
        let attribute = r#"#![cfg_attr(not(test), windows_subsystem = "windows")]"#;
        assert!(
            source.contains(attribute),
            "{} must carry {attribute} — a Windows launch of the product is \
             console-less because of it, where the test harness keeps its console",
            path.display()
        );
    }
}

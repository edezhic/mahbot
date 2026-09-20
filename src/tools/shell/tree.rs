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
//!   [`Tree::terminate`] is one `kill(-pgid, SIGKILL)` — best-effort, a failure is
//!   logged — reaching every descendant.
//! - **windows**: one fresh job object per run, created with
//!   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, and the just-spawned child assigned to
//!   it ([`Tree::attach`]). Descent is inheritance — the child of a job member
//!   joins the job — and breakaway is deliberately not permitted, so
//!   `CREATE_BREAKAWAY_FROM_JOB` cannot hand a descendant to another job.
//!   [`Tree::terminate`] ends the job with `TerminateJobObject`; nothing here asks
//!   politely. The handle is the guarantee: closing the last one ends every process
//!   in the job, which is what an abrupt death of the service does.
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
    /// The child's pid, which `process_group(0)` also made its process group's
    /// id. `None` until [`Tree::attach`], and after a spawn that produced no pid
    /// at all.
    pid: Option<u32>,
}

#[cfg(unix)]
impl Tree {
    /// Nothing to set up per run before the spawn: the group is a property of
    /// the spawn itself. `_owner` decides the Windows job only, so both owners
    /// are contained the same way here — deliberately.
    pub(super) fn new(_owner: RunOwner) -> Self {
        Self { pid: None }
    }

    /// Record the spawned child's process group.
    pub(super) fn attach(&mut self, pid: u32) {
        self.pid = Some(pid);
    }

    /// End every process in the run's group — the direct child included, which is
    /// why a caller that got `true` only has to reap it. `true` means the group was
    /// signalled, not that every member died: a signal to a group with no live
    /// member left is not an error here ([`super::kill_process_group`] logs its own
    /// failure).
    pub(super) fn terminate(&self) -> bool {
        if let Some(pid) = self.pid {
            super::kill_process_group(pid, libc::SIGKILL);
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

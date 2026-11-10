//! The platform's own exemption from the way it slows a process whose window is not visible.
//!
//! The service dispatches every 100 ms for its whole life, and both platforms treat exactly
//! that: macOS naps an application and coalesces its timers once its windows are not visible
//! and it is not audible, and Windows drops the Quality of Service of an occluded or
//! minimised window owner far enough to ignore the timer-resolution requests it makes. Both
//! document a per-process exemption, and asking for it is all this module does — once per
//! start, on the daemon's start-up path. Desktop Linux has no such treatment for a process of
//! our kind, so nothing is asked and nothing is reported.
//!
//! What is deliberately *not* asked for is any scheduling privilege: this process's priority
//! is left alone, no real-time class is requested, and nothing machine-wide is changed (which
//! rules out Windows' battery-only downgrade of a foreground application — a registry value
//! applying to every application). Nor does [`exempt_from_slowdown`] keep the display on, keep
//! the machine from entering low power at the normal time, or add a keep-awake assertion: the
//! macOS option set is the arm of the activity family that leaves the sleep-disabling bit out
//! (`OPTIONS`). A platform that suspends desktop applications once the display turns off
//! (modern standby) takes the service with the machine: accepted.
//!
//! Accepted trade-off, without an escape hatch: the service keeps its full cadence while its
//! window is hidden, minimised or behind others, which costs power and heat, and no setting,
//! environment variable or flag turns it off. A child of the service owns no window of its
//! own, so where the platform classifies it by its ancestor's window it is part of the same
//! situation — only a third-party tool the service runs is out of scope.
//!
//! Nothing here is fatal: a platform that refuses, ignores or does not support the request
//! leaves the launch exactly as it would have been. The start-up line (`main.rs`) reports what
//! came back — accepted, refused, or not supported — and never as protection.

#[cfg(target_os = "macos")]
use objc2_foundation::{NSActivityOptions, NSProcessInfo, ns_string};

/// What the platform answered when the exemption was asked for.
#[derive(Clone, Copy, Debug)]
pub enum Reply {
    Accepted,
    Refused,
    /// This platform, or this build of it, does not have what was asked for — a different
    /// fact from a refusal, and never reported as one.
    Unsupported,
}

impl std::fmt::Display for Reply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Accepted => "accepted",
            Self::Refused => "refused",
            Self::Unsupported => "not supported",
        })
    }
}

/// The one request a start-up makes and what came back, for the start-up status line.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub request: &'static str,
    pub reply: Reply,
}

/// The activity macOS is asked for, and the whole of the sleep boundary on this platform:
/// `UserInitiatedAllowingIdleSystemSleep` is the activity App Nap's guide names for work that
/// must not be deferred, and the arm whose mask leaves the sleep-disabling bit out;
/// `LatencyCritical` is the platform's only *process-wide* lever against timer coalescing (the
/// per-timer levers cover one timer each). Together they are `NSActivityUserInteractive` with
/// the idle-sleep bit removed, pinned by the assertion below. Holding this set for the
/// service's life rather than for the burst the platform documents is deliberate: the platform
/// offers nothing narrower, and a request that left the precision half in force would not turn
/// the treatment off completely.
#[cfg(target_os = "macos")]
const OPTIONS: NSActivityOptions = NSActivityOptions::UserInitiatedAllowingIdleSystemSleep
    .union(NSActivityOptions::LatencyCritical);

// The one slip that would pass every other check while quietly keeping the machine awake for
// good is an option family that also disables idle sleep, so the mask is pinned at compile
// time, where the mask is.
#[cfg(target_os = "macos")]
const _: () = {
    assert!(
        OPTIONS.contains(NSActivityOptions::LatencyCritical),
        "the activity must carry what turns timer coalescing off"
    );
    assert!(
        !OPTIONS.contains(NSActivityOptions::IdleSystemSleepDisabled),
        "the machine must still be allowed to sleep"
    );
    assert!(
        !OPTIONS.contains(NSActivityOptions::IdleDisplaySleepDisabled),
        "the display must still be allowed to sleep"
    );
};

/// Ask macOS not to nap this application and not to coalesce its timers. Holding the returned
/// object *is* the exemption: the platform ends the activity the moment it is deallocated, so the
/// token is deliberately leaked for the life of the process. A dropped token would silently put the
/// service back under App Nap. The termination bits riding inside every `UserInitiated`-family mask
/// are inert here: sudden and automatic termination are Info.plist opt-ins and the service is a
/// bare binary with no bundle. The token is never null by contract, so a refusal cannot be observed
/// here; whether the activity really keeps App Nap off is not readable from inside the process
/// either, so the start-up line reports the request and the effect is read from the platform's own
/// records.
#[cfg(target_os = "macos")]
#[must_use]
pub fn exempt_from_slowdown() -> Option<Report> {
    const REQUEST: &str = "NSProcessInfo beginActivityWithOptions:NSActivityUserInitiated\
                           AllowingIdleSystemSleep|NSActivityLatencyCritical";

    let token = NSProcessInfo::processInfo()
        .beginActivityWithOptions_reason(OPTIONS, ns_string!("MahBot dispatches work"));
    std::mem::forget(token);
    Some(Report {
        request: REQUEST,
        reply: Reply::Accepted,
    })
}

/// Ask Windows for High Quality of Service, with both of its throttling mechanisms off.
///
/// The whole mechanism set goes into a single call: `ControlMask` declares the mechanisms this
/// process controls and `StateMask` which are on, so a second call would leave the process
/// controlling only what it names. The platform's own examples ask for its two mechanisms
/// separately; this deliberately does not.
///
/// A Windows 10 build — which the x86_64 product still supports — does not know the
/// timer-resolution flag and rejects the whole request over it, so the execution-speed mechanism
/// is asked for alone and that narrower request is reported; nothing of the treatment is left in
/// force, since the mechanism that flag turns off is a Windows 11 behaviour such a build does not
/// have. What the call left behind is readable back where the build supports it
/// (`GetProcessInformation` with the same class).
#[cfg(windows)]
#[must_use]
pub fn exempt_from_slowdown() -> Option<Report> {
    use std::mem::size_of;

    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, GetLastError};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        PROCESS_POWER_THROTTLING_STATE, ProcessPowerThrottling, SetProcessInformation,
    };

    const REQUEST_BOTH: &str = "SetProcessInformation(ProcessPowerThrottling, ControlMask=\
                                EXECUTION_SPEED|IGNORE_TIMER_RESOLUTION, StateMask=0)";
    const REQUEST_SPEED_ONLY: &str = "SetProcessInformation(ProcessPowerThrottling, \
                                      ControlMask=EXECUTION_SPEED, StateMask=0)";

    fn ask(control: u32) -> Result<(), u32> {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: control,
            StateMask: 0,
        };
        // The call is handed the structure's own size, as that class documents. `allow`, not
        // `expect`: this cast cannot truncate on a 32-bit target, and an unfired expectation is
        // itself a warning.
        #[allow(clippy::cast_possible_truncation)]
        let size = size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32;
        // SAFETY: `state` is a `PROCESS_POWER_THROTTLING_STATE` that outlives the call
        // and is handed over with its own size, and `GetCurrentProcess` is the
        // documented pseudo-handle for this process.
        let ok = unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                std::ptr::from_ref(&state).cast(),
                size,
            )
        };
        if ok != 0 {
            Ok(())
        } else {
            // SAFETY: `GetLastError` takes no arguments and observes the calling
            // thread's last error, which is the failed call's.
            Err(unsafe { GetLastError() })
        }
    }

    let (request, error) =
        match ask(PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION)
        {
            Ok(()) => {
                return Some(Report {
                    request: REQUEST_BOTH,
                    reply: Reply::Accepted,
                });
            }
            Err(ERROR_INVALID_PARAMETER) => match ask(PROCESS_POWER_THROTTLING_EXECUTION_SPEED) {
                Ok(()) => {
                    return Some(Report {
                        request: REQUEST_SPEED_ONLY,
                        reply: Reply::Accepted,
                    });
                }
                Err(error) => (REQUEST_SPEED_ONLY, error),
            },
            Err(error) => (REQUEST_BOTH, error),
        };

    Some(Report {
        request,
        // Even the mechanism every supported build has was refused: the class or the function
        // is not there on this build — not the same fact as a refusal by a platform that has them.
        reply: if error == ERROR_INVALID_PARAMETER {
            Reply::Unsupported
        } else {
            Reply::Refused
        },
    })
}

#[cfg(not(any(target_os = "macos", windows)))]
#[must_use]
pub fn exempt_from_slowdown() -> Option<Report> {
    None
}

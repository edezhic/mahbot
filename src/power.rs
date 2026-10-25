//! The platform's own exemption from the way it slows a process whose window is not
//! visible.
//!
//! The service dispatches every 100 ms for its whole life, and both platforms this product
//! runs on have a treatment aimed exactly at a process in that situation: macOS naps an
//! application and coalesces its timers once its windows are not visible and it is not
//! audible, and Windows drops the Quality of Service of a window owner that is occluded or
//! minimised and not audible, far enough to ignore the timer-resolution requests it makes.
//! Both platforms document a per-process exemption from that treatment, and asking for it is
//! all this module does — once per start, on the daemon's own start-up path, so it holds for
//! the whole life of the service instead of being tied to a window, to a user action or to
//! one operation. Desktop Linux has no such treatment for a process of our kind, so nothing
//! is asked there and nothing is reported.
//!
//! What is deliberately *not* asked for is any scheduling privilege: this process's own
//! priority is left alone, no real-time class is requested, and nothing machine-wide is
//! changed. That last one rules out Windows' battery-only downgrade of a foreground
//! application after a period without user input: it exists only as a value in the machine's
//! registry and would apply to every application on the machine, so it is left alone rather
//! than switched off, and the request below claims nothing about it.
//!
//! The boundary with the machine's own sleep, decided once and applied on every platform:
//! [`exempt_from_slowdown`] does not keep the display on, does not keep the machine from
//! entering its own low-power state at the normal time, and adds no keep-awake assertion of
//! any kind — the macOS option set is the arm of the activity family that leaves the
//! sleep-disabling bit out (`OPTIONS`). Voice input and speech output take no part in this:
//! they do not run through this module, and nothing here adds or removes a keep-awake of
//! theirs. A platform that instead suspends desktop applications once the display turns off
//! (modern standby) takes the service with the machine: that is accepted, and it is not
//! something this request may change.
//!
//! Accepted trade-off, deliberate and without an escape hatch: with the exemption in force
//! the service keeps its full cadence while its window is hidden, minimised or behind
//! others, which costs power and heat, and no setting, environment variable or command-line
//! flag turns it off.
//!
//! The request names one process and is made once, on the service's start-up path, so no
//! spawn site makes a second one. A child of the service owns no window of its own: where
//! the platform classifies such a child by the window its ancestor owns rather than by
//! anything the child declares (Windows assigns a descendant of a window-owning process its
//! Quality of Service from that window), the child is part of the service's own situation and
//! is not a third party the platform treats on its own — which is the whole reason the single
//! request above is the only one made. Only a third-party tool the service runs — a shell, a
//! compiler, a browser, which the platform classifies for itself — is out of scope, and this
//! request says nothing about it.
//!
//! Nothing here is fatal: a platform that refuses, ignores or does not support the request
//! leaves the launch exactly as it would have been without it. What came back is stated by
//! the start-up line (`main.rs`) rather than kept quiet, so the exemption is reported as a
//! request — accepted, refused, or not supported on this platform or build — and never as
//! protection.

#[cfg(target_os = "macos")]
use objc2_foundation::{NSActivityOptions, NSProcessInfo, ns_string};

/// What the platform answered when the exemption was asked for.
#[derive(Clone, Copy, Debug)]
pub enum Reply {
    /// The platform took the request.
    Accepted,
    /// The platform was asked and said no.
    Refused,
    /// This platform, or this build of it, does not have what was asked for. A different
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
    /// The request, in the platform's own vocabulary.
    pub request: &'static str,
    /// What the platform answered.
    pub reply: Reply,
}

/// The activity macOS is asked for, and the whole of the sleep boundary on this platform:
/// `UserInitiatedAllowingIdleSystemSleep` is the activity the platform's App Nap guide names
/// for work that must not be deferred, and it is the arm of the family whose mask leaves the
/// sleep-disabling bit out; `LatencyCritical` is the platform's only *process-wide* lever on
/// the other half of the treatment — "the highest amount of timer and I/O precision
/// available", the flag it documents for latency-sensitive work — and is what this request
/// carries against timer coalescing. The per-timer levers (`DISPATCH_TIMER_STRICT`, the
/// kqueue note flags) cover one timer each and cannot carry a service whose whole cadence has
/// to survive. The two together are `NSActivityUserInteractive` with the idle-sleep bit
/// removed; the assertion below pins that, and measures it the only way the platform allows —
/// on macOS 26.5 a process holding this option set is absent from `pmset -g assertions` for
/// the whole time it holds it, while one holding `NSActivityUserInitiated` or
/// `NSActivityUserInteractive`, whose masks carry that bit, is listed there as
/// `PreventUserIdleSystemSleep`.
///
/// The other effect this carries, held for the life of the service rather than for the short
/// burst the platform documents `LatencyCritical` for, is exactly that above-baseline timer
/// and I/O precision: accepted deliberately, because the platform warns against holding such
/// an activity for an extended period and offers nothing narrower, and a request that left
/// the precision half in force would not turn the treatment off completely.
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

/// Ask macOS not to nap this application and not to coalesce its timers.
///
/// Holding the returned object *is* the exemption: the platform ends the activity the moment
/// it is deallocated, so the token is deliberately leaked for the life of the process (which
/// is what ends it). A dropped token would silently put the service back under App Nap.
///
/// The termination bits that ride inside every `UserInitiated`-family mask are counted here
/// as well and are inert for this product: sudden and automatic termination are opt-ins
/// through a bundle's Info.plist (`NSSupportsSuddenTermination` /
/// `NSSupportsAutomaticTermination`), and the service is a bare binary with no bundle. The
/// product's own exit routes are unaffected (see `shutdown`, which owns that list).
///
/// The token is the platform's whole answer and is never null by contract, so a refusal
/// cannot be observed here and none is invented; whether the activity really keeps App Nap
/// off is not readable from inside the process either, the platform stating outright that
/// user preferences may override the request. So the start-up line reports the request, and
/// the effect is read from the platform's own records instead — this process's App Nap state,
/// and the machine's sleep-assertion list for the boundary above.
#[cfg(target_os = "macos")]
#[must_use]
pub fn exempt_from_slowdown() -> Option<Report> {
    /// The request, in the platform's own vocabulary.
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
/// The whole mechanism set goes into a single call: `ControlMask` declares the mechanisms
/// this process controls and `StateMask` which of them are on, so a second call would leave
/// the process controlling only what the second one names. The platform's own examples ask
/// for its two mechanisms separately; this deliberately does not.
///
/// A Windows 10 build — which the x86_64 product still supports — does not know the
/// timer-resolution flag and rejects the whole request over it, so the execution-speed
/// mechanism is asked for on its own instead, and that narrower request is what gets
/// reported. Nothing of the treatment is left in force by the fallback: the mechanism the
/// flag turns off is a Windows 11 behaviour such a build does not have.
///
/// The declaration stands for the life of the service, so the platform's window-state
/// heuristics no longer classify this process — not by its window's visibility, not by its
/// audibility. What the call left behind is readable back where the build supports it
/// (`GetProcessInformation` with the same class), which is what a check reads. Accepted
/// deliberately: keeping the dispatch cadence is the point of the request, and the platform
/// documents nothing narrower that turns the mechanisms off.
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

    /// One request, with every mechanism it names turned off.
    fn ask(control: u32) -> Result<(), u32> {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: control,
            StateMask: 0,
        };
        // The call is handed the structure's own size, as that class documents.
        // `allow`, not `expect`: this cast cannot truncate on a 32-bit target, and an
        // unfired expectation is itself a warning.
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
        // Even the mechanism every supported build has was refused: the class or the
        // function is not there on this build, which is not the same fact as a refusal
        // by a platform that has them.
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

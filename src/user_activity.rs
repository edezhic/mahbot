//! Who is at the computer: the one machine-wide notion of user activity.
//!
//! Active while activity is reported, absent after [`ABSENCE_THRESHOLD`] without any, and
//! *cannot tell* where the machine cannot report input — which behaves exactly like active,
//! because the product must never claim the user is away when it cannot see them. Nothing
//! else is kept: no history (nothing is logged when the state changes), no per-application
//! detail. The state is the machine's, not one account's.
//!
//! Activity is what the operating system reports as real input — mouse and keyboard — plus a
//! recognised wake phrase, reported through [`report_activity`] the moment the phrase is
//! recognised, whatever the voice pipeline then does with the capture. An open microphone is
//! not activity.
//!
//! Sources report and the rule in [`activity`] combines them; there is no registry, and a
//! platform without a source simply has none, which is the *cannot tell* case. Absence is
//! concluded only from the operating system's continuous reading: a wake phrase is a point
//! event, so no amount of silence from it adds up to evidence that someone left.
//!
//! The reading is taken when asked ([`is_absent`]), never sampled, so it always reflects the
//! machine's own last five minutes — a restart into an untouched machine reports absent at
//! once instead of manufacturing fresh presence.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::util::UnwrapPoison as _;

/// How long without activity still counts as being at the computer.
const ABSENCE_THRESHOLD: Duration = Duration::from_mins(5);

/// When the last report from a source other than the operating system's reading arrived — a
/// recognised wake phrase, in practice.
static LAST_REPORT: Mutex<Option<Instant>> = Mutex::new(None);

/// Whether the user is absent right now. Never true on a machine that cannot tell.
pub(crate) fn is_absent() -> bool {
    activity(
        os_idle(),
        LAST_REPORT.lock().unwrap_poison().map(|at| at.elapsed()),
    ) == Activity::Absent
}

/// A source saw the user: last activity is now. Called by the wake word at the moment it
/// recognises the spoken phrase.
pub(crate) fn report_activity() {
    *LAST_REPORT.lock().unwrap_poison() = Some(Instant::now());
}

/// What the machine says about the person in front of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activity {
    Active,
    Absent,
    /// The machine cannot say; behaves exactly like [`Activity::Active`].
    Unknown,
}

/// The rule, over how long since the operating system saw real input (`None` where it cannot
/// say) and how long since the last report: the freshest of the two decides, and absence
/// needs the operating system's reading to exist at all.
fn activity(os_idle: Option<Duration>, since_report: Option<Duration>) -> Activity {
    let Some(os_idle) = os_idle else {
        return Activity::Unknown;
    };
    let since = since_report.map_or(os_idle, |report| os_idle.min(report));
    if since >= ABSENCE_THRESHOLD {
        Activity::Absent
    } else {
        Activity::Active
    }
}

// ── The operating system's input reading ────────────────────────────────
// One reader per platform, where the platform answers without asking for a permission.
// Each gives "how long since real input", or `None` when it cannot tell.

/// CoreGraphics' combined-session idle time: seconds since any real input in the login
/// session.
#[cfg(target_os = "macos")]
fn os_idle() -> Option<Duration> {
    use objc2_core_graphics::{CGEventSource, CGEventSourceStateID, CGEventType};

    crate::util::with_block_in_place(|| {
        // `kCGAnyInputEventType` is the header's `(CGEventType)~0`.
        let seconds = CGEventSource::seconds_since_last_event_type(
            CGEventSourceStateID::CombinedSessionState,
            CGEventType(u32::MAX),
        );
        // A negative or non-finite answer is the API saying it cannot tell (no window-server
        // session), not an enormous absence.
        Duration::try_from_secs_f64(seconds).ok()
    })
}

/// `GetLastInputInfo`: the tick count of the session's last mouse or keyboard input.
#[cfg(target_os = "windows")]
fn os_idle() -> Option<Duration> {
    use windows::Win32::System::SystemInformation::GetTickCount;
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetLastInputInfo, LASTINPUTINFO};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "LASTINPUTINFO is two u32 fields"
    )]
    let size = std::mem::size_of::<LASTINPUTINFO>() as u32;
    let mut last = LASTINPUTINFO {
        cbSize: size,
        dwTime: 0,
    };
    // SAFETY: `last` is live, initialised and exactly the size the call is told, and the
    // call only writes into it. A FALSE answer is the session declining to say — a service
    // session, a desktop it cannot see — which is the `None` arm, not an idle of zero.
    if !unsafe { GetLastInputInfo(&raw mut last) }.as_bool() {
        return None;
    }
    // SAFETY: takes no arguments, reads nothing and always succeeds.
    let now = unsafe { GetTickCount() };
    // Both are 32-bit tick counts, so the difference is taken in the same wrapping space. A
    // difference past the current tick value is either `dwTime` running ahead of this clock
    // (documented as not necessarily incremental) or a last input just before the ~49.7-day
    // wrap, where the real idle is seconds; both are read as input just now.
    let delta = now.wrapping_sub(last.dwTime);
    Some(Duration::from_millis(u64::from(if delta > now {
        0
    } else {
        delta
    })))
}

/// The X screen-saver's idle time: one request over a short-lived X11 connection, which this
/// daemon makes rarely enough that a connection per question beats the state a cached one
/// would need.
#[cfg(target_os = "linux")]
fn os_idle() -> Option<Duration> {
    use x11rb::connection::Connection as _;
    use x11rb::protocol::screensaver::ConnectionExt as _;

    // Only an X11 session's input goes through a server this can ask; a Wayland session's
    // idle time belongs to its compositor.
    if crate::util::display_server::wayland() {
        return None;
    }
    crate::util::with_block_in_place(|| {
        let (connection, screen) = x11rb::connect(None).ok()?;
        let root = connection.setup().roots.get(screen)?.root;
        let reply = connection.screensaver_query_info(root).ok()?.reply().ok()?;
        Some(Duration::from_millis(u64::from(reply.ms_since_user_input)))
    })
}

/// No reader on this platform, so the state stays *cannot tell*.
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn os_idle() -> Option<Duration> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clock reading well inside the threshold, and one well past it.
    const RECENT: Duration = ABSENCE_THRESHOLD.saturating_sub(Duration::from_secs(1));
    const STALE: Duration = ABSENCE_THRESHOLD.saturating_add(Duration::from_mins(1));

    #[test]
    fn activity_stops_counting_at_the_threshold() {
        assert_eq!(activity(Some(Duration::ZERO), None), Activity::Active);
        assert_eq!(activity(Some(RECENT), None), Activity::Active);
        assert_eq!(activity(Some(ABSENCE_THRESHOLD), None), Activity::Absent);
        assert_eq!(activity(Some(STALE), None), Activity::Absent);
    }

    /// A report is a point event, so it cannot stand in for the machine's own reading: where
    /// that reading is silent the answer is *cannot tell*, never absence.
    #[test]
    fn a_machine_that_cannot_tell_is_never_absent() {
        for report in [None, Some(RECENT), Some(STALE)] {
            assert_eq!(activity(None, report), Activity::Unknown);
        }
    }

    /// A wake phrase holds the user present for five minutes after the operating system has
    /// stopped seeing input.
    #[test]
    fn the_freshest_source_decides() {
        assert_eq!(activity(Some(STALE), Some(RECENT)), Activity::Active);
        assert_eq!(activity(Some(RECENT), Some(STALE)), Activity::Active);
        assert_eq!(activity(Some(STALE), Some(STALE)), Activity::Absent);
        assert_eq!(activity(Some(STALE), None), Activity::Absent);
    }
}

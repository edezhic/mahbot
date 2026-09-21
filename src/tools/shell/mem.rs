//! How much memory a command's process tree is holding.
//!
//! The watchdog [`super::run_command_with_timeout`] runs needs one number: the
//! resident memory of the child *and every descendant*, sampled while the
//! command runs, against a ceiling of [`MEMORY_LIMIT_FRACTION`] of physical RAM.
//! The ceiling exists because the shell tool imposes no memory bound of its own
//! — a single command can allocate the whole machine away — and the sample is
//! what lets the run be ended before the operating system starts thrashing.
//!
//! One mechanism, three platforms; only the two queries below are spelled per
//! platform:
//!
//! - the machine's *total*, for the ceiling: `sysctl hw.memsize` on macOS,
//!   `sysconf(_SC_PHYS_PAGES)` times the page size on Linux,
//!   `GlobalMemoryStatusEx` on Windows;
//! - the tree's *usage*: each process's resident footprint (`proc_pid_rusage`
//!   on macOS, `/proc/<pid>/statm` on Linux, `GetProcessMemoryInfo` on
//!   Windows), summed over the tree the platform's own child discovery yields
//!   (`proc_listchildpids`, the `ppid` of `/proc/<pid>/stat`, a Toolhelp
//!   parent-pid walk).
//!
//! Everything here fails open. A query that cannot answer returns `None`, a
//! descendant that is already gone by the time it is read simply contributes
//! nothing, and no measurement failure ever ends a run or panics one: a
//! watchdog that killed a command because it could not read a number would be
//! worse than the unbounded command it was added to stop.
//!
//! One accepted limitation follows from discovering the tree by parent pid: a
//! descendant whose own parent has already exited is reparented away and is no
//! longer reachable from the run's root, so its memory is not counted. The
//! measurement is a lower bound — it can miss memory, never invent it.

#[cfg(any(target_os = "linux", windows))]
use std::collections::HashMap;
#[cfg(any(target_os = "macos", target_os = "linux", windows))]
use std::collections::HashSet;

/// Fraction of physical RAM a command's process tree may use before the
/// watchdog ends the run.
const MEMORY_LIMIT_FRACTION: f64 = 0.8;

/// The ceiling a run is watched against — [`MEMORY_LIMIT_FRACTION`] of physical
/// RAM — or `None` when the machine's total cannot be read (fail open: a run is
/// never ended for memory when the total is unknown).
#[must_use]
pub(super) fn default_limit() -> Option<u64> {
    // The share is the only float involved; physical RAM is far below 2^53, so
    // the round-trip through `f64` is exact.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let limit = (physical_memory()? as f64 * MEMORY_LIMIT_FRACTION) as u64;
    Some(limit)
}

/// A byte count as a one-decimal GiB figure — the unit a memory failure reports
/// in (`~15.4 GiB`). Integer arithmetic, so the figure is exact to the tenth.
#[must_use]
pub(super) fn gib(bytes: u64) -> String {
    /// One gibibyte; a tenth of it is `bytes % GIB * 10 / GIB`.
    const GIB: u64 = 1 << 30;
    let whole = bytes / GIB;
    let tenth = bytes % GIB * 10 / GIB;
    format!("{whole}.{tenth} GiB")
}

/// The one-line memory failure every run shape reports, e.g.
/// `terminated: exceeded memory limit (used ~15.4 GiB of 18.0 GiB)`. Spelled
/// once so the tool's error block and a direct run's typed error cannot drift.
///
/// `used` is reported against the machine's total RAM — the figure a reader can
/// weigh the amount against, since the ceiling itself is a share of it — and
/// against `limit` only when the total cannot be read. `limit` is the ceiling
/// that was actually crossed.
#[must_use]
pub(super) fn exceeded_failure(used: u64, limit: u64) -> String {
    let total = physical_memory().unwrap_or(limit);
    format!(
        "terminated: exceeded memory limit (used ~{} of {})",
        gib(used),
        gib(total),
    )
}

// ── Physical RAM (the ceiling) ────────────────────────────────────────

/// Total physical RAM in bytes, or `None` when it cannot be read.
#[cfg(target_os = "macos")]
#[must_use]
pub(super) fn physical_memory() -> Option<u64> {
    let mut size = 0_u64;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is a NUL-terminated literal; `size` is a live `u64`
    // whose length `len` states; the new-value arguments are null and zero,
    // which `sysctl` reads as a pure query.
    let ok = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            std::ptr::addr_of_mut!(size).cast(),
            std::ptr::addr_of_mut!(len),
            std::ptr::null_mut(),
            0,
        )
    };
    (ok == 0).then_some(size)
}

/// Total physical RAM in bytes, or `None` when it cannot be read.
#[cfg(target_os = "linux")]
#[must_use]
pub(super) fn physical_memory() -> Option<u64> {
    // SAFETY: `sysconf` is a read-only query with no arguments to misuse.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    // SAFETY: as above.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page_size <= 0 {
        return None;
    }
    Some(
        u64::try_from(pages)
            .ok()?
            .saturating_mul(u64::try_from(page_size).ok()?),
    )
}

/// Total physical RAM in bytes, or `None` when it cannot be read.
#[cfg(windows)]
#[must_use]
pub(super) fn physical_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: all-zero is the documented starting value; `dwLength` is set to
    // the struct's size, which is what the call requires.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength =
        u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).expect("memory status fits in u32");
    // SAFETY: `status` is live and its first field states its size.
    let ok = unsafe { GlobalMemoryStatusEx(std::ptr::addr_of_mut!(status)) };
    (ok != 0).then_some(status.ullTotalPhys)
}

/// No physical-RAM reading on a platform that is neither macOS, Linux nor
/// Windows (fail open).
#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
#[must_use]
pub(super) fn physical_memory() -> Option<u64> {
    None
}

// ── Resident usage of a process tree ──────────────────────────────────

/// Resident bytes `root` and every live descendant hold, or `None` when the
/// platform's helper cannot answer (fail open). The root's own reading is
/// required; a descendant that cannot be read — it exited mid-walk — is
/// skipped rather than failing the whole sample.
#[cfg(target_os = "macos")]
#[must_use]
pub(super) fn tree_rss(root: u32) -> Option<u64> {
    // macOS has no whole-table snapshot to walk: descend from `root` with the
    // kernel's own child listing, one level at a time.
    let mut total = process_rss(root)?;
    let mut seen = HashSet::from([root]);
    let mut queue = vec![root];
    while let Some(pid) = queue.pop() {
        for child in children_of(pid) {
            if !seen.insert(child) {
                continue;
            }
            if let Some(rss) = process_rss(child) {
                total = total.saturating_add(rss);
            }
            queue.push(child);
        }
    }
    Some(total)
}

/// Resident bytes `root` and every live descendant hold — the same contract
/// and fail-open rule as the macOS arm above, walked from this platform's
/// process table.
#[cfg(target_os = "linux")]
#[must_use]
pub(super) fn tree_rss(root: u32) -> Option<u64> {
    sum_descendants(root, &children_map())
}

/// Resident bytes `root` and every live descendant hold — the same contract
/// and fail-open rule as the macOS arm above, walked from this platform's
/// process table.
#[cfg(windows)]
#[must_use]
pub(super) fn tree_rss(root: u32) -> Option<u64> {
    sum_descendants(root, &children_map()?)
}

/// No resident-memory reading on a platform that is neither macOS, Linux nor
/// Windows (fail open).
#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
#[must_use]
pub(super) fn tree_rss(_root: u32) -> Option<u64> {
    None
}

/// Sum [`process_rss`] over `root` and every pid `children` reaches from it.
/// Shared by the two platforms that discover the tree from a whole-process
/// table; macOS descends with `proc_listchildpids` instead and does its own
/// walk.
#[cfg(any(target_os = "linux", windows))]
fn sum_descendants(root: u32, children: &HashMap<u32, Vec<u32>>) -> Option<u64> {
    let mut total = process_rss(root)?;
    let mut seen = HashSet::from([root]);
    let mut queue = vec![root];
    while let Some(pid) = queue.pop() {
        for &child in children.get(&pid).into_iter().flatten() {
            if !seen.insert(child) {
                continue;
            }
            if let Some(rss) = process_rss(child) {
                total = total.saturating_add(rss);
            }
            queue.push(child);
        }
    }
    Some(total)
}

// ── Per-process resident memory ───────────────────────────────────────

/// Resident bytes one macOS process holds: its physical footprint, the figure
/// the system's own memory accounting reports.
#[cfg(target_os = "macos")]
fn process_rss(pid: u32) -> Option<u64> {
    let id = libc::c_int::try_from(pid).ok()?;
    // SAFETY: all-zero is a valid starting value for the flavor's struct.
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a live `rusage_info_v2` and `RUSAGE_INFO_V2` is the
    // flavor whose layout the kernel fills into exactly that type; the address
    // passed is the struct's own, which is the shape the C API's
    // `rusage_info_t *` parameter takes.
    let ok = unsafe {
        libc::proc_pid_rusage(
            id,
            libc::RUSAGE_INFO_V2,
            std::ptr::addr_of_mut!(info).cast(),
        )
    };
    (ok == 0).then_some(info.ri_phys_footprint)
}

/// The pids macOS reports as `pid`'s direct children, as the kernel sees them
/// now. A refusal or an empty answer yields an empty list (fail open: the
/// parent's own footprint still answers).
#[cfg(target_os = "macos")]
fn children_of(pid: u32) -> Vec<u32> {
    /// Direct children one process may have before this walk stops growing its
    /// buffer; every real command is far below it.
    const CHILD_CAPACITY: usize = 1024;

    let Ok(parent) = libc::pid_t::try_from(pid) else {
        return Vec::new();
    };
    let mut buffer: Vec<libc::pid_t> = vec![0; CHILD_CAPACITY];
    let bytes = libc::c_int::try_from(std::mem::size_of_val(buffer.as_slice()))
        .expect("the child buffer fits in a C int");
    // SAFETY: `buffer` is live for the call and `bytes` is exactly its length,
    // so the kernel writes no more than the buffer holds.
    let written = unsafe { libc::proc_listchildpids(parent, buffer.as_mut_ptr().cast(), bytes) };
    // The call reports how many pids it filled in, capped at the buffer.
    let Ok(filled) = usize::try_from(written) else {
        return Vec::new();
    };
    buffer.truncate(filled.min(CHILD_CAPACITY));
    buffer
        .into_iter()
        .filter_map(|p| u32::try_from(p).ok())
        .collect()
}

/// Resident bytes one Linux process holds, from `/proc/<pid>/statm` whose
/// second field is the resident set in pages.
#[cfg(target_os = "linux")]
fn process_rss(pid: u32) -> Option<u64> {
    let statm = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
    let mut fields = statm.split_whitespace();
    fields.next()?; // `size` — the resident field is next
    let resident_pages: u64 = fields.next()?.parse().ok()?;
    Some(resident_pages.saturating_mul(page_size()))
}

/// The system page size, which `/proc/<pid>/statm` counts resident memory in.
#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    // SAFETY: `sysconf` is a read-only query with no arguments to misuse.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    // A non-positive answer cannot happen for a valid name; falling back to the
    // near-universal 4 KiB only keeps the sample usable (fail open either way).
    u64::try_from(size).unwrap_or(4096)
}

/// Resident bytes one Windows process holds: its working set, the resident
/// pages the system has charged to it.
#[cfg(windows)]
fn process_rss(pid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };

    // SAFETY: a plain open by pid, asking for the two rights
    // `GetProcessMemoryInfo` documents.
    let process =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid) };
    if process == 0 {
        return None;
    }
    // SAFETY: all-zero is valid; `cb` is set to the struct's size below.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>())
        .expect("memory counters fit in u32");
    // SAFETY: `process` is a live handle and `counters` is sized as stated.
    let ok =
        unsafe { GetProcessMemoryInfo(process, std::ptr::addr_of_mut!(counters), counters.cb) };
    // SAFETY: `process` came from the `OpenProcess` above and is not used again.
    unsafe { CloseHandle(process) };
    if ok == 0 {
        return None;
    }
    u64::try_from(counters.WorkingSetSize).ok()
}

// ── Process-tree discovery ────────────────────────────────────────────

/// Every live pid mapped to its parent's, walked from `/proc`. A walk that
/// cannot read the table yields an empty index (fail open: no descendant is
/// found, and the root's own footprint still answers).
#[cfg(target_os = "linux")]
fn children_map() -> HashMap<u32, Vec<u32>> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return HashMap::new();
    };
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if let Some(parent) = parent_of(pid) {
            children.entry(parent).or_default().push(pid);
        }
    }
    children
}

/// The pid that spawned `pid`, from `/proc/<pid>/stat`. Its `ppid` is the
/// second whitespace-separated field after the command name, and the command
/// name may itself contain spaces or parentheses — hence the parse starts at
/// the last `)`, not at a field index.
#[cfg(target_os = "linux")]
fn parent_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = stat.get(stat.rfind(')')? + 1..)?;
    let mut fields = rest.split_whitespace();
    fields.next()?; // `state` — `ppid` is next
    fields.next()?.parse().ok()
}

/// Every live process mapped to its parent's, from one Toolhelp snapshot — the
/// Windows spelling of Linux's `/proc` walk. `None` when the snapshot itself
/// fails (fail open).
#[cfg(windows)]
fn children_map() -> Option<HashMap<u32, Vec<u32>>> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
        TH32CS_SNAPPROCESS,
    };

    // SAFETY: a whole-system process snapshot. The handle is a real one or the
    // invalid value checked below, and it is closed before this returns.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return None;
    }
    // SAFETY: all-zero is a valid starting value except `dwSize`, which the
    // call requires to be the struct's own size.
    let mut entry: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
    entry.dwSize =
        u32::try_from(std::mem::size_of::<PROCESSENTRY32W>()).expect("process entry fits in u32");
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    // SAFETY: `snapshot` is live and the entry is sized as the call requires.
    let mut more = unsafe { Process32FirstW(snapshot, std::ptr::addr_of_mut!(entry)) };
    while more != 0 {
        children
            .entry(entry.th32ParentProcessID)
            .or_default()
            .push(entry.th32ProcessID);
        // SAFETY: as above.
        more = unsafe { Process32NextW(snapshot, std::ptr::addr_of_mut!(entry)) };
    }
    // SAFETY: `snapshot` came from the call above and is not used again.
    unsafe { CloseHandle(snapshot) };
    Some(children)
}

#[cfg(test)]
mod tests {
    use super::{default_limit, exceeded_failure, gib, physical_memory, tree_rss};

    #[test]
    fn gib_renders_one_decimal_place() {
        assert_eq!(gib(0), "0.0 GiB");
        assert_eq!(gib(1 << 30), "1.0 GiB");
        assert_eq!(gib(3 * (1 << 30) / 2), "1.5 GiB");
        assert_eq!(gib(15 * (1 << 30) / 2 + 1), "7.5 GiB");
    }

    #[test]
    fn the_memory_failure_names_the_amount_against_the_machines_total() {
        // The denominator is this machine's RAM, so only the prefix is pinned.
        let line = exceeded_failure(0, 1 << 30);
        assert!(
            line.starts_with("terminated: exceeded memory limit (used ~0.0 GiB of "),
            "line: {line}"
        );
    }

    #[test]
    fn the_default_limit_is_eighty_percent_of_physical_memory() {
        let Some(total) = physical_memory() else {
            return;
        };
        let limit = default_limit().expect("a known total has a limit");
        // Four fifths, the fraction `default_limit` documents; written here as
        // integer arithmetic so the assertion cannot drift with the float.
        assert_eq!(limit, total.saturating_mul(4) / 5);
    }

    /// The watchdog's own process is measurable: its tree holds at least its own
    /// footprint. Guards each platform's helper against a total regression (a
    /// wrong field, a bad path) without allocating anything.
    #[test]
    fn the_current_process_tree_has_a_nonzero_footprint() {
        let rss = tree_rss(std::process::id()).expect("the measuring process must be readable");
        assert!(rss > 0, "footprint should be non-zero, got {rss}");
    }
}

//! The excerpt of a failed command's output that names the failure.
//!
//! Command output is unbounded and the record that keeps it is not, but the part that diagnoses a
//! failure is not a positional window: a test run prints the failing test's own block in the middle
//! and its verdict at the end, a compiler prints an error before the span and code frame that
//! locate it, and the first and last lines of a long run say nothing about what broke. So the
//! excerpt keeps the lines that *name* the failure ([`FAILURE_LINES`]) with their context and the
//! run's closing lines, taken from both ends of the output so a wall of failures cannot crowd out
//! the verdict; where nothing names it, the output's end comes first, then the lines that only point
//! at a failure ([`WEAK_LINES`]), then any line that mentions one — the rest replaced by omission
//! markers, in output order. The run's closing lines are kept whatever the rule, where a verdict
//! the patterns do not know still sits. The header claims only how much of the output the excerpt
//! holds: a heuristic cannot promise that the line it kept is the one that broke.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::RegexSet;

use super::{TRUNCATED_MIDDLE, omitted_lines, push_line};

/// Bytes the whole excerpt may carry: its header, its lines and the markers between them.
const FAILURE_EXCERPT_BYTES: usize = 12_000;

/// Bytes the body may carry — its lines and the markers between them. The other 600 of
/// [`FAILURE_EXCERPT_BYTES`] are the header and the trailing marker.
const BODY_BUDGET: usize = FAILURE_EXCERPT_BYTES - 600;

/// Bytes the selected lines may carry: the other 800 of [`BODY_BUDGET`] are the markers a
/// marker-dense excerpt needs, so it keeps fewer lines rather than losing its end to them.
const LINE_BUDGET: usize = BODY_BUDGET - 800;

/// Bytes one kept line may carry: a minified or single-line output can print a whole document on
/// one line, and it is the head of that line that identifies the failure.
const MAX_LINE_BYTES: usize = 2_000;

/// Lines kept before a marked line: a test's captured-output header sits above its assertion, and a
/// panic's location above its message.
const LEAD_LINES: usize = 2;

/// Lines kept after a marked line: a compiler's span, code frame and notes follow its error, and a
/// traceback's frames follow its header.
const TRAIL_LINES: usize = 14;

/// Lines the output's end may contribute where nothing names the failure — its verdict is the only
/// content left to keep.
const TAIL_LINES: usize = 60;

/// Lines the output's end may contribute where a line does name the failure: enough for a closing
/// verdict, so an unrecognised one is not lost with the rest of the tail.
const VERDICT_LINES: usize = 5;

/// The lines that name a failure outright — the failing test and its assertion, the compiler's
/// error and the span that locates it, a formatter's diff header, a panic, a traceback, a bare
/// `FAIL`.
static FAILURE_LINES: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        // libtest / cargo test
        r"^test .* \.\.\. FAILED",
        r"^test result: FAILED",
        r"^---- .* stdout ----",
        // rustc diagnostics and a rust panic
        r"^error(\[E\d+\])?:",
        r"^thread '.*' panicked",
        // pytest / python
        r"^E\s{2,}\S",
        r"^_{3,} .* _{3,}$",
        r"^FAILED \S",
        r"^Traceback \(most recent call last\)",
        r"^[\w.]*Error:",
        r"^\s*AssertionError",
        // go
        r"^panic: ",
        r"^--- FAIL: ",
        // javascript / typescript
        r"^\s*● ",
        r"^error TS\d+:",
        r"^\S+\(\d+,\d+\): error TS\d+:",
        r"^\s*\d+:\d+\s+error\s",
        // a location line that names an error, where a linter prints warnings as well
        r"^\S+\.\w+:\d+(:\d+)?: error\b",
        // formatters that print a diff or a file list instead of a diagnostic — rustfmt's current
        // `Diff in <path>:<line>:` and its older `Diff in <path> at line <n>:`
        r"^Diff in \S+",
        r"^would reformat ",
        // bare verdicts and other ecosystems
        r"^(FAILED|FAIL)\b",
        r"^not ok \d+",
        r"\*\*\* .*Error \d+",
        r"^fatal: ",
    ])
    .expect("the failure-line patterns are valid")
});

/// Lines that mark where a failure is without naming it: a test run's list of failing names, a
/// linter's location lines and a formatter's `[warn]`, all of which a run prints for warnings too,
/// so they are kept only with the room that is left.
static WEAK_LINES: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"^failures:",
        r"^=+ FAILURES",
        r"^\S+\.\w+:\d+(:\d+)?: ",
        r"^\[warn\] \S",
    ])
    .expect("the weak-marker patterns are valid")
});

/// The last try, for a tool whose own failure line none of the other patterns knows.
static LOOSE_LINES: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([r"(?i)\b(error|failed|failure|panic|assert|assertion|exception|fatal)\b"])
        .expect("the loose failure-line pattern is valid")
});

/// The part of `output` that names the failure, with context, bounded by
/// [`FAILURE_EXCERPT_BYTES`] — or `output` itself when it already fits, so a short failure reads
/// exactly as the command printed it. The excerpt is stored verbatim in a durable record.
#[must_use]
pub(crate) fn failure_excerpt(output: &str) -> String {
    if output.len() <= FAILURE_EXCERPT_BYTES {
        return output.to_string();
    }
    let lines: Vec<&str> = output.lines().collect();
    let accepted = select(&lines);
    let mut body = String::new();
    let mut shown = 0usize;
    let mut omitted = 0usize;
    for kept in accepted {
        let Some(text) = kept else {
            omitted += 1;
            continue;
        };
        // A gap's marker is part of the record's size too, so a marker-dense excerpt keeps fewer
        // lines rather than growing past the bound.
        let marker = (omitted > 0).then(|| omitted_lines(omitted, "omitted"));
        let cost = text.len() + 1 + marker.as_ref().map_or(0, |marker| marker.len() + 1);
        if body.len() + cost > BODY_BUDGET {
            omitted += 1;
            continue;
        }
        if let Some(marker) = marker {
            push_line(&mut body, &marker);
        }
        omitted = 0;
        push_line(&mut body, &text);
        shown += 1;
    }
    if omitted > 0 {
        push_line(&mut body, &omitted_lines(omitted, "omitted"));
    }
    format!("{}\n{body}", header(shown, lines.len()))
}

/// The header every excerpt starts with: how much of the output it holds, and that a heuristic
/// cannot promise the failure is among it.
fn header(shown: usize, total: usize) -> String {
    format!(
        "[failure excerpt: {shown} of {total} lines — the lines that name a failure and the run's \
         end, where they fit; the failure may not be among them]"
    )
}

/// Which lines the excerpt keeps. The run's closing lines come first, whatever else is found: a
/// verdict the patterns do not know is still the line a reader needs. Then the lines that name the
/// failure, then the ones that only point at it — and, where nothing is marked at all, any line
/// that mentions one, so a stray mention in the head cannot hide the verdict.
fn select<'a>(lines: &[&'a str]) -> Vec<Option<Cow<'a, str>>> {
    let strong = windowed(lines, &FAILURE_LINES);
    let weak = windowed(lines, &WEAK_LINES);
    let names = strong.iter().any(|kept| *kept);
    let points = weak.iter().any(|kept| *kept);
    let mut accepted = vec![None; lines.len()];
    let mut used = 0usize;
    // A capture gap is the output's own disclosure that part of it was never captured: it is kept
    // wherever the failure sits, before anything else can spend the room.
    for (index, line) in lines.iter().enumerate() {
        if line.trim() == TRUNCATED_MIDDLE {
            accepted[index] = Some(Cow::Borrowed(*line));
            used += line.len() + 1;
        }
    }
    let verdict = if names || points {
        VERDICT_LINES
    } else {
        TAIL_LINES
    };
    accept(&tail_mask(lines, verdict), &mut accepted, &mut used, lines);
    if names {
        accept(&strong, &mut accepted, &mut used, lines);
        accept(&weak, &mut accepted, &mut used, lines);
    } else if points {
        accept(&weak, &mut accepted, &mut used, lines);
    } else {
        accept(
            &windowed(lines, &LOOSE_LINES),
            &mut accepted,
            &mut used,
            lines,
        );
    }
    accepted
}

/// `patterns`' matches widened by [`LEAD_LINES`] before and [`TRAIL_LINES`] after each, merged where
/// the windows overlap.
fn windowed(lines: &[&str], patterns: &RegexSet) -> Vec<bool> {
    let mut keep = vec![false; lines.len()];
    for (index, line) in lines.iter().enumerate() {
        if patterns.is_match(line) {
            let start = index.saturating_sub(LEAD_LINES);
            let end = (index + TRAIL_LINES + 1).min(lines.len());
            keep[start..end].fill(true);
        }
    }
    keep
}

/// Take the lines `mask` marks while the budget lasts, alternately from the output's two ends: a
/// wall of markers cannot spend the room before the failure the run ends on. A marked line the
/// budget cannot hold whole keeps its head — dropping it would lose the identification the excerpt
/// exists for — and the selection ends there.
fn accept<'a>(
    mask: &[bool],
    accepted: &mut [Option<Cow<'a, str>>],
    used: &mut usize,
    lines: &[&'a str],
) {
    let marked: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, marked)| marked.then_some(index))
        .collect();
    let mut front = 0usize;
    let mut back = marked.len();
    let mut from_front = true;
    while front < back {
        let index = if from_front {
            front += 1;
            marked[front - 1]
        } else {
            back -= 1;
            marked[back]
        };
        from_front = !from_front;
        if accepted[index].is_some() {
            continue;
        }
        let line = lines[index];
        let text = if line.len() > MAX_LINE_BYTES {
            Cow::Owned(cut_line(line, MAX_LINE_BYTES))
        } else {
            Cow::Borrowed(line)
        };
        let remaining = LINE_BUDGET.saturating_sub(*used);
        if text.len() < remaining {
            *used += text.len() + 1;
            accepted[index] = Some(text);
            continue;
        }
        // A marked line the budget cannot hold whole keeps its head: dropping it would lose the
        // identification the excerpt exists for.
        let head_bytes = remaining.saturating_sub(1);
        if line.floor_char_boundary(head_bytes) == 0 {
            break;
        }
        accepted[index] = Some(Cow::Owned(cut_line(line, head_bytes)));
        *used = LINE_BUDGET;
        break;
    }
}

/// The head of `line` up to `bytes`, then what the reader cannot see of it — a one-line output can
/// print a whole document, and a bare ellipsis would leave the loss unstated.
fn cut_line(line: &str, bytes: usize) -> String {
    let head = &line[..line.floor_char_boundary(bytes)];
    format!(
        "{head}… ({} more chars on this line)",
        line[head.len()..].chars().count()
    )
}

/// The last `count` lines that fit the budget — where a run prints its verdict. Sized to the budget
/// up front, so the very end is never what the cut drops.
fn tail_mask(lines: &[&str], count: usize) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    let mut used = 0usize;
    for (index, line) in lines.iter().enumerate().rev().take(count) {
        used += line.len().min(MAX_LINE_BYTES) + 1;
        if used > LINE_BUDGET {
            break;
        }
        mask[index] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filler that names no failure.
    fn noise(prefix: &str, count: usize) -> String {
        (1..=count)
            .map(|i| format!("{prefix} {i} of the run"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn short_output_is_returned_whole() {
        let output = "error: broken\n  --> src/lib.rs:1:1";
        assert_eq!(failure_excerpt(output), output);
    }

    #[test]
    fn a_compiler_error_keeps_its_span_and_code_frame() {
        let output = format!(
            "{}\nerror[E0308]: mismatched types\n  --> src/main.rs:3:9\n   |\n\
             3 |     let x: u8 = \"s\";\n   |         ^^^^^^ expected `u8`, found `&str`\n{}",
            noise("Compiling", 300),
            noise("Finished", 300),
        );
        let excerpt = failure_excerpt(&output);
        for needle in [
            "error[E0308]: mismatched types",
            "--> src/main.rs:3:9",
            "3 |     let x: u8",
            "expected `u8`, found `&str`",
        ] {
            assert!(excerpt.contains(needle), "{excerpt} lacks {needle:?}");
        }
    }

    #[test]
    fn a_format_check_diff_is_named_by_its_header() {
        let mut lines = vec!["Diff in /repo/src/main.rs:1:".to_string()];
        lines.extend((1..=600).map(|i| format!(" unchanged context line {i}")));
        let output = lines.join("\n");
        let excerpt = failure_excerpt(&output);
        assert!(
            excerpt.contains("Diff in /repo/src/main.rs:1:"),
            "{excerpt}"
        );
        assert!(excerpt.starts_with("[failure excerpt:"), "{excerpt}");
    }

    #[test]
    fn nothing_named_keeps_the_end_of_the_output() {
        let excerpt = failure_excerpt(&noise("ok", 1000));
        assert!(excerpt.starts_with("[failure excerpt:"), "{excerpt}");
        assert!(excerpt.contains("ok 1000 of the run"), "{excerpt}");
        assert!(!excerpt.contains("\nok 1 of the run"), "{excerpt}");
    }

    #[test]
    fn a_tail_too_long_for_the_budget_keeps_its_very_end() {
        let output = (1..=100)
            .map(|i| format!("ok {i} {}", "x".repeat(300)))
            .collect::<Vec<_>>()
            .join("\n");
        let excerpt = failure_excerpt(&output);
        assert!(excerpt.contains("ok 100 "), "the last line is kept");
        assert!(
            !excerpt.contains("\nok 1 x"),
            "the front of the tail is what goes"
        );
    }

    /// However much names the failure — one wall of it, or many scattered ones — the record stays
    /// within its bound and keeps the first and the last.
    #[test]
    fn the_excerpt_is_bounded_whatever_names_the_failure() {
        let wall = (1..=5000)
            .map(|i| format!("error: failure {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut scattered = Vec::new();
        for i in 1..=200 {
            scattered.push(format!("error: failure {i}"));
            scattered.extend((1..=40).map(|j| format!("noise {j} around failure {i}")));
        }
        for (output, last) in [
            (wall, "error: failure 5000"),
            (scattered.join("\n"), "error: failure 200"),
        ] {
            let excerpt = failure_excerpt(&output);
            assert!(
                excerpt.len() <= FAILURE_EXCERPT_BYTES,
                "{} bytes",
                excerpt.len()
            );
            assert!(excerpt.contains("error: failure 1"), "{excerpt}");
            assert!(excerpt.contains(last), "{excerpt} lacks {last:?}");
            assert!(excerpt.contains("lines omitted"), "{excerpt}");
        }
    }

    /// A run whose head names nothing: the head is dropped and the middle kept — the failing
    /// block with its own wording, and the run's closing verdict.
    #[test]
    fn a_failing_test_in_the_middle_survives_its_surroundings() {
        let output = format!(
            "{}\n---- test_add stdout ----\n\
             thread 'test_add' panicked at src/lib.rs:10:5:\n\
             assertion `left == right` failed\n  left: 1\n right: 2\n\
             test result: FAILED. 800 passed; 1 failed\n\
             error: test failed, to rerun pass `--lib`\n{}",
            noise("checking", 400),
            noise("cleanup", 400),
        );
        let excerpt = failure_excerpt(&output);
        for needle in [
            "---- test_add stdout ----",
            "assertion `left == right` failed",
            "left: 1",
            "test result: FAILED. 800 passed; 1 failed",
            "error: test failed, to rerun pass `--lib`",
        ] {
            assert!(excerpt.contains(needle), "{excerpt} lacks {needle:?}");
        }
        assert!(!excerpt.contains("checking 1 of the run"), "{excerpt}");
        assert!(!excerpt.contains("cleanup 100 of the run"), "{excerpt}");
        assert!(excerpt.contains("lines omitted"), "{excerpt}");
        assert!(excerpt.starts_with("[failure excerpt:"), "{excerpt}");
    }

    /// A verdict no pattern knows is still kept: with nothing else marked, and behind an
    /// incidental marker that looks like a failure of its own.
    #[test]
    fn a_verdict_the_patterns_do_not_know_is_kept() {
        let mut quiet = vec!["note: failed to load optional plugin config, continuing".to_string()];
        quiet.extend((1..=600).map(|i| format!("step {i} of the build")));
        let mut incidental = vec![
            "thread 'idle_worker' panicked at src/pool.rs:4:5: the worker gave up".to_string(),
        ];
        incidental.extend((1..=600).map(|i| format!("step {i} of the build")));
        for mut output in [quiet, incidental] {
            output.push("BUILD BROKEN: cannot find module 'left-pad'".to_string());
            let excerpt = failure_excerpt(&output.join("\n"));
            assert!(
                excerpt.contains("BUILD BROKEN: cannot find module 'left-pad'"),
                "{excerpt}"
            );
        }
    }

    /// A linter prints a location line for every warning, so a wall of them must not spend the room
    /// the run's own verdict needs.
    #[test]
    fn a_wall_of_locations_does_not_spend_the_verdict() {
        let mut lines: Vec<String> = (1..=600)
            .map(|i| format!("src/app.js:{}:3: warning: unused variable `v`", i + 10))
            .collect();
        lines.extend((1..=60).map(|i| format!("step {i} of the build")));
        lines.push("BUILD BROKEN: cannot find module 'left-pad'".to_string());
        let excerpt = failure_excerpt(&lines.join("\n"));
        assert!(
            excerpt.contains("BUILD BROKEN: cannot find module 'left-pad'"),
            "{excerpt}"
        );
    }

    /// `[warn]` names a warning, so hundreds of them must not take the room of the error the run
    /// fails on — even with the warnings on both sides of it.
    #[test]
    fn a_wall_of_warnings_does_not_spend_the_errors_room() {
        let output = format!(
            "{}\nerror[E0308]: mismatched types\n  --> src/main.rs:3:9\n   |\n\
             3 |     let x: u8 = \"s\";\n   |         ^^^^^^ expected `u8`, found `&str`\n{}",
            noise("[warn] unused variable", 400),
            noise("[warn] unused import", 400),
        );
        let excerpt = failure_excerpt(&output);
        for needle in [
            "error[E0308]: mismatched types",
            "--> src/main.rs:3:9",
            "expected `u8`, found `&str`",
        ] {
            assert!(excerpt.contains(needle), "{excerpt} lacks {needle:?}");
        }
    }

    /// One line longer than the whole budget keeps its head, and does not take the room a later
    /// failure needs.
    #[test]
    fn a_line_too_long_for_the_budget_keeps_its_head() {
        let long = format!("error: the module failed to load: {}", "x".repeat(20_000));
        let output = format!(
            "{}\n{long}\n\nerror: the second thing that broke\n{}",
            noise("Compiling", 300),
            noise("Finished", 300),
        );
        let excerpt = failure_excerpt(&output);
        assert!(
            excerpt.contains("error: the module failed to load"),
            "{excerpt}"
        );
        assert!(
            excerpt.contains("error: the second thing that broke"),
            "{excerpt}"
        );
        assert!(
            excerpt.contains("more chars on this line"),
            "the cut says how much: {excerpt}"
        );
        assert!(
            excerpt.len() <= FAILURE_EXCERPT_BYTES,
            "{} bytes",
            excerpt.len()
        );
    }

    /// Many failures: the record keeps the first of them and the run's closing verdict, not only
    /// as many early ones as the budget holds.
    #[test]
    fn many_failures_keep_the_runs_verdict() {
        let mut lines = Vec::new();
        for i in 1..=100 {
            lines.extend([
                format!("test case_{i} ... FAILED"),
                format!("---- case_{i} stdout ----"),
                format!("thread 'case_{i}' panicked at src/lib.rs:{i}:5:"),
                "assertion `left == right` failed".to_string(),
                "  left: 1".to_string(),
                " right: 2".to_string(),
            ]);
        }
        lines.push("test result: FAILED. 0 passed; 100 failed".to_string());
        let excerpt = failure_excerpt(&lines.join("\n"));
        for needle in [
            "test case_1 ... FAILED",
            "test result: FAILED. 0 passed; 100 failed",
        ] {
            assert!(excerpt.contains(needle), "{excerpt} lacks {needle:?}");
        }
    }

    /// A capture gap is the output's own disclosure that its middle was never captured: the record
    /// keeps it wherever the failure sits, so it never reads as a positional window.
    #[test]
    fn a_capture_gap_is_kept_wherever_the_failure_sits() {
        let output = format!(
            "{}\n{}\n{}\n---- test_x stdout ----\n\
             thread 'test_x' panicked at src/lib.rs:1:1:\nassertion failed\n",
            noise("checking", 500),
            crate::util::TRUNCATED_MIDDLE,
            noise("still checking", 500),
        );
        let excerpt = failure_excerpt(&output);
        assert!(excerpt.contains("was not captured"), "{excerpt}");
        assert!(excerpt.contains("thread 'test_x' panicked"), "{excerpt}");
    }
}

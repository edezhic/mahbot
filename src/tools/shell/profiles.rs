//! Shell command output filter profiles.
//!
//! Each profile defines how to filter and truncate output for a specific
//! command or command family. Profiles are compiled once at module init
//! via [`LazyLock`] and selected by [`crate::tools::shell::select_profile`].

use regex::{Regex, RegexSet};
use std::sync::LazyLock;

// ── Profile data structures ───────────────────────────────────────────

/// A short-circuit rule: if output matches `pattern` (and `unless` is absent or not matched),
/// return `message` without further processing.
pub(crate) struct ShortCircuit {
    pub(crate) pattern: Regex,
    pub(crate) message: &'static str,
    pub(crate) unless: Option<Regex>,
}

/// A compiled output filter profile. Built once at init via [`Profile::new`] + builder methods.
pub(super) struct Profile {
    pub(super) match_command: Regex,
    pub(super) strip_lines: Option<RegexSet>,
    pub(super) keep_stderr: Option<RegexSet>,
    pub(super) short_circuits: Vec<ShortCircuit>,
    pub(super) max_line_len: Option<usize>,
    pub(super) head_lines: Option<usize>,
    pub(super) tail_lines: Option<usize>,
    pub(super) max_lines: Option<usize>,
    pub(super) on_empty: Option<&'static str>,
    /// Optional transform that replaces the pipeline output after line-level
    /// processing but before `combine_output` and `finish_shell_output`.
    /// Receives the processed output and exit code, returns the transformed output.
    pub(super) output_transform: Option<fn(&str, exit_code: i32) -> String>,
    /// When true, the output_transform is only applied to standalone commands
    /// (single segment). For chained commands (`&&`, `||`, `;`, `|`), the
    /// transform is skipped and the output passes through as-is. This prevents
    /// transforms that assume homogeneous output (e.g., compact_ls) from
    /// silently dropping output produced by later command segments.
    pub(super) standalone_only: bool,
}

impl Profile {
    /// Create a new profile that matches `command_pattern` (a regex against the canonical command).
    pub(super) fn new(match_command: &str) -> Self {
        Self {
            match_command: Regex::new(match_command).expect("bad match_command regex"),
            strip_lines: None,
            keep_stderr: None,
            short_circuits: Vec::new(),
            max_line_len: None,
            head_lines: None,
            tail_lines: None,
            max_lines: None,
            on_empty: None,
            output_transform: None,
            standalone_only: false,
        }
    }

    fn strip(mut self, patterns: &[&str]) -> Self {
        self.strip_lines = Some(RegexSet::new(patterns).expect("bad strip_lines regex"));
        self
    }

    /// Like [`Self::strip`], but takes a pre-compiled [`RegexSet`] directly (e.g. from a
    /// `LazyLock`-cached set). Clones the set (cheap — internally reference-counted).
    fn strip_set(mut self, patterns: &RegexSet) -> Self {
        self.strip_lines = Some(patterns.clone());
        self
    }

    fn keep_stderr(mut self, patterns: &[&str]) -> Self {
        self.keep_stderr = Some(RegexSet::new(patterns).expect("bad keep_stderr regex"));
        self
    }

    fn short_circuit(mut self, pattern: &str, message: &'static str) -> Self {
        self.short_circuits.push(ShortCircuit {
            pattern: Regex::new(pattern).expect("bad short_circuit pattern"),
            message,
            unless: None,
        });
        self
    }

    fn short_circuit_unless(mut self, pattern: &str, message: &'static str, unless: &str) -> Self {
        self.short_circuits.push(ShortCircuit {
            pattern: Regex::new(pattern).expect("bad short_circuit pattern"),
            message,
            unless: Some(Regex::new(unless).expect("bad unless pattern")),
        });
        self
    }

    const fn max_line_len(mut self, n: usize) -> Self {
        self.max_line_len = Some(n);
        self
    }

    pub(super) const fn head(mut self, n: usize) -> Self {
        self.head_lines = Some(n);
        self
    }

    pub(super) const fn tail(mut self, n: usize) -> Self {
        self.tail_lines = Some(n);
        self
    }

    pub(super) const fn max(mut self, n: usize) -> Self {
        self.max_lines = Some(n);
        self
    }

    const fn on_empty(mut self, msg: &'static str) -> Self {
        self.on_empty = Some(msg);
        self
    }

    /// Set an output transform that replaces the pipeline output after line-level
    /// processing but before `combine_output` and `finish_shell_output`.
    fn output_transform(mut self, transform: fn(&str, exit_code: i32) -> String) -> Self {
        self.output_transform = Some(transform);
        self
    }

    /// Restrict the `output_transform` to standalone commands only (no chaining).
    /// When set, `select_profile` skips this profile entirely for chained commands
    /// (`&&`, `||`, `;`, `|`), causing them to fall through to `GEN_FALLBACK`
    /// with its sensible truncation defaults. This is useful for transforms like
    /// `compact_ls` that assume homogeneous output and would silently drop
    /// output from later command segments.
    const fn standalone_only(mut self) -> Self {
        self.standalone_only = true;
        self
    }
}

// ── Shared strip/keep patterns ──────────────────────────────────────────

/// Canonical list of cargo compilation noise prefixes (plain text).
///
/// This is the single source of truth for cargo compilation noise patterns.
/// - `profiles.rs` builds regex patterns (`^\s*<prefix>\s`) from these for the profile pipeline.
/// - `filter_cargo_test_output` uses `starts_with` checks against these to skip compilation noise.
///
/// NOTE: `Running` is intentionally included — it is noise in cargo build output
/// (`Running \`rustc ...\``). However, `filter_cargo_test_output` explicitly skips `Running`
/// because `Running unittests src/lib.rs` is useful context in test output.
pub(super) const CARGO_COMPILE_PREFIXES: &[&str] = &[
    "Compiling",
    "Checking",
    "Downloading",
    "Downloaded",
    "Finished",
    "Fresh",
    "Blocking",
    "Documenting",
    "Running",
];

/// Derived [`RegexSet`] from [`CARGO_COMPILE_PREFIXES`], compiled once at init.
/// Used by `cargo_tool` and other cargo profiles via `Profile::strip_set`.
static CARGO_COMPILE_STRIP: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new(
        CARGO_COMPILE_PREFIXES
            .iter()
            .map(|p| format!(r"^\s*{}\s", regex::escape(p))),
    )
    .expect("bad cargo compile strip regex")
});

/// Stderr lines *kept* through cargo compile output filtering.
const CARGO_COMPILE_KEEP_STDERR: &[&str] = &[r"warning:", r"^error"];

/// Blank/empty line pattern. Used in `.strip()` calls to remove whitespace-only lines.
const BLANK_LINE: &str = r"^\s*$";

// ── Helper builders ─────────────────────────────────────────────────

/// Build a "small utility" profile: strips blanks, truncates long lines, caps at `max_lines`.
fn small_util(matcher: &str, max_lines: usize) -> Profile {
    Profile::new(matcher)
        .strip(&[BLANK_LINE])
        .max_line_len(120)
        .max(max_lines)
}

/// Build a cargo subcommand profile (build/check). Strips compilation noise,
/// keeps stderr warnings/errors, signals clean success via on_empty.
fn cargo_tool(matcher: &str, max_lines: usize, ok_msg: &'static str) -> Profile {
    Profile::new(matcher)
        .strip_set(&CARGO_COMPILE_STRIP)
        .keep_stderr(CARGO_COMPILE_KEEP_STDERR)
        .max(max_lines)
        .on_empty(ok_msg)
}

// ── Profile definitions ─────────────────────────────────────────────

/// Generic fallback applied to all commands that don't match a specific profile.
pub(super) static GEN_FALLBACK: LazyLock<Profile> = LazyLock::new(|| {
    Profile::new("")
        .max_line_len(500)
        .head(10)
        .tail(10)
        .max(200)
});

/// Compiled profiles, built once on first access.
pub(super) static PROFILES: LazyLock<Vec<Profile>> = LazyLock::new(|| {
    vec![
        // ── General shell ────────────────────────────────────────────────
        Profile::new(r"^df\b").max_line_len(80).max(20),
        // Simple utilities: strip blanks, truncate long lines, cap at 50
        small_util(r"^(?:du|shellcheck|stat|ps|rustc)\b", 50),
        Profile::new(r"^make\b")
            .strip(&[r"make\[\d+\]:", BLANK_LINE, r"Nothing to be done"])
            .max(50)
            .on_empty("make: ok"),
        Profile::new(r"^ping\b")
            .strip(&[r"^\d+ bytes from ", r"^Reply from "])
            .tail(4),
        Profile::new(r"^rsync\b")
            .strip(&[BLANK_LINE, r"sending incremental", r"sent \d+"])
            .short_circuit_unless(
                r"total size is",
                "ok (synced)",
                r"error|failed|No such file",
            )
            .max(20),
        Profile::new(r"^ssh\b")
            .strip(&[
                BLANK_LINE,
                r"Warning: Permanently added",
                r"^debug1:",
                r"Authenticated to",
            ])
            .max(30),
        Profile::new(r"^systemctl\s+status\b")
            .strip(&[BLANK_LINE])
            .head(5)
            .tail(10),
        // ── Container & infra ─────────────────────────────────────────────
        Profile::new(r"^docker\b")
            .strip(&[
                BLANK_LINE,
                r"^Step\s+\d+/",
                r"^ --->",
                r"^ ---> Using cache",
                r"^Successfully built",
                r"^Successfully tagged",
            ])
            .short_circuit_unless(
                r"Successfully built",
                "[docker build: ok]",
                r"error|Error|failed|FAILED",
            )
            .max(40)
            .on_empty("[docker: ok]"),
        Profile::new(r"^gh\b")
            .strip(&[BLANK_LINE, r"^\s*-\s", r"warning:.*(?:gh|GitHub)"])
            .short_circuit_unless(
                r"(\u{2713}|Done|Successfully)",
                "[gh: ok]",
                r"error|Error|failed|FAILED",
            )
            .tail(10)
            .max(30)
            .on_empty("[gh: ok]"),
        Profile::new(r"^helm\b")
            .strip(&[BLANK_LINE, r"^\s*STATUS:", r"^\s*v\d+\.\d+\.\d+\s"])
            .tail(10)
            .max(30)
            .on_empty("[helm: ok]"),
        // ── VCS ─────────────────────────────────────────────────────────
        Profile::new(r"^git\s+log\b")
            .strip(&[BLANK_LINE])
            .max_line_len(200)
            .head(20)
            .max(50),
        Profile::new(r"^git\s+diff\b").short_circuit(r"^\s*$", "[git diff: no changes]"),
        // ── Rust toolchain ────────────────────────────────────────────────
        cargo_tool(r"^cargo\s+(build|check)\b", 50, "[cargo: ok]"),
        cargo_tool(r"^cargo\s+clippy\b", 50, "[cargo clippy: ok]"),
        Profile::new(r"^cargo\s+fmt\b")
            .short_circuit(r"^Formatted\s", "[cargo fmt: ok]")
            .max(50)
            .on_empty("[cargo fmt: ok]"),
        Profile::new(r"^cargo\s+install\b")
            .strip_set(&CARGO_COMPILE_STRIP)
            .max(30),
        Profile::new(r"^rustup\b")
            .strip(&[r"^info:", r"syncing", r"downloading", r"installing"])
            .max(20),
        // ── JS/TS ecosystem ──────────────────────────────────────────────
        Profile::new(r"^npm\s+install\b")
            .strip(&[
                r"^\s*added\s",
                r"^\s*removed\s",
                r"^\s*changed\s",
                r"^\s*audited\s",
            ])
            .short_circuit_unless(r"up to date", "npm: up to date", r"ERR|error|failed")
            .max(30),
        Profile::new(r"^npm\s+audit\b|^pnpm\s+audit\b")
            .strip(&[BLANK_LINE])
            .head(5)
            .tail(10)
            .on_empty("[npm audit: clean]"),
        Profile::new(r"^npm\s+run\b|^pnpm\s+run\b|^yarn\s+(run\b)")
            .strip(&[BLANK_LINE, r"^> .+@", r"^npm ERR!", r"^ERR!"])
            .short_circuit_unless(
                r"success|Done in",
                "[npm run: ok]",
                r"error|Error|failed|FAILED|ERR",
            )
            .tail(15)
            .max(40),
        small_util(r"^(?:npx|biome|oxlint|ruff)\b", 30),
        Profile::new(r"^pnpm\s+install\b")
            .strip(&[
                r"Already up to date",
                r"Progress:",
                r"Resolving:",
                r"Downloading:",
            ])
            .max(30),
        Profile::new(r"^tsc\b")
            .strip(&[BLANK_LINE])
            .keep_stderr(CARGO_COMPILE_KEEP_STDERR)
            .max(50)
            .on_empty("[tsc: ok]"),
        Profile::new(r"^vitest\b")
            .strip(&[r"^\s*(stdout|PASS|SKIP)\s"])
            .tail(20)
            .max(60),
        Profile::new(r"^eslint\b")
            .strip(&[BLANK_LINE])
            .keep_stderr(CARGO_COMPILE_KEEP_STDERR)
            .max(50)
            .on_empty("[eslint: ok]"),
        Profile::new(r"^prettier\b")
            .strip(&[BLANK_LINE])
            .short_circuit(r"unchanged", "prettier: ok")
            .max(20),
        Profile::new(r"^next\s+build\b")
            .strip(&[BLANK_LINE, r"^\s*✓", r"^info\s+-"])
            .tail(10)
            .max(40),
        small_util(r"^playwright\b", 40).tail(15),
        Profile::new(r"^prisma\b")
            .strip(&[BLANK_LINE, r"Environment variables", r"Prisma schema"])
            .max(30),
        small_util(r"^nx\b", 30)
            .strip(&[BLANK_LINE, r"NX\s"])
            .on_empty("[nx: ok]"),
        Profile::new(r"^turbo\b")
            .strip(&[BLANK_LINE, r"^\s*•"])
            .tail(10)
            .max(40),
        Profile::new(r"^jest\b")
            .strip(&[r"^\s*PASS\s", r"^\s*Tests:\s"])
            .tail(15)
            .max(40),
        Profile::new(r"^yarn\b")
            .strip(&[r"^info\s", r"\[\d+/\d+\]"])
            .max(20),
        // ── Python ecosystem ──────────────────────────────────────────────
        // NOTE: Only `^pytest\b` is needed here — `python -m pytest` and
        // `poetry run pytest` intentionally fall through to GEN_FALLBACK
        // because canonical_command() strips intermediate flags (like `-m`)
        // and doesn't treat `pytest` as a subcommand after `run`. The
        // canonical forms become `python pytest` and `poetry run`
        // respectively, neither of which match `^pytest\b`.
        Profile::new(r"^pytest\b")
            .strip(&[
                BLANK_LINE,
                r"^\s*\.+\s*$",
                r"^\s*collected\s+\d+",
                r"^\s*={3,}\s",
            ])
            .short_circuit_unless(r"passed", "[pytest: passed]", r"failed|error|FAILED|ERROR")
            .tail(20)
            .max(60)
            .on_empty("[pytest: passed]"),
        Profile::new(r"^pip\s+install\b")
            .strip(&[
                BLANK_LINE,
                r"^Collecting\s",
                r"^\s*Downloading\s",
                r"^\s*Installing\s",
                r"^Successfully installed\s",
            ])
            .short_circuit(r"already satisfied", "[pip: already satisfied]")
            .max(30),
        // ── Query & search ──────────────────────────────────────────────
        Profile::new(r"^fd\b")
            .strip(&[BLANK_LINE])
            .head(20)
            .tail(10)
            .max(50),
        Profile::new(r"^rg\b|^grep\b")
            .strip(&[BLANK_LINE, r"^Binary\s+file\s+\S+\s+matches"])
            .head(20)
            .tail(10)
            .max(60),
        // ── Test frameworks ──────────────────────────────────────────────
        Profile::new(r"^mocha\b")
            .strip(&[BLANK_LINE, r"^\s*(✓|✗|√|×)\s"])
            .tail(15)
            .max(40),
        // ── Terraform ────────────────────────────────────────────────────
        Profile::new(r"^terraform\b")
            .strip(&[BLANK_LINE, r"^Initializing", r"^Terraform has been"])
            .short_circuit_unless(r"No changes", "[terraform: no changes]", r"error|Error")
            .head(5)
            .tail(15)
            .max(40)
            .on_empty("[terraform: ok]"),
        // ── Cargo test / nextest ────────────────────────────────────────
        Profile::new(r"^cargo\s+(test|nextest)\b")
            .output_transform(super::filter_cargo_test_output)
            .standalone_only(),
        // ── ls ─────────────────────────────────────────────────────────────
        Profile::new(r"^ls\b")
            .output_transform(super::compact_ls)
            .standalone_only(),
    ]
});

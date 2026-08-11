//! Subprocess end-to-end harness for the transparent grep engine.
//!
//! The parent role rewrites commands via the engine's analysis, spawns
//! `sh -c` with the bench binary acting as the hidden `__grep-engine`
//! subcommand, and diffs the result against the real system grep. The child
//! role dispatches the engine from argv (the same code path `main()` uses).
//!
//! Run with: `cargo bench --no-default-features --features grep-engine-e2e --bench grep_engine_e2e`
//!
//! Exit codes: 0 = all differential rows byte-identical, 1 = mismatch, 2 = harness error.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("__grep-engine") {
        let code = mahbot::run_grep_engine(&args[1..]);
        std::process::exit(code);
    }
    if args.first().map(String::as_str) == Some("--probe") {
        std::process::exit(0);
    }
    std::process::exit(run_matrix());
}

/// Full subprocess-path differential: the rewritten command through a real
/// shell against the real system grep (macOS BSD).
fn run_matrix() -> i32 {
    if !cfg!(target_os = "macos") {
        eprintln!("subprocess matrix is macOS-gated (BSD grep parity target)");
        return 2;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let ws = tmp.path().join("ws");
    let home = tmp.path().join("home");
    fs::create_dir_all(&ws).expect("ws");
    fs::create_dir_all(&home).expect("home");
    build_fixture(&ws, &home);

    let rows: &[&str] = &[
        "grep -rn x plain",
        "grep -n x a.txt b.txt",
        "grep -rn needle sub",
        "cd sub && grep -rn needle .",
        "grep -rn needle ~/htree",
        "grep -rn needle 'weird dir'",
        "grep -rn --exclude='c.txt' x plain",
        "grep -rn needle bindir",
        "grep -rn needle bindir/bin5.dat bindir/bin4.dat",
        "grep x missing.txt a.txt",
        "grep -on o a.txt b.txt",
        "grep -n -A1 -B1 'd\\|h' ctx.txt a.txt",
        "grep -x foo x1.txt a.txt",
        "grep -v foo a.txt b.txt",
        "grep -m2 a m.txt c.txt",
        "grep -rn x plain | head -3",
        // ── Raw-token operands: quoted globs/~ stay literal (multi-operand,
        //    so the production serve gate applies) ──
        "grep -rn needle '*.txt' a.txt",
        "grep -rn needle '~' a.txt",
        "grep -rn needle 'a\"b.txt' a.txt",
        // Pin: double-quoted glob stays literal. Red pre-fix (the glob leaked
        // a"b.txt's needle); keep that fixture line needle-bearing.
        "grep -rn needle \"*.txt\" a.txt",
        // ── Relaxed pipeline gate: any length, grep first, >256 KiB stream ──
        "grep -rn needle bigdir | wc -l",
        "grep -rn needle bigdir | tail -1",
        "grep -rn needle bigdir | sort",
        "grep -rn needle bigdir | tee teeout.txt",
        // 3+ members and grep chains: every member after the served first
        // grep is a real tool on the byte-identical uncapped stream.
        "grep -rn needle bigdir | grep -v '^bigdir/big' | head -5",
        "grep -rn needle bigdir | grep needle | grep needle | head -3",
        "grep -rn needle bigdir | grep -c needle", // 2-member grep|grep
        "grep -rn needle bigdir | sort | head -3",
        "grep -rn needle bigdir | sort | uniq | head -3",
        // Grep-introducer tails (triage: preserved verbatim — real tools on the
        // byte-identical stream): xargs grep, sh -c, cd.
        "grep -rln needle bigdir | xargs grep needle",
        "grep -rn needle bigdir | sh -c 'grep -c needle'",
        "grep -rn needle bigdir | cd /tmp | wc -l",
        // 2>&1: benign ordering only (missing operand after matches; wc is
        // order-insensitive). Engine stderr flushes at finish vs BSD at
        // operand-open — missing.txt-first or sort tails diverge (accepted).
        "grep -rn needle bigdir missing.txt 2>&1 | wc -l",
        "grep -rn needle 'weird dir' | wc -l", // quoted spaced operand through the pipe
        // Quoted `~` + unquoted glob: literal `sub/~` dir in cwd (no home-strip).
        "grep -rn needle 'sub/~'/*.txt a.txt",
        // ── Producer-first pipelines: the first grep in a pipeline is served
        //    from the producer's stdin, wherever it sits ──
        "cat a.txt | grep foo", // the flipped negative oracle
        "cat a.txt | grep -n foo | head -2",
        "cat a.txt | grep -c foo | wc -l", // bare count through a tail
        "cat a.txt | grep -l --null foo",  // NUL-terminated stdin name
        "cat a.txt | grep -m1 foo | head -1", // -m stops consuming stdin
        "cat a.txt | sort | grep foo",     // multi-member producer
        "cd sub && cat s.txt | grep needle", // cd chain before the pipeline
        "seq 1 100 | grep 5 | head -5",
        "seq 1 100 | grep 5 | grep 5 | head -3", // grep-on-grep: second is preserved
        "seq 1 100 | grep 5 | grep -v 5 | head -3", // -v on the preserved chain
        "seq 1 100000 | grep -m3 5",             // early exit on an unbounded producer
        "seq 1 1000000 | grep zzz",              // never-match full scan (reads to EOF)
        // Real producers (deterministic): git log and pager-env.
        "git log --oneline | grep Initial | head -1",
        "GIT_PAGER=cat git log --oneline | grep Initial",
        // 2>&1 merged-stderr ordering (producer-side, before the engine).
        "ls /nonexistent-zzz 2>&1 | grep No",
        "ls /nonexistent-aaa /nonexistent-bbb 2>&1 | grep -n No",
        // Member-side 2>&1 on a stdin-fed member: the stream-size marker must
        // be suppressed (it would corrupt the merged pipeline).
        "cat a.txt | grep foo 2>&1 | wc -l",
        // Shell-level `exec 2>&1` merge before the pipeline: the marker would
        // leak into the agent-visible stdout, where the parent's strip cannot
        // reach it — suppressed fail-closed (exec is POSIX; `|&` is not, and
        // /bin/sh rejects it, so the connector-level merge is unreachable).
        "exec 2>&1; cat a.txt | grep foo",
        // Binary stdin: NUL in the 32 KiB window → the Binary message; NUL
        // after the window → text mode, the NUL-containing line still matches.
        "perl -e 'print \"\\0needle\\n\"' | grep needle",
        "perl -e 'print \"x\" x 32768, \"\\0needle\\n\"' | grep needle",
        // `>|` noclobber override preserved on a stdin-fed member.
        "seq 1 100 | grep 5 >| /dev/null",
        // ── -c / -l ──
        "grep -c foo a.txt b.txt",
        "grep -c foo a.txt b.txt c.txt",
        "grep -l foo a.txt b.txt c.txt",
        "grep -cl foo a.txt b.txt c.txt",
        "grep -ch foo a.txt b.txt",
        "grep -cn foo a.txt b.txt",
        "grep -co foo a.txt b.txt",
        "grep -cv foo a.txt b.txt",
        "grep -c -m2 a m.txt c.txt",
        "grep -cl -m5 a m.txt c.txt",
        "grep -cr x plain",
        "grep -lr x plain",
        "grep -clr x plain",
        "grep -cr needle bindir",
        "grep -lr needle bindir",
        "grep -clr needle bindir",
        "grep -cr needle sub",
        "grep -c x missing.txt a.txt",
        "grep -l --null foo a.txt b.txt",
        "grep -c --null foo a.txt b.txt",
        "grep -cl --null foo a.txt b.txt",
        "grep -lr --null x plain",
        "grep -cr --null x plain",
    ];

    let mut failures = 0;
    for row in rows {
        match check_row(row, &ws, &home) {
            Ok(()) => println!("PASS: {row}"),
            Err(msg) => {
                println!("FAIL: {row}: {msg}");
                failures += 1;
            }
        }
    }

    // Fail-closed rows: the parent must NOT rewrite these (the original
    // command — a shell syntax error / a real grep pipeline — runs instead).
    for (label, row) in [
        ("empty pipeline member", "grep -rn needle bigdir | | wc -l"),
        ("stdin-fed with file operands", "cat a.txt | grep foo b.txt"),
        ("stdin-fed -r without operands", "printf 'x' | grep -r foo"),
        ("stdin-fed with a '-' operand", "cat a.txt | grep foo -"),
    ] {
        match check_fallback(row, &ws, &home) {
            Ok(()) => println!("PASS (fallback): {row}"),
            Err(msg) => {
                println!("FAIL: {label}: {row}: {msg}");
                failures += 1;
            }
        }
    }

    // The exec-in-place fallback: a spec whose expected cwd does not match
    // the process cwd must exec the real grep and produce ITS output.
    match check_exec_fallback(&ws) {
        Ok(()) => println!("PASS: exec-in-place cwd fallback"),
        Err(msg) => {
            println!("FAIL: exec-in-place cwd fallback: {msg}");
            failures += 1;
        }
    }

    // SIGPIPE: a served grep piped to `head` must not hang (a Rust child
    // ignoring SIGPIPE would scan everything; the engine dies like grep).
    for (label, cmd) in [
        ("SIGPIPE with head", "grep -rn needle bigdir | head -1"),
        (
            "SIGPIPE through a preserved grep",
            "grep -rn needle bigdir | grep needle | head -1",
        ),
        // Producer-first chain: the engine is the MIDDLE member — cat feeds it
        // via stdin and must die on EPIPE when the engine dies under head.
        (
            "SIGPIPE chain with a real producer",
            "cat bigdir/big.txt | grep needle | head -1",
        ),
    ] {
        match check_sigpipe(cmd, &ws, &home) {
            Ok(()) => println!("PASS: {label}"),
            Err(msg) => {
                println!("FAIL: {label}: {msg}");
                failures += 1;
            }
        }
    }

    if failures == 0 {
        println!("grep-engine-e2e: all rows passed");
        0
    } else {
        println!("grep-engine-e2e: {failures} failures");
        1
    }
}

fn build_fixture(ws: &Path, home: &Path) {
    fs::write(ws.join("a.txt"), "foo\nbar\nfoo\nbaz\n").unwrap();
    fs::write(ws.join("b.txt"), "qux\nfoo\n").unwrap();
    fs::write(ws.join("c.txt"), "apple\n").unwrap();
    fs::write(ws.join("m.txt"), "a\nb\na\nb\na\n").unwrap();
    fs::write(ws.join("ctx.txt"), "a\nb\nc\nd\ne\nf\ng\nh\ni\n").unwrap();
    fs::write(ws.join("x1.txt"), "foo\n^foo\n").unwrap();
    fs::create_dir_all(ws.join("plain/d1")).unwrap();
    fs::write(ws.join("plain/a.txt"), "x1\n").unwrap();
    fs::write(ws.join("plain/d1/b.txt"), "x2\n").unwrap();
    fs::write(ws.join("plain/d1/c.txt"), "x3\n").unwrap();
    fs::write(ws.join("plain/e.txt"), "x5\n").unwrap();
    fs::create_dir_all(ws.join("bindir")).unwrap();
    fs::write(ws.join("bindir/bin1.dat"), b"hello\x00world\nneedle\n").unwrap();
    fs::write(ws.join("bindir/bin2.dat"), b"needle\x00world\n").unwrap();
    fs::write(ws.join("bindir/bin4.dat"), b"needle\nhello\x00world\n").unwrap();
    fs::write(ws.join("bindir/bin5.dat"), b"\xff\x00needle\n").unwrap();
    fs::create_dir_all(ws.join("sub")).unwrap();
    fs::write(ws.join("sub/s.txt"), "needle\n").unwrap();
    fs::write(ws.join("-x1"), "x\n").unwrap();
    fs::create_dir_all(home.join("htree")).unwrap();
    fs::write(home.join("htree/f.txt"), "needle\n").unwrap();
    // Pathological filename tokens: space, quotes, dollar, newline.
    fs::create_dir_all(ws.join("weird dir")).unwrap();
    fs::write(ws.join("weird dir/need'le.txt"), "needle\n").unwrap();
    fs::write(ws.join("weird dir/do\"ll$ar.txt"), "needle\n").unwrap();
    fs::write(ws.join("weird dir/line\nbreak.txt"), "needle\n").unwrap();
    // Literal filename operands: quoted/escaped glob and ~ are literal to BSD.
    fs::write(ws.join("~"), "needle\n").unwrap();
    fs::write(ws.join("*.txt"), "needle\n").unwrap();
    fs::write(ws.join("a\"b.txt"), "needle\n").unwrap();
    // Quoted `~` + unquoted glob: a literal `~` dir relative to cwd.
    fs::create_dir_all(ws.join("sub/~")).unwrap();
    fs::write(ws.join("sub/~/q.txt"), "needle\n").unwrap();
    // >256 KiB of matches (200k lines): exercises the piped-member cap lift
    // and the SIGPIPE path (>64 KiB pipe-buffer bound). mixed.txt adds
    // needle lines from a second path so grep-chain tails can filter on the
    // path prefix.
    fs::create_dir_all(ws.join("bigdir")).unwrap();
    fs::write(ws.join("bigdir/big.txt"), "needle\n".repeat(200_000)).unwrap();
    fs::write(
        ws.join("bigdir/mixed.txt"),
        "other\nneedle\n".repeat(50_000),
    )
    .unwrap();
    // A real git repo with pinned identity + dates: `git log` output is
    // deterministic across the two runs of every parity row.
    git_fixture(ws);
}

/// Initialize a repo in `ws` with one commit (pinned author/committer dates
/// make the log output deterministic).
fn git_fixture(ws: &Path) {
    let run = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(ws)
            .env("GIT_AUTHOR_DATE", "2026-01-02T03:04:05Z")
            .env("GIT_COMMITTER_DATE", "2026-01-02T03:04:05Z")
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run(&["init", "-q"]);
    run(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "Initial commit",
    ]);
}

fn check_row(row: &str, ws: &Path, home: &Path) -> Result<(), String> {
    let rewritten = mahbot::grep_engine_rewrite_for_test(row, ws, home)
        .ok_or_else(|| "not servable".to_string())?;
    let engine_out = run_sh(&rewritten, ws)?;
    // The real grep: parse the original row's command via sh -c (the authentic
    // execution path) — this also validates the rewrite against the original.
    let real_out = run_sh(row, ws)?;
    if engine_out == real_out {
        Ok(())
    } else {
        Err(format!(
            "engine {:?} != real {:?}",
            String::from_utf8_lossy(&engine_out.0),
            String::from_utf8_lossy(&real_out.0)
        ))
    }
}

fn run_sh(command: &str, cwd: &Path) -> Result<(Vec<u8>, i32), String> {
    let home = cwd.parent().expect("ws parent").join("home");
    let out = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .env("LC_ALL", "C.UTF-8")
        .env("HOME", &home)
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    Ok((out.stdout, out.status.code().unwrap_or(-1)))
}

fn check_exec_fallback(ws: &Path) -> Result<(), String> {
    // A spec whose expected cwd does not match the process cwd must exec the
    // real grep, which then runs in the ACTUAL cwd.
    check_exec_case(
        ws,
        r#"{"version":5,"verb":"grep","mode":"Basic","flags":{"n":true,"i":false,"v":false,"w":false,"x":false,"a":false,"h":false,"H":false,"s":false,"r":true,"o":false,"null":false,"c":false,"l":false,"m":null,"before":0,"after":0},"filters":[],"exclude_dir":[],"patterns":["x"],"operands":[{"display":".","resolved":"/nonexistent","trailing_slash":false}],"cwd":"/nonexistent","fallback":["grep","-rn","x","."],"piped":false,"stdin":false,"report_stream_bytes":false}"#,
        "grep -rn x .",
    )?;
    // A v4 spec (old parent binary after a self-update swap) into the v5
    // engine: the version mismatch must exec the real grep in place — the
    // fail-closed swap direction the rollout depends on.
    check_exec_case(
        ws,
        r#"{"version":4,"verb":"grep","mode":"Basic","flags":{"n":true,"i":false,"v":false,"w":false,"x":false,"a":false,"h":false,"H":false,"s":false,"r":true,"o":false,"null":false,"c":false,"l":false,"m":null,"before":0,"after":0},"filters":[],"exclude_dir":[],"patterns":["x"],"operands":[{"display":".","resolved":"/nonexistent","trailing_slash":false}],"cwd":"/nonexistent","fallback":["grep","-rn","x","."],"piped":false}"#,
        "grep -rn x .",
    )?;
    // A dash-prefixed operand must survive the exec'd grep via the `--`
    // separator the parent inserts into the fallback argv.
    let dash_abs = ws.join("-x1").to_string_lossy().into_owned();
    let json = format!(
        r#"{{"version":5,"verb":"grep","mode":"Basic","flags":{{"n":false,"i":false,"v":false,"w":false,"x":false,"a":false,"h":false,"H":false,"s":false,"r":false,"o":false,"null":false,"c":false,"l":false,"m":null,"before":0,"after":0}},"filters":[],"exclude_dir":[],"patterns":["x"],"operands":[{{"display":"-x1","resolved":"{dash_abs}","trailing_slash":false}}],"cwd":"/nonexistent","fallback":["grep","-e","x","--","-x1"],"piped":false,"stdin":false,"report_stream_bytes":false}}"#
    );
    check_exec_case(ws, &json, "grep -e x -- -x1")
}

fn check_exec_case(ws: &Path, json: &str, real_cmd: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let cmd = format!(
        "{} __grep-engine '{}'",
        exe.to_string_lossy(),
        json.replace('\'', "'\\''")
    );
    let (out, code) = run_sh(&cmd, ws)?;
    let real = run_sh(real_cmd, ws)?;
    let (rout, rcode) = &real;
    if &out == rout && code == *rcode {
        Ok(())
    } else {
        Err(format!(
            "fallback ({}, {code}) != real ({}, {rcode})",
            String::from_utf8_lossy(&out),
            String::from_utf8_lossy(rout)
        ))
    }
}

fn check_fallback(row: &str, ws: &Path, home: &Path) -> Result<(), String> {
    if mahbot::grep_engine_rewrite_for_test(row, ws, home).is_some() {
        return Err("expected fallback, got served".to_string());
    }
    Ok(())
}

fn check_sigpipe(command: &str, ws: &Path, home: &Path) -> Result<(), String> {
    // A huge-output grep piped to head must terminate promptly (SIGPIPE):
    // the >64 KiB fixture fills the pipe, so the engine must die on the
    // closed pipe instead of scanning everything.
    let rewritten = mahbot::grep_engine_rewrite_for_test(command, ws, home)
        .ok_or_else(|| "not servable".to_string())?;
    let start = std::time::Instant::now();
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(&rewritten)
        .current_dir(ws)
        .env("LC_ALL", "C.UTF-8")
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn: {e}"))?;
    // Poll for exit (wait_timeout is unstable); 10 s is the hang threshold.
    loop {
        if child
            .try_wait()
            .map_err(|e| format!("wait: {e}"))?
            .is_some()
        {
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            return Err("engine hung under head (SIGPIPE not effective)".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

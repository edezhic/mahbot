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
    match check_sigpipe(&ws, &home) {
        Ok(()) => println!("PASS: SIGPIPE with head"),
        Err(msg) => {
            println!("FAIL: SIGPIPE with head: {msg}");
            failures += 1;
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
        r#"{"version":3,"verb":"grep","mode":"Basic","flags":{"n":true,"i":false,"v":false,"w":false,"x":false,"a":false,"h":false,"H":false,"s":false,"r":true,"o":false,"null":false,"c":false,"l":false,"m":null,"before":0,"after":0},"filters":[],"exclude_dir":[],"patterns":["x"],"operands":[{"display":".","resolved":"/nonexistent","trailing_slash":false}],"cwd":"/nonexistent","fallback":["grep","-rn","x","."]}"#,
        "grep -rn x .",
    )?;
    // A dash-prefixed operand must survive the exec'd grep via the `--`
    // separator the parent inserts into the fallback argv.
    let dash_abs = ws.join("-x1").to_string_lossy().into_owned();
    let json = format!(
        r#"{{"version":3,"verb":"grep","mode":"Basic","flags":{{"n":false,"i":false,"v":false,"w":false,"x":false,"a":false,"h":false,"H":false,"s":false,"r":false,"o":false,"null":false,"c":false,"l":false,"m":null,"before":0,"after":0}},"filters":[],"exclude_dir":[],"patterns":["x"],"operands":[{{"display":"-x1","resolved":"{dash_abs}","trailing_slash":false}}],"cwd":"/nonexistent","fallback":["grep","-e","x","--","-x1"]}}"#
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

fn check_sigpipe(ws: &Path, home: &Path) -> Result<(), String> {
    // A huge-output grep piped to head must terminate promptly (SIGPIPE).
    let rewritten = mahbot::grep_engine_rewrite_for_test("grep -rn x plain | head -1", ws, home)
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

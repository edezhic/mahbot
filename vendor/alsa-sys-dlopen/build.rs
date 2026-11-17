//! Generates the ALSA bindings from the system header and wraps every function in a `dlopen`
//! lookup.
//!
//! The declarations come from `<alsa/asoundlib.h>` with the allowlist the published `alsa-sys`
//! uses, so the `alsa` crate compiles against exactly the surface it expects. Each `pub fn` is
//! replaced by a Rust function of the same name and signature that resolves the symbol on first
//! call; variadic declarations are dropped, because Rust cannot express a variadic wrapper and
//! `alsa` does not call them.
//!
//! The build needs the ALSA development headers (`libasound2-dev` on Debian/Ubuntu) and
//! libclang, exactly as the published crate's bindgen path does. It deliberately emits no
//! `rustc-link-lib=asound`: the point of the crate is that the file does not name the library.

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Include paths only. `cargo_metadata(false)` keeps pkg-config from emitting the
    // `-lasound` this crate exists to remove.
    let alsa = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("alsa")
        .expect(
            "the ALSA development headers are needed to build the audio backend \
             (libasound2-dev on Debian/Ubuntu)",
        );

    let mut builder = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <alsa/asoundlib.h>")
        .use_core()
        .size_t_is_usize(true)
        .allowlist_recursively(false)
        .prepend_enum_name(false)
        .layout_tests(false)
        .allowlist_function("snd_.*")
        .allowlist_type("_?snd_.*")
        .allowlist_type(".*va_list.*")
        .with_codegen_config(bindgen::CodegenConfig::FUNCTIONS | bindgen::CodegenConfig::TYPES);
    for include in &alsa.include_paths {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    let bindings = builder
        .generate()
        .expect("bindgen could not read <alsa/asoundlib.h>")
        .to_string();

    let (items, wrappers) = split(&bindings);
    let generated = format!("{MACHINERY}\n{items}\n{wrappers}");
    let out =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("generated.rs");
    std::fs::write(out, generated).expect("write the generated ALSA bindings");
}

/// The dlopen layer the wrappers call through, emitted ahead of the bindings.
const MACHINERY: &str = r#"
use std::sync::OnceLock;

struct __AlsaLibrary(*mut core::ffi::c_void);
// The handle lives for the process and is only read through `dlsym`.
unsafe impl Send for __AlsaLibrary {}
unsafe impl Sync for __AlsaLibrary {}

fn __alsa_library() -> Option<&'static __AlsaLibrary> {
    static LIBRARY: OnceLock<Option<__AlsaLibrary>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            // The soname every glibc distribution ships. `RTLD_LAZY` keeps one symbol this
            // build does not know from failing the whole load; `RTLD_LOCAL` keeps ALSA's
            // symbols out of every other library's namespace.
            let handle = unsafe {
                libc::dlopen(
                    c"libasound.so.2".as_ptr(),
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };
            (!handle.is_null()).then_some(__AlsaLibrary(handle))
        })
        .as_ref()
}

unsafe fn __alsa_symbol<F>(name: &[u8]) -> Option<F> {
    let library = __alsa_library()?;
    let symbol = unsafe { libc::dlsym(library.0, name.as_ptr().cast()) };
    if symbol.is_null() {
        None
    } else {
        Some(unsafe { core::mem::transmute_copy(&symbol) })
    }
}

/// What a function that cannot be resolved returns when it returns a string: a caller that
/// builds an error message from `snd_strerror` must get text, never a null pointer.
fn __alsa_unavailable() -> *const core::ffi::c_char {
    static TEXT: &[u8] = b"libasound is not available on this machine\0";
    TEXT.as_ptr().cast()
}
"#;

/// Split bindgen's output into the items to keep and the wrappers to emit in place of the
/// `extern "C"` declarations.
fn split(source: &str) -> (String, String) {
    let mut items = String::new();
    let mut wrappers = String::new();
    let mut lines = source.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        // A link attribute belongs to the extern block below it; the wrappers define those
        // symbols, so the attribute is dropped with the block.
        if trimmed.starts_with("#[link(") {
            continue;
        }
        if trimmed.starts_with("unsafe extern \"C\" {") || trimmed.starts_with("extern \"C\" {") {
            let mut block = String::from(line);
            block.push('\n');
            let mut depth = brace_delta(trimmed);
            while depth > 0 {
                let Some(next) = lines.next() else { break };
                block.push_str(next);
                block.push('\n');
                depth += brace_delta(next);
            }
            for declaration in declarations(&block) {
                if let Some(wrapper) = wrapper(&declaration) {
                    wrappers.push_str(&wrapper);
                }
            }
            continue;
        }
        items.push_str(line);
        items.push('\n');
    }
    (items, wrappers)
}

fn brace_delta(line: &str) -> i32 {
    line.chars().fold(0, |depth, ch| match ch {
        '{' => depth + 1,
        '}' => depth - 1,
        _ => depth,
    })
}

/// Every `pub fn …;` declaration in one extern block, without the surrounding block.
fn declarations(block: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = block;
    while let Some(start) = rest.find("pub fn ") {
        let declaration = &rest[start + "pub fn ".len()..];
        let Some(end) = declaration_end(declaration) else {
            break;
        };
        found.push(format!("pub fn {}", &declaration[..end]));
        rest = &declaration[end..];
    }
    found
}

/// The end of one declaration: its top-level `;`. Parentheses, brackets and generics nest, and a
/// `->` at depth zero only lowers the already-zero depth.
fn declaration_end(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            ';' if depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

/// One function's wrapper, or `None` for a variadic declaration (which Rust cannot express and
/// `alsa` does not call).
fn wrapper(declaration: &str) -> Option<String> {
    let rest = declaration.strip_prefix("pub fn ")?;
    let open = rest.find('(')?;
    let name = rest[..open].trim();
    let after_open = &rest[open + 1..];
    let close = matching_paren(after_open)?;
    let args = &after_open[..close];
    let tail = after_open[close + 1..].trim();
    let returns = tail.strip_prefix("->").map_or("()", str::trim);
    if args.contains("...") {
        return None;
    }
    let names = arg_names(args)?;
    let call = names.join(", ");

    let mut out = String::new();
    writeln!(out, "pub unsafe fn {name}({args}) -> {returns} {{").ok()?;
    writeln!(
        out,
        "    type F = unsafe extern \"C\" fn({args}) -> {returns};"
    )
    .ok()?;
    // The lookup is cached per function: the audio path calls `snd_pcm_writei` once per period,
    // and a `dlsym` on every one of those is pure overhead after the first.
    writeln!(
        out,
        "    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();"
    )
    .ok()?;
    writeln!(
        out,
        "    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b\"{name}\\0\")) {{"
    )
    .ok()?;
    writeln!(out, "        Some(function) => function({call}),").ok()?;
    if matches!(name, "snd_strerror" | "snd_asoundlib_version") {
        writeln!(out, "        None => __alsa_unavailable(),").ok()?;
    } else {
        writeln!(out, "        None => {},", fallback(returns)).ok()?;
    }
    writeln!(out, "    }}").ok()?;
    writeln!(out, "}}").ok()?;
    Some(out)
}

/// The closing `)` of the argument list, counting nested parentheses.
fn matching_paren(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' if depth == 0 => return Some(index),
            ')' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Each argument's name, in order — what the wrapper forwards to the real function.
fn arg_names(args: &str) -> Option<Vec<String>> {
    let mut names = Vec::new();
    for part in split_top_level(args) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, _) = part.split_once(':')?;
        names.push(name.trim().to_string());
    }
    Some(names)
}

/// Split on commas that are not inside parentheses, brackets or generics.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// What a wrapper returns when the symbol is not there. ALSA reports failure as a negative
/// integer, so the C integer types get `-1` rather than the zero that would read as success;
/// pointers get null and everything else a zeroed value (all ALSA returns are primitives).
fn fallback(returns: &str) -> &'static str {
    let returns = returns.trim();
    if returns == "()" || returns.is_empty() {
        "()"
    } else if returns.contains("*mut") {
        "core::ptr::null_mut()"
    } else if returns.contains("*const") {
        "core::ptr::null()"
    } else if returns.contains("c_int") || returns.contains("i32") {
        "-1"
    } else if returns.contains("c_uint")
        || returns.contains("u32")
        || returns.contains("usize")
        || returns.contains("u64")
        || returns.contains("i64")
        || returns.contains("c_ulong")
        || returns.contains("c_long")
        || returns.contains("u8")
        || returns.contains("i8")
    {
        "0"
    } else {
        "core::mem::zeroed()"
    }
}

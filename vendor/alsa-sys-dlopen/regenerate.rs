//! The transform behind `regenerate.sh`: it turns the published `alsa-sys` bindings into this
//! crate's `src/generated.rs`, replacing every `extern "C"` declaration with a `dlopen` wrapper of
//! the same name and signature.
//!
//! Not part of the build. `regenerate.sh` compiles and runs it by hand when the audio crate moves;
//! a normal build compiles the committed file as it stands, so no build reads a C header or runs a
//! binding generator.

use std::fmt::Write as _;

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("usage: transform <path to alsa-sys src/generated.rs>");
    let source = std::fs::read_to_string(&input).expect("read the published bindings");
    let (items, wrappers) = split(&source);
    print!("{MACHINERY}\n{items}\n{wrappers}");
}

/// The dlopen layer the wrappers call through, emitted ahead of the bindings.
const MACHINERY: &str = r#"// The bindings `alsa` 0.11 compiles against, wrapped so every call is resolved through `dlopen`.
//
// Generated — do not edit by hand: `regenerate.sh` beside this crate is the recipe. The items below
// are the published `alsa-sys` 0.4.0 `src/generated.rs` (MIT, see `LICENSE-ALSA-SYS`), byte for byte
// except that every `extern "C"` declaration is replaced by a `dlopen` wrapper of the same name and
// signature, and the variadic ones — which Rust cannot express and `alsa` does not call — are
// dropped. One surface serves both Linux targets: it was generated on x86_64, so `__va_list_tag`
// carries that layout on either architecture, which is safe because `alsa` only forwards one as an
// opaque pointer.
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
///
/// A block ends at the first line that is exactly `}`: bindgen closes it at column zero. Counting
/// braces instead would mis-read any line carrying an unbalanced brace — a doc comment bindgen
/// retains from a non-system header is enough — and then run on to the end of the file, which is how
/// a build-time copy of this transform truncated its own output.
fn split(source: &str) -> (String, String) {
    let mut items = String::new();
    let mut wrappers = String::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("unsafe extern \"C\" {") || trimmed.starts_with("extern \"C\" {") {
            let mut block = String::from(line);
            block.push('\n');
            for next in lines.by_ref() {
                block.push_str(next);
                block.push('\n');
                if next == "}" {
                    break;
                }
            }
            // Nothing in the block may drop out with nothing said: an item of another shape, an
            // attributed one, or a declaration `declaration_end` gave up on would leave the crate
            // that needs it failing to compile with no clue why. What is parsed is cut out of the
            // block, so what remains must be the block's own frame and nothing else. The digest pin
            // fixes today's input, so this fires only when the recipe is pointed at another
            // `alsa-sys` — which is exactly when it matters.
            let parsed = declarations(&block);
            let mut rest = block.clone();
            for declaration in &parsed {
                rest = rest.replacen(&format!("{declaration};"), "", 1);
            }
            let leftover: Vec<&str> = rest
                .lines()
                .skip(1)
                .map(str::trim)
                .filter(|line| !line.is_empty() && *line != "}")
                .collect();
            assert!(
                leftover.is_empty(),
                "an extern item this transform does not wrap: {leftover:?}"
            );
            // Every declaration is wrapped, or — the variadic ones alone — dropped on purpose; one
            // this transform cannot read panics inside `wrapper`.
            for declaration in parsed {
                match wrapper(&declaration) {
                    Some(text) => wrappers.push_str(&text),
                    None if declaration.contains("...") => {}
                    None => panic!("a declaration this transform could not wrap: {declaration}"),
                }
            }
            continue;
        }
        items.push_str(line);
        items.push('\n');
    }
    (items, wrappers)
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

/// A declaration this transform cannot read: said with the declaration, because an item dropped in
/// silence is exactly what the frozen surface must never do.
fn unparsable(declaration: &str) -> ! {
    panic!("a declaration this transform cannot read: {declaration}")
}

/// One function's wrapper, or `None` for a variadic declaration: Rust cannot express one and `alsa`
/// does not call it, so that is the transform's only deliberate drop. A declaration it cannot read
/// is a panic, not a drop — a symbol lost in silence is the defect this crate exists to prevent.
fn wrapper(declaration: &str) -> Option<String> {
    let rest = declaration
        .strip_prefix("pub fn ")
        .unwrap_or_else(|| unparsable(declaration));
    let open = rest.find('(').unwrap_or_else(|| unparsable(declaration));
    let name = rest[..open].trim();
    let after_open = &rest[open + 1..];
    let close = matching_paren(after_open).unwrap_or_else(|| unparsable(declaration));
    let args = &after_open[..close];
    let tail = after_open[close + 1..].trim();
    let returns = tail.strip_prefix("->").map_or("()", str::trim);
    if args.contains("...") {
        return None;
    }
    let names = arg_names(args).unwrap_or_else(|| unparsable(declaration));
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

/// What a wrapper returns when the symbol is not there: `-1` for `c_int` (and a spelled `i32`), `0`
/// for the other integer types the surface names by spelling, null for a pointer, and a zeroed value
/// for a typedef'd type such as `snd_pcm_sframes_t` — read as written, not resolved. Only a host with
/// the library absent ever sees one of these, and the product asks the library first
/// (`mahbot::audio::library_available`) and opens no device when it is missing, so no stream exists
/// to read a wrapper's value from.
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

//! The transform behind `regenerate.sh`: it narrows the published `alsa-sys` bindings to the
//! surface `alsa` compiles against, then replaces every `extern "C"` declaration with a `dlopen`
//! wrapper of the same name and signature.
//!
//! Not part of the build. `regenerate.sh` compiles and runs it by hand when the audio crate moves;
//! a normal build compiles the committed file as it stands, so no build reads a C header or runs a
//! binding generator.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

const USAGE: &str = "usage: transform <published alsa-sys generated.rs> <alsa crate src directory>";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let (Some(bindings), Some(layer)) = (arguments.next(), arguments.next()) else {
        panic!("{USAGE}");
    };
    let published = std::fs::read_to_string(&bindings).expect("read the published bindings");
    let surface = Surface::parse(&published);
    print!(
        "{MACHINERY}\n{}",
        surface.render(&surface.reachable(&referenced(&layer)))
    );
}

/// The dlopen layer the wrappers call through, emitted ahead of the bindings.
const MACHINERY: &str = r#"// The bindings `alsa` 0.11 compiles against, wrapped so every call is resolved through `dlopen`.
//
// Generated — do not edit by hand: `regenerate.sh` beside this crate is the recipe. The items below
// are the part of the published `alsa-sys` 0.4.0 `src/generated.rs` (MIT, see `LICENSE-ALSA-SYS`)
// that `alsa` 0.11 reaches — the names its own sources spell, and whatever the signatures of those
// need — with every `extern "C"` declaration replaced by a `dlopen` wrapper of the same name and
// signature. The rest of the published file binds libasound beyond any caller of the crate and is
// not carried. The variadic declarations, which Rust cannot express and `alsa` does not call, are
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
"#;

/// The published bindings, item by item: what each one provides, and what it names.
struct Surface {
    units: Vec<Unit>,
    /// Every unit that provides one name, by that name — the declaration itself, the alias, and any
    /// `impl` block written for a type.
    providers: BTreeMap<String, Vec<usize>>,
}

/// One top-level item of the published bindings.
struct Unit {
    kind: Kind,
    /// The surface name this item provides: what it declares, or the type an `impl` block is written
    /// for. `None` for an item that provides nothing at all — an `impl` for a type the surface does
    /// not declare.
    name: Option<String>,
    text: String,
}

/// How one item is emitted.
enum Kind {
    /// A type, constant, re-export or `impl` block: emitted as it stands.
    Item,
    /// An `extern "C"` declaration: emitted as a `dlopen` wrapper.
    Declaration,
    /// A variadic declaration. Rust cannot express one and `alsa` does not call one, so it is never
    /// emitted — and reaching one is a panic, because a symbol lost in silence is the defect this
    /// crate exists to prevent.
    Variadic,
}

impl Surface {
    fn parse(source: &str) -> Self {
        let mut units = Vec::new();
        for text in top_level(source) {
            // Bindgen may put attributes in front of a block; everything after them is its frame.
            let block: Vec<&str> = text
                .lines()
                .skip_while(|line| line.starts_with("#["))
                .collect();
            if block.first().is_some_and(|frame| {
                frame.starts_with("unsafe extern \"C\" {") || frame.starts_with("extern \"C\" {")
            }) {
                for declaration in declarations(&block.join("\n")) {
                    let kind = if declaration.contains("...") {
                        Kind::Variadic
                    } else {
                        Kind::Declaration
                    };
                    units.push(Unit {
                        kind,
                        name: provided(&declaration),
                        text: declaration,
                    });
                }
                continue;
            }
            units.push(Unit {
                kind: Kind::Item,
                name: provided(&text),
                text,
            });
        }
        // An `impl` provides nothing of its own: it belongs to the type it is written for, and is
        // kept with it.
        let names: BTreeSet<String> = units.iter().filter_map(|unit| unit.name.clone()).collect();
        for unit in &mut units {
            if unit.name.is_none() {
                unit.name = impl_target(&unit.text, &names);
            }
        }
        // Nothing may drop out unannounced: an item of a shape this transform cannot name — an enum,
        // a `static mut`, an `impl` for a type the surface does not declare — would vanish from the
        // surface at the next bump with no word. Same rule as `declarations`.
        assert!(
            units.iter().all(|unit| unit.name.is_some()),
            "an item this transform cannot name: {:?}",
            units
                .iter()
                .find(|unit| unit.name.is_none())
                .map(|unit| unit.text.as_str())
        );
        let mut providers: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, unit) in units.iter().enumerate() {
            if let Some(name) = &unit.name {
                providers.entry(name.clone()).or_default().push(index);
            }
        }
        Self { units, providers }
    }

    /// The names `alsa` can reach: every name its sources spell, and whatever the signature of a
    /// reachable item needs. A function is reachable only by being called; a type or a constant only
    /// by appearing in the signature of something already reachable.
    fn reachable(&self, roots: &BTreeSet<String>) -> BTreeSet<String> {
        let mut keep = BTreeSet::new();
        let mut queue: Vec<String> = self
            .providers
            .keys()
            .filter(|name| roots.contains(*name))
            .cloned()
            .collect();
        while let Some(name) = queue.pop() {
            if !keep.insert(name.clone()) {
                continue;
            }
            for index in self.providers.get(&name).into_iter().flatten() {
                for reference in identifiers(&self.units[*index].text) {
                    if self.providers.contains_key(&reference) && !keep.contains(&reference) {
                        queue.push(reference);
                    }
                }
            }
        }
        keep
    }

    /// The surface as this crate commits it: the reachable items in the published order, then the
    /// wrappers in theirs — the layout the frozen file has always had.
    fn render(&self, keep: &BTreeSet<String>) -> String {
        let mut items = String::new();
        let mut wrappers = String::new();
        for unit in &self.units {
            let Some(name) = &unit.name else {
                continue;
            };
            if !keep.contains(name) {
                continue;
            }
            match unit.kind {
                Kind::Item => {
                    items.push_str(&unit.text);
                    items.push('\n');
                }
                Kind::Declaration => wrappers.push_str(&wrapper(&unit.text)),
                Kind::Variadic => panic!("a variadic declaration `alsa` reaches: {}", unit.text),
            }
        }
        format!("{items}\n{wrappers}")
    }
}

/// The published bindings as top-level items, verbatim.
///
/// Bindgen writes each item's keyword first and indents every line that continues it, so a line at
/// column zero opens the next item — except the item's own closing frame and the `where`/`{` of a
/// multi-line `impl` header, which `starts_item` deliberately does not name, and the attribute lines
/// that belong to the item after them. Counting brackets instead would mis-read any line carrying an
/// unbalanced one — a doc comment bindgen retains is enough — and then swallow the items after it.
fn top_level(source: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in source.lines() {
        if current.is_empty() {
            // Bindgen's banner and the blank lines between items carry no item: the file's own
            // header is the provenance now.
            let trimmed = line.trim();
            if trimmed.is_empty() || (trimmed.starts_with("/*") && trimmed.ends_with("*/")) {
                continue;
            }
        } else if starts_item(line) && !current.iter().all(|kept| kept.starts_with("#[")) {
            items.push(current.join("\n"));
            current.clear();
        }
        current.push(line);
    }
    if !current.is_empty() {
        items.push(current.join("\n"));
    }
    items
}

/// Whether one line at column zero opens a top-level item.
fn starts_item(line: &str) -> bool {
    !line.starts_with([' ', '\t'])
        && ["pub ", "#[", "impl", "unsafe extern ", "extern \"C\" "]
            .iter()
            .any(|keyword| line.starts_with(keyword))
}

/// The name one item declares: bindgen writes the item's attributes, then its keyword, then the name
/// — and a `use` renames on `as`.
fn provided(text: &str) -> Option<String> {
    let declaration = text.lines().find(|line| !line.starts_with("#["))?;
    for keyword in [
        "pub struct ",
        "pub union ",
        "pub type ",
        "pub const ",
        "pub static ",
        "pub fn ",
    ] {
        if let Some(rest) = declaration.strip_prefix(keyword) {
            return identifiers(rest).first().cloned();
        }
    }
    let (_, alias) = declaration
        .strip_prefix("pub use self::")?
        .split_once(" as ")?;
    identifiers(alias).first().cloned()
}

/// The surface type an `impl` block is written for: the first name in its header that the surface
/// provides. A trait's name is not one of them, so `impl<T> Debug for X<T>` resolves to `X`.
fn impl_target(text: &str, names: &BTreeSet<String>) -> Option<String> {
    let header = &text[..text.find('{')?];
    identifiers(header)
        .into_iter()
        .find(|name| names.contains(name))
}

/// Every identifier in one piece of text, in order.
fn identifiers(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            names.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        names.push(current);
    }
    names
}

/// Every name the ALSA layer's own sources spell. The names in its comments and string literals are
/// prose, not references — the crate labels its own errors `from_const("snd_hctl_elem_id_get_name",
/// …)` beside the call it makes — so both are dropped before the scan. Only the files directly under
/// `src`: `src/direct` holds the crate's own copy of the ioctl structures, whose names are its own.
fn referenced(directory: &str) -> BTreeSet<String> {
    let mut sources: Vec<std::path::PathBuf> = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read the alsa crate's src ({directory}): {error}"))
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no Rust sources under {directory}");
    let mut names = BTreeSet::new();
    for path in sources {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        names.extend(identifiers(&without_comments(&source)));
    }
    names
}

/// One source file with its comments and string literals removed: `//` to the end of the line,
/// `/* */`, and `"…"` whole.
///
/// This reads the shapes `alsa`'s sources are written in, not Rust: no raw string with a quote inside
/// it, no char literal holding one, no nested block comment. A source that had one would be mis-read
/// silently, which is why the assumption is written here beside the code that makes it.
///
/// `src/audio/mod.rs` carries the same scan for its in-tree checks; the two cannot share code, so a
/// change here is a change there.
fn without_comments(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut code = String::with_capacity(source.len());
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '/' if chars.get(index + 1) == Some(&'/') => {
                while index < chars.len() && chars[index] != '\n' {
                    index += 1;
                }
            }
            '/' if chars.get(index + 1) == Some(&'*') => {
                index += 2;
                while index < chars.len()
                    && !(chars[index] == '*' && chars.get(index + 1) == Some(&'/'))
                {
                    index += 1;
                }
                index += 2;
            }
            '"' => {
                index += 1;
                while index < chars.len() {
                    if chars[index] == '\\' {
                        index += 2;
                        continue;
                    }
                    let done = chars[index] == '"';
                    index += 1;
                    if done {
                        break;
                    }
                }
            }
            ch => {
                code.push(ch);
                index += 1;
            }
        }
    }
    code
}

/// Every `pub fn …;` declaration in one extern block, without the surrounding block.
///
/// Nothing in the block may drop out with nothing said: an item of another shape, an attributed one,
/// or a declaration `declaration_end` gave up on would leave the crate that needs it failing to
/// compile with no clue why. What is parsed is cut out of the block, so what remains must be the
/// block's own frame and nothing else. The digest pin fixes today's input, so this fires only when
/// the recipe is pointed at another `alsa-sys` — which is exactly when it matters.
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
    let mut leftover = block.to_owned();
    for declaration in &found {
        leftover = leftover.replacen(&format!("{declaration};"), "", 1);
    }
    let leftover: Vec<&str> = leftover
        .lines()
        .skip(1)
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != "}")
        .collect();
    assert!(
        leftover.is_empty(),
        "an extern item this transform does not wrap: {leftover:?}"
    );
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

/// `writeln!` into the `String` a wrapper is built in, one line at a time: writing to a `String`
/// cannot fail, so the eight lines below carry no error path.
macro_rules! emit {
    ($out:expr, $($line:tt)*) => {
        writeln!($out, $($line)*).expect("writing to a String cannot fail")
    };
}

/// One function's wrapper. A declaration it cannot read is a panic, not a drop — a symbol lost in
/// silence is the defect this crate exists to prevent. Variadic declarations never reach here: the
/// parse step gives them their own kind.
fn wrapper(declaration: &str) -> String {
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
    let names = arg_names(args).unwrap_or_else(|| unparsable(declaration));
    let call = names.join(", ");

    let mut out = String::new();
    emit!(out, "pub unsafe fn {name}({args}) -> {returns} {{");
    emit!(
        out,
        "    type F = unsafe extern \"C\" fn({args}) -> {returns};"
    );
    // The lookup is cached per function: the audio path calls `snd_pcm_writei` once per period,
    // and a `dlsym` on every one of those is pure overhead after the first.
    emit!(
        out,
        "    static SYMBOL: std::sync::OnceLock<Option<F>> = std::sync::OnceLock::new();"
    );
    emit!(
        out,
        "    match *SYMBOL.get_or_init(|| __alsa_symbol::<F>(b\"{name}\\0\")) {{"
    );
    emit!(out, "        Some(function) => function({call}),");
    emit!(out, "        None => {},", fallback(returns));
    emit!(out, "    }}");
    emit!(out, "}}");
    out
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

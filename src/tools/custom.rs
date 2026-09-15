//! `custom` — one native tool forwarding calls to admin-authored scripts in the
//! `shared/` folder of the admin's personal workspace.
//!
//! A tool is a file in that folder: one file per tool, flat (subfolders are
//! never walked), the file's name without its extension being the tool's name.
//! The file's leading comment block (`//` lines or a block comment) is the only
//! place a tool is defined — `@description …` plus one
//! `@param <name> <type> <required|optional> …` per argument. There is no
//! registry and no publishing step, and a malformed or missing header is never
//! repaired or guessed: the file is simply not described and any call to it
//! fails.
//!
//! The scripts are the product's single-file `bun` scripts, run like the
//! Assistant's own — never through a shell, so nothing in the call can be
//! reinterpreted as shell syntax. The Assistant reaches its own scripts through
//! the shell's `PATH` (where a user-installed bun can outrank the managed one);
//! a custom tool always runs the managed binary. A script receives the caller's
//! arguments as one argv entry: a JSON object of the declared parameters the
//! caller supplied, under the names it used. The payload is nested so the
//! product's own argument normalization can never rewrite it. It runs in place,
//! from the calling session's workspace root — and the Assistant is pinned to
//! its own personal workspace, so a custom call always runs in the caller's
//! `userspaces/<user>` directory.
//!
//! Availability is per user: the admin may call anything, a guest only
//! what is granted to them (`users.granted_tools`). The `<custom-tools>`
//! context block is the model-facing catalogue and follows the same split.

use crate::Workspace;
use crate::prompt::{load_prompt, substitute};
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Folder holding the catalogue, inside the admin's personal workspace.
const SHARED_DIR: &str = "shared";

/// Extensions the managed `bun` runtime executes as a single-file script. Any
/// other extension is not a tool — not described and not callable — so a stray
/// file (a note, a module a script imports) never becomes one.
const RUNNABLE_EXTENSIONS: &[&str] = &["ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"];

/// How much of a tool file is read to parse its header. The header is the
/// file's leading comment block, so this is orders of magnitude more than any
/// real one, and it bounds the read of an arbitrarily large script.
const MAX_HEADER_BYTES: u64 = 16 * 1024;

// ── Header grammar ───────────────────────────────────────────────────────

/// The four argument types the shallow validation knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamType {
    Str,
    Integer,
    Boolean,
    List,
}

impl ParamType {
    fn parse(token: &str) -> Option<Self> {
        match token {
            "string" => Some(Self::Str),
            "integer" => Some(Self::Integer),
            "boolean" => Some(Self::Boolean),
            "list" => Some(Self::List),
            _ => None,
        }
    }

    /// Name used in the catalogue block.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Str => "string",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::List => "list",
        }
    }

    /// Wording for a type mismatch, for the `usage:` error.
    const fn expected(self) -> &'static str {
        match self {
            Self::Str => "a string",
            Self::Integer => "an integer",
            Self::Boolean => "a boolean",
            Self::List => "an array of strings",
        }
    }
}

/// One declared parameter.
struct Param {
    name: String,
    ty: ParamType,
    required: bool,
    /// Author-supplied prose, rendered in the catalogue block.
    description: String,
}

/// A usable custom tool: its parsed header plus the file that defines it.
struct CustomToolEntry {
    name: String,
    description: String,
    params: Vec<Param>,
    path: PathBuf,
}

/// Split `line` into its first `n` whitespace-separated words plus the rest of
/// the line (`""` when nothing follows them).
fn split_words(line: &str, n: usize) -> (Vec<&str>, &str) {
    let mut words = Vec::with_capacity(n);
    let mut rest = line;
    for _ in 0..n {
        let trimmed = rest.trim_start();
        if trimmed.is_empty() {
            rest = "";
            break;
        }
        let end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        words.push(&trimmed[..end]);
        rest = &trimmed[end..];
    }
    (words, rest.trim())
}

/// The comment openers a runnable single-file script really uses: `//`, `/*`,
/// a block-comment `*` (which also covers `*/` and `**`), and a `#!` shebang.
/// Nothing else starts a comment, so any other line ends the header block
/// instead of being folded into it.
const COMMENT_OPENERS: &[&str] = &["//", "/*", "*", "#!"];

/// The text of a comment-only line with its comment decoration stripped, or
/// `None` for a line that carries code (which ends the header block). A block
/// comment decorates both ends (`/** @description … */`), so its trailing `*/`
/// goes too — otherwise it would land in the catalogue prose. A `//` line has
/// no closer, so everything after the marker is text.
fn comment_body(line: &str) -> Option<&str> {
    let (opener, rest) = COMMENT_OPENERS
        .iter()
        .find_map(|opener| line.strip_prefix(opener).map(|rest| (*opener, rest)))?;
    let rest = if opener == "//" {
        rest
    } else {
        rest.strip_suffix("*/").unwrap_or(rest)
    };
    Some(rest.trim_start_matches('*').trim())
}

/// Parse a script's leading comment block into `(description, params)`.
/// `None` when the header is missing or malformed.
///
/// An unknown `@`-directive is malformed rather than ignored: the header *is*
/// the tool's interface, so a misspelled `@param` (`@parm`) that was silently
/// skipped would drop a declared argument, and the caller's value for it would
/// come back as an ignored argument — a quietly wrong tool. The author instead
/// sees the call fail loudly and fixes the header.
fn parse_header(source: &str) -> Option<(String, Vec<Param>)> {
    let mut description: Option<String> = None;
    let mut params: Vec<Param> = Vec::new();

    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(body) = comment_body(line) else {
            // The first line carrying code ends the header block.
            break;
        };
        let Some(directive) = body.strip_prefix('@') else {
            continue;
        };
        let (words, text) = split_words(directive, 1);
        let [keyword] = words[..] else {
            return None;
        };
        match keyword {
            "description" => {
                // A header declares the tool's description once, like each
                // parameter of it.
                if text.is_empty() || description.is_some() {
                    return None;
                }
                description = Some(text.to_string());
            }
            "param" => {
                let (words, text) = split_words(text, 3);
                let [name, ty, required] = words[..] else {
                    return None;
                };
                let ty = ParamType::parse(ty)?;
                let required = match required {
                    "required" => true,
                    "optional" => false,
                    _ => return None,
                };
                if params.iter().any(|p| p.name == name) {
                    return None;
                }
                params.push(Param {
                    name: name.to_string(),
                    ty,
                    required,
                    description: text.to_string(),
                });
            }
            _ => return None,
        }
    }

    Some((description?, params))
}

/// Whether `name` is a usable tool name: a plain file name, so no call can name
/// a path, traverse out of the folder, or address a dot-file. Discovery, the
/// call path and the grant action share this one predicate, so no surface can
/// hold or advertise a name another would refuse.
pub(crate) fn is_tool_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && !name.contains(['/', '\\'])
}

// ── Discovery ────────────────────────────────────────────────────────────

/// The catalogue folder: `<admin's personal workspace>/shared`.
fn shared_dir() -> PathBuf {
    crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME).join(SHARED_DIR)
}

/// Whether `bun` can execute `path` as a single-file script.
fn is_runnable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            RUNNABLE_EXTENSIONS
                .iter()
                .any(|k| ext.eq_ignore_ascii_case(k))
        })
}

/// Read just enough of a tool file to parse its header.
fn read_header_source(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;
    let mut buf = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_HEADER_BYTES)
        .read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Load the catalogue: every usable tool in the folder, ordered by name
/// (byte-wise — the same deterministic ordering the product's other
/// file-defined descriptions use). Files with a non-runnable extension, a
/// malformed header, or a name another file already took are skipped.
fn load_catalogue(dir: &Path) -> Vec<CustomToolEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        // `DirEntry::file_type` does not follow symlinks: only regular files in
        // the folder itself are candidates.
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.path())
        .filter(|p| is_runnable(p))
        .collect();
    // Sort by path so a same-stem collision (`weather.ts` vs `weather.js`)
    // resolves the same way on every read: the first *usable* file wins, so a
    // malformed `.js` leaves the tool to a later `.ts` rather than erasing it.
    files.sort();

    let mut tools: Vec<CustomToolEntry> = Vec::new();
    for path in files {
        let Some(name) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if !is_tool_name(&name) || tools.iter().any(|t| t.name == name) {
            continue;
        }
        let Ok(source) = read_header_source(&path) else {
            continue;
        };
        let Some((description, params)) = parse_header(&source) else {
            continue;
        };
        tools.push(CustomToolEntry {
            name,
            description,
            params,
            path,
        });
    }
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    tools
}

/// Load the catalogue off the async runtime (it reads the folder).
async fn catalogue() -> Vec<CustomToolEntry> {
    tokio::task::spawn_blocking(|| load_catalogue(&shared_dir()))
        .await
        .unwrap_or_default()
}

// ── Catalogue block ──────────────────────────────────────────────────────

/// Render one line per tool: its description, then its parameters narrated in
/// prose (name, basic type, required-ness). The admin-authored text is
/// credential-scrubbed on the way into the prompt, like the product's other
/// user-provided text that enters one.
fn render_lines(tools: &[&CustomToolEntry]) -> String {
    let mut out = String::new();
    for tool in tools {
        let _ = write!(
            out,
            "- {}: {}",
            tool.name,
            crate::util::scrub_credentials(&tool.description)
        );
        if !tool.params.is_empty() {
            out.push_str(" Parameters: ");
            for (i, param) in tool.params.iter().enumerate() {
                if i > 0 {
                    out.push_str("; ");
                }
                let _ = write!(
                    out,
                    "{} ({}, {})",
                    param.name,
                    param.ty.as_str(),
                    if param.required {
                        "required"
                    } else {
                        "optional"
                    }
                );
                if !param.description.is_empty() {
                    let _ = write!(
                        out,
                        " — {}",
                        crate::util::scrub_credentials(&param.description)
                    );
                }
            }
            out.push('.');
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// The `<custom-tools>` context block for an Assistant session.
///
/// Unlike the other assistant blocks this one is emitted even when there is
/// nothing to list — a guest with no grants (and the admin's Assistant with no
/// tools yet) still learns the feature exists.
pub(crate) async fn context_block(user_name: &str, is_admin: bool) -> String {
    let granted = if is_admin {
        Vec::new()
    } else {
        crate::users::granted_tools(user_name).await
    };
    // A caller with no grants can only receive the no-grants form, so the
    // folder is not read for them at all.
    let catalogue = if is_admin || !granted.is_empty() {
        catalogue().await
    } else {
        Vec::new()
    };
    block_for(&catalogue, &granted, is_admin)
}

/// Render the block for a loaded catalogue: every tool for the admin, only the
/// granted ones for a guest — the same split the call itself enforces. A grant
/// that matches no usable file contributes nothing, and an empty listing falls
/// back to the matching brief form.
fn block_for(catalogue: &[CustomToolEntry], granted: &[String], is_admin: bool) -> String {
    let tools: Vec<&CustomToolEntry> = catalogue
        .iter()
        .filter(|t| is_admin || granted.iter().any(|g| g == &t.name))
        .collect();
    if tools.is_empty() {
        return load_prompt(if is_admin {
            "context/custom_tools_none.md"
        } else {
            "context/custom_tools_no_grants.md"
        });
    }
    substitute(
        &load_prompt("context/custom_tools.md"),
        &[("{{tools}}", &render_lines(&tools))],
    )
}

// ── Call path ────────────────────────────────────────────────────────────

/// A resolved custom-tool call: the script to run and the arguments it receives.
pub(crate) struct ResolvedCall {
    /// The tool's file.
    pub path: PathBuf,
    /// The declared arguments the caller supplied, coerced to their types.
    pub args: Map<String, Value>,
    /// The supplied argument names the tool does not declare. The call path
    /// reports them back with the run's output; a strict caller has none,
    /// because it refuses them instead.
    pub ignored: Vec<String>,
}

impl ResolvedCall {
    /// The JSON object the script receives as its single argv entry.
    pub(crate) fn payload(&self) -> String {
        serde_json::to_string(&self.args).expect("a custom tool's arguments are serializable")
    }
}

/// Why a custom-tool call could not be resolved.
///
/// The call path renders these as its `forbidden:` / `not-found:` / `usage:`
/// refusals; an alarm's arming and firing paths turn them into their own
/// wording, which is why the reason is a value rather than a formatted error.
pub(crate) enum CallRefusal {
    /// The name is not a callable tool name at all (a path, a dot-file, empty).
    Name { name: String },
    /// The caller may not call this name: not the admin and not granted it.
    Unavailable { name: String },
    /// No usable tool of that name: no such file, or a header the catalogue
    /// refuses.
    NotUsable { name: String },
    /// The arguments do not fit the tool's declared interface.
    Arguments(anyhow::Error),
}

impl CallRefusal {
    /// The refusal as a normal call reports it to the model.
    pub(crate) fn into_error(self) -> anyhow::Error {
        match self {
            Self::Name { name } => anyhow::anyhow!(
                "forbidden: \"{name}\" is not a tool name — hint: a tool's name is the file \
                 name without its extension"
            ),
            Self::Unavailable { name } => anyhow::anyhow!(
                "forbidden: custom tool \"{name}\" is not granted to you — hint: only tools \
                 granted to your account can be called"
            ),
            Self::NotUsable { name } => anyhow::anyhow!(
                "not-found: custom tool \"{name}\" is not usable — hint: a tool is a \
                 `{name}.ts` (or .js/.tsx/…) script in the admin's `shared` folder whose \
                 leading comment block declares a `@description` and one `@param` per \
                 argument"
            ),
            Self::Arguments(e) => e,
        }
    }
}

/// Refuse a name that is not a callable tool name at all (see [`is_tool_name`]).
///
/// A caller that must settle the name before anything else about the call — the
/// normal call path, which reads the argument object next — calls this itself,
/// as [`resolve_tool_call`] does first for every path.
fn check_name(name: &str) -> Result<(), CallRefusal> {
    if is_tool_name(name) {
        Ok(())
    } else {
        Err(CallRefusal::Name {
            name: name.to_string(),
        })
    }
}

/// Resolve `name` for `caller` and validate `supplied` against the tool's
/// declared interface, returning the script to run and the arguments it gets.
///
/// Availability is settled before the catalogue is read — exactly as a normal
/// call already does it — so no path that resolves a tool can be used to learn
/// what exists. `strict` additionally refuses arguments the tool does not
/// declare; the default (a normal call) ignores them and reports them back.
pub(crate) async fn resolve_tool_call(
    caller: &str,
    name: &str,
    supplied: &Map<String, Value>,
    strict: bool,
) -> Result<ResolvedCall, CallRefusal> {
    check_name(name)?;
    // The admin may call anything; a guest only what is granted to them. The
    // refusal comes before the catalogue is read so it cannot probe which tools
    // exist — nor leak anything about their declared parameters.
    if !crate::users::is_admin(caller).await
        && !crate::users::granted_tools(caller)
            .await
            .iter()
            .any(|granted| granted == name)
    {
        return Err(CallRefusal::Unavailable {
            name: name.to_string(),
        });
    }
    let Some(tool) = catalogue().await.into_iter().find(|t| t.name == name) else {
        return Err(CallRefusal::NotUsable {
            name: name.to_string(),
        });
    };

    let ignored = ignored_arguments(&tool, supplied);
    if strict && !ignored.is_empty() {
        // The note rides the refusal, exactly as it rides a refused run: the
        // assistant is shown which arguments were ignored.
        return Err(CallRefusal::Arguments(report_ignored_failure(
            anyhow::anyhow!(
                "usage: custom tool \"{name}\" was given arguments it does not declare — \
                 hint: a trigger passes only the arguments the tool declares"
            ),
            &ignored,
        )));
    }
    let args = match check_arguments(&tool, supplied) {
        Ok(args) => args,
        Err(e) => return Err(CallRefusal::Arguments(report_ignored_failure(e, &ignored))),
    };
    Ok(ResolvedCall {
        path: tool.path,
        args,
        ignored,
    })
}

/// A JSON number that denotes an integer — `3` and `3.0` alike. The declared
/// type is a 64-bit integer, so a magnitude beyond `i64` does not denote one
/// and is reported as a type mismatch like any other wrong value.
#[expect(clippy::cast_possible_truncation)] // the guard keeps the cast exact
fn integer(value: &Value) -> Option<i64> {
    if let Some(n) = value.as_i64() {
        return Some(n);
    }
    let float = value.as_f64()?;
    // `2^53` is where an f64 stops representing integers exactly, so the cast
    // below is exact for every value that reaches it.
    let integral = float.fract() == 0.0 && float.abs() <= 9_007_199_254_740_992.0;
    integral.then_some(float as i64)
}

/// Coerce one supplied value to its parameter's declared basic type.
fn checked(param: &Param, value: &Value) -> anyhow::Result<Value> {
    let mismatch = || super::wrong_type(&param.name, param.ty.expected(), value);
    match param.ty {
        ParamType::Str => value
            .as_str()
            .map(str::to_string)
            .map(Value::from)
            .ok_or_else(mismatch),
        ParamType::Integer => integer(value).map(Value::from).ok_or_else(mismatch),
        ParamType::Boolean => value.as_bool().map(Value::from).ok_or_else(mismatch),
        ParamType::List => {
            let items = value.as_array().ok_or_else(mismatch)?;
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item.as_str() {
                    Some(item) => out.push(Value::from(item.to_string())),
                    None => anyhow::bail!(
                        "usage: argument \"{}\" must be an array of strings, got a non-string \
                         element — hint: pass a JSON array of strings, e.g. {}: [\"a\", \"b\"]",
                        param.name,
                        param.name
                    ),
                }
            }
            Ok(Value::Array(out))
        }
    }
}

/// The whole of the argument contract: the declared parameters are checked
/// shallowly — string, integer, boolean and list-of-strings values, required
/// ones present — and anything deeper is the script's own job.
///
/// Returns the payload the script receives: only the declared parameters the
/// caller supplied, under the names it used. An explicit `null` counts as
/// omitted — the usual way a model says "no value for this one" — so it is
/// neither forwarded nor a type mismatch. The unrecognised names come from
/// [`ignored_arguments`], which the caller reports back alongside this.
fn check_arguments(
    tool: &CustomToolEntry,
    supplied: &Map<String, Value>,
) -> anyhow::Result<Map<String, Value>> {
    let mut payload = Map::new();
    for param in &tool.params {
        match supplied.get(&param.name) {
            None | Some(Value::Null) => {
                if param.required {
                    anyhow::bail!(
                        "usage: argument \"{}\" is required by \"{}\" — hint: pass it in the \
                         `args` object of the call",
                        param.name,
                        tool.name
                    );
                }
            }
            Some(value) => {
                payload.insert(param.name.clone(), checked(param, value)?);
            }
        }
    }
    Ok(payload)
}

/// The supplied argument names the tool does not declare: ignored and reported
/// back to the caller, never fatal.
fn ignored_arguments(tool: &CustomToolEntry, supplied: &Map<String, Value>) -> Vec<String> {
    let mut ignored: Vec<String> = supplied
        .keys()
        .filter(|key| !tool.params.iter().any(|p| &p.name == *key))
        .cloned()
        .collect();
    ignored.sort();
    ignored
}

/// The single native tool that forwards a call to one admin-authored script.
/// Available to every Assistant (the admin's and a guest's alike); the grant
/// decides what each call may reach.
pub(crate) struct CustomTool;

#[async_trait]
impl Tool for CustomTool {
    fn name(&self) -> &'static str {
        "custom"
    }

    fn parameters_schema(&self) -> Value {
        super::tool_params_schema(
            &json!({
                "tool": {
                    "type": "string",
                    "description": "Name of the custom tool to call — one of the names listed in the <custom-tools> block."
                },
                "args": {
                    "type": "object",
                    "description": "The tool's arguments, keyed by parameter name as declared in the <custom-tools> block. Values are validated against the declared types; unknown keys are ignored and reported back."
                }
            }),
            &["tool"],
        )
    }

    async fn execute(&self, ws: &Workspace, args: Value) -> Result<String> {
        // The caller is resolved from the live call and fails closed: an
        // unresolvable identity is refused outright, never treated as the
        // admin.
        let caller = crate::agent::tool_user_name();
        if caller.trim().is_empty() {
            anyhow::bail!(
                "forbidden: this call has no acting user — hint: custom tools are only \
                 callable from an Assistant session"
            );
        }

        let name = super::get_str(&args, "tool")?;
        // The call's identity is settled before its arguments, as it was before
        // the resolver below was extracted: a bad name outranks a malformed
        // `args`.
        check_name(name).map_err(CallRefusal::into_error)?;
        let supplied = super::get_object(&args, "args")?;
        // A normal call is not strict: an argument the tool does not declare is
        // ignored and reported back with the run rather than refused.
        let call = resolve_tool_call(&caller, name, &supplied, false)
            .await
            .map_err(CallRefusal::into_error)?;

        let Some(bun) = crate::tools::bun::bun_binary_path() else {
            return Err(report_ignored_failure(
                super::internal_fault("the managed bun runtime is unavailable"),
                &call.ignored,
            ));
        };
        let run = crate::tools::shell::run_program_with_timeout(
            ws,
            &bun,
            &[call.path.display().to_string(), call.payload()],
            &format!("custom tool \"{name}\""),
        )
        .await;
        match run {
            Ok(output) => Ok(report_ignored(&output, &call.ignored)),
            Err(e) => Err(report_ignored_failure(e, &call.ignored)),
        }
    }
}

/// Append the unrecognised-argument note to a run's output or a failure's
/// reason — the note rides whatever the caller is about to see.
fn report_ignored(text: &str, ignored: &[String]) -> String {
    if ignored.is_empty() {
        return text.to_string();
    }
    crate::tools::shell::with_note(
        text,
        &format!("[ignored arguments: {}]", ignored.join(", ")),
    )
}

/// A failure the caller sees, carrying the note when there is something
/// unrecognised to report. An empty note leaves the error exactly as it was.
///
/// The note is appended to the message because every producer on this path is
/// message-only, while anyhow's `.context` would render it before the reason it
/// annotates.
fn report_ignored_failure(e: anyhow::Error, ignored: &[String]) -> anyhow::Error {
    if ignored.is_empty() {
        return e;
    }
    anyhow::Error::msg(report_ignored(&e.to_string(), ignored))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::ProbeFile;

    #[test]
    fn header_parses_comments_and_all_four_types() {
        let source = "#!/usr/bin/env bun\n\
                      // @description Fetches the weather for a city.\n\
                      // @param city string required the city to look up\n\
                      // @param days integer optional forecast days\n\
                      /* @param metric boolean required use metric units */\n\
                      /**\n\
                       * @param tags list optional labels\n\
                       */\n\
                      \n\
                      const city = process.argv[2];\n\
                      // @param late string optional after the code\n";
        let (description, params) = parse_header(source).expect("valid header");
        assert_eq!(description, "Fetches the weather for a city.");
        // The last line sits past the first line of code, so it is not part of
        // the header block at all.
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["city", "days", "metric", "tags"]);
        assert_eq!(params[0].ty, ParamType::Str);
        assert!(params[0].required);
        assert_eq!(params[0].description, "the city to look up");
        assert_eq!(params[1].ty, ParamType::Integer);
        assert!(!params[1].required);
        assert_eq!(params[2].ty, ParamType::Boolean);
        // A block comment's closing punctuation is not part of its prose.
        assert_eq!(params[2].description, "use metric units");
        assert_eq!(params[3].ty, ParamType::List);
        assert_eq!(params[3].description, "labels");
    }

    #[test]
    fn malformed_headers_are_rejected() {
        let cases: [&str; 8] = [
            // no description at all
            "// @param city string required\n",
            // no header at all
            "const x = 1;\n",
            // empty description
            "// @description\n",
            // a second description
            "// @description x\n// @description y\n",
            // a misspelled directive is a typo, not a silent no-op
            "// @description x\n// @parm city string required\n",
            // unknown type
            "// @description x\n// @param city float required\n",
            // missing required/optional
            "// @description x\n// @param city string\n",
            // duplicate parameter
            "// @description x\n// @param c string required\n// @param c string optional\n",
        ];
        for case in cases {
            assert!(parse_header(case).is_none(), "must reject: {case:?}");
        }
    }

    #[test]
    fn catalogue_lines_narrate_parameters() {
        let tools = [
            CustomToolEntry {
                name: "weather".to_string(),
                description: "Fetches the weather.".to_string(),
                params: vec![
                    Param {
                        name: "city".to_string(),
                        ty: ParamType::Str,
                        required: true,
                        description: "the city to look up".to_string(),
                    },
                    Param {
                        name: "days".to_string(),
                        ty: ParamType::Integer,
                        required: false,
                        description: String::new(),
                    },
                ],
                path: PathBuf::from("weather.ts"),
            },
            CustomToolEntry {
                name: "disk".to_string(),
                description: "Reports disk usage.".to_string(),
                params: Vec::new(),
                path: PathBuf::from("disk.js"),
            },
        ];
        let refs: Vec<&CustomToolEntry> = tools.iter().collect();
        assert_eq!(
            render_lines(&refs),
            "- weather: Fetches the weather. Parameters: city (string, required) — the city to \
             look up; days (integer, optional).\n- disk: Reports disk usage."
        );
    }

    #[test]
    fn catalogue_skips_unusable_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let write = |name: &str, body: &str| {
            std::fs::write(tmp.path().join(name), body).expect("write");
        };
        write(
            "weather.ts",
            "// @description Weather.\n// @param city string required city\n",
        );
        write("helper.js", "// @description Helper.\n");
        // Not runnable by the runtime.
        write("notes.txt", "// @description Notes.\n");
        write("script.py", "// @description Python.\n");
        // A dot-file is not a tool: discovery and the call path share one name
        // predicate, so the block never advertises a name a call would refuse.
        write(".hidden.ts", "// @description Hidden.\n");
        // Malformed header.
        write("broken.ts", "// @param city string required\n");
        // Same stem as weather.ts — the first file by path wins, so the
        // collision is not order-dependent.
        write("weather.js", "// @description Other weather.\n");
        std::fs::create_dir(tmp.path().join("nested")).expect("mkdir");
        std::fs::write(
            tmp.path().join("nested/hidden.ts"),
            "// @description Nested.\n",
        )
        .expect("write");

        let tools = load_catalogue(tmp.path());
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["helper", "weather"]);
        let weather = tools.iter().find(|t| t.name == "weather").expect("weather");
        assert_eq!(weather.description, "Other weather.");
        assert!(weather.path.ends_with("weather.js"));
    }

    /// The block's listing split is what keeps a guest from being told about
    /// tools it cannot call: the whole catalogue for the admin, the
    /// granted subset otherwise, and the matching brief form when nothing is
    /// left.
    #[test]
    fn block_lists_the_whole_catalogue_or_the_granted_subset() {
        let entry = |name: &str| CustomToolEntry {
            name: name.to_string(),
            description: format!("The {name} tool."),
            params: Vec::new(),
            path: PathBuf::from(format!("{name}.ts")),
        };
        let catalogue = [entry("alpha"), entry("beta")];

        let admin = block_for(&catalogue, &[], true);
        assert!(
            admin.contains("alpha") && admin.contains("beta"),
            "got: {admin}"
        );

        let granted = block_for(&catalogue, &["beta".to_string()], false);
        assert!(
            granted.contains("beta") && !granted.contains("alpha"),
            "got: {granted}"
        );

        // A grant whose file is gone contributes nothing.
        assert_eq!(
            block_for(&catalogue, &["gone".to_string()], false),
            load_prompt("context/custom_tools_no_grants.md")
        );
        assert_eq!(
            block_for(&[], &[], true),
            load_prompt("context/custom_tools_none.md")
        );
    }

    /// The access gate and the argument contract, end to end: an unresolvable
    /// identity is refused (never treated as the admin), grants are checked
    /// before the catalogue is read, only plain file names are accepted, and the
    /// unrecognised-argument note rides a refusal as well as a run.
    #[tokio::test]
    async fn call_gate_and_argument_contract() {
        crate::util::test::init_management_test_stores().await;
        let ws = crate::workspace::test_ws("/tmp/custom_tool_gate");
        let tool = CustomTool;
        let call = |args: Value| tool.execute(&ws, args);
        let as_user = |user: &str, args: Value| {
            crate::agent::CURRENT_TOOL_USER_NAME.scope(user.to_string(), call(args))
        };

        // No acting user: refused, not resolved to the admin.
        let err = call(json!({ "tool": "ghost" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("forbidden:"), "got: {err}");

        // An ungranted caller is refused before the tool's existence is
        // consulted, so the refusal is not an existence oracle. The name is
        // unique to this test: every test in the process shares one users
        // store, so a common name could carry another test's grants.
        let err = as_user("custom_gate_guest", json!({ "tool": "ghost" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not granted to you"), "got: {err}");

        // The admin bypasses the grant check and reaches existence.
        let err = as_user("admin", json!({ "tool": "ghost" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("not-found:"), "got: {err}");

        // Only a plain file name is a tool name, so no call can address a file
        // outside the folder.
        for bad in ["../probe", ".hidden", "a/b", "a\\b", ""] {
            let err = as_user("admin", json!({ "tool": bad }))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.starts_with("forbidden:"), "{bad:?} got: {err}");
        }

        // Author a usable tool in the admin's own folder (the test root's), then
        // refuse a call to it: the declared interface decides what counts as
        // unrecognised, and the note rides the refusal.
        let dir = shared_dir();
        std::fs::create_dir_all(&dir).expect("create the shared folder");
        let probe = ProbeFile(dir.join("probe.ts"));
        std::fs::write(
            &probe.0,
            "// @description Probe.\n// @param city string required the city\n",
        )
        .expect("write the probe tool");
        let err = as_user("admin", json!({ "tool": "probe", "args": { "extra": 1 } }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("usage: "), "got: {err}");
        assert!(err.contains("[ignored arguments: extra]"), "got: {err}");
    }

    /// The one difference a strict caller (an alarm) has: an argument the tool
    /// does not declare is refused rather than ignored, and the availability
    /// gate is settled before anything is looked up either way.
    #[tokio::test]
    async fn strict_calls_refuse_undeclared_arguments() {
        crate::util::test::init_management_test_stores().await;
        let dir = shared_dir();
        std::fs::create_dir_all(&dir).expect("create the shared folder");
        let probe = ProbeFile(dir.join("strict_probe.ts"));
        std::fs::write(
            &probe.0,
            "// @description Strict probe.\n// @param city string required the city\n",
        )
        .expect("write the probe tool");
        let supplied = json!({ "city": "Minsk", "extra": 1 });
        let supplied = supplied.as_object().unwrap();

        let Err(CallRefusal::Arguments(e)) =
            resolve_tool_call("admin", "strict_probe", supplied, true).await
        else {
            panic!("a strict call must refuse an undeclared argument");
        };
        assert!(e.to_string().starts_with("usage: "), "got: {e}");

        // The same arguments are a normal call's business as usual: they
        // resolve, and the unrecognised one comes back for the caller to see.
        let resolved = resolve_tool_call("admin", "strict_probe", supplied, false)
            .await
            .unwrap_or_else(|_| panic!("a normal call ignores an undeclared argument"));
        assert_eq!(resolved.ignored, ["extra"]);
        assert_eq!(resolved.args["city"], json!("Minsk"));

        // A guest without the grant is refused whatever the name resolves to —
        // here to nothing at all — so no path that arms an alarm can probe the
        // catalogue.
        let unavailable = resolve_tool_call("strict_probe_guest", "ghost", supplied, true).await;
        assert!(
            matches!(unavailable, Err(CallRefusal::Unavailable { .. })),
            "an ungranted caller must be refused before the tool is looked up"
        );
    }

    #[test]
    fn arguments_are_checked_shallowly() {
        let tool = CustomToolEntry {
            name: "weather".to_string(),
            description: String::new(),
            params: vec![
                Param {
                    name: "city".to_string(),
                    ty: ParamType::Str,
                    required: true,
                    description: String::new(),
                },
                Param {
                    name: "days".to_string(),
                    ty: ParamType::Integer,
                    required: false,
                    description: String::new(),
                },
                Param {
                    name: "tags".to_string(),
                    ty: ParamType::List,
                    required: false,
                    description: String::new(),
                },
            ],
            path: PathBuf::from("weather.ts"),
        };

        // Only declared parameters reach the script, under the names supplied.
        let supplied = json!({"city": "Minsk", "days": 3.0, "tags": ["a", "b"], "extra": true});
        let supplied = supplied.as_object().unwrap();
        let payload = check_arguments(&tool, supplied).expect("valid arguments");
        assert_eq!(payload["city"], json!("Minsk"));
        assert_eq!(payload["days"], json!(3));
        assert_eq!(payload["tags"], json!(["a", "b"]));
        assert_eq!(ignored_arguments(&tool, supplied), ["extra"]);

        // A missing required parameter, a wrong type and a bad list element are
        // refused with the product's `usage:` shape.
        for supplied in [
            json!({}),
            json!({"city": 5}),
            json!({"city": "x", "tags": [1]}),
        ] {
            let err =
                check_arguments(&tool, supplied.as_object().unwrap()).expect_err("must refuse");
            assert!(err.to_string().starts_with("usage: "), "{err}");
        }
    }
}

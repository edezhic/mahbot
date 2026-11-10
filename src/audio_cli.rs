//! `mahbot audio` — the product's own speech engines on the command line.
//!
//! Two operations, both the product's own engines and nothing else: `transcribe` turns an
//! existing recording into text on standard output, `speak` turns text into a spoken WAV file.
//! Both run on every platform. Neither needs the running product — this command line is
//! dispatched before the instance lock — and neither touches its stores or settings: both read
//! the machine-wide speech model folder ([`crate::speech`]) and nothing else, each set obtained
//! the way the product obtains it (once, saying so on standard error, no retry loop).
//!
//! The product's speech switches are not this command's: they govern what the app says aloud
//! and transcribes, while these operations are explicit requests carrying their own
//! destination — nothing is read from the settings and nothing changed there. `speak` renders
//! through the very path the app speaks a phrase through, so a file made here sounds like the
//! product speaking; the whole text is spoken, and a text the engine cannot speak to its end
//! fails the command rather than being cut short.
//!
//! The four exit classes are the product's own ([`crate::cli::Exit`]), and what separates them
//! is who judges. This command judges the words and the values they carry (a flag no operation
//! takes, a recording the recogniser does not take, a destination it cannot write) — the
//! caller's own misuse. What the ENGINE judges is a failed operation. What is missing to run
//! at all is the environment: a model folder that is not there or cannot be written.
//!
//! The order is deliberate: the words first (an operation no word names, a flag this operation
//! does not take, one given twice, one with no value), then what the caller handed over, then
//! the work itself — because a typo must not cost a model download. The models are obtained
//! only after all of that, the work is done, and only then is anything printed or written.
//!
//! A read-only shell must never admit this family: `speak` writes a file at a path the caller
//! names, so the guard the inspecting roles hold lists it among its refusals.

use std::fmt::Write as _;
use std::path::PathBuf;

use crate::cli::{Exit, grammar};
use crate::util::{print_stderr, print_stdout};

// One flag of one operation: the value it carries and the line the help shows.
struct Flag {
    /// The spelling without its dashes (`--text` is `text`).
    name: &'static str,
    /// Whether the caller has to give it — the help says so beside the flag, and [`parse`] refuses
    /// its absence before reading any value.
    required: bool,
    /// One line for the operation's help.
    purpose: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    /// An existing recording into text.
    Transcribe,
    /// Text into a spoken file.
    Speak,
}

impl Operation {
    const ALL: [Self; 2] = [Self::Transcribe, Self::Speak];

    const fn word(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Speak => "speak",
        }
    }

    const fn purpose(self) -> &'static str {
        match self {
            Self::Transcribe => "Turn a recording into text, printed on standard output",
            Self::Speak => "Turn text into a spoken WAV file",
        }
    }

    const fn usage(self) -> &'static str {
        match self {
            Self::Transcribe => "mahbot audio transcribe --input <file>",
            Self::Speak => "mahbot audio speak --text <text> --output <path>",
        }
    }

    /// The flags this operation takes — one list for the parser and the help both, so neither can
    /// name a flag the other does not read, nor disagree about which are required.
    const fn flags(self) -> &'static [Flag] {
        match self {
            Self::Transcribe => &[Flag {
                name: "input",
                required: true,
                purpose: "The recording to turn into text — WAV, OGG/Opus or MP3, by name or by \
                          content",
            }],
            Self::Speak => &[
                Flag {
                    name: "text",
                    required: true,
                    purpose: "The text to speak",
                },
                Flag {
                    name: "output",
                    required: true,
                    purpose: "Where to write the file: a name ending in '.wav', or an existing \
                              directory to have one named for you",
                },
            ],
        }
    }

    fn from_word(word: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|operation| operation.word() == word)
    }
}

/// One command line, read: what was asked for, or an end to it.
enum Invocation {
    /// `mahbot audio -h` — this command's own help.
    Help,
    /// `mahbot audio <operation> -h` — that operation's help.
    OperationHelp(Operation),
    Transcribe {
        input: PathBuf,
    },
    Speak(Speak),
}

/// What `speak` carries. The text is kept as the caller spelled it: the preparation the product
/// speaks through is applied to it, never the caller's words rewritten.
struct Speak {
    text: String,
    output: PathBuf,
}

// One operation's flag tail, read.

enum Tail {
    /// A `-h`/`--help` stood where a flag could — that operation's help.
    Help,
    /// The values given, in the order they were given.
    Values(Vec<(&'static str, String)>),
}

/// Read one operation's flag tail — the tokens after the operation word. The grammar is
/// [`crate::cli::grammar`]'s and a value is read VERBATIM: the very next token, whatever it looks
/// like (`--text -h` is the text `-h`). This command refuses only what is its own — a token that is
/// no flag, a flag this operation does not take, one given twice, one with no value — and names the
/// flag, never its value: a mistyped value is the caller's business.
fn parse_flags(operation: Operation, tokens: &[String]) -> Result<Tail, String> {
    let flags = operation.flags();
    let mut values: Vec<(&'static str, String)> = Vec::new();
    let mut rest = tokens.iter();
    while let Some(token) = rest.next() {
        let (name, inline) = match grammar::read_token(token)? {
            grammar::Token::Help => return Ok(Tail::Help),
            grammar::Token::Flag { name, inline } => (name, inline),
        };
        let Some(flag) = flags.iter().find(|flag| flag.name == name) else {
            return Err(grammar::unknown_flag(name, operation.word()));
        };
        if values.iter().any(|(given, _)| *given == flag.name) {
            return Err(grammar::repeated_flag(flag.name));
        }
        let value = match inline {
            Some(value) => value.to_string(),
            None => rest
                .next()
                .cloned()
                .ok_or_else(|| grammar::flag_value_missing(name))?,
        };
        values.push((flag.name, value));
    }
    Ok(Tail::Values(values))
}

fn parse(args: &[String]) -> Result<Invocation, String> {
    let Some(word) = args.first() else {
        return Err("no operation was given".to_string());
    };
    if grammar::is_help(word) {
        return Ok(Invocation::Help);
    }
    let Some(operation) = Operation::from_word(word) else {
        return Err(format!("unknown operation '{word}'"));
    };
    let values = match parse_flags(operation, &args[1..]) {
        Ok(Tail::Help) => return Ok(Invocation::OperationHelp(operation)),
        Ok(Tail::Values(values)) => values,
        Err(reason) => return Err(reason),
    };

    let given = |name: &str| {
        values
            .iter()
            .find(|(given, _)| *given == name)
            .map(|(_, value)| value.clone())
    };
    // The operation's own list says which flags are required, so the first one it calls required
    // and the tail lacks ends the parse here — which makes the lookups below safe: every required
    // value is present.
    if let Some(flag) = operation
        .flags()
        .iter()
        .find(|flag| flag.required && given(flag.name).is_none())
    {
        return Err(format!(
            "flag '--{}' is required for '{}'",
            flag.name,
            operation.word()
        ));
    }
    let required = |name: &str| given(name).expect("a required flag, checked just above");
    match operation {
        Operation::Transcribe => Ok(Invocation::Transcribe {
            input: PathBuf::from(required("input")),
        }),
        Operation::Speak => Ok(Invocation::Speak(Speak {
            text: required("text"),
            output: PathBuf::from(required("output")),
        })),
    }
}

// What these operations are — one block for both helps, so a caller entering through one
// operation's own help reads the same contract as one reading the family's.
const ENGINE_HELP: &str = "\nA frontend onto the product's own speech engines: everything happens \
     on this machine and nothing is sent anywhere. The engines need their model set and nothing \
     else — both read the machine-wide speech model folder — not the product's settings, not its \
     databases, and not the running product, so this works while it is stopped and running it \
     while the product is up does not disturb it. The first use of an operation fetches the \
     models it needs, once, saying so on standard error; if it cannot obtain them, it fails \
     rather than trying again.\n\
     \nBoth operations run on every platform. The product's speech switches govern what the app \
     itself speaks and transcribes; these operations are explicit requests with their own \
     destination — they neither need those switches nor change them.\n";

/// The exit classes, word for word the numbering a script reads (see [`Exit`]).
const EXIT_CLASS_HELP: &str = "\nExit codes:\n  0  the operation was carried out\n  1  the engine \
     could not do the work\n  2  the environment — the models cannot be obtained\n  3  this \
     command was misused — what was asked for cannot be done as asked\n\
     \nThe words are read before anything is fetched, so a malformed invocation is misuse and \
     never an environment failure.\n";

/// Where the two streams go — shared by both helps for the same reason.
const OUTPUT_HELP: &str = "\nOutput:\n  stdout  transcribe: the recognised text, and nothing at \
     all when the recording holds no speech; speak: the path of the file, when the destination \
     was a directory and this command named the file itself\n  stderr  everything this command \
     itself has to say\n";

/// `mahbot audio -h`: what this command is and what a caller reads back from it.
#[must_use]
fn top_help() -> String {
    let mut out =
        String::from("mahbot audio — the product's own speech engines on the command line\n");
    out.push_str("\nUsage: mahbot audio <operation> [flags]\n\nOperations:\n");
    let width = Operation::ALL
        .iter()
        .map(|operation| operation.word().len())
        .max()
        .unwrap_or_default();
    for operation in Operation::ALL {
        let _ = writeln!(
            out,
            "  {:<width$}  {}",
            operation.word(),
            operation.purpose()
        );
    }
    out.push_str(
        "\nA flag is `--<name> <value>` or `--<name>=<value>`, and its value is taken verbatim \
         (so a text may begin with a dash). `mahbot audio <operation> -h` names one operation's \
         flags in full.\n",
    );
    out.push_str(ENGINE_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out
}

/// `mahbot audio <operation> -h`: what one operation does and the flags it takes.
#[must_use]
fn operation_help(operation: Operation) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "mahbot audio {} — {}",
        operation.word(),
        operation.purpose()
    );
    let _ = write!(out, "\nUsage: {}\n", operation.usage());
    out.push_str("\nFlags:\n");
    let flags = operation.flags();
    let rows: Vec<(String, &str)> = flags
        .iter()
        .map(|flag| {
            let required = if flag.required { " (required)" } else { "" };
            (format!("--{} <value>{required}", flag.name), flag.purpose)
        })
        .chain(std::iter::once((
            "-h, --help".to_string(),
            "Print this help and exit",
        )))
        .collect();
    let width = rows
        .iter()
        .map(|(spelling, _)| spelling.len())
        .max()
        .unwrap_or_default();
    for (spelling, purpose) in rows {
        let _ = writeln!(out, "  {spelling:<width$}  {purpose}");
    }
    out.push_str("\nA flag's value is the token right after it, taken verbatim.\n");
    out.push_str(match operation {
        Operation::Transcribe => {
            "The product's own local recognition, and no language is named: the engine detects \
             it. A recording the engine finds no speech in is not a failure — stdout then carries \
             nothing at all, which is how a script tells it from an empty line. A recording \
             longer than the engine will take fails, never a transcript with a part of it \
             missing.\n"
        }
        Operation::Speak => {
            "The whole text is spoken as one reading, through the same preparation of the text \
             the product speaks with: it is not split, and no length is imposed here — a text the \
             engine cannot speak to its end fails rather than being cut short. The destination is \
             never overwritten, the directory it names must already exist, and nothing is written \
             at all if the text cannot be spoken.\n"
        }
    });
    out.push_str(ENGINE_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out
}

// `mahbot audio` — the process exit code. Help goes to stdout and the caller's misuse is answered
// from the words alone. The two operations themselves live in [`engine`].

pub async fn run(args: &[String]) -> i32 {
    match parse(args) {
        Ok(Invocation::Help) => {
            print_stdout(&top_help());
            Exit::Ok.code()
        }
        Ok(Invocation::OperationHelp(operation)) => {
            print_stdout(&operation_help(operation));
            Exit::Ok.code()
        }
        Ok(Invocation::Transcribe { input }) => engine::transcribe(&input).await,
        Ok(Invocation::Speak(request)) => engine::speak(&request).await,
        Err(reason) => misuse(&reason),
    }
}

/// Refuse what was asked for as the caller's own mistake: his words cannot mean it, or cannot be
/// done as asked.
fn misuse(reason: &str) -> i32 {
    print_stderr(&format!("mahbot audio: {reason}"));
    print_stderr("run 'mahbot audio -h' for usage.");
    Exit::Usage.code()
}

mod engine {
    use super::{Speak, misuse};
    use crate::cli::Exit;
    use crate::speech::{
        AUDIO_FORMATS, INFERENCE_TIMEOUT, InferenceTimedOut, ensure_ready_once, synth, takes_audio,
        transcribe_file_async,
    };
    use crate::util::{print_stderr, print_stdout};
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};

    /// How many names to try before giving up on picking a free one: the chosen name is a fresh
    /// identifier, so the first is free for any practical purpose — the loop makes "never collides"
    /// true in principle too.
    const CHOSEN_NAME_TRIES: usize = 3;

    /// `transcribe`: the recording into text on standard output.
    pub(super) async fn transcribe(input: &Path) -> i32 {
        // What the caller handed over, before anything is fetched: a typo must not cost a model
        // download. The type is read before anything is opened, because opening is what waits — a
        // FIFO named like a recording would hold it until somebody writes into it.
        match std::fs::metadata(input) {
            Ok(metadata) if metadata.is_file() => {}
            // A directory has metadata too, and is not a recording either.
            Ok(_) => {
                return misuse(&format!(
                    "'{}' is not a file to transcribe",
                    input.display()
                ));
            }
            Err(e) => return misuse(&cannot_read(input, &e)),
        }
        // The recogniser's own rule, asked of the recogniser itself: a name it knows, or content
        // beginning with Ogg's magic. Opening is part of asking, so an unreadable input is the
        // caller's mistake, not a failed operation.
        match takes_audio(input) {
            Err(e) => return misuse(&cannot_read(input, &e)),
            Ok(true) => {}
            Ok(false) => {
                return misuse(&format!(
                    "'{}' is not a recording the local recogniser takes — it takes the names {}, \
                     and a name outside them only when the file's content is Ogg",
                    input.display(),
                    listed(AUDIO_FORMATS.iter().map(|(name, _)| format!("'.{name}'"))),
                ));
            }
        }

        if let Err(e) = ensure_ready_once(|dir| {
            announce_fetch(dir, "recognition");
        })
        .await
        {
            print_stderr(&format!(
                "mahbot audio: the recognition models cannot be made ready: {e:#}"
            ));
            return Exit::Environment.code();
        }

        match transcribe_file_async(input, INFERENCE_TIMEOUT).await {
            Ok(text) => {
                // No speech found is an answer, not a failure: nothing at all on stdout, so a
                // script can tell it from an empty line.
                let text = text.trim();
                if !text.is_empty() {
                    print_stdout(text);
                }
                Exit::Ok.code()
            }
            // The bound is told by the TYPE the failure carries, never by its wording.
            Err(e) if e.downcast_ref::<InferenceTimedOut>().is_some() => {
                print_stderr(&format!(
                    "mahbot audio: the recording is longer than the local recogniser will take — \
                     it reached the engine's own {}-minute inference bound, and nothing was \
                     transcribed rather than a transcript with a part of it missing",
                    INFERENCE_TIMEOUT.as_secs() / 60,
                ));
                Exit::Failed.code()
            }
            Err(e) => {
                print_stderr(&format!(
                    "mahbot audio: the recording could not be transcribed: {e:#}"
                ));
                Exit::Failed.code()
            }
        }
    }

    /// `speak`: the text into a spoken file, where the caller says to write it.
    pub(super) async fn speak(request: &Speak) -> i32 {
        // What the caller asked for, before anything is fetched.
        if request.text.trim().is_empty() {
            return misuse("the text to speak says nothing — it is empty or only whitespace");
        }
        let destination = match resolve_destination(&request.output) {
            Ok(destination) => destination,
            Err(reason) => return misuse(&reason),
        };

        // The set is obtained before the reading: a failure here is the environment's.
        if let Err(e) = synth::ensure_ready_once(|dir| {
            announce_fetch(dir, "speech synthesis");
        })
        .await
        {
            print_stderr(&format!(
                "mahbot audio: the speech synthesis model cannot be obtained: {e:#}"
            ));
            return Exit::Environment.code();
        }

        // The whole reading is rendered before anything is written, so a text that cannot be spoken
        // in full leaves nothing at the destination.
        let wav = match synth::render_speech_wav(&request.text).await {
            Ok(wav) => wav,
            Err(e) => {
                print_stderr(&format!(
                    "mahbot audio: nothing was written — the text could not be spoken: {e:#}"
                ));
                return Exit::Failed.code();
            }
        };

        match write_audio(&destination, &wav) {
            Ok(Some(path)) => {
                print_stdout(&path.display().to_string());
                Exit::Ok.code()
            }
            Ok(None) => Exit::Ok.code(),
            Err(reason) => misuse(&reason),
        }
    }

    /// Say on standard error that a model set is being fetched — a command line has only its
    /// standard error. Said as a fetch begins (before the first byte), and only when one begins at
    /// all: a set already on disk is silent, a first use that downloads gigabytes is not.
    fn announce_fetch(dir: &Path, what: &str) {
        print_stderr(&format!(
            "mahbot audio: fetching the {what} model set into {} — fetched once, and only when it \
             is not already there",
            dir.display()
        ));
    }

    /// The refusal for an input that cannot be read at all — the same whether the type probe or the
    /// open found it.
    fn cannot_read(path: &Path, e: &std::io::Error) -> String {
        format!("'{}' cannot be read: {e}", path.display())
    }

    /// Where the spoken file goes: the path the caller named, or a fresh name this command picks
    /// inside a directory he named.
    pub(super) enum Destination {
        /// The caller named the file itself.
        Named(PathBuf),
        /// The caller named a directory; the name is chosen when the file is created, so it cannot
        /// collide.
        Chosen(PathBuf),
    }

    /// Read what `--output` names, before anything is fetched or rendered. The name is also where
    /// the caller says which container he wants, and uncompressed WAV is the only one the product
    /// writes. The destination directory is the caller's to make: this command never creates one.
    pub(super) fn resolve_destination(output: &Path) -> Result<Destination, String> {
        if output.is_dir() {
            return Ok(Destination::Chosen(output.to_path_buf()));
        }
        // A name with an extension names a container, and uncompressed WAV is the only one the
        // product writes. Absence is not a reason here — every new destination is absent — so a
        // container this command cannot write is refused for what it is, whether or not something
        // of that name is already there.
        if output
            .extension()
            .is_some_and(|extension| !extension.eq_ignore_ascii_case("wav"))
        {
            return Err(format!(
                "'{}' is not a file this command can write — the spoken audio is uncompressed \
                 WAV, so name it '….wav', or name a directory to have one named for you",
                output.display()
            ));
        }
        // Nothing after a dot and nothing at that path: the caller named the directory he expected
        // to find, and this command creates none.
        if output.extension().is_none() && !output.exists() {
            return Err(format!(
                "'{}' is not there — this command creates nothing, so name a '.wav' file in a \
                 directory that is already there, or an existing directory to have one named for \
                 you",
                output.display()
            ));
        }
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
            && !parent.is_dir()
        {
            // A parent that is there and is not a directory is not a missing one.
            return Err(if parent.exists() {
                format!(
                    "'{}' cannot be written: '{}' is not a directory, and this command does not \
                     create one",
                    output.display(),
                    parent.display()
                )
            } else {
                format!(
                    "the destination directory {} does not exist — this command does not create it",
                    parent.display()
                )
            });
        }
        if output.exists() {
            return Err(already_exists(output));
        }
        Ok(Destination::Named(output.to_path_buf()))
    }

    /// The refusal for a destination already taken, said the same way whether the name was checked
    /// before the work or lost the race at the create.
    fn already_exists(path: &Path) -> String {
        format!(
            "{} already exists — this command never overwrites a file",
            path.display()
        )
    }

    /// The refusal for audio that could not be written — the same whether the caller named the file
    /// or this command chose the name.
    fn cannot_write(path: &Path, e: &std::io::Error) -> String {
        format!("{} could not be written: {e}", path.display())
    }

    /// Write the rendered audio where the destination points, returning the path when this command
    /// picked the name.
    fn write_audio(destination: &Destination, wav: &[u8]) -> Result<Option<PathBuf>, String> {
        match destination {
            Destination::Named(path) => match write_new(path, wav) {
                Ok(()) => Ok(None),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => Err(already_exists(path)),
                Err(e) => Err(cannot_write(path, &e)),
            },
            Destination::Chosen(dir) => {
                for _ in 0..CHOSEN_NAME_TRIES {
                    let path = dir.join(format!("mahbot-speech-{}.wav", crate::generate_id()));
                    match write_new(&path, wav) {
                        Ok(()) => return Ok(Some(path)),
                        // The name was taken between choosing and creating: another name is one
                        // call away.
                        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                        Err(e) => return Err(cannot_write(&path, &e)),
                    }
                }
                Err(format!("no free name could be picked in {}", dir.display()))
            }
        }
    }

    /// Create `path` and write the audio into it. `create_new` makes "never overwritten" true even
    /// against a file that appeared a moment ago, and a failed write removes what it created — a
    /// failure leaves nothing behind.
    fn write_new(path: &Path, wav: &[u8]) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        if let Err(e) = file.write_all(wav) {
            drop(file);
            let _ = std::fs::remove_file(path);
            return Err(e);
        }
        Ok(())
    }

    /// A list of names as a sentence reads them: `a, b or c`, and just `a` for one.
    fn listed<I, S>(names: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names: Vec<String> = names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect();
        match names.split_last() {
            Some((last, [])) => last.clone(),
            Some((last, head)) => format!("{} or {last}", head.join(", ")),
            None => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grammar a caller meets before he hears anything: what a flag tail means, what one that
    /// means nothing says, and which required flags the operation's own list refuses by name.
    #[test]
    fn flag_tails_and_required_flags() {
        let values = |tokens: &[&str]| match parse_flags(Operation::Speak, &owned(tokens)) {
            Ok(Tail::Values(values)) => Ok(values),
            Ok(Tail::Help) => Err("help".to_string()),
            Err(reason) => Err(reason),
        };
        assert_eq!(
            values(&["--text", "-h", "--output", "a.wav"]).unwrap(),
            vec![("text", "-h".to_string()), ("output", "a.wav".to_string())],
            "a value is taken verbatim, even one that looks like a flag"
        );
        assert_eq!(
            values(&["--text=hi", "--output=b.wav"]).unwrap(),
            vec![("text", "hi".to_string()), ("output", "b.wav".to_string())]
        );
        // Neither of the removed spellings is accepted, silently or otherwise.
        for gone in ["--voice", "--language"] {
            assert!(
                values(&[gone, "x"]).is_err_and(|r| r.contains(gone)),
                "{gone} names no flag of this operation"
            );
        }
        assert!(values(&["--nope", "x"]).is_err_and(|r| r.contains("--nope")));
        assert!(
            values(&["--text", "a", "--text", "b"]).is_err_and(|r| r.contains("more than once"))
        );
        assert!(values(&["--text"]).is_err_and(|r| r.contains("needs a value")));
        assert!(values(&["text"]).is_err_and(|r| r.contains("--flag")));
        assert_eq!(values(&["--text", "hi", "-h"]).unwrap_err(), "help");

        // A required flag that was left out is refused by name, and the operation's own list says
        // which those are.
        let invocation = |tokens: &[&str]| parse(&owned(tokens));
        assert!(invocation(&["speak", "--text", "hi"]).is_err_and(|r| r.contains("--output")));
        assert!(invocation(&["transcribe"]).is_err_and(|r| r.contains("--input")));
        assert!(invocation(&["frobnicate"]).is_err_and(|r| r.contains("unknown operation")));
        assert!(invocation(&[]).is_err(), "no operation is misuse");
    }

    fn owned(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| (*token).to_string()).collect()
    }

    #[test]
    fn destination_rules() {
        use super::engine::{Destination, resolve_destination};

        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("taken.wav");
        std::fs::write(&existing, b"x").unwrap();

        assert!(matches!(
            resolve_destination(dir.path()),
            Ok(Destination::Chosen(_))
        ));
        assert!(matches!(
            resolve_destination(&dir.path().join("fresh.wav")),
            Ok(Destination::Named(_))
        ));
        assert!(resolve_destination(&existing).is_err_and(|r| r.contains("never overwrites")));
        // A container this command cannot write is refused for THAT, whether or not the name is
        // already there.
        std::fs::write(dir.path().join("taken.ogg"), b"x").unwrap();
        for unsupported in ["absent.ogg", "taken.ogg"] {
            assert!(
                resolve_destination(&dir.path().join(unsupported)).is_err_and(|reason| {
                    reason.contains("is not a file this command can write")
                        && !reason.contains("is not there")
                }),
                "{unsupported} names a container the product does not write, and must be refused \
                 for that whether or not it is there"
            );
        }
        // A name with nothing after a dot and nothing at that path is the directory the caller
        // expected to find.
        for absent_dir in ["no-such-dir", "no-extension"] {
            assert!(
                resolve_destination(&dir.path().join(absent_dir)).is_err_and(|reason| {
                    reason.contains("is not there") && reason.contains(".wav")
                }),
                "{absent_dir} is an absent directory, not a file name"
            );
        }
        assert!(
            resolve_destination(&dir.path().join("absent").join("out.wav"))
                .is_err_and(|r| r.contains("does not exist"))
        );
        // A parent that is there and is not a directory is refused for what it is.
        assert!(
            resolve_destination(&existing.join("out.wav")).is_err_and(|r| {
                r.contains("is not a directory") && !r.contains("does not exist")
            })
        );
    }
}

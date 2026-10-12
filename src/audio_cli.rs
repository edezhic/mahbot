//! `mahbot audio` — the product's own speech engine on the command line.
//!
//! Two operations, both the product's own engine and nothing else: `transcribe`
//! turns an existing recording into text on standard output, and `speak` turns
//! text into a spoken WAV file. Neither needs the running product — this command
//! line is dispatched before the instance lock — and neither touches its stores
//! or its settings: the only thing either reads or writes is the speech model set
//! under the product's own data folder, which it obtains the way the product does
//! when it needs it — once, saying so on standard error, and no retry loop.
//!
//! The product's read-aloud switches are not this command's. They govern what the
//! app does with its own replies and its own voice notes, while these operations
//! are explicit requests carrying their own voice, language and destination
//! (defaults and all) — nothing is read from the settings, and nothing is changed
//! there. `speak` renders its file through the very path the app reads aloud
//! through, so a file made here sounds like the product speaking; that path must
//! also draw the same text, so a part of the text that fails to render fails the
//! command rather than being left out of the file.
//!
//! The four exit classes are the product's own ([`crate::cli::Exit`]), and what
//! separates them is who judges. This command judges the words it read and the
//! values they carry — a flag no operation takes, a recording the recogniser
//! does not take, a value the engine has no such thing as, a destination it
//! cannot write — and those are the caller's own misuse. Everything the ENGINE
//! judges is a failed operation instead: the work it could not do. What is
//! missing to run at all — no engine on this platform, or models that cannot be
//! obtained — is the environment. All four are stated wherever this command is
//! listed, while only a platform that has the engine can report a failed one.
//!
//! The order the failures are answered in is deliberate, and it is the order the
//! code has. The words come first, and only the words — an operation no word
//! names, a flag this operation does not take, one given twice, one with no value
//! — which is misuse read from the words alone. Then the product's own data
//! folder, which is the environment's failure when it cannot be resolved. Then
//! what the values mean to the engine: the recording and the format it comes in,
//! a text that says nothing, a voice or a language the engine has no such thing
//! as, a destination this command cannot write — because a typo must not cost a
//! model download. The models are obtained only after all of that, the work is
//! done, and only then is anything printed or written.
//!
//! Where there is no engine at all (everywhere but macOS) both operations are the
//! same refusal naming the platform, and no VALUE is judged: the engine is what
//! judges a value, and there is none to ask. The words are read the same way
//! there as anywhere — they are this frontend's own grammar — and the help is
//! where a caller finds how the two classes divide when they, too, are wrong.
//!
//! What a read-only shell must never admit is this whole family: `speak` writes a
//! file at a path the caller names, so the guard the inspecting roles hold lists
//! it among the refusals it spells out (`tool/shell_readonly_banner.md`).

use std::fmt::Write as _;
use std::path::PathBuf;

use crate::cli::{Exit, grammar};
use crate::util::{print_stderr, print_stdout};

// ── The operations and their flags ────────────────────────────────

/// One flag of one operation: the value it carries and the line the help shows.
struct Flag {
    /// The spelling without its dashes (`--voice` is `voice`).
    name: &'static str,
    /// Whether the caller has to give it — the help says so beside the flag, and
    /// [`parse`] refuses its absence before it reads the values.
    required: bool,
    /// One line for the operation's help.
    purpose: &'static str,
}

/// The two operations this command has.
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
            Self::Speak => "mahbot audio speak --text <text> --output <path> [flags]",
        }
    }

    /// The flags this operation takes — one list for the parser and the help
    /// both, so the help can never name a flag the parser does not read, nor the
    /// parser take one the help leaves out, nor the two disagree about which of
    /// them is required.
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
                Flag {
                    name: "voice",
                    required: false,
                    purpose: "The voice to speak with — F1–F5 or M1–M5, with or without '.json'; \
                              the product's own voice when not given",
                },
                Flag {
                    name: "language",
                    required: false,
                    purpose: "The language to speak in — a code the engine knows ('en', 'ko', \
                              …), or 'na' for its language-agnostic default",
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

/// What `speak` carries. The values are kept as the caller spelled them: what
/// they MEAN is the engine's own vocabulary, and it is read where the engine
/// exists (see the module docs on the order the failures are answered in). Where
/// there is none, the refusal is the platform's and none of this is read — the
/// request is spelled the same way on every platform, and dropped unread where it
/// cannot be carried out.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct Speak {
    text: String,
    voice: Option<String>,
    language: Option<String>,
    output: PathBuf,
}

// ── Parsing ───────────────────────────────────────────────────────

/// One operation's flag tail, read.
enum Tail {
    /// A `-h`/`--help` stood where a flag could — that operation's help.
    Help,
    /// The values given, in the order they were given.
    Values(Vec<(&'static str, String)>),
}

/// Read one operation's flag tail — the tokens after the operation word.
///
/// The grammar is [`crate::cli::grammar`]'s, and a value is read VERBATIM: the
/// very next token, whatever it looks like (`--text -h` is the text `-h`). What
/// is this command's own is what it refuses — a token that is no flag at all, a
/// flag this operation does not take, a flag given twice, a flag with no value —
/// and that is misuse read from the words alone: it names the flag and never its
/// value, because a value the caller mistyped is his business.
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

/// Read a whole `mahbot audio …` command line.
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
    // The operation's own list is what says which flags are required, so the
    // first one it calls required and the tail does not carry ends the parse
    // here — which is also what makes the lookups below safe: every required
    // value they take is guaranteed present by this check, while the optional
    // ones stay `Option`.
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
            voice: given("voice"),
            language: given("language"),
        })),
    }
}

// ── Help ──────────────────────────────────────────────────────────

/// What these operations are — one block for both helps, so a caller entering
/// through one operation's own help reads the same contract as one reading the
/// family's.
const ENGINE_HELP: &str = "\nA frontend onto the product's own speech engine: everything \
     happens on this machine and nothing is sent anywhere. The engine needs the speech models \
     under the product's own data folder and nothing else — not the product's settings, not its \
     databases, and not the running product, so this works while it is stopped and running it \
     while the product is up does not disturb it. The first use of an operation fetches the \
     models it needs, once, saying so on standard error; if it cannot obtain them, it fails \
     rather than trying again.\n\
     \nThe product's voice switches govern what the app itself reads aloud and transcribes; \
     these operations are explicit requests with their own voice, language and destination — \
     they neither need those switches nor change them.\n";

/// The exit classes, word for word the numbering a script reads (see [`Exit`]).
const EXIT_CLASS_HELP: &str = "\nExit codes:\n  0  the operation was carried out\n  1  the engine \
     could not do the work\n  2  the environment — there is no speech engine on this platform, \
     or its models cannot be obtained\n  3  this command was misused — what was asked for cannot \
     be done as asked\n";

/// Where the two streams go — shared by both helps for the same reason.
const OUTPUT_HELP: &str = "\nOutput:\n  stdout  transcribe: the recognised text, and nothing at \
     all when the recording holds no speech; speak: the path of the file, when the destination \
     was a directory and this command named the file itself\n  stderr  everything this command \
     itself has to say\n";

/// The one thing about this command that is not the same everywhere — and the
/// only place two classes meet, so the order between them is said here too.
const PLATFORM_HELP: &str = "\nThe speech engine exists on macOS only: on a platform that has \
     none, both operations say so and name it as the reason.\n\
     \nThat is the one place where two of these classes meet, and which one is reported is \
     settled the same way everywhere, because the words are read before anything else is. An \
     invocation that is malformed — no operation, an operation no word names, an unknown or \
     repeated flag, a required flag left without a value — is the caller's own misuse on every \
     platform, since nothing about the engine was reached. Only once the invocation is well \
     formed does the platform come into it: a platform with no engine is then the environment, \
     and so is a value the engine would have judged, had there been one there to judge it.\n";

/// `mahbot audio -h`: what this command is and what a caller reads back from it.
#[must_use]
fn top_help() -> String {
    let mut out =
        String::from("mahbot audio — the product's own speech engine on the command line\n");
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
    out.push_str(PLATFORM_HELP);
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
            "The file is written the way the product speaks: the same preparation of the text, \
             the same splitting of long text at sentence boundaries, the same pause between the \
             parts written into the audio. The destination is never overwritten, the directory \
             it names must already exist, and nothing is written at all if any part of the text \
             fails to render.\n"
        }
    });
    out.push_str(ENGINE_HELP);
    out.push_str(EXIT_CLASS_HELP);
    out.push_str(OUTPUT_HELP);
    out.push_str(PLATFORM_HELP);
    out
}

// ── The run ───────────────────────────────────────────────────────

/// `mahbot audio` — returns the process exit code.
///
/// Help goes to stdout and the caller's own misuse is answered from the words
/// alone (see the module docs for the order the failures are answered in). The
/// two operations themselves live in [`engine`], one arm per platform.
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

/// Refuse what was asked for as the caller's own mistake: his own words cannot
/// mean it, or what they ask for cannot be done as asked.
fn misuse(reason: &str) -> i32 {
    print_stderr(&format!("mahbot audio: {reason}"));
    print_stderr("run 'mahbot audio -h' for usage.");
    Exit::Usage.code()
}

// ── The platform with no engine ───────────────────────────────────

/// The speech engine is macOS-only, so everywhere else both operations are this
/// one refusal. It is reached only through the two entry points below, and the
/// words are read before they are — so a malformed invocation is answered as the
/// caller's mistake here too, while no value is judged at all (the help puts the
/// two classes in order).
#[cfg(not(target_os = "macos"))]
fn no_engine() -> i32 {
    print_stderr(&format!(
        "mahbot audio: the speech engine is macOS-only — on {} there is none to run, so neither \
         operation can be carried out here",
        std::env::consts::OS
    ));
    Exit::Environment.code()
}

/// The two operations where there is no engine to run them: the same refusal,
/// with the same entry points the arm below has, so the caller is one line
/// either way and nothing about the request is judged here.
#[cfg(not(target_os = "macos"))]
mod engine {
    use super::{Speak, no_engine};
    use std::path::Path;

    #[expect(clippy::unused_async)] // the entry point the macOS arm has: there is nothing to await here
    pub(super) async fn transcribe(_input: &Path) -> i32 {
        no_engine()
    }

    #[expect(clippy::unused_async)] // the entry point the macOS arm has: there is nothing to await here
    pub(super) async fn speak(_request: &Speak) -> i32 {
        no_engine()
    }
}

// ── The two operations ────────────────────────────────────────────

/// The operations themselves, on the one platform that has a speech engine:
/// what the caller's values mean to it, and the work each operation asks for.
#[cfg(target_os = "macos")]
mod engine {
    use super::{Speak, misuse};
    use crate::audio::local_transcriber::{self, InferenceTimedOut};
    use crate::audio::tts;
    use crate::cli::Exit;
    use crate::util::{print_stderr, print_stdout};
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};

    /// How many names to try before giving up on picking a free one in a
    /// directory: the chosen name is a fresh identifier, so the first is
    /// already free for any practical purpose and the loop is only there to
    /// make "never collides" true in principle too.
    const CHOSEN_NAME_TRIES: usize = 3;

    /// `transcribe`: the recording into text on standard output.
    pub(super) async fn transcribe(input: &Path) -> i32 {
        if let Some(code) = prepare() {
            return code;
        }
        // What the caller handed over, before anything is fetched: a typo must
        // not cost a model download. The type is read before anything is opened,
        // because opening is what waits: a FIFO named like a recording would hold
        // it until somebody writes into it.
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
        // The recogniser's own rule, asked of the recogniser itself: a name it
        // knows, or content that begins with Ogg's magic. Opening the file is part
        // of asking, so an input that cannot be read is refused as the caller's
        // own mistake rather than as a failure of the work.
        match local_transcriber::takes_audio(input) {
            Err(e) => return misuse(&cannot_read(input, &e)),
            Ok(true) => {}
            Ok(false) => {
                return misuse(&format!(
                    "'{}' is not a recording the local recogniser takes — it takes the names {}, \
                     and a name outside them only when the file's content is Ogg",
                    input.display(),
                    listed(
                        local_transcriber::AUDIO_FORMATS
                            .iter()
                            .map(|(name, _)| format!("'.{name}'")),
                    ),
                ));
            }
        }

        if let Err(e) = local_transcriber::ensure_ready_once(|dir| {
            announce_fetch(dir, "recognition");
        })
        .await
        {
            print_stderr(&format!(
                "mahbot audio: the recognition model cannot be obtained: {e:#}"
            ));
            return Exit::Environment.code();
        }

        match local_transcriber::transcribe_file_async(input, local_transcriber::INFERENCE_TIMEOUT)
            .await
        {
            Ok(text) => {
                // A recording the engine found no speech in is an answer, not a
                // failure: nothing at all on standard output, so a script can
                // tell it from an empty line.
                let text = text.trim();
                if !text.is_empty() {
                    print_stdout(text);
                }
                Exit::Ok.code()
            }
            // The bound is told by the TYPE the failure carries and never by its
            // wording: the recogniser reports that bound as this one error alone.
            Err(e) if e.downcast_ref::<InferenceTimedOut>().is_some() => {
                print_stderr(&format!(
                    "mahbot audio: the recording is longer than the local recogniser will take — \
                     it reached the engine's own {}-minute inference bound, and nothing was \
                     transcribed rather than a transcript with a part of it missing",
                    local_transcriber::INFERENCE_TIMEOUT.as_secs() / 60,
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
        if let Some(code) = prepare() {
            return code;
        }
        // What the caller asked for, before anything is fetched.
        if request.text.trim().is_empty() {
            return misuse("the text to speak says nothing — it is empty or only whitespace");
        }
        let Some(voice) = resolve_voice(request.voice.as_deref()) else {
            return misuse(&format!(
                "the speech engine has no voice '{}' — its voices are {}",
                request.voice.as_deref().unwrap_or_default(),
                listed(tts::ALL_VOICE_STYLE_NAMES),
            ));
        };
        let Some(language) = resolve_language(request.language.as_deref()) else {
            return misuse(&format!(
                "the speech engine has no language '{}' — it takes {}",
                request.language.as_deref().unwrap_or_default(),
                listed(tts::TTS_LANGUAGES),
            ));
        };
        let destination = match resolve_destination(&request.output) {
            Ok(destination) => destination,
            Err(reason) => return misuse(&reason),
        };

        // The set is obtained and the voice that was asked for is loaded with it,
        // in one step: what comes back is what the reading speaks with, and the
        // style file is parsed once.
        let voice = match tts::ensure_ready_once(&voice, |dir| {
            announce_fetch(dir, "speech synthesis");
        })
        .await
        {
            Ok(voice) => voice,
            Err(e) => {
                print_stderr(&format!(
                    "mahbot audio: the speech synthesis model cannot be obtained: {e:#}"
                ));
                return Exit::Environment.code();
            }
        };

        // The whole reading is rendered before anything is written, so a text
        // that cannot be spoken in full leaves nothing at the destination.
        let wav = match tts::render_speech_wav(&request.text, voice, &language).await {
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

    /// Say on standard error that a model set is being fetched.
    ///
    /// The product's own daemon logs this and shows it in its window; a command
    /// line has only its standard error. It is said as a fetch begins — before
    /// the bytes of the first file that has to be fetched — and only when a fetch
    /// begins at all, so a set already on disk is silent while a first use that
    /// downloads gigabytes is not.
    fn announce_fetch(dir: &Path, what: &str) {
        print_stderr(&format!(
            "mahbot audio: fetching the {what} model into {} — the product's own copy of it, \
             fetched once",
            dir.display()
        ));
    }

    /// The first thing either operation does: point the product's own data folder
    /// at the caller's home, the way the product resolves it at boot
    /// ([`crate::config::default_config_dir`] reads `HOME`), because that is where
    /// the speech models live.
    ///
    /// This is the ONE setting-side fact this command needs: no store is opened
    /// and no config field is read. The root is process-global and immutable once
    /// set, so the value resolved here is the only one that will ever be used —
    /// which is also why it is set only when nothing has set it, since
    /// `set_storage_root` panics on a second call.
    ///
    /// `Some` is the exit class to return immediately, with the reason already
    /// told on standard error.
    fn prepare() -> Option<i32> {
        match crate::config::default_config_dir() {
            Ok(root) => {
                if crate::config::CONFIG.try_storage_root().is_none() {
                    crate::config::CONFIG.set_storage_root(root);
                }
                None
            }
            Err(e) => {
                print_stderr(&format!(
                    "mahbot audio: the product's data folder cannot be resolved: {e:#}"
                ));
                Some(Exit::Environment.code())
            }
        }
    }

    /// The voice style file the caller's `--voice` names, or `None` when the
    /// engine has no such voice.
    ///
    /// A voice is spelled the way the product spells it: the style file's own
    /// name, as the settings list them (`M1.json`). One `.json` may be left off
    /// and the case does not matter, so `M1`, `m1` and `M1.json` all name the same
    /// voice — but nothing else is accepted, because a voice the engine does not
    /// have must be refused rather than quietly replaced by another. An unnamed
    /// voice is the product's own, never a setting of its.
    pub(super) fn resolve_voice(named: Option<&str>) -> Option<String> {
        let Some(named) = named else {
            return Some(tts::DEFAULT_VOICE_NAME.to_string());
        };
        // Lower case first, then one `.json` off: what is left of `M1.JSON` is
        // `m1`, and the style files are read the same way.
        let asked = named.trim().to_ascii_lowercase();
        let asked = bare_name(&asked);
        tts::ALL_VOICE_STYLE_NAMES
            .iter()
            .find(|style| bare_name(style).eq_ignore_ascii_case(asked))
            .map(|style| (*style).to_string())
    }

    /// A style file's name without its extension: the way both a caller and the
    /// engine's own list spell the same voice (`M1.json` and `M1` are one voice).
    fn bare_name(style: &str) -> &str {
        style.strip_suffix(".json").unwrap_or(style)
    }

    /// The language code the caller's `--language` names, or `None` when the
    /// engine does not know it.
    ///
    /// The engine's own codes are the product's spelling of a language (the
    /// read-aloud setting's values), so a code is matched against them as it
    /// stands; an unnamed language is the engine's language-agnostic default
    /// ([`crate::config::DEFAULT_TTS_LANGUAGE`]), never the setting's value.
    pub(super) fn resolve_language(named: Option<&str>) -> Option<String> {
        let Some(named) = named else {
            return Some(crate::config::DEFAULT_TTS_LANGUAGE.to_string());
        };
        let asked = named.trim();
        tts::TTS_LANGUAGES
            .iter()
            .find(|code| code.eq_ignore_ascii_case(asked))
            .map(|code| (*code).to_string())
    }

    /// The refusal for an input that cannot be read at all — said the same way
    /// whether the check that found it out was the type probe or the open.
    fn cannot_read(path: &Path, e: &std::io::Error) -> String {
        format!("'{}' cannot be read: {e}", path.display())
    }

    /// Where the spoken file goes: the path the caller named, or a fresh name
    /// this command picks inside a directory he named.
    pub(super) enum Destination {
        /// The caller named the file itself.
        Named(PathBuf),
        /// The caller named a directory; the name is chosen when the file is
        /// created, so it can never collide with what is already there.
        Chosen(PathBuf),
    }

    /// Read what `--output` names, before anything is fetched or rendered.
    ///
    /// The name is also where the caller says which container he wants, and
    /// uncompressed WAV is the only one the product writes. The destination
    /// directory is the caller's to make: this command never creates one.
    pub(super) fn resolve_destination(output: &Path) -> Result<Destination, String> {
        if output.is_dir() {
            return Ok(Destination::Chosen(output.to_path_buf()));
        }
        // A name with an extension names a container, and the name is where the
        // caller says which one: uncompressed WAV is the only one the product
        // writes. Absence is not a reason here — every new destination is absent
        // — so a container this command cannot write is refused for what it is,
        // whether or not something of that name is already there.
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
        // Nothing after a dot and nothing at that path: the caller named the
        // directory he expected to find, and this command creates none.
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
            // Why it cannot be used is what the caller needs to hear: a parent
            // that is there and is not a directory is not a missing one.
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

    /// The refusal for a destination that is already taken, said the same way
    /// whether the name was checked before the work or lost the race at the
    /// create.
    fn already_exists(path: &Path) -> String {
        format!(
            "{} already exists — this command never overwrites a file",
            path.display()
        )
    }

    /// The refusal for audio that could not be written, said the same way whether
    /// the caller named the file or this command chose the name.
    fn cannot_write(path: &Path, e: &std::io::Error) -> String {
        format!("{} could not be written: {e}", path.display())
    }

    /// Write the rendered audio where the destination points, returning the path
    /// when this command picked the name.
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
                        // The name was taken in the moment between choosing and
                        // creating: another name is one call away.
                        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                        Err(e) => return Err(cannot_write(&path, &e)),
                    }
                }
                Err(format!("no free name could be picked in {}", dir.display()))
            }
        }
    }

    /// Create `path` and write the audio into it.
    ///
    /// `create_new` is what makes "never overwritten" true even against a file
    /// that appeared a moment ago, and a write that fails removes what it
    /// created — a failure leaves nothing at the destination.
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

    /// A list of names as a sentence reads them: `a, b or c`, and just `a` for
    /// one.
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

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The words a caller reads before he hears anything: what a flag tail means,
    /// and what one that means nothing says.
    #[test]
    fn flag_tails() {
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
            values(&["--text=hi", "--voice=M1"]).unwrap(),
            vec![("text", "hi".to_string()), ("voice", "M1".to_string())]
        );
        assert!(values(&["--nope", "x"]).is_err_and(|r| r.contains("--nope")));
        assert!(
            values(&["--text", "a", "--text", "b"]).is_err_and(|r| r.contains("more than once"))
        );
        assert!(values(&["--text"]).is_err_and(|r| r.contains("needs a value")));
        assert!(values(&["text"]).is_err_and(|r| r.contains("--flag")));
        assert_eq!(values(&["--text", "hi", "-h"]).unwrap_err(), "help");
    }

    /// A required flag that was left out is refused by name, and it is the
    /// operation's own list that says which those are.
    #[test]
    fn required_flags_are_named() {
        let invocation = |tokens: &[&str]| parse(&owned(tokens));
        assert!(invocation(&["speak", "--text", "hi"]).is_err_and(|r| r.contains("--output")));
        assert!(invocation(&["transcribe"]).is_err_and(|r| r.contains("--input")));
        assert!(invocation(&["frobnicate"]).is_err_and(|r| r.contains("unknown operation")));
        assert!(invocation(&[]).is_err(), "no operation is misuse");
    }

    fn owned(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| (*token).to_string()).collect()
    }

    /// The engine's own vocabulary: the names it takes, read the way a caller may
    /// spell them, and nothing else.
    #[cfg(target_os = "macos")]
    #[test]
    fn voice_and_language_names() {
        use super::engine::{resolve_language, resolve_voice};

        assert_eq!(resolve_voice(None), Some("M1.json".to_string()));
        for asked in ["M1", "m1", "M1.json", "m1.JSON"] {
            assert_eq!(
                resolve_voice(Some(asked)),
                Some("M1.json".to_string()),
                "{asked} names the product's M1.json"
            );
        }
        assert_eq!(resolve_voice(Some("F5.json")), Some("F5.json".to_string()));
        assert_eq!(resolve_voice(Some("M6")), None);
        assert_eq!(resolve_voice(Some("")), None);
        assert_eq!(resolve_voice(Some("M1.json.json")), None);

        assert_eq!(resolve_language(None), Some("na".to_string()));
        assert_eq!(resolve_language(Some("EN")), Some("en".to_string()));
        assert_eq!(resolve_language(Some(" na ")), Some("na".to_string()));
        assert_eq!(resolve_language(Some("xx")), None);
    }

    /// Where the file may go: a name this command can write, or a directory it
    /// picks one in — and every refusal the caller can act on.
    #[cfg(target_os = "macos")]
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
        // A container this command cannot write is refused for THAT, whether or
        // not something of the name is already there: every new destination is
        // absent, so absence can never be the reason for it.
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
        // A name with nothing after a dot and nothing at that path is the
        // directory the caller expected to find — refused for that, and told
        // which names do work.
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
        // A parent that is there and is not a directory is refused for what it
        // is, never as a missing directory.
        assert!(
            resolve_destination(&existing.join("out.wav")).is_err_and(|r| {
                r.contains("is not a directory") && !r.contains("does not exist")
            })
        );
    }
}

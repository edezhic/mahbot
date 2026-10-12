//! The product's flag grammar, in one place.
//!
//! The command lines behind [`crate::cli`] (the `config` and `workspace`
//! families) and `mahbot audio` read a flag tail the same way: a token is
//! `--<flag>`, `--<flag>=<value>` or `--<flag> <value>`, and a value is the very
//! next token taken VERBATIM — so a value may itself look like a flag, and a
//! value mistyped as a positional is never echoed back out. `-h`/`--help` asks
//! for help where a FLAG could stand and never where a VALUE does. The refusals
//! a token can meet are worded here too, so the frontends that share this module
//! cannot drift apart in what they say about the same mistake.
//!
//! It is not every command line the product has. The browser-automation command
//! ([`crate::chrome::cli`]) and the OpenRouter benchmark read their own flags and
//! word their own refusals, and are left as they are.

/// Whether a token asks for help (`-h`/`--help`).
pub(crate) fn is_help(token: &str) -> bool {
    token == "-h" || token == "--help"
}

/// One token of a flag tail, read.
pub(crate) enum Token<'a> {
    /// `-h`/`--help`.
    Help,
    /// `--<name>`, carrying the inline value of `--<name>=<value>` when it has one.
    Flag {
        name: &'a str,
        inline: Option<&'a str>,
    },
}

/// Read one token of a flag tail.
///
/// `Err` is a token that is no flag at all — refused WITHOUT being echoed, so a
/// value the caller forgot to give a flag to can never travel back out.
pub(crate) fn read_token(token: &str) -> Result<Token<'_>, String> {
    if is_help(token) {
        return Ok(Token::Help);
    }
    let Some(spelled) = token.strip_prefix("--") else {
        return Err("unexpected argument where a '--flag' was expected".to_string());
    };
    Ok(match spelled.split_once('=') {
        Some((name, inline)) => Token::Flag {
            name,
            inline: Some(inline),
        },
        None => Token::Flag {
            name: spelled,
            inline: None,
        },
    })
}

/// A flag this operation does not take.
pub(crate) fn unknown_flag(name: &str, operation: &str) -> String {
    format!("unknown flag '--{name}' for '{operation}'")
}

/// A flag that was given a second time.
pub(crate) fn repeated_flag(name: &str) -> String {
    format!("flag '--{name}' was given more than once")
}

/// A flag standing without the value it takes — the form is spelled out, so the
/// caller is told what to write and not only what was wrong.
pub(crate) fn flag_value_missing(name: &str) -> String {
    format!("flag '--{name}' needs a value: '--{name} <value>'")
}

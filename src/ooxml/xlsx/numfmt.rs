//! Excel number formats for `.xlsx` cells: a stored numeric value rendered the
//! way Excel would display it, or `None` when this reader cannot reproduce the
//! format.
//!
//! # Invariants
//!
//! - **Never panics.** Nothing a workbook can carry trips it: the format code
//!   ([`MAX_FORMAT_LENGTH`]), the placeholders a section names
//!   ([`MAX_PLACEHOLDERS`]), the fractional-second digits it names
//!   ([`MAX_SUB_SECOND_DIGITS`]), the scaled integer its display may need
//!   ([`MAX_SCALED`]) and the serial domain of a value
//!   ([`VALUE_LIMIT`]/[`SERIAL_LIMIT`]) are all bounded before they are used.
//! - **Never silently wrong.** A format this reader does not implement, a value
//!   outside the format's own domain, a scaled value with no exact display, a
//!   format no section of which serves the value, a section that shows nothing, a
//!   section holding only the format's own text that would have to carry the
//!   value's own sign, and a scientific section naming a dot outside its
//!   placeholders or anywhere in its exponent region, an exponent placeholder
//!   outside that region's own run, or a zero mantissa its own width would widen
//!   all come back as `None` — one outcome, which the caller turns into the value
//!   the cell stores, marked as stored — never an empty or invented rendering.
//! - **The shown value is the one Excel displays.** Excel works with the 15
//!   significant decimal digits of a number, so `1.005` under `0.00` shows as
//!   `1.01` even though the binary double holding it is a hair below the tie;
//!   [`scaled_magnitude`] performs that canonicalisation and the
//!   half-away-from-zero rounding after it.
//! - **A section serves a value by Excel's own rule.** A format whose sections
//!   carry a comparison is a conditional format, not one ordered by sign: the
//!   conditions of its first two sections are tested in order and the one that holds
//!   serves the value, through a section that shows the magnitude when the condition
//!   is Excel's own way of naming the negative side (`[<0]0.0;0.00` shows -1 as
//!   `1.0`) and lets the value keep its own minus otherwise (`[>=1]0;0` shows -2 as
//!   `-2`). A value no condition serves falls to the section Excel keeps for what
//!   none of them takes — the second of a two-section format, the third of a longer
//!   one — where the minus is dropped for a format that has a section of its own for
//!   the negative side, and only for that: one of three or more sections, or one
//!   whose own condition covers every non-negative number (`[>=-1]0;0` shows -2 as
//!   `2`, and `[>=1]0;0;0.00` shows it as `2` too). Otherwise the
//!   sections stand for positive;negative;zero;text, the negative one carrying its own
//!   sign, and a section standing alone shows the minus for a value that did not round
//!   away to nothing. A value no section serves — which only a conditional format
//!   can have — is declined, so what the cell stores stands for it, marked as stored.
//! - **A code is read as Excel reads it, up to what a line of text can carry.**
//!   The digits go in the places the section puts them, with the literals the
//!   section stands *between* them (`000-00-0000`), the symbol of a currency
//!   directive and every `%` are printed where they stand, and what shows nothing
//!   of its own — a colour, a condition, a bare locale, a `*` fill out to the
//!   cell's width — leaves nothing behind. What is left over — a fraction, a text
//!   placeholder, a localized format word this reader does not know, a decimal point
//!   standing anywhere but between the placeholders, a digit the layout has no place
//!   for or a place the digits do not fill — is declined rather than guessed at.
//!   A section naming a fractional part keeps the decimal point even when the
//!   value shows no digit behind it (`#.#####` shows 1 as `1.`), the point
//!   standing where the section puts it.

use std::cmp::Ordering;

/// Excel's own limit on a format code's length; a longer one is not a format the
/// application could store, so it is declined rather than parsed.
const MAX_FORMAT_LENGTH: usize = 255;
/// The placeholders a section may name before it stops being one this reader
/// reproduces: a code can hold hundreds of them, and every one costs at most one
/// digit of a rendered number.
const MAX_PLACEHOLDERS: usize = 30;
/// The fractional-second digits a time section may name (Excel writes at most
/// three, a millisecond).
const MAX_SUB_SECOND_DIGITS: usize = 9;
/// The largest serial a value may be for a time of day or a duration — far past
/// anything Excel's own display reaches, but still a value whose days, seconds
/// and hours have an exact integer form.
const VALUE_LIMIT: f64 = 1e9;
/// The sections a format may carry: one per sign and one for text, Excel's own
/// limit.
const MAX_SECTIONS: usize = 4;
/// The serial of 9999-12-31 in the 1900 system: the last day Excel's calendar
/// holds, and the last one this reader renders a date for there.
const SERIAL_LIMIT: i64 = 2_958_465;
/// The same last day in the 1904 system, whose day 0 is four years later: a
/// serial is that much shorter for the same calendar date.
const SERIAL_LIMIT_1904: i64 = SERIAL_LIMIT - 1_462;
/// The largest scaled integer a number's display may need, so that a value
/// without any plausible display is declined rather than rendered into a
/// three-hundred-digit line.
const MAX_SCALED: i128 = 10_i128.pow(30);

/// The built-in number-format code for `id`, or `None` when the id has no
/// ECMA-376/POI builtin meaning.
const fn builtin(id: u32) -> Option<&'static str> {
    Some(match id {
        0 => "General",
        1 => "0",
        2 => "0.00",
        3 => "#,##0",
        4 => "#,##0.00",
        5 => "\"$\"#,##0_);(\"$\"#,##0)",
        6 => "\"$\"#,##0_);[Red](\"$\"#,##0)",
        7 => "\"$\"#,##0.00_);(\"$\"#,##0.00)",
        8 => "\"$\"#,##0.00_);[Red](\"$\"#,##0.00)",
        9 => "0%",
        10 => "0.00%",
        11 => "0.00E+00",
        12 => "# ?/?",
        13 => "# ??/??",
        14 => "mm-dd-yy",
        15 => "d-mmm-yy",
        16 => "d-mmm",
        17 => "mmm-yy",
        18 => "h:mm AM/PM",
        19 => "h:mm:ss AM/PM",
        20 => "h:mm",
        21 => "h:mm:ss",
        22 => "m/d/yy h:mm",
        37 => "#,##0_);(#,##0)",
        38 => "#,##0_);[Red](#,##0)",
        39 => "#,##0.00_);(#,##0.00)",
        40 => "#,##0.00_);[Red](#,##0.00)",
        41 => "_(* #,##0_);_(* (#,##0);_(* \"-\"_);_(@_)",
        42 => "_(\"$\"* #,##0_);_(\"$\"* (#,##0);_(\"$\"* \"-\"_);_(@_)",
        43 => "_(* #,##0.00_);_(* (#,##0.00);_(* \"-\"??_);_(@_)",
        44 => "_(\"$\"* #,##0.00_);_(\"$\"* (#,##0.00);_(\"$\"* \"-\"??_);_(@_)",
        45 => "mm:ss",
        46 => "[h]:mm:ss",
        47 => "mm:ss.0",
        48 => "##0.0E+0",
        49 => "@",
        _ => return None,
    })
}

/// Whether a cell shows its stored text: the format its `id` and declared `code`
/// resolve to is `General`, whose display is the value's own stored text.
#[must_use]
pub(crate) fn is_general(id: u32, code: Option<&str>) -> bool {
    match code {
        // A declared code that is empty names no format this reader can show, so
        // the caller falls through to the stored value *marked* as stored rather
        // than passed off as the display.
        Some("") => false,
        Some(code) => is_general_code(code),
        None => builtin(id).is_some_and(is_general_code),
    }
}

/// Whether a format code is `General` under the directives a workbook wraps it in
/// (`[DBNum1][$-804]General`), the padding it may carry (` General `) and the
/// repetition across sections (`General;General`): every one of those shows the
/// stored text and nothing else.
fn is_general_code(code: &str) -> bool {
    split_sections(code).map_or_else(
        || general_section(code),
        |sections| sections.iter().all(|section| general_section(section)),
    )
}

/// Whether a section's body is `General`. The body does not depend on the value
/// its condition is read against.
fn general_section(section: &str) -> bool {
    split_section(section, 0.0)
        .0
        .trim()
        .eq_ignore_ascii_case("General")
}

/// `value` as the number format `id` displays it, or `None` when this reader has
/// no display to show for it — a format it cannot parse or does not reproduce, a
/// value outside the format's own domain, a format no section of which serves it,
/// a section that shows nothing and a section that would have to carry the value's
/// own sign. The caller shows the value the cell stores, marked as stored, in that
/// case. `General` is never asked here — it names no placeholder and no text of its
/// own, its display being the value's own stored text, whose width depends on the
/// cell, so the caller shows that text itself ([`is_general`]).
#[must_use]
pub(crate) fn display(value: f64, id: u32, code: Option<&str>, date1904: bool) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    // A declared non-empty code always wins over the built-in meaning of `id`;
    // an empty declared code is a format this reader does not reproduce.
    let format = match code {
        Some("") => return None,
        Some(code) => code,
        None => builtin(id)?,
    };
    if format.len() > MAX_FORMAT_LENGTH {
        return None;
    }
    let sections = split_sections(format)?;
    let Selection { body, signed } = select_section(&sections, value)?;
    if body.trim().is_empty() {
        return None;
    }
    let tokens = tokenize(body);
    let rendered = if is_date(&tokens) {
        render_date(value, date1904, &tokens)?
    } else {
        let Rendered { text, shows_number } = render_number(value.abs(), &tokens)?;
        // A section naming no digit placeholder at all shows text of the format's
        // own and nothing of the value. For a negative value whose sign is this
        // reader's to write, that sign has no place in the display — Excel writes
        // its own minus into some such sections and not others — so the format is
        // declined rather than shown with the sign dropped.
        if signed && value < 0.0 && !tokens.iter().any(|tok| matches!(tok, Tok::Digit(_))) {
            return None;
        }
        // The minus is the reader's to write only where the section writes none of
        // its own (`-$0.00`), and only for a value that did not round away to
        // nothing: Excel shows `0` for -0.4 under `0`, never `-0`.
        if signed && value < 0.0 && shows_number && !text.starts_with('-') {
            format!("-{text}")
        } else {
            text
        }
    };
    let rendered = rendered.trim();
    if rendered.is_empty() {
        return None;
    }
    Some(rendered.to_owned())
}

// ── Sections ────────────────────────────────────────────────────

/// The format's sections, split on top-level `;`. `None` when the format has
/// more than four — the most Excel defines, one per sign and one for text — and
/// a `;` inside `"..."`, escaped by `\`, or inside `[...]` never splits.
fn split_sections(format: &str) -> Option<Vec<&str>> {
    let bytes = format.as_bytes();
    let mut sections = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index += 1;
                while index < bytes.len() && bytes[index] != b'"' {
                    index += 1;
                }
                index += 1;
            }
            b'\\' => index += 2,
            b'[' => {
                index += 1;
                while index < bytes.len() && bytes[index] != b']' {
                    index += 1;
                }
                index += 1;
            }
            b';' => {
                sections.push(&format[start..index]);
                if sections.len() == MAX_SECTIONS {
                    return None;
                }
                index += 1;
                start = index;
            }
            _ => index += 1,
        }
    }
    sections.push(&format[start..]);
    Some(sections)
}

/// The section `value` renders with, and whether the minus it shows is the
/// reader's to write. The rule that picks it, and that decides this, is the one
/// the module's own sections states.
struct Selection<'a> {
    body: &'a str,
    /// Whether a negative value keeps its own minus: a section standing alone for
    /// every value shows it, and a conditional format shows it unless its own
    /// condition serves the value through a section written to display a negative
    /// side. A format ordered by sign has a negative section written to carry its own
    /// sign.
    signed: bool,
}

/// The section `value` renders with: for a conditional format the one Excel's own
/// condition rule serves ([`conditional_section`]), otherwise the one its sign
/// names. `None` when no section serves the value at all, which the caller shows
/// as the stored value marked as stored.
fn select_section<'a>(sections: &[&'a str], value: f64) -> Option<Selection<'a>> {
    // One walk reads each section once, for both questions: whether the format
    // compares at all, and which section serves this value.
    let mut split = [("", None); MAX_SECTIONS];
    for (slot, section) in split.iter_mut().zip(sections) {
        *slot = split_section(section, value);
    }
    let split = &split[..sections.len()];
    // Only the format's first two sections can carry a condition; one standing
    // later is no part of Excel's rule, so such a format is read by sign as an
    // unconditional one is.
    if split
        .iter()
        .take(2)
        .any(|(_, condition)| condition.is_some())
    {
        return conditional_section(split, value);
    }
    let position = sign_position(split.len(), value)?;
    Some(Selection {
        body: split[position].0,
        // One section stands for every value, so a negative one has no section of
        // its own to carry the minus.
        signed: split.len() == 1,
    })
}

/// The section a *conditional* format serves `value` with, and whether its minus
/// is the reader's to write. The rule is the one this module's invariants state;
/// what is worked out here is its order — a first section stating no condition,
/// then the conditions of the first two sections, then the section Excel keeps for
/// what none of them takes. `None` where no section serves the value at all.
fn conditional_section<'a>(
    split: &[(&'a str, Option<Condition>)],
    value: f64,
) -> Option<Selection<'a>> {
    // The sections Excel's rule works on: a format of one section draws every value
    // through it, and a two-section one repeats its first for a zero value.
    let at = |index: usize| split.get(index).copied().unwrap_or(split[0]);
    let (first, second) = (at(0).1, at(1).1);
    if first.is_none() && value > 0.0 {
        return Some(Selection {
            body: at(0).0,
            signed: false,
        });
    }
    if let Some(held) = first.filter(|it| it.held) {
        return Some(Selection {
            body: at(0).0,
            signed: !held.names_the_negative(),
        });
    }
    if let Some(held) = second.filter(|it| it.held) {
        return Some(Selection {
            body: at(1).0,
            signed: !held.names_the_negative(),
        });
    }
    let (index, signed) = if second.is_none() && value < 0.0 {
        // The second section states no condition, so it is the format's own negative
        // side. A format of three or more sections carries a section for that side
        // written to show its own minus, and a zero boundary named from above
        // ([`Condition::covers_the_positive`]) leaves a two-section one no positive
        // side for the minus to belong to: either way the value goes through the
        // second as its magnitude (`[>=1]0;0;0.00` shows -2 as `2`, and `[>=-1]0;0`
        // as `2` too). A two-section format with a condition that does not cover the
        // positive side has no such section — its second stands for the negative side
        // the value did not fall into — so the value keeps its own minus
        // (`[>=1]0;0` shows -2 as `-2`).
        (
            1,
            !(split.len() >= 3 || first.is_some_and(Condition::covers_the_positive)),
        )
    } else if split.len() >= 3 {
        // A longer format keeps its third section for the value none of its
        // conditions takes, and that section shows the value's own minus.
        (2, true)
    } else if value < 0.0 || split.len() == 1 {
        // A value no section serves — and a format whose only section is gated by a
        // condition — is declined rather than shown through a section whose condition
        // does not hold.
        return None;
    } else {
        // A positive value every condition of a two-section format let past: it goes
        // through the second section when the two do not compare — that one is the
        // format's negative side, the only section left for a value the first did not
        // take — and through the first when both compare, which is the place a
        // two-section format repeats its first for a zero value.
        (usize::from(!(first.is_some() && second.is_some())), false)
    };
    Some(Selection {
        body: at(index).0,
        signed,
    })
}

/// The section of `count` a value's sign names: one section serves every value,
/// two pick the second for a negative value, and three or four pick positive,
/// negative and zero in order.
fn sign_position(count: usize, value: f64) -> Option<usize> {
    match count {
        1 => Some(0),
        2 => Some(usize::from(value < 0.0)),
        3 | 4 => Some(if value > 0.0 {
            0
        } else if value < 0.0 {
            1
        } else {
            2
        }),
        _ => None,
    }
}

/// A section split into the text it renders and the condition it is gated on
/// (`[>1000]0.0;[<=1000]0.00`), already evaluated for `value`. Every bracket of
/// the run a section leads with is read, wherever it stands in that run: a
/// comparison is evaluated (`[Red][<100]0.0` keeps its colour and its condition,
/// and one standing after a currency directive — `[$$-409][>100]0.00` — gates the
/// section too), a directive that prints nothing of its own leaves the body, and a
/// bracket that *is* a format element (`[h]`, `[$€-407]`) begins the body — the
/// tokenizer then drops from it whatever the walk has already read (`[>100]`), so
/// the display keeps the symbol and the condition that chose it both.
fn split_section(section: &str, value: f64) -> (&str, Option<Condition>) {
    let mut rest = section;
    let mut condition = None;
    let mut body = None;
    while let Some(inner) = rest.trim_start().strip_prefix('[') {
        let Some((bracket, after)) = inner.split_once(']') else {
            break;
        };
        if body.is_none()
            && (elapsed_unit(bracket).is_some() || !currency_symbol(bracket).is_empty())
        {
            body = Some(rest.trim_start());
        }
        if condition.is_none()
            && let Some(read) = condition_of(bracket, value)
        {
            condition = Some(read);
        }
        rest = after;
    }
    (body.unwrap_or(rest), condition)
}

/// The operators a numbered condition may carry, longest first so a two-character
/// operator is never read as one of its halves.
const CONDITIONS: [&str; 6] = ["<=", ">=", "<>", "<", ">", "="];

/// One of a section's own conditions: how Excel tests a value against it, and
/// whether the test passed.
#[derive(Clone, Copy)]
struct Condition {
    operator: &'static str,
    threshold: f64,
    held: bool,
}

impl Condition {
    /// Whether the condition is Excel's way of naming the negative side — `[<0]`,
    /// `[<=-1]`, `[=-3]`. Such a section is written to display a negative value, so
    /// Excel shows the value's magnitude through it and writes no minus of its own.
    fn names_the_negative(self) -> bool {
        match self.operator {
            // The negative side named from below includes its own zero boundary;
            // named from above, a zero boundary still names a positive range.
            "=" | "<=" => self.threshold < 0.0,
            "<" => self.threshold <= 0.0,
            _ => false,
        }
    }

    /// Whether the condition holds for every non-negative number — `[>-1]`,
    /// `[>=0]`. A format conditioned that way has no positive section of its own, so
    /// a value the condition does not serve keeps no minus Excel would have written
    /// through one.
    fn covers_the_positive(self) -> bool {
        match self.operator {
            ">" => self.threshold < 0.0,
            ">=" => self.threshold <= 0.0,
            _ => false,
        }
    }
}

/// The condition a bracketed run is, evaluated for `value` — `None` when the run
/// is not a comparison at all (a colour, a locale), which is what tells a section
/// the condition gates apart from one the value's sign selects.
///
/// A condition whose operand is not a number is one no value satisfies: the value
/// falls to another section or is declined and marked as stored, never shown
/// through a condition that does not hold.
fn condition_of(bracket: &str, value: f64) -> Option<Condition> {
    let (operator, operand) = CONDITIONS
        .iter()
        .find_map(|operator| bracket.strip_prefix(operator).map(|rest| (*operator, rest)))?;
    // An operand that is not a number is a condition no value satisfies, and one no
    // reading of it as a negative or a positive side either.
    let Some(threshold) = operand.trim().parse::<f64>().ok() else {
        return Some(Condition {
            operator,
            threshold: f64::NAN,
            held: false,
        });
    };
    // The comparison is Excel's own: the serial the format names is compared with
    // the same exactness the stored value carries.
    let ordering = value.partial_cmp(&threshold);
    let held = match operator {
        "<" => ordering == Some(Ordering::Less),
        ">" => ordering == Some(Ordering::Greater),
        "<=" => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        ">=" => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
        "<>" => ordering != Some(Ordering::Equal),
        _ => ordering == Some(Ordering::Equal),
    };
    Some(Condition {
        operator,
        threshold,
        held,
    })
}

// ── Tokens ──────────────────────────────────────────────────────

/// A date/time unit. `m` is emitted as `Month` and reclassified as a minute by
/// its neighbours; `Minute` itself is only ever produced by an elapsed `[m]`.
#[derive(Clone, Copy)]
enum Unit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    AmPm,
}

/// One meaningful token of a section, with the literal text a number format
/// prints around its placeholders.
#[derive(Clone, Copy)]
enum Tok {
    /// A literal character the format quotes or escapes, so its author asked for
    /// it to be printed.
    Lit(char),
    /// A character no rule claimed: Excel prints it too, but in a section with no
    /// placeholder at all it is as likely to be a format word this reader does not
    /// know (a localized date pattern) as text the author meant to show.
    Raw(char),
    /// `0`, `#` or `?`.
    Digit(char),
    Dot,
    Comma,
    Percent,
    Slash,
    /// The unquoted `@` text placeholder. A cell holding a number has no text
    /// for it to stand for, so its section is one this reader declines.
    Text,
    /// A bracketed directive that is not part of a date and prints a symbol into
    /// the run of placeholders around it — `[$€-407]`. Anything else bracketed
    /// (`[Red]`, `[>100]`, a bare locale) contributes nothing to a number.
    Directive(char),
    /// A `*` jump: Excel pads the cell out to its width with the character that
    /// follows, which a reader cannot reproduce inside a line of text.
    Star,
    /// `E+`/`E-`/`e+`/`e-`: the sign that follows the exponent marker.
    ExpSign(char),
    /// A date/time unit.
    Date(Unit),
    /// An elapsed `[h]`/`[m]`/`[s]`.
    Elapsed(Unit),
}

/// Walk a section into tokens, dropping everything that carries no meaning:
/// quoted and escaped text is literal, `_x`/`*x` emit nothing, and a bracketed
/// run emits nothing unless it is an elapsed unit or a currency directive
/// (`[$€-407]`, and the symbol it names is printed). Conditions and colours
/// carry no text, and a condition is read where the sections are selected.
fn tokenize(section: &str) -> Vec<Tok> {
    let chars: Vec<char> = section.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let current = chars[index];
        match current {
            '"' => {
                index += 1;
                while index < chars.len() && chars[index] != '"' {
                    tokens.push(Tok::Lit(chars[index]));
                    index += 1;
                }
                index += 1;
            }
            '\\' => {
                if let Some(&escaped) = chars.get(index + 1) {
                    tokens.push(Tok::Lit(escaped));
                }
                index += 2;
            }
            '_' => index += 2,
            '*' => {
                tokens.push(Tok::Star);
                index += 2;
            }
            '[' => {
                let close = chars[index + 1..]
                    .iter()
                    .position(|&c| c == ']')
                    .map_or(chars.len(), |offset| index + 1 + offset);
                let inner: String = chars[index + 1..close].iter().collect();
                if let Some(unit) = elapsed_unit(&inner) {
                    tokens.push(Tok::Elapsed(unit));
                } else {
                    for symbol in currency_symbol(&inner).chars() {
                        tokens.push(Tok::Directive(symbol));
                    }
                }
                index = close + 1;
            }
            '0' | '#' | '?' => {
                tokens.push(Tok::Digit(current));
                index += 1;
            }
            '.' => {
                tokens.push(Tok::Dot);
                index += 1;
            }
            ',' => {
                tokens.push(Tok::Comma);
                index += 1;
            }
            '%' => {
                tokens.push(Tok::Percent);
                index += 1;
            }
            '/' => {
                tokens.push(Tok::Slash);
                index += 1;
            }
            '@' => {
                tokens.push(Tok::Text);
                index += 1;
            }
            'y' | 'Y' => push_date_run(&chars, &mut index, Unit::Year, &mut tokens),
            'd' | 'D' => push_date_run(&chars, &mut index, Unit::Day, &mut tokens),
            'h' | 'H' => push_date_run(&chars, &mut index, Unit::Hour, &mut tokens),
            's' | 'S' => push_date_run(&chars, &mut index, Unit::Second, &mut tokens),
            'm' | 'M' => push_date_run(&chars, &mut index, Unit::Month, &mut tokens),
            'a' | 'A' if starts_with_ignore_ascii_case(&chars, index, "am/pm") => {
                tokens.push(Tok::Date(Unit::AmPm));
                index += 5;
            }
            'a' | 'A' if starts_with_ignore_ascii_case(&chars, index, "a/p") => {
                tokens.push(Tok::Date(Unit::AmPm));
                index += 3;
            }
            'e' | 'E' if matches!(chars.get(index + 1), Some('+' | '-')) => {
                tokens.push(Tok::ExpSign(chars[index + 1]));
                index += 2;
            }
            other => {
                tokens.push(Tok::Raw(other));
                index += 1;
            }
        }
    }
    tokens
}

/// A run of one date letter (`y`, `d`, `h`, `s` or `m`, any case, repeated) as
/// one token.
fn push_date_run(chars: &[char], index: &mut usize, unit: Unit, tokens: &mut Vec<Tok>) {
    let letter = chars[*index].to_ascii_lowercase();
    tokens.push(Tok::Date(unit));
    *index += 1;
    while *index < chars.len() && chars[*index].to_ascii_lowercase() == letter {
        *index += 1;
    }
}

/// Whether `chars` at `at` spell `needle`, ignoring ASCII case.
fn starts_with_ignore_ascii_case(chars: &[char], at: usize, needle: &str) -> bool {
    let needle: Vec<char> = needle.chars().collect();
    chars.len() >= at + needle.len()
        && chars[at..at + needle.len()]
            .iter()
            .zip(&needle)
            .all(|(current, expected)| current.to_ascii_lowercase() == *expected)
}

/// The elapsed unit a bracketed run names, or `None` for any other bracket
/// (`[Red]`, `[$-409]`, `[>=100]`, …).
fn elapsed_unit(inner: &str) -> Option<Unit> {
    match inner.to_ascii_lowercase().as_str() {
        "h" | "hh" => Some(Unit::Hour),
        "m" | "mm" => Some(Unit::Minute),
        "s" | "ss" => Some(Unit::Second),
        _ => None,
    }
}

/// The currency symbol a `[$symbol-locale]` directive names — `€` from
/// `[$€-407]`, `$` from `[$$-409]`, `USD` from `[$USD-409]` — or `""` when the
/// bracket names no symbol at all: a colour, a condition, or a directive
/// carrying the locale alone, which is both the hex LCID Excel writes
/// (`[$-409]`) and the BCP-47 tags a workbook also holds (`[$-en-US]`,
/// `[$-x-sysdate]`). The locale is dropped either way: the code's own separators
/// are what is printed, so the format stays locale-independent.
fn currency_symbol(inner: &str) -> &str {
    let Some(rest) = inner.strip_prefix('$') else {
        return "";
    };
    match rest.split_once('-') {
        Some((symbol, _)) if !symbol.is_empty() => symbol,
        Some(_) => "",
        None => rest,
    }
}

// ── Rendering ───────────────────────────────────────────────────

/// Whether any token makes the section a date/time format.
fn is_date(tokens: &[Tok]) -> bool {
    tokens
        .iter()
        .any(|tok| matches!(tok, Tok::Date(_) | Tok::Elapsed(_)))
}

/// Which date/time units a section names, and whether a minute-role `m`, a day,
/// a month or a year is present — the shape the date renderer works from.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each unit is an independent yes/no the renderer reads"
)]
struct DateShape {
    has_date: bool,
    has_time: bool,
    hour: bool,
    minute: bool,
    second: bool,
    elapsed_hour: bool,
    elapsed_minute: bool,
    elapsed_second: bool,
}

/// Resolve the section's date tokens into [`DateShape`].
fn date_shape(tokens: &[Tok]) -> DateShape {
    let mut shape = DateShape::default();
    for (index, tok) in tokens.iter().enumerate() {
        match tok {
            Tok::Date(Unit::Month) if minute_role(tokens, index) => {
                shape.has_time = true;
                shape.minute = true;
            }
            Tok::Date(Unit::Year | Unit::Month | Unit::Day) => shape.has_date = true,
            Tok::Date(Unit::Hour) => {
                shape.has_time = true;
                shape.hour = true;
            }
            Tok::Date(Unit::Second) => {
                shape.has_time = true;
                shape.second = true;
            }
            Tok::Date(Unit::AmPm) => shape.has_time = true,
            Tok::Elapsed(Unit::Hour) => {
                shape.has_time = true;
                shape.hour = true;
                shape.elapsed_hour = true;
            }
            Tok::Elapsed(Unit::Minute) => {
                shape.has_time = true;
                shape.minute = true;
                shape.elapsed_minute = true;
            }
            Tok::Elapsed(Unit::Second) => {
                shape.has_time = true;
                shape.second = true;
                shape.elapsed_second = true;
            }
            _ => {}
        }
    }
    shape
}

/// Whether the `m` at `index` is a minute: it follows an hour, or precedes a
/// second, else it is a month.
fn minute_role(tokens: &[Tok], index: usize) -> bool {
    let is_date_token = |tok: &&Tok| matches!(tok, Tok::Date(_) | Tok::Elapsed(_));
    let preceding = tokens[..index].iter().rev().find(is_date_token);
    if matches!(
        preceding,
        Some(Tok::Date(Unit::Hour) | Tok::Elapsed(Unit::Hour))
    ) {
        return true;
    }
    let following = tokens[index + 1..].iter().find(is_date_token);
    matches!(
        following,
        Some(Tok::Date(Unit::Second) | Tok::Elapsed(Unit::Second))
    )
}

/// Render a date/time section, or `None` for a value outside the domain a serial
/// has: a negative value or one past [`VALUE_LIMIT`] (Excel shows `#####`, this
/// reader declines to guess), and a day the calendar Excel's epochs define does
/// not hold.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a value bounded by VALUE_LIMIT has a day count that fits i64"
)]
fn render_date(value: f64, date1904: bool, tokens: &[Tok]) -> Option<String> {
    if !(0.0..=VALUE_LIMIT).contains(&value) {
        return None;
    }
    let shape = date_shape(tokens);
    let places = fractional_second_places(tokens);
    if places > MAX_SUB_SECOND_DIGITS {
        return None;
    }
    let mut days = value.trunc() as i64;
    let (mut seconds, ticks) = split_seconds(value.fract() * 86_400.0, places);
    if seconds >= 86_400 {
        seconds -= 86_400;
        days += 1;
    }
    let fraction = if places == 0 {
        String::new()
    } else {
        format!(".{ticks:0places$}")
    };
    if shape.elapsed_hour || shape.elapsed_minute || shape.elapsed_second {
        let total = days * 86_400 + seconds;
        return Some(duration(&shape, total, &fraction));
    }
    let hour = seconds / 3_600;
    let minute = (seconds % 3_600) / 60;
    let second = seconds % 60;
    let clock = format!("{hour:02}:{minute:02}:{second:02}{fraction}");
    if shape.has_date {
        let (year, month, day) = civil_date(days, date1904)?;
        let mut out = format!("{year:04}-{month:02}-{day:02}");
        if shape.has_time || seconds != 0 {
            out.push(' ');
            out.push_str(&clock);
        }
        return Some(out);
    }
    // Time-only: the units the section names, hour/minute/second order.
    let mut parts = Vec::new();
    if shape.hour {
        parts.push(format!("{hour:02}"));
    }
    if shape.minute {
        parts.push(format!("{minute:02}"));
    }
    if shape.second {
        parts.push(format!("{second:02}{fraction}"));
    }
    if parts.is_empty() {
        return Some(clock);
    }
    Some(parts.join(":"))
}

/// The whole seconds of a time of day, and the fractional-second digits a
/// section naming `places` of them rounds to. The section's own precision
/// decides both: a section naming no fractional digit rounds the second itself,
/// and one naming them carries a rounded-up fraction into the second.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the seconds a day holds bound every count here, so each fits both the integer type it \
              is cast to and the 53-bit mantissa it is cast from"
)]
fn split_seconds(raw: f64, places: usize) -> (i64, i64) {
    if places == 0 {
        return (raw.round() as i64, 0);
    }
    let scale = 10_i64.pow(u32::try_from(places).unwrap_or(0));
    let mut ticks = (raw.fract() * scale as f64).round() as i64;
    let mut seconds = raw.trunc() as i64;
    if ticks >= scale {
        ticks -= scale;
        seconds += 1;
    }
    (seconds, ticks)
}

/// The fractional-second digits a section names (`ss.0` names one, `ss.000`
/// three), or zero when the smallest unit it names is not a second.
fn fractional_second_places(tokens: &[Tok]) -> usize {
    let Some(dot) = tokens.iter().rposition(|tok| matches!(tok, Tok::Dot)) else {
        return 0;
    };
    let places = tokens[dot + 1..]
        .iter()
        .filter(|tok| matches!(tok, Tok::Digit(_)))
        .count();
    if places == 0 {
        return 0;
    }
    match tokens[..dot]
        .iter()
        .rev()
        .find(|tok| matches!(tok, Tok::Date(_) | Tok::Elapsed(_)))
    {
        Some(Tok::Date(Unit::Second) | Tok::Elapsed(Unit::Second)) => places,
        _ => 0,
    }
}

/// An elapsed format's total-count rendering: the elapsed unit counts the whole
/// span, the smaller named units the remainder, and the fraction belongs to the
/// seconds.
fn duration(shape: &DateShape, total_seconds: i64, fraction: &str) -> String {
    let mut parts = Vec::new();
    if shape.hour {
        parts.push(format!("{:02}", total_seconds / 3_600));
    }
    if shape.minute {
        let minutes = if shape.elapsed_minute {
            total_seconds / 60
        } else {
            (total_seconds % 3_600) / 60
        };
        parts.push(format!("{minutes:02}"));
    }
    if shape.second {
        let seconds = if shape.elapsed_second {
            total_seconds
        } else {
            total_seconds % 60
        };
        parts.push(format!("{seconds:02}{fraction}"));
    }
    parts.join(":")
}

// ── Number rendering ────────────────────────────────────────────

/// A section's rendered body: the text it shows, and whether that text holds a
/// non-zero digit of the value's own.
struct Rendered {
    text: String,
    /// A body of nothing but zeros is what a value rounded away to nothing shows,
    /// and the minus on it is a zero Excel never writes (`-0.4` under `0` is `0`),
    /// while a digit the format's own literals carry (`"v1 "0`) is no number of the
    /// cell's: neither earns the minus a section standing alone adds.
    shows_number: bool,
}

/// Whether the digits a section laid out hold a non-zero one — the digits alone,
/// the format's own literal text excluded.
fn has_digits(digits: &str) -> bool {
    digits
        .bytes()
        .any(|byte| byte.is_ascii_digit() && byte != b'0')
}

/// Where the one decimal point a section body may name stands.
#[derive(Clone, Copy)]
enum Point {
    /// The section names no point at all: its layout has no fraction.
    Absent,
    /// The section's point, as its position in the section's tokens.
    At(usize),
}

impl Point {
    /// Whether the placeholder at `index` is one of the section's own integer ones:
    /// every one of them when the section names no point.
    fn is_integer(self, index: usize) -> bool {
        match self {
            Self::Absent => true,
            Self::At(at) => index < at,
        }
    }

    /// Whether the placeholder at `index` is one of the section's fractional ones.
    fn is_fraction(self, index: usize) -> bool {
        match self {
            Self::Absent => false,
            Self::At(at) => index > at,
        }
    }
}

/// The one decimal point a section body may name. `None` is a section this reader
/// has no place for — one naming a second point (`0.0.0`), or one whose point stands
/// outside the placeholders `first..=last`: before the first (`.00`, whose digits
/// would be counted as the whole part) or after the last (`0.5`, whose `5` is literal
/// in Excel) — either of which would be dropped along with the part of the layout
/// around it.
fn decimal_point(tokens: &[Tok], first: usize, last: usize) -> Option<Point> {
    let mut point = Point::Absent;
    for (index, tok) in tokens.iter().enumerate() {
        // A second point is no layout this reader has a place for either.
        if matches!(tok, Tok::Dot) {
            if matches!(point, Point::At(_)) {
                return None;
            }
            point = Point::At(index);
        }
    }
    if let Point::At(at) = point
        && (at <= first || at >= last)
    {
        return None;
    }
    Some(point)
}

/// Render a non-date section, or `None` when it is a form this reader does not
/// implement: a fraction, the text placeholder, or a section that names no
/// placeholder at all — which is a format string (`[$-419]ДД.ММ.ГГГГ`) or a
/// directive, never a number a cell holds.
fn render_number(value: f64, tokens: &[Tok]) -> Option<Rendered> {
    if tokens
        .iter()
        .any(|tok| matches!(tok, Tok::Slash | Tok::Text))
    {
        return None;
    }
    if tokens.iter().any(|tok| matches!(tok, Tok::ExpSign(_))) {
        return render_scientific(value, tokens);
    }
    let (Some(first), Some(last)) = (
        tokens.iter().position(|tok| matches!(tok, Tok::Digit(_))),
        tokens.iter().rposition(|tok| matches!(tok, Tok::Digit(_))),
    ) else {
        // No placeholder at all: only a section whose text the format asked for is
        // shown. A quoted or escaped run (`"neg"`, `_(* "-"_)`) is text the author
        // put there; anything else — a localized format word this reader does not
        // know (`[$-419]ДД.ММ.ГГГГ`), a lone marker or directive (`%`, `,`,
        // `[$USD-409]`) or `General` — is a format string, never a number a cell
        // holds, so it is declined rather than passed off as the cell's value.
        return tokens
            .iter()
            .any(|tok| matches!(tok, Tok::Lit(_)))
            .then(|| Rendered {
                text: literal_text(tokens),
                shows_number: false,
            });
    };
    // The point the section names must stand between its placeholders; one anywhere
    // else would be dropped along with the part of the layout around it, so the
    // section is declined ([`decimal_point`]).
    let point = decimal_point(tokens, first, last)?;
    // A literal standing *between* the placeholders (`000-00-0000`) is laid
    // between the digits; one only around them is a plain prefix or suffix.
    if tokens[first..=last].iter().copied().any(is_text) {
        return render_layered(value, tokens, first, last, point);
    }
    render_plain(value, tokens, first, last, point)
}

/// Whether a token prints text of its own, in the place the section puts it: a
/// literal, a directive's symbol or the sign a `%` stands for. A body holding one
/// cannot be a plain layout of digits — the text would drop out of it — so such a
/// section is laid out in place instead ([`render_layered`]).
fn is_text(tok: Tok) -> bool {
    matches!(
        tok,
        Tok::Lit(_) | Tok::Raw(_) | Tok::Directive(_) | Tok::Percent
    )
}

/// The literal text tokens carry, placeholders and markers contributing nothing:
/// the `%` a section prints, a currency directive's symbol and escaped or quoted
/// characters.
fn literal_text(tokens: &[Tok]) -> String {
    let mut out = String::new();
    for tok in tokens {
        push_literal(*tok, &mut out);
    }
    out
}

/// Append a token's literal text; placeholders and markers contribute nothing.
fn push_literal(tok: Tok, out: &mut String) {
    match tok {
        Tok::Lit(c) | Tok::Raw(c) | Tok::Directive(c) => out.push(c),
        Tok::Percent => out.push('%'),
        Tok::Comma => out.push(','),
        _ => {}
    }
}

/// The digits a section's placeholders ask for: how many are in the integer part
/// (the `0`s force that part's width), how many are fractional and how many of
/// those are forced to a digit, and whether a `,` anywhere groups the integer
/// part.
#[derive(Default)]
struct PlainShape {
    int_zeros: usize,
    frac_zeros: usize,
    frac_placeholders: usize,
    grouping: bool,
}

/// Count the shape of the placeholders in `tokens[first..=last]`, whose section
/// names its decimal point `point` (at its own index in `tokens`): a `,` inside the
/// body groups rather than scales.
fn plain_shape(tokens: &[Tok], first: usize, last: usize, point: Point) -> PlainShape {
    let mut shape = PlainShape::default();
    for (offset, tok) in tokens[first..=last].iter().enumerate() {
        let fractional = point.is_fraction(first + offset);
        match *tok {
            Tok::Digit('0') if fractional => {
                shape.frac_placeholders += 1;
                shape.frac_zeros += 1;
            }
            Tok::Digit('0') => shape.int_zeros += 1,
            Tok::Digit(_) if fractional => shape.frac_placeholders += 1,
            Tok::Comma => shape.grouping = true,
            _ => {}
        }
    }
    shape
}

/// How many `,` standing right at the start of `tokens` scale the value by a
/// thousand each; a `,` between placeholders groups instead.
fn leading_commas(tokens: &[Tok]) -> usize {
    tokens
        .iter()
        .take_while(|tok| matches!(tok, Tok::Comma))
        .count()
}

/// How many `,` right after the last placeholder scale the value by a thousand
/// each (`,`, `,,`).
fn scaling_commas(tokens: &[Tok], last: usize) -> usize {
    leading_commas(&tokens[last + 1..])
}

/// The scaled integer a section's placeholders show: `value` after the `%` and
/// trailing-comma scaling the section asks for, rounded at its own precision.
fn scaled_plain(value: f64, tokens: &[Tok], last: usize, places: i32) -> Option<i128> {
    let mut value = value;
    for tok in tokens {
        if matches!(tok, Tok::Percent) {
            value *= 100.0;
        }
    }
    for _ in 0..scaling_commas(tokens, last) {
        value /= 1_000.0;
    }
    scaled_magnitude(value, places)
}

/// A scaled integer as the digit body a section shows: the integer part padded to
/// the `0`s that force its width and grouped when the section asks, the fraction
/// trimmed to the digits that force one, behind the point a section naming any
/// fractional part always carries.
fn digit_text(scaled: i128, shape: &PlainShape) -> Option<String> {
    let rendered = magnitude_text(scaled, u32::try_from(shape.frac_placeholders).ok()?)?;
    let (int_raw, mut frac_digits) = rendered.find('.').map_or_else(
        || (rendered.clone(), String::new()),
        |at| (rendered[..at].to_owned(), rendered[at + 1..].to_owned()),
    );
    while frac_digits.len() > shape.frac_zeros && frac_digits.ends_with('0') {
        frac_digits.pop();
    }
    // The integer part: `0`s force its width, and a bare zero prints nothing.
    let mut int_digits = if int_raw == "0" && shape.int_zeros == 0 {
        String::new()
    } else {
        int_raw
    };
    while int_digits.len() < shape.int_zeros {
        int_digits.insert(0, '0');
    }
    if shape.grouping {
        int_digits = group_thousands(&int_digits);
    }
    // The point is a literal of the section, wherever it names a fractional part:
    // it stands even when the value shows no digit after it (`#.#####` shows 1 as
    // `1.` and 0 as `.`), and a section naming no fraction never writes one.
    if shape.frac_placeholders > 0 {
        int_digits.push('.');
        int_digits.push_str(&frac_digits);
    }
    Some(int_digits)
}

/// The literal text standing before and after a digit body, and the body itself.
struct Pieces {
    digits: String,
    prefix: String,
    suffix: String,
}

/// The digit body a plain fixed-point section shows — the integer part padded to
/// the `0`s that force its width and grouped when the section asks, the fraction
/// trimmed to the digits that force one — with the literal text standing before
/// and after it (the scaling commas excluded: they are not printed). `grouping`
/// says whether the form being rendered can carry the section's grouping at all:
/// a body whose literals stand *between* its placeholders cannot, since the
/// commas the digits then carry would land in a placeholder.
fn plain_pieces(
    value: f64,
    tokens: &[Tok],
    first: usize,
    last: usize,
    grouping: bool,
    point: Point,
) -> Option<Pieces> {
    let shape = plain_shape(tokens, first, last, point);
    if shape.int_zeros > MAX_PLACEHOLDERS
        || shape.frac_placeholders > MAX_PLACEHOLDERS
        || (shape.grouping && !grouping)
    {
        return None;
    }
    let places = i32::try_from(shape.frac_placeholders).ok()?;
    let digits = digit_text(scaled_plain(value, tokens, last, places)?, &shape)?;
    Some(Pieces {
        digits,
        prefix: literal_text(&tokens[..first]),
        suffix: literal_text(&tokens[last + 1 + scaling_commas(tokens, last)..]),
    })
}

/// A plain fixed-point section: literals around a digit body of integer and
/// fractional placeholders, with `%`, scaling commas and grouping.
fn render_plain(
    value: f64,
    tokens: &[Tok],
    first: usize,
    last: usize,
    point: Point,
) -> Option<Rendered> {
    let Pieces {
        digits,
        prefix,
        suffix,
    } = plain_pieces(value, tokens, first, last, true, point)?;
    Some(Rendered {
        text: format!("{prefix}{digits}{suffix}"),
        shows_number: has_digits(&digits),
    })
}

/// A section whose literals stand *between* its placeholders (`000-00-0000`,
/// `(###) ###-####`): the digits are laid into the placeholder positions and the
/// literals keep their own. A body naming a fractional part or a grouping comma is
/// not one this layout can carry, and neither is a value whose digits do not fill
/// it exactly — a value with more digits than there are places would be shown
/// shorter than it is, and one with fewer leaves a literal nothing stands beside.
fn render_layered(
    value: f64,
    tokens: &[Tok],
    first: usize,
    last: usize,
    point: Point,
) -> Option<Rendered> {
    let body = &tokens[first..=last];
    // The section's own point is the one its body names ([`decimal_point`]): a body
    // between whose placeholders it stands cannot carry a fractional part.
    if matches!(point, Point::At(_)) {
        return None;
    }
    let Pieces {
        digits,
        prefix,
        suffix,
    } = plain_pieces(value, tokens, first, last, false, point)?;
    let places = body
        .iter()
        .filter(|tok| matches!(tok, Tok::Digit(_)))
        .count();
    if digits.len() != places {
        return None;
    }
    // Walk the body, giving each placeholder the next digit. Every one of them has
    // a digit: the lengths above are equal.
    let mut chars = digits.chars();
    let mut out = prefix;
    for tok in body {
        match *tok {
            Tok::Digit(_) => out.push(chars.next()?),
            other => push_literal(other, &mut out),
        }
    }
    out.push_str(&suffix);
    Some(Rendered {
        text: out,
        shows_number: has_digits(&digits),
    })
}

/// Scientific notation: a mantissa the section's placeholders lay out, then `E`,
/// the sign and the exponent. Excel normalises the mantissa into
/// `[1, 10^width)` — `width` the integer placeholders the section names — so the
/// exponent is the largest multiple of that width at or below the value's decade:
/// `##0.0E+0` shows 1234 as `1.2E+3`, not as `123.4E+1`, and `00.0E+0` shows
/// 12345 as `01.2E+4`.
fn render_scientific(value: f64, tokens: &[Tok]) -> Option<Rendered> {
    let marker = tokens
        .iter()
        .position(|tok| matches!(tok, Tok::ExpSign(_)))?;
    let Tok::ExpSign(sign) = tokens[marker] else {
        return None;
    };
    let mantissa_tokens = &tokens[..marker];
    let first = mantissa_tokens
        .iter()
        .position(|tok| matches!(tok, Tok::Digit(_)))?;
    let last = mantissa_tokens
        .iter()
        .rposition(|tok| matches!(tok, Tok::Digit(_)))?;
    let exponent_tokens = &tokens[marker + 1..];
    // The exponent prints its own placeholders — one run of them — with the literal
    // text around them and the scaling commas after them. A decimal point anywhere in
    // the region, or a placeholder the run does not reach (`E+0.0`, `E+0,0`), is a
    // character this reader would drop from the display, so the section is declined
    // rather than shown without it.
    let digits_at = exponent_tokens
        .iter()
        .position(|tok| matches!(tok, Tok::Digit(_)))?;
    let last_digit = exponent_tokens
        .iter()
        .rposition(|tok| matches!(tok, Tok::Digit(_)))?;
    if exponent_tokens.iter().any(|tok| matches!(tok, Tok::Dot))
        || exponent_tokens[digits_at..=last_digit]
            .iter()
            .any(|tok| !matches!(tok, Tok::Digit(_)))
    {
        return None;
    }
    let exponent_width = last_digit - digits_at + 1;

    // The point the section names must stand between the mantissa's placeholders;
    // one anywhere else would be dropped along with the part of the mantissa around
    // it, so the section is declined ([`decimal_point`]), exactly as the plain path
    // declines the same shapes.
    let point = decimal_point(mantissa_tokens, first, last)?;
    // The mantissa's integer placeholders that force a digit.
    let int_zeros = plain_shape(mantissa_tokens, first, last, point).int_zeros;
    let width = mantissa_tokens
        .iter()
        .enumerate()
        .filter(|(index, tok)| matches!(tok, Tok::Digit(_)) && point.is_integer(*index))
        .count()
        .max(1);
    let places = mantissa_tokens
        .iter()
        .enumerate()
        .filter(|(index, tok)| matches!(tok, Tok::Digit(_)) && point.is_fraction(*index))
        .count();
    if width > MAX_PLACEHOLDERS || places > MAX_PLACEHOLDERS || exponent_width > MAX_PLACEHOLDERS {
        return None;
    }
    let width = i32::try_from(width).ok()?;
    let places = i32::try_from(places).ok()?;

    // A `%` scales the value by a hundred each and is printed where it stands, and
    // a `,` run after the mantissa's or the exponent's placeholders scales it by a
    // thousand each and is not printed at all — exactly as in a plain section.
    let mut value = value;
    for tok in tokens {
        if matches!(tok, Tok::Percent) {
            value *= 100.0;
        }
    }
    let exponent_end = digits_at + exponent_width;
    let scaling = leading_commas(&exponent_tokens[exponent_end..]);
    let mantissa_scaling = scaling_commas(mantissa_tokens, last);
    for _ in 0..scaling + mantissa_scaling {
        value /= 1_000.0;
    }

    let mut exponent = 0i32;
    let mut scaled = 0i128;
    if value != 0.0 {
        exponent = decade(value.abs())?.div_euclid(width) * width;
        scaled = scaled_magnitude(value, places - exponent)?;
        // The mantissa's own display rounding can carry it into the next decade
        // (`0.00E+00` shows 999.5 as `1.00E+03`), which raises the exponent.
        let ceiling = pow10_int(usize::try_from(places).ok()?)?
            .checked_mul(pow10_int(usize::try_from(width).ok()?)?)?;
        if scaled >= ceiling {
            exponent += width;
            scaled = scaled_magnitude(value, places - exponent)?;
        }
    }

    // Excel lays a zero mantissa out to the section's own integer width, so
    // `##0.0E+0` shows 0 as `000.0E+0`; this reader does not, so a section whose
    // integer placeholders force a wider zero than they show is declined rather
    // than shown with a mantissa Excel would fill — one whose integer run is all
    // `0` (`0.00E+00`) keeps its zero.
    if scaled == 0 && usize::try_from(width).ok()? > int_zeros {
        return None;
    }

    let mantissa = mantissa_text(mantissa_tokens, first, last, scaled, point)?;
    if mantissa.trim().is_empty() {
        return None;
    }
    let mut exponent_digits = exponent.unsigned_abs().to_string();
    while exponent_digits.len() < exponent_width {
        exponent_digits.insert(0, '0');
    }
    let mut out = mantissa;
    out.push('E');
    if exponent < 0 {
        out.push('-');
    } else if sign == '+' {
        out.push('+');
    }
    out.push_str(&literal_text(&exponent_tokens[..digits_at]));
    out.push_str(&exponent_digits);
    out.push_str(&literal_text(&exponent_tokens[exponent_end + scaling..]));
    Some(Rendered {
        text: out,
        shows_number: scaled != 0,
    })
}

/// The mantissa a scientific section shows: the digits `scaled` stands for, laid
/// out by the section's own placeholders with the literal text around them. A
/// mantissa whose literals stand between its placeholders, or whose placeholders
/// a display cannot hold, is declined rather than laid out wrong.
fn mantissa_text(
    tokens: &[Tok],
    first: usize,
    last: usize,
    scaled: i128,
    point: Point,
) -> Option<String> {
    if tokens[first..=last].iter().copied().any(is_text) {
        return None;
    }
    let shape = plain_shape(tokens, first, last, point);
    if shape.int_zeros > MAX_PLACEHOLDERS || shape.frac_placeholders > MAX_PLACEHOLDERS {
        return None;
    }
    Some(format!(
        "{}{}{}",
        literal_text(&tokens[..first]),
        digit_text(scaled, &shape)?,
        literal_text(&tokens[last + 1 + scaling_commas(tokens, last)..])
    ))
}

/// A scaled magnitude as the digit text a section shows: its digits and the
/// decimal point `places` digits from the right. The sign is the section's own
/// business — [`display`] writes the minus a lone section adds, and every caller
/// here works from a magnitude.
fn magnitude_text(scaled: i128, places: u32) -> Option<String> {
    let digits = scaled.to_string();
    let places = usize::try_from(places).ok()?;
    if places == 0 {
        return Some(digits);
    }
    Some(match digits.len().checked_sub(places) {
        Some(point) => format!("{}.{}", &digits[..point], &digits[point..]),
        // Fewer digits than places: the magnitude is a fraction of one.
        None => format!("0.{}{digits}", "0".repeat(places - digits.len())),
    })
}

/// `|value| * 10^shift` as the integer Excel's display rounding produces: the
/// magnitude canonicalised to the 15 significant decimal digits Excel works with,
/// then rounded half away from zero at the place `shift` names. `None` when the
/// value has no exact display at that place.
///
/// The 15-digit step is what displays `1.005` under `0.00` as `1.01` — the binary
/// double holding it is a hair below the tie — and the mantissa of a scientific
/// section as the digits the section's own width normalises to. A `shift` no
/// integer can reach means the value rounds to nothing there, which is the zero a
/// display of that precision shows.
fn scaled_magnitude(value: f64, shift: i32) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }
    let magnitude = value.abs();
    // `d.dddddddddddddde±XX`: the magnitude's 15 significant decimal digits, so
    // `magnitude == digits * 10^(exponent - 14)`.
    let text = format!("{magnitude:.14e}");
    let (mantissa, exponent) = text.split_once('e')?;
    let exponent: i32 = exponent.parse().ok()?;
    let digits: i128 = mantissa.replace('.', "").parse().ok()?;
    let shift = exponent.checked_sub(14)?.checked_add(shift)?;
    let rounded = if shift >= 0 {
        // The digits reach past the place: the product is exact.
        let mut value = digits;
        for _ in 0..usize::try_from(shift).ok()? {
            value = value.checked_mul(10)?;
        }
        value
    } else {
        // The place is finer than the digits reach: round at it. A divisor past
        // what an integer holds is far beyond them, so they round to nothing.
        let Some(divisor) = pow10_int(usize::try_from(shift.unsigned_abs()).ok()?) else {
            return Some(0);
        };
        let quotient = digits / divisor;
        let remainder = digits % divisor;
        if remainder * 2 >= divisor {
            quotient + 1
        } else {
            quotient
        }
    };
    (rounded <= MAX_SCALED).then_some(rounded)
}

/// The decade of a finite non-zero magnitude: the exponent of its leading
/// significant digit, read from the same canonical decimal the display rounding
/// works from rather than from a logarithm a binary value can land just under.
fn decade(magnitude: f64) -> Option<i32> {
    let text = format!("{magnitude:.14e}");
    text.split_once('e')?.1.parse().ok()
}

/// `10^power` as an exact integer, or `None` when it has no integer form here.
fn pow10_int(power: usize) -> Option<i128> {
    let mut value: i128 = 1;
    for _ in 0..power {
        value = value.checked_mul(10)?;
    }
    Some(value)
}

/// Insert Excel's thousands separators into a run of ASCII digits.
fn group_thousands(digits: &str) -> String {
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

// ── Civil dates ─────────────────────────────────────────────────

/// Days from the Unix epoch to 1899-12-31, the 1900 system's day 0.
const EPOCH_1900: i64 = days_from_civil(1899, 12, 31);
/// Days from the Unix epoch to 1904-01-01, the 1904 system's day 0.
const EPOCH_1904: i64 = days_from_civil(1904, 1, 1);

/// The serial's civil date, applying Excel's phantom 1900-02-29 (serial 60) in
/// the 1900 system and the 1904 base otherwise. `None` for a serial day outside
/// the calendar that system defines — day 0 of the 1900 system, the fake
/// `1900-01-00` Excel writes for a zero date, and anything past 9999-12-31 in
/// either.
fn civil_date(serial_days: i64, date1904: bool) -> Option<(i64, u32, u32)> {
    if date1904 {
        return (0..=SERIAL_LIMIT_1904)
            .contains(&serial_days)
            .then(|| civil_from_days(EPOCH_1904 + serial_days));
    }
    if !(1..=SERIAL_LIMIT).contains(&serial_days) {
        return None;
    }
    Some(match serial_days.cmp(&60) {
        Ordering::Equal => (1900, 2, 29),
        Ordering::Greater => civil_from_days(EPOCH_1900 + serial_days - 1),
        Ordering::Less => civil_from_days(EPOCH_1900 + serial_days),
    })
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the civil year is bounded by the callers, so every era and day count fits the type it \
              is cast to and is non-negative where it is cast to an unsigned one"
)]
const fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = (year - era * 400) as u64;
    let day_of_year =
        (153 * (if month > 2 { month - 3 } else { month + 9 }) as u64 + 2) / 5 + (day as u64 - 1);
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era as i64 - 719_468
}

/// The proleptic Gregorian `(year, month, day)` for days since 1970-01-01
/// (Howard Hinnant's `civil_from_days`).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the day count is bounded by the callers, so every era, year and day-of-month here \
              fits the type it is cast to and is non-negative where it is cast to an unsigned one"
)]
const fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = (days - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_phase = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_phase + 2) / 5 + 1) as u32;
    let month = if month_phase < 10 {
        month_phase + 3
    } else {
        month_phase - 9
    } as u32;
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`display`] for a builtin format of the 1900 system, the shape most of these
    /// tests use.
    fn by_id(value: f64, id: u32) -> Option<String> {
        display(value, id, None, false)
    }

    #[test]
    fn general_is_detected_from_the_id_or_the_declared_code() {
        assert!(is_general(0, None));
        assert!(!is_general(0, Some("")));
        assert!(!is_general(14, None));
        assert!(is_general(5, Some("General")));
        assert!(is_general(5, Some("general")));
        assert!(!is_general(0, Some("#,##0")));
        // A directive, padding or a repeated section does not hide `General`.
        assert!(is_general(164, Some("[DBNum1][$-804]General")));
        assert!(is_general(164, Some(" General ")));
        assert!(is_general(164, Some("General;General")));
        assert!(!is_general(164, Some("General;0")));
        assert_eq!(by_id(5.0, 0), None);
        assert_eq!(display(5.0, 164, Some("General"), false), None);
        assert_eq!(display(5.0, 164, Some("[$-409]general"), false), None);
    }

    #[test]
    fn plain_formats_round_pad_and_group() {
        assert_eq!(by_id(5.4, 1), Some("5".to_owned()));
        assert_eq!(by_id(5.5, 1), Some("6".to_owned()));
        assert_eq!(by_id(5.5, 2), Some("5.50".to_owned()));
        assert_eq!(by_id(1234.0, 3), Some("1,234".to_owned()));
        assert_eq!(by_id(1234.5, 3), Some("1,235".to_owned()));
        assert_eq!(by_id(1234.5, 4), Some("1,234.50".to_owned()));
    }

    #[test]
    fn percentages_scale_the_value_by_a_hundred() {
        assert_eq!(by_id(0.25, 9), Some("25%".to_owned()));
        assert_eq!(by_id(0.25, 10), Some("25.00%".to_owned()));
    }

    #[test]
    fn currency_formats_wrap_negatives_in_parentheses() {
        assert_eq!(by_id(1234.0, 5), Some("$1,234".to_owned()));
        assert_eq!(by_id(-1234.0, 5), Some("($1,234)".to_owned()));
        assert_eq!(by_id(-1234.0, 6), Some("($1,234)".to_owned()));
        assert_eq!(by_id(-1234.5, 7), Some("($1,234.50)".to_owned()));
        assert_eq!(by_id(1234.5, 8), Some("$1,234.50".to_owned()));
    }

    #[test]
    fn accounting_formats_render_zero_as_the_dash_section() {
        assert_eq!(by_id(1234.0, 41), Some("1,234".to_owned()));
        assert_eq!(by_id(-1234.0, 41), Some("(1,234)".to_owned()));
        assert_eq!(by_id(0.0, 41), Some("-".to_owned()));
        assert_eq!(by_id(0.0, 42), Some("$-".to_owned()));
        assert_eq!(by_id(0.0, 43), Some("-".to_owned()));
        assert_eq!(by_id(0.0, 44), Some("$-".to_owned()));
    }

    /// Excel displays a number rounded from the 15 significant decimal digits it
    /// works with, so a value that is really the decimal `1.005` shows `1.01`
    /// under `0.00` even though the binary double holding it is a hair below the
    /// tie. A value that rounds away to nothing is never shown as a negative
    /// zero.
    #[test]
    fn a_tie_rounds_the_way_excel_displays_it() {
        assert_eq!(by_id(1.005, 2), Some("1.01".to_owned()));
        assert_eq!(by_id(0.145, 2), Some("0.15".to_owned()));
        assert_eq!(by_id(0.285, 2), Some("0.29".to_owned()));
        assert_eq!(by_id(8.575, 2), Some("8.58".to_owned()));
        assert_eq!(by_id(2.5, 1), Some("3".to_owned()));
        assert_eq!(by_id(999.5, 11), Some("1.00E+03".to_owned()));
        assert_eq!(by_id(0.9995, 11), Some("1.00E+00".to_owned()));
        assert_eq!(by_id(-0.4, 1), Some("0".to_owned()));
        assert_eq!(by_id(-0.04, 2), Some("-0.04".to_owned()));
        assert_eq!(by_id(-0.5, 1), Some("-1".to_owned()));
    }

    /// Excel normalises a scientific mantissa into `[1, 10^width)` — `width` the
    /// integer placeholders the section names — so the exponent is a multiple of
    /// that width and the neighbouring section shapes disagree.
    #[test]
    fn scientific_formats_normalise_the_mantissa() {
        assert_eq!(by_id(12345.678, 11), Some("1.23E+04".to_owned()));
        assert_eq!(by_id(12345.0, 48), Some("12.3E+3".to_owned()));
        assert_eq!(by_id(1234.0, 48), Some("1.2E+3".to_owned()));
        assert_eq!(by_id(1.0, 48), Some("1.0E+0".to_owned()));
        assert_eq!(by_id(1_000_000.0, 48), Some("1.0E+6".to_owned()));
        // A `0` placeholder pads the mantissa to its own width; a `#` shows
        // nothing where the mantissa has no digit.
        assert_eq!(
            display(12_345.0, 164, Some("00.0E+0"), false),
            Some("01.2E+4".to_owned())
        );
    }

    /// A value far below the place a format shows rounds to the zero that place
    /// holds — the display Excel gives it — rather than being declined as a value
    /// with no display at all.
    #[test]
    fn a_value_below_the_shown_place_rounds_to_zero() {
        assert_eq!(by_id(1e-300, 2), Some("0.00".to_owned()));
        assert_eq!(by_id(5e-324, 2), Some("0.00".to_owned()));
        assert_eq!(by_id(1e-30, 2), Some("0.00".to_owned()));
        assert_eq!(by_id(-1e-300, 2), Some("0.00".to_owned()));
    }

    #[test]
    fn unsupported_formats_come_back_as_none() {
        assert_eq!(by_id(1.5, 12), None);
        assert_eq!(by_id(1.5, 999), None);
        assert_eq!(
            display(1.5, 999, Some("0.0"), false),
            Some("1.5".to_owned())
        );
    }

    /// A format no value can be shown through is declined rather than rendered
    /// into a wrong or unrepresentable line: a code past what Excel stores, a
    /// section naming more placeholders than a display could hold, a value
    /// without a finite display at the section's precision, and a serial outside
    /// the domain a date or a duration has.
    #[test]
    fn a_format_or_value_with_no_display_is_declined() {
        let long_code = format!("\"{}\"0", "x".repeat(MAX_FORMAT_LENGTH));
        assert_eq!(display(1.5, 164, Some(&long_code), false), None);
        let many_places = format!("0.{}", "0".repeat(MAX_PLACEHOLDERS + 1));
        assert_eq!(display(1.5, 164, Some(&many_places), false), None);
        assert_eq!(
            display(1.5, 164, Some(&format!("0{}.0", "0".repeat(100))), false),
            None
        );
        assert_eq!(by_id(1e308, 2), None);
        assert_eq!(by_id(1e308, 9), None);
        assert_eq!(by_id(1e19, 14), None);
        assert_eq!(display(1e19, 164, Some("[h]"), false), None);
        // A magnitude no integer can hold as a display: declined, not rendered
        // into a line that could not be a cell's value.
        assert_eq!(display(f64::MAX, 164, Some("0.000"), false), None);
        // A decimal point outside the placeholders' run — a section whose whole
        // part is empty (`.00`), one with a literal where a fraction belongs
        // (`0.5`) or one naming two (`.0.0`) — is a layout whose dot and the part
        // around it would be dropped, so it is declined.
        for code in [".00", "0.5", "0.5%", "0.0.0", "0.", "#,##0."] {
            assert_eq!(display(0.5, 164, Some(code), false), None, "{code}");
        }
        assert_eq!(display(5.0, 164, Some("0.0.0"), false), None);
        // The exponent region is held to the same rule: the placeholders it prints
        // are one run, so a point between them (`0E+0.0`), a token between them
        // (`0E+0,0`) or a point outside the run, before the placeholders (`0E+.0`)
        // or after them (`0E+0.`), is declined.
        for code in ["0E+0.0", "0E+0,0", "0E+.0", "0E+0."] {
            assert_eq!(display(5.0, 164, Some(code), false), None, "{code}");
        }
        // A scientific mantissa's point is held to the same rule: one outside the
        // placeholders, or a second beside it, is declined too.
        for code in [".0E+0", "0.E+0", "0.0.0E+0", "#.E+0"] {
            assert_eq!(display(1234.0, 164, Some(code), false), None, "{code}");
        }
    }

    /// Excel uses the first section whose condition holds — not simply the
    /// section the value's sign names.
    #[test]
    fn a_conditional_section_serves_the_value_its_condition_names() {
        let code = Some("[>1000]0.0;[<=1000]0.00");
        assert_eq!(display(500.0, 164, code, false), Some("500.00".to_owned()));
        assert_eq!(display(1500.0, 164, code, false), Some("1500.0".to_owned()));
        let code = Some("[<0]\"neg\";0");
        assert_eq!(display(1234.0, 164, code, false), Some("1234".to_owned()));
        assert_eq!(display(-1234.0, 164, code, false), Some("neg".to_owned()));
        // A colour is not a condition, so the section still serves its sign.
        assert_eq!(
            display(1234.0, 164, Some("[Red]0.0"), false),
            Some("1234.0".to_owned())
        );
        // A condition standing after a currency directive gates the section too,
        // and the directive still prints its symbol.
        let code = Some("[$$-409][>100]0.00;0.0");
        assert_eq!(
            display(5000.0, 164, code, false),
            Some("$5000.00".to_owned())
        );
        assert_eq!(display(5.0, 164, code, false), Some("5.0".to_owned()));
        assert_eq!(display(-5.0, 164, code, false), Some("-5.0".to_owned()));
    }

    /// A colour before the condition is still a colour, and the condition still
    /// gates the section.
    #[test]
    fn a_colour_before_a_condition_does_not_hide_it() {
        let code = Some("[Red][<100]0.0;0");
        assert_eq!(display(50.0, 164, code, false), Some("50.0".to_owned()));
        assert_eq!(display(5000.0, 164, code, false), Some("5000".to_owned()));
    }

    /// Excel's own conditional rule, as the engines that reproduce it measure it:
    /// the first two sections' conditions are tested in order, the one that holds
    /// serves the value through its own section — as a magnitude when the condition
    /// itself names the negative side (`[<0]`, `[<=-1]`, `[=-3]`) — and a value no
    /// condition serves falls to the section Excel keeps for what none of them takes.
    #[test]
    fn a_conditional_format_keeps_a_negative_values_own_sign() {
        let code = Some("[>=1]0;0");
        assert_eq!(display(5.0, 164, code, false), Some("5".to_owned()));
        // The second section serves it, and carries no minus of its own.
        assert_eq!(display(-2.0, 164, code, false), Some("-2".to_owned()));
        assert_eq!(
            display(-12.3, 164, Some("[>=1000000]#,,\" M\";####.00"), false),
            Some("-12.30".to_owned())
        );
        // A longer format keeps a section of its own for the negative side, so a value
        // no condition serves goes through the second as its magnitude.
        assert_eq!(
            display(-2.0, 164, Some("[>=1]0;0;0.00"), false),
            Some("2".to_owned())
        );
        // No condition holds, and the third section a longer format keeps for what
        // none of them takes is the one that serves the value, minus and all.
        assert_eq!(
            display(-2.0, 164, Some("[>=1]0.0;[>=0]0.0;0.00"), false),
            Some("-2.00".to_owned())
        );
        // A section that scales its value keeps its own literals and its sign.
        let code = Some("[>=1000000]0.0,,\"M\";[>=1000]0.0,\"K\";0");
        assert_eq!(
            display(1_021_021.0, 164, code, false),
            Some("1.0M".to_owned())
        );
        assert_eq!(
            display(102_102.0, 164, code, false),
            Some("102.1K".to_owned())
        );
        assert_eq!(display(-500.0, 164, code, false), Some("-500".to_owned()));
        // The format's own first section carries no minus: a condition that serves
        // a negative value through it shows the magnitude, as Excel does.
        assert_eq!(
            display(-2.0, 164, Some("[>=-1]0;0"), false),
            Some("2".to_owned())
        );
        assert_eq!(
            display(-1.0, 164, Some("[<0]0.0;0.00"), false),
            Some("1.0".to_owned())
        );
        assert_eq!(
            display(-2.0, 164, Some("[<=-1]0;0"), false),
            Some("2".to_owned())
        );
        assert_eq!(
            display(-3.0, 164, Some("[=-3]0;0"), false),
            Some("3".to_owned())
        );
        // A minus the first section writes itself is still shown.
        assert_eq!(
            display(-2.0, 164, Some("[<0]-0.0;0.00"), false),
            Some("-2.0".to_owned())
        );
        // A section stating no condition after one that compares is not the section
        // Excel falls back to: a value no condition takes draws the third one a
        // longer format keeps, or the second of a two-section format.
        assert_eq!(
            display(5.0, 164, Some("[>10]0;0.0;\"n/a\""), false),
            Some("n/a".to_owned())
        );
        assert_eq!(
            display(-1.0, 164, Some("[>10]0;0.0;\"n/a\""), false),
            Some("1.0".to_owned())
        );
        // A positive value is read through a first section that states no condition.
        assert_eq!(
            display(200.0, 164, Some("0.00;[>100]#,##0"), false),
            Some("200.00".to_owned())
        );
        // A value no section serves is declined: the caller shows the stored text
        // marked as stored.
        assert_eq!(display(-5.0, 164, Some("0.00;[>100]#,##0"), false), None);
        assert_eq!(display(-2.0, 164, Some("[>=1]0;[>=0]0"), false), None);
        // A text-only section whose minus would be the reader's to write is
        // declined too: Excel writes its own minus into some such sections and not
        // others, so the format is not shown with the sign dropped.
        assert_eq!(
            display(-1.0, 164, Some("[>0]\"plus\";\"minus\""), false),
            None
        );
    }

    /// The minus the reader writes follows the cell's own number and the section's
    /// own text: a digit standing in that text (`"v1 "0`) is no number of the
    /// cell's, and a minus standing there is not written a second time.
    #[test]
    fn the_reader_writes_the_minus_only_where_no_section_does() {
        assert_eq!(
            display(-0.4, 164, Some("\"v1 \"0"), false),
            Some("v1 0".to_owned())
        );
        assert_eq!(
            display(-1.4, 164, Some("\"v1 \"0"), false),
            Some("-v1 1".to_owned())
        );
        assert_eq!(
            display(-2.0, 164, Some("-$0.00"), false),
            Some("-$2.00".to_owned())
        );
    }

    #[test]
    fn a_section_that_shows_nothing_is_none() {
        assert_eq!(display(5.0, 164, Some(";;;"), false), None);
        assert_eq!(display(-5.0, 164, Some("0;"), false), None);
        assert_eq!(display(0.0, 164, Some("#"), false), None);
        assert_eq!(display(5.0, 164, Some("#"), false), Some("5".to_owned()));
    }

    /// The point a section names stands even when the value shows no digit behind
    /// it, and a zero scientific mantissa is laid out to the section's own integer
    /// width: one whose width would widen the zero is declined rather than shown
    /// with a mantissa Excel would fill, while one whose integer run is all `0`
    /// keeps its zero.
    #[test]
    fn a_named_point_stands_and_a_zero_mantissa_follows_its_width() {
        assert_eq!(
            display(1.0, 164, Some("#.#####"), false),
            Some("1.".to_owned())
        );
        assert_eq!(
            display(0.0, 164, Some("#.#####"), false),
            Some(".".to_owned())
        );
        assert_eq!(
            display(0.5, 164, Some("#.##"), false),
            Some(".5".to_owned())
        );
        assert_eq!(
            display(1000.0, 164, Some("#,##0.##"), false),
            Some("1,000.".to_owned())
        );
        assert_eq!(display(0.0, 48, None, false), None);
        assert_eq!(display(0.0, 11, None, false), Some("0.00E+00".to_owned()));
        assert_eq!(by_id(1234.0, 48), Some("1.2E+3".to_owned()));
    }

    #[test]
    fn a_declared_code_wins_with_literals_and_a_negative_section() {
        let code = Some("#,##0.00\" kr\";[Red](#,##0.00\" kr\")");
        assert_eq!(
            display(1234.5, 164, code, false),
            Some("1,234.50 kr".to_owned())
        );
        assert_eq!(
            display(-1234.5, 164, code, false),
            Some("(1,234.50 kr)".to_owned())
        );
    }

    #[test]
    fn dates_follow_the_1900_system_by_default() {
        assert_eq!(by_id(45000.0, 14), Some("2023-03-15".to_owned()));
        assert_eq!(by_id(60.0, 14), Some("1900-02-29".to_owned()));
        assert_eq!(by_id(61.0, 14), Some("1900-03-01".to_owned()));
    }

    #[test]
    fn dates_follow_the_1904_system_when_asked() {
        assert_eq!(
            display(45000.0, 14, None, true),
            Some("2027-03-16".to_owned())
        );
        assert_eq!(display(0.0, 14, None, true), Some("1904-01-01".to_owned()));
    }

    #[test]
    fn date_and_time_sections_append_the_clock() {
        assert_eq!(by_id(45000.5, 22), Some("2023-03-15 12:00:00".to_owned()));
    }

    #[test]
    fn time_only_sections_print_the_units_they_name() {
        assert_eq!(by_id(0.5, 20), Some("12:00".to_owned()));
        assert_eq!(by_id(1805.0 / 86_400.0, 45), Some("30:05".to_owned()));
    }

    /// A section naming fractional seconds shows them, and the section's own
    /// precision decides the rounding — a `mm:ss.0` rounds at the tenth of a
    /// second, a `hh:mm:ss` at the second.
    #[test]
    fn a_section_naming_fractional_seconds_shows_them() {
        assert_eq!(by_id(0.500_046_3, 47), Some("00:04.0".to_owned()));
        assert_eq!(by_id(0.500_046_3, 45), Some("00:04".to_owned()));
        assert_eq!(
            display(4.492 / 86_400.0, 164, Some("hh:mm:ss.000"), false),
            Some("00:00:04.492".to_owned())
        );
        assert_eq!(
            display(1.5, 164, Some("[h]:mm:ss.0"), false),
            Some("36:00:00.0".to_owned())
        );
    }

    #[test]
    fn an_elapsed_unit_counts_the_whole_span() {
        assert_eq!(by_id(1.5, 46), Some("36:00:00".to_owned()));
        assert_eq!(display(1.5, 164, Some("[h]"), false), Some("36".to_owned()));
    }

    #[test]
    fn a_negative_date_serial_is_declined() {
        assert_eq!(by_id(-1.0, 14), None);
        // The 1900 system's day 0 is the fake `1900-01-00`, which is no date.
        assert_eq!(by_id(0.0, 14), None);
    }

    #[test]
    fn a_currency_directive_prints_the_symbol_it_names() {
        assert_eq!(
            display(1234.5, 164, Some("#,##0.00 [$€-407]"), false),
            Some("1,234.50 €".to_owned())
        );
        assert_eq!(
            display(1234.5, 164, Some("[$$-409]#,##0.00"), false),
            Some("$1,234.50".to_owned())
        );
        assert_eq!(
            display(1234.5, 164, Some("[$USD-409]#,##0.00"), false),
            Some("USD1,234.50".to_owned())
        );
        // A directive naming no symbol is only a locale, and prints nothing —
        // both the hex LCID Excel writes and the BCP-47 tags a workbook holds.
        assert_eq!(
            display(1234.5, 164, Some("#,##0.00[$-409]"), false),
            Some("1,234.50".to_owned())
        );
        assert_eq!(
            display(1234.5, 164, Some("[$-en-US]#,##0.00"), false),
            Some("1,234.50".to_owned())
        );
        assert_eq!(
            display(1234.5, 164, Some("[$-x-sysdate]0.0"), false),
            Some("1234.5".to_owned())
        );
    }

    /// A literal standing *between* the placeholders keeps its place among the
    /// digits: a phone number or an identifier is never rendered as its digits
    /// followed by the format's punctuation.
    #[test]
    fn a_literal_between_placeholders_lays_out_with_the_digits() {
        assert_eq!(
            display(
                5_551_234.0,
                164,
                Some("[<=9999999]###-####;(###) ###-####"),
                false
            ),
            Some("555-1234".to_owned())
        );
        assert_eq!(
            display(
                5_551_234_567.0,
                164,
                Some("[<=9999999]###-####;(###) ###-####"),
                false
            ),
            Some("(555) 123-4567".to_owned())
        );
        assert_eq!(
            display(123_456_789.0, 164, Some("000-00-0000"), false),
            Some("123-45-6789".to_owned())
        );
        // The sign a `%` stands for is a literal like any other: it keeps its place
        // among the digits, which a value whose digits do not fill the layout cannot.
        assert_eq!(
            display(1.0, 164, Some("0%00"), false),
            Some("1%00".to_owned())
        );
        assert_eq!(display(5.0, 164, Some("0%0"), false), None);
        // A literal only around the body is still a prefix or a suffix.
        assert_eq!(
            display(1234.5, 164, Some("\\$#,##0.00"), false),
            Some("$1,234.50".to_owned())
        );
        assert_eq!(
            display(-1234.5, 164, Some("#,##0.00\" kr\""), false),
            Some("-1,234.50 kr".to_owned())
        );
    }

    /// An over-wide value, an under-wide one, a fractional part and a grouping
    /// comma are all layouts this reader cannot put the digits into, so they are
    /// declined rather than shown as a shorter, longer, ungrouped or misplaced
    /// number.
    #[test]
    fn a_layered_body_declines_what_it_cannot_hold() {
        assert_eq!(
            display(1_234_567_890.0, 164, Some("000-00-0000"), false),
            None
        );
        assert_eq!(display(12_345_678.0, 164, Some("###-####"), false), None);
        assert_eq!(
            display(55_512_345_678.0, 164, Some("(###) ###-####"), false),
            None
        );
        // Fewer digits than places: a literal would stand with no digits beside it.
        assert_eq!(display(123.0, 164, Some("###-####"), false), None);
        assert_eq!(
            display(5_551_234.0, 164, Some("(###) ###-####"), false),
            None
        );
        assert_eq!(display(12_345.0, 164, Some("#,##0-#"), false), None);
        assert_eq!(display(1234.5, 164, Some("##-###.0"), false), None);
    }

    /// A `*` fill pads the cell out to its own width, which a line of text has no
    /// column for: it contributes nothing, wherever it stands.
    #[test]
    fn a_fill_contributes_nothing() {
        assert_eq!(by_id(1234.0, 41), Some("1,234".to_owned()));
        assert_eq!(
            display(5_551_234.0, 164, Some("###-####* "), false),
            Some("555-1234".to_owned())
        );
    }

    /// A section with no placeholder at all is a format's own wording, and only
    /// the text its author asked for is shown: a quoted or escaped literal is text
    /// the format prints, while an unquoted run, a lone marker or a directive is a
    /// format string this reader does not know, so it is declined rather than
    /// passed off as the cell's value.
    #[test]
    fn a_placeholder_less_section_is_text_only_when_the_format_asks_for_it() {
        assert_eq!(by_id(0.0, 41), Some("-".to_owned()));
        assert_eq!(
            display(-5.0, 164, Some("0;\"neg\""), false),
            Some("neg".to_owned())
        );
        assert_eq!(
            display(5.0, 164, Some("\"-\""), false),
            Some("-".to_owned())
        );
        for code in [
            "[$-419]ДД.ММ.ГГГГ",
            "General",
            " General ",
            "General;General",
            "%",
            ",",
            "[$USD-409]",
        ] {
            assert_eq!(display(5.0, 164, Some(code), false), None, "{code}");
        }
    }

    /// A scientific section keeps everything it names after the exponent: a `%`
    /// scales the value and is printed, and a literal stands where it is written.
    /// A `%` inside the mantissa is applied once, where it is written.
    #[test]
    fn a_scientific_section_keeps_its_scaling_and_its_literals() {
        assert_eq!(
            display(0.25, 164, Some("0.00E+00%"), false),
            Some("2.50E+01%".to_owned())
        );
        assert_eq!(
            display(0.25, 164, Some("0.00%E+00"), false),
            Some("2.50%E+01".to_owned())
        );
        assert_eq!(
            display(0.25, 164, Some("%0.00E+00"), false),
            Some("%2.50E+01".to_owned())
        );
        assert_eq!(
            display(1234.5, 164, Some("0.00E+00\" kg\""), false),
            Some("1.23E+03 kg".to_owned())
        );
        assert_eq!(by_id(-1234.5, 11), Some("-1.23E+03".to_owned()));
        // A trailing `,` run scales the value there too, and is not printed.
        assert_eq!(
            display(1.5, 164, Some("0.0E+0,,"), false),
            Some("1.5E-6".to_owned())
        );
    }

    /// The `@` text placeholder has no text to stand for in a cell holding a
    /// number, so the caller shows the stored value rather than this section's
    /// literal text.
    #[test]
    fn the_text_placeholder_is_declined_for_a_number() {
        assert_eq!(by_id(1.5, 49), None);
        assert_eq!(display(1.5, 164, Some("@"), false), None);
        assert_eq!(display(1.5, 164, Some("pre @"), false), None);
    }

    #[test]
    fn a_scientific_exponent_is_as_wide_as_the_section_names() {
        assert_eq!(by_id(123_456.0, 11), Some("1.23E+05".to_owned()));
        assert_eq!(by_id(123_456.0, 48), Some("123.5E+3".to_owned()));
        assert_eq!(by_id(0.000_123_4, 11), Some("1.23E-04".to_owned()));
    }
}

//! The platform's own text: how the bytes a program writes are read back.
//!
//! On unix this module is the lossy UTF-8 reading the shell has always had, and
//! nothing about it changes. On Windows the machine's own programs — the
//! interpreter's built-ins, the console utilities, a localised system message —
//! write in the code page the machine uses for console output rather than in
//! UTF-8, so reading their bytes as UTF-8 alone turns that text into replacement
//! characters: a localised error message is the visible case. [`decode`] reads
//! such output with the code page the machine itself reports (its OEM code page,
//! `GetOEMCP`) through the platform's own converter (`MultiByteToWideChar`), so
//! every code page a machine can be set to is covered rather than a fixed table
//! of the common ones.
//!
//! The reading is per line, and the two guarantees it is built for are: output
//! that is valid UTF-8 is untouched — a native toolchain that writes UTF-8 is
//! read exactly as it was before — and one line that is not UTF-8 does not make
//! the rest of the same output unreadable, because only that line is handed to
//! the code page. Splitting at a line break is safe under either reading: a
//! `\n` is never a byte inside a multi-byte character of UTF-8, and none of the
//! code pages the platform converts has one inside a character either.
//!
//! Three residuals are stated rather than repaired. A line whose bytes happen to
//! be valid UTF-8 *and* valid text in the machine's code page reads as UTF-8. A
//! line that is mostly UTF-8 but carries one byte that is not — the capture cap
//! cutting a character in half is the case that reaches this in practice — is
//! handed over whole, so its readable part is read through the machine's page too
//! instead of as the text it was. And a program that writes some encoding of its
//! own still reads back wrong: a text file's contents, a dump in a legacy code
//! page the machine does not use for its own output, or a program that formats its
//! message in the ANSI code page, which is the visible case on a machine whose
//! regional code page differs from its console's. Only output written the way the
//! machine writes it is read here; nothing guesses a per-program encoding.
//!
//! Reading an arbitrary file an agent or the owner points at is a different
//! question, and stays the read tool's own reading; so is a datum the product reads
//! back for a purpose of its own — git's plumbing, a managed runtime's version
//! probe — which keeps the lossy reading it has always had.

/// Decode one program's output bytes into the text the agent reads.
///
/// The lines that are valid UTF-8 are kept byte for byte; each line that is not
/// is handed to this platform's fallback whole (`decode_fallback` on Windows, a
/// lossy UTF-8 read everywhere else — where the result is exactly the one a
/// whole-buffer lossy read gives, since a replacement never reaches across a line
/// break).
pub(super) fn decode(bytes: &[u8]) -> String {
    decode_lines(bytes, decode_fallback)
}

/// The per-line reading, with the fallback as an argument so both halves of it —
/// which lines are kept and which are handed over, and that the kept ones really
/// are untouched — are driven from any host's test lane.
fn decode_lines(bytes: &[u8], mut fallback: impl FnMut(&[u8]) -> String) -> String {
    let mut out = String::with_capacity(bytes.len());
    // The break is the reading's own, never the fallback's: `\n` is the same
    // character under every reading, so it is written back between the lines
    // rather than handed to a code page that could only be asked to change it.
    let mut first = true;
    for line in bytes.split(|b| *b == b'\n') {
        if !first {
            out.push('\n');
        }
        first = false;
        match std::str::from_utf8(line) {
            Ok(text) => out.push_str(text),
            Err(_) => out.push_str(&fallback(line)),
        }
    }
    out
}

/// The reading of a line that is not UTF-8, for this platform.
#[cfg(not(windows))]
fn decode_fallback(bytes: &[u8]) -> String {
    lossy(bytes)
}

/// The reading of a line that is not UTF-8, for this platform: the machine's own
/// console code page, as the machine reports it.
#[cfg(windows)]
fn decode_fallback(bytes: &[u8]) -> String {
    // SAFETY: `GetOEMCP` takes no arguments, reads the code page the process is
    // configured with and touches no memory this module owns.
    let code_page = unsafe { windows_sys::Win32::Globalization::GetOEMCP() };
    // Zero reports nothing, and the converter would read it as "the ANSI code
    // page" rather than as an answer, so it gets the reading the shell had before
    // this module existed: lossy UTF-8, still the better of the two guesses for
    // text that is not this machine's console output.
    if code_page == 0 {
        return lossy(bytes);
    }
    decode_code_page(bytes, code_page)
}

/// The reading of a line no converter of this platform's describes: the whole
/// reading where there is no converter (unix, where the result is exactly the
/// whole-buffer lossy read), and on Windows the backstop for a line the converter
/// was not asked about — a length past its own `int` (never the shell's: its
/// capture cap is far below `i32::MAX`) or a call that reports nothing. This is
/// the display of output a program has already written, so nothing here may drop
/// the line.
fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Convert one line with `code_page` through the platform's own converter
/// (`MultiByteToWideChar`) — the reason no table of code pages lives here: the
/// machine converts with the page it actually has, installed pages included, and
/// a byte the page does not define comes back as that converter's own substitute
/// character rather than as a refusal.
#[cfg(windows)]
fn decode_code_page(bytes: &[u8], code_page: u32) -> String {
    // The shell's capture cap is far below `i32::MAX` (`SHELL_PIPE_READ_CAP`), so a
    // line's length always fits the platform's own `int`.
    let Ok(len) = i32::try_from(bytes.len()) else {
        return lossy(bytes);
    };
    // SAFETY: both calls pass a valid pointer to a buffer of exactly `len` bytes
    // and the length that describes it; the second also passes a wide buffer of
    // the size the first call reported. Neither call retains either pointer, and
    // `bytes` is not mutated by this function while they run.
    let width = unsafe {
        windows_sys::Win32::Globalization::MultiByteToWideChar(
            code_page,
            0,
            bytes.as_ptr(),
            len,
            std::ptr::null_mut(),
            0,
        )
    };
    let Ok(code_units) = usize::try_from(width) else {
        return lossy(bytes);
    };
    if code_units == 0 {
        return lossy(bytes);
    }
    let mut wide = vec![0u16; code_units];
    // SAFETY: as above; `wide` holds the `width` code units the first call
    // reported, which is what the second is told to fill.
    let written = unsafe {
        windows_sys::Win32::Globalization::MultiByteToWideChar(
            code_page,
            0,
            bytes.as_ptr(),
            len,
            wide.as_mut_ptr(),
            width,
        )
    };
    match usize::try_from(written) {
        Ok(written) => {
            wide.truncate(written);
            String::from_utf16_lossy(&wide)
        }
        Err(_) => lossy(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for a machine's code page in the tests below: every byte stands
    /// for itself, so what the fallback was handed is readable in the result —
    /// which is the only thing the split decides.
    fn bytewise(bytes: &[u8]) -> String {
        bytes.iter().map(|b| char::from(*b)).collect()
    }

    /// A line that is valid UTF-8 is kept byte for byte, whatever its neighbours
    /// are: the guarantee a native toolchain's output rests on, and the one a
    /// single line in another encoding must not take away from them.
    #[test]
    fn valid_utf8_lines_are_kept_beside_lines_that_are_not() {
        // The middle line is genuinely not UTF-8 (`\x80`/`\x81` are
        // continuation bytes with no lead); the lines around it are.
        let mixed = b"ok: h\xc3\xa9llo\nnot utf-8: \x80\x81\nok: done\n";
        assert_eq!(
            decode_lines(mixed, bytewise),
            "ok: héllo\nnot utf-8: \u{80}\u{81}\nok: done\n"
        );
        // The same input with the other reading on the line that is not UTF-8.
        assert_eq!(
            decode_lines(mixed, |_| "<other>".to_string()),
            "ok: héllo\n<other>\nok: done\n"
        );
    }

    /// Only the line that is not UTF-8 reaches the fallback — a whole-buffer read
    /// would have handed the readable lines over with it.
    #[test]
    fn only_the_line_that_is_not_utf8_is_handed_over() {
        let mut lines = Vec::new();
        let text = decode_lines(b"a\n\xff\nb\n", |line| {
            lines.push(line.to_vec());
            bytewise(line)
        });
        assert_eq!(text, "a\n\u{ff}\nb\n");
        assert_eq!(lines, vec![b"\xff".to_vec()]);
    }

    /// The reading leaves the input's own line breaks where they were: the break
    /// between two lines is the reader's and not the fallback's to change, a `\r`
    /// before it belongs to the line the fallback is handed (Windows programs
    /// write `\r\n`), and a capture that ends without a break keeps none.
    #[test]
    fn the_line_breaks_of_the_input_are_kept() {
        assert_eq!(decode_lines(b"\xff\r\nplain", bytewise), "\u{ff}\r\nplain");
        assert_eq!(decode_lines(b"", bytewise), "");
    }
}

## Command lines

A line break is how two commands are written in one call here: the lines run in order, a failing line does not stop the lines after it, and the status the call reports is the last line's. A trailing break, and blank lines, add no command of their own. `&&` and `||` keep their own conditional meaning.

What a break cannot do:

- A break straight after a trailing caret is not a separator — the caret escapes it, so the lines are one command: `copy a.txt ^` on one line and `b.txt` on the next is the single command `copy a.txt b.txt`. The caret also escapes the first character of the line it continues: {{escaped_target}}
- A break inside a bracketed group belongs to the block, so a multi-line `for … do ( … )` body runs as written, and so does a `for` set written across lines (`for %f in (a` on one line and `b) do echo %f` on the next).
- A `(` opens a group only where a command can start: at the start of the text, after a separator the interpreter runs itself (`&`, `&&`, `||` — not a single `|`, whose right-hand side a child interpreter counts), first inside a group, after the `)` of a block it runs itself, after an `@` that prefixes it, and where an `if`/`for` statement writes its body — the command right after an `if` condition, the `do`/`else` after the statement's own `)`, and the `in ( … )` set.
- A `(` in any other position is written text (`echo a (`), a `(` as a pipeline's right-hand side included, and a command carrying a break inside one is refused rather than run as the block it may have been.
- A break that cuts a command's own statement in half — a `for` without its `do`, an `if` without the command its condition governs — is refused, as is a `)` standing where a command can start and closing no group, and a command whose caret is left with nothing to escape (the end of the text, or a blank line).
- A line this platform's reader runs past the break cannot be followed by another line: an `if`/`for` statement whose command is written on its line would run the line after the break under the condition — or once per iteration — instead of as a command of its own, and a word that is a comment (`rem …`) or a label (`:…`) where a command can start, a separator's position included, would take the rest of that line as the comment's text or the label's name, swallowing the line after the break with no error and no status of its own. Such a text is refused rather than run as less than it says.
- A break straight after `&`, `&&`, `|`, `||`, `<` or `>` is that operator taking the rest of its operand from the next line, so `dir &&` on one line and `next` on the next is the single command `dir && next`. A line that begins with `&` or `|`, and a join that would put two of `<`/`>`/`&` together into an operator the command did not write, are refused instead.

Any text refused here is refused whole — no part of it runs. {{remedy}}

A literal line break inside an argument cannot be carried to a program by any quoting — write that text to a file and pass the file. A command whose break sits inside a quoted argument is refused rather than run with a truncated argument.

Text blocks of the heredoc kind (`<<`) do not exist here, and a command that feeds one to the interpreter is refused.

A variable set on one line and read on a later one does not see the value it was set to: the whole command is expanded before its first line runs. Set and read a variable inside one line, or read it in a later call.

A command carrying more text than this platform's own command line holds — {{cap}} units in all, of which the interpreter's path, the settings it is started with and its `/C` switch take about {{overhead}}, leaving {{limit}} for the command text — is refused as a tool error naming the limit.

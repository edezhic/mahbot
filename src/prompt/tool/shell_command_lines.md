## Command lines

A line break writes two commands in one call: lines run in order, a failing line does not stop later lines, and the reported status is the last line's. A trailing break and blank lines add no command. `&&` and `||` keep their meaning.

What a break cannot do:

- A break straight after a trailing caret is not a separator — the caret escapes it, so the lines are one command. The caret also escapes the first character of the continued line: {{escaped_target}}
- A break inside a bracketed group belongs to the block, so a multi-line `for … do ( … )` body runs as written.
- A `(` opens a group only where a command can start: at the start, after `&`/`&&`/`||` (not a single `|`), first inside a group, after the `)` of a block, after `@`, and in an `if`/`for` body. A `(` elsewhere is written text, and a break inside one is refused.
- A break cutting a statement in half (a `for` without `do`, an `if` without its command), a `)` closing no group, or a caret with nothing to escape is refused.
- A line the platform's reader runs past the break cannot be followed by another: an `if`/`for` whose command is on its line would run the next line under the condition, and a `rem` comment or `:label` where a command can start would swallow it. Such text is refused.
- A break straight after `&`, `&&`, `|`, `||`, `<` or `>` continues that operator's operand on the next line. A line beginning with `&` or `|`, or a join making two operators, is refused.

Any text refused here is refused whole. {{remedy}}

A literal line break inside an argument cannot be carried by any quoting — write the text to a file and pass the file. A command whose break sits inside a quoted argument is refused.

Heredoc text blocks (`<<`) do not exist here and are refused.

A variable set on one line and read on a later one does not see the value: the whole command is expanded before its first line runs.

A command longer than this platform's command line — {{cap}} units in all, of which the interpreter path, settings and `/C` take about {{overhead}}, leaving {{limit}} for the text — is refused as a tool error naming the limit.

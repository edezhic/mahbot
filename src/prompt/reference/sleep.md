# sleep — reference

## Calling it

`sleep` takes no arguments. Call it alone in its round: no other tool call and no text beside it. A bundled or text-accompanied call is rejected at runtime.

## What it does

A successful `sleep` ends the run silently and delivers nothing. The session is preserved and resumed automatically by the next event — a user message, an async sub-agent result (`analyze`, or `send` when available), or a fired automation.

## When not to use it

Text in any round that contains a tool call is never delivered to the user, including text written alongside that call. If you have something to say, write a plain-text round with no tool calls: that ends your turn and delivers the message. Never use `sleep` to wait inside a turn; use it only when there is genuinely nothing to tell the user.

`sleep` ends your turn silently: nothing is delivered. Call it with no arguments, alone in your round — no other tool call, no text; a bundled or text-accompanied call is rejected.

Use it only when you have nothing to say and await the next event (a user message, an async result, a fired automation). The session resumes then; never sleep to wait inside a turn.

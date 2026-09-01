Bind the admin's own Telegram @username so incoming bot messages are routed to them. Pass the handle with or without the leading `@`.

This is admin-only: it operates on the admin account (the Support agent's personal workspace owner). The handle must not already be bound to another user — if it is, the tool refuses rather than silently reassigning it. The handle "unknown" is reserved (Telegram reports username-less senders under it) and the tool refuses to bind it.

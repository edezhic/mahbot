Create a regular user and bind them to Telegram with a chosen default agent. Pass the user's display `name`, their Telegram `@username`, and the `default_agent` they start with — `assistant` or `artist`.

They are bound to the Telegram handle so their messages are routed to them. The handle "unknown" is reserved and cannot be bound. The tool refuses a Telegram handle already bound to a different user rather than reassigning it.

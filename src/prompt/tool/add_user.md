Create a regular (non-admin) user and bind them to Telegram with a chosen default agent. Pass the user's display `name`, their Telegram `@username`, and the `default_agent` they start with — `assistant` or `artist`.

The user is always created as a regular user: there is no permissions argument, so they cannot reach the manager/engineer/admin toolset. They are bound to the Telegram handle so their messages are routed to them.

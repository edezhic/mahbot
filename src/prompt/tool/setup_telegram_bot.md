Persist a Telegram bot token so the daemon can receive messages from the admin's bot. Pass the bot token from BotFather (the `NNN:AAA...` string).

The token is saved and the Telegram listener hot-reloads it immediately — no restart is required.

Next step: ask the user to send `/start` to the bot in Telegram, then use `bind_telegram` to bind their @username so their messages are routed to them.

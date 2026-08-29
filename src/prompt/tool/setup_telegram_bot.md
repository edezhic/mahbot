Persist a Telegram bot token so the service can receive messages through the telegram bot and send back the replies. Pass the bot token from BotFather (the `NNN:AAA...` string). The token is saved and the Telegram listener hot-reloads it immediately — no restart is required.

Next step: use `bind_telegram` to bind user's `@username` so their messages are routed to them, and ask the user to send `/start` to the bot.

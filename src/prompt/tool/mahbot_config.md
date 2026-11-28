Change this product's own settings from here — the model provider and its endpoint, the model
slots, the media catalogues and their provider routing, web search, Telegram and its local Bot
API mode, the audio switches, auto-update and start-at-login, accounts and their tool grants,
custom roles, and a managed workspace's profile (who staffs each stage, verification's pass
threshold, whether every project check must pass — read it, confirm it, edit part of it, or reset
it to the default). `get` reads one setting back whole with its effective value, `list` summarises
every one (a long value is clipped), and `set` changes one; every change takes effect at once, and a
credential is never read back in full.

The exact fields for every action are available on request: call this tool with
`action: "reference"` (or run `mahbot config reference` in a terminal) for the whole text — every
action, every field, and what is inert where. Guessing a field fails, and the error names the
`reference` action.


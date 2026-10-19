Configure the service for the admin. Pick one `action`; only that action's fields are needed.

- `setup_telegram_bot` — persist a Telegram bot token so the service receives and answers messages through the bot. It is validated live via getMe and saved; the Telegram listener hot-reloads it immediately, no restart. Then `bind_telegram` the admin's account and ask them to send `/start`.
- `bind_telegram` — bind the admin's own Telegram account so incoming bot messages reach them; the admin supplies the handle by hand, nothing is looked up. A number is the binding's address from the start; a nickname gains its address when its owner writes (Telegram forbids the bot to open the conversation until then).
- `add_user` — create a guest account bound to Telegram. The account is created with its personal workspace folder; if the folder cannot be created the account still exists and the answer says so. The `name` is used verbatim, never adjusted — a refused name is a question back to the admin. Case and Unicode composition are ignored: `ALICE` collides with `Alice`, and a composed accent with its decomposed form, because on macOS and Windows those are one folder, so the new account would resolve into the existing one's workspace. A name with no binding completes a half-finished earlier run.
- `setup_web_search` — register a web-search backend so agents can use `web_search`; provider and key are persisted, and supporting agents can call it immediately.

Custom tools are script files in the `shared` folder of the admin's personal workspace; a tool is referenced by its file name without extension and becomes available to another user only by being granted.

- `grant_tool` — grant one custom tool to a user. Granting an already-held tool is a no-op.
- `revoke_tool` — revoke one custom tool from a user. Revoking an absent grant is a no-op.
- `list_grants` — show the recorded grants: pass `user` for one user, or omit it for every user with any. Grants are listed even when the granted tool's file no longer exists, so the list reflects what was recorded, not what currently works.

Custom roles are the administrator's own agents: each has a name, an authorial prompt that fully defines it, the subset of the assistant's tools it may use, and its launch behaviour. A role either exists or it does not; these actions need no confirmation.

- `create_custom_role` — define a new role. The `prompt` is its whole definition — the only thing the role is told besides the folder it works in, so everything it must know and do has to be written there. An existing role is never silently overwritten.
- `update_custom_role` — change an existing role by `name`; omitted fields keep their stored value. An empty `tools` array clears the role's tools.
- `list_custom_roles` — show every defined role: name, behaviour, chosen tools and its authorial prompt. Reports when there are none.
- `delete_custom_role` — remove a role by `name`. Alarms addressed to it can no longer wake anyone, and the owner is told through the assistant.

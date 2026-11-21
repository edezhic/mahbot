Every action and field of the `mahbot_config` and `workspace` tools; `mahbot config reference` and
`mahbot workspace reference` print it.

In a terminal the operation word is the action with `_` written `-` (`setup-telegram-bot`), each
field a flag of the same name, `_` written `-`; `action` is no flag. A value starting with `@` names
a file (`./@name` for a literal one), but a path field is resolved where the command runs and takes
no file form; a boolean is a bare switch (`--maintenance`, `=false` for false); a list repeats.

`get` reads one setting whole, `list` summarises every one — a long value is clipped — and `set`
changes one. Values are effective — defaults resolved, never a stored row — and a change takes
effect at once. A credential is never read back in full: `get` gives a hint, `list` omits it. A
value too long for one read-back is refused. An unlisted name or field is refused, naming this
reference; an unlisted terminal flag, by the command itself.

## mahbot_config

Bracketed fields are the ones the action reads.

- `reference` — this text, whole.
- `about` — the product's own README, whole: what this system is, what it does and how it is set
  up. Static text, so it needs no running instance.
- `get` (`setting`) — one setting's effective value, whole.
- `list` — every setting summarised, credentials left out, `provider_order` for the two slot models
  only.
- `set` (`setting`, `value`) — change one setting, answering what it did. The provider's credential
  and endpoint go through `setup`: the answer is the value, then its report; neither is cleared here,
  and the endpoint's own key is refused alone, naming `setup`.
- `setup` (`provider_key`, `provider_endpoint`, `provider_endpoint_key`, `clear_provider_endpoint`,
  `telegram_token`, `owner`, `unbind_owner`) — the model provider (an OpenRouter key, or a custom
  endpoint and its optional key), the bot token and the owner, changing only what it names; a
  configured provider also finishes onboarding. The key is checked live; a rejected one is a warning
  on a saved key. `clear_provider_endpoint` goes back to OpenRouter's own, dropping its key row,
  leftover ones included.
- `setup_telegram_bot` (`token`) — save the bot token; checked live, and the listener reloads.
- `bind_telegram` (`handle`, `user`) / `unbind_telegram` (`user`) — bind a Telegram account (the
  owner's own, or a guest named by `user`) or remove its binding. A numeric handle is an address at
  once; a nickname gains one when its owner writes. One binding per account: a second is refused, so
  `unbind_telegram` frees one first.
- `add_user` (`name`, `telegram`) — create a guest account and, when `telegram` is given, bind it;
  without one it has no way in yet. The name is the account's folder too; names are caseless
  (`ALICE` collides with `Alice`), the owner's name is reserved, and an account already bound is
  refused.
- `delete_user` (`user`) — remove an account and its bindings; its sessions, chat history and folder
  stay on disk, and the answer names them. The owner cannot be deleted.
- `list_users` — every account with its Telegram binding.
- `setup_web_search` (`provider`, `key`) — register `deepseek` or `exa` with its key; both are
  required.
- `grant_tool` / `revoke_tool` (`user`, `tool`) — grant or revoke one custom tool for one account;
  idempotent.
- `list_grants` (`user`) — the recorded grants: `user` for one account, omitted for every account
  with any.
- `create_custom_role` / `update_custom_role` / `list_custom_roles` / `delete_custom_role`
  (`name`, `tools`, `behavior`, `prompt`) — the owner's own agents. `create_custom_role` needs
  `prompt`, `update_custom_role` one of `prompt`, `tools`, `behavior`; a name is up to 32 letters,
  digits, `_` and `-`, starting with a letter, and must not shadow an agent-id namespace (`role`,
  `ticket`, `maintainer`, `discovery`, …). A `tools` entry is a catalogue tool name, optionally
  `name:kind1,kind2` for one with kinds, or `custom:<script>` for one script in the owner's `shared`
  folder — one entry per script, so several scripts sit in the same list. A role holds the `custom`
  dispatcher exactly when its list names at least one script, and the plain `custom` entry means
  nothing; an empty array clears the list.

`user` names the account an action acts on: `bind_telegram` defaults to the owner and `list_grants`
to every account; the other four require it.

## settings for get / set

- `provider_key` — the OpenRouter key. Secret; written by `setup`, which checks it live. Cannot be
  cleared.
- `provider_endpoint` — the chat endpoint; OpenRouter's own unless `setup` set a custom one, which
  `get` reports as the value in use. `setup` probes it and warns when it does not answer;
  `clear_provider_endpoint` settles it back. A custom endpoint is capped at 640 bytes.
- `provider_endpoint_key` — the custom endpoint's own key. Secret; written by `setup` with the
  endpoint (alone it names nothing, so `set` refuses it and names `setup`), and dropped when the
  endpoint settles back.
- `manager_model`, `worker_model` — the model slot for the manager-side roles and for workers; empty
  means the built-in default, which `get` reports.
- `image_gen_models`, `video_models` — catalogues of model names, newline- or comma-separated, at
  most 32, each name at most 128 characters; empty means the default model, which `get` reports. An
  image model the catalogue lacks or lists as text-only is refused (an unreachable catalogue passes
  silently); video models are taken as given.
- `provider_order:<model>` — that model's provider priority, comma-separated; empty means the
  product's own order. Inert for chat under a custom chat endpoint (a media model's order keeps
  applying — media always talks to OpenRouter), and `get` says so.
- `web_search_provider` — `deepseek`, `exa`, `auto` (or `automatic`), or empty for automatic. `get`
  names the backends text and picture search really use, and says when there is no search tool.
- `deepseek_key`, `exa_key` — the search backends' keys. Secrets.
- `telegram_bot_token` — Secret; checked live, and the listener hot-reloads.
- `telegram_local_server` — `local` or `cloud`: where the bot is served. `local` starts a switch and
  answers at once; it needs a bot token and the two credentials below. `get` reports the side served
  and whether a switch is in flight or blocked.
- `telegram_api_id`, `telegram_api_hash` — the local server's credentials; the hash is a secret.
  Each is shape-checked as the window's dialog does. Set the pair before switching.
- `audio_transcription_use_local` — `on` or `off`: local speech recognition. An absent row means on,
  but a fresh install starts off. Turning it on loads nothing: the recogniser is fetched when first
  needed, and turning it off turns voice off too, since they share it.
- `voice_enabled` — wake-word voice input; off unless turned on. Refuses to turn on while
  transcription is off (they share the recogniser) or with an enrolment the current encoder cannot
  score. Needs a microphone: without one the pipeline reports the device failure.
- `tts_enabled` — spoken replies; off unless turned on. Turning it on fetches the voice model;
  speaking needs an audio output, and says so.
- `autoupdate` — install newer versions by itself. Only a downloaded build updates itself (a
  registry-cache install included); elsewhere the window hides the control and `set` refuses for the
  same reason.
- `autolaunch` — start at login. Only a published copy registers itself (not a registry-cache
  install); a system-managed host refuses the registration, naming the command to run, and a masked
  unit refuses both ways, naming the unmask. `get` reports the decision and whether the registration
  is in place — `masked` when the unit is; `list` the decision alone.

Wake-word enrolment needs a microphone and stays in the window.

## workspace

Bracketed fields are the ones the action reads.

- `reference` — this text, whole: every action, every field, and what is inert where.
- `list` (`workspace`) — the managed workspaces, read live: name, status, whether the pipeline is
  paused, whether background maintenance is on, and the project folder; the active one is marked, or
  the listing says none is. Without `workspace`, every managed workspace; with it, only the one
  named. Use it for any question about current state — the `<managed-workspaces>` block is a
  session-start snapshot and carries neither state.
- `show` (`workspace`) — one workspace: path, status, paused, maintenance, active, rules, commands.
- `add` (`name`, `path`) — add a project directory and make it active; it starts pending and paused,
  and awaits its first discovery pass, which begins by itself once the model provider is configured.
  Managed workspaces are for software projects only — documents, notes and research files go in the
  personal workspace. The name is up to 40 ASCII letters and underscores, starting with a letter;
  `path` must be an existing directory (absolute, relative or `~`, resolved where the call runs).
- `delete` (`workspace`) — remove a workspace, its jobs and its research runs; the folder and the
  tickets stay on disk.
- `pause` / `resume` (`workspace`) — freeze or unfreeze a pipeline: no stage advances, and agents
  stop at their next round boundary; a resume lets work in flight continue from where it stopped,
  and neither direction loses anything. Only a `ready` workspace has one, and a named one that is
  not ready is refused; for a workspace that is not `ready`, the `, paused` that `list` prints is
  its discovery's own analysis pause, not a freeze you applied. With no `workspace`, all of them:
  the ready ones are acted on, the rest skipped. That is a fleet-wide write — use it only when the
  owner asked for all of them, and then pass nothing else. The answer reports workspace by
  workspace: paused, resumed, already in that state, or skipped because it was not ready.
- `maintenance` (`workspace`, `maintenance`) — turn the background maintainer on or off.
- `rediscover` (`workspace`) — re-analyse a project: its per-role contexts are cleared and discovery
  starts again; the pipeline is paused until it finishes, and the answer says so.
- `rediscover_diagnostics` (`workspace`) — re-detect only the project's commands: the stored ones
  are cleared while they are; status, pause and per-role contexts are untouched.
- `set_diagnostics` (`workspace`, `format`, `format_check`, `lint`, `lint_fix`, `type_check`,
  `build`, `unit_test`) — write the project's commands by hand: omit a field to keep it, pass an
  empty string to clear it, and name at least one.
- `set_rules` (`workspace`, `rules`) — write the project rules, at most 4000 characters (longer
  text is truncated). They reach a session's prompt when it is built or next compacted, not a
  running one.
- `use` (`workspace`) — make a workspace the active project.
- `board` (`workspace`, `phase`) — the tickets of the addressed workspaces, one line each: who
  filed it, its phase, its id and its title. Without `workspace`, every managed workspace's tickets
  come back together, each id carrying its project. Every non-archived ticket is listed, finished
  ones included; `phase` is one exact snake_case phase word (`in_development`, `done`, …) and
  narrows a long listing; `archived` is refused, and archived tickets are never listed — archiving
  is a one-way visibility switch. A listing that is too long is cut short and says it is partial.
- `ticket` (`ticket_id`, `full`) — one ticket by its full id, with the fields the project's manager
  sees: title, description, phase, filer, project, created and updated times, priority, the
  supersedes / superseded-by / prerequisites / archived markers, and the whole comment thread. A
  ticket is shortened by default — description cut at 500 characters, the three newest comments in
  full and older ones at 200 — and `full: true` is it whole and unclipped; the heaviest tickets
  carry hundreds of kilobytes of comments. An archived ticket is still readable here by its id.
- `comment` (`ticket_id`, `content`) — leave a clarification note on a ticket by its full id, as
  `assistant`: an ordinary comment in that ticket's thread. One left while an agent works on the
  ticket reaches it at the start of its next round without interrupting it; otherwise it waits in
  the thread until the ticket is next worked on. It never advances, blocks, cancels or unblocks a
  ticket, never changes a phase or a prerequisite, and cannot be edited or removed afterwards — so
  leave one only when the owner asks.

`workspace` names the managed workspace an action covers, spelled exactly as `list` does. Omit it
for `list` and `board` (every one) and for `pause` / `resume` (a fleet-wide freeze); `reference`,
`add`, `ticket` and `comment` do not read it. `ticket_id` is the full `<workspace>-<number>` id,
never a bare number.

`list`, `show`, `board` and `ticket` change nothing and work for any managed workspace, ready or
not, frozen or not. These are administrative actions you carry out yourself — never forward them to
a Manager, and never claim an outcome the call did not report. Close with the outcome in the reply
that ends your turn, naming the workspace — and, for a change you carried out, its resulting state.

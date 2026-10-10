Start an existing custom role: hand it a piece of work and let it run as its own agent. The role's final answer is delivered to the administrator's assistant conversation — which decides what to relay onward — naming the role and the folder it ran in, so there is nothing to poll.

A custom role is defined by the administrator: a name, an authorial prompt, the subset of the assistant's tools it may use and its behaviour. The `name` must be one that exists — use `mahbot_config`'s `list_custom_roles` to see the defined roles. The `message` is delivered to the role as an ordinary user message and is what the role is asked to work on.

The role works in a folder with the assistant's full shell/read/edit reach. It runs in the administrator's personal folder unless you pass `workspace` — the exact name of a registered workspace, or an absolute path to any folder on the machine.

Optional `identifier` names a branch: with it the launch continues that named branch of the role and its history is restored; without it the role's own default behaviour applies (one common continuable branch, or a fresh one-off run). An unknown role, an invalid name or identifier, or an unresolvable workspace is refused. The launch is durable — it is persisted before it is routed, so it survives a restart.

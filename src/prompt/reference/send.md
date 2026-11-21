# send — reference

## Arguments

- `message` — the work to hand over. It is self-contained: the addressee sees this text and nothing of your conversation.
- `addressee` — required, exactly one per call, from the ones your schema offers: `worker`, `worker_branch`, `manager`, `role` or `guest`. The command line also takes `assistant` (the owner's own assistant, which is what it means when the address is unnamed); no agent's copy of this tool has it.
- `place` — the folder the addressee works in, where your schema accepts it; a tool that does not accept it runs the work in your own workspace. For `worker` and `worker_branch`: a directory outside the managed workspaces (a managed workspace, anything inside one, the filesystem root and the product's data directory are refused); omit to run in your own workspace. For `manager`: required, the managed workspace to message by its exact name or its own folder. For `role`: a managed workspace, a directory path, or `personal:<user>`.
- `branch` — required for `worker_branch`: the conversation to continue, ASCII letters, digits, `_` and `-`, at most 32 characters. Created on first use; naming it again restores its history, and a later call may point it at another folder. For `role`: a branch of the role to continue, its history restored; omit for the role's own behaviour.
- `name` — required for `role` (the name of an existing custom role) and for `guest` (the guest's account name, exactly as it is written — no case or composition folding; your own account is refused).

The command line refuses an argument the chosen address does not take; a tool call ignores it, so pass an argument only to the address that takes it.

## What each addressee does

- `worker` — one sub-agent with full read/search/edit/shell reach, run once, whose report comes back as a `<worker-report>` naming the folder it worked in. A synchronous send (the Engineer's) returns that report in the call's result instead.
- `worker_branch` — a continuable worker conversation on `branch`, run as a normal worker. Its report returns to the conversation that asked, as a `<worker-report>`.
- `manager` — an internal message to the Manager of one managed workspace. The Manager sees it wrapped in an `<assistant-message from="<user>">` envelope naming the caller's user; users never see it. Its answer arrives as a `<manager-message>` naming its workspace.
- `role` — starts an administrator-defined custom role. Its answer arrives as a `<custom-role-message>` naming the role, its branch when it has one, and the folder it ran in.
- `guest` — another account's own assistant runs the message as that account, in its personal workspace. The recipient sees an `<owner-assistant-message one-way="no reply goes back through it">` envelope. Nothing — no answer, no notice, no failure — comes back.

## Sync and async

The Engineer dispatches a `worker` synchronously: the call returns the report, and nothing arrives later. The admin Assistant and custom roles dispatch asynchronously: the call returns before the work finishes and the answer is injected into this conversation later.

A message to a continuous role is handled one at a time, in the order sent, so a second never overtakes the first. A one-off send starts a fresh agent right away and runs in parallel. The command line's handover is written down before the caller is answered.

## Refusals

- A folder another worker or custom role is already working in: two agents in one folder overwrite each other's edits. Split the work across folders, or wait for the running one to finish.
- An address your tool does not offer, and an address's own missing argument.
- A `role` name no custom role answers to, and a `guest` name that is not an existing account (or is your own).

## Result

An acknowledgement naming where the work went — `Work handed to a worker in <folder>; its report arrives as a <worker-report> message.`, `Worker branch '<branch>' is on it in <folder>; …`, `Message delivered to the manager of workspace '<name>'.`, `custom role '<name>' launched on branch <agent> in <workspace>`, or the guest's one-way notice. Only a synchronous send returns the work's own result.

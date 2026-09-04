Read the recent conversation of a workspace's manager chat: user messages and
manager/assistant responses in chronological order. This is the user-facing
chat plane, not the Manager's internal session.

Parameters:
- `workspace` (string, required): name of the workspace whose chat to read.
- `limit` (integer, optional, default 5, max 50): how many recent messages to
  return. Messages can be long — prefer small limits and raise only if needed.

Each entry is prefixed with its author (`<user>: ...` for user messages,
`[<role>]: ...` for manager/assistant responses).

Send a message to the Manager agent of a project workspace. Use it to surface
findings, ask a product-level question on the user's behalf, or hand over
context the Manager needs. The message is delivered to the Manager as an
internal agent message attributed to you, and the workspace users also see it
in their chat history.

Parameters:
- `workspace` (string, required): name of the target workspace. Only
  project/shared workspaces have a Manager — personal workspaces cannot be
  targeted.
- `message` (string, required): the text to deliver. Make it self-contained:
  the Manager does not see your conversation with the user.
- `wait` (boolean, optional, default false): when `true`, your turn ends right
  after sending and you stay quiet until the Manager's reply arrives (delivered
  as a `<manager-reply>` internal message), a new user message, or an alarm
  wakes you. When `false`, you continue your current turn and the reply arrives
  later whenever the Manager responds.

Use `read_manager_chat` first when you need context on what was already said.

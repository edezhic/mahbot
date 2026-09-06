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

The Manager's responses are delivered to you automatically as internal <manager-message> messages.

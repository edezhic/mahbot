Register a workspace (a project directory to manage) and switch the admin's active workspace to it. Pass a short unique `name` (used in ticket ids and the GUI) and the absolute `path` to the project directory.

The workspace is then picked up automatically: if the LLM provider is configured, the pipeline claims and discovers it without further action.

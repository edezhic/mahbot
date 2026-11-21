Call one of the custom tools listed in the `<custom-tools>` context block: its `tool` name, and its arguments in `args`, keyed by the parameter names the block declares.

Arguments are validated against the declared types; a missing required one is refused, and a key the block does not declare is ignored and reported back with the run. The script's own output is returned, with the exit status when it is not clean.

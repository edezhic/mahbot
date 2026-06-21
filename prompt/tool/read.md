Read file contents with line numbers. Preferred over shell `cat` for reading files — it provides line numbers for reference, supports offset/limit for partial reads, and can list code structure via AST symbols.

When the path is a directory, the tool lists its contents (using `ls -lA`) instead of returning an error. The directory listing groups subdirectories and files with sizes and an extension summary. Note that `mode`, `offset`, and `limit` parameters only apply to file reads — they are silently ignored when a directory is passed.

Modes:
- `content` (default): Outputs an entire file or a range (using `offset`+`limit`) with line numbers. Handles any file type; binary files are read with lossy UTF-8 conversion. When the path is a directory, lists the directory contents.
- `symbols`: Lists all AST-level symbols (functions, structs, impl blocks, etc.) with line ranges. Supports Rust, JavaScript/TypeScript, and Python files. Use this to get a quick overview of a file's structure.
- `zoom`: Extract a single symbol's full source by name (requires `symbol` parameter). Use with the output of `symbols` mode to drill into specific definitions.

Path restrictions: paths must be within the project workspace, or within common dependency source directories (see below). Absolute paths are allowed for temp files (e.g. /tmp/* spill files from shell output) and dependency sources. Files larger than 10 MB are rejected. PDF files are automatically extracted to plain text.

## Dependency source access

The read tool can access dependency source code from common package manager cache directories, including:

- **Rust**: `~/.cargo/registry/src/`, `~/.cargo/git/checkouts/`
- **Python**: `~/.local/lib/`, `~/Library/Python/`, `/usr/local/lib/`, `/usr/lib/`, conda, poetry, pipenv, uv, rye directories
- **JavaScript/TypeScript**: bun, pnpm, npm global cache directories
- **Go**: `~/go/pkg/mod/`
- **Ruby**: `~/.gem/`, `~/.bundle/`
- **PHP**: `~/.composer/`
- **C/C++**: `~/.conan/`, `~/.conan2/`, Homebrew Cellar, Chocolatey, MSYS2/MinGW, Windows SDK, MSVC
- **Swift**: SwiftPM cache and Xcode DerivedData
- **Dart/Flutter**: `~/.pub-cache/`
- **Elixir/Erlang**: `~/.hex/`, `~/.mix/`
- **Haskell**: cabal, stack directories
- **Lua**: LuaRocks directories
- **R**: macOS/Linux/Windows R package libraries
- **OCaml**: `~/.opam/`
- **Julia**: `~/.julia/`
- **Nix**: `/nix/store/` (read-only)
- **System**: MacPorts (`/opt/local/`), pipx (`~/.local/pipx/`)

To discover the exact path for a specific dependency, use the shell tool's `ls` to list package directories (the search tool is workspace-scoped and won't find packages in dependency caches). For example: `ls ~/.cargo/registry/src/*/` to find all cached crate sources.
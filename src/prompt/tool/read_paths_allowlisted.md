Path restrictions: paths must be within the project workspace, or within common dependency source directories (see below). Absolute paths are allowed for temp files (e.g. spill files from shell output under `$TMPDIR` on unix, `%TMP%`/`%TEMP%` on Windows) and dependency sources.

## Dependency source access

The read tool can access dependency source code from common package manager cache directories, including:

- **Rust**: `~/.cargo/registry/src/`, `~/.cargo/git/checkouts/`, rustup toolchains `~/.rustup/toolchains/` (std sources)
- **Python**: `~/.local/lib/`, `~/Library/Python/`, `/usr/local/lib/`, `/usr/lib/`, conda, poetry, pipenv, uv, rye directories
- **Java/JVM**: Maven `~/.m2/repository/`, Gradle `~/.gradle/caches/` (caches only — not the whole `~/.gradle`), JDK headers via `$JAVA_HOME/include` or the system JVM locations (headers only)
- **JavaScript/TypeScript**: bun, pnpm, npm global caches, `~/.npm`, nvm/volta/yarn caches
- **Go**: module cache `~/go/pkg/mod/` (or `$GOMODCACHE`/`$GOPATH`), GOROOT sources (`$GOROOT/src`, `/usr/local/go`, Homebrew)
- **Ruby**: `~/.gem/`, `~/.bundle/`
- **PHP**: `~/.composer/`
- **C/C++**: `~/.conan/`, `~/.conan2/`, Homebrew Cellar, system + Homebrew headers (`/usr/include`, `/usr/local/include`, `include/`, `opt/`, `Frameworks/`), Chocolatey, MSYS2/MinGW, Windows SDK, MSVC, Xcode / Command Line Tools SDK roots
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

When `CARGO_HOME`, `RUSTUP_HOME`, `GOMODCACHE`, `GOPATH`, `GRADLE_USER_HOME`, `JAVA_HOME`, or `GOROOT` is set, the relocated root is honored alongside the HOME default. `XDG_CACHE_HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_STATE_HOME` are honored for the `~/.cache/`, `~/.config/`, `~/.local/share/`, `~/.local/state/` entries.

To discover the exact path for a specific dependency, list the package directory with the read tool (the search tool is workspace-scoped and won't find packages in dependency caches). For example, read `~/.cargo/registry/src` to find the cached crate sources.
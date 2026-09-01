Read file contents with line numbers. Preferred over shell `cat` for reading files — it provides line numbers for reference, supports offset/limit for partial reads, and can list code structure via AST symbols.

When the path is a directory, the tool lists its contents (using `ls -lA`) instead of returning an error. The directory listing groups subdirectories and files with sizes and an extension summary. Note that `mode`, `offset`, and `limit` parameters only apply to file reads — they are silently ignored when a directory is passed.

Modes:
- `content` (default): Outputs a file or a range (using `offset`+`limit`) with line numbers. Large outputs are truncated to a small budget (~5 KB) — for big files, read in slices with `offset`/`limit`, or navigate via `symbols`/`zoom`. Handles any file type; binary files are read with lossy UTF-8 conversion. When the path is a directory, lists the directory contents.
- `symbols`: Lists all AST-level symbols (functions, structs, impl blocks, etc.) with line ranges — the quickest way to map a large file's structure without reading it whole. Works for supported code formats only (Rust, JS/TS, Python, Go, C, Ruby, SQL, Markdown, JSON, TOML, CSS, HTML, shell), not arbitrary formats.
- `zoom`: Extract a single symbol's full source by name (requires `symbol` parameter). Same supported-formats limit as `symbols`. Use with the output of `symbols` mode to drill into specific definitions.

When a content-mode path is a raster image (PNG, JPEG, or WebP), the tool reads and attaches it to the conversation as a native image the model can inspect, instead of returning lossy text. Other binary formats that cannot be decoded (e.g. GIF, BMP, HEIC) are reported as unsupported. Reading the same image twice returns a reference to the already-attached image rather than adding it again.

Path restrictions: paths must be within the project workspace, or within common dependency source directories (see below). Absolute paths are allowed for temp files (e.g. $TMPDIR/* spill files from shell output) and dependency sources. Files larger than 10 MB are rejected.

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

To discover the exact path for a specific dependency, use the shell tool's `ls` to list package directories (the search tool is workspace-scoped and won't find packages in dependency caches). For example: `ls ~/.cargo/registry/src/*/` to find all cached crate sources.

## Protected credentials

Some paths are denied even though they exist, with the distinct "Path is a protected credential location" error — `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gcloud`, `~/.docker`, `~/.kube`, and private-key files (`id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `*.ppk`) anywhere, including inside the workspace. Credential-bearing config files (`.env`, `.pem`/`.cer`/`.crt` certs, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, Maven `settings.xml`/`settings-security.xml`, Gradle `gradle.properties`/`init.gradle*`, cargo `credentials.toml`) are scrubbed for credentials rather than denied — where they are readable at all (workspace files or paths inside the allowlisted roots); e.g. `~/.m2/settings.xml` and `~/.gradle/gradle.properties` sit outside the read allowlist and are simply not readable.
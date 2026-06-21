You are analyzing a workspace to identify its dev tooling commands. Scan build files, config files, README, CI configs, and directory structure to discover the shell commands used for formatting, linting, type-checking, building, and testing.

Prefer documented package scripts, Makefile targets, and CI workflow commands over inventing raw tool invocations. Confirm tooling exists in project files before including a command. Distinguish absent tooling from tooling that exists only for heavy/integration workflows.

## What to Find

For each category below, identify the minimal shell command that works from the workspace root. If no such tooling exists for a category, note it as absent.

1. **format** — auto-formatter command (e.g., `cargo fmt`, `prettier --write .`, `black .`)
2. **format-check** — format verification without changes (e.g., `cargo fmt -- --check`, `prettier --check .`, `black --check .`)
3. **lint** — linter, idempotent (e.g., `cargo clippy -- -D warnings`, `eslint .`, `ruff check .`)
4. **lint-fix** — auto-fix lint issues (e.g., `cargo clippy --fix --allow-dirty`, `eslint --fix .`, `ruff check --fix .`)
5. **type-check** — type checking without full compilation (e.g., `cargo check`, `tsc --noEmit`, `mypy .`)
6. **build** — full build command (e.g., `cargo build`, `npm run build`, `make`)
7. **unit-test** — run unit tests, fast (e.g., `cargo test`, `pytest`, `npm test`)

## Multi-Language Projects

If multiple build systems coexist (e.g., Rust + Python, Node + Go), the command for each category must be a **compound command** that covers ALL languages in a single invocation. Use `&&` to chain them:

- `cargo test && pytest` (not two separate entries)
- `cargo clippy -- -D warnings && ruff check .` (not two separate entries)

The goal is a single command per category that validates the entire workspace.

## What NOT to Include

- Heavy/integration tests (unit tests only)
- E2E tests
- Docker/container orchestration
- Deployment scripts
- Benchmarks
- Code generation steps

## Exploration Steps

1. Read build system files: Cargo.toml, package.json, pyproject.toml, go.mod, Makefile, CMakeLists.txt, etc.
2. Read CI configs: .github/workflows/, .gitlab-ci.yml, Jenkinsfile, etc. — these often contain the canonical commands.
3. Read README.md, CONTRIBUTING.md, BUILDING.md for documented commands.
4. Read config files for formatters/linters: .rustfmt.toml, .prettierrc, .eslintrc, pyproject.toml [tool.ruff], etc.
5. Check for lock files: Cargo.lock, package-lock.json, yarn.lock, poetry.lock, Pipfile.lock, go.sum.
6. Verify commands would actually work — check that required files exist (e.g., Cargo.toml exists for `cargo` commands).
7. Search the web for official tool documentation when command flags or version-specific behavior is unclear. Anchor to versions from project files and lockfiles.

## Exit Code Convention

All discovered commands must exit 0 on success and non-zero only when something actually went wrong. Some common tools violate this (e.g., `diff` exits 1 when files differ, `grep` exits 1 when no matches), so use normalization flags where needed (e.g., `--check` instead of `diff`, or `|| true` for optional checks). The `&&` chaining pattern for compound commands assumes this convention.

Do NOT output JSON during exploration. The extraction prompt will request a structured JSON object in a subsequent pass.

# MahBot

Mahbot is an agentic development environment built for **reliability**. It treats software work as a managed pipeline, not a chat session: you talk to a **Manager** about product intent and scope; specialist agents research, implement, review and test. Reliability comes from **orchestration and process**, not from betting that the current frontier model will one-shot the any task. It also includes a background maintenance process to clean up usual videcoding bloat and other code quality issues.

Mahbot packages the best practices into one system: subagents, persistent state, deterministic diagnostics, adversarial review and QA, fix & maintenance loops. It is designed to make inexpensive API/local models useful through structure instead of relying on subsidized subscriptions.

Includes a smooth native GUI that lets you easily inspect the agents, diffs, code, as well as basic editor capabilities and a built-in shell. Also packs a fully-local speech-to-text & text-to-speech voice control. Passive wake-word detection is unstable yet but it is the current development focus.

## Getting Started

Currently there are two ways to start using mahbot:

**Install from crates.io**:

```bash
cargo install mahbot
```

OR

**Build from source**:

```bash
git clone https://github.com/edezhic/mahbot
cd mahbot
cargo run --release
```

Then run `mahbot` to start the dashboard and configure your OpenRouter key in **Settings**. OpenRouter API key and the [`chrome-use`](https://github.com/leeguooooo/chrome-use) CLI (browser tool and link enrichment) are needed for full functionality — see [Prerequisites](#prerequisites) below. Also, the same binary can be run with `mahbot debug ...` to execute read-only SQL queries over the service's DBs, which is particularly useful for agents working on mahbot itself.

As of now mahbot is only regularly tested on macos, so it might have unexpected bugs on other platforms. However, all the core components are cross-platform so it should work just fine on windows & linux in the future.

## Why not just Claude Code, Codex, Pi or Cursor?

Most coding assistants optimize for **interactive pair programming**: you prompt, the model edits, you review. That works for focused tasks, but autonomous work tends to drift into low-level implementation details, skip verification, or hit subscription limits when agents run for hours. Mahbot is designed for the real-world development from the ground up:

### Product-focused Manager

You talk to a **Manager** that owns intent, scope, tickets, and progress—not an implementation agent that keeps derailing into low-level details. The Manager creates and refines work on a ticket board, delegates research to Analysts asynchronously, and only escalates real product decisions to you. This allows manager to easily fit large chunks of work across many tickets into the context without compaction and remain focused on the desired goals.

### Mandatory validation pipeline

Every ticket runs through a fixed lifecycle with **redundant checks**:

| Phase | What happens |
|-------|----------------|
| **Backlog → Analysis** | Parallel analysts research and score the ticket |
| **Analysis → Planning** | Manager notified; moves the ticket into ready for development once the scope is confirmed |
| **Ready → In development** | Engineer implements sequentially, using subagents when needed |
| **In diagnostics** | Discovered project commands run (format, lint, build, test) |
| **Diagnostics done → In review** | Parallel reviewers; bounces back into dev with the feedback if issues are found |
| **Reviewed → In QA** | Parallel QA agents; same bounce mechanics |
| **QA passed → Sanitation** | Check for untracked/new files in the working tree. If found → dispatch **Sanitation** agent |
| **Sanitation passed → Done** | Auto git commit with the ticket's title if the tree is dirty |

Circuit breakers pause the ticket's workspace if a ticket goes through too many bounces, escalate to the manager and he handles from there. This matches what SOTA agentic systems are converging on: **execution-grounded verification** and separate agents for validation. Also includes:

- **Workspace discovery** — per-role codebase summaries + auto-detected diagnostics commands. No need for AGENTS/CLAUDE/other md files.
- **Background Maintainer** — scans for refactor opportunities and creates **planning tickets only** (no silent edits)
- **Built-in dev surface** — native dashboard: chat, ticket board, editor, diff, shell, sessions, logs, tool-failure stats, settings; optional **Telegram** channel on the same pipeline
- **Archived ticket search** — hybrid FTS + local embeddings, Manager can efficiently search over all the past work

## Prerequisites

**Required:**

- OpenRouter API key (or compatible endpoint configured in settings)
- [`chrome-use`](https://github.com/leeguooooo/chrome-use) CLI (browser tool and link enrichment)
  Install: `curl -fsSL https://raw.githubusercontent.com/leeguooooo/chrome-use/main/install.sh | sh`
  (Windows users: download the `.exe` from [Releases](https://github.com/leeguooooo/chrome-use/releases))
  Requires the [Chrome extension](https://chromewebstore.google.com/detail/chrome-use) and native messaging host
  (run `chrome-use extension install` after installing the binary)

**Optional:**

- Firecrawl API key — enables `web_search` tool
- Telegram bot token — remote chat on the same agent backend

**Defaults (configurable):** per-role models via OpenRouter; image/video generation and transcription models in settings. See `src/config.rs` and the dashboard **Settings** page.

## Known issues

This is beta quality software. It has already processed thousands of tickets on it's own codebase and improving every day, and I already prefer it over Cursor + GPT 5.5 xhigh despite having expiring credits left there. However, it heavily uses Turso which is in beta itself and multiple times the board DB got corrupted, fff-search seems to reindex workspaces way more than required, chrome-use is a somewhat questionable choice, and most likely there are still some bugs left. Nevertheless, it's already very much usable with the default configs.

## License

MIT OR Apache-2.0 — see `Cargo.toml`.
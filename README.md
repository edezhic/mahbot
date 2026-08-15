# MahBot

Mahbot(i.e. __my bot__) is an agentic development environment built for **reliability** and **autonomy**. It treats software work as a managed pipeline, not a chat session: you talk to the **Manager** about intent and scope; specialist agents research, implement, review and test. The Manager creates and refines work on the ticket board and only escalates real product decisions to you. Efficiency comes from **orchestration and process**, not based on the expectation that the current frontier model will one-shot any task. 

Batteries included:
- **Smooth native GUI** for the core pipeline management as well as code editor, diff viewer and shell
- **Modern agentic stuff**: subagents, adversarial review and QA, deterministic diagnostics
- **Telegram bot** integration that allows you to easily manage the work from your smartphone
- **Voice control** using a CPU-optimized local speech-to-text model that turns babble into features (passive wake-word detection wip)
- **Background maintenance** process to clean up the usual videcoding bloat as well as other code quality issues
- **Full history** of the previous work in the tickets with efficient hybrid search over it
- **Out-of-the-box** workspace discovery for per-role contexts, auto-detected diagnostics commands. No need for plugins, AGENTS/CLAUDE/other.md files or custom configurations. Just add the API key and state your wishes
- **Specialized artist** agent for image & video generation/editing as a little treat on top

At the moment only supports OpenRouter as the provider, and by default configured to use the DeepSeek 4 Flash. More providers will definitely be added in the future, but as of now that's the best price/performance/simplicity combo known to me. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

## The Pipeline

Every ticket has a lifecycle with **redundant checks**:

| Phase | What happens |
|-------|----------------|
| **Backlog → Analysis** | Parallel analysts research and score the ticket's scope & premises |
| **→ Planning** | Manager checks the analysis results and either refines, cancels, moves into dev or asks the user |
| **→ Ready for dev** | Awaits in the engineer's queue according to it's priority |
| **→ In development** | Engineer implements the ticket or the required fixes |
| **→ In diagnostics** | Deterministic verification (format, lint, build, test) |
| **→ In review** | Agentic verification focused on the code quality |
| **→ In QA** | Agentic verification focused on the ticket's expectations |
| **→ Sanitation** | Check untracked/new files in the working tree |
| **→ Done** | Auto git commit with the ticket's title if the tree is dirty |

Circuit breaker pauses the work if a ticket goes through too many bounces, escalates to the manager and he handles from there.

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

Then run `mahbot` to start the dashboard and configure your OpenRouter key in **Settings**. OpenRouter API key and the [`chrome-use`](https://github.com/leeguooooo/chrome-use) CLI (browser tool and link enrichment) are needed for full functionality — see [Prerequisites](#prerequisites) below. Also, the same binary can be run with `mahbot debug ...` to execute read-only SQL queries over the service's DBs, which is particularly useful for the agents working on mahbot itself.

As of now mahbot is only regularly tested on macos, so it might have unexpected bugs on other platforms. However, all the core components are cross-platform so it should work just fine on windows & linux in the future.

## Prerequisites

**Required:**

- OpenRouter API key
- [`protoc`](https://grpc.io/docs/protoc-installation/) (Protocol Buffers compiler) — the ONNX voice-pipeline dependency (candle-onnx-mahbot) compiles `onnx.proto3` at build time, so it is required for `cargo install mahbot` and source builds. Install: `brew install protobuf` (macOS) or your distro's `protobuf` package
- [`chrome-use`](https://github.com/leeguooooo/chrome-use) CLI (browser tool and link enrichment)
  Install: `curl -fsSL https://raw.githubusercontent.com/leeguooooo/chrome-use/main/install.sh | sh`
  (Windows users: download the `.exe` from [Releases](https://github.com/leeguooooo/chrome-use/releases))
  Requires the [Chrome extension](https://chromewebstore.google.com/detail/chrome-use) and native messaging host
  (run `chrome-use extension install` after installing the binary)

**Optional:**

- Exa and/or Firecrawl API keys — for the `web_search` tool
- Telegram bot token — remote chat on the same agent backend
# MahBot

Mahbot(i.e. __my bot__) is an agentic development environment built for reliability and autonomy. It treats software work as a managed pipeline, not a chat session: you talk to the **Manager** about intent and scope; specialist agents do the actual work. Manager creates and refines tickets on the board, and only escalates real product decisions to you. 

**Reliability** comes from orchestration and process, not based on the expectation that the current frontier model will one-shot any task. 

**Autonomy** is achieved using the pipeline - you can request a large amount of work and manager will ensure that every piece is analyzed, implemented, reviewed, tested and commited.

Batteries included:
- __Smooth native GUI__ for the core pipeline management as well as code editor, diff viewer and shell
- __Modern agentic stuff__: subagents, adversarial review and QA, deterministic diagnostics
- __Telegram bot__ integration that allows you to easily manage the work from your smartphone
- __Voice control__ using a CPU-optimized local speech-to-text model that turns babble into features (passive wake-word detection wip)
- __Background maintenance__ process to clean up the usual videcoding bloat as well as other code quality issues
- __Full history__ of the previous work in the tickets with efficient hybrid search over it
- __Out-of-the-box__ workspace discovery for per-role contexts, auto-detected diagnostics commands. No need for plugins, AGENTS/CLAUDE/other.md files or custom configurations. Just add the API key and state your wishes
- __Specialized artist__ agent for image & video generation/editing as a little treat on top

OpenRouter is the default provider, and by default mahbot is configured to use the DeepSeek 4 Flash. A custom self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, or alike) can be configured in Settings for chat requests. Note that artists tools are currently tied to OpenRouter — so its key is still needed for those even when a custom endpoint handles dev agents. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

## The Pipeline

Every ticket has a lifecycle with **redundant checks**:

| Phase | What happens |
|-------|----------------|
| **→ Analysis** | Parallel analysts research the ticket's assumptions & scope |
| **→ Planning** | Manager sees the analysis and refines/cancels/approved or escalates |
| **→ Queued** | Awaits in the engineer's queue according to it's priority |
| **→ Development** | Engineer implements the ticket (or the required fixes) |
| **→ Diagnostics** | Deterministic verification (format, lint, build, test) |
| **→ Review** | Agentic verification focused on the code quality |
| **→ QA** | Agentic verification focused on the product behaviour |
| **→ Sanitation** | Audit untracked/new files in the working tree |
| **→ Done** | Auto git commit with the ticket's title if the tree is dirty |

Circuit breaker pauses the work if a ticket goes through too many bounces, escalates to the manager and he handles from there.

## Getting Started

Currently mahbot can only be installed from `crates.io`:

```bash
cargo install mahbot
```

Then run `mahbot` to start the service, and you'll be asked to configure:

**Required (one of):**
- OpenRouter API key, or
- a custom OpenAI-compatible endpoint

**Optional:**
- Telegram bot token — comfortably direct mahbot from your phone using text, voice msgs & built-in commands
- Exa and/or Firecrawl API keys — for the `web_search` tool, so that agents can easily search the docs & other information
- [`chrome-use`](https://github.com/leeguooooo/chrome-use) for reliable browser automation. Install: `curl -fsSL https://raw.githubusercontent.com/leeguooooo/chrome-use/main/install.sh | sh` (Windows users: download the `.exe` from [Releases](https://github.com/leeguooooo/chrome-use/releases)). Requires also the chrome extension - run `chrome-use extension install` after installing the binary.

As of now mahbot is only regularly tested on macos, so it might have unexpected bugs on other platforms. However, all the core components are cross-platform so it should work just fine on linux & windows in the future. Also, the same binary can be run with `mahbot debug -h` to execute read-only SQL queries over the service's DBs in an agent-friendly way.
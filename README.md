# MahBot

Mahbot(i.e. __my bot__) is an agentic system that automates coding while also providing built-in tools for everyday automations, research, as well as media editing and generation.

Mahbot treats software development as a managed pipeline, not a chat session: you drop a request to your assistant about any of the projects while it orchestrates other agents to clarify the details and only escalates real product decisions to you.

**Reliability** comes from orchestration and process, not based on the expectation that the current frontier model will one-shot any task. 

**Autonomy** is achieved using the pipeline - you can request a large amount of work and agents in the pipeline will ensure that every piece is analyzed, implemented, reviewed, tested and commited.

Batteries included:
- __Smooth native GUI__ for the core pipeline management as well as code editor, diff viewer and shell
- __Telegram bot__ integration that allows you to easily manage the work from your smartphone
- __Voice control__ using a CPU-optimized local speech-to-text model that turns babble into features (passive wake-word detection wip)
- __Modern agentic__ adversarial analysis before dev, review and QA after dev
- __Good old deterministic__ CI-style diagnostics after every dev round
- __Background maintenance__ agentic process to clean up the usual videcoding bloat and other code quality issues
- __Full history__ of the previous work in the tickets with efficient hybrid search over it
- __Out-of-the-box__ workspace discovery for per-role contexts, auto-detected diagnostics commands. No need for plugins, AGENTS/CLAUDE/other.md files or custom configurations. Just add the API key, select the workspace and state your wishes
- __One to rule them all__ adaptive agent for automations, Q&A, research, prototyping, image & video generation/editing and other purposes

OpenRouter is the default provider; manager-side roles default to GLM 5.3 Flash and worker-side roles to DeepSeek 4 Flash. A custom self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, or alike) can be configured in Settings for chat requests. Note that media tools are currently tied to OpenRouter — so its key is still needed for those even when a custom endpoint handles dev agents. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

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

Then run `mahbot` to start the service, and you'll be asked to provide one of:
- OpenRouter API key, or
- a custom OpenAI-compatible endpoint

The rest of the setup will be explained and done through the agent. It will help you add a workspace, other users, connect the Telegram bot, add the search providers & browser tooling for your agents.

Beware that as of now mahbot is only regularly tested on macos & linux, so it might still have unexpected bugs on other platforms.

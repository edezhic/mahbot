<p align="center">
  <img src="assets/mb-icon-256.png" width="64" alt="MahBot">
</p>

# MahBot

Mahbot(i.e. __my bot__) is an agentic system that automates coding, office work, research, media editing and generation, while also providing generic agentic tools for your custom workflows.

Batteries included:
- __Tightly optimized__ LLM cache and device resources usage for lower costs and wider support
- __Telegram bot__ integration that allows you to easily manage the work from your smartphone
- __Voice control__ (currently macOS only) using local speech-to-text model with wake-word detection
- __Smooth native GUI__ for the core pipeline management as well as code editor, diff viewer and shell
- __Modern agentic dev__ process with analysis before and adversarial verification after every change
- __Background maintenance__ agentic process to clean up the usual videcoding bloat and other code quality issues
- __Full change history__ of the previous work in the tickets with efficient hybrid search over it
- __Out-of-the-box__ workspace discovery for per-role contexts, auto-detected diagnostics commands. No need for plugins, AGENTS/CLAUDE/other.md files or custom configurations. Just add the API key, select the workspace and state your wishes
- __One to rule them all__ adaptive agent for automations, Q&A, research, prototyping, image & video generation/editing and other purposes

OpenRouter is the default provider, and DeepSeek V4.1 Flash is the default model. A custom self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, or alike) can be configured in Settings for chat requests. Note that media generation/editing tools are currently tied to OpenRouter — so its key is still needed for them even when a custom endpoint is used for the agents. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

## Software development

Mahbot treats software engineering as a managed pipeline, not a chat session: you drop a request to your assistant about any of the projects while it orchestrates other agents to clarify the details and only escalates real product decisions to you.

**Reliability** comes from orchestration and process, not based on the expectation that the current frontier model will one-shot any task. 

**Autonomy** is achieved using the pipeline - you can request a large amount of work and agents in the pipeline will ensure that every piece is analyzed, implemented, verified and commited.

Every request/ticket has a lifecycle with **redundant checks**:

| Phase | What happens |
|-------|----------------|
| **→ Analysis** | Parallel analysts research the ticket's assumptions & scope |
| **→ Planning** | Manager sees the analysis and refines/cancels/approved or escalates |
| **→ Queued** | Awaits in the engineer's queue according to it's priority |
| **→ Development** | Engineer implements the ticket (or the required fixes) |
| **→ Verification** | Deterministic checks with code reviewers and a tester |
| **→ Sanitation** | Audit untracked/new files in the working tree |
| **→ Done** | Auto git commit with the ticket's title if the tree is dirty |

Circuit breaker pauses the work if a ticket goes through too many bounces, escalates to the manager and he handles from there.

## Getting Started

Install and start it with one command. On unix-like systems:

```sh
curl -fsSL https://raw.githubusercontent.com/edezhic/mahbot/main/install.sh | sh
```

On Windows (PowerShell):

```sh
irm https://raw.githubusercontent.com/edezhic/mahbot/main/install.ps1 | iex
```

This script will install the latest release and start it for you. Further updates will be installed & restarted automatically, with very little downtime for your agents. If you would rather build it yourself, the toolchain way is the alternative:

```bash
cargo install mahbot
```

Either way the product's window opens and asks you to provide one of:
- OpenRouter API key, or
- a custom OpenAI-compatible endpoint

The rest of the setup will be explained and done through the agent. It will help you add a workspace, other users, connect the Telegram bot, add the search providers & browser tooling for your agents.

<p align="center">
  <img src="assets/mb-icon-256.png" width="64" alt="MahBot">
</p>

# MahBot

Mahbot(i.e. __my bot__) is an agentic system that automates coding, design and office work, while also providing adaptive setup for your custom integrations and workflows. Batteries included:

- __One to rule them all__ assistant for automations, research, prototyping, and many other purposes
- __Others to help him out__ specialized and custom agents that the assistant can delegate tasks to and remain responsive for any volume of your requests
- __Work from anywhere__ through a Telegram bot, smooth GUI or voice control with full sync between their states, without context switches for you
- __Work with anything__ using the custom tools & integrations that the agents can build themselves on-the-fly to serve your needs
- __Tightly optimized__ LLM cache and device resources usage for lower costs and wider support
- __Out-of-the-box__ without skills, plugins, AGENTS/CLAUDE/other.md files or manual configurations

OpenRouter is the default provider, and DeepSeek V4.1 Flash is the default model. A custom self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, or alike) can be configured in Settings for chat requests. Note that media generation/editing tools are currently tied to OpenRouter — so its key is still needed for them even when a custom endpoint is used for the agents. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

# Getting Started

Install and start it with one command. On unix-like systems:

```sh
curl -fsSL https://raw.githubusercontent.com/edezhic/mahbot/main/install.sh | sh
```

On Windows (PowerShell):

```sh
irm https://raw.githubusercontent.com/edezhic/mahbot/main/install.ps1 | iex
```

This script will install the latest release and start it for you. Further updates will be installed & restarted automatically, with graceful pause&resume and very little downtime for your agents. After the install mahbot asks you to provide one of:
- OpenRouter API key, or
- a custom OpenAI-compatible endpoint

The rest of the onboarding will be explained and done through the assistant.

# Software development

Mahbot treats software engineering as a managed pipeline, not a chat session: you drop a request to your assistant about any of the projects while it orchestrates other agents to clarify the details and only escalates real product decisions to you.

The best modern agentic dev practices built-in: scope analysis before and adversarial verification after every change, workspace discovery for per-role contexts, automatically detected & executed diagnostics commands, background maintenance process, and many more.

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
| **→ Done** | Auto git commit with the ticket's title |

Circuit breaker pauses the work if a ticket goes through too many bounces, escalates to the manager and he handles from there.

# Command line

MahBot's capabilities are also available from a terminal, so your own scripts and custom tools can use them: manage accounts, users, custom-tool grants and custom roles; list and add managed workspaces, pause or resume a pipeline, read the board or a ticket and comment on it; hand a message to any agent role; read a document with your assistant's own reader — a PDF, a Word, Excel or PowerPoint file (modern or old binary), or any other file — its full text and every image it produces written into a folder of their own; produce documents (Word, Excel, PowerPoint and PDF — creating, filling your own template, editing, merging, splitting, rotating, cropping, resizing, rendering pages, filling forms), pictures and video; transcribe speech and turn text into audio; drive the browser you are already logged in to; and run read-only queries against MahBot's own live stores with `mahbot debug`. Produced files land where you ask and nothing you already have is overwritten. Most commands work against the running product; speech, the browser and debug work on their own. Every command reports its outcome in its exit code; run `mahbot -h`, then `<command> -h`, for the operations and flags.

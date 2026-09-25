# MahBot

Mahbot(i.e. __my bot__) is an agentic system that automates coding while also providing built-in tools for everyday automations, research, as well as media editing and generation.

Mahbot treats software development as a managed pipeline, not a chat session: you drop a request to your assistant about any of the projects while it orchestrates other agents to clarify the details and only escalates real product decisions to you.

**Reliability** comes from orchestration and process, not based on the expectation that the current frontier model will one-shot any task. 

**Autonomy** is achieved using the pipeline - you can request a large amount of work and agents in the pipeline will ensure that every piece is analyzed, implemented, reviewed, tested and commited.

Batteries included:
- __Smooth native GUI__ for the core pipeline management as well as code editor, diff viewer and shell
- __Telegram bot__ integration that allows you to easily manage the work from your smartphone
- __Voice control__ (macOS only) using a CPU-optimized local speech-to-text model that turns babble into features (passive wake-word detection wip)
- __Modern agentic__ adversarial analysis before dev, review and QA after dev
- __Good old deterministic__ CI-style diagnostics after every dev round
- __Background maintenance__ agentic process to clean up the usual videcoding bloat and other code quality issues
- __Full history__ of the previous work in the tickets with efficient hybrid search over it
- __Out-of-the-box__ workspace discovery for per-role contexts, auto-detected diagnostics commands. No need for plugins, AGENTS/CLAUDE/other.md files or custom configurations. Just add the API key, select the workspace and state your wishes
- __One to rule them all__ adaptive agent for automations, Q&A, research, prototyping, image & video generation/editing and other purposes

OpenRouter is the default provider; manager-side and worker-side roles default to DeepSeek V4.1 Flash. A custom self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, or alike) can be configured in Settings for chat requests. Note that media tools are currently tied to OpenRouter — so its key is still needed for those even when a custom endpoint handles dev agents. Also, should work quite well with smaller models like Qwen 3.8 27b, and local + open-source mode is the primary long-term focus.

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

Install and start it with one command. On unix-like systems:

```bash
curl -fsSL https://raw.githubusercontent.com/edezhic/mahbot/main/install.sh | sh
```

On Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/edezhic/mahbot/main/install.ps1 | iex
```

That fetches the newest published release, puts it in the standard per-user location, adds that location to your search path only when it is missing, and starts the product. From then on mahbot updates itself from those same published releases, so there is nothing to do by hand. `install.sh <version>` installs one particular release instead of the newest; with the piped form above, set `MAHBOT_INSTALL_VERSION` to that version instead, because a piped script takes no argument of its own — that is how a release is installed by hand, since the newest-release lookup never names a test release.

If you would rather build it yourself, the toolchain way is the alternative:

```bash
cargo install mahbot
```

Either way the product's window opens and asks you to provide one of:
- OpenRouter API key, or
- a custom OpenAI-compatible endpoint

The rest of the setup will be explained and done through the agent. It will help you add a workspace, other users, connect the Telegram bot, add the search providers & browser tooling for your agents.

Mahbot supports macOS 12.3 and newer (Intel and Apple silicon), Linux on x86_64 and ARM against glibc 2.35 or newer, Windows 10 1809 and newer on x86_64, and Windows 11 on ARM. A system with no published file for it is refused plainly by the install command — musl Linux, older macOS and older glibc, and Windows 7, 8 and 10 on ARM among them. The Windows files are built and published like the others, but they have not been run by this project, so they ship what compiles rather than what has been exercised.

### Audio is macOS only

Everything local about audio — the microphone button, wake-word detection and enrolment, voice commands, local transcription of incoming voice messages, read-aloud replies and audio cues — exists **only on macOS**. The speech engine behind it has no other platform support and nothing here has a fallback, so on Linux and Windows mahbot builds with no audio capability whatsoever: no audio dependency is compiled in, and nothing audio-shaped is offered — no microphone button, no recording popup, no audio section on the settings page, no voice status anywhere.

On those platforms an incoming **voice message** is refused before anything is downloaded: the sender gets a notice saying voice messages are not supported there, and the agent gets an honest note instead of the contentless marker. A plain **sound file** is not refused — it is saved and handed to the agent like any other attachment. Nothing is transcribed from audio and there is no cloud transcription fallback.

Any local model directory an earlier release downloaded on those platforms (`~/.mahbot/models/qwen3-asr-0.6b/` and `~/.mahbot/models/supertonic3/` on unix, the same paths under `%USERPROFILE%\.mahbot\` on Windows) is simply unreachable now. Nothing deletes it and no stored setting is purged; if you want the disk space back, remove those directories by hand.

### Building from source on Windows

Building mahbot there from source needs a C/C++ toolchain plus the Windows SDK it links against (the Visual Studio Build Tools `Desktop development with C++` workload provides both), because several dependencies (ring, zstd-sys, libz-sys, onig_sys, libgit2-sys) compile C during the build. The product is a windowed program, so starting it opens no console window — its own window is the one it shows — and a failed start is recorded in the durable `error.log` block in the storage root. The product's own subcommands are the exception: the daemon starts those with their streams wired and waits for them, so their output reaches the agent.

Our own source type-checks, lints and compiles for Windows — `scripts/windows-cross-check.sh` in the repository verifies that by hand (dev-only tooling, not part of the published crate). That check only compiles: nothing is linked into a runnable binary and nothing is executed.

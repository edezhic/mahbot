# MahBot

Experimental harness aiming at reliable autonomous development with minimal supervision while heavily using cheap models. As of now integrated with telegram as primary user interface, and includes a native GUI dashboard if you want to see the details.

## Features

- You are talking to the manager; he orchestrates ticket-based pipeline with pre- and post-dev verification stages
- Background maintainer always looking for refactoring opportunities

## Goals

* Reach the point where I don't need to open IDE while working on the MahBot itself, BUT without any special setup in the mahbot repo like tuned agents.md, specific skills etc. And code quality should be kept at least ~fine.
* Generalize the rest of the rust-specific treatment and make it work with any project.
* More tools, more agents, more edge cases covered. As self-contained as possible, require only LLM and web search providers.

## Non-goals 

- Support for many providers / channels. Mahbot should be able to add support for the ones you need in ~1$ worth of tokens in a single prompt. I don't want to bloat the codebase with tons of integrations and their quirks, at least for now.
- Deployment / release automation. Even frontier models can easily screw up. Even after tons of reviews, linters and tests. Mahbot isn't supposed to replace engineers, it's aiming to be a reliable assistant in the development process.
- One-size-fits-all agent. While mahbot already has a special artist agent that has nothing to do with regular development, and will likely include more specialied agents, it doesn't aim to include magic AGI that solves completely arbitrary tasks.

## Presumptions

1. **Tokens are cheap.** Doesn't mean that all tokens are cheap, but you can run a model capable of writing code and testing it even locally on consumer hardware. As of now I'm (ab)using DeepSeek v4 Flash, and burning a billion of tokens costs about 10$. Should probably work with even smaller models like Qwen 3.6 or Gemma 4. You'd be right saying that these models don't have as much general knowledge - that's where web browser + search tools save the day.
2. **Even frontier models aren't 100% reliable.** You don't want to just pick the best model and blindly push the code to prod. You probably want at least one more agent to review the changes. Mahbot spawns at least 3 reviewers and 3 QA agents for each ticket, even for a single line change. And 3 analysts before a ticket even goes into development. And a manager to keep track of the project goals. And a maintainer to refactor the slop. And many agents can spawn additional analysts to get deeper understanding without bloating the context. And most likely there will be more.
3. Decent tokens will get even cheaper but even GPT 10 won't be able to one-shot all real-world engineering tasks. So, the optimal performance would most likely be achieved through orchestration of multiple cheaper agents.


# Prerequisites
Required:
- OpenRouter token
- `agent-browser` CLI

Optional:
- Exa token (to enable the web search tool)

# ForestCode

[中文](https://github.com/hsghcop/ForestCode/blob/main/README.md)

[![PyPI](https://img.shields.io/pypi/v/forestcode)](https://pypi.org/project/forestcode/)
[![Python](https://img.shields.io/pypi/pyversions/forestcode)](https://pypi.org/project/forestcode/)
[![License](https://img.shields.io/pypi/l/forestcode)](https://github.com/hsghcop/ForestCode/blob/main/LICENSE)

ForestCode is a terminal Coding Agent written in Python. It combines a language model, code tools, session memory, and terminal interaction into an Agent runtime that can work inside real code repositories.

ForestCode does not depend on Agent frameworks such as LangChain or LangGraph. Its core loop, tool system, context, memory, Patch-first editing, and Subagent scheduling are implemented by the project itself.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- macOS, Linux, or WSL
- Access to a DeepSeek or OpenAI-compatible API

## Installation

ForestCode is published on [PyPI](https://pypi.org/project/forestcode/). Install it as a global terminal tool with uv:

```sh
uv tool install forestcode
```

After installation, run it from any directory:

```sh
forestcode
```

## API Configuration

On first run, ForestCode creates the following file if it does not exist:

```text
~/.forestcode/settings.json
```

Open the file, provide your API key, model, and endpoint, then run `forestcode` again.

DeepSeek example:

```json
{
  "api_key": "your_api_key",
  "model": "deepseek-v4-pro",
  "base_url": "https://api.deepseek.com",
  "api_type": "deepseek",
  "timeout": 60,
  "max_turns": 10
}
```

OpenAI-compatible example:

```json
{
  "api_key": "your_api_key",
  "model": "your_model",
  "base_url": "https://your-provider.example/v1",
  "api_type": "openai-compatible",
  "timeout": 60,
  "max_turns": 10
}
```

`settings.json` supports whole-line `//` comments, but not inline comments. A `.env` file in the current directory can override standard model and tuning values, while CLI arguments have the highest priority. Never commit a real API key to Git.

## Getting Started

Enter the repository you want to work on and start ForestCode:

```sh
cd /path/to/your-project
forestcode
```

The current directory becomes the default workspace. You can then enter tasks such as:

```text
Read this project and explain its entry point and main modules.
Find the cause of the failing tests without modifying any files.
Fix this issue and show me the diff before applying it.
```

Common startup options:

| Option | Purpose |
| --- | --- |
| `--workspace <path>` | Set the workspace; defaults to the current directory |
| `--session <id>` | Use a named session; defaults to `default` |
| `--no-session` | Disable session recording and resume |
| `--command-tools` | Enable command execution tools |
| `--show-reasoning` | Display reasoning content provided by the model |
| `--no-color` | Use plain-text output |

See the complete option list with:

```sh
forestcode --help
```

The interactive CLI can manage sessions, compact context, show long-term memory, and select a Skill or Subagent. Type `/` to see the currently available commands.

## Features

- Chat with a model in the terminal and let it read, search, and understand the current repository.
- Patch-first file editing: review a diff before approving any write.
- Optional command execution, disabled by default and confirmed before running.
- JSONL sessions, session resume, automatic context compaction, and long-term memory.
- A `rich` and `prompt_toolkit` terminal UI with Markdown, completion, a spinner, and turn cancellation.
- Skills for loading reusable local instructions into one task.
- Subagents for running independent child tasks concurrently under a parent Agent.
- DeepSeek and OpenAI-compatible API support.

## How It Works

ForestCode combines the current workspace, project rules, session history, long-term memory, and tool definitions into each model request. The model can call file and search tools. For file changes, ForestCode shows a Patch and writes only after user approval.

Command tools are disabled by default. Enable them when the Agent needs to run tests or other commands:

```sh
forestcode --command-tools
```

Skills add specialized instructions to the current task. Subagents let the parent Agent delegate independent tasks to concurrent children. Both use the same Agent Loop, tool permissions, and session mechanisms as the main runtime.

## Permissions and Runtime Boundaries

- Tools are restricted to the current workspace by default.
- File changes are shown as a diff and written only after user approval.
- Command tools are disabled by default; when enabled, each command still requires approval.
- Subagents support one parent-child level. The parent manages delegation, waiting, and cancellation; children cannot delegate further or communicate with each other.
- MCP, Agent Team, nested children, and cross-process task recovery are not currently supported.
- API keys belong in user configuration and should never be committed to Git.

## Development

Clone the repository and sync dependencies:

```sh
git clone https://github.com/hsghcop/ForestCode.git
cd ForestCode
uv sync
```

Run from source:

```sh
uv run forestcode
```

Run the full test suite:

```sh
uv run python -m unittest discover -s tests -p "test_*.py"
```

Use the project script for the complete test and build pipeline:

```sh
./scripts/check.sh               # test, then rebuild packages
./scripts/check.sh --skip-build  # test only
```

After a successful build, `dist/` contains a wheel and a source distribution. Install the local wheel globally for isolated testing:

```sh
uv tool install --force dist/forestcode-*.whl
```

## Project Structure

```text
src/forestcode/
  core/        Agent Loop, run state, and tool execution orchestration
  models/      model configuration, adapters, and routing
  tools/       file, search, editing, command, and permission tools
  context/     per-request model context construction
  memory/      sessions, compaction, and long-term memory
  terminal/    input, rendering, confirmation, and cancellation
  skills/      Skill discovery, loading, and run-scoped activation
  subagents/   Subagent configuration, scheduling, and child runtime
```

## License

ForestCode is licensed under the [MIT License](https://github.com/hsghcop/ForestCode/blob/main/LICENSE).

# ForestCode

[English](https://github.com/hsghcop/ForestCode/blob/main/README_EN.md)

[![PyPI](https://img.shields.io/pypi/v/forestcode)](https://pypi.org/project/forestcode/)
[![Python](https://img.shields.io/pypi/pyversions/forestcode)](https://pypi.org/project/forestcode/)
[![License](https://img.shields.io/pypi/l/forestcode)](LICENSE)

ForestCode 是一个用 Python 实现的终端 Coding Agent。它把语言模型、代码工具、会话记忆和终端交互组合成一个可以在真实代码仓库中工作的 Agent runtime。

ForestCode 不依赖 LangChain、LangGraph 等 Agent 框架，核心循环、工具系统、上下文、记忆、Patch-first 编辑和 Subagent 调度都由项目自身实现。

## 环境要求

- Python 3.13 或更高版本
- [uv](https://docs.astral.sh/uv/)
- macOS、Linux 或 WSL
- 一个可用的 DeepSeek 或 OpenAI-compatible API

## 安装

ForestCode 已发布到 [PyPI](https://pypi.org/project/forestcode/)。使用 uv 安装为全局终端工具：

```sh
uv tool install forestcode
```

安装后可以在任意目录运行：

```sh
forestcode
```

## 配置 API

第一次运行时，如果配置不存在，ForestCode 会创建：

```text
~/.forestcode/settings.json
```

打开这个文件，填入 API Key、模型和接口地址，然后重新运行 `forestcode`。

DeepSeek 示例：

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

OpenAI-compatible 示例：

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

`settings.json` 支持独占一行的 `//` 注释，不支持行尾注释。当前目录的 `.env` 可以覆盖普通模型和调优参数，CLI 参数优先级最高。不要把真实 API Key 提交到 Git 仓库。

## 开始使用

进入需要处理的代码仓库，然后启动 ForestCode：

```sh
cd /path/to/your-project
forestcode
```

ForestCode 默认把当前目录作为 workspace。你可以直接输入任务，例如：

```text
先阅读这个项目，告诉我它的入口和主要模块。
检查当前测试失败的原因，不要修改代码。
修复这个问题，修改前先展示 diff。
```

常用启动参数：

| 参数 | 作用 |
| --- | --- |
| `--workspace <path>` | 指定工作目录，默认是当前目录 |
| `--session <id>` | 使用指定会话，默认是 `default` |
| `--no-session` | 不记录或恢复会话 |
| `--command-tools` | 开启命令执行工具 |
| `--show-reasoning` | 显示模型提供的 reasoning 内容 |
| `--no-color` | 使用纯文本输出 |

完整参数可以通过以下命令查看：

```sh
forestcode --help
```

交互中可以管理会话、压缩上下文、查看长期记忆、选择 Skill 或 Subagent。输入 `/` 可以查看当前可用命令。

## 主要功能

- 终端内与模型对话，并让模型读取、搜索和理解当前代码仓库。
- Patch-first 文件编辑：修改前展示 diff，由用户确认后写入。
- 可选的命令执行工具，默认关闭，执行前需要确认。
- JSONL 会话记录、会话恢复、自动上下文压缩和长期记忆。
- 基于 `rich` 与 `prompt_toolkit` 的终端界面，支持 Markdown、补全、spinner 和回合取消。
- Skills：为一次任务加载可复用的本地工作说明。
- Subagents：由父 Agent 并发派发独立子任务，并集中等待、取消和收集结果。
- 支持 DeepSeek 和 OpenAI-compatible API。

## 工作方式

ForestCode 会把当前 workspace、项目规则、会话历史、长期记忆和工具定义组合成本轮上下文。模型可以调用文件与搜索工具；涉及文件修改时，终端先展示 Patch，用户确认后才写入。

命令工具默认关闭。如需让 Agent 执行测试或其他命令：

```sh
forestcode --command-tools
```

Skills 可以为当前任务加入专门的工作说明；Subagents 可以把多个独立任务交给不同子 Agent 并发处理。它们都建立在同一套 Agent Loop、工具权限和会话机制之上。

## 权限与运行边界

- 工具默认限制在当前 workspace。
- 文件修改先展示 diff，用户确认后才写入。
- 命令工具默认关闭；启用后，执行命令仍需确认。
- Subagents 只支持一层父子关系，由父 Agent 统一派发、等待和取消；child 不能继续派生或互相通信。
- 当前不支持 MCP、Agent Team、多层 child 或跨进程任务恢复。
- API Key 保存在用户配置中，不要提交到 Git 仓库。

## 从源码开发

克隆仓库并同步依赖：

```sh
git clone https://github.com/hsghcop/ForestCode.git
cd ForestCode
uv sync
```

从源码启动：

```sh
uv run forestcode
```

运行全部测试：

```sh
uv run python -m unittest discover -s tests -p "test_*.py"
```

项目提供统一的测试与构建脚本：

```sh
./scripts/check.sh               # 运行测试，通过后重新构建包
./scripts/check.sh --skip-build  # 只运行测试
```

构建成功后，`dist/` 中会生成 wheel 和源码包。可以把本地 wheel 安装为全局工具进行独立测试：

```sh
uv tool install --force dist/forestcode-*.whl
```

## 代码结构

```text
src/forestcode/
  core/        Agent Loop、run state 与工具执行编排
  models/      模型配置、适配器与路由
  tools/       文件、搜索、编辑、命令和权限工具
  context/     本轮模型上下文构建
  memory/      会话、压缩与长期记忆
  terminal/    输入、渲染、确认和取消
  skills/      Skill 发现、加载和一次性激活
  subagents/   Subagent 配置、调度和 child runtime
```

## License

ForestCode 使用 [MIT License](LICENSE)。

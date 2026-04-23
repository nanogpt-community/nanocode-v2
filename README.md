# Pebble

Pebble is a rust based agentic coding harness

It supports:

- Nano-GPT
- Synthetic
- OpenAI Codex / ChatGPT plans
- OpenCode Go

Pebble is designed around an interactive REPL, local tools, managed sessions, MCP servers, and a user-controlled permission model. Web retrieval is provider-agnostic and always runs through Exa.

<img width="1571" height="504" alt="CleanShot 2026-04-21 at 21 54 52" src="https://github.com/user-attachments/assets/21eeb498-7f84-4c72-8fc2-718176ddf0ad" />

Check the [Changelog](CHANGELOG.md) for update/patch notes


## First-run setup

### 1. Authenticate a model service

Pebble can prompt you for credentials interactively:

```bash
pebble login
```

You can also target a specific service directly:

```bash
pebble login synthetic
pebble login openai-codex
pebble login opencode-go
pebble login nanogpt
```

For API-key services, you can also pass the key inline:

```bash
pebble login opencode-go --api-key "$OPENCODE_GO_API_KEY"
```

`openai-codex` uses ChatGPT device-code auth instead of an API key.

Inside the REPL, the equivalent commands are:

```text
/login
/auth
/login openai-codex
/login opencode-go
/auth synthetic
/logout openai-codex
```

If you run `/login` or `/auth` without a service, Pebble opens a picker with:

- `nanogpt`
- `synthetic`
- `openai-codex`
- `opencode-go`
- `exa`

### 2. Save your Exa key

Pebble uses Exa for all web search and scrape functionality.

Save it with:

```bash
pebble login exa
```

Or from inside the REPL:

```text
/login exa
/auth exa
```

You can also provide it inline:

```bash
pebble login exa --api-key "$EXA_API_KEY"
```

Or export it in your shell:

```bash
export EXA_API_KEY=...
```

## Daily usage

### Interactive REPL

Launch the REPL:

```bash
pebble
```

Useful first commands:

```text
/help
/status
/model
/login
/logout
/sessions
```

Basic prompt flow:

```text
> summarize this project
> inspect Cargo.toml and explain the workspace layout
> find the session restore logic
```

### One-shot prompt mode

For a single command without entering the REPL:

```bash
pebble prompt "Summarize this repository"
```

Or:

```bash
pebble "Inspect the current Rust workspace and explain the top-level crates"
```

### Restrict tool access

```bash
pebble --allowedTools read,glob "Summarize Cargo.toml"
```

## Core REPL commands

Common commands:

- `/help`
- `/help auth`
- `/help sessions`
- `/help extensions`
- `/help web`
- `/status`
- `/model`
- `/login`
- `/logout`
- `/provider`
- `/permissions`
- `/bypass`
- `/proxy`
- `/mcp`
- `/skills`
- `/plugins`
- `/sessions`
- `/resume`
- `/resume last`
- `/session switch <id>`

Notes:

- `/provider` only applies to NanoGPT-backed models.
- `Shift+Enter` and `Ctrl+J` insert a newline in the input editor.

## Authentication and config

Pebble stores user config under:

```text
~/.pebble/
```

Credentials are stored in:

```text
~/.pebble/credentials.json
```

Possible stored keys:

- `nanogpt_api_key`
- `synthetic_api_key`
- `openai_codex_auth`
- `opencode_go_api_key`
- `exa_api_key`

Environment variables still take precedence over saved credentials.

Useful environment variables:

- `NANOGPT_API_KEY`
- `SYNTHETIC_API_KEY`
- `OPENAI_CODEX_ACCESS_TOKEN`
- `OPENAI_CODEX_REFRESH_TOKEN`
- `OPENAI_CODEX_ACCOUNT_ID`
- `OPENAI_CODEX_EXPIRES_AT`
- `OPENCODE_GO_API_KEY`
- `EXA_API_KEY`
- `NANOGPT_BASE_URL`
- `SYNTHETIC_BASE_URL`
- `OPENAI_CODEX_BASE_URL`
- `OPENCODE_GO_BASE_URL`
- `EXA_BASE_URL`
- `PEBBLE_CONFIG_HOME`

`EXA_BASE_URL` defaults to `https://api.exa.ai`.
`OPENAI_CODEX_BASE_URL` defaults to `https://chatgpt.com/backend-api/codex`.

## Sessions and restore

Pebble keeps managed sessions under:

```text
.pebble/sessions/
```

Useful flows:

- `/sessions` lists recent sessions
- `/resume` opens the picker
- `/resume last` restores the most recently modified session
- `/session switch <session-id>` switches inside the REPL
- `pebble resume [SESSION_ID_OR_PATH]` resumes from the CLI

Session restore includes more than just transcript history. Pebble persists and restores:

- active model
- permission mode
- thinking toggle
- proxy tool-call toggle
- allowed tool set

That makes restored sessions behave much closer to the original live session.

## Permissions

Pebble supports:

- `read-only`
- `workspace-write`
- `danger-full-access`

Examples:

```text
/permissions
/permissions workspace-write
/bypass
```

`/bypass` is a shortcut for `danger-full-access` in the current session.

## Web search and scrape

Pebble keeps the tool names `WebSearch` and `WebScrape`, but both use Exa.

### WebSearch

- uses Exa `POST /search`
- defaults to Exa search type `auto`
- promotes to `deep` for deeper or more structured requests
- maps allowed and blocked domains into Exa domain filters

### WebScrape

- uses Exa `POST /contents`
- supports one or more URLs
- validates URLs before sending requests
- returns normalized previews in the TUI

### Check readiness

Run:

```text
/status
```

Pebble reports Exa readiness separately from the active model backend.

## Extensions

Pebble has three main extension surfaces:

- skills
- MCP servers
- plugins

### Skills

Create a project-local skill:

```text
/skills init my-skill
```

This creates:

```text
.pebble/skills/my-skill/SKILL.md
```

Useful commands:

```text
/skills
/skills help
```

### MCP servers

Create a starter MCP server entry:

```text
/mcp add my-server
```

This updates:

```text
.pebble/settings.json
```

Inspect what is configured:

```text
/mcp
/mcp tools
/mcp reload
```

Enable or disable a configured server locally:

```text
/mcp disable context7
/mcp enable context7
```

These local toggles are written to:

```text
.pebble/settings.local.json
```

That lets you keep a shared project MCP config while turning specific servers on or off per machine.

### Plugins

Useful commands:

```text
/plugins
/plugins help
/plugins install ./plugins/my-plugin
/plugins enable my-plugin-id
```

Pebble expects plugins to expose:

```text
.pebble-plugin/plugin.json
```

## Proxy mode

Pebble can run in XML proxy tool-call mode:

```text
/proxy status
/proxy on
/proxy off
```

When proxy mode is enabled, tool use is expected through XML `<tool_call>` blocks rather than native tool schemas.

## Troubleshooting

### A model won’t answer

- run `/status`
- confirm you saved credentials with `pebble login`
- or export the matching `*_API_KEY`
- verify the active model with `/model`

### Web tools are unavailable

- run `pebble login exa`
- or export `EXA_API_KEY`
- check `/status` for Exa readiness

### MCP server loads but shows no tools

- run `/mcp`
- run `/mcp tools`
- run `/mcp reload`
- check `.pebble/settings.json`
- check `.pebble/settings.local.json`
- if the server is marked `disabled`, run `/mcp enable <name>`

### Session restore feels wrong

- inspect `/status`
- use `/resume last` or `/session switch <id>`
- verify the session was saved after changing model, permissions, proxy, or thinking state

### Plugin setup is unclear

- run `/plugins help`
- confirm the plugin root contains `.pebble-plugin/plugin.json`

## Install, build, and development

### Build a release binary

```bash
cargo build --release -p pebble
```

Binary output:

```bash
./target/release/pebble
```

On Windows, the binary output is `target\\release\\pebble.exe`. Pebble resolves config and
credentials from `PEBBLE_CONFIG_HOME` first, then `%USERPROFILE%\\.pebble` when `HOME` is not set.

### Run from source

```bash
cargo run -p pebble --
```

### Run tests

```bash
cargo test --workspace -- --test-threads=1
```

### Project config files

Common project-local files:

- `PEBBLE.md`
- `.pebble/settings.json`
- `.pebble/settings.local.json`
- `.pebble/skills/`
- `.pebble/sessions/`

### Release/update behavior

Pebble’s self-update flow targets the GitHub releases for:

```text
nanogpt-community/pebble
```

# NanoCode v2

NanoCode v2 is a rust based agentic coding harness for the nanogpt API

## Includes

- XML tool calling via /proxy command
- Select any nanogpt model via /model
- Provider selection via /provider (WIP - This currently uses openai api instead of anthropic v1)

## Prerequisites

- Rust toolchain installed (`rustup`, stable toolchain)
- NanoGPT API key

## Build

```bash
cd rust
cargo build --release -p nanocode
```

The optimized binary will be written to:

```bash
./target/release/nanocode
```

## Config and auth

The CLI reads:

- `~/.nanocode/credentials.json` for persisted NanoGPT authentication
- `~/.nanocode/state.json` for default model, favorites, provider overrides, and proxy mode
- `NANOGPT_API_KEY` for shell-scoped NanoGPT authentication
- `NANOGPT_BASE_URL` to override the default `https://nano-gpt.com/api`
- `NANOCODE_PERMISSION_MODE` for local permission defaults
- `NANOCODE_CONFIG_HOME`, `.nanocode/settings*.json`, and `NANOCODE.md` files for runtime configuration and instructions

## Quick start

### Show help

```bash
cd rust
cargo run -p nanocode -- --help
```

### Print version

```bash
cd rust
cargo run -p nanocode -- --version
```

### Restrict enabled tools

```bash
cd rust
cargo run -p nanocode -- --allowedTools read,glob prompt "Summarize Cargo.toml"
```

## Notes

- The fallback default model is `zai-org/glm-5.1`.
- Provider overrides are stored per model.
- `--allowedTools` restricts both advertised tools and executable tools for prompt mode and REPL sessions.
- `compat-harness` exists to compare the Rust port against the upstream TypeScript codebase and is intentionally excluded from the requested release test run.
- The CLI currently focuses on a practical integrated workflow: prompt execution, REPL operation, session inspection/resume, config discovery, NanoGPT routing, and tool/runtime plumbing.

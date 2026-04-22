# PEBBLE.md

## Project Overview
Pebble is a Rust agentic coding harness built around an interactive REPL, local tools, managed sessions, MCP servers, and a user-controlled permission model. It supports Nano-GPT, Synthetic, and OpenCode Go. Web retrieval (search/scrape) always runs through Exa.

## Repository Shape
- Rust workspace at the repository root; `Cargo.toml` includes all crates under `crates/*`.
- Active refactor in progress: the legacy `crates/nanocode/` crate has been removed and replaced by `crates/pebble/`. Verify `crates/pebble/Cargo.toml` defines the binary target.
- Notable crates:
  - `crates/pebble/` — intended main CLI binary.
  - `crates/api/` — client, types, errors; contains integration tests in `tests/`.
  - `crates/runtime/` — core runtime: bootstrap, config, conversation, MCP clients/stdio, sandbox, session, file ops, prompts.
  - `crates/commands/`, `crates/plugins/`, `crates/tools/` — command dispatch, plugin system, and tool implementations.
  - `crates/compat-harness/` — compatibility utilities.
- CI: `.github/workflows/release-pebble.yml`.
- Shared assistant settings live in `.pebble/`.

## Commands
- Build the workspace: `cargo build --workspace`
- Run all tests: `cargo test --workspace`
- Run a single crate's tests: `cargo test -p <crate-name>`
- Check formatting: `cargo fmt --all`
- Run lints: `cargo clippy --workspace` (respects workspace-level `forbid(unsafe_code)` and clippy

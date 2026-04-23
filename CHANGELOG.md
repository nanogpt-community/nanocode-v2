# CHANGELOG (STARTING FROM v0.4.0)

## v0.4.4

- Fix CI instability in tools tests by resolving non-Windows bash commands through `/bin/sh` when available, avoiding sensitivity to tests that temporarily mutate `PATH`.
- Recover from poisoned test environment locks in the local skill-loading test so later tests can continue after prior panics.
- Address clippy's needless pass-by-value warning in runtime file operation tests.

## v0.4.3

- Clarify timeout units in built-in `bash`, `REPL`, and `PowerShell` tool schemas and descriptions so model-facing docs explicitly state milliseconds.
- Enforce `REPL.timeout_ms` for Python, JavaScript/Node, and shell snippets, returning structured output with `timedOut: true` and `Command exceeded timeout of {timeout_ms} ms` on timeout.
- Harden `grep_search.output_mode` handling by validating supported modes (`files_with_matches`, `content`, and `count`) and returning a clear invalid-input error for unknown values.
- Update `grep_search` content-mode pagination so `head_limit` and `offset` apply to returned content lines, with `filenames` and `numFiles` derived from the displayed lines.
- Add targeted tests for REPL timeout enforcement and grep output-mode/pagination semantics.

## v0.4.2

- Add repo-aware defaults for native search tools: broad `grep_search` and `glob_search` now skip hidden/project-state directories, respect `.gitignore`, and avoid `.git`, `target`, `.pebble/sessions`, `.pebble/tool-results`, `.pebble/agents`, `.sandbox-home`, and `.sandbox-tmp` unless explicitly targeted.
- Add workspace-bound path enforcement for `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`, and `apply_patch` targets, including lexical and symlink escape checks with clear `path escapes workspace: ...` errors.
- Add focused runtime coverage for workspace path safety, symlink escapes, missing file creation, patch target safety, and repo-aware search behavior.
- Replace direct runtime `walkdir` traversal with `ignore::WalkBuilder` and remove the direct `walkdir` runtime dependency.

## v0.4.1

- Add safer atomic writes for JSON, session, config, plugin, tool, and file-edit persistence to reduce the risk of truncated files.
- Add CI coverage for formatting, clippy, and serial workspace tests.
- Fix runtime prompt tests so they isolate temp roots from ambient project instruction and memory files.

## v0.4.0

- Completely redesigned compaction

Before - haha delete messages go brr 
Now - Full and complete compactions (I totally didn't spend hours recreating opencode's compaction system in rust)

- Fixed openai models not having a context length
- Add fallback for context length to 200K if can't find or is unknown for whatever reason
- Removed Vim Mode
- some black magic to make things better in the backend
- add snapshots (/undo and /redo)
- add session fork/rename/timeline
- add custom command templates
- add @file references
- add changelog.md so github has changelogs
- persist things like permission mode and reasoning effort across sessions and startups
- add patch/diff editing for improved multi-file editing
- many more improvements and fixes

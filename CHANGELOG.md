# CHANGELOG (STARTING FROM v0.4.0)

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

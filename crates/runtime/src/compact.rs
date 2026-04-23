use std::borrow::Cow;
use std::cmp;
use std::fs;
use std::path::PathBuf;

use platform::write_atomic;

use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session};

const COMPACT_CONTINUATION_PREAMBLE: &str =
    "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";
const COMPACT_RECENT_MESSAGES_NOTE: &str = "Recent messages are preserved verbatim.";
const COMPACT_DIRECT_RESUME_INSTRUCTION: &str = "Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, and do not preface with continuation text.";
const MIN_PRESERVE_RECENT_TOKENS: usize = 2_000;
const MAX_PRESERVE_RECENT_TOKENS: usize = 8_000;
const TOOL_RESULT_PRUNE_MIN_TOKENS: usize = 20_000;
const TOOL_RESULT_PRUNE_PROTECT_TOKENS: usize = 40_000;
const TOOL_RESULT_PRUNE_RECENT_TURNS: usize = 2;
const TOOL_RESULT_CONTEXT_PLACEHOLDER: &str = "[Old tool result content cleared]";
const TOOL_RESULT_PRUNE_PROTECTED_TOOLS: &[&str] = &["skill"];
const TOOL_RESULT_ARCHIVE_DIR: &str = "tool-results";
const SUMMARY_TEMPLATE: &str = r#"Output exactly this Markdown structure and keep the section order unchanged:
---
## Goal
- [single-sentence task summary]

## Constraints & Preferences
- [user constraints, preferences, specs, or "(none)"]

## Progress
### Done
- [completed work or "(none)"]

### In Progress
- [current work or "(none)"]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- [decision and why, or "(none)"]

## Next Steps
- [ordered next actions or "(none)"]

## Critical Context
- [important technical facts, errors, open questions, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
---

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, commands, error strings, and identifiers when known.
- Do not mention the summary process or that context was compacted."#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub preserve_recent_messages: usize,
    pub max_estimated_tokens: usize,
    pub preserve_recent_tokens: Option<usize>,
    pub auto: bool,
    pub overflow: bool,
}

impl CompactionConfig {
    #[must_use]
    pub fn preserve_recent_token_budget(&self) -> usize {
        self.preserve_recent_tokens.unwrap_or_else(|| {
            if self.max_estimated_tokens == 0 {
                return MAX_PRESERVE_RECENT_TOKENS;
            }

            self.max_estimated_tokens
                .saturating_div(4)
                .clamp(MIN_PRESERVE_RECENT_TOKENS, MAX_PRESERVE_RECENT_TOKENS)
        })
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent_messages: 2,
            max_estimated_tokens: 10_000,
            preserve_recent_tokens: None,
            auto: false,
            overflow: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    pub summary: String,
    pub formatted_summary: String,
    pub compacted_session: Session,
    pub removed_message_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCompaction {
    pub previous_summary: Option<String>,
    pub prompt: String,
    pub summary_input_session: Session,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolResultPruneResult {
    pub session: Session,
    pub pruned_tool_result_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedTail {
    keep_from: usize,
    tail_start_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Turn {
    start: usize,
    end: usize,
}

#[must_use]
pub fn estimate_session_tokens(session: &Session) -> usize {
    session.messages.iter().map(estimate_message_tokens).sum()
}

#[must_use]
pub fn should_compact(session: &Session, config: CompactionConfig) -> bool {
    let start = compacted_summary_prefix_len(session);
    let compactable = &session.messages[start..];
    let total_tokens = compactable
        .iter()
        .map(estimate_message_tokens)
        .sum::<usize>();

    if total_tokens < config.max_estimated_tokens {
        return false;
    }

    compactable.len() > config.preserve_recent_messages
        || (start > 0 && total_tokens > config.preserve_recent_token_budget())
}

#[must_use]
pub fn format_compact_summary(summary: &str) -> String {
    let without_analysis = strip_tag_block(summary, "analysis");
    let formatted = if let Some(content) = extract_tag_block(&without_analysis, "summary") {
        without_analysis.replace(
            &format!("<summary>{content}</summary>"),
            &format!("Summary:\n{}", content.trim()),
        )
    } else {
        without_analysis
    };

    collapse_blank_lines(&formatted).trim().to_string()
}

#[must_use]
pub fn get_compact_continuation_message(
    summary: &str,
    suppress_follow_up_questions: bool,
    recent_messages_preserved: bool,
) -> String {
    let mut base = format!(
        "{COMPACT_CONTINUATION_PREAMBLE}{}",
        format_compact_summary(summary)
    );

    if recent_messages_preserved {
        base.push_str("\n\n");
        base.push_str(COMPACT_RECENT_MESSAGES_NOTE);
    }

    if suppress_follow_up_questions {
        base.push('\n');
        base.push_str(COMPACT_DIRECT_RESUME_INSTRUCTION);
    }

    base
}

#[must_use]
pub fn get_tool_result_context_output(output: &str, compacted: bool) -> Cow<'_, str> {
    if compacted {
        Cow::Borrowed(TOOL_RESULT_CONTEXT_PLACEHOLDER)
    } else {
        Cow::Borrowed(output)
    }
}

#[must_use]
pub(crate) fn prune_old_tool_results(
    session: &Session,
    max_estimated_tokens: usize,
) -> ToolResultPruneResult {
    if max_estimated_tokens > 0 && estimate_session_tokens(session) < max_estimated_tokens {
        return ToolResultPruneResult {
            session: session.clone(),
            pruned_tool_result_count: 0,
        };
    }

    let mut recent_user_turns = 0;
    let mut protected_tokens = 0;
    let mut prunable_tokens = 0;
    let mut candidate_indexes = Vec::new();
    let start = compacted_summary_prefix_len(session);

    for index in (start..session.messages.len()).rev() {
        let message = &session.messages[index];
        if message.role == MessageRole::User {
            recent_user_turns += 1;
        }
        if recent_user_turns < TOOL_RESULT_PRUNE_RECENT_TURNS {
            continue;
        }

        let Some(ContentBlock::ToolResult {
            tool_name,
            output,
            is_error,
            compacted,
            ..
        }) = message.blocks.first()
        else {
            continue;
        };

        if *is_error
            || *compacted
            || TOOL_RESULT_PRUNE_PROTECTED_TOOLS.contains(&tool_name.as_str())
        {
            continue;
        }

        let estimate = estimate_tool_result_tokens(tool_name, output, false);
        protected_tokens += estimate;
        if protected_tokens <= TOOL_RESULT_PRUNE_PROTECT_TOKENS {
            continue;
        }

        prunable_tokens += estimate;
        candidate_indexes.push(index);
    }

    if prunable_tokens < TOOL_RESULT_PRUNE_MIN_TOKENS {
        return ToolResultPruneResult {
            session: session.clone(),
            pruned_tool_result_count: 0,
        };
    }

    let mut pruned_session = session.clone();
    let mut pruned_tool_result_count = 0;

    for index in candidate_indexes {
        let Some(message) = pruned_session.messages.get_mut(index) else {
            continue;
        };
        let message_id = message.id.clone();
        let Some(ContentBlock::ToolResult {
            tool_name,
            output,
            compacted,
            archived_output_path,
            ..
        }) = message.blocks.first_mut()
        else {
            continue;
        };
        if !*compacted {
            let Some(path) = archive_tool_result_output(&message_id, tool_name, output) else {
                continue;
            };
            output.clear();
            *compacted = true;
            *archived_output_path = Some(path);
            pruned_tool_result_count += 1;
        }
    }

    ToolResultPruneResult {
        session: pruned_session,
        pruned_tool_result_count,
    }
}

#[must_use]
pub fn compact_session(session: &Session, config: CompactionConfig) -> CompactionResult {
    compact_session_with_summary(session, config, None)
}

#[must_use]
pub fn compact_session_with_summary(
    session: &Session,
    config: CompactionConfig,
    generated_summary: Option<String>,
) -> CompactionResult {
    if !should_compact(session, config) {
        return CompactionResult {
            summary: String::new(),
            formatted_summary: String::new(),
            compacted_session: session.clone(),
            removed_message_count: 0,
        };
    }

    let existing_summary = session
        .messages
        .first()
        .and_then(extract_existing_compacted_summary);
    let compacted_prefix_len = usize::from(existing_summary.is_some());
    let selected_tail = select_retained_tail(session, config);
    let summarize_all =
        selected_tail.tail_start_id.is_none() && selected_tail.keep_from == compacted_prefix_len;
    let removed = if summarize_all {
        &session.messages[compacted_prefix_len..]
    } else {
        &session.messages[compacted_prefix_len..selected_tail.keep_from]
    };
    let preserved = if summarize_all {
        Vec::new()
    } else {
        session.messages[selected_tail.keep_from..].to_vec()
    };
    let summary = match generated_summary.filter(|summary| !summary.trim().is_empty()) {
        Some(summary) => summary.trim().to_string(),
        None => merge_compact_summaries(existing_summary.as_deref(), &summarize_messages(removed)),
    };
    let formatted_summary = format_compact_summary(&summary);
    let mut compacted_messages = vec![ConversationMessage::compaction_summary(
        summary.clone(),
        !preserved.is_empty(),
        config.auto,
        config.overflow,
        selected_tail.tail_start_id,
    )];
    compacted_messages.extend(preserved);

    CompactionResult {
        summary,
        formatted_summary,
        compacted_session: Session {
            version: session.version,
            messages: compacted_messages,
            metadata: session.metadata.clone(),
        },
        removed_message_count: removed.len(),
    }
}

#[must_use]
pub fn prepare_compaction(
    session: &Session,
    config: CompactionConfig,
) -> Option<PreparedCompaction> {
    if !should_compact(session, config) {
        return None;
    }

    let previous_summary = session
        .messages
        .first()
        .and_then(extract_existing_compacted_summary);
    let compacted_prefix_len = usize::from(previous_summary.is_some());
    let selected_tail = select_retained_tail(session, config);
    let summarize_all =
        selected_tail.tail_start_id.is_none() && selected_tail.keep_from == compacted_prefix_len;
    let input_messages = if summarize_all {
        session.messages[compacted_prefix_len..].to_vec()
    } else {
        session.messages[compacted_prefix_len..selected_tail.keep_from].to_vec()
    };
    if input_messages.is_empty() && previous_summary.is_none() {
        return None;
    }

    Some(PreparedCompaction {
        prompt: build_compaction_prompt(previous_summary.as_deref(), &[]),
        previous_summary,
        summary_input_session: Session {
            version: session.version,
            messages: input_messages,
            metadata: session.metadata.clone(),
        },
    })
}

#[must_use]
pub fn build_compaction_prompt(previous_summary: Option<&str>, context: &[String]) -> String {
    let anchor = previous_summary.map_or_else(
        || "Create a new anchored summary from the conversation history above.".to_string(),
        |summary| {
            [
                "Update the anchored summary below using the conversation history above.",
                "Preserve still-true details, remove stale details, and merge in the new facts.",
                "<previous-summary>",
                summary,
                "</previous-summary>",
            ]
            .join("\n")
        },
    );
    let mut sections = vec![anchor, SUMMARY_TEMPLATE.to_string()];
    sections.extend(
        context
            .iter()
            .filter(|value| !value.trim().is_empty())
            .cloned(),
    );
    sections.join("\n\n")
}

fn select_retained_tail(session: &Session, config: CompactionConfig) -> SelectedTail {
    let start = compacted_summary_prefix_len(session);
    let compactable = &session.messages[start..];
    if compactable.is_empty() {
        return SelectedTail {
            keep_from: session.messages.len(),
            tail_start_id: None,
        };
    }

    let budget = config.preserve_recent_token_budget();
    if config.preserve_recent_messages == 0 {
        return SelectedTail {
            keep_from: start,
            tail_start_id: None,
        };
    }

    let turns = collect_turns(session, start);
    if turns.is_empty() {
        let preferred_tail = cmp::min(config.preserve_recent_messages, compactable.len());
        let mut keep_from = session.messages.len().saturating_sub(preferred_tail);
        while keep_from < session.messages.len()
            && estimate_messages_tokens(&session.messages[keep_from..]) > budget
        {
            keep_from += 1;
        }
        if keep_from <= start {
            return SelectedTail {
                keep_from: start,
                tail_start_id: None,
            };
        }
        return SelectedTail {
            tail_start_id: session
                .messages
                .get(keep_from)
                .map(|message| message.id.clone()),
            keep_from,
        };
    }

    let recent_start = turns.len().saturating_sub(config.preserve_recent_messages);
    let recent = &turns[recent_start..];
    let mut total = 0;
    let mut keep_from = None;

    for turn in recent.iter().rev() {
        let size = estimate_messages_tokens(&session.messages[turn.start..turn.end]);
        if total + size <= budget {
            total += size;
            keep_from = Some(turn.start);
            continue;
        }

        let remaining = budget.saturating_sub(total);
        let split = split_turn(session, *turn, remaining);
        if split.is_some() {
            keep_from = split;
        }
        break;
    }

    let keep_from = keep_from.unwrap_or(start);
    if keep_from <= start {
        return SelectedTail {
            keep_from: start,
            tail_start_id: None,
        };
    }
    SelectedTail {
        tail_start_id: session
            .messages
            .get(keep_from)
            .map(|message| message.id.clone()),
        keep_from,
    }
}

fn collect_turns(session: &Session, start: usize) -> Vec<Turn> {
    let mut turns = Vec::new();
    for index in start..session.messages.len() {
        if session.messages[index].role != MessageRole::User {
            continue;
        }
        turns.push(Turn {
            start: index,
            end: session.messages.len(),
        });
    }
    for index in 0..turns.len().saturating_sub(1) {
        turns[index].end = turns[index + 1].start;
    }
    turns
}

fn split_turn(session: &Session, turn: Turn, budget: usize) -> Option<usize> {
    if budget == 0 || turn.end.saturating_sub(turn.start) <= 1 {
        return None;
    }

    for start in (turn.start + 1)..turn.end {
        if estimate_messages_tokens(&session.messages[start..turn.end]) <= budget {
            return Some(start);
        }
    }
    None
}

fn compacted_summary_prefix_len(session: &Session) -> usize {
    usize::from(
        session
            .messages
            .first()
            .and_then(extract_existing_compacted_summary)
            .is_some(),
    )
}

#[cfg(test)]
fn render_memory_file(formatted_summary: &str) -> String {
    format!("# Project memory\n\n{}\n", formatted_summary.trim())
}

fn archive_tool_result_output(message_id: &str, tool_name: &str, output: &str) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let archive_dir = cwd.join(".pebble").join(TOOL_RESULT_ARCHIVE_DIR);
    fs::create_dir_all(&archive_dir).ok()?;

    let filename = format!(
        "{}-{}.txt",
        sanitize_filename_component(message_id),
        sanitize_filename_component(tool_name)
    );
    let absolute_path = archive_dir.join(filename);
    write_atomic(&absolute_path, output).ok()?;

    Some(
        PathBuf::from(".pebble")
            .join(TOOL_RESULT_ARCHIVE_DIR)
            .join(absolute_path.file_name()?)
            .to_string_lossy()
            .into_owned(),
    )
}

fn sanitize_filename_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() || matches!(char, '-' | '_') {
                char
            } else {
                '-'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "item".to_string()
    } else {
        sanitized
    }
}

fn summarize_messages(messages: &[ConversationMessage]) -> String {
    let user_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::User)
        .count();
    let assistant_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .count();
    let tool_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .count();

    let mut tool_names = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
            ContentBlock::ToolResult { tool_name, .. } => Some(tool_name.as_str()),
            ContentBlock::Text { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::CompactionSummary { .. } => None,
        })
        .collect::<Vec<_>>();
    tool_names.sort_unstable();
    tool_names.dedup();

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={}, assistant={}, tool={}).",
            messages.len(),
            user_messages,
            assistant_messages,
            tool_messages
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    let recent_user_requests = collect_recent_role_summaries(messages, MessageRole::User, 3);
    if !recent_user_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        lines.extend(
            recent_user_requests
                .into_iter()
                .map(|request| format!("  - {request}")),
        );
    }

    let pending_work = infer_pending_work(messages);
    if !pending_work.is_empty() {
        lines.push("- Pending work:".to_string());
        lines.extend(pending_work.into_iter().map(|item| format!("  - {item}")));
    }

    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    if let Some(current_work) = infer_current_work(messages) {
        lines.push(format!("- Current work: {current_work}"));
    }

    lines.push("- Key timeline:".to_string());
    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let content = message
            .blocks
            .iter()
            .map(summarize_block)
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("  - {role}: {content}"));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn merge_compact_summaries(existing_summary: Option<&str>, new_summary: &str) -> String {
    let Some(existing_summary) = existing_summary else {
        return new_summary.to_string();
    };

    let previous_highlights = extract_summary_highlights(existing_summary);
    let new_formatted_summary = format_compact_summary(new_summary);
    let new_highlights = extract_summary_highlights(&new_formatted_summary);
    let new_timeline = extract_summary_timeline(&new_formatted_summary);

    let mut lines = vec!["<summary>".to_string(), "Conversation summary:".to_string()];

    if !previous_highlights.is_empty() {
        lines.push("- Previously compacted context:".to_string());
        lines.extend(
            previous_highlights
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }

    if !new_highlights.is_empty() {
        lines.push("- Newly compacted context:".to_string());
        lines.extend(new_highlights.into_iter().map(|line| format!("  {line}")));
    }

    if !new_timeline.is_empty() {
        lines.push("- Key timeline:".to_string());
        lines.extend(new_timeline.into_iter().map(|line| format!("  {line}")));
    }

    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn summarize_block(block: &ContentBlock) -> String {
    let raw = match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Thinking { text, .. } => format!("thinking {text}"),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult {
            tool_name,
            output,
            is_error,
            compacted,
            ..
        } => format!(
            "tool_result {tool_name}: {}{}",
            if *is_error { "error " } else { "" },
            get_tool_result_context_output(output, *compacted)
        ),
        ContentBlock::CompactionSummary { summary, .. } => {
            format!("compaction summary {}", format_compact_summary(summary))
        }
    };
    truncate_summary(&raw, 160)
}

fn collect_recent_role_summaries(
    messages: &[ConversationMessage],
    role: MessageRole,
    limit: usize,
) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role == role)
        .rev()
        .filter_map(|message| first_text_block(message))
        .take(limit)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn infer_pending_work(messages: &[ConversationMessage]) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .filter(|text| {
            let lowered = text.to_ascii_lowercase();
            lowered.contains("todo")
                || lowered.contains("next")
                || lowered.contains("pending")
                || lowered.contains("follow up")
                || lowered.contains("remaining")
        })
        .take(3)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .flat_map(|block| extract_file_candidates(block_context_text(block).as_ref()))
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn infer_current_work(messages: &[ConversationMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .find(|text| !text.trim().is_empty())
        .map(|text| truncate_summary(text, 200))
}

fn first_text_block(message: &ConversationMessage) -> Option<&str> {
    message.blocks.iter().find_map(|block| match block {
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
        ContentBlock::Thinking { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::CompactionSummary { .. }
        | ContentBlock::Text { .. } => None,
    })
}

fn has_interesting_extension(candidate: &str) -> bool {
    std::path::Path::new(candidate)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            ["rs", "ts", "tsx", "js", "json", "md"]
                .iter()
                .any(|expected| extension.eq_ignore_ascii_case(expected))
        })
}

fn extract_file_candidates(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|char: char| {
                matches!(char, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') && has_interesting_extension(candidate) {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn truncate_summary(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut truncated = content.chars().take(max_chars).collect::<String>();
    truncated.push('…');
    truncated
}

fn estimate_message_tokens(message: &ConversationMessage) -> usize {
    message
        .blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } | ContentBlock::Thinking { text, .. } => text.len() / 4 + 1,
            ContentBlock::ToolUse { name, input, .. } => (name.len() + input.len()) / 4 + 1,
            ContentBlock::ToolResult {
                tool_name,
                output,
                compacted,
                ..
            } => estimate_tool_result_tokens(tool_name, output, *compacted),
            ContentBlock::CompactionSummary { summary, .. } => summary.len() / 4 + 1,
        })
        .sum()
}

fn estimate_tool_result_tokens(tool_name: &str, output: &str, compacted: bool) -> usize {
    let output = get_tool_result_context_output(output, compacted);
    (tool_name.len() + output.len()) / 4 + 1
}

fn block_context_text(block: &ContentBlock) -> Cow<'_, str> {
    match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text, .. } => {
            Cow::Borrowed(text.as_str())
        }
        ContentBlock::ToolUse { input, .. } => Cow::Borrowed(input.as_str()),
        ContentBlock::ToolResult {
            output, compacted, ..
        } => get_tool_result_context_output(output, *compacted),
        ContentBlock::CompactionSummary { summary, .. } => Cow::Borrowed(summary.as_str()),
    }
}

fn estimate_messages_tokens(messages: &[ConversationMessage]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

fn extract_tag_block(content: &str, tag: &str) -> Option<String> {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    let start_index = content.find(&start)? + start.len();
    let end_index = content[start_index..].find(&end)? + start_index;
    Some(content[start_index..end_index].to_string())
}

fn strip_tag_block(content: &str, tag: &str) -> String {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    if let (Some(start_index), Some(end_index_rel)) = (content.find(&start), content.find(&end)) {
        let end_index = end_index_rel + end.len();
        let mut stripped = String::new();
        stripped.push_str(&content[..start_index]);
        stripped.push_str(&content[end_index..]);
        stripped
    } else {
        content.to_string()
    }
}

fn collapse_blank_lines(content: &str) -> String {
    let mut result = String::new();
    let mut last_blank = false;
    for line in content.lines() {
        let is_blank = line.trim().is_empty();
        if is_blank && last_blank {
            continue;
        }
        result.push_str(line);
        result.push('\n');
        last_blank = is_blank;
    }
    result
}

fn extract_existing_compacted_summary(message: &ConversationMessage) -> Option<String> {
    if message.role != MessageRole::System {
        return None;
    }

    if let Some(ContentBlock::CompactionSummary { summary, .. }) = message.blocks.first() {
        return Some(summary.trim().to_string());
    }

    let text = first_text_block(message)?;
    let summary = text.strip_prefix(COMPACT_CONTINUATION_PREAMBLE)?;
    let summary = summary
        .split_once(&format!("\n\n{COMPACT_RECENT_MESSAGES_NOTE}"))
        .map_or(summary, |(value, _)| value);
    let summary = summary
        .split_once(&format!("\n{COMPACT_DIRECT_RESUME_INSTRUCTION}"))
        .map_or(summary, |(value, _)| value);
    Some(summary.trim().to_string())
}

fn extract_summary_highlights(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;

    for line in format_compact_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed == "Summary:" || trimmed == "Conversation summary:" {
            continue;
        }
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if in_timeline {
            continue;
        }
        lines.push(trimmed.to_string());
    }

    lines
}

fn extract_summary_timeline(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;

    for line in format_compact_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if !in_timeline {
            continue;
        }
        if trimmed.is_empty() {
            break;
        }
        lines.push(trimmed.to_string());
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::{
        collect_key_files, compact_session, estimate_session_tokens, format_compact_summary,
        get_tool_result_context_output, infer_pending_work, prune_old_tool_results,
        render_memory_file, should_compact, CompactionConfig,
    };
    use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn formats_compact_summary_like_upstream() {
        let summary = "<analysis>scratch</analysis>\n<summary>Kept work</summary>";
        assert_eq!(format_compact_summary(summary), "Summary:\nKept work");
        assert_eq!(
            render_memory_file("Summary:\nKept work"),
            "# Project memory\n\nSummary:\nKept work\n"
        );
    }

    #[test]
    fn does_not_compact_when_only_preserved_recent_messages_remain() {
        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("small one"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "small two".to_string(),
                }]),
            ],
            metadata: None,
        };

        assert!(!should_compact(
            &session,
            CompactionConfig {
                preserve_recent_messages: 4,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            }
        ));

        let result = compact_session(
            &session,
            CompactionConfig {
                preserve_recent_messages: 4,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            },
        );

        assert_eq!(result.removed_message_count, 0);
        assert_eq!(result.compacted_session, session);
    }

    #[test]
    fn does_not_persist_compacted_summaries_as_project_memory() {
        let _guard = crate::test_env_lock();
        let temp = std::env::temp_dir().join(format!(
            "runtime-compact-memory-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&temp).expect("set cwd");

        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("one ".repeat(200)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "two ".repeat(200),
                }]),
                ConversationMessage::tool_result("1", "bash", "ok ".repeat(200), false),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent".to_string(),
                }]),
            ],
            metadata: None,
        };

        let result = compact_session(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            },
        );
        assert_eq!(result.removed_message_count, 4);
        let memory_dir = temp.join(".pebble").join("memory");
        assert!(!memory_dir.exists());

        std::env::set_current_dir(previous).expect("restore cwd");
        fs::remove_dir_all(temp).expect("cleanup temp dir");
    }

    #[test]
    fn compacts_older_messages_into_a_system_summary() {
        let _guard = crate::test_env_lock();
        let temp = std::env::temp_dir().join(format!(
            "runtime-compact-session-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&temp).expect("set cwd");

        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("one ".repeat(200)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "two ".repeat(200),
                }]),
                ConversationMessage::tool_result("1", "bash", "ok ".repeat(200), false),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent".to_string(),
                }]),
            ],
            metadata: None,
        };

        let result = compact_session(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            },
        );

        assert_eq!(result.removed_message_count, 4);
        assert_eq!(
            result.compacted_session.messages[0].role,
            MessageRole::System
        );
        assert_eq!(result.compacted_session.messages.len(), 1);
        assert!(matches!(
            &result.compacted_session.messages[0].blocks[0],
            ContentBlock::CompactionSummary { summary, .. }
                if summary.contains("Conversation summary")
        ));
        assert!(result.formatted_summary.contains("Scope:"));
        assert!(result.formatted_summary.contains("Key timeline:"));
        assert!(should_compact(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            }
        ));
        assert!(
            estimate_session_tokens(&result.compacted_session) < estimate_session_tokens(&session)
        );

        std::env::set_current_dir(previous).expect("restore cwd");
        fs::remove_dir_all(temp).expect("cleanup temp dir");
    }

    #[test]
    fn keeps_previous_compacted_context_when_compacting_again() {
        let initial_session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("Investigate rust/crates/runtime/src/compact.rs"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "I will inspect the compact flow.".to_string(),
                }]),
                ConversationMessage::user_text(
                    "Also update rust/crates/runtime/src/conversation.rs",
                ),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "Next: preserve prior summary context during auto compact.".to_string(),
                }]),
            ],
            metadata: None,
        };
        let config = CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 1,
            preserve_recent_tokens: Some(1),
            ..CompactionConfig::default()
        };

        let first = compact_session(&initial_session, config);
        let mut follow_up_messages = first.compacted_session.messages.clone();
        follow_up_messages.extend([
            ConversationMessage::user_text("Please add regression tests for compaction."),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "Working on regression coverage now.".to_string(),
            }]),
        ]);

        let second = compact_session(
            &Session {
                version: 1,
                messages: follow_up_messages,
                metadata: None,
            },
            config,
        );

        assert!(second
            .formatted_summary
            .contains("Previously compacted context:"));
        assert!(second
            .formatted_summary
            .contains("Scope: 4 earlier messages compacted"));
        assert!(second
            .formatted_summary
            .contains("Newly compacted context:"));
        assert!(matches!(
            &second.compacted_session.messages[0].blocks[0],
            ContentBlock::CompactionSummary { summary, .. }
                if summary.contains("Previously compacted context:")
                    && summary.contains("Newly compacted context:")
        ));
        assert_eq!(second.compacted_session.messages.len(), 1);
    }

    #[test]
    fn ignores_existing_compacted_summary_when_deciding_to_recompact() {
        let summary = "<summary>Conversation summary:\n- Scope: earlier work preserved.\n- Key timeline:\n  - user: large preserved context\n</summary>";
        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::compaction_summary(summary, true, false, false, None),
                ConversationMessage::user_text("tiny"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent".to_string(),
                }]),
            ],
            metadata: None,
        };

        assert!(!should_compact(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
                preserve_recent_tokens: None,
                ..CompactionConfig::default()
            }
        ));
    }

    #[test]
    fn compacts_more_than_fixed_recent_window_when_needed() {
        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("old request"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "old response".to_string(),
                }]),
                ConversationMessage::user_text("recent but large ".repeat(120)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent answer ".repeat(120),
                }]),
            ],
            metadata: None,
        };

        let result = compact_session(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 450,
                preserve_recent_tokens: Some(450),
                ..CompactionConfig::default()
            },
        );

        assert!(result.removed_message_count >= 3);
        assert_eq!(result.compacted_session.messages.len(), 2);
        assert!(matches!(
            &result.compacted_session.messages[1].blocks[0],
            ContentBlock::Text { text } if text.contains("recent answer")
        ));
    }

    #[test]
    fn compacts_all_non_summary_messages_when_recent_tail_alone_is_too_large() {
        let summary = "<summary>Conversation summary:\n- Scope: earlier work preserved.\n- Key timeline:\n  - user: large preserved context\n</summary>";
        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::compaction_summary(summary, true, false, false, None),
                ConversationMessage::user_text("recent request ".repeat(120)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent response ".repeat(120),
                }]),
            ],
            metadata: None,
        };

        let result = compact_session(
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 300,
                preserve_recent_tokens: Some(300),
                ..CompactionConfig::default()
            },
        );

        assert_eq!(result.removed_message_count, 2);
        assert_eq!(result.compacted_session.messages.len(), 1);
        assert!(matches!(
            &result.compacted_session.messages[0].blocks[0],
            ContentBlock::CompactionSummary { summary, .. }
                if summary.contains("Previously compacted context:")
                    && summary.contains("recent request")
        ));
    }

    #[test]
    fn prunes_old_tool_results_before_full_compaction() {
        let _guard = crate::test_env_lock();
        let temp = std::env::temp_dir().join(format!(
            "runtime-pruned-tool-results-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&temp).expect("set cwd");

        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("first"),
                ConversationMessage::tool_result("tool-1", "bash", "x".repeat(250_000), false),
                ConversationMessage::user_text("second"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "working".to_string(),
                }]),
                ConversationMessage::user_text("third"),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "still working".to_string(),
                }]),
            ],
            metadata: None,
        };

        let result = prune_old_tool_results(&session, 10_000);

        std::env::set_current_dir(previous).expect("restore cwd");

        assert_eq!(result.pruned_tool_result_count, 1);
        assert!(estimate_session_tokens(&result.session) < estimate_session_tokens(&session));
        let archived_path = match &result.session.messages[1].blocks[0] {
            ContentBlock::ToolResult {
                compacted: true,
                output,
                archived_output_path,
                ..
            } => {
                assert!(output.is_empty());
                archived_output_path
                    .clone()
                    .expect("archived output path should be recorded")
            }
            _ => panic!("expected compacted tool result"),
        };
        let archived_output =
            fs::read_to_string(temp.join(&archived_path)).expect("archived tool result readable");
        fs::remove_dir_all(temp).expect("cleanup temp dir");

        assert_eq!(archived_output.len(), 250_000);
    }

    #[test]
    fn uses_placeholder_for_compacted_tool_result_context() {
        assert_eq!(
            get_tool_result_context_output("secret output", true),
            "[Old tool result content cleared]"
        );
        assert_eq!(
            get_tool_result_context_output("secret output", false),
            "secret output"
        );
    }

    #[test]
    fn extracts_key_files_from_message_content() {
        let files = collect_key_files(&[ConversationMessage::user_text(
            "Update rust/crates/runtime/src/compact.rs and rust/crates/rusty-claude-cli/src/main.rs next.",
        )]);
        assert!(files.contains(&"rust/crates/runtime/src/compact.rs".to_string()));
        assert!(files.contains(&"rust/crates/rusty-claude-cli/src/main.rs".to_string()));
    }

    #[test]
    fn infers_pending_work_from_recent_messages() {
        let pending = infer_pending_work(&[
            ConversationMessage::user_text("done"),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "Next: update tests and follow up on remaining CLI polish.".to_string(),
            }]),
        ]);
        assert_eq!(pending.len(), 1);
        assert!(pending[0].contains("Next: update tests"));
    }
}

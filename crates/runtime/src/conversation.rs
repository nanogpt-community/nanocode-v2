use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use crate::compact::{
    compact_session_with_summary, estimate_session_tokens, prepare_compaction,
    prune_old_tool_results, CompactionConfig, CompactionResult,
};
use crate::config::{RuntimeCompactionConfig, RuntimeFeatureConfig};
use crate::hooks::{HookAbortSignal, HookProgressReporter, HookRunResult, HookRunner};
use crate::permissions::{
    PermissionContext, PermissionOutcome, PermissionPolicy, PermissionPrompter,
};
use crate::session::{ContentBlock, ConversationMessage, Session};
use crate::usage::{TokenUsage, UsageTracker};

const DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD: u32 = 200_000;
const AUTO_COMPACTION_THRESHOLD_ENV_VAR: &str = "PEBBLE_AUTO_COMPACT_INPUT_TOKENS";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiRequest {
    pub system_prompt: Vec<String>,
    pub messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ThinkingSignature(String),
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    Usage(TokenUsage),
    MessageStop,
}

pub trait ApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError>;
}

pub trait ToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    message: String,
}

impl ToolError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ToolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ToolError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    message: String,
}

impl RuntimeError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RuntimeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnSummary {
    pub assistant_messages: Vec<ConversationMessage>,
    pub tool_results: Vec<ConversationMessage>,
    pub iterations: usize,
    pub usage: TokenUsage,
    pub auto_compaction: Option<AutoCompactionEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AutoCompactionEvent {
    pub removed_message_count: usize,
    pub pruned_tool_result_count: usize,
}

pub struct ConversationRuntime<C, T> {
    session: Session,
    api_client: C,
    tool_executor: T,
    permission_policy: PermissionPolicy,
    system_prompt: Vec<String>,
    max_iterations: usize,
    usage_tracker: UsageTracker,
    hook_runner: HookRunner,
    compaction_config: RuntimeCompactionConfig,
    auto_compaction_input_tokens_threshold: u32,
    hook_abort_signal: HookAbortSignal,
    hook_progress_reporter: Option<Box<dyn HookProgressReporter>>,
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    #[must_use]
    pub fn new(
        session: Session,
        api_client: C,
        tool_executor: T,
        permission_policy: PermissionPolicy,
        system_prompt: Vec<String>,
    ) -> Self {
        Self::new_with_features(
            session,
            api_client,
            tool_executor,
            permission_policy,
            system_prompt,
            &RuntimeFeatureConfig::default(),
        )
    }

    #[must_use]
    pub fn new_with_features(
        session: Session,
        api_client: C,
        tool_executor: T,
        permission_policy: PermissionPolicy,
        system_prompt: Vec<String>,
        feature_config: &RuntimeFeatureConfig,
    ) -> Self {
        let usage_tracker = UsageTracker::from_session(&session);
        Self {
            session,
            api_client,
            tool_executor,
            permission_policy,
            system_prompt,
            max_iterations: usize::MAX,
            usage_tracker,
            hook_runner: HookRunner::from_feature_config(feature_config),
            compaction_config: feature_config.compaction(),
            auto_compaction_input_tokens_threshold: auto_compaction_threshold_from_env(),
            hook_abort_signal: HookAbortSignal::default(),
            hook_progress_reporter: None,
        }
    }

    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub fn with_auto_compaction_input_tokens_threshold(mut self, threshold: u32) -> Self {
        self.auto_compaction_input_tokens_threshold = threshold;
        self
    }

    #[must_use]
    pub fn with_hook_abort_signal(mut self, hook_abort_signal: HookAbortSignal) -> Self {
        self.hook_abort_signal = hook_abort_signal;
        self
    }

    #[must_use]
    pub fn with_hook_progress_reporter(
        mut self,
        hook_progress_reporter: Box<dyn HookProgressReporter>,
    ) -> Self {
        self.hook_progress_reporter = Some(hook_progress_reporter);
        self
    }

    fn run_pre_tool_use_hook(&mut self, tool_name: &str, input: &str) -> HookRunResult {
        if let Some(reporter) = self.hook_progress_reporter.as_mut() {
            self.hook_runner.run_pre_tool_use_with_context(
                tool_name,
                input,
                Some(&self.hook_abort_signal),
                Some(reporter.as_mut()),
            )
        } else {
            self.hook_runner.run_pre_tool_use_with_context(
                tool_name,
                input,
                Some(&self.hook_abort_signal),
                None,
            )
        }
    }

    fn run_post_tool_use_hook(
        &mut self,
        tool_name: &str,
        input: &str,
        output: &str,
        is_error: bool,
    ) -> HookRunResult {
        if let Some(reporter) = self.hook_progress_reporter.as_mut() {
            self.hook_runner.run_post_tool_use_with_context(
                tool_name,
                input,
                output,
                is_error,
                Some(&self.hook_abort_signal),
                Some(reporter.as_mut()),
            )
        } else {
            self.hook_runner.run_post_tool_use_with_context(
                tool_name,
                input,
                output,
                is_error,
                Some(&self.hook_abort_signal),
                None,
            )
        }
    }

    fn run_post_tool_use_failure_hook(
        &mut self,
        tool_name: &str,
        input: &str,
        output: &str,
    ) -> HookRunResult {
        if let Some(reporter) = self.hook_progress_reporter.as_mut() {
            self.hook_runner.run_post_tool_use_failure_with_context(
                tool_name,
                input,
                output,
                Some(&self.hook_abort_signal),
                Some(reporter.as_mut()),
            )
        } else {
            self.hook_runner.run_post_tool_use_failure_with_context(
                tool_name,
                input,
                output,
                Some(&self.hook_abort_signal),
                None,
            )
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn run_turn(
        &mut self,
        user_input: impl Into<String>,
        mut prompter: Option<&mut dyn PermissionPrompter>,
    ) -> Result<TurnSummary, RuntimeError> {
        self.session
            .messages
            .push(ConversationMessage::user_text(user_input.into()));

        let mut assistant_messages = Vec::new();
        let mut tool_results = Vec::new();
        let mut iterations = 0;
        let mut max_turn_input_tokens = 0;
        let mut auto_compaction_event = AutoCompactionEvent::default();

        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(RuntimeError::new(
                    "conversation loop exceeded the maximum number of iterations",
                ));
            }

            let mut compacted_for_overflow = false;
            let events = loop {
                if !compacted_for_overflow {
                    if let Some(event) = self.maybe_auto_compact_by_estimate() {
                        auto_compaction_event.merge(event);
                    }
                }

                let request = ApiRequest {
                    system_prompt: self.system_prompt.clone(),
                    messages: self.session.messages.clone(),
                };

                match self.api_client.stream(request) {
                    Ok(events) => break events,
                    Err(error) if context_length_exceeded_error(&error) => {
                        let Some(event) = self.force_compact_for_context_overflow() else {
                            return Err(error);
                        };
                        compacted_for_overflow = true;
                        auto_compaction_event.merge(event);
                    }
                    Err(error) => return Err(error),
                }
            };
            let (assistant_message, usage) = build_assistant_message(events)?;
            if let Some(usage) = usage {
                max_turn_input_tokens = max_turn_input_tokens.max(usage.input_tokens);
                self.usage_tracker.record(usage);
            }
            let pending_tool_uses = assistant_message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => {
                        Some((id.clone(), name.clone(), input.clone()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();

            self.session.messages.push(assistant_message.clone());
            assistant_messages.push(assistant_message);

            if pending_tool_uses.is_empty() {
                break;
            }

            for (tool_use_id, tool_name, input) in pending_tool_uses {
                let pre_hook_result = self.run_pre_tool_use_hook(&tool_name, &input);
                let effective_input = pre_hook_result
                    .updated_input()
                    .map_or_else(|| input.clone(), ToOwned::to_owned);
                let permission_context = PermissionContext::new(
                    pre_hook_result.permission_override(),
                    pre_hook_result.permission_reason().map(ToOwned::to_owned),
                );

                let permission_outcome = if let Some(prompt) = prompter.as_mut() {
                    if pre_hook_result.is_cancelled() {
                        PermissionOutcome::Deny {
                            reason: format_hook_message(
                                &pre_hook_result,
                                &format!("PreToolUse hook cancelled tool `{tool_name}`"),
                            ),
                        }
                    } else if pre_hook_result.is_denied() {
                        PermissionOutcome::Deny {
                            reason: format_hook_message(
                                &pre_hook_result,
                                &format!("PreToolUse hook denied tool `{tool_name}`"),
                            ),
                        }
                    } else {
                        self.permission_policy.authorize_with_context(
                            &tool_name,
                            &effective_input,
                            &permission_context,
                            Some(*prompt),
                        )
                    }
                } else {
                    if pre_hook_result.is_cancelled() {
                        PermissionOutcome::Deny {
                            reason: format_hook_message(
                                &pre_hook_result,
                                &format!("PreToolUse hook cancelled tool `{tool_name}`"),
                            ),
                        }
                    } else if pre_hook_result.is_denied() {
                        PermissionOutcome::Deny {
                            reason: format_hook_message(
                                &pre_hook_result,
                                &format!("PreToolUse hook denied tool `{tool_name}`"),
                            ),
                        }
                    } else {
                        self.permission_policy.authorize_with_context(
                            &tool_name,
                            &effective_input,
                            &permission_context,
                            None,
                        )
                    }
                };

                let result_message = match permission_outcome {
                    PermissionOutcome::Allow => {
                        let (mut output, mut is_error) =
                            match self.tool_executor.execute(&tool_name, &effective_input) {
                                Ok(output) => (output, false),
                                Err(error) => (error.to_string(), true),
                            };
                        output = merge_hook_feedback(pre_hook_result.messages(), output, false);

                        let hook_output = output.clone();
                        let post_hook_result = if is_error {
                            self.run_post_tool_use_failure_hook(
                                &tool_name,
                                &effective_input,
                                &hook_output,
                            )
                        } else {
                            self.run_post_tool_use_hook(
                                &tool_name,
                                &effective_input,
                                &hook_output,
                                false,
                            )
                        };
                        if post_hook_result.is_denied() || post_hook_result.is_cancelled() {
                            is_error = true;
                        }
                        output = merge_hook_feedback(
                            post_hook_result.messages(),
                            output,
                            post_hook_result.is_denied() || post_hook_result.is_cancelled(),
                        );

                        ConversationMessage::tool_result(tool_use_id, tool_name, output, is_error)
                    }
                    PermissionOutcome::Deny { reason } => ConversationMessage::tool_result(
                        tool_use_id,
                        tool_name,
                        merge_hook_feedback(pre_hook_result.messages(), reason, true),
                        true,
                    ),
                };
                self.session.messages.push(result_message.clone());
                tool_results.push(result_message);
            }
        }

        if auto_compaction_event.is_empty() {
            if let Some(event) = self.maybe_auto_compact(max_turn_input_tokens) {
                auto_compaction_event.merge(event);
            }
        }
        let auto_compaction = (!auto_compaction_event.is_empty()).then_some(auto_compaction_event);

        Ok(TurnSummary {
            assistant_messages,
            tool_results,
            iterations,
            usage: self.usage_tracker.cumulative_usage(),
            auto_compaction,
        })
    }

    #[must_use]
    pub fn compact(&mut self, config: CompactionConfig) -> CompactionResult {
        let config = self.apply_configured_compaction_defaults(config);
        let summary = self.generate_compaction_summary(config).ok();
        compact_session_with_summary(&self.session, config, summary)
    }

    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        estimate_session_tokens(&self.session)
    }

    #[must_use]
    pub fn usage(&self) -> &UsageTracker {
        &self.usage_tracker
    }

    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn replace_session(&mut self, session: Session) {
        self.usage_tracker = UsageTracker::from_session(&session);
        self.session = session;
    }

    #[must_use]
    pub fn into_session(self) -> Session {
        self.session
    }

    fn maybe_auto_compact(&mut self, turn_input_tokens: u32) -> Option<AutoCompactionEvent> {
        if !self.compaction_config.auto {
            return None;
        }
        if turn_input_tokens < self.auto_compaction_input_tokens_threshold {
            return None;
        }

        self.apply_context_reduction(
            usize::try_from(self.auto_compaction_input_tokens_threshold).unwrap_or(usize::MAX),
            self.apply_configured_compaction_defaults(CompactionConfig {
                max_estimated_tokens: usize::try_from(self.auto_compaction_input_tokens_threshold)
                    .unwrap_or(usize::MAX),
                auto: true,
                ..CompactionConfig::default()
            }),
        )
    }

    fn maybe_auto_compact_by_estimate(&mut self) -> Option<AutoCompactionEvent> {
        if !self.compaction_config.auto {
            return None;
        }
        if self.estimated_tokens()
            < usize::try_from(self.auto_compaction_input_tokens_threshold).unwrap_or(usize::MAX)
        {
            return None;
        }
        self.apply_context_reduction(
            usize::try_from(self.auto_compaction_input_tokens_threshold).unwrap_or(usize::MAX),
            self.apply_configured_compaction_defaults(CompactionConfig {
                max_estimated_tokens: usize::try_from(self.auto_compaction_input_tokens_threshold)
                    .unwrap_or(usize::MAX),
                auto: true,
                ..CompactionConfig::default()
            }),
        )
    }

    fn force_compact_for_context_overflow(&mut self) -> Option<AutoCompactionEvent> {
        if !self.compaction_config.auto {
            return None;
        }
        self.apply_context_reduction(
            usize::try_from(self.auto_compaction_input_tokens_threshold).unwrap_or(usize::MAX),
            self.apply_configured_compaction_defaults(CompactionConfig {
                max_estimated_tokens: 0,
                preserve_recent_messages: 1,
                preserve_recent_tokens: Some(usize::MAX / 2),
                auto: true,
                overflow: true,
            }),
        )
    }

    fn apply_context_reduction(
        &mut self,
        prune_target_tokens: usize,
        config: CompactionConfig,
    ) -> Option<AutoCompactionEvent> {
        let mut event = AutoCompactionEvent::default();
        if self.compaction_config.prune {
            let prune_result = prune_old_tool_results(&self.session, prune_target_tokens);
            if prune_result.pruned_tool_result_count > 0 {
                self.session = prune_result.session;
                event.pruned_tool_result_count = prune_result.pruned_tool_result_count;
                if prune_target_tokens > 0 && self.estimated_tokens() < prune_target_tokens {
                    return Some(event);
                }
            }
        }

        let summary = self.generate_compaction_summary(config).ok();
        let result = compact_session_with_summary(&self.session, config, summary);
        if result.removed_message_count > 0 {
            self.session = result.compacted_session;
            event.removed_message_count = result.removed_message_count;
        }

        (!event.is_empty()).then_some(event)
    }

    fn apply_configured_compaction_defaults(
        &self,
        mut config: CompactionConfig,
    ) -> CompactionConfig {
        if let Some(tail_turns) = self.compaction_config.tail_turns {
            config.preserve_recent_messages = tail_turns;
        }
        if config.preserve_recent_tokens.is_none() {
            config.preserve_recent_tokens = self.compaction_config.preserve_recent_tokens;
        }
        config
    }

    fn generate_compaction_summary(
        &mut self,
        config: CompactionConfig,
    ) -> Result<String, RuntimeError> {
        let Some(prepared) = prepare_compaction(&self.session, config) else {
            return Err(RuntimeError::new(
                "session is below the compaction threshold",
            ));
        };
        let mut messages = prepared.summary_input_session.messages;
        messages.push(ConversationMessage::user_text(prepared.prompt));
        let events = self.api_client.stream(ApiRequest {
            system_prompt: vec![
                "You are a compaction agent. Summarize the supplied conversation state without using tools."
                    .to_string(),
            ],
            messages,
        })?;
        let (message, usage) = build_assistant_message(events)?;
        if let Some(usage) = usage {
            self.usage_tracker.record(usage);
        }
        let summary = message
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.trim()),
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if summary.is_empty() {
            return Err(RuntimeError::new(
                "compaction summary response did not include text",
            ));
        }
        Ok(summary)
    }
}

impl AutoCompactionEvent {
    fn is_empty(self) -> bool {
        self.removed_message_count == 0 && self.pruned_tool_result_count == 0
    }

    fn merge(&mut self, other: Self) {
        self.removed_message_count += other.removed_message_count;
        self.pruned_tool_result_count += other.pruned_tool_result_count;
    }
}

#[must_use]
pub fn auto_compaction_threshold_from_env() -> u32 {
    parse_auto_compaction_threshold(
        std::env::var(AUTO_COMPACTION_THRESHOLD_ENV_VAR)
            .ok()
            .as_deref(),
    )
}

#[must_use]
fn parse_auto_compaction_threshold(value: Option<&str>) -> u32 {
    value
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .filter(|threshold| *threshold > 0)
        .unwrap_or(DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD)
}

fn context_length_exceeded_error(error: &RuntimeError) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("context_length_exceeded")
        || message.contains("context length exceeded")
        || message.contains("maximum context length")
}

fn build_assistant_message(
    events: Vec<AssistantEvent>,
) -> Result<(ConversationMessage, Option<TokenUsage>), RuntimeError> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut blocks = Vec::new();
    let mut finished = false;
    let mut usage = None;

    for event in events {
        match event {
            AssistantEvent::TextDelta(delta) => {
                flush_thinking_block(&mut thinking, &mut thinking_signature, &mut blocks);
                text.push_str(&delta);
            }
            AssistantEvent::ThinkingDelta(delta) => {
                flush_text_block(&mut text, &mut blocks);
                thinking.push_str(&delta);
            }
            AssistantEvent::ThinkingSignature(signature) => {
                flush_text_block(&mut text, &mut blocks);
                thinking_signature = Some(signature);
            }
            AssistantEvent::ToolUse { id, name, input } => {
                flush_text_block(&mut text, &mut blocks);
                flush_thinking_block(&mut thinking, &mut thinking_signature, &mut blocks);
                blocks.push(ContentBlock::ToolUse { id, name, input });
            }
            AssistantEvent::Usage(value) => usage = Some(value),
            AssistantEvent::MessageStop => {
                finished = true;
            }
        }
    }

    flush_text_block(&mut text, &mut blocks);
    flush_thinking_block(&mut thinking, &mut thinking_signature, &mut blocks);

    if !finished {
        return Err(RuntimeError::new(
            "assistant stream ended without a message stop event",
        ));
    }
    if blocks.is_empty() {
        return Err(RuntimeError::new("assistant stream produced no content"));
    }

    Ok((
        ConversationMessage::assistant_with_usage(blocks, usage),
        usage,
    ))
}

fn flush_text_block(text: &mut String, blocks: &mut Vec<ContentBlock>) {
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text: std::mem::take(text),
        });
    }
}

fn format_hook_message(result: &HookRunResult, fallback: &str) -> String {
    if result.messages().is_empty() {
        fallback.to_string()
    } else {
        result.messages().join("\n")
    }
}

fn merge_hook_feedback(messages: &[String], output: String, denied: bool) -> String {
    if messages.is_empty() {
        return output;
    }

    let mut sections = Vec::new();
    if !output.trim().is_empty() {
        sections.push(output);
    }
    let label = if denied {
        "Hook feedback (denied)"
    } else {
        "Hook feedback"
    };
    sections.push(format!("{label}:\n{}", messages.join("\n")));
    sections.join("\n\n")
}

fn flush_thinking_block(
    text: &mut String,
    signature: &mut Option<String>,
    blocks: &mut Vec<ContentBlock>,
) {
    if text.is_empty() && signature.is_none() {
        return;
    }
    blocks.push(ContentBlock::Thinking {
        text: std::mem::take(text),
        signature: signature.take(),
    });
}

type ToolHandler = Box<dyn FnMut(&str) -> Result<String, ToolError>>;

#[derive(Default)]
pub struct StaticToolExecutor {
    handlers: BTreeMap<String, ToolHandler>,
}

impl StaticToolExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn register(
        mut self,
        tool_name: impl Into<String>,
        handler: impl FnMut(&str) -> Result<String, ToolError> + 'static,
    ) -> Self {
        self.handlers.insert(tool_name.into(), Box::new(handler));
        self
    }
}

impl ToolExecutor for StaticToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        self.handlers
            .get_mut(tool_name)
            .ok_or_else(|| ToolError::new(format!("unknown tool: {tool_name}")))?(input)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_auto_compaction_threshold, ApiClient, ApiRequest, AssistantEvent,
        AutoCompactionEvent, ConversationRuntime, RuntimeError, StaticToolExecutor,
        DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD,
    };
    use crate::compact::CompactionConfig;
    use crate::config::{RuntimeFeatureConfig, RuntimeHookConfig};
    use crate::permissions::{
        PermissionMode, PermissionPolicy, PermissionPromptDecision, PermissionPrompter,
        PermissionRequest,
    };
    use crate::prompt::{ProjectContext, SystemPromptBuilder};
    use crate::session::{ContentBlock, MessageRole, Session};
    use crate::usage::TokenUsage;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct ScriptedApiClient {
        call_count: usize,
    }

    impl ApiClient for ScriptedApiClient {
        fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
            self.call_count += 1;
            match self.call_count {
                1 => {
                    assert!(request
                        .messages
                        .iter()
                        .any(|message| message.role == MessageRole::User));
                    Ok(vec![
                        AssistantEvent::TextDelta("Let me calculate that.".to_string()),
                        AssistantEvent::ToolUse {
                            id: "tool-1".to_string(),
                            name: "add".to_string(),
                            input: "2,2".to_string(),
                        },
                        AssistantEvent::Usage(TokenUsage {
                            input_tokens: 20,
                            output_tokens: 6,
                            cache_creation_input_tokens: 1,
                            cache_read_input_tokens: 2,
                        }),
                        AssistantEvent::MessageStop,
                    ])
                }
                2 => {
                    let last_message = request
                        .messages
                        .last()
                        .expect("tool result should be present");
                    assert_eq!(last_message.role, MessageRole::Tool);
                    Ok(vec![
                        AssistantEvent::TextDelta("The answer is 4.".to_string()),
                        AssistantEvent::Usage(TokenUsage {
                            input_tokens: 24,
                            output_tokens: 4,
                            cache_creation_input_tokens: 1,
                            cache_read_input_tokens: 3,
                        }),
                        AssistantEvent::MessageStop,
                    ])
                }
                _ => Err(RuntimeError::new("unexpected extra API call")),
            }
        }
    }

    struct PromptAllowOnce;

    impl PermissionPrompter for PromptAllowOnce {
        fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
            assert_eq!(request.tool_name, "add");
            PermissionPromptDecision::Allow
        }
    }

    #[test]
    fn runs_user_to_tool_to_result_loop_end_to_end_and_tracks_usage() {
        let api_client = ScriptedApiClient { call_count: 0 };
        let tool_executor = StaticToolExecutor::new().register("add", |input| {
            let total = input
                .split(',')
                .map(|part| part.parse::<i32>().expect("input must be valid integer"))
                .sum::<i32>();
            Ok(total.to_string())
        });
        let permission_policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("add", PermissionMode::DangerFullAccess);
        let system_prompt = SystemPromptBuilder::new()
            .with_project_context(ProjectContext {
                cwd: PathBuf::from("/tmp/project"),
                current_date: "2026-03-31".to_string(),
                git_status: None,
                repository: None,
                instruction_files: Vec::new(),
                memory_files: Vec::new(),
            })
            .with_os("linux", "6.8")
            .build();
        let mut runtime = ConversationRuntime::new(
            Session::new(),
            api_client,
            tool_executor,
            permission_policy,
            system_prompt,
        );

        let summary = runtime
            .run_turn("what is 2 + 2?", Some(&mut PromptAllowOnce))
            .expect("conversation loop should succeed");

        assert_eq!(summary.iterations, 2);
        assert_eq!(summary.assistant_messages.len(), 2);
        assert_eq!(summary.tool_results.len(), 1);
        assert_eq!(runtime.session().messages.len(), 4);
        assert_eq!(summary.usage.output_tokens, 10);
        assert_eq!(summary.auto_compaction, None);
        assert!(matches!(
            runtime.session().messages[1].blocks[1],
            ContentBlock::ToolUse { .. }
        ));
        assert!(matches!(
            runtime.session().messages[2].blocks[0],
            ContentBlock::ToolResult {
                is_error: false,
                ..
            }
        ));
    }

    #[test]
    fn records_denied_tool_results_when_prompt_rejects() {
        struct RejectPrompter;
        impl PermissionPrompter for RejectPrompter {
            fn decide(&mut self, _request: &PermissionRequest) -> PermissionPromptDecision {
                PermissionPromptDecision::Deny {
                    reason: "not now".to_string(),
                }
            }
        }

        struct SingleCallApiClient;
        impl ApiClient for SingleCallApiClient {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                if request
                    .messages
                    .iter()
                    .any(|message| message.role == MessageRole::Tool)
                {
                    return Ok(vec![
                        AssistantEvent::TextDelta("I could not use the tool.".to_string()),
                        AssistantEvent::MessageStop,
                    ]);
                }
                Ok(vec![
                    AssistantEvent::ToolUse {
                        id: "tool-1".to_string(),
                        name: "blocked".to_string(),
                        input: "secret".to_string(),
                    },
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            SingleCallApiClient,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::WorkspaceWrite)
                .with_tool_requirement("blocked", PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        );

        let summary = runtime
            .run_turn("use the tool", Some(&mut RejectPrompter))
            .expect("conversation should continue after denied tool");

        assert_eq!(summary.tool_results.len(), 1);
        assert!(matches!(
            &summary.tool_results[0].blocks[0],
            ContentBlock::ToolResult { is_error: true, output, .. } if output == "not now"
        ));
    }

    #[test]
    fn denies_tool_use_when_pre_tool_hook_blocks() {
        struct SingleCallApiClient;
        impl ApiClient for SingleCallApiClient {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                if request
                    .messages
                    .iter()
                    .any(|message| message.role == MessageRole::Tool)
                {
                    return Ok(vec![
                        AssistantEvent::TextDelta("blocked".to_string()),
                        AssistantEvent::MessageStop,
                    ]);
                }
                Ok(vec![
                    AssistantEvent::ToolUse {
                        id: "tool-1".to_string(),
                        name: "blocked".to_string(),
                        input: r#"{"path":"secret.txt"}"#.to_string(),
                    },
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let mut runtime = ConversationRuntime::new_with_features(
            Session::new(),
            SingleCallApiClient,
            StaticToolExecutor::new().register("blocked", |_input| {
                panic!("tool should not execute when hook denies")
            }),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
            &RuntimeFeatureConfig::default().with_hooks(RuntimeHookConfig::new(
                vec![shell_snippet("printf 'blocked by hook'; exit 2")],
                Vec::new(),
                Vec::new(),
            )),
        );

        let summary = runtime
            .run_turn("use the tool", None)
            .expect("conversation should continue after hook denial");

        assert_eq!(summary.tool_results.len(), 1);
        let ContentBlock::ToolResult {
            is_error, output, ..
        } = &summary.tool_results[0].blocks[0]
        else {
            panic!("expected tool result block");
        };
        assert!(
            *is_error,
            "hook denial should produce an error result: {output}"
        );
        assert!(
            output.contains("denied tool") || output.contains("blocked by hook"),
            "unexpected hook denial output: {output:?}"
        );
    }

    #[test]
    fn appends_post_tool_hook_feedback_to_tool_result() {
        struct TwoCallApiClient {
            calls: usize,
        }

        impl ApiClient for TwoCallApiClient {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                self.calls += 1;
                match self.calls {
                    1 => Ok(vec![
                        AssistantEvent::ToolUse {
                            id: "tool-1".to_string(),
                            name: "add".to_string(),
                            input: r#"{"lhs":2,"rhs":2}"#.to_string(),
                        },
                        AssistantEvent::MessageStop,
                    ]),
                    2 => {
                        assert!(request
                            .messages
                            .iter()
                            .any(|message| message.role == MessageRole::Tool));
                        Ok(vec![
                            AssistantEvent::TextDelta("done".to_string()),
                            AssistantEvent::MessageStop,
                        ])
                    }
                    _ => Err(RuntimeError::new("unexpected extra API call")),
                }
            }
        }

        let mut runtime = ConversationRuntime::new_with_features(
            Session::new(),
            TwoCallApiClient { calls: 0 },
            StaticToolExecutor::new().register("add", |_input| Ok("4".to_string())),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
            &RuntimeFeatureConfig::default().with_hooks(RuntimeHookConfig::new(
                vec![shell_snippet("printf 'pre hook ran'")],
                vec![shell_snippet("printf 'post hook ran'")],
                Vec::new(),
            )),
        );

        let summary = runtime
            .run_turn("use add", None)
            .expect("tool loop succeeds");

        assert_eq!(summary.tool_results.len(), 1);
        let ContentBlock::ToolResult {
            is_error, output, ..
        } = &summary.tool_results[0].blocks[0]
        else {
            panic!("expected tool result block");
        };
        assert!(
            !*is_error,
            "post hook should preserve non-error result: {output:?}"
        );
        assert!(
            output.contains('4'),
            "tool output missing value: {output:?}"
        );
        assert!(
            output.contains("pre hook ran"),
            "tool output missing pre hook feedback: {output:?}"
        );
        assert!(
            output.contains("post hook ran"),
            "tool output missing post hook feedback: {output:?}"
        );
    }

    #[test]
    fn runs_post_tool_use_failure_hook_for_tool_errors() {
        struct TwoCallApiClient {
            calls: usize,
        }

        impl ApiClient for TwoCallApiClient {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                self.calls += 1;
                match self.calls {
                    1 => Ok(vec![
                        AssistantEvent::ToolUse {
                            id: "tool-1".to_string(),
                            name: "explode".to_string(),
                            input: r#"{"path":"boom"}"#.to_string(),
                        },
                        AssistantEvent::MessageStop,
                    ]),
                    2 => {
                        assert!(request
                            .messages
                            .iter()
                            .any(|message| message.role == MessageRole::Tool));
                        Ok(vec![
                            AssistantEvent::TextDelta("done".to_string()),
                            AssistantEvent::MessageStop,
                        ])
                    }
                    _ => Err(RuntimeError::new("unexpected extra API call")),
                }
            }
        }

        let mut runtime = ConversationRuntime::new_with_features(
            Session::new(),
            TwoCallApiClient { calls: 0 },
            StaticToolExecutor::new()
                .register("explode", |_input| Err(super::ToolError::new("kaboom"))),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
            &RuntimeFeatureConfig::default().with_hooks(RuntimeHookConfig::new(
                Vec::new(),
                Vec::new(),
                vec![shell_snippet("printf 'failure hook ran'")],
            )),
        );

        let summary = runtime
            .run_turn("use explode", None)
            .expect("tool loop succeeds");

        let ContentBlock::ToolResult {
            is_error, output, ..
        } = &summary.tool_results[0].blocks[0]
        else {
            panic!("expected tool result block");
        };
        assert!(*is_error, "tool error should stay an error: {output:?}");
        assert!(output.contains("kaboom"));
        assert!(output.contains("failure hook ran"));
    }

    #[test]
    fn reconstructs_usage_tracker_from_restored_session() {
        struct SimpleApi;
        impl ApiClient for SimpleApi {
            fn stream(
                &mut self,
                _request: ApiRequest,
            ) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let mut session = Session::new();
        session
            .messages
            .push(crate::session::ConversationMessage::assistant_with_usage(
                vec![ContentBlock::Text {
                    text: "earlier".to_string(),
                }],
                Some(TokenUsage {
                    input_tokens: 11,
                    output_tokens: 7,
                    cache_creation_input_tokens: 2,
                    cache_read_input_tokens: 1,
                }),
            ));

        let runtime = ConversationRuntime::new(
            session,
            SimpleApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::ReadOnly),
            vec!["system".to_string()],
        );

        assert_eq!(runtime.usage().turns(), 1);
        assert_eq!(runtime.usage().cumulative_usage().total_tokens(), 21);
    }

    #[test]
    fn compacts_session_after_turns() {
        struct SimpleApi;
        impl ApiClient for SimpleApi {
            fn stream(
                &mut self,
                _request: ApiRequest,
            ) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let _guard = crate::test_env_lock();
        let temp = std::env::temp_dir().join(format!(
            "runtime-conversation-compact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&temp).expect("set cwd");

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            SimpleApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::ReadOnly),
            vec!["system".to_string()],
        );
        runtime.run_turn("a", None).expect("turn a");
        runtime.run_turn("b", None).expect("turn b");
        runtime.run_turn("c", None).expect("turn c");

        let result = runtime.compact(CompactionConfig {
            preserve_recent_messages: 2,
            max_estimated_tokens: 1,
            preserve_recent_tokens: None,
            ..CompactionConfig::default()
        });
        assert_eq!(result.summary, "done");
        assert_eq!(
            result.compacted_session.messages[0].role,
            MessageRole::System
        );

        std::env::set_current_dir(previous).expect("restore cwd");
        std::fs::remove_dir_all(temp).expect("cleanup temp dir");
    }

    #[cfg(windows)]
    fn shell_snippet(script: &str) -> String {
        script.replace('\'', "\"")
    }

    #[cfg(not(windows))]
    fn shell_snippet(script: &str) -> String {
        script.to_string()
    }

    #[test]
    fn auto_compacts_when_turn_input_threshold_is_crossed() {
        struct SimpleApi;
        impl ApiClient for SimpleApi {
            fn stream(
                &mut self,
                _request: ApiRequest,
            ) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::Usage(TokenUsage {
                        input_tokens: 120_000,
                        output_tokens: 4,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let session = Session {
            version: 1,
            messages: vec![
                crate::session::ConversationMessage::user_text("one"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "two".to_string(),
                }]),
                crate::session::ConversationMessage::user_text("three"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "four".to_string(),
                }]),
            ],
            metadata: None,
        };

        let mut runtime = ConversationRuntime::new(
            session,
            SimpleApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_auto_compaction_input_tokens_threshold(1);

        let summary = runtime
            .run_turn("trigger", None)
            .expect("turn should succeed");

        assert_eq!(
            summary.auto_compaction,
            Some(AutoCompactionEvent {
                removed_message_count: 2,
                pruned_tool_result_count: 0,
            })
        );
        assert_eq!(runtime.session().messages[0].role, MessageRole::System);
    }

    #[test]
    fn preflight_compacts_before_sending_oversized_request() {
        struct AssertCompactedApi;
        impl ApiClient for AssertCompactedApi {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                if request
                    .system_prompt
                    .first()
                    .is_some_and(|prompt| prompt.contains("compaction agent"))
                {
                    return Ok(vec![
                        AssistantEvent::TextDelta("summary".to_string()),
                        AssistantEvent::MessageStop,
                    ]);
                }
                assert_eq!(
                    request.messages.first().map(|message| message.role),
                    Some(MessageRole::System)
                );
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let session = Session {
            version: 1,
            messages: vec![
                crate::session::ConversationMessage::user_text("one ".repeat(80)),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "two ".repeat(80),
                }]),
                crate::session::ConversationMessage::user_text("three ".repeat(80)),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "four ".repeat(80),
                }]),
            ],
            metadata: None,
        };

        let mut runtime = ConversationRuntime::new(
            session,
            AssertCompactedApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_auto_compaction_input_tokens_threshold(1);

        let summary = runtime
            .run_turn("trigger", None)
            .expect("turn should succeed");

        assert_eq!(
            summary.auto_compaction,
            Some(AutoCompactionEvent {
                removed_message_count: 2,
                pruned_tool_result_count: 0,
            })
        );
        assert_eq!(runtime.session().messages[0].role, MessageRole::System);
    }

    #[test]
    fn preflight_prunes_old_tool_results_before_full_compaction() {
        struct AssertPrunedApi;
        impl ApiClient for AssertPrunedApi {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                assert_ne!(
                    request.messages.first().map(|message| message.role),
                    Some(MessageRole::System)
                );
                assert!(request.messages.iter().any(|message| {
                    matches!(
                        message.blocks.first(),
                        Some(ContentBlock::ToolResult {
                            compacted: true,
                            ..
                        })
                    )
                }));
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let _guard = crate::test_env_lock();
        let temp = std::env::temp_dir().join(format!(
            "runtime-conversation-pruned-tool-results-{}",
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
                crate::session::ConversationMessage::user_text("first"),
                crate::session::ConversationMessage::tool_result(
                    "tool-1",
                    "bash",
                    "x".repeat(250_000),
                    false,
                ),
                crate::session::ConversationMessage::user_text("second"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "working".to_string(),
                }]),
                crate::session::ConversationMessage::user_text("third"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "still working".to_string(),
                }]),
            ],
            metadata: None,
        };

        let mut runtime = ConversationRuntime::new(
            session,
            AssertPrunedApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_auto_compaction_input_tokens_threshold(10_000);

        let summary = runtime
            .run_turn("trigger", None)
            .expect("turn should succeed");
        std::env::set_current_dir(previous).expect("restore cwd");

        assert_eq!(
            summary.auto_compaction,
            Some(AutoCompactionEvent {
                removed_message_count: 0,
                pruned_tool_result_count: 1,
            })
        );
        let archived_path = match &runtime.session().messages[1].blocks[0] {
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
    fn retries_after_context_length_exceeded_by_compacting_session() {
        struct OverflowThenSuccessApi {
            calls: usize,
        }

        impl ApiClient for OverflowThenSuccessApi {
            fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                self.calls += 1;
                if request
                    .system_prompt
                    .first()
                    .is_some_and(|prompt| prompt.contains("compaction agent"))
                {
                    return Ok(vec![
                        AssistantEvent::TextDelta("summary".to_string()),
                        AssistantEvent::MessageStop,
                    ]);
                }
                match self.calls {
                    1 => Err(RuntimeError::new(
                        "api stream returned context_length_exceeded",
                    )),
                    3 => {
                        assert_eq!(
                            request.messages.first().map(|message| message.role),
                            Some(MessageRole::System)
                        );
                        Ok(vec![
                            AssistantEvent::TextDelta("done".to_string()),
                            AssistantEvent::MessageStop,
                        ])
                    }
                    _ => Err(RuntimeError::new(format!(
                        "unexpected API call {}",
                        self.calls
                    ))),
                }
            }
        }

        let session = Session {
            version: 1,
            messages: vec![
                crate::session::ConversationMessage::user_text("one"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "two".to_string(),
                }]),
                crate::session::ConversationMessage::user_text("three"),
                crate::session::ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "four".to_string(),
                }]),
            ],
            metadata: None,
        };

        let mut runtime = ConversationRuntime::new(
            session,
            OverflowThenSuccessApi { calls: 0 },
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_auto_compaction_input_tokens_threshold(100_000);

        let summary = runtime
            .run_turn("trigger", None)
            .expect("turn should succeed after compaction retry");

        assert_eq!(
            summary.auto_compaction,
            Some(AutoCompactionEvent {
                removed_message_count: 4,
                pruned_tool_result_count: 0,
            })
        );
        assert_eq!(runtime.session().messages[0].role, MessageRole::System);
    }

    #[test]
    fn skips_auto_compaction_below_threshold() {
        struct SimpleApi;
        impl ApiClient for SimpleApi {
            fn stream(
                &mut self,
                _request: ApiRequest,
            ) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![
                    AssistantEvent::TextDelta("done".to_string()),
                    AssistantEvent::Usage(TokenUsage {
                        input_tokens: 99_999,
                        output_tokens: 4,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                    AssistantEvent::MessageStop,
                ])
            }
        }

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            SimpleApi,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_auto_compaction_input_tokens_threshold(100_000);

        let summary = runtime
            .run_turn("trigger", None)
            .expect("turn should succeed");
        assert_eq!(summary.auto_compaction, None);
        assert_eq!(runtime.session().messages.len(), 2);
    }

    #[test]
    fn auto_compaction_threshold_defaults_and_parses_values() {
        assert_eq!(
            parse_auto_compaction_threshold(None),
            DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD
        );
        assert_eq!(parse_auto_compaction_threshold(Some("4321")), 4321);
        assert_eq!(
            parse_auto_compaction_threshold(Some("not-a-number")),
            DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD
        );
        assert_eq!(
            parse_auto_compaction_threshold(Some("0")),
            DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD
        );
    }
}

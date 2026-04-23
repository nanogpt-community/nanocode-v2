use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use platform::write_atomic;

use crate::json::{JsonError, JsonValue};
use crate::usage::TokenUsage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    ToolResult {
        tool_use_id: String,
        tool_name: String,
        output: String,
        is_error: bool,
        compacted: bool,
        archived_output_path: Option<String>,
    },
    CompactionSummary {
        summary: String,
        recent_messages_preserved: bool,
        auto: bool,
        overflow: bool,
        tail_start_id: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationMessage {
    pub id: String,
    pub role: MessageRole,
    pub blocks: Vec<ContentBlock>,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMetadata {
    pub title: Option<String>,
    pub started_at: String,
    pub model: String,
    pub message_count: u32,
    pub last_prompt: Option<String>,
    pub permission_mode: Option<String>,
    pub thinking_enabled: Option<bool>,
    pub collaboration_mode: Option<String>,
    pub reasoning_effort: Option<String>,
    pub fast_mode: Option<bool>,
    pub proxy_tool_calls: Option<bool>,
    pub allowed_tools: Option<Vec<String>>,
    pub edit_history: Option<Vec<EditHistoryEntry>>,
    pub undo_stack: Option<Vec<SessionTurnSnapshot>>,
    pub redo_stack: Option<Vec<SessionTurnSnapshot>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditHistoryEntry {
    pub timestamp: String,
    pub tool_name: String,
    pub files: Vec<EditHistoryFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditHistoryFile {
    pub path: String,
    pub before: String,
    pub after: String,
    pub before_exists: bool,
    pub after_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTurnSnapshot {
    pub timestamp: String,
    pub message_count_before: u32,
    pub prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub files: Vec<EditHistoryFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub version: u32,
    pub messages: Vec<ConversationMessage>,
    pub metadata: Option<SessionMetadata>,
}

#[derive(Debug)]
pub enum SessionError {
    Io(std::io::Error),
    Json(JsonError),
    Format(String),
}

impl Display for SessionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::Format(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<std::io::Error> for SessionError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<JsonError> for SessionError {
    fn from(value: JsonError) -> Self {
        Self::Json(value)
    }
}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 1,
            messages: Vec::new(),
            metadata: None,
        }
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), SessionError> {
        write_atomic(path, self.to_json().render())?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let contents = fs::read_to_string(path)?;
        Self::from_json(&JsonValue::parse(&contents)?)
    }

    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        object.insert(
            "version".to_string(),
            JsonValue::Number(i64::from(self.version)),
        );
        object.insert(
            "messages".to_string(),
            JsonValue::Array(
                self.messages
                    .iter()
                    .map(ConversationMessage::to_json)
                    .collect(),
            ),
        );
        if let Some(metadata) = &self.metadata {
            object.insert("metadata".to_string(), metadata.to_json());
        }
        JsonValue::Object(object)
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value
            .as_object()
            .ok_or_else(|| SessionError::Format("session must be an object".to_string()))?;
        let version = object
            .get("version")
            .and_then(JsonValue::as_i64)
            .ok_or_else(|| SessionError::Format("missing version".to_string()))?;
        let version = u32::try_from(version)
            .map_err(|_| SessionError::Format("version out of range".to_string()))?;
        let messages = object
            .get("messages")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SessionError::Format("missing messages".to_string()))?
            .iter()
            .map(ConversationMessage::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let metadata = object
            .get("metadata")
            .map(SessionMetadata::from_json)
            .transpose()?;
        Ok(Self {
            version,
            messages,
            metadata,
        })
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

static MESSAGE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_message_id() -> String {
    let counter = MESSAGE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("msg-{timestamp:x}-{counter:x}")
}

impl SessionMetadata {
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        if let Some(title) = &self.title {
            object.insert("title".to_string(), JsonValue::String(title.clone()));
        }
        object.insert(
            "started_at".to_string(),
            JsonValue::String(self.started_at.clone()),
        );
        object.insert("model".to_string(), JsonValue::String(self.model.clone()));
        object.insert(
            "message_count".to_string(),
            JsonValue::Number(i64::from(self.message_count)),
        );
        if let Some(last_prompt) = &self.last_prompt {
            object.insert(
                "last_prompt".to_string(),
                JsonValue::String(last_prompt.clone()),
            );
        }
        if let Some(permission_mode) = &self.permission_mode {
            object.insert(
                "permission_mode".to_string(),
                JsonValue::String(permission_mode.clone()),
            );
        }
        if let Some(thinking_enabled) = self.thinking_enabled {
            object.insert(
                "thinking_enabled".to_string(),
                JsonValue::Bool(thinking_enabled),
            );
        }
        if let Some(collaboration_mode) = &self.collaboration_mode {
            object.insert(
                "collaboration_mode".to_string(),
                JsonValue::String(collaboration_mode.clone()),
            );
        }
        if let Some(reasoning_effort) = &self.reasoning_effort {
            object.insert(
                "reasoning_effort".to_string(),
                JsonValue::String(reasoning_effort.clone()),
            );
        }
        if let Some(fast_mode) = self.fast_mode {
            object.insert("fast_mode".to_string(), JsonValue::Bool(fast_mode));
        }
        if let Some(proxy_tool_calls) = self.proxy_tool_calls {
            object.insert(
                "proxy_tool_calls".to_string(),
                JsonValue::Bool(proxy_tool_calls),
            );
        }
        if let Some(allowed_tools) = &self.allowed_tools {
            object.insert(
                "allowed_tools".to_string(),
                JsonValue::Array(
                    allowed_tools
                        .iter()
                        .cloned()
                        .map(JsonValue::String)
                        .collect(),
                ),
            );
        }
        if let Some(edit_history) = &self.edit_history {
            object.insert(
                "edit_history".to_string(),
                JsonValue::Array(edit_history.iter().map(EditHistoryEntry::to_json).collect()),
            );
        }
        if let Some(undo_stack) = &self.undo_stack {
            object.insert(
                "undo_stack".to_string(),
                JsonValue::Array(
                    undo_stack
                        .iter()
                        .map(SessionTurnSnapshot::to_json)
                        .collect(),
                ),
            );
        }
        if let Some(redo_stack) = &self.redo_stack {
            object.insert(
                "redo_stack".to_string(),
                JsonValue::Array(
                    redo_stack
                        .iter()
                        .map(SessionTurnSnapshot::to_json)
                        .collect(),
                ),
            );
        }
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value.as_object().ok_or_else(|| {
            SessionError::Format("session metadata must be an object".to_string())
        })?;
        Ok(Self {
            title: optional_string(object, "title"),
            started_at: required_string(object, "started_at")?,
            model: required_string(object, "model")?,
            message_count: required_u32(object, "message_count")?,
            last_prompt: optional_string(object, "last_prompt"),
            permission_mode: optional_string(object, "permission_mode"),
            thinking_enabled: optional_bool(object, "thinking_enabled"),
            collaboration_mode: optional_string(object, "collaboration_mode"),
            reasoning_effort: optional_string(object, "reasoning_effort"),
            fast_mode: optional_bool(object, "fast_mode"),
            proxy_tool_calls: optional_bool(object, "proxy_tool_calls"),
            allowed_tools: optional_string_array(object, "allowed_tools")?,
            edit_history: optional_edit_history(object, "edit_history")?,
            undo_stack: optional_turn_snapshots(object, "undo_stack")?,
            redo_stack: optional_turn_snapshots(object, "redo_stack")?,
        })
    }
}

impl EditHistoryEntry {
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        object.insert(
            "timestamp".to_string(),
            JsonValue::String(self.timestamp.clone()),
        );
        object.insert(
            "tool_name".to_string(),
            JsonValue::String(self.tool_name.clone()),
        );
        object.insert(
            "files".to_string(),
            JsonValue::Array(self.files.iter().map(EditHistoryFile::to_json).collect()),
        );
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value.as_object().ok_or_else(|| {
            SessionError::Format("edit history entry must be an object".to_string())
        })?;
        Ok(Self {
            timestamp: required_string(object, "timestamp")?,
            tool_name: required_string(object, "tool_name")?,
            files: required_edit_history_files(object, "files")?,
        })
    }
}

impl EditHistoryFile {
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        object.insert("path".to_string(), JsonValue::String(self.path.clone()));
        object.insert("before".to_string(), JsonValue::String(self.before.clone()));
        object.insert("after".to_string(), JsonValue::String(self.after.clone()));
        object.insert(
            "before_exists".to_string(),
            JsonValue::Bool(self.before_exists),
        );
        object.insert(
            "after_exists".to_string(),
            JsonValue::Bool(self.after_exists),
        );
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value.as_object().ok_or_else(|| {
            SessionError::Format("edit history file must be an object".to_string())
        })?;
        Ok(Self {
            path: required_string(object, "path")?,
            before: required_string(object, "before")?,
            after: required_string(object, "after")?,
            before_exists: object
                .get("before_exists")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true),
            after_exists: object
                .get("after_exists")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true),
        })
    }
}

impl SessionTurnSnapshot {
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        object.insert(
            "timestamp".to_string(),
            JsonValue::String(self.timestamp.clone()),
        );
        object.insert(
            "message_count_before".to_string(),
            JsonValue::Number(i64::from(self.message_count_before)),
        );
        if let Some(prompt) = &self.prompt {
            object.insert("prompt".to_string(), JsonValue::String(prompt.clone()));
        }
        object.insert(
            "messages".to_string(),
            JsonValue::Array(
                self.messages
                    .iter()
                    .map(ConversationMessage::to_json)
                    .collect(),
            ),
        );
        object.insert(
            "files".to_string(),
            JsonValue::Array(self.files.iter().map(EditHistoryFile::to_json).collect()),
        );
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value.as_object().ok_or_else(|| {
            SessionError::Format("session turn snapshot must be an object".to_string())
        })?;
        let messages = object
            .get("messages")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SessionError::Format("missing messages".to_string()))?
            .iter()
            .map(ConversationMessage::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            timestamp: required_string(object, "timestamp")?,
            message_count_before: required_u32(object, "message_count_before")?,
            prompt: optional_string(object, "prompt"),
            messages,
            files: required_edit_history_files(object, "files")?,
        })
    }
}

impl ConversationMessage {
    #[must_use]
    pub fn new(role: MessageRole, blocks: Vec<ContentBlock>, usage: Option<TokenUsage>) -> Self {
        Self {
            id: generate_message_id(),
            role,
            blocks,
            usage,
        }
    }

    #[must_use]
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::new(
            MessageRole::User,
            vec![ContentBlock::Text { text: text.into() }],
            None,
        )
    }

    #[must_use]
    pub fn assistant(blocks: Vec<ContentBlock>) -> Self {
        Self::new(MessageRole::Assistant, blocks, None)
    }

    #[must_use]
    pub fn assistant_with_usage(blocks: Vec<ContentBlock>, usage: Option<TokenUsage>) -> Self {
        Self::new(MessageRole::Assistant, blocks, usage)
    }

    #[must_use]
    pub fn compaction_summary(
        summary: impl Into<String>,
        recent_messages_preserved: bool,
        auto: bool,
        overflow: bool,
        tail_start_id: Option<String>,
    ) -> Self {
        Self::new(
            MessageRole::System,
            vec![ContentBlock::CompactionSummary {
                summary: summary.into(),
                recent_messages_preserved,
                auto,
                overflow,
                tail_start_id,
            }],
            None,
        )
    }

    #[must_use]
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            MessageRole::Tool,
            vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                tool_name: tool_name.into(),
                output: output.into(),
                is_error,
                compacted: false,
                archived_output_path: None,
            }],
            None,
        )
    }

    #[must_use]
    pub fn compacted_tool_result(
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            MessageRole::Tool,
            vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                tool_name: tool_name.into(),
                output: output.into(),
                is_error,
                compacted: true,
                archived_output_path: None,
            }],
            None,
        )
    }

    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        object.insert("id".to_string(), JsonValue::String(self.id.clone()));
        object.insert(
            "role".to_string(),
            JsonValue::String(
                match self.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                }
                .to_string(),
            ),
        );
        object.insert(
            "blocks".to_string(),
            JsonValue::Array(self.blocks.iter().map(ContentBlock::to_json).collect()),
        );
        if let Some(usage) = self.usage {
            object.insert("usage".to_string(), usage_to_json(usage));
        }
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value
            .as_object()
            .ok_or_else(|| SessionError::Format("message must be an object".to_string()))?;
        let role = match object
            .get("role")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| SessionError::Format("missing role".to_string()))?
        {
            "system" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            other => {
                return Err(SessionError::Format(format!(
                    "unsupported message role: {other}"
                )))
            }
        };
        let blocks = object
            .get("blocks")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SessionError::Format("missing blocks".to_string()))?
            .iter()
            .map(ContentBlock::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let usage = object.get("usage").map(usage_from_json).transpose()?;
        Ok(Self {
            id: optional_string(object, "id").unwrap_or_else(generate_message_id),
            role,
            blocks,
            usage,
        })
    }
}

impl ContentBlock {
    #[must_use]
    pub fn to_json(&self) -> JsonValue {
        let mut object = BTreeMap::new();
        match self {
            Self::Text { text } => {
                object.insert("type".to_string(), JsonValue::String("text".to_string()));
                object.insert("text".to_string(), JsonValue::String(text.clone()));
            }
            Self::Thinking { text, signature } => {
                object.insert(
                    "type".to_string(),
                    JsonValue::String("thinking".to_string()),
                );
                object.insert("text".to_string(), JsonValue::String(text.clone()));
                if let Some(signature) = signature {
                    object.insert(
                        "signature".to_string(),
                        JsonValue::String(signature.clone()),
                    );
                }
            }
            Self::ToolUse { id, name, input } => {
                object.insert(
                    "type".to_string(),
                    JsonValue::String("tool_use".to_string()),
                );
                object.insert("id".to_string(), JsonValue::String(id.clone()));
                object.insert("name".to_string(), JsonValue::String(name.clone()));
                object.insert("input".to_string(), JsonValue::String(input.clone()));
            }
            Self::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
                compacted,
                archived_output_path,
            } => {
                object.insert(
                    "type".to_string(),
                    JsonValue::String("tool_result".to_string()),
                );
                object.insert(
                    "tool_use_id".to_string(),
                    JsonValue::String(tool_use_id.clone()),
                );
                object.insert(
                    "tool_name".to_string(),
                    JsonValue::String(tool_name.clone()),
                );
                object.insert("output".to_string(), JsonValue::String(output.clone()));
                object.insert("is_error".to_string(), JsonValue::Bool(*is_error));
                if *compacted {
                    object.insert("compacted".to_string(), JsonValue::Bool(true));
                }
                if let Some(archived_output_path) = archived_output_path {
                    object.insert(
                        "archived_output_path".to_string(),
                        JsonValue::String(archived_output_path.clone()),
                    );
                }
            }
            Self::CompactionSummary {
                summary,
                recent_messages_preserved,
                auto,
                overflow,
                tail_start_id,
            } => {
                object.insert(
                    "type".to_string(),
                    JsonValue::String("compaction_summary".to_string()),
                );
                object.insert("summary".to_string(), JsonValue::String(summary.clone()));
                object.insert(
                    "recent_messages_preserved".to_string(),
                    JsonValue::Bool(*recent_messages_preserved),
                );
                object.insert("auto".to_string(), JsonValue::Bool(*auto));
                object.insert("overflow".to_string(), JsonValue::Bool(*overflow));
                if let Some(tail_start_id) = tail_start_id {
                    object.insert(
                        "tail_start_id".to_string(),
                        JsonValue::String(tail_start_id.clone()),
                    );
                }
            }
        }
        JsonValue::Object(object)
    }

    fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value
            .as_object()
            .ok_or_else(|| SessionError::Format("block must be an object".to_string()))?;
        match object
            .get("type")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| SessionError::Format("missing block type".to_string()))?
        {
            "text" => Ok(Self::Text {
                text: required_string(object, "text")?,
            }),
            "thinking" => Ok(Self::Thinking {
                text: required_string(object, "text")?,
                signature: optional_string(object, "signature"),
            }),
            "tool_use" => Ok(Self::ToolUse {
                id: required_string(object, "id")?,
                name: required_string(object, "name")?,
                input: required_string(object, "input")?,
            }),
            "tool_result" => Ok(Self::ToolResult {
                tool_use_id: required_string(object, "tool_use_id")?,
                tool_name: required_string(object, "tool_name")?,
                output: required_string(object, "output")?,
                is_error: object
                    .get("is_error")
                    .and_then(JsonValue::as_bool)
                    .ok_or_else(|| SessionError::Format("missing is_error".to_string()))?,
                compacted: object
                    .get("compacted")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                archived_output_path: optional_string(object, "archived_output_path"),
            }),
            "compaction_summary" => Ok(Self::CompactionSummary {
                summary: required_string(object, "summary")?,
                recent_messages_preserved: object
                    .get("recent_messages_preserved")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                auto: object
                    .get("auto")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                overflow: object
                    .get("overflow")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                tail_start_id: optional_string(object, "tail_start_id"),
            }),
            other => Err(SessionError::Format(format!(
                "unsupported block type: {other}"
            ))),
        }
    }
}

fn usage_to_json(usage: TokenUsage) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert(
        "input_tokens".to_string(),
        JsonValue::Number(i64::from(usage.input_tokens)),
    );
    object.insert(
        "output_tokens".to_string(),
        JsonValue::Number(i64::from(usage.output_tokens)),
    );
    object.insert(
        "cache_creation_input_tokens".to_string(),
        JsonValue::Number(i64::from(usage.cache_creation_input_tokens)),
    );
    object.insert(
        "cache_read_input_tokens".to_string(),
        JsonValue::Number(i64::from(usage.cache_read_input_tokens)),
    );
    JsonValue::Object(object)
}

fn usage_from_json(value: &JsonValue) -> Result<TokenUsage, SessionError> {
    let object = value
        .as_object()
        .ok_or_else(|| SessionError::Format("usage must be an object".to_string()))?;
    Ok(TokenUsage {
        input_tokens: required_u32(object, "input_tokens")?,
        output_tokens: required_u32(object, "output_tokens")?,
        cache_creation_input_tokens: required_u32(object, "cache_creation_input_tokens")?,
        cache_read_input_tokens: required_u32(object, "cache_read_input_tokens")?,
    })
}

fn required_string(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<String, SessionError> {
    object
        .get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| SessionError::Format(format!("missing {key}")))
}

fn required_u32(object: &BTreeMap<String, JsonValue>, key: &str) -> Result<u32, SessionError> {
    let value = object
        .get(key)
        .and_then(JsonValue::as_i64)
        .ok_or_else(|| SessionError::Format(format!("missing {key}")))?;
    u32::try_from(value).map_err(|_| SessionError::Format(format!("{key} out of range")))
}

fn optional_string(object: &BTreeMap<String, JsonValue>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

fn optional_bool(object: &BTreeMap<String, JsonValue>, key: &str) -> Option<bool> {
    object.get(key).and_then(JsonValue::as_bool)
}

fn optional_string_array(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Option<Vec<String>>, SessionError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| SessionError::Format(format!("{key} must be an array")))?;
    let mut strings = Vec::with_capacity(values.len());
    for item in values {
        strings.push(
            item.as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| SessionError::Format(format!("{key} entries must be strings")))?,
        );
    }
    Ok(Some(strings))
}

fn optional_edit_history(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Option<Vec<EditHistoryEntry>>, SessionError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| SessionError::Format(format!("{key} must be an array")))?;
    values
        .iter()
        .map(EditHistoryEntry::from_json)
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn optional_turn_snapshots(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Option<Vec<SessionTurnSnapshot>>, SessionError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| SessionError::Format(format!("{key} must be an array")))?;
    values
        .iter()
        .map(SessionTurnSnapshot::from_json)
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn required_edit_history_files(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Vec<EditHistoryFile>, SessionError> {
    let values = object
        .get(key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| SessionError::Format(format!("missing {key}")))?;
    values.iter().map(EditHistoryFile::from_json).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ContentBlock, ConversationMessage, EditHistoryEntry, EditHistoryFile, MessageRole, Session,
        SessionMetadata, SessionTurnSnapshot,
    };
    use crate::usage::TokenUsage;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    #[allow(clippy::too_many_lines)]
    fn persists_and_restores_session_json() {
        let mut session = Session::new();
        session.metadata = Some(SessionMetadata {
            title: Some("Demo session".to_string()),
            started_at: "2026-04-01T00:00:00Z".to_string(),
            model: "zai-org/glm-5.1".to_string(),
            message_count: 3,
            last_prompt: Some("hello".to_string()),
            permission_mode: Some("workspace-write".to_string()),
            thinking_enabled: Some(true),
            collaboration_mode: Some("plan".to_string()),
            reasoning_effort: Some("medium".to_string()),
            fast_mode: Some(true),
            proxy_tool_calls: Some(false),
            allowed_tools: Some(vec!["read_file".to_string(), "glob_search".to_string()]),
            edit_history: Some(vec![EditHistoryEntry {
                timestamp: "2026-04-01T00:01:00Z".to_string(),
                tool_name: "edit_file".to_string(),
                files: vec![EditHistoryFile {
                    path: "src/lib.rs".to_string(),
                    before: "old".to_string(),
                    after: "new".to_string(),
                    before_exists: true,
                    after_exists: true,
                }],
            }]),
            undo_stack: Some(vec![SessionTurnSnapshot {
                timestamp: "2026-04-01T00:02:00Z".to_string(),
                message_count_before: 0,
                prompt: Some("hello".to_string()),
                messages: vec![ConversationMessage::user_text("hello")],
                files: vec![EditHistoryFile {
                    path: "src/main.rs".to_string(),
                    before: String::new(),
                    after: "fn main() {}\n".to_string(),
                    before_exists: false,
                    after_exists: true,
                }],
            }]),
            redo_stack: Some(Vec::new()),
        });
        session
            .messages
            .push(ConversationMessage::user_text("hello"));
        session
            .messages
            .push(ConversationMessage::assistant_with_usage(
                vec![
                    ContentBlock::Text {
                        text: "thinking".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "bash".to_string(),
                        input: "echo hi".to_string(),
                    },
                ],
                Some(TokenUsage {
                    input_tokens: 10,
                    output_tokens: 4,
                    cache_creation_input_tokens: 1,
                    cache_read_input_tokens: 2,
                }),
            ));
        session.messages.push(ConversationMessage::tool_result(
            "tool-1", "bash", "hi", false,
        ));
        session
            .messages
            .push(ConversationMessage::compacted_tool_result(
                "tool-2",
                "bash",
                "long output",
                false,
            ));
        if let ContentBlock::ToolResult {
            archived_output_path,
            ..
        } = &mut session.messages[3].blocks[0]
        {
            *archived_output_path = Some(".pebble/tool-results/tool-2-bash.txt".to_string());
        }

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("runtime-session-{nanos}.json"));
        session.save_to_path(&path).expect("session should save");
        let restored = Session::load_from_path(&path).expect("session should load");
        fs::remove_file(&path).expect("temp file should be removable");

        assert_eq!(restored, session);
        assert_eq!(restored.messages[2].role, MessageRole::Tool);
        assert!(matches!(
            &restored.messages[3].blocks[0],
            ContentBlock::ToolResult {
                compacted: true,
                archived_output_path: Some(path),
                ..
            } if path == ".pebble/tool-results/tool-2-bash.txt"
        ));
        assert_eq!(
            restored.messages[1].usage.expect("usage").total_tokens(),
            17
        );
    }
}

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::usage::TokenUsage;

const TRACE_PREVIEW_CHARS: usize = 512;
pub const TURN_TRACE_SCHEMA_VERSION: u32 = 2;
pub const LEGACY_TURN_TRACE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TracePayloadSummary {
    pub chars: usize,
    pub sha256: String,
    pub preview: String,
    pub truncated: bool,
    #[serde(default)]
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiCallTrace {
    pub iteration: usize,
    pub request_message_count: usize,
    pub request_estimated_tokens: usize,
    pub duration_ms: u128,
    pub result_event_count: Option<usize>,
    pub usage: Option<TokenUsage>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionTrace {
    pub iteration: usize,
    pub tool_use_id: String,
    pub tool_name: String,
    pub outcome: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallTrace {
    pub iteration: usize,
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: TracePayloadSummary,
    pub effective_input: Option<TracePayloadSummary>,
    pub output: TracePayloadSummary,
    pub duration_ms: u128,
    pub permission_outcome: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionTrace {
    pub trigger: String,
    pub removed_message_count: usize,
    pub pruned_tool_result_count: usize,
    pub estimated_tokens_after: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnTrace {
    #[serde(default = "legacy_turn_trace_schema_version")]
    pub schema_version: u32,
    pub started_at_unix_ms: u128,
    pub duration_ms: Option<u128>,
    pub initial_message_count: usize,
    pub final_message_count: Option<usize>,
    pub user_input: TracePayloadSummary,
    pub api_calls: Vec<ApiCallTrace>,
    pub permissions: Vec<PermissionTrace>,
    pub tool_calls: Vec<ToolCallTrace>,
    pub compactions: Vec<CompactionTrace>,
    pub errors: Vec<String>,
}

impl TracePayloadSummary {
    #[must_use]
    pub fn from_text(value: &str) -> Self {
        let chars = value.chars().count();
        let (redacted_value, redacted) = redact_trace_text(value);
        let preview = redacted_value
            .chars()
            .take(TRACE_PREVIEW_CHARS)
            .collect::<String>();
        let truncated =
            chars > TRACE_PREVIEW_CHARS || redacted_value.chars().count() > TRACE_PREVIEW_CHARS;
        let sha256 = sha256_hex(&redacted_value);
        Self {
            chars,
            sha256,
            preview,
            truncated,
            redacted,
        }
    }

    pub fn redact_sensitive_data(&mut self) {
        let (redacted_preview, redacted) = redact_trace_text(&self.preview);
        if redacted {
            self.preview = redacted_preview
                .chars()
                .take(TRACE_PREVIEW_CHARS)
                .collect::<String>();
            self.sha256 = sha256_hex(&self.preview);
            self.redacted = true;
        }
    }
}

impl TurnTrace {
    #[must_use]
    pub fn start(user_input: &str, initial_message_count: usize) -> Self {
        Self {
            schema_version: TURN_TRACE_SCHEMA_VERSION,
            started_at_unix_ms: unix_timestamp_ms(),
            duration_ms: None,
            initial_message_count,
            final_message_count: None,
            user_input: TracePayloadSummary::from_text(user_input),
            api_calls: Vec::new(),
            permissions: Vec::new(),
            tool_calls: Vec::new(),
            compactions: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn finish(&mut self, final_message_count: usize) {
        self.final_message_count = Some(final_message_count);
        self.duration_ms = unix_timestamp_ms().checked_sub(self.started_at_unix_ms);
    }

    pub fn record_error(&mut self, error: impl Into<String>) {
        let (error, _) = redact_trace_text(&error.into());
        self.errors.push(error);
    }

    #[must_use]
    pub fn redacted(&self) -> Self {
        let mut trace = self.clone();
        trace.redact_sensitive_data();
        trace
    }

    pub fn redact_sensitive_data(&mut self) {
        self.user_input.redact_sensitive_data();
        for api_call in &mut self.api_calls {
            if let Some(error) = api_call.error.as_mut() {
                *error = redact_trace_text(error).0;
            }
        }
        for permission in &mut self.permissions {
            if let Some(reason) = permission.reason.as_mut() {
                *reason = redact_trace_text(reason).0;
            }
        }
        for tool_call in &mut self.tool_calls {
            tool_call.input.redact_sensitive_data();
            if let Some(effective_input) = tool_call.effective_input.as_mut() {
                effective_input.redact_sensitive_data();
            }
            tool_call.output.redact_sensitive_data();
        }
        for error in &mut self.errors {
            *error = redact_trace_text(error).0;
        }
    }

    pub fn normalize_loaded(&mut self) {
        if self.schema_version == 0 {
            self.schema_version = LEGACY_TURN_TRACE_SCHEMA_VERSION;
        }
        self.redact_sensitive_data();
    }
}

const fn legacy_turn_trace_schema_version() -> u32 {
    LEGACY_TURN_TRACE_SCHEMA_VERSION
}

#[must_use]
pub fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[must_use]
pub fn redact_trace_text(value: &str) -> (String, bool) {
    if value.is_empty() {
        return (String::new(), false);
    }

    let mut redacted = value.to_string();
    for (pattern, replacement) in redaction_patterns() {
        redacted = pattern.replace_all(&redacted, *replacement).into_owned();
    }
    let changed = redacted != value;
    (redacted, changed)
}

fn redaction_patterns() -> &'static [(Regex, &'static str)] {
    static PATTERNS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            (
                Regex::new(r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
                    .expect("private key redaction regex should compile"),
                "[REDACTED PRIVATE KEY]",
            ),
            (
                Regex::new(r"(?i)\b(authorization\s*[:=]\s*)(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
                    .expect("authorization redaction regex should compile"),
                "$1$2 [REDACTED]",
            ),
            (
                Regex::new(r#"(?i)(["']?)(api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|session[_-]?token|client[_-]?secret|secret|password|passwd|pwd|stripe[_-]?key)(["']?)(\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,}\]]+)"#)
                    .expect("key-value secret redaction regex should compile"),
                "$1$2$3$4[REDACTED]",
            ),
            (
                Regex::new(r"://[^/\s:@]+:[^/\s:@]+@")
                    .expect("url credential redaction regex should compile"),
                "://[REDACTED]@",
            ),
            (
                Regex::new(r"(?i)\b(sk[-_](?:live|test|proj|ant)?[-_]?[A-Za-z0-9_-]{16,}|gh[psuor]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{16,})\b")
                    .expect("token prefix redaction regex should compile"),
                "[REDACTED]",
            ),
            (
                Regex::new(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
                    .expect("jwt redaction regex should compile"),
                "[REDACTED JWT]",
            ),
        ]
    })
}

fn sha256_hex(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::{
        redact_trace_text, TracePayloadSummary, TurnTrace, LEGACY_TURN_TRACE_SCHEMA_VERSION,
        TURN_TRACE_SCHEMA_VERSION,
    };

    #[test]
    fn payload_summary_redacts_common_secret_shapes() {
        let summary = TracePayloadSummary::from_text(
            r#"{"api_key":"sk-live-abcdefghijklmnopqrstuvwxyz","safe":"visible"}"#,
        );

        assert!(summary.redacted);
        assert!(summary.preview.contains("[REDACTED]"));
        assert!(summary.preview.contains("visible"));
        assert!(!summary.preview.contains("abcdefghijklmnopqrstuvwxyz"));
    }

    #[test]
    fn redacts_authorization_headers_and_jwts() {
        let (redacted, changed) = redact_trace_text(
            "Authorization: Bearer eyJabc.def.ghi\naccess_token=github_pat_abcdefghijklmnopqrstuvwxyz",
        );

        assert!(changed);
        assert!(redacted.contains("Bearer [REDACTED]"));
        assert!(redacted.contains("access_token=[REDACTED]"));
        assert!(!redacted.contains("github_pat_abcdefghijklmnopqrstuvwxyz"));
    }

    #[test]
    fn turn_trace_redacts_error_fields() {
        let mut trace = TurnTrace::start("hello", 0);
        trace.record_error("request failed with password=hunter2");

        assert_eq!(
            trace.errors,
            vec!["request failed with password=[REDACTED]"]
        );
    }

    #[test]
    fn new_turn_traces_use_current_schema_version() {
        let trace = TurnTrace::start("hello", 0);

        assert_eq!(trace.schema_version, TURN_TRACE_SCHEMA_VERSION);
    }

    #[test]
    fn missing_turn_trace_schema_version_defaults_to_legacy() {
        let mut trace = serde_json::from_str::<TurnTrace>(
            r#"{
              "started_at_unix_ms": 1,
              "duration_ms": null,
              "initial_message_count": 0,
              "final_message_count": null,
              "user_input": {
                "chars": 5,
                "sha256": "abc",
                "preview": "hello",
                "truncated": false
              },
              "api_calls": [],
              "permissions": [],
              "tool_calls": [],
              "compactions": [],
              "errors": []
            }"#,
        )
        .expect("legacy trace should load");

        trace.normalize_loaded();

        assert_eq!(trace.schema_version, LEGACY_TURN_TRACE_SCHEMA_VERSION);
    }
}

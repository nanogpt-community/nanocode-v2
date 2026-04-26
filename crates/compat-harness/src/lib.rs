use std::fs;
use std::path::{Path, PathBuf};

use commands::{CommandManifestEntry, CommandRegistry, CommandSource};
use runtime::{BootstrapPhase, BootstrapPlan, TurnTrace};
use serde::{Deserialize, Serialize};
use tools::{ToolManifestEntry, ToolRegistry, ToolSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamPaths {
    repo_root: PathBuf,
}

impl UpstreamPaths {
    #[must_use]
    pub fn from_repo_root(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: repo_root.into(),
        }
    }

    #[must_use]
    pub fn from_workspace_dir(workspace_dir: impl AsRef<Path>) -> Self {
        let workspace_dir = workspace_dir
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| workspace_dir.as_ref().to_path_buf());
        let primary_repo_root = workspace_dir
            .parent()
            .map_or_else(|| PathBuf::from(".."), Path::to_path_buf);
        let repo_root = resolve_upstream_repo_root(&primary_repo_root);
        Self { repo_root }
    }

    #[must_use]
    pub fn commands_path(&self) -> PathBuf {
        self.repo_root.join("src/commands.ts")
    }

    #[must_use]
    pub fn tools_path(&self) -> PathBuf {
        self.repo_root.join("src/tools.ts")
    }

    #[must_use]
    pub fn cli_path(&self) -> PathBuf {
        self.repo_root.join("src/entrypoints/cli.tsx")
    }

    #[must_use]
    pub fn commands_snapshot_path(&self) -> PathBuf {
        self.repo_root
            .join("src/reference_data/commands_snapshot.json")
    }

    #[must_use]
    pub fn tools_snapshot_path(&self) -> PathBuf {
        self.repo_root
            .join("src/reference_data/tools_snapshot.json")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedManifest {
    pub commands: CommandRegistry,
    pub tools: ToolRegistry,
    pub bootstrap: BootstrapPlan,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub required_tools: Vec<String>,
    #[serde(default)]
    pub required_tool_sequence: Vec<String>,
    #[serde(default)]
    pub forbidden_tools: Vec<String>,
    #[serde(default)]
    pub required_permission_outcomes: Vec<EvalPermissionExpectation>,
    pub max_iterations: Option<usize>,
    pub max_tool_calls: Option<usize>,
    pub max_api_calls: Option<usize>,
    #[serde(default)]
    pub require_successful_tool: bool,
    #[serde(default)]
    pub required_answer_substrings: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalPermissionExpectation {
    pub tool_name: String,
    pub outcome: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalFailureKind {
    RequiredToolMissing,
    RequiredToolSequenceMismatch,
    ForbiddenToolUsed,
    RequiredPermissionOutcomeMissing,
    MaxIterationsExceeded,
    MaxToolCallsExceeded,
    MaxApiCallsExceeded,
    NoSuccessfulToolCall,
    MissingAnswerSubstring,
    SetupError,
    RuntimeError,
}

impl EvalFailureKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequiredToolMissing => "required_tool_missing",
            Self::RequiredToolSequenceMismatch => "required_tool_sequence_mismatch",
            Self::ForbiddenToolUsed => "forbidden_tool_used",
            Self::RequiredPermissionOutcomeMissing => "required_permission_outcome_missing",
            Self::MaxIterationsExceeded => "max_iterations_exceeded",
            Self::MaxToolCallsExceeded => "max_tool_calls_exceeded",
            Self::MaxApiCallsExceeded => "max_api_calls_exceeded",
            Self::NoSuccessfulToolCall => "no_successful_tool_call",
            Self::MissingAnswerSubstring => "missing_answer_substring",
            Self::SetupError => "setup_error",
            Self::RuntimeError => "runtime_error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCaseResult {
    pub id: String,
    pub passed: bool,
    pub failures: Vec<String>,
    #[serde(default)]
    pub failure_categories: Vec<EvalFailureKind>,
    pub iterations: usize,
    pub tool_calls: usize,
    pub api_calls: usize,
    pub duration_ms: Option<u128>,
}

fn resolve_upstream_repo_root(primary_repo_root: &Path) -> PathBuf {
    let candidates = upstream_repo_candidates(primary_repo_root);
    candidates
        .into_iter()
        .find(|candidate| candidate.join("src/commands.ts").is_file())
        .unwrap_or_else(|| primary_repo_root.to_path_buf())
}

fn upstream_repo_candidates(primary_repo_root: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![primary_repo_root.to_path_buf()];

    if let Some(explicit) = std::env::var_os("CLAUDE_CODE_UPSTREAM") {
        candidates.push(PathBuf::from(explicit));
    }

    for ancestor in primary_repo_root.ancestors().take(4) {
        candidates.push(ancestor.join("claude-code"));
        candidates.push(ancestor.join("clawd-code"));
    }

    candidates.push(
        primary_repo_root
            .join("reference-source")
            .join("claude-code"),
    );
    candidates.push(primary_repo_root.join("vendor").join("claude-code"));

    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|seen: &PathBuf| seen == &candidate) {
            deduped.push(candidate);
        }
    }
    deduped
}

pub fn extract_manifest(paths: &UpstreamPaths) -> std::io::Result<ExtractedManifest> {
    let commands = if paths.commands_path().exists() {
        extract_commands(&fs::read_to_string(paths.commands_path())?)
    } else {
        load_commands_snapshot(&paths.commands_snapshot_path())?
    };
    let tools = if paths.tools_path().exists() {
        extract_tools(&fs::read_to_string(paths.tools_path())?)
    } else {
        load_tools_snapshot(&paths.tools_snapshot_path())?
    };
    let bootstrap = if paths.cli_path().exists() {
        extract_bootstrap_plan(&fs::read_to_string(paths.cli_path())?)
    } else {
        BootstrapPlan::pebble_default()
    };

    Ok(ExtractedManifest {
        commands,
        tools,
        bootstrap,
    })
}

#[must_use]
#[allow(clippy::too_many_lines)]
pub fn evaluate_trace(case: &EvalCase, trace: &TurnTrace, final_answer: &str) -> EvalCaseResult {
    let mut failures = Vec::new();
    let mut failure_categories = Vec::new();
    let iterations = trace
        .api_calls
        .iter()
        .map(|call| call.iteration)
        .max()
        .unwrap_or(0);
    let tool_calls = trace.tool_calls.len();
    let api_calls = trace.api_calls.len();

    for required in &case.required_tools {
        if !trace
            .tool_calls
            .iter()
            .any(|call| call.tool_name == *required)
        {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::RequiredToolMissing,
                format!("required tool `{required}` was not used"),
            );
        }
    }

    if !case.required_tool_sequence.is_empty() {
        let actual_tools = trace
            .tool_calls
            .iter()
            .map(|call| call.tool_name.as_str())
            .collect::<Vec<_>>();
        if !tool_sequence_matches(&case.required_tool_sequence, &actual_tools) {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::RequiredToolSequenceMismatch,
                format!(
                    "tool sequence did not include expected order `{}`; actual `{}`",
                    case.required_tool_sequence.join(" -> "),
                    actual_tools.join(" -> ")
                ),
            );
        }
    }

    for forbidden in &case.forbidden_tools {
        if trace
            .tool_calls
            .iter()
            .any(|call| call.tool_name == *forbidden)
        {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::ForbiddenToolUsed,
                format!("forbidden tool `{forbidden}` was used"),
            );
        }
    }

    for expected in &case.required_permission_outcomes {
        if !trace.permissions.iter().any(|permission| {
            permission.tool_name == expected.tool_name && permission.outcome == expected.outcome
        }) {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::RequiredPermissionOutcomeMissing,
                format!(
                    "permission outcome `{}` for tool `{}` was not recorded",
                    expected.outcome, expected.tool_name
                ),
            );
        }
    }

    if let Some(max_iterations) = case.max_iterations {
        if iterations > max_iterations {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::MaxIterationsExceeded,
                format!("iterations {iterations} exceeded maximum {max_iterations}"),
            );
        }
    }

    if let Some(max_tool_calls) = case.max_tool_calls {
        if tool_calls > max_tool_calls {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::MaxToolCallsExceeded,
                format!("tool calls {tool_calls} exceeded maximum {max_tool_calls}"),
            );
        }
    }

    if let Some(max_api_calls) = case.max_api_calls {
        if api_calls > max_api_calls {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::MaxApiCallsExceeded,
                format!("api calls {api_calls} exceeded maximum {max_api_calls}"),
            );
        }
    }

    if case.require_successful_tool
        && !trace
            .tool_calls
            .iter()
            .any(|call| call.permission_outcome == "allow" && !call.is_error)
    {
        record_eval_failure(
            &mut failures,
            &mut failure_categories,
            EvalFailureKind::NoSuccessfulToolCall,
            "no successful allowed tool call was recorded".to_string(),
        );
    }

    let normalized_answer = final_answer.to_ascii_lowercase();
    for substring in &case.required_answer_substrings {
        if !normalized_answer.contains(&substring.to_ascii_lowercase()) {
            record_eval_failure(
                &mut failures,
                &mut failure_categories,
                EvalFailureKind::MissingAnswerSubstring,
                format!("final answer did not contain required substring `{substring}`"),
            );
        }
    }

    EvalCaseResult {
        id: case.id.clone(),
        passed: failures.is_empty(),
        failures,
        failure_categories,
        iterations,
        tool_calls,
        api_calls,
        duration_ms: trace.duration_ms,
    }
}

fn tool_sequence_matches(expected: &[String], actual: &[&str]) -> bool {
    let mut expected = expected.iter();
    let Some(mut next_expected) = expected.next() else {
        return true;
    };
    for actual_tool in actual {
        if actual_tool == next_expected {
            match expected.next() {
                Some(value) => next_expected = value,
                None => return true,
            }
        }
    }
    false
}

fn record_eval_failure(
    failures: &mut Vec<String>,
    failure_categories: &mut Vec<EvalFailureKind>,
    kind: EvalFailureKind,
    message: String,
) {
    failures.push(message);
    failure_categories.push(kind);
}

#[must_use]
pub fn extract_commands(source: &str) -> CommandRegistry {
    let mut entries = Vec::new();
    let mut in_internal_block = false;

    for raw_line in source.lines() {
        let line = raw_line.trim();

        if line.starts_with("export const INTERNAL_ONLY_COMMANDS = [") {
            in_internal_block = true;
            continue;
        }

        if in_internal_block {
            if line.starts_with(']') {
                in_internal_block = false;
                continue;
            }
            if let Some(name) = first_identifier(line) {
                entries.push(CommandManifestEntry {
                    name,
                    source: CommandSource::InternalOnly,
                });
            }
            continue;
        }

        if line.starts_with("import ") {
            for imported in imported_symbols(line) {
                entries.push(CommandManifestEntry {
                    name: imported,
                    source: CommandSource::Builtin,
                });
            }
        }

        if line.contains("feature('") && line.contains("./commands/") {
            if let Some(name) = first_assignment_identifier(line) {
                entries.push(CommandManifestEntry {
                    name,
                    source: CommandSource::FeatureGated,
                });
            }
        }
    }

    dedupe_commands(entries)
}

#[must_use]
pub fn extract_tools(source: &str) -> ToolRegistry {
    let mut entries = Vec::new();

    for raw_line in source.lines() {
        let line = raw_line.trim();
        if line.starts_with("import ") && line.contains("./tools/") {
            for imported in imported_symbols(line) {
                if imported.ends_with("Tool") {
                    entries.push(ToolManifestEntry {
                        name: imported,
                        source: ToolSource::Base,
                    });
                }
            }
        }

        if line.contains("feature('") && line.contains("Tool") {
            if let Some(name) = first_assignment_identifier(line) {
                if name.ends_with("Tool") || name.ends_with("Tools") {
                    entries.push(ToolManifestEntry {
                        name,
                        source: ToolSource::Conditional,
                    });
                }
            }
        }
    }

    dedupe_tools(entries)
}

#[must_use]
pub fn extract_bootstrap_plan(source: &str) -> BootstrapPlan {
    let mut phases = vec![BootstrapPhase::CliEntry];

    if source.contains("--version") {
        phases.push(BootstrapPhase::FastPathVersion);
    }
    if source.contains("startupProfiler") {
        phases.push(BootstrapPhase::StartupProfiler);
    }
    if source.contains("--dump-system-prompt") {
        phases.push(BootstrapPhase::SystemPromptFastPath);
    }
    if source.contains("--claude-in-chrome-mcp") {
        phases.push(BootstrapPhase::ChromeMcpFastPath);
    }
    if source.contains("--daemon-worker") {
        phases.push(BootstrapPhase::DaemonWorkerFastPath);
    }
    if source.contains("remote-control") {
        phases.push(BootstrapPhase::BridgeFastPath);
    }
    if source.contains("args[0] === 'daemon'") {
        phases.push(BootstrapPhase::DaemonFastPath);
    }
    if source.contains("args[0] === 'ps'") || source.contains("args.includes('--bg')") {
        phases.push(BootstrapPhase::BackgroundSessionFastPath);
    }
    if source.contains("args[0] === 'new' || args[0] === 'list' || args[0] === 'reply'") {
        phases.push(BootstrapPhase::TemplateFastPath);
    }
    if source.contains("environment-runner") {
        phases.push(BootstrapPhase::EnvironmentRunnerFastPath);
    }
    phases.push(BootstrapPhase::MainRuntime);

    BootstrapPlan::from_phases(phases)
}

fn imported_symbols(line: &str) -> Vec<String> {
    let Some(after_import) = line.strip_prefix("import ") else {
        return Vec::new();
    };

    let before_from = after_import
        .split(" from ")
        .next()
        .unwrap_or_default()
        .trim();
    if before_from.starts_with('{') {
        return before_from
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
            .filter_map(|part| {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    return None;
                }
                Some(trimmed.split_whitespace().next()?.to_string())
            })
            .collect();
    }

    let first = before_from.split(',').next().unwrap_or_default().trim();
    if first.is_empty() {
        Vec::new()
    } else {
        vec![first.to_string()]
    }
}

fn first_assignment_identifier(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let candidate = trimmed.split('=').next()?.trim();
    first_identifier(candidate)
}

fn first_identifier(line: &str) -> Option<String> {
    let mut out = String::new();
    for ch in line.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else if !out.is_empty() {
            break;
        }
    }
    (!out.is_empty()).then_some(out)
}

fn dedupe_commands(entries: Vec<CommandManifestEntry>) -> CommandRegistry {
    let mut deduped = Vec::new();
    for entry in entries {
        let exists = deduped.iter().any(|seen: &CommandManifestEntry| {
            seen.name == entry.name && seen.source == entry.source
        });
        if !exists {
            deduped.push(entry);
        }
    }
    CommandRegistry::new(deduped)
}

fn dedupe_tools(entries: Vec<ToolManifestEntry>) -> ToolRegistry {
    let mut deduped = Vec::new();
    for entry in entries {
        let exists = deduped
            .iter()
            .any(|seen: &ToolManifestEntry| seen.name == entry.name && seen.source == entry.source);
        if !exists {
            deduped.push(entry);
        }
    }
    ToolRegistry::new(deduped)
}

fn load_commands_snapshot(path: &Path) -> std::io::Result<CommandRegistry> {
    let entries = serde_json::from_str::<Vec<SnapshotEntry>>(&fs::read_to_string(path)?)
        .map_err(json_error)?;
    Ok(CommandRegistry::new(
        entries
            .into_iter()
            .map(|entry| CommandManifestEntry {
                name: entry.name,
                source: CommandSource::Builtin,
            })
            .collect(),
    ))
}

fn load_tools_snapshot(path: &Path) -> std::io::Result<ToolRegistry> {
    let entries = serde_json::from_str::<Vec<SnapshotEntry>>(&fs::read_to_string(path)?)
        .map_err(json_error)?;
    Ok(ToolRegistry::new(
        entries
            .into_iter()
            .map(|entry| ToolManifestEntry {
                name: entry.name,
                source: ToolSource::Base,
            })
            .collect(),
    ))
}

fn json_error(error: serde_json::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error)
}

#[derive(Debug, Deserialize)]
struct SnapshotEntry {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_paths() -> UpstreamPaths {
        let workspace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        UpstreamPaths::from_workspace_dir(workspace_dir)
    }

    fn has_upstream_fixture(paths: &UpstreamPaths) -> bool {
        paths.commands_path().is_file()
            && paths.tools_path().is_file()
            && paths.cli_path().is_file()
    }

    #[test]
    fn extracts_non_empty_manifests_from_upstream_repo() {
        let paths = fixture_paths();
        if !has_upstream_fixture(&paths) {
            return;
        }
        let manifest = extract_manifest(&paths).expect("manifest should load");
        assert!(!manifest.commands.entries().is_empty());
        assert!(!manifest.tools.entries().is_empty());
        assert!(!manifest.bootstrap.phases().is_empty());
    }

    #[test]
    fn detects_known_upstream_command_symbols() {
        let paths = fixture_paths();
        if !has_upstream_fixture(&paths) {
            return;
        }
        let commands = extract_manifest(&paths)
            .expect("manifest should load")
            .commands;
        let names: Vec<_> = commands
            .entries()
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(names.contains(&"add-dir"));
        assert!(names.contains(&"review"));
    }

    #[test]
    fn detects_known_upstream_tool_symbols() {
        let paths = fixture_paths();
        if !has_upstream_fixture(&paths) {
            return;
        }
        let tools = extract_manifest(&paths)
            .expect("manifest should load")
            .tools;
        let names: Vec<_> = tools
            .entries()
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(names.contains(&"AgentTool"));
        assert!(names.contains(&"BashTool"));
    }

    #[test]
    fn evaluates_trace_against_basic_case_constraints() {
        let case = EvalCase {
            id: "read-task".to_string(),
            prompt: "read the file".to_string(),
            required_tools: vec!["read_file".to_string()],
            required_tool_sequence: Vec::new(),
            forbidden_tools: vec!["bash".to_string()],
            required_permission_outcomes: Vec::new(),
            max_iterations: Some(2),
            max_tool_calls: Some(1),
            max_api_calls: Some(2),
            require_successful_tool: true,
            required_answer_substrings: vec!["done".to_string()],
        };
        let mut trace = TurnTrace::start("read the file", 0);
        trace.api_calls.push(runtime::ApiCallTrace {
            iteration: 1,
            request_message_count: 1,
            request_estimated_tokens: 10,
            duration_ms: 5,
            result_event_count: Some(2),
            usage: None,
            error: None,
        });
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-1".to_string(),
            tool_name: "read_file".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"path":"README.md"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("contents"),
            duration_ms: 2,
            permission_outcome: "allow".to_string(),
            is_error: false,
        });
        trace.api_calls.push(runtime::ApiCallTrace {
            iteration: 2,
            request_message_count: 3,
            request_estimated_tokens: 20,
            duration_ms: 5,
            result_event_count: Some(1),
            usage: None,
            error: None,
        });
        trace.finish(4);

        let result = evaluate_trace(&case, &trace, "done");

        assert!(result.passed, "{:?}", result.failures);
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tool_calls, 1);
        assert!(result.failure_categories.is_empty());
    }

    #[test]
    fn evaluates_tool_sequence_and_permission_outcomes() {
        let case = EvalCase {
            id: "ordered-tools".to_string(),
            prompt: "read then write".to_string(),
            required_tool_sequence: vec!["read_file".to_string(), "write_file".to_string()],
            required_permission_outcomes: vec![EvalPermissionExpectation {
                tool_name: "write_file".to_string(),
                outcome: "allow".to_string(),
            }],
            ..EvalCase::default()
        };
        let mut trace = TurnTrace::start("read then write", 0);
        trace.permissions.push(runtime::PermissionTrace {
            iteration: 1,
            tool_use_id: "tool-2".to_string(),
            tool_name: "write_file".to_string(),
            outcome: "allow".to_string(),
            reason: None,
        });
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-1".to_string(),
            tool_name: "read_file".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"path":"README.md"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("contents"),
            duration_ms: 2,
            permission_outcome: "allow".to_string(),
            is_error: false,
        });
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-2".to_string(),
            tool_name: "write_file".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"path":"out.txt"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("ok"),
            duration_ms: 3,
            permission_outcome: "allow".to_string(),
            is_error: false,
        });

        let result = evaluate_trace(&case, &trace, "done");

        assert!(result.passed, "{:?}", result.failures);
    }

    #[test]
    fn categorizes_sequence_and_permission_failures() {
        let case = EvalCase {
            id: "ordered-tools".to_string(),
            prompt: "read then write".to_string(),
            required_tool_sequence: vec!["read_file".to_string(), "write_file".to_string()],
            required_permission_outcomes: vec![EvalPermissionExpectation {
                tool_name: "write_file".to_string(),
                outcome: "allow".to_string(),
            }],
            ..EvalCase::default()
        };
        let mut trace = TurnTrace::start("read then write", 0);
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-1".to_string(),
            tool_name: "write_file".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"path":"out.txt"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("denied"),
            duration_ms: 3,
            permission_outcome: "deny".to_string(),
            is_error: true,
        });
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-2".to_string(),
            tool_name: "read_file".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"path":"README.md"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("contents"),
            duration_ms: 2,
            permission_outcome: "allow".to_string(),
            is_error: false,
        });

        let result = evaluate_trace(&case, &trace, "done");

        assert!(!result.passed);
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::RequiredToolSequenceMismatch));
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::RequiredPermissionOutcomeMissing));
    }

    #[test]
    fn categorizes_eval_failures() {
        let case = EvalCase {
            id: "bad".to_string(),
            prompt: "avoid bash".to_string(),
            required_tools: vec!["read_file".to_string()],
            forbidden_tools: vec!["bash".to_string()],
            max_api_calls: Some(0),
            required_answer_substrings: vec!["done".to_string()],
            ..EvalCase::default()
        };
        let mut trace = TurnTrace::start("avoid bash", 0);
        trace.api_calls.push(runtime::ApiCallTrace {
            iteration: 1,
            request_message_count: 1,
            request_estimated_tokens: 10,
            duration_ms: 5,
            result_event_count: Some(2),
            usage: None,
            error: None,
        });
        trace.tool_calls.push(runtime::ToolCallTrace {
            iteration: 1,
            tool_use_id: "tool-1".to_string(),
            tool_name: "bash".to_string(),
            input: runtime::TracePayloadSummary::from_text(r#"{"cmd":"pwd"}"#),
            effective_input: None,
            output: runtime::TracePayloadSummary::from_text("/tmp"),
            duration_ms: 2,
            permission_outcome: "allow".to_string(),
            is_error: false,
        });

        let result = evaluate_trace(&case, &trace, "not it");

        assert!(!result.passed);
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::RequiredToolMissing));
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::ForbiddenToolUsed));
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::MaxApiCallsExceeded));
        assert!(result
            .failure_categories
            .contains(&EvalFailureKind::MissingAnswerSubstring));
    }
}

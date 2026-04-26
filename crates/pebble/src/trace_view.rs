use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use compat_harness::{EvalCase, EvalPermissionExpectation};
use runtime::TurnTrace;
use serde_json::Value as JsonValue;

use crate::report::{
    list_or_none, report_label, report_section, report_title, sequence_or_none, shell_quote,
    short_sha, truncate_for_summary,
};

pub(crate) fn load_turn_trace(path: &Path) -> Result<TurnTrace, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let mut trace = serde_json::from_str::<TurnTrace>(&raw)?;
    migrate_turn_trace(&mut trace);
    Ok(trace)
}

fn migrate_turn_trace(trace: &mut TurnTrace) {
    trace.normalize_loaded();
}

pub(crate) fn trace_json_report(trace_path: &Path, trace: &TurnTrace) -> JsonValue {
    let case_name = suggested_eval_case_name(trace_path, trace);
    let case = eval_case_from_trace(trace, &case_name, trace_path);
    serde_json::json!({
        "kind": "trace",
        "trace_path": trace_path.display().to_string(),
        "trace": trace,
        "suggested_eval": {
            "command": format!(
                "pebble eval capture {} --suite evals/regressions.json --name {}",
                shell_quote(&trace_path.display().to_string()),
                shell_quote(&case_name)
            ),
            "case": case,
        },
    })
}

pub(crate) fn replay_json_report(trace_path: &Path, trace: &TurnTrace) -> JsonValue {
    serde_json::json!({
        "kind": "replay",
        "trace_path": trace_path.display().to_string(),
        "schema_version": trace.schema_version,
        "started_at_unix_ms": trace.started_at_unix_ms,
        "duration_ms": trace.duration_ms,
        "timeline": replay_timeline_json(trace),
    })
}

pub(crate) fn replay_timeline_json(trace: &TurnTrace) -> Vec<JsonValue> {
    let mut steps = Vec::new();
    steps.push(serde_json::json!({
        "kind": "user",
        "summary": {
            "chars": trace.user_input.chars,
            "sha256": short_sha(&trace.user_input.sha256),
        },
        "preview": &trace.user_input.preview,
    }));

    let max_iteration = trace
        .api_calls
        .iter()
        .map(|call| call.iteration)
        .chain(
            trace
                .permissions
                .iter()
                .map(|permission| permission.iteration),
        )
        .chain(trace.tool_calls.iter().map(|call| call.iteration))
        .max()
        .unwrap_or(0);

    for iteration in 0..=max_iteration {
        for api_call in trace
            .api_calls
            .iter()
            .filter(|call| call.iteration == iteration)
        {
            steps.push(serde_json::json!({
                "kind": "api",
                "iteration": api_call.iteration,
                "request_message_count": api_call.request_message_count,
                "request_estimated_tokens": api_call.request_estimated_tokens,
                "duration_ms": api_call.duration_ms,
                "result_event_count": api_call.result_event_count,
                "usage": &api_call.usage,
                "error": &api_call.error,
            }));
        }
        for permission in trace
            .permissions
            .iter()
            .filter(|permission| permission.iteration == iteration)
        {
            steps.push(serde_json::json!({
                "kind": "permission",
                "permission": permission,
            }));
        }
        for tool_call in trace
            .tool_calls
            .iter()
            .filter(|call| call.iteration == iteration)
        {
            steps.push(serde_json::json!({
                "kind": "tool",
                "tool_call": tool_call,
            }));
            steps.push(serde_json::json!({
                "kind": "tool_result",
                "tool_use_id": &tool_call.tool_use_id,
                "tool_name": &tool_call.tool_name,
                "output": &tool_call.output,
            }));
        }
    }

    for compaction in &trace.compactions {
        steps.push(serde_json::json!({
            "kind": "compaction",
            "compaction": compaction,
        }));
    }
    for error in &trace.errors {
        steps.push(serde_json::json!({
            "kind": "error",
            "error": error,
        }));
    }
    steps
}

#[allow(clippy::too_many_lines)]
pub(crate) fn render_replay_report(trace_path: &Path, trace: &TurnTrace) -> String {
    let mut report = String::new();
    let _ = writeln!(report, "{}", report_title("Pebble Replay"));
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("file"),
        trace_path.display()
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("schema_version"),
        trace.schema_version
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("started_at_unix_ms"),
        trace.started_at_unix_ms
    );
    if let Some(duration_ms) = trace.duration_ms {
        let _ = writeln!(report, "  {} {}ms", report_label("duration"), duration_ms);
    }
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Timeline"));

    let mut step = 1usize;
    append_replay_step(
        &mut report,
        &mut step,
        "user",
        format!(
            "input chars={} sha256={}",
            trace.user_input.chars,
            short_sha(&trace.user_input.sha256)
        ),
        Some(&trace.user_input.preview),
    );

    let max_iteration = trace
        .api_calls
        .iter()
        .map(|call| call.iteration)
        .chain(
            trace
                .permissions
                .iter()
                .map(|permission| permission.iteration),
        )
        .chain(trace.tool_calls.iter().map(|call| call.iteration))
        .max()
        .unwrap_or(0);

    for iteration in 0..=max_iteration {
        for api_call in trace
            .api_calls
            .iter()
            .filter(|call| call.iteration == iteration)
        {
            let usage = api_call.usage.as_ref().map_or_else(
                || "usage=unknown".to_string(),
                |usage| format!("usage={}", usage.total_tokens()),
            );
            append_replay_step(
                &mut report,
                &mut step,
                "api",
                format!(
                    "iter={} messages={} est_tokens={} duration={}ms events={} {}{}",
                    api_call.iteration,
                    api_call.request_message_count,
                    api_call.request_estimated_tokens,
                    api_call.duration_ms,
                    api_call
                        .result_event_count
                        .map_or_else(|| "unknown".to_string(), |count| count.to_string()),
                    usage,
                    api_call
                        .error
                        .as_ref()
                        .map_or_else(String::new, |error| format!(" error={error}"))
                ),
                None,
            );
        }

        for permission in trace
            .permissions
            .iter()
            .filter(|permission| permission.iteration == iteration)
        {
            append_replay_step(
                &mut report,
                &mut step,
                "permission",
                format!(
                    "iter={} tool={} id={} outcome={}{}",
                    permission.iteration,
                    permission.tool_name,
                    permission.tool_use_id,
                    permission.outcome,
                    permission
                        .reason
                        .as_ref()
                        .map_or_else(String::new, |reason| format!(" reason={reason}"))
                ),
                None,
            );
        }

        for tool_call in trace
            .tool_calls
            .iter()
            .filter(|call| call.iteration == iteration)
        {
            append_replay_step(
                &mut report,
                &mut step,
                "tool",
                format!(
                    "iter={} {} id={} duration={}ms permission={} status={}",
                    tool_call.iteration,
                    tool_call.tool_name,
                    tool_call.tool_use_id,
                    tool_call.duration_ms,
                    tool_call.permission_outcome,
                    if tool_call.is_error { "error" } else { "ok" }
                ),
                Some(&tool_call.input.preview),
            );
            append_replay_step(
                &mut report,
                &mut step,
                "tool_result",
                format!(
                    "{} id={} output_chars={} sha256={}",
                    tool_call.tool_name,
                    tool_call.tool_use_id,
                    tool_call.output.chars,
                    short_sha(&tool_call.output.sha256)
                ),
                Some(&tool_call.output.preview),
            );
        }
    }

    for compaction in &trace.compactions {
        append_replay_step(
            &mut report,
            &mut step,
            "compaction",
            format!(
                "trigger={} removed_messages={} pruned_tool_results={} estimated_tokens_after={}",
                compaction.trigger,
                compaction.removed_message_count,
                compaction.pruned_tool_result_count,
                compaction.estimated_tokens_after
            ),
            None,
        );
    }

    for error in &trace.errors {
        append_replay_step(&mut report, &mut step, "error", error.clone(), None);
    }

    report.trim_end().to_string()
}

fn append_replay_step(
    report: &mut String,
    step: &mut usize,
    kind: &str,
    summary: impl std::fmt::Display,
    preview: Option<&str>,
) {
    let _ = writeln!(report, "    {:>2}. {:<12} {}", *step, kind, summary);
    *step += 1;
    if let Some(preview) = preview {
        let preview = truncate_for_summary(preview, 140);
        if !preview.is_empty() {
            let _ = writeln!(report, "        {preview}");
        }
    }
}

pub(crate) fn render_trace_report(trace_path: &Path, trace: &TurnTrace) -> String {
    let mut report = String::new();
    let duration = trace
        .duration_ms
        .map_or_else(|| "unknown".to_string(), |duration| format!("{duration}ms"));
    let final_messages = trace
        .final_message_count
        .map_or_else(|| "unknown".to_string(), |count| count.to_string());
    let total_api_ms: u128 = trace.api_calls.iter().map(|call| call.duration_ms).sum();
    let total_tool_ms: u128 = trace.tool_calls.iter().map(|call| call.duration_ms).sum();
    let tool_errors = trace.tool_calls.iter().filter(|call| call.is_error).count();
    let denied_permissions = trace
        .permissions
        .iter()
        .filter(|permission| !permission.outcome.eq_ignore_ascii_case("allow"))
        .count();
    let peak_estimated_tokens = trace
        .api_calls
        .iter()
        .map(|call| call.request_estimated_tokens)
        .max()
        .unwrap_or(0);

    let _ = writeln!(report, "{}", report_title("Pebble Trace"));
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("file"),
        trace_path.display()
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("schema_version"),
        trace.schema_version
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("started_at_unix_ms"),
        trace.started_at_unix_ms
    );
    let _ = writeln!(report, "  {} {}", report_label("duration"), duration);
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("messages"),
        trace.initial_message_count,
        final_messages
    );
    let _ = writeln!(
        report,
        "  {} api={} tool={} permission={} compaction={} errors={}",
        report_label("events"),
        trace.api_calls.len(),
        trace.tool_calls.len(),
        trace.permissions.len(),
        trace.compactions.len(),
        trace.errors.len()
    );
    let _ = writeln!(
        report,
        "  {} api={}ms tool={}ms tool_errors={} denied_permissions={} peak_estimated_tokens={}",
        report_label("totals"),
        total_api_ms,
        total_tool_ms,
        tool_errors,
        denied_permissions,
        peak_estimated_tokens
    );
    let _ = writeln!(
        report,
        "  {} chars={} sha256={}{}",
        report_label("user_input"),
        trace.user_input.chars,
        short_sha(&trace.user_input.sha256),
        if trace.user_input.truncated {
            " truncated"
        } else {
            ""
        }
    );
    let input_preview = truncate_for_summary(&trace.user_input.preview, 160);
    if !input_preview.is_empty() {
        let _ = writeln!(report, "    {input_preview}");
    }

    append_trace_suggested_eval(&mut report, trace_path, trace);
    append_trace_api_calls(&mut report, trace);
    append_trace_tool_calls(&mut report, trace);
    append_trace_permissions(&mut report, trace);
    append_trace_compactions(&mut report, trace);
    append_trace_errors(&mut report, trace);

    report.trim_end().to_string()
}

fn append_trace_suggested_eval(report: &mut String, trace_path: &Path, trace: &TurnTrace) {
    let case_name = suggested_eval_case_name(trace_path, trace);
    let case = eval_case_from_trace(trace, &case_name, trace_path);
    let permissions = case
        .required_permission_outcomes
        .iter()
        .map(|permission| format!("{}={}", permission.tool_name, permission.outcome))
        .collect::<Vec<_>>();
    let command = format!(
        "pebble eval capture {} --suite evals/regressions.json --name {}",
        shell_quote(&trace_path.display().to_string()),
        shell_quote(&case_name)
    );

    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Suggested Eval"));
    let _ = writeln!(report, "    {} {}", report_label("command"), command);
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("tools"),
        sequence_or_none(&case.required_tool_sequence)
    );
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("perms"),
        list_or_none(&permissions)
    );
    let _ = writeln!(
        report,
        "    {} iterations={} tool_calls={} api_calls={}",
        report_label("limits"),
        case.max_iterations
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        case.max_tool_calls
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        case.max_api_calls
            .map_or_else(|| "none".to_string(), |value| value.to_string())
    );
    if trace.user_input.truncated {
        let _ = writeln!(
            report,
            "    {} prompt preview was truncated before capture",
            report_label("note")
        );
    }
}

fn suggested_eval_case_name(trace_path: &Path, trace: &TurnTrace) -> String {
    if trace.tool_calls.is_empty() {
        return slugify_eval_case_id(
            trace_path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("captured-trace"),
        );
    }
    slugify_eval_case_id(
        &trace
            .tool_calls
            .iter()
            .map(|call| call.tool_name.as_str())
            .collect::<Vec<_>>()
            .join("-then-"),
    )
}

fn append_trace_api_calls(report: &mut String, trace: &TurnTrace) {
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("API Calls"));
    if trace.api_calls.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for (index, call) in trace.api_calls.iter().enumerate() {
        let usage = call.usage.as_ref().map_or_else(
            || "usage=unknown".to_string(),
            |usage| format!("usage={}", usage.total_tokens()),
        );
        let result = call.result_event_count.map_or_else(
            || "events=unknown".to_string(),
            |count| format!("events={count}"),
        );
        let error = call
            .error
            .as_ref()
            .map_or_else(String::new, |error| format!(" error={error}"));
        let _ = writeln!(
            report,
            "    #{:<2} iter={} messages={} est_tokens={} duration={}ms {} {}{}",
            index + 1,
            call.iteration,
            call.request_message_count,
            call.request_estimated_tokens,
            call.duration_ms,
            result,
            usage,
            error
        );
    }
}

fn append_trace_tool_calls(report: &mut String, trace: &TurnTrace) {
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Tool Calls"));
    if trace.tool_calls.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for (index, call) in trace.tool_calls.iter().enumerate() {
        let state = if call.is_error { "error" } else { "ok" };
        let input_preview = truncate_for_summary(&call.input.preview, 96);
        let output_preview = truncate_for_summary(&call.output.preview, 96);
        let _ = writeln!(
            report,
            "    #{:<2} iter={} {} id={} duration={}ms permission={} status={}",
            index + 1,
            call.iteration,
            call.tool_name,
            call.tool_use_id,
            call.duration_ms,
            call.permission_outcome,
            state
        );
        let _ = writeln!(
            report,
            "        input chars={} sha256={} {}",
            call.input.chars,
            short_sha(&call.input.sha256),
            input_preview
        );
        let _ = writeln!(
            report,
            "        output chars={} sha256={} {}",
            call.output.chars,
            short_sha(&call.output.sha256),
            output_preview
        );
    }
}

fn append_trace_permissions(report: &mut String, trace: &TurnTrace) {
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Permissions"));
    if trace.permissions.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for permission in &trace.permissions {
        let reason = permission
            .reason
            .as_ref()
            .map_or_else(String::new, |reason| format!(" reason={reason}"));
        let _ = writeln!(
            report,
            "    iter={} {} id={} outcome={}{}",
            permission.iteration,
            permission.tool_name,
            permission.tool_use_id,
            permission.outcome,
            reason
        );
    }
}

fn append_trace_compactions(report: &mut String, trace: &TurnTrace) {
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Compactions"));
    if trace.compactions.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for compaction in &trace.compactions {
        let _ = writeln!(
            report,
            "    trigger={} removed_messages={} pruned_tool_results={} estimated_tokens_after={}",
            compaction.trigger,
            compaction.removed_message_count,
            compaction.pruned_tool_result_count,
            compaction.estimated_tokens_after
        );
    }
}

fn append_trace_errors(report: &mut String, trace: &TurnTrace) {
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Errors"));
    if trace.errors.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for error in &trace.errors {
        let _ = writeln!(report, "    - {error}");
    }
}

pub(crate) fn slugify_eval_case_id(value: &str) -> String {
    let mut output = String::new();
    let mut pending_dash = false;
    for ch in value.trim().chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !output.is_empty() {
                output.push('-');
            }
            output.push(ch);
            pending_dash = false;
        } else if !output.is_empty() {
            pending_dash = true;
        }
    }
    if output.is_empty() {
        "captured-trace".to_string()
    } else {
        output
    }
}

pub(crate) fn eval_case_from_trace(trace: &TurnTrace, id: &str, trace_path: &Path) -> EvalCase {
    let prompt = trace_prompt_preview(trace, trace_path);
    let required_tool_sequence = trace
        .tool_calls
        .iter()
        .map(|call| call.tool_name.clone())
        .collect::<Vec<_>>();
    let required_tools = unique_preserving_order(required_tool_sequence.iter().cloned());
    let required_permission_outcomes = unique_permission_expectations(trace);
    let max_iterations = trace.api_calls.iter().map(|call| call.iteration).max();
    let require_successful_tool = trace
        .tool_calls
        .iter()
        .any(|call| call.permission_outcome == "allow" && !call.is_error);

    EvalCase {
        id: id.to_string(),
        prompt,
        required_tools,
        required_tool_sequence,
        forbidden_tools: Vec::new(),
        required_permission_outcomes,
        max_iterations,
        max_tool_calls: Some(trace.tool_calls.len()),
        max_api_calls: Some(trace.api_calls.len()),
        require_successful_tool,
        required_answer_substrings: Vec::new(),
    }
}

fn trace_prompt_preview(trace: &TurnTrace, trace_path: &Path) -> String {
    let prompt = trace.user_input.preview.trim();
    if prompt.is_empty() {
        format!("Captured trace from {}", trace_path.display())
    } else {
        prompt.to_string()
    }
}

fn unique_preserving_order(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            output.push(value);
        }
    }
    output
}

fn unique_permission_expectations(trace: &TurnTrace) -> Vec<EvalPermissionExpectation> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for permission in &trace.permissions {
        let key = (permission.tool_name.clone(), permission.outcome.clone());
        if seen.insert(key.clone()) {
            output.push(EvalPermissionExpectation {
                tool_name: key.0,
                outcome: key.1,
            });
        }
    }
    output
}

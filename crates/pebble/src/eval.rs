use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use compat_harness::{EvalCase, EvalCaseResult};
use platform::write_atomic;
use runtime::TurnTrace;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::report::{
    list_or_none, report_label, report_section, report_title, truncate_for_summary,
};
use crate::trace_view::{
    eval_case_from_trace, load_turn_trace, render_replay_report, slugify_eval_case_id,
};

pub(crate) const EVAL_REPORT_SCHEMA_VERSION: u32 = 2;
pub(crate) const LEGACY_EVAL_REPORT_SCHEMA_VERSION: u32 = 1;
const EVAL_HISTORY_INDEX_SCHEMA_VERSION: u32 = 1;
pub(crate) const EVAL_HISTORY_INDEX_FILE: &str = "index.json";

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum EvalSuiteInput {
    Cases(Vec<EvalCase>),
    Document(EvalSuiteDocument),
}

#[derive(Debug, Clone, Deserialize)]
struct EvalSuiteDocument {
    name: Option<String>,
    cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalSuiteDocumentOutput {
    name: String,
    cases: Vec<EvalCase>,
}

#[derive(Debug, Clone)]
pub(crate) struct EvalSuite {
    pub(crate) name: String,
    pub(crate) cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalCaptureOptions {
    pub(crate) trace_path: PathBuf,
    pub(crate) suite_path: PathBuf,
    pub(crate) name: Option<String>,
    pub(crate) force: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalCaptureOutcome {
    pub(crate) case: EvalCase,
    pub(crate) replaced: bool,
    pub(crate) suite_case_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalReplayOptions {
    pub(crate) report_path: PathBuf,
    pub(crate) case_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct EvalHistoryFilter {
    pub(crate) suite: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) limit: usize,
}

impl Default for EvalHistoryFilter {
    fn default() -> Self {
        Self {
            suite: None,
            model: None,
            limit: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EvalRunReport {
    #[serde(default = "legacy_eval_report_schema_version")]
    pub(crate) schema_version: u32,
    pub(crate) run_id: String,
    pub(crate) suite: String,
    pub(crate) model: String,
    pub(crate) started_at_unix_ms: u128,
    pub(crate) duration_ms: u128,
    pub(crate) passed: usize,
    pub(crate) failed: usize,
    pub(crate) cases: Vec<EvalRunCaseReport>,
}

const fn legacy_eval_report_schema_version() -> u32 {
    LEGACY_EVAL_REPORT_SCHEMA_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EvalHistoryIndex {
    pub(crate) schema_version: u32,
    pub(crate) updated_at_unix_ms: u128,
    pub(crate) runs: Vec<EvalHistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EvalHistoryEntry {
    pub(crate) run_id: String,
    pub(crate) suite: String,
    pub(crate) model: String,
    pub(crate) started_at_unix_ms: u128,
    pub(crate) duration_ms: u128,
    pub(crate) passed: usize,
    pub(crate) failed: usize,
    pub(crate) report_file: PathBuf,
    pub(crate) schema_version: u32,
}

#[derive(Debug, Clone, Serialize)]
struct EvalHistoryRow {
    entry: EvalHistoryEntry,
    pass_rate_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EvalRunCaseReport {
    pub(crate) case: EvalCase,
    pub(crate) result: EvalCaseResult,
    pub(crate) final_answer: String,
    pub(crate) trace_file: Option<PathBuf>,
    pub(crate) session_file: Option<PathBuf>,
    pub(crate) error: Option<String>,
    pub(crate) changed_files: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct EvalReportMetrics {
    cases: usize,
    passed: usize,
    failed: usize,
    errored: usize,
    duration_ms: u128,
    tool_calls: usize,
    api_calls: usize,
    iterations: usize,
    changed_files: usize,
    peak_estimated_tokens: Option<usize>,
}

pub(crate) fn load_eval_suite(path: &Path) -> Result<EvalSuite, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let input = serde_json::from_str::<EvalSuiteInput>(&raw)?;
    let default_name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("eval-suite")
        .to_string();
    let suite = match input {
        EvalSuiteInput::Cases(cases) => EvalSuite {
            name: default_name,
            cases,
        },
        EvalSuiteInput::Document(document) => EvalSuite {
            name: document
                .name
                .filter(|name| !name.trim().is_empty())
                .unwrap_or(default_name),
            cases: document.cases,
        },
    };
    if suite.cases.is_empty() {
        return Err(format!("eval suite `{}` does not contain any cases", path.display()).into());
    }
    for case in &suite.cases {
        if case.id.trim().is_empty() {
            return Err("eval suite contains a case with an empty id".into());
        }
        if case.prompt.trim().is_empty() {
            return Err(format!("eval case `{}` has an empty prompt", case.id).into());
        }
    }
    Ok(suite)
}

pub(crate) fn evals_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = env::current_dir()?.join(".pebble").join("evals");
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub(crate) fn run_eval_capture(
    options: &EvalCaptureOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let trace = load_turn_trace(&options.trace_path)?;
    let id = eval_capture_case_id(options, &options.trace_path);
    let case = eval_case_from_trace(&trace, &id, &options.trace_path);
    let outcome = write_captured_eval_case(&options.suite_path, case, options.force)?;
    println!("{}", render_eval_capture_report(options, &trace, &outcome));
    Ok(())
}

fn eval_capture_case_id(options: &EvalCaptureOptions, trace_path: &Path) -> String {
    let raw = options.name.as_deref().unwrap_or_else(|| {
        trace_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("captured-trace")
    });
    slugify_eval_case_id(raw)
}

pub(crate) fn write_captured_eval_case(
    suite_path: &Path,
    case: EvalCase,
    force: bool,
) -> Result<EvalCaptureOutcome, Box<dyn std::error::Error>> {
    let mut suite = if suite_path.exists() {
        load_eval_suite(suite_path)?
    } else {
        EvalSuite {
            name: suite_path
                .file_stem()
                .and_then(|name| name.to_str())
                .filter(|name| !name.trim().is_empty())
                .unwrap_or("captured-regressions")
                .to_string(),
            cases: Vec::new(),
        }
    };

    let replaced = if let Some(index) = suite
        .cases
        .iter()
        .position(|existing| existing.id == case.id)
    {
        if !force {
            return Err(format!(
                "eval suite `{}` already contains case `{}`; pass --force to replace it",
                suite_path.display(),
                case.id
            )
            .into());
        }
        suite.cases[index] = case.clone();
        true
    } else {
        suite.cases.push(case.clone());
        false
    };

    if let Some(parent) = suite_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let document = EvalSuiteDocumentOutput {
        name: suite.name,
        cases: suite.cases,
    };
    write_atomic(suite_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(EvalCaptureOutcome {
        case,
        replaced,
        suite_case_count: document.cases.len(),
    })
}

pub(crate) fn render_eval_capture_report(
    options: &EvalCaptureOptions,
    trace: &TurnTrace,
    outcome: &EvalCaptureOutcome,
) -> String {
    let mut report = String::new();
    let _ = writeln!(report, "{}", report_title("Eval Capture"));
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("trace"),
        options.trace_path.display()
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("suite"),
        options.suite_path.display()
    );
    let _ = writeln!(report, "  {} {}", report_label("case"), outcome.case.id);
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("action"),
        if outcome.replaced {
            "replaced"
        } else {
            "appended"
        }
    );
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("suite_cases"),
        outcome.suite_case_count
    );
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Generated Assertions"));
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("prompt_chars"),
        outcome.case.prompt.chars().count()
    );
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("required_tools"),
        list_or_none(&outcome.case.required_tools)
    );
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("tool_sequence"),
        list_or_none(&outcome.case.required_tool_sequence)
    );
    let permissions = outcome
        .case
        .required_permission_outcomes
        .iter()
        .map(|permission| format!("{}={}", permission.tool_name, permission.outcome))
        .collect::<Vec<_>>();
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("permissions"),
        list_or_none(&permissions)
    );
    let _ = writeln!(
        report,
        "    {} iterations={} tool_calls={} api_calls={}",
        report_label("limits"),
        outcome
            .case
            .max_iterations
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        outcome
            .case
            .max_tool_calls
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        outcome
            .case
            .max_api_calls
            .map_or_else(|| "none".to_string(), |value| value.to_string())
    );
    if trace.user_input.truncated {
        let _ = writeln!(
            report,
            "    {} prompt came from a truncated trace preview",
            report_label("note")
        );
    }
    report.trim_end().to_string()
}

pub(crate) fn load_eval_report(path: &Path) -> Result<EvalRunReport, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let mut report = serde_json::from_str::<EvalRunReport>(&raw)?;
    migrate_eval_report(&mut report);
    Ok(report)
}

fn migrate_eval_report(report: &mut EvalRunReport) {
    if report.schema_version == 0 {
        report.schema_version = LEGACY_EVAL_REPORT_SCHEMA_VERSION;
    }
}

pub(crate) fn eval_replay_json_report(
    options: &EvalReplayOptions,
    eval: &EvalRunReport,
) -> JsonValue {
    let cases = eval_replay_cases(eval, options.case_id.as_deref())
        .into_iter()
        .map(|case| eval_replay_case_json(&options.report_path, case))
        .collect::<Vec<_>>();
    serde_json::json!({
        "kind": "eval_replay",
        "report_path": options.report_path.display().to_string(),
        "run_id": &eval.run_id,
        "schema_version": eval.schema_version,
        "suite": &eval.suite,
        "model": &eval.model,
        "filter": {
            "case": &options.case_id,
            "default": if options.case_id.is_some() { "case" } else { "failed_cases" },
        },
        "cases": cases,
    })
}

fn eval_replay_case_json(report_path: &Path, case: &EvalRunCaseReport) -> JsonValue {
    let trace_path = case
        .trace_file
        .as_deref()
        .map(|path| resolve_eval_trace_path(report_path, path));
    let session_path = case
        .session_file
        .as_deref()
        .map(|path| resolve_eval_trace_path(report_path, path));
    let trace = trace_path.as_ref().map(|path| match load_turn_trace(path) {
        Ok(trace) => serde_json::json!({
            "path": path.display().to_string(),
            "loaded": true,
            "timeline": crate::trace_view::replay_timeline_json(&trace),
        }),
        Err(error) => serde_json::json!({
            "path": path.display().to_string(),
            "loaded": false,
            "error": error.to_string(),
        }),
    });
    serde_json::json!({
        "id": &case.result.id,
        "status": if case.error.is_some() {
            "error"
        } else if case.result.passed {
            "pass"
        } else {
            "fail"
        },
        "case": &case.case,
        "result": &case.result,
        "failure_categories": case.result.failure_categories.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
        "final_answer": &case.final_answer,
        "artifacts": {
            "trace_file": trace_path.as_ref().map(|path| path.display().to_string()),
            "session_file": session_path.as_ref().map(|path| path.display().to_string()),
            "changed_files": case.changed_files,
        },
        "error": &case.error,
        "trace": trace,
    })
}

pub(crate) fn render_eval_replay_report(
    options: &EvalReplayOptions,
    eval: &EvalRunReport,
) -> String {
    let cases = eval_replay_cases(eval, options.case_id.as_deref());
    let mut report = String::new();
    let _ = writeln!(report, "{}", report_title("Eval Replay"));
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("report"),
        options.report_path.display()
    );
    let _ = writeln!(report, "  {} {}", report_label("run_id"), eval.run_id);
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("schema_version"),
        eval.schema_version
    );
    let _ = writeln!(report, "  {} {}", report_label("suite"), eval.suite);
    let _ = writeln!(report, "  {} {}", report_label("model"), eval.model);
    let _ = writeln!(
        report,
        "  {} {}",
        report_label("filter"),
        options
            .case_id
            .as_ref()
            .map_or("failed cases".to_string(), |case_id| format!(
                "case={case_id}"
            ))
    );
    let _ = writeln!(report, "  {} {}", report_label("cases"), cases.len());

    if cases.is_empty() {
        let _ = writeln!(report);
        let _ = writeln!(report, "  {}", report_section("Cases"));
        let _ = writeln!(report, "    none");
        return report.trim_end().to_string();
    }

    for case in cases {
        append_eval_replay_case(&mut report, &options.report_path, case);
    }

    report.trim_end().to_string()
}

fn eval_replay_cases<'a>(
    eval: &'a EvalRunReport,
    case_id: Option<&str>,
) -> Vec<&'a EvalRunCaseReport> {
    eval.cases
        .iter()
        .filter(|case| {
            case_id.map_or_else(
                || !case.result.passed || case.error.is_some(),
                |case_id| case.case.id == case_id || case.result.id == case_id,
            )
        })
        .collect()
}

fn append_eval_replay_case(report: &mut String, report_path: &Path, case: &EvalRunCaseReport) {
    let status = if case.error.is_some() {
        "error"
    } else if case.result.passed {
        "pass"
    } else {
        "fail"
    };
    let categories = case
        .result
        .failure_categories
        .iter()
        .map(|category| category.as_str().to_string())
        .collect::<Vec<_>>();
    let _ = writeln!(report);
    let _ = writeln!(report, "  {} {}", report_section("Case"), case.result.id);
    let _ = writeln!(report, "    {} {}", report_label("status"), status);
    let _ = writeln!(
        report,
        "    {} iterations={} tool_calls={} api_calls={} duration={}",
        report_label("metrics"),
        case.result.iterations,
        case.result.tool_calls,
        case.result.api_calls,
        case.result
            .duration_ms
            .map_or_else(|| "unknown".to_string(), |duration| format!("{duration}ms"))
    );
    let _ = writeln!(
        report,
        "    {} {}",
        report_label("categories"),
        list_or_none(&categories)
    );
    append_eval_replay_failures(report, case);
    append_eval_replay_artifacts(report, report_path, case);
    append_eval_replay_final_answer(report, case);
    append_eval_replay_trace(report, report_path, case);
}

fn append_eval_replay_failures(report: &mut String, case: &EvalRunCaseReport) {
    let _ = writeln!(report, "    {}", report_section("Failures"));
    if case.result.failures.is_empty() && case.error.is_none() {
        let _ = writeln!(report, "      none");
        return;
    }
    for failure in &case.result.failures {
        let _ = writeln!(report, "      - {}", truncate_for_summary(failure, 180));
    }
    if let Some(error) = &case.error {
        let _ = writeln!(
            report,
            "      - runtime error: {}",
            truncate_for_summary(error, 180)
        );
    }
}

fn append_eval_replay_artifacts(report: &mut String, report_path: &Path, case: &EvalRunCaseReport) {
    let trace = case
        .trace_file
        .as_deref()
        .map(|path| resolve_eval_trace_path(report_path, path));
    let session = case
        .session_file
        .as_deref()
        .map(|path| resolve_eval_trace_path(report_path, path));
    let _ = writeln!(report, "    {}", report_section("Artifacts"));
    let _ = writeln!(
        report,
        "      {} {}",
        report_label("trace"),
        trace
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string())
    );
    let _ = writeln!(
        report,
        "      {} {}",
        report_label("session"),
        session
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string())
    );
    let _ = writeln!(
        report,
        "      {} {}",
        report_label("changed_files"),
        case.changed_files
    );
}

fn append_eval_replay_final_answer(report: &mut String, case: &EvalRunCaseReport) {
    let _ = writeln!(report, "    {}", report_section("Final Answer"));
    let preview = truncate_for_summary(case.final_answer.trim(), 240);
    if preview.is_empty() {
        let _ = writeln!(report, "      none");
    } else {
        let _ = writeln!(report, "      {preview}");
    }
}

fn append_eval_replay_trace(report: &mut String, report_path: &Path, case: &EvalRunCaseReport) {
    let _ = writeln!(report, "    {}", report_section("Trace Replay"));
    let Some(trace_file) = case.trace_file.as_deref() else {
        let _ = writeln!(report, "      trace unavailable");
        return;
    };
    let trace_path = resolve_eval_trace_path(report_path, trace_file);
    match load_turn_trace(&trace_path) {
        Ok(trace) => {
            let replay = render_replay_report(&trace_path, &trace);
            for line in replay.lines() {
                let _ = writeln!(report, "      {line}");
            }
        }
        Err(error) => {
            let _ = writeln!(
                report,
                "      unable to load trace `{}`: {error}",
                trace_path.display()
            );
        }
    }
}

pub(crate) fn eval_history_json_report(
    filter: &EvalHistoryFilter,
    index: &EvalHistoryIndex,
) -> JsonValue {
    serde_json::json!({
        "kind": "eval_history",
        "filter": filter,
        "index_schema_version": index.schema_version,
        "updated_at_unix_ms": index.updated_at_unix_ms,
        "rows": eval_history_rows(filter, index),
    })
}

pub(crate) fn rebuild_eval_history_index(
    cwd: &Path,
) -> Result<EvalHistoryIndex, Box<dyn std::error::Error>> {
    let evals = cwd.join(".pebble").join("evals");
    let mut runs = Vec::new();
    match fs::read_dir(&evals) {
        Ok(entries) => {
            for entry in entries {
                let path = entry?.path();
                if path
                    .file_name()
                    .is_some_and(|name| name == EVAL_HISTORY_INDEX_FILE)
                    || path.extension().is_none_or(|ext| ext != "json")
                {
                    continue;
                }
                let Ok(report) = load_eval_report(&path) else {
                    continue;
                };
                runs.push(eval_history_entry_from_report(cwd, &path, &report));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(Box::new(error)),
    }
    runs.sort_by(|left, right| {
        left.started_at_unix_ms
            .cmp(&right.started_at_unix_ms)
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    Ok(EvalHistoryIndex {
        schema_version: EVAL_HISTORY_INDEX_SCHEMA_VERSION,
        updated_at_unix_ms: unix_timestamp_ms(),
        runs,
    })
}

pub(crate) fn write_eval_history_index(
    cwd: &Path,
    index: &EvalHistoryIndex,
) -> io::Result<PathBuf> {
    let path = cwd
        .join(".pebble")
        .join("evals")
        .join(EVAL_HISTORY_INDEX_FILE);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(index)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_atomic(&path, bytes)?;
    Ok(path)
}

fn eval_history_entry_from_report(
    cwd: &Path,
    report_path: &Path,
    report: &EvalRunReport,
) -> EvalHistoryEntry {
    EvalHistoryEntry {
        run_id: report.run_id.clone(),
        suite: report.suite.clone(),
        model: report.model.clone(),
        started_at_unix_ms: report.started_at_unix_ms,
        duration_ms: report.duration_ms,
        passed: report.passed,
        failed: report.failed,
        report_file: report_path
            .strip_prefix(cwd)
            .map_or_else(|_| report_path.to_path_buf(), Path::to_path_buf),
        schema_version: report.schema_version,
    }
}

pub(crate) fn render_eval_history_report(
    filter: &EvalHistoryFilter,
    index: &EvalHistoryIndex,
) -> String {
    let rows = eval_history_rows(filter, index);
    let mut report = String::new();
    let _ = writeln!(report, "{}", report_title("Eval History"));
    let _ = writeln!(
        report,
        "  {} schema_version={} index_runs={} updated_at_unix_ms={}",
        report_label("index"),
        index.schema_version,
        index.runs.len(),
        index.updated_at_unix_ms
    );
    let _ = writeln!(
        report,
        "  {} suite={} model={} limit={}",
        report_label("filter"),
        filter.suite.as_deref().unwrap_or("any"),
        filter.model.as_deref().unwrap_or("any"),
        filter.limit
    );
    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Runs"));
    if rows.is_empty() {
        let _ = writeln!(report, "    none");
        return report.trim_end().to_string();
    }
    for row in rows.iter().rev().take(filter.limit) {
        let total = row.entry.passed + row.entry.failed;
        let pass_rate = optional_percent_label(percent_value(row.entry.passed, total));
        let delta = row
            .pass_rate_delta
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:+.1}pp"));
        let _ = writeln!(
            report,
            "    {} suite={} model={} pass={} ({}/{}) delta={} duration={}ms schema={} report={}",
            row.entry.run_id,
            row.entry.suite,
            row.entry.model,
            pass_rate,
            row.entry.passed,
            total,
            delta,
            row.entry.duration_ms,
            row.entry.schema_version,
            row.entry.report_file.display()
        );
    }
    report.trim_end().to_string()
}

fn eval_history_rows(filter: &EvalHistoryFilter, index: &EvalHistoryIndex) -> Vec<EvalHistoryRow> {
    let filtered = index
        .runs
        .iter()
        .filter(|entry| {
            filter
                .suite
                .as_ref()
                .is_none_or(|suite| entry.suite == *suite)
                && filter
                    .model
                    .as_ref()
                    .is_none_or(|model| entry.model == *model)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(filtered.len());
    let mut previous_pass_rate = None;
    for entry in filtered {
        let pass_rate = percent_value(entry.passed, entry.passed + entry.failed);
        let pass_rate_delta = pass_rate
            .zip(previous_pass_rate)
            .map(|(current, previous)| current - previous);
        rows.push(EvalHistoryRow {
            entry,
            pass_rate_delta,
        });
        previous_pass_rate = pass_rate;
    }
    rows
}

pub(crate) fn eval_compare_json_report(
    old_path: &Path,
    old_report: &EvalRunReport,
    new_path: &Path,
    new_report: &EvalRunReport,
) -> JsonValue {
    serde_json::json!({
        "kind": "eval_compare",
        "old_path": old_path.display().to_string(),
        "new_path": new_path.display().to_string(),
        "old_report": old_report,
        "new_report": new_report,
        "old_metrics": eval_report_metrics(old_path, old_report),
        "new_metrics": eval_report_metrics(new_path, new_report),
        "failure_categories": {
            "old": eval_report_failure_categories(old_report),
            "new": eval_report_failure_categories(new_report),
        },
        "case_changes": eval_case_changes_json(old_report, new_report),
    })
}

fn eval_case_changes_json(old_report: &EvalRunReport, new_report: &EvalRunReport) -> JsonValue {
    let old_cases = old_report
        .cases
        .iter()
        .map(|case| (case.result.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let new_cases = new_report
        .cases
        .iter()
        .map(|case| (case.result.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut regressions = Vec::new();
    let mut fixes = Vec::new();
    let mut added = Vec::new();
    let mut removed = Vec::new();

    for (id, new_case) in &new_cases {
        match old_cases.get(id) {
            Some(old_case) => {
                let old_passed = old_case.result.passed && old_case.error.is_none();
                let new_passed = new_case.result.passed && new_case.error.is_none();
                if old_passed && !new_passed {
                    regressions.push((*id).to_string());
                } else if !old_passed && new_passed {
                    fixes.push((*id).to_string());
                }
            }
            None => added.push((*id).to_string()),
        }
    }
    for id in old_cases.keys() {
        if !new_cases.contains_key(id) {
            removed.push((*id).to_string());
        }
    }

    serde_json::json!({
        "regressions": regressions,
        "fixes": fixes,
        "added": added,
        "removed": removed,
    })
}

#[allow(clippy::too_many_lines)]
pub(crate) fn render_eval_compare_report(
    old_path: &Path,
    old_report: &EvalRunReport,
    new_path: &Path,
    new_report: &EvalRunReport,
) -> String {
    let old_metrics = eval_report_metrics(old_path, old_report);
    let new_metrics = eval_report_metrics(new_path, new_report);
    let mut report = String::new();

    let _ = writeln!(report, "{}", report_title("Eval Compare"));
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("reports"),
        old_path.display(),
        new_path.display()
    );
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("run_id"),
        old_report.run_id,
        new_report.run_id
    );
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("schema_version"),
        old_report.schema_version,
        new_report.schema_version
    );
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("suite"),
        old_report.suite,
        new_report.suite
    );
    let _ = writeln!(
        report,
        "  {} {} -> {}",
        report_label("model"),
        old_report.model,
        new_report.model
    );

    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Metrics"));
    append_eval_metric_usize(&mut report, "cases", old_metrics.cases, new_metrics.cases);
    append_eval_metric_percent(
        &mut report,
        "pass_rate",
        old_metrics.passed,
        old_metrics.cases,
        new_metrics.passed,
        new_metrics.cases,
    );
    append_eval_metric_usize(
        &mut report,
        "passed",
        old_metrics.passed,
        new_metrics.passed,
    );
    append_eval_metric_usize(
        &mut report,
        "failed",
        old_metrics.failed,
        new_metrics.failed,
    );
    append_eval_metric_usize(
        &mut report,
        "errored",
        old_metrics.errored,
        new_metrics.errored,
    );
    append_eval_metric_u128(
        &mut report,
        "duration_ms",
        old_metrics.duration_ms,
        new_metrics.duration_ms,
    );
    append_eval_metric_usize(
        &mut report,
        "tool_calls",
        old_metrics.tool_calls,
        new_metrics.tool_calls,
    );
    append_eval_metric_usize(
        &mut report,
        "api_calls",
        old_metrics.api_calls,
        new_metrics.api_calls,
    );
    append_eval_metric_usize(
        &mut report,
        "iterations",
        old_metrics.iterations,
        new_metrics.iterations,
    );
    append_eval_metric_usize(
        &mut report,
        "changed_files",
        old_metrics.changed_files,
        new_metrics.changed_files,
    );
    append_eval_metric_optional_usize(
        &mut report,
        "peak_estimated_tokens",
        old_metrics.peak_estimated_tokens,
        new_metrics.peak_estimated_tokens,
    );

    append_eval_failure_category_changes(&mut report, old_report, new_report);
    append_eval_case_changes(&mut report, old_report, new_report);

    report.trim_end().to_string()
}

fn eval_report_metrics(report_path: &Path, report: &EvalRunReport) -> EvalReportMetrics {
    EvalReportMetrics {
        cases: report.cases.len(),
        passed: report.passed,
        failed: report.failed,
        errored: report
            .cases
            .iter()
            .filter(|case| case.error.is_some())
            .count(),
        duration_ms: report.duration_ms,
        tool_calls: report.cases.iter().map(|case| case.result.tool_calls).sum(),
        api_calls: report.cases.iter().map(|case| case.result.api_calls).sum(),
        iterations: report.cases.iter().map(|case| case.result.iterations).sum(),
        changed_files: report.cases.iter().map(|case| case.changed_files).sum(),
        peak_estimated_tokens: eval_report_peak_estimated_tokens(report_path, report),
    }
}

fn eval_report_peak_estimated_tokens(report_path: &Path, report: &EvalRunReport) -> Option<usize> {
    report
        .cases
        .iter()
        .filter_map(|case| case.trace_file.as_deref())
        .filter_map(|trace_file| {
            let trace_path = resolve_eval_trace_path(report_path, trace_file);
            load_turn_trace(&trace_path).ok()
        })
        .flat_map(|trace| {
            trace
                .api_calls
                .into_iter()
                .map(|call| call.request_estimated_tokens)
        })
        .max()
}

fn resolve_eval_trace_path(report_path: &Path, trace_path: &Path) -> PathBuf {
    if trace_path.is_absolute() {
        return trace_path.to_path_buf();
    }
    report_path.parent().map_or_else(
        || trace_path.to_path_buf(),
        |parent| parent.join(trace_path),
    )
}

fn append_eval_metric_usize(report: &mut String, label: &str, old_value: usize, new_value: usize) {
    let _ = writeln!(
        report,
        "    {} {:>8} -> {:<8} {}",
        report_label(label),
        old_value,
        new_value,
        signed_i128_delta(new_value as i128 - old_value as i128)
    );
}

fn append_eval_metric_u128(report: &mut String, label: &str, old_value: u128, new_value: u128) {
    let _ = writeln!(
        report,
        "    {} {:>8} -> {:<8} {}",
        report_label(label),
        old_value,
        new_value,
        signed_u128_delta(old_value, new_value)
    );
}

fn append_eval_metric_optional_usize(
    report: &mut String,
    label: &str,
    old_value: Option<usize>,
    new_value: Option<usize>,
) {
    let delta = old_value.zip(new_value).map_or_else(
        || "n/a".to_string(),
        |(old, new)| signed_i128_delta(new as i128 - old as i128),
    );
    let _ = writeln!(
        report,
        "    {} {:>8} -> {:<8} {}",
        report_label(label),
        optional_usize_label(old_value),
        optional_usize_label(new_value),
        delta
    );
}

fn append_eval_metric_percent(
    report: &mut String,
    label: &str,
    old_numerator: usize,
    old_denominator: usize,
    new_numerator: usize,
    new_denominator: usize,
) {
    let old_percent = percent_value(old_numerator, old_denominator);
    let new_percent = percent_value(new_numerator, new_denominator);
    let delta = old_percent.zip(new_percent).map_or_else(
        || "n/a".to_string(),
        |(old, new)| format!("{:+.1}pp", new - old),
    );
    let _ = writeln!(
        report,
        "    {} {:>8} -> {:<8} {}",
        report_label(label),
        optional_percent_label(old_percent),
        optional_percent_label(new_percent),
        delta
    );
}

fn append_eval_case_changes(
    report: &mut String,
    old_report: &EvalRunReport,
    new_report: &EvalRunReport,
) {
    let old_cases = old_report
        .cases
        .iter()
        .map(|case| (case.result.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let new_cases = new_report
        .cases
        .iter()
        .map(|case| (case.result.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut regressions = Vec::new();
    let mut fixes = Vec::new();
    let mut added = Vec::new();
    let mut removed = Vec::new();

    for (id, new_case) in &new_cases {
        match old_cases.get(id) {
            Some(old_case) => {
                let old_passed = old_case.result.passed && old_case.error.is_none();
                let new_passed = new_case.result.passed && new_case.error.is_none();
                if old_passed && !new_passed {
                    regressions.push((*id, *new_case));
                } else if !old_passed && new_passed {
                    fixes.push((*id, *new_case));
                }
            }
            None => added.push((*id, *new_case)),
        }
    }
    for (id, old_case) in &old_cases {
        if !new_cases.contains_key(id) {
            removed.push((*id, *old_case));
        }
    }

    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Case Changes"));
    append_eval_case_change_group(report, "regressions", &regressions);
    append_eval_case_change_group(report, "fixes", &fixes);
    append_eval_case_change_group(report, "added", &added);
    append_eval_case_change_group(report, "removed", &removed);
}

fn append_eval_failure_category_changes(
    report: &mut String,
    old_report: &EvalRunReport,
    new_report: &EvalRunReport,
) {
    let old_counts = eval_report_failure_categories(old_report);
    let new_counts = eval_report_failure_categories(new_report);
    let keys = old_counts
        .keys()
        .chain(new_counts.keys())
        .cloned()
        .collect::<BTreeSet<_>>();

    let _ = writeln!(report);
    let _ = writeln!(report, "  {}", report_section("Failure Categories"));
    if keys.is_empty() {
        let _ = writeln!(report, "    none");
        return;
    }
    for key in keys {
        let old_value = old_counts.get(&key).copied().unwrap_or(0);
        let new_value = new_counts.get(&key).copied().unwrap_or(0);
        append_eval_metric_usize(report, &key, old_value, new_value);
    }
}

fn eval_report_failure_categories(report: &EvalRunReport) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for case in &report.cases {
        for category in &case.result.failure_categories {
            *counts.entry(category.as_str().to_string()).or_insert(0) += 1;
        }
        if case.error.is_some() && case.result.failure_categories.is_empty() {
            *counts.entry("runtime_error".to_string()).or_insert(0) += 1;
        }
    }
    counts
}

fn append_eval_case_change_group(
    report: &mut String,
    label: &str,
    cases: &[(&str, &EvalRunCaseReport)],
) {
    let _ = writeln!(report, "    {} {}", report_label(label), cases.len());
    for (id, case) in cases.iter().take(8) {
        let failures = if case.result.failures.is_empty() {
            case.error.clone().unwrap_or_default()
        } else {
            case.result.failures.join("; ")
        };
        let preview = truncate_for_summary(&failures, 120);
        if preview.is_empty() {
            let _ = writeln!(report, "      - {id}");
        } else {
            let _ = writeln!(report, "      - {id}: {preview}");
        }
    }
    if cases.len() > 8 {
        let _ = writeln!(report, "      ... {} more", cases.len() - 8);
    }
}

fn signed_i128_delta(delta: i128) -> String {
    format!("{delta:+}")
}

fn signed_u128_delta(old_value: u128, new_value: u128) -> String {
    if new_value >= old_value {
        format!("+{}", new_value - old_value)
    } else {
        format!("-{}", old_value - new_value)
    }
}

fn optional_usize_label(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

#[allow(clippy::cast_precision_loss)]
fn percent_value(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| (numerator as f64 / denominator as f64) * 100.0)
}

fn optional_percent_label(value: Option<f64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| format!("{value:.1}%"))
}

pub(crate) fn print_eval_suite_check(suite: &EvalSuite, suite_path: &Path) {
    println!("Eval suite valid");
    println!("  file    {}", suite_path.display());
    println!("  suite   {}", suite.name);
    println!("  cases   {}", suite.cases.len());
    for case in &suite.cases {
        println!(
            "  - {} prompt_chars={} required_tools={} tool_sequence={} forbidden_tools={} permissions={} max_iterations={} max_tool_calls={} max_api_calls={}",
            case.id,
            case.prompt.chars().count(),
            case.required_tools.len(),
            case.required_tool_sequence.len(),
            case.forbidden_tools.len(),
            case.required_permission_outcomes.len(),
            case.max_iterations.map_or_else(|| "none".to_string(), |value| value.to_string()),
            case.max_tool_calls.map_or_else(|| "none".to_string(), |value| value.to_string()),
            case.max_api_calls.map_or_else(|| "none".to_string(), |value| value.to_string()),
        );
    }
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

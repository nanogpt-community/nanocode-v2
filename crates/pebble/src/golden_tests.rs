use std::fs;
use std::path::{Path, PathBuf};

use runtime::{
    ConfigSource, McpServerConfig, McpStdioServerConfig, McpStdioStderrMode, McpTransport,
    ScopedMcpServerConfig,
};
use serde_json::json;

use crate::mcp::{McpCatalog, McpServerStatus, McpToolBinding};
use crate::tool_render::render_tool_result_block;
use crate::trace_view::{
    load_turn_trace, render_replay_report, render_trace_report, replay_json_report,
    trace_json_report,
};
use crate::ui::{format_context_window_usage, ContextWindowInfo};

const GOLDEN_UPDATE_ENV: &str = "PEBBLE_UPDATE_GOLDENS";

#[test]
fn golden_trace_reports_match_committed_output() {
    let fixture = fixture_path("traces/tool-permission.json");
    let trace = load_turn_trace(&fixture).expect("golden trace should load");
    let display_path = Path::new("crates/pebble/tests/golden/traces/tool-permission.json");

    assert_golden(
        "trace-report.txt",
        strip_ansi(&render_trace_report(display_path, &trace)),
    );
    assert_golden(
        "replay-report.txt",
        strip_ansi(&render_replay_report(display_path, &trace)),
    );
    assert_golden(
        "trace-json.json",
        pretty_json(&trace_json_report(display_path, &trace)),
    );
    assert_golden(
        "replay-json.json",
        pretty_json(&replay_json_report(display_path, &trace)),
    );
}

#[test]
fn golden_tool_previews_match_committed_output() {
    let previews = [
        (
            "read_file",
            json!({
                "file": {
                    "filePath": "src/main.rs",
                    "startLine": 10,
                    "numLines": 3,
                    "totalLines": 120
                }
            }),
        ),
        (
            "bash",
            json!({
                "stdout": "alpha\nbeta\ngamma\n",
                "stderr": "",
                "exitCode": 0
            }),
        ),
        (
            "grep_search",
            json!({
                "numFiles": 2,
                "numMatches": 3,
                "content": "src/app.rs:10:context window\nsrc/ui.rs:20:context window\nREADME.md:3:context window"
            }),
        ),
        (
            "apply_patch",
            json!({
                "summary": "2 files changed",
                "dryRun": false,
                "changedFiles": [
                    {"action": "update", "filePath": "crates/pebble/src/app.rs"},
                    {"action": "add", "filePath": "crates/pebble/src/golden_tests.rs"}
                ]
            }),
        ),
    ];

    let mut rendered = String::new();
    for (tool_name, payload) in previews {
        rendered.push_str("== ");
        rendered.push_str(tool_name);
        rendered.push_str(" ==\n");
        rendered.push_str(&strip_ansi(&render_tool_result_block(
            tool_name,
            &payload.to_string(),
        )));
        rendered.push('\n');
    }

    assert_golden("tool-previews.txt", rendered);
}

#[test]
fn golden_context_window_percentages_match_committed_output() {
    let cases = [
        ContextWindowInfo {
            used_tokens: 0,
            max_tokens: 200_000,
        },
        ContextWindowInfo {
            used_tokens: 12_345,
            max_tokens: 200_000,
        },
        ContextWindowInfo {
            used_tokens: 199_999,
            max_tokens: 200_000,
        },
        ContextWindowInfo {
            used_tokens: 10,
            max_tokens: 0,
        },
    ];

    let rendered = cases
        .into_iter()
        .map(format_context_window_usage)
        .collect::<Vec<_>>()
        .join("\n");
    assert_golden("context-window.txt", format!("{rendered}\n"));
}

#[test]
fn golden_mcp_tool_specs_match_committed_output() {
    let catalog = McpCatalog {
        servers: vec![McpServerStatus {
            server_name: "docs".to_string(),
            scope: ConfigSource::Project,
            enabled: true,
            transport: McpTransport::Stdio,
            loaded: true,
            tool_count: 1,
            note: "stdio tools loaded".to_string(),
        }],
        tools: vec![McpToolBinding {
            exposed_name: "mcp__docs__search".to_string(),
            server_name: "docs".to_string(),
            upstream_name: "search".to_string(),
            description: "Search project documentation".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
            config: ScopedMcpServerConfig {
                scope: ConfigSource::Project,
                enabled: true,
                config: McpServerConfig::Stdio(McpStdioServerConfig {
                    command: "docs-mcp".to_string(),
                    args: Vec::new(),
                    env: Default::default(),
                    stderr: McpStdioStderrMode::Inherit,
                }),
            },
        }],
    };

    let specs = catalog
        .tool_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "required_permission": format!("{:?}", spec.required_permission),
            })
        })
        .collect::<Vec<_>>();
    assert_golden("mcp-tool-specs.json", pretty_json(&json!(specs)));
}

fn assert_golden(name: &str, actual: String) {
    let actual = normalize_newlines(&actual);
    let expected_path = fixture_path(name);
    if update_goldens() {
        fs::write(&expected_path, actual).expect("golden fixture should update");
        return;
    }

    let expected = fs::read_to_string(&expected_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", expected_path.display()));
    assert_eq!(
        normalize_newlines(&expected),
        actual,
        "golden fixture mismatch: {}\nrerun with {GOLDEN_UPDATE_ENV}=1 to bless intentional changes",
        expected_path.display()
    );
}

fn pretty_json(value: &serde_json::Value) -> String {
    let mut output = serde_json::to_string_pretty(value).expect("golden JSON should serialize");
    output.push('\n');
    output
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name)
}

fn update_goldens() -> bool {
    std::env::var_os(GOLDEN_UPDATE_ENV).is_some()
}

fn normalize_newlines(value: &str) -> String {
    value.replace("\r\n", "\n")
}

fn strip_ansi(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        output.push(ch);
    }
    output
}

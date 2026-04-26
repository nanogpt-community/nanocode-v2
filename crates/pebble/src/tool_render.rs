use crate::ui;

/// Render the compact, already-ANSI-styled result block that visually hangs
/// off the tool-call header we printed when the model invoked the tool.
///
/// Where possible we use [`render_structured_tool_preview`] to produce a
/// hand-crafted summary for the tool family. When the output is unknown we
/// fall back to a single "N bytes / N lines" one-liner so the transcript
/// stays compact. Full payloads are always retained in conversation context
/// regardless of what the TUI shows.
pub(crate) fn render_tool_result_block(tool_name: &str, output: &str) -> String {
    if let Some(block) = render_structured_tool_preview(tool_name, output) {
        return block;
    }

    if output.trim().is_empty() {
        return ui::tool_result_block(&[("", "(empty result)")]);
    }
    let line_count = output.lines().count();
    let char_count = output.chars().count();
    let summary = if line_count > 1 {
        format!("{line_count} lines · {char_count} chars")
    } else {
        format!("{char_count} chars")
    };
    ui::tool_result_block(&[("", &summary)])
}

fn render_structured_tool_preview(tool_name: &str, output: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    match tool_name {
        "read_file" => render_read_file_preview(&value),
        "glob_search" => render_glob_search_preview(&value),
        "grep_search" => render_grep_search_preview(&value),
        "bash" => render_bash_preview(&value),
        "write_file" | "edit_file" => render_write_edit_preview(&value),
        "apply_patch" => render_apply_patch_preview(&value),
        _ => None,
    }
}

fn render_read_file_preview(value: &serde_json::Value) -> Option<String> {
    let file = value.get("file")?;
    let path = file.get("filePath")?.as_str()?;
    let start_line = file.get("startLine").and_then(serde_json::Value::as_u64);
    let num_lines = file.get("numLines").and_then(serde_json::Value::as_u64);
    let total_lines = file.get("totalLines").and_then(serde_json::Value::as_u64);
    let end_line = match (start_line, num_lines) {
        (Some(start), Some(count)) if count > 0 => Some(start + count - 1),
        _ => start_line,
    };
    let range = format!(
        "lines {}–{} of {}",
        start_line.unwrap_or(1),
        end_line.unwrap_or(start_line.unwrap_or(1)),
        total_lines.unwrap_or(num_lines.unwrap_or(0))
    );
    Some(ui::tool_result_block(&[("", path), ("range", &range)]))
}

fn render_glob_search_preview(value: &serde_json::Value) -> Option<String> {
    let num_files = value.get("numFiles").and_then(serde_json::Value::as_u64)?;
    let filenames = value.get("filenames")?.as_array()?;
    let truncated = filenames.len() > 5
        || value.get("truncated").and_then(serde_json::Value::as_bool) == Some(true);

    // Only show a handful of filenames — just enough to make the match set
    // recognisable without flooding the transcript.
    let preview_names: Vec<String> = filenames
        .iter()
        .filter_map(serde_json::Value::as_str)
        .take(5)
        .map(std::string::ToString::to_string)
        .collect();

    let summary = if truncated {
        format!("{num_files} files (showing first {})", preview_names.len())
    } else {
        format!("{num_files} files")
    };

    let mut block = ui::tool_result_block(&[("matched", &summary)]);
    if !preview_names.is_empty() {
        block.push_str(&ui::tool_result_body("files", &preview_names.join("\n")));
    }
    Some(block)
}

fn render_grep_search_preview(value: &serde_json::Value) -> Option<String> {
    let num_files = value.get("numFiles").and_then(serde_json::Value::as_u64)?;
    let num_matches = value.get("numMatches").and_then(serde_json::Value::as_u64);
    let content = value
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    // Keep the preview tight — five lines is plenty of context in the
    // transcript; the full result is still in conversation state.
    let preview = truncate_tool_text(content, 5, 320);
    let heading = match num_matches {
        Some(matches) => format!("{num_files} files · {matches} matches"),
        None => format!("{num_files} files"),
    };

    let mut block = ui::tool_result_block(&[("matched", &heading)]);
    if !preview.trim().is_empty() {
        block.push_str(&ui::tool_result_body("hits", &preview));
    }
    Some(block)
}

fn render_bash_preview(value: &serde_json::Value) -> Option<String> {
    let stdout = value
        .get("stdout")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let stderr = value
        .get("stderr")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let exit_code = value
        .get("exitCode")
        .or_else(|| value.get("exit_code"))
        .and_then(serde_json::Value::as_i64);

    let mut block = String::new();

    // Always show the exit code so failed commands are instantly scannable.
    if let Some(code) = exit_code {
        let label = if code == 0 { "ok" } else { "exit" };
        block.push_str(&ui::tool_result_block(&[(label, &code.to_string())]));
    }

    if !stdout.trim().is_empty() {
        let preview = truncate_tool_text(stdout, 8, 600);
        block.push_str(&ui::tool_result_body("stdout", &preview));
    }
    if !stderr.trim().is_empty() {
        let preview = truncate_tool_text(stderr, 4, 400);
        block.push_str(&ui::tool_result_body("stderr", &preview));
    }

    if block.is_empty() {
        block.push_str(&ui::tool_result_block(&[("", "(no output)")]));
    }
    Some(block)
}

fn render_write_edit_preview(value: &serde_json::Value) -> Option<String> {
    let path = value
        .get("path")
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.get("filePath").and_then(serde_json::Value::as_str))
        .unwrap_or("unknown");
    let message = value
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let lines = value
        .get("content")
        .and_then(serde_json::Value::as_str)
        .map(|content| content.lines().count());

    let mut rows: Vec<(&str, String)> = vec![("", path.to_string())];
    if let Some(lines) = lines {
        let word = if lines == 1 { "line" } else { "lines" };
        rows.push(("content", format!("{lines} {word}")));
    }
    if !message.trim().is_empty() {
        rows.push(("result", message.trim().to_string()));
    }

    // Borrow the string values to match `tool_result_block`'s signature.
    let pairs: Vec<(&str, &str)> = rows.iter().map(|(k, v)| (*k, v.as_str())).collect();
    Some(ui::tool_result_block(&pairs))
}

fn render_apply_patch_preview(value: &serde_json::Value) -> Option<String> {
    let summary = value
        .get("summary")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("patch processed");
    let mode = if value
        .get("dryRun")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        "check"
    } else {
        "apply"
    };
    let mut block = ui::tool_result_block(&[("mode", mode), ("summary", summary)]);
    let files = value
        .get("changedFiles")
        .and_then(serde_json::Value::as_array)
        .map(|files| {
            files
                .iter()
                .take(8)
                .filter_map(|file| {
                    let action = file.get("action")?.as_str()?;
                    let path = file.get("filePath")?.as_str()?;
                    Some(format!("{action:<8} {path}"))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !files.is_empty() {
        block.push_str(&ui::tool_result_body("files", &files.join("\n")));
    }
    Some(block)
}

fn truncate_tool_text(input: &str, max_lines: usize, max_chars: usize) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    let mut line_count = 0usize;
    let mut truncated = false;

    for line in input.lines() {
        if line_count == max_lines {
            truncated = true;
            break;
        }
        let next_len = if output.is_empty() {
            line.len()
        } else {
            output.len() + 1 + line.len()
        };
        if next_len > max_chars {
            let remaining = max_chars.saturating_sub(output.len());
            if remaining > 1 {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(
                    &line
                        .chars()
                        .take(remaining.saturating_sub(1))
                        .collect::<String>(),
                );
            }
            truncated = true;
            break;
        }
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(line);
        line_count += 1;
    }

    if truncated {
        if !output.ends_with("...") {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str("...");
        }
    } else if input.ends_with('\n') && !output.is_empty() {
        output.push('\n');
    }

    output
}

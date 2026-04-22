#![allow(dead_code)]
//! Shared terminal UI primitives for the Pebble REPL.
//!
//! This module concentrates the visual vocabulary of the interactive shell:
//! the welcome banner, status panels, colored badges for tool calls, turn
//! separators, prompt styling, and other reusable widgets. Keeping these in
//! one place means the REPL presents a consistent, cohesive look instead of
//! a grab-bag of ad-hoc `println!`s.

use crossterm::style::{Color, Stylize};

/// Width of framed panels (banner, status cards, tool-call boxes).
pub const PANEL_WIDTH: usize = 72;

/// Unicode/ANSI palette used by the shell. Centralising the palette makes it
/// easy to tweak the whole look and feel in one place.
pub mod palette {
    use crossterm::style::Color;

    pub const ACCENT: Color = Color::Cyan;
    pub const ACCENT_DIM: Color = Color::DarkCyan;
    pub const BRAND: Color = Color::Magenta;
    pub const BRAND_DIM: Color = Color::DarkMagenta;
    pub const MUTED: Color = Color::DarkGrey;
    pub const OK: Color = Color::Green;
    pub const WARN: Color = Color::Yellow;
    pub const ERR: Color = Color::Red;
    pub const INFO: Color = Color::Blue;

    /// Color used to highlight the name of a tool call.
    pub const TOOL_NAME: Color = Color::Cyan;
    /// Color for arguments rendered inline next to a tool call.
    pub const TOOL_ARG: Color = Color::DarkGrey;
    /// Color for permission-mode badges.
    pub const PERMISSION: Color = Color::Yellow;
}

/// Compute the printable width of a string, ignoring ANSI escape sequences.
/// Falls back to a simple char count for anything that isn't an escape.
#[must_use]
pub fn visible_width(text: &str) -> usize {
    let mut width = 0usize;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            // Eat a CSI sequence (`ESC [ ... letter`).
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
        }
        width += 1;
    }
    width
}

/// A single row to place inside a bordered panel.
#[derive(Debug, Clone)]
pub enum PanelRow {
    /// Blank spacer row inside the frame.
    Blank,
    /// Horizontal rule row inside the frame.
    Divider,
    /// A label/value pair rendered in two columns.
    Field { label: String, value: String },
    /// A free-form already-styled text line.
    Line(String),
    /// A subdued section heading.
    Section(String),
}

/// Build a rounded, colour-accented box around the provided rows. The title
/// bar is printed in the accent color; inside rows may carry their own ANSI
/// styling without breaking alignment thanks to [`visible_width`].
#[must_use]
pub fn panel(title: &str, rows: &[PanelRow]) -> String {
    let width = PANEL_WIDTH;
    let inner = width.saturating_sub(4); // borders + one pad on each side

    let mut out = String::new();

    // Top border with inline title: ╭── Title ──────────────╮
    let title_segment = format!(" {} ", title.bold().with(palette::ACCENT));
    let title_visible = visible_width(&title_segment);
    let remaining = inner.saturating_sub(title_visible).saturating_add(2);
    let top = format!(
        "{}{}{}{}",
        "╭─".with(palette::ACCENT_DIM),
        title_segment,
        "─".repeat(remaining).with(palette::ACCENT_DIM),
        "╮".with(palette::ACCENT_DIM),
    );
    out.push_str(&top);
    out.push('\n');

    for row in rows {
        let body = match row {
            PanelRow::Blank => String::new(),
            PanelRow::Divider => "┈".repeat(inner).with(palette::MUTED).to_string(),
            PanelRow::Section(text) => {
                format!("{}", text.clone().bold().with(palette::BRAND))
            }
            PanelRow::Field { label, value } => {
                let label_width = 14usize;
                let label_styled = format!("{label:<label_width$}");
                format!(
                    "{} {}",
                    label_styled.with(palette::MUTED),
                    value.clone().with(Color::White),
                )
            }
            PanelRow::Line(text) => text.clone(),
        };
        out.push_str(&frame_line(&body, inner));
        out.push('\n');
    }

    // Bottom border
    let bottom = format!(
        "{}{}{}",
        "╰".with(palette::ACCENT_DIM),
        "─"
            .repeat(width.saturating_sub(2))
            .with(palette::ACCENT_DIM),
        "╯".with(palette::ACCENT_DIM),
    );
    out.push_str(&bottom);
    out
}

fn frame_line(content: &str, inner_width: usize) -> String {
    let content_visible = visible_width(content);
    let pad = inner_width.saturating_sub(content_visible);
    format!(
        "{} {}{} {}",
        "│".with(palette::ACCENT_DIM),
        content,
        " ".repeat(pad),
        "│".with(palette::ACCENT_DIM),
    )
}

/// Render the welcome banner shown at REPL startup. Keeps the most important
/// signal (what model we're talking to and how privileged we are) front and
/// centre so the user can see it immediately.
#[must_use]
pub fn welcome_banner(info: &BannerInfo<'_>) -> String {
    let logo_line = format!(
        "{}  {}",
        "◆ pebble".bold().with(palette::BRAND),
        format!("v{}", info.version).with(palette::MUTED),
    );
    let tagline = "rust agentic coding shell"
        .italic()
        .with(palette::MUTED)
        .to_string();

    let mut rows = vec![
        PanelRow::Line(logo_line),
        PanelRow::Line(tagline),
        PanelRow::Blank,
        PanelRow::Section("Session".to_string()),
        PanelRow::Field {
            label: "service".to_string(),
            value: info.service.to_string(),
        },
        PanelRow::Field {
            label: "model".to_string(),
            value: info.model.to_string(),
        },
    ];
    if let Some(provider) = info.provider {
        rows.push(PanelRow::Field {
            label: "provider".to_string(),
            value: provider.to_string(),
        });
    }
    rows.push(PanelRow::Field {
        label: "permissions".to_string(),
        value: format!(
            "{}",
            info.permission_mode.to_string().with(palette::PERMISSION)
        ),
    });
    if let Some(cwd) = info.cwd {
        rows.push(PanelRow::Field {
            label: "cwd".to_string(),
            value: cwd.to_string(),
        });
    }
    rows.push(PanelRow::Blank);
    rows.push(PanelRow::Section("Shortcuts".to_string()));
    rows.push(PanelRow::Line(shortcut_hints()));
    rows.push(PanelRow::Line(help_hint()));

    panel("pebble", &rows)
}

fn shortcut_hints() -> String {
    let items = [
        ("Shift+Enter", "newline"),
        ("Ctrl+C", "cancel input"),
        ("Ctrl+D", "exit"),
    ];
    items
        .iter()
        .map(|(key, desc)| {
            format!(
                "{} {}",
                format!(" {key} ")
                    .on(palette::ACCENT_DIM)
                    .with(Color::Black),
                desc.to_string().with(palette::MUTED),
            )
        })
        .collect::<Vec<_>>()
        .join("  ")
}

fn help_hint() -> String {
    format!(
        "{} {}",
        "type".to_string().with(palette::MUTED),
        "/help".bold().with(palette::ACCENT),
    )
}

/// Public entry point for caller-provided banner data.
#[derive(Debug, Clone, Copy)]
pub struct BannerInfo<'a> {
    pub version: &'a str,
    pub service: &'a str,
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub permission_mode: &'a str,
    pub cwd: Option<&'a str>,
}

/// Emit a subdued horizontal separator used between turns. The separator is
/// visually quiet so it doesn't steal attention from assistant output.
#[must_use]
pub fn turn_separator() -> String {
    let line = "─".repeat(PANEL_WIDTH);
    format!("\n{}\n", line.with(palette::MUTED))
}

/// Render the primary line-editor prompt. We use a chevron glyph in the
/// brand color so users can find the input cursor at a glance, even on
/// output-heavy sessions.
#[must_use]
pub fn prompt_string() -> String {
    #[cfg(windows)]
    {
        "> ".to_string()
    }
    #[cfg(not(windows))]
    {
        format!("{} ", "❯".bold().with(palette::BRAND))
    }
}

/// Render a compact one-line hint shown immediately above the prompt. This
/// is where we surface ambient state (permission mode, thinking toggle, etc.)
/// so users never have to `/status` to know what they're about to send.
#[must_use]
pub fn prompt_status_line(info: &PromptStatusInfo<'_>) -> String {
    let mut segments = Vec::new();

    segments.push(styled_pill("model", info.model, palette::ACCENT));
    segments.push(styled_pill("mode", info.collaboration_mode, palette::BRAND));
    segments.push(styled_pill(
        "reasoning",
        info.reasoning_effort,
        palette::BRAND,
    ));
    segments.push(styled_pill(
        "perms",
        info.permission_mode,
        palette::PERMISSION,
    ));
    if info.fast_mode {
        segments.push(styled_pill("fast", "on", palette::OK));
    }
    if info.proxy_tool_calls {
        segments.push(styled_pill("proxy", "on", palette::INFO));
    }
    if let Some(tokens) = info.estimated_tokens {
        segments.push(styled_pill(
            "ctx",
            &format_compact_tokens(tokens),
            palette::MUTED,
        ));
    }

    segments.join(" ")
}

fn styled_pill(label: &str, value: &str, color: Color) -> String {
    let label_segment = format!(" {label} ").with(Color::Black).on(color);
    let value_segment = format!(" {value} ").with(color).on(palette::MUTED);
    format!("{label_segment}{value_segment}")
}

#[allow(clippy::cast_precision_loss)]
fn format_compact_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

/// Inputs for [`prompt_status_line`].
#[derive(Debug, Clone, Copy)]
pub struct PromptStatusInfo<'a> {
    pub model: &'a str,
    pub permission_mode: &'a str,
    pub collaboration_mode: &'a str,
    pub reasoning_effort: &'a str,
    pub fast_mode: bool,
    pub proxy_tool_calls: bool,
    pub estimated_tokens: Option<u64>,
}

/// Render the header that precedes a tool call in the transcript.
///
/// Example:
///   ⏺ Bash › git status
///
/// The leading glyph makes tool activity skimmable even in a dense log.
#[must_use]
pub fn tool_call_header(name: &str, summary: &str) -> String {
    let icon = tool_icon(name);
    let name_styled = name.bold().with(palette::TOOL_NAME).to_string();
    if summary.is_empty() {
        format!("{} {}", icon.with(palette::ACCENT), name_styled)
    } else {
        let summary_styled = summary.to_owned().with(palette::TOOL_ARG);
        format!(
            "{} {} {} {}",
            icon.with(palette::ACCENT),
            name_styled,
            "›".with(palette::MUTED),
            summary_styled,
        )
    }
}

/// Pick a glyph per tool family so users can visually distinguish reads from
/// writes from shell from search.
fn tool_icon(tool_name: &str) -> &'static str {
    match tool_name {
        "bash" | "Bash" => "▶",
        "read_file" | "Read" => "📖",
        "write_file" | "Write" | "edit_file" | "Edit" | "MultiEdit" => "✎",
        "glob_search" | "Glob" => "◇",
        "grep_search" | "Grep" => "◈",
        "web_search" | "WebSearch" | "WebFetch" | "web_scrape" | "WebScrape" => "🌐",
        "ls" | "Ls" => "📁",
        _ => "⏺",
    }
}

/// Emit a short dim note. Used for auxiliary notices (auto-compaction, budget
/// warnings, tool result truncations) where we want to inform without
/// distracting from the conversation flow.
#[must_use]
pub fn dim_note(text: &str) -> String {
    format!("{} {}", "ℹ".with(palette::MUTED), text.with(palette::MUTED))
}

/// Emit a warning-styled note (yellow, with a warning glyph).
#[must_use]
pub fn warning_note(text: &str) -> String {
    format!("{} {}", "⚠".with(palette::WARN), text.with(palette::WARN))
}

/// Emit a success note (green, with a check glyph).
#[must_use]
pub fn success_note(text: &str) -> String {
    format!("{} {}", "✔".with(palette::OK), text.with(palette::OK))
}

/// Emit an error note (red, with a cross glyph).
#[must_use]
pub fn error_note(text: &str) -> String {
    format!("{} {}", "✘".with(palette::ERR), text.with(palette::ERR))
}

/// Styling for the "thinking" stream that surfaces model-internal reasoning
/// in a clearly-distinct, dim colour so it never competes visually with the
/// primary response.
#[must_use]
pub fn thinking_chunk(text: &str) -> String {
    format!("{}", text.italic().with(palette::MUTED))
}

/// Styled leader printed once before a stream of thinking output.
#[must_use]
pub fn thinking_lead() -> String {
    format!(
        "{} {}\n",
        "✦".with(palette::BRAND_DIM),
        "thinking".italic().with(palette::MUTED),
    )
}

/// Render a compact, indented tool-result block that visually hangs off the
/// tool-call header above it. The layout intentionally does NOT use markdown
/// because the tool-call header already announced the call; what we want
/// underneath is a terse, dim confirmation, not a second heading.
///
/// `summary_lines` is an ordered list of `(label, value)` pairs. Labels are
/// rendered muted, values are rendered normally. Pass an empty `label` to
/// render a single freeform line.
///
/// Example output:
/// ```text
///   ⎿ wrote 1 line to /tmp/test.txt
/// ```
#[must_use]
pub fn tool_result_block(summary_lines: &[(&str, &str)]) -> String {
    if summary_lines.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (idx, (label, value)) in summary_lines.iter().enumerate() {
        let glyph = if idx == 0 { "⎿" } else { " " };
        let label_part = if label.is_empty() {
            String::new()
        } else {
            format!("{} ", format!("{label}:").with(palette::MUTED))
        };
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(
                "  {} {label_part}{}\n",
                glyph.to_string().with(palette::MUTED),
                value.to_string().with(palette::MUTED),
            ),
        );
    }
    out
}

/// Render a multi-line tool result whose body is an already-captured chunk of
/// text (bash stdout, grep hits, file range). The body is indented and
/// dimmed, with a `⎿` tree-drawing prefix on the first line. Long bodies
/// should be truncated by the caller before being passed in.
#[must_use]
pub fn tool_result_body(heading: &str, body: &str) -> String {
    let mut out = String::new();
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!(
            "  {} {}\n",
            "⎿".with(palette::MUTED),
            heading.to_string().with(palette::MUTED),
        ),
    );
    for line in body.lines() {
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("    {}\n", line.to_string().with(palette::MUTED)),
        );
    }
    out
}

/// Prefix emitted once per assistant response to visually anchor the model's
/// voice. Rendered as a bold brand-colored bullet on its own line so the
/// reply is easy to locate in a tool-heavy transcript.
#[must_use]
pub fn assistant_lead() -> String {
    format!(
        "{} {}\n",
        "●".with(palette::BRAND),
        "pebble".bold().with(palette::BRAND),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_width_strips_ansi() {
        let raw = format!("{}", "hello".with(palette::ACCENT));
        assert_eq!(visible_width(&raw), 5);
    }

    #[test]
    fn panel_contains_title_and_rows() {
        let rendered = panel(
            "Demo",
            &[
                PanelRow::Field {
                    label: "key".into(),
                    value: "value".into(),
                },
                PanelRow::Blank,
                PanelRow::Line("freeform".into()),
            ],
        );
        assert!(rendered.contains("Demo"));
        assert!(rendered.contains("key"));
        assert!(rendered.contains("value"));
        assert!(rendered.contains("freeform"));
        assert!(rendered.contains("╭"));
        assert!(rendered.contains("╯"));
    }

    #[test]
    fn welcome_banner_mentions_core_fields() {
        let banner = welcome_banner(&BannerInfo {
            version: "0.2.0",
            service: "NanoGPT",
            model: "zai-org/glm-5.1",
            provider: Some("fireworks"),
            permission_mode: "workspace-write",
            cwd: Some("/tmp/project"),
        });
        assert!(banner.contains("pebble"));
        assert!(banner.contains("v0.2.0"));
        assert!(banner.contains("NanoGPT"));
        assert!(banner.contains("zai-org/glm-5.1"));
        assert!(banner.contains("fireworks"));
        assert!(banner.contains("workspace-write"));
        assert!(banner.contains("/tmp/project"));
    }

    #[test]
    fn tool_call_header_includes_icon_and_name() {
        let header = tool_call_header("Bash", "git status");
        assert!(header.contains("Bash"));
        assert!(header.contains("git status"));
    }

    #[test]
    fn format_compact_tokens_scales() {
        assert_eq!(format_compact_tokens(500), "500");
        assert_eq!(format_compact_tokens(1_500), "1.5k");
        assert_eq!(format_compact_tokens(2_400_000), "2.4M");
    }

    #[test]
    fn prompt_status_line_contains_permission_and_model() {
        let line = prompt_status_line(&PromptStatusInfo {
            model: "glm-5.1",
            permission_mode: "read-only",
            collaboration_mode: "build",
            reasoning_effort: "medium",
            fast_mode: true,
            proxy_tool_calls: false,
            estimated_tokens: Some(12_000),
        });
        assert!(line.contains("glm-5.1"));
        assert!(line.contains("read-only"));
        assert!(line.contains("build"));
        assert!(line.contains("medium"));
        assert!(line.contains("fast"));
        assert!(line.contains("12.0k"));
    }
}

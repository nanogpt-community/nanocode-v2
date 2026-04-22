use std::borrow::Cow;
use std::cell::RefCell;
use std::io::{self, IsTerminal, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossterm::cursor::MoveToColumn;
use crossterm::execute;
use crossterm::terminal::{size, Clear, ClearType};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Context, Editor, Event, EventContext,
    EventHandler, Helper, KeyCode, KeyEvent, Modifiers,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Submit(String),
    Cancel,
    Exit,
    ToggleMode,
}

struct SlashCommandHelper {
    completions: Vec<String>,
    current_line: RefCell<String>,
}

impl SlashCommandHelper {
    fn new(completions: Vec<String>) -> Self {
        Self {
            completions,
            current_line: RefCell::new(String::new()),
        }
    }

    fn reset_current_line(&self) {
        self.current_line.borrow_mut().clear();
    }

    fn set_completions(&mut self, completions: Vec<String>) {
        self.completions = completions;
    }

    fn current_line(&self) -> String {
        self.current_line.borrow().clone()
    }

    fn set_current_line(&self, line: &str) {
        let mut current = self.current_line.borrow_mut();
        current.clear();
        current.push_str(line);
    }
}

impl Completer for SlashCommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let Some((start, prefix)) = slash_command_prefix(line, pos) else {
            return Ok((0, Vec::new()));
        };

        let matches = self
            .completions
            .iter()
            .filter(|candidate| candidate.starts_with(prefix))
            .map(|candidate| Pair {
                display: candidate.clone(),
                replacement: candidate.clone(),
            })
            .collect();

        Ok((start, matches))
    }
}

impl Hinter for SlashCommandHelper {
    type Hint = String;
}

impl Highlighter for SlashCommandHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        self.set_current_line(line);
        Cow::Borrowed(line)
    }

    fn highlight_char(&self, line: &str, _pos: usize, _kind: CmdKind) -> bool {
        self.set_current_line(line);
        false
    }
}

impl Validator for SlashCommandHelper {}
impl Helper for SlashCommandHelper {}

struct PasteSafeSubmitHandler;

impl ConditionalEventHandler for PasteSafeSubmitHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        if ctx.line().is_empty() {
            None
        } else {
            Some(Cmd::AcceptLine)
        }
    }
}

struct EmptyLineTabHandler {
    toggled: Arc<AtomicBool>,
}

impl ConditionalEventHandler for EmptyLineTabHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        if ctx.line().is_empty() {
            self.toggled.store(true, Ordering::SeqCst);
            Some(Cmd::Interrupt)
        } else {
            None
        }
    }
}

fn paste_safe_mode_enabled() -> bool {
    env_flag_enabled("PEBBLE_PASTE_SAFE")
}

pub struct LineEditor {
    prompt: String,
    status_line: Option<String>,
    completions: Vec<String>,
    editor: Editor<SlashCommandHelper, DefaultHistory>,
    pending_mode_toggle: Arc<AtomicBool>,
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>, completions: Vec<String>) -> Self {
        let pending_mode_toggle = Arc::new(AtomicBool::new(false));
        let editor = Self::build_editor(completions.clone(), pending_mode_toggle.clone());

        Self {
            prompt: prompt.into(),
            status_line: None,
            completions,
            editor,
            pending_mode_toggle,
        }
    }

    /// Update the visible input prompt glyph. Callers typically do this when
    /// switching modes (e.g. toggling thinking) so the prompt stays in sync
    /// with ambient state.
    #[allow(dead_code)]
    pub fn set_prompt(&mut self, prompt: impl Into<String>) {
        self.prompt = prompt.into();
    }

    /// Set (or clear) a single-line status indicator that is printed just
    /// above the input prompt on every `read_line` call. This is where the
    /// REPL advertises the current model, permission mode, token budget,
    /// etc. `None` disables the line.
    pub fn set_status_line(&mut self, status_line: Option<String>) {
        self.status_line = status_line;
    }

    pub fn push_history(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        if entry.trim().is_empty() {
            return;
        }

        let _ = self.editor.add_history_entry(entry);
    }

    pub fn set_completions(&mut self, completions: Vec<String>) {
        self.completions = completions.clone();
        if let Some(helper) = self.editor.helper_mut() {
            helper.set_completions(completions);
        }
    }

    fn print_status_line(&self) -> io::Result<()> {
        if let Some(status) = self.status_line.as_ref() {
            let rendered = fit_status_line(status, terminal_status_width());
            let mut stdout = io::stdout();
            execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
            writeln!(stdout, "{rendered}")?;
            stdout.flush()?;
        }
        Ok(())
    }

    pub fn read_line(&mut self) -> io::Result<ReadOutcome> {
        loop {
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                return self.read_line_fallback();
            }
            self.pending_mode_toggle.store(false, Ordering::SeqCst);

            if let Some(helper) = self.editor.helper_mut() {
                helper.reset_current_line();
            }

            self.print_status_line()?;

            match self.editor.readline(&self.prompt) {
                Ok(line) => {
                    if self.handle_submission(&line)? {
                        continue;
                    }
                    return Ok(ReadOutcome::Submit(line));
                }
                Err(ReadlineError::Interrupted) => {
                    if self.pending_mode_toggle.swap(false, Ordering::SeqCst) {
                        self.finish_interrupted_read()?;
                        return Ok(ReadOutcome::ToggleMode);
                    }
                    let has_input = !self.current_line().is_empty();
                    self.finish_interrupted_read()?;
                    return if has_input {
                        Ok(ReadOutcome::Cancel)
                    } else {
                        Ok(ReadOutcome::Exit)
                    };
                }
                Err(ReadlineError::Eof) => {
                    self.finish_interrupted_read()?;
                    return Ok(ReadOutcome::Exit);
                }
                Err(error) => return Err(io::Error::other(error)),
            }
        }
    }

    fn build_editor(
        completions: Vec<String>,
        pending_mode_toggle: Arc<AtomicBool>,
    ) -> Editor<SlashCommandHelper, DefaultHistory> {
        let paste_safe_mode = paste_safe_mode_enabled();
        let config = Config::builder()
            .completion_type(CompletionType::List)
            .build();
        let mut editor = Editor::<SlashCommandHelper, DefaultHistory>::with_config(config)
            .expect("rustyline editor should initialize");
        editor.set_helper(Some(SlashCommandHelper::new(completions)));
        editor.bind_sequence(KeyEvent(KeyCode::Enter, Modifiers::SHIFT), Cmd::Newline);
        editor.bind_sequence(KeyEvent(KeyCode::Char('J'), Modifiers::CTRL), Cmd::Newline);
        if paste_safe_mode {
            editor.bind_sequence(KeyEvent(KeyCode::Enter, Modifiers::NONE), Cmd::Newline);
            editor.bind_sequence(
                KeyEvent(KeyCode::Char('D'), Modifiers::CTRL),
                EventHandler::Conditional(Box::new(PasteSafeSubmitHandler)),
            );
        }
        editor.bind_sequence(KeyEvent(KeyCode::Up, Modifiers::NONE), Cmd::PreviousHistory);
        editor.bind_sequence(KeyEvent(KeyCode::Down, Modifiers::NONE), Cmd::NextHistory);
        editor.bind_sequence(
            KeyEvent(KeyCode::Tab, Modifiers::NONE),
            EventHandler::Conditional(Box::new(EmptyLineTabHandler {
                toggled: pending_mode_toggle,
            })),
        );
        editor
    }

    fn current_line(&self) -> String {
        self.editor
            .helper()
            .map_or_else(String::new, SlashCommandHelper::current_line)
    }

    fn finish_interrupted_read(&mut self) -> io::Result<()> {
        if let Some(helper) = self.editor.helper_mut() {
            helper.reset_current_line();
        }
        let mut stdout = io::stdout();
        writeln!(stdout)
    }

    fn read_line_fallback(&mut self) -> io::Result<ReadOutcome> {
        loop {
            self.print_status_line()?;
            let mut stdout = io::stdout();
            write!(stdout, "{}", self.prompt)?;
            stdout.flush()?;

            let Some(buffer) = read_full_fallback_input(&mut io::stdin())? else {
                return Ok(ReadOutcome::Exit);
            };

            if self.handle_submission(&buffer)? {
                continue;
            }

            return Ok(ReadOutcome::Submit(buffer));
        }
    }

    fn handle_submission(&mut self, _line: &str) -> io::Result<bool> {
        Ok(false)
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn read_full_fallback_input(reader: &mut impl Read) -> io::Result<Option<String>> {
    let mut buffer = String::new();
    reader.read_to_string(&mut buffer)?;
    if buffer.is_empty() {
        return Ok(None);
    }

    while matches!(buffer.chars().last(), Some('\n' | '\r')) {
        buffer.pop();
    }

    Ok(Some(buffer))
}

fn terminal_status_width() -> Option<usize> {
    size()
        .ok()
        .map(|(columns, _)| usize::from(columns).saturating_sub(1))
        .filter(|width| *width > 0)
}

fn fit_status_line(status: &str, max_width: Option<usize>) -> String {
    let status = status.trim_end_matches(['\r', '\n']);
    let Some(max_width) = max_width else {
        return status.to_string();
    };
    if crate::ui::visible_width(status) <= max_width {
        return status.to_string();
    }
    truncate_ansi_visible_width(status, max_width)
}

fn truncate_ansi_visible_width(input: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let mut output = String::new();
    let mut visible = 0usize;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            output.push(ch);
            output.push(chars.next().expect("peeked ansi introducer"));
            for next in chars.by_ref() {
                output.push(next);
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }

        if visible >= max_width {
            break;
        }

        output.push(ch);
        visible += 1;
    }

    if output.contains('\u{1b}') {
        output.push_str("\u{1b}[0m");
    }
    output
}

fn slash_command_prefix(line: &str, pos: usize) -> Option<(usize, &str)> {
    if pos != line.len() {
        return None;
    }

    let prefix = &line[..pos];
    if !prefix.starts_with('/') {
        return None;
    }

    Some((0, prefix))
}

#[cfg(test)]
mod tests {
    use super::{
        fit_status_line, paste_safe_mode_enabled, read_full_fallback_input, slash_command_prefix,
        truncate_ansi_visible_width, LineEditor, SlashCommandHelper,
    };
    use rustyline::completion::Completer;
    use rustyline::highlight::Highlighter;
    use rustyline::history::{DefaultHistory, History};
    use rustyline::Context;
    use std::io::Cursor;

    #[test]
    fn extracts_only_terminal_slash_command_prefixes() {
        assert_eq!(slash_command_prefix("/he", 3), Some((0, "/he")));
        assert_eq!(slash_command_prefix("/help me", 5), None);
        assert_eq!(slash_command_prefix("hello", 5), None);
        assert_eq!(slash_command_prefix("/help", 2), None);
    }

    #[test]
    fn completes_matching_slash_commands() {
        let helper = SlashCommandHelper::new(vec![
            "/help".to_string(),
            "/hello".to_string(),
            "/status".to_string(),
        ]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (start, matches) = helper
            .complete("/he", 3, &ctx)
            .expect("completion should work");

        assert_eq!(start, 0);
        assert_eq!(
            matches
                .into_iter()
                .map(|candidate| candidate.replacement)
                .collect::<Vec<_>>(),
            vec!["/help".to_string(), "/hello".to_string()]
        );
    }

    #[test]
    fn completes_contextual_slash_commands_with_arguments() {
        let helper = SlashCommandHelper::new(vec![
            "/help auth".to_string(),
            "/help sessions".to_string(),
            "/permissions workspace-write".to_string(),
        ]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (start, matches) = helper
            .complete("/help a", 7, &ctx)
            .expect("completion should work");

        assert_eq!(start, 0);
        assert_eq!(
            matches
                .into_iter()
                .map(|candidate| candidate.replacement)
                .collect::<Vec<_>>(),
            vec!["/help auth".to_string()]
        );
    }

    #[test]
    fn ignores_non_slash_command_completion_requests() {
        let helper = SlashCommandHelper::new(vec!["/help".to_string()]);
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (_, matches) = helper
            .complete("hello", 5, &ctx)
            .expect("completion should work");

        assert!(matches.is_empty());
    }

    #[test]
    fn tracks_current_buffer_through_highlighter() {
        let helper = SlashCommandHelper::new(Vec::new());
        let _ = helper.highlight("draft", 5);

        assert_eq!(helper.current_line(), "draft");
    }

    #[test]
    fn push_history_ignores_blank_entries() {
        let mut editor = LineEditor::new("› ", vec!["/help".to_string()]);
        editor.push_history("   ");
        editor.push_history("/help");

        assert_eq!(editor.editor.history().len(), 1);
    }

    #[test]
    fn handle_submission_does_not_intercept_plain_input() {
        let mut editor = LineEditor::new("> ", vec!["/help".to_string()]);
        editor.push_history("/help");

        let handled = editor
            .handle_submission("hello")
            .expect("submission handling should succeed");

        assert!(!handled);
        assert_eq!(editor.editor.history().len(), 1);
    }

    #[test]
    fn fallback_input_reads_full_multiline_payload() {
        let mut input = Cursor::new("first line\nsecond line\nthird line\n");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(
            result,
            Some("first line\nsecond line\nthird line".to_string())
        );
    }

    #[test]
    fn fallback_input_trims_final_crlf_without_touching_internal_newlines() {
        let mut input = Cursor::new("alpha\r\nbeta\r\n");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(result, Some("alpha\r\nbeta".to_string()));
    }

    #[test]
    fn fallback_input_returns_none_for_empty_stdin() {
        let mut input = Cursor::new("");

        let result = read_full_fallback_input(&mut input).expect("fallback read should succeed");

        assert_eq!(result, None);
    }

    #[test]
    fn truncates_status_line_by_visible_width() {
        let status = "\u{1b}[31mabcdef\u{1b}[0m";
        let rendered = truncate_ansi_visible_width(status, 4);

        assert_eq!(crate::ui::visible_width(&rendered), 4);
        assert!(rendered.ends_with("\u{1b}[0m"));
    }

    #[test]
    fn fit_status_line_leaves_short_lines_unchanged() {
        let status = "\u{1b}[31mshort\u{1b}[0m";

        assert_eq!(fit_status_line(status, Some(10)), status);
    }

    #[test]
    fn paste_safe_mode_reads_env_flag() {
        std::env::set_var("PEBBLE_PASTE_SAFE", "1");
        assert!(paste_safe_mode_enabled());

        std::env::remove_var("PEBBLE_PASTE_SAFE");
        assert!(!paste_safe_mode_enabled());
    }
}

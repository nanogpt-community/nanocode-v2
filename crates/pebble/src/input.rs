use std::borrow::Cow;
use std::cell::RefCell;
use std::io::{self, IsTerminal, Read, Write};

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Context, EditMode, Editor, Event,
    EventContext, EventHandler, Helper, KeyCode, KeyEvent, Modifiers,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Submit(String),
    Cancel,
    Exit,
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

fn paste_safe_mode_enabled() -> bool {
    env_flag_enabled("PEBBLE_PASTE_SAFE")
}

pub struct LineEditor {
    prompt: String,
    status_line: Option<String>,
    completions: Vec<String>,
    vim_enabled: bool,
    editor: Editor<SlashCommandHelper, DefaultHistory>,
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>, completions: Vec<String>) -> Self {
        let vim_enabled = env_flag_enabled("PEBBLE_VIM");
        let editor = Self::build_editor(completions.clone(), vim_enabled);

        Self {
            prompt: prompt.into(),
            status_line: None,
            completions,
            vim_enabled,
            editor,
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
            let mut stdout = io::stdout();
            writeln!(stdout, "{status}")?;
            stdout.flush()?;
        }
        Ok(())
    }

    pub fn read_line(&mut self) -> io::Result<ReadOutcome> {
        loop {
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                return self.read_line_fallback();
            }

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
        vim_enabled: bool,
    ) -> Editor<SlashCommandHelper, DefaultHistory> {
        let paste_safe_mode = paste_safe_mode_enabled();
        let config = Config::builder()
            .completion_type(CompletionType::List)
            .edit_mode(if vim_enabled {
                EditMode::Vi
            } else {
                EditMode::Emacs
            })
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

    fn handle_submission(&mut self, line: &str) -> io::Result<bool> {
        if line.trim() == "/vim" {
            self.toggle_vim()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn toggle_vim(&mut self) -> io::Result<()> {
        let history = self
            .editor
            .history()
            .iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        self.vim_enabled = !self.vim_enabled;
        self.editor = Self::build_editor(self.completions.clone(), self.vim_enabled);
        for entry in history {
            let _ = self.editor.add_history_entry(entry);
        }

        let mut stdout = io::stdout();
        writeln!(
            stdout,
            "Vim mode {}.",
            if self.vim_enabled {
                "enabled"
            } else {
                "disabled"
            }
        )
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
        paste_safe_mode_enabled, read_full_fallback_input, slash_command_prefix, LineEditor,
        SlashCommandHelper,
    };
    use std::io::Cursor;
    use rustyline::completion::Completer;
    use rustyline::highlight::Highlighter;
    use rustyline::history::{DefaultHistory, History};
    use rustyline::Context;

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
    fn vim_toggle_rebuilds_editor_and_preserves_history() {
        let mut editor = LineEditor::new("> ", vec!["/help".to_string(), "/vim".to_string()]);
        editor.push_history("/help");

        assert!(!editor.vim_enabled);

        let toggled = editor
            .handle_submission("/vim")
            .expect("toggle should succeed");

        assert!(toggled);
        assert!(editor.vim_enabled);
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
    fn paste_safe_mode_reads_env_flag() {
        std::env::set_var("PEBBLE_PASTE_SAFE", "1");
        assert!(paste_safe_mode_enabled());

        std::env::remove_var("PEBBLE_PASTE_SAFE");
        assert!(!paste_safe_mode_enabled());
    }
}

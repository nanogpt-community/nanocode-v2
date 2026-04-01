use std::io::{self, IsTerminal, Write};

use crossterm::cursor::{MoveDown, MoveToColumn, MoveUp};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::queue;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size, Clear, ClearType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputBuffer {
    buffer: String,
    cursor: usize,
}

impl InputBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
        }
    }

    pub fn insert(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn insert_newline(&mut self) {
        self.insert('\n');
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let previous = self.buffer[..self.cursor]
            .char_indices()
            .last()
            .map_or(0, |(idx, _)| idx);
        self.buffer.drain(previous..self.cursor);
        self.cursor = previous;
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.buffer[..self.cursor]
            .char_indices()
            .last()
            .map_or(0, |(idx, _)| idx);
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        if let Some(next) = self.buffer[self.cursor..].chars().next() {
            self.cursor += next.len_utf8();
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.buffer.len();
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    #[cfg(test)]
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
    }

    pub fn replace(&mut self, value: impl Into<String>) {
        self.buffer = value.into();
        self.cursor = self.buffer.len();
    }

    #[must_use]
    fn current_command_prefix(&self) -> Option<&str> {
        if self.cursor != self.buffer.len() {
            return None;
        }
        let prefix = &self.buffer[..self.cursor];
        if prefix.contains(char::is_whitespace) || !prefix.starts_with('/') {
            return None;
        }
        Some(prefix)
    }

    pub fn complete_slash_command(&mut self, candidates: &[String]) -> bool {
        let Some(prefix) = self.current_command_prefix() else {
            return false;
        };

        let matches = candidates
            .iter()
            .filter(|candidate| candidate.starts_with(prefix))
            .map(String::as_str)
            .collect::<Vec<_>>();
        if matches.is_empty() {
            return false;
        }

        let replacement = longest_common_prefix(&matches);
        if replacement == prefix {
            return false;
        }

        self.replace(replacement);
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedBuffer {
    lines: Vec<String>,
    visual_line_count: usize,
    cursor_row: u16,
    cursor_col: u16,
}

impl RenderedBuffer {
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.visual_line_count
    }

    fn write(&self, out: &mut impl Write) -> io::Result<()> {
        for (index, line) in self.lines.iter().enumerate() {
            if index > 0 {
                writeln!(out)?;
            }
            write!(out, "{line}")?;
        }
        Ok(())
    }

    #[cfg(test)]
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    #[cfg(test)]
    #[must_use]
    pub fn cursor_position(&self) -> (u16, u16) {
        (self.cursor_row, self.cursor_col)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Submit(String),
    Cancel,
    Exit,
}

pub struct LineEditor {
    prompt: String,
    continuation_prompt: String,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: Option<String>,
    completions: Vec<String>,
}

impl LineEditor {
    #[must_use]
    pub fn new(prompt: impl Into<String>, completions: Vec<String>) -> Self {
        Self {
            prompt: prompt.into(),
            continuation_prompt: String::from("> "),
            history: Vec::new(),
            history_index: None,
            draft: None,
            completions,
        }
    }

    pub fn push_history(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        if entry.trim().is_empty() {
            return;
        }
        self.history.push(entry);
        self.history_index = None;
        self.draft = None;
    }

    pub fn read_line(&mut self) -> io::Result<ReadOutcome> {
        let use_raw_editor = std::env::var("NANOCODE_RAW_REPL")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"));

        if !use_raw_editor || !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return self.read_line_fallback();
        }

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        let mut input = InputBuffer::new();
        let mut rendered_lines = 1usize;
        self.redraw(&mut stdout, &input, rendered_lines)?;

        loop {
            let event = event::read()?;
            if let Event::Key(key) = event {
                match self.handle_key(key, &mut input) {
                    EditorAction::Continue => {
                        rendered_lines = self.redraw(&mut stdout, &input, rendered_lines)?;
                    }
                    EditorAction::Submit => {
                        disable_raw_mode()?;
                        writeln!(stdout)?;
                        self.history_index = None;
                        self.draft = None;
                        return Ok(ReadOutcome::Submit(input.as_str().to_owned()));
                    }
                    EditorAction::Cancel => {
                        disable_raw_mode()?;
                        writeln!(stdout)?;
                        self.history_index = None;
                        self.draft = None;
                        return Ok(ReadOutcome::Cancel);
                    }
                    EditorAction::Exit => {
                        disable_raw_mode()?;
                        writeln!(stdout)?;
                        self.history_index = None;
                        self.draft = None;
                        return Ok(ReadOutcome::Exit);
                    }
                }
            }
        }
    }

    fn read_line_fallback(&self) -> io::Result<ReadOutcome> {
        let mut stdout = io::stdout();
        write!(stdout, "{}", self.prompt)?;
        stdout.flush()?;

        let mut buffer = String::new();
        let bytes_read = io::stdin().read_line(&mut buffer)?;
        if bytes_read == 0 {
            return Ok(ReadOutcome::Exit);
        }

        while matches!(buffer.chars().last(), Some('\n' | '\r')) {
            buffer.pop();
        }
        Ok(ReadOutcome::Submit(buffer))
    }

    #[allow(clippy::too_many_lines)]
    fn handle_key(&mut self, key: KeyEvent, input: &mut InputBuffer) -> EditorAction {
        match key {
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                if input.as_str().is_empty() {
                    EditorAction::Exit
                } else {
                    input.clear();
                    self.history_index = None;
                    self.draft = None;
                    EditorAction::Cancel
                }
            }
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                input.insert_newline();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SHIFT) => {
                input.insert_newline();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => EditorAction::Submit,
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                input.backspace();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                input.move_left();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                input.move_right();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.navigate_history_up(input);
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.navigate_history_down(input);
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                input.complete_slash_command(&self.completions);
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                input.move_home();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::End, ..
            } => {
                input.move_end();
                EditorAction::Continue
            }
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                input.clear();
                self.history_index = None;
                self.draft = None;
                EditorAction::Cancel
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            } if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                input.insert(ch);
                self.history_index = None;
                self.draft = None;
                EditorAction::Continue
            }
            _ => EditorAction::Continue,
        }
    }

    fn navigate_history_up(&mut self, input: &mut InputBuffer) {
        if self.history.is_empty() {
            return;
        }

        match self.history_index {
            Some(0) => {}
            Some(index) => {
                let next_index = index - 1;
                input.replace(self.history[next_index].clone());
                self.history_index = Some(next_index);
            }
            None => {
                self.draft = Some(input.as_str().to_owned());
                let next_index = self.history.len() - 1;
                input.replace(self.history[next_index].clone());
                self.history_index = Some(next_index);
            }
        }
    }

    fn navigate_history_down(&mut self, input: &mut InputBuffer) {
        let Some(index) = self.history_index else {
            return;
        };

        if index + 1 < self.history.len() {
            let next_index = index + 1;
            input.replace(self.history[next_index].clone());
            self.history_index = Some(next_index);
            return;
        }

        input.replace(self.draft.take().unwrap_or_default());
        self.history_index = None;
    }

    fn redraw(
        &self,
        out: &mut impl Write,
        input: &InputBuffer,
        previous_line_count: usize,
    ) -> io::Result<usize> {
        let width = terminal_width();
        let rendered = render_buffer(&self.prompt, &self.continuation_prompt, input, width);
        if previous_line_count > 1 {
            queue!(out, MoveUp(saturating_u16(previous_line_count - 1)))?;
        }
        queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
        rendered.write(out)?;
        queue!(
            out,
            MoveUp(saturating_u16(rendered.line_count().saturating_sub(1))),
            MoveToColumn(0),
        )?;
        if rendered.cursor_row > 0 {
            queue!(out, MoveDown(rendered.cursor_row))?;
        }
        queue!(out, MoveToColumn(rendered.cursor_col))?;
        out.flush()?;
        Ok(rendered.line_count())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditorAction {
    Continue,
    Submit,
    Cancel,
    Exit,
}

#[must_use]
pub fn render_buffer(
    prompt: &str,
    continuation_prompt: &str,
    input: &InputBuffer,
    width: u16,
) -> RenderedBuffer {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut visual_line_count = 0usize;
    let mut cursor_row = 0usize;
    let mut cursor_col = 0usize;
    let mut byte_offset = 0usize;
    for (index, line) in input.as_str().split('\n').enumerate() {
        let prefix = if index == 0 {
            prompt
        } else {
            continuation_prompt
        };
        let rendered_line = format!("{prefix}{line}");
        if input.cursor >= byte_offset && input.cursor <= byte_offset + line.len() {
            let cursor_line_chars = input.as_str()[byte_offset..input.cursor].chars().count();
            let absolute_cursor_col = prefix.chars().count() + cursor_line_chars;
            let wrapped_before_cursor = absolute_cursor_col / usize::from(width);
            cursor_row = visual_line_count + wrapped_before_cursor;
            cursor_col = absolute_cursor_col % usize::from(width);
        }
        visual_line_count += visual_rows(rendered_line.chars().count(), width);
        lines.push(rendered_line);
        byte_offset += line.len() + 1;
    }
    if lines.is_empty() {
        lines.push(prompt.to_string());
        visual_line_count = 1;
    }

    RenderedBuffer {
        lines,
        visual_line_count,
        cursor_row: saturating_u16(cursor_row),
        cursor_col: saturating_u16(cursor_col),
    }
}

#[must_use]
fn longest_common_prefix(values: &[&str]) -> String {
    let Some(first) = values.first() else {
        return String::new();
    };

    let mut prefix = (*first).to_string();
    for value in values.iter().skip(1) {
        while !value.starts_with(&prefix) {
            prefix.pop();
            if prefix.is_empty() {
                break;
            }
        }
    }
    prefix
}

#[must_use]
fn saturating_u16(value: usize) -> u16 {
    u16::try_from(value).unwrap_or(u16::MAX)
}

fn terminal_width() -> u16 {
    size().map(|(width, _)| width).unwrap_or(80).max(1)
}

fn visual_rows(char_count: usize, width: u16) -> usize {
    let width = usize::from(width.max(1));
    if char_count == 0 {
        return 1;
    }
    (char_count.saturating_sub(1) / width) + 1
}

#[cfg(test)]
mod tests {
    use super::{render_buffer, InputBuffer, LineEditor};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn supports_basic_line_editing() {
        let mut input = InputBuffer::new();
        input.insert('h');
        input.insert('i');
        input.move_left();
        input.insert('!');
        assert_eq!(input.as_str(), "h!i");
        assert_eq!(input.cursor(), 2);
        input.backspace();
        assert_eq!(input.as_str(), "hi");
        input.move_end();
        input.insert_newline();
        input.insert('o');
        input.insert('k');
        assert_eq!(input.as_str(), "hi\nok");
    }

    #[test]
    fn renders_multiline_buffers_with_cursor_position() {
        let mut input = InputBuffer::new();
        for ch in "hello\nworld".chars() {
            input.insert(ch);
        }
        input.move_left();
        input.move_left();
        let rendered = render_buffer("› ", "> ", &input, 80);
        assert_eq!(rendered.lines(), &["› hello".to_string(), "> world".to_string()]);
        assert_eq!(rendered.cursor_position(), (1, 5));
    }

    #[test]
    fn counts_terminal_wrapping_for_cursor_and_height() {
        let mut input = InputBuffer::new();
        for ch in "abcdef".chars() {
            input.insert(ch);
        }
        let rendered = render_buffer("› ", "> ", &input, 4);
        assert_eq!(rendered.line_count(), 2);
        assert_eq!(rendered.cursor_position(), (2, 0));
    }

    #[test]
    fn tab_completion_expands_slash_commands() {
        let mut editor = LineEditor::new("› ", vec!["/status".to_string(), "/start".to_string()]);
        let mut input = InputBuffer::new();
        input.insert('/');
        input.insert('s');
        editor.handle_key(key(KeyCode::Tab), &mut input);
        assert_eq!(input.as_str(), "/sta");
    }

    #[test]
    fn history_navigation_preserves_current_draft() {
        let mut editor = LineEditor::new("› ", Vec::new());
        editor.push_history("first");
        editor.push_history("second");
        let mut input = InputBuffer::new();
        input.replace("dr");

        editor.handle_key(key(KeyCode::Up), &mut input);
        assert_eq!(input.as_str(), "second");
        editor.handle_key(key(KeyCode::Up), &mut input);
        assert_eq!(input.as_str(), "first");
        editor.handle_key(key(KeyCode::Down), &mut input);
        assert_eq!(input.as_str(), "second");
        editor.handle_key(key(KeyCode::Down), &mut input);
        assert_eq!(input.as_str(), "dr");
    }
}

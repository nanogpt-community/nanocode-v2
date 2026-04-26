use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use api::{ApiService, ReasoningEffort};
use platform::write_atomic;
use runtime::{
    ContentBlock, ConversationMessage, EditHistoryFile, MessageRole, PermissionMode, Session,
    SessionMetadata, SessionTurnSnapshot,
};
use serde_json::Value as JsonValue;

use crate::app::{
    parse_collaboration_mode_arg, parse_permission_mode_arg, parse_reasoning_effort_arg,
    reasoning_effort_label, AllowedToolSet, CollaborationMode, FastMode, DEFAULT_MODEL,
    MAX_TURN_SNAPSHOT_STACK_ENTRIES,
};
use crate::models::infer_service_for_model;
use crate::report::truncate_for_summary;

#[derive(Debug, Clone)]
pub(crate) struct SessionHandle {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ManagedSessionSummary {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) modified_epoch_secs: u64,
    pub(crate) message_count: usize,
    pub(crate) title: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) started_at: Option<String>,
    pub(crate) last_prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionRuntimeState {
    pub(crate) model: String,
    pub(crate) service: ApiService,
    pub(crate) allowed_tools: Option<AllowedToolSet>,
    pub(crate) permission_mode: PermissionMode,
    pub(crate) collaboration_mode: CollaborationMode,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) fast_mode: FastMode,
    pub(crate) proxy_tool_calls: bool,
}

pub(crate) fn sessions_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let path = cwd.join(".pebble").join("sessions");
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub(crate) fn create_managed_session_handle() -> Result<SessionHandle, Box<dyn std::error::Error>> {
    let id = generate_session_id();
    let path = sessions_dir()?.join(format!("{id}.json"));
    Ok(SessionHandle { id, path })
}

fn generate_session_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("session-{millis}")
}

pub(crate) fn resolve_session_reference(
    reference: &str,
) -> Result<SessionHandle, Box<dyn std::error::Error>> {
    if reference.trim().eq_ignore_ascii_case("last") {
        let Some(session) = list_managed_sessions()?.into_iter().next() else {
            return Err("no saved sessions available".into());
        };
        return Ok(SessionHandle {
            id: session.id,
            path: session.path,
        });
    }
    let direct = PathBuf::from(reference);
    let cwd_relative = env::current_dir()?.join(reference);
    let path = if direct.exists() {
        direct
    } else if cwd_relative.exists() {
        cwd_relative
    } else {
        sessions_dir()?.join(format!("{reference}.json"))
    };
    if !path.exists() {
        return Err(format!("session not found: {reference}").into());
    }
    let id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(reference)
        .to_string();
    Ok(SessionHandle { id, path })
}

pub(crate) fn list_managed_sessions(
) -> Result<Vec<ManagedSessionSummary>, Box<dyn std::error::Error>> {
    let mut sessions = Vec::new();
    for entry in fs::read_dir(sessions_dir()?)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let metadata = entry.metadata()?;
        let modified_epoch_secs = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or_default();
        let session = Session::load_from_path(&path).ok();
        let derived_message_count = session.as_ref().map_or(0, |session| session.messages.len());
        let stored = session
            .as_ref()
            .and_then(|session| session.metadata.as_ref());
        let id = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_string();
        sessions.push(ManagedSessionSummary {
            id,
            path,
            modified_epoch_secs,
            message_count: stored.map_or(derived_message_count, |metadata| {
                metadata.message_count as usize
            }),
            title: stored.and_then(|metadata| metadata.title.clone()),
            model: stored.map(|metadata| metadata.model.clone()),
            started_at: stored.map(|metadata| metadata.started_at.clone()),
            last_prompt: stored.and_then(|metadata| metadata.last_prompt.clone()),
        });
    }
    sessions.sort_by_key(|session| Reverse(session.modified_epoch_secs));
    Ok(sessions)
}

pub(crate) fn render_session_list(
    active_session_id: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let sessions = list_managed_sessions()?;
    let mut lines = vec![
        "Sessions".to_string(),
        format!("  Directory         {}", sessions_dir()?.display()),
    ];
    if sessions.is_empty() {
        lines.push("  No managed sessions saved yet.".to_string());
        return Ok(lines.join("\n"));
    }
    for session in sessions {
        let marker = if session.id == active_session_id {
            "● current"
        } else {
            "○ saved"
        };
        let model = session.model.as_deref().unwrap_or("unknown");
        let started = session.started_at.as_deref().unwrap_or("unknown");
        let last_prompt = session.last_prompt.as_deref().map_or_else(
            || "-".to_string(),
            |prompt| truncate_for_summary(prompt, 36),
        );
        let title = session
            .title
            .as_deref()
            .map_or_else(|| "-".to_string(), |title| truncate_for_summary(title, 24));
        lines.push(format!(
            "  {id:<20} {marker:<10} msgs={msgs:<4} model={model:<24} title={title:<24} started={started} modified={modified} last={last_prompt} path={path}",
            id = session.id,
            msgs = session.message_count,
            model = model,
            title = title,
            started = started,
            modified = session.modified_epoch_secs,
            last_prompt = last_prompt,
            path = session.path.display(),
        ));
    }
    Ok(lines.join("\n"))
}

pub(crate) fn render_session_timeline(session: &Session) -> String {
    let mut lines = vec!["Timeline".to_string()];
    if session.messages.is_empty() {
        lines.push("  No messages in this session.".to_string());
        return lines.join("\n");
    }
    for (index, message) in session.messages.iter().enumerate() {
        lines.push(format!(
            "  {idx:>3}. {id:<24} {role:<9} {preview}",
            idx = index + 1,
            id = short_message_id(&message.id),
            role = message_role_label(message.role),
            preview = truncate_for_summary(&message_preview(message), 88),
        ));
    }
    lines.join("\n")
}

pub(crate) fn resolve_timeline_target(session: &Session, target: &str) -> Option<usize> {
    if let Ok(index) = target.parse::<usize>() {
        return (index > 0 && index <= session.messages.len()).then_some(index - 1);
    }
    session
        .messages
        .iter()
        .position(|message| message.id == target || message.id.starts_with(target))
}

fn short_message_id(id: &str) -> String {
    if id.chars().count() <= 22 {
        id.to_string()
    } else {
        format!("{}…", id.chars().take(21).collect::<String>())
    }
}

fn message_role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn message_preview(message: &ConversationMessage) -> String {
    let Some(block) = message.blocks.first() else {
        return String::new();
    };
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Thinking { text, .. } => format!("thinking: {text}"),
        ContentBlock::ToolUse { name, input, .. } => {
            format!("tool use: {name} {}", summarize_tool_payload(input))
        }
        ContentBlock::ToolResult {
            tool_name,
            output,
            is_error,
            ..
        } => {
            let prefix = if *is_error {
                "tool error"
            } else {
                "tool result"
            };
            format!("{prefix}: {tool_name} {}", summarize_tool_payload(output))
        }
        ContentBlock::CompactionSummary { summary, .. } => {
            format!("compaction: {summary}")
        }
    }
}

pub(crate) fn summarize_tool_payload(payload: &str) -> String {
    let compact = match serde_json::from_str::<JsonValue>(payload) {
        Ok(value) => value.to_string(),
        Err(_) => payload.trim().to_string(),
    };
    truncate_for_summary(&compact, 96)
}

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

pub(crate) fn current_timestamp_rfc3339ish() -> String {
    format!("{}Z", current_epoch_secs())
}

fn last_prompt_from_session(session: &Session) -> Option<String> {
    session
        .messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::User)
        .and_then(|message| {
            message.blocks.iter().find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.trim().to_string()),
                _ => None,
            })
        })
        .filter(|text| !text.is_empty())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_session_metadata(
    session: &Session,
    model: &str,
    allowed_tools: Option<&AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    proxy_tool_calls: bool,
) -> SessionMetadata {
    let started_at = session
        .metadata
        .as_ref()
        .map_or_else(current_timestamp_rfc3339ish, |metadata| {
            metadata.started_at.clone()
        });
    SessionMetadata {
        title: session
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.title.clone()),
        started_at,
        model: model.to_string(),
        message_count: session.messages.len().try_into().unwrap_or(u32::MAX),
        last_prompt: last_prompt_from_session(session),
        permission_mode: Some(permission_mode.as_str().to_string()),
        thinking_enabled: Some(reasoning_effort.is_some()),
        collaboration_mode: Some(collaboration_mode.as_str().to_string()),
        reasoning_effort: reasoning_effort
            .map(|effort| reasoning_effort_label(Some(effort)).to_string()),
        fast_mode: Some(fast_mode.enabled()),
        proxy_tool_calls: Some(proxy_tool_calls),
        allowed_tools: allowed_tools.map(|allowed| allowed.iter().cloned().collect()),
        edit_history: session
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.edit_history.clone()),
        undo_stack: session
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.undo_stack.clone()),
        redo_stack: session
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.redo_stack.clone()),
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WorktreeSnapshot {
    files: BTreeMap<String, SnapshotFileState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotFileState {
    exists: bool,
    content: String,
}

impl WorktreeSnapshot {
    pub(crate) fn capture(cwd: &Path) -> Self {
        let mut files = BTreeMap::new();
        for path in git_dirty_paths(cwd) {
            files.insert(path.clone(), read_snapshot_file_state(cwd, &path));
        }
        Self { files }
    }
}

pub(crate) fn build_turn_snapshot(
    cwd: &Path,
    before_files: &WorktreeSnapshot,
    before_message_count: usize,
    after_session: &Session,
) -> Option<SessionTurnSnapshot> {
    let messages = after_session
        .messages
        .iter()
        .skip(before_message_count)
        .cloned()
        .collect::<Vec<_>>();
    let mut file_changes = file_changes_from_turn_messages(cwd, &messages);
    let mut changed_paths = before_files.files.keys().cloned().collect::<BTreeSet<_>>();
    changed_paths.extend(git_dirty_paths(cwd));

    for path in changed_paths {
        file_changes.entry(path.clone()).or_insert_with(|| {
            let before = before_files.files.get(&path).cloned().unwrap_or_else(|| {
                read_clean_tracked_snapshot_state(cwd, &path).unwrap_or(SnapshotFileState {
                    exists: false,
                    content: String::new(),
                })
            });
            let after = read_snapshot_file_state(cwd, &path);
            EditHistoryFile {
                path,
                before: before.content,
                after: after.content,
                before_exists: before.exists,
                after_exists: after.exists,
            }
        });
    }

    let files = file_changes
        .into_values()
        .filter(|file| file.before_exists != file.after_exists || file.before != file.after)
        .collect::<Vec<_>>();
    if files.is_empty() && messages.is_empty() {
        return None;
    }

    Some(SessionTurnSnapshot {
        timestamp: current_timestamp_rfc3339ish(),
        message_count_before: before_message_count.try_into().unwrap_or(u32::MAX),
        prompt: messages.iter().find_map(user_text_from_message),
        messages,
        files,
    })
}

pub(crate) fn file_changes_from_turn_messages(
    cwd: &Path,
    messages: &[ConversationMessage],
) -> BTreeMap<String, EditHistoryFile> {
    let mut changes = BTreeMap::new();
    for message in messages {
        for block in &message.blocks {
            let ContentBlock::ToolResult {
                tool_name,
                output,
                is_error,
                ..
            } = block
            else {
                continue;
            };
            if *is_error {
                continue;
            }
            for change in file_changes_from_tool_output(cwd, tool_name, output) {
                changes
                    .entry(change.path.clone())
                    .and_modify(|existing: &mut EditHistoryFile| {
                        existing.after.clone_from(&change.after);
                        existing.after_exists = change.after_exists;
                    })
                    .or_insert(change);
            }
        }
    }
    changes
}

fn file_changes_from_tool_output(
    cwd: &Path,
    tool_name: &str,
    output: &str,
) -> Vec<EditHistoryFile> {
    let Ok(value) = serde_json::from_str::<JsonValue>(output) else {
        return Vec::new();
    };
    match tool_name {
        "write_file" => file_change_from_write_file_output(cwd, &value)
            .into_iter()
            .collect(),
        "edit_file" => file_change_from_edit_file_output(cwd, &value)
            .into_iter()
            .collect(),
        "apply_patch" => file_changes_from_apply_patch_output(cwd, &value),
        _ => Vec::new(),
    }
}

fn file_change_from_write_file_output(cwd: &Path, value: &JsonValue) -> Option<EditHistoryFile> {
    let path = value.get("filePath")?.as_str()?;
    let after = value.get("content")?.as_str()?.to_string();
    let before_value = value.get("originalFile")?;
    let before = before_value.as_str().map(ToOwned::to_owned);
    Some(EditHistoryFile {
        path: snapshot_path_for_output(cwd, path),
        before: before.clone().unwrap_or_default(),
        after,
        before_exists: before.is_some(),
        after_exists: true,
    })
}

fn file_change_from_edit_file_output(cwd: &Path, value: &JsonValue) -> Option<EditHistoryFile> {
    let path = value.get("filePath")?.as_str()?;
    let before = value.get("originalFile")?.as_str()?.to_string();
    let old_string = value.get("oldString")?.as_str()?;
    let new_string = value.get("newString")?.as_str()?;
    let replace_all = value
        .get("replaceAll")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let after = if replace_all {
        before.replace(old_string, new_string)
    } else {
        before.replacen(old_string, new_string, 1)
    };
    Some(EditHistoryFile {
        path: snapshot_path_for_output(cwd, path),
        before,
        after,
        before_exists: true,
        after_exists: true,
    })
}

fn file_changes_from_apply_patch_output(cwd: &Path, value: &JsonValue) -> Vec<EditHistoryFile> {
    value
        .get("changedFiles")
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
        .filter_map(|file| {
            let path = file.get("filePath")?.as_str()?;
            let before_exists = file
                .get("beforeExists")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let after_exists = file
                .get("afterExists")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let before = file
                .get("beforeContent")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            let after = file
                .get("afterContent")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            Some(EditHistoryFile {
                path: snapshot_path_for_output(cwd, path),
                before,
                after,
                before_exists,
                after_exists,
            })
        })
        .collect()
}

fn snapshot_path_for_output(cwd: &Path, path: &str) -> String {
    let path = Path::new(path);
    if path.is_absolute() {
        path.strip_prefix(cwd).ok().map_or_else(
            || path.to_string_lossy().into_owned(),
            |path| path.to_string_lossy().into_owned(),
        )
    } else {
        path.to_string_lossy().into_owned()
    }
}

pub(crate) fn append_undo_snapshot(session: &mut Session, snapshot: SessionTurnSnapshot) {
    push_undo_snapshot(session, snapshot, true);
}

pub(crate) fn push_undo_snapshot(
    session: &mut Session,
    snapshot: SessionTurnSnapshot,
    clear_redo: bool,
) {
    if session.metadata.is_none() {
        session.metadata = Some(SessionMetadata {
            title: None,
            started_at: current_timestamp_rfc3339ish(),
            model: DEFAULT_MODEL.to_string(),
            message_count: session.messages.len().try_into().unwrap_or(u32::MAX),
            last_prompt: last_prompt_from_session(session),
            permission_mode: None,
            thinking_enabled: None,
            collaboration_mode: None,
            reasoning_effort: None,
            fast_mode: None,
            proxy_tool_calls: None,
            allowed_tools: None,
            edit_history: None,
            undo_stack: None,
            redo_stack: None,
        });
    }
    if let Some(metadata) = &mut session.metadata {
        metadata
            .undo_stack
            .get_or_insert_with(Vec::new)
            .push(snapshot);
        trim_turn_snapshot_stack(metadata.undo_stack.as_mut());
        if clear_redo {
            metadata.redo_stack = Some(Vec::new());
        }
    }
}

fn trim_turn_snapshot_stack(stack: Option<&mut Vec<SessionTurnSnapshot>>) {
    let Some(stack) = stack else {
        return;
    };
    let overflow = stack.len().saturating_sub(MAX_TURN_SNAPSHOT_STACK_ENTRIES);
    if overflow > 0 {
        stack.drain(0..overflow);
    }
}

fn user_text_from_message(message: &ConversationMessage) -> Option<String> {
    (message.role == MessageRole::User)
        .then(|| {
            message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
}

fn git_dirty_paths(cwd: &Path) -> BTreeSet<String> {
    let output = Command::new("git")
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .current_dir(cwd)
        .output();
    let Ok(output) = output else {
        return BTreeSet::new();
    };
    if !output.status.success() {
        return BTreeSet::new();
    }

    let mut paths = BTreeSet::new();
    let mut pending_rename = false;
    for record in output.stdout.split(|byte| *byte == 0) {
        if record.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(record);
        if pending_rename {
            paths.insert(text.to_string());
            pending_rename = false;
            continue;
        }
        if text.len() < 4 {
            continue;
        }
        let status = &text[..2];
        let path = text[3..].to_string();
        if status.starts_with('R') || status.ends_with('R') {
            pending_rename = true;
        }
        paths.insert(path);
    }
    paths
}

fn read_snapshot_file_state(cwd: &Path, relative_path: &str) -> SnapshotFileState {
    let path = cwd.join(relative_path);
    match fs::read_to_string(path) {
        Ok(content) => SnapshotFileState {
            exists: true,
            content,
        },
        Err(_) => SnapshotFileState {
            exists: false,
            content: String::new(),
        },
    }
}

fn read_clean_tracked_snapshot_state(cwd: &Path, relative_path: &str) -> Option<SnapshotFileState> {
    let output = Command::new("git")
        .args(["show", &format!("HEAD:{relative_path}")])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|content| SnapshotFileState {
            exists: true,
            content,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SnapshotDirection {
    Before,
    After,
}

pub(crate) fn pop_undo_snapshot(session: &mut Session) -> Option<SessionTurnSnapshot> {
    session
        .metadata
        .as_mut()
        .and_then(|metadata| metadata.undo_stack.as_mut())
        .and_then(Vec::pop)
}

pub(crate) fn pop_redo_snapshot(session: &mut Session) -> Option<SessionTurnSnapshot> {
    session
        .metadata
        .as_mut()
        .and_then(|metadata| metadata.redo_stack.as_mut())
        .and_then(Vec::pop)
}

pub(crate) fn push_redo_snapshot(session: &mut Session, snapshot: SessionTurnSnapshot) {
    if let Some(metadata) = &mut session.metadata {
        metadata
            .redo_stack
            .get_or_insert_with(Vec::new)
            .push(snapshot);
        trim_turn_snapshot_stack(metadata.redo_stack.as_mut());
    }
}

pub(crate) fn restore_snapshot_files(
    cwd: &Path,
    snapshot: &SessionTurnSnapshot,
    direction: SnapshotDirection,
) -> io::Result<()> {
    for file in &snapshot.files {
        let (exists, content) = match direction {
            SnapshotDirection::Before => (file.before_exists, file.before.as_str()),
            SnapshotDirection::After => (file.after_exists, file.after.as_str()),
        };
        let path = cwd.join(&file.path);
        if exists {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            write_atomic(path, content)?;
        } else if path.is_file() || path.is_symlink() {
            fs::remove_file(path)?;
        } else if path.is_dir() {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn session_runtime_state(
    session: &Session,
    fallback_model: &str,
    fallback_allowed_tools: Option<&AllowedToolSet>,
    fallback_permission_mode: PermissionMode,
    fallback_collaboration_mode: CollaborationMode,
    fallback_reasoning_effort: Option<ReasoningEffort>,
    fallback_fast_mode: FastMode,
    fallback_proxy_tool_calls: bool,
) -> SessionRuntimeState {
    let metadata = session.metadata.as_ref();
    let model = metadata.map_or_else(
        || fallback_model.to_string(),
        |metadata| metadata.model.clone(),
    );
    let service = infer_service_for_model(&model);
    let allowed_tools = metadata
        .and_then(|metadata| metadata.allowed_tools.as_ref())
        .map(|tools| tools.iter().cloned().collect::<AllowedToolSet>())
        .or_else(|| fallback_allowed_tools.cloned());
    let permission_mode = metadata
        .and_then(|metadata| metadata.permission_mode.as_deref())
        .and_then(|value| parse_permission_mode_arg(value).ok())
        .unwrap_or(fallback_permission_mode);
    let collaboration_mode = metadata
        .and_then(|metadata| metadata.collaboration_mode.as_deref())
        .and_then(|value| parse_collaboration_mode_arg(value).ok())
        .unwrap_or(fallback_collaboration_mode);
    let reasoning_effort = metadata
        .and_then(|metadata| metadata.reasoning_effort.as_deref())
        .and_then(|value| parse_reasoning_effort_arg(value).ok())
        .flatten()
        .or(fallback_reasoning_effort);
    let fast_mode =
        metadata
            .and_then(|metadata| metadata.fast_mode)
            .map_or(fallback_fast_mode, |enabled| {
                if enabled {
                    FastMode::On
                } else {
                    FastMode::Off
                }
            });
    let proxy_tool_calls = metadata
        .and_then(|metadata| metadata.proxy_tool_calls)
        .unwrap_or(fallback_proxy_tool_calls);

    SessionRuntimeState {
        model,
        service,
        allowed_tools,
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
        proxy_tool_calls,
    }
}
